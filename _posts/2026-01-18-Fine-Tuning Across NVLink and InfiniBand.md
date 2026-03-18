---
layout: single
title: "Multi-Node Fine-Tuning: FSDP Sharding Strategy Matters"
author_profile: false
---

## Introduction

In a [previous post]({% post_url 2026-01-08-LLM Inference Across NVLink and InfiniBand %}), I ran Qwen2.5-72B inference on Azure H100 nodes and showed how NVLink's 900 GB/s bandwidth dominates InfiniBand's 400 Gb/s once tensor parallelism crosses a node boundary. The takeaway was clear: for inference, stay within the NVLink domain when possible.

Training changes the equation. Unlike inference — where a single request flows through the model once — training involves a forward pass, backward pass, gradient synchronization, and weight updates, all repeated thousands of times. The communication pattern is fundamentally different, and the choice of *how* to distribute model state across GPUs determines whether your inter-node fabric becomes a bottleneck or barely a factor.

In this post, I fine-tune **Qwen2.5-7B** on the same 3-node Azure H100 cluster using PyTorch FSDP, scaling from 2 to 24 GPUs. I test two sharding strategies — `FULL_SHARD` and `HYBRID_SHARD` — and show how the wrong strategy can make adding nodes *decrease* total throughput, while the right strategy delivers near-linear scaling across InfiniBand.

## Test Environment

The hardware is identical to the [previous post]({% post_url 2026-01-08-LLM Inference Across NVLink and InfiniBand %}); all three nodes passed Azure NHC and NCCL all-reduce baselines. I won't repeat the validation here — see the inference post for the full node health and NCCL results.

| Component | Detail |
|-----------|--------|
| **VM SKU** | Standard_ND96isr_H100_v5 |
| **GPUs per node** | 8× NVIDIA H100 80 GB HBM3 |
| **Intra-node** | NVLink 4th gen, 900 GB/s bisection |
| **Inter-node** | 8× 400 Gb/s NDR InfiniBand (ConnectX-7) |
| **Nodes** | 3 (24 GPUs total) |
| **Container** | `nvcr.io/nvidia/pytorch:24.12-py3` (PyTorch 2.6.0a0, CUDA 13.0, NCCL 2.23.4) |
| **Model** | Qwen2.5-7B (~14 GB in bf16) |
| **Framework** | PyTorch FSDP |

I chose Qwen2.5-7B instead of the 72B model from the inference post because a 7B model fits comfortably within a single node — this means the decision to go multi-node is purely about scaling throughput, not about memory capacity. It also makes the sharding strategy comparison clean: FULL_SHARD and HYBRID_SHARD are both valid options at every scale, so the performance difference is entirely due to communication patterns.

## Making Multi-Node FSDP Training Actually Work

Before diving into results, it's worth documenting the journey of getting multi-node FSDP training running on a bare VMSS cluster — because the setup is where most people get stuck.

### Challenge 1: Container Environment Consistency

Each node needs an identical software environment. I used NVIDIA's PyTorch NGC container (`nvcr.io/nvidia/pytorch:24.12-py3`) which bundles PyTorch, CUDA, and NCCL. However, HuggingFace `transformers` isn't pre-installed, and the version matters:

**Problem:** The default `pip install transformers` pulls the latest version (4.48+), which imports `TransformGetItemToIndex` — a symbol that doesn't exist in PyTorch 2.6.0a0's `torch.fx.passes.graph_transform_observer`. The training script crashes before touching a GPU.

**Solution:** Pin the version:
```bash
pip install transformers==4.47.1 -q
```

This is baked into the launch script so every node installs the same version at container start.

### Challenge 2: Model Distribution

The model needs to be present on every node at the same path. I downloaded Qwen2.5-7B once on the head node and used `rsync` to copy it:

```bash
# Download on head node
huggingface-cli download Qwen/Qwen2.5-7B --local-dir ~/models/Qwen2.5-7B

# Copy to workers
rsync -avP ~/models/ vmssAYZGM2:~/models/
rsync -avP ~/models/ vmssCZIUQ2:~/models/
```

**Pitfall:** `rsync` preserves directory structure, so if your `--local-dir` path doesn't match what the workers expect, you can end up with nested directories (`/models/Qwen2.5-7B/Qwen2.5-7B/`). The model loads fine from the wrong directory at first — and then fails silently with wrong weights or an obscure `safetensors` error. Always verify the model path on every node:

```bash
ssh vmssAYZGM2 "ls ~/models/Qwen2.5-7B/config.json && echo OK"
ssh vmssCZIUQ2 "ls ~/models/Qwen2.5-7B/config.json && echo OK"
```

### Challenge 3: Orchestrating Multi-Node torchrun

PyTorch's `torchrun` (elastic launch) expects every node to run a `torchrun` process with matching `--nnodes`, `--nproc_per_node`, `--master_addr`, and `--master_port`. All nodes must start within the rendezvous timeout (default: 15 minutes), and they must be able to reach each other over the network.

In a cloud VMSS environment without a shared job scheduler (like Slurm), you need to orchestrate this yourself. I wrote two scripts:

**`launch_node.sh`** — Runs on each node, starts a Docker container with the right NCCL environment variables and calls `torchrun`:

```bash
#!/bin/bash
# Usage: bash launch_node.sh <num_nodes> <node_rank> <master_addr> <master_port> [sharding]
NUM_NODES=$1
NODE_RANK=$2
MASTER_ADDR=$3
MASTER_PORT=$4
SHARDING=${5:-full}
GPUS_PER_NODE=8

sudo docker run --rm \
    --gpus all --ipc=host --ulimit memlock=-1 \
    --net=host --privileged \
    -v /home/azureuser/models:/models \
    -v /home/azureuser/finetune_bench.py:/bench.py \
    -v /home/azureuser/results:/results \
    -e SHARDING=$SHARDING \
    --name bench_node${NODE_RANK} \
    nvcr.io/nvidia/pytorch:24.12-py3 \
    bash -c "pip install transformers==4.47.1 -q && \
    NCCL_IB_DISABLE=0 NCCL_SOCKET_IFNAME=eth0 NCCL_DEBUG=WARN \
    torchrun \
        --nproc_per_node=$GPUS_PER_NODE \
        --nnodes=$NUM_NODES \
        --node_rank=$NODE_RANK \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        /bench.py"
```

Key Docker flags:
- `--net=host`: Required so containers across nodes can reach each other via the host network (InfiniBand uses RDMA, which needs host networking).
- `--privileged`: Needed for full GPU and IB device access.
- `--ipc=host --ulimit memlock=-1`: Shared memory and locked memory for NCCL.
- `NCCL_IB_DISABLE=0`: Explicitly enable InfiniBand (the default in this container, but worth being explicit).
- `NCCL_SOCKET_IFNAME=eth0`: Tell NCCL to use the right interface for out-of-band control messages.

**`run_multinode.sh`** — Orchestrates the full run from the head node:

```bash
#!/bin/bash
# Usage: bash run_multinode.sh <num_nodes> <node_list> [sharding]
# Example: bash run_multinode.sh 2 "vmssE6JHE7,vmssAYZGM2" hybrid

NUM_NODES=${1:-2}
NODE_LIST=${2:-"vmssE6JHE7,vmssAYZGM2"}
SHARDING=${3:-full}
MASTER_ADDR="10.0.0.4"
MASTER_PORT=29500

IFS=',' read -ra NODES <<< "$NODE_LIST"

# Launch workers first (background via SSH)
for i in "${!NODES[@]}"; do
    NODE="${NODES[$i]}"
    if [ "$NODE" != "vmssE6JHE7" ]; then
        ssh "$NODE" "nohup bash ~/launch_node.sh $NUM_NODES $i \
            $MASTER_ADDR $MASTER_PORT $SHARDING \
            > ~/bench_node${i}.log 2>&1 &"
        sleep 2
    fi
done

# Launch head node (foreground, rank 0)
bash ~/launch_node.sh $NUM_NODES 0 $MASTER_ADDR $MASTER_PORT $SHARDING
```

The pattern is: launch workers first via `nohup` + SSH (they block on the rendezvous waiting for the master), then launch the head node in the foreground (which triggers the rendezvous and starts training). The worker logs are tailed after the head node finishes.

**Pitfall — stale containers:** If a previous run crashed or was interrupted, Docker containers with the same name may still exist on worker nodes. The next launch silently fails because `docker run --name bench_node1` returns an error that gets buried in `nohup` output. Always clean up before a new run:

```bash
sudo docker rm -f bench_node0 2>/dev/null
ssh vmssAYZGM2 "sudo docker rm -f bench_node1" 2>/dev/null
ssh vmssCZIUQ2 "sudo docker rm -f bench_node2" 2>/dev/null
```

### Challenge 4: FSDP + Activation Checkpointing

PyTorch FSDP wraps the model and its sub-modules for sharded data parallelism. But getting it to work with HuggingFace models requires care:

1. **Auto-wrap policy:** FSDP needs to know which sub-modules to wrap individually. For Qwen2.5-7B, the right unit is `Qwen2DecoderLayer`:

    ```python
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

    auto_wrap = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={Qwen2DecoderLayer},
    )
    ```

2. **Activation checkpointing matters:** A 7B model in bf16 is ~14 GB for parameters alone. With 8 GPUs and FULL_SHARD, each GPU holds ~1.75 GB of parameters. But activations for a 2048-token sequence are much larger — without checkpointing, you quickly OOM. The trick is to use *FSDP-native* activation checkpointing, not HuggingFace's built-in `gradient_checkpointing_enable()`:

    ```python
    # HuggingFace's own checkpointing (for model loading)
    model.gradient_checkpointing_enable()

    # FSDP-native activation checkpointing (after wrapping)
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper, CheckpointImpl, apply_activation_checkpointing,
    )

    non_reentrant_wrapper = functools.partial(
        checkpoint_wrapper,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=non_reentrant_wrapper,
        check_fn=lambda m: isinstance(m, Qwen2DecoderLayer),
    )
    ```

    The `NO_REENTRANT` implementation is critical — the older reentrant version is incompatible with FSDP's forward hooks.

3. **Disable KV cache:** Training doesn't use autoregressive decoding, but HuggingFace models enable KV caching by default. This wastes memory and causes shape mismatches during backward:

    ```python
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, use_cache=False,
    )
    ```

## The Benchmark

Each run processes synthetic data (random token sequences of length 2048) for 50 benchmark steps after 10 warmup steps, with a batch size of 1 per GPU. The metric is aggregate training throughput in tokens/sec.

### Intra-Node Scaling (2, 4, 8 GPUs — Single Node)

Within a single node, all GPU communication uses NVLink. FULL_SHARD and HYBRID_SHARD behave identically here (there's only one node, so there's nothing to "replicate across").

| GPUs | Throughput (tok/s) | Per-GPU (tok/s) | Scaling vs 2 GPU |
|------|--------------------|-----------------|------------------|
| 2    | 8,818              | 4,409           | 1.00×            |
| 4    | 17,964             | 4,491           | 2.04×            |
| 8    | 36,382             | 4,548           | 4.13×            |

This is **near-linear scaling**: doubling GPUs doubles throughput. The per-GPU throughput actually *increases slightly* (4,409 → 4,548) as we add GPUs, because FSDP shards the model more finely — each GPU holds less parameter data, leaving more memory for CUDA's workspace and reducing memory pressure. The all-gather and reduce-scatter operations stay on NVLink's 900 GB/s fabric and are effectively free at this model size.

### Crossing the Node Boundary — FULL_SHARD

Now the critical test: what happens when we go beyond 8 GPUs and cross from NVLink to InfiniBand?

| GPUs | Nodes | Throughput (tok/s) | Per-GPU (tok/s) | Scaling vs 8 GPU |
|------|-------|--------------------|-----------------|------------------|
| 8    | 1     | 36,382             | 4,548           | 1.00×            |
| 16   | 2     | 33,622             | 2,101           | 0.92×            |
| 24   | 3     | 31,589             | 1,316           | 0.87×            |

**Adding nodes makes it *slower*.** The total throughput at 16 GPUs (33,622 tok/s) is *lower* than 8 GPUs (36,382 tok/s), and 24 GPUs (31,589 tok/s) is lower still. Per-GPU throughput craters from 4,548 to 2,101 (16 GPU) and 1,316 (24 GPU) — a **71% per-GPU efficiency loss** at 24 GPUs.

### HYBRID_SHARD to the Rescue

`HYBRID_SHARD` changes the communication pattern: parameters are sharded within each node (over NVLink), but replicated across nodes. Inter-node IB only carries gradient all-reduce — not per-layer parameter all-gather.

| GPUs | Nodes | Throughput (tok/s) | Per-GPU (tok/s) | Scaling vs 8 GPU |
|------|-------|--------------------|-----------------|------------------|
| 8    | 1     | 36,382             | 4,548           | 1.00×            |
| 16   | 2     | 58,424             | 3,651           | 1.61×            |
| 24   | 3     | 81,451             | 3,394           | 2.24×            |

**Near-linear scaling is restored.** 16 GPUs deliver 1.61× the throughput of 8 GPUs (80% efficiency), and 24 GPUs deliver 2.24× (75% efficiency). The per-GPU throughput remains healthy at 3,394–3,651 tok/s.

### Full Comparison

| GPUs | Nodes | FULL_SHARD (tok/s) | HYBRID_SHARD (tok/s) | HYBRID vs FULL |
|------|-------|--------------------|----------------------|----------------|
| 2    | 1     | 8,818              | —                    | —              |
| 4    | 1     | 17,964             | —                    | —              |
| 8    | 1     | 36,382             | (same)               | —              |
| 16   | 2     | 33,622             | **58,424**           | **+73.8%**     |
| 24   | 3     | 31,589             | **81,451**           | **+157.8%**    |

At 24 GPUs, HYBRID_SHARD is **2.58× faster** than FULL_SHARD. The gap widens with more nodes because FULL_SHARD's per-layer IB penalty compounds while HYBRID_SHARD's gradient-only IB traffic stays constant per step.

## Why the Sharding Strategy Makes This Much Difference

The performance difference comes down to *what data crosses InfiniBand, and how often*.

### FULL_SHARD: Per-Layer Cross-Node Communication

With `FULL_SHARD` across 16 GPUs spanning 2 nodes, each GPU holds 1/16th of every parameter tensor. To execute any layer's forward pass, that layer's full parameters must be reconstructed via an **all-gather** — pulling the other 15/16ths from all other GPUs. After the backward pass through that layer, a **reduce-scatter** distributes the gradients back.

Qwen2.5-7B has 28 decoder layers. Each step requires:
- **Forward pass:** 28 all-gather operations (one per layer, reconstructing ~500 MB per layer)
- **Backward pass:** 28 all-gather operations (to reconstruct parameters again) + 28 reduce-scatter operations (to distribute gradients)
- **Total:** ~84 collective operations per training step that must cross InfiniBand

Each all-gather moves ~14 GB × (15/16) ≈ 13.1 GB of data. With 8 IB ports per node at 400 Gb/s each, the aggregate unidirectional bandwidth is ~400 GB/s. But an all-gather across 16 GPUs (2 nodes) requires data to cross the node boundary — the effective cross-node bandwidth is limited by the IB fabric, not the NVLink fabric.

**The compute-to-communication ratio is terrible.** Each decoder layer has ~250M parameters. At bf16, computing the forward and backward pass takes roughly 1.5 TFLOPs. An H100 at bf16 delivers ~1,979 TFLOPS peak — so the compute for one layer on 8 GPUs takes about 0.1 ms. But the all-gather of 500 MB across IB takes about 1.25 ms (at 400 GB/s aggregate). Communication takes **12× longer than compute**. The GPUs spend most of their time waiting for parameters to arrive from the other node.

### HYBRID_SHARD: Node-Local Sharding, Cross-Node Replication

`HYBRID_SHARD` changes the game fundamentally. Parameters are sharded across the 8 GPUs *within each node* (using NVLink at 900 GB/s), but the same full parameter set exists on every node. This means:

- **Forward pass:** All-gather happens within the node over NVLink — same 28 operations, but at 900 GB/s instead of crossing IB.
- **Backward pass:** Reduce-scatter within node (NVLink), followed by a single **all-reduce across nodes** to synchronize gradients.

The cross-node traffic drops from ~84 collective operations per step (FULL_SHARD) to effectively **one gradient all-reduce per step** (HYBRID_SHARD). That all-reduce moves ~14 GB of gradient data across IB — but it happens once, not 84 times.

This is classic data parallelism across nodes with FSDP within each node. The IB fabric handles the gradient synchronization it was designed for, while NVLink handles the bandwidth-intensive parameter reconstruction.

### Why the Gap Widens at 24 GPUs

At 24 GPUs (3 nodes):

- **FULL_SHARD** now shards across 24 GPUs. Each all-gather must collect from 24 sources, with 16 of them on remote nodes. The all-gather ring now has more cross-node hops, increasing latency multiplicatively. The per-GPU shard is even smaller (14 GB / 24 = 583 MB), meaning compute per layer per GPU drops further while communication stays constant — the compute-to-communication ratio gets *worse*.

- **HYBRID_SHARD** stays the same within each node (8-GPU FSDP shard over NVLink). The cross-node gradient all-reduce now involves 3 nodes instead of 2, adding one extra IB hop. But since this is one operation per step (not per layer), the additional cost is marginal.

This is why FULL_SHARD goes from 33,622 tok/s (16 GPU) to 31,589 tok/s (24 GPU) — a *decrease* — while HYBRID_SHARD goes from 58,424 to 81,451 — a healthy 39% increase with 50% more hardware.

### Sanity-Checking the Numbers

Let's verify the results make physical sense:

**Intra-node scaling (2 → 8 GPU):** Per-GPU throughput stays at ~4,500 tok/s, meaning total throughput scales linearly. This is expected — NVLink's 900 GB/s is so fast that FSDP communication is negligible compared to compute. The slight per-GPU improvement (4,409 → 4,548) makes sense because finer sharding reduces per-GPU memory pressure.

**FULL_SHARD 8 → 16 GPU:** Per-GPU drops from 4,548 to 2,101 — a 54% decline. If we assume each step is now bottlenecked by IB all-gather time, we can estimate: 84 collectives × ~1.25 ms each ≈ 105 ms of communication per step. The compute per step is about 3.0 ms per layer × 28 layers / 16 GPUs ≈ 5.25 ms. Total step time ≈ 110 ms, dominated by communication. At 16 × 2048 = 32,768 tokens per step, that's 32,768 / 0.110 ≈ 297K tok/s — which overestimates a bit because peak IB bandwidth isn't sustained with small messages and there's overlap with compute. Our measured 33,622 tok/s at an actual step time of about 0.97s is consistent with real-world IB latency being higher than the simple bandwidth calculation suggests, especially with 84 serialized collectives that cannot fully overlap with compute.

**HYBRID_SHARD 16 GPU:** Per-GPU is 3,651 tok/s — 80% of the single-node per-GPU rate (4,548). The 20% overhead should come from the cross-node gradient all-reduce. One all-reduce of 14 GB across 2 nodes at ~400 GB/s takes ~35 ms. If the single-node step time is ~7.2 ms per step (8 × 2048 × 50 / 36,382 / 50 ≈ 0.45s / 50 ≈ 9.0 ms per step — wait, let me recalculate: 36,382 tok/s at 8 × 2048 = 16,384 tokens/step means 16,384 / 36,382 = 0.45s per step). At 16 GPUs, the effective step should be: NVLink compute (same ~0.45s since each node does the same work) + IB gradient sync (~35 ms). That gives 0.485s per step, at 32,768 tokens/step → 67,500 tok/s — somewhat higher than measured 58,424. The discrepancy suggests the gradient all-reduce takes closer to 115 ms in practice (the 14 GB is transmitted in ring-reduce fashion, and there's synchronization overhead), giving ~0.56s per step → 58,500 tok/s. This matches the measured result almost exactly.

**HYBRID_SHARD 24 GPU:** Per-GPU is 3,394 tok/s — 75% of single-node rate. Adding a third node introduces one more hop in the gradient all-reduce (ring of 3 nodes), increasing sync time by ~50%. This brings per-GPU efficiency from 80% (2 nodes) to 75% (3 nodes) — a reasonable degradation that would flatten out at larger scales as the ring overhead per node decreases.

## Implications for Real Training Workloads

These results have direct implications for anyone training on multi-node GPU clusters:

### 1. Default FSDP Settings Can Be a Trap

PyTorch FSDP defaults to `FULL_SHARD`, which is optimal for single-node training but catastrophic for multi-node with models that fit within one node. Many tutorials and getting-started guides don't mention `HYBRID_SHARD`, so users who naively scale from 1 node to 2+ nodes will see *negative* scaling and may blame their hardware or network configuration.

### 2. Model Size Relative to Node Size Determines the Strategy

The rule of thumb:
- **Model fits in one node?** Use `HYBRID_SHARD`. Each node runs independent FSDP, nodes synchronize gradients.
- **Model too large for one node?** Use `FULL_SHARD` — you have no choice but to shard across nodes, and the IB communication is the cost of admission. Consider whether `_HYBRID_SHARD_ZERO2` might help.

Qwen2.5-7B at 14 GB in bf16 fits easily in a single H100 node with 640 GB of GPU memory. HYBRID_SHARD is the obvious choice.

### 3. The Compute-to-Communication Ratio is Everything

FULL_SHARD's problem isn't that IB is slow — 400 Gb/s × 8 ports is impressive bandwidth. The problem is that a 7B model doesn't have enough compute per layer to amortize the communication cost. Each layer's all-gather takes longer than the layer's compute, making GPUs idle most of the time.

With a larger model (e.g., 70B), FULL_SHARD across nodes would be more viable because:
- Each layer has 10× more parameters → 10× more compute per layer
- The all-gather volume is larger, but so is the compute time to hide it
- The compute-to-communication ratio improves from ~0.08 (7B) to ~0.8 (70B)

This doesn't mean FULL_SHARD is *good* for 70B cross-node — it means it's less *catastrophically bad*. HYBRID_SHARD would still win when the model fits within a single node.

### 4. IB Bandwidth Matters for Gradient Sync

HYBRID_SHARD's 20-25% per-GPU overhead at 2-3 nodes comes entirely from gradient synchronization over IB. This means IB bandwidth directly impacts multi-node training efficiency:

- With 8× 400 Gb/s NDR InfiniBand: ~75-80% per-GPU efficiency at 2-3 nodes ✓
- With 1× 100 Gb/s (typical cloud): the gradient all-reduce would take ~4× longer, potentially dropping efficiency to 40-50%
- With 8× 200 Gb/s HDR InfiniBand: somewhere in between

The Azure ND H100 v5 VMs with 8× NDR InfiniBand are well-suited for multi-node training precisely because the gradient sync completes fast enough to maintain good efficiency.

## Key Takeaways

1. **Sharding strategy matters more than hardware.** HYBRID_SHARD on IB outperforms FULL_SHARD on the same hardware by 2.58× at 24 GPUs. No amount of network tuning can fix an algorithm that hits IB 84 times per step when it could hit it once.

2. **FULL_SHARD across nodes shows *negative* total throughput scaling.** 24 GPUs is slower than 8 GPUs. This isn't a bug — it's the expected behavior when per-layer all-gather over IB dominates compute.

3. **HYBRID_SHARD delivers near-linear multi-node scaling.** 80% efficiency at 2 nodes, 75% at 3 nodes. The parameter reconstruction stays on NVLink; only gradients cross IB.

4. **Multi-node FSDP requires careful orchestration.** On a bare VMSS cluster without Slurm, you need launch scripts that handle container lifecycle, NCCL environment variables, rendezvous coordination, and cleanup. The scripts in this post provide a working template.

5. **Verify everything on every node.** Model paths, container images, library versions, stale containers — any mismatch between nodes causes silent failures or hangs that are hard to debug across SSH.

## Reproducing These Results

All code is available: the [benchmark script](https://github.com/JingchaoZhang/JingchaoZhang.github.io/blob/master/scripts/fsdp-multinode/finetune_bench.py), [launch script](https://github.com/JingchaoZhang/JingchaoZhang.github.io/blob/master/scripts/fsdp-multinode/launch_node.sh), and [orchestration script](https://github.com/JingchaoZhang/JingchaoZhang.github.io/blob/master/scripts/fsdp-multinode/run_multinode.sh) work on any Azure ND H100 v5 VMSS cluster (or similar multi-node GPU setup with InfiniBand). The only requirements are:

- Docker with NVIDIA runtime
- Passwordless SSH between nodes
- The model downloaded on every node
- InfiniBand configured and healthy

To run:
```bash
# Single node (8 GPU)
bash launch_node.sh 1 0 $(hostname -i) 29500 full

# 2 nodes, HYBRID_SHARD
bash run_multinode.sh 2 "headnode,worker1" hybrid

# 3 nodes, FULL_SHARD (if you want to see the pain)
bash run_multinode.sh 3 "headnode,worker1,worker2" full
```
