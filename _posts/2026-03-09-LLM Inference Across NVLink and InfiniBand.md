---
layout: single
title: "LLM Inference Across NVLink and InfiniBand"
author_profile: false
---

## Introduction

When serving a large language model across multiple GPUs, the choice of parallelism strategy directly determines *which hardware interconnect carries your data*. Tensor parallelism (TP) triggers all-reduce collectives — bandwidth-hungry operations that favor NVLink's 900 GB/s bisection bandwidth within a node. Pipeline parallelism (PP) uses point-to-point transfers between pipeline stages — latency-sensitive operations that must cross InfiniBand when stages span nodes. And when you scale tensor parallelism beyond a single node, every all-reduce must traverse both NVLink *and* InfiniBand, making the slower link the bottleneck.

In this post, I run **Qwen2.5-72B** inference on 2 Azure H100 nodes using [vLLM](https://github.com/vllm-project/vllm), systematically varying the parallelism configuration to isolate each communication pattern. (We chose Qwen2.5-72B as a representative dense transformer; the communication patterns are architecture-independent, so similar results are expected for other models of comparable size such as Llama 2 70B.)

| Config | TP | PP | Nodes | Communication Pattern |
|--------|----|----|-------|-----------------------|
| A | 4 | 1 | 1 | All-reduce over **NVLink** (half-node baseline) |
| B | 8 | 1 | 1 | All-reduce over **NVLink only** |
| C | 8 | 2 | 2 | NVLink intra-node + **IB point-to-point** (pipeline) |
| D | 16 | 1 | 2 | All-reduce over **NVLink + IB** (mixed) |

The goal is to show, with real numbers, how each strategy maps to hardware and what the performance implications are.

## Test Environment

| Component | Detail |
|-----------|--------|
| **VM SKU** | Standard_ND96isr_H100_v5 |
| **GPUs per node** | 8x NVIDIA H100 80GB HBM3 |
| **Inter-node** | 8x 400 Gb/s NDR InfiniBand (ConnectX-7) |
| **Intra-node** | NVLink 4th gen, 900 GB/s bisection |
| **Nodes** | 2 (16 GPUs total) |
| **OS** | Ubuntu 22.04, kernel 5.15.0-1103-azure |
| **NCCL** | 2.28.9+cuda13.0 |
| **MPI** | HPC-X (OpenMPI-based) |
| **Topology** | `/opt/microsoft/ndv5-topo.xml` |
| **Model** | Qwen2.5-72B (FP16, 140 GB) |
| **Serving** | vLLM |

The two test nodes were selected from a 3-node VMSS deployment in Azure East US. All three nodes passed [Azure Node Health Checks (NHC)](https://github.com/Azure/azurehpc-health-checks) with flying colors — we picked two for the experiments and kept the third as a spare.

## Node Health Validation

Before running any workload, every node was validated with Azure NHC:

```bash
parallel-ssh -i -t 0 -h hostfile \
  "sudo /opt/azurehpc/run-health-checks.sh" &> all_nodes_NHC.log
```

**Result: 3/3 nodes passed all checks.**

| Check | vmssE6JHE7 | vmssAYZGM2 | vmssCZIUQ2 |
|-------|-----------|-----------|-----------|
| GPU count (8) | PASS | PASS | PASS |
| GPU ECC | PASS | PASS | PASS |
| NVLink (all links active) | PASS | PASS | PASS |
| NCCL all-reduce | 482.5 GB/s | 482.6 GB/s | 482.5 GB/s |
| IB bandwidth (per port) | ~392.7 Gb/s | ~392.8 Gb/s | ~392.7 Gb/s |
| IB link flapping | PASS | PASS | PASS |

All IB ports measured within a tight band of 392.5–393.2 Gb/s (threshold: 380 Gb/s). Intra-node NCCL all-reduce hit ~482.5 GB/s — near the theoretical maximum for 8 H100s over NVLink. GPU clocks were locked at 1980 MHz (`nvidia-smi -lgc 1980 --mode 1`) to eliminate frequency scaling variability.

## NCCL Baseline: 2-Node All-Reduce

Before touching any LLM, we established a communication baseline with `nccl-tests` across both nodes (16 GPUs, 8GB message):

```bash
mpirun -np $(( SCALE * DEVICES )) \
  --map-by ppr:8:node \
  -hostfile $HOSTFILE \
  -mca plm_rsh_no_tree_spawn 1 \
  -mca plm_rsh_num_concurrent 800 \
  -mca coll_hcoll_enable 0 \
  -x LD_LIBRARY_PATH \
  -x CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -x UCX_TLS=rc \
  -x UCX_NET_DEVICES=mlx5_ib0:1 \
  -x NCCL_SOCKET_IFNAME=eth0 \
  -x NCCL_DEBUG=WARN \
  -x NCCL_MIN_NCHANNELS=32 \
  -x NCCL_IB_QPS_PER_CONNECTION=4 \
  -x NCCL_P2P_NET_CHUNKSIZE=$((512*1024)) \
  -x NCCL_PXN_DISABLE=1 \
  -x NCCL_TOPO_FILE=/opt/microsoft/ndv5-topo.xml \
  -x NCCL_IGNORE_CPU_AFFINITY=1 \
  /opt/nccl-tests/build/all_reduce_perf -i 0 -g 1 -t 1 -b 8G -e 8G -f 0 -R 1
```

**Result: ~483 GB/s bus bandwidth (in-place, steady state)**

| Metric | Value |
|--------|-------|
| Message size | 8 GB |
| Bus bandwidth (in-place, median) | **483.3 GB/s** |
| Bus bandwidth (out-of-place, median) | **484.1 GB/s** |
| Outlier range | 346–399 GB/s (occasional, out-of-place only) |

This tells us the NVLink + IB fabric is healthy and performing at expected levels. The occasional out-of-place dips are typical NCCL warm-up artifacts and don't indicate a problem.

## Setting Up the LLM Workload

We use **Qwen2.5-72B** (`Qwen/Qwen2.5-72B`) as our benchmark model — a 72-billion parameter dense transformer, open-access with no gating, and ~140 GB in FP16. It does not fit in a single H100's 80 GB memory, which makes multi-GPU parallelism a practical necessity rather than an artificial benchmark setup.

### Container and Model Setup

vLLM v0.17.0 ships as a ready-to-run Docker image with CUDA, PyTorch, FlashAttention 3, and NCCL 2.27.5 baked in:

```bash
# Pull vLLM container on both nodes
parallel-ssh -i -h hostfile_2nodes -t 0 \
  "docker pull vllm/vllm-openai:latest"

# Download model via HuggingFace CLI (head node)
hf download Qwen/Qwen2.5-72B --local-dir /home/azureuser/models/Qwen2.5-72B

# Rsync model to second node
rsync -avP /home/azureuser/models/ vmssAYZGM2:/home/azureuser/models/
```

The model is ~140 GB in FP16 safetensor format (37 shards).

## Experiment Results

### Config A: TP=4, Single Node (Half-Node Baseline)

This configuration uses only 4 of the 8 GPUs on a single node, splitting the 72B model across GPUs 0–3 via tensor parallelism. All communication stays within the NVLink domain — this serves as our **lower bound** for what partial-node serving looks like.

```bash
sudo docker run --gpus '"device=0,1,2,3"' \
  --ipc=host --network=host \
  -v /home/azureuser/models:/models \
  vllm/vllm-openai:latest \
  --model /models/Qwen2.5-72B \
  --tensor-parallel-size 4 \
  --dtype float16 --max-model-len 2048
```

**Startup observations:**
- Model loading: **31.3 seconds** (37 safetensor shards)
- Memory per shard: **34.01 GiB** across 4 GPUs (~8.5 GiB/GPU for weights)
- Available KV cache: **32.87 GiB** → 430,800 tokens capacity
- FlashAttention v3 selected, CUDA graphs captured for both prefill and decode

**Results (Qwen2.5-72B, FP16):**

| Metric | Value |
|--------|-------|
| Throughput | **60.8 tok/s** |
| Avg Latency (512 tokens) | **8.42s** |
| Avg TTFT (Time To First Token) | **25 ms** |
| Consistency (5 runs) | ±0.0 tok/s |

The throughput is rock-solid at 60.8 tok/s with zero variance across 5 runs — a sign that the workload is compute-bound (not communication-bound) at this scale. The 25 ms TTFT is excellent: the prompt is short (57 tokens) and the prefill fits comfortably in GPU memory.

### Config B: TP=8, Single Node (Full NVLink)

This is the natural deployment for a 72B model on a single H100 node — all 8 GPUs participate in tensor parallelism, and all all-reduce traffic stays on NVLink.

```bash
sudo docker run --gpus all \
  --ipc=host --network=host \
  -v /home/azureuser/models:/models \
  vllm/vllm-openai:latest \
  --model /models/Qwen2.5-72B \
  --tensor-parallel-size 8 \
  --dtype float16 --max-model-len 2048
```

**Startup observations:**
- torch.compile took **371s** (vs 40s for TP=4) — first-run penalty for new graph shapes with 8-way sharding. Subsequent launches reuse the AOT cache.
- Available KV cache: **49.6 GiB** → 1,300,336 tokens (3x more than TP=4, since each GPU holds a smaller model shard)
- CUDA graph capture: ~6 minutes total (8 GPUs need more synchronization during capture)

**Results (Qwen2.5-72B, FP16):**

| Metric | Value |
|--------|-------|
| Throughput | **99.5 tok/s** |
| Avg Latency (512 tokens) | **5.14s** |
| Avg TTFT | **19 ms** |
| Consistency (5 runs) | ±0.0 tok/s |

Doubling the GPU count from 4 to 8 yields a **1.64x throughput gain** (60.8 → 99.5 tok/s). This is solid intra-node scaling — NVLink's 900 GB/s bisection bandwidth keeps the all-reduce overhead low. TTFT also improved from 25ms to 19ms, reflecting faster prefill with more parallel compute.

### Config C: TP=8 + PP=2, Two Nodes (NVLink + IB Pipeline)

This is our first cross-node configuration. Each node runs TP=8 over NVLink (identical to Config B), and the two nodes form a 2-stage pipeline connected via InfiniBand. Activations are sent point-to-point between pipeline stages — small messages, but latency-sensitive.

This required a Ray cluster spanning both nodes:

```bash
# Head node (10.0.0.4)
sudo docker run -d --gpus all --ipc=host --network=host \
  --name vllm-head --entrypoint bash \
  -e RAY_health_check_period_ms=30000 \
  -e RAY_health_check_timeout_ms=120000 \
  -v /home/azureuser/models:/models \
  vllm/vllm-openai:latest \
  -c "ray start --head --port=6379 && sleep infinity"

# Worker node (10.0.0.5)
sudo docker run -d --gpus all --ipc=host --network=host \
  --name vllm-worker --entrypoint bash \
  -e RAY_health_check_period_ms=30000 \
  -e RAY_health_check_timeout_ms=120000 \
  -v /home/azureuser/models:/models \
  vllm/vllm-openai:latest \
  -c "ray start --address=10.0.0.4:6379 && sleep infinity"

# Launch vLLM server on the head container
sudo docker exec vllm-head bash -c "python3 -m vllm.entrypoints.openai.api_server \
  --model /models/Qwen2.5-72B \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 2 \
  --dtype float16 --max-model-len 2048 --port 8000 \
  --distributed-executor-backend ray"
```

The increased Ray health check timeouts are critical — CUDA graph capture across 16 GPUs on 2 nodes takes long enough that the default 10-second timeout would kill the worker before compilation finishes.

**Results (Qwen2.5-72B, FP16):**

| Metric | Value |
|--------|-------|
| Throughput | **67.5 tok/s** |
| Avg Latency (512 tokens) | **7.59s** |
| Avg TTFT | **25.8 ms** |
| Consistency (5 runs) | ±0.1 tok/s |

Compared to Config B (99.5 tok/s on a single node), adding a second node with pipeline parallelism **reduces throughput by 32%**. The overhead comes from the sequential nature of pipeline stages during autoregressive decoding: each token must complete a full forward pass through stage 1 (40 layers on node 1), transfer activations over IB, then complete stage 2 (40 layers on node 2) before the next token can begin. At any given moment, only one stage is actively computing — the other is idle waiting. The activation tensors sent over IB are small, but the IB round-trip latency (~7 µs per transfer) accumulates over hundreds of generated tokens. TTFT increased from 19ms to 25.8ms because the prefill must also traverse both pipeline stages sequentially.

### Config D: TP=16, Two Nodes (NVLink + IB All-Reduce)

This config extends tensor parallelism across both nodes — all 16 GPUs form a single TP group. Every all-reduce operation now has a cross-node component that must traverse InfiniBand.

We reused the same Ray cluster (after a fresh restart to free the placement group from Config C):

```bash
sudo docker exec vllm-head bash -c "python3 -m vllm.entrypoints.openai.api_server \
  --model /models/Qwen2.5-72B \
  --tensor-parallel-size 16 \
  --pipeline-parallel-size 1 \
  --dtype float16 --max-model-len 2048 --port 8000 \
  --distributed-executor-backend ray"
```

**Startup observations:**
- vLLM disabled async scheduling (not supported with Ray backend)
- `fuse_allreduce_rms: True` — the compiler attempts to fuse all-reduce with RMSNorm to hide communication latency
- CUDA graph capture completed successfully across all 16 GPUs (~8s per graph, 51 capture sizes)

**Results (Qwen2.5-72B, FP16):**

| Metric | Value |
|--------|-------|
| Throughput | **34.3 tok/s** |
| Avg Latency (512 tokens) | **14.93s** |
| Avg TTFT | **58.5 ms** |
| Consistency (5 runs) | ±0.4 tok/s |

This is the slowest configuration despite using the most GPUs. Compared to Config B (99.5 tok/s with 8 GPUs), Config D delivers only **34.5% of the throughput while using 2x the hardware**. TTFT tripled from 19ms to 58.5ms — every prefill token must wait for cross-node all-reduce at each of the 80 transformer layers. The per-layer IB all-reduce tax is catastrophic: Qwen2.5-72B has 80 layers, and each layer requires at least 2 all-reduce operations (after attention and after MLP), meaning ~160 cross-node collective operations per token.

## Comparison

| Config | Parallelism | Nodes | GPUs | Interconnect | Throughput | Latency (512 tok) | TTFT |
|--------|------------|-------|------|--------------|------------|-------------------|------|
| A | TP=4 | 1 | 4 | NVLink only | **60.8 tok/s** | 8.42s | 25 ms |
| B | TP=8 | 1 | 8 | NVLink only | **99.5 tok/s** | 5.14s | 19 ms |
| C | TP=8, PP=2 | 2 | 16 | NVLink + IB (pipeline) | **67.5 tok/s** | 7.59s | 25.8 ms |
| D | TP=16 | 2 | 16 | NVLink + IB (all-reduce) | **34.3 tok/s** | 14.93s | 58.5 ms |

## Analysis

### Why Config B is the fastest

Config B (TP=8, single node) achieves the highest throughput at **99.5 tok/s** despite using only 8 GPUs. The reason is simple: all inter-GPU communication stays on NVLink, which provides 900 GB/s bisection bandwidth. Tensor parallelism shards every linear layer across GPUs and requires an all-reduce after each layer — but when all 8 GPUs are connected via NVLink, this all-reduce completes in microseconds.

The 1.64x scaling from TP=4 (Config A, 60.8 tok/s) to TP=8 (Config B, 99.5 tok/s) shows that the workload is still compute-bound at this scale — doubling the GPU count nearly doubles throughput, meaning communication overhead is negligible.

### Why Config C is slower than a single node

Config C uses the same TP=8 per node as Config B, but adds a second pipeline stage on a second node. With 2x the GPUs (16 total), you might expect 2x the throughput. Instead, it delivers only **67.5 tok/s** — 32% *slower* than the 8-GPU single-node Config B.

The culprit is **sequential stage execution**. In a 2-stage pipeline serving a single request, the stages cannot overlap: stage 1 (layers 1–40 on node 1) must finish before stage 2 (layers 41–80 on node 2) can begin. During autoregressive decoding, each token requires a full sequential pass through both stages — stage 1 computes, sends activations over IB, then stage 2 computes and produces the next token logit. At any instant, only one stage is actively working while the other waits. This is distinct from the "pipeline bubble" in training, where microbatch scheduling can partially fill the idle time; with a single decode request generating one token at a time, there is nothing to overlap.

The math tells the story: Config B processes tokens at 99.5 tok/s using 8 GPUs that are all active on every layer. Config C has 16 GPUs but only ~8 are computing at any moment (the other 8 wait for their stage), and each stage transition adds an IB round-trip. The result — 67.5 tok/s — reflects the cost of serialized pipeline stages plus per-token IB latency.

TTFT increased from 19ms to 25.8ms because the prefill prompt must now flow through both pipeline stages sequentially. The additional ~7ms is the IB round-trip latency for activation transfers.

### Why Config D is the slowest — the per-layer IB tax

Config D (TP=16, 2 nodes) was the biggest surprise: **34.3 tok/s with 16 GPUs** — worse than TP=4 on just 4 GPUs (60.8 tok/s), and a staggering 2.9x slower than Config B's 99.5 tok/s on 8 GPUs.

The root cause is the **frequency of cross-node communication**. Unlike pipeline parallelism (Config C), which only crosses IB at pipeline stage boundaries, cross-node tensor parallelism forces every all-reduce to traverse InfiniBand. Qwen2.5-72B has 80 transformer layers, and each layer performs at least 2 all-reduce operations (post-attention and post-MLP). That's ~160 cross-node collectives per token generated.

Each all-reduce follows a hierarchical pattern: first reduce within the 8 NVLink-connected GPUs on each node, then exchange partial results across IB, then broadcast back. While the intra-node NVLink portion completes in microseconds, the IB portion is bounded by ~50 GB/s effective bandwidth — roughly 18x slower than NVLink's 900 GB/s. Multiply that per-layer penalty by 80 layers and the overhead becomes dominant.

The TTFT impact is equally telling: 58.5ms vs 19ms for Config B — a 3x increase. The prefill (which processes all prompt tokens at once) should benefit from more parallel compute, but the per-layer IB synchronization completely negates the compute advantage of additional GPUs.

**The lesson: Config C (PP across IB) loses to sequential stage execution (only half the GPUs active at a time), but Config D (TP across IB) loses worse because it hits the IB bottleneck on every single layer, not just at stage boundaries.**

## Key Takeaways

1. **NVLink is the performance boundary.** The single biggest takeaway: TP=8 on one node (99.5 tok/s) outperforms every 2-node configuration, despite using half the GPUs. When your model fits within a single NVLink domain, adding more nodes *hurts* rather than helps for single-request latency.

2. **Pipeline parallelism pays a steep serialization tax for single requests.** PP=2 across IB reduced throughput by 32% compared to the single-node baseline. With only one request decoding at a time, the two pipeline stages execute sequentially — only half the GPUs are active at any moment. This is an algorithmic penalty inherent to serving single requests through a pipeline, not a hardware limitation. PP makes sense for models that *don't fit* in one node's memory, or for high-throughput batch serving where multiple requests can fill the pipeline — not for single-request latency optimization.

3. **Cross-node TP is worse than pipeline parallelism.** This was the most counterintuitive result. Config D (TP=16) was 2x slower than Config C (PP=2) and 2.9x slower than Config B — despite all 16 GPUs actively computing every token. The per-layer IB all-reduce (160 collectives/token across 80 layers) overwhelms the compute benefit. Pipeline parallelism at least confines IB traffic to stage boundaries.

4. **For models that fit in one node, stay in one node.** The Qwen2.5-72B model (140 GB in FP16) fits across 8x H100 80GB GPUs. The right answer for latency-optimized serving is TP=8 on a single node — full stop. Both 16-GPU configs were slower than the 8-GPU config.

5. **More GPUs ≠ more speed.** Config B (8 GPUs) → 99.5 tok/s. Config C (16 GPUs) → 67.5 tok/s. Config D (16 GPUs) → 34.3 tok/s. Doubling the GPU count while crossing the NVLink boundary made things *worse*. Multi-node scaling adds value for throughput at the *system* level (serving more concurrent requests), not for single-request latency.
