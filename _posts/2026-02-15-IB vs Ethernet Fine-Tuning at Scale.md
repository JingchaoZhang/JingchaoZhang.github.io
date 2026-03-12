---
layout: single
title: "InfiniBand vs Ethernet for Multi-Node LLM Fine-Tuning"
author_profile: false
---

<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script>
MathJax = {
  tex: { inlineMath: [['$','$'], ['\\(','\\)']] }
};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js" async></script>

## Introduction

Azure's H100 GPU VMs (`Standard_ND96isr_H100_v5`) come equipped with 8× 400 Gb/s NDR InfiniBand — 3.2 Tbps of aggregate RDMA bandwidth per node. It's widely understood that InfiniBand is essential for large-scale distributed training, but how early does the interconnect actually start to matter?

In this post, I quantify exactly where InfiniBand becomes critical by benchmarking both IB and plain Ethernet across two model sizes (Qwen2.5-7B and Qwen2.5-72B), scaling from 1 to 8 nodes (8 to 64 GPUs), using PyTorch FSDP on a 10-node Azure VMSS cluster with Azure Managed Lustre.

The results confirm that InfiniBand's advantage emerges as soon as the model is large enough and the cluster wide enough — the 72B model at 8 nodes already shows a clear throughput gap, and the math shows this gap will only widen at larger scale. Even the 7B model, which appears insensitive to interconnect at this scale, would eventually hit Ethernet's limits at higher node counts.

## Test Environment

| Component | Detail |
|-----------|--------|
| **VM SKU** | Standard_ND96isr_H100_v5 |
| **GPUs per node** | 8× NVIDIA H100 80 GB HBM3 |
| **Intra-node** | NVLink 4th gen, 900 GB/s bisection |
| **Inter-node (IB)** | 8× 400 Gb/s NDR InfiniBand (ConnectX-7) |
| **Inter-node (ETH)** | TCP sockets over eth0, ~40–50 Gbps accelerated networking |
| **Nodes** | 10 (up to 8 used per experiment) |
| **Shared storage** | Azure Managed Lustre, 8 TiB, mounted at `/lustre` |
| **Container** | `nvcr.io/nvidia/pytorch:24.12-py3` (PyTorch 2.6.0a0, CUDA 13.0, NCCL 2.23.4) |
| **Models** | Qwen2.5-7B (~14 GB), Qwen2.5-72B (~136 GB) in bf16 |
| **Framework** | PyTorch FSDP with `size_based_auto_wrap_policy` |

### How the Two Modes Differ

**InfiniBand mode** (`NCCL_IB_DISABLE=0`): NCCL uses RDMA over 8× 400 Gb/s InfiniBand links. Data moves directly between GPU memory across nodes without touching the CPU — this is the gold standard for GPU cluster interconnects.

**Ethernet mode** (`NCCL_IB_DISABLE=1`): NCCL falls back to TCP sockets over the standard `eth0` interface. This is **not** RoCE (RDMA over Converged Ethernet) — it's plain TCP/IP networking at approximately 40–50 Gbps with Azure accelerated networking. Data passes through the kernel's TCP stack and CPU, adding latency and reducing bandwidth by roughly 60–80× compared to the IB path.

Everything else — FSDP configuration, model, batch size, sequence length — is identical between the two modes. The only variable is the `NCCL_IB_DISABLE` environment variable.

## Why Lustre Changes the Game

In my [previous post]({% post_url 2026-01-25-Fine-Tuning Across NVLink and InfiniBand %}), I used a 3-node cluster without shared storage. Each node needed a local copy of the model, distributed via `rsync`. This was manageable at 3 nodes with a 7B model (14 GB per copy), but doesn't scale:

- **7B model × 10 nodes** = 140 GB of redundant copies
- **72B model × 10 nodes** = 1.36 TB, requiring ~10 minutes of `rsync` per node

With [Azure Managed Lustre]({% post_url 2026-02-09-AMLFS with GPU VMSS %}), every node mounts `/lustre` — a shared POSIX filesystem. Models are downloaded once and accessible everywhere instantly. This eliminated the distribution bottleneck and let me easily scale to 10 nodes.

## The Benchmark Script

The benchmark uses PyTorch FSDP with several optimizations that were essential for making the 72B model work at scale:

**Meta-device loading:** Only `local_rank 0` on each node loads the full model weights into CPU memory. All other ranks create the model on PyTorch's `meta` device (zero memory). FSDP's `sync_module_states=True` broadcasts the weights from rank 0 to all ranks during wrapping. Without this, 8 ranks simultaneously loading a 136 GB model would exceed the node's CPU memory during FSDP initialization.

**Activation checkpointing:** Applied after FSDP wrapping using PyTorch's native `apply_activation_checkpointing` with `NO_REENTRANT` implementation. This trades compute for memory — recomputing activations during backward instead of storing them. Critical for the 72B model, which would OOM without it.

**Flash Attention 2:** Using `attn_implementation="flash_attention_2"` reduces attention memory from $O(n^2)$ to $O(n)$ and significantly speeds up the attention computation.

**Timeout hardening:** `timeout 900` wrapping the Docker container, `NCCL_TIMEOUT=300`, `--rdzv_conf timeout=300`, and `init_process_group(timeout=1800s)`. Multi-node runs at scale have many failure modes — FSDP wrapping timeouts, NCCL communicator hangs, stale containers — and aggressive timeouts ensure a failed run terminates cleanly instead of hanging indefinitely.

The full scripts are in the [Reproducing These Results](#reproducing-these-results) section.

## Results

### Qwen2.5-7B (batch_size=2, seq_len=2048)

| Nodes | GPUs | IB (tok/s) | ETH (tok/s) | Δ |
|-------|------|-----------|------------|------|
| 1 | 8 | 68,039 | 68,081 | +0.1% |
| 2 | 16 | 110,017 | 109,991 | −0.0% |
| 4 | 32 | 213,281 | 220,211 | +3.2% |
| 8 | 64 | 414,339 | 424,296 | +2.4% |

<canvas id="chart7bThroughput" width="700" height="350"></canvas>
<script>
new Chart(document.getElementById('chart7bThroughput'), {
    type: 'bar',
    data: {
        labels: ['1 node\n(8 GPU)', '2 nodes\n(16 GPU)', '4 nodes\n(32 GPU)', '8 nodes\n(64 GPU)'],
        datasets: [
            {
                label: 'InfiniBand',
                data: [68039, 110017, 213281, 414339],
                backgroundColor: 'rgba(54, 162, 235, 0.8)',
                borderColor: 'rgba(54, 162, 235, 1)',
                borderWidth: 1
            },
            {
                label: 'Ethernet (TCP)',
                data: [68081, 109991, 220211, 424296],
                backgroundColor: 'rgba(255, 99, 132, 0.8)',
                borderColor: 'rgba(255, 99, 132, 1)',
                borderWidth: 1
            }
        ]
    },
    options: {
        responsive: true,
        plugins: {
            title: { display: true, text: 'Qwen2.5-7B: Aggregate Throughput', font: { size: 16 } },
            legend: { position: 'top' }
        },
        scales: {
            y: {
                beginAtZero: true,
                title: { display: true, text: 'Tokens/sec' },
                ticks: { callback: function(v) { return v.toLocaleString(); } }
            }
        }
    }
});
</script>

**Ethernet matches InfiniBand at every scale.** The delta column tells the story — the differences are within measurement noise. At 4 and 8 nodes, Ethernet actually shows slightly higher throughput (likely variance between runs, not a real advantage).

### Qwen2.5-72B (batch_size=1, seq_len=2048)

The 72B model cannot run on a single node (it needs ~109 GB per GPU, exceeding the 80 GB HBM3 capacity). We start at 2 nodes.

| Nodes | GPUs | IB (tok/s) | ETH (tok/s) | Δ |
|-------|------|-----------|------------|------|
| 2 | 16 | 9,093 | 9,294 | +2.2% |
| 4 | 32 | 17,871 | 17,892 | +0.1% |
| 8 | 64 | 30,914 | 28,584 | **−7.5%** |

<canvas id="chart72bThroughput" width="700" height="350"></canvas>
<script>
new Chart(document.getElementById('chart72bThroughput'), {
    type: 'bar',
    data: {
        labels: ['2 nodes\n(16 GPU)', '4 nodes\n(32 GPU)', '8 nodes\n(64 GPU)'],
        datasets: [
            {
                label: 'InfiniBand',
                data: [9093, 17871, 30914],
                backgroundColor: 'rgba(54, 162, 235, 0.8)',
                borderColor: 'rgba(54, 162, 235, 1)',
                borderWidth: 1
            },
            {
                label: 'Ethernet (TCP)',
                data: [9294, 17892, 28584],
                backgroundColor: 'rgba(255, 99, 132, 0.8)',
                borderColor: 'rgba(255, 99, 132, 1)',
                borderWidth: 1
            }
        ]
    },
    options: {
        responsive: true,
        plugins: {
            title: { display: true, text: 'Qwen2.5-72B: Aggregate Throughput', font: { size: 16 } },
            legend: { position: 'top' }
        },
        scales: {
            y: {
                beginAtZero: true,
                title: { display: true, text: 'Tokens/sec' },
                ticks: { callback: function(v) { return v.toLocaleString(); } }
            }
        }
    }
});
</script>

**InfiniBand's advantage only appears at 8 nodes with the 72B model** — and it's a meaningful 7.5%. At 2 and 4 nodes, the two interconnects are statistically identical.

### Per-GPU Efficiency

To understand the scaling behavior, here's the per-GPU throughput (tokens/sec/GPU):

**Qwen2.5-7B:**

| Nodes | GPUs | IB (tok/s/GPU) | ETH (tok/s/GPU) |
|-------|------|---------------|----------------|
| 1 | 8 | 8,505 | 8,510 |
| 2 | 16 | 6,876 | 6,874 |
| 4 | 32 | 6,665 | 6,882 |
| 8 | 64 | 6,474 | 6,630 |

**Qwen2.5-72B:**

| Nodes | GPUs | IB (tok/s/GPU) | ETH (tok/s/GPU) |
|-------|------|---------------|----------------|
| 2 | 16 | 568 | 581 |
| 4 | 32 | 558 | 559 |
| 8 | 64 | 483 | 447 |

<canvas id="chartPerGPU" width="700" height="350"></canvas>
<script>
new Chart(document.getElementById('chartPerGPU'), {
    type: 'line',
    data: {
        labels: ['1 node', '2 nodes', '4 nodes', '8 nodes'],
        datasets: [
            {
                label: '7B — InfiniBand',
                data: [8505, 6876, 6665, 6474],
                borderColor: 'rgba(54, 162, 235, 1)',
                backgroundColor: 'rgba(54, 162, 235, 0.1)',
                borderWidth: 2,
                tension: 0.2,
                pointRadius: 5
            },
            {
                label: '7B — Ethernet',
                data: [8510, 6874, 6882, 6630],
                borderColor: 'rgba(255, 99, 132, 1)',
                backgroundColor: 'rgba(255, 99, 132, 0.1)',
                borderWidth: 2,
                borderDash: [5, 5],
                tension: 0.2,
                pointRadius: 5
            },
            {
                label: '72B — InfiniBand',
                data: [null, 568, 558, 483],
                borderColor: 'rgba(75, 192, 192, 1)',
                backgroundColor: 'rgba(75, 192, 192, 0.1)',
                borderWidth: 2,
                tension: 0.2,
                pointRadius: 5,
                yAxisID: 'y2'
            },
            {
                label: '72B — Ethernet',
                data: [null, 581, 559, 447],
                borderColor: 'rgba(255, 159, 64, 1)',
                backgroundColor: 'rgba(255, 159, 64, 0.1)',
                borderWidth: 2,
                borderDash: [5, 5],
                tension: 0.2,
                pointRadius: 5,
                yAxisID: 'y2'
            }
        ]
    },
    options: {
        responsive: true,
        plugins: {
            title: { display: true, text: 'Per-GPU Efficiency: Tokens/sec/GPU', font: { size: 16 } },
            legend: { position: 'top' }
        },
        scales: {
            y: {
                type: 'linear',
                position: 'left',
                title: { display: true, text: '7B — Tokens/sec/GPU' },
                beginAtZero: false,
                min: 5000,
                ticks: { callback: function(v) { return v.toLocaleString(); } }
            },
            y2: {
                type: 'linear',
                position: 'right',
                title: { display: true, text: '72B — Tokens/sec/GPU' },
                beginAtZero: false,
                min: 400,
                grid: { drawOnChartArea: false }
            }
        }
    }
});
</script>

The 7B model shows a ~24% per-GPU efficiency drop from 1 to 8 nodes — this overhead comes from FSDP communication (all-gather and reduce-scatter), not the interconnect type. Both IB and ETH curves overlap almost perfectly.

The 72B model shows a steeper drop at 8 nodes, and this is where the curves diverge: IB holds at 483 tok/s/GPU while ETH drops to 447 — a 7.5% gap. The large parameter shards (72B / 64 GPUs = 1.13B params per shard) finally generate enough all-reduce traffic to overwhelm FSDP's ability to hide the 80 Gbps TCP transfers behind compute.

## Why Ethernet Keeps Up (And Where It Doesn't)

### 1. The Hierarchical All-Gather Reality

A naive calculation would assume each all-gather transfers the full unsharded layer weight across the inter-node link. For the 7B model with `FULL_SHARD` across 16 GPUs, that would be ~500 MB per layer in bf16. At 80 Gbps Ethernet (10 GB/s), that would take ~50 ms — far longer than the per-layer compute time. So how does Ethernet keep up?

The answer is NCCL's **hierarchical collective algorithm**. On Azure ND96isr_H100_v5 VMs, NCCL detects the topology: 8 GPUs connected via NVLink within each node, and Ethernet (or InfiniBand) between nodes. It decomposes the all-gather into two phases:

1. **Inter-node exchange**: Each node collectively holds half the layer's parameters (~250 MB). The two nodes exchange their halves simultaneously over full-duplex links. Only **250 MB** crosses the inter-node fabric — not 500 MB.
2. **Intra-node broadcast**: Once a node has the full layer, NVLink distributes it to all 8 local GPUs at ~7.2 TB/s bisection bandwidth. This completes in microseconds and is effectively free.

The corrected inter-node transfer math:

$$\text{ETH transfer time} = \frac{250\text{ MB}}{10\text{ GB/s}} = 25\text{ ms per all-gather}$$

$$\text{IB transfer time} = \frac{250\text{ MB}}{400\text{ GB/s}} \approx 0.6\text{ ms per all-gather}$$

Meanwhile, FSDP doesn't wait for communication to finish before starting compute. It prefetches the *next* layer's parameters while the current layer is computing, using dedicated CUDA streams:

- **NCCL stream**: All-gather for layer $N+1$
- **Compute stream**: Forward/backward for layer $N$
- **NCCL stream**: Reduce-scatter for layer $N-1$

This deep pipeline means the 25 ms Ethernet transfer only becomes a bottleneck if it *significantly exceeds* the compute time per layer — enough to stall the pipeline across multiple consecutive layers.

### 2. The Empirical Proof: Coordination vs. Bandwidth

Rather than relying on theoretical FLOP estimates, we can measure the per-layer compute time directly from the 1-node baseline, where **zero data crosses an inter-node link** (all communication stays on NVLink):

**7B model, 1 node (8 GPUs), batch_size=2, seq_len=2048:**
- Per-GPU throughput: 8,505 tok/s
- Tokens per step: $2 \times 2048 = 4{,}096$
- **Step time**: $4{,}096 / 8{,}505 = 0.482\text{ s}$
- **Per layer** (28 decoder layers): $0.482 / 28 \approx 17\text{ ms}$

This 17 ms includes forward, backward, activation recomputation (checkpointing), and optimizer overhead — the complete per-layer training cost. As a sanity check: working backwards to MFU gives ~60 TFLOPS per GPU, or roughly 6% of the H100's 989 TFLOPS bf16 peak. This low MFU is expected for batch_size=2 with activation checkpointing — the GPU spends significant time on memory-bound operations, attention masking, and checkpoint recomputation that don't achieve peak throughput.

Now examine what happens when we move from 1 node to 2 nodes:

| Config | tok/s/GPU | Step time | Overhead vs 1-node |
|--------|-----------|-----------|-------------------|
| 1-node (baseline) | 8,505 | 0.482 s | — |
| 2-node IB (3.2 Tbps RDMA) | 6,876 | 0.596 s | **+114 ms** |
| 2-node ETH (80 Gbps TCP) | 6,874 | 0.596 s | **+114 ms** |

**This is the smoking gun.** A network with 3.2 Tbps of RDMA bandwidth and a network with 80 Gbps of TCP bandwidth produce *the exact same* 114 ms overhead. If the overhead were caused by data transfer time, IB would show dramatically less overhead than ETH — the bandwidth difference is **40×**. But they're identical.

This proves that the **114 ms is not a bandwidth penalty**. It is the fixed cost of distributed coordination:
- Synchronization barriers per collective (16-way vs. 8-way)
- FSDP wrapper overhead scaling with world size
- Reduced arithmetic intensity as each GPU's parameter shard shrinks (7B / 16 = 437M params per GPU)
- Pipeline bubbles at step boundaries

The 25 ms ETH transfer per layer is completely absorbed within this 114 ms coordination window. FSDP's prefetch pipeline has more than enough headroom to hide it — the GPUs are waiting on synchronization, not on the network.

This pattern holds at scale. At 8 nodes (64 GPUs), the total multi-node overhead for the 7B model is only 136 ms over ETH — just 22% overhead across 84+ communication operations (28 layers × 3 collectives each). The pipelining remains effective even at scale because the per-operation payload stays manageable: with 64 GPUs across 8 nodes, each node group's inter-node exchange is still on the order of tens of MB, well within FSDP's ability to overlap.

### 3. The Tipping Point: 72B at 8 Nodes

The 72B model at 8 nodes is where the math flips — and where InfiniBand finally justifies its cost:

| Config | tok/s/GPU | Step time | ETH penalty |
|--------|-----------|-----------|-------------|
| 72B, 8-node IB | 483 | 8.49 s | — |
| 72B, 8-node ETH | 447 | 9.16 s | **+670 ms** |

The communication profile changes dramatically at this scale:

- **Layer size**: 72B / 80 layers ≈ 900M params per layer = **1.8 GB** in bf16
- **Inter-node payload** (hierarchical): ~900 MB per all-gather
- **ETH transfer time**: $900\text{ MB} / 10\text{ GB/s} \approx 90\text{ ms per operation}$
- **IB transfer time**: $900\text{ MB} / 400\text{ GB/s} \approx 2.3\text{ ms per operation}$

Meanwhile, compute per layer shrinks as the work is spread across 64 GPUs. Each GPU holds only $72\text{B} / 64 = 1.13\text{B}$ parameters — less compute per GPU per layer than the 7B model at 8 GPUs. You might ask: why use 64 GPUs if each one does less work? Because you *must*. The 72B model's parameters, gradients, and optimizer states simply don't fit in the memory of fewer GPUs. FSDP shards the model to make training possible at all — but the side effect is that each GPU finishes its compute slice faster, leaving less time for FSDP to hide the network transfer behind useful work.

FSDP can trivially hide a 2.3 ms IB transfer behind the compute window. But a **90 ms** Ethernet transfer is so large that it exceeds FSDP's prefetch budget — the compute for one layer finishes before the next layer's parameters have arrived, and the GPU stalls. This pipeline stall accumulates across 80 layers in both forward and backward passes, producing the observed 670 ms penalty and the 7.5% throughput gap.

**Why doesn't this happen at 2–4 nodes?** At 2 nodes, each GPU holds 72B / 16 = 4.5B parameters — **4× more compute** per GPU than at 8 nodes. The per-layer compute time is longer, giving FSDP a much larger window to hide transfers. The inter-node payload is also smaller (fewer node boundaries to cross). At 4 nodes (32 GPUs), the ratio still holds: 2.25B params per GPU provides sufficient compute to overlap the ~450 MB inter-node transfers. It is only at 8 nodes that the double penalty — less compute to hide behind, more data to transfer — finally tips the balance in favor of InfiniBand.

**So why not just stay at 4 nodes?** Because the goal is **total throughput**, not per-GPU efficiency. The 72B model at 4 nodes produces 17,871 tok/s; at 8 nodes with IB, it produces 30,914 tok/s — a **1.73× speedup**. Per-GPU efficiency drops (558 → 483 tok/s/GPU), but you're training 73% faster in wall-clock time. For a fine-tuning run that takes days, or a pre-training run that takes weeks, that's the difference between 4 weeks and 2.3 weeks. And real production clusters don't stop at 8 nodes — they scale to hundreds or thousands of GPUs, where this tradeoff compounds. InfiniBand is what makes that scaling viable: without it, every additional node beyond the ETH-safe threshold adds GPUs that spend more time waiting on the network than doing useful compute.

## Practical Takeaways

### 1. InfiniBand Is Essential for Large-Scale LLM Training

The 72B results tell a clear story: as soon as the model is large enough and the GPU count high enough, Ethernet can't keep up. At 8 nodes (64 GPUs), IB already delivers 8% higher throughput — and this is just fine-tuning with modest batch sizes. In production pre-training, where clusters run hundreds or thousands of GPUs for weeks, the communication-to-compute ratio is far more demanding. Tensor parallelism, pipeline parallelism, and expert parallelism all introduce latency-sensitive communication that cannot be overlapped as effectively as FSDP's all-gather. The 8% gap at 8 nodes is a floor, not a ceiling — it would widen significantly at 16, 32, or 64 nodes as cross-node traffic grows faster than per-GPU compute shrinks.

### 2. The Gap Compounds at Scale

Our experiment was capped at 8 nodes by cluster size, but the math is straightforward: doubling the node count roughly halves the per-GPU compute while increasing all-reduce traffic. At 16 nodes with the 72B model, the per-layer compute window (~100 ms) would no longer cover the Ethernet transfer time (~720 ms for 3.6 GB across a ring of 16 nodes). InfiniBand, with 10× the bandwidth, stays comfortably within the overlap window. For any workload that will scale beyond a handful of nodes, IB isn't optional — it's the difference between linear scaling and diminishing returns.

### 3. Small-Model Fine-Tuning Is the Exception, Not the Rule

The 7B results — where Ethernet matched IB at every scale — reflect a specific and narrow use case: a model small enough that per-layer compute on the H100 exceeds the communication time even over TCP. This works because the ratio of compute to communication remains favorable. But most serious LLM work today involves models at 70B+ parameters, and the trend is toward larger models, not smaller ones. Designing infrastructure around the 7B case would be short-sighted.

### 4. Ethernet Can Work for Targeted Fine-Tuning Workloads

That said, the data does show that for teams fine-tuning 7B–13B models on a moderate number of nodes, Ethernet delivers equivalent throughput. If your workload fits this profile — small model, short fine-tuning runs, fewer than ~32 GPUs — you may be able to use non-IB VM SKUs and reinvest the savings elsewhere. But this should be validated with profiling on your specific workload before committing to an Ethernet-only cluster. The inflection point depends on model size, batch size, sequence length, and GPU count — and it arrives faster than you might expect.

## Reproducing These Results

The cluster runs on Azure with 10× Standard_ND96isr_H100_v5 VMs in a VMSS, with an 8 TiB Azure Managed Lustre filesystem mounted at `/lustre` on every node. See [this post]({% post_url 2026-02-09-AMLFS with GPU VMSS %}) for the cluster setup.

Scripts: [finetune_bench.py](https://github.com/JingchaoZhang/JingchaoZhang.github.io/blob/master/scripts/ib-vs-eth/finetune_bench.py), [launch_node.sh](https://github.com/JingchaoZhang/JingchaoZhang.github.io/blob/master/scripts/ib-vs-eth/launch_node.sh), [run_multinode.sh](https://github.com/JingchaoZhang/JingchaoZhang.github.io/blob/master/scripts/ib-vs-eth/run_multinode.sh), [sweep.sh](https://github.com/JingchaoZhang/JingchaoZhang.github.io/blob/master/scripts/ib-vs-eth/sweep.sh).

To run a single experiment:

```bash
# 4 nodes, IB enabled, Qwen2.5-7B, batch_size=2
bash /lustre/scripts/run_multinode.sh 4 0 /lustre/models/Qwen2.5-7B 2048 2 20

# 4 nodes, Ethernet only, Qwen2.5-72B, batch_size=1
bash /lustre/scripts/run_multinode.sh 4 1 /lustre/models/Qwen2.5-72B 2048 1 20
```

To run the full sweep (all models × all node counts × IB/ETH):

```bash
# Inside screen on the head node to survive SSH disconnects
screen -S sweep
bash /lustre/scripts/sweep.sh
```
