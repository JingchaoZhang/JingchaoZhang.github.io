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

Azure's H100 GPU VMs (`Standard_ND96isr_H100_v5`) come equipped with 8× 400 Gb/s NDR InfiniBand — 3.2 Tbps of aggregate RDMA bandwidth per node. Everyone knows InfiniBand matters for distributed training. But how much does it *actually* matter — measured in throughput, not marketing?

In this post, I benchmark InfiniBand (RDMA) against genuine Ethernet (TCP sockets) across two dense models (Qwen2.5-7B and Qwen2.5-72B) and one Mixture-of-Experts model (Mixtral 8x7B), scaling from 1 to 8 nodes (8–64 GPUs), using PyTorch FSDP on an 11-node Azure VMSS cluster with Azure Managed Lustre.

**The results are dramatic.** InfiniBand delivers **26–28× higher multi-node throughput** than Ethernet for the 7B dense model, **38–45× higher** for the 72B dense model, and **56–57× higher** for the Mixtral 8x7B MoE model. The MoE architecture — with its expert routing and 46.7B total parameters sharded across nodes — is the most communication-intensive of the three, making InfiniBand even more critical.

> **Correction:** An earlier version of this post reported near-identical IB and Ethernet performance using `NCCL_IB_DISABLE=1`. That environment variable [does not work](#the-nccl_ib_disable-bug) with NCCL 2.23.4 on Azure ND-series VMs — the RDMA network plugin ignores it entirely. All previous "Ethernet" results were actually running on InfiniBand. This update uses `NCCL_NET=Socket` to force genuine TCP sockets, [verified via nccl-tests](#the-nccl_ib_disable-bug).

## Test Environment

| Component | Detail |
|-----------|--------|
| **VM SKU** | Standard_ND96isr_H100_v5 |
| **GPUs per node** | 8× NVIDIA H100 80 GB HBM3 |
| **Intra-node** | NVLink 4th gen, 900 GB/s bisection |
| **Inter-node (IB)** | 8× 400 Gb/s NDR InfiniBand (ConnectX-7) |
| **Inter-node (ETH)** | TCP sockets over eth0, ~40–50 Gbps accelerated networking |
| **Nodes** | 11 (up to 8 used per experiment) |
| **Shared storage** | Azure Managed Lustre, 8 TiB, mounted at `/lustre` |
| **Container** | `nvcr.io/nvidia/pytorch:24.12-py3` (PyTorch 2.6.0a0, CUDA 12.6, NCCL 2.23.4) |
| **Models** | Qwen2.5-7B (~14 GB), Qwen2.5-72B (~136 GB), Mixtral-8x7B-v0.1 (~87 GB) in bf16 |
| **Framework** | PyTorch FSDP (`size_based_auto_wrap_policy` for dense, `transformer_auto_wrap_policy` for MoE) |

### The `NCCL_IB_DISABLE` Bug

The standard way to disable InfiniBand in NCCL is `NCCL_IB_DISABLE=1`. This is documented in NCCL's official docs, used in Azure tutorials, and appears in countless blog posts and Stack Overflow answers. But on Azure ND-series H100 VMs, **it doesn't work.**

These VMs use an **external RDMA network plugin** (`libnccl-net.so`) that registers with NCCL as a transport provider. NCCL 2.23.4 routes all inter-node traffic through this plugin regardless of the `NCCL_IB_DISABLE` setting — the flag only controls NCCL's *built-in* IB transport, not external plugins.

I discovered this when my initial "Ethernet" benchmarks produced suspiciously identical results to InfiniBand. Verification with `nccl-tests all_reduce_perf` (1 GB payload, 2 nodes, 16 GPUs):

```
# NCCL_IB_DISABLE=0 (InfiniBand "enabled")
#   busbw: ~78 GB/s per GPU → ~392 GB/s aggregate

# NCCL_IB_DISABLE=1 (InfiniBand "disabled" — supposedly Ethernet)
#   busbw: ~78 GB/s per GPU → IDENTICAL. Still using RDMA.

# NCCL_NET=Socket (the actual fix)
#   busbw: ~0.64 GB/s per GPU → ~3.19 GB/s aggregate. Genuine TCP.
```

Both `NCCL_IB_DISABLE=0` and `NCCL_IB_DISABLE=1` produce identical ~78 GB/s per GPU — they're both running on RDMA. The environment variable is simply ignored when the external network plugin is loaded.

**The fix:** `NCCL_NET=Socket` overrides all network plugins and forces NCCL to use the kernel's TCP stack. The measured bandwidth gap: **392 GB/s (RDMA) vs. 3.19 GB/s (TCP) — a 122× difference.**

### How the Two Modes Actually Differ

**InfiniBand mode** (`NCCL_IB_DISABLE=0`): NCCL uses RDMA over 8× 400 Gb/s InfiniBand links via the Azure RDMA network plugin. Data moves directly between GPU memory across nodes without CPU involvement. Measured aggregate bandwidth: ~392 GB/s between 2 nodes.

**Ethernet mode** (`NCCL_NET=Socket`, `NCCL_IB_DISABLE=1`): NCCL bypasses all network plugins and uses plain TCP sockets over `eth0`. Data passes through the kernel's TCP stack and CPU. Measured aggregate bandwidth: ~3.19 GB/s. We set both flags as belt-and-suspenders — `NCCL_NET=Socket` is the operative one.

Everything else — FSDP configuration, model, batch size, sequence length, number of training steps — is identical between the two modes. The only variable is the network transport.

## Why Lustre Matters

In my [previous post]({% post_url 2026-01-25-Fine-Tuning Across NVLink and InfiniBand %}), I used a 3-node cluster without shared storage. Each node needed a local copy of the model, distributed via `rsync`. This was manageable at 3 nodes with a 7B model (14 GB per copy), but doesn't scale:

- **7B model × 10 nodes** = 140 GB of redundant copies
- **72B model × 10 nodes** = 1.36 TB, requiring ~10 minutes of `rsync` per node

With [Azure Managed Lustre]({% post_url 2026-02-09-AMLFS with GPU VMSS %}), every node mounts `/lustre` — a shared POSIX filesystem. Models are downloaded once and accessible everywhere. This eliminated the distribution bottleneck and enabled painless scaling to 11 nodes.

## The Benchmark Script

The benchmark uses PyTorch FSDP with several optimizations essential for the 72B model:

- **Meta-device loading:** Only `local_rank 0` per node loads model weights to CPU. All other ranks create the model on PyTorch's `meta` device (zero memory). FSDP's `sync_module_states=True` broadcasts weights during wrapping.
- **Activation checkpointing:** `NO_REENTRANT` mode via `apply_activation_checkpointing` — trades compute for memory by recomputing activations during backward.
- **Flash Attention 2:** Reduces attention memory from $O(n^2)$ to $O(n)$ and accelerates the attention computation.
- **Auto-wrap policy:** `size_based_auto_wrap_policy(min_num_params=100_000_000)` with `FULL_SHARD` strategy.

The full scripts are in the [Reproducing These Results](#reproducing-these-results) section.

## Results

### Qwen2.5-7B (batch_size=2, seq_len=2048)

| Nodes | GPUs | IB (tok/s) | ETH (tok/s) | IB / ETH |
|-------|------|-----------|------------|----------|
| 1 | 8 | 65,274 | 65,230 | 1.0× |
| 2 | 16 | 131,018 | 4,968 | **26.4×** |
| 4 | 32 | 262,201 | 9,439 | **27.8×** |
| 8 | 64 | 502,991 | 18,710 | **26.9×** |

<canvas id="chart7bThroughput" width="700" height="400"></canvas>
<script>
new Chart(document.getElementById('chart7bThroughput'), {
    type: 'bar',
    data: {
        labels: ['1 node\n(8 GPU)', '2 nodes\n(16 GPU)', '4 nodes\n(32 GPU)', '8 nodes\n(64 GPU)'],
        datasets: [
            {
                label: 'InfiniBand (RDMA)',
                data: [65274, 131018, 262201, 502991],
                backgroundColor: 'rgba(54, 162, 235, 0.8)',
                borderColor: 'rgba(54, 162, 235, 1)',
                borderWidth: 1
            },
            {
                label: 'Ethernet (TCP)',
                data: [65230, 4968, 9439, 18710],
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

**The single-node baseline is identical** — 65,274 (IB) vs. 65,230 (ETH) tok/s. No communication crosses the inter-node link, so the interconnect is irrelevant. This confirms the test is fair.

**At 2+ nodes, InfiniBand is 26–28× faster.** The gap appears immediately at 2 nodes and remains remarkably consistent as the cluster scales. InfiniBand shows near-perfect linear scaling — 65K → 131K → 262K → 503K tok/s — while Ethernet throughput is strangled by inter-node communication. The Ethernet per-GPU efficiency drops from 8,154 tok/s (1-node) to just 292–311 tok/s (multi-node): a **26× collapse** in per-GPU utilization the moment FSDP must communicate across the network.

### Qwen2.5-72B (batch_size=1, seq_len=2048)

The 72B model requires FSDP sharding across at least 2 nodes (16 GPUs) — a single node's 640 GB of HBM3 cannot hold the model parameters, gradients, and optimizer states simultaneously.

| Nodes | GPUs | IB (tok/s) | ETH (tok/s) | IB / ETH |
|-------|------|-----------|------------|----------|
| 2 | 16 | 9,127 | 203 | **45.0×** |
| 4 | 32 | 17,604 | 402 | **43.8×** |
| 8 | 64 | 30,175 | 784 | **38.5×** |

<canvas id="chart72bThroughput" width="700" height="400"></canvas>
<script>
new Chart(document.getElementById('chart72bThroughput'), {
    type: 'bar',
    data: {
        labels: ['2 nodes\n(16 GPU)', '4 nodes\n(32 GPU)', '8 nodes\n(64 GPU)'],
        datasets: [
            {
                label: 'InfiniBand (RDMA)',
                data: [9127, 17604, 30175],
                backgroundColor: 'rgba(54, 162, 235, 0.8)',
                borderColor: 'rgba(54, 162, 235, 1)',
                borderWidth: 1
            },
            {
                label: 'Ethernet (TCP)',
                data: [203, 402, 784],
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

The 72B model shows a consistently larger gap than the 7B: **45× at 2 nodes, 44× at 4 nodes, and 39× at 8 nodes**. The 72B has 10× more parameters, generating proportionally more inter-node traffic per FSDP collective. InfiniBand scaling remains solid — 9,127 → 17,604 → 30,175 tok/s — though per-GPU efficiency drops from 571 to 472 tok/s (17%) as the massive collectives begin to saturate even the RDMA fabric. On Ethernet, each step takes over **2.5 minutes** regardless of scale, confirming completely network-dominated execution. The slight decrease in speedup ratio at 8 nodes reflects InfiniBand's growing (but still tolerable) communication overhead, while Ethernet is already so saturated that adding nodes barely changes per-GPU throughput.

### Mixtral 8x7B MoE (batch_size=1, seq_len=2048)

Mixtral 8x7B is a **Mixture-of-Experts** (MoE) model: 46.7B total parameters across 32 decoder layers, each containing 8 expert FFN modules, with top-2 expert routing selecting ~12.9B active parameters per token. This architecture creates a distinctive FSDP challenge: the communication volume scales with the **total** parameter count (46.7B), but the useful compute per token scales with only the **active** subset (12.9B) — a 3.6× worse compute-to-communication ratio than an equivalently sized dense model.

| Nodes | GPUs | IB (tok/s) | ETH (tok/s) | IB / ETH |
|-------|------|-----------|------------|----------|
| 1 | 8 | 11,577 | 11,634 | 1.0× |
| 2 | 16 | 22,520 | 398 | **56.6×** |
| 4 | 32 | 44,239 | 774 | **57.2×** |
| 8 | 64 | 83,583 | 1,501 | **55.7×** |

<canvas id="chartMoeThroughput" width="700" height="400"></canvas>
<script>
new Chart(document.getElementById('chartMoeThroughput'), {
    type: 'bar',
    data: {
        labels: ['1 node\n(8 GPU)', '2 nodes\n(16 GPU)', '4 nodes\n(32 GPU)', '8 nodes\n(64 GPU)'],
        datasets: [
            {
                label: 'InfiniBand (RDMA)',
                data: [11577, 22520, 44239, 83583],
                backgroundColor: 'rgba(54, 162, 235, 0.8)',
                borderColor: 'rgba(54, 162, 235, 1)',
                borderWidth: 1
            },
            {
                label: 'Ethernet (TCP)',
                data: [11634, 398, 774, 1501],
                backgroundColor: 'rgba(255, 99, 132, 0.8)',
                borderColor: 'rgba(255, 99, 132, 1)',
                borderWidth: 1
            }
        ]
    },
    options: {
        responsive: true,
        plugins: {
            title: { display: true, text: 'Mixtral 8x7B MoE: Aggregate Throughput', font: { size: 16 } },
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

**The MoE model doubles the IB/ETH gap.** At 56–57× across all multi-node configurations, the MoE speedup is roughly **twice** the 7B dense model's 27× and exceeds even the 72B dense model's 38–45×. The single-node baseline is again identical (11,577 IB vs. 11,634 ETH), confirming the gap is purely an interconnect effect.

InfiniBand scaling remains excellent: **97.3% efficiency at 2 nodes and 90.2% at 8 nodes** (relative to the single-node baseline of 11,577 tok/s). On Ethernet, per-GPU throughput collapses from 1,454 tok/s (1-node) to 23–25 tok/s (multi-node) — each GPU spends over 98% of its time waiting for network transfers.

### Per-GPU Efficiency

<canvas id="chartPerGPU" width="700" height="400"></canvas>
<script>
new Chart(document.getElementById('chartPerGPU'), {
    type: 'line',
    data: {
        labels: ['1 node\n(8 GPU)', '2 nodes\n(16 GPU)', '4 nodes\n(32 GPU)', '8 nodes\n(64 GPU)'],
        datasets: [
            {
                label: '7B Dense — InfiniBand',
                data: [8159, 8189, 8194, 7859],
                borderColor: 'rgba(54, 162, 235, 1)',
                backgroundColor: 'rgba(54, 162, 235, 0.1)',
                borderWidth: 2,
                tension: 0.2,
                pointRadius: 5
            },
            {
                label: '7B Dense — Ethernet',
                data: [8154, 311, 295, 292],
                borderColor: 'rgba(255, 99, 132, 1)',
                backgroundColor: 'rgba(255, 99, 132, 0.1)',
                borderWidth: 2,
                borderDash: [5, 5],
                tension: 0.2,
                pointRadius: 5
            },
            {
                label: '72B Dense — InfiniBand',
                data: [null, 571, 550, 472],
                borderColor: 'rgba(153, 102, 255, 1)',
                backgroundColor: 'rgba(153, 102, 255, 0.1)',
                borderWidth: 2,
                tension: 0.2,
                pointRadius: 5
            },
            {
                label: '72B Dense — Ethernet',
                data: [null, 13, 13, 12],
                borderColor: 'rgba(201, 128, 232, 1)',
                backgroundColor: 'rgba(201, 128, 232, 0.1)',
                borderWidth: 2,
                borderDash: [5, 5],
                tension: 0.2,
                pointRadius: 5
            },
            {
                label: 'MoE 8x7B — InfiniBand',
                data: [1447, 1408, 1382, 1306],
                borderColor: 'rgba(75, 192, 192, 1)',
                backgroundColor: 'rgba(75, 192, 192, 0.1)',
                borderWidth: 2,
                tension: 0.2,
                pointRadius: 5
            },
            {
                label: 'MoE 8x7B — Ethernet',
                data: [1454, 25, 24, 23],
                borderColor: 'rgba(255, 159, 64, 1)',
                backgroundColor: 'rgba(255, 159, 64, 0.1)',
                borderWidth: 2,
                borderDash: [5, 5],
                tension: 0.2,
                pointRadius: 5
            }
        ]
    },
    options: {
        responsive: true,
        plugins: {
            title: { display: true, text: 'Per-GPU Throughput: Dense vs. MoE', font: { size: 16 } },
            legend: { position: 'top' }
        },
        scales: {
            y: {
                type: 'logarithmic',
                title: { display: true, text: 'Tokens/sec/GPU (log scale)' },
                min: 10,
                max: 10000,
                ticks: { callback: function(v) { return v.toLocaleString(); } }
            }
        }
    }
});
</script>

The per-GPU chart (log scale) reveals the scaling story clearly. For the 7B dense model, InfiniBand maintains ~8,100–8,200 tok/s per GPU from 1 to 4 nodes — **near-perfect linear scaling** — with only a 3.7% dip at 8 nodes. The 72B dense model shows lower per-GPU throughput (~571 tok/s at 2 nodes) due to its heavier per-layer computation, dropping to 472 tok/s at 8 nodes (17% loss) as the larger collectives stress even the RDMA fabric. The MoE model follows a similar pattern at ~1,400 tok/s per GPU on IB.

On Ethernet, all three models collapse. Qwen 7B drops from 8,154 to ~300 tok/s per GPU. The 72B model is the most extreme: just 12–13 tok/s per GPU across all scales — a **97.7% loss** in utilization. Mixtral MoE lands at 23–25 tok/s per GPU. The 72B and MoE collapses are more severe than the 7B's because their larger collective payloads amplify network stalls relative to useful compute.

### IB-to-ETH Speedup

<canvas id="chartSpeedup" width="700" height="350"></canvas>
<script>
new Chart(document.getElementById('chartSpeedup'), {
    type: 'line',
    data: {
        labels: ['1 node\n(8 GPU)', '2 nodes\n(16 GPU)', '4 nodes\n(32 GPU)', '8 nodes\n(64 GPU)'],
        datasets: [
            {
                label: 'Qwen2.5-7B',
                data: [1.0, 26.4, 27.8, 26.9],
                borderColor: 'rgba(54, 162, 235, 1)',
                backgroundColor: 'rgba(54, 162, 235, 0.1)',
                borderWidth: 2,
                tension: 0.2,
                pointRadius: 6,
                fill: false
            },
            {
                label: 'Qwen2.5-72B',
                data: [null, 45.0, 43.8, 38.5],
                borderColor: 'rgba(255, 159, 64, 1)',
                backgroundColor: 'rgba(255, 159, 64, 0.1)',
                borderWidth: 2,
                tension: 0.2,
                pointRadius: 6,
                pointStyle: 'triangle',
                fill: false
            },
            {
                label: 'Mixtral 8x7B (MoE)',
                data: [1.0, 56.6, 57.2, 55.7],
                borderColor: 'rgba(75, 192, 192, 1)',
                backgroundColor: 'rgba(75, 192, 192, 0.1)',
                borderWidth: 2,
                tension: 0.2,
                pointRadius: 6,
                fill: false
            }
        ]
    },
    options: {
        responsive: true,
        plugins: {
            title: { display: true, text: 'IB-to-ETH Throughput Speedup', font: { size: 16 } },
            legend: { position: 'top' }
        },
        scales: {
            y: {
                beginAtZero: true,
                title: { display: true, text: 'Speedup (×)' },
                ticks: {
                    callback: function(v) { return v + '×'; }
                }
            }
        }
    }
});
</script>

The 7B speedup is remarkably consistent at **~27×** across 2, 4, and 8 nodes — the gap doesn't grow or shrink with scale. The Mixtral MoE model shows an even more dramatic and consistent **~57×** gap, nearly double the 7B's ratio, reflecting the MoE architecture's heavier communication burden. The 72B dense model sits between the two at **45–39×**, with the gap gradually narrowing from 2 to 8 nodes as InfiniBand's overhead grows with the massive 72B-scale collectives — yet even at 8 nodes the 39× gap dwarfs the 7B's 27×. At 1 node (no inter-node communication), the 7B and MoE models show a 1.0× speedup, confirming the gap is purely an interconnect effect.

## Why the Gap Is So Large

### The Bandwidth Reality

The nccl-tests measurements establish the fundamental constraint:

- **InfiniBand**: ~392 GB/s aggregate inter-node bandwidth (RDMA, zero-copy)
- **Ethernet**: ~3.19 GB/s aggregate inter-node bandwidth (TCP sockets, CPU-mediated)
- **Ratio: 122×**

FSDP with `FULL_SHARD` performs three types of collective operations per layer per training step: an all-gather in the forward pass, an all-gather during backward recomputation (activation checkpointing), and a reduce-scatter for gradients. For the 7B model with 28 decoder layers, that's **~84 inter-node collective operations per step.**

Each collective for a 7B layer transfers roughly 250 MB across the inter-node link (hierarchical algorithm: each node holds half, exchanges with the other). The transfer time:

$$\text{IB: } \frac{250\text{ MB}}{392\text{ GB/s}} \approx 0.6\text{ ms per collective}$$

$$\text{ETH: } \frac{250\text{ MB}}{3.19\text{ GB/s}} \approx 78\text{ ms per collective}$$

On InfiniBand, 84 collectives × 0.6 ms = ~50 ms of total inter-node transfer — easily hidden behind the ~500 ms compute budget. On Ethernet, 84 collectives × 78 ms = ~6,500 ms — **13× the compute time**, impossible to hide.

### The Empirical Proof

The step time measurements confirm this directly:

| Config | Step Time | Network Overhead |
|--------|-----------|-----------------|
| 1 node (baseline) | 502 ms | — |
| 8-node IB | 521 ms | **+19 ms** |
| 8-node ETH | 14,011 ms | **+13,509 ms** |

**InfiniBand adds 19 milliseconds** of network overhead to scale from 8 GPUs to 64 GPUs — a 3.8% increase. The IB transfers are almost entirely hidden behind GPU compute thanks to FSDP's prefetch pipeline.

**Ethernet adds 13.5 seconds** of network overhead — a 2,691% increase. The GPU spends most of each training step stalled, waiting for the next layer's parameters to arrive over TCP.

The **711× difference** in network overhead (13,509 ms vs. 19 ms) directly reflects the interconnect bandwidth gap. It doesn't reach the full 122× bandwidth ratio because FSDP's pipelining can partially overlap *some* Ethernet transfers with compute — but the overlap fraction is small when each transfer takes 78 ms against a 17 ms per-layer compute window.

### Why Not 122× Throughput Gap?

The throughput gap is ~27× (7B dense), 38–45× (72B dense), and ~57× (MoE), not 122×, because:

1. **Intra-node NVLink is unaffected.** All 8 GPUs within each node still communicate at 900 GB/s on both IB and ETH. Only the inter-node link changes.
2. **GPU compute is nonzero.** Even on Ethernet, the GPUs perform some useful work between network stalls. The step time is compute + network, not network alone.
3. **FSDP pipelining overlaps *some* transfers.** While one layer computes, FSDP prefetches the next layer's parameters. On IB this hides everything; on ETH it hides a fraction.

The 72B model shows a larger gap (38–45×) because each layer is ~10× larger, generating ~10× more inter-node traffic per collective. The gap decreases slightly from 45× (2 nodes) to 39× (8 nodes) because even InfiniBand takes longer to coordinate 72B-scale collectives across more participants, while Ethernet is already completely saturated and barely changes per-GPU throughput as nodes increase.

### Why the MoE Gap Is Even Wider

The Mixtral 8x7B MoE model shows a **56–57× IB/ETH gap** — roughly double the 7B dense model's ~27×. This amplification comes from the MoE architecture's fundamental property: **total parameters far exceed active parameters.**

Each Mixtral decoder layer contains 8 expert FFN modules, but only 2 are activated per token (top-2 routing). FSDP shards the entire model — all 46.7B parameters — across GPUs. Every forward pass all-gathers the complete layer (including all 8 experts), but only ~25% of those parameters participate in the actual matrix multiplies for any given token.

This creates a 3.6× worse **compute-to-communication ratio** than an equivalently sized dense model:

- **Dense model**: communicates $P$ parameters, computes with $P$ parameters → ratio 1:1
- **MoE model**: communicates 46.7B parameters, computes with ~12.9B → ratio **3.6:1**

The extra communication volume cannot be hidden behind useful compute:

| Config | Step Time | Network Overhead |
|--------|-----------|-----------------|
| 1-node MoE (baseline) | 1,415 ms | — |
| 8-node MoE, IB | 1,568 ms | **+153 ms** (10.8%) |
| 8-node MoE, ETH | 87,345 ms | **+85,930 ms** (6,072%) |

Compare this to the 7B dense model: IB added only 19 ms of overhead at 8 nodes, vs. 153 ms for the MoE model. Even InfiniBand feels the weight of all-gathering 46.7B parameters across 8 nodes — but 153 ms of overhead is still tolerable. On Ethernet, the 85.9 *seconds* of per-step network overhead renders multi-node MoE training completely impractical.

## Practical Takeaways

### 1. InfiniBand Is Non-Negotiable for Multi-Node Training

A 27× throughput gap is not a performance optimization — it's the difference between feasible and infeasible. Training that takes 1 day on InfiniBand takes **4 weeks** on Ethernet. No amount of software tuning can close a 122× bandwidth gap.

### 2. The Gap Is Immediate and Constant

The slowdown doesn't creep in at large scale — it appears at **2 nodes** and stays at ~27× regardless of cluster size (for the 7B model). There is no "safe" multi-node Ethernet regime. The moment FSDP communicates across an Ethernet link, throughput collapses.

### 3. Larger and Sparse Models Widen the Gap

The 72B dense model shows 38–45× slowdown vs. the 7B's ~27×, and the Mixtral MoE model pushes it to **~57×**. Larger models mean more data per collective. MoE architectures are even worse — they communicate *all* expert parameters but compute with only a fraction, creating the worst compute-to-communication ratio. As LLMs trend toward both larger sizes and sparse MoE designs, InfiniBand's advantage will only grow.

### 4. Verify Your Ethernet Mode

If you're benchmarking IB vs. Ethernet on Azure ND-series VMs, **do not trust `NCCL_IB_DISABLE=1`**. Use `NCCL_NET=Socket` and verify with `nccl-tests` that bandwidth drops to TCP levels (~3 GB/s rather than ~392 GB/s). Without this verification, you may be unknowingly benchmarking IB against IB.

### 5. Single-Node Fine-Tuning Is the Exception

The single-node results — identical IB and ETH performance — show that interconnect doesn't matter when all communication stays on NVLink. If your model and batch fit in one node's memory, Ethernet VMs are perfectly fine. But this is increasingly rare as model sizes grow.

## Reproducing These Results

The cluster runs on Azure with 11× Standard_ND96isr_H100_v5 VMs in a VMSS, with an 8 TiB Azure Managed Lustre filesystem mounted at `/lustre` on every node. See [this post]({% post_url 2026-02-09-AMLFS with GPU VMSS %}) for the cluster setup.

**Dense model scripts:** [finetune_bench.py](https://github.com/JingchaoZhang/JingchaoZhang.github.io/blob/master/scripts/ib-vs-eth/finetune_bench.py), [launch_node.sh](https://github.com/JingchaoZhang/JingchaoZhang.github.io/blob/master/scripts/ib-vs-eth/launch_node.sh), [run_multinode.sh](https://github.com/JingchaoZhang/JingchaoZhang.github.io/blob/master/scripts/ib-vs-eth/run_multinode.sh), [sweep.sh](https://github.com/JingchaoZhang/JingchaoZhang.github.io/blob/master/scripts/ib-vs-eth/sweep.sh).

**MoE model scripts:** [finetune_bench_moe.py](https://github.com/JingchaoZhang/JingchaoZhang.github.io/blob/master/scripts/ib-vs-eth/finetune_bench_moe.py), [launch_node_moe.sh](https://github.com/JingchaoZhang/JingchaoZhang.github.io/blob/master/scripts/ib-vs-eth/launch_node_moe.sh), [run_multinode_moe.sh](https://github.com/JingchaoZhang/JingchaoZhang.github.io/blob/master/scripts/ib-vs-eth/run_multinode_moe.sh), [sweep_moe.sh](https://github.com/JingchaoZhang/JingchaoZhang.github.io/blob/master/scripts/ib-vs-eth/sweep_moe.sh).

To run a single experiment:

```bash
# Dense: 4 nodes, IB enabled, Qwen2.5-7B, batch_size=2
bash /lustre/scripts/run_multinode.sh 4 0 /lustre/models/Qwen2.5-7B 2048 2 20

# Dense: 4 nodes, Ethernet only, Qwen2.5-72B, batch_size=1
bash /lustre/scripts/run_multinode.sh 4 1 /lustre/models/Qwen2.5-72B 2048 1 20

# MoE: 4 nodes, IB enabled, Mixtral-8x7B, batch_size=1
bash /lustre/scripts/run_multinode_moe.sh 4 0 /lustre/models/Mixtral-8x7B-v0.1 2048 1 20

# MoE: 4 nodes, Ethernet only, Mixtral-8x7B, batch_size=1
bash /lustre/scripts/run_multinode_moe.sh 4 1 /lustre/models/Mixtral-8x7B-v0.1 2048 1 20
```

To run the full sweep (all models × all node counts × IB/ETH):

```bash
# Inside screen on the head node to survive SSH disconnects
screen -S sweep
bash /lustre/scripts/sweep.sh       # Dense models
bash /lustre/scripts/sweep_moe.sh   # MoE model
```
