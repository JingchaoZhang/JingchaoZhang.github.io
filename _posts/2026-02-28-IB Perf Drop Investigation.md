---
layout: single
title: "Debugging InfiniBand Performance Degradation Caused by Background GPU Stress Tests"
author_profile: false
---

## The Problem

A customer operating an H200 GPU cluster (~10k GPUs) reported that their distributed training workload had slowed significantly. NCCL `all_reduce` benchmarks confirmed the regression: bus bandwidth had dropped from ~320 GB/s to ~170 GB/s — nearly a 50% reduction. The customer was able to reproduce the issue on as few as two nodes.

## Initial Investigation: Ruling Out Fabric Issues

The InfiniBand engineering team performed a fabric-wide scan and identified several links with high Bit Error Rate (BER) and degraded ports. These were disabled and isolated. No congestion or switch-level issues were found. However, the customer confirmed the performance degradation persisted after the fabric remediation.

## Setting Up a Controlled Environment

Fortunately, the cluster had a handful of **Healthy Empty Nodes (HENs)** — idle, known-good nodes — where I could deploy a VMSS and conduct isolated testing.

### Node Preparation

First, I enabled persistence mode and locked GPU clocks across all nodes. In some configurations, the power cap may also need adjustment.

```bash
export HOSTFILE="./hostfile"
export N_THREADS=$(wc -l < "$HOSTFILE")

# Disable host key checking for parallel-ssh
cat > "$HOME/.ssh/config" <<EOL
Host *
    StrictHostKeyChecking no
    UserKnownHostsFile=/dev/null
EOL
chmod 600 "$HOME/.ssh/config"

parallel-ssh -i -h $HOSTFILE -p $N_THREADS -t 0 "sudo nvidia-smi -pm 1"
parallel-ssh -i -h $HOSTFILE -p $N_THREADS -t 0 "sudo nvidia-smi -lgc 1980 --mode 1"
```

### Health Checks

I ran [Azure NHC (Node Health Checks)](https://github.com/Azure/azurehpc-health-checks) across all nodes to validate hardware topology, GPU/NVLink/IB status, and baseline bandwidth.

```bash
parallel-ssh -i -h $HOSTFILE -p $N_THREADS -t 0 \
  "bash ~/azurehpc-health-checks/run-health-checks.sh 2>&1 | tee health.log"
```

All checks passed — topology, GPU/NVMe counts, ECC, NVLink, and IB write bandwidth (all 8x ConnectX-7 NICs at ~392 Gbps):
```
SUCCESS:  nhc:  Health check passed:  check_hw_topology: Topology passed check against /azure-nhc/topofiles/ndv5-topo.xml
SUCCESS:  nhc:  Health check passed:  check_gpu_count: Expected 8 and found 8
SUCCESS:  nhc:  Health check passed:  check_gpu_ecc: ECC checks passed
SUCCESS:  nhc:  Health check passed:  check_nccl_allreduce: NCCL all reduce bandwidth test passed, 482.559 GB/s
SUCCESS:  nhc:  Health check passed:  check_ib_bw_gdr: IB_WRITE_BW passed for all 8 IB ports, BW=~392 Gbps each
SUCCESS:  nhc:  Health check passed:  check_nvlink_status: All 8 GPUs have all NVLinks active
SUCCESS:  nhc:  Health check passed:  check_ib_link_flapping: No excessive IB link flapping found
```

### NCCL Baseline with Tuned Parameters

My initial NCCL `all_reduce` run with properly tuned parameters showed healthy bandwidth — consistently around ~390 GB/s bus bandwidth:

```
NCCL version 2.28.9+cuda13.0
#       size         count      type   redop     time   algbw   busbw      time   algbw   busbw
#        (B)    (elements)                        (us)  (GB/s)  (GB/s)      (us)  (GB/s)  (GB/s)
  8589934592    2147483648     float     sum   42392.9  202.63  392.59   42582.7  201.72  390.84
  8589934592    2147483648     float     sum   42457.2  202.32  391.99   42567.6  201.80  390.98
  8589934592    2147483648     float     sum   42435.2  202.42  392.20   42581.4  201.73  390.85
  ...
  8589934592    2147483648     float     sum   42396.0  202.61  392.56   42569.3  201.79  390.96
```

Aside from a few minor dips, the results were within expected range. This naturally led me to examine the customer's test parameters.

### Investigating NCCL Tuning Parameters

The customer's NCCL configuration was missing several optimization flags critical for multi-rail InfiniBand performance:

```bash
-x NCCL_MIN_NCHANNELS=32 \
-x NCCL_IB_QPS_PER_CONNECTION=4 \
-x NCCL_P2P_NET_CHUNKSIZE=$((512*1024)) \
-x NCCL_IGNORE_CPU_AFFINITY=1 \
```

Removing these flags from my test produced noticeably lower throughput — peak bus bandwidth dropped from ~392 GB/s to ~337 GB/s. However, this alone did not fully explain the customer's reported ~170 GB/s:

```
  8589934592    2147483648     float     sum   49377.7  173.96  337.05   49782.2  172.55  334.32
  8589934592    2147483648     float     sum   49365.9  174.01  337.14   59351.0  144.73  280.42
  ...
  8589934592    2147483648     float     sum   104298    82.36  159.57   73555.0  116.78  226.27
```

The missing tuning flags accounted for some degradation, but the customer was seeing something worse. A different root cause had to be at play.

## Narrowing Down: VM-Level State

Since the customer reported that performance degraded over time (not immediately at boot), I suspected accumulated VM-level state — degraded PCIe links, uncorrectable ECC errors, or other transient hardware issues.

I asked the customer to reboot four affected VMs and rerun their benchmark. The result: **performance returned to normal**. This confirmed the issue was rooted in VM-level state, not the InfiniBand fabric.

The question remained: **what was accumulating on these VMs to cause the degradation?**

## Finding the Root Cause: A Background GPU Stress Test

I asked the customer to run `nvbandwidth` and `dcgmi diag -r 4` (Level 4 DCGM diagnostics) on the degraded nodes. Both came back clean.

Then a crucial detail emerged: the customer disclosed that they had a **GPU diagnostic stress test — specifically, the Extended Utility Diagnostics (EUD) — running continuously in the background** on all nodes. They did not believe it was the root cause because, according to their observation, "EUD only uses ~10% of GPU resources."

This immediately raised a red flag. EUD is not a lightweight monitoring tool — it is a **full GPU stress diagnostic** that exercises memory, compute units, and NVLink interconnects.

## Reproduction

I set up a controlled experiment on my test cluster to definitively confirm the impact.

**Environment:**
```bash
$ nvidia-smi -q | grep -E 'Driver Version|CUDA Version'
Driver Version                            : 550.90.07
CUDA Version                              : 12.4
```

**EUD installation:**
```bash
sudo dpkg -i nvidia-diagnostic-local-repo-ubuntu2204-550.26-mode1_1.0-1_amd64.deb
sudo cp /var/nvidia-diagnostic-local-repo-ubuntu2204-550.26-mode1/nvidia-diagnostic-local-041C3BF3-keyring.gpg /usr/share/keyrings/
sudo apt update
sudo apt install nvidia-diagnostic-550
```

**EUD validation (run in isolation — all GPUs pass):**
```
$ dcgmi diag -r eud
+---------------------------+------------------------------------------------+
| Diagnostic                | Result                                         |
+===========================+================================================+
| DCGM Version              | 3.3.9                                          |
| Driver Version Detected   | 550.90.07                                      |
| GPU Device IDs Detected   | 2330,2330,2330,2330,2330,2330,2330,2330         |
| EUD Test Version          | eud.550.26                                     |
| EUD Test                  | Pass - All                                     |
+---------------------------+------------------------------------------------+
```

### Test 1: NCCL Without EUD (Baseline)

NCCL `all_reduce` on 4 nodes / 32 GPUs with 8 GB messages — rock-solid ~390 GB/s bus bandwidth:

```
NCCL version 2.22.3+cuda12.5
#       size         count      type   redop     time   algbw   busbw      time   algbw   busbw
#        (B)    (elements)                        (us)  (GB/s)  (GB/s)      (us)  (GB/s)  (GB/s)
  8589934592    2147483648     float     sum    42664  201.34  390.09    42750  200.93  389.31
  8589934592    2147483648     float     sum    42638  201.46  390.33    42635  201.48  390.36
  8589934592    2147483648     float     sum    42680  201.26  389.95    42727  201.04  389.52
  8589934592    2147483648     float     sum    42831  200.56  388.58    42578  201.75  390.88
  ...
  8589934592    2147483648     float     sum    42660  201.36  390.13    42709  201.13  389.68
```

Consistent, with negligible variance (< 1%).

### Test 2: NCCL With Concurrent GPU Stress Test

I launched EUD simultaneously across all nodes while NCCL was running:

```bash
parallel-ssh -h ~/hostfile -t 600 -i "sudo dcgmi diag -r eud"
```

The impact was dramatic:

```
NCCL version 2.22.3+cuda12.5
#       size         count      type   redop     time   algbw   busbw      time   algbw   busbw
#        (B)    (elements)                        (us)  (GB/s)  (GB/s)      (us)  (GB/s)  (GB/s)
  8589934592    2147483648     float     sum    66340  129.48  250.87   443092   19.39   37.56
  8589934592    2147483648     float     sum   283475   30.30   58.71   207513   41.39   80.20
  8589934592    2147483648     float     sum   507865   16.91   32.77    47528  180.73  350.17
  8589934592    2147483648     float     sum   262626   32.71   63.37   443742   19.36   37.51
  ...
  8589934592    2147483648     float     sum    42905  200.21  387.90    42754  200.92  389.27
  ...
  8589934592    2147483648     float     sum   176794   48.59   94.14   180657   47.55   92.12
  ...
  8589934592    2147483648     float     sum    81973  104.79  203.03    66950  128.30  248.59
```

Bus bandwidth swung wildly between **~33 GB/s and ~390 GB/s** — an order-of-magnitude variation within the same run.

## Analysis

The erratic bandwidth pattern is a direct consequence of GPU resource contention between EUD and NCCL:

- **High-bandwidth iterations (~350–390 GB/s):** EUD is between test phases or performing lightweight checks. NCCL has uncontested access to GPU compute, memory, NVLink, and IB RDMA resources.
- **Low-bandwidth iterations (~30–90 GB/s):** EUD is actively stressing one or more GPUs — saturating memory bandwidth, exercising compute units, or running NVLink stress tests. NCCL operations on the affected GPUs stall, and because `all_reduce` is a **synchronous collective**, the slowest GPU across all 32 devices dictates the iteration time.

The pattern appears random because EUD's internal test phases are not synchronized with NCCL iterations.

**A note on the reported numbers:** The customer ran their NCCL test with `-i 1000` (1000 iterations), which causes `nccl-tests` to report a single **averaged** result per message size. This is why they observed a steady ~170 GB/s — the high and low iterations were averaged together, masking the bimodal distribution. My reproduction used the default per-iteration output, which exposes the raw swings between ~33 GB/s and ~390 GB/s. Both views are consistent: the averaged ~170 GB/s is exactly what you'd expect from a mixture of near-line-rate iterations and severely degraded ones.

## Key Takeaways

1. **GPU stress diagnostics must not run concurrently with production workloads.** Despite appearing to use minimal resources in aggregate metrics, tools like EUD take exclusive control of GPU subsystems during their stress phases, causing severe interference with NCCL collectives and any workload dependent on GPU-to-GPU communication.

2. **The synchronous nature of collectives amplifies the impact.** Even a single GPU being held by a stress test on one node will stall the entire `all_reduce` across all participating GPUs, making the degradation cluster-wide.

3. **Always check for background GPU processes** when investigating intermittent performance regressions on GPU clusters — particularly processes that are "always on" and assumed to be benign.
