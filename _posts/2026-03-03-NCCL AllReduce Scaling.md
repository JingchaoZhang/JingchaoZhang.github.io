---
layout: single
title: "NCCL All-Reduce Scaling: From 2 to 64 H100 Nodes on Azure"
author_profile: false
---

## Introduction

How does NCCL `all_reduce` really scale as you add more GPU nodes? Most benchmarks you see online test one or two configurations. In this post, I scale from 2 to 64 nodes (16 to 512 H100 GPUs) on Azure `Standard_ND96isr_H100_v5` VMs connected with 8x 400 Gb/s InfiniBand (ConnectX-7), and present real measured bandwidth across the full sweep.

This is part of a hands-on series where I build, validate, and benchmark an HPC/AI cluster from scratch — including the messy parts like failing health checks and nodes that need reboots.

## Test Environment

| Component | Detail |
|-----------|--------|
| **VM SKU** | Standard_ND96isr_H100_v5 |
| **GPUs per node** | 8x NVIDIA H100 80GB HBM3 |
| **Interconnect** | 8x 400 Gb/s NDR InfiniBand (ConnectX-7) |
| **Intra-node** | NVLink 4th generation, 900 GB/s bisection |
| **Deployment** | Azure VMSS, 66 nodes provisioned |
| **OS / Kernel** | Ubuntu 22.04, 5.15.0-1103-azure |
| **NCCL Tests** | `/opt/nccl-tests/build/all_reduce_perf` |
| **MPI** | HPC-X v2.25.1 (OpenMPI-based) |
| **Topology file** | `/opt/microsoft/ndv5-topo.xml` |

## Phase 1: Node Health Checks — The Reality of Large Clusters

Before running any NCCL benchmarks, you need to validate every node. At scale, hardware failures are not the exception — they're the expectation. I ran [Azure NHC (Node Health Checks)](https://github.com/Azure/azurehpc-health-checks) across all 66 nodes using `parallel-ssh`:

```bash
parallel-ssh -i -t 0 -h hostfile -x "-o StrictHostKeyChecking=no" \
  "sudo /opt/azurehpc/run-health-checks.sh" &> all_nodes_NHC.log
```

To find which nodes failed:

```bash
grep -B50 'ERROR:  nhc:' all_nodes_NHC.log \
  | grep -oP '\[(SUCCESS|FAILURE)\] \K\S+' | sort -u
```

**Result: 3 out of 66 nodes failed health checks** (95.5% pass rate).

### Failure 1: GPU ECC Error — vmss8EYIHY

```
ERROR:  nhc:  Health check failed:  check_ecc: GPU id 3: SRAM Uncorrectable ECC
error count detected, (0,1). FaultCode: NHC2019
```

GPU 3 on this node had an uncorrectable SRAM ECC error. SRAM ECC errors in the "uncorrectable" category are serious — they mean the GPU's error-correcting code could not fix a bit flip in the on-chip memory (registers, caches, or shared memory). The `(0,1)` indicates the volatile (current-session) count is 0 but the aggregate (lifetime) count is 1, meaning the error occurred in a previous session and has been sticky across resets.

Note that NHC stopped early on this node — it never reached the NCCL, NVLink, or IB bandwidth tests. The ECC check is ordered before the more expensive tests as a fast-fail optimization.

**Impact**: Any computation scheduled on GPU 3 risks silent data corruption. This node must be excluded from all benchmarks.

### Failure 2: IB Bandwidth Degradation (All 8 Ports) — vmssBU4WM4

```
ERROR:  nhc:  Health check failed:  check_ib_bw_gdr: ib_write_bw, IB=mlx5_ib0,
  IB BW (expected > 380 Gbps, but measured 351.09 Gbps. FaultCode: NHC2003
ERROR:  nhc:  Health check failed:  check_ib_bw_gdr: ib_write_bw, IB=mlx5_ib1,
  IB BW (expected > 380 Gbps, but measured 351.06 Gbps. FaultCode: NHC2003
ERROR:  nhc:  Health check failed:  check_ib_bw_gdr: ib_write_bw, IB=mlx5_ib2,
  IB BW (expected > 380 Gbps, but measured 351.10 Gbps. FaultCode: NHC2003
ERROR:  nhc:  Health check failed:  check_ib_bw_gdr: ib_write_bw, IB=mlx5_ib3,
  IB BW (expected > 380 Gbps, but measured 351.04 Gbps. FaultCode: NHC2003
ERROR:  nhc:  Health check failed:  check_ib_bw_gdr: ib_write_bw, IB=mlx5_ib4,
  IB BW (expected > 380 Gbps, but measured 351.07 Gbps. FaultCode: NHC2003
ERROR:  nhc:  Health check failed:  check_ib_bw_gdr: ib_write_bw, IB=mlx5_ib5,
  IB BW (expected > 380 Gbps, but measured 350.93 Gbps. FaultCode: NHC2003
ERROR:  nhc:  Health check failed:  check_ib_bw_gdr: ib_write_bw, IB=mlx5_ib6,
  IB BW (expected > 380 Gbps, but measured 351.06 Gbps. FaultCode: NHC2003
ERROR:  nhc:  Health check failed:  check_ib_bw_gdr: ib_write_bw, IB=mlx5_ib7,
  IB BW (expected > 380 Gbps, but measured 351.14 Gbps. FaultCode: NHC2003
```

All 8 InfiniBand ports on this node measured ~351 Gbps against a 380 Gbps threshold (healthy nodes measured ~392 Gbps). The fact that *all 8 ports* are affected uniformly — each reading within a tight band of 350.93–351.14 Gbps — strongly suggests this is not a cable or switch port issue. When individual cables degrade, you typically see one or two ports affected, not all eight.

The uniform degradation pattern across every port points to a host-side issue, likely the PCIe bus or a firmware/driver state problem. A soft reboot often resolves this.

**Impact**: ~10% IB bandwidth reduction across all ports. While GPU/NCCL/NVLink checks all passed (NCCL intra-node scored 482.6 GB/s), any multi-node workload touching this node would see degraded inter-node communication.

### Failure 3: IB Bandwidth Degradation (2 of 8 Ports) — vmss1P6OZ0

```
ERROR:  nhc:  Health check failed:  check_ib_bw_gdr: ib_write_bw, IB=mlx5_ib7,
  IB BW (expected > 380 Gbps, but measured 321.24 Gbps. FaultCode: NHC2003
ERROR:  nhc:  Health check failed:  check_ib_bw_gdr: ib_write_bw, IB=mlx5_ib0,
  IB BW (expected > 380 Gbps, but measured 321.27 Gbps. FaultCode: NHC2003
```

Only 2 of 8 ports are affected, but the degradation is more severe: ~321 Gbps versus the expected >380 Gbps (~18% below threshold). The remaining 6 ports are healthy at ~392 Gbps. This pattern — a subset of ports with deeper degradation — is more consistent with specific cable or switch port issues, though a reboot is still worth trying first.

**Impact**: Two GPUs on this node would have significantly slower inter-node communication, creating hot spots during collective operations.

### Summary Table

| Node | Error Type | Fault Code | Details | Severity |
|------|-----------|------------|---------|----------|
| vmss8EYIHY | GPU ECC | NHC2019 | GPU 3: SRAM Uncorrectable ECC (0,1) | High — data corruption risk |
| vmssBU4WM4 | IB Bandwidth | NHC2003 | All 8 ports: ~351 Gbps (expect >380) | Medium — 10% IB loss |
| vmss1P6OZ0 | IB Bandwidth | NHC2003 | 2 ports: ~321 Gbps (expect >380) | Medium — 18% IB loss on 2 ports |

**63 out of 66 nodes passed all health checks** and are available for NCCL scaling experiments.

### Remediation: Soft Reboot

For the IB bandwidth failures, the standard first step is a soft reboot. ECC errors may or may not clear after reboot — if the uncorrectable count persists, the GPU needs to be replaced (which on Azure means reimaging or replacing the VM).

```bash
# Reboot the 3 failed nodes
parallel-ssh -i -h failed_nodes.txt -x "-o StrictHostKeyChecking=no" \
  "sudo reboot"
```

After reboot, I re-ran NHC on just these 3 nodes:

```bash
parallel-ssh -i -h hostfile_bad -t 0 \
  "bash ~/azurehpc-health-checks/run-health-checks.sh 2>&1 | tee health.log"
```

**Post-reboot results:**

| Node | Before Reboot | After Reboot | Verdict |
|------|--------------|--------------|---------|
| vmss1P6OZ0 | 2 IB ports at ~321 Gbps | All 8 ports at ~392 Gbps | **Fixed** |
| vmssBU4WM4 | All 8 IB ports at ~351 Gbps | All 8 ports at ~392 Gbps | **Fixed** |
| vmss8EYIHY | GPU 3 ECC error (0,1) | GPU 3 ECC error (0,1) | **Still failing** |

The IB bandwidth issues on both nodes were completely resolved by a soft reboot — confirming these were transient host-side state issues, not hardware faults. Both nodes now report full line rate (~392 Gbps per port) and passed all other checks including intra-node NCCL all-reduce (482.6 GB/s).

The ECC error on vmss8EYIHY persists as expected. The `(0,1)` count means the volatile counter (current session) is 0, but the aggregate (lifetime) counter is 1. This aggregate counter is sticky across reboots — it only resets with a GPU reset at the driver level or when the VM is redeployed to fresh hardware. Since this is the head node, I will exclude it from the GPU compute hostfile but continue using it for orchestration (running `mpirun`, `parallel-ssh`, etc.).

**Final tally: 65 out of 66 nodes healthy and ready for NCCL scaling experiments.**

## Phase 2: First 64-Node Run — Houston, We Have a Problem

With 64 healthy nodes selected (excluding vmss8EYIHY due to the ECC error), I ran the NCCL all-reduce benchmark using our standard test configuration:

```bash
mpirun -np 512 --map-by ppr:8:node -hostfile hostfile_good \
  -x UCX_TLS=rc -x UCX_NET_DEVICES=mlx5_ib0:1 \
  -x NCCL_MIN_NCHANNELS=32 -x NCCL_IB_QPS_PER_CONNECTION=4 \
  -x NCCL_P2P_NET_CHUNKSIZE=524288 -x NCCL_PXN_DISABLE=1 \
  -x NCCL_TOPO_FILE=/opt/microsoft/ndv5-topo.xml \
  -x NCCL_IGNORE_CPU_AFFINITY=1 \
  /opt/nccl-tests/build/all_reduce_perf -b 8G -e 8G -g 1 -t 1 -i 0 -R 1
```

The `-i 0` flag prints every iteration individually (no averaging). Here's what came back:

```
#       size         count      type   redop     time   algbw   busbw  #wrong
#        (B)    (elements)                        (us)  (GB/s)  (GB/s)
  8589934592    2147483648     float     sum    47850   179.52  358.33       0
  8589934592    2147483648     float     sum   100855    85.17  170.01       0
  8589934592    2147483648     float     sum    97920    87.72  175.10       0
  8589934592    2147483648     float     sum    79879   107.54  214.65       0
  8589934592    2147483648     float     sum   117113    73.35  146.41       0
  8589934592    2147483648     float     sum    74107   115.91  231.37       0
  8589934592    2147483648     float     sum    93984    91.40  182.44       0
  8589934592    2147483648     float     sum   102055    84.17  168.01       0
  8589934592    2147483648     float     sum    86831    98.93  197.47       0
  8589934592    2147483648     float     sum    89917    95.53  190.69       0
  ...
  8589934592    2147483648     float     sum   131082    65.53  130.81       0
  8589934592    2147483648     float     sum    94940    90.48  180.60       0
  8589934592    2147483648     float     sum   130218    65.97  131.67       0
  8589934592    2147483648     float     sum    82236   104.45  208.50       0
  8589934592    2147483648     float     sum   117674    73.00  145.71       0
  8589934592    2147483648     float     sum   104503    82.20  164.07       0
  8589934592    2147483648     float     sum    97453    88.14  175.94       0
```

**This is terrible.** Out-of-place busbw swings wildly between **130 and 358 GB/s** across iterations. For 64 nodes with 8x 400 Gb/s IB per node, we should see stable numbers above 350 GB/s. Instead we're seeing:

- **Best iteration**: 358 GB/s (first iteration — before congestion builds up)
- **Worst iteration**: 126 GB/s (in-place)
- **Median**: ~180 GB/s
- **Coefficient of variation**: ~30%

This is not a stable fabric. Something is causing massive jitter.

### What Could Be Wrong?

Looking at the test parameters, I noticed two things:

1. **No warmup iterations** — NCCL needs time to establish QP connections and set up transport. The first few iterations include setup overhead, and without warmup the results are noisy from the start.

2. **No adaptive routing (`NCCL_IB_SL=2`)** — The script had this commented out. On Azure's NDR IB fabric with fat-tree topology, adaptive routing (`SL=2`) allows the subnet manager to distribute traffic across multiple paths. Without it, all flows between the same pair of nodes take the same static path, which at 64 nodes creates severe congestion hot spots on specific spine switches.

At smaller node counts (2–16 nodes), the number of IB flows is low enough that static routing works fine. But at 64 nodes, 512 GPUs are generating 3,200+ simultaneous flows — enough to saturate individual switch ports and create congestion cascades.

### The Plan: Systematic Diagnosis

Rather than guessing, I decided to run a systematic sweep:

1. **Phase A (Baseline)**: Sweep from 2 to 64 nodes *without* adaptive routing to find exactly where the instability begins
2. **Phase B (Adaptive Routing)**: Same sweep *with* `NCCL_IB_SL=2` to measure the improvement
3. Both phases use 20 warmup + 50 measured iterations for statistically meaningful results

```bash
# The sweep script handles everything:
nohup bash nccl_sweep.sh hostfile_good > nccl_sweep.log 2>&1 &
```

### Sweep Results: 2 → 64 Nodes

The sweep ran 14 tests total (7 node counts × 2 phases), each with 20 warmup + 50 measured iterations at 8 GB message size. Here are the results:

**Avg Bus Bandwidth (GB/s) — 8 GB All-Reduce**

| Nodes | GPUs | Baseline (SL=0) | Adaptive (SL=2) | Δ |
|-------|------|-----------------|-----------------|---|
| 2 | 16 | 482.0 | 482.2 | +0.0% |
| 4 | 32 | 350.3 | 344.9 | −1.5% |
| 8 | 64 | 373.8 | 350.8 | −6.2% |
| 16 | 128 | 274.8 | 309.2 | **+12.5%** |
| 32 | 256 | 243.4 | 284.3 | **+16.8%** |
| 48 | 384 | 228.9 | 243.5 | **+6.4%** |
| 64 | 512 | 235.0 | 251.7 | **+7.1%** |

**Detailed Out-of-Place / In-Place Breakdown (busbw in GB/s)**

| Nodes | Baseline OOP | Baseline IP | Adaptive OOP | Adaptive IP |
|-------|-------------|-------------|--------------|-------------|
| 2 | 482.0 | 482.0 | 482.2 | 482.2 |
| 4 | 335.3 | 365.3 | 331.3 | 358.6 |
| 8 | 392.3 | 355.4 | 350.1 | 351.6 |
| 16 | 253.0 | 296.7 | 305.7 | 312.8 |
| 32 | 240.2 | 246.7 | 259.8 | 308.9 |
| 48 | 229.2 | 228.5 | 213.2 | 273.9 |
| 64 | 223.5 | 246.4 | 181.5 | 321.8 |

### Analysis: Where Does Scaling Break Down?

Several patterns emerge from this data:

**1. The 2-node baseline is near-ideal.** At 482 GB/s busbw with 2 nodes (16 GPUs), we're close to the theoretical limit for ring all-reduce over 8x 400 Gb/s IB links: `8 × 400 / 8 × (15/16) = 375 GB/s` for network bandwidth, but NVLink enables intra-node transfers at 900 GB/s, so the effective busbw exceeds the per-link network limit. This confirms the hardware is healthy.

**2. The first major drop happens at 4 nodes.** From 482 → 350 GB/s (−27%) when going from 2 to 4 nodes. This is the "multi-hop penalty" — at 2 nodes, every IB flow has a single switch hop. At 4+ nodes, some flows must traverse spine switches, adding latency and contention.

**3. Adaptive routing helps at 16+ nodes, but not as much as expected.** SL=2 provides a clear benefit starting at 16 nodes (+12.5%) and peaking at 32 nodes (+16.8%). However, even with adaptive routing, 64-node busbw is only 252 GB/s — well below the 350+ GB/s we'd expect from properly tuned H100 clusters.

**4. The OOP/IP divergence at large scale is striking.** At 64 nodes with adaptive routing, out-of-place drops to 181.5 GB/s while in-place reaches 321.8 GB/s — a **77% gap**. This asymmetry suggests memory allocation pressure: out-of-place requires a separate output buffer, which at 8 GB × 512 GPUs may be causing CUDA memory allocation overhead or fragmenting the memory pool, affecting DMA registration performance.

**5. PXN is not the issue — this is a rail-optimized topology.** My initial hypothesis was that `NCCL_PXN_DISABLE=1` was the bottleneck. However, ND96isr_H100_v5 uses a **rail-optimized topology** where each GPU has a dedicated 1:1 mapping to its own IB NIC. PXN (PCIe cross-node relay) reduces hops when GPUs *don't* have direct NIC access — but with rail-optimized, every GPU already does. Testing confirmed this: re-running at 64 nodes with `NCCL_PXN_DISABLE=0` produced **even worse results** — busbw oscillating between 120–332 GB/s with enormous jitter:

```
# NCCL_PXN_DISABLE=0 at 64 nodes — worse, not better
#       size         count      type   redop     time   algbw   busbw     time   algbw   busbw
  8589934592    2147483648     float     sum    75242  114.16  227.88   113131   75.93  151.56
  8589934592    2147483648     float     sum   142248   60.39  120.54   122715   70.00  139.73
  8589934592    2147483648     float     sum    62079  138.37  276.20    77438  110.93  221.42
  ...  (wild swings from 120 to 332 GB/s)
```

Enabling PXN on a rail-optimized topology actually adds unnecessary indirection — PXN tries to relay traffic through peer GPUs' NICs, creating extra hops and contention when every GPU already has the shortest possible path. The original `NCCL_PXN_DISABLE=1` setting was correct.

**6. The real culprit: network congestion from bad nodes or switch-level issues.** The wild per-iteration jitter (not just low throughput) is the key diagnostic signal. Consistent low bandwidth would suggest a configuration issue; *variable* bandwidth that swings 2–3× between iterations points to congestion — specific nodes or switch paths that intermittently saturate and cause backpressure across the fabric. The next step is to isolate which nodes are causing the congestion using a binary search approach.

### Scaling Efficiency

To quantify the degradation, here's the scaling efficiency relative to the 2-node baseline:

| Nodes | Ideal busbw | Baseline (SL=0) | Efficiency | Adaptive (SL=2) | Efficiency |
|-------|-------------|-----------------|------------|-----------------|------------|
| 2 | 482 | 482.0 | 100% | 482.2 | 100% |
| 4 | 482 | 350.3 | 72.7% | 344.9 | 71.6% |
| 8 | 482 | 373.8 | 77.6% | 350.8 | 72.8% |
| 16 | 482 | 274.8 | 57.0% | 309.2 | 64.2% |
| 32 | 482 | 243.4 | 50.5% | 284.3 | 59.0% |
| 48 | 482 | 228.9 | 47.5% | 243.5 | 50.5% |
| 64 | 482 | 235.0 | 48.8% | 251.7 | 52.2% |

At 64 nodes, we're only achieving **~50% scaling efficiency**. Published Azure benchmarks for ND H100 v5 at similar scale typically report 350–400 GB/s busbw. With PXN ruled out and adaptive routing already enabled, the evidence points to network-level congestion from specific bad nodes or switch paths.

## Phase 3: Hunting Bad Nodes — Binary Search and Pair Testing

When you have 64 nodes and inconsistent NCCL performance, the challenge is isolating *which* nodes are causing the problem. I used two complementary approaches: a **binary search** to find bad groups, and a **pair test** to rank every individual node.

### Approach 1: Binary Search (nccl_bisect.sh)

The idea is simple: split nodes into halves, test each half independently, and recursively narrow down the problematic group. If both halves perform similarly, the issue is fabric-wide rather than localized.

```bash
nohup bash nccl_bisect.sh hostfile_good 4 > bisect.log 2>&1 &
```

**Bisect results (recursive split from 64 → 2 nodes):**

| Level | Group Size | Busbw Range | Spread |
|-------|-----------|-------------|--------|
| Full | 64 nodes | 203 GB/s | baseline |
| 32+32 split | 32 nodes | 304–315 GB/s | <5% — both halves similar |
| 16-node groups | 16 nodes | ~309 GB/s | all groups similar |
| 8-node groups | 8 nodes | 357–381 GB/s | all groups healthy |
| 2-node pairs | 2 nodes | 447–484 GB/s | all pairs healthy |

**Key finding: there is no single bad group.** At every split level, both halves performed within 5% of each other. The bisect script flagged every split as "FABRIC-WIDE" rather than localizing to one side. Individual 2-node pairs all achieved 447–484 GB/s — well above the 380 Gbps NHC threshold.

This means the degradation is **emergent at scale**: it doesn't exist in any subset of 32 or fewer nodes, but appears once you cross a critical mass of simultaneous flows on the fabric.

### Approach 2: Individual Node Ranking (nccl_pairtest.sh)

Since the bisect couldn't isolate bad groups, I built a pair-test script that measures every node individually against a fixed reference node. Each node is tested as a 2-node pair (16 GPUs), giving a clean per-node ranking:

```bash
bash nccl_pairtest.sh hostfile_good 8
# Tests each of 63 nodes against vmss1P6OZ0 (reference)
# Parameters: 8 GB message, 10 warmup + 30 measured iterations
# Total runtime: ~17 minutes for all 63 nodes
```

**Full rankings (best → worst):**

| Rank | Node | Busbw (GB/s) | | Rank | Node | Busbw (GB/s) |
|------|------|-------------|---|------|------|-------------|
| #1 | vmssPRQ0UI | 484.6 | | #33 | vmssQF6J0Z | 483.3 |
| #2 | vmssKRLAAI | 484.6 | | #34 | vmss14GB0H | 483.2 |
| #3 | vmssULS7RH | 484.5 | | #35 | vmssDHM7KW | 483.2 |
| #4 | vmssYHO48J | 484.4 | | #36 | vmssC594SO | 483.1 |
| #5 | vmssM4W53P | 484.2 | | #37 | vmssG33UBF | 483.1 |
| #6 | vmssLQ1K7I | 484.2 | | #38 | vmssST7QJU | 483.0 |
| #7 | vmss6RBNRQ | 484.2 | | #39 | vmssXJURFT | 483.0 |
| #8 | vmssORMSFU | 484.2 | | #40 | vmss18XTSZ | 482.9 |
| #9 | vmss36ZC3H | 484.1 | | #41 | vmssSN0OJU | 482.8 |
| #10 | vmssWIU0TW | 484.1 | | #42 | vmss3HOCA3 | 482.8 |
| #11 | vmssPZ2NP3 | 484.1 | | #43 | vmssT0DAH9 | 482.8 |
| #12 | vmssCLQONY | 484.1 | | #44 | vmssBU4WM4 | 482.6 |
| #13 | vmssA9LRGP | 484.0 | | #45 | vmssD2RJZ7 | 482.5 |
| #14 | vmssGDYRS7 | 484.0 | | #46 | vmss8WQTVM | 482.5 |
| #15 | vmssKVDVDM | 483.9 | | #47 | vmssGWUAIA | 482.0 |
| #16 | vmss0KV98K | 483.8 | | #48 | vmssTU4DED | 482.0 |
| #17 | vmssXTVQ9P | 483.8 | | #49 | vmssM0L1TW | 481.9 |
| #18 | vmssHPOKQW | 483.7 | | #50 | vmssU1ZIUQ | 481.9 |
| #19 | vmss70MXOJ | 483.7 | | #51 | vmssYIQ30G | 481.4 |
| #20 | vmss4G7YWX | 483.7 | | #52 | vmss27Z8Y3 | 481.0 |
| #21 | vmssO6BFK7 | 483.7 | | #53 | vmssR7ROF1 | 480.8 |
| #22 | vmssP8SP3P | 483.7 | | #54 | vmss142A5O | **477.4** |
| #23 | vmssGV70CJ | 483.7 | | #55 | vmssVWTZGG | **457.3** |
| #24 | vmssP7OA9N | 483.6 | | #56 | vmssXWTZY4 | **454.7** |
| #25 | vmssX9PI9I | 483.6 | | #57 | vmssDWKXEI | **450.7** |
| #26 | vmssYQ2GK5 | 483.6 | | #58 | vmssOGJHWA | **448.9** |
| #27 | vmssEOM8CR | 483.6 | | #59 | vmssL72NDU | **448.0** |
| #28 | vmssFEUDWG | 483.6 | | #60 | vmssGO3XCF | **447.5** |
| #29 | vmssCFOHQQ | 483.5 | | #61 | vmssTA30SB | **441.4** |
| #30 | vmssSUJHMO | 483.5 | | #62 | vmssWW94RX | **439.0** |
| #31 | vmss58XZZS | 483.4 | | #63 | vmssNUU8H7 | **435.5** |
| #32 | vmssHB3SG8 | 483.3 | | | *ref: vmss1P6OZ0* | |

The distribution reveals a clear bimodal pattern:

| Category | Nodes | Busbw Range | Spread |
|----------|-------|-------------|--------|
| **Healthy** | 53 | 480.8–484.6 GB/s | 0.8% |
| **Marginal** | 1 (vmss142A5O) | 477.4 GB/s | −1.5% |
| **Degraded** | 9 | 435.5–457.3 GB/s | −5.6% to −10.1% |

All 9 degraded nodes **passed NHC** (which requires >380 Gbps per port). They're not "broken" — they're *marginal*. Their IB bandwidth is 6–10% below the best nodes, well above the health check threshold but consistently at the bottom of the pack.

### The Critical Test: Does Exclusion Help?

Here's the moment of truth. If these marginal nodes are causing the fabric congestion, removing them should dramatically improve the multi-node result:

| Configuration | Nodes | GPUs | Busbw (GB/s) |
|--------------|-------|------|-------------|
| Full cluster | 64 | 512 | ~203–252 |
| Exclude 8 worst | 56 | 448 | **248.7** |
| Exclude 4 worst | 60 | 480 | **229.4** |

**Excluding the worst nodes barely helps.** Even after removing 8 marginal nodes (12.5% of the cluster), the clean 56-node cluster only achieves 249 GB/s — essentially the same as the full 64-node cluster. Excluding just 4 gets even worse at 229 GB/s.

This is the most important finding of this entire investigation: **the performance degradation at scale is not caused by individual bad nodes.** It's an emergent property of the network fabric under load.

### Approach 3: GPU Thermal Throttling Test (dcgmproftester13)

The pair test measures IB network performance, but what about the GPUs themselves? A thermally throttling GPU slows down its clock during collective operations, and since `all_reduce` is synchronous across all ranks, **one slow GPU stalls all 512**. To test this, I ran `dcgmproftester13` (from NVIDIA DCGM 4.5.2) across all nodes — a 120-second sustained FP16 Tensor Core stress test (test ID 1004) that pushes GPUs to their thermal limits:

```bash
parallel-ssh -i -t 0 -p 66 -h hostfile_good \
  'dcgmproftester13 --no-dcgm-validation -t 1004 -d 120 2>&1 | tee thermal_results.$(hostname).1004.120.$(date +%Y-%m-%d.%Hh%Mm%Ss).log'
```

During the test, I monitored GPU temperatures and clock throttling events via `nvidia-smi`. Any node where a GPU triggered HW Thermal Slowdown (hardware-enforced clock reduction at ~83°C) or SW Thermal Slowdown (driver-initiated power capping) was flagged as failing.

**Result: 9 out of 65 nodes exhibited thermal throttling** (13.8% failure rate):

| Node | GPU | Throttle Type |
|------|-----|--------------|
| vmss142A5O | GPU 5 | HW Thermal Slowdown |
| vmss1P6OZ0 | GPU 2 | HW Thermal Slowdown |
| vmssLQ1K7I | GPU 11 | HW Thermal Slowdown |
| vmssGDYRS7 | GPU 11 | HW Thermal Slowdown |
| vmss58XZZS | GPU 11 | HW Thermal Slowdown |
| vmssXWTZY4 | GPU 0 | SW Thermal Slowdown |
| vmssR7ROF1 | GPU 9 | SW Thermal Slowdown |
| vmssKRLAAI | GPU 9 | SW Thermal Slowdown |
| vmssA9LRGP | GPU 3 | SW Thermal Slowdown |

**HW vs. SW Thermal Slowdown**: HW Thermal Slowdown is more severe — the GPU hardware itself forces clock reduction when junction temperature exceeds the hard limit (~83°C on H100). SW Thermal Slowdown is a softer, driver-level power cap that kicks in earlier to prevent reaching the hardware limit. Both reduce compute throughput, but HW throttling is a stronger indicator of cooling issues (inadequate airflow, fan failure, or thermal paste degradation).

**Interesting overlap with pair-test results**: Most of the thermally throttling nodes actually ranked as *healthy* in the NCCL pair test:

| Node | Thermal Result | NCCL Pair-Test Rank | Pair Busbw |
|------|---------------|--------------------:|-----------|
| vmssKRLAAI | SW Throttle (GPU 9) | #2 | 484.6 GB/s |
| vmssLQ1K7I | HW Throttle (GPU 11) | #6 | 484.2 GB/s |
| vmssA9LRGP | SW Throttle (GPU 3) | #13 | 484.0 GB/s |
| vmssGDYRS7 | HW Throttle (GPU 11) | #14 | 484.0 GB/s |
| vmss58XZZS | HW Throttle (GPU 11) | #31 | 483.4 GB/s |
| vmssR7ROF1 | SW Throttle (GPU 9) | #53 | 480.8 GB/s |
| vmss142A5O | HW Throttle (GPU 5) | #54 | 477.4 GB/s |
| vmssXWTZY4 | SW Throttle (GPU 0) | #56 | 454.7 GB/s |
| vmss1P6OZ0 | HW Throttle (GPU 2) | *ref node* | — |

This makes sense: the NCCL pair test is **network-bound** (IB bandwidth limited), so a GPU throttling 5–10% on compute doesn't visibly affect the 2-node busbw result. But during a large-scale all-reduce where compute and network overlap, even one thermally throttling GPU can create a straggler effect.

**Did excluding thermal nodes help?** I removed all 9 thermally throttling nodes and re-ran the 56-node NCCL sweep:

| Configuration | Nodes | GPUs | Busbw (GB/s) |
|--------------|-------|------|-------------|
| Full cluster (64 nodes) | 64 | 512 | ~203–252 |
| Exclude IB-degraded (pair test) | 56 | 448 | 248.7 |
| Exclude thermal throttlers | 56 | 448 | **~248** |

**Still no improvement.** Removing thermally throttling nodes produced essentially the same result as removing IB-degraded nodes. The thermal throttling is a real issue for these specific GPUs under sustained compute load, but it's not the root cause of the NCCL scaling degradation. The all-reduce at 8 GB message size is network-dominated, and the straggler effect from thermal throttling is dwarfed by the fabric congestion.

**However, thermal throttling matters for real workloads.** While it doesn't explain the NCCL busbw scaling cliff, thermally throttling GPUs *would* impact training throughput in production — every `all_reduce` in a training step waits for the slowest rank, and if that rank is also computing its backward pass on a throttled GPU, the compound effect is significant. These 9 nodes should be flagged for investigation (cooling, airflow, rack placement) regardless of the NCCL results.

### What's Actually Happening

Putting all the evidence together:

1. **Every node is individually healthy** — 2-node pairs all achieve 435–485 GB/s
2. **Small groups are healthy** — 8-node groups hit 357–381 GB/s
3. **The degradation emerges gradually** — from 482 GB/s (2 nodes) → 350 GB/s (4 nodes) → 309 GB/s (16 nodes) → 252 GB/s (64 nodes)
4. **Removing bad nodes doesn't fix it** — 56 "clean" nodes still get 249 GB/s
5. **Jitter is massive** — busbw swings 2–3× between iterations at 64 nodes

This pattern is consistent with **fabric-level congestion**: at 64 nodes, the 3,200+ simultaneous IB flows create hot spots on spine switches that adaptive routing (`SL=2`) can *reduce* but not eliminate. The marginal nodes (435–455 GB/s) likely have suboptimal switch placement or share congested uplinks, but they're a symptom, not the cause.

The root issue is that this particular VMSS deployment has nodes scattered across the fat-tree topology, and at 64-node scale, the aggregate bandwidth demand exceeds what the spine tier can provide without congestion. This is a well-known challenge of running at scale on shared cloud infrastructure — your performance depends not just on your nodes, but on how those nodes are *placed* in the physical network topology.

## Key Takeaways

1. **Always run NHC before benchmarking.** At 66 nodes, we caught 3 failures (4.5% failure rate). Two were fixable with a reboot; one required node exclusion. Running NCCL tests on unhealthy nodes wastes hours and produces misleading data.

2. **Warmup iterations matter.** The first few NCCL iterations include connection establishment overhead. Without warmup (`-w 0`), your "measured" busbw includes setup cost. Always use `-w 20` or more.

3. **Adaptive routing (`NCCL_IB_SL=2`) is essential at scale.** Static routing works fine at 2–8 nodes but causes 10–17% degradation at 16–64 nodes due to congestion hot spots on spine switches. On Azure NDR IB fabrics, always enable SL=2 for multi-node workloads.

4. **Understand your topology before tuning.** On rail-optimized VMs like ND96isr_H100_v5, `NCCL_PXN_DISABLE=1` is correct — PXN adds unnecessary hops. On non-rail-optimized topologies, PXN is beneficial. Blindly copying NCCL flags from one SKU to another can hurt performance.

5. **Not all degradation is node-level.** NHC catches broken nodes, but it can't detect fabric-level congestion that only manifests at scale. Pair tests and bisection are useful diagnostics, but if both come back clean, the problem is likely switch-level placement in the fat-tree topology.

6. **Jitter is the diagnostic signal.** Consistent low busbw points to configuration issues; *variable* busbw (swinging 2–3× between iterations) points to network congestion. If binary search can't isolate the source, it's fabric-wide.

7. **Measure systematically.** Sweeping from 2 to 64 nodes revealed that the scaling cliff starts at 4 nodes (multi-hop IB penalty) and steepens at 16+ nodes (congestion). The progression tells a story that no single data point can.

8. **NHC thresholds have a blind spot.** Nodes passing NHC (>380 Gbps) can still vary by 10% in pair-test performance (435 vs. 485 GB/s). These "marginal" nodes don't cause large-scale congestion on their own, but they indicate suboptimal hardware or placement. Consider tightening NHC thresholds for performance-critical workloads.

9. **Run thermal stress tests separately from network tests.** NHC and NCCL pair tests are network-focused — they won't catch GPUs that thermally throttle under sustained compute load. Running `dcgmproftester13 -t 1004 -d 120` across all nodes caught 9 thermally throttling GPUs (13.8% of the cluster) that NHC missed entirely. While thermal throttling didn't explain the NCCL scaling cliff (which is network-dominated), it matters for real training workloads where compute and communication overlap. Add DCGM thermal testing to your pre-benchmark validation checklist alongside NHC.
