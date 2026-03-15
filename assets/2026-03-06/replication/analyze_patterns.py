#!/usr/bin/env python3
"""Deep analysis of Dense vs MoE IB + GPU patterns for blog post."""
import json, os
import numpy as np

DATA = "/home/jingchao/Azure_Cluster_Test/VMSS/create_VMSS_noBastion/logs/pattern_export_healthy"
GB = 1e9

def load(name):
    with open(os.path.join(DATA, f"{name}.json")) as f:
        return json.load(f)["data"]["result"]

def extract_ib(results):
    ports = {}
    for r in results:
        port = r["metric"].get("ib_port", "")
        if not port.startswith("mlx5_ib"):
            continue
        ts = np.array([float(v[0]) for v in r["values"]])
        vals = np.array([float(v[1]) for v in r["values"]])
        ports[port] = (ts, vals)
    return ports

def extract_gpu(results):
    gpus = {}
    for r in results:
        gid = r["metric"].get("gpu_id", r["metric"].get("gpu", "?"))
        ts = np.array([float(v[0]) for v in r["values"]])
        vals = np.array([float(v[1]) for v in r["values"]])
        gpus[gid] = (ts, vals)
    return gpus

def analyze_workload(name, ib_xmit_data, ib_rcv_data, gpu_data):
    ib_xmit = extract_ib(ib_xmit_data)
    ib_rcv = extract_ib(ib_rcv_data)
    gpus = extract_gpu(gpu_data)
    
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    
    # Aggregate IB
    ref_ts = list(ib_xmit.values())[0][0]
    total_xmit = np.zeros_like(ref_ts)
    total_rcv = np.zeros_like(ref_ts)
    for p, (_, v) in ib_xmit.items():
        total_xmit += v
    for p, (_, v) in ib_rcv.items():
        total_rcv += v
    
    total_xmit_gb = total_xmit / GB
    total_rcv_gb = total_rcv / GB
    
    # GPU average
    gpu_avg = np.zeros_like(list(gpus.values())[0][1])
    for _, (_, v) in gpus.items():
        gpu_avg += v
    gpu_avg /= len(gpus)
    
    # Find active IB window
    ib_active = total_xmit_gb > 0.1  # >100 MB/s
    ib_indices = np.where(ib_active)[0]
    
    # Find active GPU window
    gpu_active = gpu_avg > 5  # >5%
    gpu_indices = np.where(gpu_active)[0]
    
    print(f"\n  IB Ports: {len(ib_xmit)} (mlx5_ib0-ib7)")
    print(f"  GPUs: {len(gpus)}")
    
    # IB stats
    print(f"\n  --- IB Throughput (total across all 8 ports) ---")
    if len(ib_indices) > 0:
        ib_start, ib_end = ib_indices[0], ib_indices[-1]
        ib_duration = ref_ts[ib_end] - ref_ts[ib_start]
        print(f"  Active window: {ib_duration:.0f}s ({len(ib_indices)} samples)")
        print(f"  Peak total xmit: {np.max(total_xmit_gb):.1f} GB/s")
        print(f"  Peak total rcv:  {np.max(total_rcv_gb):.1f} GB/s")
        print(f"  Avg during active (xmit): {np.mean(total_xmit_gb[ib_active]):.1f} GB/s")
        print(f"  Avg during active (rcv):  {np.mean(total_rcv_gb[ib_active]):.1f} GB/s")
    else:
        print(f"  No IB activity detected!")
    
    # Per-port breakdown
    print(f"\n  --- Per-Port Peak Xmit ---")
    for port_name in sorted(ib_xmit.keys()):
        _, vals = ib_xmit[port_name]
        peak = np.max(vals) / GB
        nz = np.sum(vals > 0)
        print(f"    {port_name}: peak={peak:.1f} GB/s, active_samples={nz}")
    
    # GPU stats
    print(f"\n  --- GPU Utilization ---")
    if len(gpu_indices) > 0:
        gpu_start, gpu_end = gpu_indices[0], gpu_indices[-1]
        gpu_duration = ref_ts[gpu_end] - ref_ts[gpu_start]
        print(f"  Active window: {gpu_duration:.0f}s ({len(gpu_indices)} samples)")
        print(f"  Peak avg util: {np.max(gpu_avg):.0f}%")
        print(f"  Mean during active: {np.mean(gpu_avg[gpu_active]):.0f}%")
    else:
        print(f"  No GPU activity detected!")
    
    # Per-GPU breakdown
    print(f"\n  --- Per-GPU Peak Util ---")
    for gid in sorted(gpus.keys()):
        _, vals = gpus[gid]
        peak = np.max(vals)
        nz = np.sum(vals > 5)
        print(f"    GPU {gid}: peak={peak:.0f}%, active_samples={nz}")
    
    # Correlation analysis: Compute/Communication overlap
    if len(ib_indices) > 0 and len(gpu_indices) > 0:
        print(f"\n  --- Compute vs Communication Overlap ---")
        both_active = ib_active & gpu_active
        ib_only = ib_active & ~gpu_active
        gpu_only = ~ib_active & gpu_active
        
        print(f"  Both IB+GPU active: {np.sum(both_active)} samples")
        print(f"  IB only (no GPU):   {np.sum(ib_only)} samples")
        print(f"  GPU only (no IB):   {np.sum(gpu_only)} samples")
        
        # GPU util during IB bursts
        if np.sum(ib_active) > 0:
            print(f"  GPU util during IB bursts: {np.mean(gpu_avg[ib_active]):.0f}% avg")
        # IB during GPU active
        if np.sum(gpu_active) > 0:
            print(f"  IB xmit during GPU active: {np.mean(total_xmit_gb[gpu_active]):.1f} GB/s avg")
    
    # Communication pattern: bursty vs sustained
    if len(ib_indices) > 0:
        print(f"\n  --- Communication Pattern ---")
        # Look at consecutive active samples
        diffs = np.diff(ib_indices)
        bursts = np.sum(diffs > 1) + 1  # number of separate bursts
        max_burst_len = 0
        current_burst = 1
        for d in diffs:
            if d == 1:
                current_burst += 1
            else:
                max_burst_len = max(max_burst_len, current_burst)
                current_burst = 1
        max_burst_len = max(max_burst_len, current_burst)
        print(f"  Number of separate IB bursts: {bursts}")
        print(f"  Longest continuous burst: {max_burst_len}s")
        print(f"  Duty cycle (IB active / total window): {len(ib_indices)/len(ref_ts)*100:.1f}%")

# Run analysis
print("Dense (Qwen 7B) vs MoE (Mixtral 8x7B) — Communication Pattern Analysis")
print(f"Data source: Moneo net_exporter on vmss72VTKQ (head node)")
print(f"Cluster: 2-node FSDP, Azure ND H100 v5, InfiniBand")

analyze_workload("Dense: Qwen2.5-7B (7.6B params, FSDP FULL_SHARD)",
                 load("qwen_ib_xmit"), load("qwen_ib_rcv"), load("qwen_gpu_util"))

analyze_workload("MoE: Mixtral-8x7B-v0.1 (46.7B params, FSDP FULL_SHARD)",
                 load("mixtral_ib_xmit"), load("mixtral_ib_rcv"), load("mixtral_gpu_util"))

# Key comparison
print("\n" + "="*60)
print("  KEY COMPARISON")
print("="*60)
q_xmit = extract_ib(load("qwen_ib_xmit"))
m_xmit = extract_ib(load("mixtral_ib_xmit"))
q_total = sum(v for _, (_, v) in q_xmit.items())
m_total = sum(v for _, (_, v) in m_xmit.items())
print(f"  Peak IB (all ports): Qwen={np.max(q_total)/GB:.0f} GB/s, Mixtral={np.max(m_total)/GB:.0f} GB/s")
print(f"  Ratio: Mixtral/Qwen = {np.max(m_total)/np.max(q_total):.1f}x")
