#!/usr/bin/env python3
"""
Dense (Qwen 7B) vs MoE (Mixtral 8x7B) — IB + GPU Communication Patterns
Data: Moneo net_exporter gauges (bytes/s per port) + dcgm_gpu_utilization
Node: vmss72VTKQ (head, 8x H100), 2-node FSDP, InfiniBand
"""
import json, os, sys
import numpy as np

# Try matplotlib with Agg backend for headless rendering
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

DATA = "/home/jingchao/Azure_Cluster_Test/VMSS/create_VMSS_noBastion/logs/pattern_export_healthy"
OUT  = DATA

def load_prom(path):
    with open(path) as f:
        d = json.load(f)
    return d["data"]["result"]

def extract_ib_timeseries(results):
    """Extract per-port IB timeseries, filtering to mlx5_ib* only.
    Returns: {port_name: (timestamps, values_bytes_per_sec)}
    """
    ports = {}
    for r in results:
        port = r["metric"].get("ib_port", "")
        if not port.startswith("mlx5_ib"):
            continue
        ts = np.array([float(v[0]) for v in r["values"]])
        vals = np.array([float(v[1]) for v in r["values"]])
        ports[port] = (ts, vals)
    return ports

def extract_gpu_timeseries(results):
    """Extract per-GPU utilization timeseries.
    Returns: {gpu_id: (timestamps, values_percent)}
    """
    gpus = {}
    for r in results:
        gpu_id = r["metric"].get("gpu_id", r["metric"].get("gpu", "unknown"))
        ts = np.array([float(v[0]) for v in r["values"]])
        vals = np.array([float(v[1]) for v in r["values"]])
        gpus[gpu_id] = (ts, vals)
    return gpus

def aggregate_ib(ports):
    """Sum all IB ports into total bytes/sec. Returns (relative_time_s, total_bytes_per_sec)."""
    if not ports:
        return np.array([]), np.array([])
    ref_port = list(ports.values())[0]
    ts = ref_port[0]
    total = np.zeros_like(ts)
    for _, (_, vals) in ports.items():
        total += vals
    t0 = ts[0]
    return ts - t0, total

def aggregate_gpu(gpus):
    """Average GPU util across all GPUs. Returns (relative_time_s, avg_percent)."""
    if not gpus:
        return np.array([]), np.array([])
    ref = list(gpus.values())[0]
    ts = ref[0]
    total = np.zeros_like(ts)
    for _, (_, vals) in gpus.items():
        total += vals
    avg = total / len(gpus)
    t0 = ts[0]
    return ts - t0, avg

def find_active_window(ts, vals, threshold_frac=0.05):
    """Find the time window where there's actual activity.
    Returns (start_idx, end_idx) with some padding.
    """
    if len(vals) == 0:
        return 0, 0
    max_val = np.max(vals)
    if max_val == 0:
        return 0, len(vals)
    active = vals > max_val * threshold_frac
    indices = np.where(active)[0]
    if len(indices) == 0:
        return 0, len(vals)
    pad = max(5, int(len(ts) * 0.05))
    start = max(0, indices[0] - pad)
    end = min(len(ts), indices[-1] + pad)
    return start, end

# ─── Load data ───
print("Loading data...")
qwen_xmit = extract_ib_timeseries(load_prom(f"{DATA}/qwen_ib_xmit.json"))
qwen_rcv  = extract_ib_timeseries(load_prom(f"{DATA}/qwen_ib_rcv.json"))
qwen_gpu  = extract_gpu_timeseries(load_prom(f"{DATA}/qwen_gpu_util.json"))

mix_xmit = extract_ib_timeseries(load_prom(f"{DATA}/mixtral_ib_xmit.json"))
mix_rcv  = extract_ib_timeseries(load_prom(f"{DATA}/mixtral_ib_rcv.json"))
mix_gpu  = extract_gpu_timeseries(load_prom(f"{DATA}/mixtral_gpu_util.json"))

print(f"  Qwen IB ports: {len(qwen_xmit)}, GPUs: {len(qwen_gpu)}")
print(f"  Mixtral IB ports: {len(mix_xmit)}, GPUs: {len(mix_gpu)}")

# ─── Aggregate ───
q_ib_t, q_ib_xmit = aggregate_ib(qwen_xmit)
_, q_ib_rcv = aggregate_ib(qwen_rcv)
q_gpu_t, q_gpu_avg = aggregate_gpu(qwen_gpu)

m_ib_t, m_ib_xmit = aggregate_ib(mix_xmit)
_, m_ib_rcv = aggregate_ib(mix_rcv)
m_gpu_t, m_gpu_avg = aggregate_gpu(mix_gpu)

# Convert to GB/s
GB = 1e9
q_ib_xmit_gb = q_ib_xmit / GB
q_ib_rcv_gb = q_ib_rcv / GB
m_ib_xmit_gb = m_ib_xmit / GB
m_ib_rcv_gb = m_ib_rcv / GB

# ─── Find active windows ───
q_ib_start, q_ib_end = find_active_window(q_ib_t, q_ib_xmit_gb)
q_gpu_start, q_gpu_end = find_active_window(q_gpu_t, q_gpu_avg)
# Use union of IB and GPU active windows
q_start = min(q_ib_start, q_gpu_start)
q_end = max(q_ib_end, q_gpu_end)

m_ib_start, m_ib_end = find_active_window(m_ib_t, m_ib_xmit_gb)
m_gpu_start, m_gpu_end = find_active_window(m_gpu_t, m_gpu_avg)
m_start = min(m_ib_start, m_gpu_start)
m_end = max(m_ib_end, m_gpu_end)

print(f"  Qwen active: {q_end-q_start}s, peak IB xmit: {np.max(q_ib_xmit_gb):.1f} GB/s total")
print(f"  Mixtral active: {m_end-m_start}s, peak IB xmit: {np.max(m_ib_xmit_gb):.1f} GB/s total")

# ════════════════════════════════════════════════════════════════
# FIGURE 1: Aggregate IB + GPU comparison (2x2)
# ════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("Dense (Qwen 7B) vs MoE (Mixtral 8×7B) — IB & GPU Patterns\n"
             "2-node FSDP on Azure ND H100 v5 (8× IB per node, 8× H100 per node)",
             fontsize=14, fontweight='bold')

# Convert active window times to relative from each window start
q_t_rel = q_ib_t[q_start:q_end] - q_ib_t[q_start]
m_t_rel = m_ib_t[m_start:m_end] - m_ib_t[m_start]

# Top-left: Qwen IB
ax = axes[0, 0]
ax.fill_between(q_t_rel, q_ib_xmit_gb[q_start:q_end], alpha=0.6, color='#2196F3', label='IB Xmit')
ax.fill_between(q_t_rel, q_ib_rcv_gb[q_start:q_end], alpha=0.4, color='#FF9800', label='IB Rcv')
ax.set_ylabel('Total IB Throughput (GB/s)')
ax.set_title('Dense (Qwen 7B) — IB Traffic', fontsize=12, fontweight='bold')
ax.legend(loc='upper right')
ax.set_xlim(0, q_t_rel[-1] if len(q_t_rel) else 1)
ax.grid(True, alpha=0.3)

# Top-right: Mixtral IB
ax = axes[0, 1]
ax.fill_between(m_t_rel, m_ib_xmit_gb[m_start:m_end], alpha=0.6, color='#2196F3', label='IB Xmit')
ax.fill_between(m_t_rel, m_ib_rcv_gb[m_start:m_end], alpha=0.4, color='#FF9800', label='IB Rcv')
ax.set_ylabel('Total IB Throughput (GB/s)')
ax.set_title('MoE (Mixtral 8×7B) — IB Traffic', fontsize=12, fontweight='bold')
ax.legend(loc='upper right')
ax.set_xlim(0, m_t_rel[-1] if len(m_t_rel) else 1)
ax.grid(True, alpha=0.3)

# Bottom-left: Qwen GPU
ax = axes[1, 0]
q_gpu_t_rel = q_gpu_t[q_start:q_end] - q_gpu_t[q_start]
ax.fill_between(q_gpu_t_rel, q_gpu_avg[q_start:q_end], alpha=0.6, color='#4CAF50')
ax.set_ylabel('Avg GPU Utilization (%)')
ax.set_xlabel('Time (s)')
ax.set_title('Dense (Qwen 7B) — GPU Utilization', fontsize=12, fontweight='bold')
ax.set_ylim(0, 105)
ax.set_xlim(0, q_gpu_t_rel[-1] if len(q_gpu_t_rel) else 1)
ax.grid(True, alpha=0.3)

# Bottom-right: Mixtral GPU
ax = axes[1, 1]
m_gpu_t_rel = m_gpu_t[m_start:m_end] - m_gpu_t[m_start]
ax.fill_between(m_gpu_t_rel, m_gpu_avg[m_start:m_end], alpha=0.6, color='#4CAF50')
ax.set_ylabel('Avg GPU Utilization (%)')
ax.set_xlabel('Time (s)')
ax.set_title('MoE (Mixtral 8×7B) — GPU Utilization', fontsize=12, fontweight='bold')
ax.set_ylim(0, 105)
ax.set_xlim(0, m_gpu_t_rel[-1] if len(m_gpu_t_rel) else 1)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(f"{OUT}/dense_vs_moe_patterns.png", dpi=150, bbox_inches='tight')
print(f"\nSaved: {OUT}/dense_vs_moe_patterns.png")

# ════════════════════════════════════════════════════════════════
# FIGURE 2: Per-port IB breakdown (shows individual NIC usage)
# ════════════════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(2, 2, figsize=(18, 12))
fig2.suptitle("Per-Port IB Throughput — Dense vs MoE\n"
              "Each line = one mlx5_ibN:1 port on vmss72VTKQ (head node)",
              fontsize=14, fontweight='bold')

colors = plt.cm.Set2(np.linspace(0, 1, 8))

# Top-left: Qwen per-port xmit
ax = axes2[0, 0]
for i, (port, (ts, vals)) in enumerate(sorted(qwen_xmit.items())):
    t_rel = ts[q_start:q_end] - ts[q_start]
    ax.plot(t_rel, vals[q_start:q_end] / GB, label=port.split(':')[0],
            color=colors[i % 8], alpha=0.8, linewidth=1)
ax.set_ylabel('IB Xmit (GB/s per port)')
ax.set_title('Dense (Qwen 7B) — Per-Port Xmit', fontsize=11, fontweight='bold')
ax.legend(fontsize=7, ncol=2, loc='upper right')
ax.set_xlim(0, q_t_rel[-1] if len(q_t_rel) else 1)
ax.grid(True, alpha=0.3)

# Top-right: Mixtral per-port xmit
ax = axes2[0, 1]
for i, (port, (ts, vals)) in enumerate(sorted(mix_xmit.items())):
    t_rel = ts[m_start:m_end] - ts[m_start]
    ax.plot(t_rel, vals[m_start:m_end] / GB, label=port.split(':')[0],
            color=colors[i % 8], alpha=0.8, linewidth=1)
ax.set_ylabel('IB Xmit (GB/s per port)')
ax.set_title('MoE (Mixtral 8×7B) — Per-Port Xmit', fontsize=11, fontweight='bold')
ax.legend(fontsize=7, ncol=2, loc='upper right')
ax.set_xlim(0, m_t_rel[-1] if len(m_t_rel) else 1)
ax.grid(True, alpha=0.3)

# Bottom-left: Qwen per-port rcv
ax = axes2[1, 0]
for i, (port, (ts, vals)) in enumerate(sorted(qwen_rcv.items())):
    t_rel = ts[q_start:q_end] - ts[q_start]
    ax.plot(t_rel, vals[q_start:q_end] / GB, label=port.split(':')[0],
            color=colors[i % 8], alpha=0.8, linewidth=1)
ax.set_ylabel('IB Rcv (GB/s per port)')
ax.set_xlabel('Time (s)')
ax.set_title('Dense (Qwen 7B) — Per-Port Rcv', fontsize=11, fontweight='bold')
ax.legend(fontsize=7, ncol=2, loc='upper right')
ax.set_xlim(0, q_t_rel[-1] if len(q_t_rel) else 1)
ax.grid(True, alpha=0.3)

# Bottom-right: Mixtral per-port rcv
ax = axes2[1, 1]
for i, (port, (ts, vals)) in enumerate(sorted(mix_rcv.items())):
    t_rel = ts[m_start:m_end] - ts[m_start]
    ax.plot(t_rel, vals[m_start:m_end] / GB, label=port.split(':')[0],
            color=colors[i % 8], alpha=0.8, linewidth=1)
ax.set_ylabel('IB Rcv (GB/s per port)')
ax.set_xlabel('Time (s)')
ax.set_title('MoE (Mixtral 8×7B) — Per-Port Rcv', fontsize=11, fontweight='bold')
ax.legend(fontsize=7, ncol=2, loc='upper right')
ax.set_xlim(0, m_t_rel[-1] if len(m_t_rel) else 1)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig2.savefig(f"{OUT}/dense_vs_moe_perport.png", dpi=150, bbox_inches='tight')
print(f"Saved: {OUT}/dense_vs_moe_perport.png")

# ════════════════════════════════════════════════════════════════
# FIGURE 3: Overlaid IB + GPU on same timeline (key insight figure)
# ════════════════════════════════════════════════════════════════
fig3, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
fig3.suptitle("IB vs GPU Utilization Correlation — Dense vs MoE\n"
              "Showing computation (GPU) vs communication (IB) overlap patterns",
              fontsize=14, fontweight='bold')

# Qwen overlaid
ax1_ib = ax1
ax1_gpu = ax1.twinx()
ax1_ib.fill_between(q_t_rel, q_ib_xmit_gb[q_start:q_end], alpha=0.5, color='#2196F3', label='IB Total Xmit')
ax1_gpu.plot(q_gpu_t_rel, q_gpu_avg[q_start:q_end], color='#E91E63', linewidth=2, alpha=0.8, label='GPU Util %')
ax1_ib.set_ylabel('IB Throughput (GB/s)', color='#2196F3')
ax1_gpu.set_ylabel('GPU Utilization (%)', color='#E91E63')
ax1_gpu.set_ylim(0, 105)
ax1.set_title('Dense (Qwen 7B FSDP) — Computation vs Communication', fontweight='bold')
ax1.set_xlabel('Time (s)')
ax1.grid(True, alpha=0.3)
# Combined legend
lines1, labels1 = ax1_ib.get_legend_handles_labels()
lines2, labels2 = ax1_gpu.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

# Mixtral overlaid
ax2_ib = ax2
ax2_gpu = ax2.twinx()
ax2_ib.fill_between(m_t_rel, m_ib_xmit_gb[m_start:m_end], alpha=0.5, color='#2196F3', label='IB Total Xmit')
ax2_gpu.plot(m_gpu_t_rel, m_gpu_avg[m_start:m_end], color='#E91E63', linewidth=2, alpha=0.8, label='GPU Util %')
ax2_ib.set_ylabel('IB Throughput (GB/s)', color='#2196F3')
ax2_gpu.set_ylabel('GPU Utilization (%)', color='#E91E63')
ax2_gpu.set_ylim(0, 105)
ax2.set_title('MoE (Mixtral 8×7B FSDP) — Computation vs Communication', fontweight='bold')
ax2.set_xlabel('Time (s)')
ax2.grid(True, alpha=0.3)
lines1, labels1 = ax2_ib.get_legend_handles_labels()
lines2, labels2 = ax2_gpu.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.tight_layout()
fig3.savefig(f"{OUT}/dense_vs_moe_overlay.png", dpi=150, bbox_inches='tight')
print(f"Saved: {OUT}/dense_vs_moe_overlay.png")

# ─── Summary stats ───
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
q_peak_xmit = np.max(q_ib_xmit_gb)
q_peak_rcv = np.max(q_ib_rcv_gb)
m_peak_xmit = np.max(m_ib_xmit_gb)
m_peak_rcv = np.max(m_ib_rcv_gb)

# Per-port peaks
q_port_peaks = [np.max(vals) / GB for _, (_, vals) in qwen_xmit.items()]
m_port_peaks = [np.max(vals) / GB for _, (_, vals) in mix_xmit.items()]

print(f"Dense (Qwen 7B):")
print(f"  Peak total IB xmit: {q_peak_xmit:.1f} GB/s  ({q_peak_xmit/8:.1f} GB/s avg/port)")
print(f"  Peak total IB rcv:  {q_peak_rcv:.1f} GB/s")
print(f"  Per-port peak range: {min(q_port_peaks):.1f} - {max(q_port_peaks):.1f} GB/s")
print(f"  GPU util during IB burst: {np.mean(q_gpu_avg[q_start:q_end][q_gpu_avg[q_start:q_end] > 0]):.0f}% avg")
print(f"")
print(f"MoE (Mixtral 8x7B):")
print(f"  Peak total IB xmit: {m_peak_xmit:.1f} GB/s  ({m_peak_xmit/8:.1f} GB/s avg/port)")
print(f"  Peak total IB rcv:  {m_peak_rcv:.1f} GB/s")
print(f"  Per-port peak range: {min(m_port_peaks):.1f} - {max(m_port_peaks):.1f} GB/s")
print(f"  GPU util during IB burst: {np.mean(m_gpu_avg[m_start:m_end][m_gpu_avg[m_start:m_end] > 0]):.0f}% avg")
