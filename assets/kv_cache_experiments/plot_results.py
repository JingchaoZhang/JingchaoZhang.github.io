#!/usr/bin/env python3
"""Generate all KV cache experiment plots for the blog post.
Uses hardcoded data from actual H100 experiment runs — no GPU needed.
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

OUT_DIR = "/workspace/kv_exp/figures"
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    'figure.figsize': (10, 6),
    'font.size': 13,
    'axes.titlesize': 15,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.2,
})

BLUE = '#2196F3'
RED = '#F44336'
GREEN = '#4CAF50'
ORANGE = '#FF9800'
PURPLE = '#9C27B0'
GREY = '#757575'

# ============================================================
# Figure 1: KV Cache Size vs Sequence Length (Experiment 1)
# ============================================================
print("Generating Figure 1: KV cache vs sequence length...")

seq_lens = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
kv_mb = [7.0, 14.0, 28.0, 56.0, 112.0, 224.0, 448.0, 896.0]
gpu_delta_mb = [77.0, 88.25, 176.5, 354.0, 712.0, 1412.5, 2824.5, 5648.0]

fig, ax1 = plt.subplots()

ax1.plot(seq_lens, kv_mb, 'o-', color=BLUE, linewidth=2.5, markersize=8,
         label='KV Cache (tensors only)', zorder=5)
ax1.plot(seq_lens, gpu_delta_mb, 's--', color=ORANGE, linewidth=2, markersize=7,
         label='Total GPU Delta (KV + activations)', zorder=4)

# Theoretical line
theory_mb = [s * 57344 / 1024**2 for s in seq_lens]
ax1.plot(seq_lens, theory_mb, ':', color=GREY, linewidth=1.5,
         label='Theoretical (56 KB/token)', zorder=3)

ax1.set_xscale('log', base=2)
ax1.set_yscale('log', base=2)
ax1.set_xlabel('Sequence Length (tokens)')
ax1.set_ylabel('Memory (MB)')
ax1.set_title('KV Cache Scales Linearly with Sequence Length\nQwen2.5-7B (FP16) on NVIDIA H100')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)
ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

# Annotate key points
ax1.annotate(f'896 MB', xy=(16384, 896), xytext=(8000, 1400),
             arrowprops=dict(arrowstyle='->', color=BLUE), fontsize=10, color=BLUE)
ax1.annotate(f'7 MB', xy=(128, 7), xytext=(200, 20),
             arrowprops=dict(arrowstyle='->', color=BLUE), fontsize=10, color=BLUE)

plt.savefig(f'{OUT_DIR}/fig1_kv_vs_seqlen.png')
plt.close()

# ============================================================
# Figure 2: KV Cache vs Batch Size (Experiment 2)
# ============================================================
print("Generating Figure 2: KV cache vs batch size...")

batch_sizes = [1, 2, 4, 8, 16, 32, 64]
kv_cache_gb = [0.11, 0.22, 0.44, 0.88, 1.75, 3.50, 7.00]
total_used_gb = [14.95, 15.64, 17.02, 19.77, 25.29, 36.32, 58.38]
model_weight_gb = 14.2
gpu_total_gb = 79.2

fig, ax = plt.subplots()

# Stacked-ish view: model weights (constant) + KV cache (growing)
ax.bar(range(len(batch_sizes)), [model_weight_gb]*len(batch_sizes),
       label='Model Weights (fixed)', color=GREY, alpha=0.6, width=0.6)
ax.bar(range(len(batch_sizes)), kv_cache_gb, bottom=[model_weight_gb]*len(batch_sizes),
       label='KV Cache', color=BLUE, alpha=0.85, width=0.6)

# GPU delta (activations etc)
gpu_delta_other = [t - model_weight_gb - k for t, k in zip(total_used_gb, kv_cache_gb)]
ax.bar(range(len(batch_sizes)), gpu_delta_other,
       bottom=[model_weight_gb + k for k in kv_cache_gb],
       label='Activations + Other', color=ORANGE, alpha=0.6, width=0.6)

ax.axhline(y=gpu_total_gb, color=RED, linestyle='--', linewidth=1.5, label=f'GPU Capacity ({gpu_total_gb:.0f} GB)')
ax.set_xticks(range(len(batch_sizes)))
ax.set_xticklabels([str(b) for b in batch_sizes])
ax.set_xlabel('Batch Size (Concurrent Requests)')
ax.set_ylabel('GPU Memory (GB)')
ax.set_title('GPU Memory Usage vs Concurrent Requests\nQwen2.5-7B, seq_len=2048, H100 80GB')
ax.legend(loc='upper left')
ax.set_ylim(0, 85)
ax.grid(True, alpha=0.2, axis='y')

# Annotate percentage
for i, (b, t) in enumerate(zip(batch_sizes, total_used_gb)):
    pct = t / gpu_total_gb * 100
    ax.text(i, t + 1.5, f'{pct:.0f}%', ha='center', fontsize=9, color=GREY)

plt.savefig(f'{OUT_DIR}/fig2_kv_vs_batch.png')
plt.close()

# ============================================================
# Figure 3: Concurrent Request Capacity (Experiment 3)
# ============================================================
print("Generating Figure 3: Concurrent request capacity...")

ctx_lens = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
max_batch = [2019, 1009, 504, 252, 126, 63, 31, 15, 7]

fig, ax = plt.subplots()

bars = ax.bar(range(len(ctx_lens)), max_batch, color=BLUE, alpha=0.8, width=0.65,
              edgecolor='white', linewidth=0.5)

# Color gradient: green for high capacity, red for low
for i, (bar, mb) in enumerate(zip(bars, max_batch)):
    if mb > 500:
        bar.set_color(GREEN)
    elif mb > 50:
        bar.set_color(BLUE)
    elif mb > 10:
        bar.set_color(ORANGE)
    else:
        bar.set_color(RED)
    bar.set_alpha(0.8)
    ax.text(i, mb + max(max_batch)*0.02, str(mb), ha='center', fontsize=9, fontweight='bold')

ax.set_xticks(range(len(ctx_lens)))
ax.set_xticklabels([f'{c//1024}K' if c >= 1024 else str(c) for c in ctx_lens], rotation=45)
ax.set_xlabel('Context Length')
ax.set_ylabel('Max Concurrent Requests')
ax.set_title('How Many Users Can One H100 Serve Simultaneously?\nQwen2.5-7B (FP16), 85% memory utilization')
ax.set_yscale('log')
ax.grid(True, alpha=0.2, axis='y')
ax.set_ylim(1, 5000)

# Add annotation
ax.annotate('128K context:\n7 users max', xy=(8, 7), xytext=(6.5, 40),
            arrowprops=dict(arrowstyle='->', color=RED, lw=1.5),
            fontsize=11, color=RED, fontweight='bold')

plt.savefig(f'{OUT_DIR}/fig3_capacity.png')
plt.close()

# ============================================================
# Figure 4: GQA Comparison (Experiment 4)
# ============================================================
print("Generating Figure 4: GQA comparison...")

models = ['Qwen2.5\n3B', 'Qwen2.5\n7B', 'Mistral\n7B', 'Mixtral\n8x7B', 'Qwen2.5\n72B']
kv_kb_per_tok = [36.0, 56.0, 128.0, 128.0, 320.0]
kv_gb_128k = [4.5, 7.0, 16.0, 16.0, 40.0]
gqa_ratios = [8, 7, 4, 4, 8]
kv_heads = [2, 4, 8, 8, 8]
# Hypothetical MHA equivalent
mha_kb_per_tok = [36.0*8, 56.0*7, 128.0*4, 128.0*4, 320.0*8]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left: KV cache per token
x = np.arange(len(models))
width = 0.35
bars1 = ax1.bar(x - width/2, mha_kb_per_tok, width, label='Without GQA (MHA)', color=RED, alpha=0.5)
bars2 = ax1.bar(x + width/2, kv_kb_per_tok, width, label='With GQA (actual)', color=GREEN, alpha=0.8)

for i, (mha, gqa, ratio) in enumerate(zip(mha_kb_per_tok, kv_kb_per_tok, gqa_ratios)):
    savings = (1 - gqa/mha) * 100
    ax1.text(i, max(mha, gqa) + 50, f'{ratio}x\n-{savings:.0f}%', ha='center',
             fontsize=9, color=GREEN, fontweight='bold')

ax1.set_xticks(x)
ax1.set_xticklabels(models)
ax1.set_ylabel('KV Cache (KB/token)')
ax1.set_title('GQA Reduces KV Cache Per Token')
ax1.legend()
ax1.grid(True, alpha=0.2, axis='y')

# Right: KV cache at 128K context
bars3 = ax2.bar(x, kv_gb_128k, color=BLUE, alpha=0.8, width=0.5)
ax2.axhline(y=79.2, color=RED, linestyle='--', linewidth=1.5, alpha=0.5)
ax2.text(len(models)-1, 81, 'H100 80GB', fontsize=10, color=RED, ha='right')

for i, (bar, gb) in enumerate(zip(bars3, kv_gb_128k)):
    ax2.text(i, gb + 1, f'{gb:.0f} GB', ha='center', fontsize=10, fontweight='bold')
    if gb > 35:
        bar.set_color(RED)
        bar.set_alpha(0.7)

ax2.set_xticks(x)
ax2.set_xticklabels(models)
ax2.set_ylabel('KV Cache for Single 128K Request (GB)')
ax2.set_title('KV Cache at 128K Context Length')
ax2.grid(True, alpha=0.2, axis='y')
ax2.set_ylim(0, 85)

plt.suptitle('Grouped Query Attention (GQA) Impact on KV Cache', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig4_gqa_comparison.png')
plt.close()

# ============================================================
# Figure 5: Prefill vs Decode (Experiment 5)
# ============================================================
print("Generating Figure 5: Prefill vs decode...")

steps = [0, 50, 100, 150, 200, 250, 299]
total_tokens = [37, 87, 137, 187, 237, 287, 336]
kv_cache_mb_gen = [2.02, 4.76, 7.49, 10.23, 12.96, 15.70, 18.38]
time_ms = [718.9, 481.1, 479.1, 470.0, 472.4, 478.7, 473.4]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left: KV cache growth during generation
ax1.plot(total_tokens, kv_cache_mb_gen, 'o-', color=BLUE, linewidth=2.5, markersize=8)
ax1.fill_between(total_tokens, kv_cache_mb_gen, alpha=0.15, color=BLUE)

# Mark prefill vs decode
ax1.axvline(x=37, color=RED, linestyle='--', alpha=0.7)
ax1.text(25, max(kv_cache_mb_gen)*0.9, 'PREFILL', fontsize=10, color=RED,
         fontweight='bold', rotation=90, va='top')
ax1.text(50, max(kv_cache_mb_gen)*0.9, 'DECODE →', fontsize=10, color=GREEN,
         fontweight='bold', va='top')

ax1.set_xlabel('Total Tokens (prompt + generated)')
ax1.set_ylabel('KV Cache Size (MB)')
ax1.set_title('KV Cache Grows Linearly During Generation')
ax1.grid(True, alpha=0.3)

# Right: Throughput comparison — prefill vs decode
phases = ['Prefill\n(37 tokens)', 'Decode\n(per token)']
throughput = [51, 2]  # tokens/s
latency = [718.9, 475.8]  # ms

ax2_bar = ax2.bar(phases, throughput, color=[ORANGE, BLUE], alpha=0.8, width=0.5,
                  edgecolor='white', linewidth=1)

for i, (bar, tp, lat) in enumerate(zip(ax2_bar, throughput, latency)):
    ax2.text(i, tp + 1.5, f'{tp} tok/s\n({lat:.0f} ms)', ha='center',
             fontsize=11, fontweight='bold')

ax2.set_ylabel('Throughput (tokens/s)')
ax2.set_title('Prefill is Compute-Bound,\nDecode is Memory-Bandwidth-Bound')
ax2.grid(True, alpha=0.2, axis='y')
ax2.set_ylim(0, 65)

# Add explanation
ax2.text(0.5, 0.15, 'Prefill processes 37 tokens in one shot\nDecode generates 1 token at a time\n→ 24x higher prefill throughput',
         transform=ax2.transAxes, fontsize=10, ha='center', va='center',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('Token Generation: Prefill vs Decode Phases', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig5_prefill_vs_decode.png')
plt.close()

# ============================================================
# Figure 6: Memory Budget Breakdown (Experiment 3 — visual)
# ============================================================
print("Generating Figure 6: Memory budget breakdown...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

scenarios = [
    ("1 user @ 2K ctx", 14.2, 0.11, 79.2),
    ("64 users @ 2K ctx", 14.2, 7.0, 79.2),
    ("7 users @ 128K ctx", 14.2, 49.0, 79.2),
]

for ax, (title, weights, kv, total) in zip(axes, scenarios):
    other = total - weights - kv
    sizes = [weights, kv, max(other, 0)]
    labels = [f'Weights\n{weights:.1f} GB', f'KV Cache\n{kv:.1f} GB', f'Free\n{other:.1f} GB']
    colors_pie = [GREY, BLUE, '#E8E8E8']
    explode = (0, 0.05, 0)

    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors_pie,
                                       explode=explode, autopct='%1.0f%%',
                                       startangle=90, pctdistance=0.75)
    for t in autotexts:
        t.set_fontsize(10)
    for t in texts:
        t.set_fontsize(9)
    ax.set_title(title, fontsize=12, fontweight='bold')

plt.suptitle('H100 80GB Memory Budget: Weights vs KV Cache\nQwen2.5-7B (FP16)',
             fontsize=14, fontweight='bold', y=1.05)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig6_memory_budget.png')
plt.close()

print(f"\nAll figures saved to {OUT_DIR}/")
for f in sorted(os.listdir(OUT_DIR)):
    print(f"  {f}")
