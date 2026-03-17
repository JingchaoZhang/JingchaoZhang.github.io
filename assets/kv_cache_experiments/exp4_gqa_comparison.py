#!/usr/bin/env python3
"""Experiment 4: GQA impact on KV cache size.
Compares models with different KV head counts to show how
Grouped Query Attention reduces KV cache memory.
"""
from transformers import AutoConfig

print("=" * 70)
print("Experiment 4: GQA Impact on KV Cache Size")
print("=" * 70)

models = [
    ("google/gemma-2-9b",               "Gemma-2 9B (MHA)"),
    ("Qwen/Qwen2.5-7B",                "Qwen2.5 7B (GQA)"),
    ("Qwen/Qwen2.5-3B",                "Qwen2.5 3B (GQA)"),
    ("mistralai/Mistral-7B-v0.1",       "Mistral 7B (GQA)"),
    ("mistralai/Mixtral-8x7B-v0.1",     "Mixtral 8x7B (GQA+MoE)"),
    ("Qwen/Qwen2.5-72B",               "Qwen2.5 72B (GQA)"),
    ("deepseek-ai/DeepSeek-V2-Lite",    "DeepSeek-V2-Lite (MLA)"),
]

print(f"\n{'Model':<30} {'Params':>8} {'Layers':>6} {'Attn H':>6} {'KV H':>5} "
      f"{'H Dim':>5} {'GQA':>4} {'KV KB/tok':>9} {'KV GB/128K':>10}")
print("-" * 95)

results = []

for model_id, label in models:
    try:
        c = AutoConfig.from_pretrained(model_id)
        layers = c.num_hidden_layers
        attn_heads = c.num_attention_heads
        kv_heads = getattr(c, 'num_key_value_heads', attn_heads)
        head_dim = c.hidden_size // attn_heads
        gqa_ratio = attn_heads // kv_heads

        # KV cache per token in bytes (FP16)
        kv_bytes_per_token = 2 * layers * kv_heads * head_dim * 2

        # What it would be without GQA (full MHA)
        mha_bytes_per_token = 2 * layers * attn_heads * head_dim * 2

        # KV cache for 128K context
        kv_128k_gb = kv_bytes_per_token * 131072 / 1024**3

        # Approximate param count
        params = getattr(c, 'num_parameters', None)

        results.append({
            "label": label,
            "kv_bytes": kv_bytes_per_token,
            "mha_bytes": mha_bytes_per_token,
            "gqa_ratio": gqa_ratio,
        })

        print(f"{label:<30} {'~7B' if '7' in label or '8B' in label else '~70B':>8} "
              f"{layers:>6} {attn_heads:>6} {kv_heads:>5} {head_dim:>5} "
              f"{gqa_ratio:>3}x {kv_bytes_per_token/1024:>8.1f} {kv_128k_gb:>9.2f}")
    except Exception as e:
        print(f"{label:<30} (skipped: {e})")

# Analysis
print(f"\n--- GQA Savings Analysis ---")
for r in results:
    if r["gqa_ratio"] > 1:
        savings = (1 - r["kv_bytes"] / r["mha_bytes"]) * 100
        print(f"{r['label']}: GQA {r['gqa_ratio']}x → KV cache is {savings:.0f}% smaller than MHA")

print(f"\n--- Why GQA Matters ---")
print(f"Without GQA (MHA): every attention head has its own K and V projections")
print(f"With GQA: multiple query heads share the same K,V heads")
print(f"  → KV cache shrinks by factor of (num_attn_heads / num_kv_heads)")
print(f"  → Compute stays the same (queries still attend independently)")
print(f"  → Strictly a memory optimization for inference serving")
