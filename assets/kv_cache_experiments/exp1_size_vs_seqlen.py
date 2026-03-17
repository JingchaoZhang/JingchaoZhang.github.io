#!/usr/bin/env python3
"""Experiment 1: Measure KV cache size vs sequence length.
Shows that KV cache grows linearly with sequence length and verifies
the theoretical formula against actual GPU memory measurements.
"""
import torch
from transformers import AutoModelForCausalLM, AutoConfig

MODEL = "Qwen/Qwen2.5-7B"

print("=" * 70)
print("Experiment 1: KV Cache Size vs Sequence Length")
print("=" * 70)

# Load model
print(f"\nLoading {MODEL} in FP16...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float16, device_map="cuda:0"
)
model.eval()

# Print KV cache config
config = model.config
n_layers = config.num_hidden_layers
n_kv_heads = config.num_key_value_heads
head_dim = config.hidden_size // config.num_attention_heads

theoretical_bytes_per_token = 2 * n_layers * n_kv_heads * head_dim * 2  # 2 for K+V, 2 for FP16

print(f"\nModel config:")
print(f"  Layers:          {n_layers}")
print(f"  Attention heads: {config.num_attention_heads}")
print(f"  KV heads (GQA):  {n_kv_heads}")
print(f"  Head dimension:  {head_dim}")
print(f"  GQA ratio:       {config.num_attention_heads // n_kv_heads}x")
print(f"  Theoretical KV bytes/token: {theoretical_bytes_per_token:,} "
      f"({theoretical_bytes_per_token / 1024:.1f} KB)")

# Measure at different sequence lengths
seq_lengths = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
results = []

print(f"\n{'SeqLen':>8} | {'KV Tensors':>12} | {'GPU Delta':>12} | {'Bytes/Token':>12} | {'Theory':>12} | {'Match':>6}")
print("-" * 75)

for seq_len in seq_lengths:
    torch.cuda.empty_cache()

    input_ids = torch.randint(100, 30000, (1, seq_len), device="cuda:0")

    mem_before = torch.cuda.memory_allocated()

    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
        past_kv = outputs.past_key_values

    mem_after = torch.cuda.memory_allocated()
    gpu_delta = mem_after - mem_before

    # Measure actual KV tensor sizes
    kv_tensor_bytes = 0
    for layer_kv in past_kv:
        for tensor in layer_kv:
            kv_tensor_bytes += tensor.nelement() * tensor.element_size()

    actual_bytes_per_token = kv_tensor_bytes / seq_len
    match = "YES" if abs(actual_bytes_per_token - theoretical_bytes_per_token) < 1 else "NO"

    results.append({
        "seq_len": seq_len,
        "kv_mb": kv_tensor_bytes / 1024**2,
        "gpu_delta_mb": gpu_delta / 1024**2,
        "bytes_per_token": actual_bytes_per_token,
    })

    print(f"{seq_len:>8} | {kv_tensor_bytes/1024**2:>9.2f} MB | {gpu_delta/1024**2:>9.2f} MB | "
          f"{actual_bytes_per_token:>10,.0f} B | {theoretical_bytes_per_token:>10,.0f} B | {match:>6}")

    del outputs, past_kv, input_ids
    torch.cuda.empty_cache()

# Summary
print(f"\n--- Summary ---")
print(f"KV cache scales linearly: {results[-1]['kv_mb'] / results[0]['kv_mb']:.1f}x memory "
      f"for {results[-1]['seq_len'] / results[0]['seq_len']:.0f}x sequence length")
print(f"Theoretical formula: 2 * {n_layers} layers * {n_kv_heads} kv_heads * {head_dim} head_dim * 2 bytes = {theoretical_bytes_per_token:,} bytes/token")
print(f"At 128K context: {theoretical_bytes_per_token * 131072 / 1024**3:.2f} GB KV cache per request")
