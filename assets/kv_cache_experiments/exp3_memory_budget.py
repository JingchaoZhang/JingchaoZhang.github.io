#!/usr/bin/env python3
"""Experiment 3: GPU memory budget breakdown.
Shows how GPU memory is partitioned: model weights vs KV cache vs activations,
and calculates max concurrent requests at various context lengths.
"""
import torch
from transformers import AutoModelForCausalLM

MODEL = "Qwen/Qwen2.5-7B"

print("=" * 70)
print("Experiment 3: GPU Memory Budget — Weights vs KV Cache")
print("=" * 70)

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

mem_empty = torch.cuda.memory_allocated() / 1024**3

print(f"\nLoading {MODEL} in FP16...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float16, device_map="cuda:0"
)
model.eval()

mem_model = torch.cuda.memory_allocated() / 1024**3
total_gpu = torch.cuda.get_device_properties(0).total_memory / 1024**3
model_size = mem_model - mem_empty
free_for_kv = total_gpu - mem_model

print(f"\n{'='*50}")
print(f"  GPU:              {torch.cuda.get_device_name(0)}")
print(f"  Total GPU memory: {total_gpu:.1f} GB")
print(f"  CUDA overhead:    {mem_empty:.2f} GB")
print(f"  Model weights:    {model_size:.1f} GB ({model_size/total_gpu*100:.0f}% of GPU)")
print(f"  Free for KV+act:  {free_for_kv:.1f} GB ({free_for_kv/total_gpu*100:.0f}% of GPU)")
print(f"{'='*50}")

# Verify with an actual forward pass
print(f"\nVerifying with actual forward pass (batch=1, seq=4096)...")
torch.cuda.reset_peak_memory_stats()
input_ids = torch.randint(100, 30000, (1, 4096), device="cuda:0")

mem_pre_fwd = torch.cuda.memory_allocated() / 1024**3
with torch.no_grad():
    outputs = model(input_ids, use_cache=True)
    past_kv = outputs.past_key_values

mem_post_fwd = torch.cuda.memory_allocated() / 1024**3
peak_mem = torch.cuda.max_memory_allocated() / 1024**3

kv_bytes = sum(t.nelement() * t.element_size() for layer in past_kv for t in layer)
kv_gb = kv_bytes / 1024**3

# Activations = peak during forward - steady state after
activation_overhead = peak_mem - mem_post_fwd

print(f"  KV cache (4096 tokens):       {kv_gb:.3f} GB")
print(f"  Activation peak overhead:     {activation_overhead:.3f} GB")
print(f"  Peak GPU usage:               {peak_mem:.1f} GB ({peak_mem/total_gpu*100:.0f}%)")

del outputs, past_kv, input_ids
torch.cuda.empty_cache()

# Calculate capacity table
config = model.config
kv_bytes_per_token = (2 * config.num_hidden_layers * config.num_key_value_heads
                      * (config.hidden_size // config.num_attention_heads) * 2)

# Reserve some memory for activations and fragmentation
usable_fraction = 0.85
usable_bytes = free_for_kv * usable_fraction * 1024**3

print(f"\n--- Concurrent Request Capacity (85% memory utilization) ---")
print(f"Usable memory for KV cache: {usable_bytes/1024**3:.1f} GB")
print(f"KV bytes per token: {kv_bytes_per_token:,}")
print(f"\n{'Context Len':>12} | {'KV/Request':>12} | {'Max Batch':>10} | {'Total KV':>10} | {'Pct GPU':>8}")
print("-" * 60)

for ctx_len in [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]:
    kv_per_request = kv_bytes_per_token * ctx_len
    max_batch = int(usable_bytes / kv_per_request)
    total_kv_gb = (max_batch * kv_per_request) / 1024**3
    pct = (model_size + total_kv_gb) / total_gpu * 100

    if max_batch < 1:
        print(f"{ctx_len:>12,} | {kv_per_request/1024**2:>9.1f} MB | {'< 1':>10} | {'N/A':>10} | {'N/A':>8}")
    else:
        print(f"{ctx_len:>12,} | {kv_per_request/1024**2:>9.1f} MB | {max_batch:>10} | {total_kv_gb:>7.1f} GB | {pct:>6.0f}%")

print(f"\n--- Key Insight ---")
print(f"Model weights ({model_size:.1f} GB) are FIXED cost.")
print(f"KV cache is VARIABLE cost: batch_size * context_length * {kv_bytes_per_token:,} bytes.")
print(f"At long contexts, KV cache dominates total GPU memory.")
