#!/usr/bin/env python3
"""Experiment 2: KV cache scaling with batch size.
Shows how KV cache grows with concurrent requests (batch dimension),
demonstrating why serving many users simultaneously is memory-bound.
"""
import torch
from transformers import AutoModelForCausalLM

MODEL = "Qwen/Qwen2.5-7B"

print("=" * 70)
print("Experiment 2: KV Cache vs Batch Size (Concurrent Users)")
print("=" * 70)

print(f"\nLoading {MODEL} in FP16...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float16, device_map="cuda:0"
)
model.eval()

total_gpu_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
model_mem_gb = torch.cuda.memory_allocated() / 1024**3

print(f"\nGPU: {torch.cuda.get_device_name(0)}")
print(f"GPU total memory: {total_gpu_gb:.1f} GB")
print(f"Model weights:    {model_mem_gb:.1f} GB")
print(f"Available:        {total_gpu_gb - model_mem_gb:.1f} GB")

SEQ_LEN = 2048
batch_sizes = [1, 2, 4, 8, 16, 32, 64]

print(f"\nKV Cache scaling with batch size (seq_len={SEQ_LEN})")
print(f"\n{'Batch':>6} | {'KV Cache':>12} | {'GPU Delta':>12} | {'Total Used':>12} | {'% GPU':>6} | {'Status':>8}")
print("-" * 70)

for bs in batch_sizes:
    torch.cuda.empty_cache()
    input_ids = torch.randint(100, 30000, (bs, SEQ_LEN), device="cuda:0")

    mem_before = torch.cuda.memory_allocated()

    try:
        with torch.no_grad():
            outputs = model(input_ids, use_cache=True)
            past_kv = outputs.past_key_values

        mem_after = torch.cuda.memory_allocated()
        gpu_delta_gb = (mem_after - mem_before) / 1024**3

        kv_bytes = sum(t.nelement() * t.element_size() for layer in past_kv for t in layer)
        kv_gb = kv_bytes / 1024**3
        total_used_gb = mem_after / 1024**3
        pct = total_used_gb / total_gpu_gb * 100

        print(f"{bs:>6} | {kv_gb:>9.2f} GB | {gpu_delta_gb:>9.2f} GB | "
              f"{total_used_gb:>9.2f} GB | {pct:>5.1f}% | {'OK':>8}")

        del outputs, past_kv
    except torch.cuda.OutOfMemoryError:
        print(f"{bs:>6} | {'---':>12} | {'---':>12} | {'---':>12} | {'---':>6} | {'OOM':>8}")
        torch.cuda.empty_cache()

    del input_ids
    torch.cuda.empty_cache()

# Show the math
config = model.config
kv_bytes_per_token = (2 * config.num_hidden_layers * config.num_key_value_heads
                      * (config.hidden_size // config.num_attention_heads) * 2)
print(f"\n--- Key Takeaway ---")
print(f"KV cache per request at {SEQ_LEN} tokens: {kv_bytes_per_token * SEQ_LEN / 1024**2:.1f} MB")
print(f"KV cache scales as: batch_size * seq_len * {kv_bytes_per_token:,} bytes/token")
print(f"This is why concurrent user count is limited by GPU memory, not compute.")
