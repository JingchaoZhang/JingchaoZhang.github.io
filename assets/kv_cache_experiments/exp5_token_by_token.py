#!/usr/bin/env python3
"""Experiment 5: Watch KV cache grow token by token during generation.
Shows the difference between prefill (bulk KV creation) and decode
(incremental KV append), and how memory grows during autoregressive generation.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

MODEL = "Qwen/Qwen2.5-7B"

print("=" * 70)
print("Experiment 5: KV Cache Growth During Generation")
print("=" * 70)

print(f"\nLoading {MODEL} in FP16...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float16, device_map="cuda:0"
)
model.eval()

total_gpu_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3

prompt = ("Explain the theory of relativity in detail, covering both special relativity "
          "and general relativity, including key equations, experimental evidence, and "
          "modern applications in GPS and gravitational wave detection.")
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda:0")
prompt_len = input_ids.shape[1]

MAX_NEW = 300
past_kv = None

print(f"\nPrompt: {prompt[:80]}...")
print(f"Prompt tokens: {prompt_len}")
print(f"Generating {MAX_NEW} new tokens...\n")

print(f"{'Phase':<8} {'Step':>5} {'Total Tok':>9} {'KV Cache':>10} {'GPU Used':>10} "
      f"{'KV Shape':>25} {'Time ms':>8}")
print("-" * 80)

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

gen_start = time.time()
prefill_time = 0
decode_times = []

for i in range(MAX_NEW):
    step_start = time.time()

    with torch.no_grad():
        if past_kv is None:
            # PREFILL: process entire prompt at once, create full KV cache
            out = model(input_ids, use_cache=True)
            phase = "PREFILL"
        else:
            # DECODE: only feed the last token, reuse KV cache
            out = model(input_ids[:, -1:], past_key_values=past_kv, use_cache=True)
            phase = "DECODE"

        past_kv = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=-1)

    step_time = (time.time() - step_start) * 1000  # ms

    if i == 0:
        prefill_time = step_time
    else:
        decode_times.append(step_time)

    # Print at key intervals
    if i == 0 or i % 50 == 0 or i == MAX_NEW - 1:
        kv_bytes = sum(t.nelement() * t.element_size() for layer in past_kv for t in layer)
        seq_len = past_kv[0][0].shape[2]  # (batch, kv_heads, seq_len, head_dim)
        gpu_used = torch.cuda.memory_allocated() / 1024**3
        kv_shape = f"({past_kv[0][0].shape[2]}, {past_kv[0][0].shape[1]}h, {past_kv[0][0].shape[3]}d)"

        print(f"{phase:<8} {i:>5} {seq_len:>9} {kv_bytes/1024**2:>7.2f} MB "
              f"{gpu_used:>7.2f} GB {kv_shape:>25} {step_time:>7.1f}")

total_time = time.time() - gen_start
peak_mem = torch.cuda.max_memory_allocated() / 1024**3

# Decode generated text
generated = tokenizer.decode(input_ids[0, prompt_len:], skip_special_tokens=True)

print(f"\n--- Performance Summary ---")
print(f"Prefill time ({prompt_len} tokens):          {prefill_time:.1f} ms "
      f"({prompt_len / prefill_time * 1000:.0f} tokens/s)")
if decode_times:
    avg_decode = sum(decode_times) / len(decode_times)
    print(f"Avg decode time (per token):        {avg_decode:.1f} ms "
          f"({1000 / avg_decode:.0f} tokens/s)")
print(f"Total generation time:              {total_time:.1f} s")
print(f"Peak GPU memory:                    {peak_mem:.2f} GB ({peak_mem/total_gpu_gb*100:.0f}%)")

print(f"\n--- KV Cache Anatomy ---")
print(f"Number of layers with KV cache: {len(past_kv)}")
print(f"Each layer stores: Key [{past_kv[0][0].shape}] + Value [{past_kv[0][1].shape}]")
print(f"  - dim 0: batch size = {past_kv[0][0].shape[0]}")
print(f"  - dim 1: KV heads   = {past_kv[0][0].shape[1]}")
print(f"  - dim 2: seq length = {past_kv[0][0].shape[2]} (grows each step)")
print(f"  - dim 3: head dim   = {past_kv[0][0].shape[3]}")

final_kv_bytes = sum(t.nelement() * t.element_size() for layer in past_kv for t in layer)
print(f"\nFinal KV cache size: {final_kv_bytes/1024**2:.2f} MB for {past_kv[0][0].shape[2]} tokens")
print(f"  = {final_kv_bytes / past_kv[0][0].shape[2]:.0f} bytes/token")

print(f"\n--- Key Insight ---")
print(f"Prefill: computed KV for {prompt_len} tokens in one shot ({prefill_time:.0f} ms) — COMPUTE bound")
if decode_times:
    print(f"Decode:  appended 1 KV entry per step ({avg_decode:.0f} ms each) — MEMORY BANDWIDTH bound")
    print(f"Prefill is {prefill_time/avg_decode:.0f}x slower wall-clock BUT processes "
          f"{prompt_len}x more tokens → {prompt_len*avg_decode/prefill_time:.0f}x higher throughput")
print(f"\nGenerated text preview: {generated[:200]}...")
