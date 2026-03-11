"""
Fine-tuning benchmark for Qwen2.5-7B using PyTorch FSDP.
Measures training throughput (tokens/sec) across varying GPU counts
to demonstrate NVLink vs InfiniBand scaling.
"""

import os
import time
import json
import functools
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

# === Configuration ===
MODEL_PATH = os.environ.get("MODEL_PATH", "/models/Qwen2.5-7B")
SEQ_LEN = 2048
BATCH_SIZE_PER_GPU = 1
NUM_STEPS = 50        # benchmark steps (after warmup)
WARMUP_STEPS = 10
GRAD_ACCUM_STEPS = 1
SHARDING = os.environ.get("SHARDING", "full")  # "full" or "hybrid"

def create_synthetic_dataset(tokenizer, num_samples=2000, seq_len=SEQ_LEN):
    """Create synthetic training data — random token sequences."""
    vocab_size = tokenizer.vocab_size
    data = []
    for _ in range(num_samples):
        input_ids = torch.randint(100, vocab_size - 100, (seq_len,))
        data.append({"input_ids": input_ids, "labels": input_ids.clone()})
    return data

def collate_fn(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    return {"input_ids": input_ids, "labels": labels}

def main():
    # Initialize distributed
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if global_rank == 0:
        print(f"=== Fine-Tuning Benchmark ===")
        print(f"World size: {world_size} GPUs")
        print(f"Model: {MODEL_PATH}")
        print(f"Seq length: {SEQ_LEN}")
        print(f"Batch size per GPU: {BATCH_SIZE_PER_GPU}")
        print(f"Global batch size: {BATCH_SIZE_PER_GPU * world_size}")
        print(f"Sharding strategy: {SHARDING}")
        print(f"Warmup steps: {WARMUP_STEPS}")
        print(f"Benchmark steps: {NUM_STEPS}")
        print(f"================================")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    if global_rank == 0:
        print("Loading model...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        use_cache=False,  # disable KV cache for training
    )
    model.gradient_checkpointing_enable()

    # FSDP configuration: wrap each Qwen2DecoderLayer individually
    auto_wrap = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={Qwen2DecoderLayer},
    )

    bf16_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    # Select sharding strategy
    if SHARDING == "hybrid":
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
        if global_rank == 0:
            print("Using HYBRID_SHARD: FSDP within node, replicate across nodes")
    else:
        sharding_strategy = ShardingStrategy.FULL_SHARD
        if global_rank == 0:
            print("Using FULL_SHARD: FSDP across all GPUs")

    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap,
        sharding_strategy=sharding_strategy,
        mixed_precision=bf16_policy,
        device_id=local_rank,
        use_orig_params=True,
    )

    # Apply FSDP-native activation checkpointing to each decoder layer
    non_reentrant_wrapper = functools.partial(
        checkpoint_wrapper,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=non_reentrant_wrapper,
        check_fn=lambda submodule: isinstance(submodule, Qwen2DecoderLayer),
    )

    if global_rank == 0:
        print("Model loaded and wrapped with FSDP")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

    # Synthetic dataset
    dataset = create_synthetic_dataset(tokenizer)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=global_rank, shuffle=True)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE_PER_GPU,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    # === Training loop ===
    model.train()
    data_iter = iter(dataloader)

    def get_batch():
        nonlocal data_iter
        try:
            return next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            return next(data_iter)

    # Warmup
    if global_rank == 0:
        print(f"Running {WARMUP_STEPS} warmup steps...")
    for step in range(WARMUP_STEPS):
        batch = get_batch()
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Synchronize before benchmark
    torch.cuda.synchronize()
    dist.barrier()

    # Benchmark
    if global_rank == 0:
        print(f"Running {NUM_STEPS} benchmark steps...")

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    total_tokens = 0
    for step in range(NUM_STEPS):
        batch = get_batch()
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        tokens_this_step = BATCH_SIZE_PER_GPU * world_size * SEQ_LEN
        total_tokens += tokens_this_step

        if global_rank == 0 and (step + 1) % 10 == 0:
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time
            print(f"  Step {step+1}/{NUM_STEPS} | "
                  f"Throughput: {total_tokens / elapsed:,.0f} tokens/sec | "
                  f"Loss: {loss.item():.4f}")

    torch.cuda.synchronize()
    dist.barrier()
    end_time = time.perf_counter()

    # Results
    elapsed = end_time - start_time
    throughput = total_tokens / elapsed
    tokens_per_gpu = throughput / world_size

    if global_rank == 0:
        print(f"\n=== RESULTS ===")
        print(f"GPUs: {world_size}")
        print(f"Total time: {elapsed:.2f}s")
        print(f"Total tokens: {total_tokens:,}")
        print(f"Throughput: {throughput:,.0f} tokens/sec")
        print(f"Per-GPU throughput: {tokens_per_gpu:,.0f} tokens/sec/GPU")
        print(f"Batch size (global): {BATCH_SIZE_PER_GPU * world_size}")
        print(f"Seq length: {SEQ_LEN}")
        print(f"Steps: {NUM_STEPS}")

        # Save results to JSON
        results = {
            "gpus": world_size,
            "sharding": SHARDING,
            "throughput_tokens_per_sec": round(throughput, 1),
            "per_gpu_throughput": round(tokens_per_gpu, 1),
            "total_time_sec": round(elapsed, 2),
            "total_tokens": total_tokens,
            "batch_size_per_gpu": BATCH_SIZE_PER_GPU,
            "global_batch_size": BATCH_SIZE_PER_GPU * world_size,
            "seq_len": SEQ_LEN,
            "num_steps": NUM_STEPS,
        }
        out_file = f"/results/finetune_{world_size}gpu_{SHARDING}.json"
        os.makedirs("/results", exist_ok=True)
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {out_file}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
