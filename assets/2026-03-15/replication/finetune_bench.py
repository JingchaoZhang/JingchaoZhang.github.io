#!/usr/bin/env python3
"""
Fine-tuning benchmark for measuring IB vs Ethernet performance.
Uses FSDP for multi-node distributed training with causal LM.
Measures tokens/sec throughput over synthetic data.

Key fix: Uses NCCL_NET=Socket for true Ethernet mode.
NCCL_IB_DISABLE alone doesn't work with NCCL 2.28+ external RDMA plugin.
"""
import argparse
import functools
import os
import time
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from transformers import AutoModelForCausalLM, AutoConfig

try:
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper,
        CheckpointImpl,
        apply_activation_checkpointing,
    )
    HAS_ACTIVATION_CKPT = True
except ImportError:
    HAS_ACTIVATION_CKPT = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to model directory (e.g. /lustre/models/Qwen2.5-7B)")
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Per-GPU micro batch size")
    parser.add_argument("--steps", type=int, default=20,
                        help="Number of training steps to benchmark")
    parser.add_argument("--warmup_steps", type=int, default=3,
                        help="Warmup steps (excluded from timing)")
    return parser.parse_args()


def main():
    args = parse_args()

    dist.init_process_group("nccl", timeout=timedelta(minutes=30))
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)

    if global_rank == 0:
        print(f"=== Fine-tuning Benchmark ===")
        print(f"Model: {args.model_path}")
        print(f"World size: {world_size} GPUs")
        print(f"Seq len: {args.seq_len}, Micro BS: {args.batch_size}")
        print(f"Steps: {args.steps} (warmup: {args.warmup_steps})")
        print(f"NCCL_IB_DISABLE={os.environ.get('NCCL_IB_DISABLE', 'not set')}")
        print(f"NCCL_NET={os.environ.get('NCCL_NET', 'not set')}")
        print()

    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    config.use_cache = False

    bf16_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=100_000_000
    )

    # Detect large model (needs meta-device loading + activation checkpointing)
    model_name = os.path.basename(args.model_path).lower()
    is_large = "72b" in model_name or "70b" in model_name

    if is_large:
        if global_rank == 0:
            print("Large model detected - using meta-device initialization")

        # local_rank 0 on each node loads model on CPU; others use meta device
        if local_rank == 0:
            if global_rank == 0:
                print("Loading model on CPU (local_rank 0 per node)...")
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                config=config,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                trust_remote_code=True,
            )
        else:
            with torch.device("meta"):
                model = AutoModelForCausalLM.from_config(
                    config,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2",
                    trust_remote_code=True,
                )

        dist.barrier()

        if global_rank == 0:
            param_count = sum(p.numel() for p in model.parameters()) / 1e9
            print(f"Model: {param_count:.1f}B parameters")

        def param_init_fn(module):
            module.to_empty(device=torch.cuda.current_device(), recurse=False)

        model = FSDP(
            model,
            mixed_precision=bf16_policy,
            device_id=local_rank,
            auto_wrap_policy=auto_wrap_policy,
            sync_module_states=True,
            param_init_fn=param_init_fn,
        )

        # Activation checkpointing to reduce memory for large models
        if HAS_ACTIVATION_CKPT:
            try:
                from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
                check_fn = lambda m: isinstance(m, Qwen2DecoderLayer)
            except ImportError:
                check_fn = lambda m: m.__class__.__name__.endswith("DecoderLayer")

            non_reentrant = functools.partial(
                checkpoint_wrapper,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            )
            apply_activation_checkpointing(
                model, checkpoint_wrapper_fn=non_reentrant, check_fn=check_fn
            )
            if global_rank == 0:
                print("Activation checkpointing applied")

    else:
        # Standard loading for smaller models (7B etc.)
        if global_rank == 0:
            print("Loading model...")

        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            config=config,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
        )
        model = model.to(local_rank)

        if global_rank == 0:
            param_count = sum(p.numel() for p in model.parameters()) / 1e9
            print(f"Model: {param_count:.1f}B parameters")

        model = FSDP(
            model,
            mixed_precision=bf16_policy,
            device_id=local_rank,
            auto_wrap_policy=auto_wrap_policy,
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    vocab_size = config.vocab_size
    input_ids = torch.randint(0, vocab_size, (args.batch_size, args.seq_len),
                              device=local_rank)
    labels = input_ids.clone()

    # Warmup
    if global_rank == 0:
        print(f"Running {args.warmup_steps} warmup steps...")
    for step in range(args.warmup_steps):
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    torch.cuda.synchronize()
    dist.barrier()

    # Benchmark
    if global_rank == 0:
        print(f"Running {args.steps} benchmark steps...")

    tokens_per_step = args.batch_size * args.seq_len * world_size
    torch.cuda.synchronize()
    dist.barrier()
    t0 = time.perf_counter()

    for step in range(args.steps):
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if global_rank == 0 and (step + 1) % 5 == 0:
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            tps = tokens_per_step * (step + 1) / elapsed
            print(f"  Step {step+1}/{args.steps} | loss={loss.item():.4f} | "
                  f"tokens/sec={tps:.0f}")

    torch.cuda.synchronize()
    dist.barrier()
    t1 = time.perf_counter()

    total_time = t1 - t0
    total_tokens = tokens_per_step * args.steps
    tokens_per_sec = total_tokens / total_time

    if global_rank == 0:
        print()
        print(f"=== RESULTS ===")
        print(f"Model: {os.path.basename(args.model_path)}")
        print(f"Nodes: {world_size // 8}")
        print(f"GPUs: {world_size}")
        print(f"NCCL_IB_DISABLE: {os.environ.get('NCCL_IB_DISABLE', 'not set')}")
        print(f"NCCL_NET: {os.environ.get('NCCL_NET', 'not set')}")
        print(f"Seq len: {args.seq_len}")
        print(f"Micro BS: {args.batch_size}")
        print(f"Total steps: {args.steps}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Tokens/sec: {tokens_per_sec:.0f}")
        print(f"Tokens/sec/GPU: {tokens_per_sec / world_size:.0f}")
        print(f"Time/step: {total_time / args.steps * 1000:.1f}ms")
        print(f"=== END RESULTS ===")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
