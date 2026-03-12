#!/usr/bin/env python3
"""
Fine-tuning benchmark for measuring IB vs Ethernet performance.
Uses FSDP for multi-node distributed training with causal LM.
Measures tokens/sec throughput over synthetic data.

v2: Meta-device loading for non-rank-0 + longer timeout + barriers
"""
import argparse
import functools
import os
import sys
import time
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    checkpoint_wrapper,
    CheckpointImpl,
)
from transformers import AutoModelForCausalLM, AutoConfig


def log(msg, rank=None):
    ts = time.strftime("%H:%M:%S")
    prefix = f"[{ts}][rank{rank}]" if rank is not None else f"[{ts}]"
    print(f"{prefix} {msg}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup_steps", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # Use a long timeout to avoid premature NCCL communicator init failures
    dist.init_process_group("nccl", timeout=timedelta(minutes=30))
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)

    if global_rank == 0:
        log(f"=== Fine-tuning Benchmark ===", global_rank)
        log(f"Model: {args.model_path}", global_rank)
        log(f"World size: {world_size} GPUs", global_rank)
        log(f"Seq len: {args.seq_len}, Micro BS: {args.batch_size}", global_rank)
        log(f"Steps: {args.steps} (warmup: {args.warmup_steps})", global_rank)
        log(f"NCCL_IB_DISABLE={os.environ.get('NCCL_IB_DISABLE', 'not set')}", global_rank)
        print()

    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    config.use_cache = False

    bf16_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    # Only local_rank 0 on each node loads the full model to save CPU memory
    # and reduce FSDP wrapping contention. Other ranks use meta device.
    if local_rank == 0:
        log(f"Loading model on CPU (local_rank=0)...", global_rank)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            config=config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
        if global_rank == 0:
            param_count = sum(p.numel() for p in model.parameters()) / 1e9
            log(f"Model loaded: {param_count:.1f}B parameters", global_rank)
    else:
        log(f"Creating model on meta device (local_rank={local_rank})...", global_rank)
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(
                config,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
            )
        log(f"Meta model created", global_rank)

    # Barrier: ensure all ranks are ready before FSDP wrapping
    log(f"Waiting at pre-FSDP barrier...", global_rank)
    dist.barrier()
    log(f"Pre-FSDP barrier passed", global_rank)

    # param_init_fn materializes meta tensors on GPU for non-local_rank-0 ranks
    def param_init_fn(module):
        is_meta = any(
            p.device == torch.device("meta")
            for p in module.parameters(recurse=False)
        ) or any(
            b.device == torch.device("meta")
            for b in module.buffers(recurse=False)
        )
        if is_meta:
            module.to_empty(device=torch.cuda.current_device(), recurse=False)

    auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=100_000_000
    )

    log(f"Wrapping with FSDP (sync_module_states=True)...", global_rank)
    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=bf16_policy,
        device_id=local_rank,
        sync_module_states=True,
        param_init_fn=param_init_fn if local_rank != 0 else None,
    )
    log(f"FSDP wrapping done", global_rank)

    # Apply activation checkpointing AFTER FSDP (required for FSDP1)
    non_reentrant_wrapper = functools.partial(
        checkpoint_wrapper,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=non_reentrant_wrapper,
        check_fn=lambda submodule: isinstance(submodule, Qwen2DecoderLayer),
    )
    log(f"Activation checkpointing applied", global_rank)

    # Barrier: ensure all ranks have FSDP + checkpointing before proceeding
    dist.barrier()
    log(f"Post-FSDP barrier passed", global_rank)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    vocab_size = config.vocab_size
    input_ids = torch.randint(0, vocab_size, (args.batch_size, args.seq_len),
                              device=local_rank)
    labels = input_ids.clone()

    if global_rank == 0:
        log(f"Running {args.warmup_steps} warmup steps...", global_rank)
    for step in range(args.warmup_steps):
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if global_rank == 0:
            log(f"  warmup step {step+1}/{args.warmup_steps} done", global_rank)
    torch.cuda.synchronize()
    dist.barrier()

    if global_rank == 0:
        log(f"Running {args.steps} benchmark steps...", global_rank)

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
            log(f"  Step {step+1}/{args.steps} | loss={loss.item():.4f} | "
                  f"tokens/sec={tps:.0f}", global_rank)

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
        print(f"IB_DISABLE: {os.environ.get('NCCL_IB_DISABLE', 'not set')}")
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
