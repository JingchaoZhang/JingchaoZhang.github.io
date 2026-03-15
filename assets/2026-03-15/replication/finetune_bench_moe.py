#!/usr/bin/env python3
"""
MoE fine-tuning benchmark for measuring IB vs Ethernet performance.
Designed for Mixtral-8x7B: FSDP must all-gather ALL expert weights per layer,
but only 2 experts compute per token - creating a communication-heavy workload
that stresses the inter-node fabric.

Uses FSDP FULL_SHARD with activation checkpointing.
All ranks load model independently from shared Lustre filesystem,
eliminating the NCCL BROADCAST bottleneck that fails at 4+ nodes.
"""
import argparse
import functools
import os
import time

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    checkpoint_wrapper,
    CheckpointImpl,
)
from transformers import AutoModelForCausalLM, AutoConfig
from transformers.models.mixtral.modeling_mixtral import MixtralDecoderLayer

from datetime import timedelta


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

    dist.init_process_group("nccl", timeout=timedelta(seconds=1800))
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)

    if global_rank == 0:
        print(f"=== MoE Fine-tuning Benchmark ===")
        print(f"Model: {args.model_path}")
        print(f"World size: {world_size} GPUs ({world_size // 8} nodes)")
        print(f"Seq len: {args.seq_len}, Micro BS: {args.batch_size}")
        print(f"Steps: {args.steps} (warmup: {args.warmup_steps})")
        print(f"NCCL_IB_DISABLE={os.environ.get('NCCL_IB_DISABLE', 'not set')}")
        print(f"NCCL_NET={os.environ.get('NCCL_NET', 'not set')}")
        print()

    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)

    bf16_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={MixtralDecoderLayer},
    )

    # All ranks load model from Lustre independently.
    # Stagger by local_rank to limit per-node CPU memory peak and Lustre contention.
    # Each round: 1 rank per node loads simultaneously (N_nodes concurrent readers).
    for lr in range(8):
        if local_rank == lr:
            if global_rank == 0:
                print(f"Loading model (staggered round {lr}/7)...")
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                config=config,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
        dist.barrier()

    if global_rank == 0:
        total_params = sum(p.numel() for p in model.parameters()) / 1e9
        print(f"Total parameters: {total_params:.1f}B")
        print(f"Architecture: {config.num_hidden_layers} layers, "
              f"{config.num_local_experts} experts/layer, "
              f"top-{config.num_experts_per_tok} routing")

    dist.barrier()
    if global_rank == 0:
        print("Wrapping with FSDP (FULL_SHARD)...")

    # No sync_module_states needed - all ranks loaded identical weights from disk
    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=bf16_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=local_rank,
        sync_module_states=False,
        limit_all_gathers=True,
    )

    check_fn = lambda submodule: isinstance(submodule, MixtralDecoderLayer)
    non_reentrant_wrapper = functools.partial(
        checkpoint_wrapper,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=non_reentrant_wrapper,
        check_fn=check_fn,
    )

    if global_rank == 0:
        print("FSDP wrapping + activation checkpointing done.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    vocab_size = config.vocab_size
    input_ids = torch.randint(0, vocab_size, (args.batch_size, args.seq_len),
                              device=local_rank)
    labels = input_ids.clone()

    if global_rank == 0:
        print(f"Running {args.warmup_steps} warmup steps...")
    for step in range(args.warmup_steps):
        outputs = model(input_ids=input_ids, labels=labels, use_cache=False)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if global_rank == 0:
            print(f"  Warmup step {step + 1}/{args.warmup_steps} | loss={loss.item():.4f}")

    torch.cuda.synchronize()
    dist.barrier()

    if global_rank == 0:
        print(f"Running {args.steps} benchmark steps...")

    tokens_per_step = args.batch_size * args.seq_len * world_size
    torch.cuda.synchronize()
    dist.barrier()
    t0 = time.perf_counter()

    for step in range(args.steps):
        outputs = model(input_ids=input_ids, labels=labels, use_cache=False)
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
