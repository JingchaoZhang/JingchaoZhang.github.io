#!/usr/bin/env python3
"""
AllReduce Benchmark: Custom Ring AllReduce vs NCCL AllReduce
===========================================================
Implements a manual ring allreduce using point-to-point send/recv,
then benchmarks it against NCCL's optimized all_reduce.

Usage (via torchrun on 2 nodes, 8 GPUs each):
  torchrun --nnodes=2 --nproc_per_node=8 --master_addr=<IP> --master_port=29500 \
           allreduce_benchmark.py

The script:
1. Implements ring allreduce from scratch (reduce-scatter + allgather)
2. Runs NCCL's built-in allreduce for comparison
3. Verifies correctness (both produce identical results)
4. Measures throughput at various tensor sizes
"""
import os
import sys
import time
import torch
import torch.distributed as dist


def ring_allreduce(tensor, group=None):
    """
    Manual ring allreduce using point-to-point send/recv.
    
    Algorithm:
      Phase 1 - Reduce-Scatter: N-1 steps. Each GPU sends a chunk to its
        right neighbor and receives from its left neighbor, accumulating
        partial sums. After this phase, each GPU holds the final reduced
        result for exactly one chunk.
      Phase 2 - AllGather: N-1 steps. Each GPU sends its completed chunk
        around the ring until every GPU has all chunks.
    
    This is bandwidth-optimal: each GPU sends and receives exactly
    2*(N-1)/N * data_size bytes total.
    """
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)

    if world_size == 1:
        return tensor

    # Split tensor into world_size chunks
    # Pad if not evenly divisible
    numel = tensor.numel()
    chunk_size = (numel + world_size - 1) // world_size
    padded_size = chunk_size * world_size

    if padded_size != numel:
        padded = torch.zeros(padded_size, dtype=tensor.dtype, device=tensor.device)
        padded[:numel] = tensor.view(-1)
    else:
        padded = tensor.view(-1).clone()

    chunks = list(padded.chunk(world_size))

    # Ring neighbors
    left = (rank - 1) % world_size
    right = (rank + 1) % world_size

    # Receive buffer
    recv_buf = torch.zeros(chunk_size, dtype=tensor.dtype, device=tensor.device)

    # =====================================================
    # Phase 1: Reduce-Scatter
    # N-1 steps. Use batch_isend_irecv to avoid NCCL deadlock.
    # =====================================================
    for step in range(world_size - 1):
        send_idx = (rank - step) % world_size
        recv_idx = (rank - step - 1) % world_size

        ops = [
            dist.P2POp(dist.isend, chunks[send_idx], right, group=group),
            dist.P2POp(dist.irecv, recv_buf, left, group=group),
        ]
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()

        chunks[recv_idx].add_(recv_buf)

    # =====================================================
    # Phase 2: AllGather
    # N-1 steps. Rotate completed chunks around the ring.
    # =====================================================
    for step in range(world_size - 1):
        send_idx = (rank - step + 1) % world_size
        recv_idx = (rank - step) % world_size

        ops = [
            dist.P2POp(dist.isend, chunks[send_idx], right, group=group),
            dist.P2POp(dist.irecv, recv_buf, left, group=group),
        ]
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()

        chunks[recv_idx].copy_(recv_buf)

    # Reconstruct full tensor
    result = torch.cat(chunks)[:numel]
    tensor.view(-1).copy_(result)
    return tensor


def nccl_allreduce(tensor, group=None):
    """NCCL's optimized allreduce — our baseline."""
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    return tensor


def verify_correctness(rank, world_size, device):
    """Verify that custom ring allreduce produces correct results."""
    torch.manual_seed(42)

    # Each GPU has rank-dependent data
    custom_tensor = torch.arange(1024, dtype=torch.float32, device=device) + rank
    nccl_tensor = custom_tensor.clone()
    expected = torch.zeros(1024, dtype=torch.float32, device=device)
    for r in range(world_size):
        expected += torch.arange(1024, dtype=torch.float32, device=device) + r

    ring_allreduce(custom_tensor)
    nccl_allreduce(nccl_tensor)

    ring_ok = torch.allclose(custom_tensor, expected, atol=1e-3)
    nccl_ok = torch.allclose(nccl_tensor, expected, atol=1e-3)

    if rank == 0:
        print(f"  Custom Ring AllReduce: {'PASS ✓' if ring_ok else 'FAIL ✗'}")
        print(f"  NCCL AllReduce:        {'PASS ✓' if nccl_ok else 'FAIL ✗'}")
        print(f"  Ring == NCCL:          {'PASS ✓' if torch.allclose(custom_tensor, nccl_tensor, atol=1e-3) else 'FAIL ✗'}")

    return ring_ok and nccl_ok


def benchmark_allreduce(fn, tensor_size, device, warmup=5, iters=20, label=""):
    """Benchmark an allreduce function, return avg time and bandwidth."""
    tensor = torch.randn(tensor_size, dtype=torch.float32, device=device)
    tensor_bytes = tensor.nelement() * tensor.element_size()

    # Warmup
    for _ in range(warmup):
        t = tensor.clone()
        fn(t)
    torch.cuda.synchronize()
    dist.barrier()

    # Timed iterations
    times = []
    for _ in range(iters):
        t = tensor.clone()
        torch.cuda.synchronize()
        dist.barrier()

        start = time.perf_counter()
        fn(t)
        torch.cuda.synchronize()
        end = time.perf_counter()

        times.append(end - start)

    avg_time = sum(times) / len(times)
    min_time = min(times)

    # Algorithm bandwidth: for allreduce, ideal data moved = 2*(N-1)/N * size
    world_size = dist.get_world_size()
    algo_factor = 2.0 * (world_size - 1) / world_size
    algo_bw = (tensor_bytes * algo_factor) / avg_time / 1e9  # GB/s
    algo_bw_peak = (tensor_bytes * algo_factor) / min_time / 1e9  # GB/s

    return {
        "label": label,
        "size_bytes": tensor_bytes,
        "avg_ms": avg_time * 1000,
        "min_ms": min_time * 1000,
        "algo_bw_avg": algo_bw,
        "algo_bw_peak": algo_bw_peak,
    }


def format_size(nbytes):
    """Human-readable size."""
    if nbytes >= 1024**3:
        return f"{nbytes/1024**3:.1f} GB"
    elif nbytes >= 1024**2:
        return f"{nbytes/1024**2:.1f} MB"
    elif nbytes >= 1024:
        return f"{nbytes/1024:.1f} KB"
    return f"{nbytes} B"


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    if rank == 0:
        print("=" * 80)
        print("AllReduce Benchmark: Custom Ring vs NCCL")
        print("=" * 80)
        print(f"  World size:    {world_size} GPUs")
        print(f"  GPU:           {torch.cuda.get_device_name(device)}")
        print(f"  Nodes:         {world_size // 8} (assuming 8 GPUs/node)")
        print(flush=True)

    # --- Correctness check ---
    if rank == 0:
        print("Correctness verification:", flush=True)
    verify_correctness(rank, world_size, device)
    dist.barrier()

    # --- Benchmark at various sizes ---
    sizes = [
        1024,               # 4 KB
        64 * 1024,          # 256 KB
        1024 * 1024,        # 4 MB
        16 * 1024 * 1024,   # 64 MB
        64 * 1024 * 1024,   # 256 MB
        128 * 1024 * 1024,  # 512 MB
        256 * 1024 * 1024,  # 1 GB
        512 * 1024 * 1024,  # 2 GB
        1024 * 1024 * 1024, # 4 GB
        2048 * 1024 * 1024, # 8 GB (same as nccl-tests)
    ]

    results_custom = []
    results_nccl = []

    if rank == 0:
        print()
        print(f"{'Size':>10} │ {'Custom Ring':>12} {'BW':>10} │ {'NCCL':>12} {'BW':>10} │ {'Speedup':>8}")
        print(f"{'':>10} │ {'(avg ms)':>12} {'(GB/s)':>10} │ {'(avg ms)':>12} {'(GB/s)':>10} │ {'(NCCL/Ring)':>8}")
        print("─" * 80)

    for numel in sizes:
        rc = benchmark_allreduce(ring_allreduce, numel, device, label="Ring")
        rn = benchmark_allreduce(nccl_allreduce, numel, device, label="NCCL")

        results_custom.append(rc)
        results_nccl.append(rn)

        if rank == 0:
            speedup = rc["avg_ms"] / rn["avg_ms"] if rn["avg_ms"] > 0 else 0
            print(f"{format_size(rc['size_bytes']):>10} │ "
                  f"{rc['avg_ms']:>10.3f}ms {rc['algo_bw_avg']:>8.1f}  │ "
                  f"{rn['avg_ms']:>10.3f}ms {rn['algo_bw_avg']:>8.1f}  │ "
                  f"{speedup:>7.1f}x")

    if rank == 0:
        print("─" * 80)
        print()
        print("Analysis:")
        print(f"  • 'BW' = Algorithm Bandwidth = 2*(N-1)/N * data_size / time")
        print(f"  • Speedup > 1 means NCCL is faster than custom Ring")
        print()

        # Summary
        large_custom = [r for r in results_custom if r["size_bytes"] >= 64 * 1024 * 1024]
        large_nccl = [r for r in results_nccl if r["size_bytes"] >= 64 * 1024 * 1024]
        if large_custom and large_nccl:
            avg_custom_bw = sum(r["algo_bw_avg"] for r in large_custom) / len(large_custom)
            avg_nccl_bw = sum(r["algo_bw_avg"] for r in large_nccl) / len(large_nccl)
            print(f"  Large message (≥64MB) avg algorithm bandwidth:")
            print(f"    Custom Ring: {avg_custom_bw:.1f} GB/s")
            print(f"    NCCL:        {avg_nccl_bw:.1f} GB/s")
            print(f"    NCCL is {avg_nccl_bw/avg_custom_bw:.1f}x faster")
        print()

        small_custom = [r for r in results_custom if r["size_bytes"] <= 256 * 1024]
        small_nccl = [r for r in results_nccl if r["size_bytes"] <= 256 * 1024]
        if small_custom and small_nccl:
            avg_custom_lat = sum(r["avg_ms"] for r in small_custom) / len(small_custom)
            avg_nccl_lat = sum(r["avg_ms"] for r in small_nccl) / len(small_nccl)
            print(f"  Small message (≤256KB) avg latency:")
            print(f"    Custom Ring: {avg_custom_lat:.3f} ms")
            print(f"    NCCL:        {avg_nccl_lat:.3f} ms")
            print(f"    NCCL is {avg_custom_lat/avg_nccl_lat:.1f}x faster")

        print()
        print("Why NCCL is faster:")
        print("  1. Kernel fusion: NCCL fuses send/recv/reduce into single GPU kernels")
        print("  2. Pipelining: NCCL overlaps computation and communication")
        print("  3. Protocol selection: LL (low-latency) for small, LL128/Simple for large")
        print("  4. Channel parallelism: NCCL uses multiple parallel rings/trees")
        print("  5. Direct GPU-GPU: NCCL uses GPUDirect RDMA, bypassing CPU entirely")
        print("  6. Our Ring uses Python-level batch_isend_irecv with sync waits")
        sys.stdout.flush()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
