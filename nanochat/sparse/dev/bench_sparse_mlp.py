"""Module-level benchmark for sparse vs dense MLP decode latency.

Run with:
    python -m nanochat.sparse.dev.bench_sparse_mlp \
        --n-embd 768 --k 256 --iters 200

This benchmarks an isolated MLP forward pass. For end-to-end decode tokens/sec
through the full Engine, train an actual checkpoint and measure via
engine.generate(). See docs/test-request.md for the Train/Test plan.
"""

import argparse
import time

import torch

from nanochat.gpt import MLP
from nanochat.sparse.sparse_mlp import TopKSparseMLP, init_all_router_weights


class _BenchConfig:
    """Minimal stand-in for GPTConfig for the bench."""

    def __init__(self, n_embd, sparse_k, sparse_use_router, sparse_router_rank, sparse_router_oversample):
        self.n_embd = n_embd
        self.sparse_k = sparse_k
        self.sparse_use_router = sparse_use_router
        self.sparse_router_rank = sparse_router_rank
        self.sparse_router_oversample = sparse_router_oversample


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def bench_module(module, x, n_iters, warmup, device):
    module.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            _ = module(x)
        _sync(device)
        t0 = time.perf_counter()
        for _ in range(n_iters):
            _ = module(x)
        _sync(device)
        t1 = time.perf_counter()
    return (t1 - t0) / n_iters * 1000.0  # ms per call


def bench_module_random_x(module, n_iters, warmup, device, dtype, shape):
    """Bench with a fresh random x each iter (worst case for caches)."""
    module.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            x = torch.randn(*shape, device=device, dtype=dtype)
            _ = module(x)
        _sync(device)
        t0 = time.perf_counter()
        for _ in range(n_iters):
            x = torch.randn(*shape, device=device, dtype=dtype)
            _ = module(x)
        _sync(device)
        t1 = time.perf_counter()
    return (t1 - t0) / n_iters * 1000.0


def main():
    parser = argparse.ArgumentParser(description="Benchmark sparse vs dense MLP")
    parser.add_argument("--n-embd", type=int, default=768)
    parser.add_argument("--k", type=int, default=256)
    parser.add_argument("--router-rank", type=int, default=64)
    parser.add_argument("--router-oversample", type=int, default=2)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq", type=int, default=1)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda":
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    else:
        dtype = torch.float32

    cfg_dense = _BenchConfig(args.n_embd, args.k, False, args.router_rank, args.router_oversample)
    cfg_router = _BenchConfig(args.n_embd, args.k, True, args.router_rank, args.router_oversample)

    dense = MLP(cfg_dense).to(device=device, dtype=dtype)
    sparse_t1 = TopKSparseMLP(cfg_dense).to(device=device, dtype=dtype)
    sparse_t2 = TopKSparseMLP(cfg_router).to(device=device, dtype=dtype)
    init_all_router_weights(sparse_t2)

    sparse_t3 = TopKSparseMLP(cfg_dense).to(device=device, dtype=dtype)
    sparse_t3.enable_cache(max_entries=4096)

    x_fixed = torch.randn(args.batch, args.seq, args.n_embd, device=device, dtype=dtype)
    shape = (args.batch, args.seq, args.n_embd)

    print(f"Config: n_embd={args.n_embd}, intermediate={4*args.n_embd}, K={args.k}, batch={args.batch}, seq={args.seq}")
    print(f"Device: {device}, dtype: {dtype}")
    print(f"Iters: {args.iters} (warmup: {args.warmup})")
    print()

    t_dense = bench_module(dense, x_fixed, args.iters, args.warmup, device)
    t_t1 = bench_module(sparse_t1, x_fixed, args.iters, args.warmup, device)
    t_t2 = bench_module(sparse_t2, x_fixed, args.iters, args.warmup, device)
    t_t3_warm = bench_module(sparse_t3, x_fixed, args.iters, args.warmup, device)

    t_t3_cold = bench_module_random_x(sparse_t3, args.iters, args.warmup, device, dtype, shape)

    print(f"{'Mode':<35} {'ms/call':>10} {'speedup':>10}")
    print("-" * 57)
    print(f"{'Dense MLP (baseline)':<35} {t_dense:>10.4f} {1.0:>10.2f}x")
    print(f"{'Sparse Tier 1 (TopK exact)':<35} {t_t1:>10.4f} {t_dense/t_t1:>10.2f}x")
    print(f"{'Sparse Tier 2 (TopK + Router)':<35} {t_t2:>10.4f} {t_dense/t_t2:>10.2f}x")
    print(f"{'Sparse Tier 1 + cache (warm)':<35} {t_t3_warm:>10.4f} {t_dense/t_t3_warm:>10.2f}x")
    print(f"{'Sparse Tier 1 + cache (cold, random x)':<35} {t_t3_cold:>10.4f} {t_dense/t_t3_cold:>10.2f}x")

    # Print sparse_t3 cache stats for sanity
    hits, misses = sparse_t3.cache_stats()
    print()
    print(f"Tier 3 final cache stats: hits={hits}, misses={misses}, hit_rate={hits/(hits+misses+1e-9):.2%}")


if __name__ == "__main__":
    main()
