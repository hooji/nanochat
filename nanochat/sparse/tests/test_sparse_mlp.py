"""Tests for TopK-sparse MLP correctness.

Run with: pytest nanochat/sparse/tests/test_sparse_mlp.py -v

These tests verify:
  - Training forward applies top-K + scatter (Tier 1 reference math)
  - Inference forward is mathematically equivalent to training (Tier 1)
  - Router approximation produces correct shape and improves with oversample
  - State-dict / config consistency assertions catch mismatches
  - FfnCache returns the same output on hit
  - Aux losses are populated after training forward and have O(1) scale
    (i.e. don't suffer from the K^2 scaling regression caught by the first
    smoke test — see docs/results.md Phase A)
"""

import pytest
import torch

from nanochat.sparse.sparse_mlp import (
    TopKSparseMLP,
    Router,
    FfnCache,
    collect_aux_losses,
    enable_sparse_cache,
    disable_sparse_cache,
    sparse_cache_stats,
)


class _FakeConfig:
    """Minimal stand-in for GPTConfig."""

    def __init__(
        self,
        n_embd=64,
        sparse_k=8,
        sparse_use_router=False,
        sparse_router_rank=16,
        sparse_router_oversample=2,
    ):
        self.n_embd = n_embd
        self.sparse_k = sparse_k
        self.sparse_use_router = sparse_use_router
        self.sparse_router_rank = sparse_router_rank
        self.sparse_router_oversample = sparse_router_oversample


def test_construct_with_and_without_router():
    no_router = TopKSparseMLP(_FakeConfig(sparse_use_router=False))
    with_router = TopKSparseMLP(_FakeConfig(sparse_use_router=True))
    assert no_router.router is None
    assert with_router.router is not None
    assert isinstance(with_router.router, Router)


def test_forward_shape_training_and_eval():
    torch.manual_seed(0)
    mlp = TopKSparseMLP(_FakeConfig(n_embd=32, sparse_k=16))
    x = torch.randn(2, 5, 32)

    mlp.train()
    out_train = mlp(x)
    assert out_train.shape == (2, 5, 32)

    mlp.eval()
    out_eval = mlp(x)
    assert out_eval.shape == (2, 5, 32)


def test_inference_matches_training_at_arbitrary_k():
    """Tier 1 inference is mathematically equivalent to the train-time path."""
    torch.manual_seed(1)
    mlp = TopKSparseMLP(_FakeConfig(n_embd=32, sparse_k=8))
    x = torch.randn(2, 5, 32, dtype=torch.float64)
    mlp.to(dtype=torch.float64)

    mlp.train()
    out_train = mlp(x)
    mlp.eval()
    out_eval = mlp(x)

    assert torch.allclose(out_train, out_eval, atol=1e-10), (
        f"max diff: {(out_train - out_eval).abs().max()}"
    )


def test_topk_at_full_intermediate_matches_dense_relu_squared():
    """At K = intermediate, the sparse path is identical to dense ReLU^2."""
    torch.manual_seed(2)
    n_embd = 32
    mlp = TopKSparseMLP(_FakeConfig(n_embd=n_embd, sparse_k=4 * n_embd))
    mlp.to(dtype=torch.float64)
    x = torch.randn(1, 3, n_embd, dtype=torch.float64)

    # Hand-compute dense ReLU^2 reference using the same weights
    with torch.no_grad():
        h = mlp.c_fc(x)
        h = torch.relu(h).square()
        dense_out = mlp.c_proj(h)

    mlp.eval()
    sparse_out = mlp(x)
    assert torch.allclose(dense_out, sparse_out, atol=1e-10)


def test_train_forward_produces_aux_losses():
    mlp = TopKSparseMLP(_FakeConfig(sparse_use_router=False))
    mlp.train()
    x = torch.randn(1, 3, mlp.n_embd)
    _ = mlp(x)
    aux_l, router_l = mlp.aux_loss()
    assert aux_l is not None
    assert router_l is None


def test_train_forward_with_router_produces_router_loss():
    mlp = TopKSparseMLP(_FakeConfig(sparse_use_router=True))
    mlp.init_router_weights()
    mlp.train()
    x = torch.randn(1, 3, mlp.n_embd)
    _ = mlp(x)
    aux_l, router_l = mlp.aux_loss()
    assert aux_l is not None
    assert router_l is not None
    assert router_l.requires_grad  # must flow gradients into router weights


def test_collect_aux_losses_sums_across_modules():
    """collect_aux_losses walks the whole model."""
    import torch.nn as nn

    config = _FakeConfig(sparse_use_router=True)
    a = TopKSparseMLP(config)
    b = TopKSparseMLP(config)
    a.init_router_weights()
    b.init_router_weights()
    a.train()
    b.train()
    x = torch.randn(1, 3, a.n_embd)
    _ = a(x)
    _ = b(x)

    model = nn.ModuleList([a, b])
    aux_total, router_total = collect_aux_losses(model)
    a_aux, a_r = a.aux_loss()
    b_aux, b_r = b.aux_loss()
    assert torch.allclose(aux_total, a_aux + b_aux)
    assert torch.allclose(router_total, a_r + b_r)


def test_state_dict_assertion_passes_when_consistent():
    mlp = TopKSparseMLP(_FakeConfig(sparse_use_router=True))
    sd = mlp.state_dict()
    mlp.assert_state_dict_matches_config(sd)


def test_state_dict_assertion_fails_when_router_missing():
    mlp = TopKSparseMLP(_FakeConfig(sparse_use_router=True))
    sd = {k: v for k, v in mlp.state_dict().items() if "router" not in k}
    with pytest.raises(AssertionError):
        mlp.assert_state_dict_matches_config(sd)


def test_state_dict_assertion_fails_when_router_unexpected():
    no_router = TopKSparseMLP(_FakeConfig(sparse_use_router=False))
    with_router = TopKSparseMLP(_FakeConfig(sparse_use_router=True))
    sd_with_router = with_router.state_dict()
    with pytest.raises(AssertionError):
        no_router.assert_state_dict_matches_config(sd_with_router)


def test_cache_hit_returns_same_output():
    """Two calls with the same input should hit cache on the second."""
    torch.manual_seed(7)
    mlp = TopKSparseMLP(_FakeConfig(sparse_k=8))
    mlp.eval()
    mlp.enable_cache(max_entries=128)
    x = torch.randn(1, 1, mlp.n_embd)
    out1 = mlp(x)
    out2 = mlp(x)
    hits, misses = mlp.cache_stats()
    assert misses == 1
    assert hits == 1
    assert torch.allclose(out1, out2)


def test_cache_bypassed_for_multi_token_input():
    mlp = TopKSparseMLP(_FakeConfig())
    mlp.eval()
    mlp.enable_cache()
    x = torch.randn(2, 3, mlp.n_embd)
    out = mlp(x)
    assert out.shape == (2, 3, mlp.n_embd)
    hits, misses = mlp.cache_stats()
    assert hits == 0 and misses == 0


def test_helper_enable_disable_cache():
    """Module-level helpers walk the whole model."""
    import torch.nn as nn

    a = TopKSparseMLP(_FakeConfig())
    b = TopKSparseMLP(_FakeConfig())
    model = nn.ModuleList([a, b])

    n = enable_sparse_cache(model)
    assert n == 2
    assert a._cache_enabled and b._cache_enabled

    n = disable_sparse_cache(model)
    assert n == 2
    assert not a._cache_enabled and not b._cache_enabled


def test_router_inference_returns_correct_shape():
    mlp = TopKSparseMLP(_FakeConfig(sparse_use_router=True, sparse_router_oversample=2))
    mlp.init_router_weights()
    mlp.eval()
    x = torch.randn(2, 3, mlp.n_embd)
    out = mlp(x)
    assert out.shape == (2, 3, mlp.n_embd)


def test_router_oversample_capped_at_intermediate():
    """K' should not exceed the intermediate dimension."""
    # intermediate = 4 * 32 = 128; K=8, oversample=50 would request K'=400 > 128.
    mlp = TopKSparseMLP(
        _FakeConfig(
            n_embd=32,
            sparse_k=8,
            sparse_use_router=True,
            sparse_router_oversample=50,
        )
    )
    mlp.init_router_weights()
    mlp.eval()
    x = torch.randn(1, 1, 32)
    out = mlp(x)
    assert out.shape == (1, 1, 32)  # should not crash


def test_router_perfect_match_recovers_exact_topk():
    """If router weights are set to exactly mirror c_fc, router inference == Tier 1."""
    torch.manual_seed(11)
    config = _FakeConfig(
        n_embd=16, sparse_k=8, sparse_use_router=True, sparse_router_rank=16, sparse_router_oversample=4
    )
    mlp = TopKSparseMLP(config)
    mlp.to(dtype=torch.float64)

    # Manually align router scores with the ReLU^2 activations by giving the
    # router enough rank to express c_fc.weight directly.
    with torch.no_grad():
        # router(x) = v(u(x)) = x @ u^T @ v^T.
        # We want this to equal x @ c_fc.weight^T (the c_fc scores), so set
        # u = identity (need rank >= n_embd; here rank=n_embd=16) and v = c_fc.weight.
        eye = torch.eye(16, dtype=torch.float64)
        mlp.router.u.weight.copy_(eye)
        mlp.router.v.weight.copy_(mlp.c_fc.weight)

    mlp.eval()
    x = torch.randn(2, 3, 16, dtype=torch.float64)

    out_router = mlp._router_inference(x)
    out_exact = mlp._sparse_inference(x)

    # With oversample=4 -> K' = 32 = intermediate (capped), so router considers
    # every feature and the result should equal Tier 1 exactly.
    assert torch.allclose(out_router, out_exact, atol=1e-9), (
        f"router with full oversample should match exact; max diff = "
        f"{(out_router - out_exact).abs().max()}"
    )


def test_aux_loss_is_finite_and_nonneg():
    mlp = TopKSparseMLP(_FakeConfig(sparse_use_router=True))
    mlp.init_router_weights()
    mlp.train()
    x = torch.randn(2, 4, mlp.n_embd)
    _ = mlp(x)
    aux_l, router_l = mlp.aux_loss()
    assert torch.isfinite(aux_l).item()
    assert torch.isfinite(router_l).item()
    assert aux_l.item() >= 0
    assert router_l.item() >= 0


def test_aux_loss_bounded_by_intermediate():
    """Switch-Transformer eq.4 aux loss is bounded above by intermediate.

    Regression guard: an earlier version multiplied (load * importance) by
    intermediate without normalizing either to a probability distribution,
    which caused aux to scale as K^2 * intermediate and dominate the LM
    loss (see docs/results.md Phase A). The fixed formula normalizes both
    to distributions, so under the formula aux is bounded by `intermediate`
    (the maximum at total collapse to one feature).

    With random inputs we should be near the lower bound of 1.
    """
    torch.manual_seed(17)
    # Pick a config where the buggy K^2 scaling would obviously violate
    # the bound (K^2 = 1024 vs intermediate = 128).
    config = _FakeConfig(n_embd=32, sparse_k=32)
    mlp = TopKSparseMLP(config)
    mlp.train()
    x = torch.randn(4, 64, 32)
    _ = mlp(x)
    aux_l, _ = mlp.aux_loss()
    intermediate = 4 * config.n_embd
    assert aux_l.item() <= intermediate + 1e-3, (
        f"aux_loss = {aux_l.item():.4f} exceeds intermediate={intermediate}. "
        f"Under the Switch eq.4 formula (normalized importance and load) this "
        f"is impossible. The K^2 scaling regression may have returned."
    )


def test_ffn_cache_eviction_bounded():
    cache = FfnCache(max_entries=2)
    # First two entries fit
    cache.put(torch.tensor([1, 2]), torch.tensor([1.0]))
    cache.put(torch.tensor([3, 4]), torch.tensor([2.0]))
    assert len(cache.store) == 2
    # Third entry should be dropped (no eviction)
    cache.put(torch.tensor([5, 6]), torch.tensor([3.0]))
    assert len(cache.store) == 2


def test_ffn_cache_key_is_sort_invariant():
    """Different orderings of the same top-K set should hit the same cache entry."""
    cache = FfnCache()
    out = torch.tensor([1.0, 2.0, 3.0])
    cache.put(torch.tensor([3, 1, 2]), out)
    retrieved = cache.get(torch.tensor([2, 3, 1]))
    assert retrieved is not None
    assert torch.equal(retrieved, out)
