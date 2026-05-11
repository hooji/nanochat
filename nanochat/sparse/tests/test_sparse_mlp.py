"""Tests for TopK-sparse MLP correctness.

Run with: pytest nanochat/sparse/tests/test_sparse_mlp.py -v

These tests verify:
  - Training forward applies top-K + scatter (Tier 1 reference math)
  - Inference forward is mathematically equivalent to training (Tier 1)
  - Router has a real nonlinearity (V(GELU(U(·))) form, not collapsed linear)
  - Router approximation produces correct shape and improves with oversample
  - State-dict / config consistency assertions catch mismatches
  - FfnCache returns the same output on hit
  - Aux losses are populated after training forward, have provable [1, I] bounds
    (catches both the K^2 scaling bug and the anti-correlation bug; see
    docs/results.md Phase A for the diagnostic traces)
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
    gather_router_params,
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


def test_router_has_nonlinearity():
    """Router must be a true 2-layer MLP V(GELU(U(x))), not a collapsed linear V@U.

    Detection: for a purely linear router, router(-x) == -router(x). GELU
    breaks this antisymmetry. If router(x) + router(-x) is meaningfully
    nonzero, a nonlinearity is present.
    """
    torch.manual_seed(31)
    config = _FakeConfig(sparse_use_router=True, sparse_router_rank=4)
    mlp = TopKSparseMLP(config)
    # Set non-trivial weights so the test isn't trivially satisfied by zeros.
    with torch.no_grad():
        mlp.router.u.weight.normal_(std=0.5)
        mlp.router.v.weight.normal_(std=0.5)

    x = torch.randn(1, 1, mlp.n_embd)
    out_pos = mlp.router(x)
    out_neg = mlp.router(-x)
    sym_diff = (out_pos + out_neg).abs().mean()
    assert sym_diff.item() > 1e-3, (
        f"router appears to be linear: |router(x) + router(-x)| ≈ 0 "
        f"(got {sym_diff.item():.2e}). Expected a nonlinearity between U and V; "
        f"without it V(U(x)) collapses to a rank-{config.sparse_router_rank} "
        f"linear projection."
    )


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


def test_gather_router_params_returns_only_router_params():
    """gather_router_params should return router params and nothing else."""
    import torch.nn as nn

    config = _FakeConfig(sparse_use_router=True, sparse_router_rank=8)
    mlp = TopKSparseMLP(config)
    mlp.init_router_weights()
    model = nn.ModuleList([mlp])

    router_params = gather_router_params(model)
    # Router has two Linear modules (u and v), each with a single weight
    assert len(router_params) == 2
    # Verify they are exactly the router weights
    router_param_ids = {id(p) for p in router_params}
    expected_ids = {id(mlp.router.u.weight), id(mlp.router.v.weight)}
    assert router_param_ids == expected_ids


def test_gather_router_params_empty_when_no_router():
    """gather_router_params should return empty when no router is configured."""
    import torch.nn as nn

    mlp = TopKSparseMLP(_FakeConfig(sparse_use_router=False))
    model = nn.ModuleList([mlp])
    assert gather_router_params(model) == []


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


def test_aux_loss_bounded_in_one_to_intermediate():
    """Participation-ratio aux loss is provably bounded in [1, intermediate].

    Regression guard for two distinct bugs:
      - The K^2 scaling bug (un-normalized load*importance times intermediate)
        produced aux ~ K^2 * avg_h, far above intermediate. Caught by the
        upper bound check.
      - The anti-correlation bug (normalized load*importance) allowed aux < 1
        when load and importance were anti-correlated specialist/generalist.
        Caught by the lower bound check.

    The current formula `I * sum(imp_norm^2)` is provably in [1, I]:
      - Lower bound by Cauchy-Schwarz / power-mean: sum(p_i^2) >= 1/n for p
        a distribution over n points, with equality at uniform p_i = 1/n.
      - Upper bound: sum(p_i^2) <= 1 with equality at full concentration.
      - Multiplied by I gives [1, I].
    """
    torch.manual_seed(17)
    # Pick K large enough that the old K^2 bug would obviously violate the
    # upper bound (K^2 = 1024 vs intermediate = 128).
    config = _FakeConfig(n_embd=32, sparse_k=32)
    mlp = TopKSparseMLP(config)
    mlp.train()
    x = torch.randn(4, 64, 32)
    _ = mlp(x)
    aux_l, _ = mlp.aux_loss()
    intermediate = 4 * config.n_embd
    assert aux_l.item() >= 1.0 - 1e-3, (
        f"aux_loss = {aux_l.item():.4f} below the theoretical floor of 1.0. "
        f"Under the participation-ratio form (I * sum(p_i^2)) this is impossible "
        f"— the anti-correlation regression may have returned."
    )
    assert aux_l.item() <= intermediate + 1e-3, (
        f"aux_loss = {aux_l.item():.4f} exceeds intermediate={intermediate}. "
        f"Under the participation-ratio form this is impossible — the K^2 "
        f"scaling regression may have returned."
    )


def test_router_target_is_magnitude_weighted_not_uniform():
    """The router CE target should be h_sparse / h_sparse.sum, not uniform 1/K.

    Detection: with a non-trivial router (post-init normal weights on V), the
    router CE loss with the magnitude-weighted target should differ from what
    the uniform-1/K target would produce, given that top_vals vary in magnitude.
    """
    torch.manual_seed(43)
    config = _FakeConfig(n_embd=32, sparse_k=4, sparse_use_router=True)
    mlp = TopKSparseMLP(config)
    mlp.init_router_weights()
    # Push router.v away from zero so router_scores are nontrivial.
    with torch.no_grad():
        mlp.router.v.weight.normal_(std=0.5)
    mlp.train()
    x = torch.randn(1, 1, 32)
    _ = mlp(x)
    _, router_l_magnitude = mlp.aux_loss()

    # Reproduce relevant forward bits to compare against the uniform-target form.
    h = mlp.c_fc(x)
    h = torch.relu(h).square()
    top_vals, top_idx = torch.topk(h, mlp.k, dim=-1)
    router_scores = mlp.router(x)
    log_probs = torch.log_softmax(router_scores, dim=-1)

    with torch.no_grad():
        uniform_target = torch.zeros_like(router_scores)
        uniform_target.scatter_(-1, top_idx, 1.0 / mlp.k)
    uniform_loss = -(uniform_target * log_probs).sum(dim=-1).mean()

    assert not torch.allclose(router_l_magnitude, uniform_loss, atol=1e-5), (
        f"Router loss matches uniform-target form (mag={router_l_magnitude.item():.4f}, "
        f"uniform={uniform_loss.item():.4f}); target may have regressed to uniform."
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
