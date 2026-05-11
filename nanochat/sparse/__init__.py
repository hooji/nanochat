"""Opt-in TopK-sparse FFN training and inference for nanochat.

This subpackage is loaded only when GPTConfig.sparse_ffn=True. The vanilla
nanochat code path does not import from here.

See docs/README.md for the high-level overview and docs/test-request.md for
the Train/Test team handoff.
"""

from nanochat.sparse.sparse_mlp import (
    TopKSparseMLP,
    Router,
    FfnCache,
    collect_aux_losses,
    enable_sparse_cache,
    disable_sparse_cache,
    sparse_cache_stats,
    init_all_router_weights,
    assert_state_dict_consistency,
)

__all__ = [
    "TopKSparseMLP",
    "Router",
    "FfnCache",
    "collect_aux_losses",
    "enable_sparse_cache",
    "disable_sparse_cache",
    "sparse_cache_stats",
    "init_all_router_weights",
    "assert_state_dict_consistency",
]
