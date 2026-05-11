"""TopK-sparse FFN for nanochat (opt-in).

Inspired by the LARQL "FFN as graph database" view: the intermediate dimension
of an MLP behaves like a content-addressable bank of features, and only a small
fraction fires meaningfully on any given token. This module makes that sparsity
explicit at training time (so the model learns to keep its active set small)
and exploits it at inference (so weight bandwidth scales with K, not the full
intermediate dim).

Three tiers, all opt-in via GPTConfig flags:

  Tier 1 (TopK):     Forward computes c_fc, applies ReLU^2, then top-K + scatter.
                     Mathematically equivalent at training and inference; the
                     inference path skips the dense c_proj matmul in favor of
                     a K-sparse gather.

  Tier 2 (Router):   A low-rank predictor learns to identify the active set
                     before c_fc. At inference, only K' = K * oversample rows
                     of c_fc are computed (gather instead of full matmul).
                     Approximation controlled by router_oversample.

  Tier 3 (FfnCache): Hashes the top-K index set, caches the projection output.
                     Inference-only, off by default, enable via helper.

When GPTConfig.sparse_ffn is False (the default), this module is not imported
or instantiated and the model behaves bit-identically to vanilla nanochat.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# We reuse the cast-on-forward Linear from gpt.py. This is safe because gpt.py
# only imports from us conditionally inside Block.__init__, so there is no
# circular import at module load time.
from nanochat.gpt import Linear


class Router(nn.Module):
    """Low-rank predictor of which intermediate features will fire.

    Computes a score per intermediate feature as V(U(x)) where U is
    [n_embd, rank] and V is [rank, intermediate]. The caller selects the
    top-K' candidates and computes c_fc only on those rows.
    """

    def __init__(self, n_embd: int, intermediate: int, rank: int):
        super().__init__()
        self.u = Linear(n_embd, rank, bias=False)
        self.v = Linear(rank, intermediate, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.v(self.u(x))


class FfnCache:
    """L1 FFN output cache: hash(sorted top-K indices) -> output tensor.

    Used at inference only. Single-token (B=1, T=1) entries; larger inputs
    bypass the cache. Bounded by max_entries with no LRU (matches LARQL).
    """

    def __init__(self, max_entries: int = 4096):
        self.max_entries = max_entries
        self.store: dict = {}
        self.hits = 0
        self.misses = 0

    def _key(self, top_idx_1d: torch.Tensor):
        sorted_idx, _ = torch.sort(top_idx_1d)
        return tuple(sorted_idx.cpu().tolist())

    def get(self, top_idx_1d: torch.Tensor):
        key = self._key(top_idx_1d)
        out = self.store.get(key)
        if out is None:
            self.misses += 1
        else:
            self.hits += 1
        return out

    def put(self, top_idx_1d: torch.Tensor, output_1d: torch.Tensor) -> None:
        if len(self.store) >= self.max_entries:
            return
        key = self._key(top_idx_1d)
        self.store[key] = output_1d.detach()

    def stats(self):
        return self.hits, self.misses

    def reset(self) -> None:
        self.store.clear()
        self.hits = 0
        self.misses = 0


class TopKSparseMLP(nn.Module):
    """Top-K sparse FFN, drop-in replacement for nanochat's vanilla MLP.

    Shape signature matches MLP exactly so the surrounding Block code is
    unaware of the swap:
        c_fc:   Linear(n_embd,   4*n_embd, bias=False)
        c_proj: Linear(4*n_embd, n_embd,   bias=False)

    forward() branches on training vs. inference mode. Training runs the full
    dense c_fc and applies top-K + scatter so gradients flow only through the K
    active features. Inference gathers only the K columns of c_proj. With the
    router enabled, the full c_fc is skipped in favor of computing only the
    K' candidate rows the router selects.

    Auxiliary losses (load-balancing + router training) are stored on the
    module after each training forward and collected via collect_aux_losses
    for inclusion in the training loss.
    """

    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.intermediate = 4 * config.n_embd
        self.k = int(config.sparse_k)
        assert 0 < self.k <= self.intermediate, (
            f"sparse_k must be in (0, {self.intermediate}], got {self.k}"
        )

        self.c_fc = Linear(self.n_embd, self.intermediate, bias=False)
        self.c_proj = Linear(self.intermediate, self.n_embd, bias=False)

        self.use_router = bool(config.sparse_use_router)
        if self.use_router:
            self.router = Router(
                self.n_embd,
                self.intermediate,
                rank=int(config.sparse_router_rank),
            )
            self.router_oversample = int(config.sparse_router_oversample)
        else:
            self.router = None
            self.router_oversample = 1

        # Inference state (not persisted, not part of state_dict).
        self._cache_enabled = False
        self._cache = None

        # Aux losses set by _train_forward; collected by collect_aux_losses.
        self._last_aux_loss = None
        self._last_router_loss = None

    # ---------- public mode controls (inference only) ----------

    def enable_cache(self, max_entries: int = 4096) -> None:
        self._cache = FfnCache(max_entries=max_entries)
        self._cache_enabled = True

    def disable_cache(self) -> None:
        self._cache_enabled = False
        self._cache = None

    def cache_stats(self):
        return self._cache.stats() if self._cache is not None else (0, 0)

    def aux_loss(self):
        return self._last_aux_loss, self._last_router_loss

    def init_router_weights(self) -> None:
        """Called by GPT.init_weights() after the main weight init.

        Convention:
          - router.u: uniform with std = 1/sqrt(n_embd) (same scale as Q/K/V)
          - router.v: zero init so the router starts neutral; cross-entropy
            against true top-K shapes it during training.
        """
        if self.router is None:
            return
        s = 3 ** 0.5 * self.n_embd ** -0.5
        torch.nn.init.uniform_(self.router.u.weight, -s, s)
        torch.nn.init.zeros_(self.router.v.weight)

    def assert_state_dict_matches_config(self, state_dict, prefix: str = "") -> None:
        """Defensive check that the state_dict matches our config.

        Catches the case where someone hand-edits sparse_use_router in a config
        but the checkpoint was trained without it (or vice versa).
        """
        router_keys = [k for k in state_dict.keys() if k.startswith(f"{prefix}router.")]
        if self.use_router:
            assert len(router_keys) > 0, (
                f"sparse_use_router=True but no '{prefix}router.*' keys in state_dict; "
                f"checkpoint and config are out of sync."
            )
        else:
            assert len(router_keys) == 0, (
                f"sparse_use_router=False but found {len(router_keys)} "
                f"'{prefix}router.*' keys in state_dict; checkpoint and config "
                f"are out of sync."
            )

    # ---------- forward dispatch ----------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self._train_forward(x)
        if self._cache_enabled:
            return self._cached_inference(x)
        if self.router is not None:
            return self._router_inference(x)
        return self._sparse_inference(x)

    # ---------- training reference path ----------

    def _train_forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, n_embd]
        h = self.c_fc(x)                                       # [B, T, I]
        h = F.relu(h).square()                                 # natural sparsity
        top_vals, top_idx = torch.topk(h, self.k, dim=-1)      # [B, T, K]
        h_sparse = torch.zeros_like(h)
        h_sparse.scatter_(-1, top_idx, top_vals)               # K-nonzero per row

        # Load-balancing aux loss (Switch-Transformer eq. 4, Fedus et al. 2022).
        #
        # Switch's L_aux = N * sum_i(f_i * P_i), where f and P are both
        # probability distributions over N experts:
        #   f_i = fraction of tokens routed to expert i      (sum_i f_i = 1)
        #   P_i = mean router gate probability for expert i   (sum_i P_i = 1)
        # Under uniform routing this gives aux = N * sum(1/N^2) = 1.
        # Under collapse to one expert it gives aux = N.
        #
        # For TopK FFN the analogs are:
        #   load[f]       = E[1{f in top-K}] over (batch, seq)
        #   importance[f] = E[h_sparse[f]]   over (batch, seq)
        # Neither sums to 1 by construction (load sums to K; importance has no
        # natural scale). Without explicit normalization the product of
        # (load * importance) scales as K^2, and multiplying by I over-scales
        # the loss by an additional factor — making aux dominate the LM loss
        # for any reasonable K. This was the bug that took out the first
        # smoke test; see docs/results.md Phase A.
        importance = h_sparse.mean(dim=(0, 1))                  # [I]
        with torch.no_grad():
            load = (h_sparse > 0).float().mean(dim=(0, 1))      # [I]
        imp_norm = importance / (importance.sum() + 1e-9)       # sums to 1
        load_norm = load / (load.sum() + 1e-9)                  # sums to 1
        # aux ∈ [1, I]: 1 at uniform usage, I at total collapse to one feature.
        self._last_aux_loss = self.intermediate * (load_norm * imp_norm).sum()

        # Router training: cross-entropy of router scores against true top-K.
        if self.router is not None:
            router_scores = self.router(x)                      # [B, T, I]
            with torch.no_grad():
                target = torch.zeros_like(router_scores)
                target.scatter_(-1, top_idx, 1.0 / self.k)      # uniform over K
            log_probs = F.log_softmax(router_scores, dim=-1)
            self._last_router_loss = -(target * log_probs).sum(dim=-1).mean()
        else:
            self._last_router_loss = None

        return self.c_proj(h_sparse)

    # ---------- inference paths ----------

    def _sparse_inference(self, x: torch.Tensor) -> torch.Tensor:
        """Tier 1: full c_fc, top-K, gather c_proj columns. Exact."""
        h = self.c_fc(x)
        h = F.relu(h).square()
        top_vals, top_idx = torch.topk(h, self.k, dim=-1)
        return self._sparse_proj(top_vals, top_idx)

    def _router_inference(self, x: torch.Tensor) -> torch.Tensor:
        """Tier 2: router picks K' candidates, then top-K from those. Approximate."""
        assert self.router is not None
        k_prime = min(self.k * self.router_oversample, self.intermediate)
        router_scores = self.router(x)                          # [B, T, I]
        _, candidates = torch.topk(router_scores, k_prime, dim=-1)  # [B, T, K']

        # Gather K' rows of c_fc.weight and compute scores for those features only.
        w_fc = self.c_fc.weight.to(x.dtype)                     # [I, n_embd]
        B, T, Kp = candidates.shape
        flat_cand = candidates.reshape(-1, Kp)                  # [B*T, K']
        w_fc_g = w_fc[flat_cand]                                # [B*T, K', n_embd]
        x_flat = x.reshape(-1, 1, self.n_embd)                  # [B*T, 1, n_embd]
        h_cand = (x_flat * w_fc_g).sum(dim=-1).reshape(B, T, Kp)  # [B, T, K']

        h_cand = F.relu(h_cand).square()
        top_vals_cand, top_pos = torch.topk(h_cand, self.k, dim=-1)    # [B, T, K]
        top_idx = torch.gather(candidates, -1, top_pos)                # global ids
        return self._sparse_proj(top_vals_cand, top_idx)

    def _sparse_proj(self, top_vals: torch.Tensor, top_idx: torch.Tensor) -> torch.Tensor:
        """Shared K-sparse output projection.

        top_vals, top_idx: [B, T, K]
        c_proj.weight:     [n_embd, I]
        Output:            [B, T, n_embd]
        """
        w_proj_t = self.c_proj.weight.to(top_vals.dtype).t()    # [I, n_embd]
        B, T, K = top_idx.shape
        flat_idx = top_idx.reshape(-1, K)                       # [B*T, K]
        w_g = w_proj_t[flat_idx]                                # [B*T, K, n_embd]
        v_flat = top_vals.reshape(-1, K, 1)                     # [B*T, K, 1]
        out = (v_flat * w_g).sum(dim=1)                         # [B*T, n_embd]
        return out.reshape(B, T, self.n_embd)

    def _cached_inference(self, x: torch.Tensor) -> torch.Tensor:
        """Tier 3: hash top-K indices, return cached output on hit.

        Only fires for B=1, T=1 (decode). Multi-token inputs bypass to the
        exact path.
        """
        if x.shape[0] != 1 or x.shape[1] != 1:
            return self._router_inference(x) if self.router is not None else self._sparse_inference(x)
        assert self._cache is not None

        # We still derive the top-K indices to key the cache. Use the router
        # if available (cheaper) else fall back to full c_fc.
        if self.router is not None:
            k_prime = min(self.k * self.router_oversample, self.intermediate)
            router_scores = self.router(x)
            _, candidates = torch.topk(router_scores, k_prime, dim=-1)
            w_fc = self.c_fc.weight.to(x.dtype)
            flat_cand = candidates.reshape(-1, k_prime)
            w_fc_g = w_fc[flat_cand]
            x_flat = x.reshape(-1, 1, self.n_embd)
            h_cand = (x_flat * w_fc_g).sum(dim=-1).reshape(1, 1, k_prime)
            h_cand = F.relu(h_cand).square()
            top_vals_cand, top_pos = torch.topk(h_cand, self.k, dim=-1)
            top_idx = torch.gather(candidates, -1, top_pos)
            top_vals = top_vals_cand
        else:
            h = self.c_fc(x)
            h = F.relu(h).square()
            top_vals, top_idx = torch.topk(h, self.k, dim=-1)

        cached = self._cache.get(top_idx.reshape(-1))
        if cached is not None:
            return cached.view(1, 1, self.n_embd)
        out = self._sparse_proj(top_vals, top_idx)
        self._cache.put(top_idx.reshape(-1), out.reshape(-1))
        return out


# ---------- module-level helpers (used by gpt.py and inference scripts) ----------

def collect_aux_losses(model: nn.Module):
    """Sum aux losses across all TopKSparseMLP submodules.

    Returns (load_balance_total, router_total) as scalar tensors on the model's
    device. Returns zero tensors if no sparse MLPs are present.
    """
    device = next(model.parameters()).device
    aux_total = torch.zeros((), device=device)
    router_total = torch.zeros((), device=device)
    for module in model.modules():
        if isinstance(module, TopKSparseMLP):
            aux_l, router_l = module.aux_loss()
            if aux_l is not None:
                aux_total = aux_total + aux_l
            if router_l is not None:
                router_total = router_total + router_l
    return aux_total, router_total


def enable_sparse_cache(model: nn.Module, max_entries: int = 4096) -> int:
    """Enable FfnCache on every TopKSparseMLP in `model`. Returns count enabled."""
    n = 0
    for module in model.modules():
        if isinstance(module, TopKSparseMLP):
            module.enable_cache(max_entries=max_entries)
            n += 1
    return n


def disable_sparse_cache(model: nn.Module) -> int:
    n = 0
    for module in model.modules():
        if isinstance(module, TopKSparseMLP):
            module.disable_cache()
            n += 1
    return n


def sparse_cache_stats(model: nn.Module):
    """Sum cache stats across all TopKSparseMLPs in `model`."""
    hits, misses = 0, 0
    for module in model.modules():
        if isinstance(module, TopKSparseMLP):
            h, m = module.cache_stats()
            hits += h
            misses += m
    return hits, misses


def init_all_router_weights(model: nn.Module) -> int:
    """Walk model and init router weights for every TopKSparseMLP. Returns count."""
    n = 0
    for module in model.modules():
        if isinstance(module, TopKSparseMLP):
            module.init_router_weights()
            n += 1
    return n


def assert_state_dict_consistency(model: nn.Module, state_dict) -> None:
    """For every TopKSparseMLP, assert state_dict has matching router keys."""
    for name, module in model.named_modules():
        if isinstance(module, TopKSparseMLP):
            module.assert_state_dict_matches_config(state_dict, prefix=f"{name}.")
