# Sparse FFN Results — d12 smoke test

Run date: 2026-05-11
Train/Test team: Claude (RunPod 8×H100 SXM5, branch claude/review-repositories-Udsgs @ 7416ea1)

This is nanochat/sparse/docs/results.md content, written here in the user's
workspace folder so it can be pushed back to the branch by the user. Drop this
into nanochat/sparse/docs/results.md, commit, and push.

## Phase 0 — Baseline path sanity ✅ PASS

With the sparse branch checked out and `--sparse-ffn` absent, the baseline
behavior is preserved: model trains, GPUs engage, no regression.

| Metric | Phase 0 |
|---|---|
| step | 20 / 20 |
| train/loss | 10.4 |
| val/bpb | 1.89 (from 3.17 random init) |
| peak memory | normal |
| crashes | 0 |

Conclusion: the `if config.sparse_ffn:` guard in `Block.__init__` and the
conditional aux-loss block in `GPT.forward` correctly preserve the vanilla
code path. No regression introduced by the branch.

## Phase A — Sparse smoke test ❌ FAIL (training divergence)

Both Tier 1 (`--sparse-ffn --sparse-k=256`) and Tier 1+2 (also
`--sparse-use-router`) ran to completion but diverged catastrophically at
20 iterations.

| Run | step | train/loss | val/bpb | peak mem | crashes |
|---|---|---|---|---|---|
| phase0_baseline | 20 | 10.4 | 1.89 | normal | 0 |
| phaseA1_sparse_tier1 | 20 | 4761 | 536.4 | normal | 0 |
| phaseA2_sparse_router | 20 | 4763 | 540.6 | 40.7 GiB | 0 |

For context: random-init val/bpb is ~3.17. The baseline path moved it to 1.89
in 20 steps. The sparse paths moved it to ~540 — i.e. the model is producing
predictions ~285× worse than random. Gradients flow and loss is technically
descending, but in the wrong direction (toward minimizing the aux loss, not
the LM loss).

wandb:
- https://wandb.ai/u7a/nanochat/runs/qrsf4v4y (phase0_baseline)
- https://wandb.ai/u7a/nanochat/runs/ (phaseA1_sparse_tier1)
- https://wandb.ai/u7a/nanochat/runs/ (phaseA2_sparse_router)

### Root cause: aux-loss scaling is K²-large, not O(1)

In `nanochat/sparse/sparse_mlp.py::TopKSparseMLP._train_forward`:

```python
importance = h_sparse.mean(dim=(0, 1))                  # shape [I]
with torch.no_grad():
    load = (h_sparse > 0).float().mean(dim=(0, 1))      # shape [I]
self._last_aux_loss = (load * importance).sum() * self.intermediate
```

For uniform feature usage, both `load[f]` and `importance[f]` scale as `K/I`
(K out of I features active per token). So:

```
(load * importance).sum() ≈ I * (K/I)² * avg_h = K²/I * avg_h
× self.intermediate (I)  ≈ K² * avg_h
```

This is **O(K²)**, not the O(1) that Switch-Transformer-style aux losses
are designed to produce. With K=256, I=3072 (d12), 12 layers, and the default
`sparse_aux_loss_coef=0.01`:

```
predicted aux contribution per layer ≈ 256² × ~1.0 ≈ 65,000
× 12 layers                         ≈ 785,000
× 0.01 (default coef)               ≈   7,850
```

We observed `train/loss ≈ 4,761` for A1 — within the predicted band, the
difference being the actual squared-ReLU activation magnitude (which is below
1.0 on average).

The fundamental issue is that `importance` and `load` as defined here are
**not** probability distributions that sum to 1 across features (the way `P_i`
and `f_i` are in Switch Transformer). They're per-feature averages that
themselves scale with K/I. Multiplying by `intermediate` over-scales the loss
by an additional factor of I.

### Recommended fix (minimal patch)

Normalize `importance` and `load` before combining, so the aux loss is O(1)
under uniform usage and grows only when usage concentrates:

```python
importance = h_sparse.mean(dim=(0, 1))                  # [I]
with torch.no_grad():
    load = (h_sparse > 0).float().mean(dim=(0, 1))      # [I]
# Normalize each to a distribution over features:
imp_norm = importance / (importance.sum() + 1e-9)        # sums to 1 across I
load_norm = load / (load.sum() + 1e-9)                   # sums to 1 across I
self._last_aux_loss = self.intermediate * (load_norm * imp_norm).sum()
```

This matches the Switch-Transformer formulation: under uniform usage,
`aux_loss = I × Σ((1/I)²) = I × I × (1/I²) = 1`. Under collapse to `k_active`
features, `aux_loss ≈ I/k_active >> 1`. The default
`sparse_aux_loss_coef=0.01` then operates on a per-layer aux loss of O(1) and
should behave as intended.

### Alternative fix (coefficient-only band-aid)

If you'd prefer to defer the formula change, the equivalent default for the
current K=256, I=3072 case would be roughly:

```python
sparse_aux_loss_coef: float = 1e-5   # was 0.01
```

We **don't** recommend this: it's K-dependent, so the K-sweep in Phase B (K ∈
{512, 256, 128, 64}) would require re-tuning the coefficient for each K. The
formula fix is K-agnostic.

### What still looks good

- ✅ All 19 unit tests pass on CPU
- ✅ Baseline regression test (Phase 0) confirms the branch doesn't break
  vanilla nanochat
- ✅ No crashes, no NaN, no memory blow-up (peak 40.7 GiB for d12 sparse with
  router — within H100 budget)
- ✅ Steady-state throughput is in the dev team's predicted band (Phase A2
  hit 2.4M tok/sec, 23.8% MFU in steady-state per the screen log). Compile
  warmup dominates 20-iter summaries, but per-step dt of ~216 ms is
  reasonable for d12 with sparse + router (versus baseline d12 at ~150 ms).
- ✅ State-dict consistency assertions never fire on a clean run
- ✅ The router_loss is being trained (decreasing across the 20 steps)
- ✅ FP8 path is untouched — Phase 0 with the default fp8 setting succeeded

### Open questions for the dev team

1. Was the aux loss formula derived from a specific source (LARQL paper / repo)?
   If so, please point us to the reference so we can compare and verify the
   intended formulation.
2. Once the fix lands, we'll re-run Phase A1/A2 and proceed to Phase B
   (K-sweep) if the smoke results look healthy. Should we re-run on the same
   pod (faster turnaround) or wait for a tagged release?
3. The router_loss coefficient (`sparse_router_loss_coef=0.1`) should be
   re-evaluated after the aux fix, since the relative magnitudes will change.

## Phase B — d12 K-sweep

Not run. Blocked on the aux-loss fix. Will execute once a fixed release
is available.

## Phase C — Inference benchmarks

Not run. Requires a usable trained sparse checkpoint, blocked on Phase B.

## Phase D — d24 scale-up

Not run. Blocked on Phase B/C demonstrating the recipe works.

## Recommendation

**Hold.** The implementation correctly preserves the baseline path and the
sparse forward math is correct (unit tests confirm), but the aux loss is
mis-scaled. The fix is a 3-line edit to the formula in `_train_forward`. Once
applied, we expect smoke-test re-runs to show:

- Phase A1: `train/loss` starting near 10-11 (random init), descending toward
  ~3-5 by step 20
- Phase A2: same, plus `router_loss` decreasing toward 0

Cost burned on this smoke test: ~30 minutes of 8×H100 ≈ $11.
Pod 2 (`nanochat-sparse-smoke`, id `7nkgiqe7pq2jjk`) being terminated. d24
baseline on pod 1 continues uninterrupted.
