# Phase A Fix Notes — for the Train/Test team

**Branch:** `claude/review-repositories-Udsgs` (HEAD at commit `cc68ae2`)
**Context:** Response to the d12 K=512 smoke test diagnostic in `results.md`
(now archived in git history at commit `ece92e3`).

This doc tells you what changed in the fix bundle, what new CLI flag exists,
and what to watch on the re-run. Read time: ~3 minutes.

## What changed

Four coupled fixes, two commits:

**Commit `a46e2bf` — `sparse_mlp.py` + tests:**

1. **Router architecture: `V(U(x))` → `V(GELU(U(x)))`.** Adds a real nonlinearity
   between U and V so the router is a true 2-layer MLP, not a collapsed
   rank-r linear projection. ~1 line of code.

2. **Router target: uniform 1/K → magnitude-weighted.** The cross-entropy target
   is now the normalized `h_sparse` magnitudes (so mass is concentrated on the
   highest-activating features), not uniform 1/K over the top-K positions.
   This dramatically strengthens the gradient signal at K=512.

3. **Aux loss: `I * sum(load_norm * imp_norm)` → `I * sum(imp_norm**2)`.**
   The participation-ratio form (a.k.a. inverse Simpson). Provably bounded
   in [1, I] with minimum at uniform usage; the old form could be minimized
   by anti-correlating load and importance, which pushed the model *away*
   from balance.

**Commit `cc68ae2` — `gpt.py` + `base_train.py`:**

4. **Router gets its own Muon param group with multiplied LR.** Default 5x
   matrix_lr. The router target is non-stationary (top-K of c_fc, which
   keeps moving), so the router needs sustained LR to track. The same
   warmup + cosine-decay shape applies via the existing lrm multiplier.

## New CLI argument

One new flag in `scripts/base_train.py`:

```
--sparse-router-lr-multiplier FLOAT    (default 5.0)
```

Only takes effect when `--sparse-use-router` is also passed. When the multiplier
is 1.0, router behaves identically to other matrix params; values > 1 give
the router a proportionally higher initial LR. Schedule is unchanged.

There are **no other new arguments.** All existing sparse-related flags
(`--sparse-ffn`, `--sparse-k`, `--sparse-use-router`, `--sparse-router-rank`,
`--sparse-router-oversample`, `--sparse-aux-loss-coef`, `--sparse-router-loss-coef`)
are unchanged in name and default.

## What to use for the re-run

Same flags as the previous K=512 run, plus the default router LR multiplier:

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=12 \
    --sparse-ffn \
    --sparse-k=512 \
    --sparse-use-router
```

The `--sparse-router-lr-multiplier` defaults to 5.0; no need to pass it
explicitly unless you want to sweep.

## What to watch in wandb

**Primary signal: `train/router_loss_raw` trajectory.**

- Previous run: stalled at ~85 total (≈7.08/layer), ≈53% of the way to
  the floor of log(K)=6.24/layer.
- This run: floor is now the entropy of the magnitude distribution
  (typically well below log(K)), so the absolute number isn't directly
  comparable. **Look at the curve shape and the trajectory past the prior
  stall point.**
- Healthy: monotonic descent throughout training, no plateau under LR decay.
  Should drop well past 85 total in the first few hundred steps.
- Unhealthy: another plateau, especially during the cosine-decay phase.
  If you see this, the next thing to try is bumping the LR multiplier
  (e.g. `--sparse-router-lr-multiplier=10.0`).

**Secondary signal: `train/aux_loss_raw` bounds.**

- Previous run: observed [0.38, 1.28] per layer (× 12 layers = [4.6, 15.3] total).
  Below the claimed floor of 1.0 — the bug.
- This run: provably bounded in [1.0, 3072.0] per layer (where 3072 =
  intermediate = 4*n_embd for d12). Total across 12 layers should be in
  [12, ~36000]. Expect realistic values around [12, ~50] under healthy
  training (close to the lower bound, indicating near-uniform feature usage).
- If aux drops below 12 total, that's a regression — ping us with the wandb
  link. There's a unit test (`test_aux_loss_bounded_in_one_to_intermediate`)
  guarding this; it would have caught a code-level regression, so this is
  unlikely.

**Tertiary signal: `val/bpb` is still a *lagging* indicator.**

Validation goes through `_router_inference` (the approximate path), so until
the router learns enough to surface what the LM has learned, val/bpb will
look rough. Don't panic if val/bpb is still poor at step 500 — wait until
router_loss has dropped meaningfully (say below 60 total) before reading
val/bpb as a signal of model quality.

If you want to disentangle model quality from router quality during this
run, the exact (Tier 1) inference path is what we ultimately care about,
and it bypasses the router entirely. We don't currently surface that as a
validation hook — if it'd help, ping us and we'll add a flag.

## What we kept the same

- `--sparse-aux-loss-coef` default is still 0.01. The aux loss is on a
  different scale now (bounded in [1, I] instead of free-running), so the
  effective contribution is smaller. We're keeping the same default for
  now to see what natural behavior emerges; if aux becomes too dominant
  or too negligible, we'll iterate.

- `--sparse-router-loss-coef` default is still 0.1. Magnitude-weighted
  target gives lower router_loss values than uniform, so 0.1 might be a
  bit high in absolute terms, but it's a reasonable starting point. We'd
  rather see the data first.

- All inference paths are unchanged (Tier 1 exact gather still works,
  Tier 2 router-predicted still works, Tier 3 cache still works). The
  router architecture change does mean the inference router also has GELU
  now — same change applies symmetrically.

## Compute economy notes

The expected re-run cost is roughly the same as the previous d12 sparse smoke:
~30 minutes of 8×H100 ≈ $11. Both fix commits are pure architecture / loss /
optimizer changes; no FLOPs added beyond the router GELU (negligible).

The MFU gap (22% sparse vs 33% dense) you observed last time is still
structural — it comes from the gather kernels in the inference path and
the scatter in training. We're not fixing that in this round; it remains
on the to-do list as a fused Triton kernel project.

## What to report back

Overwrite `nanochat/sparse/docs/results.md` (we reset it to the template
in commit `81cb178` for exactly this purpose). The phase structure
(Phase 0 / A / B / C / D) is in the test-request doc.

If you hit anything weird, reply with:
- The failing config (CLI line)
- The relevant wandb panel screenshots or links
- The traceback if there's a crash

We'll iterate.

## Open questions we'd appreciate your read on

1. **Router LR multiplier value.** Is 5x enough to keep the router learning
   through cosine decay? Wider sweep welcome if you have the budget:
   `{1.0, 3.0, 5.0, 10.0, 20.0}` would map out the curve.

2. **K-sweep ordering.** With the bug fixed, we'd suggest the K-sweep order
   stay the same (512 → 256 → 128 → 64). 512 is the easiest target so we
   want to confirm router/aux are healthy there before tightening K.

3. **Aux coefficient.** If aux ends up dominating LM loss in the early
   training phase (say >20% of total loss), the right move is probably to
   anneal it (start at 0 in warmup, ramp up). Easy change in `gpt.py::forward`
   if needed — just flag it.

All of these are nice-to-knows, not blockers. Default flags should be fine
for the first re-run.
