# Test Request: TopK-Sparse FFN

**To:** Train/Test team
**From:** Sparse-FFN development team
**Branch:** `claude/review-repositories-Udsgs`

This document describes what we'd like measured on the new opt-in sparse FFN
code path. The default code path (no `--sparse-ffn` flag) is unchanged and
should still produce the existing baseline numbers bit-for-bit. If you observe
a regression on the baseline path with this branch checked out, that's a bug —
please report it.

## Quick summary

We're testing whether nanochat can be trained with **top-K sparse FFN
activations** (only K of `4 * n_embd` features fire per token) so that
inference can skip most weight reads. The mathematical claim is that ReLU^2
already produces heavy-tailed sparsity; we're making it explicit during
training so K can be small (e.g. K=256 of intermediate=5120 ≈ 20× sparsity)
and inference can exploit it via gather kernels.

Three optional features on top of the base sparse model:

1. **Tier 1 (TopK)** — required for the rest. Hard top-K + scatter in training,
   gather-c_proj at inference.
2. **Tier 2 (Router)** — a low-rank predictor of which features will fire.
   Enables skipping the dense c_fc at inference.
3. **Tier 3 (FfnCache)** — inference-only cache keyed on the sorted top-K
   indices. Off by default. We're including it for measurement; do **not**
   assume it's part of the main story until results are in.

## What we'd like you to measure

### Phase 0 — Sanity check (a few minutes)

With the branch checked out, confirm the **baseline path still works**:

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=12 \
    --run=baseline_smoke \
    --model-tag=d12_baseline \
    --num-iterations=20 \
    --core-metric-every=-1 \
    --sample-every=-1 \
    --save-every=-1
```

Expected: no behavioral difference from master. If anything changes, the
integration patches need review.

Then run our unit tests:

```bash
pytest nanochat/sparse/tests/test_sparse_mlp.py -v
```

All should pass.

### Phase A — d12 sparse smoke test (~5 min per run)

Goal: confirm the new code path trains without crashing.

| Config | Flags |
|---|---|
| Tier 1, K=256 | `--sparse-ffn --sparse-k=256` |
| Tier 1+2, K=256 | `--sparse-ffn --sparse-k=256 --sparse-use-router` |

Use the same `--depth=12 --num-iterations=20` quick setup as Phase 0.

Report for each:

- Final train loss
- Training throughput (tok/sec) — expect 5-25% slowdown vs baseline
- Peak memory — should not regress significantly
- For Tier 1+2: does `router_loss` decrease monotonically?

### Phase B — d12 sweep across K (after Phase A passes)

Vary K to find the quality/speedup tradeoff curve. All runs with
`--sparse-ffn --sparse-use-router`. Use full d12 training horizon (not 20
iterations).

| K  | Approx sparsity | Hypothesis |
|---|---|---|
| 512 | ~10×  | Minimal quality regression, modest inference win |
| **256** | **~20×**  | **Main proposal**; <5% CORE regression target |
| 128 | ~40×  | Aggressive; quality may suffer |
| 64  | ~80×  | LARQL's prediction; will likely degrade |

For each: report `val_bpb`, CORE score, training tok/sec.

### Phase C — Inference benchmarks

For the K=256 Phase B checkpoint (and ideally one other K for comparison), run:

1. **Module microbench**:
   ```bash
   python -m nanochat.sparse.dev.bench_sparse_mlp \
       --n-embd <model_dim> --k 256 --batch 1 --seq 1 --iters 200
   ```
   Reports per-call latency for dense vs Tier 1 vs Tier 2 vs Tier 3.

2. **End-to-end decode tok/sec** via the existing `Engine`:
   - Baseline (vanilla d12 checkpoint): decode tok/sec for 64 tokens at
     temperature=0 on "The capital of France is".
   - Sparse checkpoint, default config (Tier 2 inference): same prompt and
     length.
   - Sparse checkpoint with cache on: enable via
     `from nanochat.sparse import enable_sparse_cache; enable_sparse_cache(model)`
     before generation.

We expect (rough targets):

- Tier 1: ~2-3× decode speedup vs dense baseline
- Tier 2: ~4-6× decode speedup vs dense baseline
- Tier 3 warm: another ~1.5-2× on top of Tier 2 for templated prefixes

If you see substantially different numbers, that's important signal — please
report.

### Phase D — d24 scale-up (only after Phase B/C confirm the recipe)

Once a Phase B run hits the quality bar (e.g. CORE within 5% of d24 baseline
at K=256), repeat the speedrun config at d24 with sparse:

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=24 \
    --target-param-data-ratio=8 \
    --device-batch-size=16 \
    --fp8 \
    --sparse-ffn \
    --sparse-k=256 \
    --sparse-use-router \
    --sparse-router-rank=64 \
    --run=$WANDB_RUN
```

Report:

- `val_bpb`, CORE
- Training time vs baseline d24
- Decode tok/sec vs baseline d24
- Side-by-side sample outputs on the prompts in `runs/speedrun.sh`

## Diagnostics worth capturing

Optional but valuable for the writeup:

- **Active-set entropy**: how often does the same top-K set recur across
  consecutive tokens? (Tests LARQL's paraphrase-collapse hypothesis.)
- **Router recall**: fraction of true top-K (from full c_fc) that appears in
  the router's top-K candidates. Wire a buffer into
  `TopKSparseMLP._router_inference` and log to wandb if you'd like.
- **Layer-wise sparsity**: do the same K values work at every layer? LARQL
  found distinct phases (computation L7-L18, knowledge L19-L29, format gate
  L30-L33 on Gemma 3 4B). nanochat may show something similar at d24.

## What changes about training (heads-up)

- **Step time**: expect 5-25% slower per step from topk + aux loss overhead.
  Tier 2 adds another 5-10% from the router forward and its CE loss.
- **Memory**: should not regress materially — the intermediate tensor has the
  same shape as the dense `h`.
- **Convergence**: this is the unknown. Sparse-trained models *might* need
  similar steps (Switch Transformer experience) or up to 1.5× more steps to
  reach equivalent loss. Worth budgeting.

## How to report back

Overwrite `nanochat/sparse/docs/results.md` with your findings. Sections:

1. Phase 0 — Baseline path sanity (pass/fail + any anomalies)
2. Phase A — Sparse smoke test outcome
3. Phase B — d12 K-sweep table + plots (or wandb links)
4. Phase C — Inference benchmark table
5. Phase D — d24 result + recommendation (ship as-is / iterate / abandon)

If you hit anything weird (training divergence, NaN, memory blow-up,
torch.compile errors, state-dict consistency assertion firing), reply with
the failing config + traceback and we'll iterate on the implementation.

## Open questions we'd like your read on

- Should the aux loss coefficient `--sparse-aux-loss-coef` anneal during
  training (start higher, decrease)? Default is 0.01 constant.
- For very small K (≤64), should we also anneal K itself (start with K=512
  and reduce)? Currently K is a fixed hyperparameter.
- Tier 3 cache: currently bounded by `max_entries` with no LRU (matches
  LARQL). Should we add LRU eviction? Probably not until we have a real
  workload-driven measurement.

If any of these become blocking during your runs, flag them and we'll iterate.

## File map

- `nanochat/sparse/sparse_mlp.py` — implementation
- `nanochat/sparse/tests/test_sparse_mlp.py` — unit tests
- `nanochat/sparse/dev/bench_sparse_mlp.py` — microbench
- `nanochat/sparse/docs/README.md` — high-level overview
- `nanochat/sparse/docs/results.md` — your scratch pad for outcomes

Outside the sparse directory, only three files are touched:

- `nanochat/gpt.py` — config fields + `if config.sparse_ffn:` in `Block.__init__`
- `nanochat/checkpoint_manager.py` — defaults patcher + consistency assertion
- `scripts/base_train.py` — CLI flag plumbing into `GPTConfig`
