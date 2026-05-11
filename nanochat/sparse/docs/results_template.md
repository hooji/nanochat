# Sparse FFN Results

_Canonical empty template. Copy this file (e.g. to `results.md` for the
primary d12 run, or to a new `results_<tag>.md` for a follow-on d24 run)
and fill it in as each phase completes._

_To be populated by the Train/Test team as runs complete._

## Phase 0 — Baseline Path Sanity

(pending)

## Phase A — d12 Sparse Smoke Test

(pending)

## Phase B — d12 K-Sweep

| K | val_bpb | CORE | train tok/sec | wandb |
|---|---|---|---|---|
| (vanilla baseline) | — | — | — | — |
| 512 | — | — | — | — |
| 256 | — | — | — | — |
| 128 | — | — | — | — |
| 64  | — | — | — | — |

## Phase C — Inference Benchmarks

| Mode | ms / call (microbench) | decode tok/sec (engine) | speedup |
|---|---|---|---|
| Dense baseline | — | — | 1.00× |
| Tier 1 (TopK exact) | — | — | — |
| Tier 2 (TopK + Router) | — | — | — |
| Tier 2 + Tier 3 (cache warm) | — | — | — |

## Phase D — d24 Scale-up

(pending)

## Recommendation

(populate once results are in)
