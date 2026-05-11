# Sparse FFN for nanochat (opt-in)

This directory contains an experimental opt-in training and inference path that
adds top-K sparsity to nanochat's FFN. It is inspired by the LARQL "FFN as graph
database" observation: the intermediate dimension of an MLP is a
content-addressable bank of features, and only a small fraction fires
meaningfully on any given token.

## What's in here

| File | Purpose |
|---|---|
| `sparse_mlp.py` | `TopKSparseMLP`, `Router`, `FfnCache`, helper functions |
| `tests/test_sparse_mlp.py` | Unit tests: forward correctness, state-dict guards, cache |
| `dev/bench_sparse_mlp.py` | Microbench: dense vs Tier 1 vs Tier 2 vs Tier 3 |
| `docs/test-request.md` | Test plan + handoff for the Train/Test team |
| `docs/results.md` | Populated as Train/Test reports come back |

## Integration story

The default code path (vanilla nanochat MLP) runs **bit-identically** when
`--sparse-ffn` is absent. The integration is:

- 7 new fields on `GPTConfig` (all default to disabled).
- 1 conditional in `Block.__init__` that swaps `MLP` for `TopKSparseMLP`.
- 1 `hasattr` line in `GPT.init_weights` to init router weights if present.
- 4 lines in `GPT.forward` that add aux losses to the training loss (only
  when `sparse_ffn=True`).
- ~10 lines in `checkpoint_manager.py` for forward-compat defaults and a
  state-dict consistency assertion.
- 7 new CLI flags on `scripts/base_train.py`.

Nothing in `engine.py`, `chat_cli.py`, `chat_web.py`, or any eval script
changes. Sparse inference threads through automatically because the loaded
checkpoint's `GPTConfig` already specifies `sparse_ffn=True`.

## How to opt in

### Training

```bash
torchrun --nproc_per_node=8 -m scripts.base_train -- \
    --depth=24 \
    --sparse-ffn \
    --sparse-k=256 \
    --sparse-use-router \
    --sparse-router-rank=64
```

All sparse flags default to disabled; omit `--sparse-ffn` for the unchanged
baseline.

### Inference

When a checkpoint trained with `--sparse-ffn` is loaded, the inference path
automatically routes through the sparse MLP (the config rides in the
checkpoint). No additional flags are needed for Tier 1+2. For Tier 3 (optional,
off by default):

```python
from nanochat.sparse import enable_sparse_cache, sparse_cache_stats
enable_sparse_cache(model, max_entries=4096)
# ... run engine.generate() as usual ...
hits, misses = sparse_cache_stats(model)
```

## Three tiers

### Tier 1 — TopK sparsification

After `c_fc(x)` and `ReLU^2`, zero everything outside the K largest activations.
The model trains with this hard sparsity so its features cluster into a
smaller working set. At inference we gather only K columns of `c_proj` (instead
of the full matmul), reducing FFN bandwidth on `c_proj` by `intermediate / K`.

Mathematically identical at training and inference (the gather kernel produces
the same output as the dense matmul on the scattered `h_sparse`). The training
backward only flows gradients through the K active features.

### Tier 2 — Router (predicted active sets)

A low-rank predictor `R = V @ U` (rank ~64) learns to predict which K features
will fire ahead of the c_fc matmul. At inference, the router selects
`K' = K * oversample` candidates, and only those rows of `c_fc.weight` are
computed.

This eliminates the dense c_fc, completing the "graph traversal" view of the
forward pass: predict candidates, verify with exact dots, sum the contributions.

**Approximation.** Recall of the true top-K is empirically ≥99% with
oversample=2 once the router is trained, but this is a hypothesis the
Train/Test team will measure on real checkpoints. To force exact Tier 1 even on
a Tier 2-trained model (for ablations), construct the model with
`sparse_use_router=False` at load time, or remove the router from the
state_dict (the consistency assertion will fire — adjust accordingly).

### Tier 3 — FfnCache

Hashes the sorted top-K index set per layer. If the same active set appears
again, return the cached output without computing `_sparse_proj`. Works best on
chat workloads with repeated system-prompt prefixes.

Off by default. Measure and decide whether to enable. See LARQL's `ffn-cache.md`
for expected hit rates on different workloads.

## Config fields

All added to `GPTConfig` (`nanochat/gpt.py`):

```python
sparse_ffn: bool = False               # master switch
sparse_k: int = 256                    # active set size (intermediate/K = sparsity)
sparse_use_router: bool = False        # enable Tier 2
sparse_router_rank: int = 64           # router low-rank dim
sparse_router_oversample: int = 2      # K' = K * this at inference
sparse_aux_loss_coef: float = 0.01     # load-balancing aux loss weight
sparse_router_loss_coef: float = 0.1   # router cross-entropy weight
```

Defaults preserve vanilla behavior — `sparse_ffn=False` means `TopKSparseMLP`
is never instantiated.

## State-dict consistency

When a checkpoint is loaded by `checkpoint_manager.build_model`, we assert that
the state_dict's router keys match the config. This catches the case where
someone hand-edits the config but the checkpoint was trained without it (or
vice versa). The assertion lives in
`sparse_mlp.assert_state_dict_consistency`.

## Run the tests

```bash
pytest nanochat/sparse/tests/test_sparse_mlp.py -v
```

All tests use small synthetic shapes and run in seconds on CPU. They cover
correctness equivalence between training and inference paths, router
approximation behavior, cache semantics, and state-dict guards.

## Run the bench

```bash
python -m nanochat.sparse.dev.bench_sparse_mlp \
    --n-embd 768 --k 256 --batch 1 --seq 1 --iters 200
```

This times an isolated MLP forward in dense, Tier 1, Tier 2, and Tier 3
(warm + cold) modes. End-to-end decode tok/sec via the full Engine requires a
trained checkpoint — see `docs/test-request.md`.
