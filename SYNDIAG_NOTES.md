# SYNDIAG — Synergy-OOD Diagnostic Logging for BF-CQL (Part A)

Non-invasive instrumentation of `BFCQLAgent` that observes whether group-level
actor drift co-occurs with coalition Q-overestimation signals (coalition value
`v(M)` and synergy residual `Delta(M)`) during normal offline training.
**Logging only**: no loss, gradient, optimizer, update-order, or RNG changes.

Definitions (all in the normalized `[-1, 1]` action space the critic consumes):

```
d_g(s, a_D) = sqrt( mean_{j in g} ((a_pi_j - a_D_j) / (sigma_D_j + 1e-6))^2 )   # [B, G]
v(M)        = min(Q1,Q2)(s, a_cf(M)) - min(Q1,Q2)(s, a_D)                       # [B]
Delta(M)    = v(M) - sum_{g in M} v({g})                                        # [B]
a_cf(M)     = where(mask(M), a_pi, a_D)
```

`a_pi` is the squashed deterministic actor action `tanh(mean)` from
`FactorizedActor.forward(...)[0]` — BF-CQL enforces `use_tanh=True` with
`action_scale=1, action_bias=0`, so this is already in the same normalized
space as `dataset_actions = _to_normalized_actions(data["actions"])` and as
the critic input. It is the exact tensor the training loop already computes
for `q_pi_minus_q_data`; syndiag reuses it.

## Hook sites (exactly two)

1. **`_update_q` — existing `torch.no_grad()` diagnostic block**
   (`src/holosoma/holosoma/agents/bf_cql/bf_cql_agent.py`, the block computing
   `q_pi_minus_q_data`). It now additionally exports two *already-computed*,
   detached tensors through the return tuple:
   `syndiag_q_data_min = min(q1, q2).detach()` (per-sample `[B]`) and
   `syndiag_pi_actions = pi_actions_det.detach()` (`[B, A]`).
   `Q(s, a_D)` is therefore **reused, never re-evaluated**.
   The tick itself cannot live inside `_update_q` because that function is
   `torch.compile`'d — step-dependent Python branching, `.item()` calls and
   npz file I/O would graph-break or recompile every step. Exporting two
   detached graph outputs is compile-safe and adds no computation.

2. **`offline_learn` — logging section.** Right next to the existing
   `_compute_action_ood_stats(data)` call, `_syndiag_maybe_tick(...)` runs and
   its scalars are merged into `training_metrics.add({...})` (standard
   TensorBoard flow via `post_epoch_logging`).

All diagnostic code lives in clearly-marked `_syndiag_*` methods on the agent
plus the pure-math module `src/holosoma/holosoma/agents/bf_cql/syndiag.py`.

## Schedule, safety, zero-impact guarantees

- Tick condition: `self._critic_update_step % syndiag.interval == 0`
  (interval counts **critic updates**; with `num_updates=4` one `global_step`
  performs 4 critic updates). On non-tick steps the overhead is the enabled
  flag plus one modulo check.
- All diagnostic math runs under `torch.no_grad()`; the coalition critic
  forward runs under the same autocast (`_maybe_amp`) as training so `q_cf`
  and the reused `q_data` share precision.
- No new tensors on the autograd graph, no optimizer/schedule changes, no
  draws on the default RNG (all syndiag math is deterministic; the dump
  subsample takes the *first* `dump_max_rows` rows of an already-random batch).
- The observation normalizers are **never called** by syndiag (calling them
  would update their running stats); it reuses already-normalized tensors and
  keeps a zero-copy reference to the raw observations.
- `sigma_D` is a **dedicated** running per-dim std of normalized dataset
  actions (bootstrap-initialized from the first tick's batch std, then EMA
  with momentum 0.999 on each tick). It is not checkpointed and resets on
  resume; it never touches `obs_normalizer`/`critic_obs_normalizer`.
- Never crashes training: the tick body is wrapped in try/except — the first
  failure logs one exception, and 3 *consecutive* failures disable syndiag for
  the rest of the run.
- With `syndiag.enabled=false` the training path is bit-identical to the
  pre-patch behavior (no extra TensorDict keys, no tick, unchanged sampler
  usage). Verified by `tests/test_syndiag.py::test_noop_equivalence_losses_bit_identical`
  (atol=0, with an RNG-consuming `rsample` in the loop so any hidden RNG draw
  would be caught).

## Config (`BFCQLConfig.syndiag`, `SynDiagSettings` in `config_types/algo.py`)

```yaml
syndiag:
  enabled: true
  interval: 200        # run diagnostics every N critic updates
  dump_interval: 50    # dump raw npz every N diagnostic ticks (0 = never)
  dump_topk: 3         # top coalitions per sample included in the dump
  delta_min: 0.0       # activity threshold for recall metrics
  max_coalitions: 32   # safety cap; warn and truncate (singletons+pairs first)
  dump_max_rows: 2048  # deterministic row cap per dump file
```

## Coalition list `M_list`

Derived generically from the G action groups (`bf_cql_group_indices` /
`bf_cql_group_names`): all singletons (needed for `Delta`), all pairs `i<j`,
plus named physical blocks when the grouping contains
`left_leg / right_leg / waist / left_arm / right_arm` (i.e. `coarse_5`):
`{LL,RL}`, `{LL,RL,W}`, `{LA,RA}`, `{LL,RL,LA,RA}` (the two pairs dedupe into
the all-pairs list → 17 coalitions total). For other groupings (e.g. the
default `functional_9`) the named blocks are skipped with a warning; 9
singletons + 36 pairs exceed the cap, so the list truncates to 32 with a
warning (all singletons + the first 23 pairs — raise `max_coalitions` to 45 to
keep every pair).

Coalition names are deterministic and collision-free:
`{sing|pair|tri|quad}_<ABBR>[_<ABBR>...]`, where `ABBR` is the first letter of
each underscore token of the group name (`left_leg → LL`, `waist → W`,
`left_knee_ankle → LKA`; numeric suffix on collision).

All coalition evaluations are batched into **one** twin-critic forward of
shape `[B * C, ...]` (`syndiag.coalition_q_values`); no per-coalition loop
over the batch.

## Logged TensorBoard keys (prefix `syndiag/`)

- `syndiag/drift_{group_name}` — batch mean of `d_g` per group.
- `syndiag/v_{coal}`, `syndiag/delta_{coal}` — per-coalition batch means
  (e.g. `v_pair_LL_RA`, `delta_tri_LL_RL_W`). Singleton `delta_sing_*` is 0 by
  construction (identity check).
- Binned drift–Delta correlation: per-coalition block drift
  `mean_{g in M} d_g`, pooled across coalitions of the same size, split into
  drift quartiles → `syndiag/delta_pairs_driftQ{1..4}`,
  `syndiag/delta_triples_driftQ{1..4}` (triples only when named blocks exist),
  plus the paper-facing `syndiag/delta_pairs_q4_over_q1` (Q4 divided by
  `|Q1|` clamped at 1e-8, sign of Q4 preserved) and
  `syndiag/delta_pairs_q4_minus_q1`.
- `syndiag/recall_pair_top{2,3}` — fraction of active samples whose top-Delta
  pair has both groups inside the top-K groups by `d_g`; active means the top
  Delta exceeds `delta_min`. `syndiag/active_frac` — fraction of active samples.
- `syndiag/superadditivity_quad` — mean of `Delta(4-limb) − Σ Delta(pairs ⊂ quad)`
  (positive ⇒ higher-order synergy beyond pairs; only with the coarse grouping).

Keys appear only on tick steps; `TensorAverageMeterDict` averages whatever was
added within each logging window, so intermittent keys are safe.

## Raw dump (input contract for Part B)

Every `dump_interval` ticks (main process only):
`{log_dir}/syndiag/dump_step{global_step:08d}.npz` — full schema documented in
`tools/eval_counterfactual_gap.py` (the Part B stub). Highlights: global
`dataset_index` into the source HDF5 (plumbed through `GPUTransitionCache.sample`
and `RAMShuffleBuffer.refill` as a logging-only batch field; `-1` when symmetry
augmentation makes rows untraceable), raw (unnormalized) observations and
env-scale dataset actions as stored, `a_pi` in both spaces, `d_g`, per-coalition
masks / `v` / `Delta` / `q_cf`, reused `q_data_min`, and env-scale `a_cf` for the
top-`dump_topk` coalitions per sample. Row cap `dump_max_rows=2048` keeps files
in the single-digit-MB range (≪ 50MB; float32 throughout).

## Tests

`tests/test_syndiag.py` (CPU-only, no IsaacSim):

1. **No-op equivalence** — fixed seed, fixed synthetic batch, real
   `FactorizedActor`/`DoubleQCritic` update loop; critic and actor loss
   sequences are bit-identical (`==`, atol=0) with the tick running every
   update vs disabled.
2. **Additive toy critic** `Q = Σ_g w_gᵀ a^g` ⇒ `|Delta(M)| < 1e-5` for all
   coalitions.
3. **Interaction toy critic** `Q = a¹·a²` ⇒ `Delta({1,2})` non-zero and equal
   to the analytic `(a_pi¹−a_D¹)·(a_pi²−a_D²)`; singleton v's don't explain it.
4. **Shape/abbrev/dump sanity** — deterministic collision-free names, drift
   `[B, G]`, npz round-trip through `np.load` with internal `Delta`/`v`
   consistency, truncation order, and the 3-consecutive-failure disable path.

Run: `python -m pytest tests/test_syndiag.py -q` (12 passed; existing
`test_sync_cql.py` + `test_pbf_cql.py` still pass).

## Measured overhead

RTX 5000 Ada (laptop), torch 2.7.0+cu128, bf16 autocast, `B=1024`, `A=29`,
critic hidden 768 / actor hidden 512, `functional_9` grouping (C=32 after cap):

| what | wall-clock |
|---|---|
| one syndiag tick (drift + 32-coalition critic forward + aggregates) | **10.7 ms** |
| one approximate BF-CQL update step (bellman + 9-group CQL block + actor, fwd+bwd+step) | **48.0 ms** |
| tick / update ratio | 0.22× |
| amortized overhead at `interval=200` | **≈ 0.11 %** |

The update-step baseline mirrors the module-level structure of
`_update_q` + `_update_actor` on the same hardware (the real step additionally
does batch sampling and the Lagrange update, so the true relative overhead is
slightly lower). Dump ticks add CPU-side npz compression (~tens of ms every
`interval × dump_interval` = 10k critic updates by default) off the training
hot path.

## Known limitations

- Under bf16 autocast the `Delta` resolution is limited by bf16 rounding of Q
  (~0.4% relative); the coalition forward intentionally runs at the same
  precision as the reused `q_data` so that `a_cf = a_D` ⇒ `v = 0` exactly.
- `sigma_D` resets on checkpoint resume (diagnostic-only buffer, not saved).
- With `use_symmetry=True` the augmented rows carry `dataset_index = -1`.
- `functional_9` + `max_coalitions=32` drops 13 of 36 pairs (warned at setup).
- `SyncCQLAgent`/`PBFCQLAgent` are untouched: they override `_update_q`/
  `offline_learn`, and the borrowed `_sample_offline_batch` guards syndiag
  state with `getattr(..., False)`.
