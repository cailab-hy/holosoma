# SYNC-QL Notes

SYNC-QL is implemented in `src/holosoma/holosoma/agents/bf_cql/sync_cql_agent.py`.

## Reused From BF-CQL

- `SyncCQLAgent` is a standalone `BaseAlgo` subclass, not a `BFCQLAgent` subclass.
- Reuses BF-CQL-compatible setup utilities, environment wrapper, normalized action convention, factorized actor, global twin critic, target critic, replay/dataset loading, observation normalization, optimizers, checkpoint/export helpers, and vectorized evaluation.
- Reuses the BF-CQL group partition from `resolve_action_groups`; SYNC-QL never defines a new action grouping.
- Keeps the existing Bellman target, terminal/bootstrap handling, target-network update, alpha autotune, and singleton CFCQL term structure.

## Added

- `SyncCQLAgent` adds a second conservative term on the synergy residual:
  - `v(M) = Q(s, a_cf(M)) - Q(s, a_D)`
  - `Delta(M) = v(M) - sum_{g in M} v({g})`
- The hinge is applied to `Delta(M*)`, not `v(M*)`, to avoid double-counting the singleton CFCQL penalty.
- `M*` is selected per sample by:
  - drift screening using normalized RMSE,
  - `topk`, `greedy`, `random`, or `none` selection.
- Optional actor counterfactual objective uses the same selected `M*`, controlled by `sync_cql.lambda_cf`.
- Optional `alpha2_lagrange` tunes the synergy coefficient toward `sync_cql.tau_syn`.

## Config Reference

The config is nested under `algo.config.sync_cql`:

- `K`: maximum selected block size.
- `delta_threshold`: drift threshold for candidate screening.
- `selection_mode`: `topk | greedy | random | none`.
- `drift_mode`: `rmse | density`; `density` is a CVAE placeholder.
- `eps_gain`: greedy early-stop marginal gain.
- `margin_m`: hinge margin for `Delta(M*)`.
- `alpha2`: fixed synergy weight or initial Lagrange value.
- `alpha2_lagrange`: enable dual tuning for `alpha2`.
- `tau_syn`: target hinge penalty for dual tuning.
- `lambda_cf`: actor counterfactual objective weight.
- `drift_ema`: EMA smoothing for group drift.
- `drift_std_momentum`: running dataset-action std momentum.
- `freeze_drift_stats`: freeze running std updates.

## Extra Q-Evaluation Cost

Let `K = sync_cql.K`.

- `selection_mode=none`: `0` extra SYNC-QL Q forwards.
- `selection_mode=topk`: `K + 1` extra Q evaluations per sample for the penalty:
  - one block `a_cf(M*)`,
  - `K` singleton counterfactuals.
  - `Q(s,a_D)` is reused from the Bellman/CFCQL update.
- `selection_mode=random`: same as top-k, `K + 1`.
- `selection_mode=greedy`: up to `2K * K` detached Q evaluations for selection, plus `K + 1` penalty evaluations.

All counterfactual Q evaluations are batched by stacking along the batch dimension.
