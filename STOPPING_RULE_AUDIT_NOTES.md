# Stopping-Rule Audit Registration

## §9 Registered stopping rule

This document fixes the retrospective rule before the final audit table is
assembled. `stopping_rule_audit.py` only reads saved checkpoints and fixed
`probe_rows.npz` rows; it never regenerates labels or changes training.

The primary-wall contrast channel is

`Delta_hat = (mean Q_SURV - mean Q_FAIL) / (p99(Q_probe) - p01(Q_probe))`.

Contrast firing is enabled only after `Delta_hat >= 2 * tau` has occurred at
least once. After activation, the channel fires at the first checkpoint where
the previous checkpoint satisfies `Delta_hat >= tau` and the current checkpoint
satisfies `Delta_hat < tau`. This guard formalizes that firing means the loss of
an already formed contrast. It prevents early warmup noise below `tau` from
being misclassified as a collapse; it is not a post-hoc threshold adjustment.

The level baseline is the median `q_target_min` in the inclusive 5k--20k
window. The level channel fires at the first checkpoint below
`baseline - abs(level_offset)`. A supplied log CSV is authoritative. Without a
log, the audit uses the minimum target-twin Q on the fixed global probe rows and
records that fallback explicitly as `probe:target_twin_min_q_data`.

The overall firing time is the earlier channel time. The adopted checkpoint is
the largest saved step strictly below that firing time. If neither channel
fires, the final checkpoint is adopted.

## Batch YAML

```yaml
defaults:
  tau: 0.05
  level_offset: -30
  level_baseline_window: 5k-20k
  device: cuda:0
  batch_size: 4096

runs:
  - label: AW-s1 cell1
    ckpt_dir: logs/WholeBodyTracking/aw_run
    dataset: offline_data/cell1.h5
    probe_rows: probe_rows_cell1.npz
    wall_bins: [4, 5]
    agent_type: aw_cql
    level_from_log: logs/WholeBodyTracking/aw_run/q_target_min.csv
    out: audit_aw_s1_cell1.csv

  - label: IQL-s1 cell1
    ckpt_dir: logs/WholeBodyTracking/iql_run
    dataset: offline_data/cell1.h5
    probe_rows: probe_rows_cell1.npz
    wall_bins: [4, 5]
    agent_type: iql
    out: audit_iql_s1_cell1.csv
```

Run all entries with:

```bash
python stopping_rule_audit.py --batch-config stopping_runs.yaml
```
