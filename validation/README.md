# Paper validation sweeps

Seed-repeated training runs for the paper tables. Each leaf script runs one method over its
paper seeds; `run_*.sh` runs a whole table. All scripts are idempotent: a seed whose final
checkpoint (`model_0100000.pt`) already exists under `logs/WholeBodyTracking/` is skipped.

| Group | Script | Seeds | Paper |
| --- | --- | --- | --- |
| Main G1-WBT | `g1_wbt/run_main_table.sh` (BC, IQL, CQL a=5, ACL-QL, AW-CQL H50) | 1-5 | Main table |
| Structural ablation | `g1_wbt/ablation/run_ablation_table.sh` (wBC, B-arm, C-arm, Asym-CQL) | 1-3 | Ablation table / figure |
| H robustness | `g1_wbt/H_sweep/run_h_sweep.sh` (H25, H100; H50 = main AW-CQL seeds 1-3) | 1-3 | Appendix |
| LAFAN cross-motion | `g1_wbt_lafan/run_lafan_table.sh` (CQL, IQL, ACL-QL, AW-CQL) | 1-3 | Cross-motion table |
| LAFAN single-kick | `g1_wbt_kick/run_kick_table.sh` (CQL, IQL, ACL-QL, AW-CQL; motion fightAndSports1_subject4 f1728-1858, one hazard) | 1-5 | Cross-motion table |
| Optional | `optional/g1_wbt_td3_bc.sh`, `optional/g1_wbt_cql_alpha_sweep.sh` | 1-3 | - |

```bash
bash validation/run_all.sh                          # every table below, in order (skips finished seeds)
PART=1 bash validation/run_all.sh                   # terminal 1: main table + H sweep (31 runs)
PART=2 bash validation/run_all.sh                   # terminal 2: ablation + LAFAN tables (33 runs), no overlap with PART=1
bash validation/g1_wbt/g1_wbt_cql.sh                 # seeds 1-5 of the CQL reference
SEEDS="4 5" bash validation/g1_wbt/g1_wbt_cql.sh     # only seeds 4 and 5
DRY_RUN=1 bash validation/g1_wbt/run_main_table.sh   # print every command
JOBS=2 bash validation/g1_wbt_lafan/run_lafan_table.sh   # two runs at a time on one GPU
LOGGER=disabled bash validation/optional/g1_wbt_td3_bc.sh
```

Run names are `<experiment>_seed<N>` (e.g. `g1_29dof_wbt_cql_seed3`, `g1_29dof_wbt_aw_cql_H50_seed1`); they are used for the log directory and the wandb run.
Per-run console output goes to `validation/logs/<run>.<timestamp>.log`.

Q-level diagnostics: every CQL-family run logs `Loss/probe/*` (`probe/q_level` etc.; fixed 4096-transition probe set,
`q_probe_seed=12345`) and `Loss/lse_mean`, `Loss/q_policy_mean`; for runs trained before these metrics existed,
`scripts/probe_q_level.py` recomputes them from the saved checkpoints.

Environment variables: `SEEDS`, `JOBS`, `SIMULATOR`, `LOGGER`, `EXTRA_ARGS`, `DRY_RUN`, `FORCE`,
`FINAL_STEP`, `ENTRY` (see `common.sh`). Sidecars (`.aw_weights.H*.npz`, `.acl_quality.npz`)
are computed on demand when a script needs one that does not exist yet.
