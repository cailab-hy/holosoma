# Paired Q-level displacement vs g1_29dof_wbt_aw_cql_H50 (same seed, same checkpoint step)

traj = mean over checkpoints with step >= 10000; final-10 = mean over the last 10 checkpoints.

| Method | n | dQ_level traj | dQ_level final-10 | dLSE traj | dLSE final-10 | dQ_pi final-10 |
| --- | --- | --- | --- | --- | --- | --- |
| AW-CQL delta=-0.25 (lift) | 3 | 39.33 ± 0.98 | 37.71 ± 1.30 | 41.48 ± 1.07 | 39.98 ± 1.40 | 38.76 ± 1.34 |
| AW-CQL delta=-0.10 (lift) | 3 | 15.28 ± 0.22 | 15.86 ± 0.19 | 16.03 ± 0.23 | 16.65 ± 0.29 | 16.27 ± 0.22 |
| AW-CQL delta=+0.10 (sink) | 2 | -45.41 ± 16.56 | -39.89 ± 81.85 | -42.51 ± 16.83 | -32.13 ± 83.71 | -36.75 ± 84.06 |
| AW-CQL delta=+0.25 (sink) | 3 | -105.96 ± 5.98 | -159.11 ± 58.29 | -98.48 ± 9.34 | -149.95 ± 54.01 | -157.38 ± 57.37 |
| CQL (alpha=5) | 5 | -5.01 ± 0.14 | -5.12 ± 0.19 | -5.63 ± 0.13 | -5.82 ± 0.17 | -5.77 ± 0.22 |
| B-arm (LSE - wQ_D) | 5 | -5.47 ± 0.23 | -6.54 ± 0.32 | -4.93 ± 0.23 | -6.02 ± 0.30 | -6.20 ± 0.30 |
| Residual B-arm lambda=0.25 | 3 | -1.45 ± 0.06 | -1.65 ± 0.14 | -1.45 ± 0.08 | -1.70 ± 0.15 | -1.50 ± 0.19 |
