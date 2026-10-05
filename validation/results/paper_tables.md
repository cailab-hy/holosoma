# Paper performance tables (all arms: 5 seeds, 100k iterations, eval every 1k on 4096 envs)

Success = `motion_ends` percentage. **Final-10** = mean over the last 10 evals (91k-100k), mean ± std over seeds
(primary number). **Best** = per-seed max over training, then mean ± std. **Last** = eval at 100k.
Secondary: Final-10 return, episode length (steps), bad-tracking termination rate (%).

## Table 1. Main comparison, G1 largebox (sub3_largebox_003)

| Method | Final-10 success (%) | Best (%) | Last (%) | Return | Length | Bad-tracking (%) |
|---|---|---|---|---|---|---|
| BC | 0.1 ± 0.1 | 12.9 ± 5.2 | 0.0 ± 0.0 | 7.1 ± 0.4 | 66 ± 3 | 99.9 ± 0.1 |
| IQL | 66.3 ± 5.8 | 83.5 ± 2.6 | 73.8 ± 10.7 | 43.6 ± 2.2 | 447 ± 23 | 32.6 ± 5.9 |
| CQL (α=5) | 0.3 ± 0.4 | 58.4 ± 5.5 | 0.0 ± 0.0 | 17.3 ± 0.5 | 161 ± 6 | 99.7 ± 0.4 |
| ACL-QL | 0.0 ± 0.0 | 0.9 ± 1.0 | 0.0 ± 0.1 | 14.5 ± 1.0 | 146 ± 15 | 100.0 ± 0.0 |
| **AW-CQL (H=50, ours)** | **70.2 ± 2.3** | **83.9 ± 1.6** | 67.6 ± 6.3 | **47.0 ± 0.7** | **474 ± 6** | **28.6 ± 2.3** |

## Table 2. Structural ablation, G1 largebox (same H=50 sidecar weights)

| Variant | Penalty form | Final-10 success (%) | Best (%) | Last (%) | Return | Bad-tracking (%) |
|---|---|---|---|---|---|---|
| wBC | weighted BC, no critic | 0.5 ± 0.4 | 37.3 ± 14.3 | 0.2 ± 0.4 | 12.5 ± 1.1 | 99.5 ± 0.4 |
| B-arm | LSE − w·Q_D | 61.2 ± 2.7 | 78.1 ± 5.1 | 58.0 ± 12.5 | 44.0 ± 1.1 | 37.7 ± 2.7 |
| C-arm | w·LSE − Q_D | 0.0 ± 0.0 | 0.6 ± 0.7 | 0.0 ± 0.0 | 15.2 ± 0.2 | 100.0 ± 0.0 |
| Asym-CQL | independent mean-one weights on both terms | 3.5 ± 1.6 | 19.6 ± 8.4 | 2.9 ± 0.4 | 18.5 ± 1.7 | 96.5 ± 1.6 |
| AW-CQL, global baseline | w(LSE − Q_D), A^H vs dataset-wide baseline (no phase) | 62.2 ± 3.6 | 77.4 ± 2.7 | 60.2 ± 4.9 | 43.9 ± 0.9 | 36.7 ± 3.6 |
| **AW-CQL (full)** | w(LSE − Q_D), per-phase-bin baseline | **70.2 ± 2.3** | **83.9 ± 1.6** | **67.6 ± 6.3** | **47.0 ± 0.7** | **28.6 ± 2.3** |

Paired same-seed differences vs AW-CQL (Final-10): B-arm −9.0, global baseline −8.0 ± 3.9 (5/5 seeds lower).

## Table 3. Advantage horizon H, G1 largebox

| H | Final-10 success (%) | Best (%) | Last (%) | Return | Length | Bad-tracking (%) |
|---|---|---|---|---|---|---|
| 25 | 30.4 ± 9.3 | 74.0 ± 2.3 | 23.8 ± 21.8 | 33.9 ± 2.9 | 336 ± 30 | 69.1 ± 9.5 |
| 50 (main) | 70.2 ± 2.3 | 83.9 ± 1.6 | 67.6 ± 6.3 | 47.0 ± 0.7 | 474 ± 6 | 28.6 ± 2.3 |
| 100 | 72.1 ± 3.3 | 86.3 ± 1.2 | 70.0 ± 15.7 | 47.1 ± 0.7 | 478 ± 7 | 26.7 ± 3.4 |

## Table 4. Cross-motion, G1 LAFAN dance1 (dance1_subject3, frames 172-412)

| Method | Final-10 success (%) | Best (%) | Last (%) | Return | Length | Bad-tracking (%) |
|---|---|---|---|---|---|---|
| IQL | 78.9 ± 3.0 | 94.7 ± 0.9 | 73.1 ± 15.5 | 50.9 ± 1.5 | 440 ± 12 | 20.8 ± 3.0 |
| CQL (α=5) | 93.2 ± 0.7 | 97.0 ± 0.7 | 94.0 ± 0.5 | 58.0 ± 0.4 | 500 ± 3 | 6.4 ± 0.7 |
| ACL-QL | 49.7 ± 9.9 | 81.1 ± 7.1 | 46.6 ± 9.5 | 42.4 ± 3.4 | 390 ± 27 | 50.0 ± 10.0 |
| AW-CQL, global baseline | 94.6 ± 1.9 | 98.0 ± 0.3 | 94.9 ± 3.4 | 59.7 ± 0.3 | 510 ± 2 | 5.0 ± 1.9 |
| **AW-CQL (H=50, ours)** | **97.1 ± 0.6** | **98.8 ± 0.3** | **97.1 ± 0.7** | **60.1 ± 0.2** | **512 ± 1** | **2.5 ± 0.6** |

Paired same-seed difference, global baseline vs AW-CQL (Final-10): −2.5 ± 1.9 (4/5 seeds lower; 95% CI [−4.9, −0.05]).

## Table 5. Progress-bin sensitivity of the baseline b(kappa), G1 largebox (AW-CQL H=50, 5 seeds)

| K (bins) | Final-10 success (%) | Best (%) | Last (%) | Paired vs K=20 (Final-10) |
|---|---|---|---|---|
| 1 (= dataset-wide baseline) | 62.2 ± 3.6 | 77.4 ± 2.7 | 60.2 ± 4.9 | −8.0 ± 3.9, 95% CI [−12.8, −3.2], 5/5 lower |
| 10 | 69.4 ± 3.2 | 84.0 ± 1.8 | 71.6 ± 5.5 | −0.8 ± 2.7, [−4.3, +2.6] |
| **20 (main)** | **70.2 ± 2.3** | 83.9 ± 1.6 | 67.6 ± 6.3 | — |
| 40 | 68.3 ± 3.9 | 83.4 ± 2.8 | 64.5 ± 15.3 | −1.9 ± 2.2, [−4.7, +0.9] |

K = 10, 20, 40 are statistically indistinguishable; only K = 1 (no progress conditioning) is worse. K=40 "Last" spread comes from seeds 1–2 dipping at the 100k checkpoint (56.1, 43.8); Final-10 is unaffected.

## Table 6. Global-baseline control on both motions (AW-CQL H=50, 5 seeds, Final-10 success %)

| Motion | Phase baseline (ours) | Dataset-wide baseline | Paired difference |
|---|---|---|---|
| G1 largebox (2 low-return regions) | 70.2 ± 2.3 | 62.2 ± 3.6 | −8.0 ± 3.9, [−12.8, −3.2], 5/5 lower |
| LAFAN dance1 (no low-return region) | 97.1 ± 0.6 | 94.6 ± 1.9 | −2.5 ± 1.9, [−4.9, −0.1], 4/5 lower |

## Table 7. Conservative weight alpha, G1 largebox (5 seeds)

| Method | alpha | Final-10 success (%) | Best (%) | Last (%) |
|---|---|---|---|---|
| CQL | 1 | 72.2 ± 4.6 | 84.9 ± 2.9 | 74.0 ± 5.1 |
| CQL | 5 (main table) | 0.3 ± 0.4 | 58.4 ± 5.5 | 0.0 ± 0.0 |
| AW-CQL (H=50) | 1 | 72.1 ± 4.0 | 87.8 ± 1.6 | 76.4 ± 7.3 |
| AW-CQL (H=50) | 5 (main table) | 70.2 ± 2.3 | 83.9 ± 1.6 | 67.6 ± 6.3 |

Same-seed paired AW-CQL − CQL at alpha=1: −0.1 ± 8.7 (no difference). CQL's collapse in the main table is alpha-specific (alpha=5); AW-CQL is stable across alpha = 1 and 5. alpha = 10 not run yet.

## Table 8. LAFAN front kick (fight1_subject3 f4999-5116, one low-return region), 5 seeds

| Method | Final-10 success (%) | Best (%) | Last (%) | Return | Length | Bad-tracking (%) |
|---|---|---|---|---|---|---|
| IQL | 68.1 ± 5.3 | 96.0 ± 1.1 | 59.3 ± 22.3 | 33.9 ± 2.0 | 308.0 ± 18.5 | 31.9 ± 5.3 |
| CQL (alpha=5) | 95.9 ± 0.9 | 98.1 ± 0.5 | 96.2 ± 1.1 | 45.0 ± 0.3 | 400.6 ± 2.6 | 4.1 ± 0.9 |
| AW-CQL global baseline | 97.7 ± 0.3 | 99.0 ± 0.1 | 97.6 ± 0.5 | 45.8 ± 0.1 | 406.2 ± 0.9 | 2.3 ± 0.3 |
| AW-CQL (H=50, ours) | 98.7 ± 0.4 | 99.6 ± 0.1 | 98.8 ± 0.1 | 46.2 ± 0.2 | 408.2 ± 0.7 | 1.3 ± 0.4 |

Paired same-seed differences (Final-10): AW-CQL − CQL +2.8 ± 0.7, 95% CI [+1.9, +3.6], 5/5 seeds higher; AW-CQL − global baseline +1.0 ± 0.3, 95% CI [+0.6, +1.4], 5/5 seeds higher; AW-CQL − IQL +30.6 ± 5.0, 95% CI [+24.4, +36.8], 5/5 seeds higher.

Source: `validation/results/summary.csv` (regenerated 2026-10-05 by `scripts/summarize_validation_results.py`).
