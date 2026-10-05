# OPER-A (ODPR-A) vs AW-CQL weights

dataset `offline_data/g1_29dof_wbt_fastsac_episode1m_env256_dataset.h5` rhash 94fc3c2c85c17e69, N=2,525,605

- rhash AW phase: OK
- rhash AW global: OK
- rhash OPER seed file 0: OK

## Weight distributions (mean-one weights)

| weighting | ESS/N | std | max | p1 | p10 | p50 | p90 | p99 | share w<0.5 | share w>2 |
|---|---|---|---|---|---|---|---|---|---|---|
| AW phase (ours) | 0.716 | 0.629 | 6.12 | 0.022 | 0.195 | 0.967 | 1.813 | 2.627 | 25.9% | 6.1% |
| OPER-A raw (iteration weights, mean one) | 0.996 | 0.066 | 2.75 | 0.828 | 0.947 | 0.998 | 1.057 | 1.197 | 0.1% | 0.0% |
| OPER-A as used (linear std=2.0, eps=0.1) | 0.425 | 1.163 | 42.42 | 0.079 | 0.079 | 0.739 | 2.150 | 5.465 | 37.7% | 11.7% |
| OPER-A ESS-matched to AW (ODPR std param 0.809) | 0.716 | 0.629 | 21.32 | 0.096 | 0.335 | 0.930 | 1.621 | 3.242 | 15.4% | 4.9% |
| AW global | 0.597 | 0.822 | 4.77 | 0.071 | 0.156 | 0.785 | 2.250 | 3.282 | 38.1% | 13.6% |

OPER-A per iteration (seed file 0): iter 1: ESS 1.000, std 0.019, adv mean 0.0363, |adv| 0.0897; iter 2: ESS 0.999, std 0.034, adv mean 0.0327, |adv| 0.0850; iter 3: ESS 0.998, std 0.046, adv mean 0.0305, |adv| 0.0834; iter 4: ESS 0.997, std 0.057, adv mean 0.0289, |adv| 0.0827; iter 5: ESS 0.996, std 0.066, adv mean 0.0277, |adv| 0.0823

OPER-A seed files averaged: 1 (ODPR averages 2-3 seeds before normalisation).

## Agreement between weightings

| pair | Pearson(w) | Spearman(w) | Jaccard top-10% | Jaccard bottom-10% | Spearman(within-bin rank) |
|---|---|---|---|---|---|
| OPER-A raw vs AW phase | 0.079 | 0.075 | 0.062 | 0.085 | 0.060 |
| OPER-A as used vs AW phase | 0.037 | 0.065 | 0.062 | 0.034 | 0.047 |
| OPER-A vs AW global | 0.076 | 0.069 | 0.076 | 0.074 | 0.060 |
| AW phase vs AW global | 0.652 | 0.754 | 0.166 | 0.343 | 1.000 |

Advantages: Pearson(OPER adv, AW A^H) = 0.061, Spearman = 0.063; Pearson(OPER adv, raw G^H) = 0.059; Pearson(OPER V(s), G^H) = 0.908; Pearson(OPER TD target, G^H) = 0.912

## Per progress bin

| bin | region | data mass % | AW phase w-mass % | AW global w-mass % | OPER raw w-mass % | OPER as-used w-mass % | mean OPER w (as used) | mean OPER adv | mean AW A^H | mean G^H |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | high | 12.10 | 14.52 | 25.29 | 12.22 | 16.73 | 1.382 | +0.0370 | +0.000 | 2.826 |
| 1 | high | 7.46 | 7.23 | 10.95 | 7.46 | 7.46 | 0.999 | +0.0250 | -0.000 | 2.594 |
| 2 |  | 8.02 | 7.44 | 7.99 | 7.98 | 6.82 | 0.851 | +0.0190 | +0.000 | 2.187 |
| 3 |  | 8.34 | 7.77 | 4.34 | 8.29 | 6.54 | 0.784 | +0.0195 | +0.000 | 1.448 |
| 4 | low | 6.70 | 5.87 | 1.70 | 6.66 | 5.28 | 0.788 | +0.0207 | -0.000 | 0.670 |
| 5 | low | 2.82 | 3.28 | 1.15 | 2.82 | 2.65 | 0.940 | +0.0233 | -0.000 | 1.011 |
| 6 |  | 2.53 | 2.60 | 2.56 | 2.54 | 2.80 | 1.107 | +0.0306 | +0.000 | 2.128 |
| 7 |  | 3.34 | 3.40 | 3.68 | 3.36 | 3.77 | 1.128 | +0.0352 | +0.000 | 2.231 |
| 8 |  | 3.99 | 4.19 | 3.73 | 4.01 | 4.44 | 1.112 | +0.0337 | +0.000 | 2.022 |
| 9 |  | 4.31 | 4.67 | 4.01 | 4.32 | 4.73 | 1.096 | +0.0338 | +0.000 | 1.995 |
| 10 |  | 4.59 | 4.68 | 3.88 | 4.60 | 4.68 | 1.020 | +0.0312 | -0.000 | 1.932 |
| 11 |  | 4.59 | 4.39 | 2.66 | 4.58 | 4.18 | 0.911 | +0.0273 | +0.000 | 1.549 |
| 12 | low | 4.61 | 4.02 | 1.51 | 4.60 | 3.94 | 0.855 | +0.0279 | -0.000 | 0.962 |
| 13 | low | 3.26 | 2.93 | 0.74 | 3.25 | 2.92 | 0.895 | +0.0246 | +0.000 | 0.532 |
| 14 | low | 2.00 | 2.13 | 0.96 | 2.00 | 2.37 | 1.186 | +0.0320 | -0.000 | 1.254 |
| 15 |  | 2.63 | 2.84 | 2.70 | 2.65 | 3.41 | 1.299 | +0.0418 | -0.000 | 2.107 |
| 16 | high | 3.35 | 3.37 | 5.40 | 3.36 | 3.83 | 1.143 | +0.0312 | -0.000 | 2.672 |
| 17 | high | 4.28 | 4.24 | 8.24 | 4.28 | 4.22 | 0.984 | +0.0252 | -0.000 | 2.885 |
| 18 | high | 5.22 | 5.19 | 6.62 | 5.20 | 4.43 | 0.849 | +0.0223 | -0.000 | 2.416 |
| 19 | low | 5.83 | 5.25 | 1.91 | 5.82 | 4.80 | 0.822 | +0.0266 | +0.000 | 0.945 |

## Region weight mass (share of total weight; first column = data share)

| region | data | AW phase | AW global | OPER-A raw | OPER-A as used |
|---|---|---|---|---|---|
| wall bins [4, 5, 13] | 12.8% | 12.1% | 3.6% | 12.7% | 10.9% |
| low-return bins [4, 5, 12, 13, 14, 19] | 25.2% | 23.5% | 8.0% | 25.1% | 22.0% |
| high-return bins [0, 1, 16, 17, 18] | 32.4% | 34.5% | 56.5% | 32.5% | 36.7% |

Share of weight variance explained by the progress bin alone (R^2 of per-bin means): AW phase 0.025, AW global 0.534, OPER-A raw 0.006, OPER-A as used 0.027
Phase dependence of the OPER advantage: R^2 of per-bin means = 0.002 (AW A^H by construction 0: -0.000)

Wrote `validation/results/oper_vs_aw/oper_a_pseudo_sidecar.npz` (advantage = OPER-A TD(0) advantage, gH = OPER TD target, H=50 for the validation window) for scripts/aw_post_h_validity.py --npz. Only its within-bin Panel A and the bin-FE logistic are meaningful for OPER-A: gH here is a 1-step TD target, not a 50-step return, so Panel B's raw-G^H curve does not apply.
