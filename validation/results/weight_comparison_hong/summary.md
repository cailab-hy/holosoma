# Weight signal comparison: PRe(AW-H50), Hong-AW

dataset `g1_29dof_wbt_fastsac_episode1m_env256_dataset.h5`, N=2,525,605, episodes=34,914 (bad tracking 81.9%, motion completed 18.2%)

## 1. Distribution (all weights have mean 1)

| weights | ESS/N | std | min | p1 | p10 | p50 | p90 | p99 | max | share w<0.5 | share w>2 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PRe(AW-H50) | 0.716 | 0.629 | 0.000 | 0.02 | 0.19 | 0.97 | 1.81 | 2.63 | 6.1 | 25.9% | 6.1% |
| Hong-AW | 0.454 | 1.097 | 0.085 | 0.24 | 0.40 | 0.63 | 1.85 | 6.04 | 12.6 | 25.2% | 8.8% |

## 2. Agreement between weightings

| pair | Pearson | Spearman | within-bin Spearman | top-10% Jaccard | bottom-10% Jaccard |
|---|---|---|---|---|---|
| PRe(AW-H50) vs Hong-AW | 0.242 | 0.494 | 0.485 | 0.085 | 0.473 |

(top/bottom-10% Jaccard of two independent weightings would be 0.053.)

## 3. Per progress bin (K=20)

| bin | data % | bad-tracking hazard %/step | PRe(AW-H50) mass % | Hong-AW mass % | PRe(AW-H50) mean w | Hong-AW mean w | PRe(AW-H50) std w | Hong-AW std w |
|---|---|---|---|---|---|---|---|---|
| 0 | 12.10 | 0.43 | 14.52 | 9.05 | 1.199 | 0.748 | 0.676 | 0.864 |
| 1 | 7.46 | 0.46 | 7.23 | 5.64 | 0.968 | 0.756 | 0.469 | 0.819 |
| 2 | 8.02 | 0.42 | 7.44 | 6.16 | 0.928 | 0.768 | 0.470 | 0.815 |
| 3 | 8.34 | 0.60 | 7.77 | 6.51 | 0.931 | 0.780 | 0.603 | 0.827 |
| 4 | 6.70 | 3.30 | 5.87 | 5.80 | 0.876 | 0.865 | 0.585 | 0.937 |
| 5 | 2.82 | 4.55 | 3.28 | 3.77 | 1.164 | 1.337 | 1.079 | 1.375 |
| 6 | 2.53 | 1.10 | 2.60 | 3.65 | 1.029 | 1.443 | 0.605 | 1.417 |
| 7 | 3.34 | 0.82 | 3.40 | 4.23 | 1.018 | 1.264 | 0.588 | 1.295 |
| 8 | 3.99 | 0.99 | 4.19 | 4.70 | 1.049 | 1.176 | 0.664 | 1.214 |
| 9 | 4.31 | 1.07 | 4.67 | 4.95 | 1.084 | 1.149 | 0.720 | 1.178 |
| 10 | 4.59 | 0.97 | 4.68 | 5.25 | 1.019 | 1.144 | 0.650 | 1.167 |
| 11 | 4.59 | 1.13 | 4.39 | 5.12 | 0.957 | 1.115 | 0.603 | 1.152 |
| 12 | 4.61 | 1.47 | 4.02 | 5.04 | 0.872 | 1.094 | 0.531 | 1.152 |
| 13 | 3.26 | 4.52 | 2.93 | 4.11 | 0.898 | 1.262 | 0.679 | 1.309 |
| 14 | 2.00 | 2.54 | 2.13 | 3.29 | 1.065 | 1.648 | 0.799 | 1.524 |
| 15 | 2.63 | 1.09 | 2.84 | 3.78 | 1.079 | 1.439 | 0.685 | 1.409 |
| 16 | 3.35 | 0.55 | 3.37 | 4.12 | 1.006 | 1.230 | 0.514 | 1.272 |
| 17 | 4.28 | 0.47 | 4.24 | 4.64 | 0.990 | 1.084 | 0.472 | 1.141 |
| 18 | 5.22 | 0.38 | 5.19 | 5.09 | 0.994 | 0.976 | 0.650 | 1.039 |
| 19 | 5.83 | 0.37 | 5.25 | 5.08 | 0.900 | 0.871 | 0.583 | 0.970 |

| weights | Var(w) explained by bin (R2) | corr(bin mean w, bin hazard) | weight mass on the 4 highest-hazard bins (data share 14.8%) |
|---|---|---|---|
| PRe(AW-H50) | 0.0251 | +0.007 | 14.21% |
| Hong-AW | 0.0458 | +0.404 | 16.98% |

Highest-hazard bins: [5, 13, 4, 14].

## 4. What each weight correlates with (Spearman)

| weights | reward r_t | G^H (H-step return) | steps until episode end | episode completes motion (AUC) | within-bin: G^H | within-bin: episode completes (AUC) |
|---|---|---|---|---|---|---|
| PRe(AW-H50) | +0.420 | +0.754 | +0.695 | 0.626 | +1.000 | 0.741 |
| Hong-AW | +0.066 | +0.357 | +0.615 | 0.733 | +0.485 | 0.941 |

AUC = probability that a transition from an episode that completes the motion gets a higher weight than one from an episode that ends by bad tracking (0.5 = no information).

## 5. Transitions that are about to terminate by bad tracking

`fail<=K`: the episode ends by bad tracking within the next K steps (K counted from the transition). Mean weight of those rows (1.0 = dataset mean), and AUC of the weight for separating them from all other rows (AUC < 0.5 means the weighting DOWN-weights soon-to-fail transitions; within-bin AUC removes the effect of where in the motion they are).

| K | share of rows | PRe(AW-H50) mean w | Hong-AW mean w | PRe(AW-H50) AUC | Hong-AW AUC | PRe(AW-H50) within-bin AUC | Hong-AW within-bin AUC |
|---|---|---|---|---|---|---|---|
| 1 | 1.1% | 0.204 | 0.497 | 0.098 | 0.229 | 0.035 | 0.155 |
| 5 | 5.7% | 0.212 | 0.497 | 0.084 | 0.216 | 0.028 | 0.144 |
| 10 | 10.5% | 0.229 | 0.506 | 0.072 | 0.213 | 0.024 | 0.138 |
| 25 | 22.5% | 0.313 | 0.529 | 0.062 | 0.204 | 0.029 | 0.130 |
| 50 | 37.1% | 0.500 | 0.565 | 0.112 | 0.197 | 0.066 | 0.120 |
| 100 | 57.3% | 0.765 | 0.615 | 0.238 | 0.182 | 0.187 | 0.097 |
| 200 | 71.5% | 0.910 | 0.684 | 0.350 | 0.215 | 0.249 | 0.063 |

## 6. Failure AFTER the H=50 window (rows whose episode is still alive at t+H; 52.9% of rows)

Outcome: bad tracking within [t+H, t+H+M). This cannot be read off the H-step return itself.

| M | failure rate | PRe(AW-H50) mean w (fail / survive) | Hong-AW mean w (fail / survive) | PRe(AW-H50) within-bin AUC | Hong-AW within-bin AUC |
|---|---|---|---|---|---|
| 50 | 38.1% | 1.251 / 1.414 | 0.708 / 1.690 | 0.381 | 0.160 |
| 100 | 58.4% | 1.333 / 1.379 | 0.761 / 2.094 | 0.427 | 0.112 |
| 150 | 64.9% | 1.353 / 1.350 | 0.814 / 2.245 | 0.449 | 0.095 |

## 7. Where the weight mass goes

| weights | mass on rows of completing episodes (data share 24.7%) | mass on the last 25 steps before a bad-tracking end (data share 22.5%) |
|---|---|---|
| PRe(AW-H50) | 29.6% | 7.0% |
| Hong-AW | 43.0% | 11.9% |
