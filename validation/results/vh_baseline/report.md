# Frozen V_H baseline vs PRe phase-bin baseline

dataset `g1_29dof_wbt_fastsac_episode1m_env256_dataset.h5` (rhash 94fc3c2c85c17e69), N=2,525,605, episodes=34,914, H=50, gamma=0.99

V_H: MLP 3x512 ReLU, MSE on G^H, input critic_observations + kappa, 5-fold cross-fitting by episode, early stopping on 10% of training episodes. Total time 1.8 min.

## Fit quality (R2 of G^H)

| fold | rows train | rows OOF | best epoch | R2 train | R2 val | R2 out-of-fold | train - OOF |
|---|---|---|---|---|---|---|---|
| 1 | 1,822,430 | 503,440 | 10 | 0.9350 | 0.9078 | 0.9029 | +0.0320 |
| 2 | 1,808,342 | 512,907 | 9 | 0.9329 | 0.9014 | 0.9021 | +0.0308 |
| 3 | 1,811,412 | 505,329 | 9 | 0.9326 | 0.8994 | 0.9020 | +0.0306 |
| 4 | 1,827,427 | 501,533 | 9 | 0.9321 | 0.9014 | 0.9034 | +0.0288 |
| 5 | 1,821,404 | 502,396 | 7 | 0.9277 | 0.9007 | 0.9016 | +0.0261 |

All rows, out-of-fold: R2(G^H, V_H) = **0.9024**; phase-bin baseline: R2(G^H, b_k) = **0.4084**.

## How much of V_H is progress

- share of Var(V_H) explained by the 20 kappa bins: **0.4426**
- corr(V_H(s_i), b_k(i)) = **0.6652**
- sigma_A: PRe 0.88318 -> V_H 0.35873
- corr(A^V, A^PRe) = 0.3742; within-bin Spearman = 0.3927

## Weights

| weights | sigma used | ESS/N | clip % | max w | std w | Pearson vs PRe | within-bin Spearman vs PRe |
|---|---|---|---|---|---|---|---|
| pre | 0.88318 | 0.716 | 0.00 | 6.12 | 0.629 | nan | nan |
| vh | 0.35873 | 0.495 | 1.13 | 7.06 | 1.010 | 0.213 | 0.393 |
| vh_presigma | 0.88318 | 0.856 | 0.01 | 9.30 | 0.411 | 0.276 | 0.393 |

## Weight mass per progress bin (%)

| bin | data | pre | vh | vh_presigma |
|---|---|---|---|---|
| 0 | 12.10 | 14.52 | 12.70 | 12.37 |
| 1 | 7.46 | 7.23 | 7.02 | 7.43 |
| 2 | 8.02 | 7.44 | 7.47 | 7.94 |
| 3 | 8.34 | 7.77 | 7.95 | 8.21 |
| 4 | 6.70 | 5.87 | 6.08 | 6.44 |
| 5 | 2.82 | 3.28 | 2.99 | 2.86 |
| 6 | 2.53 | 2.60 | 2.75 | 2.60 |
| 7 | 3.34 | 3.40 | 3.70 | 3.42 |
| 8 | 3.99 | 4.19 | 4.71 | 4.14 |
| 9 | 4.31 | 4.67 | 5.41 | 4.59 |
| 10 | 4.59 | 4.68 | 5.05 | 4.65 |
| 11 | 4.59 | 4.39 | 4.73 | 4.58 |
| 12 | 4.61 | 4.02 | 4.34 | 4.50 |
| 13 | 3.26 | 2.93 | 3.19 | 3.20 |
| 14 | 2.00 | 2.13 | 2.19 | 2.03 |
| 15 | 2.63 | 2.84 | 2.91 | 2.72 |
| 16 | 3.35 | 3.37 | 3.34 | 3.36 |
| 17 | 4.28 | 4.24 | 4.22 | 4.29 |
| 18 | 5.22 | 5.19 | 4.76 | 5.14 |
| 19 | 5.83 | 5.25 | 4.50 | 5.54 |
