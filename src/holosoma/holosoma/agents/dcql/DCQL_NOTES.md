# DCQL v1 Notes

DCQL is an escape-ray conservative Q-learning baseline for high-dimensional humanoid offline RL.
It keeps the BF-CQL runtime shell but replaces the conservative machinery.

## Reused from BF/SYNC-CQL

- Factorized actor heads with a global twin critic.
- Normalized critic action space: actor and critic train in `[-1, 1]`; env scaling is applied only at rollout/export.
- Offline HDF5 dataset loading, GPU cache, observation normalization, symmetry augmentation, target Q networks, AMP, logging, save/load, and eval/export flow.
- Bellman target construction, optional max target backup, `q_min/q_max`, Huber/MSE Bellman loss, and standard SAC actor objective.

## Added by DCQL

For each sample, DCQL builds an escape direction from an on-support reference action to the deterministic actor action:

```text
a_ref = a_D                         # v1 dataset reference
a_pi  = tanh(mean_actor(s))
delta = a_pi - a_ref
```

The gate is computed with dataset-action running std:

```text
||delta_sigma|| = sqrt(mean_j (((a_pi_j - a_ref_j) / (sigma_D_j + 1e-6))^2))
active = ||delta_sigma|| >= dcql.delta_thr
```

If active, rays are evaluated in normalized action space:

```text
a_ray_i = clamp(a_ref + t_i * delta + eps_i, -1, 1)
eps_i ~ Normal(0, dcql.ray_noise_std^2)
```

The critic loss adds:

```text
L_dcql = cql_weight * mean_gate(
  tau * log(1/N * sum_i exp(Q(s, a_ray_i) / tau)) - Q(s, a_D)
)
```

The anchor `Q(s, a_D)` is not detached, so descent pushes ray Q down and data-action Q up.
Ray actions and gates are built under `torch.no_grad()`, so DCQL does not update the actor through rays.

## Config Reference

`BFCQLConfig.dcql` defaults:

```yaml
enabled: true
t_grid: [0.5, 1.0, 1.5, 2.0]
ray_noise_std: 0.05
delta_thr: 0.7
gate_norm: batch        # batch | active
a_ref_mode: dataset     # knn is reserved and raises NotImplementedError
drift_std_momentum: 0.999
freeze_drift_stats: false
warmup_ballast_steps: 0
ballast_alpha: 0.1
ballast_num_samples: 8
```

## Extra Q Evaluations

Let `B` be batch size, `R=len(t_grid)`, and `M=ballast_num_samples`.

- Default DCQL top path: one batched critic forward for `B * R` ray actions.
- Existing Bellman/data Q forwards are unchanged.
- Optional warmup ballast for the first `warmup_ballast_steps`: one extra batched critic forward for `B * M` random actions.

So extra critic action evaluations per update are `B * R`, or `B * (R + M)` during ballast warmup.
