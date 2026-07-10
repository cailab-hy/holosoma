# PSC v0 — Principal-Subspace Conservatism

BF-CQL with ONE change: the conservative counterfactual blocks live over the
eigen-directions of the dataset action covariance (data geometry) instead of
joint-index groups. Bellman loss, twin critics, per-block logsumexp-anchor gap,
sum over blocks, SAC actor (full-dim update, `functional_9` heads), data
loading and eval are the BF-CQL machinery unchanged.

## Loss as implemented

Per critic update, for each batch element (s, a_D) with u = U^T(a - mu):

```
for each eigen-block S_g (g = 0..G-1, DESCENDING eigenvalue order):
    u_rand[S_g] ~ Uniform(-m*s_i, +m*s_i)  per direction i in S_g
    u_curr[S_g] = (U^T(a_pi - mu))[S_g],  a_pi ~ pi(.|s)        (curr proposal)
    u_next[S_g] = same with next-state policy samples            (next proposal)
    all other coordinates anchored: u_tilde[-S_g] = u_D[-S_g]
    a_tilde = clamp(U u_tilde + mu, action_low, action_high)     (training-space bounds)

    gap_g = tau * logsumexp_over_proposals( [Q(s,a_tilde) - logdens] / tau ) - Q(s, a_D)

L_cons = alpha_cql * mean_g gap_g          (uniform block weighting, v0)
L      = L_Bellman + L_cons
```

- **Eigen-block convention**: `psc.block_sizes` (default `(3,3,3,3,3,3,4,3,4)`)
  slices the eigen-index in descending-eigenvalue order; block 0 = top
  directions, last block = lowest-variance directions. Sizes must sum to A.
- **Rand scale** (`sqrt_eig_floored`): `s_i = max(sqrt(eig_i),
  quantile(sqrt(eig), scale_floor_quantile))`, coefficient range
  `U(-m*s_i, +m*s_i)` with `m = rand_range_mult`. The floor keeps near-null
  directions probed (we WANT off-manifold counterfactuals there).
- **Rand density**: exact — `-sum_i log(2 m s_i)` over the block. The rotation
  is volume-preserving (`|det U| = 1`), so the uniform density defined in
  u-space needs no extra constant when the critic is evaluated in a-space.
- **Bootstrap**: `bootstrap = (truncations | ~dones)` — restored per spec with
  an inline comment (recurring regression; truncated ends must bootstrap).

## Deviation (documented): curr/next density partition

BF-CQL's importance correction for policy proposals is the exact per-group
marginal log-prob (available because policy dims are independent and groups are
axis-aligned). For a rotated basis the exact block marginal of the tanh-Gaussian
is intractable. v0 uses a **projection-energy partition** of the exact per-dim
log-probs:

```
logp_block_g = sum_j W[g, j] * logp_dim_j,   W[g, j] = sum_{i in S_g} U[j, i]^2
```

Properties: `sum_g W[g, j] = 1` (total log-density preserved across blocks),
commensurate with the block-dimensional rand density, and **exact for
axis-aligned bases** — with `U = I` it reduces to BF-CQL's group log-probs,
which is enforced by the identity-equivalence test (test 5).

## Per-update critic-eval count

Identical to BF-CQL: per block, one twin-critic forward over
`3 * batch * num_repeat` counterfactuals (rand/curr/next stacked), G blocks per
update, plus the same target/bellman/diagnostic evaluations. The extra PSC cost
is only the rotations (a few [B*R, A] x [A, A] matmuls per update).

## Setup-time guards

- basis loaded from `psc.basis_path` ({mu, U, eigvals, meta} from
  `scripts/psc_spectrum.py`); hard fail if U is not orthonormal
  (`||U^T U - I||_inf >= 1e-4`), A != action_dim, or `meta.space` != this run's
  action-space config (normalized vs env).
- `recompute_check=true`: Sigma_D recomputed on a deterministic <=100k
  subsample of the training dataset; hard fail if the top-k (k = first block
  size) projection energy against the loaded basis is <= 0.95.

## Logging (prefix `psc/`)

- `psc/gap_block_{g}` — per-block conservative gap in eigen order. Key
  diagnostic: the story predicts large gaps concentrated in LOW-variance
  (high-g) blocks.
- `psc/rand_scale_mean` (constant), effective rank logged once at setup.
- All standard BF-CQL metrics kept for overlay against the monolithic /
  fixed-rand / physical-grouping baselines.

## Part 0: spectrum measurement

```
# tracking (reference config: env-scaled actions, control action_scale = 1.0)
python scripts/psc_spectrum.py \
    offline_data/g1_29dof_wbt_fastsac_offline_collect_5m_dataset.h5 \
    --space env --control-action-scale 1.0 \
    --out offline_data/psc_basis_wbt.pt --plot offline_data/psc_spectrum_wbt.png

# locomotion
python scripts/psc_spectrum.py \
    offline_data/g1_29dof_loco_fastsac_dataset.h5 \
    --space env \
    --out offline_data/psc_basis_loco.pt --plot offline_data/psc_spectrum_loco.png
```

Reports: full 29-eigenvalue spectrum, cumulative variance, effective rank
`exp(H(lambda))`, and per-physical-group projection energy onto the top-k
eigenspace (the "physical groups approximate the data geometry" measurement).

## Runs

- tracking: `exp:g1-29dof-wbt-psc` (reference config wired in: no action
  normalization + target_entropy_ratio 0.5, all other params = BF-CQL entry)
- loco: `exp:g1-29dof-psc`

## Tests (tests/test_psc.py, 8)

1. rotation round-trip (atol 1e-5)  2. block splice anchors dataset outside S_g
3. counterfactuals within action bounds after clamp  4. basis mismatch guards
(action_dim / space meta / non-orthonormal U / missing keys / block-size sum)
5. **identity-basis equivalence**: U=I, mu=0, blocks == coordinate grouping,
m*s_i == 1 -> PSC critic step == BF-CQL critic step (allclose; also proves
"PSC generalizes BF-CQL")  +rotated-basis-changes-loss sanity
6. checkpoint basis round-trip + action_dim assertion on load.
