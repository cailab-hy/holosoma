# Offline-RL Golden Reference — Audit Summary

> Snapshot date: **2026-05-11**.  
> Manifest: [configs/golden/offline_rl_golden_manifest.yaml](configs/golden/offline_rl_golden_manifest.yaml).  
> No source code was modified to produce this audit.

## 1. Representative configuration table

| Field | CQL | SMQR anchor | SMQR-SG (current best) |
|---|---|---|---|
| run_name | exp_80_perf_cql_seed1_bs4096_300k | exp_81_perf_smqr_qtimesg_seed1_bs4096_300k | exp_S6_1m_fixed05_cqls05_bc30_seed1_bs4096 |
| `algo_mode` | `auto` (→ `cql`) | `smqr_anchor` | `smqr_anchor` |
| `critic_penalty_mode` | `vanilla_cql` | `smqr_cont_self` | `smqr_cont_self` |
| `smqr_anchor_objective` | n/a (`vanilla`) | `vanilla` | `vanilla` |
| `smqr_lse_mode` | n/a (`q_times_g`) | `q_times_g` | **`sg_blend`** |
| `smqr_blend_*` | — | — | schedule=`fixed`, λ_start=λ_end=0.5 |
| `sc_tau_res_scale` | 2.0 (inert) | **0.0** (anchor-only) | **0.0** (anchor-only) |
| `sc_tau_beta` | 1.0 | 1.0 | 1.0 |
| `actor_update_mode` | `sac_bc` | `sac_bc` | `sac_bc` |
| `bc_weight` | 2.5 | 2.5 | **3.0** |
| `cql_loss_scale` | 1.0 (default) | 1.0 (default) | **0.5** |
| `cql_alpha_mode` | `td_relative` | `td_relative` | `td_relative` |
| `cql_td_ratio` | 0.5 | 0.5 | 0.5 |
| `cql_alpha_floor` | 0.008 | 0.008 | 0.008 |
| `batch_size` | 4096 | 4096 | 4096 |
| `critic_learning_rate` | 3.0e-4 | 3.0e-4 | 3.0e-4 |
| `num_learning_iterations` | 300 000 | 300 000 | **1 000 000** |
| `dataset_path` | `offline_data/fastsac_dataset.h5` | same | same |
| Reward / task | g1_29dof WBT + box (sub3_largebox_003) | same | same |
| Train entrypoint | `python -m holosoma.train_agent` | same | same |
| Eval entrypoint | `python -m holosoma.eval_agent` | same | same |
| Checkpoint (golden) | `logs/hv-g1-manager/exp_80_perf_cql_seed1_bs4096_300k/model_0300000.pt` | `logs/hv-g1-manager/exp_81_perf_smqr_qtimesg_seed1_bs4096_300k/model_0300000.pt` | `logs/hv-g1-manager/exp_S6_1m_fixed05_cqls05_bc30_seed1_bs4096/model_1000000.pt` |
| Saved config snapshot | `…/holosoma_config.yaml` (same dir) | same | same |
| Eval results dir | `eval_out/perf_stage1_cql/step_300000/` | `eval_out/perf_stage1_smqr_qtg/step_300000/` | `eval_out/s6_1m/step_1000000/` |

Secondary references pinned in the manifest (for regression coverage of the other SMQR-SG code paths):

| Variant | Run | Step | Why pinned |
|---|---|---|---|
| sg_blend @ 300k | exp_S6_stageS_fixed05_cqls05_bc30_seed1_bs4096_300k | 300 000 | Early-horizon snapshot of the current best |
| pure `sg_weighted_lse` | exp_82_perf_smqrsg_critlr05_seed1_bs4096_300k | 300 000 | Locks the standalone SG code path (critic_lr=1.5e-4) |

## 2. Golden checkpoint manifest

See [configs/golden/offline_rl_golden_manifest.yaml](configs/golden/offline_rl_golden_manifest.yaml). Contains, per algorithm:
- `algorithm_name`, `run_name`, `role`
- `checkpoint_path`, `config_path`, `eval_results_dir`, `train_log`, `tfevents_glob`
- `train_script` (reconstructed CLI), `eval_command` (canonical paired-eval invocation)
- `key_hyperparameters` (every distinguishing flag)
- `expected_eval_metrics` (verbatim from `eval_summary.json`)
- `expected_train_metrics_required_keys` (key-level pinning — see §3)
- `notes`

## 3. Golden metric summary

### Eval metrics (recorded values)

Eval settings (shared across all three): `num_envs=256`, `max_steps=600`, `seed=20240601`, `torch_deterministic=True`, `grasp_radius=0.12`, `lift_height_margin=0.05`, `headless=True`, `single_episode_per_env=True`.

| Metric | CQL @ 300k | SMQR anchor @ 300k | SMQR-SG (sg_blend) @ 1M | SMQR-SG (sg_blend) @ 300k | SMQR-SG (pure sgw) @ 300k |
|---|---|---|---|---|---|
| success_rate | 0.0000 | 0.0938 | **0.2500** | 0.1133 | 0.0039 |
| first_contact_rate | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| grasp_rate_v1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| grasp_rate_v2 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| lift_rate_v1 | 0.3438 | 0.4297 | **0.5703** | 0.4141 | 0.3281 |
| lift_after_contact_rate | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| mean_min_hand_obj_dist | 0.4425 | 0.4120 | **0.3373** | 0.4093 | 0.4343 |
| mean_reward | 11.31 | 17.51 | **26.63** | 18.13 | 12.50 |
| mean_length | 119.31 | 180.44 | **262.99** | 183.90 | 135.47 |

Source files: `eval_out/perf_stage1_cql/step_300000/eval_summary.json`, `eval_out/perf_stage1_smqr_qtg/step_300000/eval_summary.json`, `eval_out/s6_1m/step_1000000/eval_summary.json`, `eval_out/stageS_S6/step_300000/eval_summary.json`, `eval_out/perf_stage1_smqrsg/step_300000/eval_summary.json`.

Note: `first_contact_rate`, `grasp_rate_v1`, `grasp_rate_v2`, `lift_after_contact_rate` are uniformly 0 at these checkpoint steps — they are nonetheless pinned as required keys because future refactors must continue to emit them.

### Train metrics (required keys, not numeric baselines)

Every algorithm's `learn()` step must continue to emit the following keys in its returned metric dict (verified present in [offline_cql_agent.py](src/holosoma/holosoma/agents/offline_cql/offline_cql_agent.py#L3921-L4500)):

- `td_loss`
- `critic_grad_norm`
- `cql_penalty`
- `q_data_mean`
- `cql_q_rand_mean`
- `q_overestimation_gap`
- `actor_loss`
- `bc_loss`
- `action_std`
- `policy_entropy`

Plus universally available: `cql_alpha`.

> Numeric baselines for these scalars live in the TensorBoard event files (`events.out.tfevents.*` under each run dir). They are intentionally **not** transcribed into the manifest in this pass; a follow-up scrape script should emit a sidecar `*_train_metrics_golden.yaml` per run before the refactor lands.

## 4. Must-preserve behaviours (refactor invariants)

These are duplicated in the manifest's `must_preserve_behaviors` section; condensed here:

**Router (`algo_mode.resolve_algo_mode`)**
- `auto` + `vanilla_cql` → `MODE_CQL`.
- `auto` + `smqr_cont_self` + `sc_tau_res_scale==0` → `MODE_SMQR_ANCHOR`.
- Explicit mode contradicting legacy keys raises `ValueError`.

**Numerics**
- τ(s) for anchor-only SMQR ≡ `q_data.detach().min(dim=0).values` (critic axis), shape `[B]`.
- `q_times_g` logits ≡ `Q · g − log p` (no detach on g, no log floor).
- `sg_weighted_lse` logits ≡ `Q + log(g.detach().clamp_min(ε)) − log p`.
- `sg_blend` final per-state penalty ≡ `(1−λ)·P_qg + λ·P_sgw` (loss-level blend, not logit-level).
- Fixed schedule `λ_start=λ_end=0.5` is bit-equivalent to the prior 50/50 blend at all steps.
- `cql_logsumexp` computed in float32 with `clamp(±q_clip)` under AMP.
- `cql_loss = cql_loss_scale · α_effective · penalty_for_loss`; the three α dispatch branches (`td_relative`, `fixed_effective`, Lagrangian) must each remain bit-exact.
- `cql_penalty` telemetry tensor is **unfiltered** even when `cql_penalty_floor_optin=True`; only the loss copy is clamped to ≥0.

**Cross-contamination guards** (must continue to hard-fail; see [offline_cql_agent.py](src/holosoma/holosoma/agents/offline_cql/offline_cql_agent.py#L307-L350))
- `smqr_lse_mode ∈ {q_times_detached_g, sg_weighted_lse, sg_blend}` ⇒ requires `algo_mode='smqr_anchor'` + `smqr_anchor_objective='vanilla'` + `sc_tau_res_scale=0.0` + no F1/G1/H1/B2/Phase-F flags.
- `smqr_anchor_objective='stabilized'` ⇒ requires `smqr_anchor_phase_e_optin=True`.
- `smqr_h1_alpha_floor` and `smqr_b2_alpha_floor` mutually exclusive.

**Logging / IO**
- Train and eval key sets above must not regress.
- Checkpoint layout `model_{step:07d}.pt` + `.onnx`; per-run `holosoma_config.yaml` snapshot remains canonical config record.

## 5. Refactor risk register

| ID | Area | Risk | Mitigation hook |
|---|---|---|---|
| R1 | algo_mode resolver | Splitting modes into Strategy classes can drop explicit-vs-legacy validation | Property-test `resolve_algo_mode()` against the (legacy_key, expected_mode) matrix |
| R2 | τ(s) computation | `min(dim=0)` is the **critic** axis — easy to swap with batch axis during refactor | Shape assert + fixed-fixture numerical equality |
| R3 | 5 SMQR logit branches | ε defaults / detach positions / per-branch clamp can drift | Bit-exact reference-tensor test per logit form |
| R4 | sg_blend λ schedule | 4 schedule modes + warmup/ramp boundaries; off-by-one risk | Test λ(t) at boundary steps for every schedule |
| R5 | `cql_loss_scale` chain | Scale applied AFTER α-dispatch, BEFORE summation; V1 shrinkage is **not** scaled | Toggle test asserts scale affects only `cql_loss`, leaves shrinkage untouched |
| R6 | Cross-contamination guards | Moving to pydantic `__post_init__` can lose allowlist context | Keep guards as a standalone validator + negative tests |
| R7 | `getattr(args, ..., default)` defaults | Centralised config layer must preserve every per-key default | Diff defaults table across refactor |
| R8 | AMP / dtype boundaries | `cql_logsumexp` upcasts to fp32 + clamp; consolidation risks fp16/bf16 overflow | High-\|Q\| smoke test under `amp_dtype=bf16` |
| R9 | Eval reproducibility | Eval scalars depend on actor sampling determinism | Eval-byte regression: rerun eval against each golden checkpoint, require scalars within ±1e-4 |
| R10 | Checkpoint compatibility | Renaming `qnet.tau_head` etc. breaks load of golden checkpoints | State-dict migration table + load smoke test per pinned checkpoint |

## 6. Caveats / known gaps

- Train-metric numeric baselines are pinned by **key** only, not by value. TensorBoard event files (`events.out.tfevents.*`) remain the authoritative numeric source; a follow-up scrape into a sidecar YAML is recommended before the refactor lands.
- Original interactive CLI commands for these runs are not preserved on disk; `train_script` in the manifest is reconstructed from `holosoma_config.yaml` and is functionally equivalent (every leaf maps 1:1 to a tyro flag).
- `exp_82_perf_smqrsg_critlr05` is the only run that diverges from the shared envelope (`critic_learning_rate=1.5e-4` vs 3e-4). It is retained intentionally as the canonical pure-SG-with-halved-LR regression point.
- A large number of currently-inert flags (`smqr_learned_phase_*_optin`, `sc_mask_*`, `sc_severity_*`, `sc_phase_*`) remain in `holosoma_config.yaml`. They must continue to deserialise without raising in any future config layer, even though no production track currently activates them.
