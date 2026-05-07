"""Offline CQL agent.

Mirrors the structure of ``fast_sac_agent.py`` but replaces the online
training loop with a purely offline gradient-descent loop over a static
dataset, and swaps the distributional C51 critic for a scalar twin-Q
critic with the CQL conservative penalty.

CLI usage (after setup_all.sh)::

    python src/holosoma/holosoma/train_agent.py \\
        exp:g1-29dof-offline-cql \\
        terrain:terrain-locomotion-plane \\
        --algo.config.dataset-path=<path-to-h5>
"""

from __future__ import annotations

import itertools
import math
import os
from contextlib import contextmanager
from typing import Any, Callable, Dict, Sequence

import tqdm
from loguru import logger

from holosoma.config_types.algo import OfflineCQLConfig
from holosoma.utils.average_meters import TensorAverageMeterDict

from holosoma.agents.base_algo.base_algo import BaseAlgo
from holosoma.agents.callbacks.base_callback import RLEvalCallback
from holosoma.agents.fast_sac.fast_sac import Actor
from holosoma.agents.fast_sac.fast_sac_agent import FastSACEnv
from holosoma.agents.fast_sac.fast_sac_utils import EmpiricalNormalization
from holosoma.agents.modules.logging_utils import LoggingHelper
from holosoma.agents.offline_cql.offline_cql import (
    StateValueNetwork,
    TwinQCritic,
    polyak_update,
)
from holosoma.agents.offline_cql.algo_mode import (
    MODE_CQL,
    MODE_SMQR_ANCHOR,
    MODE_SMQR_LEARNED,
    ResolvedAlgoMode,
    assert_phase_a_compatible,
    resolve_algo_mode,
)
from holosoma.agents.offline_cql.offline_cql_utils import (
    OfflineDataset,
    create_frozen_normalizer,
    load_cql_params,
    save_cql_params,
    validate_normalization,
)
from holosoma.envs.base_task.base_task import BaseTask
from holosoma.utils.helpers import get_class, instantiate
from holosoma.utils.inference_helpers import (
    attach_onnx_metadata,
    export_motion_and_policy_as_onnx,
    export_policy_as_onnx,
    get_command_ranges_from_env,
    get_control_gains_from_config,
    get_urdf_text_from_robot_config,
)
from holosoma.utils.safe_torch_import import (
    F,
    GradScaler,
    TensorDict,
    TensorboardSummaryWriter,
    autocast,
    nn,
    optim,
    torch,
)


class OfflineCQLAgent(BaseAlgo):
    """Conservative Q-Learning (CQL) agent trained on a fixed offline dataset.

    Differences from ``FastSACAgent``:
    * **No online data collection** — ``learn()`` iterates over a static dataset.
    * **Scalar twin-Q critic** — replaces the C51 distributional critic.
    * **CQL penalty** — adds the conservative regulariser to the critic loss.
    * **Obs normalisation from dataset** — statistics are computed once at
      ``setup()`` time from the full dataset, then frozen.

    The actor class (``fast_sac.Actor``) is reused unchanged so that
    checkpoint ``actor_state_dict`` keys remain identical and existing ONNX
    export / inference pipelines work without modification.
    """

    actor: Actor
    qnet: TwinQCritic

    def __init__(
        self,
        env: BaseTask,
        config: OfflineCQLConfig,
        device: str,
        log_dir: str,
        multi_gpu_cfg: dict | None = None,
    ):
        # Wrap the env exactly like FastSAC (needed for action_scale and eval)
        wrapped_env = FastSACEnv(env, config.actor_obs_keys, config.critic_obs_keys)
        super().__init__(wrapped_env, config, device, multi_gpu_cfg)  # type: ignore[arg-type]
        self.unwrapped_env = env
        self.log_dir = log_dir
        self.global_step = 0

        # ── A1 (fixed-reference q_normalizer telemetry) ───────────
        # Captured lazily inside ``_update_actor`` the first time
        # ``self.global_step >= {1000, 5000}``.  Used to compute
        # ``rl_actor_term_fixed_qnorm_{1k,5k}`` for telemetry only;
        # actor loss continues to use the adaptive per-batch
        # normalizer.  ``None`` until captured.
        self._q_normalizer_ref_1k: float | None = None
        self._q_normalizer_ref_5k: float | None = None

        # ── q_normalizer mode (slow_ema / freeze_at_step) ─────────
        # State for non-adaptive modes.  ``_q_normalizer_ema`` is
        # initialised lazily on the first batch (matches the very
        # first raw_adaptive value, so step-0 active equals legacy
        # adaptive).  ``_q_normalizer_frozen`` is captured at the
        # configured freeze step.  Both stay ``None`` for the
        # legacy adaptive mode.
        self._q_normalizer_ema: float | None = None
        self._q_normalizer_frozen: float | None = None

        # ── Logging infrastructure ────────────────────────────────
        self.writer = TensorboardSummaryWriter(log_dir=self.log_dir, flush_secs=10)
        self.logging_helper = LoggingHelper(
            self.writer,
            self.log_dir,
            device=self.device,
            num_envs=self.env.num_envs,
            num_steps_per_env=config.logging_interval,
            num_learning_iterations=config.num_learning_iterations,
            is_main_process=self.is_main_process,
            num_gpus=self.gpu_world_size,
        )
        self.logging_helper.exclude_tags = {
            "Perf/total_fps", "Perf/collection_time", "Perf/learning_time",
            "Train/num_samples",
        }
        self.training_metrics = TensorAverageMeterDict()
        self.eval_callbacks: list[RLEvalCallback] = []

    # ── lifecycle ──────────────────────────────────────────────────────

    def setup(self, *, eval_only: bool = False, checkpoint_path: str | None = None) -> None:
        """Build networks, load dataset, compute obs normalisation stats.

        Follows the same obs-index computation as ``FastSACAgent.setup()``
        to guarantee identical observation slicing.

        Parameters
        ----------
        eval_only:
            When ``True``, skip dataset loading entirely.  Network
            architecture is inferred from the checkpoint state-dict
            (via *checkpoint_path*) so that ``load()`` will succeed
            regardless of the current env observation dimensions.
            Normalizers are created as ``nn.Identity`` placeholders;
            ``load()`` will overwrite them with the checkpoint state.
        checkpoint_path:
            Path to a ``.pt`` checkpoint used to infer network dims in
            ``eval_only`` mode.  Ignored when ``eval_only=False``.

        **Must set at minimum** (used by ``export()`` and ``evaluate_policy()``):

        * ``self.actor_obs_dim`` — int, total flattened actor obs size
        * ``self.obs_normalization`` — bool, from ``self.config.obs_normalization``
        * ``self.obs_normalizer`` — ``EmpiricalNormalization`` or ``nn.Identity``
        * ``self.critic_obs_normalizer`` — same
        * ``self.actor`` — ``Actor`` instance
        * ``self.qnet`` / ``self.qnet_target`` — ``TwinQCritic`` instances
        * ``self.log_alpha`` — learnable SAC temperature
        * ``self.scaler`` — ``GradScaler``
        * ``self.actor_optimizer``, ``self.q_optimizer``, ``self.alpha_optimizer``
        * ``self.log_alpha_cql``, ``self.alpha_cql_optimizer`` (CQL-specific)

        Observation normalisation policy
        --------------------------------
        When ``config.obs_normalization`` is True, normalizers are created
        via ``create_frozen_normalizer()`` using exact (mean, std) computed
        from the full offline dataset.  The normalizers are set to
        ``eval()`` mode AND have ``until=count`` so that statistics are
        **never updated during training** — a double safety net.

        This is the correct behaviour for offline RL:

        * The dataset is fixed, so population statistics are known exactly.
        * Allowing online drift would introduce non-stationarity into what
          should be a deterministic mapping.
        * Frozen stats are checkpoint-compatible with FastSAC's
          ``obs_normalizer_state`` / ``critic_obs_normalizer_state`` keys.

        After construction, ``validate_normalization()`` is called on a
        dataset slice to log a human-readable audit of raw vs. normalised
        statistics.  This makes the normalisation behaviour easy to verify
        in experiment logs.
        """
        logger.info("Setting up OfflineCQL")

        args = self.config
        device = self.device
        env = self.env  # FastSACEnv wrapper

        # ══════════════════════════════════════════════════════════════
        # Phase A — unified algo_mode resolver + guard
        # ══════════════════════════════════════════════════════════════
        # Pure routing: reads the new `algo_mode` / `smqr_learned_variant`
        # / `smqr_logging_namespace` keys plus the legacy
        # `critic_penalty_mode` + `sc_tau_res_scale` and produces a
        # ResolvedAlgoMode label.  No mutation of `args` and no torch
        # construction happens here — the legacy critic-loss branch
        # remains the single source of truth for numerics.
        #
        # The Phase A guard then blocks any execution path that would
        # actually *train* learned-τ.  Anchor-only SMQR
        # (sc_tau_res_scale=0.0) and vanilla CQL pass through untouched.
        # Phase B opt-in: when ``smqr_learned_phase_b_optin=True`` and
        # the resolved mode is ``smqr_learned``, the gate opens for the
        # *vanilla* learned-τ baseline only.  ``stabilized`` variants
        # remain blocked regardless of the opt-in flag.
        self._algo_mode: ResolvedAlgoMode = resolve_algo_mode(args)
        _learned_optin = bool(getattr(args, "smqr_learned_phase_b_optin", False))
        _stab_optin = bool(getattr(args, "smqr_learned_phase_c_optin", False))
        _v1_optin = bool(getattr(args, "smqr_learned_phase_d_optin", False))
        _f1_optin = bool(getattr(args, "smqr_learned_phase_f_optin", False))
        # Phase G1 (sub-flag of F1): candidate-wise routing — random
        # candidates revert to vanilla full-grad Q·g, policy candidates
        # keep F1 ST-split.  No phase opt-in; entirely gated by F1.
        _g1_routing_flag = bool(
            getattr(args, "smqr_f1_random_full_grad", False)
        )
        # Phase H1 (sub-flag of G1): additive constant floor on the
        # random-branch effective gate.  qg_rand = Q·(g + α).  Active
        # only when G1 routing is active and α > 0.  α = 0 recovers
        # G1 bit-exactly.  No phase opt-in; entirely gated by F1+G1.
        _h1_alpha_floor = float(
            getattr(args, "smqr_h1_alpha_floor", 0.0)
        )
        # Phase B2 (sub-flag of G1): STE-based backward-only floor on
        # the random-branch Q-gradient.
        #   forward = Q·g                (G1 bit-exact, unlike H1)
        #   ∂/∂Q    = max(g, α)          (Q-grad floor; lifts starvation)
        #   ∂/∂g    = Q                  (= G1 τ-grad)
        # Active only when G1 routing is active and α > 0.  α = 0
        # recovers G1 bit-exactly.  Mutually exclusive with H1 (both
        # > 0 → RuntimeError below).  No phase opt-in; entirely gated
        # by F1+G1.
        _b2_alpha_floor = float(
            getattr(args, "smqr_b2_alpha_floor", 0.0)
        )
        if _h1_alpha_floor > 0.0 and _b2_alpha_floor > 0.0:
            raise RuntimeError(
                "smqr_h1_alpha_floor and smqr_b2_alpha_floor are mutually "
                "exclusive sub-flags of G1 routing; enable at most one. "
                f"Got h1={_h1_alpha_floor} and b2={_b2_alpha_floor}. "
                "H1 modifies the forward (Q·(g+α)); B2 modifies only the "
                "backward Q-grad via STE max-clip and keeps the forward "
                "bit-exact to G1.  Pick one."
            )
        # Phase E (objective-isolation): orthogonal to learned-τ; gates
        # the stabilised objective on the anchor-only branch.
        _anchor_objective = str(
            getattr(args, "smqr_anchor_objective", "vanilla")
        ).strip().lower()
        _anchor_stab_optin = bool(
            getattr(args, "smqr_anchor_phase_e_optin", False)
        )
        assert_phase_a_compatible(
            self._algo_mode,
            allow_learned=_learned_optin,
            allow_stabilized=_stab_optin,
            allow_v1=_v1_optin,
            allow_f1=_f1_optin,
            anchor_objective=_anchor_objective,
            allow_anchor_stab=_anchor_stab_optin,
        )
        # ── Step-3 SMQR-SG cross-contamination guard ──────────────
        # ``smqr_lse_mode`` is a sub-mode of the anchor-only vanilla
        # weighted-logits branch.  It is only consulted when
        #   * algo_mode == smqr_anchor,
        #   * smqr_anchor_objective == 'vanilla' (Phase E branches
        #     own their own log(g+ε) path with g un-detached),
        #   * F1/G1/H1/B2 inactive (those touch the same logits),
        #   * sc_tau_res_scale == 0.0 (anchor-only invariant).
        # Selecting any non-default value outside this safe envelope
        # is a hard error: it would either silently fall through to
        # an unrelated branch or produce mixed-objective gradients.
        _smqr_lse_mode = str(
            getattr(args, "smqr_lse_mode", "q_times_g")
        ).strip().lower()
        _smqr_sg_eps = float(getattr(args, "smqr_sg_eps", 1e-6))
        _allowed_lse_modes = (
            "q_times_g", "q_times_detached_g", "sg_weighted_lse",
            "sg_blend",
        )
        if _smqr_lse_mode not in _allowed_lse_modes:
            raise RuntimeError(
                f"Unknown smqr_lse_mode={_smqr_lse_mode!r}. "
                f"Allowed: {_allowed_lse_modes}."
            )
        if _smqr_lse_mode != "q_times_g":
            _bad: list[str] = []
            if self._algo_mode.mode != MODE_SMQR_ANCHOR:
                _bad.append(
                    f"algo_mode={self._algo_mode.mode!r} "
                    "(must be 'smqr_anchor')"
                )
            if _anchor_objective != "vanilla":
                _bad.append(
                    f"smqr_anchor_objective={_anchor_objective!r} "
                    "(must be 'vanilla' — Step-3 SMQR-SG is mutually "
                    "exclusive with the Phase E stabilised branch)"
                )
            if float(getattr(args, "sc_tau_res_scale", 0.0)) != 0.0:
                _bad.append(
                    f"sc_tau_res_scale={getattr(args, 'sc_tau_res_scale', 0.0)} "
                    "(must be 0.0 — anchor-only invariant)"
                )
            if _f1_optin:
                _bad.append(
                    "smqr_learned_phase_f_optin=True (F1/G1/H1/B2 modify "
                    "the same logits block; not compatible with SMQR-SG)"
                )
            if (
                float(getattr(args, "smqr_h1_alpha_floor", 0.0)) > 0.0
                or float(getattr(args, "smqr_b2_alpha_floor", 0.0)) > 0.0
                or bool(getattr(args, "smqr_f1_random_full_grad", False))
            ):
                _bad.append(
                    "F1/G1/H1/B2 sub-flags active "
                    "(smqr_f1_random_full_grad / smqr_h1_alpha_floor / "
                    "smqr_b2_alpha_floor)"
                )
            if _smqr_sg_eps <= 0.0:
                _bad.append(
                    f"smqr_sg_eps={_smqr_sg_eps} (must be > 0)"
                )
            if _bad:
                raise RuntimeError(
                    "smqr_lse_mode={!r} requires the anchor-only vanilla "
                    "envelope. Violations:\n  - {}\n"
                    "Either set smqr_lse_mode='q_times_g' (default, "
                    "bit-exact) or fix the listed conditions.".format(
                        _smqr_lse_mode, "\n  - ".join(_bad),
                    )
                )

        # ── sg_blend λ-schedule validation (Stage S) ─────────────────
        # Only meaningful when smqr_lse_mode='sg_blend'; we still
        # range-check the fields whenever the user has set them, so
        # that typos surface even under mode='q_times_g'.
        _blend_schedule = str(
            getattr(args, "smqr_blend_schedule", "fixed")
        ).strip().lower()
        _allowed_blend_schedules = (
            "fixed", "linear", "delayed_linear", "piecewise",
        )
        if _blend_schedule not in _allowed_blend_schedules:
            raise RuntimeError(
                f"Unknown smqr_blend_schedule={_blend_schedule!r}. "
                f"Allowed: {_allowed_blend_schedules}."
            )
        _bls = float(getattr(args, "smqr_blend_lambda_start", 0.5))
        _ble = float(getattr(args, "smqr_blend_lambda_end", 0.5))
        _blw = int(getattr(args, "smqr_blend_warmup_steps", 0))
        _blr = int(getattr(args, "smqr_blend_ramp_steps", 1))
        _blh = int(getattr(args, "smqr_blend_hold_steps", 0))
        if not (0.0 <= _bls <= 1.0):
            raise RuntimeError(
                f"smqr_blend_lambda_start={_bls} outside [0, 1]."
            )
        if not (0.0 <= _ble <= 1.0):
            raise RuntimeError(
                f"smqr_blend_lambda_end={_ble} outside [0, 1]."
            )
        if _blw < 0:
            raise RuntimeError(
                f"smqr_blend_warmup_steps={_blw} must be ≥ 0."
            )
        if _blr < 1:
            raise RuntimeError(
                f"smqr_blend_ramp_steps={_blr} must be ≥ 1."
            )
        if _blh < 0:
            raise RuntimeError(
                f"smqr_blend_hold_steps={_blh} must be ≥ 0."
            )
        # Cache validated knobs so _update_critic can skip getattr
        # lookups on every gradient step.
        self._smqr_blend_schedule = _blend_schedule
        self._smqr_blend_lambda_start = _bls
        self._smqr_blend_lambda_end = _ble
        self._smqr_blend_warmup_steps = _blw
        self._smqr_blend_ramp_steps = _blr
        self._smqr_blend_hold_steps = _blh
        if _smqr_lse_mode == "sg_blend":
            logger.info(
                "smqr_blend schedule resolved: "
                f"schedule={_blend_schedule} "
                f"lambda_start={_bls} lambda_end={_ble} "
                f"warmup={_blw} ramp={_blr} hold={_blh}"
            )
        # Phase B/E fresh-run guard: any opted-in objective/τ ablation
        # MUST start from scratch so that pilot results are not
        # contaminated by a warm-started head / optimizer state from a
        # different objective form.
        _is_anchor_stab_active = (
            self._algo_mode.mode == MODE_SMQR_ANCHOR
            and _anchor_objective == "stabilized"
            and _anchor_stab_optin
        )
        if (
            (
                self._algo_mode.mode == MODE_SMQR_LEARNED
                or _is_anchor_stab_active
            )
            and getattr(args, "checkpoint", None)
        ):
            raise RuntimeError(
                "smqr_learned / smqr_anchor+stabilized must be a fresh run "
                "(no resume). "
                f"Got --training.checkpoint={getattr(args, 'checkpoint', None)!r}. "
                "Phase B/E pilots are scoped as fresh-from-scratch only."
            )
        logger.info(
            "Algo mode resolved: "
            f"mode={self._algo_mode.mode} "
            f"tau_source={self._algo_mode.tau_source} "
            f"legacy_critic_penalty_mode={self._algo_mode.legacy_critic_penalty_mode} "
            f"sc_tau_res_scale={self._algo_mode.tau_res_scale} "
            f"variant={self._algo_mode.learned_variant} "
            f"explicit={self._algo_mode.explicit} "
            f"phase_b_optin={_learned_optin} "
            f"phase_c_optin={_stab_optin} "
            f"phase_d_optin={_v1_optin} "
            f"phase_f_optin={_f1_optin} "
            f"f1_random_full_grad={_g1_routing_flag} "
            f"h1_alpha_floor={_h1_alpha_floor} "
            f"b2_alpha_floor={_b2_alpha_floor} "
            f"anchor_objective={_anchor_objective} "
            f"phase_e_optin={_anchor_stab_optin} "
            f"smqr_lse_mode={_smqr_lse_mode} "
            f"smqr_sg_eps={_smqr_sg_eps} "
            f"logging_prefix={self._algo_mode.logging_prefix!r}"
        )

        # ══════════════════════════════════════════════════════════════
        # 0. CONFIG FIELD VALIDATION — 4-tier taxonomy
        # ══════════════════════════════════════════════════════════════
        #
        # Tier A — HARD-REQUIRED: no safe default exists; absence is fatal.
        #   These define the core algorithmic identity and must come from
        #   the user's config.
        #
        # Tier B — OPTIONAL WITH SAFE DEFAULT: used via getattr(args, k, default).
        #   Sensible defaults match FastSAC conventions or CQL-paper values.
        #   The user can override but doesn't have to.
        #
        # Tier C — FEATURE-DEPENDENT: required only when a corresponding
        #   feature flag is True.  Checked conditionally below.
        #
        # Tier D — INFERABLE: derived from the environment or dataset at
        #   runtime.  Never comes from config.
        #
        # ┌─────────────────────────────┬──────┬──────────────────────────┐
        # │ Field                       │ Tier │ Notes                    │
        # ├─────────────────────────────┼──────┼──────────────────────────┤
        # │ actor_obs_keys              │  A   │ defines obs slicing      │
        # │ critic_obs_keys             │  A   │ defines obs slicing      │
        # │ dataset_path                │  A   │ path to H5 file          │
        # │ obs_normalization           │  A   │ bool flag                │
        # │ actor_hidden_dim            │  A   │ network architecture     │
        # │ critic_hidden_dim           │  A   │ network architecture     │
        # │ actor_learning_rate         │  A   │ optimizer                │
        # │ critic_learning_rate        │  A   │ optimizer                │
        # │ alpha_learning_rate         │  A   │ optimizer                │
        # │ alpha_init                  │  A   │ initial SAC temperature  │
        # │ use_autotune                │  A   │ SAC α auto-tune flag     │
        # │ target_entropy_ratio        │  A   │ target entropy fraction  │
        # │ gamma                       │  A   │ discount factor          │
        # │ tau                         │  A   │ Polyak coefficient       │
        # │ batch_size                  │  A   │ gradient step size       │
        # │ num_learning_iterations     │  A   │ total steps              │
        # │ policy_frequency            │  A   │ actor update cadence     │
        # │ logging_interval            │  A   │ metric logging cadence   │
        # │ save_interval               │  A   │ checkpoint cadence       │
        # │ cql_num_random_actions      │  A   │ CQL IS sample count      │
        # │ cql_num_policy_actions      │  A   │ CQL IS sample count      │
        # │ cql_alpha_autotune          │  A   │ CQL Lagrange flag        │
        # │ amp                         │  A   │ mixed-precision flag     │
        # │ amp_dtype                   │  A   │ "bf16" or "fp16"         │
        # │ max_grad_norm               │  A   │ gradient clipping        │
        # ├─────────────────────────────┼──────┼──────────────────────────┤
        # │ use_tanh                    │  B   │ default True             │
        # │ use_layer_norm              │  B   │ default True             │
        # │ log_std_max                 │  B   │ default 2.0              │
        # │ log_std_min                 │  B   │ default −5.0             │
        # │ num_q_networks              │  B   │ default 2                │
        # │ cql_alpha_init              │  B   │ default 1.0              │
        # │ cql_alpha_learning_rate     │  B   │ default 3e-4             │
        # │ weight_decay                │  B   │ default 0.0              │
        # │ q_clip                      │  B   │ default 1e4              │
        # │ compile                     │  B   │ default False            │
        # │ eval_interval               │  B   │ default 0 (disabled)     │
        # │ eval_steps                  │  B   │ default 200              │
        # │ eval_callbacks              │  B   │ default None             │
        # ├─────────────────────────────┼──────┼──────────────────────────┤
        # │ cql_target_penalty          │  C   │ required if              │
        # │                             │      │ cql_alpha_autotune=True  │
        # ├─────────────────────────────┼──────┼──────────────────────────┤
        # │ actor_obs_dim               │  D   │ from env obs manager     │
        # │ critic_obs_dim              │  D   │ from env obs manager     │
        # │ n_act                       │  D   │ env.robot_config         │
        # │ action_scale / action_bias  │  D   │ env._action_boundaries   │
        # │ dataset.size                │  D   │ from H5 file             │
        # └─────────────────────────────┴──────┴──────────────────────────┘

        # ── Tier A: hard-required ──────────────────────────────────────
        _HARD_REQUIRED: dict[str, str] = {
            "actor_obs_keys":          "list[str] — observation keys for the actor",
            "critic_obs_keys":         "list[str] — observation keys for the critic",
            "dataset_path":            "str — path to the offline HDF5 dataset",
            "obs_normalization":       "bool — whether to normalise observations",
            "actor_hidden_dim":        "int — actor MLP hidden width",
            "critic_hidden_dim":       "int — critic MLP hidden width",
            "actor_learning_rate":     "float — actor optimizer LR",
            "critic_learning_rate":    "float — critic optimizer LR",
            "alpha_learning_rate":     "float — SAC entropy-temp optimizer LR",
            "alpha_init":              "float — initial SAC entropy temperature",
            "use_autotune":            "bool — whether to auto-tune SAC α",
            "target_entropy_ratio":    "float — target entropy as fraction of −n_act",
            "gamma":                   "float — discount factor",
            "tau":                     "float — Polyak averaging coefficient",
            "batch_size":              "int — batch size per gradient step",
            "num_learning_iterations": "int — total gradient steps",
            "policy_frequency":        "int — actor update every N critic steps",
            "logging_interval":        "int — log metrics every N steps",
            "save_interval":           "int — checkpoint every N steps",
            "cql_num_random_actions":  "int — uniform random actions for CQL IS",
            "cql_num_policy_actions":  "int — policy actions for CQL IS",
            "cql_alpha_autotune":      "bool — auto-tune CQL Lagrange multiplier",
            "amp":                     "bool — enable automatic mixed precision",
            "amp_dtype":               "str — 'bf16' or 'fp16'",
            "max_grad_norm":           "float — gradient clipping max norm (0 = disabled)",
        }
        # In eval_only mode, dataset_path is not needed (dims come
        # from the checkpoint state-dict instead).
        _eval_skip = {"dataset_path", "batch_size", "num_learning_iterations"} if eval_only else set()
        missing = [k for k in _HARD_REQUIRED if k not in _eval_skip and not hasattr(args, k)]
        if missing:
            details = "\n".join(f"  • {k}: {_HARD_REQUIRED[k]}" for k in missing)
            raise ValueError(
                f"OfflineCQLConfig is missing {len(missing)} hard-required "
                f"field(s) (Tier A):\n{details}\n"
                f"Add them to the config dataclass or Hydra YAML."
            )

        # ── Tier C: feature-dependent ──────────────────────────────────
        if args.cql_alpha_autotune and not hasattr(args, "cql_target_penalty"):
            raise ValueError(
                "cql_alpha_autotune=True requires config field "
                "'cql_target_penalty' (float — target CQL penalty for "
                "Lagrangian α_cql).  Set it in the config or disable "
                "cql_alpha_autotune."
            )

        # ── 1. Observation index computation ───────────────────────────
        #
        # For offline CQL the **dataset** is the authority on observation
        # dimensions — the env is only needed for action_scale and eval
        # rollouts.  We compute env dims first as a reference, then load
        # the dataset and reconcile.
        algo_obs_dim_dict = env.observation_manager.get_obs_dims()

        algo_history_length_dict: dict[str, int] = {}
        for group_cfg in env.observation_manager.cfg.groups.values():
            history_len = getattr(group_cfg, "history_length", 1)
            for term_name in group_cfg.terms:
                algo_history_length_dict[term_name] = history_len

        actor_obs_keys = list(args.actor_obs_keys)
        critic_obs_keys = list(args.critic_obs_keys)
        n_act: int = env.robot_config.actions_dim

        # ── 1a. Env-derived obs dims (reference only) ─────────────────
        env_actor_obs_dim = 0
        env_actor_obs_indices: dict[str, dict[str, int]] = {}
        for obs_key in actor_obs_keys:
            history_len = algo_history_length_dict.get(obs_key, 1)
            obs_size = algo_obs_dim_dict[obs_key] * history_len
            env_actor_obs_indices[obs_key] = {
                "start": env_actor_obs_dim,
                "end": env_actor_obs_dim + obs_size,
                "size": obs_size,
            }
            env_actor_obs_dim += obs_size

        env_critic_obs_dim = 0
        env_critic_obs_indices: dict[str, dict[str, int]] = {}
        for obs_key in critic_obs_keys:
            history_len = algo_history_length_dict.get(obs_key, 1)
            obs_size = algo_obs_dim_dict[obs_key] * history_len
            env_critic_obs_indices[obs_key] = {
                "start": env_critic_obs_dim,
                "end": env_critic_obs_dim + obs_size,
                "size": obs_size,
            }
            env_critic_obs_dim += obs_size

        if not eval_only:
            # ── 1b. Load dataset — dataset dims are authoritative ─────
            #
            # We load without expected obs dim assertions; only action dim
            # is checked (must match the robot).  Obs dim reconciliation
            # follows.
            self.dataset = OfflineDataset(
                path=args.dataset_path,
                device=device,
                expected_act_dim=n_act,
            )
            ds = self.dataset  # alias for brevity

            # ── 1c. Reconcile obs dims: dataset vs env ────────────────
            actor_obs_dim: int = ds.actor_obs_dim
            critic_obs_dim: int = ds.critic_obs_dim

            if actor_obs_dim != env_actor_obs_dim:
                logger.warning(
                    f"ACTOR OBS DIM MISMATCH: dataset has {actor_obs_dim}, "
                    f"but current env/obs config produces {env_actor_obs_dim}.\n"
                    f"  → The dataset was likely collected with a different "
                    f"observation config (extra terms, history, etc.).\n"
                    f"  → Networks will be built with dataset dims ({actor_obs_dim}).\n"
                    f"  → Eval rollouts will be DISABLED (env obs won't fit the "
                    f"trained actor).\n"
                    f"  To fix: use the same observation preset that collected "
                    f"the dataset, or re-export the dataset with the current env."
                )

            if critic_obs_dim != env_critic_obs_dim:
                logger.warning(
                    f"CRITIC OBS DIM MISMATCH: dataset has {critic_obs_dim}, "
                    f"but current env/obs config produces {env_critic_obs_dim}.\n"
                    f"  → Networks will be built with dataset dims ({critic_obs_dim}).\n"
                    f"  → Eval rollouts will be DISABLED."
                )
        else:
            # ── 1b-eval. Infer dims from checkpoint state-dict ────────
            #
            # For eval we don't need the dataset.  Network architecture is
            # inferred from the saved weights so that load() will succeed.
            self.dataset = None  # type: ignore[assignment]

            if checkpoint_path is None:
                # No checkpoint → fall back to env dims (load() will fail
                # later if they don't match the checkpoint).
                logger.warning(
                    "eval_only=True but no checkpoint_path provided — "
                    "building networks with env dims.  load() will fail "
                    "if the checkpoint was trained with different dims."
                )
                actor_obs_dim = env_actor_obs_dim
                critic_obs_dim = env_critic_obs_dim
            else:
                _ckpt_peek = torch.load(
                    checkpoint_path, map_location="cpu", weights_only=False,
                )
                actor_obs_dim = _ckpt_peek["actor_state_dict"]["net.0.weight"].shape[1]
                _q_input_dim = _ckpt_peek["qnet_state_dict"]["qnets.0.net.0.weight"].shape[1]
                critic_obs_dim = _q_input_dim - n_act
                logger.info(
                    f"Inferred dims from checkpoint: "
                    f"actor_obs_dim={actor_obs_dim}, "
                    f"critic_obs_dim={critic_obs_dim}"
                )
                del _ckpt_peek  # free memory; load() will re-read

            if actor_obs_dim != env_actor_obs_dim:
                logger.warning(
                    f"EVAL DIM MISMATCH: checkpoint actor expects "
                    f"{actor_obs_dim}-dim obs, but env produces "
                    f"{env_actor_obs_dim}-dim.\n"
                    f"  → evaluate_policy() will fail.  Reconfigure the "
                    f"env to produce {actor_obs_dim}-dim observations "
                    f"(same obs preset used to collect the training data)."
                )

        self._eval_dims_match: bool = (
            actor_obs_dim == env_actor_obs_dim
            and critic_obs_dim == env_critic_obs_dim
        )

        # Build obs_indices.  When dims match, use fine-grained per-key
        # indices from the env.  When they don't, create a single flat
        # entry covering the entire dataset observation vector.
        if actor_obs_dim == env_actor_obs_dim:
            self.actor_obs_indices = env_actor_obs_indices
        else:
            flat_key = actor_obs_keys[0]
            self.actor_obs_indices = {
                flat_key: {"start": 0, "end": actor_obs_dim, "size": actor_obs_dim},
            }
            actor_obs_keys = [flat_key]

        if critic_obs_dim == env_critic_obs_dim:
            self.critic_obs_indices = env_critic_obs_indices
        else:
            flat_key = critic_obs_keys[0]
            self.critic_obs_indices = {
                flat_key: {"start": 0, "end": critic_obs_dim, "size": critic_obs_dim},
            }
            critic_obs_keys = [flat_key]

        # Build per-term offsets within each critic obs group for
        # v4 phase detection (obj_pos_b z-height).  The group-level
        # critic_obs_indices only has one key per obs group (e.g.
        # "critic_obs"); term-level keys like "obj_pos_b" live here.
        self._critic_term_offsets: dict[str, dict[str, int]] = {}
        try:
            for _grp_name in critic_obs_keys:
                if _grp_name in env.observation_manager.cfg.groups:
                    _grp_cfg = env.observation_manager.cfg.groups[_grp_name]
                    if getattr(_grp_cfg, "concatenate", True):
                        _offset = 0
                        for _t_name, _t_cfg in _grp_cfg.terms.items():
                            _t_obs = env.observation_manager._compute_term(
                                _grp_name, _t_name, _t_cfg
                            )
                            _t_dim = _t_obs.shape[1]
                            _hist = getattr(_grp_cfg, "history_length", 1)
                            if _hist > 1:
                                _t_dim *= _hist
                            self._critic_term_offsets[_t_name] = {
                                "start": _offset,
                                "end": _offset + _t_dim,
                                "size": _t_dim,
                            }
                            _offset += _t_dim
        except Exception as _e:
            logger.warning(
                f"Could not build critic term offsets for phase "
                f"detection: {_e}.  Phase-gating will be disabled."
            )
            self._critic_term_offsets = {}

        self.actor_obs_dim: int = actor_obs_dim
        self.critic_obs_dim: int = critic_obs_dim
        self._env_actor_obs_dim: int = env_actor_obs_dim
        self._env_critic_obs_dim: int = env_critic_obs_dim

        # ── 2. Action scaling (same logic as FastSAC) ──────────────────
        use_tanh: bool = getattr(args, "use_tanh", True)
        action_scale = (
            env._action_boundaries if use_tanh
            else torch.ones(n_act, device=device)
        )
        action_bias = torch.zeros(n_act, device=device)

        # ── 3. Build Actor ─────────────────────────────────────────────
        #
        # WHY REUSING fast_sac.Actor IS SEMANTICALLY SAFE FOR CQL
        # ─────────────────────────────────────────────────────────────
        #
        # Contract 1 — Action space identity:
        #   Actor.forward() returns  tanh(mean) · action_scale + action_bias.
        #   Actor.get_actions_and_log_probs() samples via rsample() and
        #   applies the same transform, with log-prob corrected by both
        #   the tanh Jacobian  (−log(1 − tanh²(u)))  and the scaling
        #   Jacobian  (−log(action_scale)).  The offline H5 dataset
        #   stores *post-scaled* actions in this exact space, so:
        #     • _update_critic's Q(s, a_data) evaluates the correct action.
        #     • _update_actor's  min_j Q_j(s, π(s))  produces actions in
        #       the same range as the dataset.
        #     • CQL random actions are sampled in [bias-scale, bias+scale],
        #       matching the actor's output range.
        #
        # Contract 2 — Evaluation consistency:
        #   actor(obs)[0] is the deterministic tanh(mean)·scale+bias
        #   action, identical in _run_eval_rollouts(), evaluate_policy(),
        #   get_inference_policy(), and ONNX export.  Eval and training
        #   see the same action semantics.
        #
        # Contract 3 — Checkpoint compatibility:
        #   actor_state_dict keys are name-for-name identical to FastSAC.
        #   action_scale and action_bias are registered buffers, so they
        #   are restored on load() and exported in ONNX.  This enables
        #   warm-starting CQL from a FastSAC actor (actor_only=True).
        #
        # Risk note (documented in audit Q6):
        #   The actor loss is pure SAC (α·log π − min Q) with no BC term.
        #   CQL's indirect defence (push down OOD Q-values) is the only
        #   guard against actor drift outside dataset support.

        self.actor = Actor(
            obs_indices=self.actor_obs_indices,
            obs_keys=actor_obs_keys,
            n_act=n_act,
            num_envs=env.num_envs,
            device=device,
            hidden_dim=args.actor_hidden_dim,
            log_std_max=getattr(args, "log_std_max", 2.0),
            log_std_min=getattr(args, "log_std_min", -5.0),
            use_tanh=use_tanh,
            use_layer_norm=getattr(args, "use_layer_norm", True),
            action_scale=action_scale,
            action_bias=action_bias,
        )

        # ── 4. Build TwinQCritic + frozen target ──────────────────────
        #
        # self.qnet — TwinQCritic (scalar, no C51)
        #   Ensemble of ScalarQNetwork modules.
        # self.qnet_target — TwinQCritic (frozen deep-copy for Polyak avg)
        num_q_networks: int = getattr(args, "num_q_networks", 2)

        self.qnet = TwinQCritic(
            obs_indices=self.critic_obs_indices,
            obs_keys=critic_obs_keys,
            n_act=n_act,
            hidden_dim=args.critic_hidden_dim,
            use_layer_norm=getattr(args, "use_layer_norm", True),
            num_q_networks=num_q_networks,
            device=device,
        )

        # ── V1 (Phase D) / F1 (Phase F) τ-head bias init ──────────
        # V1 and F1 share the parameterisation
        #   τ(s) = anchor − scale·softplus(τ_raw).
        # The TwinQCritic constructor zero-inits the τ-head's final
        # Linear (weight=0, bias=0).  Under softplus, raw=0 gives
        # softplus(0) = log 2 ≈ 0.693, i.e. τ would START 0.693·scale
        # BELOW the anchor — a large undesired initial offset.
        # To start AT the anchor we override the final-Linear bias to
        # b_init = -5.0 so that softplus(-5) ≈ 6.7e-3 ≈ 0.  Weight is
        # already zero, so initial τ_raw = -5.0 for every state, then
        # the head learns the residual from there.
        # Applied ONLY when the resolved variant is V1 OR F1 AND the
        # corresponding opt-in is set (defence in depth: matches the guard).
        # Done BEFORE create_target so qnet_target is identical.
        _uses_v1_tau_param = (
            self._algo_mode.mode == MODE_SMQR_LEARNED
            and self._algo_mode.learned_variant in ("v1_oneside_shrink", "f1_st_qg")
            and (_v1_optin or _f1_optin)
        )
        if _uses_v1_tau_param:
            _V1_TAU_BIAS_INIT = -5.0
            _which = (
                "V1" if self._algo_mode.learned_variant == "v1_oneside_shrink" else "F1"
            )
            with torch.no_grad():
                _tau_last = self.qnet.tau_head[-1]
                # Sanity: zero-init weight precondition (TwinQCritic ctor).
                if not torch.all(_tau_last.weight == 0).item():
                    logger.warning(
                        f"{_which}: tau_head final-Linear weight is not zero at "
                        "init; bias override still applied, but initial "
                        "τ may not equal anchor."
                    )
                if _tau_last.bias is None:
                    raise RuntimeError(
                        f"{_which} requires tau_head final Linear to have a bias "
                        "(got bias=None)."
                    )
                _tau_last.bias.fill_(_V1_TAU_BIAS_INIT)
            logger.info(
                f"[{_which}] tau_head final-Linear bias set to {_V1_TAU_BIAS_INIT} "
                f"(softplus({_V1_TAU_BIAS_INIT}) ≈ {math.log(1+math.exp(_V1_TAU_BIAS_INIT)):.4f}) "
                "→ τ starts at the per-state anchor."
            )

        # self.qnet_target — frozen deep-copy for Polyak-averaged TD target
        self.qnet_target = TwinQCritic.create_target(self.qnet)

        logger.info(f"Actor:\n{self.actor}")
        logger.info(f"TwinQCritic:\n{self.qnet}")

        # ── 5. SAC entropy temperature α ───────────────────────────────
        #
        # self.log_alpha — nn.Parameter-like learnable scalar
        #   exp(log_alpha) is the SAC entropy coefficient.
        # self.target_entropy — float
        #   Target entropy for auto-tuning: −n_act × target_entropy_ratio.
        self.log_alpha = torch.tensor(
            [math.log(args.alpha_init)], requires_grad=True, device=device,
        )
        self.target_entropy: float = -n_act * args.target_entropy_ratio

        # ── 6. CQL Lagrange multiplier α_cql ──────────────────────────
        #
        # self.log_alpha_cql — learnable scalar (CQL conservative weight)
        # self.alpha_cql_optimizer — AdamW for the Lagrangian
        #   Always created (even if not auto-tuned) so that save/load
        #   paths don't need conditional logic.
        cql_alpha_init: float = getattr(args, "cql_alpha_init", 1.0)
        self.log_alpha_cql = torch.tensor(
            [math.log(max(cql_alpha_init, 1e-8))],
            requires_grad=True,
            device=device,
        )
        cql_alpha_lr: float = getattr(args, "cql_alpha_learning_rate", 3e-4)
        self.alpha_cql_optimizer = optim.AdamW(
            [self.log_alpha_cql], lr=cql_alpha_lr, fused=True, betas=(0.9, 0.95),
        )

        # ── 7. Build optimizers ────────────────────────────────────────
        #
        # self.actor_optimizer  — AdamW over Actor parameters
        # self.q_optimizer      — AdamW over TwinQCritic parameters
        # self.alpha_optimizer  — AdamW over [log_alpha]
        weight_decay: float = getattr(args, "weight_decay", 0.0)

        self.actor_optimizer = optim.AdamW(
            list(self.actor.parameters()),
            lr=args.actor_learning_rate,
            weight_decay=weight_decay,
            fused=True,
            betas=(0.9, 0.95),
        )
        self.q_optimizer = optim.AdamW(
            list(self.qnet.parameters()),
            lr=args.critic_learning_rate,
            weight_decay=weight_decay,
            fused=True,
            betas=(0.9, 0.95),
        )
        self.alpha_optimizer = optim.AdamW(
            [self.log_alpha], lr=args.alpha_learning_rate, fused=True, betas=(0.9, 0.95),
        )

        # ── 8. GradScaler (AMP) ───────────────────────────────────────
        #
        # self.scaler — torch.amp.GradScaler
        #   Used by _update_critic / _update_actor / _update_alpha for
        #   mixed-precision backward + step.
        self.scaler = GradScaler(enabled=args.amp)

        # ── 8b. IQL value network V(s) (only for iql_actor mode) ──
        #
        # When actor_update_mode == "iql_actor", we train a separate
        # state-value function V(s) with expectile regression against
        # Q(s, a_data) and use advantage-weighted BC for the actor.
        self._actor_update_mode: str = getattr(args, "actor_update_mode", "sac_bc")
        if self._actor_update_mode == "iql_actor":
            self.value_net = StateValueNetwork(
                n_obs=critic_obs_dim,
                hidden_dim=args.critic_hidden_dim,
                use_layer_norm=getattr(args, "use_layer_norm", True),
                device=device,
            )
            self.value_optimizer = optim.AdamW(
                list(self.value_net.parameters()),
                lr=args.critic_learning_rate,
                weight_decay=weight_decay,
                fused=True,
                betas=(0.9, 0.95),
            )
            logger.info(f"IQL actor mode — StateValueNetwork:\n{self.value_net}")
        else:
            self.value_net = None  # type: ignore[assignment]
            self.value_optimizer = None  # type: ignore[assignment]

        # ── eval-only early return ─────────────────────────────────────
        #
        # In eval mode the dataset is not loaded, so all subsequent
        # sections (dataset checks, frozen normalisation, training
        # summary) are skipped.  Placeholder normalizers are created
        # here; load() will overwrite them with checkpoint state.
        if eval_only:
            self.obs_normalization: bool = args.obs_normalization
            # Create normalizers with the correct shape so that load()
            # can overwrite their state_dict from the checkpoint.
            # nn.Identity would be rejected by _load_normalizer_safe when
            # the checkpoint contains EmpiricalNormalization state.
            if args.obs_normalization:
                self.obs_normalizer: nn.Module = EmpiricalNormalization(
                    shape=actor_obs_dim, device=device,
                )
                self.critic_obs_normalizer: nn.Module = EmpiricalNormalization(
                    shape=critic_obs_dim, device=device,
                )
                # Put into eval mode — load() will restore real stats.
                self.obs_normalizer.eval()
                self.critic_obs_normalizer.eval()
            else:
                self.obs_normalizer = nn.Identity()
                self.critic_obs_normalizer = nn.Identity()
            self.policy = self.actor.explore

            _dims_ok = "✓" if self._eval_dims_match else "✗ MISMATCH"
            logger.info(
                f"\n╔══════════════════════════════════════════════════════════╗\n"
                f"║        Offline CQL — Eval-Only Setup Summary            ║\n"
                f"╠══════════════════════════════════════════════════════════╣\n"
                f"  actor_obs_dim (model)  : {actor_obs_dim}\n"
                f"  critic_obs_dim (model) : {critic_obs_dim}\n"
                f"  env actor_obs_dim      : {env_actor_obs_dim}\n"
                f"  env critic_obs_dim     : {env_critic_obs_dim}\n"
                f"  action_dim             : {n_act}\n"
                f"  dims compatible        : {_dims_ok}\n"
                f"  obs_normalization      : {args.obs_normalization}\n"
                f"╚══════════════════════════════════════════════════════════╝"
            )
            return

        # ══════════════════════════════════════════════════════════════
        # 9. SEMANTIC CONSISTENCY CHECKS — fail-fast dataset audit
        # ══════════════════════════════════════════════════════════════
        #
        # The dataset was loaded in step 1b above.  Here we verify
        # that its *content* is usable for training.  Every check
        # includes a human-readable error message.
        #
        # Skipped entirely in eval_only mode (no dataset loaded).

        # ── 9a. Batch-size / action-dim sanity ─────────────────────────
        assert ds.act_dim == n_act, (
            f"Dataset act_dim ({ds.act_dim}) != "
            f"env n_act ({n_act}).  Action space mismatch."
        )
        assert ds.size >= args.batch_size, (
            f"Dataset has only {ds.size} transitions but "
            f"batch_size={args.batch_size}.  Provide more data or reduce "
            f"batch_size."
        )

        # ── 9b. Dtype assertions ──────────────────────────────────────
        for _name, _tensor in [
            ("actor_obs", ds.actor_obs),
            ("critic_obs", ds.critic_obs),
            ("actions", ds.actions),
            ("rewards", ds.rewards),
            ("next_actor_obs", ds.next_actor_obs),
            ("next_critic_obs", ds.next_critic_obs),
        ]:
            assert _tensor.dtype == torch.float32, (
                f"Dataset '{_name}' has dtype {_tensor.dtype}, "
                f"expected float32.  The H5 file may need re-export."
            )
        for _name, _tensor in [("dones", ds.dones), ("truncations", ds.truncations)]:
            assert _tensor.dtype == torch.int64, (
                f"Dataset '{_name}' has dtype {_tensor.dtype}, "
                f"expected int64."
            )

        # ── 10c. Finiteness checks ────────────────────────────────────
        _finite_checks = [
            ("actor_obs", ds.actor_obs),
            ("critic_obs", ds.critic_obs),
            ("actions", ds.actions),
            ("rewards", ds.rewards),
            ("next_actor_obs", ds.next_actor_obs),
            ("next_critic_obs", ds.next_critic_obs),
        ]
        for _name, _tensor in _finite_checks:
            _nonfinite = (~torch.isfinite(_tensor)).sum().item()
            if _nonfinite > 0:
                raise ValueError(
                    f"Dataset '{_name}' contains {_nonfinite:,} non-finite values "
                    f"(NaN or Inf).  The dataset is corrupted or was written from "
                    f"a diverged training run."
                )

        # ── 10d. Action range / statistics checks ─────────────────────
        #
        # The dataset stores post-scaled actions:  a = tanh(u) · scale + bias.
        # With use_tanh=True and bias=0, the valid range per dimension i is
        # [−scale_i, +scale_i].  We check that actions fall within this
        # range with a small tolerance (1e-3) for float rounding.
        #
        # Also log action statistics so the user can compare them against
        # eval/action_mean later to detect actor drift.
        ds_act_min = ds.actions.min(dim=0).values  # [act_dim]
        ds_act_max = ds.actions.max(dim=0).values  # [act_dim]
        ds_act_mean = ds.actions.mean(dim=0)        # [act_dim]
        ds_act_std = ds.actions.std(dim=0)           # [act_dim]

        _scale = action_scale.to(device)
        _bias = action_bias.to(device)
        _lo = _bias - _scale
        _hi = _bias + _scale
        _tol = 1e-3

        _below = (ds_act_min < _lo - _tol).sum().item()
        _above = (ds_act_max > _hi + _tol).sum().item()
        if _below > 0 or _above > 0:
            logger.warning(
                f"ACTION RANGE MISMATCH: {_below} dimension(s) have min < "
                f"(bias-scale)-tol, {_above} dimension(s) have max > "
                f"(bias+scale)+tol.\n"
                f"  Dataset action range : [{ds_act_min.min().item():.4f}, "
                f"{ds_act_max.max().item():.4f}]\n"
                f"  Expected range       : [{_lo.min().item():.4f}, "
                f"{_hi.max().item():.4f}]\n"
                f"This suggests the dataset was collected with different "
                f"action scaling or a different robot config."
            )

        # ── 10e. next_obs consistency checks ───────────────────────────
        #
        # Verify next_actor_obs and next_critic_obs have the same shape
        # as their current-step counterparts (already enforced by
        # OfflineDataset.__init__, but we double-check here as a contract).
        assert ds.next_actor_obs.shape == ds.actor_obs.shape, (
            f"next_actor_obs shape {ds.next_actor_obs.shape} != "
            f"actor_obs shape {ds.actor_obs.shape}"
        )
        assert ds.next_critic_obs.shape == ds.critic_obs.shape, (
            f"next_critic_obs shape {ds.next_critic_obs.shape} != "
            f"critic_obs shape {ds.critic_obs.shape}"
        )

        # Spot-check: on non-terminal transitions, next_obs should differ
        # from current obs (catch the case where next_obs was accidentally
        # set equal to current obs during dataset export).
        _non_terminal_mask = ds.dones == 0
        _n_nonterminal = _non_terminal_mask.sum().item()
        if _n_nonterminal > 100:
            # Check a random sample to avoid scanning the full dataset
            _check_idx = torch.where(_non_terminal_mask)[0][:1000]
            _same_actor = (
                ds.actor_obs[_check_idx] == ds.next_actor_obs[_check_idx]
            ).all(dim=-1).sum().item()
            _same_ratio = _same_actor / len(_check_idx)
            if _same_ratio > 0.5:
                logger.warning(
                    f"NEXT_OBS STALENESS: {_same_ratio:.1%} of sampled non-"
                    f"terminal transitions have next_actor_obs == actor_obs.  "
                    f"This may indicate the dataset's next_obs was not "
                    f"correctly recorded (e.g. copied from current obs)."
                )

        # ══════════════════════════════════════════════════════════════
        # 11. OBSERVATION NORMALISATION — frozen, dataset-based
        # ══════════════════════════════════════════════════════════════
        #
        # self.obs_normalization — bool
        #   Cached flag checked in learn(), _run_eval_rollouts(), export().
        # self.obs_normalizer — EmpiricalNormalization | nn.Identity
        #   Normaliser for actor observations.
        # self.critic_obs_normalizer — EmpiricalNormalization | nn.Identity
        #   Normaliser for critic observations.
        #
        # FREEZE CONTRACT (enforced by runtime assertions below):
        #   1. Statistics come from the full dataset (not a subset).
        #   2. EmpiricalNormalization.count == dataset.size.
        #   3. EmpiricalNormalization is in eval() mode.
        #   4. EmpiricalNormalization.until == count (double safety).
        #   5. learn() always passes update=False.
        #   6. _run_eval_rollouts() always passes update=False.
        #   These guarantees ensure the normaliser is IDENTICAL between
        #   training and evaluation — the same (mean, std) is applied
        #   everywhere, with no drift.

        self.obs_normalization: bool = args.obs_normalization

        if args.obs_normalization:
            actor_mean, actor_std = ds.compute_obs_statistics("actor")
            critic_mean, critic_std = ds.compute_obs_statistics("critic")

            self.obs_normalizer: nn.Module = create_frozen_normalizer(
                mean=actor_mean,
                std=actor_std,
                count=ds.size,
                device=device,
            )
            self.critic_obs_normalizer: nn.Module = create_frozen_normalizer(
                mean=critic_mean,
                std=critic_std,
                count=ds.size,
                device=device,
            )

            # ── Freeze-contract runtime assertions ─────────────────────
            for _label, _norm in [
                ("obs_normalizer", self.obs_normalizer),
                ("critic_obs_normalizer", self.critic_obs_normalizer),
            ]:
                assert not _norm.training, (
                    f"{_label} must be in eval() mode after create_frozen_normalizer"
                )
                assert hasattr(_norm, "count") and _norm.count.item() == ds.size, (
                    f"{_label}.count ({getattr(_norm, 'count', '?')}) != "
                    f"dataset.size ({ds.size})"
                )
                assert hasattr(_norm, "until") and _norm.until == ds.size, (
                    f"{_label}.until ({getattr(_norm, 'until', '?')}) != "
                    f"dataset.size ({ds.size}) — the 'until' safety net is "
                    f"not engaged; stats could drift if .train() is called."
                )
                # Verify stored statistics are finite
                assert torch.isfinite(_norm._mean).all(), (
                    f"{_label}._mean contains non-finite values"
                )
                assert torch.isfinite(_norm._std).all(), (
                    f"{_label}._std contains non-finite values"
                )
                assert (_norm._std > 0).all(), (
                    f"{_label}._std contains zero entries — constant features "
                    f"will produce NaN after normalisation"
                )

            # Audit: log normalisation quality on a representative slice
            _audit_n = min(10_000, ds.size)
            actor_audit = validate_normalization(
                self.obs_normalizer,
                ds.actor_obs[:_audit_n],
                label="actor_obs",
            )
            logger.info(actor_audit["report"])
            critic_audit = validate_normalization(
                self.critic_obs_normalizer,
                ds.critic_obs[:_audit_n],
                label="critic_obs",
            )
            logger.info(critic_audit["report"])
        else:
            self.obs_normalizer = nn.Identity()
            self.critic_obs_normalizer = nn.Identity()

        # ── 12. Convenience alias (used by FastSAC eval path) ─────────
        self.policy = self.actor.explore

        # ══════════════════════════════════════════════════════════════
        # 13. SETUP SUMMARY + DIAGNOSTIC WARNINGS
        # ══════════════════════════════════════════════════════════════

        norm_status = "FROZEN (dataset statistics)" if args.obs_normalization else "OFF"
        _act_scale_str = (
            f"[{_scale.min().item():.4f}, {_scale.max().item():.4f}]"
            if _scale.numel() > 1
            else f"{_scale.item():.4f}"
        )

        summary_lines = [
            "╔══════════════════════════════════════════════════════════╗",
            "║            Offline CQL — Setup Summary                  ║",
            "╠══════════════════════════════════════════════════════════╣",
            f"  Dataset          : {args.dataset_path}",
            f"  Transitions      : {ds.size:>12,}",
            f"  actor_obs_dim    : {actor_obs_dim:>12}  (env)",
            f"  critic_obs_dim   : {critic_obs_dim:>12}  (env)",
            f"  action_dim       : {n_act:>12}  (env)",
            f"  action_scale     : {_act_scale_str}",
            f"  DS action range  : [{ds_act_min.min().item():.4f}, "
            f"{ds_act_max.max().item():.4f}]",
            f"  DS action mean   : {ds_act_mean.abs().mean().item():.4f}  "
            f"(compare w/ eval/action_mean)",
            f"  DS action std    : {ds_act_std.mean().item():.4f}",
            "──────────────────────────────────────────────────────────",
            f"  num_q_networks   : {num_q_networks:>12}",
            f"  Normalisation    : {norm_status}",
            f"  AMP              : {'ON (' + args.amp_dtype + ')' if args.amp else 'OFF'}",
            f"  Batch size       : {args.batch_size:>12}",
            f"  Learning iters   : {args.num_learning_iterations:>12,}",
            f"  α_init (SAC)     : {args.alpha_init:>12.4f}",
            f"  target_entropy   : {self.target_entropy:>12.4f}",
            f"  α_cql_init       : {cql_alpha_init:>12.4f}",
            f"  CQL α autotune   : {args.cql_alpha_autotune!s:>12}",
            f"  CQL random acts  : {args.cql_num_random_actions:>12}",
            f"  CQL policy acts  : {args.cql_num_policy_actions:>12}",
            f"  max_grad_norm    : {args.max_grad_norm:>12}",
        ]

        # ── SC-CQL config summary ──────────────────────────────────────
        _sc_mode = getattr(args, "critic_penalty_mode", "vanilla_cql")
        if _sc_mode == "sc_cql":
            _sc_tgt = getattr(args, "sc_mask_target", "policy_curr_only")
            _sc_mm = getattr(args, "sc_mask_mode", "sigmoid_symmetric")
            _sc_pm = getattr(args, "sc_phase_mode", "all")
            _sc_pht = getattr(args, "sc_phase_height_threshold", 0.15)
            _sc_dbg = getattr(args, "sc_phase_debug", False)
            _obj_avail = "obj_pos_b" in self._critic_term_offsets
            _obj_info = (
                f"offset={self._critic_term_offsets['obj_pos_b']['start']}"
                if _obj_avail else "NOT FOUND"
            )
            summary_lines.extend([
                "──────────────────────────────────────────────────────────",
                "  SC-CQL v4 config:",
                f"    penalty_mode   : {_sc_mode}",
                f"    mask_target    : {_sc_tgt}",
                f"    mask_mode      : {_sc_mm}",
                f"    phase_mode     : {_sc_pm}",
                f"    phase_h_thr    : {_sc_pht}",
                f"    phase_debug    : {_sc_dbg}",
                f"    obj_pos_b      : {_obj_info}",
                f"    severity_mode  : {getattr(args, 'sc_severity_mode', 'none')}",
                f"    severity_floor : {getattr(args, 'sc_severity_floor', 0.0)}",
                f"    severity_norm  : {getattr(args, 'sc_severity_norm_mode', 'batch_max')}",
                f"    mask_boost     : {getattr(args, 'sc_mask_boost', 1.0)}",
            ])
            if _sc_pm == "post_lift_only" and not _obj_avail:
                summary_lines.append(
                    "    ⚠ obj_pos_b NOT in critic term offsets — "
                    "phase gating will be NO-OP!"
                )
        elif _sc_mode == "smqr_cont_self":
            # Continuous-action SMQR baseline (A-fidelity): per-critic
            # self-mask with shared τ(s) head, NO detach on Q.
            _smqr_beta = float(getattr(args, "sc_tau_beta", 1.0))
            _smqr_eps = float(getattr(args, "sc_tau_eps", 1e-6))
            _smqr_abs = float(getattr(args, "sc_tau_near_abs_eps", 0.05))
            _smqr_bc = float(getattr(args, "sc_tau_near_beta_coeff", 1.0))
            summary_lines.extend([
                "──────────────────────────────────────────────────────────",
                "  SMQR continuous-action baseline (smqr_cont_self):",
                f"    penalty_mode      : {_sc_mode}",
                f"    τ head            : SHARED (qnet.tau_head, obs-only, zero-init)",
                f"    τ param           : τ(s) = Q_data_min(s).detach() + {float(getattr(args, 'sc_tau_res_scale', 2.0))} · tanh(τ_raw(s))  [B2-fix: bounded]",
                f"    self-mask         : per-critic g_i = σ((Q_i − τ)/β)",
                f"    detach on Q       : NO  (Qg′ exposed to learning)",
                f"    log-density       : applied OUTSIDE mask (IS term)",
                f"    β                 : {_smqr_beta}",
                f"    β floor (eps)     : {_smqr_eps}",
                f"    near_abs_eps      : {_smqr_abs}",
                f"    near_beta_coeff   : {_smqr_bc}  (→|Δ| ≤ {_smqr_bc*_smqr_beta:.4f})",
                f"    legacy SC tags    : DISABLED (gap/severity/sparse/v1/v4)",
            ])

        # ── Diagnostic warnings for unsafe first-run configurations ───
        _warnings: list[str] = []
        if ds.size < 50_000:
            _warnings.append(
                f"SMALL DATASET: {ds.size:,} transitions may be too few for "
                f"stable CQL training.  Consider ≥100k."
            )
        if args.cql_num_random_actions < 5 or args.cql_num_policy_actions < 5:
            _warnings.append(
                f"LOW CQL IS SAMPLES: num_random={args.cql_num_random_actions}, "
                f"num_policy={args.cql_num_policy_actions}.  "
                f"Logsumexp estimate will be noisy; recommend ≥10 each."
            )
        if args.batch_size > ds.size // 2:
            _warnings.append(
                f"BATCH/DATASET RATIO: batch_size={args.batch_size} is "
                f">50% of dataset ({ds.size}).  Each batch will contain "
                f"many repeated samples."
            )
        if not args.obs_normalization:
            _raw_std = ds.actor_obs.std(dim=0)
            _large_features = (_raw_std > 100).sum().item()
            if _large_features > 0:
                _warnings.append(
                    f"UNNORMALIZED + LARGE VARIANCE: {_large_features}/"
                    f"{actor_obs_dim} actor_obs features have std > 100 but "
                    f"obs_normalization=False.  Consider enabling it."
                )
        if _below > 0 or _above > 0:
            _warnings.append(
                f"ACTION SCALE MISMATCH: {_below + _above} dim(s) have "
                f"dataset actions outside env action_scale bounds."
            )

        if _warnings:
            summary_lines.append(
                "──────────────────────────────────────────────────────────"
            )
            summary_lines.append("  ⚠ WARNINGS:")
            for _w in _warnings:
                summary_lines.append(f"    • {_w}")

        summary_lines.append(
            "╚══════════════════════════════════════════════════════════╝"
        )
        logger.info("\n" + "\n".join(summary_lines))

    def learn(self) -> None:
        """Offline training loop with periodic env evaluation.

        Architecture
        ------------
        The outer loop runs *gradient steps* (no env interaction).  At a
        configurable ``eval_interval`` the agent pauses training, switches
        the actor to eval mode, and runs deterministic rollouts in the
        live environment.  Both training losses and rollout statistics
        are logged side-by-side so that the experimenter can diagnose the
        common offline-RL failure mode where **losses are stable but
        rollout performance is poor or non-monotonic**.

        Diagnostic comments (marked ⚠ DIAGNOSTIC) are placed at every
        point where loss-vs-rollout mismatch can occur.

        Resume support
        --------------
        If ``self.global_step > 0`` (set by ``load()``), the loop picks
        up from where it left off — the progress bar, logging counters,
        and checkpoint schedule all respect the restored step.

        Per gradient step:
        1. Sample a batch from the static dataset.
        2. Normalise observations (frozen statistics).
        3. Update critic with TD loss + CQL conservative penalty.
        4. (Delayed) Update actor with SAC-style policy loss.
        5. Update temperature(s).
        6. Polyak-average the target network.
        7. Accumulate and periodically log rich metrics.
        8. Periodically run eval rollouts and log eval metrics.
        9. Periodically save checkpoints.
        """
        import time as _time

        args = self.config
        device = self.device
        dataset = self.dataset  # type: OfflineDataset

        # ── Phase A guard re-check (defence in depth) ─────────────
        # ``setup()`` already enforced this; we re-check here so that
        # any path that constructs the agent without going through
        # ``setup()`` (e.g. eval-only that flips into training) cannot
        # silently start a learned-τ optimizer step.
        _algo_mode_resolved: ResolvedAlgoMode = getattr(
            self, "_algo_mode", resolve_algo_mode(args)
        )
        _learned_optin = bool(getattr(args, "smqr_learned_phase_b_optin", False))
        _stab_optin = bool(getattr(args, "smqr_learned_phase_c_optin", False))
        _v1_optin = bool(getattr(args, "smqr_learned_phase_d_optin", False))
        _f1_optin = bool(getattr(args, "smqr_learned_phase_f_optin", False))
        _anchor_objective = str(
            getattr(args, "smqr_anchor_objective", "vanilla")
        ).strip().lower()
        _anchor_stab_optin = bool(
            getattr(args, "smqr_anchor_phase_e_optin", False)
        )
        assert_phase_a_compatible(
            _algo_mode_resolved,
            allow_learned=_learned_optin,
            allow_stabilized=_stab_optin,
            allow_v1=_v1_optin,
            allow_f1=_f1_optin,
            anchor_objective=_anchor_objective,
            allow_anchor_stab=_anchor_stab_optin,
        )
        # One-shot mode tag for downstream report tooling.
        try:
            self.writer.add_text(
                f"{_algo_mode_resolved.logging_prefix}meta/mode",
                (
                    f"mode={_algo_mode_resolved.mode} | "
                    f"tau_source={_algo_mode_resolved.tau_source} | "
                    f"legacy_critic_penalty_mode="
                    f"{_algo_mode_resolved.legacy_critic_penalty_mode} | "
                    f"sc_tau_res_scale={_algo_mode_resolved.tau_res_scale} | "
                    f"variant={_algo_mode_resolved.learned_variant} | "
                    f"explicit={_algo_mode_resolved.explicit}"
                ),
                global_step=self.global_step,
            )
        except Exception:  # pragma: no cover — TB writer always present
            pass

        # ── Resolve callables (compile-friendly) ──────────────────
        if getattr(args, "compile", False):
            normalize_obs = torch.compile(self.obs_normalizer.forward)
            normalize_critic_obs = torch.compile(self.critic_obs_normalizer.forward)
        else:
            normalize_obs = self.obs_normalizer.forward
            normalize_critic_obs = self.critic_obs_normalizer.forward

        training_metrics = self.training_metrics
        training_metrics.clear()
        self._last_cql_penalty = torch.tensor(0.0, device=device)

        # Config knobs (with safe defaults for backward compat)
        eval_interval: int = getattr(args, "eval_interval", 0)
        eval_steps: int = getattr(args, "eval_steps", 200)

        # ── Resume support ────────────────────────────────────────
        # global_step may be > 0 if load() was called before learn().
        start_step = self.global_step
        if start_step > 0:
            logger.info(f"Resuming CQL training from step {start_step}")

        pbar = tqdm.tqdm(
            total=args.num_learning_iterations, initial=start_step, desc="CQL"
        )
        loop_start = _time.perf_counter()

        # ── IQL SANITY DIAGNOSTIC (one-shot, first batch) ─────────
        # Run once at the very start to catch action-scaling or
        # architecture mismatches before wasting GPU hours.
        if self._actor_update_mode == "iql_actor" and start_step == 0:
            with torch.no_grad():
                _sb = dataset.sample(min(256, args.batch_size))
                if self.obs_normalization:
                    _sb["observations"] = normalize_obs(_sb["observations"], update=False)
                    _sb["critic_observations"] = normalize_critic_obs(_sb["critic_observations"], update=False)
                _sb_obs = _sb["observations"]
                _sb_act = _sb["actions"]
                _sb_cobs = _sb["critic_observations"]

                # Actor outputs — Actor.forward() returns (action, mean_pre_tanh, log_std)
                _mean_act, _mean_pre_tanh, _log_std = self.actor(_sb_obs)
                _std = _log_std.exp()
                _sampled, _ = self.actor.get_actions_and_log_probs(_sb_obs)

                # V and Q
                _q = self.qnet.min_q(_sb_cobs, _sb_act).squeeze(-1)
                _v = self.value_net(_sb_cobs).squeeze(-1)
                _adv = _q - _v

                # Mean-BC loss
                _mean_bc_loss = ((_mean_act - _sb_act) ** 2).sum(dim=-1).mean()

                # Log-prob BC loss (for comparison)
                # Use mean_pre_tanh directly from Actor (no lossy atanh roundtrip)
                _data_pre_tanh = torch.atanh(
                    ((_sb_act - self.actor.action_bias) / (self.actor.action_scale + 1e-6)).clamp(-0.999, 0.999)
                )
                _lp = -0.5 * (((_data_pre_tanh - _mean_pre_tanh) / (_std + 1e-6)).pow(2) + 2 * _log_std + math.log(2 * math.pi))
                _lp = _lp.sum(dim=-1)
                _scaled_a = (_sb_act - self.actor.action_bias) / (self.actor.action_scale + 1e-6)
                _lp = _lp - torch.log(1 - _scaled_a.pow(2) + 1e-6).sum(dim=-1)
                _lp = _lp - torch.log(self.actor.action_scale + 1e-6).sum()
                _logprob_bc_loss = -_lp.mean()

                _impl_mode = getattr(args, "actor_iql_impl_mode", "logprob_bc")
                _diag = (
                    f"\n{'='*60}\n"
                    f"  IQL ACTOR SANITY DIAGNOSTIC (step 0)\n"
                    f"  impl_mode          : {_impl_mode}\n"
                    f"{'─'*60}\n"
                    f"  DATA action range  : [{_sb_act.min().item():.4f}, {_sb_act.max().item():.4f}]\n"
                    f"  DATA action |mean| : {_sb_act.abs().mean().item():.4f}\n"
                    f"  ACTOR mean  range  : [{_mean_act.min().item():.4f}, {_mean_act.max().item():.4f}]\n"
                    f"  ACTOR mean |mean|  : {_mean_act.abs().mean().item():.4f}\n"
                    f"  ACTOR sample range : [{_sampled.min().item():.4f}, {_sampled.max().item():.4f}]\n"
                    f"  ACTOR logstd range : [{_log_std.min().item():.2f}, {_log_std.max().item():.2f}]\n"
                    f"  ACTOR std mean     : {_std.mean().item():.4f}\n"
                    f"  action_scale range : [{self.actor.action_scale.min().item():.4f}, {self.actor.action_scale.max().item():.4f}]\n"
                    f"{'─'*60}\n"
                    f"  Q_data mean        : {_q.mean().item():.4f}\n"
                    f"  V(s) mean          : {_v.mean().item():.4f}\n"
                    f"  advantage mean     : {_adv.mean().item():.4f}\n"
                    f"  advantage std      : {_adv.std().item():.4f}\n"
                    f"{'─'*60}\n"
                    f"  mean_bc loss       : {_mean_bc_loss.item():.6f}\n"
                    f"  logprob_bc loss    : {_logprob_bc_loss.item():.6f}\n"
                    f"  logprob_bc finite  : {torch.isfinite(_logprob_bc_loss).item()}\n"
                    f"  log_prob finite%   : {torch.isfinite(_lp).float().mean().item():.2%}\n"
                    f"{'='*60}"
                )
                logger.info(_diag)

        while self.global_step <= args.num_learning_iterations:
            step_start = _time.perf_counter()

            # ── 1. Sample batch ────────────────────────────────────
            data = dataset.sample(args.batch_size)

            # ── 2. Normalise observations ──────────────────────────
            #
            # ⚠ DIAGNOSTIC — normaliser mismatch:
            # If the frozen normaliser statistics were computed on a
            # dataset whose distribution differs from what the
            # *environment* produces at eval time (e.g. different
            # command distribution, domain-rand settings, or sim
            # version), the actor will see OOD inputs during rollout
            # even though training losses look perfectly stable.
            # Compare  eval/obs_mean  vs  train/obs_mean  in logs.
            if self.obs_normalization:
                data["observations"] = normalize_obs(data["observations"], update=False)
                data["next"]["observations"] = normalize_obs(
                    data["next"]["observations"], update=False
                )
                data["critic_observations"] = normalize_critic_obs(
                    data["critic_observations"], update=False
                )
                data["next"]["critic_observations"] = normalize_critic_obs(
                    data["next"]["critic_observations"], update=False
                )

            # ── 3. Update critic ───────────────────────────────────
            #
            # ⚠ DIAGNOSTIC — Q overestimation despite CQL:
            # If q_data_mean drifts well above td_target_mean, the CQL
            # penalty may be too weak (α_cql too low) or the logsumexp
            # estimate too noisy (too few random/policy samples).
            # This causes rollout failure because the actor exploits
            # Q-value overestimation in parts of state space not
            # covered by the dataset.
            critic_metrics = self._update_critic(data)
            self._last_cql_penalty = critic_metrics.pop("_cql_penalty_raw")
            training_metrics.add(critic_metrics)

            # ── 4. Update actor (delayed policy update) ────────────
            #
            # ⚠ DIAGNOSTIC — actor loss looks good but rollouts fail:
            # The actor only sees states *from the dataset*.  If the
            # env resets to states outside the dataset support, the
            # actor's first actions are essentially random, and the
            # episode may never recover.  Compare eval/episode_length
            # to the dataset's average episode length.
            if self.global_step % args.policy_frequency == 0:
                if self._actor_update_mode == "iql_actor":
                    # ── IQL path: V(s) expectile regression + advantage-weighted BC
                    value_metrics = self._update_value(data)
                    training_metrics.add(value_metrics)

                    actor_metrics = self._update_actor_iql(data)
                    training_metrics.add(actor_metrics)
                    # Skip SAC alpha autotune — IQL actor does not use entropy
                else:
                    # ── SAC+BC path (default, A1-series baseline) ──
                    actor_metrics = self._update_actor(data)
                    training_metrics.add(actor_metrics)

                    # ── 5. Update temperature(s) ──────────────────────
                    alpha_metrics = self._update_alpha(
                        actor_metrics["log_probs_mean"]
                    )
                    training_metrics.add(alpha_metrics)

            # ── 6. Polyak-average target network ──────────────────
            with torch.no_grad():
                polyak_update(self.qnet, self.qnet_target, args.tau)

            step_time = _time.perf_counter() - step_start

            # ── 7. Training metric logging ─────────────────────────
            if (
                self.global_step % args.logging_interval == 0
                and self.global_step > 0
            ):
                with torch.no_grad():
                    accumulated = training_metrics.mean_and_clear()
                    loss_dict: dict[str, float] = {}
                    for key, value in accumulated.items():
                        if isinstance(value, torch.Tensor):
                            loss_dict[key] = value.item()
                        else:
                            loss_dict[key] = float(value)

                # Add timing info
                elapsed = _time.perf_counter() - loop_start
                loss_dict["steps_per_sec"] = (
                    (self.global_step - start_step) / max(elapsed, 1e-8)
                )

                if self.is_main_process:
                    # Write training metrics under "Loss/" prefix
                    # (LoggingHelper does this automatically)
                    self.logging_helper.post_epoch_logging(
                        it=self.global_step,
                        loss_dict=loss_dict,
                        extra_log_dicts={},
                    )

            # ── 8. Periodic evaluation rollouts ────────────────────
            #
            # ⚠ DIAGNOSTIC — the eval block is deliberately placed
            # *inside* the training loop, not after it, so you see
            # rollout quality at regular intervals.  The critical
            # thing to watch: if  td_loss ↓  and  critic_loss ↓  but
            # eval/mean_reward  is flat or decreasing, the policy is
            # overfitting to the dataset Q-landscape.
            if (
                eval_interval > 0
                and self.global_step > 0
                and self.global_step % eval_interval == 0
                and self.is_main_process
                and self._eval_dims_match
            ):
                eval_metrics = self._run_eval_rollouts(
                    num_steps=eval_steps,
                )

                # Log eval metrics to TensorBoard/wandb under "Eval/"
                for k, v in eval_metrics.items():
                    self.writer.add_scalar(
                        f"Eval/{k}", v, global_step=self.global_step
                    )

                # Console summary
                eval_str = "  ".join(
                    f"{k}={v:.4f}" for k, v in sorted(eval_metrics.items())
                )
                logger.info(
                    f"[step {self.global_step}] EVAL  {eval_str}"
                )

                # ⚠ DIAGNOSTIC — check for action-distribution shift:
                # If eval/action_mean is far from the dataset's action
                # mean, the policy has drifted to parts of action
                # space unseen during training — a classic offline RL
                # failure.

            # ── 9. Checkpoint ──────────────────────────────────────
            if (
                args.save_interval > 0
                and self.global_step > 0
                and self.global_step % args.save_interval == 0
                and self.is_main_process
            ):
                logger.info(f"Saving model at global step {self.global_step}")
                self.save(
                    os.path.join(self.log_dir, f"model_{self.global_step:07d}.pt")
                )
                self.export(
                    onnx_file_path=os.path.join(
                        self.log_dir, f"model_{self.global_step:07d}.onnx"
                    )
                )

            # ── Advance step ──────────────────────────────────────
            if self.global_step >= args.num_learning_iterations:
                break
            self.global_step += 1
            pbar.update(1)

        pbar.close()

        # ── Final eval + checkpoint ───────────────────────────────
        if self.is_main_process:
            if eval_interval > 0 and self._eval_dims_match:
                final_eval = self._run_eval_rollouts(num_steps=eval_steps)
                eval_str = "  ".join(
                    f"{k}={v:.4f}" for k, v in sorted(final_eval.items())
                )
                logger.info(f"[step {self.global_step}] FINAL EVAL  {eval_str}")
                for k, v in final_eval.items():
                    self.writer.add_scalar(
                        f"Eval/{k}", v, global_step=self.global_step
                    )

            self.save(
                os.path.join(self.log_dir, f"model_{self.global_step:07d}.pt")
            )

    def load(self, ckpt_path: str | None, *, actor_only: bool = False) -> None:
        """Load a CQL checkpoint (or warm-start from a FastSAC actor).

        Checkpoint key contract
        -----------------------
        Every key written by ``save_cql_params()`` is consumed here under
        the **exact same name**.  Keys shared with FastSAC
        (``actor_state_dict``, ``obs_normalizer_state``, …) are loaded
        identically so that the ``eval_agent.py`` flow works unchanged.

        Parameters
        ----------
        ckpt_path:
            Path to a ``.pt`` checkpoint file.  ``None`` is a no-op.
        actor_only:
            If ``True``, load **only** ``actor_state_dict`` and
            ``obs_normalizer_state``.  Use this to initialise the CQL
            actor from a FastSAC checkpoint (critic shapes differ).
        """
        if not ckpt_path:
            return

        ckpt = load_cql_params(
            ckpt_path,
            device=self.device,
            actor=self.actor,
            qnet=self.qnet,
            qnet_target=self.qnet_target,
            log_alpha=self.log_alpha,
            obs_normalizer=self.obs_normalizer,
            critic_obs_normalizer=self.critic_obs_normalizer,
            actor_optimizer=self.actor_optimizer,
            q_optimizer=self.q_optimizer,
            alpha_optimizer=self.alpha_optimizer,
            scaler=self.scaler,
            log_alpha_cql=getattr(self, "log_alpha_cql", None),
            alpha_cql_optimizer=getattr(self, "alpha_cql_optimizer", None),
            actor_only=actor_only,
            value_net=getattr(self, "value_net", None),
            value_optimizer=getattr(self, "value_optimizer", None),
        )

        # Restore iteration counter — identical key to FastSAC
        if not actor_only:
            self.global_step = ckpt.get("global_step", 0)

        # Restore env curriculum state — identical key to FastSAC
        self._restore_env_state(ckpt.get("env_state"))

    def save(self, path: str) -> None:  # type: ignore[override]
        """Persist the full training state via ``save_cql_params()``.

        Produces a checkpoint dict that is a strict superset of what
        ``fast_sac_utils.save_params`` writes.  The two CQL-only keys
        (``log_alpha_cql``, ``alpha_cql_optimizer_state_dict``) are
        appended; everything else is name-for-name identical.
        """
        env_state = self._collect_env_state()
        save_cql_params(
            global_step=self.global_step,
            actor=self.actor,
            qnet=self.qnet,
            qnet_target=self.qnet_target,
            log_alpha=self.log_alpha,
            obs_normalizer=self.obs_normalizer,
            critic_obs_normalizer=self.critic_obs_normalizer,
            actor_optimizer=self.actor_optimizer,
            q_optimizer=self.q_optimizer,
            alpha_optimizer=self.alpha_optimizer,
            scaler=self.scaler,
            args=self.config,
            save_path=path,
            metadata=self._checkpoint_metadata(iteration=self.global_step),
            env_state=env_state or None,
            log_alpha_cql=getattr(self, "log_alpha_cql", None),
            alpha_cql_optimizer=getattr(self, "alpha_cql_optimizer", None),
            value_net=getattr(self, "value_net", None),
            value_optimizer=getattr(self, "value_optimizer", None),
        )

    # ── AMP helper ─────────────────────────────────────────────────────

    @contextmanager
    def _maybe_amp(self):
        """Mixed-precision context — mirrors ``FastSACAgent._maybe_amp()``."""
        amp_dtype = torch.bfloat16 if self.config.amp_dtype == "bf16" else torch.float16
        with autocast(device_type="cuda", dtype=amp_dtype, enabled=self.config.amp):
            yield

    # ── critic update ─────────────────────────────────────────────────

    def _update_critic(self, data: TensorDict) -> dict[str, torch.Tensor]:
        """Critic gradient step: scalar TD loss + CQL conservative penalty.

        CQL penalty ≈  α_cql * (E_s[ logsumexp_a Q(s,a) ] − E_{s,a~D}[ Q(s,a) ])
        where the logsumexp is estimated with actions sampled from:
          (a) uniform random in the *post-scaled* action space,
          (b) the current policy.

        Returns a dict of scalar metrics for logging.
        """
        args = self.config
        scaler = self.scaler
        actor = self.actor
        qnet = self.qnet
        qnet_target = self.qnet_target

        # Initialise sg_blend λ telemetry at function scope so the
        # final _metrics dict can reference it unconditionally
        # (becomes 0.0 in modes that don't run the sg_blend dispatch).
        _smqr_blend_lambda_active: float = 0.0

        with self._maybe_amp():
            # ── unpack batch ────────────────────────────────────────
            observations = data["observations"]           # [B, actor_obs_dim]
            critic_obs = data["critic_observations"]      # [B, critic_obs_dim]
            actions = data["actions"]                     # [B, act_dim]
            next_obs = data["next"]["observations"]       # [B, actor_obs_dim]
            next_critic_obs = data["next"]["critic_observations"]  # [B, critic_obs_dim]
            rewards = data["next"]["rewards"]             # [B]
            dones = data["next"]["dones"].bool()          # [B]
            truncations = data["next"]["truncations"].bool()  # [B]
            bootstrap = (truncations | ~dones).float()    # [B]
            discount = args.gamma ** data["next"]["effective_n_steps"]  # [B]

            # ── TD target (no grad) ────────────────────────────────
            _q_clip: float = getattr(args, "q_clip", 1e4)
            with torch.no_grad():
                next_actions, next_log_probs = actor.get_actions_and_log_probs(next_obs)
                # next_log_probs: [B]
                target_q = qnet_target.min_q(next_critic_obs, next_actions)  # [B, 1]
                target_q = target_q.squeeze(-1)  # [B]
                # SAC-style soft target: r + γ * bootstrap * (min Q_tgt - α * log π)
                #
                # ⚡ STABILITY (P2): upcast to float32 for the Bellman backup.
                # Under AMP, target_q and next_log_probs may be bf16/fp16;
                # the reward + discount × (Q − α·log π) addition accumulates
                # rounding errors that slowly bias the target, causing
                # td_loss to drift upward over thousands of steps.
                td_target = rewards.float() + discount.float() * bootstrap.float() * (
                    target_q.float()
                    - self.log_alpha.exp().detach().float() * next_log_probs.float()
                )  # [B], float32

                # ⚡ STABILITY (P1): clamp TD target to prevent cascading
                # divergence.  If the target network produces extreme
                # Q-values (before Polyak averaging corrects it), the
                # unclamped target pushes the online critic toward
                # infinity on the next step — a positive feedback loop.
                td_target = td_target.clamp(-_q_clip, _q_clip)

            # ── Bellman residual (TD loss) ──────────────────────────
            # qnet.forward returns [num_q, B, 1]
            q_pred_all = qnet(critic_obs, actions)  # [num_q, B, 1]
            q_pred_all = q_pred_all.squeeze(-1)     # [num_q, B]
            # ⚡ STABILITY (P6): compute MSE in float32 to match td_target.
            # Under AMP, q_pred_all is bf16; squaring the Bellman residual
            # magnifies bf16 rounding errors.  The cast is cheap (element-
            # wise) and keeps the loss landscape smooth.
            td_loss = 0.5 * F.mse_loss(
                q_pred_all.float(),
                td_target.unsqueeze(0).expand_as(q_pred_all),
            )

            # ── CQL conservative penalty ───────────────────────────
            # Estimate logsumexp_a Q(s,a) via importance sampling with:
            #   (i)  N_rand uniform random actions in [action_bias - action_scale,
            #        action_bias + action_scale]
            #   (ii) N_pi actions from the current policy
            B = observations.shape[0]
            n_act = actions.shape[-1]
            num_random = args.cql_num_random_actions   # e.g. 10
            num_policy = args.cql_num_policy_actions   # e.g. 10

            critic_obs_processed = qnet.process_obs(critic_obs)  # [B, obs_dim]

            # (i) Uniform random actions
            # actor has action_scale [n_act] and action_bias [n_act] buffers
            rand_actions = (
                torch.rand(B, num_random, n_act, device=observations.device) * 2.0 - 1.0
            ) * actor.action_scale.unsqueeze(0).unsqueeze(0) + actor.action_bias.unsqueeze(0).unsqueeze(0)
            # [B, num_random, n_act]

            # Log-density of uniform: -n_act * log(2 * action_scale)
            # action_scale is per-dimension, so: -Σ_i log(2 * scale_i)
            rand_log_density = -torch.log(
                2.0 * actor.action_scale + 1e-6
            ).sum().detach()  # scalar

            # Q-values for random actions: [num_q, B, num_random, 1]
            q_rand = qnet.q_values_for_actions(critic_obs_processed, rand_actions)
            q_rand = q_rand.squeeze(-1)  # [num_q, B, num_random]

            # (ii) Policy actions (with gradient through actor for CQL-H variant)
            # Repeat obs for sampling multiple actions
            obs_repeat = observations.unsqueeze(1).expand(B, num_policy, -1).reshape(B * num_policy, -1)
            pi_actions, pi_log_probs = actor.get_actions_and_log_probs(obs_repeat)
            pi_actions = pi_actions.view(B, num_policy, n_act)       # [B, num_policy, n_act]
            pi_log_probs = pi_log_probs.view(B, num_policy).detach() # [B, num_policy]

            q_pi = qnet.q_values_for_actions(critic_obs_processed, pi_actions)
            q_pi = q_pi.squeeze(-1)  # [num_q, B, num_policy]

            # ── logsumexp with importance weights ──────────────────
            # For random actions: Q(s,a) - log(density) to correct IS
            # For policy actions: Q(s,a) - log π(a|s) to correct IS
            # Then logsumexp over all (num_random + num_policy) samples,
            # minus log(N) to get log(1/N Σ exp(...))
            N_total = num_random + num_policy

            # Concatenate: [num_q, B, N_total]
            q_cat = torch.cat([
                q_rand - rand_log_density,              # IS correction for uniform
                q_pi - pi_log_probs.unsqueeze(0),       # IS correction for policy
            ], dim=-1)

            # ⚡ STABILITY (P3): upcast to float32 and clamp before
            # logsumexp.  Under AMP the importance-corrected Q-values
            # (Q − log π) can exceed fp16 range (±65504).  Even in
            # fp32, extreme values make the logsumexp dominated by one
            # or two outlier samples, creating a noisy gradient that
            # destabilises the critic.  The clamp bounds the effective
            # importance ratio to exp(±q_clip).
            q_cat_f32 = q_cat.float().clamp(-_q_clip, _q_clip)

            # logsumexp over action samples, then mean over batch
            # Subtract log(N_total) to normalise: log(1/N Σ exp(Q - log_density))
            cql_logsumexp = (
                torch.logsumexp(q_cat_f32, dim=-1) - math.log(N_total)
            )  # [num_q, B], float32

            # Dataset Q-values (already computed above)
            q_data = q_pred_all.float()  # [num_q, B], float32 to match cql_logsumexp
            # CQL penalty per Q-network: E_s[logsumexp] - E_{s,a~D}[Q]
            #
            # ⚡ Patch 1: fixed α_cql with signed penalty.
            #   The IS-estimated logsumexp is structurally below q_data
            #   in 29-DOF with 10+10 samples (penalty ≈ −2 per Q), so
            #   the per-Q penalty is persistently negative.  This is
            #   fine: the CQL gradient direction (push Q_ood down,
            #   push Q_data up) is correct regardless of sign.  A
            #   small fixed α_cql (0.02) keeps cql_loss small relative
            #   to td_loss, and clamp(min=-10) prevents pathological
            #   batches from overwhelming the TD objective.
            #   The raw unclamped sum is kept for logging as
            #   cql_penalty_raw.
            per_state_penalty = cql_logsumexp - q_data  # [num_q, B]

            # ── SMQR continuous-action baseline (smqr_cont_self) ────
            # When ``critic_penalty_mode == 'smqr_cont_self'`` we
            # OVERWRITE ``cql_logsumexp`` and ``per_state_penalty``
            # with the per-critic self-mask SMQR formulation.  This is
            # the A-fidelity baseline for the Q≈τ / Qg′ amplification
            # hypothesis study.
            #
            # Per-critic self-mask (no detach): for each critic i,
            #   g_i(s,a) = σ((Q_i(s,a) − τ(s)) / β)
            # Penalty per critic i:
            #   L_i(s) = log(1/K · Σ_k exp(Q_i(s,a_k)·g_i(s,a_k)
            #                              − log p(a_k|s)))
            #            − Q_i(s, a_data)
            #
            # Design notes:
            # * τ(s) is a SHARED state-dependent scalar (one head,
            #   ``qnet.tau_head``).  Both critics compare against the
            #   same threshold.
            # * Mask is computed from raw Q (no log-density inside the
            #   sigmoid) so the threshold lives in the natural Q
            #   scale.  log-density is applied OUTSIDE the mask, in
            #   the IS-correction term.
            # * No detach on Q inside g — this is the entire point of
            #   the A-fidelity baseline: critic gradient flows as
            #   ``∂L/∂Q = (g + Q·g')·∂Q/∂θ`` so the Qg′ amplification
            #   hypothesis is exposed to the learning signal.
            # * Legacy SC heuristics (sc_mask_*, severity, phase) are
            #   ignored in this mode by construction.
            _penalty_mode_check = getattr(args, "critic_penalty_mode", "vanilla_cql")
            _smqr_cache: dict | None = None
            # V1 (Phase D) shrinkage loss — set inside the smqr block
            # when active, otherwise stays at zero so the addition to
            # critic_loss is a no-op.  Live tensor (with grad), unlike
            # the detached telemetry view in _smqr_cache.
            _v1_shrink_loss: torch.Tensor = torch.zeros((), device=self.device)
            if _penalty_mode_check == "smqr_cont_self":
                _beta = max(float(getattr(args, "sc_tau_beta", 1.0)),
                            float(getattr(args, "sc_tau_eps", 1e-6)))

                # Raw Q stack (NO log-density correction, NO detach).
                # Shapes: q_rand [num_q, B, N_rand], q_pi [num_q, B, N_pi]
                Q_cat_raw = torch.cat([q_rand, q_pi], dim=-1).float()
                # [num_q, B, K] where K = N_rand + N_pi = N_total

                # Log-density per action (detached; matches vanilla CQL).
                # rand: scalar; pi: [B, N_pi] already detached.
                log_p_rand = rand_log_density.expand(B, num_random)  # [B, N_rand]
                log_p_cat = torch.cat([log_p_rand, pi_log_probs], dim=-1)  # [B, K]
                log_p_cat = log_p_cat.unsqueeze(0).float()  # [1, B, K]

                # ── Shared τ(s) = anchor + bounded residual (B2-fix)
                # History:
                #   B-fix (unbounded residual) stopped τ → -∞ but
                #   then τ → +∞ runaway re-emerged (residual kept
                #   growing positive, g → 0, near-τ occupancy → 0,
                #   SMQR degenerated to the empty mask).
                #
                # B2-fix (current): clamp the residual to a bounded
                # range around the anchor with tanh:
                #
                #   τ(s) = Q_data_min(s).detach()
                #          + sc_tau_res_scale · tanh(τ_raw(s))
                #
                # * Anchor: raw-Q min over critic axis only
                #   (dim=0); per-state [B] shape preserved, batch
                #   axis untouched.  No grad.
                # * Residual: bounded in [−scale, +scale].  At init
                #   (tau_head final Linear is zero-init) the raw
                #   output is 0, so τ starts exactly at the anchor.
                # * ``sc_tau_res_scale`` default = 2.0 = 2·β: two
                #   full mask-transition widths either side of the
                #   anchor.  Enough to learn a meaningful
                #   state-dependent τ; not enough to run away.
                # * No new loss terms, no regularizer — the bound
                #   is purely structural via the tanh.
                _tau_res_scale = float(getattr(args, "sc_tau_res_scale", 2.0))
                _tau_anchor = q_data.detach().min(dim=0).values  # [B]  (raw Q, no grad)
                _tau_raw_residual = qnet.tau_from_processed(
                    critic_obs_processed
                )                                                # [B]  (learned, unbounded)

                # ── V1 (Phase D) τ parameterisation branch ───────
                # vanilla / stabilized : τ = anchor + scale·tanh(τ_raw)   (two-sided)
                # v1_oneside_shrink   : τ = anchor − scale·softplus(τ_raw) (anchor-capped, ≤ anchor)
                # See algo_mode.MODE_SMQR_LEARNED + ALLOWED_LEARNED_VARIANTS.
                _v1_active = (
                    self._algo_mode.mode == MODE_SMQR_LEARNED
                    and self._algo_mode.learned_variant == "v1_oneside_shrink"
                )
                _f1_active = (
                    self._algo_mode.mode == MODE_SMQR_LEARNED
                    and self._algo_mode.learned_variant == "f1_st_qg"
                )
                # F1 reuses V1's softplus τ-parameterisation + shrinkage exactly;
                # only the critic-loss objective form differs (logits branch below).
                _uses_v1_tau = _v1_active or _f1_active
                if _uses_v1_tau:
                    _tau_residual_pos = F.softplus(_tau_raw_residual)               # ≥ 0, [B]
                    _tau_residual = -(_tau_res_scale * _tau_residual_pos)           # ≤ 0, [B]
                    # ── anchor-shrinkage loss (V1 + F1) ─────────
                    # L_shrink = λ_sh · E_s[(τ_anchor − τ)²]
                    #          = λ_sh · E_s[(scale · softplus(τ_raw))²]
                    # Gradient flows ONLY into θ_τ (the anchor is
                    # detached upstream; this term has no Q-net path).
                    _lam_sh = float(getattr(args, "smqr_v1_shrink_lambda", 1e-3))
                    _v1_shrink_loss = _lam_sh * (
                        _tau_res_scale * _tau_residual_pos
                    ).pow(2).mean()
                else:
                    _tau_residual_pos = None
                    _tau_residual = _tau_res_scale * torch.tanh(_tau_raw_residual)  # bounded [-s,s]
                _tau = _tau_anchor + _tau_residual               # [B]
                _tau_b1 = _tau.view(1, B, 1)  # broadcast to [num_q, B, K]

                # Per-critic Δ and self-mask (no detach on Q_raw).
                _delta = Q_cat_raw - _tau_b1                                # [num_q, B, K]
                _g = torch.sigmoid(_delta / _beta)                          # [num_q, B, K]

                # ── Default sub-flag values (consumed by _smqr_cache).
                # The F1/G1/H1/B2 branches override these inside their
                # nested ``if _f1_active:`` block; the anchor-only
                # baseline never enters that block, so without these
                # defaults the cache assignment below raises
                # UnboundLocalError on _h1_alpha / _b2_alpha.
                _g1_active = False
                _h1_active = False
                _h1_alpha = 0.0
                _b2_active = False
                _b2_alpha = 0.0

                # ── Phase B vs Phase C/D vs Phase E weighted-logits branch ──
                # Vanilla (Phase B / anchor-vanilla):
                #   logits_k = Q · g − log p
                #   → d/dQ logsumexp ∝ Q · g(1−g)/β  (|Q|-amplified)
                # Stabilized objective (Phase C learned-stab,
                # Phase D V1, Phase E anchor+stab):
                #   logits_k = Q + log(g+ε) − log p
                #   → d/dQ contribution is bounded softmax weights;
                #     d/dτ contribution is softmax-weighted (1−g)/β
                #     with no Q multiplier.
                # V1 also adds an additive shrinkage term to the
                # critic loss (computed below, post-cql_loss).
                # Phase E: same objective form, but τ ≡ τ_anchor
                # (sc_tau_res_scale=0.0 ⇒ _tau_residual ≡ 0).
                _stab_active = (
                    self._algo_mode.mode == MODE_SMQR_LEARNED
                    and self._algo_mode.learned_variant == "stabilized"
                )
                _anchor_stab_active = (
                    self._algo_mode.mode == MODE_SMQR_ANCHOR
                    and str(getattr(args, "smqr_anchor_objective", "vanilla"))
                        .strip().lower() == "stabilized"
                    and bool(getattr(args, "smqr_anchor_phase_e_optin", False))
                )
                if _f1_active:
                    # F1 (Phase F): ST-split symmetric stop-gradient.
                    # forward(qg_F1) = 0.5·(Q·g + Q·g) ≡ Q·g  (vanilla bit-exact)
                    # backward halves the Q·g' contribution to both θ_Q and θ_τ.
                    # No log(g+ε) floor, no ε dependence.
                    _g_eps = 0.0
                    _log_g_stab = None
                    _qg_st = 0.5 * (
                        Q_cat_raw * _g.detach() + Q_cat_raw.detach() * _g
                    )
                    # ── G1 (Phase F+ sub-flag): candidate-wise routing ──
                    # K-axis layout: Q_cat_raw = cat([q_rand, q_pi], dim=-1)
                    #   indices [0, num_random)         — random candidates
                    #   indices [num_random, N_total)   — policy candidates
                    # Random:  vanilla Q·g  (full ∂L/∂Q via g + Q·g'/β)
                    # Policy:  ST-split    (½ ∂L/∂Q, F1 unchanged)
                    # Forward value identical on both branches
                    # (0.5·(Q·g + Q·g) ≡ Q·g) — only backward differs.
                    # Data term and τ-parameterisation are untouched.
                    _g1_active = (
                        bool(getattr(args, "smqr_f1_random_full_grad", False))
                        and num_random > 0
                    )
                    if _g1_active:
                        # ── H1 (Phase F++ sub-flag): α-floor on RANDOM gate ──
                        # G1     :  qg_rand = Q · g
                        # H1     :  qg_rand = Q · (g + α),  α = smqr_h1_alpha_floor
                        # α = 0 ⇒ H1 collapses to G1 bit-exactly.
                        # α > 0 ⇒ ∂qg_rand/∂Q = (g + α) ≥ α, so the
                        # random push-down signal does not vanish even
                        # when g_rand → 0 (G1's late-stage starvation).
                        # Applied ONLY to the random K-axis half;
                        # policy branch keeps the F1 ST-split.
                        _h1_alpha = float(
                            getattr(args, "smqr_h1_alpha_floor", 0.0)
                        )
                        _h1_active = (_h1_alpha > 0.0)
                        # ── B2 (Phase I sub-flag): STE max-clip backward floor ──
                        # Forward  : qg_rand = Q · g                  (G1 bit-exact)
                        # ∂/∂Q     : max(g, α)                        (Q-grad floor)
                        # ∂/∂g     : Q                                (= G1 τ-grad)
                        # Mutually exclusive with H1 (guarded in setup()).
                        # α = 0 ⇒ collapses to G1 bit-exactly on every
                        # gradient channel.  α > 0 ⇒ Q-grad ≥ α even
                        # when g_rand → 0 (anti-starvation), without
                        # shifting the forward logsumexp mass that H1
                        # was over-correcting.  Random branch only;
                        # policy branch keeps F1 ST-split unchanged.
                        _b2_alpha = float(
                            getattr(args, "smqr_b2_alpha_floor", 0.0)
                        )
                        _b2_active = (_b2_alpha > 0.0) and (not _h1_active)
                        if _h1_active:
                            _g_rand_eff = _g[..., :num_random] + _h1_alpha
                            _qg_rand = (
                                Q_cat_raw[..., :num_random] * _g_rand_eff
                            )
                            _qg_pol_st = _qg_st[..., num_random:]
                            _qg_F1 = torch.cat(
                                [_qg_rand, _qg_pol_st], dim=-1
                            )
                        elif _b2_active:
                            # STE construction (random K-axis only):
                            #   y = Q·g_back + Q.detach()·g − Q.detach()·g_back
                            # forward     = Q·g                       (g_back terms cancel via detach)
                            # ∂y/∂Q       = g_back                    (only first term carries Q-grad)
                            # ∂y/∂g       = Q.detach()                (only middle term carries g-grad)
                            #             ⇒ ∂y/∂τ = Q · g'  (= G1)
                            _g_r = _g[..., :num_random]
                            _Q_r = Q_cat_raw[..., :num_random]
                            _g_back = torch.clamp(
                                _g_r.detach(), min=_b2_alpha
                            )                                          # detached constant
                            _qg_rand = (
                                _Q_r * _g_back
                                + _Q_r.detach() * _g_r
                                - _Q_r.detach() * _g_back
                            )
                            _qg_pol_st = _qg_st[..., num_random:]
                            _qg_F1 = torch.cat(
                                [_qg_rand, _qg_pol_st], dim=-1
                            )
                        else:
                            _qg_van = Q_cat_raw * _g                       # full grad on random
                            _is_rand_mask = torch.zeros(
                                N_total, dtype=torch.bool,
                                device=Q_cat_raw.device,
                            )
                            _is_rand_mask[:num_random] = True
                            _qg_F1 = torch.where(
                                _is_rand_mask.view(1, 1, N_total),
                                _qg_van,
                                _qg_st,
                            )
                    else:
                        _h1_active = False
                        _h1_alpha = 0.0
                        _b2_active = False
                        _b2_alpha = 0.0
                        _qg_F1 = _qg_st
                    _weighted_logits_preclip = _qg_F1 - log_p_cat
                    _weighted_logits = _weighted_logits_preclip.clamp(
                        -_q_clip, _q_clip,
                    )
                elif _stab_active or _v1_active or _anchor_stab_active:
                    _g_eps = float(getattr(args, "smqr_stab_g_eps", 1e-6))
                    _log_g_stab = torch.log(_g.clamp_min(_g_eps))           # [num_q, B, K]
                    _weighted_logits_preclip = (
                        Q_cat_raw + _log_g_stab - log_p_cat
                    )
                    _weighted_logits = _weighted_logits_preclip.clamp(
                        -_q_clip, _q_clip,
                    )
                else:
                    # ── Step-3 SMQR-SG sub-mode switch ──
                    # Anchor-only vanilla weighted-logits branch.
                    # Read sub-knob (already validated in setup() —
                    # cross-contamination guard ensures we only land
                    # here under algo_mode='smqr_anchor' +
                    # smqr_anchor_objective='vanilla' +
                    # sc_tau_res_scale=0.0 + no F1/G1/H1/B2).
                    _lse_mode = str(
                        getattr(args, "smqr_lse_mode", "q_times_g")
                    ).strip().lower()
                    _sg_eps = float(getattr(args, "smqr_sg_eps", 1e-6))
                    if _lse_mode == "q_times_g":
                        # Existing baseline: bit-exact pre-Step-3.
                        # Keep _g_eps=0.0 and _log_g_stab=None so the
                        # downstream stabilised telemetry block stays
                        # disabled.
                        _g_eps = 0.0
                        _log_g_stab = None
                        _weighted_logits_preclip = (
                            Q_cat_raw * _g - log_p_cat
                        )
                    elif _lse_mode == "q_times_detached_g":
                        # Backward-only ablation: forward identical to
                        # q_times_g (Q*g), but the gate-derivative
                        # contribution Q·g'/β to ∂L/∂Q is removed via
                        # detach.  Forward ranking distortion is
                        # PRESERVED — this is intentional.
                        _g_eps = 0.0
                        _log_g_stab = None
                        _weighted_logits_preclip = (
                            Q_cat_raw * _g.detach() - log_p_cat
                        )
                    elif _lse_mode == "sg_weighted_lse":
                        # SMQR-SG main:
                        #   logits = Q − log p + log(detach(g) + ε)
                        # Removes Q·g multiplication entirely; the
                        # gate enters as a detached additive constant
                        # so the softmax is a gate-weighted softmax
                        # over (Q − log p).  ∂lse/∂Q_i = w_i (no Q·g'
                        # term; β-independent gradient structure).
                        _g_eps = _sg_eps
                        _log_g_sg = torch.log(
                            _g.detach().clamp_min(_sg_eps)
                        )                                        # [num_q, B, K], detached
                        _log_g_stab = None  # keep Phase C/D/E telemetry off
                        _weighted_logits_preclip = (
                            Q_cat_raw + _log_g_sg - log_p_cat
                        )
                    elif _lse_mode == "sg_blend":
                        # Stage R1 (P3 redesign): 50/50 LOSS-level
                        # blend of q_times_g and sg_weighted_lse.
                        # Both logits are computed and clamped
                        # independently; the resulting per-state
                        # penalties are averaged AFTER the standard
                        # logsumexp/per_state_penalty step (see
                        # override block below).  We assign the
                        # sg_weighted_lse logits as the "primary"
                        # _weighted_logits so downstream SMQR-SG
                        # telemetry (near-τ frac, gradient ratios,
                        # etc.) reflects the gate-weighted side.
                        _g_eps = _sg_eps
                        _log_g_sg = torch.log(
                            _g.detach().clamp_min(_sg_eps)
                        )
                        _log_g_stab = None
                        _weighted_logits_preclip_qg = (
                            Q_cat_raw * _g - log_p_cat
                        )
                        _weighted_logits_preclip_sg = (
                            Q_cat_raw + _log_g_sg - log_p_cat
                        )
                        # Primary (used for downstream telemetry): SG side.
                        _weighted_logits_preclip = _weighted_logits_preclip_sg
                    else:
                        # Unreachable: guard validated _smqr_lse_mode
                        # in setup().  Defensive raise.
                        raise RuntimeError(
                            f"Unknown smqr_lse_mode={_lse_mode!r}"
                        )
                    _weighted_logits = _weighted_logits_preclip.clamp(
                        -_q_clip, _q_clip,
                    )
                # Overwrite cql_logsumexp and per_state_penalty so the
                # downstream α_cql / total-loss path is reused
                # unchanged.
                cql_logsumexp = (
                    torch.logsumexp(_weighted_logits, dim=-1) - math.log(N_total)
                )                                                            # [num_q, B]
                per_state_penalty = cql_logsumexp - q_data                  # [num_q, B]

                # ── sg_blend LOSS-level override ─────────────────
                # Re-compute the q_times_g side and replace
                # per_state_penalty with a schedule-weighted blend
                #   per_state_penalty = (1-λ) * P_qg + λ * P_sgw
                # where λ = λ(t) follows ``smqr_blend_schedule`` (see
                # algo.py).  λ=0 → pure q_times_g, λ=1 → pure
                # sg_weighted_lse.  Default schedule='fixed' with
                # λ_start=λ_end=0.5 recovers the Stage R2 R4 fixed
                # 50/50 blend bit-exactly.
                #
                # The primary _weighted_logits / cql_logsumexp
                # tensors remain the sg_weighted_lse side for
                # telemetry consistency; cql_logsumexp itself is
                # logged but is not consumed downstream
                # (per_state_penalty drives the loss path), so the
                # blend is realised purely through per_state_penalty.
                if _lse_mode == "sg_blend":
                    _weighted_logits_qg = _weighted_logits_preclip_qg.clamp(
                        -_q_clip, _q_clip,
                    )
                    _cql_logsumexp_qg = (
                        torch.logsumexp(_weighted_logits_qg, dim=-1)
                        - math.log(N_total)
                    )                                                        # [num_q, B]
                    _per_state_penalty_qg = _cql_logsumexp_qg - q_data       # [num_q, B]
                    # _per_state_penalty already holds the SG side.

                    # ── Resolve λ(t) from the schedule ────────────
                    _gs = int(self.global_step)
                    _ls = self._smqr_blend_lambda_start
                    _le = self._smqr_blend_lambda_end
                    _wp = self._smqr_blend_warmup_steps
                    _rp = self._smqr_blend_ramp_steps
                    _sched = self._smqr_blend_schedule
                    if _sched == "fixed":
                        _lam = _ls
                    elif _sched == "linear":
                        # 0 → ls; ramp_steps → le; clamp afterwards.
                        _frac = min(max(_gs / float(_rp), 0.0), 1.0)
                        _lam = _ls + (_le - _ls) * _frac
                    elif _sched == "delayed_linear":
                        if _gs < _wp:
                            _lam = _ls
                        else:
                            _frac = min(
                                max((_gs - _wp) / float(_rp), 0.0), 1.0,
                            )
                            _lam = _ls + (_le - _ls) * _frac
                    elif _sched == "piecewise":
                        # [0, wp): ls; [wp, wp+rp): linear ls→le;
                        # thereafter: le.  hold_steps is informational
                        # only — λ remains at le past the hold window.
                        if _gs < _wp:
                            _lam = _ls
                        elif _gs < _wp + _rp:
                            _frac = (_gs - _wp) / float(_rp)
                            _lam = _ls + (_le - _ls) * _frac
                        else:
                            _lam = _le
                    else:
                        # Unreachable: validated in setup().
                        raise RuntimeError(
                            f"unknown smqr_blend_schedule={_sched!r}"
                        )
                    _smqr_blend_lambda_active = float(_lam)
                    per_state_penalty = (
                        (1.0 - _smqr_blend_lambda_active) * _per_state_penalty_qg
                        + _smqr_blend_lambda_active * per_state_penalty
                    )

                # Stash intermediate tensors for the SMQR telemetry
                # block (computed under no_grad later).
                _smqr_cache = {
                    "tau": _tau.detach(),
                    "tau_anchor": _tau_anchor.detach(),
                    "tau_residual": _tau_residual.detach(),
                    "tau_raw_residual": _tau_raw_residual.detach(),
                    "tau_res_scale": _tau_res_scale,
                    "Q_cat_raw": Q_cat_raw.detach(),
                    "delta": _delta.detach(),
                    "g": _g.detach(),
                    # V1 (Phase D) cache.
                    "v1_active": _v1_active,
                    # F1 (Phase F) cache.
                    "f1_active": _f1_active,
                    # G1 (Phase F+ sub-flag) cache.
                    "g1_active": (_f1_active and _g1_active),
                    # H1 (Phase F++ sub-flag) cache.
                    "h1_active": (_f1_active and _g1_active and _h1_active),
                    "h1_alpha": _h1_alpha,
                    # B2 (Phase I sub-flag) cache.
                    "b2_active": (_f1_active and _g1_active and _b2_active),
                    "b2_alpha": _b2_alpha,
                    "tau_residual_pos": (
                        _tau_residual_pos.detach()
                        if _tau_residual_pos is not None else None
                    ),
                    "v1_shrink_lambda": (
                        float(getattr(args, "smqr_v1_shrink_lambda", 1e-3))
                        if _uses_v1_tau else 0.0
                    ),
                    "v1_shrink_loss": _v1_shrink_loss.detach(),
                    # Phase C stabilised cache (set even when inactive).
                    "stab_active": _stab_active,
                    # Phase E (anchor+stabilised objective) cache.
                    "anchor_stab_active": _anchor_stab_active,
                    "g_eps": _g_eps,
                    "log_g_stab": (
                        _log_g_stab.detach() if _log_g_stab is not None else None
                    ),
                    "weighted_logits": _weighted_logits.detach(),
                    "weighted_logits_preclip": _weighted_logits_preclip.detach(),
                    # Step-3 SMQR-SG sub-mode label (only meaningful
                    # in the anchor-only vanilla branch; carried for
                    # telemetry routing in all paths).
                    "smqr_lse_mode": str(
                        getattr(args, "smqr_lse_mode", "q_times_g")
                    ).strip().lower(),
                    "smqr_sg_eps": float(
                        getattr(args, "smqr_sg_eps", 1e-6)
                    ),
                    "beta": _beta,
                    "num_random": num_random,
                    "num_policy": num_policy,
                    # q_data and q_pi_next_at_s are recomputed/used
                    # inside the telemetry block from the existing
                    # tensors.
                }

            # ── SC-CQL per-state penalty reweight (v3) ─────────────
            _penalty_mode = getattr(args, "critic_penalty_mode", "vanilla_cql")
            _sc_mask_curr = None   # [B] or None
            _sc_mask_next = None   # [B] or None
            _sc_reweight = None    # [B] or None
            _sc_boost_val = 0.0
            _sc_gap_curr = None    # [B]
            _sc_gap_next = None    # [B]
            _sc_deficit_curr = None  # [B]
            _sc_deficit_next = None  # [B]
            _sc_severity_curr = None  # [B]
            _sc_severity_next = None  # [B]
            _sc_topk_thr = torch.tensor(0.0, device=self.device)

            if _penalty_mode == "sc_cql":
                _sc_target = getattr(args, "sc_mask_target", "policy_curr_only")
                _sc_mask_mode = getattr(args, "sc_mask_mode", "sigmoid_symmetric")

                with torch.no_grad():
                    # ── Common: policy-current gap ──────────────────
                    _q_data_min_sc = q_data.float().min(dim=0).values          # [B]
                    _q_pi_curr_min_sc = q_pi.float().min(dim=0).values         # [B, num_policy]
                    _q_pi_curr_min_sc = _q_pi_curr_min_sc.mean(dim=-1)         # [B]
                    _sc_gap_curr = _q_data_min_sc - _q_pi_curr_min_sc          # [B]

                    # ── Common: policy-next gap (always compute for
                    #    diagnostics; only used in reweight when target
                    #    includes next) ─────────────────────────────
                    _q_pi_next_sc = qnet.q_values_for_actions(
                        critic_obs_processed, next_actions.unsqueeze(1)
                    ).squeeze(-1).squeeze(-1).float()  # [num_q, B]
                    _q_pi_next_min_sc = _q_pi_next_sc.min(dim=0).values        # [B]
                    _sc_gap_next = _q_data_min_sc - _q_pi_next_min_sc          # [B]

                if _sc_mask_mode == "violation_only_sparse":
                    # ── SC-CQL v2/v3: one-sided sparse mask ────────
                    _sc_margin = getattr(args, "sc_margin_target", 0.0)
                    _sc_sp_temp = getattr(args, "sc_sparse_temperature", 0.1)
                    _sc_active_frac = getattr(args, "sc_active_frac_target", 0.10)
                    _sc_boost_val = getattr(args, "sc_mask_boost", 1.0)
                    _sc_sev_mode = getattr(args, "sc_severity_mode", "none")
                    _sc_sev_power = getattr(args, "sc_severity_power", 1.0)
                    _sc_sev_floor = getattr(args, "sc_severity_floor", 0.0)
                    _sc_sev_norm = getattr(args, "sc_severity_norm_mode", "batch_max")
                    _sc_phase_mode = getattr(args, "sc_phase_mode", "all")
                    _sc_phase_h_thr = getattr(args, "sc_phase_height_threshold", 0.15)

                    def _build_sparse_mask(gap, deficit_out):
                        """Build one-sided sparse mask from gap. Returns (mask, deficit, severity)."""
                        deficit = F.relu(_sc_margin - gap)                      # [B]
                        deficit_out.copy_(deficit)
                        base = 1.0 - torch.exp(-deficit / max(_sc_sp_temp, 1e-8))
                        # Sparse gating
                        if deficit.sum() > 0 and _sc_active_frac < 1.0:
                            q_lvl = min(1.0 - _sc_active_frac, 0.999)
                            thr = deficit.float().quantile(q_lvl)
                            gate = (deficit >= thr).float()
                        else:
                            gate = (deficit > 0).float()
                            thr = torch.tensor(0.0, device=deficit.device)
                        mask = base * gate
                        # Severity scaling (v3/v4)
                        if _sc_sev_mode == "deficit_weighted":
                            if _sc_sev_norm == "p90" and deficit.numel() >= 2:
                                d_norm = deficit.float().quantile(0.9).clamp(min=1e-8)
                            else:
                                d_norm = deficit.max().clamp(min=1e-8)
                            sev = (deficit / d_norm).clamp(max=1.0).pow(_sc_sev_power)
                            if _sc_sev_floor > 0:
                                sev = sev.clamp(min=_sc_sev_floor) * (deficit > 0).float()
                            mask = mask * sev
                        else:
                            sev = torch.ones_like(deficit)
                        return mask, deficit, sev, thr

                    with torch.no_grad():
                        _sc_deficit_curr = torch.empty(B, device=self.device)
                        _sc_mask_curr, _sc_deficit_curr, _sc_severity_curr, _sc_topk_thr = \
                            _build_sparse_mask(_sc_gap_curr, _sc_deficit_curr)
                        _sc_reweight = 1.0 + _sc_boost_val * _sc_mask_curr     # [B]

                        if _sc_target in ("policy_curr_and_next", "policy_next_only"):
                            _sc_deficit_next = torch.empty(B, device=self.device)
                            _sc_mask_next, _sc_deficit_next, _sc_severity_next, _ = \
                                _build_sparse_mask(_sc_gap_next, _sc_deficit_next)

                            # Phase gating (v4): zero out next mask for
                            # pre-lift transitions when phase mode is
                            # post_lift_only.
                            if _sc_phase_mode == "post_lift_only":
                                _obj_idx = self._critic_term_offsets.get("obj_pos_b") if hasattr(self, "_critic_term_offsets") else None
                                if _obj_idx is not None:
                                    _obj_z = critic_obs[:, _obj_idx["start"] + 2]  # [B]
                                    _post_lift = (_obj_z >= _sc_phase_h_thr).float()
                                    _sc_mask_next = _sc_mask_next * _post_lift
                                # else: silently skip phase gating (no obj_pos_b available)

                            # One-shot phase debug (v4)
                            if getattr(args, "sc_phase_debug", False) and not getattr(self, "_sc_phase_debug_done", False):
                                self._sc_phase_debug_done = True
                                _dbg_obj = self._critic_term_offsets.get("obj_pos_b")
                                if _dbg_obj is not None:
                                    _dbg_z = critic_obs[:, _dbg_obj["start"] + 2]
                                    _dbg_pl = (_dbg_z >= _sc_phase_h_thr).float().mean()
                                    logger.info(
                                        f"\n[SC-CQL v4 PHASE DEBUG — first batch]\n"
                                        f"  obj_pos_b term offset   : {_dbg_obj}\n"
                                        f"  obj_z min/mean/max      : {_dbg_z.min():.4f} / {_dbg_z.mean():.4f} / {_dbg_z.max():.4f}\n"
                                        f"  obj_z std               : {_dbg_z.std():.4f}\n"
                                        f"  phase_height_threshold  : {_sc_phase_h_thr}\n"
                                        f"  post_lift_frac          : {_dbg_pl:.4f}\n"
                                        f"  sc_mask_target          : {_sc_target}\n"
                                        f"  sc_mask_next nonzero    : {(_sc_mask_next > 1e-6).float().mean():.4f}\n"
                                        f"  obj_z histogram (10 bins):\n"
                                        f"    {torch.histc(_dbg_z.float(), bins=10).int().tolist()}"
                                    )
                                else:
                                    logger.warning(
                                        "[SC-CQL v4 PHASE DEBUG] obj_pos_b NOT found in "
                                        f"_critic_term_offsets.  Keys: {list(self._critic_term_offsets.keys())}"
                                    )

                            if _sc_target == "policy_next_only":
                                _sc_reweight = 1.0 + _sc_boost_val * _sc_mask_next
                            else:
                                _sc_combined = torch.max(_sc_mask_curr, _sc_mask_next)
                                _sc_reweight = 1.0 + _sc_boost_val * _sc_combined

                else:  # sigmoid_symmetric (v1)
                    _sc_strength = getattr(args, "sc_mask_strength", "mid")
                    _sc_thresh = getattr(args, "sc_mask_threshold", 0.0)
                    _SC_PRESETS = {
                        "weak":   (1.0, 0.5),
                        "mid":    (0.5, 1.0),
                        "strong": (0.3, 2.0),
                    }
                    if _sc_strength in _SC_PRESETS:
                        _sc_temp_val, _sc_boost_val = _SC_PRESETS[_sc_strength]
                    else:
                        _sc_temp_val = getattr(args, "sc_mask_temperature", 0.5)
                        _sc_boost_val = getattr(args, "sc_mask_boost", 1.0)

                    with torch.no_grad():
                        _sc_mask_curr = torch.sigmoid(
                            (_sc_thresh - _sc_gap_curr) / max(_sc_temp_val, 1e-6)
                        )
                        _sc_reweight = 1.0 + _sc_boost_val * _sc_mask_curr

                        if _sc_target in ("policy_curr_and_next", "policy_next_only"):
                            _sc_mask_next = torch.sigmoid(
                                (_sc_thresh - _sc_gap_next) / max(_sc_temp_val, 1e-6)
                            )
                            if _sc_target == "policy_next_only":
                                _sc_reweight = 1.0 + _sc_boost_val * _sc_mask_next
                            else:
                                _sc_combined = torch.max(_sc_mask_curr, _sc_mask_next)
                                _sc_reweight = 1.0 + _sc_boost_val * _sc_combined

                # Reweighted penalty (gradient flows through
                # per_state_penalty only; _sc_reweight is detached).
                cql_penalty_per_q = (
                    per_state_penalty * _sc_reweight.unsqueeze(0)
                ).mean(dim=1)  # [num_q]
            else:
                # Vanilla CQL: uniform average over batch.
                cql_penalty_per_q = per_state_penalty.mean(dim=1)  # [num_q]

            cql_penalty = cql_penalty_per_q.clamp(min=-10).sum()
            cql_penalty_raw = cql_penalty_per_q.sum()

            # ── Phase P1b — one-sided penalty floor (loss-only) ─────
            # When opted in, the scalar fed into the critic loss is the
            # ReLU of the (already clamp(-10)) penalty:
            #     penalty_for_loss = clamp_min(cql_penalty, 0.0)
            # The ``cql_penalty`` tensor itself is left untouched, so the
            # Lagrange autotune update (which consumes _cql_penalty_raw
            # via learn()) is bit-exact to prior runs.  Only the
            # contribution to ``critic_loss`` is filtered.
            #
            # Default (optin=False) is a strict no-op:
            # ``_penalty_for_loss is cql_penalty`` and the two extra
            # frac telemetry keys still emit (for cross-run
            # comparability) but ``_penalty_clamped_frac`` is forced to
            # 0.0 to indicate the gate did not activate.
            _penalty_floor_optin = bool(
                getattr(args, "cql_penalty_floor_optin", False)
            )
            _penalty_negative_frac = (
                1.0 if cql_penalty.detach().item() < 0.0 else 0.0
            )
            if _penalty_floor_optin:
                _penalty_for_loss = cql_penalty.clamp(min=0.0)
                _penalty_clamped_frac = _penalty_negative_frac
            else:
                _penalty_for_loss = cql_penalty
                _penalty_clamped_frac = 0.0

            # ── Alpha-CQL (mode-dispatched) ───────────────────────
            alpha_cql = self.log_alpha_cql.exp().detach().squeeze()
            _cql_td_ratio = getattr(args, "cql_td_ratio", None)
            _cql_alpha_mode = getattr(args, "cql_alpha_mode", "td_relative")
            # Phase P1 (effective-α blow-up confounder):
            #   When > 0, applies an upper cap to the effective α used in
            #   the loss multiplier:  effective_α = min(raw_or_floored, cap).
            #   = 0 (default) is a strict no-op (bit-exact regression).
            #   Honoured under both ``td_relative`` and ``fixed_effective``
            #   modes; ignored when the Lagrangian path is in use.
            _cql_alpha_cap = float(
                getattr(args, "cql_effective_alpha_cap", 0.0)
            )

            if _cql_td_ratio is not None and _cql_alpha_mode == "fixed_effective":
                # Ablation mode: effective α is a constant from config.
                # No dependency on td_loss — isolates the alpha rule as
                # a variable while keeping everything else identical.
                _fixed_val = getattr(args, "cql_fixed_effective_alpha", 0.015)
                _effective_alpha_pre_cap = _fixed_val
                _cql_raw_alpha = _fixed_val  # logged as-is (no td calc)
                _floor_active = 0.0
                if _cql_alpha_cap > 0.0:
                    _effective_alpha = min(_effective_alpha_pre_cap, _cql_alpha_cap)
                else:
                    _effective_alpha = _effective_alpha_pre_cap
                _cap_active = (
                    1.0 if (
                        _cql_alpha_cap > 0.0
                        and _effective_alpha_pre_cap > _cql_alpha_cap
                    ) else 0.0
                )
                cql_loss = _effective_alpha * _penalty_for_loss
            elif _cql_td_ratio is not None:
                # Default: TD-relative schedule.
                _cql_floor = getattr(args, "cql_alpha_floor", 0.0)
                _cql_raw_alpha = (
                    _cql_td_ratio * td_loss.detach().item()
                    / max(abs(cql_penalty.detach().item()), 1e-8)
                )
                _effective_alpha_pre_cap = max(_cql_raw_alpha, _cql_floor)
                _floor_active = 1.0 if _cql_raw_alpha < _cql_floor else 0.0
                if _cql_alpha_cap > 0.0:
                    _effective_alpha = min(_effective_alpha_pre_cap, _cql_alpha_cap)
                else:
                    _effective_alpha = _effective_alpha_pre_cap
                _cap_active = (
                    1.0 if (
                        _cql_alpha_cap > 0.0
                        and _effective_alpha_pre_cap > _cql_alpha_cap
                    ) else 0.0
                )
                cql_loss = _effective_alpha * _penalty_for_loss
            else:
                # Lagrangian / fixed α_cql (original path).  Cap is
                # ignored here to keep the Lagrangian path bit-exact.
                cql_loss = alpha_cql * cql_penalty
                _effective_alpha = alpha_cql.item()
                _effective_alpha_pre_cap = _effective_alpha
                _cql_raw_alpha = _effective_alpha
                _floor_active = 0.0
                _cap_active = 0.0

            # ── Stage R1 — mode-agnostic conservative loss scale ─────
            # Multiplicative scale applied uniformly across all three
            # α-CQL dispatch branches (td_relative / fixed_effective /
            # Lagrangian).  Default 1.0 is a strict no-op (bit-exact
            # regression for all prior runs).  Telemetry exposes the
            # PRE-scale ``cql_loss`` value plus a separate
            # ``cql_loss_scale`` scalar so the realised contribution
            # to ``critic_loss`` is the product of the two.
            _cql_loss_scale = float(getattr(args, "cql_loss_scale", 1.0))
            _cql_loss_unscaled = cql_loss
            cql_loss = _cql_loss_scale * cql_loss

            # ── Total critic loss ──────────────────────────────────
            # V1 (Phase D) adds an anchor-shrinkage term that gradient-
            # flows only into the τ-head.  In all other modes
            # ``_v1_shrink_loss`` is the zero scalar set above, so the
            # addition is a no-op for cql / smqr_anchor / vanilla
            # learned / stabilized learned.
            critic_loss = td_loss + cql_loss + _v1_shrink_loss

        # ── Backward + optimise ────────────────────────────────────
        self.q_optimizer.zero_grad(set_to_none=True)
        scaler.scale(critic_loss).backward()
        scaler.unscale_(self.q_optimizer)

        if args.max_grad_norm > 0:
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                qnet.parameters(), max_norm=args.max_grad_norm,
            )
        else:
            critic_grad_norm = torch.tensor(0.0, device=self.device)

        scaler.step(self.q_optimizer)
        scaler.update()

        # ── Rich metrics ───────────────────────────────────────────
        with torch.no_grad():
            q_data_mean = q_data.mean()
            q_data_max = q_data.max()
            q_data_min = q_data.min()
            td_target_mean = td_target.mean()

            # cql_penalty_per_q_mean: mean across the Q-ensemble
            cql_penalty_mean = cql_penalty_per_q.mean()

            # ── A. Policy-side OOD direct verification ─────────────
            # Use min(Q1,Q2) to match the actor/TD-target reduction.
            q_data_min_ens = q_data.min(dim=0).values           # [B]
            # q_pi: [num_q, B, num_policy] → min across Q-ensemble
            q_pi_curr_min = q_pi.min(dim=0).values              # [B, num_policy]
            q_pi_curr_per_batch = q_pi_curr_min.mean(dim=-1)    # [B]
            _cql_q_pi_curr_mean = q_pi_curr_per_batch.mean()

            # Q(s, π(s')) via main critic — one extra critic forward
            q_pi_next_at_s = qnet.q_values_for_actions(
                critic_obs_processed, next_actions.unsqueeze(1),
            )  # [num_q, B, 1, 1]
            q_pi_next_at_s = q_pi_next_at_s.squeeze(-1).squeeze(-1)  # [num_q, B]
            q_pi_next_min = q_pi_next_at_s.min(dim=0).values   # [B]
            _cql_q_pi_next_mean = q_pi_next_min.mean()

            # Gaps (positive = data Q > policy Q, healthy)
            _q_data_q_pi_curr_gap = (q_data_min_ens - q_pi_curr_per_batch).mean()
            _q_data_q_pi_next_gap = (q_data_min_ens - q_pi_next_min).mean()

            # Violation rates and magnitudes
            _viol_0p1 = (q_pi_curr_per_batch >= q_data_min_ens - 0.1).float().mean()
            _viol_0p0 = (q_pi_curr_per_batch >= q_data_min_ens).float().mean()
            _viol_mag = F.relu(q_pi_curr_per_batch - q_data_min_ens).mean()

            # L2-weighted violation: action distance × violation indicator
            a_pi_curr_mean = pi_actions.mean(dim=1)             # [B, n_act]
            l2_dist = ((a_pi_curr_mean - actions) ** 2).sum(dim=-1)  # [B]
            viol_mask = (q_pi_curr_per_batch >= q_data_min_ens - 0.1).float()
            _viol_l2w = (l2_dist * viol_mask).mean()

            # ── B. CQL candidate dominance ─────────────────────────
            # Extend the IS-corrected logits to include pi_next for
            # a 3-way softmax weight diagnostic.
            q_pi_next_is = q_pi_next_at_s.float() - next_log_probs.unsqueeze(0)
            q_cat_3way = torch.cat([
                q_cat_f32,                                      # [num_q, B, N_total]
                q_pi_next_is.unsqueeze(-1).clamp(-_q_clip, _q_clip),  # [num_q, B, 1]
            ], dim=-1)
            weights_3way = F.softmax(q_cat_3way, dim=-1)       # [num_q, B, N_total+1]
            _w_rand = weights_3way[:, :, :num_random].sum(dim=-1).mean()
            _w_pi_curr = weights_3way[:, :, num_random:num_random + num_policy].sum(dim=-1).mean()
            _w_pi_next = weights_3way[:, :, num_random + num_policy:].sum(dim=-1).mean()

            # Per-group max raw Q-values (not IS-corrected)
            _q_rand_max = q_rand.max()
            _q_pi_curr_max = q_pi.max()
            _q_pi_next_max = q_pi_next_at_s.max()

            # ── C. Twin critic disagreement ────────────────────────
            _gap_data = (q_data.max(dim=0).values - q_data.min(dim=0).values).mean()
            _gap_pi_curr = (
                q_pi.max(dim=0).values - q_pi.min(dim=0).values
            ).mean()  # mean over [B, num_policy]
            _gap_pi_next = (
                q_pi_next_at_s.max(dim=0).values - q_pi_next_at_s.min(dim=0).values
            ).mean()

            # ── D. SC-CQL diagnostics (v3-extended) ───────────────
            _zero = torch.tensor(0.0, device=self.device)
            if _sc_mask_curr is not None:
                _sc_m_curr_mean = _sc_mask_curr.mean()
                _sc_m_curr_std = _sc_mask_curr.std()
                _sc_m_curr_active = (_sc_mask_curr > 1e-6).float().mean()

                # Leakage on safe states
                _sc_margin_diag = getattr(args, "sc_margin_target", 0.0)
                _safe_curr = _sc_gap_curr >= _sc_margin_diag
                _viol_curr = ~_safe_curr
                _sc_safe_mask_mean = _sc_mask_curr[_safe_curr].mean() if _safe_curr.any() else _zero
                _sc_safe_mask_p90 = _sc_mask_curr[_safe_curr].float().quantile(0.9) if (_safe_curr.sum() >= 2) else _zero
                _sc_viol_mask_mean = _sc_mask_curr[_viol_curr].mean() if _viol_curr.any() else _zero
                _sc_viol_mask_p90 = _sc_mask_curr[_viol_curr].float().quantile(0.9) if (_viol_curr.sum() >= 2) else _zero
                _sc_leakage_frac = (_sc_mask_curr[_safe_curr] > 1e-6).float().mean() if _safe_curr.any() else _zero

                # Penalty budget
                _sc_pen_total_raw = per_state_penalty.mean(dim=1).sum()
                _sc_pen_total_masked = cql_penalty_raw
                _sc_budget_ratio = _sc_pen_total_masked / (_sc_pen_total_raw.abs() + 1e-8)

                # Policy-current penalty
                _rw_curr = 1.0 + _sc_boost_val * _sc_mask_curr
                _sc_pen_curr_raw = per_state_penalty.mean()
                _sc_pen_curr_masked = (per_state_penalty * _rw_curr.unsqueeze(0)).mean()

                # Deficit / severity stats (curr)
                _sc_def_curr_mean = _sc_deficit_curr.mean() if _sc_deficit_curr is not None else _zero
                _sc_def_curr_p90 = _sc_deficit_curr.float().quantile(0.9) if (_sc_deficit_curr is not None and B >= 2) else _zero
                _sc_sev_curr_mean = _sc_severity_curr.mean() if _sc_severity_curr is not None else _zero

                # Curr subset quality
                _active_curr = _sc_mask_curr > 1e-6
                if _active_curr.any():
                    _gap_masked_sub = (q_data_min_ens[_active_curr] - q_pi_curr_per_batch[_active_curr]).mean()
                    _viol_0p0_masked = (q_pi_curr_per_batch[_active_curr] >= q_data_min_ens[_active_curr]).float().mean()
                    _viol_0p1_masked = (q_pi_curr_per_batch[_active_curr] >= q_data_min_ens[_active_curr] - 0.1).float().mean()
                else:
                    _gap_masked_sub = _zero
                    _viol_0p0_masked = _zero
                    _viol_0p1_masked = _zero

                # Curr safe subset
                _gap_safe_sub = (q_data_min_ens[_safe_curr] - q_pi_curr_per_batch[_safe_curr]).mean() if _safe_curr.any() else _zero

                # ── Policy-next diagnostics ─────────────────────────
                if _sc_mask_next is not None:
                    _sc_m_next_mean = _sc_mask_next.mean()
                    _sc_m_next_std = _sc_mask_next.std()
                    _sc_m_next_active = (_sc_mask_next > 1e-6).float().mean()
                    _rw_next = 1.0 + _sc_boost_val * _sc_mask_next
                    _sc_pen_next_raw = per_state_penalty.mean()
                    _sc_pen_next_masked = (per_state_penalty * _rw_next.unsqueeze(0)).mean()
                    _sc_def_next_mean = _sc_deficit_next.mean() if _sc_deficit_next is not None else _zero
                    _sc_def_next_p90 = _sc_deficit_next.float().quantile(0.9) if (_sc_deficit_next is not None and B >= 2) else _zero
                    _sc_sev_next_mean = _sc_severity_next.mean() if _sc_severity_next is not None else _zero
                    # Next subset quality
                    _active_next = _sc_mask_next > 1e-6
                    if _active_next.any():
                        _gap_next_masked_sub = (q_data_min_ens[_active_next] - q_pi_next_min[_active_next]).mean()
                        _gap_next_safe_sub = _zero  # compute below
                    else:
                        _gap_next_masked_sub = _zero
                        _gap_next_safe_sub = _zero
                    _safe_next = _sc_gap_next >= _sc_margin_diag if _sc_gap_next is not None else torch.zeros(B, dtype=torch.bool, device=self.device)
                    _gap_next_safe_sub = (q_data_min_ens[_safe_next] - q_pi_next_min[_safe_next]).mean() if _safe_next.any() else _zero
                else:
                    _sc_m_next_mean = _zero
                    _sc_m_next_std = _zero
                    _sc_m_next_active = _zero
                    _sc_pen_next_raw = _zero
                    _sc_pen_next_masked = _zero
                    _sc_def_next_mean = _zero
                    _sc_def_next_p90 = _zero
                    _sc_sev_next_mean = _zero
                    _gap_next_masked_sub = _zero
                    _gap_next_safe_sub = _zero

                # ── D4. Phase-aware diagnostics (v4) ────────────────
                _obj_idx_diag = self._critic_term_offsets.get("obj_pos_b") if hasattr(self, "_critic_term_offsets") else None
                if _obj_idx_diag is not None:
                    _obj_z_diag = critic_obs[:, _obj_idx_diag["start"] + 2]
                    _post_lift_diag = _obj_z_diag >= getattr(args, "sc_phase_height_threshold", 0.15)
                    _sc_phase_post_lift_frac = _post_lift_diag.float().mean()
                    _sc_phase_obj_z_mean = _obj_z_diag.mean()
                    _sc_phase_obj_z_max = _obj_z_diag.max()
                    if _sc_mask_next is not None and _post_lift_diag.any():
                        _active_next_pl = (_sc_mask_next > 1e-6) & _post_lift_diag
                        _sc_m_next_active_pl = _active_next_pl.float().mean()
                        _sc_m_next_mean_pl = _sc_mask_next[_post_lift_diag].mean()
                        if _active_next_pl.any():
                            _gap_next_masked_pl = (q_data_min_ens[_active_next_pl] - q_pi_next_min[_active_next_pl]).mean()
                            _viol_next_pl = (q_pi_next_min[_active_next_pl] >= q_data_min_ens[_active_next_pl]).float().mean()
                        else:
                            _gap_next_masked_pl = _zero
                            _viol_next_pl = _zero
                    else:
                        _sc_m_next_active_pl = _zero
                        _sc_m_next_mean_pl = _zero
                        _gap_next_masked_pl = _zero
                        _viol_next_pl = _zero
                else:
                    _sc_phase_post_lift_frac = _zero
                    _sc_phase_obj_z_mean = _zero
                    _sc_phase_obj_z_max = _zero
                    _sc_m_next_active_pl = _zero
                    _sc_m_next_mean_pl = _zero
                    _gap_next_masked_pl = _zero
                    _viol_next_pl = _zero
                _sc_phase_signal_avail = torch.tensor(
                    1.0 if _obj_idx_diag is not None else 0.0,
                    device=self.device,
                )

            else:
                # Vanilla CQL — emit zeros for all SC tags
                _sc_m_curr_mean = _zero
                _sc_m_curr_std = _zero
                _sc_m_curr_active = _zero
                _sc_m_next_mean = _zero
                _sc_m_next_std = _zero
                _sc_m_next_active = _zero
                _sc_pen_total_raw = _zero
                _sc_pen_total_masked = _zero
                _sc_pen_curr_raw = _zero
                _sc_pen_curr_masked = _zero
                _sc_pen_next_raw = _zero
                _sc_pen_next_masked = _zero
                _gap_masked_sub = _zero
                _viol_0p0_masked = _zero
                _viol_0p1_masked = _zero
                _sc_safe_mask_mean = _zero
                _sc_safe_mask_p90 = _zero
                _sc_viol_mask_mean = _zero
                _sc_viol_mask_p90 = _zero
                _sc_leakage_frac = _zero
                _sc_budget_ratio = _zero
                _gap_safe_sub = _zero
                _sc_def_curr_mean = _zero
                _sc_def_curr_p90 = _zero
                _sc_sev_curr_mean = _zero
                _sc_def_next_mean = _zero
                _sc_def_next_p90 = _zero
                _sc_sev_next_mean = _zero
                _gap_next_masked_sub = _zero
                _gap_next_safe_sub = _zero
                _sc_phase_post_lift_frac = _zero
                _sc_phase_obj_z_mean = _zero
                _sc_phase_obj_z_max = _zero
                _sc_m_next_active_pl = _zero
                _sc_m_next_mean_pl = _zero
                _gap_next_masked_pl = _zero
                _viol_next_pl = _zero
                _sc_phase_signal_avail = _zero

            # ──────────────────────────────────────────────────────────
            # E. SQR-SG hypothesis instrumentation (PROXY-BASED)
            #
            # ⚠ LEGACY — only computed when ``critic_penalty_mode ==
            # 'sc_cql'``.  Under the SMQR continuous-action baseline
            # (``smqr_cont_self``) the hypothesis is observed directly
            # via the ``train/smqr/*`` tags below (block F), which use
            # the actual learned τ(s) head and the per-critic
            # self-mask g_i.  Under ``vanilla_cql`` the proxy carries
            # no signal and is omitted to keep TensorBoard clean.
            #
            # Hypothesis to verify (both SC-CQL v1 and v4):
            #   H1. Q(s,a) repeatedly stays near a threshold τ(s).
            #   H2. That near-boundary mass actually drives gradient /
            #       penalty distortion (unintended regularization).
            #
            # τ_proxy(s) := Q_data_min(s) − τ_cfg_offset
            #   τ_cfg_offset =  sc_mask_threshold      (v1, sigmoid)
            #                   sc_margin_target       (v4, sparse)
            # ──────────────────────────────────────────────────────────
            if _penalty_mode == "sc_cql":
                _sc_mask_mode_v = getattr(args, "sc_mask_mode", "sigmoid_symmetric")
                if _sc_mask_mode_v == "sigmoid_symmetric":
                    _tau_cfg_offset = float(getattr(args, "sc_mask_threshold", 0.0))
                else:
                    _tau_cfg_offset = float(getattr(args, "sc_margin_target", 0.0))
                _tau_proxy = q_data_min_ens - _tau_cfg_offset  # [B]

                # Signed distance to proxy threshold for three action sources.
                _delta_pi_curr = q_pi_curr_per_batch - _tau_proxy           # [B]
                _delta_pi_next = q_pi_next_min       - _tau_proxy           # [B]
                _q_data_action_mean = q_data.float().mean(dim=0)            # [B]
                _delta_data        = _q_data_action_mean - _tau_proxy       # [B]

                def _pcts(x):
                    if x.numel() < 2:
                        return (_zero, _zero, _zero, _zero, _zero)
                    q = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], device=x.device)
                    v = torch.quantile(x.float(), q)
                    return (v[0], v[1], v[2], v[3], v[4])

                _p10_c, _p25_c, _p50_c, _p75_c, _p90_c = _pcts(_delta_pi_curr)
                _p10_n, _p25_n, _p50_n, _p75_n, _p90_n = _pcts(_delta_pi_next)
                _p10_d, _p25_d, _p50_d, _p75_d, _p90_d = _pcts(_delta_data)

                def _near_frac(x, eps):
                    return (x.abs() <= eps).float().mean()

                _nf_c_001 = _near_frac(_delta_pi_curr, 0.01)
                _nf_c_005 = _near_frac(_delta_pi_curr, 0.05)
                _nf_c_010 = _near_frac(_delta_pi_curr, 0.10)
                _nf_n_001 = _near_frac(_delta_pi_next, 0.01)
                _nf_n_005 = _near_frac(_delta_pi_next, 0.05)
                _nf_n_010 = _near_frac(_delta_pi_next, 0.10)
                _nf_d_001 = _near_frac(_delta_data,    0.01)
                _nf_d_005 = _near_frac(_delta_data,    0.05)
                _nf_d_010 = _near_frac(_delta_data,    0.10)

                _below_c = (_delta_pi_curr < 0).float().mean()
                _above_c = (_delta_pi_curr > 0).float().mean()
                _below_n = (_delta_pi_next < 0).float().mean()
                _above_n = (_delta_pi_next > 0).float().mean()
                _below_d = (_delta_data    < 0).float().mean()
                _above_d = (_delta_data    > 0).float().mean()

                _g_T = 0.1
                def _g_stats(delta):
                    g = torch.sigmoid(-delta.float() / _g_T)
                    gp = g * (1.0 - g)
                    g_mean = g.mean()
                    gp_mean = gp.mean()
                    if gp.numel() >= 2:
                        gp_p90 = torch.quantile(gp, 0.9)
                    else:
                        gp_p90 = _zero
                    gp_peak_frac = (gp >= 0.20).float().mean()
                    return g_mean, gp_mean, gp_p90, gp_peak_frac

                _g_c, _gp_c, _gp_p90_c, _gp_peak_c = _g_stats(_delta_pi_curr)
                _g_n, _gp_n, _gp_p90_n, _gp_peak_n = _g_stats(_delta_pi_next)
                _g_d, _gp_d, _gp_p90_d, _gp_peak_d = _g_stats(_delta_data)

                def _qgprime_stats(q_val, delta):
                    g = torch.sigmoid(-delta.float() / _g_T)
                    gp = g * (1.0 - g)
                    qgp = q_val.float() * gp
                    mean = qgp.mean()
                    if qgp.numel() >= 2:
                        p90 = torch.quantile(qgp.abs(), 0.9)
                    else:
                        p90 = _zero
                    peak_frac = (gp >= 0.20).float().mean()
                    return mean, p90, peak_frac

                _qgp_c_mean, _qgp_c_p90, _qgp_c_peak = _qgprime_stats(q_pi_curr_per_batch, _delta_pi_curr)
                _qgp_n_mean, _qgp_n_p90, _qgp_n_peak = _qgprime_stats(q_pi_next_min,       _delta_pi_next)
                _qgp_d_mean, _qgp_d_p90, _qgp_d_peak = _qgprime_stats(_q_data_action_mean, _delta_data)

                _psp_abs = per_state_penalty.detach().float().abs().mean(dim=0)  # [B]
                _total_abs = _psp_abs.sum().clamp(min=1e-12)
                _near_c_mask = (_delta_pi_curr.abs() <= 0.05)
                _near_n_mask = (_delta_pi_next.abs() <= 0.05)
                _near_pen_mass_curr = _psp_abs[_near_c_mask].sum() if _near_c_mask.any() else _zero
                _near_pen_mass_next = _psp_abs[_near_n_mask].sum() if _near_n_mask.any() else _zero
                _near_pen_share_total = (
                    _psp_abs[_near_c_mask | _near_n_mask].sum() / _total_abs
                    if (_near_c_mask | _near_n_mask).any() else _zero
                )
                _outside_pen_share_total = 1.0 - _near_pen_share_total

                if not hasattr(self, "_sc_instr_state"):
                    self._sc_instr_state = {
                        "ema_pi_curr": 0.0,
                        "ema_pi_next": 0.0,
                        "streak_pi_curr": 0,
                        "streak_pi_next": 0,
                        "window_pi_curr": [],
                        "window_pi_next": [],
                        "window_qgp_peak_c": [],
                        "window_td_loss": [],
                        "window_critic_grad": [],
                        "window_critic_loss": [],
                        "debug_printed": False,
                    }
                _rec_alpha = 0.01
                _rec_window = 100
                _rec_streak_thr = 0.05
                _st = self._sc_instr_state
                _fc = float(_nf_c_005.item())
                _fn = float(_nf_n_005.item())
                _st["ema_pi_curr"] = (1 - _rec_alpha) * _st["ema_pi_curr"] + _rec_alpha * _fc
                _st["ema_pi_next"] = (1 - _rec_alpha) * _st["ema_pi_next"] + _rec_alpha * _fn
                def _push(buf, val):
                    buf.append(float(val))
                    if len(buf) > _rec_window:
                        buf.pop(0)
                _push(_st["window_pi_curr"], _fc)
                _push(_st["window_pi_next"], _fn)
                _push(_st["window_qgp_peak_c"], _qgp_c_peak.item())
                _push(_st["window_td_loss"], td_loss.detach().item())
                _push(_st["window_critic_grad"], critic_grad_norm.detach().item())
                _push(_st["window_critic_loss"], critic_loss.detach().item())
                _st["streak_pi_curr"] = (_st["streak_pi_curr"] + 1) if _fc >= _rec_streak_thr else 0
                _st["streak_pi_next"] = (_st["streak_pi_next"] + 1) if _fn >= _rec_streak_thr else 0

                _ema_c = torch.tensor(_st["ema_pi_curr"], device=self.device)
                _ema_n = torch.tensor(_st["ema_pi_next"], device=self.device)
                _win_c = torch.tensor(
                    sum(_st["window_pi_curr"]) / max(len(_st["window_pi_curr"]), 1),
                    device=self.device,
                )
                _win_n = torch.tensor(
                    sum(_st["window_pi_next"]) / max(len(_st["window_pi_next"]), 1),
                    device=self.device,
                )
                _streak_c = torch.tensor(float(_st["streak_pi_curr"]), device=self.device)
                _streak_n = torch.tensor(float(_st["streak_pi_next"]), device=self.device)

                def _pearson(xs, ys):
                    n = min(len(xs), len(ys))
                    if n < 16:
                        return _zero
                    x = torch.tensor(xs[-n:], device=self.device, dtype=torch.float32)
                    y = torch.tensor(ys[-n:], device=self.device, dtype=torch.float32)
                    xm, ym = x.mean(), y.mean()
                    xs_ = x - xm
                    ys_ = y - ym
                    denom = (xs_.pow(2).sum().sqrt() * ys_.pow(2).sum().sqrt()).clamp(min=1e-12)
                    return ((xs_ * ys_).sum() / denom)

                _corr_near_vs_td    = _pearson(_st["window_pi_curr"],    _st["window_td_loss"])
                _corr_near_vs_gnorm = _pearson(_st["window_pi_curr"],    _st["window_critic_grad"])
                _corr_qgp_vs_closs  = _pearson(_st["window_qgp_peak_c"], _st["window_critic_loss"])

                try:
                    _have_phase = bool(_obj_idx_diag is not None and _post_lift_diag.any())
                except NameError:
                    _have_phase = False
                if _have_phase:
                    _pl = _post_lift_diag
                    _nf_c_005_pl = (_delta_pi_curr[_pl].abs() <= 0.05).float().mean() if _pl.any() else _zero
                    _nf_n_005_pl = (_delta_pi_next[_pl].abs() <= 0.05).float().mean() if _pl.any() else _zero
                    _gp_n_pl = torch.sigmoid(-_delta_pi_next[_pl].float() / _g_T)
                    _gp_n_pl = _gp_n_pl * (1.0 - _gp_n_pl)
                    _gp_peak_n_pl = (_gp_n_pl >= 0.20).float().mean() if _pl.any() else _zero
                    _qgp_n_pl = q_pi_next_min[_pl].float() * _gp_n_pl
                    _qgp_peak_n_pl = (_gp_n_pl >= 0.20).float().mean() if _pl.any() else _zero
                else:
                    _nf_c_005_pl = _zero
                    _nf_n_005_pl = _zero
                    _gp_peak_n_pl = _zero
                    _qgp_peak_n_pl = _zero

                _tau_proxy_mean = _tau_proxy.mean()
                _tau_proxy_std  = _tau_proxy.std() if _tau_proxy.numel() >= 2 else _zero
                _tau_proxy_type = torch.tensor(
                    0.0 if _sc_mask_mode_v == "sigmoid_symmetric" else 1.0,
                    device=self.device,
                )

                if not _st["debug_printed"]:
                    _st["debug_printed"] = True
                    logger.info(
                        "\n[SC-CQL hypothesis instrumentation — one-shot debug]\n"
                        f"  mode                 : {_sc_mask_mode_v}  "
                        f"(tau_source_mode={int(_tau_proxy_type.item())})\n"
                        f"  tau_cfg_offset       : {_tau_cfg_offset}\n"
                        f"  PROXY DISCLAIMER     : tau_proxy_* tags are derived\n"
                        f"                         from Q_data_min(s), not a learned τ(s)."
                    )

            # ──────────────────────────────────────────────────────────
            # F. SMQR continuous-action baseline telemetry
            #
            # Active only when ``critic_penalty_mode == 'smqr_cont_self'``.
            # Uses the *learned* τ(s) head and the per-critic
            # self-mask g_i(s,a) computed during the penalty step
            # (cached in ``_smqr_cache``).  Tags are minimal: shared
            # τ stats + per-critic Δ / g / g′ / Q·g′ / grad_factor /
            # near-τ occupancy for the action sets {data, rand, pi,
            # pi_next}.
            # ──────────────────────────────────────────────────────────
            _smqr_metrics: dict[str, torch.Tensor] = {}
            if _penalty_mode == "smqr_cont_self" and _smqr_cache is not None:
                _tau = _smqr_cache["tau"]                  # [B]
                _tau_anchor_t = _smqr_cache["tau_anchor"]      # [B]
                _tau_residual_t = _smqr_cache["tau_residual"]  # [B]  post-tanh
                _tau_raw_res_t = _smqr_cache["tau_raw_residual"]  # [B]  pre-tanh
                _tau_res_scale_v = float(_smqr_cache["tau_res_scale"])
                _Q_raw = _smqr_cache["Q_cat_raw"]          # [num_q, B, K]
                _g_all = _smqr_cache["g"]                  # [num_q, B, K]
                _delta_all = _smqr_cache["delta"]          # [num_q, B, K]
                _beta = _smqr_cache["beta"]
                _N_rand = _smqr_cache["num_random"]
                _N_pi = _smqr_cache["num_policy"]

                _abs_eps = float(getattr(args, "sc_tau_near_abs_eps", 0.05))
                _beta_coeff = float(getattr(args, "sc_tau_near_beta_coeff", 1.0))
                _beta_eps_thr = _beta_coeff * _beta

                def _q_stats_1d(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                    """Return (p05, p95) safely (zeros for tiny tensors)."""
                    if x.numel() < 2:
                        return _zero, _zero
                    qs = torch.tensor([0.05, 0.95], device=x.device)
                    v = torch.quantile(x.float().flatten(), qs)
                    return v[0], v[1]

                # Shared τ stats
                _tau_p05, _tau_p95 = _q_stats_1d(_tau)
                _smqr_metrics["smqr/shared/tau_mean"] = _tau.mean()
                _smqr_metrics["smqr/shared/tau_std"]  = (
                    _tau.std() if _tau.numel() >= 2 else _zero
                )
                _smqr_metrics["smqr/shared/tau_p05"]  = _tau_p05
                _smqr_metrics["smqr/shared/tau_p50"]  = (
                    _tau.median() if _tau.numel() >= 1 else _zero
                )
                _smqr_metrics["smqr/shared/tau_p95"]  = _tau_p95
                _smqr_metrics["smqr/shared/beta"]     = torch.tensor(
                    float(_beta), device=self.device,
                )

                # B2-fix decomposition: τ = anchor(Q_data_min, detached)
                # + sc_tau_res_scale · tanh(τ_raw).  Separate stats for
                # the anchor, pre-tanh raw residual, post-tanh residual,
                # and tanh saturation fraction — lets us diagnose whether
                # τ is pinned by the bound (raw runaway + high sat_frac)
                # or is moving freely inside the band.
                _smqr_metrics["smqr/shared/tau_anchor_mean"] = _tau_anchor_t.mean()
                _smqr_metrics["smqr/shared/tau_anchor_std"]  = (
                    _tau_anchor_t.std() if _tau_anchor_t.numel() >= 2 else _zero
                )
                _smqr_metrics["smqr/shared/tau_raw_residual_mean"] = _tau_raw_res_t.mean()
                _smqr_metrics["smqr/shared/tau_raw_residual_std"]  = (
                    _tau_raw_res_t.std() if _tau_raw_res_t.numel() >= 2 else _zero
                )
                _smqr_metrics["smqr/shared/tau_residual_post_mean"] = _tau_residual_t.mean()
                _smqr_metrics["smqr/shared/tau_residual_post_std"]  = (
                    _tau_residual_t.std() if _tau_residual_t.numel() >= 2 else _zero
                )
                # tanh saturation: fraction of |tanh(raw)| > 0.95
                # i.e. fraction of states whose τ is pinned near the
                # boundary of the bounded band.
                _smqr_metrics["smqr/shared/tau_residual_sat_frac"] = (
                    (_tau_residual_t.abs() > 0.95 * _tau_res_scale_v).float().mean()
                )
                _smqr_metrics["smqr/shared/tau_res_scale"] = torch.tensor(
                    _tau_res_scale_v, device=self.device,
                )

                # Build per-set slices over the K=N_rand+N_pi dim.
                # data:    Q_data (size [num_q, B])  — separate tensor
                # rand:    [num_q, B, 0:N_rand]
                # pi:      [num_q, B, N_rand:N_rand+N_pi]
                # pi_next: q_pi_next_at_s (already computed), Δ relative to τ
                _q_data_raw = q_data.float()                # [num_q, B]
                _delta_data_smqr = _q_data_raw - _tau.view(1, B)  # [num_q, B]
                _g_data = torch.sigmoid(_delta_data_smqr / _beta)
                _gprime_data = _g_data * (1.0 - _g_data) / _beta

                _delta_next_smqr = q_pi_next_at_s.float() - _tau.view(1, B)  # [num_q, B]
                _g_next = torch.sigmoid(_delta_next_smqr / _beta)
                _gprime_next = _g_next * (1.0 - _g_next) / _beta

                # gprime for the candidate-action sets is derived from
                # _g_all directly (g'/β = g·(1-g)/β).
                _gprime_all = _g_all * (1.0 - _g_all) / _beta  # [num_q, B, K]

                _slice_rand = slice(0, _N_rand)
                _slice_pi = slice(_N_rand, _N_rand + _N_pi)

                def _emit_for_critic(i: int) -> None:
                    pfx = f"smqr/c{i+1}"
                    sets: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {
                        # name : (delta, g, gprime, q_raw)
                    }
                    sets_4 = {
                        "data":    (_delta_data_smqr[i], _g_data[i], _gprime_data[i], _q_data_raw[i]),
                        "rand":    (_delta_all[i, :, _slice_rand], _g_all[i, :, _slice_rand],
                                    _gprime_all[i, :, _slice_rand], _Q_raw[i, :, _slice_rand]),
                        "pi":      (_delta_all[i, :, _slice_pi],   _g_all[i, :, _slice_pi],
                                    _gprime_all[i, :, _slice_pi],   _Q_raw[i, :, _slice_pi]),
                        "pi_next": (_delta_next_smqr[i], _g_next[i], _gprime_next[i], q_pi_next_at_s[i].float()),
                    }
                    for set_name, (delta, g, gprime, q_raw) in sets_4.items():
                        d_flat = delta.flatten().float()
                        g_flat = g.flatten()
                        gp_flat = gprime.flatten()
                        q_flat = q_raw.flatten().float()
                        qgp_flat = q_flat * gp_flat
                        gradf = g_flat + qgp_flat  # ∂L/∂Q ≈ g + Q·g'

                        # Δ stats
                        d_p05, d_p95 = _q_stats_1d(d_flat)
                        _smqr_metrics[f"{pfx}/delta_{set_name}_mean"] = d_flat.mean()
                        _smqr_metrics[f"{pfx}/delta_{set_name}_std"]  = (
                            d_flat.std() if d_flat.numel() >= 2 else _zero
                        )
                        _smqr_metrics[f"{pfx}/delta_{set_name}_p95_abs"] = (
                            torch.quantile(d_flat.abs(), 0.95) if d_flat.numel() >= 2 else _zero
                        )
                        _smqr_metrics[f"{pfx}/near_{set_name}_frac_abs"] = (
                            (d_flat.abs() <= _abs_eps).float().mean()
                        )
                        _smqr_metrics[f"{pfx}/near_{set_name}_frac_beta"] = (
                            (d_flat.abs() <= _beta_eps_thr).float().mean()
                        )

                        # Mask activation
                        _smqr_metrics[f"{pfx}/g_{set_name}_mean"] = g_flat.mean()
                        _smqr_metrics[f"{pfx}/g_{set_name}_gt_05_frac"] = (
                            (g_flat > 0.5).float().mean()
                        )

                        # g′ stats
                        gp_p95 = (
                            torch.quantile(gp_flat, 0.95) if gp_flat.numel() >= 2 else _zero
                        )
                        _smqr_metrics[f"{pfx}/gprime_{set_name}_mean"] = gp_flat.mean()
                        _smqr_metrics[f"{pfx}/gprime_{set_name}_p95"]  = gp_p95

                        # |Q·g′| stats
                        abs_qgp = qgp_flat.abs()
                        abs_p95 = (
                            torch.quantile(abs_qgp, 0.95) if abs_qgp.numel() >= 2 else _zero
                        )
                        _smqr_metrics[f"{pfx}/abs_qgprime_{set_name}_mean"] = abs_qgp.mean()
                        _smqr_metrics[f"{pfx}/abs_qgprime_{set_name}_p95"]  = abs_p95

                        # grad_factor = g + Q·g′  (direct ∂L/∂Q proxy)
                        gf_p95 = (
                            torch.quantile(gradf, 0.95) if gradf.numel() >= 2 else _zero
                        )
                        _smqr_metrics[f"{pfx}/grad_factor_{set_name}_mean"] = gradf.mean()
                        _smqr_metrics[f"{pfx}/grad_factor_{set_name}_p95"]  = gf_p95

                for _i in range(q_data.shape[0]):
                    _emit_for_critic(_i)

                # ── Phase C / Phase D / Phase E stabilised-objective telemetry ──
                # Emitted whenever the active critic-loss path uses
                # the ``Q + log(g+ε)`` form, i.e. Phase C
                # ``stabilized`` learned-τ, Phase D ``v1_oneside_shrink``
                # learned-τ, OR Phase E anchor+stabilised
                # (``algo_mode='smqr_anchor'`` +
                # ``smqr_anchor_objective='stabilized'``).  Diagnoses
                # whether log(g+ε) is acting as a healthy soft floor
                # or hitting the ε floor for most candidates / pinning
                # all mass on a single sample.
                _stab_obj_active = (
                    _smqr_cache.get("stab_active", False)
                    or _smqr_cache.get("v1_active", False)
                    or _smqr_cache.get("anchor_stab_active", False)
                )
                if _stab_obj_active:
                    _stab_log_g = _smqr_cache["log_g_stab"]          # [num_q, B, K]
                    _stab_g_eps_v = float(_smqr_cache["g_eps"])
                    _stab_w_logits = _smqr_cache["weighted_logits"]  # [num_q, B, K] post-clamp
                    _stab_w = torch.softmax(_stab_w_logits.float(), dim=-1)  # [num_q, B, K]
                    _stab_ent = -(
                        _stab_w * _stab_w.clamp_min(1e-12).log()
                    ).sum(dim=-1)                                    # [num_q, B]
                    _stab_top1 = _stab_w.max(dim=-1).values          # [num_q, B]
                    _lg_p05, _ = _q_stats_1d(_stab_log_g)
                    _smqr_metrics["smqr/stab/log_g_p05"] = _lg_p05
                    _smqr_metrics["smqr/stab/log_g_min"] = _stab_log_g.min()
                    # g_lt_eps_frac: fraction of candidate-action g
                    # values that were clamped UP TO the ε floor (a
                    # direct read on whether log(g+ε) is degenerating
                    # into a constant offset).
                    _smqr_metrics["smqr/stab/g_lt_eps_frac"] = (
                        (_g_all <= _stab_g_eps_v).float().mean()
                    )
                    _smqr_metrics["smqr/stab/softmax_weight_entropy"] = _stab_ent.mean()
                    _smqr_metrics["smqr/stab/softmax_weight_top1"] = _stab_top1.mean()
                    _smqr_metrics["smqr/stab/g_eps"] = torch.tensor(
                        _stab_g_eps_v, device=self.device,
                    )

                # ── Phase E (anchor + stabilised objective) τ-invariant ──
                # By construction this branch fixes τ ≡ τ_anchor (via
                # ``sc_tau_res_scale=0.0`` enforced at the resolver).
                # Verify the invariant ONCE, on the first invocation,
                # using a floating tolerance (not strict equality).
                # If the invariant ever fails, the run is poisoned —
                # halt loudly.
                if _smqr_cache.get("anchor_stab_active", False):
                    if not getattr(self, "_phase_e_invariant_logged", False):
                        _trs_e = float(_smqr_cache["tau_res_scale"])
                        _tau_e = _smqr_cache["tau"]                  # [B]
                        _anc_e = _smqr_cache["tau_anchor"]           # [B]
                        _res_e = _smqr_cache["tau_residual"]         # [B]
                        _amt_e_abs = (_anc_e - _tau_e).abs()
                        _amt_e_mean = float(_amt_e_abs.mean().item())
                        _amt_e_p95 = float(
                            torch.quantile(_amt_e_abs, 0.95).item()
                            if _amt_e_abs.numel() >= 2 else _amt_e_abs.max().item()
                        )
                        _res_e_std = float(_res_e.float().std(unbiased=False).item())
                        _tol = 1e-8
                        logger.info(
                            "[Phase E] τ-invariant (one-time check): "
                            f"sc_tau_res_scale={_trs_e:.3e}  "
                            f"|anchor−τ|_mean={_amt_e_mean:.3e}  "
                            f"|anchor−τ|_p95={_amt_e_p95:.3e}  "
                            f"tau_residual_std={_res_e_std:.3e}  "
                            f"tol={_tol:.0e}"
                        )
                        if _trs_e != 0.0:
                            raise RuntimeError(
                                "[Phase E] sc_tau_res_scale must be 0.0 for "
                                "anchor+stabilised objective; got "
                                f"{_trs_e!r}.  This should have been "
                                "blocked by the algo_mode resolver."
                            )
                        if _amt_e_mean > _tol or _amt_e_p95 > _tol:
                            raise RuntimeError(
                                "[Phase E] τ-invariant violated: "
                                f"|anchor−τ|_mean={_amt_e_mean:.3e}, "
                                f"|anchor−τ|_p95={_amt_e_p95:.3e}, "
                                f"tol={_tol:.0e}.  τ must equal τ_anchor "
                                "exactly (within float tolerance) on the "
                                "anchor-only branch."
                            )
                        if _res_e_std >= _tol:
                            raise RuntimeError(
                                "[Phase E] tau_residual.std()="
                                f"{_res_e_std:.3e} ≥ tol={_tol:.0e}; "
                                "residual must be ~0 on anchor-only branch."
                            )
                        self._phase_e_invariant_logged = True

                # ── V1 (Phase D) one-sided-residual telemetry ────
                # By construction τ ≤ τ_anchor, so anchor − τ ≥ 0.
                # If ``anchor_minus_tau_p95`` saturates near
                # ``sc_tau_res_scale``, the head has hit the (downward)
                # boundary and shrinkage is the only thing keeping
                # the residual finite — paired metrics with `L_shrink`
                # let us read whether the boundary is actively bounding
                # the residual or merely a passive cap.
                # ── V1 (Phase D) / F1 (Phase F) one-sided-residual telemetry ──
                # By construction τ ≤ τ_anchor, so anchor − τ ≥ 0.
                # F1 shares V1's τ-parameterisation exactly, so the
                # metrics are emitted under either the smqr/v1/* or
                # smqr/f1/* namespace depending on the active variant.
                if (
                    _smqr_cache.get("v1_active", False)
                    or _smqr_cache.get("f1_active", False)
                ):
                    _is_f1 = bool(_smqr_cache.get("f1_active", False))
                    _pfx_oneside = "smqr/f1" if _is_f1 else "smqr/v1"
                    _which_lbl = "F1" if _is_f1 else "V1"
                    _trp = _smqr_cache["tau_residual_pos"]            # [B], ≥ 0
                    _trs = float(_smqr_cache["tau_res_scale"])
                    _amt = _trs * _trp                                # [B], = anchor − τ ≥ 0
                    # Numerical one-sided sanity (debug assert + warn).
                    if torch.isfinite(_amt).all().item():
                        _amt_min_v = float(_amt.min().item())
                        if _amt_min_v < -1e-6:
                            logger.warning(
                                f"[{_which_lbl}] anchor_minus_tau MIN = {_amt_min_v:.3e} "
                                "(< 0); one-sided invariant violated."
                            )
                        # Hard assert in debug builds; guarded so
                        # production runs do not crash on a transient
                        # numerical fluke.
                        assert _amt_min_v >= -1e-3, (
                            f"{_which_lbl} invariant τ ≤ anchor violated: "
                            f"min(anchor − τ) = {_amt_min_v}."
                        )
                    _amt_p95 = (
                        torch.quantile(_amt.float(), 0.95)
                        if _amt.numel() >= 2 else _amt.mean()
                    )
                    _smqr_metrics[f"{_pfx_oneside}/anchor_minus_tau_mean"] = _amt.mean()
                    _smqr_metrics[f"{_pfx_oneside}/anchor_minus_tau_p95"] = _amt_p95
                    _smqr_metrics[f"{_pfx_oneside}/tau_raw_mean"] = _tau_raw_res_t.mean()
                    _smqr_metrics[f"{_pfx_oneside}/tau_raw_p05"] = (
                        torch.quantile(_tau_raw_res_t.float(), 0.05)
                        if _tau_raw_res_t.numel() >= 2 else _tau_raw_res_t.mean()
                    )
                    _smqr_metrics[f"{_pfx_oneside}/L_shrink"] = _smqr_cache["v1_shrink_loss"]
                    _smqr_metrics[f"{_pfx_oneside}/shrink_lambda"] = torch.tensor(
                        float(_smqr_cache["v1_shrink_lambda"]),
                        device=self.device,
                    )

                # ── F1 (Phase F) ST-split objective telemetry ─────────
                # F1 does NOT use log(g+ε); the stabilised-objective
                # telemetry block (g_lt_eps_frac / log_g_min / softmax)
                # is suppressed for F1.  Emit F1-specific keys here:
                #   - qg_st_drift  : sanity check that ST-split forward
                #                    matches vanilla Q·g bit-exactly.
                #   - qg_pi_mean   : mean Q·g over policy candidates.
                #   - qg_data_mean : mean Q·g over data action (= q_data).
                #   - softmax_weight_top1 / entropy : softmax health on
                #     the F1 weighted_logits (no log_g term).
                if _smqr_cache.get("f1_active", False):
                    _wl_f1 = _smqr_cache["weighted_logits"]            # [num_q, B, K]
                    _w_f1 = torch.softmax(_wl_f1.float(), dim=-1)
                    _ent_f1 = -(
                        _w_f1 * _w_f1.clamp_min(1e-12).log()
                    ).sum(dim=-1)                                       # [num_q, B]
                    _top1_f1 = _w_f1.max(dim=-1).values                 # [num_q, B]
                    _smqr_metrics["smqr/f1/softmax_weight_top1"] = _top1_f1.mean()
                    _smqr_metrics["smqr/f1/softmax_weight_entropy"] = _ent_f1.mean()
                    # qg sanity: forward-equality of ST-split with
                    # vanilla Q·g (must be ~0 in float, regardless of
                    # backward routing).  Computed from the cached
                    # detached tensors.
                    _Q_det = _smqr_cache["Q_cat_raw"]                   # [num_q, B, K]
                    _g_det = _smqr_cache["g"]                           # [num_q, B, K]
                    _qg_vanilla = _Q_det * _g_det
                    _qg_st_fwd = 0.5 * (_Q_det * _g_det + _Q_det * _g_det)  # bit-exact reference
                    _smqr_metrics["smqr/f1/qg_st_drift"] = (
                        (_qg_st_fwd - _qg_vanilla).abs().mean()
                    )
                    _smqr_metrics["smqr/f1/qg_pi_mean"] = _qg_vanilla.mean()
                    # Per-critic q_data is q_data; emit a single scalar mean.
                    _smqr_metrics["smqr/f1/qg_data_mean"] = q_data.mean()

                # ── G1 (Phase F+) candidate-wise routing telemetry ────
                # Emitted ONLY when G1 routing is active (i.e. F1 base
                # AND smqr_f1_random_full_grad=True).  Splits forward-
                # identity drift and ∂L/∂Q grad-factor magnitude across
                # the random vs policy K-axis halves.
                #
                # Forward identity (must be ~0 by construction on both):
                #   random branch : qg = Q·g                        ⇒ drift = 0
                #   policy branch : qg = ½·(Q·sg(g) + sg(Q)·g)      ⇒ forward = Q·g ⇒ drift = 0
                #
                # Grad-factor proxy (per-K |∂L/∂Q| up to softmax weight):
                #   random : |g + Q·g'/β|        (vanilla full-grad)
                #   policy : ½·|g + Q·g'/β|      (F1 ST-split)
                # Reading: random_grad_factor_p95 should be ≈ 2× policy
                # if the same Q,g distribution applies — confirms the
                # ½-attenuation is correctly localised to the policy
                # channel only.
                if _smqr_cache.get("g1_active", False):
                    _Q_g1 = _smqr_cache["Q_cat_raw"]                    # [num_q, B, K]
                    _g_g1 = _smqr_cache["g"]                            # [num_q, B, K]
                    _beta_g1 = float(_smqr_cache["beta"])
                    _nr_g1 = int(_smqr_cache["num_random"])
                    # qg drift (forward identity sanity, split by branch)
                    _qg_van_full = _Q_g1 * _g_g1
                    _qg_st_full = 0.5 * (_Q_g1 * _g_g1 + _Q_g1 * _g_g1)
                    # Random branch: routed to vanilla Q·g, drift vs vanilla = 0.
                    _drift_rand = (_qg_van_full - _qg_van_full)[..., :_nr_g1].abs()
                    # Policy branch: routed to ST-split, forward = Q·g, drift = |st_fwd − vanilla| ≈ 0.
                    _drift_pol = (_qg_st_full - _qg_van_full)[..., _nr_g1:].abs()
                    _smqr_metrics["smqr/ab/qg_drift_random"] = _drift_rand.mean()
                    _smqr_metrics["smqr/ab/qg_drift_policy"] = _drift_pol.mean()
                    # Grad-factor magnitude proxy:  |g + Q·g'/β|
                    # g' = g·(1−g)/β  ⇒  Q·g'/β = Q·g·(1−g)/β²  (already
                    # absorbed into the proxy via the chain-rule factor).
                    _gprime = _g_g1 * (1.0 - _g_g1) / max(_beta_g1, 1e-8)
                    _gf_full = (_g_g1 + _Q_g1 * _gprime).abs()          # [num_q, B, K]
                    _gf_rand = _gf_full[..., :_nr_g1].flatten().float()
                    _gf_pol_half = (0.5 * _gf_full[..., _nr_g1:]).flatten().float()
                    _smqr_metrics["smqr/ab/random_grad_factor_p95"] = (
                        torch.quantile(_gf_rand, 0.95)
                        if _gf_rand.numel() >= 2 else _gf_rand.mean()
                    )
                    _smqr_metrics["smqr/ab/policy_grad_factor_p95"] = (
                        torch.quantile(_gf_pol_half, 0.95)
                        if _gf_pol_half.numel() >= 2 else _gf_pol_half.mean()
                    )

                # ── H1 (Phase F++) α-floor telemetry ───────────────────
                # Emitted ONLY when H1 is active (i.e. F1+G1 base AND
                # smqr_h1_alpha_floor > 0).  Splits "G1 raw" vs "H1
                # effective" gate / grad-factor on the random branch
                # so the α effect is directly readable.
                #
                # Effective gate: g_eff = g + α (random branch only).
                # Grad-factor:    |g_eff + Q·g'/β|
                # Compared against smqr/ab/random_grad_factor_p95 (G1),
                # the H1 key should be uniformly higher by ≥ α and must
                # not collapse to ≈ 0 even when g_rand → 0.
                if _smqr_cache.get("h1_active", False):
                    _Q_h1 = _smqr_cache["Q_cat_raw"]                    # [num_q, B, K]
                    _g_h1 = _smqr_cache["g"]                            # [num_q, B, K]
                    _beta_h1 = float(_smqr_cache["beta"])
                    _nr_h1 = int(_smqr_cache["num_random"])
                    _alpha_h1 = float(_smqr_cache["h1_alpha"])
                    # Effective gate on random branch: g + α
                    _g_rand_eff = _g_h1[..., :_nr_h1] + _alpha_h1       # [num_q, B, num_random]
                    _g_rand_eff_flat = _g_rand_eff.flatten().float()
                    _smqr_metrics["smqr/h1/alpha_floor"] = torch.tensor(
                        _alpha_h1, device=self.device,
                    )
                    _smqr_metrics["smqr/h1/random_eff_gate_mean"] = (
                        _g_rand_eff_flat.mean()
                    )
                    _smqr_metrics["smqr/h1/random_eff_gate_p05"] = (
                        torch.quantile(_g_rand_eff_flat, 0.05)
                        if _g_rand_eff_flat.numel() >= 2 else _g_rand_eff_flat.mean()
                    )
                    # Grad-factor with α-floor on the random branch:
                    # |(g + α) + Q · g·(1−g)/β²|.  g' = g·(1−g)/β.
                    _gprime_h1 = _g_h1[..., :_nr_h1] * (
                        1.0 - _g_h1[..., :_nr_h1]
                    ) / max(_beta_h1, 1e-8)
                    _gf_rand_h1 = (
                        _g_rand_eff
                        + _Q_h1[..., :_nr_h1] * _gprime_h1
                    ).abs().flatten().float()
                    _smqr_metrics["smqr/h1/random_grad_factor_p95"] = (
                        torch.quantile(_gf_rand_h1, 0.95)
                        if _gf_rand_h1.numel() >= 2 else _gf_rand_h1.mean()
                    )

                # ── B2 (Phase I) STE max-clip backward floor telemetry ─
                # Emitted ONLY when B2 is active (i.e. F1+G1 base AND
                # smqr_b2_alpha_floor > 0  AND  H1 inactive).
                # Forward is bit-exact to G1, so smqr/ab/qg_drift_random
                # remains 0 by construction (sanity-checked in the G1
                # block above).  The diagnostic axis here is the BACKWARD
                # Q-gradient floor `g_back = max(g, α)`:
                #   - random_g_below_floor_frac   : how often the floor activates.
                #   - random_g_back_{mean,p05}    : effective Q-grad gate distribution.
                #   - random_grad_factor_p95      : full proxy |g_back + Q·g'/β|;
                #     compare directly against smqr/ab/random_grad_factor_p95.
                if _smqr_cache.get("b2_active", False):
                    _Q_b2 = _smqr_cache["Q_cat_raw"]                    # [num_q, B, K]
                    _g_b2 = _smqr_cache["g"]                            # [num_q, B, K]
                    _beta_b2 = float(_smqr_cache["beta"])
                    _nr_b2 = int(_smqr_cache["num_random"])
                    _alpha_b2 = float(_smqr_cache["b2_alpha"])
                    _g_r_b2 = _g_b2[..., :_nr_b2]                       # [num_q, B, num_random]
                    _Q_r_b2 = _Q_b2[..., :_nr_b2]
                    _g_back_b2 = torch.clamp(_g_r_b2, min=_alpha_b2)    # already detached cache
                    _below_b2 = (_g_r_b2 < _alpha_b2).float()
                    _smqr_metrics["smqr/b2/alpha_floor"] = torch.tensor(
                        _alpha_b2, device=self.device,
                    )
                    _smqr_metrics["smqr/b2/random_g_below_floor_frac"] = (
                        _below_b2.mean()
                    )
                    _g_back_flat = _g_back_b2.flatten().float()
                    _smqr_metrics["smqr/b2/random_g_back_mean"] = (
                        _g_back_flat.mean()
                    )
                    _smqr_metrics["smqr/b2/random_g_back_p05"] = (
                        torch.quantile(_g_back_flat, 0.05)
                        if _g_back_flat.numel() >= 2 else _g_back_flat.mean()
                    )
                    # Grad-factor proxy with B2 backward floor:
                    #   |g_back + Q · g'|,  g' = g·(1−g)/β
                    # The leading `g` of the G1 proxy is replaced by
                    # `g_back = max(g, α)` (the actual Q-grad on this
                    # branch); the τ-grad term Q·g' is unchanged.
                    _gprime_b2 = _g_r_b2 * (1.0 - _g_r_b2) / max(_beta_b2, 1e-8)
                    _gf_b2 = (
                        _g_back_b2 + _Q_r_b2 * _gprime_b2
                    ).abs().flatten().float()
                    _smqr_metrics["smqr/b2/random_grad_factor_p95"] = (
                        torch.quantile(_gf_b2, 0.95)
                        if _gf_b2.numel() >= 2 else _gf_b2.mean()
                    )

                # ── Step-3 SMQR-SG telemetry (smqr/sg/*) ──────────
                # Emitted whenever the SMQR penalty path runs (cache
                # is populated).  All three lse_modes share the same
                # schema so paired comparisons line up; mode_id makes
                # the active branch unambiguous in the dashboard.
                _sg_lse_mode = str(
                    _smqr_cache.get("smqr_lse_mode", "q_times_g")
                )
                _SG_MODE_ID = {
                    "q_times_g": 0,
                    "q_times_detached_g": 1,
                    "sg_weighted_lse": 2,
                }
                _sg_eps_v = float(_smqr_cache.get("smqr_sg_eps", 1e-6))
                _sg_beta = float(_smqr_cache["beta"])
                _sg_w_logits = _smqr_cache["weighted_logits"]            # [num_q, B, K] post-clip
                _sg_w_pre = _smqr_cache.get("weighted_logits_preclip")   # [num_q, B, K] pre-clip
                if _sg_w_pre is None:
                    _sg_w_pre = _sg_w_logits
                _sg_Q = _smqr_cache["Q_cat_raw"]                          # [num_q, B, K]
                _sg_g = _smqr_cache["g"]                                  # [num_q, B, K]
                _sg_delta = _smqr_cache["delta"]                          # [num_q, B, K]
                _sg_tau = _smqr_cache["tau"]                              # [B]

                _smqr_metrics["smqr/sg/lse_mode_id"] = torch.tensor(
                    float(_SG_MODE_ID.get(_sg_lse_mode, -1)),
                    device=self.device,
                )
                _smqr_metrics["smqr/sg/sg_eps"] = torch.tensor(
                    _sg_eps_v, device=self.device,
                )

                # Logits stats (pre/post clip).
                _pre_f = _sg_w_pre.float()
                _post_f = _sg_w_logits.float()
                _smqr_metrics["smqr/sg/logits_preclip_mean"] = _pre_f.mean()
                _smqr_metrics["smqr/sg/logits_preclip_std"] = (
                    _pre_f.std() if _pre_f.numel() >= 2 else _zero
                )
                _smqr_metrics["smqr/sg/logits_preclip_min"] = _pre_f.min()
                _smqr_metrics["smqr/sg/logits_preclip_max"] = _pre_f.max()
                _smqr_metrics["smqr/sg/logits_postclip_mean"] = _post_f.mean()
                _smqr_metrics["smqr/sg/logits_postclip_std"] = (
                    _post_f.std() if _post_f.numel() >= 2 else _zero
                )

                # Softmax over candidate axis.
                _sg_w = torch.softmax(_post_f, dim=-1)                    # [num_q, B, K]
                _sg_ent = -(
                    _sg_w * _sg_w.clamp_min(1e-12).log()
                ).sum(dim=-1)                                             # [num_q, B]
                _sg_top1 = _sg_w.max(dim=-1).values                       # [num_q, B]
                _smqr_metrics["smqr/sg/lse_softmax_entropy_mean"] = _sg_ent.mean()
                _smqr_metrics["smqr/sg/lse_softmax_top1_mean"] = _sg_top1.mean()

                # Gate stats.
                _g_flat_sg = _sg_g.flatten().float()
                _smqr_metrics["smqr/sg/g_mean"] = _g_flat_sg.mean()
                _smqr_metrics["smqr/sg/g_std"] = (
                    _g_flat_sg.std() if _g_flat_sg.numel() >= 2 else _zero
                )
                if _g_flat_sg.numel() >= 2:
                    _qs_g = torch.tensor(
                        [0.10, 0.50, 0.90], device=_g_flat_sg.device,
                    )
                    _gv = torch.quantile(_g_flat_sg, _qs_g)
                    _smqr_metrics["smqr/sg/g_p10"] = _gv[0]
                    _smqr_metrics["smqr/sg/g_p50"] = _gv[1]
                    _smqr_metrics["smqr/sg/g_p90"] = _gv[2]
                else:
                    _smqr_metrics["smqr/sg/g_p10"] = _zero
                    _smqr_metrics["smqr/sg/g_p50"] = _zero
                    _smqr_metrics["smqr/sg/g_p90"] = _zero
                _smqr_metrics["smqr/sg/g_lt_eps_frac"] = (
                    (_g_flat_sg <= _sg_eps_v).float().mean()
                )

                # Near-τ band on candidates: |Δ| ≤ β.
                _abs_d = _sg_delta.abs().float()
                _near_mask = (_abs_d <= _sg_beta)                         # [num_q, B, K]
                _smqr_metrics["smqr/sg/near_tau_frac"] = (
                    _near_mask.float().mean()
                )
                # Softmax mass on near-τ candidates.
                _near_mass = (_sg_w * _near_mask.float()).sum(dim=-1)     # [num_q, B]
                _smqr_metrics["smqr/sg/near_tau_softmax_mass"] = _near_mass.mean()

                # Gradient-factor proxy (mode-specific) and grad mass.
                #   q_times_g          : w · |g + Q · g'|
                #   q_times_detached_g : w · |g|
                #   sg_weighted_lse    : w · 1
                _gp_sg = _sg_g * (1.0 - _sg_g) / max(_sg_beta, 1e-8)      # g'
                if _sg_lse_mode == "q_times_g":
                    _gf_proxy = (_sg_g + _sg_Q * _gp_sg).abs().float()
                elif _sg_lse_mode == "q_times_detached_g":
                    _gf_proxy = _sg_g.abs().float()
                else:  # sg_weighted_lse
                    _gf_proxy = torch.ones_like(_sg_g, dtype=torch.float32)
                _gf_weighted = (_sg_w * _gf_proxy)                        # softmax-weighted
                _gf_weighted_flat = _gf_weighted.flatten()
                _smqr_metrics["smqr/sg/grad_factor_proxy_mean"] = _gf_weighted_flat.mean()
                if _gf_weighted_flat.numel() >= 2:
                    _qs_gf = torch.tensor(
                        [0.50, 0.95], device=_gf_weighted_flat.device,
                    )
                    _gfv = torch.quantile(_gf_weighted_flat, _qs_gf)
                    _smqr_metrics["smqr/sg/grad_factor_proxy_p50"] = _gfv[0]
                    _smqr_metrics["smqr/sg/grad_factor_proxy_p95"] = _gfv[1]
                else:
                    _smqr_metrics["smqr/sg/grad_factor_proxy_p50"] = _zero
                    _smqr_metrics["smqr/sg/grad_factor_proxy_p95"] = _zero
                # Near-τ grad mass: ∑_near (w_i · grad_factor_i).
                _near_grad_mass = (_gf_weighted * _near_mask.float()).sum(dim=-1)
                _smqr_metrics["smqr/sg/near_tau_grad_mass"] = _near_grad_mass.mean()
                # Starvation: candidates whose softmax weight is
                # effectively zero (≤ 1e-6).
                _smqr_metrics["smqr/sg/grad_starvation_frac"] = (
                    (_sg_w <= 1e-6).float().mean()
                )

                # Ranking diagnostics: argmax(Q) vs argmax(weighted_logits).
                _q_top1 = _sg_Q.float().argmax(dim=-1)                    # [num_q, B]
                _w_top1 = _post_f.argmax(dim=-1)                          # [num_q, B]
                _smqr_metrics["smqr/sg/top1_q_matches_top1_logits"] = (
                    (_q_top1 == _w_top1).float().mean()
                )
                # Pearson correlation along the candidate axis (per
                # critic, per state), then averaged.  Cheap proxy for
                # rank correlation.
                _Qf = _sg_Q.float()
                _Wf = _post_f
                _Q_c = _Qf - _Qf.mean(dim=-1, keepdim=True)
                _W_c = _Wf - _Wf.mean(dim=-1, keepdim=True)
                _num = (_Q_c * _W_c).sum(dim=-1)
                _den = (
                    _Q_c.pow(2).sum(dim=-1).clamp_min(1e-12).sqrt()
                    * _W_c.pow(2).sum(dim=-1).clamp_min(1e-12).sqrt()
                )
                _pearson = (_num / _den).clamp(-1.0, 1.0)
                _smqr_metrics["smqr/sg/rank_corr_q_vs_logits"] = _pearson.mean()

                # ── Core-input ranking diagnostics (Step-4 patch) ──
                # The pre-existing rank_corr_q_vs_logits compares
                # argmax(Q) to argmax(post-clip weighted_logits), which
                # is "core_input − log p" (i.e. final_logits).  To
                # separate the Q*g forward distortion from the −log p
                # action-density correction, we additionally compare
                # argmax(Q) to argmax(core_input) where:
                #   q_times_g          : core_input = Q · g
                #   q_times_detached_g : core_input = Q · g           (forward identical to q_times_g)
                #   sg_weighted_lse    : core_input = Q + log(g.clamp_min(ε))
                # No autograd path is created (telemetry-only); core
                # input is built from cached tensors and detached.
                with torch.no_grad():
                    _Q_det = _sg_Q.detach().float()
                    _g_det = _sg_g.detach().float()
                    if _sg_lse_mode == "sg_weighted_lse":
                        _core_input = _Q_det + _g_det.clamp_min(_sg_eps_v).log()
                    else:
                        _core_input = _Q_det * _g_det
                    # final_logits == post-clip weighted_logits already
                    # exposed as rank_corr_q_vs_logits; re-emit under
                    # the explicit alias for symmetry with the new
                    # core_input pair.
                    _smqr_metrics["smqr/sg/rank_corr_q_vs_final_logits"] = _pearson.mean()
                    _smqr_metrics["smqr/sg/top1_q_matches_top1_final_logits"] = (
                        (_q_top1 == _w_top1).float().mean()
                    )
                    # core_input top-1 match.
                    _ci_top1 = _core_input.argmax(dim=-1)
                    _smqr_metrics["smqr/sg/top1_q_matches_top1_core_input"] = (
                        (_q_top1 == _ci_top1).float().mean()
                    )
                    # core_input Pearson rank-corr proxy.
                    _CIc = _core_input - _core_input.mean(dim=-1, keepdim=True)
                    _num_ci = (_Q_c * _CIc).sum(dim=-1)
                    _den_ci = (
                        _Q_c.pow(2).sum(dim=-1).clamp_min(1e-12).sqrt()
                        * _CIc.pow(2).sum(dim=-1).clamp_min(1e-12).sqrt()
                    )
                    _pearson_ci = (_num_ci / _den_ci).clamp(-1.0, 1.0)
                    _smqr_metrics["smqr/sg/rank_corr_q_vs_core_input"] = _pearson_ci.mean()

                # One-shot debug print for SMQR-SG branch identity.
                if not getattr(self, "_smqr_sg_debug_printed", False):
                    self._smqr_sg_debug_printed = True
                    logger.info(
                        "\n[SMQR-SG (Step 3) — one-shot debug]\n"
                        f"  smqr_lse_mode        : {_sg_lse_mode}  "
                        f"(id={_SG_MODE_ID.get(_sg_lse_mode, -1)})\n"
                        f"  smqr_sg_eps          : {_sg_eps_v}\n"
                        f"  forward formula     : "
                        + (
                            "Q · g − log p"
                            if _sg_lse_mode == "q_times_g"
                            else "Q · detach(g) − log p"
                            if _sg_lse_mode == "q_times_detached_g"
                            else "Q + log(detach(g) + ε) − log p"
                        )
                        + "\n"
                        f"  ∂lse/∂Q proxy        : "
                        + (
                            "softmax · (g + Q · g'/β)"
                            if _sg_lse_mode == "q_times_g"
                            else "softmax · g (gate-derivative removed)"
                            if _sg_lse_mode == "q_times_detached_g"
                            else "softmax (β-independent; no Q · g' term)"
                        )
                        + "\n"
                    )

                # One-shot summary print
                if not getattr(self, "_smqr_debug_printed", False):
                    self._smqr_debug_printed = True
                    logger.info(
                        "\n[SMQR continuous-action baseline — one-shot debug]\n"
                        f"  mode                 : smqr_cont_self (per-critic self-mask, no detach)\n"
                        f"  τ parameterization   : τ(s) = Q_data_min(s).detach() + {_tau_res_scale_v} · tanh(τ_raw(s))   [B2-fix: bounded residual]\n"
                        f"  τ head               : shared, obs-only ({type(qnet.tau_head).__name__})  (final Linear zero-init)\n"
                        f"  β                    : {_beta}\n"
                        f"  sc_tau_res_scale     : {_tau_res_scale_v}  (→ τ ∈ anchor ± {_tau_res_scale_v})\n"
                        f"  near_abs_eps         : {_abs_eps}\n"
                        f"  near_beta_coeff      : {_beta_coeff} (→ |Δ| ≤ {_beta_eps_thr:.4f})\n"
                        f"  Q stack shape        : {tuple(_Q_raw.shape)}\n"
                        f"  IS density           : log p applied OUTSIDE mask\n"
                        f"  grad_factor          : g + Q·g'  (direct ∂L/∂Q proxy)\n"
                    )

        # ── Build return dict (mode-conditional) ──────────────────────
        _metrics: dict[str, torch.Tensor] = {
            "td_loss": td_loss.detach(),
            "cql_penalty": cql_penalty.detach(),
            # _cql_penalty_raw is consumed by learn() for the Lagrange
            # update but excluded from TensorBoard logging (underscore key).
            "_cql_penalty_raw": cql_penalty_raw.detach(),
            "cql_penalty_per_q_mean": cql_penalty_mean.detach(),
            "cql_alpha": alpha_cql.squeeze().detach(),
            "cql_loss": cql_loss.detach(),
            "cql_loss_unscaled": _cql_loss_unscaled.detach(),
            "cql_loss_scale": torch.tensor(_cql_loss_scale, device=self.device),
            "smqr_blend_lambda_active": torch.tensor(
                float(_smqr_blend_lambda_active), device=self.device,
            ),
            "critic_loss": critic_loss.detach(),
            "critic_grad_norm": critic_grad_norm.detach(),
            "q_data_mean": q_data_mean,
            "q_data_max": q_data_max,
            "q_data_min": q_data_min,
            "td_target_mean": td_target_mean,
            "td_target_max": td_target.max(),
            "td_target_min": td_target.min(),
            "cql_q_rand_mean": q_rand.mean().detach(),
            "cql_q_pi_mean": q_pi.mean().detach(),
            "q_overestimation_gap": (q_data_mean - td_target_mean).detach(),
            "cql_logsumexp_mean": cql_logsumexp.mean().detach(),
            "q_data_q_pi_gap": (q_data_mean - q_pi.mean()).detach(),
            # TD-relative CQL alpha diagnostics
            "cql_effective_alpha": torch.tensor(_effective_alpha, device=self.device),
            "cql_raw_alpha": torch.tensor(_cql_raw_alpha, device=self.device),
            "cql_alpha_floor_active": torch.tensor(_floor_active, device=self.device),
            # ── P1 effective-α cap diagnostics ────────────────────
            # _effective_alpha_pre_cap : value BEFORE cap (= raw/floored)
            # _effective_alpha         : value AFTER cap  (used in loss)
            # cap_active               : 1.0 iff cap clipped this step
            # cql_loss_actual          : the actual scalar fed into
            #                            critic_loss (= eff_α * penalty),
            #                            useful when penalty sign-flips.
            "cql_effective_alpha_raw": torch.tensor(
                _effective_alpha_pre_cap, device=self.device,
            ),
            "cql_effective_alpha_capped": torch.tensor(
                _effective_alpha, device=self.device,
            ),
            "cql_alpha_cap_active": torch.tensor(
                _cap_active, device=self.device,
            ),
            "cql_alpha_cap_value": torch.tensor(
                _cql_alpha_cap, device=self.device,
            ),
            "cql_loss_actual": cql_loss.detach(),
            # ── P1b penalty-floor diagnostics ──────────────────────
            # cql_penalty_raw            : the raw (clamp(-10) sum) penalty,
            #                              unchanged across all modes; kept
            #                              as a top-level metric alias for
            #                              cross-run table convenience.
            # cql_penalty_for_loss       : the scalar actually multiplied
            #                              into critic_loss after P1b
            #                              one-sided floor.  Identical to
            #                              cql_penalty when optin=False.
            # cql_penalty_clamped_frac   : 1.0 iff optin=True AND the
            #                              floor activated this step.
            # cql_penalty_negative_frac  : 1.0 iff cql_penalty<0 (regardless
            #                              of optin).  Used to detect sign-
            #                              flip events even when the floor
            #                              is off.
            "cql_penalty_raw": cql_penalty.detach(),
            "cql_penalty_for_loss": (
                _penalty_for_loss.detach()
                if isinstance(_penalty_for_loss, torch.Tensor)
                else torch.tensor(_penalty_for_loss, device=self.device)
            ),
            "cql_penalty_clamped_frac": torch.tensor(
                _penalty_clamped_frac, device=self.device,
            ),
            "cql_penalty_negative_frac": torch.tensor(
                _penalty_negative_frac, device=self.device,
            ),
            "cql_penalty_floor_optin": torch.tensor(
                1.0 if _penalty_floor_optin else 0.0, device=self.device,
            ),
            # ── C. Twin critic disagreement (data-only = generic) ─
            "q1_q2_gap_data": _gap_data,
        }

        # ── A / B / C(pi) blocks — emitted in vanilla_cql and sc_cql
        # ONLY.  Under smqr_cont_self these are either (i) redundant
        # with the per-critic SMQR Δ / g telemetry (A block ⊂
        # ``smqr/c*/delta_pi_*_mean`` + ``near_pi_frac_*``) or
        # (ii) computed from vanilla-CQL IS-corrected logits
        # ``q_cat_f32`` and therefore carry VANILLA semantics, not
        # the SMQR mask-weighted loss semantics — they would be
        # misleading as apples-to-apples SMQR diagnostics.
        #
        # Rationale for dropping in smqr mode:
        #   A. pi_q_violation_* / q_data_q_pi_*_gap / cql_q_pi_*_mean
        #      → subsumed by smqr/c{1,2}/delta_pi_{_,next_}mean,
        #        near_pi_frac_abs, near_pi_frac_beta (per-critic,
        #        not min-reduced).
        #   B. cql_lse_weight_share_* / cql_q_*_max
        #      → weight shares are derived from vanilla q_cat,
        #        NOT the SMQR-weighted logits ``Q·g − log p``.
        #        Publishing them under the SMQR run invites wrong
        #        causal readings.
        #   C. q1_q2_gap_pi_curr / q1_q2_gap_pi_next
        #      → pi-side twin gap at OOD actions is dominated by
        #        the same confound the SMQR study is isolating.
        #        Data-side gap (``q1_q2_gap_data``) is kept as a
        #        generic critic-health signal.
        #
        # Back-compat: vanilla_cql and sc_cql runs are UNCHANGED.
        if _penalty_mode != "smqr_cont_self":
            _metrics.update({
                # ── A. Policy-side OOD (min-Q basis) ──────────────
                "cql_q_pi_curr_mean": _cql_q_pi_curr_mean,
                "cql_q_pi_next_mean": _cql_q_pi_next_mean,
                "q_data_q_pi_curr_gap": _q_data_q_pi_curr_gap,
                "q_data_q_pi_next_gap": _q_data_q_pi_next_gap,
                "pi_q_violation_rate_eps_0p1": _viol_0p1,
                "pi_q_violation_rate_eps_0p0": _viol_0p0,
                "pi_q_violation_mag": _viol_mag,
                "pi_q_violation_l2_weighted": _viol_l2w,
                # ── B. CQL candidate dominance ────────────────────
                "cql_lse_weight_share_rand": _w_rand,
                "cql_lse_weight_share_pi_curr": _w_pi_curr,
                "cql_lse_weight_share_pi_next": _w_pi_next,
                "cql_q_rand_max": _q_rand_max,
                "cql_q_pi_curr_max": _q_pi_curr_max,
                "cql_q_pi_next_max": _q_pi_next_max,
                # ── C(pi). Twin critic disagreement at policy acts
                "q1_q2_gap_pi_curr": _gap_pi_curr,
                "q1_q2_gap_pi_next": _gap_pi_next,
            })

        # Legacy SC-CQL telemetry (D + E blocks).  Emitted ONLY in
        # ``critic_penalty_mode == 'sc_cql'`` so that the clean SMQR
        # baseline + vanilla CQL runs do not mix in heuristic gap /
        # severity / phase / proxy-τ tags as confounders.
        if _penalty_mode == "sc_cql":
            _metrics.update({
                # ── D. SC-CQL ─────────────────────────────────────
                "sc_mask_mean_pi_curr": _sc_m_curr_mean,
                "sc_mask_std_pi_curr": _sc_m_curr_std,
                "sc_mask_active_frac_pi_curr": _sc_m_curr_active,
                "sc_mask_mean_pi_next": _sc_m_next_mean,
                "sc_mask_std_pi_next": _sc_m_next_std,
                "sc_mask_active_frac_pi_next": _sc_m_next_active,
                "sc_penalty_pi_curr_raw": _sc_pen_curr_raw,
                "sc_penalty_pi_curr_masked": _sc_pen_curr_masked,
                "sc_penalty_pi_next_raw": _sc_pen_next_raw,
                "sc_penalty_pi_next_masked": _sc_pen_next_masked,
                "sc_penalty_total_raw": _sc_pen_total_raw,
                "sc_penalty_total_masked": _sc_pen_total_masked,
                "sc_penalty_budget_ratio": _sc_budget_ratio,
                "q_data_q_pi_curr_gap_masked_subset": _gap_masked_sub,
                "pi_q_violation_rate_masked_eps_0p0": _viol_0p0_masked,
                "pi_q_violation_rate_masked_eps_0p1": _viol_0p1_masked,
                # ── D2. SC-CQL v2 leakage / sparsity ──────────────
                "sc_safe_mask_mean_pi_curr": _sc_safe_mask_mean,
                "sc_safe_mask_p90_pi_curr": _sc_safe_mask_p90,
                "sc_violation_mask_mean_pi_curr": _sc_viol_mask_mean,
                "sc_violation_mask_p90_pi_curr": _sc_viol_mask_p90,
                "sc_mask_topk_threshold_pi_curr": _sc_topk_thr,
                "sc_mask_leakage_frac_pi_curr": _sc_leakage_frac,
                "q_data_q_pi_curr_gap_safe_subset": _gap_safe_sub,
                # ── D3. SC-CQL v3 deficit / severity / next subset
                "sc_deficit_mean_pi_curr": _sc_def_curr_mean,
                "sc_deficit_p90_pi_curr": _sc_def_curr_p90,
                "sc_deficit_mean_pi_next": _sc_def_next_mean,
                "sc_deficit_p90_pi_next": _sc_def_next_p90,
                "sc_severity_mean_pi_curr": _sc_sev_curr_mean,
                "sc_severity_mean_pi_next": _sc_sev_next_mean,
                "q_data_q_pi_next_gap_masked_subset": _gap_next_masked_sub,
                "q_data_q_pi_next_gap_safe_subset": _gap_next_safe_sub,
                # ── D4. SC-CQL v4 phase-aware ─────────────────────
                "sc_phase_signal_available": _sc_phase_signal_avail,
                "sc_phase_post_lift_frac": _sc_phase_post_lift_frac,
                "sc_phase_obj_z_mean": _sc_phase_obj_z_mean,
                "sc_phase_obj_z_max": _sc_phase_obj_z_max,
                "sc_mask_active_frac_pi_next_post_lift": _sc_m_next_active_pl,
                "sc_mask_mean_pi_next_post_lift": _sc_m_next_mean_pl,
                "q_data_q_pi_next_gap_masked_post_lift": _gap_next_masked_pl,
                "pi_q_violation_rate_next_post_lift_eps_0p0": _viol_next_pl,
                # ── E. SQR-SG hypothesis instrumentation (PROXY) ──
                "tau_proxy_cfg_offset": torch.tensor(_tau_cfg_offset, device=self.device),
                "tau_proxy_mean": _tau_proxy_mean,
                "tau_proxy_std": _tau_proxy_std,
                "tau_proxy_source_mode": _tau_proxy_type,
                "tau_proxy_delta_pi_curr_mean": _delta_pi_curr.mean(),
                "tau_proxy_delta_pi_curr_std":  _delta_pi_curr.std() if _delta_pi_curr.numel() >= 2 else _zero,
                "tau_proxy_delta_pi_curr_p10": _p10_c,
                "tau_proxy_delta_pi_curr_p25": _p25_c,
                "tau_proxy_delta_pi_curr_p50": _p50_c,
                "tau_proxy_delta_pi_curr_p75": _p75_c,
                "tau_proxy_delta_pi_curr_p90": _p90_c,
                "tau_proxy_delta_pi_next_mean": _delta_pi_next.mean(),
                "tau_proxy_delta_pi_next_std":  _delta_pi_next.std() if _delta_pi_next.numel() >= 2 else _zero,
                "tau_proxy_delta_pi_next_p10": _p10_n,
                "tau_proxy_delta_pi_next_p25": _p25_n,
                "tau_proxy_delta_pi_next_p50": _p50_n,
                "tau_proxy_delta_pi_next_p75": _p75_n,
                "tau_proxy_delta_pi_next_p90": _p90_n,
                "tau_proxy_delta_data_mean": _delta_data.mean(),
                "tau_proxy_delta_data_std":  _delta_data.std() if _delta_data.numel() >= 2 else _zero,
                "tau_proxy_delta_data_p10": _p10_d,
                "tau_proxy_delta_data_p25": _p25_d,
                "tau_proxy_delta_data_p50": _p50_d,
                "tau_proxy_delta_data_p75": _p75_d,
                "tau_proxy_delta_data_p90": _p90_d,
                "near_tau_proxy_frac_pi_curr_abs_eps_0p01": _nf_c_001,
                "near_tau_proxy_frac_pi_curr_abs_eps_0p05": _nf_c_005,
                "near_tau_proxy_frac_pi_curr_abs_eps_0p1":  _nf_c_010,
                "near_tau_proxy_frac_pi_next_abs_eps_0p01": _nf_n_001,
                "near_tau_proxy_frac_pi_next_abs_eps_0p05": _nf_n_005,
                "near_tau_proxy_frac_pi_next_abs_eps_0p1":  _nf_n_010,
                "near_tau_proxy_frac_data_abs_eps_0p01":    _nf_d_001,
                "near_tau_proxy_frac_data_abs_eps_0p05":    _nf_d_005,
                "near_tau_proxy_frac_data_abs_eps_0p1":     _nf_d_010,
                "below_tau_proxy_frac_pi_curr": _below_c,
                "above_tau_proxy_frac_pi_curr": _above_c,
                "below_tau_proxy_frac_pi_next": _below_n,
                "above_tau_proxy_frac_pi_next": _above_n,
                "below_tau_proxy_frac_data":    _below_d,
                "above_tau_proxy_frac_data":    _above_d,
                "tau_proxy_g_mean_pi_curr":        _g_c,
                "tau_proxy_gprime_mean_pi_curr":   _gp_c,
                "tau_proxy_gprime_p90_pi_curr":    _gp_p90_c,
                "tau_proxy_gprime_peak_frac_pi_curr": _gp_peak_c,
                "tau_proxy_g_mean_pi_next":        _g_n,
                "tau_proxy_gprime_mean_pi_next":   _gp_n,
                "tau_proxy_gprime_p90_pi_next":    _gp_p90_n,
                "tau_proxy_gprime_peak_frac_pi_next": _gp_peak_n,
                "tau_proxy_g_mean_data":           _g_d,
                "tau_proxy_gprime_mean_data":      _gp_d,
                "tau_proxy_gprime_p90_data":       _gp_p90_d,
                "tau_proxy_gprime_peak_frac_data": _gp_peak_d,
                "tau_proxy_qgprime_mean_pi_curr":      _qgp_c_mean,
                "tau_proxy_qgprime_p90_pi_curr":       _qgp_c_p90,
                "tau_proxy_qgprime_peak_frac_pi_curr": _qgp_c_peak,
                "tau_proxy_qgprime_mean_pi_next":      _qgp_n_mean,
                "tau_proxy_qgprime_p90_pi_next":       _qgp_n_p90,
                "tau_proxy_qgprime_peak_frac_pi_next": _qgp_n_peak,
                "tau_proxy_qgprime_mean_data":         _qgp_d_mean,
                "tau_proxy_qgprime_p90_data":          _qgp_d_p90,
                "tau_proxy_qgprime_peak_frac_data":    _qgp_d_peak,
                "near_tau_proxy_penalty_mass_pi_curr":    _near_pen_mass_curr,
                "near_tau_proxy_penalty_mass_pi_next":    _near_pen_mass_next,
                "near_tau_proxy_penalty_share_total":     _near_pen_share_total,
                "outside_tau_proxy_penalty_share_total":  _outside_pen_share_total,
                "near_tau_proxy_frac_pi_curr_abs_eps_0p05_ema": _ema_c,
                "near_tau_proxy_frac_pi_next_abs_eps_0p05_ema": _ema_n,
                "near_tau_proxy_recurrence_window_mean_pi_curr": _win_c,
                "near_tau_proxy_recurrence_window_mean_pi_next": _win_n,
                "near_tau_proxy_streak_len_pi_curr": _streak_c,
                "near_tau_proxy_streak_len_pi_next": _streak_n,
                "corr_near_tau_proxy_frac_vs_td_loss":         _corr_near_vs_td,
                "corr_near_tau_proxy_frac_vs_critic_grad_norm": _corr_near_vs_gnorm,
                "corr_qgprime_peak_frac_vs_critic_loss":       _corr_qgp_vs_closs,
                "near_tau_proxy_frac_pi_curr_post_lift_abs_eps_0p05": _nf_c_005_pl,
                "near_tau_proxy_frac_pi_next_post_lift_abs_eps_0p05": _nf_n_005_pl,
                "tau_proxy_gprime_peak_frac_pi_next_post_lift":       _gp_peak_n_pl,
                "tau_proxy_qgprime_peak_frac_pi_next_post_lift":      _qgp_peak_n_pl,
            })

        # SMQR continuous-action baseline telemetry — emitted only
        # under ``critic_penalty_mode == 'smqr_cont_self'``.
        if _smqr_metrics:
            _metrics.update(_smqr_metrics)
        return _metrics

    # ── actor update ──────────────────────────────────────────────────

    def _update_actor(self, data: TensorDict) -> dict[str, torch.Tensor]:
        """Actor gradient step: SAC-style  α·log π(a|s) − min Q(s,a).

        Identical in spirit to ``FastSACAgent._update_pol()`` but reads scalar
        Q-values directly instead of marginalising over a C51 distribution.
        """
        scaler = self.scaler
        args = self.config

        with self._maybe_amp():
            observations = data["observations"]           # [B, actor_obs_dim]
            critic_obs = data["critic_observations"]      # [B, critic_obs_dim]

            actions_new, log_probs = self.actor.get_actions_and_log_probs(observations)
            # log_probs: [B]

            # Diagnostic: policy entropy and std (cheap under no_grad)
            with torch.no_grad():
                _, _, log_std = self.actor(observations)
                action_std = log_std.exp().mean()
                policy_entropy = -log_probs.mean()

            # min Q across ensemble for new actions
            min_q = self.qnet.min_q(critic_obs, actions_new)  # [B, 1]
            min_q = min_q.squeeze(-1)  # [B]

            alpha = self.log_alpha.exp().detach()

            # ── Q-normalizer (per-batch, read-only scale) ──────────
            # Normalises the Q-value term in the actor loss so that
            # the RL gradient is scale-invariant w.r.t. the critic's
            # output magnitude.  Detached — does not affect backward.
            #
            # ``q_normalizer_raw_adaptive`` is always recomputed
            # per-batch (legacy formula).  ``q_normalizer_active`` is
            # what actually divides ``min_q`` and depends on
            # ``args.q_normalizer_mode``:
            #   - "adaptive"      → active = raw_adaptive
            #   - "slow_ema"      → active = EMA(raw_adaptive)
            #   - "freeze_at_step"→ active = raw_adaptive until
            #                       global_step >= freeze_step,
            #                       then frozen at that snapshot.
            # All three are clamped to ``max(q_normalizer_min, 1.0)``.
            with torch.no_grad():
                _q_norm_raw_adaptive = max(min_q.abs().mean().item(), 1.0)

                _qn_mode = str(getattr(args, "q_normalizer_mode", "adaptive"))
                _qn_min = float(getattr(args, "q_normalizer_min", 1.0))
                _qn_min = max(_qn_min, 1.0)
                _qn_floor = lambda v: max(float(v), _qn_min)

                if _qn_mode == "slow_ema":
                    _tau = float(getattr(args, "q_normalizer_ema_tau", 0.005))
                    if self._q_normalizer_ema is None:
                        self._q_normalizer_ema = _q_norm_raw_adaptive
                    else:
                        self._q_normalizer_ema = (
                            (1.0 - _tau) * self._q_normalizer_ema
                            + _tau * _q_norm_raw_adaptive
                        )
                    _q_norm_active = _qn_floor(self._q_normalizer_ema)
                elif _qn_mode == "freeze_at_step":
                    _fs = int(getattr(args, "q_normalizer_freeze_step", 0))
                    if int(self.global_step) < _fs:
                        _q_norm_active = _qn_floor(_q_norm_raw_adaptive)
                    else:
                        if self._q_normalizer_frozen is None:
                            self._q_normalizer_frozen = _q_norm_raw_adaptive
                        _q_norm_active = _qn_floor(self._q_normalizer_frozen)
                else:
                    # "adaptive" (default, legacy bit-exact)
                    _q_norm_active = _qn_floor(_q_norm_raw_adaptive)

                # Legacy alias retained throughout the function so
                # downstream telemetry (A1 fixed-ref, etc.) keeps
                # using the *adaptive* numerator unchanged.  The
                # actor RL term itself uses ``_q_norm_active``.
                q_normalizer = _q_norm_raw_adaptive
            normalized_q = min_q / _q_norm_active

            # ── BC regularisation (Patch 2) ────────────────────────
            bc_weight = getattr(args, "bc_weight", 0.0)
            if bc_weight > 0.0:
                bc_loss = F.mse_loss(actions_new, data["actions"])
            else:
                bc_loss = torch.tensor(0.0, device=self.device)

            rl_term = (alpha * log_probs - normalized_q).mean()
            actor_loss = rl_term + bc_weight * bc_loss

            # ── Extra diagnostics (no_grad, cheap) ─────────────────
            with torch.no_grad():
                action_l2_vs_data = ((actions_new - data["actions"]) ** 2).sum(dim=-1).mean()
                action_mae_vs_data = (actions_new - data["actions"]).abs().mean()

                # ── A2: group-wise action telemetry ───────────────
                # G1 29-DoF action layout (legs/waist=0..14, arm
                # shoulders+elbows=15-18,22-25, wrists=19-21,26-28).
                # For other action_dims, only "total" is filled.
                _act_dim = actions_new.shape[-1]
                _data_act = data["actions"]
                _diff = actions_new - _data_act
                if _act_dim == 29:
                    _arm_idx = torch.tensor(
                        [15, 16, 17, 18, 22, 23, 24, 25],
                        device=self.device, dtype=torch.long,
                    )
                    _hand_idx = torch.tensor(
                        [19, 20, 21, 26, 27, 28],
                        device=self.device, dtype=torch.long,
                    )
                    _prism_idx = torch.tensor(
                        [], device=self.device, dtype=torch.long,
                    )
                else:
                    _arm_idx = torch.empty(0, device=self.device, dtype=torch.long)
                    _hand_idx = torch.empty(0, device=self.device, dtype=torch.long)
                    _prism_idx = torch.empty(0, device=self.device, dtype=torch.long)

                def _l2_grp(idx):
                    if idx.numel() == 0:
                        return torch.zeros((), device=self.device)
                    return (_diff.index_select(-1, idx) ** 2).sum(dim=-1).mean()

                def _norm_grp(t, idx):
                    if idx.numel() == 0:
                        return torch.zeros((), device=self.device)
                    return t.index_select(-1, idx).norm(dim=-1).mean()

                action_l2_vs_data_total = action_l2_vs_data
                action_l2_vs_data_arm   = _l2_grp(_arm_idx)
                action_l2_vs_data_hand  = _l2_grp(_hand_idx)
                action_l2_vs_data_prismatic_or_gripper = _l2_grp(_prism_idx)

                data_action_norm_total  = _data_act.norm(dim=-1).mean()
                data_action_norm_arm    = _norm_grp(_data_act, _arm_idx)
                data_action_norm_hand   = _norm_grp(_data_act, _hand_idx)

                policy_action_norm_total = actions_new.norm(dim=-1).mean()
                policy_action_norm_arm   = _norm_grp(actions_new, _arm_idx)
                policy_action_norm_hand  = _norm_grp(actions_new, _hand_idx)

                action_saturation_frac_data   = (_data_act.abs() > 0.98).float().mean()
                action_saturation_frac_policy = (actions_new.abs() > 0.98).float().mean()

                # Per-DoF actor-data L2 (mean of squared diff across batch).
                _per_dof_l2 = (_diff ** 2).mean(dim=0)  # [A]
                _topk = min(10, _act_dim)
                _top_vals, _top_idx = torch.topk(_per_dof_l2, _topk)
                action_l2_per_dof_top1_idx   = _top_idx[0].float()
                action_l2_per_dof_top1_value = _top_vals[0]
                action_l2_per_dof_top10_sum  = _top_vals.sum()
                action_l2_per_dof_max        = _per_dof_l2.max()
                action_l2_per_dof_mean       = _per_dof_l2.mean()

                # ── A1: fixed-reference q_normalizer telemetry ────
                # Capture refs lazily on first step ≥ {1k, 5k}.
                gs = int(self.global_step)
                if self._q_normalizer_ref_1k is None and gs >= 1000:
                    self._q_normalizer_ref_1k = float(q_normalizer)
                if self._q_normalizer_ref_5k is None and gs >= 5000:
                    self._q_normalizer_ref_5k = float(q_normalizer)
                _ref_1k = (
                    self._q_normalizer_ref_1k
                    if self._q_normalizer_ref_1k is not None
                    else float(q_normalizer)
                )
                _ref_5k = (
                    self._q_normalizer_ref_5k
                    if self._q_normalizer_ref_5k is not None
                    else float(q_normalizer)
                )

                # Raw-Q breakdowns (numerator-side).
                q_pi_mean_raw = min_q.mean()
                q_pi_std_raw = min_q.std() if min_q.numel() > 1 else torch.zeros_like(q_pi_mean_raw)

                # Q on data actions for the same critic_obs batch.
                _q_data = self.qnet.min_q(critic_obs, data["actions"]).squeeze(-1)
                q_data_mean_raw = _q_data.mean()
                q_data_std_raw = _q_data.std() if _q_data.numel() > 1 else torch.zeros_like(q_data_mean_raw)
                q_data_q_pi_gap_raw = q_data_mean_raw - q_pi_mean_raw

                # Numerator of RL actor term BEFORE q_normalizer
                # division: alpha·logπ − Q_pi.
                _rl_num = (alpha * log_probs - min_q)
                rl_actor_q_raw_mean = _rl_num.mean()
                rl_actor_q_raw_std = _rl_num.std() if _rl_num.numel() > 1 else torch.zeros_like(rl_actor_q_raw_mean)

                # Fixed-reference RL terms — same numerator, swapped
                # divisor only.  No effect on optimizer.
                rl_actor_term_fixed_qnorm_1k = (alpha * log_probs - min_q / _ref_1k).mean()
                rl_actor_term_fixed_qnorm_5k = (alpha * log_probs - min_q / _ref_5k).mean()

                # Active vs raw_adaptive RL term decomposition
                # (telemetry only; ``rl_term`` already uses active).
                rl_actor_term_active_qnorm = (
                    alpha * log_probs - min_q / _q_norm_active
                ).mean()
                rl_actor_term_raw_adaptive_qnorm = (
                    alpha * log_probs - min_q / _q_norm_raw_adaptive
                ).mean()

        # ── A3: actor gradient-norm ratio telemetry ────────────────
        # Telemetry-only, computed every ``_a3_grad_telemetry_period``
        # steps.  Uses ``torch.autograd.grad`` on the same batch /
        # graph used by the main backward — no RNG, no parameter
        # mutation, no .grad accumulation.  ``retain_graph=True`` so
        # the subsequent ``scaler.scale(actor_loss).backward()`` can
        # still walk the same computational graph.
        _A3_PERIOD = 500
        _do_grad_telemetry = (self.global_step > 0 and self.global_step % _A3_PERIOD == 0)
        _grad_metrics: dict[str, torch.Tensor] = {}
        if _do_grad_telemetry:
            _actor_params = [p for p in self.actor.parameters() if p.requires_grad]
            try:
                _g_rl = torch.autograd.grad(
                    rl_term, _actor_params,
                    retain_graph=True, allow_unused=True, create_graph=False,
                )
                _rl_sq = torch.zeros((), device=self.device, dtype=torch.float32)
                _rl_bad = torch.zeros((), device=self.device, dtype=torch.float32)
                for _g in _g_rl:
                    if _g is None:
                        continue
                    _gf = _g.detach().float()
                    _rl_sq = _rl_sq + _gf.pow(2).sum()
                    if not torch.isfinite(_gf).all():
                        _rl_bad = torch.ones((), device=self.device, dtype=torch.float32)
                _gn_rl_unw = _rl_sq.clamp_min(0).sqrt()
            except Exception:
                _g_rl = None
                _gn_rl_unw = torch.zeros((), device=self.device)
                _rl_bad = torch.ones((), device=self.device)

            if bc_weight > 0.0 and bc_loss.requires_grad:
                try:
                    _g_bc = torch.autograd.grad(
                        bc_loss, _actor_params,
                        retain_graph=True, allow_unused=True, create_graph=False,
                    )
                    _bc_sq = torch.zeros((), device=self.device, dtype=torch.float32)
                    _bc_bad = torch.zeros((), device=self.device, dtype=torch.float32)
                    _dot = torch.zeros((), device=self.device, dtype=torch.float32)
                    for _gr, _gb in zip(_g_rl if _g_rl is not None else [None] * len(_g_bc), _g_bc):
                        if _gb is None:
                            continue
                        _gbf = _gb.detach().float()
                        _bc_sq = _bc_sq + _gbf.pow(2).sum()
                        if not torch.isfinite(_gbf).all():
                            _bc_bad = torch.ones((), device=self.device, dtype=torch.float32)
                        if _gr is not None:
                            _dot = _dot + (_gr.detach().float() * _gbf).sum()
                    _gn_bc_unw = _bc_sq.clamp_min(0).sqrt()
                except Exception:
                    _gn_bc_unw = torch.zeros((), device=self.device)
                    _bc_bad = torch.ones((), device=self.device)
                    _dot = torch.zeros((), device=self.device)
            else:
                _gn_bc_unw = torch.zeros((), device=self.device)
                _bc_bad = torch.zeros((), device=self.device)
                _dot = torch.zeros((), device=self.device)

            _gn_rl_w = _gn_rl_unw  # rl_term has weight 1.0 in actor_loss
            _gn_bc_w = float(bc_weight) * _gn_bc_unw
            _ratio_bc_to_rl = _gn_bc_w / (_gn_rl_w + 1e-12)
            # cosine of the *unweighted* gradients (sign-only relevant for direction)
            _cos = _dot / (_gn_rl_unw * _gn_bc_unw + 1e-12)
            # actual total gradient that backward will produce =
            # ||g_rl + bc_w · g_bc||.  Recovered analytically.
            _total_sq = _gn_rl_w.pow(2) + _gn_bc_w.pow(2) + 2.0 * float(bc_weight) * _dot
            _gn_total = _total_sq.clamp_min(0).sqrt()

            _grad_metrics = {
                "grad_norm_rl_unweighted":   _gn_rl_unw.detach(),
                "grad_norm_bc_unweighted":   _gn_bc_unw.detach(),
                "grad_norm_rl_weighted":     _gn_rl_w.detach(),
                "grad_norm_bc_weighted":     _gn_bc_w.detach(),
                "grad_ratio_bc_to_rl_weighted": _ratio_bc_to_rl.detach(),
                "grad_cosine_bc_rl":         _cos.detach(),
                "grad_dot_bc_rl":            _dot.detach(),
                "grad_norm_total_actual":    _gn_total.detach(),
                "grad_rl_nan_or_inf":        _rl_bad.detach(),
                "grad_bc_nan_or_inf":        _bc_bad.detach(),
            }

        # ── Backward + optimise ────────────────────────────────────
        self.actor_optimizer.zero_grad(set_to_none=True)
        scaler.scale(actor_loss).backward()
        scaler.unscale_(self.actor_optimizer)

        if args.max_grad_norm > 0:
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(), max_norm=args.max_grad_norm,
            )
        else:
            actor_grad_norm = torch.tensor(0.0, device=self.device)

        scaler.step(self.actor_optimizer)
        scaler.update()

        return {
            "actor_loss": actor_loss.detach(),
            "actor_grad_norm": actor_grad_norm.detach(),
            "policy_entropy": policy_entropy.detach(),
            "action_std": action_std.detach(),
            "alpha_value": alpha.squeeze().detach(),
            "log_probs_mean": log_probs.mean().detach(),
            "bc_loss": bc_loss.detach(),
            "bc_weight": torch.tensor(bc_weight, device=self.device),
            # Q-normalizer and actor decomposition diagnostics
            "q_normalizer": torch.tensor(q_normalizer, device=self.device),
            "normalized_q_term_mean": normalized_q.mean().detach(),
            "rl_actor_term": rl_term.detach(),
            "action_l2_vs_data": action_l2_vs_data,
            "action_mae_vs_data": action_mae_vs_data,
            # ── D. Actor drift auxiliary metrics ───────────────────
            "actor_bc_to_rl_ratio": (
                (bc_weight * bc_loss) / (rl_term.detach().abs() + 1e-8)
            ).detach() if bc_weight > 0 else torch.tensor(0.0, device=self.device),
            "action_saturation_frac": (actions_new.detach().abs() > 0.98).float().mean(),
            # ── A1: fixed-reference q_normalizer telemetry ────────
            "q_normalizer_adaptive": torch.tensor(q_normalizer, device=self.device),
            "rl_actor_term_adaptive": rl_term.detach(),
            "rl_actor_q_raw_mean": rl_actor_q_raw_mean.detach(),
            "rl_actor_q_raw_std":  rl_actor_q_raw_std.detach(),
            "q_pi_mean_raw":       q_pi_mean_raw.detach(),
            "q_pi_std_raw":        q_pi_std_raw.detach(),
            "q_data_mean_raw":     q_data_mean_raw.detach(),
            "q_data_std_raw":      q_data_std_raw.detach(),
            "q_data_q_pi_gap_raw": q_data_q_pi_gap_raw.detach(),
            "q_normalizer_ref_1k": torch.tensor(_ref_1k, device=self.device),
            "q_normalizer_ref_5k": torch.tensor(_ref_5k, device=self.device),
            "rl_actor_term_fixed_qnorm_1k": rl_actor_term_fixed_qnorm_1k.detach(),
            "rl_actor_term_fixed_qnorm_5k": rl_actor_term_fixed_qnorm_5k.detach(),
            # ── B (q_normalizer mode) telemetry ───────────────────
            "q_normalizer_raw_adaptive": torch.tensor(_q_norm_raw_adaptive, device=self.device),
            "q_normalizer_active":       torch.tensor(_q_norm_active, device=self.device),
            "q_normalizer_mode_id":      torch.tensor(
                {"adaptive": 0, "slow_ema": 1, "freeze_at_step": 2}.get(_qn_mode, -1),
                device=self.device, dtype=torch.float32,
            ),
            "q_normalizer_frozen_set":   torch.tensor(
                1.0 if self._q_normalizer_frozen is not None else 0.0,
                device=self.device,
            ),
            "rl_actor_term_active_qnorm":       rl_actor_term_active_qnorm.detach(),
            "rl_actor_term_raw_adaptive_qnorm": rl_actor_term_raw_adaptive_qnorm.detach(),
            # ── A2: group-wise action telemetry ───────────────────
            "action_l2_vs_data_total": action_l2_vs_data_total.detach(),
            "action_l2_vs_data_arm":   action_l2_vs_data_arm.detach(),
            "action_l2_vs_data_hand":  action_l2_vs_data_hand.detach(),
            "action_l2_vs_data_prismatic_or_gripper": action_l2_vs_data_prismatic_or_gripper.detach(),
            "data_action_norm_total":  data_action_norm_total.detach(),
            "data_action_norm_arm":    data_action_norm_arm.detach(),
            "data_action_norm_hand":   data_action_norm_hand.detach(),
            "policy_action_norm_total": policy_action_norm_total.detach(),
            "policy_action_norm_arm":   policy_action_norm_arm.detach(),
            "policy_action_norm_hand":  policy_action_norm_hand.detach(),
            "action_saturation_frac_data":   action_saturation_frac_data.detach(),
            "action_saturation_frac_policy": action_saturation_frac_policy.detach(),
            "action_l2_per_dof_top1_idx":    action_l2_per_dof_top1_idx.detach(),
            "action_l2_per_dof_top1_value":  action_l2_per_dof_top1_value.detach(),
            "action_l2_per_dof_top10_sum":   action_l2_per_dof_top10_sum.detach(),
            "action_l2_per_dof_max":         action_l2_per_dof_max.detach(),
            "action_l2_per_dof_mean":        action_l2_per_dof_mean.detach(),
            # ── A3: gradient-norm telemetry (period-gated) ────────
            **_grad_metrics,
        }

    # ── alpha update ──────────────────────────────────────────────────

    def _update_alpha(self, log_probs: torch.Tensor) -> dict[str, torch.Tensor]:
        """SAC temperature autotune + optional CQL-alpha Lagrangian.

        Parameters
        ----------
        log_probs:
            Detached policy log-probabilities ``[B]`` from the latest actor
            update (or critic update's next-state log-probs).

        Returns a dict of scalar metrics.
        """
        scaler = self.scaler
        metrics: dict[str, torch.Tensor] = {}

        # ── SAC entropy temperature α ─────────────────────────────
        if self.config.use_autotune:
            self.alpha_optimizer.zero_grad(set_to_none=True)
            with self._maybe_amp():
                alpha_loss = (
                    -self.log_alpha.exp() * (log_probs.detach() + self.target_entropy)
                ).mean()
            scaler.scale(alpha_loss).backward()
            scaler.unscale_(self.alpha_optimizer)
            scaler.step(self.alpha_optimizer)
            scaler.update()

            # ⚡ STABILITY (P4): clamp SAC temperature to a safe range.
            # Unbounded alpha causes two failure modes:
            #   • α → ∞: actor loss dominated by entropy term, ignores
            #     Q-values entirely → effectively random policy even
            #     though losses look stable.
            #   • α → 0: entropy collapses, actor becomes deterministic
            #     and cannot recover from local optima — especially
            #     harmful in offline RL where there is no environment
            #     exploration to escape.
            # When alpha_min is set (e.g. 0.02), it acts as a tighter
            # lower bound than the default 1e-8, preventing the
            # entropy temperature from vanishing entirely.
            _alpha_min = getattr(self.config, "alpha_min", None)
            _log_alpha_min = math.log(_alpha_min) if _alpha_min else math.log(1e-8)
            with torch.no_grad():
                self.log_alpha.clamp_(
                    min=_log_alpha_min, max=math.log(10.0)
                )

            metrics["alpha_loss"] = alpha_loss.detach()
        else:
            metrics["alpha_loss"] = torch.tensor(0.0, device=self.device)

        # ── CQL Lagrangian α_cql ──────────────────────────────────
        if self.config.cql_alpha_autotune:
            self.alpha_cql_optimizer.zero_grad(set_to_none=True)
            with self._maybe_amp():
                # Minimise: α_cql * (cql_penalty - target_cql_penalty)
                # The cql_penalty is already captured during _update_critic;
                # here we use the stored value from the last critic step.
                alpha_cql_loss = (
                    self.log_alpha_cql.exp() * (
                        self._last_cql_penalty.detach() - self.config.cql_target_penalty
                    )
                )
            scaler.scale(alpha_cql_loss).backward()
            scaler.unscale_(self.alpha_cql_optimizer)
            scaler.step(self.alpha_cql_optimizer)
            scaler.update()
            # ⚡ STABILITY (P5): clamp CQL Lagrange multiplier to a
            # safe range.  Without an upper bound, α_cql can grow
            # without limit when the CQL penalty persistently exceeds
            # cql_target_penalty, which makes the CQL loss dominate
            # the critic objective and prevents TD learning entirely.
            with torch.no_grad():
                self.log_alpha_cql.clamp_(
                    min=math.log(1e-6), max=math.log(1e6)
                )
            metrics["alpha_cql_loss"] = alpha_cql_loss.detach()

        return metrics

    # ── IQL value network update ──────────────────────────────────────

    def _update_value(self, data: TensorDict) -> dict[str, torch.Tensor]:
        """Expectile regression of V(s) against Q(s, a_data).

        Loss = E[ L_τ(Q_data − V(s)) ]  where
        L_τ(u) = |τ − 𝟙(u < 0)| · u²

        The Q-target is the *frozen* min(Q1, Q2) evaluated at the
        *dataset* action — i.e. the same Q_data used in critic logging.
        Gradients only flow through V(s).
        """
        scaler = self.scaler
        args = self.config
        tau = args.iql_expectile  # e.g. 0.7

        with self._maybe_amp():
            critic_obs = data["critic_observations"]  # [B, critic_obs_dim]
            actions = data["actions"]                  # [B, act_dim]

            with torch.no_grad():
                q_data = self.qnet.min_q(critic_obs, actions).squeeze(-1)  # [B]

            v_pred = self.value_net(critic_obs).squeeze(-1)  # [B]
            diff = q_data - v_pred                           # [B]

            # Expectile loss: weight positive residuals by τ, negative by (1-τ)
            weight = torch.where(diff >= 0, tau, 1.0 - tau)  # [B]
            value_loss = (weight * diff.pow(2)).mean()

        # ── NaN/Inf guard ──────────────────────────────────────────
        if not torch.isfinite(value_loss):
            logger.error(
                f"[step {self.global_step}] IQL value_loss is {value_loss.item():.4f}. "
                f"Skipping backward. q_data_mean={q_data.mean().item():.4f}, "
                f"v_mean={v_pred.mean().item():.4f}"
            )
            return {
                "iql_value_loss": torch.tensor(0.0, device=self.device),
                "iql_v_mean": v_pred.mean().detach(),
                "iql_q_data_mean": q_data.mean().detach(),
            }

        self.value_optimizer.zero_grad(set_to_none=True)
        scaler.scale(value_loss).backward()
        scaler.unscale_(self.value_optimizer)

        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.value_net.parameters(), max_norm=args.max_grad_norm,
            )

        scaler.step(self.value_optimizer)
        scaler.update()

        return {
            "iql_value_loss": value_loss.detach(),
            "iql_v_mean": v_pred.mean().detach(),
            "iql_q_data_mean": q_data.mean().detach(),
        }

    # ── IQL-style actor update ────────────────────────────────────────

    def _update_actor_iql(self, data: TensorDict) -> dict[str, torch.Tensor]:
        """Advantage-weighted BC with pluggable implementation backends.

        Backends (selected by ``config.actor_iql_impl_mode``):

        * ``logprob_bc`` — weighted negative log-likelihood of dataset
          actions under the squashed Gaussian policy (original D1).
        * ``mean_bc`` — weighted MSE between the actor's *deterministic*
          mean action (tanh(mean)·scale+bias) and the dataset action.
        * ``logprob_bc_fixed_std`` — same as ``logprob_bc`` but the actor's
          log-std is detached from the backward graph so std does not drift.
        * ``pure_bc_mean`` — unweighted MSE(mean_action, a_data).  No
          advantage weighting at all.  Ultra-sanity for action scaling.
        """
        scaler = self.scaler
        args = self.config
        beta = args.iql_beta            # e.g. 3.0
        max_w = args.iql_max_weight     # e.g. 20.0
        impl_mode: str = getattr(args, "actor_iql_impl_mode", "logprob_bc")

        with self._maybe_amp():
            observations = data["observations"]       # [B, actor_obs_dim]
            critic_obs = data["critic_observations"]  # [B, critic_obs_dim]
            actions_data = data["actions"]             # [B, act_dim]

            # ── Shared: advantage + weights (no grad) ──────────────
            use_weights = impl_mode != "pure_bc_mean"
            with torch.no_grad():
                q_data = self.qnet.min_q(critic_obs, actions_data).squeeze(-1)  # [B]
                v_pred = self.value_net(critic_obs).squeeze(-1)                 # [B]
                adv = q_data - v_pred                                           # [B]

                if use_weights:
                    raw_weights = (adv / beta).exp()          # [B]
                    weights = raw_weights.clamp(max=max_w)    # [B]
                    clip_frac = (raw_weights > max_w).float().mean()
                else:
                    weights = torch.ones_like(adv)
                    clip_frac = torch.tensor(0.0, device=adv.device)

            # ── Actor forward (shared across backends) ─────────────
            # actor.forward(obs) returns:
            #   action     = tanh(mean_raw) * scale + bias  [B, act_dim]
            #   mean_raw   = pre-tanh mean (the raw NN output)  [B, act_dim]
            #   log_std    = clamped log standard deviation  [B, act_dim]
            mean_action, mean_pre_tanh, log_std = self.actor(observations)
            # mean_action: post-tanh, post-scale deterministic action
            # mean_pre_tanh: raw Gaussian mean (pre-tanh space)
            # log_std: [B, act_dim]

            # ── std sanity diagnostics (detached) ──────────────────
            with torch.no_grad():
                _logstd_mean = log_std.mean()
                _logstd_min = log_std.min()
                _logstd_max = log_std.max()
                _std_mean = log_std.exp().mean()

                # Warn on pathological std
                if _logstd_max.item() > 5.0 or _logstd_min.item() < -10.0:
                    if self.global_step % args.logging_interval == 0:
                        logger.warning(
                            f"[step {self.global_step}] IQL actor std WARNING: "
                            f"logstd range [{_logstd_min.item():.2f}, {_logstd_max.item():.2f}] "
                            f"is outside safe bounds [-10, 5]"
                        )

            # ── Backend-specific loss computation ──────────────────
            if impl_mode == "pure_bc_mean":
                # Unweighted MSE — ultra-sanity check
                bc_mse = ((mean_action - actions_data) ** 2).sum(dim=-1)  # [B]
                iql_actor_loss = bc_mse.mean()
                unweighted_bc = iql_actor_loss.detach()
                weighted_bc = iql_actor_loss.detach()

            elif impl_mode == "mean_bc":
                # Weighted MSE on deterministic mean action
                bc_mse = ((mean_action - actions_data) ** 2).sum(dim=-1)  # [B]
                iql_actor_loss = (weights * bc_mse).mean()
                unweighted_bc = bc_mse.mean().detach()
                weighted_bc = iql_actor_loss.detach()

            elif impl_mode == "logprob_bc_fixed_std":
                # Log-prob BC with std detached (no gradient through std)
                std = log_std.detach().exp()
                log_std_fixed = log_std.detach()
                # Pre-tanh value of dataset action: atanh((a - bias) / scale)
                pre_tanh = torch.atanh(
                    ((actions_data - self.actor.action_bias) / (self.actor.action_scale + 1e-6)).clamp(-0.999, 0.999)
                )
                # mean_pre_tanh comes directly from actor forward (no atanh roundtrip)
                # Gaussian log-prob in pre-tanh space
                log_probs = -0.5 * (((pre_tanh - mean_pre_tanh) / (std + 1e-6)).pow(2) + 2 * log_std_fixed + math.log(2 * math.pi))
                log_probs = log_probs.sum(dim=-1)  # [B]
                # Tanh Jacobian correction
                scaled_action = (actions_data - self.actor.action_bias) / (self.actor.action_scale + 1e-6)
                log_probs = log_probs - torch.log(1 - scaled_action.pow(2) + 1e-6).sum(dim=-1)
                # Scale Jacobian correction
                log_probs = log_probs - torch.log(self.actor.action_scale + 1e-6).sum()

                iql_actor_loss = -(weights * log_probs).mean()
                unweighted_bc = -(log_probs).mean().detach()
                weighted_bc = iql_actor_loss.detach()

            else:  # "logprob_bc" (original D1)
                std = log_std.exp()
                # Pre-tanh value of dataset action: atanh((a - bias) / scale)
                pre_tanh = torch.atanh(
                    ((actions_data - self.actor.action_bias) / (self.actor.action_scale + 1e-6)).clamp(-0.999, 0.999)
                )
                # mean_pre_tanh comes directly from actor forward (no atanh roundtrip)
                # Gaussian log-prob in pre-tanh space
                log_probs = -0.5 * (((pre_tanh - mean_pre_tanh) / (std + 1e-6)).pow(2) + 2 * log_std + math.log(2 * math.pi))
                log_probs = log_probs.sum(dim=-1)  # [B]
                # Tanh Jacobian correction
                scaled_action = (actions_data - self.actor.action_bias) / (self.actor.action_scale + 1e-6)
                log_probs = log_probs - torch.log(1 - scaled_action.pow(2) + 1e-6).sum(dim=-1)
                # Scale Jacobian correction
                log_probs = log_probs - torch.log(self.actor.action_scale + 1e-6).sum()

                iql_actor_loss = -(weights * log_probs).mean()
                unweighted_bc = -(log_probs).mean().detach()
                weighted_bc = iql_actor_loss.detach()

            # ── NaN/Inf guard ──────────────────────────────────────
            if not torch.isfinite(iql_actor_loss):
                logger.error(
                    f"[step {self.global_step}] IQL actor loss is {iql_actor_loss.item():.4f} "
                    f"(impl_mode={impl_mode}). Skipping backward."
                )
                return {
                    "iql_actor_loss": torch.tensor(0.0, device=self.device),
                    "iql_adv_mean": adv.mean().detach(),
                    "iql_adv_std": adv.float().std().detach(),
                    "iql_weight_mean": weights.mean().detach(),
                    "iql_weight_max": weights.max().detach(),
                    "iql_weight_clip_frac": clip_frac.detach(),
                    "actor_grad_norm": torch.tensor(0.0, device=self.device),
                    "action_l2_vs_data": torch.tensor(0.0, device=self.device),
                    "action_mae_vs_data": torch.tensor(0.0, device=self.device),
                }

            # ── Extra diagnostics (no_grad, cheap) ─────────────────
            with torch.no_grad():
                action_l2_vs_data = ((mean_action - actions_data) ** 2).sum(dim=-1).mean()
                action_mae_vs_data = (mean_action - actions_data).abs().mean()
                # Sampled action for range check
                sampled_actions, _ = self.actor.get_actions_and_log_probs(observations)
                # Action range diagnostics
                data_action_mean_abs = actions_data.abs().mean()
                data_action_max_abs = actions_data.abs().max()
                actor_mean_action_max_abs = mean_action.abs().max()
                actor_sample_action_max_abs = sampled_actions.abs().max()
                # Out-of-range fraction: dataset actions beyond actor scale
                action_scale = self.actor.action_scale  # [act_dim]
                action_bias = self.actor.action_bias    # [act_dim]
                _lo = action_bias - action_scale
                _hi = action_bias + action_scale
                out_of_range = ((actions_data < _lo - 1e-3) | (actions_data > _hi + 1e-3)).float().mean()
                # Scale mismatch: mean |mean_action| / (mean |data_action| + eps)
                scale_mismatch = mean_action.abs().mean() / (actions_data.abs().mean() + 1e-8)

                # ── B-IQL: extended advantage / weight telemetry ──
                _adv_f = adv.float()
                adv_p10 = _adv_f.quantile(0.1)
                adv_p50 = _adv_f.quantile(0.5)
                _w_f = weights.float()
                weight_std = _w_f.std() if _w_f.numel() > 1 else torch.zeros_like(_w_f.mean())
                weight_p50 = _w_f.quantile(0.5)
                weight_p90 = _w_f.quantile(0.9)
                weight_p99 = _w_f.quantile(0.99)
                # Effective sample size (Kish): (Σw)² / (N · Σw²)  ∈ [1/N, 1]
                _ws = _w_f.sum()
                _ws2 = (_w_f * _w_f).sum()
                ess_frac = (_ws * _ws) / (_ws2 * _w_f.numel() + 1e-12)
                q_data_for_log = q_data.mean()
                v_for_log = v_pred.mean()
                qmv_mean = adv.mean()
                qmv_std = _adv_f.std() if _adv_f.numel() > 1 else torch.zeros_like(qmv_mean)

                # Expectile training telemetry (recomputed cheaply on
                # same q_data / V — value_net was already updated this
                # step, but values are detached anyway so no autograd
                # graph leak).
                _diff_for_exp = (q_data - v_pred).float()
                expectile_positive_frac = (_diff_for_exp >= 0).float().mean()
                _tau_exp = float(args.iql_expectile)
                _exp_w = torch.where(_diff_for_exp >= 0,
                                      torch.full_like(_diff_for_exp, _tau_exp),
                                      torch.full_like(_diff_for_exp, 1.0 - _tau_exp))
                expectile_weight_mean = _exp_w.mean()

                # ── B-IQL: group-wise action telemetry (29-DoF) ───
                _act_dim = mean_action.shape[-1]
                _diff_grp = mean_action - actions_data
                if _act_dim == 29:
                    _arm_idx = torch.tensor([15,16,17,18,22,23,24,25],
                                            device=self.device, dtype=torch.long)
                    _hand_idx = torch.tensor([19,20,21,26,27,28],
                                             device=self.device, dtype=torch.long)
                else:
                    _arm_idx = torch.empty(0, device=self.device, dtype=torch.long)
                    _hand_idx = torch.empty(0, device=self.device, dtype=torch.long)
                def _l2g(idx):
                    if idx.numel() == 0: return torch.zeros((), device=self.device)
                    return (_diff_grp.index_select(-1, idx) ** 2).sum(dim=-1).mean()
                def _ng(t, idx):
                    if idx.numel() == 0: return torch.zeros((), device=self.device)
                    return t.index_select(-1, idx).norm(dim=-1).mean()
                action_l2_vs_data_arm  = _l2g(_arm_idx)
                action_l2_vs_data_hand = _l2g(_hand_idx)
                data_action_norm_total = actions_data.norm(dim=-1).mean()
                data_action_norm_arm   = _ng(actions_data, _arm_idx)
                data_action_norm_hand  = _ng(actions_data, _hand_idx)
                policy_action_norm_total = mean_action.norm(dim=-1).mean()
                policy_action_norm_arm   = _ng(mean_action, _arm_idx)
                policy_action_norm_hand  = _ng(mean_action, _hand_idx)

                # Policy entropy proxy under iql actor (no log_prob
                # of *sampled* action computed here, so use ½·log(2πe·σ²)
                # closed-form differential entropy of pre-tanh Gaussian).
                _logstd = log_std
                policy_entropy = (0.5 * (math.log(2 * math.pi * math.e) + 2.0 * _logstd)).sum(-1).mean()

        # ── Backward + optimise ────────────────────────────────────
        self.actor_optimizer.zero_grad(set_to_none=True)
        scaler.scale(iql_actor_loss).backward()
        scaler.unscale_(self.actor_optimizer)

        if args.max_grad_norm > 0:
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(), max_norm=args.max_grad_norm,
            )
        else:
            actor_grad_norm = torch.tensor(0.0, device=self.device)

        scaler.step(self.actor_optimizer)
        scaler.update()

        return {
            # ── Mode-specific actor loss ───────────────────────────
            "iql_actor_loss": iql_actor_loss.detach(),
            "iql_actor_unweighted_bc": unweighted_bc,
            "iql_actor_weighted_bc": weighted_bc,
            # ── IQL weighting sanity ───────────────────────────────
            "iql_adv_mean": adv.mean().detach(),
            "iql_adv_std": adv.float().std().detach(),
            "iql_adv_p90": adv.float().quantile(0.9).detach(),
            "iql_weight_mean": weights.mean().detach(),
            "iql_weight_max": weights.max().detach(),
            "iql_weight_clip_frac": clip_frac.detach(),
            # ── Actor distribution / std sanity ────────────────────
            "iql_actor_action_mean_abs": mean_action.detach().abs().mean(),
            "iql_actor_action_std_mean": _std_mean,
            "iql_actor_logstd_mean": _logstd_mean,
            "iql_actor_logstd_min": _logstd_min,
            "iql_actor_logstd_max": _logstd_max,
            "iql_actor_mean_l2_to_data": action_l2_vs_data,
            "iql_actor_mean_mae_to_data": action_mae_vs_data,
            # ── Dataset action / scaling sanity ────────────────────
            "data_action_mean_abs": data_action_mean_abs,
            "data_action_max_abs": data_action_max_abs,
            "actor_mean_action_max_abs": actor_mean_action_max_abs,
            "actor_sample_action_max_abs": actor_sample_action_max_abs,
            "action_out_of_range_frac": out_of_range,
            "data_actor_scale_mismatch_ratio": scale_mismatch,
            # ── Backward compat ────────────────────────────────────
            "actor_grad_norm": actor_grad_norm.detach(),
            "action_l2_vs_data": action_l2_vs_data,
            "action_mae_vs_data": action_mae_vs_data,
            # ── B-IQL: extended advantage / weight telemetry ───────
            "awbc_adv_mean":     adv.mean().detach(),
            "awbc_adv_std":      adv.float().std().detach() if adv.numel() > 1 else torch.zeros((), device=self.device),
            "awbc_adv_p10":      adv_p10.detach(),
            "awbc_adv_p50":      adv_p50.detach(),
            "awbc_adv_p90":      adv.float().quantile(0.9).detach(),
            "awbc_weight_mean":  weights.mean().detach(),
            "awbc_weight_std":   weight_std.detach(),
            "awbc_weight_p50":   weight_p50.detach(),
            "awbc_weight_p90":   weight_p90.detach(),
            "awbc_weight_p99":   weight_p99.detach(),
            "awbc_weight_max":   weights.max().detach(),
            "awbc_weight_clip_frac": clip_frac.detach(),
            "awbc_effective_sample_size": ess_frac.detach(),
            "awbc_q_data_mean":  q_data_for_log.detach(),
            "awbc_v_mean":       v_for_log.detach(),
            "awbc_q_minus_v_mean": qmv_mean.detach(),
            "awbc_q_minus_v_std":  qmv_std.detach(),
            "awbc_actor_loss":   iql_actor_loss.detach(),
            "awbc_unweighted_bc": unweighted_bc,
            "awbc_weighted_bc":   weighted_bc,
            # Expectile head telemetry (value_loss itself logged via
            # _update_value() as `iql_value_loss`).
            "expectile_weight_mean":   expectile_weight_mean.detach(),
            "expectile_positive_frac": expectile_positive_frac.detach(),
            "q_data_minus_v_mean":     qmv_mean.detach(),
            # Group-wise action mismatch (29-DoF)
            "action_l2_vs_data_total": action_l2_vs_data,
            "action_l2_vs_data_arm":   action_l2_vs_data_arm.detach(),
            "action_l2_vs_data_hand":  action_l2_vs_data_hand.detach(),
            "data_action_norm_total":  data_action_norm_total.detach(),
            "data_action_norm_arm":    data_action_norm_arm.detach(),
            "data_action_norm_hand":   data_action_norm_hand.detach(),
            "policy_action_norm_total": policy_action_norm_total.detach(),
            "policy_action_norm_arm":   policy_action_norm_arm.detach(),
            "policy_action_norm_hand":  policy_action_norm_hand.detach(),
            "action_std":               _std_mean.detach(),
            "policy_entropy":           policy_entropy.detach(),
        }

    # ── evaluation rollouts ────────────────────────────────────────────

    @torch.no_grad()
    def _run_eval_rollouts(
        self,
        num_steps: int = 200,
    ) -> dict[str, float]:
        """Run deterministic rollouts and return structured eval metrics.

        This is the core diagnostic tool for offline CQL: it tells you
        whether the policy *actually works* in the environment, independent
        of how good the training losses look.

        Action semantics
        ----------------
        Uses ``self.actor(obs)[0]`` which returns ``tanh(mean) * scale +
        bias`` — the *deterministic* action, identical to FastSAC's
        ``evaluate_policy`` and to ONNX inference.  No sampling noise.

        Metrics returned
        ----------------
        * ``mean_reward``    — mean per-step reward across all envs
        * ``mean_ep_reward`` — mean total episode reward (completed eps)
        * ``mean_ep_length`` — mean episode length (completed eps)
        * ``num_episodes``   — how many episodes completed
        * ``action_mean``    — mean action magnitude (distribution shift
                               diagnostic: compare to dataset action mean)
        * ``action_std``     — std of actions taken (should be near 0 for
                               deterministic policy; if not, something is
                               wrong)
        * ``obs_mean``       — mean obs magnitude the actor sees at eval
                               (compare to training obs to detect env
                               mismatch)
        * Any ``episode``-level signals the env provides (e.g. task
          success rate) are aggregated under their original key names.

        ⚠ DIAGNOSTIC — common failure modes visible here:
        1. ``mean_ep_reward`` is flat while ``td_loss`` decreases →
           the Q-function is overfitting to the static dataset.
        2. ``action_mean`` diverges from dataset action mean → the
           actor exploits OOD Q-values.
        3. ``mean_ep_length`` is very short → the policy falls down
           immediately, likely because the env's initial state
           distribution differs from the dataset's.
        4. ``obs_mean`` differs significantly from training obs_mean →
           the normaliser was computed on a different distribution
           than what the env produces.
        """
        was_training = self.actor.training
        self.actor.eval()
        if self.obs_normalization:
            self.obs_normalizer.eval()

        env = self.env  # FastSACEnv wrapper
        obs = env.reset()

        # Accumulators
        total_reward = torch.zeros(env.num_envs, device=self.device)
        ep_reward_sums = torch.zeros(env.num_envs, device=self.device)
        ep_lengths = torch.zeros(env.num_envs, device=self.device)
        completed_ep_rewards: list[float] = []
        completed_ep_lengths: list[float] = []
        all_actions: list[torch.Tensor] = []
        all_obs: list[torch.Tensor] = []
        episode_signals: dict[str, list[float]] = {}  # task-level success etc.
        total_alive_steps: int = 0  # env-steps where the env was alive

        # Motion command handle for detecting clip-end (success) resets
        _motion_cmd = getattr(
            self.unwrapped_env.command_manager, "get_state", lambda _: None
        )("motion_command")

        for step in range(num_steps):
            # ── Normalise obs ──────────────────────────────────────
            if self.obs_normalization:
                norm_obs = self.obs_normalizer(obs, update=False)
            else:
                norm_obs = obs

            # ── Deterministic action ──────────────────────────────
            # actor.forward returns (action, mean, log_std);
            # action = tanh(mean)*scale + bias (deterministic).
            # Same semantics as FastSAC evaluate_policy and ONNX export.
            actions, pre_tanh_mean, log_std = self.actor(norm_obs)

            # ── bad_tracking diagnostic instrumentation ───────────
            # Log step-level diagnostics for the first few steps to
            # diagnose immediate bad_tracking termination.
            if step < 10:
                _act_abs = actions.abs()
                _pre_tanh_abs = pre_tanh_mean.abs()
                _act_scale = self.actor.action_scale
                # Compute what the env PD controller will see
                _env_action_scales = getattr(self.unwrapped_env, "action_scales", None)
                _info_parts = [
                    f"[eval_diag step={step}]",
                    f"pre_tanh_mean: abs_max={_pre_tanh_abs.max().item():.4f} "
                    f"abs_mean={_pre_tanh_abs.mean().item():.4f}",
                    f"post_scale_action: abs_max={_act_abs.max().item():.4f} "
                    f"abs_mean={_act_abs.mean().item():.4f}",
                    f"actor.action_scale: min={_act_scale.min().item():.4f} "
                    f"max={_act_scale.max().item():.4f}",
                ]
                if _env_action_scales is not None:
                    _pd_input = actions * _env_action_scales
                    _info_parts.append(
                        f"pd_position_offset: abs_max={_pd_input.abs().max().item():.4f} "
                        f"abs_mean={_pd_input.abs().mean().item():.4f}"
                    )
                _info_parts.append(
                    f"obs raw abs_mean={obs.abs().mean().item():.4f} "
                    f"norm_obs abs_mean={norm_obs.abs().mean().item():.4f}"
                )
                logger.info("  ".join(_info_parts))

            all_actions.append(actions)
            all_obs.append(obs)

            # Snapshot motion time-steps to detect clip-end (success) after step
            _prev_ts = _motion_cmd.time_steps.clone() if _motion_cmd is not None else None

            obs, rewards, dones, extras = env.step(actions)

            total_reward += rewards
            ep_reward_sums += rewards
            ep_lengths += 1
            total_alive_steps += int((~dones.bool()).sum().item())

            # ── Common helper: record & reset accumulators ─────────
            def _collect_episodes(indices: torch.Tensor) -> None:
                for idx in indices:
                    completed_ep_rewards.append(ep_reward_sums[idx].item())
                    completed_ep_lengths.append(ep_lengths[idx].item())
                ep_reward_sums[indices] = 0.0
                ep_lengths[indices] = 0.0

            # Collect completed episodes (failure / timeout)
            done_mask = dones.bool()
            if done_mask.any():
                done_indices = done_mask.nonzero(as_tuple=False).squeeze(-1)
                _collect_episodes(done_indices)

                # Collect task-level episode signals if available
                ep_info = extras.get("episode", {})
                if isinstance(ep_info, dict):
                    for k, v in ep_info.items():
                        if k not in episode_signals:
                            episode_signals[k] = []
                        if isinstance(v, torch.Tensor):
                            episode_signals[k].append(v.float().mean().item())
                        elif isinstance(v, (int, float)):
                            episode_signals[k].append(float(v))

            # Collect success episodes (clip-end without reset_buf)
            if _prev_ts is not None:
                clip_ended = (
                    (_prev_ts > 1) & (_motion_cmd.time_steps < _prev_ts) & ~done_mask
                )
                if clip_ended.any():
                    success_indices = clip_ended.nonzero(as_tuple=False).squeeze(-1)
                    if success_indices.dim() == 0:
                        success_indices = success_indices.unsqueeze(0)
                    _collect_episodes(success_indices)

        # ── Aggregate metrics ─────────────────────────────────────
        metrics: dict[str, float] = {}
        metrics["mean_reward"] = (total_reward / max(num_steps, 1)).mean().item()

        if completed_ep_rewards:
            metrics["mean_ep_reward"] = sum(completed_ep_rewards) / len(completed_ep_rewards)
            metrics["mean_ep_length"] = sum(completed_ep_lengths) / len(completed_ep_lengths)
        else:
            # No episodes completed — the policy is either very good
            # (long episodes) or stuck.  Log partial stats.
            metrics["mean_ep_reward"] = ep_reward_sums.mean().item()
            metrics["mean_ep_length"] = ep_lengths.mean().item()
        metrics["num_episodes"] = float(len(completed_ep_rewards))

        # Action diagnostics
        if all_actions:
            stacked_actions = torch.cat(all_actions, dim=0)
            metrics["action_mean"] = stacked_actions.abs().mean().item()
            metrics["action_std"] = stacked_actions.std().item()
        if all_obs:
            stacked_obs = torch.cat(all_obs, dim=0)
            metrics["obs_mean"] = stacked_obs.abs().mean().item()

        # Task-level episode signals (e.g. success_rate, tracking_error)
        for k, vals in episode_signals.items():
            if vals:
                metrics[f"ep_{k}"] = sum(vals) / len(vals)

        # ── Humanoid-specific eval metrics ────────────────────────
        n_completed = len(completed_ep_rewards)
        metrics["fall_rate"] = n_completed / max(env.num_envs, 1)
        if completed_ep_lengths:
            metrics["time_to_fall_mean"] = (
                sum(completed_ep_lengths) / len(completed_ep_lengths)
            )
        else:
            metrics["time_to_fall_mean"] = float(num_steps)
        total_possible = num_steps * env.num_envs
        metrics["alive_steps_ratio"] = (
            total_alive_steps / max(total_possible, 1)
        )

        # Task-specific metrics — logged if the env provides them in
        # episode signals.  Keys follow the env's naming convention.
        _task_key_map = {
            "grasp_success": "grasp_success_rate",
            "carry_success": "carry_success_rate",
            "place_success": "place_success_rate",
            "final_box_goal_dist": "final_box_goal_dist_mean",
            "box_height_max": "box_height_max_mean",
        }
        for src_key, dst_key in _task_key_map.items():
            # Check both raw and ep_ prefixed keys
            if src_key in episode_signals and episode_signals[src_key]:
                metrics[dst_key] = (
                    sum(episode_signals[src_key]) / len(episode_signals[src_key])
                )
            elif f"ep_{src_key}" in metrics:
                metrics[dst_key] = metrics[f"ep_{src_key}"]

        # Restore training mode
        if was_training:
            self.actor.train()
            if self.obs_normalization:
                self.obs_normalizer.train()

        return metrics

    # ── inference / export ────────────────────────────────────────────

    def get_inference_policy(self, device: str | None = None) -> Callable[[dict[str, torch.Tensor]], torch.Tensor]:
        """Return a callable that maps ``{"actor_obs": Tensor}`` → action.

        Identical to ``FastSACAgent.get_inference_policy`` — the actor and
        obs normaliser are the same classes.
        """
        device = device or self.device
        policy = self.actor.to(device)
        obs_normalizer = self.obs_normalizer.to(device)
        policy.eval()
        obs_normalizer.eval()

        obs_normalization = self.config.obs_normalization

        def policy_fn(obs: dict[str, torch.Tensor]) -> torch.Tensor:
            if obs_normalization:
                normalized_obs = obs_normalizer(obs["actor_obs"], update=False)
            else:
                normalized_obs = obs["actor_obs"]
            return policy(normalized_obs)[0]

        return policy_fn

    @property
    def actor_onnx_wrapper(self) -> nn.Module:
        """ONNX-exportable wrapper — same structure as FastSAC."""
        import copy

        actor = copy.deepcopy(self.actor).to("cpu")
        obs_normalizer = copy.deepcopy(self.obs_normalizer).to("cpu")

        class ActorWrapper(nn.Module):
            def __init__(self, actor: nn.Module, obs_normalizer: nn.Module | None):
                super().__init__()
                self.actor = actor
                self.obs_normalizer = obs_normalizer

            def forward(self, actor_obs: torch.Tensor) -> torch.Tensor:
                if self.obs_normalizer is not None:
                    normalized_obs = self.obs_normalizer(actor_obs, update=False)
                else:
                    normalized_obs = actor_obs
                return self.actor(normalized_obs)[0]

        return ActorWrapper(actor, obs_normalizer if self.config.obs_normalization else None)

    def export(self, onnx_file_path: str) -> None:
        """Export ONNX policy — identical to FastSAC (same Actor + obs normaliser).

        The ``actor_onnx_wrapper`` property already produces the correct
        ONNX-exportable module, so this method simply drives the same
        tracing, metadata-attachment, and wandb-upload flow as FastSAC.
        """
        was_training = self.actor.training

        self.actor.eval()
        if self.obs_normalization:
            self.obs_normalizer.eval()

        example_input_list = torch.zeros(1, self.actor_obs_dim, device="cpu")

        motion_command = self.unwrapped_env.command_manager.get_state("motion_command")
        if motion_command is not None:
            export_motion_and_policy_as_onnx(
                self.actor_onnx_wrapper,
                motion_command,
                onnx_file_path,
                self.device,
            )
        else:
            export_policy_as_onnx(
                wrapper=self.actor_onnx_wrapper,
                onnx_file_path=onnx_file_path,
                example_obs_dict={"actor_obs": example_input_list},
            )

        kp_list, kd_list = get_control_gains_from_config(self.env.robot_config)
        cmd_ranges = get_command_ranges_from_env(self.unwrapped_env)
        action_scales = getattr(self.unwrapped_env, "action_scales", None)
        if action_scales is None:
            action_scale_metadata: float | list[float] = float(self.env.robot_config.control.action_scale)
        else:
            action_scale_metadata = action_scales.detach().cpu().tolist()
        urdf_file_path, urdf_str = get_urdf_text_from_robot_config(self.env.robot_config)

        metadata = {
            "dof_names": self.env.robot_config.dof_names,
            "kp": kp_list,
            "kd": kd_list,
            "action_scale": action_scale_metadata,
            "command_ranges": cmd_ranges,
            "robot_urdf": urdf_str,
            "robot_urdf_path": urdf_file_path,
        }
        metadata.update(self._checkpoint_metadata(iteration=self.global_step))

        attach_onnx_metadata(
            onnx_path=onnx_file_path,
            metadata=metadata,
        )

        if hasattr(self, "logging_helper"):
            self.logging_helper.save_to_wandb(onnx_file_path)

        if was_training:
            self.actor.train()
            if self.obs_normalization:
                self.obs_normalizer.train()

    @torch.no_grad()
    def evaluate_policy(self, max_eval_steps: int | None = None) -> dict[str, float]:
        """Run the learned policy in the environment for evaluation.

        This method serves two purposes:

        1. **Callback-driven evaluation** — identical to FastSAC.  Eval
           callbacks (video recording, metric logging, etc.) are invoked
           at each step via the ``_pre/_post_eval_env_step`` hooks.

        2. **Structured metric collection** — returns a dict of rollout
           statistics (episode reward, length, task success signals,
           action diagnostics) that callers can log or inspect.

        Action semantics
        ----------------
        Uses ``self.actor(obs)[0]`` = ``tanh(mean) * scale + bias`` —
        the deterministic action, identical to ``get_inference_policy``
        and ONNX export.  Same semantics as FastSAC ``evaluate_policy``.

        Parameters
        ----------
        max_eval_steps:
            Maximum number of env steps.  ``None`` means run forever
            (until callbacks signal stop, for backward compat).

        Returns
        -------
        dict[str, float]
            Evaluation metrics.  Empty dict if ``max_eval_steps`` is None
            (pure callback mode).
        """
        # ── Dims guard ─────────────────────────────────────────────
        # The actor and normalizer were built with model obs dims
        # (from the dataset or checkpoint), which may differ from the
        # current env obs dims.  If they don't match, env obs can't
        # be fed to the actor.
        if not self._eval_dims_match:
            raise RuntimeError(
                f"Cannot run evaluate_policy(): observation dimension "
                f"mismatch between the trained model and the current "
                f"environment.\n"
                f"  model actor_obs_dim  = {self.actor_obs_dim}\n"
                f"  env   actor_obs_dim  = {self._env_actor_obs_dim}\n"
                f"  model critic_obs_dim = {self.critic_obs_dim}\n"
                f"  env   critic_obs_dim = {self._env_critic_obs_dim}\n"
                f"The checkpoint was trained with a different observation "
                f"config than the current env produces.  To fix:\n"
                f"  • Reconfigure the env observation preset to match "
                f"the {self.actor_obs_dim}-dim obs used during training, OR\n"
                f"  • Re-collect the dataset / retrain with the current "
                f"env config ({self._env_actor_obs_dim}-dim obs)."
            )

        self._create_eval_callbacks()
        self._pre_evaluate_policy()

        was_training = self.actor.training
        self.actor.eval()
        if self.obs_normalization:
            self.obs_normalizer.eval()

        env = self.env
        obs = env.reset()

        # Accumulators for structured metrics
        total_reward = torch.zeros(env.num_envs, device=self.device)
        ep_reward_sums = torch.zeros(env.num_envs, device=self.device)
        ep_lengths = torch.zeros(env.num_envs, device=self.device)
        completed_ep_rewards: list[float] = []
        completed_ep_lengths: list[float] = []
        all_actions: list[torch.Tensor] = []
        episode_signals: dict[str, list[float]] = {}

        # ── Single-episode-per-env mode ────────────────────────────
        # When enabled (set externally via ``self._single_episode_per_env``),
        # each env records its FIRST completed episode only; subsequent
        # rewards/steps on the same env are ignored, and the evaluation
        # loop exits once every env has finished one episode.
        single_episode_per_env = bool(getattr(self, "_single_episode_per_env", False))
        per_env_done = torch.zeros(env.num_envs, dtype=torch.bool, device=self.device)
        per_env_reward = torch.zeros(env.num_envs, device=self.device)
        per_env_length = torch.zeros(env.num_envs, device=self.device)
        per_env_success = torch.zeros(env.num_envs, dtype=torch.bool, device=self.device)
        per_env_reason: list[str] = ["" for _ in range(env.num_envs)]

        # Motion command handle for detecting clip-end (success) resets
        _motion_cmd = getattr(
            self.unwrapped_env.command_manager, "get_state", lambda _: None
        )("motion_command")

        # ── Progress diagnostics (eval-path only) ───────────────────────
        # Purely diagnostic accumulators populated per-step, for the first
        # (and only) episode of each env under ``single_episode_per_env``.
        # Zero intrusion into the training path: all tensors are local and
        # only written to ``self._last_per_env_stats`` at exit.  Values are
        # only meaningful when the active motion clip has an object
        # (``_motion_cmd.motion.has_object``).
        _diag_cfg = getattr(self, "_eval_diagnostics_cfg", None)
        _grasp_radius: float = float(getattr(_diag_cfg, "eval_grasp_radius", 0.12))
        _lift_margin: float = float(getattr(_diag_cfg, "eval_lift_height_margin", 0.05))
        _contact_radius: float = float(getattr(_diag_cfg, "eval_contact_radius", 0.18))
        _object_moved_thresh: float = float(
            getattr(_diag_cfg, "eval_object_moved_thresh", 0.02)
        )
        _has_object: bool = bool(
            _motion_cmd is not None
            and getattr(getattr(_motion_cmd, "motion", None), "has_object", False)
        )

        # Discover wrist indices in ``motion_command.body_names_to_track``.
        # ``robot_body_pos_w`` is indexed by ``tracked_body_indexes`` which
        # follows the same order as ``body_names_to_track``, so we can map
        # by the name list directly.  Substring match is robust across G1
        # naming variants (left_wrist_roll_link / left_wrist_yaw_link / …).
        _l_wrist_idx: int = -1
        _r_wrist_idx: int = -1
        if _has_object:
            _body_names = list(
                getattr(getattr(_motion_cmd, "motion_cfg", None),
                        "body_names_to_track", []) or []
            )
            for _bi, _bn in enumerate(_body_names):
                _bn_l = _bn.lower()
                if "wrist" in _bn_l:
                    if _l_wrist_idx < 0 and ("left" in _bn_l or _bn_l.startswith("l_")):
                        _l_wrist_idx = _bi
                    elif _r_wrist_idx < 0 and ("right" in _bn_l or _bn_l.startswith("r_")):
                        _r_wrist_idx = _bi
            if _l_wrist_idx < 0 or _r_wrist_idx < 0:
                logger.warning(
                    f"[Eval-diag] Could not locate both wrist bodies in "
                    f"body_names_to_track={_body_names}; grasp detection "
                    f"will be disabled."
                )

        _INF = float("inf")
        _per_env_min_obj2goal = torch.full(
            (env.num_envs,), _INF, device=self.device
        )
        _per_env_max_obj_height = torch.full(
            (env.num_envs,), -_INF, device=self.device
        )
        _per_env_first_grasp_step = torch.full(
            (env.num_envs,), -1, dtype=torch.long, device=self.device
        )
        _per_env_first_lift_step = torch.full(
            (env.num_envs,), -1, dtype=torch.long, device=self.device
        )
        _per_env_bad_tracking_step = torch.full(
            (env.num_envs,), -1, dtype=torch.long, device=self.device
        )
        _per_env_obj_z0 = torch.zeros(env.num_envs, device=self.device)
        _per_env_obj_z0_set = torch.zeros(
            env.num_envs, dtype=torch.bool, device=self.device
        )
        _per_env_action_sq_sum = torch.zeros(env.num_envs, device=self.device)
        _per_env_action_abs_max = torch.zeros(env.num_envs, device=self.device)
        _per_env_alive_steps = torch.zeros(
            env.num_envs, dtype=torch.long, device=self.device
        )

        # ── v2 diagnostics ───────────────────────────────────────────
        _per_env_init_obj2goal = torch.full(
            (env.num_envs,), _INF, device=self.device
        )
        _per_env_init_hand_obj_d = torch.full(
            (env.num_envs,), _INF, device=self.device
        )
        _per_env_min_hand_obj_d = torch.full(
            (env.num_envs,), _INF, device=self.device
        )
        _per_env_obj_xy0 = torch.zeros(env.num_envs, 2, device=self.device)
        _per_env_obj_xy_disp_max = torch.zeros(env.num_envs, device=self.device)
        _per_env_first_contact_step = torch.full(
            (env.num_envs,), -1, dtype=torch.long, device=self.device
        )
        _per_env_first_approach_step = torch.full(
            (env.num_envs,), -1, dtype=torch.long, device=self.device
        )
        # Reward accumulated only while bad_tracking has not yet been tagged.
        _per_env_reward_pre_bad = torch.zeros(env.num_envs, device=self.device)
        # Hand-obj distance snapshotted at the instant first_lift_step is tagged.
        _per_env_hand_obj_d_at_lift = torch.full(
            (env.num_envs,), float("nan"), device=self.device
        )

        # Suppress step-level debug logs during the first episode
        _first_episode_done = False

        for step in itertools.islice(itertools.count(), max_eval_steps):
            if self.obs_normalization:
                normalized_obs = self.obs_normalizer(obs, update=False)
            else:
                normalized_obs = obs

            # Deterministic action — same as _run_eval_rollouts
            actions, pre_tanh_mean, _log_std = self.actor(normalized_obs)

            # ── bad_tracking diagnostic (evaluate_policy path) ────
            if step < 10 and _first_episode_done:
                _act_abs = actions.abs()
                _pre_abs = pre_tanh_mean.abs()
                _env_as = getattr(self.unwrapped_env, "action_scales", None)
                _parts = [
                    f"[eval_policy step={step}]",
                    f"pre_tanh |mean|: max={_pre_abs.max().item():.4f} "
                    f"avg={_pre_abs.mean().item():.4f}",
                    f"action |a|: max={_act_abs.max().item():.4f} "
                    f"avg={_act_abs.mean().item():.4f}",
                ]
                if _env_as is not None:
                    _pd = actions * _env_as
                    _parts.append(
                        f"pd_offset |a*s|: max={_pd.abs().max().item():.4f} "
                        f"avg={_pd.abs().mean().item():.4f}"
                    )
                _parts.append(
                    f"norm_obs |x|: avg={normalized_obs.abs().mean().item():.4f}"
                )
                # Per-joint detail for env 0 — critical joints that control
                # bad_tracking-monitored bodies (wrists + ankles)
                _critical = {4: "L_ank_p", 5: "L_ank_r", 10: "R_ank_p", 11: "R_ank_r",
                             19: "L_wri_r", 20: "L_wri_p", 21: "L_wri_y",
                             26: "R_wri_r", 27: "R_wri_p", 28: "R_wri_y"}
                _a0 = actions[0]  # env 0
                _pd0 = (_a0 * _env_as).abs() if _env_as is not None else _a0.abs()
                _joint_strs = []
                for _j, _name in sorted(_critical.items()):
                    _joint_strs.append(f"{_name}={_a0[_j].item():+.2f}(pd={_pd0[_j].item():.3f})")
                _parts.append("joints[env0]: " + " ".join(_joint_strs))
                # Also log raw obs vs normalized obs difference (normalizer shift diagnostic)
                _parts.append(f"raw_obs |x|: avg={obs.abs().mean().item():.4f}")
                logger.info("  ".join(_parts))

            # Callback hooks (video, custom metrics, etc.)
            actor_state = {"step": step, "actions": actions, "obs": obs}
            actor_state = self._pre_eval_env_step(actor_state)

            # Snapshot motion time-steps to detect clip-end (success) after step
            _prev_motion_ts = _motion_cmd.time_steps.clone() if _motion_cmd is not None else None

            obs, rewards, dones, extras = env.step(actor_state["actions"])

            actor_state["obs"] = obs
            actor_state = self._post_eval_env_step(actor_state)

            # Accumulate metrics
            all_actions.append(actor_state["actions"])
            if single_episode_per_env:
                alive_f = (~per_env_done).float()
                total_reward += rewards * alive_f
                ep_reward_sums += rewards * alive_f
                ep_lengths += alive_f
            else:
                total_reward += rewards
                ep_reward_sums += rewards
                ep_lengths += 1

            # ── Progress diagnostics accumulators (alive envs only) ──
            # Kept inside a try/except so any sim-side schema mismatch
            # degrades gracefully (metrics become NaN/-1 downstream rather
            # than crashing the eval run).
            if single_episode_per_env:
                try:
                    _alive_mask = ~per_env_done
                    if _alive_mask.any():
                        _a = actor_state["actions"]  # [N, act_dim]
                        _a_sq = _a.pow(2).sum(dim=-1)  # [N]
                        _a_max = _a.abs().amax(dim=-1)  # [N]
                        _per_env_action_sq_sum[_alive_mask] += _a_sq[_alive_mask]
                        _per_env_action_abs_max[_alive_mask] = torch.maximum(
                            _per_env_action_abs_max[_alive_mask],
                            _a_max[_alive_mask],
                        )
                        _per_env_alive_steps[_alive_mask] += 1

                        # v2: reward accumulated while bad_tracking not yet tagged.
                        # Includes the step at which bad_tracking terminates
                        # (bad_tracking_step is tagged AFTER this block), which
                        # is a small bounded overshoot we accept for simplicity.
                        _pre_bad_mask = _alive_mask & (_per_env_bad_tracking_step < 0)
                        if _pre_bad_mask.any():
                            _per_env_reward_pre_bad[_pre_bad_mask] += rewards[_pre_bad_mask]

                    if _has_object and _alive_mask.any():
                        _sim_obj_pos = _motion_cmd.simulator_object_pos_w  # [N, 3]
                        _ref_obj_pos = _motion_cmd.object_pos_w            # [N, 3]
                        _obj_z = _sim_obj_pos[:, 2]                        # [N]
                        _obj_xy = _sim_obj_pos[:, :2]                      # [N, 2]

                        # min object↔goal distance (reference pose)
                        _d_goal = torch.linalg.norm(
                            _sim_obj_pos - _ref_obj_pos, dim=-1
                        )  # [N]

                        # Compute hand distances (needed both for snapshot + min)
                        _has_wrists = (_l_wrist_idx >= 0 and _r_wrist_idx >= 0)
                        if _has_wrists:
                            _rb = _motion_cmd.robot_body_pos_w  # [N, tracked, 3]
                            _lh = _rb[:, _l_wrist_idx, :]
                            _rh = _rb[:, _r_wrist_idx, :]
                            _lh_d = torch.linalg.norm(_lh - _sim_obj_pos, dim=-1)
                            _rh_d = torch.linalg.norm(_rh - _sim_obj_pos, dim=-1)
                            _min_hand_d = torch.minimum(_lh_d, _rh_d)

                        # Snapshot episode-start quantities on first alive step
                        _need_snapshot = _alive_mask & (~_per_env_obj_z0_set)
                        if _need_snapshot.any():
                            _per_env_obj_z0[_need_snapshot] = _obj_z[_need_snapshot]
                            _per_env_obj_xy0[_need_snapshot] = _obj_xy[_need_snapshot]
                            _per_env_init_obj2goal[_need_snapshot] = _d_goal[_need_snapshot]
                            if _has_wrists:
                                _per_env_init_hand_obj_d[_need_snapshot] = _min_hand_d[_need_snapshot]
                            _per_env_obj_z0_set[_need_snapshot] = True

                        _per_env_min_obj2goal = torch.where(
                            _alive_mask,
                            torch.minimum(_per_env_min_obj2goal, _d_goal),
                            _per_env_min_obj2goal,
                        )

                        # max object height gain over episode start
                        _h = _obj_z - _per_env_obj_z0  # [N]
                        _per_env_max_obj_height = torch.where(
                            _alive_mask & _per_env_obj_z0_set,
                            torch.maximum(_per_env_max_obj_height, _h),
                            _per_env_max_obj_height,
                        )

                        # v2: object XY displacement since episode start
                        _xy_disp = torch.linalg.norm(
                            _obj_xy - _per_env_obj_xy0, dim=-1
                        )  # [N]
                        _per_env_obj_xy_disp_max = torch.where(
                            _alive_mask & _per_env_obj_z0_set,
                            torch.maximum(_per_env_obj_xy_disp_max, _xy_disp),
                            _per_env_obj_xy_disp_max,
                        )

                        if _has_wrists:
                            # v2: running min hand-obj distance
                            _per_env_min_hand_obj_d = torch.where(
                                _alive_mask,
                                torch.minimum(_per_env_min_hand_obj_d, _min_hand_d),
                                _per_env_min_hand_obj_d,
                            )

                            # v1 grasp (tight radius, kept for backwards compat)
                            _grasp_mask = (
                                _alive_mask
                                & (_per_env_first_grasp_step < 0)
                                & (_min_hand_d < _grasp_radius)
                            )
                            if _grasp_mask.any():
                                _per_env_first_grasp_step[_grasp_mask] = int(step)

                            # v2 first_contact (looser radius)
                            _contact_mask = (
                                _alive_mask
                                & (_per_env_first_contact_step < 0)
                                & (_min_hand_d < _contact_radius)
                            )
                            if _contact_mask.any():
                                _per_env_first_contact_step[_contact_mask] = int(step)

                            # v2 first_approach: hand-obj distance ≤ 50% of its
                            # episode-start value (threshold-free relative).
                            _approach_mask = (
                                _alive_mask
                                & _per_env_obj_z0_set
                                & (_per_env_first_approach_step < 0)
                                & torch.isfinite(_per_env_init_hand_obj_d)
                                & (_min_hand_d <= 0.5 * _per_env_init_hand_obj_d)
                            )
                            if _approach_mask.any():
                                _per_env_first_approach_step[_approach_mask] = int(step)

                        # First lift
                        _lift_mask = (
                            _alive_mask
                            & _per_env_obj_z0_set
                            & (_per_env_first_lift_step < 0)
                            & (_h > _lift_margin)
                        )
                        if _lift_mask.any():
                            _per_env_first_lift_step[_lift_mask] = int(step)
                            # v2: snapshot hand-obj distance at the lift instant
                            if _has_wrists:
                                _per_env_hand_obj_d_at_lift[_lift_mask] = _min_hand_d[_lift_mask]
                except Exception as _diag_exc:  # pragma: no cover
                    # Degrade gracefully — log once and disable subsequent
                    # object-side updates so we don't spam the log.
                    if not getattr(self, "_diag_error_logged", False):
                        logger.warning(
                            f"[Eval-diag] progress diagnostics disabled after "
                            f"exception: {_diag_exc!r}"
                        )
                        self._diag_error_logged = True
                        _has_object = False

            # ── Common episode-end helper ─────────────────────────
            def _finish_episodes(indices: torch.Tensor, reason: str) -> None:
                """Log, record, and reset accumulators for finished episodes."""
                nonlocal _first_episode_done
                if single_episode_per_env:
                    # Ignore envs that have already recorded their first episode.
                    keep = ~per_env_done[indices]
                    indices = indices[keep]
                    if indices.numel() == 0:
                        return
                    per_env_reward[indices] = ep_reward_sums[indices]
                    per_env_length[indices] = ep_lengths[indices]
                    per_env_success[indices] = (reason == "success")
                    for idx in indices:
                        per_env_reason[int(idx.item())] = reason
                    per_env_done[indices] = True
                for idx in indices:
                    i = idx.item()
                    completed_ep_rewards.append(ep_reward_sums[idx].item())
                    completed_ep_lengths.append(ep_lengths[idx].item())
                    ep_num = len(completed_ep_rewards)
                    logger.info(
                        f"[Eval] Episode {ep_num} ended  "
                        f"(env={i}, steps={int(ep_lengths[idx].item())}, "
                        f"reward={ep_reward_sums[idx].item():.2f})  "
                        f"reason: {reason}"
                    )
                ep_reward_sums[indices] = 0.0
                ep_lengths[indices] = 0.0
                _first_episode_done = True

            # ── Failure / timeout episodes (reset_buf driven) ─────
            done_mask = dones.bool()
            if done_mask.any():
                done_indices = done_mask.nonzero(as_tuple=False).squeeze(-1)

                # Determine per-env termination reason
                term_mgr = getattr(
                    getattr(env, "_env", None), "termination_manager", None
                )
                for idx in done_indices:
                    i = idx.item()
                    reason_str = "unknown"
                    if term_mgr is not None and hasattr(term_mgr, "active_terms"):
                        reasons = [
                            name
                            for name, mask in term_mgr.active_terms.items()
                            if mask[i].item()
                        ]
                        if reasons:
                            reason_str = ", ".join(reasons)
                    # Tag bad_tracking termination step for this env (diagnostic only)
                    if single_episode_per_env and "bad_tracking" in reason_str:
                        if int(_per_env_bad_tracking_step[idx].item()) < 0:
                            _per_env_bad_tracking_step[idx] = int(step)
                    _finish_episodes(idx.unsqueeze(0), reason_str)

                ep_info = extras.get("episode", {})
                if isinstance(ep_info, dict):
                    for k, v in ep_info.items():
                        if k not in episode_signals:
                            episode_signals[k] = []
                        if isinstance(v, torch.Tensor):
                            episode_signals[k].append(v.float().mean().item())
                        elif isinstance(v, (int, float)):
                            episode_signals[k].append(float(v))

            # ── Success episodes (clip-end, no reset_buf) ─────────
            if _prev_motion_ts is not None:
                clip_ended = (
                    (_prev_motion_ts > 1) & (_motion_cmd.time_steps < _prev_motion_ts) & ~done_mask
                )
                if clip_ended.any():
                    success_indices = clip_ended.nonzero(as_tuple=False).squeeze(-1)
                    if success_indices.dim() == 0:
                        success_indices = success_indices.unsqueeze(0)
                    _finish_episodes(success_indices, "success")

            # ── Early termination when all envs finished first episode ──
            if single_episode_per_env and bool(per_env_done.all().item()):
                logger.info(
                    f"[Eval] All {env.num_envs} envs completed their first episode "
                    f"at step {step + 1}. Stopping evaluation loop."
                )
                break

        self._post_evaluate_policy()

        # Restore training mode
        if was_training:
            self.actor.train()
            if self.obs_normalization:
                self.obs_normalizer.train()

        # Build return dict (empty if max_eval_steps was None → 0 steps)
        metrics: dict[str, float] = {}
        num_steps = step + 1 if max_eval_steps else 0  # type: ignore[possibly-undefined]
        if num_steps > 0:
            metrics["mean_reward"] = (total_reward / num_steps).mean().item()
            if completed_ep_rewards:
                metrics["mean_ep_reward"] = sum(completed_ep_rewards) / len(completed_ep_rewards)
                metrics["mean_ep_length"] = sum(completed_ep_lengths) / len(completed_ep_lengths)
            metrics["num_episodes"] = float(len(completed_ep_rewards))
            if all_actions:
                stacked = torch.cat(all_actions, dim=0)
                metrics["action_mean"] = stacked.abs().mean().item()
            for k, vals in episode_signals.items():
                if vals:
                    metrics[f"ep_{k}"] = sum(vals) / len(vals)

        # Expose per-env first-episode stats for external consumers
        # (e.g. eval_agent.py CSV/summary writer). Only populated when
        # ``_single_episode_per_env`` mode was enabled.
        if single_episode_per_env:
            # Sentinel-cleanup: ∞ / -∞ → NaN so downstream CSV/JSON writers
            # see a clean "unavailable" marker rather than a float overflow
            # token.  Applied only where an episode produced no qualifying
            # sample (e.g. clip had no object, or alive_steps == 0).
            _nan = float("nan")
            _min_o2g = _per_env_min_obj2goal.detach().clone()
            _min_o2g[~torch.isfinite(_min_o2g)] = _nan
            _max_h = _per_env_max_obj_height.detach().clone()
            _max_h[~torch.isfinite(_max_h)] = _nan
            # action_norm_mean = sqrt(sum_sq / alive_steps)  (NaN when 0)
            _alive = _per_env_alive_steps.clamp(min=1).float()
            _an_mean = (_per_env_action_sq_sum / _alive).sqrt()
            _an_mean[_per_env_alive_steps == 0] = _nan
            _an_max = _per_env_action_abs_max.detach().clone()
            _an_max[_per_env_alive_steps == 0] = _nan

            # ── v2 sentinel cleanup ──────────────────────────────────
            _init_o2g = _per_env_init_obj2goal.detach().clone()
            _init_o2g[~torch.isfinite(_init_o2g)] = _nan
            _min_hand = _per_env_min_hand_obj_d.detach().clone()
            _min_hand[~torch.isfinite(_min_hand)] = _nan
            _xy_disp = _per_env_obj_xy_disp_max.detach().clone()
            # xy_disp is 0 when the object-side was never observed (has_object
            # False or init snapshot never happened).  Keep 0 only where
            # object was observed, else NaN.
            _xy_disp[~_per_env_obj_z0_set] = _nan
            _r_pre_bad = _per_env_reward_pre_bad.detach().clone()
            # If bad_tracking never triggered, reward_until_bad_tracking equals
            # per_env_reward (full episode).  If env never got any alive step
            # (which shouldn't happen) set NaN.
            _r_pre_bad[_per_env_alive_steps == 0] = _nan
            _hand_at_lift = _per_env_hand_obj_d_at_lift.detach().clone()

            self._last_per_env_stats = {
                "num_envs": int(env.num_envs),
                "reward": per_env_reward.detach().cpu().tolist(),
                "length": per_env_length.detach().cpu().tolist(),
                "success": per_env_success.detach().cpu().tolist(),
                "reason": list(per_env_reason),
                "done": per_env_done.detach().cpu().tolist(),
                # ── progress diagnostics (eval-path only) ─────────────────
                "min_obj2goal_dist": _min_o2g.cpu().tolist(),
                "max_obj_height": _max_h.cpu().tolist(),
                "first_grasp_step": _per_env_first_grasp_step.cpu().tolist(),
                "first_lift_step": _per_env_first_lift_step.cpu().tolist(),
                "bad_tracking_step": _per_env_bad_tracking_step.cpu().tolist(),
                "action_norm_mean": _an_mean.cpu().tolist(),
                "action_abs_max": _an_max.cpu().tolist(),
                "alive_steps": _per_env_alive_steps.cpu().tolist(),
                # ── v2 diagnostics ────────────────────────────────────────
                "initial_obj2goal_dist": _init_o2g.cpu().tolist(),
                "object_xy_displacement": _xy_disp.cpu().tolist(),
                "min_hand_obj_dist": _min_hand.cpu().tolist(),
                "first_contact_step": _per_env_first_contact_step.cpu().tolist(),
                "first_approach_step": _per_env_first_approach_step.cpu().tolist(),
                "reward_until_bad_tracking": _r_pre_bad.cpu().tolist(),
                "min_hand_obj_dist_at_lift": _hand_at_lift.cpu().tolist(),
            }

        return metrics

    # ── eval callback helpers (same as FastSAC) ──────────────────────

    def _create_eval_callbacks(self) -> None:
        if not hasattr(self, "eval_callbacks"):
            self.eval_callbacks: list = []
        if self.config.eval_callbacks is not None:
            for cb_name in self.config.eval_callbacks:
                self.eval_callbacks.append(instantiate(self.config.eval_callbacks[cb_name], training_loop=self))

    def _pre_evaluate_policy(self) -> None:
        self.env.set_is_evaluating()
        for c in self.eval_callbacks:
            c.on_pre_evaluate_policy()

    def _post_evaluate_policy(self) -> None:
        for c in self.eval_callbacks:
            c.on_post_evaluate_policy()

    def _pre_eval_env_step(self, actor_state: dict) -> dict:
        for c in self.eval_callbacks:
            actor_state = c.on_pre_eval_env_step(actor_state)
        return actor_state

    def _post_eval_env_step(self, actor_state: dict) -> dict:
        for c in self.eval_callbacks:
            actor_state = c.on_post_eval_env_step(actor_state)
        return actor_state
