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

            # ── Alpha-CQL (mode-dispatched) ───────────────────────
            alpha_cql = self.log_alpha_cql.exp().detach().squeeze()
            _cql_td_ratio = getattr(args, "cql_td_ratio", None)
            _cql_alpha_mode = getattr(args, "cql_alpha_mode", "td_relative")

            if _cql_td_ratio is not None and _cql_alpha_mode == "fixed_effective":
                # Ablation mode: effective α is a constant from config.
                # No dependency on td_loss — isolates the alpha rule as
                # a variable while keeping everything else identical.
                _fixed_val = getattr(args, "cql_fixed_effective_alpha", 0.015)
                _effective_alpha = _fixed_val
                _cql_raw_alpha = _fixed_val  # logged as-is (no td calc)
                cql_loss = _effective_alpha * cql_penalty
                _floor_active = 0.0
            elif _cql_td_ratio is not None:
                # Default: TD-relative schedule.
                _cql_floor = getattr(args, "cql_alpha_floor", 0.0)
                _cql_raw_alpha = (
                    _cql_td_ratio * td_loss.detach().item()
                    / max(abs(cql_penalty.detach().item()), 1e-8)
                )
                _effective_alpha = max(_cql_raw_alpha, _cql_floor)
                cql_loss = _effective_alpha * cql_penalty
                _floor_active = 1.0 if _cql_raw_alpha < _cql_floor else 0.0
            else:
                # Lagrangian / fixed α_cql (original path).
                cql_loss = alpha_cql * cql_penalty
                _effective_alpha = alpha_cql.item()
                _cql_raw_alpha = _effective_alpha
                _floor_active = 0.0

            # ── Total critic loss ──────────────────────────────────
            critic_loss = td_loss + cql_loss

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

        return {
            "td_loss": td_loss.detach(),
            "cql_penalty": cql_penalty.detach(),
            # _cql_penalty_raw is consumed by learn() for the Lagrange
            # update but excluded from TensorBoard logging (underscore key).
            "_cql_penalty_raw": cql_penalty_raw.detach(),
            "cql_penalty_per_q_mean": cql_penalty_mean.detach(),
            "cql_alpha": alpha_cql.squeeze().detach(),
            "cql_loss": cql_loss.detach(),
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
            # ── A. Policy-side OOD (min-Q basis) ──────────────────
            "cql_q_pi_curr_mean": _cql_q_pi_curr_mean,
            "cql_q_pi_next_mean": _cql_q_pi_next_mean,
            "q_data_q_pi_curr_gap": _q_data_q_pi_curr_gap,
            "q_data_q_pi_next_gap": _q_data_q_pi_next_gap,
            "pi_q_violation_rate_eps_0p1": _viol_0p1,
            "pi_q_violation_rate_eps_0p0": _viol_0p0,
            "pi_q_violation_mag": _viol_mag,
            "pi_q_violation_l2_weighted": _viol_l2w,
            # ── B. CQL candidate dominance ────────────────────────
            "cql_lse_weight_share_rand": _w_rand,
            "cql_lse_weight_share_pi_curr": _w_pi_curr,
            "cql_lse_weight_share_pi_next": _w_pi_next,
            "cql_q_rand_max": _q_rand_max,
            "cql_q_pi_curr_max": _q_pi_curr_max,
            "cql_q_pi_next_max": _q_pi_next_max,
            # ── C. Twin critic disagreement ───────────────────────
            "q1_q2_gap_data": _gap_data,
            "q1_q2_gap_pi_curr": _gap_pi_curr,
            "q1_q2_gap_pi_next": _gap_pi_next,
            # ── D. SC-CQL ─────────────────────────────────────────
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
            # ── D2. SC-CQL v2 leakage / sparsity ──────────────────
            "sc_safe_mask_mean_pi_curr": _sc_safe_mask_mean,
            "sc_safe_mask_p90_pi_curr": _sc_safe_mask_p90,
            "sc_violation_mask_mean_pi_curr": _sc_viol_mask_mean,
            "sc_violation_mask_p90_pi_curr": _sc_viol_mask_p90,
            "sc_mask_topk_threshold_pi_curr": _sc_topk_thr,
            "sc_mask_leakage_frac_pi_curr": _sc_leakage_frac,
            "q_data_q_pi_curr_gap_safe_subset": _gap_safe_sub,
            # ── D3. SC-CQL v3 deficit / severity / next subset ────
            "sc_deficit_mean_pi_curr": _sc_def_curr_mean,
            "sc_deficit_p90_pi_curr": _sc_def_curr_p90,
            "sc_deficit_mean_pi_next": _sc_def_next_mean,
            "sc_deficit_p90_pi_next": _sc_def_next_p90,
            "sc_severity_mean_pi_curr": _sc_sev_curr_mean,
            "sc_severity_mean_pi_next": _sc_sev_next_mean,
            "q_data_q_pi_next_gap_masked_subset": _gap_next_masked_sub,
            "q_data_q_pi_next_gap_safe_subset": _gap_next_safe_sub,
            # ── D4. SC-CQL v4 phase-aware ─────────────────────────
            "sc_phase_signal_available": _sc_phase_signal_avail,
            "sc_phase_post_lift_frac": _sc_phase_post_lift_frac,
            "sc_phase_obj_z_mean": _sc_phase_obj_z_mean,
            "sc_phase_obj_z_max": _sc_phase_obj_z_max,
            "sc_mask_active_frac_pi_next_post_lift": _sc_m_next_active_pl,
            "sc_mask_mean_pi_next_post_lift": _sc_m_next_mean_pl,
            "q_data_q_pi_next_gap_masked_post_lift": _gap_next_masked_pl,
            "pi_q_violation_rate_next_post_lift_eps_0p0": _viol_next_pl,
        }

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
            with torch.no_grad():
                q_normalizer = max(min_q.abs().mean().item(), 1.0)
            normalized_q = min_q / q_normalizer

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

        # Motion command handle for detecting clip-end (success) resets
        _motion_cmd = getattr(
            self.unwrapped_env.command_manager, "get_state", lambda _: None
        )("motion_command")

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
            total_reward += rewards
            ep_reward_sums += rewards
            ep_lengths += 1

            # ── Common episode-end helper ─────────────────────────
            def _finish_episodes(indices: torch.Tensor, reason: str) -> None:
                """Log, record, and reset accumulators for finished episodes."""
                nonlocal _first_episode_done
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
