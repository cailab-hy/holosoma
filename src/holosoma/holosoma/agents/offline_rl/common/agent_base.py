"""OfflineRLAgentBase — Step 7-B skeleton.

Algorithm-neutral base class for the per-family offline-RL agents that
replaced the removed monolithic ``OfflineCQLAgent`` implementation.

Design contract
---------------
This class is **the** algorithm-neutral chassis.  Subclasses
(:class:`CQLAgent` first, ``SMQRAgent`` / ``SMQRSGAgent`` later) override
the four algorithm hooks at the bottom of this file; everything else is
shared.

What lives here (Step 7-B)
~~~~~~~~~~~~~~~~~~~~~~~~~~
* ``__init__``                      — env wrap, logger, training-metric meter
* ``save`` / ``load``               — checkpoint round-trip via
                                      :mod:`holosoma.agents.offline_rl.common.checkpointing`
* ``_maybe_amp``                    — AMP context (forwards to
                                      :mod:`holosoma.agents.offline_rl.common.optim`)
* ``get_inference_policy``          — actor closure for inference
* ``actor_onnx_wrapper`` / ``export`` — ONNX export (algorithm-neutral)
* ``_create_eval_callbacks`` etc.   — trivial hooks
* Abstract hooks for the subclass:
    - ``_resolve_algo_mode``
    - ``_validate_algo_config``
    - ``_compute_conservative_penalty``
    - ``_emit_algo_telemetry``

What does NOT live here (yet)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* ``setup`` — owned by the subclass in Step 7-B because the dataset /
  network construction is currently intertwined with algo-mode resolve
  logic; refactoring into a fully algo-neutral setup is part of
  Step 7-C/7-D unification.
* ``learn`` / ``_update_critic`` / ``_update_actor`` / ``_update_alpha``
  — owned by the subclass.  ``learn`` shape is algorithm-agnostic, but
  the call graph is short and porting it verbatim into the subclass is
  the smallest-blast-radius choice for 7-B.
* ``evaluate_policy`` / ``_run_eval_rollouts`` — owned by the subclass;
  extraction into :mod:`holosoma.agents.offline_rl.common.eval_utils` is
  deferred (placeholder helpers added there in this step).

Frozen contracts preserved
~~~~~~~~~~~~~~~~~~~~~~~~~~
* ``BaseAlgo`` inheritance chain is unchanged.
* Checkpoint key schema (``save_cql_params`` / ``load_cql_params``)
    unchanged — this class invokes the canonical helpers in
    :mod:`holosoma.agents.offline_rl.common.checkpointing`.
* TensorBoard tag emission stays inside the subclass methods (no tag
  renames in this step).
* Historical ``OfflineCQLAgent`` target metadata remains supported via
    :mod:`holosoma.agents.offline_rl.common.target_compat`; it resolves
    directly to canonical ``CQLAgent`` without importing ``offline_cql``.
"""

from __future__ import annotations

import copy
import math
from contextlib import contextmanager
from typing import Any, Callable

from loguru import logger

from holosoma.agents.base_algo.base_algo import BaseAlgo
from holosoma.agents.callbacks.base_callback import RLEvalCallback
from holosoma.agents.fast_sac.fast_sac_agent import FastSACEnv
from holosoma.agents.modules.logging_utils import LoggingHelper
from holosoma.agents.offline_rl.common.checkpointing import (
    load_offline_rl_params,
    save_offline_rl_params,
)
from holosoma.envs.base_task.base_task import BaseTask
from holosoma.utils.average_meters import TensorAverageMeterDict
from holosoma.utils.inference_helpers import (
    attach_onnx_metadata,
    export_motion_and_policy_as_onnx,
    export_policy_as_onnx,
    get_command_ranges_from_env,
    get_control_gains_from_config,
    get_urdf_text_from_robot_config,
)
from holosoma.utils.safe_torch_import import (
    GradScaler,
    TensorboardSummaryWriter,
    autocast,
    nn,
    torch,
)


class OfflineRLAgentBase(BaseAlgo):
    """Algorithm-neutral base class for the new offline-RL agents.

    This class is **abstract** with respect to the four
    ``_resolve_algo_mode`` / ``_validate_algo_config`` /
    ``_compute_conservative_penalty`` / ``_emit_algo_telemetry`` hooks.
    Calling them on the base raises :class:`NotImplementedError`.

    Parameters
    ----------
    env:
        Live :class:`BaseTask` instance (wrapped internally with
        :class:`FastSACEnv` to expose ``action_scale`` / ``num_envs``).
    config:
        Hydra-instantiated config object.  For Step 7-B this is the
        legacy :class:`holosoma.config_types.algo.OfflineCQLConfig`
        (aliased to ``OfflineRLBaseConfig`` in :mod:`common.config_base`).
    device:
        Torch device string (``"cuda"`` / ``"cpu"`` / ``"cuda:0"`` …).
    log_dir:
        Directory that will receive TensorBoard scalars + checkpoints.
    multi_gpu_cfg:
        DDP configuration passed through to :class:`BaseAlgo`.
    """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def __init__(
        self,
        env: BaseTask,
        config: Any,
        device: str,
        log_dir: str,
        multi_gpu_cfg: dict | None = None,
    ):
        wrapped_env = FastSACEnv(env, config.actor_obs_keys, config.critic_obs_keys)
        super().__init__(wrapped_env, config, device, multi_gpu_cfg)  # type: ignore[arg-type]
        self.unwrapped_env = env
        self.log_dir = log_dir
        self.global_step = 0

        # q_normalizer slow-EMA / freeze-at-step state (algorithm-neutral —
        # the q_normalizer mode is a knob on the SAC actor objective).
        self._q_normalizer_ref_1k: float | None = None
        self._q_normalizer_ref_5k: float | None = None
        self._q_normalizer_ema: float | None = None
        self._q_normalizer_frozen: float | None = None

        # TensorBoard + scalar meter
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
            "Perf/total_fps",
            "Perf/collection_time",
            "Perf/learning_time",
            "Train/num_samples",
        }
        self.training_metrics = TensorAverageMeterDict()
        self.eval_callbacks: list[RLEvalCallback] = []

    # ------------------------------------------------------------------
    # AMP helper
    # ------------------------------------------------------------------
    @contextmanager
    def _maybe_amp(self):
        """Mixed-precision context — bit-exact equivalent of the legacy
        ``OfflineCQLAgent._maybe_amp`` helper.
        """
        amp_dtype = torch.bfloat16 if self.config.amp_dtype == "bf16" else torch.float16
        with autocast(device_type="cuda", dtype=amp_dtype, enabled=self.config.amp):
            yield

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:  # type: ignore[override]
        """Persist the full training state via the common checkpoint helper.

        The checkpoint key schema is **unchanged** from
        ``save_cql_params`` — :mod:`common.checkpointing` re-exports it
        with object identity (``save_offline_rl_params is
        save_cql_params``).
        """
        env_state = self._collect_env_state()
        save_offline_rl_params(
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

    def load(self, ckpt_path: str | None, *, actor_only: bool = False) -> None:
        """Load a checkpoint via the common checkpoint helper.

        Bit-exact equivalent of the legacy ``OfflineCQLAgent.load``:
        ``load_offline_rl_params is load_cql_params`` (object identity
        preserved by :mod:`common.checkpointing`).
        """
        if not ckpt_path:
            return

        ckpt = load_offline_rl_params(
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

        if not actor_only:
            self.global_step = ckpt.get("global_step", 0)
        self._restore_env_state(ckpt.get("env_state"))

    # ------------------------------------------------------------------
    # Inference / export (algorithm-neutral)
    # ------------------------------------------------------------------
    def get_inference_policy(
        self, device: str | None = None
    ) -> Callable[[dict[str, torch.Tensor]], torch.Tensor]:
        """Return a callable that maps ``{"actor_obs": Tensor}`` → action.

        Verbatim port of the legacy method — actor + obs normaliser are
        identical classes.
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
        """ONNX-exportable wrapper — verbatim port of the legacy property."""

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

        return ActorWrapper(
            actor, obs_normalizer if self.config.obs_normalization else None
        )

    def export(self, onnx_file_path: str) -> None:
        """Export ONNX policy — verbatim port of the legacy method."""
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
            action_scale_metadata: float | list[float] = float(
                self.env.robot_config.control.action_scale
            )
        else:
            action_scale_metadata = action_scales.detach().cpu().tolist()
        urdf_file_path, urdf_str = get_urdf_text_from_robot_config(
            self.env.robot_config
        )

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

        attach_onnx_metadata(onnx_path=onnx_file_path, metadata=metadata)

        if hasattr(self, "logging_helper"):
            self.logging_helper.save_to_wandb(onnx_file_path)

        if was_training:
            self.actor.train()
            if self.obs_normalization:
                self.obs_normalizer.train()

    # ------------------------------------------------------------------
    # Eval-callback hooks (identical to the legacy stubs)
    # ------------------------------------------------------------------
    def _create_eval_callbacks(self) -> None:
        # Subclasses may populate ``self.eval_callbacks``; default is a
        # no-op (matches the legacy behaviour).
        pass

    def _pre_evaluate_policy(self) -> None:
        pass

    def _post_evaluate_policy(self) -> None:
        pass

    def _pre_eval_env_step(self, actor_state: dict) -> dict:
        return actor_state

    def _post_eval_env_step(self, actor_state: dict) -> dict:
        return actor_state

    # ------------------------------------------------------------------
    # Abstract hooks — subclasses must override
    # ------------------------------------------------------------------
    def _resolve_algo_mode(self, args: Any) -> Any:
        """Resolve the algorithm-mode label.

        Concrete subclasses return a marker object (typically a
        :class:`holosoma.agents.offline_rl.common.algo_mode.ResolvedAlgoMode`)
        that downstream code can branch on without re-reading config
        flags.
        """
        raise NotImplementedError(
            "OfflineRLAgentBase._resolve_algo_mode is abstract; "
            "subclass must override."
        )

    def _validate_algo_config(self, args: Any) -> None:
        """Validate that *args* is configured for **this** algorithm.

        Subclasses raise a descriptive :class:`RuntimeError` when knobs
        belonging to a different algorithm family are set (e.g. a CQL
        agent observing ``critic_penalty_mode='smqr_cont_self'``).
        """
        raise NotImplementedError(
            "OfflineRLAgentBase._validate_algo_config is abstract; "
            "subclass must override."
        )

    def _compute_conservative_penalty(
        self, *args: Any, **kwargs: Any
    ) -> "torch.Tensor":
        """Algorithm-specific Q-side conservative penalty.

        Subclasses return the post-tail ``per_state_penalty`` shaped
        ``[num_q, B]`` (legacy contract).  The base class deliberately
        does not pre-define the keyword signature: CQL uses one set of
        intermediate tensors and SMQR / SMQR-SG use another; locking the
        signature would force a premature refactor.
        """
        raise NotImplementedError(
            "OfflineRLAgentBase._compute_conservative_penalty is abstract; "
            "subclass must override."
        )

    def _emit_algo_telemetry(self, telemetry: dict[str, Any]) -> None:
        """Optional hook for algorithm-specific TB scalar emission.

        Default is a no-op so subclasses can either delegate to this
        method or emit telemetry inline — both styles are bit-exact
        because no shared canonical tag is touched here.
        """
        return None


__all__ = ["OfflineRLAgentBase"]
