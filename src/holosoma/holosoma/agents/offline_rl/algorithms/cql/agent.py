"""CQLAgent — Step 7-B extraction of the vanilla CQL training path.

Standalone agent for **vanilla Conservative Q-Learning** (Kumar et al.
2020).  The implementation was extracted from the historical monolithic
``OfflineCQLAgent`` source before the ``offline_cql`` package was removed.

Design contract (Step 7-B)
--------------------------
This class inherits from :class:`OfflineRLAgentBase` and ports the
algorithm-specific subset of the legacy agent.  Out-of-family knobs
(SMQR ``smqr_cont_self`` / SC-CQL ``sc_cql`` / learned-τ variants /
``sg_blend`` / ``sg_weighted_lse`` / ``q_times_detached_g``) are
**rejected** in :meth:`_validate_algo_config`.

* Loss helper: :func:`compute_cql_logsumexp_penalty` (bit-exact lift
  shared with the legacy agent — extracted in Step 4-A).
* Checkpoint schema: unchanged (re-uses the legacy
  ``save_cql_params`` / ``load_cql_params`` via
  :mod:`holosoma.agents.offline_rl.common.checkpointing`).
* TensorBoard tag schema: unchanged (canonical tags emitted verbatim).
* Production train path: uses the direct canonical ``CQLAgent`` target.
    Historical ``OfflineCQLAgent`` target metadata remains loadable only through
    :mod:`holosoma.agents.offline_rl.common.target_compat`, which maps the old
    string directly to this class without importing the deleted package.

Import boundary
---------------
This module **must not** import from ``holosoma.agents.offline_cql``.  Direct
legacy imports are unsupported after Step 14; old target compatibility is
metadata-only in ``target_compat``.  Network classes, dataset helpers and
checkpoint I/O are sourced through :mod:`holosoma.agents.offline_rl.common.*`.

The Step 7-B import-boundary lint test
(``tests/offline_rl/test_import_boundary.py``) enforces this rule.
"""

from __future__ import annotations

import math
import os
from typing import Any

import tqdm
from loguru import logger

from holosoma.agents.fast_sac.fast_sac import Actor
from holosoma.agents.fast_sac.fast_sac_utils import EmpiricalNormalization
from holosoma.agents.offline_rl.algorithms.cql.losses import (
    compute_cql_logsumexp_penalty,
)
from holosoma.agents.offline_rl.common.agent_base import OfflineRLAgentBase
from holosoma.agents.offline_rl.common.datasets import (
    OfflineDataset,
    create_frozen_normalizer,
    validate_normalization,
)
from holosoma.agents.offline_rl.common.eval_utils import (
    create_eval_callbacks as _shim_create_eval_callbacks,
)
from holosoma.agents.offline_rl.common.eval_utils import (
    post_eval_env_step as _shim_post_eval_env_step,
)
from holosoma.agents.offline_rl.common.eval_utils import (
    post_evaluate_policy as _shim_post_evaluate_policy,
)
from holosoma.agents.offline_rl.common.eval_utils import (
    pre_eval_env_step as _shim_pre_eval_env_step,
)
from holosoma.agents.offline_rl.common.eval_utils import (
    pre_evaluate_policy as _shim_pre_evaluate_policy,
)
from holosoma.agents.offline_rl.common.eval_utils import (
    run_eval_rollouts as _shim_run_eval_rollouts,
)
from holosoma.agents.offline_rl.common.eval_utils import (
    run_evaluate_policy,
)
from holosoma.agents.offline_rl.common.networks import (
    TwinQCritic,
    polyak_update,
)
from holosoma.agents.offline_rl.common.optim import amp_autocast
from holosoma.utils.safe_torch_import import (
    F,
    GradScaler,
    TensorDict,
    nn,
    optim,
    torch,
)


# ---------------------------------------------------------------------------
# Algo-mode marker (lightweight — full ResolvedAlgoMode is SMQR-specific)
# ---------------------------------------------------------------------------
class _CQLAlgoMode:
    """Marker returned by :meth:`CQLAgent._resolve_algo_mode`.

    Mirrors the field set of
    :class:`holosoma.agents.offline_rl.common.algo_mode.ResolvedAlgoMode` for
    the subset relevant to vanilla CQL (no SMQR / learned-τ branches).
    """

    mode = "cql"
    tau_source = "none"
    legacy_critic_penalty_mode = "vanilla_cql"
    tau_res_scale = 0.0
    learned_variant = "none"
    explicit = False
    logging_prefix = ""


class CQLAgent(OfflineRLAgentBase):
    """Vanilla Conservative Q-Learning agent (Step 7-B extraction)."""

    # ------------------------------------------------------------------
    # Algorithm hooks
    # ------------------------------------------------------------------
    def _resolve_algo_mode(self, args: Any) -> _CQLAlgoMode:
        return _CQLAlgoMode()

    def _validate_algo_config(self, args: Any) -> None:
        """Reject SMQR / SMQR-SG / learned-τ / SC-CQL config knobs.

        Vanilla CQL must run with ``critic_penalty_mode='vanilla_cql'``
        and zero stop-gradient blend.  Anything else is routed to the
        SMQR / SMQR-SG subclasses (Step 7-C / 7-D) — flagging it here
        prevents silent fall-through to the wrong loss.
        """
        mode = str(getattr(args, "critic_penalty_mode", "vanilla_cql"))
        if mode != "vanilla_cql":
            raise RuntimeError(
                f"CQLAgent requires critic_penalty_mode='vanilla_cql', "
                f"got {mode!r}.  Use SMQRAgent (smqr_cont_self / "
                f"smqr_anchor + q_times_g) or SMQRSGAgent "
                f"(sg_weighted_lse / sg_blend) instead."
            )
        # Defence in depth — forbid the SMQR-only flags even when the
        # main penalty mode is 'vanilla_cql' so a copy-pasted config
        # cannot silently fall through.
        for forbidden_key, forbidden_val in (
            ("algo_mode", "smqr_anchor"),
            ("algo_mode", "smqr_learned"),
            ("smqr_lse_mode", "sg_blend"),
            ("smqr_lse_mode", "sg_weighted_lse"),
            ("smqr_lse_mode", "q_times_detached_g"),
        ):
            cur = str(getattr(args, forbidden_key, "")).strip().lower()
            if cur == forbidden_val:
                raise RuntimeError(
                    f"CQLAgent rejects {forbidden_key}={cur!r}; "
                    "this knob belongs to the SMQR / SMQR-SG family."
                )

    def _compute_conservative_penalty(
        self,
        *,
        q_rand: "torch.Tensor",
        q_pi: "torch.Tensor",
        rand_log_density: "torch.Tensor",
        pi_log_probs: "torch.Tensor",
        q_pred_all: "torch.Tensor",
        num_random: int,
        num_policy: int,
        q_clip: float,
    ) -> dict[str, "torch.Tensor"]:
        """Vanilla CQL importance-weighted logsumexp penalty.

        Thin wrapper around the Step 4-A extracted helper.  Returns the
        dict produced by :func:`compute_cql_logsumexp_penalty` verbatim
        so the inline call site in :meth:`_update_critic` can pick the
        intermediate tensors it needs.
        """
        return compute_cql_logsumexp_penalty(
            q_rand=q_rand,
            q_pi=q_pi,
            rand_log_density=rand_log_density,
            pi_log_probs=pi_log_probs,
            q_pred_all=q_pred_all,
            num_random=num_random,
            num_policy=num_policy,
            q_clip=q_clip,
        )

    # ------------------------------------------------------------------
    # Setup (CQL-only path; SMQR guard blocks deliberately omitted)
    # ------------------------------------------------------------------
    def setup(
        self,
        *,
        eval_only: bool = False,
        checkpoint_path: str | None = None,
    ) -> None:
        """Build networks, load dataset, compute obs normalisation.

        Bit-exact equivalent of the vanilla-CQL portion of the legacy
        :meth:`OfflineCQLAgent.setup` (Tier-A config validation +
        dataset load + frozen normaliser + network construction +
        optimizer / AMP setup).  SMQR / SMQR-SG / learned-τ guard
        blocks are absent — :meth:`_validate_algo_config` rejects any
        config that would have entered them.
        """
        logger.info("Setting up CQLAgent (Step 7-B)")

        args = self.config
        device = self.device
        env = self.env  # FastSACEnv wrapper

        # ── algo-mode marker + config validation ──────────────────
        self._algo_mode = self._resolve_algo_mode(args)
        self._validate_algo_config(args)

        # ── Tier A hard-required field check (vanilla-CQL subset) ─
        _HARD_REQUIRED = (
            "actor_obs_keys",
            "critic_obs_keys",
            "obs_normalization",
            "actor_hidden_dim",
            "critic_hidden_dim",
            "actor_learning_rate",
            "critic_learning_rate",
            "alpha_learning_rate",
            "alpha_init",
            "use_autotune",
            "target_entropy_ratio",
            "gamma",
            "tau",
            "policy_frequency",
            "logging_interval",
            "save_interval",
            "cql_num_random_actions",
            "cql_num_policy_actions",
            "cql_alpha_autotune",
            "amp",
            "amp_dtype",
            "max_grad_norm",
        )
        _eval_skip = (
            {"dataset_path", "batch_size", "num_learning_iterations"}
            if eval_only
            else set()
        )
        _extra = (
            ()
            if eval_only
            else ("dataset_path", "batch_size", "num_learning_iterations")
        )
        missing = [
            k
            for k in (*_HARD_REQUIRED, *_extra)
            if k not in _eval_skip and not hasattr(args, k)
        ]
        if missing:
            raise ValueError(
                f"CQLAgent config is missing hard-required field(s): {missing}"
            )
        if args.cql_alpha_autotune and not hasattr(args, "cql_target_penalty"):
            raise ValueError(
                "cql_alpha_autotune=True requires config field 'cql_target_penalty'."
            )

        # ── 1. Observation index computation ───────────────────────
        algo_obs_dim_dict = env.observation_manager.get_obs_dims()
        algo_history_length_dict: dict[str, int] = {}
        for group_cfg in env.observation_manager.cfg.groups.values():
            history_len = getattr(group_cfg, "history_length", 1)
            for term_name in group_cfg.terms:
                algo_history_length_dict[term_name] = history_len

        actor_obs_keys = list(args.actor_obs_keys)
        critic_obs_keys = list(args.critic_obs_keys)
        n_act: int = env.robot_config.actions_dim

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
            self.dataset = OfflineDataset(
                path=args.dataset_path,
                device=device,
                expected_act_dim=n_act,
            )
            ds = self.dataset
            actor_obs_dim: int = ds.actor_obs_dim
            critic_obs_dim: int = ds.critic_obs_dim
            if actor_obs_dim != env_actor_obs_dim:
                logger.warning(
                    f"ACTOR OBS DIM MISMATCH: ds={actor_obs_dim} env={env_actor_obs_dim}"
                )
            if critic_obs_dim != env_critic_obs_dim:
                logger.warning(
                    f"CRITIC OBS DIM MISMATCH: ds={critic_obs_dim} env={env_critic_obs_dim}"
                )
        else:
            self.dataset = None  # type: ignore[assignment]
            if checkpoint_path is None:
                logger.warning(
                    "eval_only=True but no checkpoint_path provided; using env dims."
                )
                actor_obs_dim = env_actor_obs_dim
                critic_obs_dim = env_critic_obs_dim
            else:
                _ckpt_peek = torch.load(
                    checkpoint_path, map_location="cpu", weights_only=False
                )
                actor_obs_dim = _ckpt_peek["actor_state_dict"][
                    "net.0.weight"
                ].shape[1]
                _q_input_dim = _ckpt_peek["qnet_state_dict"][
                    "qnets.0.net.0.weight"
                ].shape[1]
                critic_obs_dim = _q_input_dim - n_act
                del _ckpt_peek

        self._eval_dims_match: bool = (
            actor_obs_dim == env_actor_obs_dim
            and critic_obs_dim == env_critic_obs_dim
        )
        if actor_obs_dim == env_actor_obs_dim:
            self.actor_obs_indices = env_actor_obs_indices
        else:
            flat_key = actor_obs_keys[0]
            self.actor_obs_indices = {
                flat_key: {
                    "start": 0,
                    "end": actor_obs_dim,
                    "size": actor_obs_dim,
                }
            }
            actor_obs_keys = [flat_key]
        if critic_obs_dim == env_critic_obs_dim:
            self.critic_obs_indices = env_critic_obs_indices
        else:
            flat_key = critic_obs_keys[0]
            self.critic_obs_indices = {
                flat_key: {
                    "start": 0,
                    "end": critic_obs_dim,
                    "size": critic_obs_dim,
                }
            }
            critic_obs_keys = [flat_key]
        # Term-level offsets (kept empty for vanilla CQL — only SC-CQL
        # phase gating consumes this and SC-CQL is out-of-scope here).
        self._critic_term_offsets: dict[str, dict[str, int]] = {}

        self.actor_obs_dim: int = actor_obs_dim
        self.critic_obs_dim: int = critic_obs_dim
        self._env_actor_obs_dim: int = env_actor_obs_dim
        self._env_critic_obs_dim: int = env_critic_obs_dim

        # ── 2. Action scaling ──────────────────────────────────────
        use_tanh: bool = getattr(args, "use_tanh", True)
        action_scale = (
            env._action_boundaries
            if use_tanh
            else torch.ones(n_act, device=device)
        )
        action_bias = torch.zeros(n_act, device=device)

        # ── 3. Actor ───────────────────────────────────────────────
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

        # ── 4. TwinQCritic + frozen target ────────────────────────
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
        self.qnet_target = TwinQCritic.create_target(self.qnet)
        logger.info(f"Actor:\n{self.actor}")
        logger.info(f"TwinQCritic:\n{self.qnet}")

        # ── 5. SAC entropy temperature α ──────────────────────────
        self.log_alpha = torch.tensor(
            [math.log(args.alpha_init)], requires_grad=True, device=device,
        )
        self.target_entropy: float = -n_act * args.target_entropy_ratio

        # ── 6. CQL Lagrange multiplier α_cql ──────────────────────
        cql_alpha_init: float = getattr(args, "cql_alpha_init", 1.0)
        self.log_alpha_cql = torch.tensor(
            [math.log(max(cql_alpha_init, 1e-8))],
            requires_grad=True,
            device=device,
        )
        cql_alpha_lr: float = getattr(args, "cql_alpha_learning_rate", 3e-4)
        self.alpha_cql_optimizer = optim.AdamW(
            [self.log_alpha_cql],
            lr=cql_alpha_lr,
            fused=True,
            betas=(0.9, 0.95),
        )

        # ── 7. Optimizers ──────────────────────────────────────────
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
            [self.log_alpha],
            lr=args.alpha_learning_rate,
            fused=True,
            betas=(0.9, 0.95),
        )

        # ── 8. GradScaler ──────────────────────────────────────────
        self.scaler = GradScaler(enabled=args.amp)

        # Schema parity with the legacy agent — empty placeholders
        # so that the checkpoint key set is unchanged.
        self.value_net = None  # type: ignore[assignment]
        self.value_optimizer = None  # type: ignore[assignment]

        # ── eval-only early return ────────────────────────────────
        if eval_only:
            self.obs_normalization: bool = args.obs_normalization
            if args.obs_normalization:
                self.obs_normalizer: nn.Module = EmpiricalNormalization(
                    shape=actor_obs_dim, device=device,
                )
                self.critic_obs_normalizer: nn.Module = EmpiricalNormalization(
                    shape=critic_obs_dim, device=device,
                )
                self.obs_normalizer.eval()
                self.critic_obs_normalizer.eval()
            else:
                self.obs_normalizer = nn.Identity()
                self.critic_obs_normalizer = nn.Identity()
            self.policy = self.actor.explore
            return

        # ── 9. Frozen normaliser from dataset stats ───────────────
        self.obs_normalization = args.obs_normalization
        if args.obs_normalization:
            actor_mean, actor_std = ds.compute_obs_statistics("actor")
            critic_mean, critic_std = ds.compute_obs_statistics("critic")
            self.obs_normalizer = create_frozen_normalizer(
                mean=actor_mean, std=actor_std, count=ds.size, device=device,
            )
            self.critic_obs_normalizer = create_frozen_normalizer(
                mean=critic_mean, std=critic_std, count=ds.size, device=device,
            )
            _audit_n = min(10_000, ds.size)
            logger.info(
                validate_normalization(
                    self.obs_normalizer,
                    ds.actor_obs[:_audit_n],
                    label="actor_obs",
                )["report"]
            )
            logger.info(
                validate_normalization(
                    self.critic_obs_normalizer,
                    ds.critic_obs[:_audit_n],
                    label="critic_obs",
                )["report"]
            )
        else:
            self.obs_normalizer = nn.Identity()
            self.critic_obs_normalizer = nn.Identity()
        self.policy = self.actor.explore

    # ------------------------------------------------------------------
    # Outer training loop (CQL-only)
    # ------------------------------------------------------------------
    def learn(self) -> None:
        """Offline CQL training loop.

        Algorithm-neutral outer loop verbatim port of the legacy
        :meth:`OfflineCQLAgent.learn` minus the SMQR Phase-A/B/C/D
        guard re-check and the SMQR ``writer.add_text`` mode tag.  The
        per-iteration call graph (sample → normalise → critic → actor →
        alpha → polyak → log → save) is preserved.
        """
        import time as _time

        args = self.config
        device = self.device
        dataset = self.dataset

        if getattr(args, "compile", False):
            normalize_obs = torch.compile(self.obs_normalizer.forward)
            normalize_critic_obs = torch.compile(self.critic_obs_normalizer.forward)
        else:
            normalize_obs = self.obs_normalizer.forward
            normalize_critic_obs = self.critic_obs_normalizer.forward

        training_metrics = self.training_metrics
        training_metrics.clear()
        self._last_cql_penalty = torch.tensor(0.0, device=device)

        eval_interval: int = getattr(args, "eval_interval", 0)
        eval_steps: int = getattr(args, "eval_steps", 200)

        start_step = self.global_step
        if start_step > 0:
            logger.info(f"Resuming CQLAgent training from step {start_step}")

        pbar = tqdm.tqdm(
            total=args.num_learning_iterations,
            initial=start_step,
            desc="CQL",
        )
        loop_start = _time.perf_counter()

        while self.global_step <= args.num_learning_iterations:
            data = dataset.sample(args.batch_size)

            if self.obs_normalization:
                data["observations"] = normalize_obs(
                    data["observations"], update=False
                )
                data["next"]["observations"] = normalize_obs(
                    data["next"]["observations"], update=False
                )
                data["critic_observations"] = normalize_critic_obs(
                    data["critic_observations"], update=False
                )
                data["next"]["critic_observations"] = normalize_critic_obs(
                    data["next"]["critic_observations"], update=False
                )

            critic_metrics = self._update_critic(data)
            self._last_cql_penalty = critic_metrics.pop("_cql_penalty_raw")
            training_metrics.add(critic_metrics)

            if self.global_step % args.policy_frequency == 0:
                actor_metrics = self._update_actor(data)
                training_metrics.add(actor_metrics)
                alpha_metrics = self._update_alpha(actor_metrics["log_probs_mean"])
                training_metrics.add(alpha_metrics)

            with torch.no_grad():
                polyak_update(self.qnet, self.qnet_target, args.tau)

            if (
                self.global_step % args.logging_interval == 0
                and self.global_step > 0
            ):
                with torch.no_grad():
                    accumulated = training_metrics.mean_and_clear()
                    loss_dict: dict[str, float] = {}
                    for key, value in accumulated.items():
                        loss_dict[key] = (
                            value.item()
                            if isinstance(value, torch.Tensor)
                            else float(value)
                        )
                elapsed = _time.perf_counter() - loop_start
                loss_dict["steps_per_sec"] = (self.global_step - start_step) / max(
                    elapsed, 1e-8
                )
                if self.is_main_process:
                    self.logging_helper.post_epoch_logging(
                        it=self.global_step,
                        loss_dict=loss_dict,
                        extra_log_dicts={},
                    )

            if (
                eval_interval > 0
                and self.global_step > 0
                and self.global_step % eval_interval == 0
                and self.is_main_process
                and getattr(self, "_eval_dims_match", True)
            ):
                eval_metrics = self._run_eval_rollouts(num_steps=eval_steps)
                for key, value in eval_metrics.items():
                    self.writer.add_scalar(
                        f"Eval/{key}", value, global_step=self.global_step
                    )

                eval_str = "  ".join(
                    f"{key}={value:.4f}"
                    for key, value in sorted(eval_metrics.items())
                )
                logger.info(f"[step {self.global_step}] EVAL  {eval_str}")

            if (
                args.save_interval > 0
                and self.global_step > 0
                and self.global_step % args.save_interval == 0
                and self.is_main_process
            ):
                logger.info(f"Saving CQLAgent at step {self.global_step}")
                self.save(
                    os.path.join(
                        self.log_dir, f"model_{self.global_step:07d}.pt"
                    )
                )

            if self.global_step >= args.num_learning_iterations:
                break
            self.global_step += 1
            pbar.update(1)

        pbar.close()
        if self.is_main_process:
            if eval_interval > 0 and getattr(self, "_eval_dims_match", True):
                final_eval = self._run_eval_rollouts(num_steps=eval_steps)
                eval_str = "  ".join(
                    f"{key}={value:.4f}"
                    for key, value in sorted(final_eval.items())
                )
                logger.info(f"[step {self.global_step}] FINAL EVAL  {eval_str}")
                for key, value in final_eval.items():
                    self.writer.add_scalar(
                        f"Eval/{key}", value, global_step=self.global_step
                    )

            self.save(
                os.path.join(self.log_dir, f"model_{self.global_step:07d}.pt")
            )

    # ------------------------------------------------------------------
    # Critic update (vanilla CQL only)
    # ------------------------------------------------------------------
    def _update_critic(self, data: "TensorDict") -> dict[str, "torch.Tensor"]:
        """Critic gradient step: TD loss + CQL conservative penalty.

        Bit-exact port of the **vanilla_cql** branch of the legacy
        :meth:`OfflineCQLAgent._update_critic`.  SMQR
        (``smqr_cont_self``), SC-CQL (``sc_cql``), and learned-τ
        branches are absent — :meth:`_validate_algo_config` guarantees
        they cannot reach this method.
        """
        args = self.config
        scaler = self.scaler
        actor = self.actor
        qnet = self.qnet
        qnet_target = self.qnet_target

        with self._maybe_amp():
            observations = data["observations"]
            critic_obs = data["critic_observations"]
            actions = data["actions"]
            next_obs = data["next"]["observations"]
            next_critic_obs = data["next"]["critic_observations"]
            rewards = data["next"]["rewards"]
            dones = data["next"]["dones"].bool()
            truncations = data["next"]["truncations"].bool()
            bootstrap = (truncations | ~dones).float()
            discount = args.gamma ** data["next"]["effective_n_steps"]

            # ── TD target ─────────────────────────────────────────
            _q_clip: float = getattr(args, "q_clip", 1e4)
            with torch.no_grad():
                next_actions, next_log_probs = actor.get_actions_and_log_probs(
                    next_obs
                )
                target_q = qnet_target.min_q(next_critic_obs, next_actions)
                target_q = target_q.squeeze(-1)
                td_target = rewards.float() + discount.float() * bootstrap.float() * (
                    target_q.float()
                    - self.log_alpha.exp().detach().float()
                    * next_log_probs.float()
                )
                td_target = td_target.clamp(-_q_clip, _q_clip)

            # ── TD loss ───────────────────────────────────────────
            q_pred_all = qnet(critic_obs, actions).squeeze(-1)
            td_loss = 0.5 * F.mse_loss(
                q_pred_all.float(),
                td_target.unsqueeze(0).expand_as(q_pred_all),
            )

            # ── CQL conservative penalty (IS-weighted logsumexp) ──
            B = observations.shape[0]
            n_act = actions.shape[-1]
            num_random = args.cql_num_random_actions
            num_policy = args.cql_num_policy_actions

            critic_obs_processed = qnet.process_obs(critic_obs)

            rand_actions = (
                torch.rand(B, num_random, n_act, device=observations.device) * 2.0
                - 1.0
            ) * actor.action_scale.unsqueeze(0).unsqueeze(0) + actor.action_bias.unsqueeze(
                0
            ).unsqueeze(0)
            rand_log_density = -torch.log(
                2.0 * actor.action_scale + 1e-6
            ).sum().detach()

            q_rand = qnet.q_values_for_actions(
                critic_obs_processed, rand_actions
            ).squeeze(-1)

            obs_repeat = (
                observations.unsqueeze(1)
                .expand(B, num_policy, -1)
                .reshape(B * num_policy, -1)
            )
            pi_actions, pi_log_probs = actor.get_actions_and_log_probs(obs_repeat)
            pi_actions = pi_actions.view(B, num_policy, n_act)
            pi_log_probs = pi_log_probs.view(B, num_policy).detach()

            q_pi = qnet.q_values_for_actions(
                critic_obs_processed, pi_actions
            ).squeeze(-1)

            _cql_pen_out = self._compute_conservative_penalty(
                q_rand=q_rand,
                q_pi=q_pi,
                rand_log_density=rand_log_density,
                pi_log_probs=pi_log_probs,
                q_pred_all=q_pred_all,
                num_random=num_random,
                num_policy=num_policy,
                q_clip=_q_clip,
            )
            cql_logsumexp = _cql_pen_out["cql_logsumexp"]
            q_data = _cql_pen_out["q_data"]
            per_state_penalty = _cql_pen_out["per_state_penalty"]

            # ── CQL penalty per Q-network ─────────────────────────
            cql_penalty_per_q = per_state_penalty.mean(dim=1)
            cql_penalty = cql_penalty_per_q.clamp(min=-10).sum()
            cql_penalty_raw = cql_penalty_per_q.sum()

            # Phase P1b — one-sided penalty floor (opt-in; default no-op)
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

            # ── α_cql dispatch (td_relative / fixed_effective / Lagrangian)
            alpha_cql = self.log_alpha_cql.exp().detach().squeeze()
            _cql_td_ratio = getattr(args, "cql_td_ratio", None)
            _cql_alpha_mode = getattr(args, "cql_alpha_mode", "td_relative")
            _cql_alpha_cap = float(
                getattr(args, "cql_effective_alpha_cap", 0.0)
            )

            if (
                _cql_td_ratio is not None
                and _cql_alpha_mode == "fixed_effective"
            ):
                _fixed_val = getattr(args, "cql_fixed_effective_alpha", 0.015)
                _effective_alpha_pre_cap = _fixed_val
                _cql_raw_alpha = _fixed_val
                _floor_active = 0.0
                _effective_alpha = (
                    min(_effective_alpha_pre_cap, _cql_alpha_cap)
                    if _cql_alpha_cap > 0.0
                    else _effective_alpha_pre_cap
                )
                _cap_active = (
                    1.0
                    if (
                        _cql_alpha_cap > 0.0
                        and _effective_alpha_pre_cap > _cql_alpha_cap
                    )
                    else 0.0
                )
                cql_loss = _effective_alpha * _penalty_for_loss
            elif _cql_td_ratio is not None:
                _cql_floor = getattr(args, "cql_alpha_floor", 0.0)
                _cql_raw_alpha = (
                    _cql_td_ratio
                    * td_loss.detach().item()
                    / max(abs(cql_penalty.detach().item()), 1e-8)
                )
                _effective_alpha_pre_cap = max(_cql_raw_alpha, _cql_floor)
                _floor_active = 1.0 if _cql_raw_alpha < _cql_floor else 0.0
                _effective_alpha = (
                    min(_effective_alpha_pre_cap, _cql_alpha_cap)
                    if _cql_alpha_cap > 0.0
                    else _effective_alpha_pre_cap
                )
                _cap_active = (
                    1.0
                    if (
                        _cql_alpha_cap > 0.0
                        and _effective_alpha_pre_cap > _cql_alpha_cap
                    )
                    else 0.0
                )
                cql_loss = _effective_alpha * _penalty_for_loss
            else:
                cql_loss = alpha_cql * cql_penalty
                _effective_alpha = alpha_cql.item()
                _effective_alpha_pre_cap = _effective_alpha
                _cql_raw_alpha = _effective_alpha
                _floor_active = 0.0
                _cap_active = 0.0

            # Stage R1 — uniform conservative-loss scale (default 1.0 = no-op)
            _cql_loss_scale = float(getattr(args, "cql_loss_scale", 1.0))
            _cql_loss_unscaled = cql_loss
            cql_loss = _cql_loss_scale * cql_loss

            critic_loss = td_loss + cql_loss

        # ── Backward + optimise ───────────────────────────────────
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

        # ── Canonical TB scalars (10 golden + a few extras) ───────
        with torch.no_grad():
            q_data_mean = q_data.mean()
            q_data_max = q_data.max()
            q_data_min = q_data.min()
            td_target_mean = td_target.mean()
            cql_penalty_mean = cql_penalty_per_q.mean()
            cql_q_rand_mean = q_rand.mean()

            # Q overestimation gap — q_data − target (legacy canonical)
            q_overestimation_gap = q_data_mean - td_target_mean

        return {
            "td_loss": td_loss.detach(),
            "cql_penalty": cql_penalty.detach(),
            "cql_penalty_per_q_mean": cql_penalty_per_q.mean().detach(),
            "cql_loss": cql_loss.detach(),
            "critic_loss": critic_loss.detach(),
            "critic_grad_norm": critic_grad_norm.detach(),
            "q_data_mean": q_data_mean.detach(),
            "q_data_max": q_data_max.detach(),
            "q_data_min": q_data_min.detach(),
            "td_target_mean": td_target_mean.detach(),
            "cql_q_rand_mean": cql_q_rand_mean.detach(),
            "q_overestimation_gap": q_overestimation_gap.detach(),
            "cql_alpha": alpha_cql.detach(),
            "cql_effective_alpha": torch.tensor(
                _effective_alpha, device=self.device
            ),
            "cql_loss_scale": torch.tensor(
                _cql_loss_scale, device=self.device
            ),
            "_cql_penalty_raw": cql_penalty_raw.detach(),  # consumed by learn()
        }

    # ------------------------------------------------------------------
    # Actor update (algorithm-neutral SAC + BC; ported verbatim)
    # ------------------------------------------------------------------
    def _update_actor(self, data: "TensorDict") -> dict[str, "torch.Tensor"]:
        scaler = self.scaler
        args = self.config

        with self._maybe_amp():
            observations = data["observations"]
            critic_obs = data["critic_observations"]

            actions_new, log_probs = self.actor.get_actions_and_log_probs(
                observations
            )
            with torch.no_grad():
                _, _, log_std = self.actor(observations)
                action_std = log_std.exp().mean()
                policy_entropy = -log_probs.mean()

            min_q = self.qnet.min_q(critic_obs, actions_new).squeeze(-1)
            alpha = self.log_alpha.exp().detach()

            with torch.no_grad():
                _q_norm_raw_adaptive = max(min_q.abs().mean().item(), 1.0)
                _qn_mode = str(getattr(args, "q_normalizer_mode", "adaptive"))
                _qn_min = max(float(getattr(args, "q_normalizer_min", 1.0)), 1.0)
                _floor = lambda v: max(float(v), _qn_min)
                if _qn_mode == "slow_ema":
                    _tau = float(getattr(args, "q_normalizer_ema_tau", 0.005))
                    if self._q_normalizer_ema is None:
                        self._q_normalizer_ema = _q_norm_raw_adaptive
                    else:
                        self._q_normalizer_ema = (
                            (1.0 - _tau) * self._q_normalizer_ema
                            + _tau * _q_norm_raw_adaptive
                        )
                    _q_norm_active = _floor(self._q_normalizer_ema)
                elif _qn_mode == "freeze_at_step":
                    _fs = int(getattr(args, "q_normalizer_freeze_step", 0))
                    if int(self.global_step) < _fs:
                        _q_norm_active = _floor(_q_norm_raw_adaptive)
                    else:
                        if self._q_normalizer_frozen is None:
                            self._q_normalizer_frozen = _q_norm_raw_adaptive
                        _q_norm_active = _floor(self._q_normalizer_frozen)
                else:
                    _q_norm_active = _floor(_q_norm_raw_adaptive)
                q_normalizer = _q_norm_raw_adaptive
            normalized_q = min_q / _q_norm_active

            bc_weight = getattr(args, "bc_weight", 0.0)
            if bc_weight > 0.0:
                bc_loss = F.mse_loss(actions_new, data["actions"])
            else:
                bc_loss = torch.tensor(0.0, device=self.device)

            rl_term = (alpha * log_probs - normalized_q).mean()
            actor_loss = rl_term + bc_weight * bc_loss

            with torch.no_grad():
                action_l2_vs_data = (
                    (actions_new - data["actions"]) ** 2
                ).sum(dim=-1).mean()
                action_mae_vs_data = (actions_new - data["actions"]).abs().mean()

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
            "q_normalizer": torch.tensor(q_normalizer, device=self.device),
            "normalized_q_term_mean": normalized_q.mean().detach(),
            "rl_actor_term": rl_term.detach(),
            "action_l2_vs_data": action_l2_vs_data,
            "action_mae_vs_data": action_mae_vs_data,
        }

    # ------------------------------------------------------------------
    # Alpha update (verbatim port)
    # ------------------------------------------------------------------
    def _update_alpha(
        self, log_probs: "torch.Tensor"
    ) -> dict[str, "torch.Tensor"]:
        scaler = self.scaler
        metrics: dict[str, "torch.Tensor"] = {}

        if self.config.use_autotune:
            self.alpha_optimizer.zero_grad(set_to_none=True)
            with self._maybe_amp():
                alpha_loss = (
                    -self.log_alpha.exp()
                    * (log_probs.detach() + self.target_entropy)
                ).mean()
            scaler.scale(alpha_loss).backward()
            scaler.unscale_(self.alpha_optimizer)
            scaler.step(self.alpha_optimizer)
            scaler.update()
            _alpha_min = getattr(self.config, "alpha_min", None)
            _log_alpha_min = (
                math.log(_alpha_min) if _alpha_min else math.log(1e-8)
            )
            with torch.no_grad():
                self.log_alpha.clamp_(min=_log_alpha_min, max=math.log(10.0))
            metrics["alpha_loss"] = alpha_loss.detach()
        else:
            metrics["alpha_loss"] = torch.tensor(0.0, device=self.device)

        if self.config.cql_alpha_autotune:
            self.alpha_cql_optimizer.zero_grad(set_to_none=True)
            with self._maybe_amp():
                alpha_cql_loss = (
                    self.log_alpha_cql.exp()
                    * (
                        self._last_cql_penalty.detach()
                        - self.config.cql_target_penalty
                    )
                )
            scaler.scale(alpha_cql_loss).backward()
            scaler.unscale_(self.alpha_cql_optimizer)
            scaler.step(self.alpha_cql_optimizer)
            scaler.update()
            with torch.no_grad():
                self.log_alpha_cql.clamp_(
                    min=math.log(1e-6), max=math.log(1e6)
                )
            metrics["alpha_cql_loss"] = alpha_cql_loss.detach()

        return metrics

    # ------------------------------------------------------------------
    # Evaluation — canonical algorithm-neutral helpers in
    # :mod:`offline_rl.common.eval_utils`.
    #
    # The common helpers operate through the shared ``self.<attribute>``
    # contract populated by :class:`OfflineRLAgentBase` (actor, qnet,
    # qnet_target, log_alpha, obs_normalizer, critic_obs_normalizer,
    # env, scaler, ...).  This keeps CQL / SMQR / SMQR-SG eval routing
    # consistent without importing legacy agent descriptors.
    # ------------------------------------------------------------------
    def evaluate_policy(
        self, max_eval_steps: int | None = None
    ) -> dict[str, float]:
        return run_evaluate_policy(self, max_eval_steps)

    def _create_eval_callbacks(self) -> None:  # type: ignore[override]
        return _shim_create_eval_callbacks(self)

    def _run_eval_rollouts(self, *args, **kwargs):  # type: ignore[override]
        return _shim_run_eval_rollouts(self, *args, **kwargs)

    def _pre_evaluate_policy(self) -> None:  # type: ignore[override]
        return _shim_pre_evaluate_policy(self)

    def _post_evaluate_policy(self) -> None:  # type: ignore[override]
        return _shim_post_evaluate_policy(self)

    def _pre_eval_env_step(self, actor_state: dict) -> dict:  # type: ignore[override]
        return _shim_pre_eval_env_step(self, actor_state)

    def _post_eval_env_step(self, actor_state: dict) -> dict:  # type: ignore[override]
        return _shim_post_eval_env_step(self, actor_state)


__all__ = ["CQLAgent"]
