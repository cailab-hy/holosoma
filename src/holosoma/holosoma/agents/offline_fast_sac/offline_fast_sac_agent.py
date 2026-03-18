from __future__ import annotations

import copy
import math
import os
from contextlib import contextmanager
from typing import Any, Callable

import tqdm
from loguru import logger

from holosoma.agents.base_algo.base_algo import BaseAlgo
from holosoma.agents.fast_sac.fast_sac import Actor, Critic
from holosoma.agents.fast_sac.fast_sac_utils import (
    EmpiricalNormalization,
    save_params,
)
from holosoma.agents.modules.logging_utils import LoggingHelper
from holosoma.agents.offline_fast_sac.offline_fast_sac_utils import (
    OfflineReplayBuffer,
    init_normalizer_from_dataset,
)
from holosoma.config_types.algo import FastSACConfig
from holosoma.envs.base_task.base_task import BaseTask
from holosoma.utils.average_meters import TensorAverageMeterDict
from holosoma.utils.safe_torch_import import (
    F,
    GradScaler,
    TensorboardSummaryWriter,
    TensorDict,
    autocast,
    nn,
    optim,
    torch,
)

torch.set_float32_matmul_precision("high")


class OfflineFastSACAgent(BaseAlgo):
    """Offline variant of FastSAC that trains from a static HDF5 dataset.

    Reuses the same ``Actor`` / distributional ``Critic`` architecture and the
    same update equations (``_update_main``, ``_update_pol``) as the online
    ``FastSACAgent``.  The only structural difference is that the training loop
    samples from an ``OfflineReplayBuffer`` instead of collecting online
    rollouts from the environment.

    An environment may optionally be provided for periodic evaluation, but it
    is **never** used for data collection.
    """

    config: FastSACConfig
    actor: Actor
    qnet: Critic

    def __init__(
        self,
        config: FastSACConfig,
        device: str,
        log_dir: str,
        env: BaseTask | None = None,
        multi_gpu_cfg: dict | None = None,
    ):
        if not config.offline_mode:
            raise ValueError("OfflineFastSACAgent requires config.offline_mode=True")
        if not config.offline_dataset_path:
            raise ValueError("OfflineFastSACAgent requires config.offline_dataset_path to be set")
        if config.actor_obs_dim <= 0 or config.critic_obs_dim <= 0 or config.action_dim <= 0:
            raise ValueError(
                "OfflineFastSACAgent requires config.actor_obs_dim, critic_obs_dim, "
                "and action_dim to be positive integers"
            )

        # BaseAlgo expects an env — pass a dummy-safe env (or the real one for eval)
        super().__init__(env, config, device, multi_gpu_cfg)  # type: ignore[arg-type]
        self.eval_env = env  # may be None; only used for evaluation
        self.log_dir = log_dir
        self.global_step = 0
        self.writer = TensorboardSummaryWriter(log_dir=self.log_dir, flush_secs=10)
        self.logging_helper = LoggingHelper(
            self.writer,
            self.log_dir,
            device=self.device,
            num_envs=1,  # offline — no parallel envs
            num_steps_per_env=config.logging_interval,
            num_learning_iterations=config.num_learning_iterations,
            is_main_process=self.is_main_process,
            num_gpus=self.gpu_world_size,
        )
        self.training_metrics = TensorAverageMeterDict()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self) -> None:
        logger.info("Setting up OfflineFastSACAgent (offline mode)")
        args = self.config
        device = self.device

        actor_obs_dim = args.actor_obs_dim
        critic_obs_dim = args.critic_obs_dim
        n_act = args.action_dim

        # In offline mode, we use a single flat obs key per type.
        self.actor_obs_indices = {
            "obs": {"start": 0, "end": actor_obs_dim, "size": actor_obs_dim},
        }
        self.critic_obs_indices = {
            "obs": {"start": 0, "end": critic_obs_dim, "size": critic_obs_dim},
        }
        self.actor_obs_dim = actor_obs_dim

        # --- GradScaler ---
        self.scaler = GradScaler(enabled=args.amp)

        # --- Observation normalization ---
        self.obs_normalization = args.obs_normalization
        if args.obs_normalization:
            self.obs_normalizer: nn.Module = EmpiricalNormalization(shape=actor_obs_dim, device=device)
            self.critic_obs_normalizer: nn.Module = EmpiricalNormalization(shape=critic_obs_dim, device=device)
        else:
            self.obs_normalizer = nn.Identity()
            self.critic_obs_normalizer = nn.Identity()

        # --- Action scaling ---
        if args.offline_action_scale:
            action_scale = torch.tensor(args.offline_action_scale, device=device, dtype=torch.float)
            assert action_scale.shape == (n_act,), (
                f"offline_action_scale length {action_scale.shape[0]} != action_dim {n_act}"
            )
        else:
            action_scale = torch.ones(n_act, device=device)
        if not args.use_tanh:
            action_scale = torch.ones(n_act, device=device)
        action_bias = torch.zeros(n_act, device=device)

        # --- Actor ---
        obs_keys_for_net = ["obs"]
        self.actor = Actor(
            obs_indices=self.actor_obs_indices,
            obs_keys=obs_keys_for_net,
            n_act=n_act,
            num_envs=1,
            device=device,
            hidden_dim=args.actor_hidden_dim,
            log_std_max=args.log_std_max,
            log_std_min=args.log_std_min,
            use_tanh=args.use_tanh,
            use_layer_norm=args.use_layer_norm,
            action_scale=action_scale,
            action_bias=action_bias,
        )

        # --- Critic + target ---
        critic_obs_keys_for_net = ["obs"]
        self.qnet = Critic(
            obs_indices=self.critic_obs_indices,
            obs_keys=critic_obs_keys_for_net,
            n_act=n_act,
            num_atoms=args.num_atoms,
            v_min=args.v_min,
            v_max=args.v_max,
            hidden_dim=args.critic_hidden_dim,
            device=device,
            use_layer_norm=args.use_layer_norm,
            num_q_networks=args.num_q_networks,
        )

        print(self.actor)
        print(self.qnet)

        self.log_alpha = torch.tensor([math.log(args.alpha_init)], requires_grad=True, device=device)
        self.policy = self.actor.explore

        self.qnet_target = Critic(
            obs_indices=self.critic_obs_indices,
            obs_keys=critic_obs_keys_for_net,
            n_act=n_act,
            num_atoms=args.num_atoms,
            v_min=args.v_min,
            v_max=args.v_max,
            hidden_dim=args.critic_hidden_dim,
            device=device,
            use_layer_norm=args.use_layer_norm,
            num_q_networks=args.num_q_networks,
        )
        self.qnet_target.load_state_dict(self.qnet.state_dict())

        # --- Optimizers ---
        self.q_optimizer = optim.AdamW(
            list(self.qnet.parameters()),
            lr=args.critic_learning_rate,
            weight_decay=args.weight_decay,
            fused=True,
            betas=(0.9, 0.95),
        )
        self.actor_optimizer = optim.AdamW(
            list(self.actor.parameters()),
            lr=args.actor_learning_rate,
            weight_decay=args.weight_decay,
            fused=True,
            betas=(0.9, 0.95),
        )
        self.target_entropy = -n_act * args.target_entropy_ratio
        self.alpha_optimizer = optim.AdamW(
            [self.log_alpha], lr=args.alpha_learning_rate, fused=True, betas=(0.9, 0.95)
        )

        logger.info(f"actor_obs_dim: {actor_obs_dim}, critic_obs_dim: {critic_obs_dim}, action_dim: {n_act}")

        # --- Offline replay buffer ---
        self.rb = OfflineReplayBuffer(
            dataset_path=args.offline_dataset_path,
            device=device,
        )
        # Validate dimensions match config
        assert self.rb.n_obs == actor_obs_dim, (
            f"Dataset actor_obs_dim {self.rb.n_obs} != config actor_obs_dim {actor_obs_dim}"
        )
        assert self.rb.n_critic_obs == critic_obs_dim, (
            f"Dataset critic_obs_dim {self.rb.n_critic_obs} != config critic_obs_dim {critic_obs_dim}"
        )
        assert self.rb.n_act == n_act, (
            f"Dataset action_dim {self.rb.n_act} != config action_dim {n_act}"
        )

        # --- Initialize normalizers from dataset statistics ---
        if args.obs_normalization and args.offline_normalizer_init_mode == "dataset":
            obs_mean, obs_std = self.rb.compute_obs_statistics()
            init_normalizer_from_dataset(self.obs_normalizer, obs_mean, obs_std)  # type: ignore[arg-type]
            logger.info("Initialized obs_normalizer from dataset statistics (frozen)")

            critic_mean, critic_std = self.rb.compute_critic_obs_statistics()
            init_normalizer_from_dataset(self.critic_obs_normalizer, critic_mean, critic_std)  # type: ignore[arg-type]
            logger.info("Initialized critic_obs_normalizer from dataset statistics (frozen)")
        elif args.obs_normalization and args.offline_normalizer_init_mode == "none":
            # No normalization even though obs_normalization=True
            self.obs_normalizer = nn.Identity()
            self.critic_obs_normalizer = nn.Identity()
            self.obs_normalization = False
            logger.info("obs_normalization disabled by offline_normalizer_init_mode='none'")

        # --- Multi-GPU sync ---
        if self.is_multi_gpu:
            self._synchronize_model_parameters()

    # ------------------------------------------------------------------
    # AMP context
    # ------------------------------------------------------------------

    @contextmanager
    def _maybe_amp(self):
        amp_dtype = torch.bfloat16 if self.config.amp_dtype == "bf16" else torch.float16
        with autocast(device_type="cuda", dtype=amp_dtype, enabled=self.config.amp):
            yield

    # ------------------------------------------------------------------
    # Multi-GPU helpers (identical to FastSACAgent)
    # ------------------------------------------------------------------

    def _synchronize_model_parameters(self):
        for param in self.actor.parameters():
            torch.distributed.broadcast(param.data, src=0)
        for param in self.qnet.parameters():
            torch.distributed.broadcast(param.data, src=0)
        torch.distributed.broadcast(self.log_alpha.data, src=0)
        self.qnet_target.load_state_dict(self.qnet.state_dict())
        logger.info(f"Synchronized model parameters across {self.gpu_world_size} GPUs")

    def _all_reduce_model_grads(self, model: nn.Module) -> None:
        if not self.is_multi_gpu:
            return
        grads = [p.grad.view(-1) for p in model.parameters() if p.grad is not None]
        if not grads:
            return
        flat = torch.cat(grads)
        torch.distributed.all_reduce(flat, op=torch.distributed.ReduceOp.SUM)
        flat /= self.gpu_world_size
        offset = 0
        for p in model.parameters():
            if p.grad is not None:
                n = p.numel()
                p.grad.copy_(flat[offset : offset + n].view_as(p.grad))
                offset += n

    # ------------------------------------------------------------------
    # Update methods — reused from FastSACAgent with no equation changes
    # ------------------------------------------------------------------

    def _update_main(
        self, data: TensorDict
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Critic update + alpha update.  Identical to ``FastSACAgent._update_main``.

        The only extension point is the ``_compute_conservative_critic_loss``
        hook called after the standard distributional critic loss.
        """
        args = self.config

        scaler = self.scaler
        actor = self.actor
        qnet = self.qnet
        qnet_target = self.qnet_target
        q_optimizer = self.q_optimizer
        alpha_optimizer = self.alpha_optimizer

        with self._maybe_amp():
            next_observations = data["next"]["observations"]
            critic_observations = data["critic_observations"]
            next_critic_observations = data["next"]["critic_observations"]
            actions = data["actions"]
            rewards = data["next"]["rewards"]
            dones = data["next"]["dones"].bool()
            truncations = data["next"]["truncations"].bool()
            bootstrap = (truncations | ~dones).float()

            with torch.no_grad():
                next_state_actions, next_state_log_probs = actor.get_actions_and_log_probs(next_observations)
                discount = args.gamma ** data["next"]["effective_n_steps"]

                target_distributions = qnet_target.projection(
                    next_critic_observations,
                    next_state_actions,
                    rewards - discount * bootstrap * self.log_alpha.exp() * next_state_log_probs,
                    bootstrap,
                    discount,
                )
                target_values = qnet_target.get_value(target_distributions)
                target_value_max = target_values.max()
                target_value_min = target_values.min()

            q_outputs = qnet(critic_observations, actions)
            critic_log_probs = F.log_softmax(q_outputs, dim=-1)
            critic_losses = -torch.sum(target_distributions * critic_log_probs, dim=-1)
            qf_loss = critic_losses.mean(dim=1).sum(dim=0)

            # --- CODAC / SMQR conservative loss hook ---
            # Override ``_compute_conservative_critic_loss`` in a subclass to
            # add a conservative penalty (e.g. CODAC, SMQR).  The default
            # implementation returns 0.
            conservative_critic_loss = self._compute_conservative_critic_loss(
                data, q_outputs, critic_observations, actions
            )
            qf_loss = qf_loss + args.conservative_weight * conservative_critic_loss

        q_optimizer.zero_grad(set_to_none=True)
        scaler.scale(qf_loss).backward()

        if self.is_multi_gpu:
            self._all_reduce_model_grads(qnet)

        scaler.unscale_(q_optimizer)
        if args.max_grad_norm > 0:
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                qnet.parameters(),
                max_norm=args.max_grad_norm,
            )
        else:
            critic_grad_norm = torch.tensor(0.0, device=self.device)
        scaler.step(q_optimizer)
        scaler.update()

        alpha_loss = torch.tensor(0.0, device=self.device)
        if self.config.use_autotune:
            alpha_optimizer.zero_grad(set_to_none=True)
            with self._maybe_amp():
                alpha_loss = (-self.log_alpha.exp() * (next_state_log_probs.detach() + self.target_entropy)).mean()

            scaler.scale(alpha_loss).backward()

            if self.is_multi_gpu:
                if self.log_alpha.grad is not None:
                    torch.distributed.all_reduce(self.log_alpha.grad.data, op=torch.distributed.ReduceOp.SUM)
                    self.log_alpha.grad.data.copy_(self.log_alpha.grad.data / self.gpu_world_size)

            scaler.unscale_(alpha_optimizer)
            scaler.step(alpha_optimizer)
            scaler.update()

        return (
            rewards.mean(),
            critic_grad_norm.detach(),
            qf_loss.detach(),
            target_value_max.detach(),
            target_value_min.detach(),
            alpha_loss.detach(),
        )

    def _update_pol(self, data: TensorDict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Actor update.  Identical to ``FastSACAgent._update_pol``.

        The only extension point is the ``_compute_conservative_actor_loss``
        hook called after the standard actor loss.
        """
        actor = self.actor
        qnet = self.qnet
        actor_optimizer = self.actor_optimizer
        scaler = self.scaler
        args = self.config

        with self._maybe_amp():
            critic_observations = data["critic_observations"]

            actions, log_probs = actor.get_actions_and_log_probs(data["observations"])
            with torch.no_grad():
                _, _, log_std = actor(data["observations"])
                action_std = log_std.exp().mean()
                policy_entropy = -log_probs.mean()

            q_outputs = qnet(critic_observations, actions)
            q_probs = F.softmax(q_outputs, dim=-1)
            q_values = qnet.get_value(q_probs)
            qf_value = q_values.mean(dim=0)
            actor_loss = (self.log_alpha.exp().detach() * log_probs - qf_value).mean()

            # --- CODAC / SMQR conservative actor loss hook ---
            conservative_actor_loss = self._compute_conservative_actor_loss(data, actions, log_probs)
            actor_loss = actor_loss + conservative_actor_loss

        actor_optimizer.zero_grad(set_to_none=True)
        scaler.scale(actor_loss).backward()

        if self.is_multi_gpu:
            self._all_reduce_model_grads(actor)

        scaler.unscale_(actor_optimizer)
        if args.max_grad_norm > 0:
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                actor.parameters(),
                max_norm=args.max_grad_norm,
            )
        else:
            actor_grad_norm = torch.tensor(0.0, device=self.device)
        scaler.step(actor_optimizer)
        scaler.update()
        return (
            actor_grad_norm.detach(),
            actor_loss.detach(),
            policy_entropy.detach(),
            action_std.detach(),
        )

    # ------------------------------------------------------------------
    # Conservative loss hooks — override in subclasses for CODAC / SMQR
    # ------------------------------------------------------------------

    def _compute_conservative_critic_loss(
        self,
        data: TensorDict,
        q_outputs: torch.Tensor,
        critic_observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Compute conservative regularization for the critic.

        Default returns 0 (standard SAC).  Override in subclasses for CODAC,
        SMQR, or other conservative offline RL methods.

        Parameters
        ----------
        data : TensorDict
            Full batch (contains observations, next obs, etc.).
        q_outputs : torch.Tensor
            Raw Q-network logits ``[num_q, batch, num_atoms]``.
        critic_observations : torch.Tensor
            Critic observations for the current batch.
        actions : torch.Tensor
            Actions from the dataset for the current batch.

        Returns
        -------
        torch.Tensor
            Scalar conservative loss term (will be multiplied by
            ``config.conservative_weight``).
        """
        return torch.tensor(0.0, device=self.device)

    def _compute_conservative_actor_loss(
        self,
        data: TensorDict,
        actions: torch.Tensor,
        log_probs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute conservative regularization for the actor.

        Default returns 0 (standard SAC).  Override in subclasses.
        """
        return torch.tensor(0.0, device=self.device)

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def load(self, ckpt_path: str | None) -> None:
        if not ckpt_path:
            return
        torch_checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        self.actor.load_state_dict(torch_checkpoint["actor_state_dict"])
        self.qnet.load_state_dict(torch_checkpoint["qnet_state_dict"])
        self.qnet_target.load_state_dict(torch_checkpoint["qnet_target_state_dict"])
        self.log_alpha.data.copy_(torch_checkpoint["log_alpha"].to(self.device))
        self.actor_optimizer.load_state_dict(torch_checkpoint["actor_optimizer_state_dict"])
        self.q_optimizer.load_state_dict(torch_checkpoint["q_optimizer_state_dict"])
        self.alpha_optimizer.load_state_dict(torch_checkpoint["alpha_optimizer_state_dict"])
        self.scaler.load_state_dict(torch_checkpoint["grad_scaler_state_dict"])
        self.global_step = torch_checkpoint["global_step"]

        # Restore normalizer if saved (offline_normalizer_init_mode='checkpoint')
        if self.obs_normalization:
            obs_norm_state = torch_checkpoint.get("obs_normalizer_state")
            critic_norm_state = torch_checkpoint.get("critic_obs_normalizer_state")
            if obs_norm_state is not None:
                self.obs_normalizer.load_state_dict(obs_norm_state)
            if critic_norm_state is not None:
                self.critic_obs_normalizer.load_state_dict(critic_norm_state)
            # Keep normalizer frozen in offline mode
            self.obs_normalizer.eval()
            self.critic_obs_normalizer.eval()

        logger.info(f"Loaded checkpoint from '{ckpt_path}' (global_step={self.global_step})")

    def save(self, path: str) -> None:  # type: ignore[override]
        save_params(
            self.global_step,
            self.actor,
            self.qnet,
            self.qnet_target,
            self.log_alpha,
            self.obs_normalizer,
            self.critic_obs_normalizer,
            self.actor_optimizer,
            self.q_optimizer,
            self.alpha_optimizer,
            self.scaler,
            self.config,
            path,
            save_fn=self.logging_helper.save_checkpoint_artifact,
            metadata=self._checkpoint_metadata(iteration=self.global_step),
        )

    # ------------------------------------------------------------------
    # Main offline training loop
    # ------------------------------------------------------------------

    def learn(self) -> None:
        """Offline training loop.

        Samples mini-batches from the pre-loaded ``OfflineReplayBuffer``,
        applies the same critic / actor / alpha / target-network updates as
        online FastSAC, and periodically logs and checkpoints.
        """
        args = self.config
        device = self.device

        # --- Optionally compile hot paths ---
        def _noop_normalize(x: torch.Tensor, **_: Any) -> torch.Tensor:
            return x

        if args.compile:
            update_main = torch.compile(self._update_main)
            update_pol = torch.compile(self._update_pol)
            if self.obs_normalization:
                normalize_obs = torch.compile(self.obs_normalizer.forward)
                normalize_critic_obs = torch.compile(self.critic_obs_normalizer.forward)
            else:
                normalize_obs = _noop_normalize
                normalize_critic_obs = _noop_normalize
        else:
            update_main = self._update_main
            update_pol = self._update_pol
            if self.obs_normalization:
                normalize_obs = self.obs_normalizer.forward
                normalize_critic_obs = self.critic_obs_normalizer.forward
            else:
                normalize_obs = _noop_normalize
                normalize_critic_obs = _noop_normalize

        qnet = self.qnet
        qnet_target = self.qnet_target
        rb = self.rb
        batch_size = args.batch_size

        # Freeze normalizer during training (stats already set from dataset).
        if self.obs_normalization:
            self.obs_normalizer.eval()
            self.critic_obs_normalizer.eval()

        # --- Metric placeholders ---
        policy_entropy = torch.tensor(0.0, device=device)
        action_std = torch.tensor(0.0, device=device)
        actor_loss = torch.tensor(0.0, device=device)
        actor_grad_norm = torch.tensor(0.0, device=device)

        num_epochs = args.num_learning_iterations
        updates_per_epoch = args.offline_num_updates_per_epoch

        pbar = tqdm.tqdm(total=num_epochs, initial=self.global_step, desc="Offline FastSAC")

        while self.global_step <= num_epochs:
            with self.logging_helper.record_learn_time():
                for update_idx in range(updates_per_epoch):
                    # --- Sample from offline buffer ---
                    data = rb.sample(batch_size)

                    # --- Normalize observations (no running update) ---
                    data["observations"] = normalize_obs(data["observations"], update=False)
                    data["next"]["observations"] = normalize_obs(data["next"]["observations"], update=False)
                    data["critic_observations"] = normalize_critic_obs(
                        data["critic_observations"], update=False
                    )
                    data["next"]["critic_observations"] = normalize_critic_obs(
                        data["next"]["critic_observations"], update=False
                    )

                    # --- Critic + alpha update ---
                    (
                        buffer_rewards,
                        critic_grad_norm,
                        qf_loss,
                        qf_max,
                        qf_min,
                        alpha_loss,
                    ) = update_main(data)

                    # --- Actor update (delayed) ---
                    if update_idx % args.policy_frequency == 0:
                        actor_grad_norm, actor_loss, policy_entropy, action_std = update_pol(data)

                    # --- Accumulate metrics ---
                    current_metrics = {
                        "actor_loss": actor_loss,
                        "qf_loss": qf_loss,
                        "qf_max": qf_max,
                        "qf_min": qf_min,
                        "actor_grad_norm": actor_grad_norm,
                        "critic_grad_norm": critic_grad_norm,
                        "buffer_rewards": buffer_rewards,
                        "alpha_loss": alpha_loss,
                        "alpha_value": self.log_alpha.exp().detach().mean(),
                        "policy_entropy": policy_entropy,
                        "action_std": action_std,
                    }
                    self.training_metrics.add(current_metrics)

                    # --- Target network soft update ---
                    with torch.no_grad():
                        src_ps = [p.data for p in qnet.parameters()]
                        tgt_ps = [p.data for p in qnet_target.parameters()]
                        torch._foreach_mul_(tgt_ps, 1.0 - args.tau)
                        torch._foreach_add_(tgt_ps, src_ps, alpha=args.tau)

            # --- Logging ---
            if self.global_step % args.logging_interval == 0:
                with torch.no_grad():
                    accumulated_metrics = self.training_metrics.mean_and_clear()
                    loss_dict = {}
                    for key, value in accumulated_metrics.items():
                        if isinstance(value, torch.Tensor):
                            loss_dict[key] = value.item()
                        else:
                            loss_dict[key] = float(value)

                self.logging_helper.post_epoch_logging(
                    it=self.global_step, loss_dict=loss_dict, extra_log_dicts={}
                )

            # --- Checkpointing ---
            if args.save_interval > 0 and self.global_step > 0 and self.global_step % args.save_interval == 0:
                if self.is_main_process:
                    logger.info(f"Saving model at epoch {self.global_step}")
                    self.save(os.path.join(self.log_dir, f"model_{self.global_step:07d}.pt"))

            # --- Optional evaluation ---
            if (
                args.eval_interval > 0
                and self.global_step > 0
                and self.global_step % args.eval_interval == 0
                and self.eval_env is not None
            ):
                self.evaluate_policy()

            if self.global_step >= num_epochs:
                break
            self.global_step += 1
            pbar.update(1)

        # --- Final save ---
        if self.is_main_process:
            self.save(os.path.join(self.log_dir, f"model_{self.global_step:07d}.pt"))

    # ------------------------------------------------------------------
    # Inference / evaluation (env-dependent — optional)
    # ------------------------------------------------------------------

    def get_inference_policy(self, device: str | None = None) -> Callable[[dict[str, torch.Tensor]], torch.Tensor]:
        device = device or self.device
        policy = self.actor.to(device)
        obs_normalizer = self.obs_normalizer.to(device)
        policy.eval()
        obs_normalizer.eval()

        def policy_fn(obs: dict[str, torch.Tensor]) -> torch.Tensor:
            if self.obs_normalization:
                normalized_obs = obs_normalizer(obs["actor_obs"], update=False)
            else:
                normalized_obs = obs["actor_obs"]
            return policy(normalized_obs)[0]

        return policy_fn

    @property
    def actor_onnx_wrapper(self):
        actor = copy.deepcopy(self.actor).to("cpu")
        obs_normalizer = copy.deepcopy(self.obs_normalizer).to("cpu")

        class ActorWrapper(nn.Module):
            def __init__(self, actor, obs_normalizer):
                super().__init__()
                self.actor = actor
                self.obs_normalizer = obs_normalizer

            def forward(self, actor_obs):
                if self.obs_normalizer is not None:
                    normalized_obs = self.obs_normalizer(actor_obs, update=False)
                else:
                    normalized_obs = actor_obs
                return self.actor(normalized_obs)[0]

        return ActorWrapper(actor, obs_normalizer if self.obs_normalization else None)

    @torch.no_grad()
    def evaluate_policy(self, max_eval_steps: int | None = None):
        if self.eval_env is None:
            logger.warning("evaluate_policy called but no environment was provided — skipping.")
            return
        logger.info("Running evaluation rollout...")
        # Minimal evaluation — delegates to the environment
        from holosoma.agents.fast_sac.fast_sac_agent import FastSACEnv

        env = FastSACEnv(self.eval_env, self.config.actor_obs_keys, self.config.critic_obs_keys)
        env.set_is_evaluating()
        obs = env.reset()
        import itertools

        for _ in itertools.islice(itertools.count(), max_eval_steps):
            if self.obs_normalization:
                normalized_obs = self.obs_normalizer(obs, update=False)
            else:
                normalized_obs = obs
            actions = self.actor(normalized_obs)[0]
            obs, _, _, _ = env.step(actions)
