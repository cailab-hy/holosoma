from __future__ import annotations

from dataclasses import field
from typing import Any, List, Union

from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class OptimizerConfig:
    """Configuration for optimizer settings."""

    _target_: str
    """Target optimizer class (e.g., torch.optim.AdamW)."""

    weight_decay: float = 0.001
    """Weight decay parameter for the optimizer."""


@dataclass(frozen=True)
class LayerConfig:
    """Configuration for neural network layer settings."""

    hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    """List of hidden layer dimensions."""

    activation: str = "ELU"
    """Activation function name."""

    dropout_prob: float = 0.0
    """Dropout probability."""

    use_layer_norm: bool = False
    """Whether to use layer normalization."""

    encoder_activation: str = "ELU"
    """Activation function name for encoder layers."""

    encoder_output_dim: int | None = None
    """Output dimension for encoder. Only used for encoder modules."""

    encoder_hidden_dims: List[int] | None = None
    """Hidden dimensions for encoder. Only used for encoder modules."""

    encoder_input_name: str = ""
    """Input name for encoder. Only used for encoder modules."""

    input_channels: int = 1
    """Number of input channels. Only used for CNN modules."""

    input_height: int = 1
    """Height of input feature maps. Only used for CNN modules."""

    input_width: int = 1
    """Width of input feature maps. Only used for CNN modules."""

    hidden_channels: tuple[int, ...] | None = None
    """Hidden channel dimensions. Only used for CNN modules."""

    kernel_size: int | tuple[int, ...] = 3
    """Kernel size for convolutions. Only used for CNN modules."""

    stride: int | tuple[int, ...] = 1
    """Stride for convolutions. Only used for CNN modules."""

    padding: str | int | tuple[str | int, ...] = "same"
    """Padding mode for convolutions. Only used for CNN modules."""

    module_input_name: tuple[str, ...] = ()
    """Input names for module. Only used for encoder modules."""


@dataclass(frozen=True)
class ModuleConfig:
    """Configuration for neural network modules."""

    type: str
    """Module type (e.g., MLP)."""

    input_dim: List[str] = field(default_factory=list)
    """Input dimension specification."""

    output_dim: List[str | int] = field(default_factory=list)
    """Output dimension specification."""

    layer_config: LayerConfig = field(default_factory=LayerConfig)
    """Layer configuration settings."""

    min_noise_std: float | None = None
    """Minimum noise standard deviation."""

    min_mean_noise_std: float | None = None
    """Minimum mean noise standard deviation."""


@dataclass(frozen=True)
class PPOModuleDictConfig:
    """Configuration for PPO module dictionary."""

    actor: ModuleConfig
    """Actor module configuration."""

    critic: ModuleConfig
    """Critic module configuration."""


@dataclass(frozen=True)
class PPOConfig:
    """Configuration for PPO algorithm."""

    module_dict: PPOModuleDictConfig
    """PPO module configurations (actor, critic)."""

    num_learning_epochs: int = 8
    """Number of learning epochs per update."""

    num_mini_batches: int = 4
    """Number of mini-batches per epoch."""

    clip_param: float = 0.2
    """PPO clipping parameter."""

    gamma: float = 0.99
    """Discount factor for future rewards."""

    lam: float = 0.95
    """GAE lambda parameter."""

    value_loss_coef: float = 1.0
    """Value loss coefficient."""

    entropy_coef: float = 0.01
    """Entropy coefficient for exploration."""

    actor_learning_rate: float = 1e-5
    """Learning rate for actor network."""

    actor_optimizer: OptimizerConfig = field(default_factory=lambda: OptimizerConfig(_target_="torch.optim.AdamW"))
    """Actor optimizer configuration."""

    critic_learning_rate: float = 1e-5
    """Learning rate for critic network."""

    critic_optimizer: OptimizerConfig = field(default_factory=lambda: OptimizerConfig(_target_="torch.optim.AdamW"))
    """Critic optimizer configuration."""

    max_grad_norm: float = 1.0
    """Maximum gradient norm for clipping."""

    schedule: str = "adaptive"
    """Learning rate schedule type."""

    desired_kl: float = 0.01
    """Desired KL divergence for adaptive learning rate."""

    use_symmetry: bool = False
    """Whether to use symmetry in training."""

    symmetry_actor_coef: float = 1.0
    """Symmetry coefficient for actor."""

    symmetry_critic_coef: float = 0.0
    """Symmetry coefficient for critic."""

    num_steps_per_env: int = 24
    """Number of steps per environment."""

    save_interval: int = 100
    """Interval for saving model checkpoints."""

    load_optimizer: bool = True
    """Whether to load optimizer state."""

    init_noise_std: float = 0.8
    """Initial noise standard deviation."""

    num_learning_iterations: int = 1000000
    """Total number of learning iterations."""

    init_at_random_ep_len: bool = True
    """Whether to initialize at random episode length."""

    empirical_normalization: bool = False
    """Whether to apply empirical normalization to actor and critic observations."""

    eval_callbacks: Any = None
    """Evaluation callbacks configuration."""

    max_actor_learning_rate: float | None = None
    min_actor_learning_rate: float | None = None
    max_critic_learning_rate: float | None = None
    min_critic_learning_rate: float | None = None


@dataclass(frozen=True)
class FastSACConfig:
    num_learning_iterations: int = 25000
    """total timesteps of the experiments"""

    critic_learning_rate: float = 3e-4
    """the learning rate of the critic"""

    actor_learning_rate: float = 3e-4
    """the learning rate for the actor"""

    alpha_learning_rate: float = 3e-4
    """the learning rate for the alpha"""

    buffer_size: int = 1024
    """the replay memory buffer size per environment"""

    num_steps: int = 1
    """the number of steps to use for the multi-step return"""

    gamma: float = 0.97
    """the discount factor gamma"""

    tau: float = 0.125
    """target smoothing coefficient (default: 0.005)"""

    batch_size: int = 8192
    """the batch size of sample from the replay memory"""

    learning_starts: int = 10
    """timestep to start learning"""

    policy_frequency: int = 4
    """the frequency of training policy (delayed)"""

    num_updates: int = 8
    """the number of updates to perform per step"""

    target_entropy_ratio: float = 0.0
    """the ratio of the target entropy to the number of actions"""

    num_atoms: int = 101
    """the number of atoms"""

    v_min: float = -20.0
    """the minimum value of the support"""

    v_max: float = 20.0
    """the maximum value of the support"""

    critic_hidden_dim: int = 768
    """the hidden dimension of the critic network"""

    actor_hidden_dim: int = 512
    """the hidden dimension of the actor network"""

    use_symmetry: bool = False
    """whether to use symmetry"""

    alpha_init: float = 0.001
    """the initial value of the alpha"""

    use_autotune: bool = True
    """whether to use autotune for the alpha"""

    use_tanh: bool = True
    """whether to use tanh for the action"""

    log_std_max: float = 0.0
    """the maximum value of the log std"""

    log_std_min: float = -5.0
    """the minimum value of the log std"""

    compile: bool = True
    """whether to use torch.compile."""

    obs_normalization: bool = True
    """whether to enable observation normalization"""

    use_layer_norm: bool = True
    """whether to use layer normalization"""

    num_q_networks: int = 2
    """number of Q-networks to ensemble"""

    max_grad_norm: float = 0.0
    """the maximum gradient norm"""

    amp: bool = True
    """whether to use amp"""

    amp_dtype: str = "bf16"
    """the dtype of the amp"""

    weight_decay: float = 0.001
    """the weight decay of the optimizer"""

    save_interval: int = 1000
    """the interval to save the model"""

    logging_interval: int = 100
    """the interval to log the metrics"""

    encoder_obs_key: str = "perception_obs"
    """the key of the encoder observation. only valid if use_cnn_encoder is True"""

    encoder_obs_shape: tuple[int, int, int] = (1, 13, 9)
    """the shape of the encoder observation. only valid if use_cnn_encoder is True"""

    use_cnn_encoder: bool = False
    """whether to use CNN for the encoder"""

    actor_obs_keys: List[str] = field(default_factory=lambda: ["actor_obs"])
    critic_obs_keys: List[str] = field(default_factory=lambda: ["critic_obs"])

    eval_callbacks: Any = None
    """Evaluation callbacks configuration."""


@dataclass(frozen=True)
class PPOAlgoConfig:
    """Configuration for algorithm wrapper."""

    _target_: str
    """Target algorithm class."""

    _recursive_: bool
    """Whether to recursively instantiate."""

    config: PPOConfig
    """Algorithm-specific configuration."""


@dataclass(frozen=True)
class FastSACAlgoConfig:
    """Configuration for algorithm wrapper."""

    _target_: str
    """Target algorithm class."""

    _recursive_: bool
    """Whether to recursively instantiate."""

    config: FastSACConfig
    """Algorithm-specific configuration."""


@dataclass(frozen=True)
class OfflineCQLConfig:
    """Configuration for the Offline CQL algorithm.

    Shares many hyperparameters with :class:`FastSACConfig` (same actor,
    same SAC entropy-temperature) but replaces the online replay buffer
    with an offline HDF5 dataset and adds the CQL conservative penalty.

    Tier A — hard-required (no sensible global default):
        actor_obs_keys, critic_obs_keys, dataset_path, obs_normalization,
        actor_hidden_dim, critic_hidden_dim, actor_learning_rate,
        critic_learning_rate, alpha_learning_rate, alpha_init,
        use_autotune, target_entropy_ratio, gamma, tau, batch_size,
        num_learning_iterations, policy_frequency, logging_interval,
        save_interval, cql_num_random_actions, cql_num_policy_actions,
        cql_alpha_autotune, amp, amp_dtype, max_grad_norm.

    Tier B — optional with safe defaults (see field defaults below).

    Tier C — feature-dependent:
        cql_target_penalty — required only when cql_alpha_autotune=True.
    """

    # ── Dataset ─────────────────────────────────────────────────────
    dataset_path: str = ""
    """Path to the offline HDF5 dataset file."""

    # ── Observation keys ───────────────────────────────────────────
    actor_obs_keys: List[str] = field(default_factory=lambda: ["actor_obs"])
    """Observation group keys fed to the actor."""

    critic_obs_keys: List[str] = field(default_factory=lambda: ["critic_obs"])
    """Observation group keys fed to the critic."""

    # ── Training schedule ──────────────────────────────────────────
    num_learning_iterations: int = 100_000
    """Total number of offline gradient steps."""

    batch_size: int = 256
    """Mini-batch size per gradient step."""

    policy_frequency: int = 2
    """Actor is updated every `policy_frequency` critic steps."""

    logging_interval: int = 100
    """Log training metrics every N steps."""

    save_interval: int = 5000
    """Save a checkpoint every N steps (0 = disabled)."""

    # ── Network architecture ───────────────────────────────────────
    actor_hidden_dim: int = 256
    """Hidden dimension of the actor MLP."""

    critic_hidden_dim: int = 256
    """Hidden dimension of each Q-network MLP."""

    # ── SAC entropy temperature ────────────────────────────────────
    alpha_init: float = 0.2
    """Initial value of the SAC entropy temperature."""

    use_autotune: bool = True
    """Whether to auto-tune the SAC entropy temperature α."""

    target_entropy_ratio: float = 0.9
    """Target entropy as a fraction of −action_dim."""

    alpha_learning_rate: float = 3e-4
    """Learning rate for the SAC α optimiser."""

    # ── Optimisation ───────────────────────────────────────────────
    actor_learning_rate: float = 3e-4
    """Learning rate for the actor optimiser."""

    critic_learning_rate: float = 3e-4
    """Learning rate for the twin-Q critic optimiser."""

    gamma: float = 0.99
    """Discount factor."""

    tau: float = 0.005
    """Polyak averaging coefficient for the target Q-network."""

    # ── CQL-specific ───────────────────────────────────────────────
    cql_num_random_actions: int = 10
    """Number of uniform-random actions for CQL importance sampling."""

    cql_num_policy_actions: int = 10
    """Number of current-policy actions for CQL importance sampling."""

    cql_alpha_autotune: bool = False
    """Whether to auto-tune the CQL Lagrange multiplier α_cql."""

    cql_target_penalty: float = 5.0
    """Target CQL penalty (used only when cql_alpha_autotune=True)."""

    cql_alpha_init: float = 1.0
    """Initial value of the CQL conservative weight α_cql."""

    cql_alpha_learning_rate: float = 3e-4
    """Learning rate for the CQL α_cql optimiser."""

    bc_weight: float = 0.0
    """Weight of the BC (behaviour-cloning) MSE regulariser on the actor.

    When > 0 the actor loss becomes:
        actor_loss = (α·log π − min Q).mean() + bc_weight · MSE(π(s), a_data)
    Set to 0.0 to disable (default for backward compatibility)."""

    bc_weight_final: float | None = None
    """Reserved for future bc_weight decay scheduling.

    When ``None`` (default), bc_weight is constant throughout training.
    Checkpoint-compatibility field only — no decay logic is implemented."""

    # ── q_normalizer (actor RL term divisor) ───────────────────────
    q_normalizer_mode: str = "adaptive"
    """How the q_normalizer used in the actor RL term is updated.

    Supported values:
      - ``"adaptive"`` (default, legacy): per-batch
        ``q_normalizer = max(|min_q|.mean(), 1.0)`` — recomputed
        every actor update.
      - ``"slow_ema"``: maintain an EMA buffer
        ``ema ← (1−τ)·ema + τ·raw_adaptive`` and use it as the
        active divisor.  Initial value = first batch's raw_adaptive.
      - ``"freeze_at_step"``: behave as ``adaptive`` until
        ``global_step >= q_normalizer_freeze_step``, then freeze
        the active divisor at the value captured exactly at that
        step.

    In all modes the active divisor is clamped to
    ``[max(q_normalizer_min, 1.0), +inf)``.  Telemetry exposes
    both ``raw_adaptive`` and ``active`` simultaneously."""

    q_normalizer_ema_tau: float = 0.005
    """EMA mixing rate for ``q_normalizer_mode='slow_ema'``.

    ``ema ← (1 − τ)·ema + τ·raw_adaptive``.  Smaller = slower."""

    q_normalizer_min: float = 1.0
    """Lower bound on the active q_normalizer divisor.

    Hard floor applied AFTER ema/freeze logic.  Default 1.0
    matches legacy adaptive behaviour."""

    q_normalizer_freeze_step: int = 0
    """Global step at which to freeze the active q_normalizer.

    Only consulted when ``q_normalizer_mode='freeze_at_step'``."""

    # ── SAC α lower bound ──────────────────────────────────────────
    alpha_min: float | None = None
    """Hard lower bound for the SAC entropy temperature α.

    When set, ``log_alpha`` is clamped to ``[log(alpha_min), log(10)]``
    after each autotune step.  ``None`` uses the default floor of 1e-8."""

    # ── TD-relative CQL α ──────────────────────────────────────────
    cql_td_ratio: float | None = None
    """When set, replaces the Lagrangian / fixed α_cql with a TD-relative
    schedule:  effective_α = max(cql_td_ratio · td_loss / |penalty|,
    cql_alpha_floor).  ``None`` disables (uses Lagrangian or fixed α_cql)."""

    cql_alpha_floor: float = 0.0
    """Minimum effective CQL α when ``cql_td_ratio`` is active.

    Prevents the CQL penalty weight from vanishing when td_loss → 0."""

    cql_alpha_mode: str = "td_relative"
    """CQL effective-α scheduling mode.

    Supported values:
      - ``"td_relative"`` (default): effective_α = max(cql_td_ratio · td_loss / |penalty|, floor).
        Requires ``cql_td_ratio`` to be set.
      - ``"fixed_effective"``: effective_α = ``cql_fixed_effective_alpha`` (constant).
        Ignores cql_td_ratio / cql_alpha_floor entirely.
    When ``cql_td_ratio`` is None the Lagrangian / fixed α_cql path is used
    regardless of this field (backward-compatible)."""

    cql_fixed_effective_alpha: float = 0.015
    """Constant effective CQL α used when ``cql_alpha_mode='fixed_effective'``.

    Only read when ``cql_alpha_mode='fixed_effective'``.  Ignored otherwise."""

    cql_effective_alpha_cap: float = 0.0
    """Phase P1 (effective-α blow-up confounder isolation) — upper cap on
    the CQL effective α used in the loss multiplier.

    When ``> 0`` and ``cql_alpha_mode == 'td_relative'``, the loss weight
    becomes ``min(max(raw, floor), cap)`` instead of ``max(raw, floor)``.
    This blocks the ``effective_α = cql_td_ratio · td_loss / |penalty|``
    blow-up that occurs when ``|cql_penalty| → 0`` (or sign-flips) without
    altering anything else in the critic objective.

    ``= 0`` (default) is a strict no-op and keeps all prior runs bit-exact.
    Recommended pilot when active: ``= 1.0`` (the upper end of the
    healthy effective_α regime observed in F1/G1 mid-run telemetry).
    Honoured under ``cql_alpha_mode == 'fixed_effective'`` as well, where
    it caps the constant value (rare, primarily for safety).  Ignored
    when the Lagrangian path is in use (``cql_td_ratio is None`` and
    ``cql_alpha_autotune == True``).  No sweep is gated by this field;
    sweeps are managed externally."""

    cql_penalty_floor_optin: bool = False
    """Phase P1b (penalty sign-flip noise isolation) — one-sided floor on
    the CQL penalty used in the loss term ONLY.

    When ``True``::

        penalty_for_loss = clamp_min(cql_penalty, 0.0)
        cql_loss         = effective_alpha * penalty_for_loss

    The raw ``cql_penalty`` (= logsumexp − Q_data) is preserved unchanged
    in all telemetry and in the autotune Lagrangian update path.  Only
    the scalar fed into ``critic_loss`` is replaced with its non-negative
    half.  This eliminates the negative-loss / Q-pi-pushing-up regime
    that becomes active in late training when |penalty| → 0 and the
    sign oscillates around zero (B2 / B2+P1 sign-flip events at steps
    4700–5000).  Mathematically, this is equivalent to a one-sided ReLU
    on the conservatism gradient: when the policy is already more
    pessimistic than the random/policy-mixture logsumexp, no extra
    pessimism is injected.

    Default ``False`` is a strict no-op (bit-exact regression for all
    prior runs).  Intended to be combined with ``cql_effective_alpha_cap``
    (P1) — the cap suppresses the multiplier blow-up while this floor
    suppresses the residual sign-flip noise.  Honoured under both
    ``td_relative`` and ``fixed_effective`` modes; ignored under the
    Lagrangian path."""

    cql_loss_scale: float = 1.0
    """Stage R1 (P3 SMQR-SG redesign) — mode-agnostic multiplicative
    scale on the conservative CQL loss term that enters
    ``critic_loss``.

    Mathematically::

        critic_loss = td_loss + cql_loss_scale * cql_loss + v1_shrink_loss

    where ``cql_loss`` is the (already α_cql · penalty_for_loss)
    scalar produced by the active α-CQL dispatch branch
    (``td_relative`` / ``fixed_effective`` / Lagrangian).  Applies
    uniformly to all three branches and to all ``smqr_lse_mode``
    variants — independent of the (failed) ``cql_alpha_init`` /
    ``cql_alpha_floor`` Track B knobs that are ignored under
    ``td_relative``.

    Telemetry effect: ``cql_loss`` and ``cql_effective_alpha`` are
    logged BEFORE the scale (i.e. unchanged), and a separate scalar
    ``cql_loss_scale`` is emitted so the realised contribution to the
    critic loss is ``cql_loss_scale × cql_loss``.

    Default ``= 1.0`` is a strict no-op (bit-exact regression for all
    prior runs).  Recommended Stage R1 sweep values: ``0.5`` (half
    pressure), ``0.25`` (quarter pressure).  Setting this to ``0.0``
    disables the conservative term entirely (effectively pure SAC+BC)
    — use only for diagnostic ablations.

    Compatible with ``cql_effective_alpha_cap``: the cap is applied
    first to bound ``effective_α``; the scale is applied last to
    bound ``cql_loss``.  Both are independent confounder controls."""

    # ── IQL-style actor (diagnostic hybrid) ─────────────────────────
    actor_update_mode: str = "sac_bc"
    """Actor update strategy.

    * ``'sac_bc'`` — SAC-style policy gradient + optional BC regulariser
      (default, identical to prior A1 runs).
    * ``'iql_actor'`` — advantage-weighted BC.  A separate state-value
      network V(s) is trained via expectile regression against Q(s, a_data),
      and the actor maximises  E[ w · log π(a_data | s) ] where
      w = clamp(exp((Q - V) / β), max=iql_max_weight).
      SAC entropy/temperature updates are disabled in this mode.
    """

    iql_expectile: float = 0.7
    """Expectile τ for V(s) regression.  τ > 0.5 biases V toward higher
    quantiles of Q(s, a_data), making advantages more conservative."""

    iql_beta: float = 3.0
    """Inverse temperature for advantage weighting.  Smaller β → more
    selective (only high-advantage actions get large weight)."""

    iql_max_weight: float = 20.0
    """Hard upper clamp on exp(advantage / β) to prevent weight explosion."""

    actor_iql_impl_mode: str = "logprob_bc"
    """Implementation backend for the IQL-style actor loss (D1-sanity).

    Only effective when ``actor_update_mode='iql_actor'``.

    * ``'logprob_bc'`` — weighted log π(a_data|s) BC (original D1).
    * ``'mean_bc'`` — weighted MSE between deterministic mean action and
      dataset action.  Tests whether stochastic formulation is the issue.
    * ``'logprob_bc_fixed_std'`` — same as logprob_bc but with actor
      log-std detached from gradient (frozen at current value each step).
      Tests whether std drift is the root cause.
    * ``'pure_bc_mean'`` — unweighted MSE(mean_action, a_data).  No
      advantage weighting.  Ultra-sanity check for action-scaling and
      actor architecture correctness.
    """

    # ── SC-CQL (Selective Conservatism) ─────────────────────────────
    critic_penalty_mode: str = "vanilla_cql"
    """CQL critic penalty mode.

    * ``'vanilla_cql'`` — standard CQL penalty, uniform weight across batch
      (default, identical to A1 series).
    * ``'sc_cql'`` — Selective Conservatism CQL: per-state soft mask
      reweights the CQL penalty to concentrate on policy-side near-OOD
      states where Q(s, π(s)) is dangerously close to Q(s, a_data).
    * ``'smqr_cont_self'`` — Continuous-action SMQR baseline (A-fidelity).
      Per-critic self-mask
      ``g_i(s,a) = σ((Q_i(s,a) − τ(s)) / β)`` (no detach) combined with a
      *shared* state-dependent scalar threshold τ(s) head.  Penalty is
      ``log(1/K · Σ_k exp(Q_i(s,a_k)·g_i(s,a_k) − log p(a_k|s))) − Q_i(s,a_data)``.
      Designed to (i) preserve the original SMQR exp(Q·g(Q,τ)) gradient
      structure under continuous action sampling and (ii) expose the
      Qg′ amplification hypothesis to the critic gradient.  Legacy SC
      heuristics (``sc_mask_*``, severity, phase gating) are NOT applied
      in this mode.
    """

    sc_mask_target: str = "policy_curr_only"
    """Which policy terms the SC mask targets.

    * ``'policy_curr_only'`` — mask based on Q(s,π(s)) vs Q(s,a_data) gap.
    * ``'policy_curr_and_next'`` — also includes Q(s,π(s')) gap.
    * ``'policy_next_only'`` — mask based on Q(s,π(s')) gap only.
      Combine with ``sc_phase_mode='post_lift_only'`` to target
      carry-phase overestimation exclusively.
    """

    sc_mask_strength: str = "mid"
    """SC mask strength preset.  Maps to (temperature, boost) pairs.

    * ``'weak'``   — κ=1.0, λ=0.5  (gentle selective boost)
    * ``'mid'``    — κ=0.5, λ=1.0  (moderate boost, default)
    * ``'strong'`` — κ=0.3, λ=2.0  (aggressive selective boost)
    * ``'custom'`` — use ``sc_mask_temperature`` and ``sc_mask_boost``.
    """

    sc_mask_temperature: float = 0.5
    """Temperature κ for SC sigmoid mask.  Smaller = sharper transition.
    Only used when ``sc_mask_strength='custom'``."""

    sc_mask_boost: float = 1.0
    """Boost factor λ for SC penalty reweight: w(s) = 1 + λ · mask(s).
    Only used when ``sc_mask_strength='custom'``."""

    sc_mask_threshold: float = 0.0
    """Gap threshold τ for SC mask: mask(s) = σ((τ − gap(s)) / κ).
    τ = 0 activates at the violation boundary (Q_π ≥ Q_data).
    τ > 0 is more aggressive (activates even when Q_π < Q_data by τ)."""

    # ── SC-CQL v2 (violation-only sparse mask) ─────────────────────
    sc_mask_mode: str = "sigmoid_symmetric"
    """SC mask shape.

    * ``'sigmoid_symmetric'`` — original v1 sigmoid mask (default).
    * ``'violation_only_sparse'`` — one-sided exponential mask that is
      exactly 0 for safe states (gap ≥ margin_target), active only on
      the deficit subset, and further sparsified to the worst
      ``sc_active_frac_target`` fraction of the batch.
    """

    sc_margin_target: float = 0.0
    """Target Q-margin for SC v2.  States with gap ≥ margin_target are
    considered safe and get mask = 0.  Set > 0 to enforce a positive
    safety margin (e.g. 0.5 means Q_data must exceed Q_π by 0.5)."""

    sc_sparse_temperature: float = 0.1
    """Temperature for the one-sided exponential mask in v2:
    base_mask = 1 − exp(−deficit / temperature).  Smaller = sharper."""

    sc_active_frac_target: float = 0.10
    """Target fraction of the batch that should have nonzero mask in v2.
    A dynamic threshold is computed as the (1 − frac) quantile of the
    deficit distribution; states below this threshold are zeroed out."""

    # ── SC-CQL v3 (severity-aware) ─────────────────────────────────
    sc_severity_mode: str = "none"
    """Severity scaling within the active subset (v3).

    * ``'none'`` — uniform weight inside active subset (v2 behavior).
    * ``'deficit_weighted'`` — scale mask by normalised deficit:
      severity(s) = (deficit / max(deficit))^power.  States with
      larger deficit get proportionally stronger penalty boost.
    """

    sc_severity_power: float = 1.0
    """Exponent for deficit-weighted severity.  1.0 = linear,
    >1 = superlinear (concentrates more on the worst states).
    Only used when ``sc_severity_mode='deficit_weighted'``."""

    # ── SC-CQL v4 (phase-aware, severity improvements) ─────────────
    sc_phase_mode: str = "all"
    """Phase gating for the SC mask (v4).

    * ``'all'`` — apply mask to all transitions (v2/v3 behavior).
    * ``'post_lift_only'`` — only apply *next-state* mask to
      transitions where the object height (obj_pos_b z-component)
      exceeds ``sc_phase_height_threshold``.  This targets the
      carry phase while leaving grasp/lift unaffected.
    """

    sc_phase_height_threshold: float = 0.15
    """Object z-height (body frame) above which a transition is
    considered post-lift.  Only used when
    ``sc_phase_mode='post_lift_only'``.  Typical table-height values
    for the G1 box task are 0.10–0.25 m."""

    sc_severity_floor: float = 0.0
    """Minimum severity value.  Prevents severity from collapsing to
    near-zero when deficits are small relative to the batch max.
    Set to e.g. 0.1 to ensure all active states receive at least 10%
    of the peak severity.  0.0 = no floor (v3 behavior)."""

    sc_severity_norm_mode: str = "batch_max"
    """Normalisation denominator for deficit-weighted severity.

    * ``'batch_max'`` — divide by max deficit in the batch (v3 behavior).
    * ``'p90'`` — divide by 90th-percentile deficit.  More robust to
      outliers; states above p90 are clipped to severity = 1.0.
    """

    sc_phase_debug: bool = False
    """Enable one-shot diagnostic print of obj_pos_b z-distribution
    and phase-gating activation on the first SC-CQL update step.
    Also logs the effective SC config to console at setup time."""

    # ── SMQR continuous-action baseline (smqr_cont_self) ───────────
    sc_tau_beta: float = 1.0
    """Soft-mask temperature β for the SMQR self-mask
    ``g_i(s,a) = σ((Q_i(s,a) − τ(s)) / β)``.
    Smaller β → sharper mask transition near τ(s).  Only used when
    ``critic_penalty_mode='smqr_cont_self'``.  Recommended starting
    point is 1.0; tune so that ``train/smqr/g/g_mean`` does NOT
    saturate to 0 or 1 in early training."""

    sc_tau_eps: float = 1e-6
    """Numerical floor for divisions inside the SMQR penalty (β,
    log-density, etc.).  Only used when
    ``critic_penalty_mode='smqr_cont_self'``."""

    sc_tau_near_abs_eps: float = 0.05
    """Absolute |Δ| threshold for ``train/smqr/.../near_frac_abs``
    near-τ occupancy diagnostics, where Δ = Q_i(s,a) − τ(s).  Only
    affects logging."""

    sc_tau_near_beta_coeff: float = 1.0
    """Coefficient c_β such that
    ``train/smqr/.../near_frac_beta = mean(|Δ| ≤ c_β · β)``.  Only
    affects logging."""

    sc_tau_res_scale: float = 2.0
    """Scale of the bounded τ residual in the SMQR parameterization
    ``τ(s) = Q_data_min(s).detach() + sc_tau_res_scale · tanh(τ_raw(s))``.
    Caps how far the learned τ can move from the per-state anchor,
    preventing the ``τ → ±∞ / g → {0,1}`` collapse observed with a
    free-scalar residual.  Default 2.0 = 2·β (two mask-transition
    widths on either side of the anchor) — enough headroom to learn
    a non-trivial threshold while keeping the residual bounded.
    Only used when ``critic_penalty_mode='smqr_cont_self'``."""

    sc_tau_log_hist: bool = False
    """If True, additionally log per-tensor histograms (τ, Δ).  Off by
    default to keep TensorBoard storage small."""

    obs_normalization: bool = True
    """Whether to normalise observations using dataset statistics."""

    # ── Mixed precision ────────────────────────────────────────────
    amp: bool = True
    """Enable automatic mixed-precision training."""

    amp_dtype: str = "bf16"
    """AMP dtype: 'bf16' or 'fp16'."""

    # ── Gradient clipping ──────────────────────────────────────────
    max_grad_norm: float = 1.0
    """Maximum gradient norm (0 = disabled)."""

    # ── Tier B optional fields ─────────────────────────────────────
    use_tanh: bool = True
    """Whether to squash actions through tanh."""

    use_layer_norm: bool = True
    """Whether to use layer normalisation in actor/critic MLPs."""

    log_std_max: float = 2.0
    """Upper clamp for the actor's log-std."""

    log_std_min: float = -5.0
    """Lower clamp for the actor's log-std."""

    num_q_networks: int = 2
    """Number of Q-networks in the TwinQCritic ensemble."""

    weight_decay: float = 0.0
    """Weight decay for AdamW optimisers."""

    q_clip: float = 1e4
    """Absolute clamp for TD-target Q-values."""

    compile: bool = False
    """Whether to torch.compile the normaliser forward passes."""

    eval_interval: int = 0
    """Run eval rollouts every N training steps (0 = disabled)."""

    eval_steps: int = 200
    """Number of env steps per eval rollout."""

    eval_callbacks: Any = None
    """Optional evaluation callbacks configuration."""

    # ── Phase A: unified algorithm mode scaffold ───────────────────
    # See holosoma.agents.offline_cql.algo_mode for the resolver.
    # All three keys are *additive*; default values keep every legacy
    # call site bit-equivalent ("auto" defers to critic_penalty_mode +
    # sc_tau_res_scale).  Phase A blocks training in 'smqr_learned'.
    algo_mode: str = "auto"
    """Unified algorithm mode router.

    * ``'auto'`` (default) — infer from legacy keys
      (``critic_penalty_mode`` + ``sc_tau_res_scale``).  Backward
      compatible with every pre-Phase-A run.
    * ``'cql'`` — vanilla CQL (requires ``critic_penalty_mode='vanilla_cql'``).
    * ``'smqr_anchor'`` — anchor-only SMQR (requires
      ``critic_penalty_mode='smqr_cont_self'`` AND
      ``sc_tau_res_scale=0.0``).
    * ``'smqr_learned'`` — learnable τ-residual SMQR.  *Phase A guard
      raises NotImplementedError* — Phase B branch will lift this gate.

    Resolved by :func:`holosoma.agents.offline_cql.algo_mode.resolve_algo_mode`.
    """

    smqr_learned_variant: str = "vanilla"
    """Variant selector for the learned-τ branch (Phase A: placeholder).

    * ``'vanilla'`` — the existing
      ``τ(s) = anchor + scale·tanh(τ_raw(s))`` parameterisation.
    * ``'stabilized'`` — placeholder for a future stabilised gradient
      formulation (e.g. log-density-shifted LSE).  Not implemented in
      Phase A; selecting it raises NotImplementedError even when the
      Phase B gate is open.

    Only consulted when the resolved mode is ``smqr_learned``.
    """

    smqr_logging_namespace: Any = None
    """Optional override for the TensorBoard / metric mode-prefix.

    When ``None`` (default) the prefix is derived from the resolved
    mode name as ``train/<mode>/``.  When set to a string it is used
    verbatim (with a single trailing ``/`` enforced).  Intended for
    A/B sweeps that need to log multiple modes into a single run dir
    without key collisions.
    """

    smqr_learned_phase_b_optin: bool = False
    """Phase B opt-in gate for ``algo_mode='smqr_learned'``.

    When ``False`` (default) the Phase A guard fires and
    ``smqr_learned`` cannot be trained — protecting the anchor-only
    hypothesis track from accidental learned-τ activation.

    When ``True`` *and* ``algo_mode='smqr_learned'`` *and*
    ``smqr_learned_variant='vanilla'``, the existing
    ``critic_penalty_mode='smqr_cont_self'`` code path runs unchanged
    with ``sc_tau_res_scale > 0``.  No new numerical branches are
    added in Phase B — this flag only opens the gate.

    The ``stabilized`` variant additionally requires
    ``smqr_learned_phase_c_optin=True``.
    """

    smqr_learned_phase_c_optin: bool = False
    """Phase C opt-in gate for ``smqr_learned_variant='stabilized'``.

    When ``True`` *and* ``smqr_learned_phase_b_optin=True`` *and*
    ``algo_mode='smqr_learned'`` *and*
    ``smqr_learned_variant='stabilized'``, the critic-side weighted
    logits switch from the vanilla form

        logits_k  =  Q(s,a_k) · g(s,a_k) − log p(a_k)

    to the stabilized form

        logits_k  =  Q(s,a_k) + log(g(s,a_k) + ε) − log p(a_k)

    where ``ε = smqr_stab_g_eps``.  This bounds the d/dQ contribution
    by a softmax-weighted ``(1 - g)/β`` term instead of the vanilla
    ``g(1-g)/β`` term that is amplified by Q itself, addressing the
    tanh-saturation collapse observed in the Phase B vanilla pilot.

    All other paths (cql, smqr_anchor, smqr_learned-vanilla) are
    untouched when this flag is set.
    """

    smqr_stab_g_eps: float = 1e-6
    """Floor ε added inside ``log(g + ε)`` for the stabilized
    learned-τ variant.

    Only consulted when ``smqr_learned_variant='stabilized'`` AND the
    Phase C opt-in is granted.  Smaller ε keeps the stabilised
    gradient closer to the vanilla limit; larger ε is more
    conservative (broader effective support).  Recommended pilot
    value: ``1e-6``.  Conservative fallback: ``1e-4``.
    """

    smqr_learned_phase_d_optin: bool = False
    """Phase D opt-in gate for ``smqr_learned_variant='v1_oneside_shrink'``.

    When ``True`` *and* ``smqr_learned_phase_b_optin=True`` *and*
    ``algo_mode='smqr_learned'`` *and*
    ``smqr_learned_variant='v1_oneside_shrink'``, the τ
    parameterisation switches to the one-sided form

        τ(s) = Q_data_min(s).detach() − sc_tau_res_scale · softplus(τ_raw(s))

    so that τ can never exceed the per-state anchor.  The critic
    objective remains the stabilised form ``Q + log(g + ε)``;
    only the τ parameterisation and an additive shrinkage term
    differ from the Phase C variant.

    All other paths (cql, smqr_anchor, smqr_learned-vanilla,
    smqr_learned-stabilized) are untouched when this flag is set.
    """

    smqr_v1_shrink_lambda: float = 1e-3
    """Anchor-shrinkage coefficient λ_sh for the V1 / F1 variants.

    Consulted when ``smqr_learned_variant`` is either
    ``'v1_oneside_shrink'`` (Phase D) or ``'f1_st_qg'`` (Phase F),
    AND the corresponding opt-in is granted.  Adds the term

        L_shrink = λ_sh · E_s [ (τ_anchor(s) − τ(s))² ]

    to the critic loss.  Gradient flows only into the τ-head (the
    anchor is detached).  Default ``1e-3``.  Conservative
    fallback: ``1e-2``.  Disable: ``0.0``.
    """

    smqr_learned_phase_f_optin: bool = False
    """Phase F opt-in gate for ``smqr_learned_variant='f1_st_qg'``.

    When ``True`` *and* ``smqr_learned_phase_b_optin=True`` *and*
    ``algo_mode='smqr_learned'`` *and*
    ``smqr_learned_variant='f1_st_qg'``, the τ parameterisation
    reuses the V1 form

        τ(s) = Q_data_min(s).detach() − sc_tau_res_scale · softplus(τ_raw(s))

    AND the SMQR weighted-logits switch from the stabilised form

        logits_k = Q + log(g + ε) − log p

    to the ST-split form

        logits_k = 0.5 · ( Q · sg(g) + sg(Q) · g ) − log p

    whose forward value equals vanilla ``Q · g`` bit-exactly while
    the symmetric stop-gradient identity halves the Q·g'
    amplification on both θ_Q and θ_τ.  No ``log(g+ε)`` floor is
    used — the ε / log_g_min / g_lt_eps_frac failure modes observed
    in Phase C/D/E are eliminated by construction.

    Reuses ``smqr_v1_shrink_lambda`` for the shrinkage coefficient.
    Reuses ``sc_tau_res_scale`` for the τ-residual scale.  No new
    Phase F-specific hyperparameters.
    """

    smqr_f1_random_full_grad: bool = False
    """Phase G1 (`f1_random_full_grad`) — candidate-wise objective routing
    on top of the F1 base.

    Sub-flag of :attr:`smqr_learned_phase_f_optin`.  Has effect ONLY when
    ``algo_mode='smqr_learned'``  AND
    ``smqr_learned_variant='f1_st_qg'``  AND
    ``smqr_learned_phase_f_optin=True``  AND
    this flag is ``True``.  In all other configurations the flag is a
    silent no-op (no behavioural change, no telemetry emitted).

    When active, the SMQR weighted-logits along the K-axis switch from
    a uniform F1 ST-split to a candidate-wise mixed form::

        K-axis layout:  Q_cat_raw = cat([q_rand, q_pi], dim=-1)
            indices [0, num_random)         — uniform random candidates
            indices [num_random, N_total)   — current-policy candidates

        random  channel:  qg_k = Q · g                              (vanilla full-grad)
        policy  channel:  qg_k = 0.5 · ( Q · sg(g) + sg(Q) · g )    (F1 ST-split, unchanged)

    Forward value is bit-exactly equal to vanilla ``Q · g`` on every K
    index (since ``0.5·(Q·g + Q·g) ≡ Q·g``); only the backward routing
    differs.  The data term and the τ-parameterisation are untouched.

    Rationale (from the F1 5k short-run, exp_17_smqrlrnf1_short5k_seed1):
    F1's symmetric ½-attenuation of the Q·g' backward restored
    near_pi_frac_beta ≈ 0.57 (τ-band recovery) but halved the CQL
    push-down on random candidates, leaving cql_q_rand_mean ≈ +9.6 and
    cql_penalty ≈ −3.7.  Random candidates are τ-band irrelevant
    (they do not need the ST-split's τ-grad protection), so restoring
    vanilla full-grad on the random channel only is the minimum-change
    intervention to recover conservatism without disturbing F1's
    policy-channel τ-band recovery.

    No new opt-in is introduced: this flag is gated entirely by the
    Phase F opt-in.  Setting it to ``True`` without the F1 base active
    is silently ignored (and logged as inactive).
    """

    smqr_h1_alpha_floor: float = 0.0
    """Phase H1 (`f1_random_alpha_floor`) — additive constant floor on
    the random-branch effective gate.

    Sub-flag of :attr:`smqr_f1_random_full_grad` (G1 routing).  Has
    effect ONLY when ``algo_mode='smqr_learned'``  AND
    ``smqr_learned_variant='f1_st_qg'``  AND
    ``smqr_learned_phase_f_optin=True``  AND
    ``smqr_f1_random_full_grad=True``  AND
    ``smqr_h1_alpha_floor > 0.0``.  In all other configurations the
    field is a silent no-op (no behavioural change, no telemetry
    emitted).

    When active, the random-branch SMQR weighted-logit changes from

        qg_k_rand = Q_k · g_k                                  (G1)

    to

        qg_k_rand = Q_k · ( g_k + α )                          (H1)

    where ``α = smqr_h1_alpha_floor``.  The policy branch retains the
    F1 ST-split (G1 routing unchanged) and the data term is untouched.

    Rationale (from the G1 5k short-run): the random branch suffers
    late-stage gradient starvation when ``g_rand → 0`` because
    ``∂qg/∂Q = g + Q·g'/β → 0`` with it.  The α floor introduces a
    state-independent push-down lower bound: ``∂qg_rand/∂Q ≥ α > 0``
    even when the gate fully closes.  ``α = 0`` recovers G1 bit-exactly,
    so the field is opt-in via a strictly positive value.

    Recommended pilot: ``α = 0.05``.  Conservative fallback: ``0.1``.
    Disable: ``0.0`` (default, identical to G1).  No sweep is gated by
    this field; sweeps are managed externally.
    """

    smqr_b2_alpha_floor: float = 0.0
    """Phase B2 (`f1_random_st_max_clip_backward_floor`) — STE-based
    backward-only floor on the random-branch Q-gradient.

    Sub-flag of :attr:`smqr_f1_random_full_grad` (G1 routing).  Has
    effect ONLY when ``algo_mode='smqr_learned'``  AND
    ``smqr_learned_variant='f1_st_qg'``  AND
    ``smqr_learned_phase_f_optin=True``  AND
    ``smqr_f1_random_full_grad=True``  AND
    ``smqr_b2_alpha_floor > 0.0``.  Mutually exclusive with H1
    (``smqr_h1_alpha_floor > 0.0``); enabling both raises a
    ``RuntimeError`` at agent setup.  In all other configurations
    the field is a silent no-op.

    When active, the random-branch SMQR weighted-logit becomes a
    straight-through estimator (STE) with a backward-only floor:

        forward(qg_k_rand)        = Q_k · g_k                       (= G1, bit-exact)
        ∂qg_k_rand / ∂Q_k         = max(g_k, α)                     (Q-grad floor)
        ∂qg_k_rand / ∂g_k         = Q_k                              (= G1, τ-grad unchanged)

    where ``α = smqr_b2_alpha_floor``.  The policy branch retains the
    F1 ST-split (G1 routing unchanged) and the data term is untouched.

    Rationale (Phase I memo): under G1 the random branch suffers
    late-stage Q-gradient starvation when ``g_rand → 0`` because
    ``∂qg/∂Q ∝ g``.  H1 lifts the lower bound by changing the FORWARD
    to ``Q·(g+α)``, which over-shifts the forward logsumexp mass
    toward random and degrades the F1 policy τ-band recovery.  B2
    keeps the forward bit-exact to G1 and lifts ONLY the backward
    Q-gradient floor via a clamp on the detached gate.  Healthy
    random candidates (``g_rand ≥ α``) see G1-identical Q-grad;
    starved candidates (``g_rand < α``) get a floored ``α`` push-down
    signal.  ``α = 0`` reproduces G1 bit-exactly (forward, Q-grad,
    τ-grad).

    Recommended pilot: ``α = 0.05``.  Conservative fallback: ``0.1``.
    Disable: ``0.0`` (default, identical to G1).  No sweep is gated by
    this field; sweeps are managed externally.
    """

    smqr_anchor_objective: str = "vanilla"
    """Phase E (objective-isolation ablation) — SMQR objective selector
    for the anchor-only branch.

    Only consulted when ``algo_mode='smqr_anchor'`` (i.e.
    ``critic_penalty_mode='smqr_cont_self'`` AND
    ``sc_tau_res_scale=0.0``).

    * ``'vanilla'`` (default) — the existing weighted-logits form

          logits_k = Q(s,a_k) · g(s,a_k) − log p(a_k)

      Bit-equivalent to all pre-existing anchor-only runs.

    * ``'stabilized'`` — the Phase C stabilised form

          logits_k = Q(s,a_k) + log(g(s,a_k) + ε) − log p(a_k)

      with ``ε = smqr_stab_g_eps``, but **with τ ≡ τ_anchor** (no
      learned residual / head — ``sc_tau_res_scale`` MUST be 0.0).
      This isolates whether the stabilised objective itself is the
      1st-order cause of the Phase C/D learned-τ failures,
      independently of τ-parameterisation.  Requires
      ``smqr_anchor_phase_e_optin=True``.

    Selecting ``'stabilized'`` for any non-anchor mode raises.
    """

    smqr_anchor_phase_e_optin: bool = False
    """Phase E opt-in gate for ``smqr_anchor_objective='stabilized'``.

    When ``False`` (default) the Phase E guard fires and the
    stabilised anchor-only objective cannot be trained — protecting
    the anchor-only baseline path from accidental objective swaps.

    When ``True`` *and* ``algo_mode='smqr_anchor'`` *and*
    ``smqr_anchor_objective='stabilized'``, the SMQR weighted-logits
    branch in :class:`OfflineCQLAgent._update_critic` switches to the
    stabilised form.  All other paths (cql, smqr_anchor-vanilla,
    smqr_learned-*) are untouched when this flag is set.

    Reuses ``smqr_stab_g_eps`` for the floor ε — Phase E does not
    introduce a separate ε field, since the objective form is
    structurally identical to Phase C.
    """

    smqr_lse_mode: str = "q_times_g"
    """Step-3 SMQR-SG sub-mode selector for the **anchor-only vanilla**
    weighted-logits branch.

    Hypothesis #2 (Q·g distortion): the existing
    ``logits = Q · g − log p`` form couples the conservative gate ``g``
    multiplicatively to ``Q``, so for actions with ``Q ≈ τ`` the
    softmax/Q-grad pressure is dominated by ``Q · g'(Q-τ)/β`` rather
    than the gate ``g`` itself, distorting both the forward ranking
    and the backward conservative pressure.  SMQR-SG re-routes the
    gate to act as a (detached) action-wise weight on the logsumexp
    instead of a multiplicative critic scaler.

    Allowed values:

    * ``'q_times_g'`` (default) — the existing baseline, BIT-EXACT to
      pre-Step-3 anchor-vanilla runs:

          logits_k = Q(s,a_k) · g(s,a_k) − log p(a_k)

    * ``'q_times_detached_g'`` — backward-only ablation:

          logits_k = Q(s,a_k) · stop_grad(g(s,a_k)) − log p(a_k)

      Forward is identical to ``q_times_g`` (so forward ranking
      distortion is preserved); only the gate-derivative path
      ``Q · g'/β`` into θ_Q is removed.

    * ``'sg_weighted_lse'`` — SMQR-SG main:

          logits_k = Q(s,a_k) − log p(a_k) + log(stop_grad(g(s,a_k)) + ε)

      Removes ``Q · g`` multiplication entirely.  The gate enters as
      a detached additive constant ``log(g+ε)`` so the softmax
      effectively becomes a *gate-weighted* softmax over
      ``Q − log p``.  Q-ranking is preserved, and
      ``∂lse/∂Q_i = softmax_i(weighted_logits)`` (no ``Q·g'`` term).

    * ``'sg_blend'`` — Stage R1 (P3 redesign) 50/50 LOSS-level blend:

          per_state_penalty =
              0.5 · per_state_penalty[q_times_g]
            + 0.5 · per_state_penalty[sg_weighted_lse]

      Both logsumexps are computed independently (with the same
      ``q_clip`` and ``smqr_sg_eps``) and the resulting per-state
      penalties are averaged BEFORE the α_cql multiplier.
      Mathematically equivalent to::

          L_conservative = 0.5 · L_q_times_g + 0.5 · L_sg_weighted_lse

      Intent: keep the strong critic→actor signal of ``q_times_g``
      (which P2 SMQR-anchor benefited from) while injecting half of
      the gate-weighted-softmax (``sg_weighted_lse``) ranking
      structure that P3 was attempting to isolate.  Telemetry uses
      the ``sg_weighted_lse`` side for the SMQR mechanism keys
      (near_τ frac, gradient amplification, etc.).

    Only consulted when **all** of the following hold:

      * ``algo_mode='smqr_anchor'`` (``critic_penalty_mode='smqr_cont_self'``
        AND ``sc_tau_res_scale=0.0``)
      * ``smqr_anchor_objective='vanilla'`` (i.e. NOT the Phase E
        ``stabilized`` branch — Phase E uses its own ``Q + log(g+ε)``
        path with ``g`` un-detached)
      * F1/G1/H1/B2 sub-flags inactive (those modify the same logits
        block via different mechanisms)

    Selecting any non-``'q_times_g'`` value while any of the above
    contamination conditions are violated raises in
    :meth:`OfflineCQLAgent.setup`.

    ``'q_times_g'`` (default) preserves bit-exact behaviour for
    every pre-Step-3 anchor-vanilla run.
    """

    smqr_sg_eps: float = 1e-6
    """Floor ε used inside ``log(stop_grad(g) + ε)`` for SMQR-SG
    (``smqr_lse_mode='sg_weighted_lse'``).

    Smaller ε keeps the gate-weighting closer to ``log(g)`` (i.e.
    sub-τ candidates are more aggressively suppressed in the
    softmax); larger ε is more conservative (broader effective
    support).  Independent of ``smqr_stab_g_eps``: Step-3 SMQR-SG
    uses a *detached* ``g``, while the Phase C/D/E stabilised branch
    uses an *attached* ``g`` and therefore has a different gradient
    structure even at identical ε.
    """

    # ─────────── sg_blend λ schedule (Stage S) ──────────────────────
    # Only consulted when ``smqr_lse_mode == 'sg_blend'``.  λ is the
    # mixing weight on the sg_weighted_lse side:
    #   per_state_penalty = (1 - λ) * P_qg + λ * P_sgw
    # λ=0  → pure q_times_g, λ=1 → pure sg_weighted_lse.
    # The previous sg_blend implementation hard-coded λ=0.5; setting
    # ``smqr_blend_schedule='fixed'`` and ``smqr_blend_lambda_start=
    # smqr_blend_lambda_end=0.5`` recovers it bit-exactly.

    smqr_blend_schedule: str = "fixed"
    """λ schedule mode for ``smqr_lse_mode='sg_blend'``.  One of:

    * ``'fixed'`` (default)  — constant λ = ``smqr_blend_lambda_start``.
      ``smqr_blend_lambda_end`` and ramp/warmup/hold fields are
      ignored.  Default values (start=0.5) recover the Stage R2 R4
      behaviour bit-exactly.
    * ``'linear'`` — linear ramp from ``λ_start`` to ``λ_end`` over
      the first ``smqr_blend_ramp_steps`` global-steps, then held at
      ``λ_end`` thereafter.  ``smqr_blend_warmup_steps`` is ignored.
    * ``'delayed_linear'`` — λ held at ``λ_start`` for the first
      ``smqr_blend_warmup_steps`` global-steps (discovery phase),
      then linearly ramped to ``λ_end`` over the next
      ``smqr_blend_ramp_steps`` steps, then held at ``λ_end``.
    * ``'piecewise'`` — three-segment schedule:
      [0, warmup): ``λ_start``;
      [warmup, warmup+ramp): linear ramp to ``λ_end``;
      [warmup+ramp, warmup+ramp+hold): ``λ_end``;
      thereafter: ``λ_end`` (hold field is informational only and
      does not change behaviour after the held window).

    Outside of ``smqr_lse_mode='sg_blend'`` this field is ignored.
    """

    smqr_blend_lambda_start: float = 0.5
    """λ value at training step 0 (and during the warmup phase for
    ``delayed_linear`` / ``piecewise`` schedules).  Default 0.5 to
    match Stage R2 R4 fixed blend exactly when paired with
    ``smqr_blend_schedule='fixed'``.

    Must satisfy 0.0 ≤ λ_start ≤ 1.0.  Validated in
    :meth:`OfflineCQLAgent.setup`."""

    smqr_blend_lambda_end: float = 0.5
    """λ value reached at the end of the ramp (and held afterwards).
    Default 0.5 matches Stage R2 R4 when ``schedule='fixed'`` (in
    which case this field is ignored anyway).

    Must satisfy 0.0 ≤ λ_end ≤ 1.0."""

    smqr_blend_warmup_steps: int = 0
    """Number of global-steps to hold λ at ``λ_start`` BEFORE the ramp
    begins.  Used by ``delayed_linear`` and ``piecewise`` schedules.
    Ignored under ``fixed`` and ``linear``.  Must be ≥ 0."""

    smqr_blend_ramp_steps: int = 1
    """Length (in global-steps) of the linear ramp from ``λ_start`` to
    ``λ_end``.  Used by ``linear``, ``delayed_linear``, and
    ``piecewise``.  Ignored under ``fixed``.  Must be ≥ 1
    (1 ≈ instantaneous step)."""

    smqr_blend_hold_steps: int = 0
    """Used by ``piecewise`` schedule only — informational; the
    schedule continues to hold at ``λ_end`` after this window
    expires.  Ignored otherwise.  Must be ≥ 0."""


@dataclass(frozen=True)
class OfflineCQLAlgoConfig:
    """Algo wrapper for Offline CQL (mirrors PPOAlgoConfig / FastSACAlgoConfig)."""

    _target_: str
    """Target algorithm class."""

    _recursive_: bool
    """Whether to recursively instantiate."""

    config: OfflineCQLConfig
    """Algorithm-specific configuration."""


AlgoInitConfig = Union[PPOConfig, FastSACConfig, OfflineCQLConfig]

AlgoConfig = Union[PPOAlgoConfig, FastSACAlgoConfig, OfflineCQLAlgoConfig]
