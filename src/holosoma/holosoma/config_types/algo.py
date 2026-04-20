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
