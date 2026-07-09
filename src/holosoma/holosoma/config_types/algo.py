from __future__ import annotations

from dataclasses import field
from typing import Any, List, Literal, Union

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

    offline_dataset_path: str = "offline_data/fastsac_dataset.h5"
    """path where FastSAC exports transitions for offline RL"""

    episode_data_active_envs: int = 64
    """number of active environments recorded by FastSACEpisodeDataAgent"""

    episode_data_mc_gamma: float | None = None
    """discount used for exported mc_return; None uses gamma"""

    encoder_obs_key: str = "perception_obs"
    """the key of the encoder observation. only valid if use_cnn_encoder is True"""

    encoder_obs_shape: tuple[int, int, int] = (1, 13, 9)
    """the shape of the encoder observation. only valid if use_cnn_encoder is True"""

    use_cnn_encoder: bool = False
    """whether to use CNN for the encoder"""

    actor_obs_keys: List[str] = field(default_factory=lambda: ["actor_obs"])
    critic_obs_keys: List[str] = field(default_factory=lambda: ["critic_obs"])

    save_env_num : int = 32


@dataclass(frozen=True)
class OfflineSACConfig:
    num_learning_iterations: int = 25000
    """total gradient update iterations"""

    critic_learning_rate: float = 3e-4
    """the learning rate of the critic"""

    actor_learning_rate: float = 3e-4
    """the learning rate for the actor"""

    alpha_learning_rate: float = 3e-4
    """the learning rate for alpha autotune"""

    gamma: float = 0.97
    """the discount factor gamma"""

    tau: float = 0.125
    """target network soft-update coefficient"""

    batch_size: int = 8192
    """global batch size sampled from offline dataset"""

    num_updates: int = 8
    """number of gradient updates per outer step"""

    eval_interval: int = 1000
    """steps per offline_learn() call when max_steps is not provided"""

    policy_frequency: int = 4
    """delayed actor update frequency in critic updates"""

    target_entropy_ratio: float = 0.0
    """target entropy ratio multiplied by action dimension"""

    num_atoms: int = 101
    """number of distributional support atoms"""

    v_min: float = -20.0
    """minimum critic support value"""

    v_max: float = 20.0
    """maximum critic support value"""

    critic_hidden_dim: int = 768
    """hidden dimension of critic network"""

    actor_hidden_dim: int = 512
    """hidden dimension of actor network"""

    use_symmetry: bool = False
    """whether to apply symmetry augmentation to offline batches"""

    alpha_init: float = 0.001
    """initial value of alpha"""

    use_autotune: bool = True
    """whether to learn alpha automatically"""

    use_tanh: bool = True
    """whether to use tanh-squashed actor output"""

    log_std_max: float = 0.0
    """maximum log std for actor"""

    log_std_min: float = -5.0
    """minimum log std for actor"""

    compile: bool = True
    """whether to use torch.compile for update functions"""

    obs_normalization: bool = True
    """whether to normalize actor/critic observations"""

    use_layer_norm: bool = True
    """whether to use layer normalization in networks"""

    num_q_networks: int = 2
    """number of distributional Q networks to ensemble"""

    max_grad_norm: float = 0.0
    """max grad norm (0 disables clipping)"""

    amp: bool = True
    """whether to use AMP"""

    amp_dtype: str = "bf16"
    """AMP dtype: bf16 or fp16"""

    weight_decay: float = 0.001
    """weight decay for optimizers"""

    save_interval: int = 1000
    """checkpoint interval"""

    logging_interval: int = 100
    """logging interval"""

    offline_dataset_path: str = "offline_data/fastsac_dataset.h5"
    """path to fixed offline dataset"""

    offline_block_size: int = 65536
    """number of contiguous transitions to read per HDF5 block refill"""

    offline_buffer_capacity: int = 262144
    """maximum number of transitions held in CPU RAM shuffle buffer"""

    offline_refill_threshold: int = 65536
    """refill shuffle buffer when remaining transitions fall below this threshold"""

    offline_pin_memory: bool = True
    """pin sampled CPU batches before CPU->GPU transfer"""

    offline_shuffle_block_order: bool = True
    """shuffle HDF5 block order while preserving contiguous per-block reads"""

    use_gpu_cache: bool = False
    """whether to load the full offline dataset into GPU memory"""

    encoder_obs_key: str = "perception_obs"
    """encoder observation key, used only when use_cnn_encoder is True"""

    encoder_obs_shape: tuple[int, int, int] = (1, 13, 9)
    """encoder observation shape, used only when use_cnn_encoder is True"""

    use_cnn_encoder: bool = False
    """whether to use CNN actor/critic encoders"""

    actor_obs_keys: List[str] = field(default_factory=lambda: ["actor_obs"])
    critic_obs_keys: List[str] = field(default_factory=lambda: ["critic_obs"])


@dataclass(frozen=True)
class CODACConfig(OfflineSACConfig):
    """Configuration for Stage-1 CODAC built on top of OfflineSAC."""

    conservative_weight: float = 1.0
    """weight applied to conservative regularization term"""

    conservative_temperature: float = 1.0
    """temperature for conservative log-sum-exp aggregation"""

    num_action_samples: int = 10
    """number of random/current/next action samples per state for conservative term"""

    use_lagrange: bool = False
    """whether to auto-tune conservative multiplier with Lagrange optimization"""

    target_action_gap: float = 10.0
    """target conservative gap used when use_lagrange=True"""

    cql_lagrange_init: float = 1.0
    """initial value for conservative Lagrange multiplier"""

    cql_lagrange_learning_rate: float = 3e-4
    """learning rate for conservative Lagrange multiplier optimizer"""

    cql_lagrange_max: float = 1e6
    """maximum clamp value for conservative Lagrange multiplier"""

    actor_q_aggregation: str = "min"
    """actor objective Q aggregation over critic ensemble: 'mean' or 'min'"""

    critic_conservative_mode: str = "mean_q_stage1"
    """conservative mode. Stage-1 implements only 'mean_q_stage1'."""


#===================================== FOR CQL ===============================================
@dataclass(frozen=True)
class CQLConfig:
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

    batch_size: int = 2048
    """the batch size of sample from the replay memory"""

    learning_starts: int = 10
    """timestep to start learning"""

    policy_frequency: int = 4
    """the frequency of training policy (delayed)"""

    num_updates: int = 8
    """the number of updates to perform per step"""

    eval_interval: int = 1000
    """steps per offline_learn() call when max_steps is not provided"""

    cql_num_action_samples: int = 10
    """number of repeated action samples per state for conservative regularization"""

    cql_temperature: float = 1.0
    """temperature used in conservative log-sum-exp aggregation"""

    cql_weight: float = 5.0
    """weight of conservative quantile regularization"""

    normalized_action_training: bool = False
    """whether actor/critic train in normalized [-1, 1] action space; False restores legacy env-scaled action space"""

    cql_near_action_samples: int = 0
    """number of Gaussian-noised dataset-near actions per state for local conservative regularization"""

    cql_near_noise_std: float = 0.05
    """standard deviation of Gaussian noise added to normalized dataset actions for q_near samples"""

    cql_near_weight: float = 0.05
    """cql near loss weight"""

    cql_masked_active_dim: int = 8
    """number of action dimensions perturbed by actor std for Gaussian CQL current/next samples"""

    cql_masked_inactive_std: float = 0.01
    """small Gaussian std used on inactive action dimensions for Gaussian CQL current/next samples"""


    use_lagrange: bool = False
    """whether to use Lagrange multiplier auto-tuning for CQL conservative loss"""

    cql_target_action_gap: float = 10.0
    """target CQL gap threshold used by Lagrange mode (higher -> less conservative)"""

    cql_lagrange_learning_rate: float = 3e-4
    """learning rate for CQL Lagrange multiplier optimizer"""

    cql_lagrange_init: float = 1.0
    """initial value of CQL Lagrange multiplier"""

    cql_lagrange_max: float = 1e6
    """maximum clamp value for CQL Lagrange multiplier"""

    use_curr_tail_penalty: bool = False
    """whether to add an extra top-k tail penalty on (q_curr - curr_logp)"""

    curr_tail_weight: float = 0.0
    """weight for the additional current-proposal top-k tail penalty"""

    curr_tail_top_frac: float = 0.2
    """top fraction used for top-k tail extraction from current proposal samples"""

    bc_weight: float = 0.0
    """optional actor BC regularization weight (actor_loss += bc_weight * MSE(pi(s), a_data))"""

    target_entropy_ratio: float = 0.0
    """the ratio of the target entropy to the number of actions"""

    num_atoms: int = 101
    """number of quantile fractions (kept name for backward compatibility)"""

    v_min: float = -20.0
    """the minimum value of the support"""

    v_max: float = 20.0
    """the maximum value of the support"""

    quantile_huber_kappa: float = 1.0
    """Huber threshold for quantile regression loss"""

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

    offline_dataset_path: str = "offline_data/fastsac_dataset.h5"
    """path to fixed offline dataset"""

    offline_block_size: int = 65536
    """number of contiguous transitions to read per HDF5 block refill"""

    offline_buffer_capacity: int = 262144
    """maximum number of transitions held in CPU RAM shuffle buffer"""

    offline_refill_threshold: int = 65536
    """refill shuffle buffer when remaining unsampled transitions fall below this threshold"""

    offline_pin_memory: bool = True
    """pin sampled CPU batches before CPU->GPU transfer"""

    offline_shuffle_block_order: bool = True
    """shuffle the order of contiguous HDF5 blocks each pass while keeping each block read contiguous"""

    use_gpu_cache: bool = False
    """whether to load the full offline dataset into GPU memory and sample directly on-device"""

    encoder_obs_key: str = "perception_obs"
    """the key of the encoder observation. only valid if use_cnn_encoder is True"""

    encoder_obs_shape: tuple[int, int, int] = (1, 13, 9)
    """the shape of the encoder observation. only valid if use_cnn_encoder is True"""

    use_cnn_encoder: bool = False
    """whether to use CNN for the encoder"""

    actor_warmup_steps: int = 500
    """number of initial global steps where actor update is skipped"""

    q_min: float | None = None
    """minimum Bellman target Q value; None disables lower clipping"""

    q_max: float | None = None
    """maximum Bellman target Q value; None disables upper clipping"""

    bellman_loss_type: Literal["mse", "huber"] = "mse"
    """Bellman regression loss type for critic targets"""

    huber_beta: float = 10.0
    """Smooth L1 beta used when bellman_loss_type='huber'"""

    actor_obs_keys: List[str] = field(default_factory=lambda: ["actor_obs"])
    critic_obs_keys: List[str] = field(default_factory=lambda: ["critic_obs"])

    reward_scale: int = 5

    cql_max_target_backup: bool = False

    backup_entropy : bool = False

    cql_max_target_backup_samples : int = 10


@dataclass(frozen=True)
class PBFCQLSettings:
    """Configuration for pair-synergy BF-CQL regularization."""

    enabled: bool = True
    """whether to add the all-pair PBF-CQL conservative penalty"""

    alpha: float = 0.05
    """weight applied to the PBF-CQL pair-synergy penalty"""

    margin: float = 0.0
    """hinge/softplus margin applied to Delta_ij"""

    use_softplus: bool = False
    """use beta-scaled softplus instead of ReLU hinge"""

    softplus_beta: float = 1.0
    """temperature-like beta for the softplus penalty"""

    detach_singletons: bool = True
    """kept for config visibility; PBF-CQL always detaches singleton values"""

    log_pair_stats: bool = True
    """whether to log aggregate pair residual diagnostics"""


@dataclass(frozen=True)
class SyncCQLSettings:
    """Configuration for SYNC-QL drift-gated CFCQL regularization."""

    K: int = 2
    """legacy option kept for checkpoint/config compatibility; drift-gated CFCQL does not use K"""

    delta_threshold: float = 0.5
    """minimum normalized-RMSE drift required for a group to enter the candidate set"""

    selection_mode: Literal["topk", "greedy", "random", "none"] = "topk"
    """set to 'none' to disable drift gating and recover BF-CQL-style all-group CFCQL"""

    gate_norm: Literal["batch", "active"] = "batch"
    """normalize gated CFCQL by full batch size or active sample count"""

    drift_mode: Literal["rmse", "density"] = "rmse"
    """actor drift estimator used for candidate screening"""

    eps_gain: float = 0.0
    """legacy option kept for config compatibility"""

    margin_m: float = 0.0
    """legacy option kept for config compatibility"""

    alpha2: float = 1.0
    """legacy option kept for config compatibility; the active penalty uses cql_weight"""

    alpha2_lagrange: bool = False
    """legacy option kept for config compatibility; CQL Lagrange remains controlled by use_lagrange"""

    tau_syn: float = 5.0
    """legacy option kept for config compatibility"""

    lambda_cf: float = 0.0
    """weight of the actor counterfactual block objective"""

    drift_ema: float = 0.0
    """EMA blending for per-group drift selection stability; 0.0 uses current-batch drift only"""

    drift_std_momentum: float = 0.999
    """momentum for running dataset action std used by RMSE drift"""

    freeze_drift_stats: bool = False
    """whether to freeze running dataset action std updates"""


@dataclass(frozen=True)
class DCQLSettings:
    """Configuration for escape-ray Directional Conservative Q-Learning."""

    enabled: bool = True
    """whether to replace CFCQL with escape-ray DCQL conservative regularization"""

    t_grid: List[float] = field(default_factory=lambda: [0.5, 1.0, 1.5, 2.0])
    """ray interpolation/extrapolation coefficients from dataset action to actor action"""

    ray_noise_std: float = 0.05
    """Gaussian noise std added to each ray action in normalized action space"""

    delta_thr: float = 0.7
    """minimum sigma-normalized actor-vs-reference drift required to activate DCQL"""

    gate_norm: Literal["batch", "active"] = "batch"
    """normalize gated DCQL by full batch size or active sample count"""

    a_ref_mode: Literal["dataset", "knn"] = "dataset"
    """reference on-support action source; knn is a reserved behavior-model stub"""

    drift_std_momentum: float = 0.999
    """momentum for running dataset action std used by the drift gate"""

    freeze_drift_stats: bool = False
    """whether to freeze running dataset action std updates"""

    warmup_ballast_steps: int = 0
    """number of initial critic updates using optional random-action ballast"""

    ballast_alpha: float = 0.1
    """relative weight for optional warmup random-action ballast"""

    ballast_num_samples: int = 8
    """number of random normalized actions per state for optional warmup ballast"""


@dataclass(frozen=True)
class SynDiagSettings:
    """Configuration for synergy-OOD diagnostic logging (BF-CQL, logging only).

    Never changes losses, gradients, optimizer steps, or RNG consumption of the
    training path; with enabled=False the training path is bit-identical to the
    pre-syndiag behavior.
    """

    enabled: bool = True
    """whether to run the periodic synergy-OOD diagnostics"""

    interval: int = 200
    """run diagnostics every N critic updates"""

    dump_interval: int = 50
    """dump raw npz every N diagnostic ticks (0 = never)"""

    dump_topk: int = 3
    """top coalitions per sample (by Delta) whose counterfactual actions are dumped"""

    delta_min: float = 0.0
    """activity threshold on the per-sample top Delta for recall metrics"""

    max_coalitions: int = 32
    """safety cap on the coalition list; warn and truncate (singletons+pairs kept first)"""

    dump_max_rows: int = 2048
    """subsample cap on rows per raw dump file (keeps files well under ~50MB)"""


@dataclass(frozen=True)
class BFCQLConfig:
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

    batch_size: int = 2048
    """the batch size of sample from the replay memory"""

    learning_starts: int = 10
    """timestep to start learning"""

    policy_frequency: int = 4
    """the frequency of training policy (delayed)"""

    num_updates: int = 8
    """the number of updates to perform per step"""

    eval_interval: int = 1000
    """steps per offline_learn() call when max_steps is not provided"""

    cql_num_action_samples: int = 10
    """number of repeated action samples per state for conservative regularization"""

    cql_temperature: float = 1.0
    """temperature used in conservative log-sum-exp aggregation"""

    cql_weight: float = 5.0
    """weight of conservative quantile regularization"""

    cql_near_action_samples: int = 0
    """number of Gaussian-noised dataset-near actions per state for local conservative regularization"""

    cql_near_noise_std: float = 0.05
    """standard deviation of Gaussian noise added to normalized dataset actions for q_near samples"""

    cql_near_weight: float = 0.05
    """cql near loss weight"""

    cql_masked_active_dim: int = 8
    """number of action dimensions perturbed by actor std for Gaussian CQL current/next samples"""

    cql_masked_inactive_std: float = 0.01
    """small Gaussian std used on inactive action dimensions for Gaussian CQL current/next samples"""

    bf_cql_action_grouping: Literal["functional_9", "coarse_5", "symmetric_14", "nonphysical_9"] = "functional_9"
    """semantic action grouping preset used by body-part factorized CQL"""

    ood_actor_num: int = 1
    """number of action groups replaced by actor samples in each BF-CQL actor OOD candidate"""

    sync_cql: SyncCQLSettings = field(default_factory=SyncCQLSettings)
    """SYNC-QL synergy regularization settings"""

    dcql: DCQLSettings = field(default_factory=DCQLSettings)
    """DCQL escape-ray conservative regularization settings"""

    pbf_cql: PBFCQLSettings = field(default_factory=PBFCQLSettings)
    """PBF-CQL all-pair synergy regularization settings"""

    syndiag: SynDiagSettings = field(default_factory=SynDiagSettings)
    """synergy-OOD diagnostic logging settings (logging only, no loss changes)"""

    use_lagrange: bool = False
    """whether to use Lagrange multiplier auto-tuning for CQL conservative loss"""

    cql_target_action_gap: float = 10.0
    """target CQL gap threshold used by Lagrange mode (higher -> less conservative)"""

    cql_lagrange_learning_rate: float = 3e-4
    """learning rate for CQL Lagrange multiplier optimizer"""

    cql_lagrange_init: float = 1.0
    """initial value of CQL Lagrange multiplier"""

    cql_lagrange_max: float = 1e6
    """maximum clamp value for CQL Lagrange multiplier"""

    use_curr_tail_penalty: bool = False
    """whether to add an extra top-k tail penalty on (q_curr - curr_logp)"""

    curr_tail_weight: float = 0.0
    """weight for the additional current-proposal top-k tail penalty"""

    curr_tail_top_frac: float = 0.2
    """top fraction used for top-k tail extraction from current proposal samples"""

    bc_weight: float = 0.0
    """optional actor BC regularization weight (actor_loss += bc_weight * MSE(pi(s), a_data))"""

    target_entropy_ratio: float = 0.0
    """the ratio of the target entropy to the number of actions"""

    num_atoms: int = 101
    """number of quantile fractions (kept name for backward compatibility)"""

    v_min: float = -20.0
    """the minimum value of the support"""

    v_max: float = 20.0
    """the maximum value of the support"""

    quantile_huber_kappa: float = 1.0
    """Huber threshold for quantile regression loss"""

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

    normalized_action_training: bool = False
    """whether actor/critic train in normalized [-1, 1] action space and only scale actions for env rollout"""

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

    offline_dataset_path: str = "offline_data/fastsac_dataset.h5"
    """path to fixed offline dataset"""

    offline_block_size: int = 65536
    """number of contiguous transitions to read per HDF5 block refill"""

    offline_buffer_capacity: int = 262144
    """maximum number of transitions held in CPU RAM shuffle buffer"""

    offline_refill_threshold: int = 65536
    """refill shuffle buffer when remaining unsampled transitions fall below this threshold"""

    offline_pin_memory: bool = True
    """pin sampled CPU batches before CPU->GPU transfer"""

    offline_shuffle_block_order: bool = True
    """shuffle the order of contiguous HDF5 blocks each pass while keeping each block read contiguous"""

    use_gpu_cache: bool = False
    """whether to load the full offline dataset into GPU memory and sample directly on-device"""

    encoder_obs_key: str = "perception_obs"
    """the key of the encoder observation. only valid if use_cnn_encoder is True"""

    encoder_obs_shape: tuple[int, int, int] = (1, 13, 9)
    """the shape of the encoder observation. only valid if use_cnn_encoder is True"""

    use_cnn_encoder: bool = False
    """whether to use CNN for the encoder"""

    actor_warmup_steps: int = 500
    """number of initial global steps where actor update is skipped"""

    q_min: float | None = None
    """minimum Bellman target Q value; None disables lower clipping"""

    q_max: float | None = None
    """maximum Bellman target Q value; None disables upper clipping"""

    bellman_loss_type: Literal["mse", "huber"] = "mse"
    """Bellman regression loss type for critic targets"""

    huber_beta: float = 10.0
    """Smooth L1 beta used when bellman_loss_type='huber'"""

    actor_obs_keys: List[str] = field(default_factory=lambda: ["actor_obs"])
    critic_obs_keys: List[str] = field(default_factory=lambda: ["critic_obs"])

    reward_scale: int = 5

    cql_max_target_backup: bool = False

    backup_entropy : bool = False

    cql_max_target_backup_samples : int = 10


@dataclass(frozen=True)
class CALQLConfig:
    num_learning_iterations: int = 25000
    """total timesteps of the experiments"""

    critic_learning_rate: float = 3e-4
    """the learning rate of the critic"""

    actor_learning_rate: float = 1e-4
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

    batch_size: int = 2048
    """the batch size of sample from the replay memory"""

    learning_starts: int = 10
    """timestep to start learning"""

    policy_frequency: int = 4
    """the frequency of training policy (delayed)"""

    num_updates: int = 8
    """the number of updates to perform per step"""

    eval_interval: int = 1000
    """steps per offline_learn() call when max_steps is not provided"""

    cql_num_action_samples: int = 10
    """number of repeated action samples per state for conservative regularization"""

    cql_temperature: float = 1.0
    """temperature used in conservative log-sum-exp aggregation"""

    cql_weight: float = 5.0
    """weight of conservative quantile regularization"""

    calql_use_mc_return: bool = True
    """whether to calibrate sampled CQL Q-values with dataset/online Monte-Carlo returns"""

    calql_require_mc_return: bool = True
    """whether CAL-QL setup should fail if the offline dataset has no mc_return key"""

    calql_validate_complete_episodes: bool = True
    """whether CAL-QL setup should validate episode_data_complete when present"""

    calql_mc_gamma: float | None = None
    """discount used for online MC return calculation; None uses gamma"""

    use_lagrange: bool = False
    """whether to use Lagrange multiplier auto-tuning for CQL conservative loss"""

    cql_target_action_gap: float = 10.0
    """target CQL gap threshold used by Lagrange mode (higher -> less conservative)"""

    cql_lagrange_learning_rate: float = 3e-4
    """learning rate for CQL Lagrange multiplier optimizer"""

    cql_lagrange_init: float = 1.0
    """initial value of CQL Lagrange multiplier"""

    cql_lagrange_max: float = 1e6
    """maximum clamp value for CQL Lagrange multiplier"""

    use_curr_tail_penalty: bool = False
    """whether to add an extra top-k tail penalty on (q_curr - curr_logp)"""

    curr_tail_weight: float = 0.0
    """weight for the additional current-proposal top-k tail penalty"""

    curr_tail_top_frac: float = 0.2
    """top fraction used for top-k tail extraction from current proposal samples"""

    bc_weight: float = 0.0
    """optional actor BC regularization weight (actor_loss += bc_weight * MSE(pi(s), a_data))"""

    target_entropy_ratio: float = 0.0
    """the ratio of the target entropy to the number of actions"""

    num_atoms: int = 101
    """number of quantile fractions (kept name for backward compatibility)"""

    v_min: float = -20.0
    """the minimum value of the support"""

    v_max: float = 20.0
    """the maximum value of the support"""

    quantile_huber_kappa: float = 1.0
    """Huber threshold for quantile regression loss"""

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

    offline_dataset_path: str = "offline_data/fastsac_dataset.h5"
    """path to fixed offline dataset"""

    offline_block_size: int = 65536
    """number of contiguous transitions to read per HDF5 block refill"""

    offline_buffer_capacity: int = 262144
    """maximum number of transitions held in CPU RAM shuffle buffer"""

    offline_refill_threshold: int = 65536
    """refill shuffle buffer when remaining unsampled transitions fall below this threshold"""

    offline_pin_memory: bool = True
    """pin sampled CPU batches before CPU->GPU transfer"""

    offline_shuffle_block_order: bool = True
    """shuffle the order of contiguous HDF5 blocks each pass while keeping each block read contiguous"""

    use_gpu_cache: bool = False
    """whether to load the full offline dataset into GPU memory and sample directly on-device"""

    encoder_obs_key: str = "perception_obs"
    """the key of the encoder observation. only valid if use_cnn_encoder is True"""

    encoder_obs_shape: tuple[int, int, int] = (1, 13, 9)
    """the shape of the encoder observation. only valid if use_cnn_encoder is True"""

    use_cnn_encoder: bool = False
    """whether to use CNN for the encoder"""

    actor_warmup_steps: int = 500
    """number of initial global steps where actor update is skipped"""

    actor_obs_keys: List[str] = field(default_factory=lambda: ["actor_obs"])
    critic_obs_keys: List[str] = field(default_factory=lambda: ["critic_obs"])


    offline_pretrain_steps: int = 10000
    """number of CQL gradient steps before online finetuning"""

    online_total_steps: int = 40000
    """number of online finetuning gradient steps after offline pretraining"""

    online_eval_interval: int = 0
    """online phase evaluation interval. 0 disables online eval during finetuning"""

    online_warmup_steps: int = 1000
    """environment steps collected into online replay before online updates start"""

    online_collect_steps: int = 1
    """environment steps collected before each online update block"""

    updates_per_collect: int = 1
    """online gradient steps after each online collection block"""

    online_buffer_size: int = 524288
    """maximum number of online transitions stored in replay"""

    online_random_warmup: bool = False
    """collect warmup transitions with uniform random actions instead of the pretrained policy"""

    mixing_ratio_schedule: Literal["fixed", "linear"] = "fixed"
    """offline/online batch mixing schedule during online finetuning"""

    offline_mixing_ratio: float = 0.5
    """fixed offline fraction when mixing_ratio_schedule is fixed"""

    offline_mixing_start: float = 0.5
    """initial offline fraction when mixing_ratio_schedule is linear"""

    offline_mixing_end: float = 0.0
    """final offline fraction when mixing_ratio_schedule is linear"""

    mixing_anneal_steps: int = 10000
    """number of online gradient steps used to anneal the offline fraction"""

#============================================================================================


#===================================== FOR CQL ===============================================
@dataclass(frozen=True)
class OS_CQLConfig:
    num_learning_iterations: int = 25000
    """total timesteps of the experiments"""

    critic_learning_rate: float = 3e-4
    """the learning rate of the critic"""

    actor_learning_rate: float = 1e-4
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

    batch_size: int = 2048
    """the batch size of sample from the replay memory"""

    learning_starts: int = 10
    """timestep to start learning"""

    policy_frequency: int = 4
    """the frequency of training policy (delayed)"""

    num_updates: int = 8
    """the number of updates to perform per step"""

    eval_interval: int = 1000
    """steps per offline_learn() call when max_steps is not provided"""

    cql_num_action_samples: int = 10
    """number of repeated action samples per state for conservative regularization"""

    cql_temperature: float = 1.0
    """temperature used in conservative log-sum-exp aggregation"""

    cql_weight: float = 5.0
    """weight of conservative quantile regularization"""

    use_lagrange: bool = False
    """whether to use Lagrange multiplier auto-tuning for CQL conservative loss"""

    cql_target_action_gap: float = 10.0
    """target CQL gap threshold used by Lagrange mode (higher -> less conservative)"""

    cql_lagrange_learning_rate: float = 3e-4
    """learning rate for CQL Lagrange multiplier optimizer"""

    cql_lagrange_init: float = 1.0
    """initial value of CQL Lagrange multiplier"""

    cql_lagrange_max: float = 1e6
    """maximum clamp value for CQL Lagrange multiplier"""

    use_curr_tail_penalty: bool = False
    """whether to add an extra top-k tail penalty on (q_curr - curr_logp)"""

    curr_tail_weight: float = 0.0
    """weight for the additional current-proposal top-k tail penalty"""

    curr_tail_top_frac: float = 0.2
    """top fraction used for top-k tail extraction from current proposal samples"""

    bc_weight: float = 0.0
    """optional actor BC regularization weight (actor_loss += bc_weight * MSE(pi(s), a_data))"""

    target_entropy_ratio: float = 0.0
    """the ratio of the target entropy to the number of actions"""

    num_atoms: int = 101
    """number of quantile fractions (kept name for backward compatibility)"""

    v_min: float = -20.0
    """the minimum value of the support"""

    v_max: float = 20.0
    """the maximum value of the support"""

    quantile_huber_kappa: float = 1.0
    """Huber threshold for quantile regression loss"""

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

    offline_dataset_path: str = "offline_data/fastsac_dataset.h5"
    """path to fixed offline dataset"""

    offline_block_size: int = 65536
    """number of contiguous transitions to read per HDF5 block refill"""

    offline_buffer_capacity: int = 262144
    """maximum number of transitions held in CPU RAM shuffle buffer"""

    offline_refill_threshold: int = 65536
    """refill shuffle buffer when remaining unsampled transitions fall below this threshold"""

    offline_pin_memory: bool = True
    """pin sampled CPU batches before CPU->GPU transfer"""

    offline_shuffle_block_order: bool = True
    """shuffle the order of contiguous HDF5 blocks each pass while keeping each block read contiguous"""

    use_gpu_cache: bool = False
    """whether to load the full offline dataset into GPU memory and sample directly on-device"""

    encoder_obs_key: str = "perception_obs"
    """the key of the encoder observation. only valid if use_cnn_encoder is True"""

    encoder_obs_shape: tuple[int, int, int] = (1, 13, 9)
    """the shape of the encoder observation. only valid if use_cnn_encoder is True"""

    use_cnn_encoder: bool = False
    """whether to use CNN for the encoder"""

    actor_warmup_steps: int = 500
    """number of initial global steps where actor update is skipped"""

    actor_obs_keys: List[str] = field(default_factory=lambda: ["actor_obs"])
    critic_obs_keys: List[str] = field(default_factory=lambda: ["critic_obs"])

    cql_gap_margin : int = 5
    cql_gap_softplus_tau : int = 10
    use_softplus : bool = False
    """For One-side Conservative Q-learning"""

#============================================================================================


@dataclass(frozen=True)
class OS_CALQLConfig:
    num_learning_iterations: int = 25000
    """total timesteps of the experiments"""

    critic_learning_rate: float = 3e-4
    """the learning rate of the critic"""

    actor_learning_rate: float = 1e-4
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

    batch_size: int = 2048
    """the batch size of sample from the replay memory"""

    learning_starts: int = 10
    """timestep to start learning"""

    policy_frequency: int = 4
    """the frequency of training policy (delayed)"""

    num_updates: int = 8
    """the number of updates to perform per step"""

    eval_interval: int = 1000
    """steps per offline_learn() call when max_steps is not provided"""

    cql_num_action_samples: int = 10
    """number of repeated action samples per state for conservative regularization"""

    cql_temperature: float = 1.0
    """temperature used in conservative log-sum-exp aggregation"""

    cql_weight: float = 5.0
    """weight of conservative quantile regularization"""

    calql_use_mc_return: bool = True
    """whether to calibrate sampled CQL Q-values with dataset/online Monte-Carlo returns"""

    calql_require_mc_return: bool = True
    """whether OS-CAL-QL setup should fail if the offline dataset has no mc_return key"""

    calql_validate_complete_episodes: bool = True
    """whether OS-CAL-QL setup should validate episode_data_complete when present"""

    calql_mc_gamma: float | None = None
    """discount used for online MC return calculation; None uses gamma"""

    use_lagrange: bool = False
    """whether to use Lagrange multiplier auto-tuning for CQL conservative loss"""

    cql_target_action_gap: float = 10.0
    """target CQL gap threshold used by Lagrange mode (higher -> less conservative)"""

    cql_lagrange_learning_rate: float = 3e-4
    """learning rate for CQL Lagrange multiplier optimizer"""

    cql_lagrange_init: float = 1.0
    """initial value of CQL Lagrange multiplier"""

    cql_lagrange_max: float = 1e6
    """maximum clamp value for CQL Lagrange multiplier"""

    use_curr_tail_penalty: bool = False
    """whether to add an extra top-k tail penalty on (q_curr - curr_logp)"""

    curr_tail_weight: float = 0.0
    """weight for the additional current-proposal top-k tail penalty"""

    curr_tail_top_frac: float = 0.2
    """top fraction used for top-k tail extraction from current proposal samples"""

    bc_weight: float = 0.0
    """optional actor BC regularization weight (actor_loss += bc_weight * MSE(pi(s), a_data))"""

    target_entropy_ratio: float = 0.0
    """the ratio of the target entropy to the number of actions"""

    num_atoms: int = 101
    """number of quantile fractions (kept name for backward compatibility)"""

    v_min: float = -20.0
    """the minimum value of the support"""

    v_max: float = 20.0
    """the maximum value of the support"""

    quantile_huber_kappa: float = 1.0
    """Huber threshold for quantile regression loss"""

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

    offline_dataset_path: str = "offline_data/fastsac_dataset.h5"
    """path to fixed offline dataset"""

    offline_block_size: int = 65536
    """number of contiguous transitions to read per HDF5 block refill"""

    offline_buffer_capacity: int = 262144
    """maximum number of transitions held in CPU RAM shuffle buffer"""

    offline_refill_threshold: int = 65536
    """refill shuffle buffer when remaining unsampled transitions fall below this threshold"""

    offline_pin_memory: bool = True
    """pin sampled CPU batches before CPU->GPU transfer"""

    offline_shuffle_block_order: bool = True
    """shuffle the order of contiguous HDF5 blocks each pass while keeping each block read contiguous"""

    use_gpu_cache: bool = False
    """whether to load the full offline dataset into GPU memory and sample directly on-device"""

    encoder_obs_key: str = "perception_obs"
    """the key of the encoder observation. only valid if use_cnn_encoder is True"""

    encoder_obs_shape: tuple[int, int, int] = (1, 13, 9)
    """the shape of the encoder observation. only valid if use_cnn_encoder is True"""

    use_cnn_encoder: bool = False
    """whether to use CNN for the encoder"""

    actor_warmup_steps: int = 500
    """number of initial global steps where actor update is skipped"""

    actor_obs_keys: List[str] = field(default_factory=lambda: ["actor_obs"])
    critic_obs_keys: List[str] = field(default_factory=lambda: ["critic_obs"])


    offline_pretrain_steps: int = 10000
    """number of CQL gradient steps before online finetuning"""

    online_total_steps: int = 40000
    """number of online finetuning gradient steps after offline pretraining"""

    online_eval_interval: int = 0
    """online phase evaluation interval. 0 disables online eval during finetuning"""

    online_warmup_steps: int = 1000
    """environment steps collected into online replay before online updates start"""

    online_collect_steps: int = 1
    """environment steps collected before each online update block"""

    updates_per_collect: int = 1
    """online gradient steps after each online collection block"""

    online_buffer_size: int = 524288
    """maximum number of online transitions stored in replay"""

    online_random_warmup: bool = False
    """collect warmup transitions with uniform random actions instead of the pretrained policy"""

    mixing_ratio_schedule: Literal["fixed", "linear"] = "fixed"
    """offline/online batch mixing schedule during online finetuning"""

    offline_mixing_ratio: float = 0.5
    """fixed offline fraction when mixing_ratio_schedule is fixed"""

    offline_mixing_start: float = 0.5
    """initial offline fraction when mixing_ratio_schedule is linear"""

    offline_mixing_end: float = 0.0
    """final offline fraction when mixing_ratio_schedule is linear"""

    mixing_anneal_steps: int = 10000
    """number of online gradient steps used to anneal the offline fraction"""

    cql_gap_margin : int = 5
    cql_gap_softplus_tau : int = 10
    use_softplus : bool = False
    """For One-side Conservative Q-learning"""

#============================================================================================




@dataclass(frozen=True)
class CQLSupportAwareConfig(CQLConfig):
    """CQL config with support-aware Bellman backup selection."""

    use_support_aware_backup: bool = True
    """whether to use support-aware Bellman backup action selection"""

    backup_support_penalty: float = 1.0
    """lambda_support used in score = Q_target - lambda_support * overflow"""

    backup_mode: str = "project_select"
    """support-aware backup mode. currently only 'project_select' is supported"""

    support_percentile_low: float = 1.0
    """low percentile used for action support band in normalized u-space"""

    support_percentile_high: float = 99.0
    """high percentile used for action support band in normalized u-space"""


@dataclass(frozen=True)
class IQLConfig:
    num_learning_iterations: int = 25000
    """total gradient update iterations"""

    critic_learning_rate: float = 3e-4
    """learning rate for Q networks"""

    value_learning_rate: float = 3e-4
    """learning rate for value network"""

    actor_learning_rate: float = 3e-4
    """learning rate for actor network"""

    batch_size: int = 8192
    """global batch size"""

    num_updates: int = 8
    """number of gradient updates per outer step"""

    eval_interval: int = 1000
    """steps per offline_learn() call when max_steps is not provided"""

    discount: float = 0.97
    """discount factor"""

    tau: float = 0.005
    """soft update coefficient for Q target"""

    expectile: float = 0.7
    """expectile coefficient for value regression"""

    beta: float = 3.0
    """advantage temperature for actor weighting"""

    max_weight: float = 100.0
    """maximum clip for exp(beta * advantage)"""

    critic_hidden_dim: int = 768
    """hidden dimension of Q networks"""

    value_hidden_dim: int = 768
    """hidden dimension of value network"""

    actor_hidden_dim: int = 512
    """hidden dimension of actor network"""

    use_symmetry: bool = False
    """whether to apply symmetry augmentation to offline batches"""

    use_tanh: bool = True
    """whether to use tanh-squashed actor"""

    log_std_max: float = 0.0
    """maximum log std for actor"""

    log_std_min: float = -5.0
    """minimum log std for actor"""

    compile: bool = True
    """whether to use torch.compile for update functions"""

    obs_normalization: bool = True
    """whether to normalize actor/critic observations"""

    use_layer_norm: bool = True
    """whether to use layer normalization in networks"""

    max_grad_norm: float = 0.0
    """max grad norm (0 disables clipping)"""

    amp: bool = True
    """whether to use AMP"""

    amp_dtype: str = "bf16"
    """AMP dtype: bf16 or fp16"""

    weight_decay: float = 0.001
    """weight decay for optimizers"""

    save_interval: int = 1000
    """checkpoint interval"""

    logging_interval: int = 100
    """logging interval"""

    offline_dataset_path: str = "offline_data/fastsac_dataset.h5"
    """path to fixed offline dataset"""

    encoder_obs_key: str = "perception_obs"
    """encoder observation key, used only when use_cnn_encoder is True"""

    encoder_obs_shape: tuple[int, int, int] = (1, 13, 9)
    """encoder observation shape, used only when use_cnn_encoder is True"""

    use_cnn_encoder: bool = False
    """whether to use CNN actor encoder"""

    actor_obs_keys: List[str] = field(default_factory=lambda: ["actor_obs"])
    critic_obs_keys: List[str] = field(default_factory=lambda: ["critic_obs"])


@dataclass(frozen=True)
class BCConfig:
    num_learning_iterations: int = 25000
    """total gradient update iterations"""

    actor_learning_rate: float = 3e-4
    """learning rate for actor network"""

    batch_size: int = 8192
    """global batch size"""

    num_updates: int = 8
    """number of gradient updates per outer step"""

    eval_interval: int = 1000
    """steps per offline_learn() call when max_steps is not provided"""

    actor_hidden_dim: int = 512
    """hidden dimension of actor network"""

    use_symmetry: bool = False
    """whether to apply symmetry augmentation to offline batches"""

    use_tanh: bool = True
    """whether to use tanh-squashed actor"""

    log_std_max: float = 0.0
    """maximum log std for actor"""

    log_std_min: float = -5.0
    """minimum log std for actor"""

    compile: bool = True
    """whether to use torch.compile for update functions"""

    obs_normalization: bool = True
    """whether to normalize actor observations"""

    use_layer_norm: bool = True
    """whether to use layer normalization in actor"""

    max_grad_norm: float = 0.0
    """max grad norm (0 disables clipping)"""

    amp: bool = True
    """whether to use AMP"""

    amp_dtype: str = "bf16"
    """AMP dtype: bf16 or fp16"""

    weight_decay: float = 0.001
    """weight decay for optimizer"""

    save_interval: int = 1000
    """checkpoint interval"""

    logging_interval: int = 100
    """logging interval"""

    offline_dataset_path: str = "offline_data/fastsac_dataset.h5"
    """path to fixed offline dataset"""

    encoder_obs_key: str = "perception_obs"
    """encoder observation key, used only when use_cnn_encoder is True"""

    encoder_obs_shape: tuple[int, int, int] = (1, 13, 9)
    """encoder observation shape, used only when use_cnn_encoder is True"""

    use_cnn_encoder: bool = False
    """whether to use CNN actor encoder"""

    actor_obs_keys: List[str] = field(default_factory=lambda: ["actor_obs"])
    critic_obs_keys: List[str] = field(default_factory=lambda: ["critic_obs"])


@dataclass(frozen=True)
class TD3BCConfig:
    num_learning_iterations: int = 25000
    """total gradient update iterations"""

    critic_learning_rate: float = 3e-4
    """learning rate for critic network"""

    actor_learning_rate: float = 3e-4
    """learning rate for actor network"""

    batch_size: int = 8192
    """global batch size"""

    num_updates: int = 8
    """number of gradient updates per outer step"""

    eval_interval: int = 1000
    """steps per offline_learn() call when max_steps is not provided"""

    discount: float = 0.97
    """discount factor"""

    tau: float = 0.005
    """soft update coefficient for target networks"""

    policy_delay: int = 2
    """delayed actor/target update frequency in critic updates"""

    target_policy_noise: float = 0.2
    """std of target policy smoothing noise in normalized action space"""

    target_noise_clip: float = 0.5
    """absolute clip for target policy smoothing noise"""

    td3bc_alpha: float = 2.5
    """alpha used in lambda = alpha / mean(|Q|) for TD3+BC actor objective"""

    use_adaptive_lambda: bool = True
    """whether to use adaptive lambda scaling based on Q magnitude"""

    bc_coef: float = 1.0
    """coefficient for behavior cloning MSE loss in actor update"""

    actor_bc_warmup_steps: int = 1000
    """critic update steps to run actor as pure BC (Q-term disabled)"""

    td3bc_lambda_min: float = 0.0
    """minimum clamp for adaptive lambda"""

    td3bc_lambda_max: float = 10.0
    """maximum clamp for adaptive lambda to prevent early actor explosion"""

    critic_hidden_dim: int = 768
    """hidden dimension of Q networks"""

    actor_hidden_dim: int = 512
    """hidden dimension of actor network"""

    use_symmetry: bool = False
    """whether to apply symmetry augmentation to offline batches"""

    use_tanh: bool = True
    """whether to use tanh-bounded actor output in normalized u-space"""

    compile: bool = True
    """whether to use torch.compile for update functions"""

    obs_normalization: bool = True
    """whether to normalize actor/critic observations"""

    use_layer_norm: bool = True
    """whether to use layer normalization in networks"""

    max_grad_norm: float = 0.0
    """max grad norm (0 disables clipping)"""

    amp: bool = True
    """whether to use AMP"""

    amp_dtype: str = "bf16"
    """AMP dtype: bf16 or fp16"""

    weight_decay: float = 0.001
    """weight decay for optimizers"""

    save_interval: int = 1000
    """checkpoint interval"""

    logging_interval: int = 100
    """logging interval"""

    offline_dataset_path: str = "offline_data/fastsac_dataset.h5"
    """path to fixed offline dataset"""

    encoder_obs_key: str = "perception_obs"
    """encoder observation key, used only when use_cnn_encoder is True"""

    encoder_obs_shape: tuple[int, int, int] = (1, 13, 9)
    """encoder observation shape, used only when use_cnn_encoder is True"""

    use_cnn_encoder: bool = False
    """whether to use CNN actor encoder"""

    actor_obs_keys: List[str] = field(default_factory=lambda: ["actor_obs"])
    critic_obs_keys: List[str] = field(default_factory=lambda: ["critic_obs"])


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
class OfflineSACAlgoConfig:
    """Configuration for offline SAC algorithm wrapper."""

    _target_: str
    """Target algorithm class."""

    _recursive_: bool
    """Whether to recursively instantiate."""

    config: OfflineSACConfig
    """Algorithm-specific configuration."""


@dataclass(frozen=True)
class CODACAlgoConfig:
    """Configuration for CODAC algorithm wrapper."""

    _target_: str
    """Target algorithm class."""

    _recursive_: bool
    """Whether to recursively instantiate."""

    config: CODACConfig
    """Algorithm-specific configuration."""


@dataclass(frozen=True)
class CQLAlgoConfig:
    """Configuration for algorithm wrapper."""

    _target_: str
    """Target algorithm class."""

    _recursive_: bool
    """Whether to recursively instantiate."""

    config: CQLConfig
    """Algorithm-specific configuration."""




@dataclass(frozen=True)
class BFCQLAlgoConfig:
    """Configuration for body-part factorized CQL algorithm wrapper."""

    _target_: str
    """Target algorithm class."""

    _recursive_: bool
    """Whether to recursively instantiate."""

    config: BFCQLConfig
    """Algorithm-specific configuration."""


@dataclass(frozen=True)
class CALQLAlgoConfig:
    """Configuration for O2O/CAL-QL algorithm wrapper."""

    _target_: str
    """Target algorithm class."""

    _recursive_: bool
    """Whether to recursively instantiate."""

    config: CALQLConfig
    """Algorithm-specific configuration."""


@dataclass(frozen=True)
class OS_CQLAlgoConfig:
    """Configuration for algorithm wrapper."""

    _target_: str
    """Target algorithm class."""

    _recursive_: bool
    """Whether to recursively instantiate."""

    config: OS_CQLConfig
    """Algorithm-specific configuration."""


@dataclass(frozen=True)
class OS_CALQLAlgoConfig:
    """Configuration for O2O/CAL-QL algorithm wrapper."""

    _target_: str
    """Target algorithm class."""

    _recursive_: bool
    """Whether to recursively instantiate."""

    config: OS_CALQLConfig
    """Algorithm-specific configuration."""


@dataclass(frozen=True)
class CQLSupportAwareAlgoConfig:
    """Configuration for support-aware CQL algorithm wrapper."""

    _target_: str
    """Target algorithm class."""

    _recursive_: bool
    """Whether to recursively instantiate."""

    config: CQLSupportAwareConfig
    """Algorithm-specific configuration."""


@dataclass(frozen=True)
class IQLAlgoConfig:
    """Configuration for algorithm wrapper."""

    _target_: str
    """Target algorithm class."""

    _recursive_: bool
    """Whether to recursively instantiate."""

    config: IQLConfig
    """Algorithm-specific configuration."""


@dataclass(frozen=True)
class BCAlgoConfig:
    """Configuration for algorithm wrapper."""

    _target_: str
    """Target algorithm class."""

    _recursive_: bool
    """Whether to recursively instantiate."""

    config: BCConfig
    """Algorithm-specific configuration."""


@dataclass(frozen=True)
class TD3BCAlgoConfig:
    """Configuration for TD3+BC algorithm wrapper."""

    _target_: str
    """Target algorithm class."""

    _recursive_: bool
    """Whether to recursively instantiate."""

    config: TD3BCConfig
    """Algorithm-specific configuration."""


AlgoInitConfig = Union[
    PPOConfig,
    FastSACConfig,
    OfflineSACConfig,
    CODACConfig,
    CQLConfig,
    BFCQLConfig,
    CALQLConfig,
    OS_CQLConfig,
    OS_CALQLConfig,
    CQLSupportAwareConfig,
    IQLConfig,
    BCConfig,
    TD3BCConfig,
]

AlgoConfig = Union[
    PPOAlgoConfig,
    FastSACAlgoConfig,
    OfflineSACAlgoConfig,
    CODACAlgoConfig,
    CQLAlgoConfig,
    BFCQLAlgoConfig,
    CALQLAlgoConfig,
    OS_CQLAlgoConfig,
    OS_CALQLAlgoConfig,
    CQLSupportAwareAlgoConfig,
    IQLAlgoConfig,
    BCAlgoConfig,
    TD3BCAlgoConfig,
]
