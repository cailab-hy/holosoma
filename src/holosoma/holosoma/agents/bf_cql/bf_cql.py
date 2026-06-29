from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from holosoma.agents.cql.cql import Actor, DoubleQCritic


G1_FUNCTIONAL_9_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "left_hip",
        (
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
        ),
    ),
    (
        "left_knee_ankle",
        (
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
        ),
    ),
    (
        "right_hip",
        (
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
        ),
    ),
    (
        "right_knee_ankle",
        (
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
        ),
    ),
    (
        "waist",
        (
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
        ),
    ),
    (
        "left_shoulder",
        (
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
        ),
    ),
    (
        "left_elbow_wrist",
        (
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
        ),
    ),
    (
        "right_shoulder",
        (
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
        ),
    ),
    (
        "right_elbow_wrist",
        (
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ),
    ),
)


G1_COARSE_5_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "left_leg",
        (
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
        ),
    ),
    (
        "right_leg",
        (
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
        ),
    ),
    (
        "waist",
        (
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
        ),
    ),
    (
        "left_arm",
        (
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
        ),
    ),
    (
        "right_arm",
        (
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ),
    ),
)


G1_SYMMETRIC_14_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("hip_pitch", ("left_hip_pitch_joint", "right_hip_pitch_joint")),
    ("hip_roll", ("left_hip_roll_joint", "right_hip_roll_joint")),
    ("hip_yaw", ("left_hip_yaw_joint", "right_hip_yaw_joint")),
    ("knee", ("left_knee_joint", "right_knee_joint")),
    ("ankle_pitch", ("left_ankle_pitch_joint", "right_ankle_pitch_joint")),
    ("ankle_roll", ("left_ankle_roll_joint", "right_ankle_roll_joint")),
    ("waist", ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")),
    ("shoulder_pitch", ("left_shoulder_pitch_joint", "right_shoulder_pitch_joint")),
    ("shoulder_roll", ("left_shoulder_roll_joint", "right_shoulder_roll_joint")),
    ("shoulder_yaw", ("left_shoulder_yaw_joint", "right_shoulder_yaw_joint")),
    ("elbow", ("left_elbow_joint", "right_elbow_joint")),
    ("wrist_roll", ("left_wrist_roll_joint", "right_wrist_roll_joint")),
    ("wrist_pitch", ("left_wrist_pitch_joint", "right_wrist_pitch_joint")),
    ("wrist_yaw", ("left_wrist_yaw_joint", "right_wrist_yaw_joint")),
)


GROUP_PRESETS: dict[str, tuple[tuple[str, tuple[str, ...]], ...]] = {
    "functional_9": G1_FUNCTIONAL_9_GROUPS,
    "coarse_5": G1_COARSE_5_GROUPS,
    "symmetric_14": G1_SYMMETRIC_14_GROUPS,
}


def resolve_action_groups(
    grouping: str,
    dof_names: Sequence[str],
) -> tuple[list[str], list[tuple[int, ...]]]:
    if grouping not in GROUP_PRESETS:
        raise ValueError(f"Unknown BF-CQL action grouping '{grouping}'. Available: {sorted(GROUP_PRESETS)}")

    name_to_idx = {name: idx for idx, name in enumerate(dof_names)}
    group_names: list[str] = []
    group_indices: list[tuple[int, ...]] = []
    used_indices: list[int] = []
    for group_name, joint_names in GROUP_PRESETS[grouping]:
        missing = [joint_name for joint_name in joint_names if joint_name not in name_to_idx]
        if missing:
            raise ValueError(
                f"BF-CQL group '{group_name}' references joints not present in robot dof_names: {missing}"
            )
        indices = tuple(name_to_idx[joint_name] for joint_name in joint_names)
        group_names.append(group_name)
        group_indices.append(indices)
        used_indices.extend(indices)

    expected = list(range(len(dof_names)))
    if sorted(used_indices) != expected:
        duplicate_indices = sorted({idx for idx in used_indices if used_indices.count(idx) > 1})
        missing_indices = sorted(set(expected) - set(used_indices))
        raise ValueError(
            "BF-CQL action groups must cover every action dimension exactly once. "
            f"duplicates={duplicate_indices}, missing={missing_indices}"
        )

    return group_names, group_indices


class FactorizedActor(Actor):
    def __init__(
        self,
        *args,
        action_group_indices: Sequence[Sequence[int]],
        action_group_names: Sequence[str] | None = None,
        **kwargs,
    ):
        self.action_group_indices = [tuple(int(index) for index in group) for group in action_group_indices]
        self.action_group_names = list(action_group_names or [f"group_{idx}" for idx in range(len(self.action_group_indices))])
        super().__init__(*args, **kwargs)

    def _setup_network_with_input_dim(self, input_dim: int) -> None:
        self.net = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim, device=self.device),
            nn.LayerNorm(self.hidden_dim, device=self.device) if self.use_layer_norm else nn.Identity(),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2, device=self.device),
            nn.LayerNorm(self.hidden_dim // 2, device=self.device) if self.use_layer_norm else nn.Identity(),
            nn.SiLU(),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4, device=self.device),
            nn.LayerNorm(self.hidden_dim // 4, device=self.device) if self.use_layer_norm else nn.Identity(),
            nn.SiLU(),
        )
        latent_dim = self.hidden_dim // 4
        self.group_mu_heads = nn.ModuleList(
            [nn.Linear(latent_dim, len(group), device=self.device) for group in self.action_group_indices]
        )
        self.group_logstd_heads = nn.ModuleList(
            [nn.Linear(latent_dim, len(group), device=self.device) for group in self.action_group_indices]
        )
        for head in self.group_mu_heads:
            nn.init.constant_(head.weight, 0.0)
            nn.init.constant_(head.bias, 0.0)
        for head in self.group_logstd_heads:
            nn.init.constant_(head.weight, 0.0)
            nn.init.constant_(head.bias, 0.0)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.process_obs(obs)
        x = self.net(x)
        batch_shape = x.shape[:-1]
        mean = torch.empty(*batch_shape, self.n_act, device=x.device, dtype=x.dtype)
        log_std = torch.empty_like(mean)

        for group_indices, mu_head, logstd_head in zip(
            self.action_group_indices,
            self.group_mu_heads,
            self.group_logstd_heads,
            strict=True,
        ):
            group_index_list = list(group_indices)
            mean[..., group_index_list] = mu_head(x)
            group_log_std = torch.tanh(logstd_head(x))
            group_log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (
                group_log_std + 1
            )
            log_std[..., group_index_list] = group_log_std

        if self.use_tanh:
            action = torch.tanh(mean)
        else:
            action = mean

        return action, mean, log_std

    def get_actions_and_group_log_probs(
        self,
        obs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        _, mean, log_std = self(obs)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        raw_action = dist.rsample()

        if self.use_tanh:
            action = torch.tanh(raw_action)
            log_prob_per_dim = dist.log_prob(raw_action)
            log_prob_per_dim -= torch.log(1 - action.pow(2) + 1e-6)
        else:
            action = raw_action
            log_prob_per_dim = dist.log_prob(raw_action)

        group_log_probs = [
            log_prob_per_dim[..., list(group_indices)].sum(dim=-1)
            for group_indices in self.action_group_indices
        ]
        return action, log_prob_per_dim.sum(dim=-1), group_log_probs

    def get_actions_and_log_probs(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        action, log_prob, _ = self.get_actions_and_group_log_probs(obs)
        return action, log_prob


__all__ = [
    "DoubleQCritic",
    "FactorizedActor",
    "GROUP_PRESETS",
    "resolve_action_groups",
]
