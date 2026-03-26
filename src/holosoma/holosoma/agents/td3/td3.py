from __future__ import annotations

import torch
from torch import nn


class Actor(nn.Module):
    """Deterministic TD3 actor that outputs env/scaled actions (IQL-style semantics)."""

    def __init__(
        self,
        obs_indices: dict[str, dict[str, int]],
        obs_keys: list[str],
        n_act: int,
        num_envs: int,
        hidden_dim: int,
        use_tanh: bool = True,
        use_layer_norm: bool = True,
        device: torch.device | str | None = None,
        action_scale: torch.Tensor | None = None,
        action_bias: torch.Tensor | None = None,
        encoder_obs_key: str | None = None,
        encoder_obs_shape: tuple[int, int, int] | None = None,
    ):
        super().__init__()
        self.obs_indices = obs_indices
        self.obs_keys = obs_keys
        self.n_act = n_act
        self.use_tanh = use_tanh
        self.n_envs = num_envs
        self.device = device
        self.hidden_dim = hidden_dim
        self.use_layer_norm = use_layer_norm
        self.encoder_obs_key = encoder_obs_key
        self.encoder_obs_shape = encoder_obs_shape

        self.setup_network()

        if action_scale is not None:
            self.register_buffer("action_scale", action_scale.to(device))
        else:
            self.register_buffer("action_scale", torch.ones(n_act, device=device))

        if action_bias is not None:
            self.register_buffer("action_bias", action_bias.to(device))
        else:
            self.register_buffer("action_bias", torch.zeros(n_act, device=device))

    def setup_network(self) -> None:
        n_obs = sum(self.obs_indices[obs_key]["size"] for obs_key in self.obs_keys)
        self._setup_network_with_input_dim(n_obs)

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
        self.fc_mu = nn.Linear(self.hidden_dim // 4, self.n_act, device=self.device)
        nn.init.constant_(self.fc_mu.weight, 0.0)
        nn.init.constant_(self.fc_mu.bias, 0.0)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.process_obs(obs)
        x = self.net(x)
        pre_tanh_action = self.fc_mu(x)
        if self.use_tanh:
            tanh_action = torch.tanh(pre_tanh_action)
            action = tanh_action * self.action_scale + self.action_bias
        else:
            action = pre_tanh_action
        return action, pre_tanh_action

    @torch.no_grad()
    def explore(
        self,
        obs: torch.Tensor,
        dones: torch.Tensor | None = None,
        deterministic: bool = False,
        noise_std: float = 0.1,
        noise_clip: float = 0.5,
    ) -> torch.Tensor:
        del dones
        action, _ = self(obs)
        if deterministic or noise_std <= 0.0:
            return action

        noise = torch.randn_like(action) * noise_std
        noise = noise.clamp(-noise_clip, noise_clip)
        if self.use_tanh:
            min_action = self.action_bias - self.action_scale
            max_action = self.action_bias + self.action_scale
            return (action + noise).clamp(min_action, max_action)
        return action + noise

    def process_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                obs[..., self.obs_indices[obs_key]["start"] : self.obs_indices[obs_key]["end"]]
                for obs_key in self.obs_keys
            ],
            -1,
        )


class CNNActor(Actor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup_network(self) -> None:
        if self.encoder_obs_shape is None:
            raise ValueError("encoder_obs_shape must be provided for CNNActor")

        self.encoder = nn.Sequential(
            nn.Conv2d(self.encoder_obs_shape[0], 16, kernel_size=4, stride=2, padding=1, device=self.device),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=4, stride=2, padding=1, device=self.device),
            nn.ReLU(),
            nn.Flatten(),
        )

        cnn_output_dim = calculate_cnn_output_dim(self.encoder_obs_shape)
        state_obs_dim = sum(self.obs_indices[obs_key]["size"] for obs_key in self.obs_keys)
        total_input_dim = cnn_output_dim + state_obs_dim

        self._setup_network_with_input_dim(total_input_dim)

    def process_obs(self, obs: torch.Tensor) -> torch.Tensor:
        if self.encoder_obs_key is None or self.encoder_obs_shape is None:
            raise ValueError("encoder_obs_key and encoder_obs_shape must be provided for CNNActor")

        encoder_obs = torch.cat(
            [obs[..., self.obs_indices[self.encoder_obs_key]["start"] : self.obs_indices[self.encoder_obs_key]["end"]]],
            -1,
        )
        encoder_obs = encoder_obs.view(encoder_obs.shape[0], *self.encoder_obs_shape)
        encoder_x = self.encoder(encoder_obs)

        state_x = torch.cat(
            [
                obs[..., self.obs_indices[obs_key]["start"] : self.obs_indices[obs_key]["end"]]
                for obs_key in self.obs_keys
            ],
            -1,
        )
        return torch.cat([encoder_x, state_x], -1)


class QNetwork(nn.Module):
    def __init__(
        self,
        obs_indices: dict[str, dict[str, int]],
        obs_keys: list[str],
        n_act: int,
        hidden_dim: int,
        use_layer_norm: bool = True,
        device: torch.device | str | None = None,
    ):
        super().__init__()
        self.obs_indices = obs_indices
        self.obs_keys = obs_keys

        n_obs = sum(self.obs_indices[obs_key]["size"] for obs_key in self.obs_keys)
        self.net = nn.Sequential(
            nn.Linear(n_obs + n_act, hidden_dim, device=device),
            nn.LayerNorm(hidden_dim, device=device) if use_layer_norm else nn.Identity(),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2, device=device),
            nn.LayerNorm(hidden_dim // 2, device=device) if use_layer_norm else nn.Identity(),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4, device=device),
            nn.LayerNorm(hidden_dim // 4, device=device) if use_layer_norm else nn.Identity(),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, 1, device=device),
        )

    def _process_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                obs[..., self.obs_indices[obs_key]["start"] : self.obs_indices[obs_key]["end"]]
                for obs_key in self.obs_keys
            ],
            -1,
        )

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self._process_obs(obs), actions], dim=1)
        return self.net(x).squeeze(-1)


class DoubleQCritic(nn.Module):
    def __init__(
        self,
        obs_indices: dict[str, dict[str, int]],
        obs_keys: list[str],
        n_act: int,
        hidden_dim: int,
        use_layer_norm: bool = True,
        device: torch.device | str | None = None,
    ):
        super().__init__()
        self.q1 = QNetwork(
            obs_indices=obs_indices,
            obs_keys=obs_keys,
            n_act=n_act,
            hidden_dim=hidden_dim,
            use_layer_norm=use_layer_norm,
            device=device,
        )
        self.q2 = QNetwork(
            obs_indices=obs_indices,
            obs_keys=obs_keys,
            n_act=n_act,
            hidden_dim=hidden_dim,
            use_layer_norm=use_layer_norm,
            device=device,
        )

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q1(obs, actions), self.q2(obs, actions)


def calculate_cnn_output_dim(input_shape: tuple[int, int, int]) -> int:
    channels, height, width = input_shape

    h1 = (height + 2 * 1 - 4) // 2 + 1
    w1 = (width + 2 * 1 - 4) // 2 + 1

    h2 = (h1 + 2 * 1 - 4) // 2 + 1
    w2 = (w1 + 2 * 1 - 4) // 2 + 1

    return 16 * h2 * w2
