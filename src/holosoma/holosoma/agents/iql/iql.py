from __future__ import annotations

import torch
from torch import nn


class Actor(nn.Module):
    def __init__(
        self,
        obs_indices: dict[str, dict[str, int]],
        obs_keys: list[str],
        n_act: int,
        num_envs: int,
        hidden_dim: int,
        log_std_max: float,
        log_std_min: float,
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
        self.log_std_max = log_std_max
        self.log_std_min = log_std_min
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
        self.fc_mu = nn.Sequential(
            nn.Linear(self.hidden_dim // 4, self.n_act, device=self.device),
        )
        self.fc_logstd = nn.Linear(self.hidden_dim // 4, self.n_act, device=self.device)
        nn.init.constant_(self.fc_mu[0].weight, 0.0)
        nn.init.constant_(self.fc_mu[0].bias, 0.0)
        nn.init.constant_(self.fc_logstd.weight, 0.0)
        nn.init.constant_(self.fc_logstd.bias, 0.0)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.process_obs(obs)
        x = self.net(x)
        mean = self.fc_mu(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)

        if self.use_tanh:
            tanh_mean = torch.tanh(mean)
            action = tanh_mean * self.action_scale + self.action_bias
        else:
            action = mean

        return action, mean, log_std

    def get_actions_and_log_probs(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, mean, log_std = self(obs)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        raw_action = dist.rsample()

        if self.use_tanh:
            tanh_action = torch.tanh(raw_action)
            action = tanh_action * self.action_scale + self.action_bias

            log_prob = dist.log_prob(raw_action)
            log_prob -= torch.log(1 - tanh_action.pow(2) + 1e-6)
            log_prob -= torch.log(self.action_scale + 1e-6)
        else:
            action = raw_action
            log_prob = dist.log_prob(raw_action)

        return action, log_prob.sum(1)

    def log_prob_dataset_actions(self, obs: torch.Tensor, dataset_actions: torch.Tensor) -> torch.Tensor:
        """Compute log pi(a_data | s) for squashed Gaussian actor.

        Shapes:
        - obs: [B, actor_obs_dim]
        - dataset_actions: [B, action_dim] (already in scaled action space)
        - return: [B]
        """
        _, mean, log_std = self(obs)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)

        if self.use_tanh:
            normalized_action = (dataset_actions - self.action_bias) / (self.action_scale + 1e-6)
            normalized_action = normalized_action.clamp(-1.0 + 1e-6, 1.0 - 1e-6)

            # atanh(x) = 0.5 * (log(1 + x) - log(1 - x))
            raw_action = 0.5 * (torch.log1p(normalized_action) - torch.log1p(-normalized_action))

            log_prob = dist.log_prob(raw_action)
            log_prob -= torch.log(1 - normalized_action.pow(2) + 1e-6)
            log_prob -= torch.log(self.action_scale + 1e-6)
        else:
            log_prob = dist.log_prob(dataset_actions)

        return log_prob.sum(dim=1)

    @torch.no_grad()
    def explore(
        self,
        obs: torch.Tensor,
        dones: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> torch.Tensor:
        _, mean, log_std = self(obs)
        if deterministic:
            if self.use_tanh:
                tanh_mean = torch.tanh(mean)
                return tanh_mean * self.action_scale + self.action_bias
            return mean

        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        raw_action = dist.rsample()

        if self.use_tanh:
            tanh_action = torch.tanh(raw_action)
            action = tanh_action * self.action_scale + self.action_bias
        else:
            action = raw_action

        return action

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
        self.n_act = n_act
        self.device = device

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


class ValueNetwork(nn.Module):
    def __init__(
        self,
        obs_indices: dict[str, dict[str, int]],
        obs_keys: list[str],
        hidden_dim: int,
        use_layer_norm: bool = True,
        device: torch.device | str | None = None,
    ):
        super().__init__()
        self.obs_indices = obs_indices
        self.obs_keys = obs_keys

        n_obs = sum(self.obs_indices[obs_key]["size"] for obs_key in self.obs_keys)
        self.net = nn.Sequential(
            nn.Linear(n_obs, hidden_dim, device=device),
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

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(self._process_obs(obs)).squeeze(-1)


def calculate_cnn_output_dim(input_shape: tuple[int, int, int]) -> int:
    channels, height, width = input_shape

    h1 = (height + 2 * 1 - 4) // 2 + 1
    w1 = (width + 2 * 1 - 4) // 2 + 1

    h2 = (h1 + 2 * 1 - 4) // 2 + 1
    w2 = (w1 + 2 * 1 - 4) // 2 + 1

    return 16 * h2 * w2
