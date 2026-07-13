from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class BehaviorVAE(nn.Module):
    """BCQ-style conditional VAE behavior sampler used by the MCB operator.

    Models beta(a|s) on the actor observation vector. All learning happens in
    the unit action space ``(a - action_bias) / action_scale`` (tanh-bounded),
    so per-joint action ranges do not skew the reconstruction loss;
    ``decode()`` returns env-scale actions ready for the critic.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        latent_dim: int,
        hidden_dim: int,
        action_scale: torch.Tensor,
        action_bias: torch.Tensor,
        device: torch.device | str | None = None,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.device = device

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim, device=device),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, device=device),
            nn.SiLU(),
        )
        self.enc_mean = nn.Linear(hidden_dim, latent_dim, device=device)
        self.enc_log_std = nn.Linear(hidden_dim, latent_dim, device=device)

        self.decoder = nn.Sequential(
            nn.Linear(obs_dim + latent_dim, hidden_dim, device=device),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, device=device),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim, device=device),
        )

        self.register_buffer("action_scale", action_scale.detach().clone().to(device))
        self.register_buffer("action_bias", action_bias.detach().clone().to(device))

    def _to_unit(self, actions_env: torch.Tensor) -> torch.Tensor:
        return ((actions_env - self.action_bias) / (self.action_scale + 1e-6)).clamp(-1.0, 1.0)

    def _decode_unit(self, obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.decoder(torch.cat([obs, z], dim=-1)))

    def forward(self, obs: torch.Tensor, actions_env: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (recon_unit, target_unit, mean, std) for the ELBO loss."""
        target_unit = self._to_unit(actions_env)
        h = self.encoder(torch.cat([obs, target_unit], dim=-1))
        mean = self.enc_mean(h)
        # Clamp as in BCQ so std stays in a sane range early in training.
        log_std = self.enc_log_std(h).clamp(-4.0, 15.0)
        std = log_std.exp()
        z = mean + std * torch.randn_like(std)
        recon_unit = self._decode_unit(obs, z)
        return recon_unit, target_unit, mean, std

    def loss(self, obs: torch.Tensor, actions_env: torch.Tensor, kl_weight: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        recon_unit, target_unit, mean, std = self(obs, actions_env)
        recon_loss = F.mse_loss(recon_unit, target_unit)
        kl_loss = -0.5 * (1.0 + 2.0 * std.log() - mean.pow(2) - std.pow(2)).sum(dim=-1).mean()
        return recon_loss + kl_weight * kl_loss, recon_loss, kl_loss

    @torch.no_grad()
    def decode(self, obs: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
        """Sample env-scale actions from the learned behavior prior (z clipped as in BCQ)."""
        if z is None:
            z = torch.randn(obs.shape[0], self.latent_dim, device=obs.device, dtype=obs.dtype).clamp(-0.5, 0.5)
        return self._decode_unit(obs, z) * self.action_scale + self.action_bias
