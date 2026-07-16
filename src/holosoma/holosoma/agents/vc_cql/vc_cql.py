"""Low-rank actor and covariance math for Value-Contour CQL.

The actor remains a tanh-squashed Gaussian in the same physical action space as
scalar CQL, but its pre-tanh covariance is low-rank plus diagonal.  VC-CQL maps
that covariance through the local tanh/action-scale Jacobian and fits it to a
stop-gradient covariance estimated from physical-action target-Q contours.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from holosoma.agents.cql.cql import Actor


def low_rank_gaussian_log_prob(
    value: torch.Tensor,
    mean: torch.Tensor,
    log_std: torch.Tensor,
    factor: torch.Tensor,
) -> torch.Tensor:
    """Return log N(value; mean, FF^T + diag(exp(2 log_std)))."""

    original_dtype = value.dtype
    compute_dtype = torch.float32 if original_dtype in (torch.float16, torch.bfloat16) else original_dtype
    with torch.autocast(device_type=value.device.type, enabled=False):
        value_f = value.to(compute_dtype)
        mean_f = mean.to(compute_dtype)
        log_std_f = log_std.to(compute_dtype)
        factor_f = factor.to(compute_dtype)

        delta = value_f - mean_f
        inverse_diag = torch.exp(-2.0 * log_std_f)
        rank = factor_f.shape[-1]
        identity = torch.eye(rank, device=value.device, dtype=compute_dtype).expand(value.shape[0], rank, rank)
        capacitance = identity + torch.einsum("bdr,bd,bds->brs", factor_f, inverse_diag, factor_f)
        rhs = torch.einsum("bdr,bd->br", factor_f, inverse_diag * delta)
        solved_rhs = torch.linalg.solve(capacitance, rhs.unsqueeze(-1)).squeeze(-1)

        mahalanobis = (delta.square() * inverse_diag).sum(dim=-1) - (rhs * solved_rhs).sum(dim=-1)
        mahalanobis = mahalanobis.clamp_min(0.0)
        _, capacitance_logdet = torch.linalg.slogdet(capacitance)
        covariance_logdet = (2.0 * log_std_f).sum(dim=-1) + capacitance_logdet
        normalizer = value.shape[-1] * math.log(2.0 * math.pi)
        log_prob = -0.5 * (normalizer + covariance_logdet + mahalanobis)
    return log_prob.to(original_dtype)


def low_rank_covariance(log_std: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
    """Construct FF^T + diag(exp(2 log_std))."""

    return factor @ factor.transpose(-1, -2) + torch.diag_embed(torch.exp(2.0 * log_std))


def linearized_squashed_covariance(
    raw_covariance: torch.Tensor,
    raw_mean: torch.Tensor,
    action_scale: torch.Tensor,
) -> torch.Tensor:
    """Map raw covariance through J=diag(action_scale * (1-tanh(mean)^2))."""

    jacobian_diagonal = action_scale * (1.0 - torch.tanh(raw_mean).square())
    return raw_covariance * jacobian_diagonal.unsqueeze(-1) * jacobian_diagonal.unsqueeze(-2)


def covariance_kl(
    policy_covariance: torch.Tensor,
    target_covariance: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """Compute KL(N(0, Sigma_pi) || N(0, Sigma_Q)) per sample."""

    original_dtype = policy_covariance.dtype
    with torch.autocast(device_type=policy_covariance.device.type, enabled=False):
        policy_covariance = policy_covariance.float()
        target_covariance = target_covariance.float()
        action_dim = policy_covariance.shape[-1]
        identity = torch.eye(action_dim, device=policy_covariance.device, dtype=policy_covariance.dtype)
        policy_covariance = policy_covariance + epsilon * identity
        target_covariance = target_covariance + epsilon * identity

        target_cholesky = torch.linalg.cholesky(target_covariance)
        target_inverse_policy = torch.cholesky_solve(policy_covariance, target_cholesky)
        trace_term = torch.diagonal(target_inverse_policy, dim1=-2, dim2=-1).sum(dim=-1)
        _, policy_logdet = torch.linalg.slogdet(policy_covariance)
        _, target_logdet = torch.linalg.slogdet(target_covariance)
        kl = 0.5 * (trace_term - action_dim + target_logdet - policy_logdet)
    return kl.clamp_min(0.0).to(original_dtype)


def weighted_contour_covariance(
    deltas: torch.Tensor,
    q_drops: torch.Tensor,
    *,
    temperature: float,
    epsilon: float,
    shrinkage: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate Sigma_Q = sum_k w_k delta_k delta_k^T / (sum_k w_k + eps)."""

    with torch.autocast(device_type=deltas.device.type, enabled=False):
        deltas = deltas.float()
        q_drops = q_drops.float()
        weights = torch.exp(-torch.relu(q_drops) / temperature)
        denominator = weights.sum(dim=1).clamp_min(epsilon)
        covariance = torch.einsum("bk,bkd,bke->bde", weights, deltas, deltas)
        covariance = covariance / denominator[:, None, None]
        if shrinkage > 0.0:
            diagonal = torch.diag_embed(torch.diagonal(covariance, dim1=-2, dim2=-1))
            covariance = (1.0 - shrinkage) * covariance + shrinkage * diagonal
        identity = torch.eye(covariance.shape[-1], device=covariance.device, dtype=covariance.dtype)
        covariance = covariance + epsilon * identity
    return covariance, weights


class VCActor(Actor):
    """Tanh actor with state-conditioned low-rank-plus-diagonal covariance."""

    def __init__(self, *args, covariance_rank: int, factor_max: float, **kwargs):
        self.covariance_rank = covariance_rank
        self.factor_max = factor_max
        super().__init__(*args, **kwargs)

    def _setup_network_with_input_dim(self, input_dim: int) -> None:
        super()._setup_network_with_input_dim(input_dim)
        self.fc_cov_factor = nn.Linear(
            self.hidden_dim // 4,
            self.n_act * self.covariance_rank,
            device=self.device,
        )
        nn.init.normal_(self.fc_cov_factor.weight, mean=0.0, std=1e-3)
        nn.init.normal_(self.fc_cov_factor.bias, mean=0.0, std=1e-3)

    def _features(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(self.process_obs(obs))

    def _distribution_from_features(
        self,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = self.fc_mu(features)
        raw_log_std = torch.tanh(self.fc_logstd(features))
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (raw_log_std + 1.0)
        raw_factor = self.fc_cov_factor(features).view(-1, self.n_act, self.covariance_rank)
        factor = self.factor_max * torch.tanh(raw_factor)
        return mean, log_std, factor

    def distribution_parameters(
        self,
        obs: torch.Tensor,
        *,
        detach_features_for_covariance: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self._features(obs)
        mean, log_std, factor = self._distribution_from_features(features)
        if detach_features_for_covariance:
            _, log_std, factor = self._distribution_from_features(features.detach())
        return mean, log_std, factor

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std, _ = self.distribution_parameters(obs)
        action = self.squash_raw_action(mean) if self.use_tanh else mean
        return action, mean, log_std

    def squash_raw_action(self, raw_action: torch.Tensor) -> torch.Tensor:
        if not self.use_tanh:
            return raw_action
        return torch.tanh(raw_action) * self.action_scale + self.action_bias

    def get_actions_and_log_probs(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std, factor = self.distribution_parameters(obs)
        diagonal_noise = torch.randn_like(mean)
        low_rank_noise = torch.randn(
            mean.shape[0],
            self.covariance_rank,
            device=mean.device,
            dtype=mean.dtype,
        )
        raw_action = mean + log_std.exp() * diagonal_noise + torch.einsum(
            "bdr,br->bd", factor, low_rank_noise
        )
        log_prob = low_rank_gaussian_log_prob(raw_action, mean, log_std, factor)

        if self.use_tanh:
            tanh_action = torch.tanh(raw_action)
            action = tanh_action * self.action_scale + self.action_bias
            log_prob = log_prob - torch.log(1.0 - tanh_action.square() + 1e-6).sum(dim=-1)
            log_prob = log_prob - torch.log(self.action_scale + 1e-6).sum()
        else:
            action = raw_action
        return action, log_prob

    def log_prob_dataset_actions(self, obs: torch.Tensor, dataset_actions: torch.Tensor) -> torch.Tensor:
        mean, log_std, factor = self.distribution_parameters(obs)
        if self.use_tanh:
            squashed_action = (dataset_actions - self.action_bias) / (self.action_scale + 1e-6)
            squashed_action = squashed_action.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
            raw_action = torch.atanh(squashed_action)
            log_prob = low_rank_gaussian_log_prob(raw_action, mean, log_std, factor)
            log_prob = log_prob - torch.log(1.0 - squashed_action.square() + 1e-6).sum(dim=-1)
            log_prob = log_prob - torch.log(self.action_scale + 1e-6).sum()
            return log_prob
        return low_rank_gaussian_log_prob(dataset_actions, mean, log_std, factor)

    @torch.no_grad()
    def explore(
        self,
        obs: torch.Tensor,
        dones: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> torch.Tensor:
        mean, log_std, factor = self.distribution_parameters(obs)
        if deterministic:
            return self.squash_raw_action(mean)
        raw_action = mean + log_std.exp() * torch.randn_like(mean)
        raw_action = raw_action + torch.einsum(
            "bdr,br->bd",
            factor,
            torch.randn(mean.shape[0], self.covariance_rank, device=mean.device, dtype=mean.dtype),
        )
        return self.squash_raw_action(raw_action)
