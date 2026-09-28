"""Adaptive conservative-level weight functions for ACL-QL."""

from __future__ import annotations

from holosoma.utils.safe_torch_import import nn, torch


class ACLWeightNetwork(nn.Module):
    """One state-action MLP with two unconstrained outputs: w_mu and w_beta.

    The outputs are raw linear activations, as in the paper: positivity is not
    enforced architecturally but learned through the Eq. (23) penalty
    (:func:`compute_acl_positivity_loss`), so that penalty is a live training
    signal rather than a diagnostic.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256, num_layers: int = 3):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = obs_dim + action_dim
        for _ in range(num_layers):
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if obs.ndim != 2 or actions.ndim != 2:
            raise ValueError(f"ACL weights expect [B,D] obs/actions, got {obs.shape=} {actions.shape=}")
        out = self.net(torch.cat([obs, actions], dim=-1))
        return out[:, 0], out[:, 1]


def acl_action_distance(delta_normalized: torch.Tensor, mode: str) -> torch.Tensor:
    """ACL action distance from a dimension-normalized action delta.

    original_l2 is the paper-form baseline. rms is the only fairness variant:
    the same normalized coordinate delta divided by sqrt(action_dim).
    """
    if delta_normalized.ndim != 2:
        raise ValueError(f"ACL action delta must be [B, action_dim], got {delta_normalized.shape}")
    l2 = delta_normalized.norm(dim=-1)
    if mode == "original_l2":
        return l2
    if mode == "rms":
        return delta_normalized.square().mean(dim=-1).sqrt()
    raise ValueError(f"unknown ACL distance mode {mode!r}; expected 'original_l2' or 'rms'")


def acl_quality_raw(dataset_quality: torch.Tensor, action_distance: torch.Tensor) -> torch.Tensor:
    """Eq. (14) before range projection."""
    return 0.5 * (dataset_quality - 0.5 * action_distance + 1.0)


def clamp_acl_quality(raw_quality: torch.Tensor) -> torch.Tensor:
    """Project ACL quality targets to the numerically safe open unit interval."""
    return raw_quality.clamp(1e-6, 1.0 - 1e-6)


def acl_quality_for_ood(dataset_quality: torch.Tensor, action_distance: torch.Tensor) -> torch.Tensor:
    """Eq. (14) after the implementation's explicit range projection."""
    return clamp_acl_quality(acl_quality_raw(dataset_quality, action_distance))


def compute_acl_monotonicity_loss(
    w_mu: torch.Tensor,
    w_beta: torch.Tensor,
    m_mu: torch.Tensor,
    m_beta: torch.Tensor,
) -> torch.Tensor:
    """Eq. (15), using a one-step batch roll to form (i,j) pairs."""
    for tensor in (w_mu, w_beta, m_mu, m_beta):
        if tensor.ndim != 1:
            raise ValueError(f"ACL monotonicity tensors must be [B], got {tensor.shape}")
    sw_mu = torch.softmax(w_mu, dim=0)
    sw_beta = torch.softmax(w_beta, dim=0)
    sm_mu = torch.softmax(m_mu, dim=0)
    sm_beta = torch.softmax(m_beta, dim=0)
    mu_loss = ((sw_mu - sw_mu.roll(1)) - (sm_mu.roll(1) - sm_mu)).square().mean()
    beta_loss = ((sw_beta - sw_beta.roll(1)) - (sm_beta - sm_beta.roll(1))).square().mean()
    return mu_loss + beta_loss


def compute_acl_surrogate_loss(
    w_mu: torch.Tensor,
    w_beta: torch.Tensor,
    log_mu: torch.Tensor,
    log_beta: torch.Tensor,
    d_ord: torch.Tensor,
    d_cql: torch.Tensor,
    alpha: float | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Eqs. (20)-(21), averaged over the current batch."""
    for tensor in (w_mu, w_beta, log_mu, log_beta, d_ord, d_cql):
        if tensor.ndim != 1:
            raise ValueError(f"ACL surrogate tensors must be [B], got {tensor.shape}")
    alpha_t = torch.as_tensor(alpha, device=w_mu.device, dtype=w_mu.dtype)
    loss_ord = torch.relu(w_beta * (log_beta + 1.0) - w_mu * (log_mu + 1.0) + d_ord * (log_beta + 1.0))
    loss_cql = torch.relu(
        (w_mu - alpha_t) * (log_mu + 1.0)
        - (w_beta - alpha_t) * (log_beta + 1.0)
        + d_cql * (log_beta + 1.0)
    )
    return loss_ord.mean(), loss_cql.mean()


def compute_acl_positivity_loss(w_mu: torch.Tensor, w_beta: torch.Tensor) -> torch.Tensor:
    """Eq. (23): hinge penalty on negative weights.

    Active because :class:`ACLWeightNetwork` outputs are unconstrained; it is the
    only mechanism keeping w_mu and w_beta non-negative.
    """
    return torch.relu(-w_mu).mean() + torch.relu(-w_beta).mean()


def pearson_correlation(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Batch Pearson correlation between two [B] tensors (0 when either is constant)."""
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise ValueError(f"pearson_correlation expects matching [B] tensors, got {x.shape=} {y.shape=}")
    x = x.detach().float()
    y = y.detach().float()
    xc = x - x.mean()
    yc = y - y.mean()
    denom = xc.norm() * yc.norm()
    if denom <= eps:
        return torch.zeros((), device=x.device, dtype=x.dtype)
    return (xc * yc).sum() / denom
