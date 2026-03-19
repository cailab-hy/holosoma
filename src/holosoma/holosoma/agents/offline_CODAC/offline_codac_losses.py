"""CODAC conservative penalty helper functions.

All functions are pure (no side-effects on module state) so they can be
used inside ``torch.compile``-d update methods without issues.

Typical call order inside ``OfflineCODACAgent._compute_conservative_critic_loss``:

    1. ``sample_ood_actions``  — generate random + policy actions
    2. ``compute_q_values_for_actions`` — run critic on (obs, ood_actions)
    3. ``compute_conservative_penalty`` — logsumexp(OOD) - dataset
"""

from __future__ import annotations

from holosoma.utils.safe_torch_import import F, nn, torch


# ======================================================================
# 1. OOD action sampling
# ======================================================================


def sample_ood_actions(
    actor: nn.Module,
    observations: torch.Tensor,
    num_random: int,
    num_policy: int,
    action_scale: torch.Tensor,
    action_bias: torch.Tensor,
    action_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample out-of-distribution actions for the conservative penalty.

    Parameters
    ----------
    actor : nn.Module
        Policy network (``Actor``).  Used to sample ``num_policy`` actions.
    observations : torch.Tensor
        ``[batch, obs_dim]`` observations from the dataset.
    num_random : int
        Number of uniform-random actions per observation.
    num_policy : int
        Number of current-policy actions per observation.
    action_scale, action_bias : torch.Tensor
        Buffers from ``actor.action_scale`` / ``actor.action_bias`` that
        define the valid action range:  action ∈ [bias - scale, bias + scale].
    action_dim : int
        Dimensionality of the action space.

    Returns
    -------
    random_actions : torch.Tensor
        ``[batch * num_random, action_dim]``
    policy_actions : torch.Tensor
        ``[batch * num_policy, action_dim]``  (**detached** — no gradient
        flows back to the actor through the critic loss).
    policy_log_probs : torch.Tensor
        ``[batch * num_policy]``  log-probabilities of ``policy_actions``
        (detached).  Needed for importance-weight correction if desired.
    """
    batch_size = observations.shape[0]
    device = observations.device

    # ------------------------------------------------------------------
    # Uniform random actions in [bias − scale, bias + scale]
    # ------------------------------------------------------------------
    if num_random > 0:
        low = action_bias - action_scale   # [action_dim]
        high = action_bias + action_scale  # [action_dim]
        random_actions = (
            torch.rand(batch_size * num_random, action_dim, device=device) * (high - low) + low
        )
        assert random_actions.shape == (batch_size * num_random, action_dim)
    else:
        random_actions = torch.empty(0, action_dim, device=device)

    # ------------------------------------------------------------------
    # Current-policy actions (detached — no actor grad via critic loss)
    # ------------------------------------------------------------------
    if num_policy > 0:
        with torch.no_grad():
            obs_repeated = observations.unsqueeze(1).expand(
                batch_size, num_policy, -1
            ).reshape(batch_size * num_policy, -1)

            policy_actions, policy_log_probs = actor.get_actions_and_log_probs(obs_repeated)

        assert policy_actions.shape == (batch_size * num_policy, action_dim)
        assert policy_log_probs.shape == (batch_size * num_policy,)
    else:
        policy_actions = torch.empty(0, action_dim, device=device)
        policy_log_probs = torch.empty(0, device=device)

    return random_actions, policy_actions, policy_log_probs


# ======================================================================
# 2. Q-value computation for arbitrary (obs, action) pairs
# ======================================================================


def compute_q_values_for_actions(
    qnet: nn.Module,
    critic_observations: torch.Tensor,
    actions: torch.Tensor,
    num_samples: int,
) -> torch.Tensor:
    """Evaluate the distributional critic on repeated (obs, action) pairs and
    return scalar E[Z] Q-values.

    Parameters
    ----------
    qnet : nn.Module
        Critic network (``Critic``).
    critic_observations : torch.Tensor
        ``[batch, critic_obs_dim]`` — will be repeated ``num_samples`` times
        along the batch dimension.
    actions : torch.Tensor
        ``[batch * num_samples, action_dim]`` — already expanded.
    num_samples : int
        Number of action samples per observation (used to repeat obs).

    Returns
    -------
    torch.Tensor
        ``[num_q, batch, num_samples]`` scalar Q-values (E[Z]).
    """
    batch_size = critic_observations.shape[0]

    # Repeat observations to match the pre-expanded actions:
    # [batch, dim] → [batch, num_samples, dim] → [batch * num_samples, dim]
    obs_repeated = critic_observations.unsqueeze(1).expand(
        batch_size, num_samples, -1
    ).reshape(batch_size * num_samples, -1)

    assert obs_repeated.shape[0] == actions.shape[0], (
        f"obs_repeated batch {obs_repeated.shape[0]} != actions batch {actions.shape[0]}"
    )

    # Forward through critic: [num_q, batch*num_samples, num_atoms]
    q_logits = qnet(obs_repeated, actions)
    num_q = q_logits.shape[0]

    # Convert distributional logits → probabilities → scalar E[Z]
    q_probs = F.softmax(q_logits, dim=-1)               # [num_q, B*N, atoms]
    q_scalar = qnet.get_value(q_probs)                   # [num_q, B*N]

    # Reshape: [num_q, batch*num_samples] → [num_q, batch, num_samples]
    q_scalar = q_scalar.view(num_q, batch_size, num_samples)

    return q_scalar


# ======================================================================
# 3. Conservative penalty (CQL-style, scalar version)
# ======================================================================


def compute_conservative_penalty(
    q_ood: torch.Tensor,
    q_dataset: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Compute the CQL-style conservative penalty.

    .. math::

        \\text{penalty} = \\mathbb{E}_{q}\\Big[
            \\tau \\cdot \\text{logsumexp}\\!\\big(Q_{ood} / \\tau\\big)
            - Q_{dataset}
        \\Big]

    Parameters
    ----------
    q_ood : torch.Tensor
        ``[num_q, batch, num_ood_total]`` Q-values for all OOD actions
        (random + policy concatenated along the last dim).
    q_dataset : torch.Tensor
        ``[num_q, batch]`` Q-values for dataset actions.
    temperature : float
        Logsumexp temperature τ.  Must be > 0.

    Returns
    -------
    torch.Tensor
        Scalar conservative penalty averaged over Q-networks and batch.
    """
    assert temperature > 0, f"temperature must be > 0, got {temperature}"
    assert q_ood.dim() == 3, f"q_ood must be 3-D, got {q_ood.dim()}-D"
    assert q_dataset.dim() == 2, f"q_dataset must be 2-D, got {q_dataset.dim()}-D"

    # logsumexp over OOD actions: [num_q, batch]
    # Divide by temperature before logsumexp, multiply back after.
    logsumexp_ood = temperature * torch.logsumexp(q_ood / temperature, dim=-1)

    # Conservative penalty per (q-network, batch element)
    penalty = logsumexp_ood - q_dataset  # [num_q, batch]

    # Average over batch and sum over Q-networks (same convention as
    # parent distributional loss: mean(dim=1).sum(dim=0))
    penalty = penalty.mean(dim=1).sum(dim=0)

    return penalty
