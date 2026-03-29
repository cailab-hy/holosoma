"""Network definitions for Offline CQL.

Actor Reuse Rationale
---------------------
The ``Actor`` class from ``fast_sac`` is reused **unchanged** for CQL.
This is valid because:

1. **Identical action space** — CQL and FastSAC both operate in the same
   continuous action space with per-joint scaling.  The ``action_scale`` and
   ``action_bias`` buffers registered by the Actor encode the environment's
   action boundaries, which are task-dependent, not algorithm-dependent.

2. **Action-scaling Jacobian is already accounted for** —
   ``Actor.get_actions_and_log_probs()`` computes the log-probability with
   *both* Jacobian corrections::

       log π(a|s) = log p(u|s)
                    - Σ_i log(1 − tanh²(u_i) + ε)   # tanh correction
                    - Σ_i log(scale_i + ε)            # scaling correction

   where ``u`` is the raw (pre-tanh) sample, ``a = tanh(u) · scale + bias``
   is the final action, and ``scale_i`` is the per-joint ``action_scale``.
   This ensures the density is correct for the CQL actor loss
   ``α · log π(a|s) − min_j Q_j(s, a)``.

3. **Checkpoint compatibility** — reusing the same ``Actor`` class guarantees
   that ``actor_state_dict`` keys are name-for-name identical between FastSAC
   and CQL checkpoints.  This enables:

   * Warm-starting CQL from a FastSAC policy (``actor_only=True``).
   * Using the same ONNX export / inference pipeline (``actor_onnx_wrapper``).

4. **Dataset contract** — the offline H5 dataset stores actions that are
   already in the *post-scaled* space (i.e. after ``tanh · scale + bias``),
   matching what the actor produces.  No re-scaling is needed at training
   time.

Scalar Twin-Q Critic
---------------------
CQL's conservative penalty requires computing::

    CQL_penalty = E_s[ logsumexp_a Q(s, a) ] − E_{s,a~D}[ Q(s, a) ]

The logsumexp is estimated via importance sampling with actions drawn from
(i) a uniform distribution and (ii) the current policy.  This is natural
with a *scalar* Q output.  A distributional (C51) critic would require
marginalising the value from the atom distribution *inside* the logsumexp,
which is mathematically valid but awkward and numerically fragile.

Target Critic
-------------
The target critic is an identical ``TwinQCritic`` whose weights are
Polyak-averaged toward the online critic::

    θ_target ← τ · θ_source + (1 − τ) · θ_target

Construction follows the same pattern as FastSAC: instantiate an identical
architecture, copy weights via ``load_state_dict``, then freeze gradients.
``TwinQCritic.create_target()`` encapsulates this.
"""

from __future__ import annotations

import copy

import torch
from torch import nn

# ── Actor is imported unchanged from fast_sac ──────────────────────────
from holosoma.agents.fast_sac.fast_sac import Actor, CNNActor  # noqa: F401


# ── Scalar Q-network (replaces DistributionalQNetwork) ─────────────────
class ScalarQNetwork(nn.Module):
    """Single Q-network that outputs a scalar Q-value.

    Architecture mirrors ``DistributionalQNetwork`` (same width / depth /
    activation / layer-norm convention) but the final layer produces a single
    output instead of ``num_atoms`` logits.
    """

    def __init__(
        self,
        n_obs: int,
        n_act: int,
        hidden_dim: int,
        use_layer_norm: bool = True,
        device: torch.device | None = None,
    ):
        super().__init__()
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

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Return Q-value of shape ``[B, 1]``."""
        x = torch.cat([obs, actions], dim=-1)
        return self.net(x)


# ── Twin-Q wrapper (replaces distributional Critic) ────────────────────
class TwinQCritic(nn.Module):
    """Ensemble of scalar Q-networks with the same obs-processing interface
    as ``fast_sac.Critic`` so that checkpoint key names for shared components
    (``obs_indices``, ``obs_keys``) stay compatible.

    Key differences from ``fast_sac.Critic``:

    * No ``q_support`` buffer, ``num_atoms``, ``v_min``, ``v_max``.
    * ``forward()`` returns ``[num_q, B, 1]`` (scalar) instead of
      ``[num_q, B, num_atoms]`` (logits).
    * No ``projection()`` or ``get_value()`` methods.
    * ``min_q()`` returns ``[B, 1]`` — the minimum across the ensemble,
      used for the actor loss and TD target computation.
    * ``q_values_for_actions()`` evaluates Q for multiple action sets in
      parallel, returning ``[num_q, B, N, 1]`` for the CQL logsumexp
      penalty estimation.
    * ``create_target()`` class method produces a frozen deep-copy for
      Polyak-averaged target Q-value computation.
    """

    def __init__(
        self,
        obs_indices: dict[str, dict[str, int]],
        obs_keys: list[str],
        n_act: int,
        hidden_dim: int,
        use_layer_norm: bool = True,
        num_q_networks: int = 2,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.obs_indices = obs_indices
        self.obs_keys = obs_keys
        self.n_act = n_act
        self.hidden_dim = hidden_dim
        self.use_layer_norm = use_layer_norm
        if num_q_networks < 1:
            raise ValueError("num_q_networks must be at least 1")
        self.num_q_networks = num_q_networks
        self.device = device

        n_obs = sum(self.obs_indices[obs_key]["size"] for obs_key in self.obs_keys)
        self.qnets = nn.ModuleList(
            [
                ScalarQNetwork(
                    n_obs=n_obs,
                    n_act=n_act,
                    hidden_dim=hidden_dim,
                    use_layer_norm=use_layer_norm,
                    device=device,
                )
                for _ in range(num_q_networks)
            ]
        )

    # ── forward pass ──────────────────────────────────────────────────

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Return stacked Q-values of shape ``[num_q, B, 1]``.

        Parameters
        ----------
        obs:
            Full (unsliced) observation tensor of shape ``[B, full_obs_dim]``.
            ``process_obs()`` is called internally to extract the critic-
            relevant slice.
        actions:
            Actions of shape ``[B, act_dim]`` in the post-scaled space
            (i.e. the same space the actor outputs).
        """
        x = self.process_obs(obs)
        return torch.stack([qnet(x, actions) for qnet in self.qnets], dim=0)

    def min_q(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Return the minimum Q-value across the ensemble: ``[B, 1]``.

        Used in three places during CQL training:

        * **Actor loss**: ``α · log π(a|s) − min_j Q_j(s, a)``
        * **TD target**: ``r + γ · min_j Q_j^{target}(s', a')``
        * **CQL penalty logsumexp** (when applied per-Q, the min is taken
          after the logsumexp; but some implementations take the min first —
          this method supports both patterns).

        Parameters
        ----------
        obs:
            Full observation tensor ``[B, full_obs_dim]``.
        actions:
            Actions ``[B, act_dim]``.
        """
        q_values = self.forward(obs, actions)  # [num_q, B, 1]
        return q_values.min(dim=0).values  # [B, 1]

    def q_values_for_actions(
        self, obs_processed: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate Q for a *batch of action sets* per observation.

        This is the workhorse for the CQL penalty, which estimates::

            logsumexp_a Q(s, a) ≈ log(1/N · Σ_i exp(Q(s, a_i)))

        where ``a_i`` comes from (i) a uniform distribution and/or (ii) the
        current policy.

        Parameters
        ----------
        obs_processed:
            **Already-sliced** observations of shape ``[B, obs_dim]``.
            This should be the output of ``process_obs()`` — callers are
            responsible for slicing once to avoid redundant work when
            evaluating multiple action sets on the same observations.
        actions:
            Shape ``[B, N, act_dim]`` — ``N`` actions to evaluate per
            sample (e.g. ``N = num_random + num_policy``).

        Returns
        -------
        torch.Tensor
            Shape ``[num_q, B, N, 1]``.  Each Q-network evaluates every
            ``(obs, action)`` pair independently.
        """
        B, N, A = actions.shape
        # Expand obs to match: [B, obs_dim] → [B*N, obs_dim]
        obs_expanded = obs_processed.unsqueeze(1).expand(B, N, -1).reshape(B * N, -1)
        actions_flat = actions.reshape(B * N, A)
        # Evaluate each Q-network on the flattened batch
        q_vals = torch.stack(
            [qnet(obs_expanded, actions_flat) for qnet in self.qnets], dim=0
        )  # [num_q, B*N, 1]
        return q_vals.view(self.num_q_networks, B, N, 1)

    # ── target construction ───────────────────────────────────────────

    @classmethod
    def create_target(cls, source: TwinQCritic) -> TwinQCritic:
        """Create a frozen deep-copy of *source* for Polyak-averaged target Q.

        The returned module has:

        * Identical architecture and initial weights (via ``copy.deepcopy``).
        * ``requires_grad_(False)`` — no gradient accumulation.
        * Same device placement as *source*.

        This mirrors the target construction in FastSAC (instantiate same
        class → ``load_state_dict``) but uses ``deepcopy`` for brevity
        since the ``__init__`` args are already captured in *source*.

        Usage::

            qnet = TwinQCritic(...)
            qnet_target = TwinQCritic.create_target(qnet)

            # In the training loop:
            polyak_update(qnet, qnet_target, tau=0.005)
        """
        target = copy.deepcopy(source)
        target.requires_grad_(False)
        return target

    # ── observation processing ────────────────────────────────────────

    def process_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Slice and concatenate observation groups.

        Identical to ``fast_sac.Critic.process_obs`` — extracts the
        critic-relevant observation keys from the full observation tensor
        using the ``obs_indices`` dictionary.
        """
        return torch.cat(
            [
                obs[..., self.obs_indices[obs_key]["start"] : self.obs_indices[obs_key]["end"]]
                for obs_key in self.obs_keys
            ],
            -1,
        )


# ── Target network Polyak update ─────────────────────────────────────────


@torch.no_grad()
def polyak_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    """In-place Polyak (exponential moving average) update of target params.

    ::

        θ_target ← τ · θ_source + (1 − τ) · θ_target

    Uses ``torch._foreach_*`` for fused element-wise operations, matching
    the same pattern as ``FastSACAgent``'s inline target update.

    Parameters
    ----------
    source:
        The online network whose parameters are being tracked.
    target:
        The target network to update in-place.
    tau:
        Interpolation coefficient in ``(0, 1]``.  Typical value: 0.005.
    """
    src_params = [p.data for p in source.parameters()]
    tgt_params = [p.data for p in target.parameters()]
    torch._foreach_mul_(tgt_params, 1.0 - tau)
    torch._foreach_add_(tgt_params, src_params, alpha=tau)
