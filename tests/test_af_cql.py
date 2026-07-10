"""Unit tests for AF-CQL (plain-CQL critic + BCPA block-coordinate actor).

Covers the BCPA mask sampling, the core gradient-blocking property of

    L_A = alpha * log pi(a|s) - lambda * Q(s, m * a_pi + (1 - m) * a_D)

and a guard that the package stays decoupled from agents/bf_cql.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from holosoma.agents.af_cql.af_cql import DoubleQCritic, FactorizedActor
from holosoma.agents.af_cql.af_cql_agent import AFCQLAgent

AF_PKG = Path(__file__).resolve().parents[1] / "src/holosoma/holosoma/agents/af_cql"

GROUP_INDICES = [(0, 1), (2, 3), (4,)]
GROUP_NAMES = ["g01", "g23", "g4"]
N_ACT = 5
OBS_DIM = 12
CRITIC_OBS_DIM = 16


def _obs_indices(key: str, dim: int) -> dict[str, dict[str, int]]:
    return {key: {"start": 0, "end": dim, "size": dim}}


def _skeleton_agent(
    *,
    bcpa_lambda: float = 1.0,
    num_active_groups: int = 1,
    mask_per_sample: bool = False,
) -> AFCQLAgent:
    agent = object.__new__(AFCQLAgent)
    agent.device = "cpu"
    agent.is_multi_gpu = False
    agent.bf_cql_group_names = list(GROUP_NAMES)
    agent.bf_cql_group_indices = [tuple(g) for g in GROUP_INDICES]
    mask = torch.zeros(len(GROUP_INDICES), N_ACT)
    for gi, dims in enumerate(GROUP_INDICES):
        mask[gi, list(dims)] = 1.0
    agent._af_group_dim_mask = mask
    agent.env_action_scale = torch.ones(N_ACT)
    agent.env_action_bias = torch.zeros(N_ACT)
    agent.normalized_action_training = True
    agent.config = SimpleNamespace(
        amp=False,
        amp_dtype="bf16",
        max_grad_norm=0.0,
        af_cql=SimpleNamespace(
            bcpa_lambda=bcpa_lambda,
            num_active_groups=num_active_groups,
            mask_per_sample=mask_per_sample,
        ),
    )
    return agent


def test_package_has_no_bf_cql_dependency():
    for py_file in AF_PKG.glob("*.py"):
        source = py_file.read_text()
        assert "from holosoma.agents.bf_cql" not in source, f"{py_file.name} imports from agents/bf_cql"
        assert "import holosoma.agents.bf_cql" not in source, f"{py_file.name} imports from agents/bf_cql"


def test_bcpa_mask_single_block_per_step():
    agent = _skeleton_agent(num_active_groups=1, mask_per_sample=False)
    torch.manual_seed(0)
    action_mask, group_mask = agent._sample_bcpa_action_mask(batch_size=7)
    assert action_mask.shape == (1, N_ACT)
    assert group_mask.shape == (1, len(GROUP_INDICES))
    assert int(group_mask.sum()) == 1
    active_group = int(group_mask[0].nonzero().item())
    expected = torch.zeros(N_ACT)
    expected[list(GROUP_INDICES[active_group])] = 1.0
    assert torch.equal(action_mask[0], expected)


def test_bcpa_mask_per_sample_and_group_clamp():
    agent = _skeleton_agent(num_active_groups=99, mask_per_sample=True)
    torch.manual_seed(1)
    action_mask, group_mask = agent._sample_bcpa_action_mask(batch_size=6)
    assert action_mask.shape == (6, N_ACT)
    assert group_mask.shape == (6, len(GROUP_INDICES))
    # num_active_groups clamps to G -> every group active -> full mask
    assert torch.equal(action_mask, torch.ones(6, N_ACT))


def test_bcpa_mask_covers_all_groups_over_steps():
    agent = _skeleton_agent(num_active_groups=1)
    torch.manual_seed(2)
    seen = set()
    for _ in range(64):
        _, group_mask = agent._sample_bcpa_action_mask(batch_size=4)
        seen.add(int(group_mask[0].nonzero().item()))
    assert seen == {0, 1, 2}


def test_bcpa_gradient_blocked_on_anchored_dims():
    """Anchored dims (m=0) must receive ZERO Q-gradient; active dims must receive some."""
    torch.manual_seed(3)
    batch_size = 16
    a_pi = torch.randn(batch_size, N_ACT, requires_grad=True)
    a_data = torch.randn(batch_size, N_ACT)
    weights = torch.randn(N_ACT)

    agent = _skeleton_agent(num_active_groups=1)
    action_mask, group_mask = agent._sample_bcpa_action_mask(batch_size)
    active_group = int(group_mask[0].nonzero().item())
    active_dims = list(GROUP_INDICES[active_group])
    anchored_dims = [d for d in range(N_ACT) if d not in active_dims]

    mixed = action_mask * a_pi + (1.0 - action_mask) * a_data
    loss = -(mixed @ weights).mean()  # -Q with additive toy critic
    loss.backward()

    assert a_pi.grad is not None
    assert torch.all(a_pi.grad[:, anchored_dims] == 0.0)
    assert torch.all(a_pi.grad[:, active_dims].abs() > 0.0)


def test_update_actor_only_active_group_heads_get_q_gradient():
    """End-to-end through the real _update_actor: with alpha ~ 0, inactive group
    mu-heads must receive zero gradient (their action dims are anchored to a_D)."""
    torch.manual_seed(4)
    agent = _skeleton_agent(num_active_groups=1)

    actor = FactorizedActor(
        obs_indices=_obs_indices("actor_obs", OBS_DIM),
        obs_keys=["actor_obs"],
        n_act=N_ACT,
        num_envs=1,
        hidden_dim=32,
        log_std_max=0.0,
        log_std_min=-5.0,
        use_tanh=True,
        use_layer_norm=True,
        device="cpu",
        action_scale=torch.ones(N_ACT),
        action_bias=torch.zeros(N_ACT),
        action_group_indices=GROUP_INDICES,
        action_group_names=GROUP_NAMES,
    )
    # break the zero-init symmetry so Q-gradients are non-trivial
    for head in actor.group_mu_heads:
        torch.nn.init.normal_(head.weight, std=0.1)
    critic = DoubleQCritic(
        obs_indices=_obs_indices("critic_obs", CRITIC_OBS_DIM),
        obs_keys=["critic_obs"],
        n_act=N_ACT,
        hidden_dim=32,
        use_layer_norm=True,
        device="cpu",
    )
    agent.actor = actor
    agent.qnet = critic
    agent.log_alpha = torch.tensor([-40.0])  # alpha ~ 0: kill the entropy path
    agent.scaler = torch.amp.GradScaler(enabled=False)
    agent.actor_optimizer = torch.optim.SGD(actor.parameters(), lr=0.0)

    from holosoma.utils.safe_torch_import import TensorDict

    batch_size = 8
    data = TensorDict(
        {
            "observations": torch.randn(batch_size, OBS_DIM),
            "critic_observations": torch.randn(batch_size, CRITIC_OBS_DIM),
            "actions": torch.randn(batch_size, N_ACT).clamp(-1, 1),
        },
        batch_size=batch_size,
        device="cpu",
    )

    # pin the sampled block to a fixed group so the assertion targets are deterministic
    fixed_group = 1
    group_mask = torch.zeros(1, len(GROUP_INDICES), dtype=torch.bool)
    group_mask[0, fixed_group] = True
    action_mask = torch.zeros(1, N_ACT)
    action_mask[0, list(GROUP_INDICES[fixed_group])] = 1.0
    agent._sample_bcpa_action_mask = lambda bs: (action_mask, group_mask)  # type: ignore[method-assign]

    grad_norm, actor_loss, policy_entropy, action_std = agent._update_actor(data)
    assert torch.isfinite(actor_loss)

    for group_idx, head in enumerate(actor.group_mu_heads):
        head_grad = head.weight.grad
        assert head_grad is not None
        grad_mag = head_grad.abs().sum().item()
        if group_idx == fixed_group:
            assert grad_mag > 0.0, "active group head must receive Q-gradient"
        else:
            assert grad_mag == pytest.approx(0.0, abs=1e-12), (
                f"anchored group head {group_idx} received gradient {grad_mag}"
            )


def test_bcpa_lambda_scales_q_term():
    agent0 = _skeleton_agent(bcpa_lambda=0.0)
    assert agent0.config.af_cql.bcpa_lambda == 0.0
    agent2 = _skeleton_agent(bcpa_lambda=2.0)
    assert agent2.config.af_cql.bcpa_lambda == 2.0
