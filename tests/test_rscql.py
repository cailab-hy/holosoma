"""Unit tests for RSC-QL (random subspace conservatism).

The defining property: RSC-QL is BF-CQL with the conservative partition re-drawn
per critic update. Therefore, when the partition equals the fixed physical
grouping, one RSC critic step must be numerically IDENTICAL to one BF-CQL
critic step (same nets, same batch, same RNG seed). That equivalence is tested
end-to-end through both agents' real _update_q, plus partition-sampler
properties (valid partition, per-step variation, pair coverage) and a package
decoupling guard.
"""

from __future__ import annotations

import copy
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from holosoma.agents.bf_cql.bf_cql_agent import BFCQLAgent
from holosoma.agents.rscql.rscql import DoubleQCritic, FactorizedActor
from holosoma.agents.rscql.rscql_agent import RSCQLAgent
from holosoma.utils.safe_torch_import import TensorDict

RSC_PKG = Path(__file__).resolve().parents[1] / "src/holosoma/holosoma/agents/rscql"

GROUP_INDICES = [(0, 1, 2), (3, 4, 5), (6, 7), (8,)]
GROUP_NAMES = ["a", "b", "c", "d"]
N_ACT = 9
OBS_DIM = 10
CRITIC_OBS_DIM = 14


def _obs_indices(key: str, dim: int) -> dict[str, dict[str, int]]:
    return {key: {"start": 0, "end": dim, "size": dim}}


def _build_nets(seed: int = 0):
    torch.manual_seed(seed)
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
    for head in actor.group_mu_heads:
        torch.nn.init.normal_(head.weight, std=0.1)

    def critic():
        return DoubleQCritic(
            obs_indices=_obs_indices("critic_obs", CRITIC_OBS_DIM),
            obs_keys=["critic_obs"],
            n_act=N_ACT,
            hidden_dim=32,
            use_layer_norm=True,
            device="cpu",
        )

    qnet = critic()
    qnet_target = critic()
    qnet_target.load_state_dict(qnet.state_dict())
    return actor, qnet, qnet_target


def _shared_config() -> SimpleNamespace:
    return SimpleNamespace(
        amp=False,
        amp_dtype="bf16",
        gamma=0.99,
        cql_max_target_backup=False,
        cql_max_target_backup_samples=4,
        backup_entropy=False,
        q_min=None,
        q_max=None,
        bellman_loss_type="mse",
        huber_beta=1.0,
        use_lagrange=False,
        cql_target_action_gap=0.0,
        cql_lagrange_max=1e6,
        max_grad_norm=0.0,
        use_autotune=False,
        rscql=SimpleNamespace(resample_interval=1),
    )


def _skeleton(cls, actor, qnet, qnet_target):
    agent = object.__new__(cls)
    agent.device = "cpu"
    agent.is_multi_gpu = False
    agent.config = _shared_config()
    agent.reward_scale = 1.0
    agent._num_repeat_actions = 3
    agent._temperature = 1.0
    agent._cql_weight = 5.0
    agent._ood_actor_num = 1
    agent.normalized_action_training = True
    agent.env_action_scale = torch.ones(N_ACT)
    agent.env_action_bias = torch.zeros(N_ACT)
    agent.bf_cql_group_names = list(GROUP_NAMES)
    agent.bf_cql_group_indices = [tuple(g) for g in GROUP_INDICES]
    agent.actor = actor
    agent.qnet = qnet
    agent.qnet_target = qnet_target
    agent.log_alpha = torch.tensor([-1.0])
    agent.log_cql_alpha = None
    agent.scaler = torch.amp.GradScaler(enabled=False)
    agent.q_optimizer = torch.optim.SGD(qnet.parameters(), lr=0.0)
    if cls is RSCQLAgent:
        agent._rsc_block_sizes = [len(g) for g in GROUP_INDICES]
        agent._rsc_n_act = N_ACT
        agent._rsc_partition = None
    return agent


def _make_batch(batch_size: int = 6, seed: int = 7) -> TensorDict:
    torch.manual_seed(seed)
    return TensorDict(
        {
            "observations": torch.randn(batch_size, OBS_DIM),
            "actions": torch.randn(batch_size, N_ACT).clamp(-1.0, 1.0),
            "critic_observations": torch.randn(batch_size, CRITIC_OBS_DIM),
            "next": {
                "observations": torch.randn(batch_size, OBS_DIM),
                "critic_observations": torch.randn(batch_size, CRITIC_OBS_DIM),
                "rewards": torch.randn(batch_size),
                "dones": torch.zeros(batch_size, dtype=torch.long),
                "truncations": torch.zeros(batch_size, dtype=torch.long),
                "effective_n_steps": torch.ones(batch_size),
            },
        },
        batch_size=batch_size,
        device="cpu",
    )


def test_package_has_no_bf_cql_dependency():
    for py_file in RSC_PKG.glob("*.py"):
        source = py_file.read_text()
        assert "from holosoma.agents.bf_cql" not in source, f"{py_file.name} imports from agents/bf_cql"
        assert "import holosoma.agents.bf_cql" not in source, f"{py_file.name} imports from agents/bf_cql"


def test_partition_sampler_is_valid_partition():
    actor, qnet, qnet_target = _build_nets()
    agent = _skeleton(RSCQLAgent, actor, qnet, qnet_target)
    torch.manual_seed(0)
    blocks = agent._sample_rsc_partition()
    assert [b.numel() for b in blocks] == [len(g) for g in GROUP_INDICES]
    all_dims = torch.cat(blocks).sort().values
    assert torch.equal(all_dims, torch.arange(N_ACT))  # every dim exactly once


def test_partition_varies_across_draws_and_covers_all_pairs():
    actor, qnet, qnet_target = _build_nets()
    agent = _skeleton(RSCQLAgent, actor, qnet, qnet_target)
    torch.manual_seed(1)
    draws = [agent._sample_rsc_partition() for _ in range(300)]
    signatures = {tuple(tuple(b.tolist()) for b in d) for d in draws}
    assert len(signatures) > 250  # essentially always a fresh permutation

    covered: set[tuple[int, int]] = set()
    for d in draws:
        for block in d:
            dims = block.tolist()
            covered.update(tuple(sorted(p)) for p in combinations(dims, 2))
    all_pairs = {tuple(sorted(p)) for p in combinations(range(N_ACT), 2)}
    assert covered == all_pairs  # expectation covers every coupled direction


def test_identity_partition_reproduces_bf_cql_exactly():
    """RSC-QL == BF-CQL when the drawn partition equals the physical grouping."""
    actor_a, qnet_a, qtar_a = _build_nets(seed=0)
    actor_b, qnet_b, qtar_b = _build_nets(seed=0)  # same seed -> identical weights
    bf = _skeleton(BFCQLAgent, actor_a, qnet_a, qtar_a)
    rsc = _skeleton(RSCQLAgent, actor_b, qnet_b, qtar_b)

    identity_partition = [torch.tensor(list(g), dtype=torch.long) for g in GROUP_INDICES]
    data = _make_batch()

    torch.manual_seed(1234)
    out_bf = bf._update_q(data)
    torch.manual_seed(1234)
    out_rsc = rsc._update_q(data, identity_partition)

    names = [
        "reward_mean", "q_grad_norm", "q_loss", "q_target_max", "q_target_min",
        "alpha_loss", "conservative_loss", "bellman_loss", "cql_gap", "q_data_mean",
        "q_pi_minus_q_data", "rand_q_mean", "curr_q_mean", "next_q_mean",
        "curr_logp_mean", "next_logp_mean", "random_density_mean",
    ]
    for name, a, b in zip(names, out_bf, out_rsc):
        assert torch.allclose(a, b, atol=0.0, rtol=0.0), f"{name}: bf={a} rsc={b}"


def test_shuffled_partition_changes_conservative_loss():
    actor, qnet, qnet_target = _build_nets(seed=0)
    rsc = _skeleton(RSCQLAgent, actor, qnet, qnet_target)
    data = _make_batch()

    identity_partition = [torch.tensor(list(g), dtype=torch.long) for g in GROUP_INDICES]
    torch.manual_seed(99)
    out_identity = rsc._update_q(data, identity_partition)

    perm = torch.tensor([8, 3, 0, 6, 1, 7, 2, 5, 4])
    shuffled = list(torch.split(perm, [len(g) for g in GROUP_INDICES]))
    torch.manual_seed(99)
    out_shuffled = rsc._update_q(data, shuffled)

    assert torch.isfinite(out_shuffled[2])  # q_loss finite
    conservative_identity, conservative_shuffled = out_identity[6], out_shuffled[6]
    assert not torch.allclose(conservative_identity, conservative_shuffled), (
        "different partitions must evaluate different counterfactuals"
    )


def test_per_dim_log_probs_consistent_with_group_log_probs():
    actor, _, _ = _build_nets(seed=3)
    obs = torch.randn(5, OBS_DIM)

    torch.manual_seed(11)
    action_a, total_logp, group_logps = actor.get_actions_and_group_log_probs(obs)
    torch.manual_seed(11)
    action_b, per_dim = actor.get_actions_and_log_prob_per_dim(obs)

    assert torch.equal(action_a, action_b)
    assert torch.allclose(per_dim.sum(dim=-1), total_logp, atol=1e-6)
    for group_idx, dims in enumerate(GROUP_INDICES):
        assert torch.allclose(per_dim[:, list(dims)].sum(dim=-1), group_logps[group_idx], atol=1e-6)
