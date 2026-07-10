"""Unit tests for PSC v0 (Principal-Subspace Conservatism).

Gates (per spec):
 1. rotation round-trip  2. block splice anchors  3. bounds clamp
 4. basis mismatch guards  5. identity-basis equivalence to BF-CQL
 6. checkpoint basis save/load with action_dim assertion
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from holosoma.agents.bf_cql.bf_cql_agent import BFCQLAgent
from holosoma.agents.psc.psc import DoubleQCritic, FactorizedActor
from holosoma.agents.psc.psc_agent import PSCAgent
from holosoma.utils.safe_torch_import import TensorDict

GROUP_INDICES = [(0, 1, 2), (3, 4, 5), (6, 7), (8,)]
GROUP_NAMES = ["a", "b", "c", "d"]
BLOCK_SIZES = (3, 3, 2, 1)
N_ACT = 9
OBS_DIM = 10
CRITIC_OBS_DIM = 14


def _obs_indices(key: str, dim: int) -> dict[str, dict[str, int]]:
    return {key: {"start": 0, "end": dim, "size": dim}}


def _identity_basis(n_act: int = N_ACT, eig: float = 0.25, space: str = "normalized") -> dict:
    return {
        "mu": torch.zeros(n_act),
        "U": torch.eye(n_act),
        "eigvals": torch.full((n_act,), eig),
        "meta": {"space": space, "dataset_path": "synthetic", "timestamp": "t", "action_dim": n_act},
    }


def _random_basis(n_act: int = N_ACT, seed: int = 5) -> dict:
    torch.manual_seed(seed)
    U, _ = torch.linalg.qr(torch.randn(n_act, n_act))
    eigvals = torch.logspace(0, -2, n_act)
    basis = _identity_basis(n_act)
    basis["U"] = U
    basis["eigvals"] = eigvals
    basis["mu"] = 0.1 * torch.randn(n_act)
    return basis


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


def _psc_settings(**overrides) -> SimpleNamespace:
    base = dict(
        enabled=True,
        basis_path="synthetic",
        block_sizes=BLOCK_SIZES,
        rand_scale_mode="sqrt_eig_floored",
        rand_range_mult=2.0,
        scale_floor_quantile=0.5,
        block_weighting="uniform",
        recompute_check=False,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _shared_config(**psc_overrides) -> SimpleNamespace:
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
        psc=_psc_settings(**psc_overrides),
    )


def _skeleton(cls, actor, qnet, qnet_target, basis: dict | None = None):
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
    if cls is PSCAgent and basis is not None:
        agent._psc_init_from_basis(basis)
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


def _psc_agent_with_random_basis():
    actor, qnet, qnet_target = _build_nets(seed=2)
    return _skeleton(PSCAgent, actor, qnet, qnet_target, basis=_random_basis())


# --- 1. rotation round-trip -------------------------------------------------


def test_rotation_round_trip():
    agent = _psc_agent_with_random_basis()
    torch.manual_seed(0)
    a = torch.randn(32, N_ACT)
    u = (a - agent._psc_mu) @ agent._psc_U
    back = u @ agent._psc_U.t() + agent._psc_mu
    assert torch.allclose(back, a, atol=1e-5)


# --- 2. block splice anchors ------------------------------------------------


def test_block_splice_anchors_dataset_outside_selected_block():
    agent = _psc_agent_with_random_basis()
    torch.manual_seed(1)
    rows = 16
    u_data = torch.randn(rows, N_ACT)
    u_actor = torch.randn(rows, N_ACT)
    for group_idx, (block_start, block_end) in enumerate(agent._psc_block_slices):
        mask = torch.zeros(rows, len(agent._psc_block_slices), dtype=torch.bool)
        mask[:, group_idx] = True
        a_cf = agent._psc_counterfactual_actor_actions(
            u_data, u_actor, mask, agent._psc_U, agent._psc_mu,
            torch.full((N_ACT,), -1e9), torch.full((N_ACT,), 1e9),  # disable clamp
        )
        u_cf = (a_cf - agent._psc_mu) @ agent._psc_U
        inside = slice(block_start, block_end)
        assert torch.allclose(u_cf[:, inside], u_actor[:, inside], atol=1e-5)
        outside = [i for i in range(N_ACT) if not (block_start <= i < block_end)]
        assert torch.allclose(u_cf[:, outside], u_data[:, outside], atol=1e-5)


# --- 3. bounds clamp ----------------------------------------------------------


def test_counterfactuals_respect_action_bounds():
    agent = _psc_agent_with_random_basis()
    torch.manual_seed(2)
    rows = 64
    u_data = torch.randn(rows, N_ACT) * 5.0  # deliberately extreme
    u_actor = torch.randn(rows, N_ACT) * 5.0
    mask = torch.ones(rows, len(agent._psc_block_slices), dtype=torch.bool)
    a_cf = agent._psc_counterfactual_actor_actions(
        u_data, u_actor, mask, agent._psc_U, agent._psc_mu,
        agent._psc_action_low, agent._psc_action_high,
    )
    assert torch.all(a_cf >= agent._psc_action_low - 1e-6)
    assert torch.all(a_cf <= agent._psc_action_high + 1e-6)


# --- 4. basis mismatch guards -------------------------------------------------


def test_basis_mismatch_guards_raise():
    good = _identity_basis()
    PSCAgent._psc_validate_basis(good, n_act=N_ACT, run_space="normalized")

    wrong_dim = _identity_basis(n_act=7)
    with pytest.raises(ValueError, match="action_dim"):
        PSCAgent._psc_validate_basis(wrong_dim, n_act=N_ACT, run_space="normalized")

    wrong_space = _identity_basis(space="env")
    with pytest.raises(ValueError, match="space"):
        PSCAgent._psc_validate_basis(wrong_space, n_act=N_ACT, run_space="normalized")

    non_ortho = _identity_basis()
    non_ortho["U"] = torch.eye(N_ACT) * 1.5
    with pytest.raises(ValueError, match="orthonormal"):
        PSCAgent._psc_validate_basis(non_ortho, n_act=N_ACT, run_space="normalized")

    missing = {k: v for k, v in _identity_basis().items() if k != "eigvals"}
    with pytest.raises(ValueError, match="missing"):
        PSCAgent._psc_validate_basis(missing, n_act=N_ACT, run_space="normalized")


def test_block_sizes_must_sum_to_action_dim():
    actor, qnet, qnet_target = _build_nets(seed=2)
    agent = object.__new__(PSCAgent)
    agent.device = "cpu"
    agent.actor = actor
    agent.config = _shared_config(block_sizes=(3, 3, 2))  # sums to 8 != 9
    with pytest.raises(ValueError, match="sum to action_dim"):
        agent._psc_init_from_basis(_identity_basis())


# --- 5. identity-basis equivalence to BF-CQL ---------------------------------


def test_identity_basis_reproduces_bf_cql_loss():
    """U=I, mu=0, blocks == a contiguous coordinate grouping, m*s_i == 1
    -> PSC critic step must equal BF-CQL's with that fixed grouping."""
    actor_a, qnet_a, qtar_a = _build_nets(seed=0)
    actor_b, qnet_b, qtar_b = _build_nets(seed=0)
    bf = _skeleton(BFCQLAgent, actor_a, qnet_a, qtar_a)
    # eig = 0.25 -> sqrt = 0.5 (uniform across dims -> floor no-op); m = 2.0 -> m*s = 1.0
    # which matches BF-CQL's uniform(-1,1) * action_scale(=1) rand sampling exactly.
    psc = _skeleton(PSCAgent, actor_b, qnet_b, qtar_b, basis=_identity_basis(eig=0.25))

    data = _make_batch()
    torch.manual_seed(1234)
    out_bf = bf._update_q(data)
    torch.manual_seed(1234)
    out_psc = psc._update_q(data)

    names = [
        "reward_mean", "q_grad_norm", "q_loss", "q_target_max", "q_target_min",
        "alpha_loss", "conservative_loss", "bellman_loss", "cql_gap", "q_data_mean",
        "q_pi_minus_q_data", "rand_q_mean", "curr_q_mean", "next_q_mean",
        "curr_logp_mean", "next_logp_mean", "random_density_mean",
    ]
    for name, a, b in zip(names, out_bf, out_psc):
        assert torch.allclose(a, b, atol=1e-6, rtol=1e-6), f"{name}: bf={a} psc={b}"

    psc_block_gaps = out_psc[-1]
    assert psc_block_gaps.shape == (len(BLOCK_SIZES),)
    assert torch.isfinite(psc_block_gaps).all()


def test_rotated_basis_changes_conservative_loss():
    actor, qnet, qnet_target = _build_nets(seed=0)
    psc_id = _skeleton(PSCAgent, actor, qnet, qnet_target, basis=_identity_basis(eig=0.25))
    data = _make_batch()
    torch.manual_seed(9)
    out_id = psc_id._update_q(data)

    actor2, qnet2, qtar2 = _build_nets(seed=0)
    psc_rot = _skeleton(PSCAgent, actor2, qnet2, qtar2, basis=_random_basis())
    torch.manual_seed(9)
    out_rot = psc_rot._update_q(data)

    assert torch.isfinite(out_rot[2])
    assert not torch.allclose(out_id[6], out_rot[6]), "rotated basis must alter the conservative term"


# --- 6. checkpoint basis save/load --------------------------------------------


def test_checkpoint_basis_round_trip(tmp_path):
    agent = _psc_agent_with_random_basis()
    payload = {
        "U": agent._psc_U.detach().cpu(),
        "mu": agent._psc_mu.detach().cpu(),
        "eigvals": agent._psc_eigvals.detach().cpu(),
        "meta": dict(agent._psc_basis_meta),
    }
    ckpt = tmp_path / "psc_ckpt.pt"
    torch.save({"psc_basis": payload}, ckpt)
    loaded = torch.load(ckpt, weights_only=False)["psc_basis"]

    # load path re-validates (asserts action_dim) then re-initializes buffers
    PSCAgent._psc_validate_basis(loaded, n_act=N_ACT, run_space="normalized")
    actor, qnet, qnet_target = _build_nets(seed=3)
    fresh = _skeleton(PSCAgent, actor, qnet, qnet_target, basis=loaded)
    assert torch.equal(fresh._psc_U, agent._psc_U)
    assert torch.equal(fresh._psc_mu, agent._psc_mu)
    assert torch.equal(fresh._psc_eigvals, agent._psc_eigvals)

    with pytest.raises(ValueError, match="action_dim"):
        PSCAgent._psc_validate_basis(loaded, n_act=N_ACT + 1, run_space="normalized")
