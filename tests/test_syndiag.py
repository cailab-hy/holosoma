"""Correctness gates for the BF-CQL synergy-OOD diagnostics (syndiag).

Gate 1  No-op equivalence: with a fixed seed and a fixed synthetic batch, the
        critic/actor loss sequence is bit-identical (atol=0) whether the
        syndiag tick runs between updates or not (including RNG consumption).
Gate 2  Additive toy critic Q(s,a) = sum_g w_g^T a^g  =>  Delta(M) ~ 0.
Gate 3  Interaction toy critic Q(s,a) = a^1 . a^2  =>  Delta({1,2}) != 0 and
        the singleton v's do not explain it.
Gate 4  Shape / abbreviation / dump sanity: deterministic collision-free
        coalition names, drift shape [B, G], npz dump round-trips via np.load.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from holosoma.agents.bf_cql.bf_cql import DoubleQCritic, FactorizedActor
from holosoma.agents.bf_cql.bf_cql_agent import BFCQLAgent
from holosoma.agents.bf_cql.syndiag import (
    abbreviate_group_names,
    build_coalitions,
    coalition_group_mask,
    coalition_q_values,
    compute_group_drift,
    group_dim_mask,
    quartile_delta_stats,
    recall_top_pair,
    singleton_columns,
    superadditivity_quad,
    synergy_residuals,
)
from holosoma.config_types.algo import SynDiagSettings
from holosoma.utils.safe_torch_import import TensorDict

COARSE_5_NAMES = ["left_leg", "right_leg", "waist", "left_arm", "right_arm"]


def _partition(sizes: list[int]) -> list[tuple[int, ...]]:
    indices: list[tuple[int, ...]] = []
    cursor = 0
    for size in sizes:
        indices.append(tuple(range(cursor, cursor + size)))
        cursor += size
    return indices


def _obs_indices(key: str, dim: int) -> dict[str, dict[str, int]]:
    return {key: {"start": 0, "end": dim, "size": dim}}


def _skeleton_agent(
    *,
    group_names: list[str],
    group_indices: list[tuple[int, ...]],
    n_act: int,
    critic: DoubleQCritic,
    cfg: SynDiagSettings,
    log_dir: str,
) -> BFCQLAgent:
    agent = object.__new__(BFCQLAgent)
    agent.device = "cpu"
    agent.log_dir = log_dir
    agent.global_step = 4200
    agent._critic_update_step = 0
    agent.is_main_process = True
    agent.bf_cql_group_names = list(group_names)
    agent.bf_cql_group_indices = [tuple(g) for g in group_indices]
    agent.env_action_scale = torch.ones(n_act)
    agent.env_action_bias = torch.zeros(n_act)
    agent.normalized_action_training = True
    agent.qnet = critic
    agent._offline_dataset_path = Path("offline_data/fake_dataset.h5")
    agent.config = SimpleNamespace(amp=False, amp_dtype="bf16", syndiag=cfg)
    agent.env = SimpleNamespace(robot_config=SimpleNamespace(actions_dim=n_act))
    agent._syndiag_cfg = cfg
    agent._syndiag_enabled = cfg.enabled
    agent._syndiag_setup()
    return agent


def _make_batch(batch_size: int, obs_dim: int, critic_obs_dim: int, n_act: int) -> TensorDict:
    data = TensorDict(
        {
            "observations": torch.randn(batch_size, obs_dim),
            "actions": torch.randn(batch_size, n_act).clamp(-1.0, 1.0),
            "critic_observations": torch.randn(batch_size, critic_obs_dim),
        },
        batch_size=batch_size,
        device="cpu",
    )
    data["dataset_index"] = torch.arange(batch_size, dtype=torch.long)
    data["syndiag_raw_observations"] = data["observations"].clone()
    return data


# ---------------------------------------------------------------------------
# Gate 1: no-op equivalence (losses and RNG stream untouched by the tick)
# ---------------------------------------------------------------------------


def _run_training_losses(*, tick_enabled: bool, num_steps: int = 4) -> list[tuple[float, float]]:
    obs_dim, critic_obs_dim = 12, 16
    group_indices = _partition([2, 2, 1])
    group_names = ["g_a", "g_b", "g_c"]
    n_act = 5
    batch_size = 32

    torch.manual_seed(1234)
    actor = FactorizedActor(
        obs_indices=_obs_indices("actor_obs", obs_dim),
        obs_keys=["actor_obs"],
        n_act=n_act,
        num_envs=1,
        hidden_dim=32,
        log_std_max=0.0,
        log_std_min=-5.0,
        use_tanh=True,
        use_layer_norm=True,
        device="cpu",
        action_scale=torch.ones(n_act),
        action_bias=torch.zeros(n_act),
        action_group_indices=group_indices,
        action_group_names=group_names,
    )
    critic = DoubleQCritic(
        obs_indices=_obs_indices("critic_obs", critic_obs_dim),
        obs_keys=["critic_obs"],
        n_act=n_act,
        hidden_dim=32,
        use_layer_norm=True,
        device="cpu",
    )
    q_optimizer = torch.optim.AdamW(critic.parameters(), lr=1e-3)
    actor_optimizer = torch.optim.AdamW(actor.parameters(), lr=1e-3)

    cfg = SynDiagSettings(enabled=True, interval=1, dump_interval=0)
    agent = _skeleton_agent(
        group_names=group_names,
        group_indices=group_indices,
        n_act=n_act,
        critic=critic,
        cfg=cfg,
        log_dir="/tmp/syndiag-noop-test",
    )
    if not tick_enabled:
        agent._syndiag_enabled = False

    torch.manual_seed(999)
    data = _make_batch(batch_size, obs_dim, critic_obs_dim, n_act)
    rewards = torch.randn(batch_size)

    losses: list[tuple[float, float]] = []
    for _ in range(num_steps):
        q1, q2 = critic(data["critic_observations"], data["actions"])
        q_loss = torch.nn.functional.mse_loss(q1, rewards) + torch.nn.functional.mse_loss(q2, rewards)
        q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        q_optimizer.step()

        # rsample below consumes the default RNG stream, so any hidden RNG
        # draw inside the syndiag tick would desynchronize the two variants.
        pi_actions, log_probs = actor.get_actions_and_log_probs(data["observations"])
        q1_pi, q2_pi = critic(data["critic_observations"], pi_actions)
        actor_loss = (0.01 * log_probs - torch.minimum(q1_pi, q2_pi)).mean()
        actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_optimizer.step()

        agent._critic_update_step += 1
        with torch.no_grad():
            q_data_min = torch.minimum(q1, q2).detach()
            pi_det = actor(data["observations"])[0].detach()
        stats = agent._syndiag_maybe_tick(data, q_data_min, pi_det)
        if tick_enabled:
            assert stats, "syndiag tick was expected to run every update in this test"
        else:
            assert stats == {}

        losses.append((q_loss.item(), actor_loss.item()))
    return losses


def test_noop_equivalence_losses_bit_identical():
    losses_with = _run_training_losses(tick_enabled=True)
    losses_without = _run_training_losses(tick_enabled=False)
    for (q_a, a_a), (q_b, a_b) in zip(losses_with, losses_without, strict=True):
        assert q_a == q_b  # atol=0: bit-identical
        assert a_a == a_b


# ---------------------------------------------------------------------------
# Gate 2: additive toy critic => Delta(M) ~ 0 for all coalitions
# ---------------------------------------------------------------------------


def test_additive_critic_has_zero_synergy_residuals():
    torch.manual_seed(0)
    group_indices = _partition([2, 2, 1])
    names = ["g_a", "g_b", "g_c"]
    n_act, batch_size = 5, 64

    coalitions = build_coalitions(names, max_coalitions=32)
    dim_masks = (
        coalition_group_mask(coalitions, len(names), "cpu").float()
        @ group_dim_mask(group_indices, n_act, "cpu").float()
    ) > 0.5

    weights = torch.randn(n_act)

    def critic_fn(_obs: torch.Tensor, actions: torch.Tensor):
        q = actions @ weights
        return q, q

    a_data = torch.randn(batch_size, n_act)
    a_pi = torch.randn(batch_size, n_act)
    obs = torch.zeros(batch_size, 1)

    q_cf = coalition_q_values(critic_fn, obs, a_pi, a_data, dim_masks)
    v = q_cf - (a_data @ weights)[:, None]
    delta = synergy_residuals(
        v,
        coalition_group_mask(coalitions, len(names), "cpu"),
        singleton_columns(coalitions, len(names)),
    )
    assert delta.abs().max().item() < 1e-5


# ---------------------------------------------------------------------------
# Gate 3: interaction toy critic => Delta({1,2}) != 0, singletons don't explain it
# ---------------------------------------------------------------------------


def test_interaction_critic_pair_residual_is_nonzero_and_exact():
    torch.manual_seed(1)
    group_indices = _partition([2, 2])
    names = ["g_a", "g_b"]
    n_act, batch_size = 4, 64

    coalitions = build_coalitions(names, max_coalitions=32)
    group_masks = coalition_group_mask(coalitions, len(names), "cpu")
    dim_masks = (group_masks.float() @ group_dim_mask(group_indices, n_act, "cpu").float()) > 0.5

    def critic_fn(_obs: torch.Tensor, actions: torch.Tensor):
        q = (actions[:, 0:2] * actions[:, 2:4]).sum(dim=-1)
        return q, q

    a_data = torch.randn(batch_size, n_act)
    a_pi = torch.randn(batch_size, n_act)
    obs = torch.zeros(batch_size, 1)

    q_cf = coalition_q_values(critic_fn, obs, a_pi, a_data, dim_masks)
    q_data, _ = critic_fn(obs, a_data)
    v = q_cf - q_data[:, None]
    delta = synergy_residuals(v, group_masks, singleton_columns(coalitions, len(names)))

    pair_col = next(c for c, coal in enumerate(coalitions) if len(coal.group_ids) == 2)
    expected_pair_delta = ((a_pi[:, 0:2] - a_data[:, 0:2]) * (a_pi[:, 2:4] - a_data[:, 2:4])).sum(dim=-1)

    assert delta[:, pair_col].abs().mean().item() > 1e-2
    assert torch.allclose(delta[:, pair_col], expected_pair_delta, atol=1e-5)
    # singleton residuals are exactly zero by construction
    for c, coal in enumerate(coalitions):
        if len(coal.group_ids) == 1:
            assert delta[:, c].abs().max().item() == 0.0
    # and the singleton v's alone do not reproduce the coalition value
    singleton_sum = v[:, 0] + v[:, 1]
    assert not torch.allclose(v[:, pair_col], singleton_sum, atol=1e-3)


# ---------------------------------------------------------------------------
# Gate 4: names / shapes / dump round-trip
# ---------------------------------------------------------------------------


def test_abbreviations_are_deterministic_and_collision_free():
    assert abbreviate_group_names(COARSE_5_NAMES) == ["LL", "RL", "W", "LA", "RA"]
    assert abbreviate_group_names(COARSE_5_NAMES) == abbreviate_group_names(COARSE_5_NAMES)
    collided = abbreviate_group_names(["waist", "wrist", "left_leg", "lower_lumbar"])
    assert len(set(collided)) == len(collided)


def test_coarse5_coalition_list_contains_named_blocks_and_pairs():
    coalitions = build_coalitions(COARSE_5_NAMES, max_coalitions=32)
    names = [c.name for c in coalitions]
    # 5 singletons + 10 pairs + tri + quad (the two named pairs dedupe into all-pairs)
    assert len(coalitions) == 17
    assert names[:5] == ["sing_LL", "sing_RL", "sing_W", "sing_LA", "sing_RA"]
    assert "pair_LL_RL" in names and "pair_LA_RA" in names
    assert "tri_LL_RL_W" in names
    assert "quad_LL_RL_LA_RA" in names
    # deterministic across calls
    assert names == [c.name for c in build_coalitions(COARSE_5_NAMES, max_coalitions=32)]


def test_named_blocks_skipped_gracefully_for_other_groupings():
    warnings: list[str] = []
    coalitions = build_coalitions(["g_a", "g_b", "g_c"], max_coalitions=32, warn=warnings.append)
    assert len(coalitions) == 3 + 3  # singletons + pairs only
    assert len(warnings) == 1


def test_truncation_keeps_singletons_and_pairs_first():
    names = [f"grp_{i}" for i in range(9)]
    warnings: list[str] = []
    coalitions = build_coalitions(names, max_coalitions=32, warn=warnings.append)
    assert len(coalitions) == 32  # 9 singletons + 36 pairs truncated to the cap
    assert all(len(c.group_ids) == 1 for c in coalitions[:9])
    assert all(len(c.group_ids) == 2 for c in coalitions[9:])
    assert sum("truncating" in message for message in warnings) == 1


def test_drift_shape_and_zero_for_matching_actions():
    group_indices = _partition([3, 2, 4])
    n_act, batch_size = 9, 7
    masks = group_dim_mask(group_indices, n_act, "cpu")
    a = torch.randn(batch_size, n_act)
    sigma = torch.rand(n_act) + 0.5
    drift = compute_group_drift(a, a, sigma, masks)
    assert drift.shape == (batch_size, 3)
    assert drift.abs().max().item() == 0.0
    drift2 = compute_group_drift(a + 1.0, a, sigma, masks)
    assert (drift2 > 0).all()


def test_recall_and_quartile_helpers_basic():
    delta_pairs = torch.tensor([[0.1, 2.0], [3.0, 0.2]])
    pair_group_ids = torch.tensor([[0, 1], [1, 2]])
    drift = torch.tensor([[0.1, 5.0, 4.0], [9.0, 8.0, 0.1]])
    recall, active_frac = recall_top_pair(delta_pairs, pair_group_ids, drift, top_k=2, delta_min=0.0)
    # sample0 top pair = (1,2), drift top2 = {1,2} -> hit; sample1 top pair = (0,1), top2 = {0,1} -> hit
    assert recall is not None and recall.item() == 1.0
    assert active_frac.item() == 1.0

    stats = quartile_delta_stats(torch.linspace(0, 1, 100), torch.linspace(0, 10, 100))
    assert stats is not None
    assert stats["q4"] > stats["q1"]
    assert stats["q4_minus_q1"].item() == pytest.approx((stats["q4"] - stats["q1"]).item())


def test_superadditivity_requires_quad_and_pairs():
    coalitions = build_coalitions(COARSE_5_NAMES, max_coalitions=32)
    delta = torch.zeros(3, len(coalitions))
    quad_col = next(c for c, coal in enumerate(coalitions) if len(coal.group_ids) == 4)
    delta[:, quad_col] = 2.0
    result = superadditivity_quad(delta, coalitions)
    assert result is not None and result.item() == pytest.approx(2.0)
    assert superadditivity_quad(delta[:, :5], coalitions[:5]) is None


def test_dump_round_trips_through_np_load(tmp_path):
    torch.manual_seed(7)
    group_indices = _partition([6, 6, 3, 7, 7])
    n_act = 29
    critic_obs_dim = 24
    batch_size = 16
    critic = DoubleQCritic(
        obs_indices=_obs_indices("critic_obs", critic_obs_dim),
        obs_keys=["critic_obs"],
        n_act=n_act,
        hidden_dim=32,
        use_layer_norm=True,
        device="cpu",
    )
    cfg = SynDiagSettings(enabled=True, interval=1, dump_interval=1, dump_topk=3, dump_max_rows=8)
    agent = _skeleton_agent(
        group_names=COARSE_5_NAMES,
        group_indices=group_indices,
        n_act=n_act,
        critic=critic,
        cfg=cfg,
        log_dir=str(tmp_path),
    )

    data = _make_batch(batch_size, obs_dim=10, critic_obs_dim=critic_obs_dim, n_act=n_act)
    agent._critic_update_step = cfg.interval  # force the tick schedule to fire
    with torch.no_grad():
        q1, q2 = critic(data["critic_observations"], data["actions"])
        q_data_min = torch.minimum(q1, q2)
    pi_actions = torch.randn(batch_size, n_act).clamp(-1.0, 1.0)

    metrics = agent._syndiag_maybe_tick(data, q_data_min, pi_actions)
    assert any(key.startswith("syndiag/drift_") for key in metrics)
    assert any(key.startswith("syndiag/delta_pair_") for key in metrics)

    dump_files = sorted((tmp_path / "syndiag").glob("dump_step*.npz"))
    assert len(dump_files) == 1
    payload = np.load(dump_files[0], allow_pickle=False)

    num_coalitions = len(agent._syndiag_coalitions)
    rows = min(batch_size, cfg.dump_max_rows)
    assert payload["drift"].shape == (rows, 5)
    assert payload["v"].shape == (rows, num_coalitions)
    assert payload["delta"].shape == (rows, num_coalitions)
    assert payload["q_cf"].shape == (rows, num_coalitions)
    assert payload["a_cf_env_top"].shape == (rows, cfg.dump_topk, n_act)
    assert payload["top_coalition_ids"].shape == (rows, cfg.dump_topk)
    assert payload["dataset_index"].tolist() == list(range(rows))
    assert payload["observations_raw"].shape == (rows, 10)
    assert payload["actions_raw"].shape == (rows, n_act)
    assert list(payload["coalition_names"]) == [c.name for c in agent._syndiag_coalitions]
    # v/Delta consistency inside the dump: Delta(M) = v(M) - sum singleton v's
    v_arr, delta_arr = payload["v"], payload["delta"]
    cgm = payload["coalition_group_mask"].astype(np.float32)
    singleton_cols = [int(np.where((cgm[:, g] == 1) & (cgm.sum(axis=1) == 1))[0][0]) for g in range(5)]
    recomputed = v_arr - v_arr[:, singleton_cols] @ cgm.T
    np.testing.assert_allclose(recomputed, delta_arr, atol=1e-5)


def test_failure_disables_after_three_consecutive_errors():
    critic = DoubleQCritic(
        obs_indices=_obs_indices("critic_obs", 8),
        obs_keys=["critic_obs"],
        n_act=5,
        hidden_dim=16,
        use_layer_norm=True,
        device="cpu",
    )
    cfg = SynDiagSettings(enabled=True, interval=1, dump_interval=0)
    agent = _skeleton_agent(
        group_names=["g_a", "g_b", "g_c"],
        group_indices=_partition([2, 2, 1]),
        n_act=5,
        critic=critic,
        cfg=cfg,
        log_dir="/tmp/syndiag-fail-test",
    )
    bad_data = TensorDict({"actions": torch.randn(4, 5)}, batch_size=4, device="cpu")  # no critic_observations
    q_data_min = torch.randn(4)
    pi_actions = torch.randn(4, 5)
    for _ in range(3):
        agent._critic_update_step += 1
        assert agent._syndiag_maybe_tick(bad_data, q_data_min, pi_actions) == {}
    assert agent._syndiag_enabled is False
