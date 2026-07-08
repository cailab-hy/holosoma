"""Unit tests for the standalone SYNC-QL package (agents/sync_cql).

Covers the drift-gating math that works without a simulator, plus a guard that
the package stays decoupled from agents/bf_cql.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from holosoma.agents.sync_cql.sync_cql import GROUP_PRESETS, resolve_action_groups
from holosoma.agents.sync_cql.sync_cql_agent import SyncCQLAgent, build_group_to_action_mask

SYNC_PKG = Path(__file__).resolve().parents[1] / "src/holosoma/holosoma/agents/sync_cql"


def _minimal_agent(
    *,
    mode: str = "topk",
    k: int = 2,
    delta: float = 0.5,
    drift_ema: float = 0.0,
    momentum: float = 0.999,
    freeze: bool = False,
) -> SyncCQLAgent:
    agent = object.__new__(SyncCQLAgent)
    agent.device = "cpu"
    agent.bf_cql_group_indices = [(0,), (1,), (2, 3)]
    agent.bf_cql_group_names = ["g0", "g1", "g23"]
    agent.config = SimpleNamespace(
        sync_cql=SimpleNamespace(
            K=k,
            delta_threshold=delta,
            selection_mode=mode,
            drift_mode="rmse",
            drift_ema=drift_ema,
            drift_std_momentum=momentum,
            freeze_drift_stats=freeze,
        )
    )
    agent.sync_group_to_action_mask = build_group_to_action_mask(agent.bf_cql_group_indices, 4, device="cpu")
    agent.sync_action_std = torch.ones(4)
    agent.sync_group_drift_ema = torch.zeros(len(agent.bf_cql_group_indices))
    return agent


def test_package_has_no_bf_cql_dependency():
    for py_file in SYNC_PKG.glob("*.py"):
        source = py_file.read_text()
        assert "from holosoma.agents.bf_cql" not in source, f"{py_file.name} imports from agents/bf_cql"
        assert "import holosoma.agents.bf_cql" not in source, f"{py_file.name} imports from agents/bf_cql"


def test_group_to_action_mask_shape_and_content():
    mask = build_group_to_action_mask([(0,), (1, 2), (3,)], 4, device="cpu")
    assert mask.shape == (3, 4)
    assert mask.dtype == torch.bool
    expected = torch.tensor(
        [[True, False, False, False], [False, True, True, False], [False, False, False, True]]
    )
    assert torch.equal(mask, expected)


def test_resolve_action_groups_presets_cover_all_dims():
    dof_names = [name for _, joints in GROUP_PRESETS["coarse_5"] for name in joints]
    names, indices = resolve_action_groups("coarse_5", dof_names)
    assert names == [group for group, _ in GROUP_PRESETS["coarse_5"]]
    assert sorted(i for grp in indices for i in grp) == list(range(len(dof_names)))


def test_group_drift_zero_when_actor_matches_dataset():
    agent = _minimal_agent()
    actions = torch.randn(6, 4)
    drift = agent._compute_group_drift(actions, actions)
    assert drift.shape == (6, 3)
    assert drift.max().item() < 1e-5


def test_group_drift_is_normalized_rmse_per_group():
    agent = _minimal_agent()
    agent.sync_action_std = torch.tensor([2.0, 1.0, 1.0, 1.0])
    dataset = torch.zeros(1, 4)
    actor = torch.tensor([[2.0, 1.0, 3.0, 4.0]])
    drift = agent._compute_group_drift(dataset, actor)
    # g0: |2/2| = 1 ; g1: |1/1| = 1 ; g23: sqrt((9+16)/2)
    assert drift[0, 0].item() == pytest.approx(1.0, abs=1e-4)
    assert drift[0, 1].item() == pytest.approx(1.0, abs=1e-4)
    assert drift[0, 2].item() == pytest.approx(((9 + 16) / 2) ** 0.5, abs=1e-3)


def test_group_drift_ema_blends_batch_and_history():
    agent = _minimal_agent(drift_ema=0.5)
    agent.sync_group_drift_ema = torch.tensor([4.0, 4.0, 4.0])
    dataset = torch.zeros(2, 4)
    actor = torch.zeros(2, 4)
    drift = agent._compute_group_drift(dataset, actor)
    # batch drift ~0; ema buffer updates to 0.5*4 + 0.5*0 = 2; blended = 0.5*0 + 0.5*2 = 1
    assert torch.allclose(drift, torch.full((2, 3), 1.0), atol=1e-3)


def test_action_std_update_momentum_and_freeze():
    agent = _minimal_agent(momentum=0.9)
    actions = torch.tensor([[0.0, 0.0, 0.0, 0.0], [2.0, 4.0, 6.0, 8.0]])
    batch_std = actions.std(dim=0, unbiased=False).clamp_min(1e-3)
    agent._update_sync_action_std(actions)
    assert torch.allclose(agent.sync_action_std, 0.9 * torch.ones(4) + 0.1 * batch_std, atol=1e-5)

    frozen = _minimal_agent(freeze=True)
    before = frozen.sync_action_std.clone()
    frozen._update_sync_action_std(actions)
    assert torch.equal(frozen.sync_action_std, before)


def test_selected_subset_hash_is_injective_over_masks():
    agent = _minimal_agent()
    masks = torch.tensor(
        [
            [False, False, False],
            [True, False, False],
            [False, True, False],
            [True, True, False],
            [False, False, True],
            [True, True, True],
        ]
    )
    hashes = agent._selected_subset_hash(masks)
    assert hashes.tolist() == [0, 1, 2, 3, 4, 7]
    assert len(set(hashes.tolist())) == masks.shape[0]


def test_sync_disabled_modes():
    assert _minimal_agent(mode="none")._sync_disabled()
    assert _minimal_agent(k=0)._sync_disabled()
    assert not _minimal_agent(mode="topk", k=2)._sync_disabled()


def test_density_drift_mode_placeholder_is_explicit():
    agent = _minimal_agent()
    agent.config.sync_cql.drift_mode = "density"
    with pytest.raises(NotImplementedError, match="CVAE"):
        agent._compute_group_drift(torch.zeros(2, 4), torch.ones(2, 4))


def test_subset_composition_summary_names_groups():
    agent = _minimal_agent()
    mask = torch.tensor([[True, False, True], [True, False, True], [False, False, False]])
    summary = agent._selected_subset_composition_summary(mask)
    assert "g0+g23:0.667" in summary
    assert "empty:0.333" in summary
