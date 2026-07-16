from __future__ import annotations

from types import SimpleNamespace

import torch

from holosoma.config_values.wbt.g1.experiment import g1_29dof_wbt_fast_sac_data
from holosoma.config_values.wbt.g1.reward import (
    g1_29dof_wbt_fast_sac_reward,
    g1_29dof_wbt_fast_sac_reward_offline_collect,
)
from holosoma.config_values.wbt.g1.termination import (
    g1_29dof_wbt_termination,
    g1_29dof_wbt_termination_offline_collect,
)
from holosoma.managers.reward.terms import wbt as reward_terms


STRICT_THRESHOLDS = {
    "bad_ref_pos_threshold": 0.5,
    "bad_ref_ori_threshold": 0.8,
    "bad_motion_body_pos_threshold": 0.25,
    "bad_object_pos_threshold": 0.25,
    "bad_object_ori_threshold": 0.8,
}


def test_offline_collect_preserves_every_base_reward_term() -> None:
    base_terms = g1_29dof_wbt_fast_sac_reward.terms
    collect_terms = g1_29dof_wbt_fast_sac_reward_offline_collect.terms

    assert set(base_terms).issubset(collect_terms)
    for name, term in base_terms.items():
        assert collect_terms[name] == term


def test_offline_collect_adds_only_strict_excess_penalties() -> None:
    base_names = set(g1_29dof_wbt_fast_sac_reward.terms)
    collect_terms = g1_29dof_wbt_fast_sac_reward_offline_collect.terms
    extra_terms = {name: term for name, term in collect_terms.items() if name not in base_names}

    assert set(extra_terms) == {
        "recovery_ref_position_strict_excess",
        "recovery_ref_orientation_strict_excess",
        "recovery_body_position_strict_excess",
    }
    assert extra_terms["recovery_ref_position_strict_excess"].params["threshold"] == 0.5
    assert extra_terms["recovery_ref_orientation_strict_excess"].params["threshold"] == 0.8
    assert extra_terms["recovery_body_position_strict_excess"].params["threshold"] == 0.25
    assert all(term.weight < 0.0 for term in extra_terms.values())


def test_offline_collect_termination_is_relaxed_not_strict() -> None:
    strict = g1_29dof_wbt_termination.terms["bad_tracking"].params
    relaxed = g1_29dof_wbt_termination_offline_collect.terms["bad_tracking"].params

    for key, strict_threshold in STRICT_THRESHOLDS.items():
        assert strict[key] == strict_threshold
        assert relaxed[key] == 2.0 * strict_threshold
    assert relaxed["body_names_to_track"] == strict["body_names_to_track"]
    assert relaxed["bad_motion_body_pos_body_names"] == strict["bad_motion_body_pos_body_names"]


def test_fast_sac_data_experiment_uses_recovery_collection_configs() -> None:
    assert g1_29dof_wbt_fast_sac_data.termination == g1_29dof_wbt_termination_offline_collect
    assert g1_29dof_wbt_fast_sac_data.reward == g1_29dof_wbt_fast_sac_reward_offline_collect


def test_strict_excess_hinges_are_zero_inside_strict_region(monkeypatch) -> None:
    motion_command = SimpleNamespace(
        ref_pos_w=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        robot_ref_pos_w=torch.tensor([[0.3, 0.4, 0.0], [0.6, 0.0, 0.0]]),
        ref_quat_w=torch.zeros(2, 4),
        robot_ref_quat_w=torch.ones(2, 4),
        body_pos_relative_w=torch.zeros(2, 4, 3),
        robot_body_pos_w=torch.tensor(
            [
                [[0.25, 0.0, 0.0]] * 4,
                [[0.30, 0.0, 0.0]] * 4,
            ]
        ),
        motion_cfg=SimpleNamespace(
            body_names_to_track=[
                "left_ankle_roll_link",
                "right_ankle_roll_link",
                "left_wrist_yaw_link",
                "right_wrist_yaw_link",
            ]
        ),
    )
    env = SimpleNamespace(num_envs=2, device="cpu")
    monkeypatch.setattr(reward_terms, "_get_motion_command_and_assert_type", lambda _: motion_command)
    monkeypatch.setattr(reward_terms, "gravity_vector", lambda _: torch.zeros(2, 3))
    projected_gravity_results = iter(
        [
            torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            torch.tensor([[0.0, 0.0, 0.8], [0.0, 0.0, 0.9]]),
        ]
    )
    monkeypatch.setattr(reward_terms, "quat_rotate_inverse", lambda *args, **kwargs: next(projected_gravity_results))

    root_penalty = reward_terms.motion_global_ref_position_error_hinge(env, threshold=0.5)
    orientation_penalty = reward_terms.motion_global_ref_projected_gravity_error_hinge(env, threshold=0.8)
    body_penalty = reward_terms.motion_relative_body_position_error_hinge(
        env,
        threshold=0.25,
        body_names=motion_command.motion_cfg.body_names_to_track,
        aggregation="max",
    )

    torch.testing.assert_close(root_penalty, torch.tensor([0.0, 0.1]))
    torch.testing.assert_close(orientation_penalty, torch.tensor([0.0, 0.1]))
    torch.testing.assert_close(body_penalty, torch.tensor([0.0, 0.05]))
