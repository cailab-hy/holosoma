"""Tests for the holosoma -> cyclo_lab (ai_sapiens_sim2real) bundle exporter."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
import pytest
import torch
import yaml

from holosoma.config_values.experiment import DEFAULTS
from holosoma.utils import cyclo_sim2real_export as ex

REPO_ROOT = Path(__file__).resolve().parents[1]
DANCE1_NPZ = REPO_ROOT / "src/holosoma/holosoma/data/motions/k1_23dof/whole_body_tracking/dance1_mj.npz"
EXP = DEFAULTS["k1_23dof_wbt_lafan_dance1"]


def test_sim2real_config_matches_k1_training_setup():
    cfg = ex.build_sim2real_config(EXP)
    assert cfg["policy_joints"] == list(EXP.robot.dof_names)
    assert cfg["step_dt"] == pytest.approx(0.02)
    names = [t["name"] for t in ex.observation_layout(EXP)]
    assert names == [
        "motion_command",
        "motion_anchor_ori_b",
        "base_ang_vel",
        "joint_pos_rel",
        "joint_vel_rel",
        "last_action",
    ]
    assert sum(t["dim"] for t in ex.observation_layout(EXP)) == 124
    assert list(cfg["observations"]) == names
    hip = cfg["joint_properties"]["left_hip_pitch_joint"]
    assert hip["default_position"] == pytest.approx(-0.3)
    assert hip["stiffness"] == pytest.approx(76.451, abs=1e-2)
    # scale = 0.25 * effort / kp, offset = default (cyclo action pipeline: offset + scale * a)
    assert cfg["actions"]["joint_pos"]["scale"][0] == pytest.approx(0.25 * 96.864 / 76.451, abs=1e-3)
    assert cfg["actions"]["joint_pos"]["scale"][5] == pytest.approx(0.25 * 47.277 / 22.301, abs=1e-3)
    assert cfg["actions"]["joint_pos"]["offset"] == [
        pytest.approx(EXP.robot.init_state.default_joint_angles[j]) for j in EXP.robot.dof_names
    ]


def test_sim2real_yaml_round_trip(tmp_path: Path):
    path = tmp_path / "sim2real.yaml"
    ex.dump_sim2real_yaml(ex.build_sim2real_config(EXP), path)
    text = path.read_text()
    assert "scale: [" in text  # flow-style lists like cyclo's exporter
    loaded = yaml.safe_load(text)
    assert loaded["policy_joints"][12] == "waist_yaw_joint"
    assert loaded["observations"]["motion_command"]["params"] == {"command_name": "reference_trajectory"}
    assert len(loaded["observations"]["motion_command"]["scale"]) == 46
    assert loaded["actions"]["joint_pos"]["clip"] is None


def test_motion_csv_matches_runtime_format(tmp_path: Path):
    motion = ex.reorder_joints(ex.load_motion_npz(DANCE1_NPZ), list(EXP.robot.dof_names))
    assert motion.fps == 50.0
    defaults = np.array([EXP.robot.init_state.default_joint_angles[j] for j in EXP.robot.dof_names])
    clip_frames = motion.joint_pos.shape[0]
    motion = ex.default_pose_prepend(motion, defaults, 0.764, 2.0)
    assert motion.joint_pos.shape[0] == clip_frames + 100
    np.testing.assert_allclose(motion.joint_pos[0], defaults)
    assert motion.root_pos[0, 2] == pytest.approx(0.764)
    np.testing.assert_allclose(np.linalg.norm(motion.root_quat_xyzw, axis=1), 1.0, atol=1e-6)

    csv = tmp_path / "dance1.csv"
    ex.write_motion_csv(motion, csv)
    lines = csv.read_text().splitlines()
    header = lines[0].split(",")
    assert header[:7] == ex.ROOT_CSV_COLUMNS and header[7:] == list(EXP.robot.dof_names)
    rows = np.loadtxt(csv, delimiter=",", skiprows=1)
    assert rows.shape == (clip_frames + 100, 7 + 23)
    # root quaternion stored as xyzw (runtime builds Eigen::Quaternionf(w=col6, x=col3, y=col4, z=col5))
    np.testing.assert_allclose(np.linalg.norm(rows[:, 3:7], axis=1), 1.0, atol=1e-6)


class _BundledPolicy(torch.nn.Module):
    """Stand-in for holosoma's bundled WBT export (obs + time_step -> 5 outputs)."""

    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.actor = torch.nn.Linear(obs_dim, act_dim)
        self.register_buffer("joint_pos", torch.randn(10, act_dim))

    def forward(self, obs, time_step):
        idx = torch.clamp(time_step.long().squeeze(-1), max=9)
        return self.actor(obs), self.joint_pos[idx], self.joint_pos[idx], obs[:, :3], obs[:, :4]


def _write_bundled_onnx(path: Path, obs_dim: int, act_dim: int) -> None:
    torch.onnx.export(
        _BundledPolicy(obs_dim, act_dim),
        (torch.zeros(1, obs_dim), torch.zeros(1, 1)),
        str(path),
        input_names=["obs", "time_step"],
        output_names=["actions", "joint_pos", "joint_vel", "ref_pos_xyz", "ref_quat_xyzw"],
        dynamo=False,
    )


def test_extract_policy_onnx_keeps_only_actor(tmp_path: Path):
    src = tmp_path / "model.onnx"
    _write_bundled_onnx(src, 124, 23)
    obs_dim, act_dim = ex.extract_policy_onnx(src, tmp_path / "policy.onnx")
    assert (obs_dim, act_dim) == (124, 23)
    policy = onnx.load(str(tmp_path / "policy.onnx"))
    assert [i.name for i in policy.graph.input] == ["obs"]
    assert [o.name for o in policy.graph.output] == ["actions"]
    assert ex.run_policy_once(tmp_path / "policy.onnx", 124).shape == (1, 23)


def test_export_cyclo_bundle_end_to_end(tmp_path: Path):
    src = tmp_path / "model.onnx"
    _write_bundled_onnx(src, 124, 23)
    manifest = ex.export_cyclo_bundle("k1_23dof_wbt_lafan_dance1", EXP, src, tmp_path / "bundle", "dance1")
    assert (tmp_path / "bundle/exported/policy.onnx").is_file()
    assert (tmp_path / "bundle/params/sim2real.yaml").is_file()
    assert (tmp_path / "bundle/params/dance1.csv").is_file()
    loaded = json.loads((tmp_path / "bundle/manifest.json").read_text())
    assert loaded == manifest
    assert manifest["obs_dim"] == 124 and manifest["action_dim"] == 23
    assert manifest["prepend_s"] == 2.0 and manifest["motion_frames"] == manifest["clip_frames"] + 100
    assert manifest["anchor_body"] == ["torso_link"]


def test_export_rejects_obs_dim_mismatch(tmp_path: Path):
    src = tmp_path / "model.onnx"
    _write_bundled_onnx(src, 100, 23)
    with pytest.raises(ValueError, match="observation values"):
        ex.export_cyclo_bundle("k1_23dof_wbt_lafan_dance1", EXP, src, tmp_path / "bundle", "dance1")
