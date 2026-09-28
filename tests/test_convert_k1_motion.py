"""End-to-end tests for the K1 motion converter (requires MuJoCo)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")

REPO_ROOT = Path(__file__).resolve().parents[1]
RETARGETING_SRC = REPO_ROOT / "src/holosoma_retargeting"
if str(RETARGETING_SRC) not in sys.path:
    sys.path.insert(0, str(RETARGETING_SRC))

from holosoma_retargeting.config_types.data_conversion import K1_JOINT_NAMES  # noqa: E402
from holosoma_retargeting.data_conversion.convert_k1_motion import (  # noqa: E402
    K1MotionConversionConfig,
    convert,
    load_source,
)

STANDING = {
    "left_hip_pitch_joint": -0.3,
    "left_knee_joint": 0.63,
    "left_ankle_pitch_joint": -0.33,
    "right_hip_pitch_joint": -0.3,
    "right_knee_joint": 0.63,
    "right_ankle_pitch_joint": -0.33,
}


def _qpos_rows(frames: int, with_object: bool) -> np.ndarray:
    joints = np.array([STANDING.get(name, 0.0) for name in K1_JOINT_NAMES])
    rows = []
    for t in range(frames):
        root = [0.01 * t, 0.0, 0.764, 1.0, 0.0, 0.0, 0.0]  # wxyz
        row = root + (joints + 0.001 * t).tolist()
        if with_object:
            row += [0.4, 0.0, 0.14 + 0.002 * t, 1.0, 0.0, 0.0, 0.0]
        rows.append(row)
    return np.asarray(rows)


def _check_output(path: Path, with_object: bool) -> None:
    with np.load(path) as data:
        assert [str(n) for n in data["joint_names"]] == list(K1_JOINT_NAMES)
        body_names = [str(n) for n in data["body_names"]]
        assert "left_foot_contact_point" in body_names
        frames = data["joint_pos"].shape[0]
        assert data["joint_pos"].shape == (frames, 7 + 23)
        assert data["joint_vel"].shape == (frames, 6 + 23)
        assert data["body_pos_w"].shape == (frames, len(body_names), 3)
        assert data["body_quat_w"].shape == (frames, len(body_names), 4)
        assert float(np.asarray(data["fps"]).reshape(-1)[0]) == 50.0
        pelvis_z = data["body_pos_w"][:, body_names.index("pelvis"), 2]
        assert np.allclose(pelvis_z, 0.764, atol=1e-5)
        foot_z = data["body_pos_w"][:, body_names.index("left_foot_contact_point"), 2]
        assert foot_z.min() > -0.05 and foot_z.max() < 0.1  # feet near the ground
        assert ("object_pos_w" in data) == with_object
        if with_object:
            assert data["object_pos_w"].shape == (frames, 3)
            assert data["object_quat_w"].shape == (frames, 4)
    assert path.with_suffix(".json").is_file()


def test_convert_cyclo_csv_xyzw(tmp_path: Path):
    qpos = _qpos_rows(6, with_object=False)
    csv_rows = qpos.copy()
    csv_rows[:, 3:7] = qpos[:, [4, 5, 6, 3]]  # store as xyzw like cyclo_lab CSVs
    csv = tmp_path / "stand.csv"
    np.savetxt(csv, csv_rows, delimiter=",", fmt="%.9f")

    source = load_source(K1MotionConversionConfig(input=str(csv)))
    assert source.fps == 50.0 and not source.has_object
    assert np.allclose(source.qpos[:, 3:7], qpos[:, 3:7])

    out = convert(
        K1MotionConversionConfig(
            input=str(csv), output=str(tmp_path / "stand_mj.npz"), scratch_dir=str(tmp_path / "scratch")
        )
    )
    _check_output(out, with_object=False)


def test_convert_omni_k1_bundle_with_object(tmp_path: Path):
    bundle = tmp_path / "carry_k1"
    bundle.mkdir()
    qpos = _qpos_rows(5, with_object=True)
    # Shuffle the joint order to make sure joint_names are honoured.
    perm = np.random.default_rng(0).permutation(23)
    shuffled = qpos.copy()
    shuffled[:, 7:30] = qpos[:, 7:30][:, perm]
    np.savez(
        bundle / "motion_raw.npz",
        qpos=shuffled,
        fps=np.asarray([30]),
        joint_names=np.asarray([K1_JOINT_NAMES[i] for i in perm]),
        quat_order=np.asarray("wxyz"),
        object_size=np.asarray([0.35, 0.30, 0.28]),
    )

    source = load_source(K1MotionConversionConfig(input=str(bundle)))
    assert source.has_object and source.fps == 30.0 and source.box_size == (0.35, 0.30, 0.28)
    assert np.allclose(source.qpos, qpos)

    out = convert(
        K1MotionConversionConfig(
            input=str(bundle), output=str(tmp_path / "carry_mj_w_obj.npz"), scratch_dir=str(tmp_path / "scratch")
        )
    )
    _check_output(out, with_object=True)


def test_rejects_bad_width(tmp_path: Path):
    csv = tmp_path / "bad.csv"
    np.savetxt(csv, np.zeros((3, 12)), delimiter=",")
    with pytest.raises(ValueError, match="qpos must be"):
        load_source(K1MotionConversionConfig(input=str(csv)))
