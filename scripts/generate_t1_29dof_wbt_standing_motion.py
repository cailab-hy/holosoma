#!/usr/bin/env python3
"""Generate a small, self-contained T1 29-DoF WBT standing reference."""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = REPO_ROOT / "src/holosoma/holosoma/data/robots/t1/t1_29dof.xml"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "src/holosoma/holosoma/data/motions/t1_29dof/whole_body_tracking/t1_29dof_standing_mj.npz"
)

DEFAULT_JOINT_POSITIONS = {
    "AAHead_yaw": 0.0,
    "Head_pitch": 0.0,
    "Left_Shoulder_Pitch": 0.2,
    "Left_Shoulder_Roll": -1.35,
    "Left_Elbow_Pitch": 0.0,
    "Left_Elbow_Yaw": -0.5,
    "Left_Wrist_Pitch": 0.0,
    "Left_Wrist_Yaw": 0.0,
    "Left_Hand_Roll": 0.0,
    "Right_Shoulder_Pitch": 0.2,
    "Right_Shoulder_Roll": 1.35,
    "Right_Elbow_Pitch": 0.0,
    "Right_Elbow_Yaw": 0.5,
    "Right_Wrist_Pitch": 0.0,
    "Right_Wrist_Yaw": 0.0,
    "Right_Hand_Roll": 0.0,
    "Waist": 0.0,
    "Left_Hip_Pitch": -0.2,
    "Left_Hip_Roll": 0.0,
    "Left_Hip_Yaw": 0.0,
    "Left_Knee_Pitch": 0.4,
    "Left_Ankle_Pitch": -0.25,
    "Left_Ankle_Roll": 0.0,
    "Right_Hip_Pitch": -0.2,
    "Right_Hip_Roll": 0.0,
    "Right_Hip_Yaw": 0.0,
    "Right_Knee_Pitch": 0.4,
    "Right_Ankle_Pitch": -0.25,
    "Right_Ankle_Roll": 0.0,
}


def generate(model_path: Path, output_path: Path, fps: int, duration_s: float) -> None:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0

    for joint_name, position in DEFAULT_JOINT_POSITIONS.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise KeyError(f"Joint '{joint_name}' is missing from {model_path}")
        data.qpos[model.jnt_qposadr[joint_id]] = position

    mujoco.mj_forward(model, data)
    joint_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        for joint_id in range(model.njnt)
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE
    ]
    body_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        for body_id in range(model.nbody)
    ]
    if len(joint_names) != 29:
        raise ValueError(f"Expected 29 actuated joints, found {len(joint_names)}")

    frame_count = max(2, round(fps * duration_s) + 1)

    def repeat(value: np.ndarray) -> np.ndarray:
        return np.repeat(np.asarray(value)[None, ...], frame_count, axis=0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        fps=np.asarray(fps),
        joint_pos=repeat(data.qpos),
        joint_vel=np.zeros((frame_count, model.nv), dtype=np.float64),
        body_pos_w=repeat(data.xpos),
        body_quat_w=repeat(data.xquat),
        body_lin_vel_w=np.zeros((frame_count, model.nbody, 3), dtype=np.float64),
        body_ang_vel_w=np.zeros((frame_count, model.nbody, 3), dtype=np.float64),
        joint_names=np.asarray(joint_names),
        body_names=np.asarray(body_names),
    )
    print(f"saved: {output_path}")
    print(f"frames={frame_count} fps={fps} joints={len(joint_names)} bodies={len(body_names)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--duration-s", type=float, default=2.0)
    args = parser.parse_args()
    generate(args.model, args.out, args.fps, args.duration_s)


if __name__ == "__main__":
    main()
