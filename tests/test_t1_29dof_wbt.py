from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
RETARGETING_SRC = REPO_ROOT / "src/holosoma_retargeting"
if str(RETARGETING_SRC) not in sys.path:
    sys.path.insert(0, str(RETARGETING_SRC))

from holosoma.config_values.experiment import DEFAULTS  # noqa: E402
from holosoma.config_values.robot import t1_29dof_waist_wrist  # noqa: E402
from holosoma.managers.command.terms.wbt import MotionLoader  # noqa: E402
from holosoma_retargeting.config_types.data_type import MotionDataConfig  # noqa: E402
from holosoma_retargeting.config_types.robot import RobotConfig  # noqa: E402
from holosoma_retargeting.data_conversion.convert_data_format_mj import (  # noqa: E402
    MotionLoader as ConversionMotionLoader,
)


def test_t1_29dof_retarget_model_matches_simulator_joint_convention():
    config = RobotConfig(robot_type="t1_29dof")
    model_path = RETARGETING_SRC / "holosoma_retargeting" / config.ROBOT_URDF_FILE.replace(".urdf", ".xml")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    model_joint_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        for joint_id in range(model.njnt)
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE
    ]

    assert config.ROBOT_DOF == 29
    assert model.nq == 36
    assert model_joint_names == t1_29dof_waist_wrist.dof_names
    assert RobotConfig(robot_type="t1").ROBOT_DOF == 23


def test_t1_29dof_all_retarget_mappings_exist_in_model():
    config = RobotConfig(robot_type="t1_29dof")
    model_path = RETARGETING_SRC / "holosoma_retargeting" / config.ROBOT_URDF_FILE.replace(".urdf", ".xml")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    body_names = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        for body_id in range(model.nbody)
    }

    for data_format in ("lafan", "smplh", "smplx", "mocap"):
        mapping = MotionDataConfig(data_format=data_format, robot_type="t1_29dof").resolved_joints_mapping
        assert set(mapping.values()) <= body_names
    assert len(config.FOOT_STICKING_LINKS) == 10
    assert set(config.FOOT_STICKING_LINKS) <= body_names
    assert model.body("left_foot_sphere_5_link").pos[0] > 0.1
    assert model.body("right_foot_sphere_5_link").pos[0] > 0.1
    assert MotionDataConfig(data_format="lafan", robot_type="t1_29dof").default_scale_factor == 0.626
    assert config.MANUAL_COST == {
        "13": 0.2,
        "14": 0.2,
        "15": 0.2,
        "20": 0.2,
        "21": 0.2,
        "22": 0.2,
        "29": 5.0,
        "35": 5.0,
    }


def test_t1_29dof_standing_motion_is_wbt_compatible():
    motion_path = (
        REPO_ROOT
        / "src/holosoma/holosoma/data/motions/t1_29dof/whole_body_tracking/t1_29dof_standing_mj.npz"
    )
    with np.load(motion_path) as motion:
        assert motion["joint_names"].tolist() == t1_29dof_waist_wrist.dof_names
        assert motion["joint_pos"].shape[1] == 36
        assert motion["joint_vel"].shape[1] == 35

    loader = MotionLoader(
        "holosoma/data/motions/t1_29dof/whole_body_tracking/t1_29dof_standing_mj.npz",
        t1_29dof_waist_wrist.body_names,
        t1_29dof_waist_wrist.dof_names,
        "cpu",
    )
    assert loader.joint_pos.shape[1] == 29
    assert loader.body_pos_w.shape[1] == len(t1_29dof_waist_wrist.body_names)


def test_motion_loader_prefers_exact_t1_contact_body_before_g1_alias():
    loader = object.__new__(MotionLoader)

    exact = loader._get_index_of_a_in_b(
        ["left_foot_contact_point"],
        ["left_foot_contact_point", "left_ankle_roll_link"],
    )
    fallback = loader._get_index_of_a_in_b(
        ["left_foot_contact_point"],
        ["left_ankle_roll_link"],
    )

    assert exact.tolist() == [0]
    assert fallback.tolist() == [0]


def test_t1_29dof_conversion_loader_uses_model_dof_and_embedded_joint_names(tmp_path):
    raw_motion_path = tmp_path / "t1_raw.npz"
    qpos = np.zeros((3, 36), dtype=np.float32)
    qpos[:, 3] = 1.0
    np.savez(
        raw_motion_path,
        qpos=qpos,
        fps=np.asarray(50),
        joint_names=np.asarray(t1_29dof_waist_wrist.dof_names),
    )

    loader = ConversionMotionLoader(
        motion_file=str(raw_motion_path),
        input_fps=30,
        output_fps=50,
        device="cpu",
        line_range=None,
        has_dynamic_object=False,
        use_omniretarget_data=False,
        robot_dof=29,
    )

    assert loader.input_fps == 50
    assert loader.motion_dof_poss_input.shape == (3, 29)
    assert loader.joint_names == t1_29dof_waist_wrist.dof_names


def test_t1_29dof_wbt_experiments_are_registered_consistently():
    for experiment_name in (
        "t1_29dof_wbt",
        "t1_29dof_wbt_fast_sac",
        "t1_29dof_wbt_fast_sac_episode_data",
    ):
        experiment = DEFAULTS[experiment_name]
        assert experiment.robot.dof_names == t1_29dof_waist_wrist.dof_names
        assert experiment.robot.actions_dim == 29
        assert experiment.robot.asset.enable_self_collisions is False
        assert experiment.training.num_envs == 4096
        assert experiment.command.setup_terms["motion_command"].params["motion_config"].body_name_ref == ["Trunk"]

    motion_config = DEFAULTS["t1_29dof_wbt"].command.setup_terms["motion_command"].params["motion_config"]
    assert motion_config.ankle_body_names == ["left_foot_link", "right_foot_link"]
    assert motion_config.wrist_body_names == ["left_hand_link", "right_hand_link"]
    assert set(motion_config.ankle_body_names + motion_config.wrist_body_names) <= set(
        motion_config.body_names_to_track
    )


def test_t1_29dof_cropped_dance_experiments_are_independently_registered():
    experiment_names = (
        "t1_29dof_wbt_dance",
        "t1_29dof_wbt_dance_fast_sac",
        "t1_29dof_wbt_dance_fast_sac_episode_data",
    )
    for experiment_name in experiment_names:
        experiment = DEFAULTS[experiment_name]
        motion = experiment.command.setup_terms["motion_command"].params["motion_config"]
        assert experiment.training.num_envs == 4096
        assert experiment.robot.actions_dim == 29
        assert experiment.robot.asset.enable_self_collisions is False
        assert motion.motion_file.endswith("dance2_short_mj.npz")
        assert motion.start_at_timestep_zero_prob == 1.0
        assert motion.enable_default_pose_prepend is False
        assert motion.enable_default_pose_append is False
        assert experiment.simulator.config.sim.max_episode_length_s == 5.0

    assert (
        DEFAULTS["t1_29dof_wbt_dance_fast_sac"].algo.config.offline_dataset_path
        == "offline_data/t1_29dof_wbt_dance_fastsac_dataset.h5"
    )
    assert (
        DEFAULTS["t1_29dof_wbt_dance_fast_sac_episode_data"].algo.config.offline_dataset_path
        == "offline_data/t1_29dof_wbt_dance_fastsac_episode_dataset.h5"
    )

    dance_path = (
        REPO_ROOT
        / "src/holosoma/holosoma/data/motions/t1_29dof/whole_body_tracking/dance2_short_mj.npz"
    )
    with np.load(dance_path) as dance:
        assert dance["joint_names"].tolist() == t1_29dof_waist_wrist.dof_names
        assert dance["joint_pos"].shape[1] == 36
        assert dance["joint_vel"].shape[1] == 35
        assert dance["joint_pos"].shape[0] / dance["fps"].item() < 5.0
        wrist_names = (
            "Left_Wrist_Pitch",
            "Left_Wrist_Yaw",
            "Left_Hand_Roll",
            "Right_Wrist_Pitch",
            "Right_Wrist_Yaw",
            "Right_Hand_Roll",
        )
        for wrist_name in wrist_names:
            wrist_index = 7 + dance["joint_names"].tolist().index(wrist_name)
            assert np.abs(dance["joint_pos"][:, wrist_index]).max() < 0.5

    loader = MotionLoader(
        "holosoma/data/motions/t1_29dof/whole_body_tracking/dance2_short_mj.npz",
        t1_29dof_waist_wrist.body_names,
        t1_29dof_waist_wrist.dof_names,
        "cpu",
    )
    assert loader.joint_pos.shape[1] == 29
