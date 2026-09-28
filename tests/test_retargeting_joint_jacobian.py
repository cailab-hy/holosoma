from pathlib import Path

import mujoco
import numpy as np

from holosoma_retargeting.config_types.data_type import MotionDataConfig
from holosoma_retargeting.config_types.robot import RobotConfig
from holosoma_retargeting.config_types.task import TaskConfig
from holosoma_retargeting.examples.robot_retarget import create_task_constants
from holosoma_retargeting.src.interaction_mesh_retargeter import InteractionMeshRetargeter


def test_qdot_to_qvel_transform_keeps_all_g1_actuated_joints(monkeypatch) -> None:
    retargeting_root = Path(__file__).parents[1] / "src" / "holosoma_retargeting" / "holosoma_retargeting"
    monkeypatch.chdir(retargeting_root)

    constants = create_task_constants(
        RobotConfig(robot_type="g1"),
        MotionDataConfig(data_format="lafan", robot_type="g1"),
        TaskConfig(object_name="ground"),
        "robot_only",
    )
    retargeter = InteractionMeshRetargeter(
        constants,
        object_urdf_path=None,
        activate_foot_sticking=False,
        activate_obj_non_penetration=False,
    )
    retargeter.robot_data.qpos[3] = 1.0

    transform = retargeter._build_transform_qdot_to_qvel_fast()
    actuated_entries = []
    for joint_id in range(1, retargeter.robot_model.njnt):
        joint_type = int(retargeter.robot_model.jnt_type[joint_id])
        if joint_type in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
            qpos_idx = retargeter.robot_model.jnt_qposadr[joint_id]
            dof_idx = retargeter.robot_model.jnt_dofadr[joint_id]
            actuated_entries.append(transform[dof_idx, qpos_idx])

    np.testing.assert_array_equal(actuated_entries, np.ones(constants.ROBOT_DOF))
