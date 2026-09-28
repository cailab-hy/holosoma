from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from scripts.replay_motion_collisions import load_scene, lowest_point


@pytest.mark.parametrize(
    ("kind", "size", "angle", "expected_z"),
    [
        ("sphere", [0.2], 37, 0.8),
        ("cylinder", [0.2, 0.6], 0, 0.7),
        ("cylinder", [0.2, 0.6], 90, 0.8),
        ("cylinder", [0.2, 0.6], 45, 1 - 0.5 / np.sqrt(2)),
        ("capsule", [0.2, 0.6], 0, 0.5),
        ("capsule", [0.2, 0.6], 90, 0.8),
        ("capsule", [0.2, 0.6], 45, 0.8 - 0.3 / np.sqrt(2)),
        ("box", [0.4, 0.2, 0.6], 0, 0.7),
        ("box", [0.4, 0.2, 0.6], 90, 0.8),
    ],
)
def test_lowest_point_of_rotated_primitives(kind, size, angle, expected_z):
    rotation = Rotation.from_euler("y", [angle], degrees=True).as_matrix()
    position = np.array([[2.0, -3.0, 1.0]])
    point = lowest_point(kind, np.array(size), rotation, position)
    np.testing.assert_allclose(point[0, 2], expected_z, atol=1e-12)


def test_fk_uses_named_joints_root_quaternion_and_collision_origin(tmp_path):
    urdf = tmp_path / "robot.urdf"
    urdf.write_text(
        '<robot name="test"><link name="root"/><link name="arm">'
        '<collision><origin xyz="1 0 0"/><geometry><sphere radius="0.1"/></geometry></collision>'
        '</link><joint name="hinge" type="revolute"><parent link="root"/>'
        '<child link="arm"/><origin xyz="0 0 0.5"/><axis xyz="0 1 0"/></joint></robot>'
    )
    path = tmp_path / "motion.npz"
    # Extra motion joint comes first; the FK must resolve hinge by name.
    # Root rotates 180 degrees about y: arm origin lies below root, while the
    # hinge's pi/2 rotation sends its collision center back up by 1m.
    qpos = np.array([[4, 5, 2, 0, 0, 1, 0, 99, np.pi / 2]] * 2, dtype=float)
    np.savez(path, joint_pos=qpos, joint_names=["unused", "hinge"], fps=[50])
    loaded_q, names, fps, colliders = load_scene(path, urdf)
    np.testing.assert_array_equal(loaded_q, qpos)
    assert names == ["unused", "hinge"] and fps == 50
    np.testing.assert_allclose(colliders[0].position, [[4, 5, 2.5]] * 2, atol=1e-12)
    np.testing.assert_allclose(colliders[0].lowest[:, 2], 2.4, atol=1e-12)
