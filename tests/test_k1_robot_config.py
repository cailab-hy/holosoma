"""Consistency checks for the AI Sapiens K1 Rev.1 robot config and assets."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from holosoma.config_values.robot import DEFAULTS, k1_23dof

REPO_ROOT = Path(__file__).resolve().parents[1]
ROBOT_DIR = REPO_ROOT / "src/holosoma/holosoma/data/robots/k1"
RETARGET_DIR = REPO_ROOT / "src/holosoma_retargeting/holosoma_retargeting/models/k1"


def _urdf_joints(urdf: Path) -> list[ET.Element]:
    return ET.parse(urdf).getroot().findall("joint")  # noqa: S314


def _urdf_body_order(urdf: Path) -> list[str]:
    """Depth-first link order with fixed joints collapsed unless dont_collapse is set."""
    root = ET.parse(urdf).getroot()  # noqa: S314
    joints = root.findall("joint")
    children: dict[str, list[tuple[ET.Element, str]]] = {}
    child_links = set()
    for joint in joints:
        parent = joint.find("parent").get("link")
        child = joint.find("child").get("link")
        children.setdefault(parent, []).append((joint, child))
        child_links.add(child)
    roots = [link.get("name") for link in root.findall("link") if link.get("name") not in child_links]
    assert len(roots) == 1, roots

    order: list[str] = []

    def visit(link: str, keep: bool) -> None:
        if keep:
            order.append(link)
        for joint, child in children.get(link, []):
            collapsed = joint.get("type") == "fixed" and joint.get("dont_collapse", "false") != "true"
            visit(child, not collapsed)

    visit(roots[0], True)
    return order


def test_k1_registered_in_defaults():
    assert DEFAULTS["k1_23dof"] is k1_23dof
    assert k1_23dof.asset.robot_type == "k1_23dof"


def test_k1_dof_lists_have_23_entries():
    n = 23
    assert k1_23dof.dof_obs_size == n
    assert k1_23dof.actions_dim == n
    assert len(k1_23dof.dof_names) == n
    for name in (
        "dof_pos_lower_limit_list",
        "dof_pos_upper_limit_list",
        "dof_vel_limit_list",
        "dof_effort_limit_list",
        "dof_armature_list",
        "dof_joint_friction_list",
    ):
        assert len(getattr(k1_23dof, name)) == n, name
    assert set(k1_23dof.upper_dof_names) | set(k1_23dof.lower_dof_names) == set(k1_23dof.dof_names)
    assert not set(k1_23dof.upper_dof_names) & set(k1_23dof.lower_dof_names)
    assert set(k1_23dof.symmetry_joint_names) == set(k1_23dof.dof_names)
    assert set(k1_23dof.flip_sign_joint_names) <= set(k1_23dof.dof_names)
    assert set(k1_23dof.init_state.default_joint_angles) == set(k1_23dof.dof_names)


def test_k1_limits_match_urdf():
    joints = {j.get("name"): j for j in _urdf_joints(ROBOT_DIR / "k1_23dof.urdf") if j.get("type") == "revolute"}
    assert list(joints) == k1_23dof.dof_names
    for i, name in enumerate(k1_23dof.dof_names):
        limit = joints[name].find("limit")
        assert math.isclose(float(limit.get("lower")), k1_23dof.dof_pos_lower_limit_list[i], abs_tol=1e-4), name
        assert math.isclose(float(limit.get("upper")), k1_23dof.dof_pos_upper_limit_list[i], abs_tol=1e-4), name
        assert math.isclose(float(limit.get("effort")), k1_23dof.dof_effort_limit_list[i], abs_tol=0.1), name
        assert math.isclose(float(limit.get("velocity")), k1_23dof.dof_vel_limit_list[i], abs_tol=0.1), name


def test_k1_body_names_match_urdf_tree():
    assert _urdf_body_order(ROBOT_DIR / "k1_23dof.urdf") == k1_23dof.body_names
    assert k1_23dof.num_bodies == len(k1_23dof.body_names)
    for key_body in k1_23dof.key_bodies:
        assert key_body in k1_23dof.body_names
    assert k1_23dof.torso_name in k1_23dof.body_names
    assert sum(k1_23dof.foot_body_name in b for b in k1_23dof.body_names) == k1_23dof.num_feet
    assert sum(k1_23dof.foot_height_name in b for b in k1_23dof.body_names) == k1_23dof.num_feet


def test_k1_pd_gain_keys_match_each_dof_once():
    for dof in k1_23dof.dof_names:
        short = dof.replace("_joint", "")
        matches = [key for key in k1_23dof.control.stiffness if key in short]
        assert len(matches) == 1, (dof, matches)
        assert matches[0] in k1_23dof.control.damping


def test_k1_action_scale_matches_cyclo_formula():
    # cyclo_lab: scale = 0.25 * effort_limit / stiffness per joint
    assert k1_23dof.control.action_scales_by_effort_limit_over_p_gain
    assert k1_23dof.control.action_scale == 0.25


@pytest.mark.parametrize("xml_name", ["k1_23dof.xml"])
def test_k1_mujoco_training_model_matches_config(xml_name):
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_path(str(ROBOT_DIR / xml_name))
    bodies = [model.body(i).name for i in range(model.nbody)]
    assert bodies[0] == "world"
    assert bodies[1:] == k1_23dof.body_names
    joints = [model.joint(i).name for i in range(model.njnt) if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE]
    assert joints == k1_23dof.dof_names
    actuators = [model.actuator(i).name for i in range(model.nu)]
    assert actuators == k1_23dof.dof_names


def test_k1_retargeting_models_load():
    mujoco = pytest.importorskip("mujoco")
    for name in ("k1_23dof.xml", "k1_23dof_w_largebox.xml"):
        model = mujoco.MjModel.from_xml_path(str(RETARGET_DIR / name))
        names = {model.body(i).name for i in range(model.nbody)}
        for side in ("left", "right"):
            assert f"{side}_hand_tcp" in names
            assert f"{side}_foot_front" in names
            assert {f"{side}_foot_contact_{i}" for i in range(1, 6)} <= names
        joints = [model.joint(i).name for i in range(model.njnt) if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE]
        assert joints == k1_23dof.dof_names
    assert model.nq == 7 + 23 + 7  # largebox variant
