#!/usr/bin/env python3
"""Build AI Sapiens K1 Rev.1 (23-DoF) robot assets for holosoma and holosoma_retargeting.

The generated files are committed to the repository; this script documents their
provenance and lets them be regenerated when the upstream robot description changes.

Sources (defaults assume ``cyclo_lab`` and ``omni-k1`` checkouts next to ``holosoma``):

* URDF   ``cyclo_lab/third_party/ai_sapiens/ai_sapiens_description/urdf/k1_rev1/k1.urdf``
* MJCF   ``cyclo_lab/third_party/ai_sapiens/ai_sapiens_description/mujoco/k1/k1.xml``
* meshes ``cyclo_lab/third_party/ai_sapiens/ai_sapiens_description/meshes/k1_rev1/*.stl``
* retargeting MJCF (proxy keypoints) ``omni-k1/assets/robots/k1_rev1/k1_retarget.xml``

Outputs:

* ``src/holosoma/holosoma/data/robots/k1/{k1_23dof.urdf, k1_23dof.xml, meshes/}``
  Training assets. Adds ``{left,right}_foot_contact_point`` bodies (foot height
  computation, same convention as G1/T1), names actuators after their joints (holosoma's
  MuJoCo backend looks actuators up by joint name), and removes the free-joint name.
* ``src/holosoma_retargeting/holosoma_retargeting/models/k1/{k1_23dof.urdf, k1_23dof.xml,
  k1_23dof_w_largebox.xml, meshes/}``
  Retargeting assets. Reproduces the staging omni-k1 performs at run time: foot
  sticking markers, hand TCP bodies, hand collision geoms and a ground plane.

Usage::

    python scripts/build_k1_assets.py            # uses default source paths
    python scripts/build_k1_assets.py --cyclo-root ~/cyclo_lab --omni-k1-root ~/omni-k1
"""

from __future__ import annotations

import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = REPO_ROOT / "src/holosoma/holosoma/data/robots/k1"
RETARGET_DIR = REPO_ROOT / "src/holosoma_retargeting/holosoma_retargeting/models/k1"

PACKAGE_MESH_PREFIX = "package://ai_sapiens_description/meshes/k1_rev1/"

# Bottom of the foot collision spheres (z=-0.06, radius 0.005) in the ankle-roll frame.
FOOT_CONTACT_POINT_Z = -0.065

# Foot sticking markers used by omni-k1 (ankle-roll frame). The last one is the toe tip.
FOOT_MARKERS = (
    (-0.047, 0.025, -0.060),
    (-0.047, -0.025, -0.060),
    (0.112, 0.027, -0.060),
    (0.112, -0.027, -0.060),
    (0.125, 0.0, -0.047),
)

# Compact rubber hand (wrist-roll frame): a shaft from x=0.03 to x=0.15 (radius 28 mm) ending
# in a 30 mm ball whose centre is the tool-center-point (same values as omni-k1 ``K1_HAND_TCP``).
HAND_SHAFT_FROMTO = (0.03, 0.0, 0.0, 0.15, 0.0, 0.0)
HAND_SHAFT_RADIUS = 0.028
HAND_BALL_RADIUS = 0.03
HAND_TCP = {
    "left": (0.179, 0.0072, 0.0005),
    "right": (0.179, -0.0072, 0.0005),
}


def _fmt(values) -> str:
    return " ".join(f"{v:g}" for v in values)


def _write_xml(tree: ET.ElementTree, path: Path) -> None:
    ET.indent(tree, space="  ")
    path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(path, encoding="utf-8", xml_declaration=True)
    print(f"wrote {path.relative_to(REPO_ROOT)}")


def _rewrite_urdf_mesh_paths(root: ET.Element) -> None:
    for mesh in root.iter("mesh"):
        filename = mesh.get("filename", "")
        if filename.startswith(PACKAGE_MESH_PREFIX):
            mesh.set("filename", "meshes/" + filename[len(PACKAGE_MESH_PREFIX) :])
        elif "/" in filename:
            mesh.set("filename", "meshes/" + filename.rsplit("/", 1)[-1])


def _contact_point_link(name: str) -> ET.Element:
    link = ET.Element("link", name=name)
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "mass", value="0.001")
    ET.SubElement(inertial, "inertia", ixx="0.000001", ixy="0", ixz="0", iyy="0.000001", iyz="0", izz="0.000001")
    # Invisible visual: IsaacLab's USD converter expects a visual prim on every link.
    visual = ET.SubElement(link, "visual")
    ET.SubElement(visual, "origin", xyz="0 0 0", rpy="0 0 0")
    geometry = ET.SubElement(visual, "geometry")
    ET.SubElement(geometry, "box", size="0.001 0.001 0.001")
    material = ET.SubElement(visual, "material", name="invisible")
    ET.SubElement(material, "color", rgba="0 0 0 0")
    return link


def _contact_point_joint(name: str, parent: str) -> ET.Element:
    joint = ET.Element("joint", name=name, type="fixed", dont_collapse="true")
    ET.SubElement(joint, "parent", link=parent)
    ET.SubElement(joint, "child", link=name)
    ET.SubElement(joint, "origin", xyz=f"0 0 {FOOT_CONTACT_POINT_Z}", rpy="0 0 0")
    return joint


def build_training_urdf(src: Path, dst: Path) -> None:
    tree = ET.parse(src)  # noqa: S314
    root = tree.getroot()
    root.set("name", "k1_23dof")
    _rewrite_urdf_mesh_paths(root)

    children = list(root)
    for side in ("left", "right"):
        ankle_joint = root.find(f"joint[@name='{side}_ankle_roll_joint']")
        if ankle_joint is None:
            raise ValueError(f"{src}: missing {side}_ankle_roll_joint")
        name = f"{side}_foot_contact_point"
        # Insert right after the ankle joint so the body order (DFS over the URDF tree)
        # matches RobotConfig.body_names: ..., ankle_roll_link, foot_contact_point, ...
        index = children.index(ankle_joint) + 1
        root.insert(index, _contact_point_link(name))
        root.insert(index + 1, _contact_point_joint(name, f"{side}_ankle_roll_link"))
        children = list(root)

    _write_xml(tree, dst)


def _find_free_joint(worldbody: ET.Element) -> ET.Element:
    free_joint = worldbody.find(".//freejoint")
    if free_joint is None:
        free_joint = worldbody.find(".//joint[@type='free']")
    if free_joint is None:
        raise ValueError("MJCF has no floating-base free joint")
    return free_joint


def _fused_inertial(src: Path, body_name: str) -> dict[str, str]:
    """Return the inertial attributes of ``body_name`` after MuJoCo fuses its static children."""
    tree = ET.parse(src)  # noqa: S314
    compiler = tree.getroot().find("compiler")
    compiler.set("meshdir", str((src.parent / compiler.get("meshdir", ".")).resolve()))
    compiler.set("fusestatic", "true")
    model = mujoco.MjModel.from_xml_string(ET.tostring(tree.getroot(), encoding="unicode"))
    body = model.body(body_name)
    return {
        "pos": _fmt(body.ipos),
        "quat": _fmt(body.iquat),
        "mass": f"{float(body.mass[0]):.9g}",
        "diaginertia": " ".join(f"{v:.9g}" for v in body.inertia),
    }


def _fuse_head_into_torso(src: Path, worldbody: ET.Element) -> None:
    """Merge the fixed ``head_link`` body into ``torso_link``.

    IsaacGym/IsaacSim collapse the fixed head joint of the URDF into the torso, so the
    MuJoCo model must expose the same body set for RobotConfig.body_names to be valid
    across simulators.
    """
    torso = worldbody.find(".//body[@name='torso_link']")
    head = torso.find("body[@name='head_link']")
    if head is None:
        return
    if head.get("quat", "1 0 0 0").split() != ["1", "0", "0", "0"] or head.get("euler") is not None:
        raise ValueError("head_link is expected to be axis-aligned with torso_link")
    head_pos = [float(v) for v in head.get("pos", "0 0 0").split()]
    for geom in head.findall("geom"):
        geom_pos = [float(v) for v in geom.get("pos", "0 0 0").split()]
        geom.set("pos", _fmt(a + b for a, b in zip(head_pos, geom_pos)))
        torso.append(geom)
    torso.remove(head)

    inertial = torso.find("inertial")
    for key in ("fullinertia", "diaginertia", "quat", "pos", "mass"):
        inertial.attrib.pop(key, None)
    inertial.attrib.update(_fused_inertial(src, "torso_link"))


def build_training_mjcf(src: Path, dst: Path) -> None:
    tree = ET.parse(src)  # noqa: S314
    root = tree.getroot()
    root.set("model", "k1_23dof")
    root.find("compiler").set("meshdir", "meshes/")
    worldbody = root.find("worldbody")

    # holosoma's G1/T1 models use an unnamed free joint; keep the same convention.
    _find_free_joint(worldbody).attrib.pop("name", None)

    _fuse_head_into_torso(src, worldbody)

    pelvis = worldbody.find("body[@name='pelvis']")
    if pelvis.find("site[@name='imu']") is None:
        ET.SubElement(pelvis, "site", name="imu", size="0.01", pos="0 0 0")

    for side in ("left", "right"):
        ankle = worldbody.find(f".//body[@name='{side}_ankle_roll_link']")
        contact = ET.SubElement(ankle, "body", name=f"{side}_foot_contact_point", pos=f"0 0 {FOOT_CONTACT_POINT_Z}")
        ET.SubElement(contact, "inertial", pos="0 0 0", mass="0.001", diaginertia="0.000001 0.000001 0.000001")

    # holosoma's MuJoCo backend resolves actuators by joint name.
    for motor in root.find("actuator"):
        motor.set("name", motor.get("joint"))

    _write_xml(tree, dst)


def _add_marker(parent: ET.Element, name: str, xyz) -> None:
    body = ET.SubElement(parent, "body", name=name, pos=_fmt(xyz))
    ET.SubElement(
        body,
        "geom",
        type="sphere",
        size="0.005",
        contype="0",
        conaffinity="0",
        group="3",
        rgba="0.2 0.8 0.2 0.35",
        mass="0.0001",
    )


def build_retarget_mjcf(src: Path, dst: Path, dst_with_box: Path) -> None:
    tree = ET.parse(src)  # noqa: S314
    root = tree.getroot()
    root.set("model", "k1_23dof")
    root.find("compiler").set("meshdir", "meshes")
    worldbody = root.find("worldbody")

    # holosoma_retargeting builds its actuated-joint limit vector from *named* joints
    # and assumes the free joint is unnamed (as in its G1 model).
    _find_free_joint(worldbody).attrib.pop("name", None)

    ET.SubElement(
        worldbody, "geom", name="ground", type="plane", size="10 10 0.1", pos="0 0 0", contype="1", conaffinity="1"
    )

    for side in ("left", "right"):
        ankle = worldbody.find(f".//body[@name='{side}_ankle_roll_link']")
        for index, xyz in enumerate(FOOT_MARKERS, start=1):
            _add_marker(ankle, f"{side}_foot_contact_{index}", xyz)
        _add_marker(ankle, f"{side}_foot_front", FOOT_MARKERS[-1])

        hand = worldbody.find(f".//body[@name='{side}_wrist_roll_rubber_hand']")
        if hand is None:
            raise ValueError(f"{src}: missing {side}_wrist_roll_rubber_hand")
        tcp = hand.find(f"body[@name='{side}_hand_tcp']")
        if tcp is None:
            tcp = ET.SubElement(hand, "body", name=f"{side}_hand_tcp")
        tcp.set("pos", _fmt(HAND_TCP[side]))
        if tcp.find("geom") is None:
            ET.SubElement(
                tcp,
                "geom",
                type="sphere",
                size="0.005",
                contype="0",
                conaffinity="0",
                group="3",
                rgba="0.2 0.8 0.2 0.35",
                mass="0.0001",
            )
        has_collision = any(geom.get("contype", "1") != "0" for geom in hand.findall("geom"))
        if not has_collision:
            # Hand-object non-penetration uses these colliders; primitives match the
            # training model and avoid convex-hull artefacts of the (open) STL.
            common = {"contype": "1", "conaffinity": "1", "group": "3", "rgba": "0 0 0 0", "mass": "0.0001"}
            ET.SubElement(
                hand,
                "geom",
                name=f"{side}_hand_shaft_collision",
                type="capsule",
                fromto=_fmt(HAND_SHAFT_FROMTO),
                size=f"{HAND_SHAFT_RADIUS:g}",
                **common,
            )
            ET.SubElement(
                hand,
                "geom",
                name=f"{side}_hand_ball_collision",
                type="sphere",
                pos=_fmt(HAND_TCP[side]),
                size=f"{HAND_BALL_RADIUS:g}",
                **common,
            )

    _write_xml(tree, dst)

    # Scene variant with the shared largebox asset (same layout as models/g1/g1_29dof_w_largebox.xml).
    ET.SubElement(root.find("asset"), "mesh", name="largebox_mesh", file="../../largebox/largebox.obj", scale="1 1 1")
    box = ET.SubElement(worldbody, "body", name="largebox_link", pos="0 0 0")
    ET.SubElement(box, "freejoint")
    ET.SubElement(box, "inertial", pos="0 0 0", mass="0.1", diaginertia="0.002 0.002 0.002")
    ET.SubElement(
        box,
        "geom",
        name="largebox",
        type="mesh",
        mesh="largebox_mesh",
        friction="0.9 0.005 0.0001",
        rgba="0.65 0.45 0.25 0.65",
        contype="1",
        conaffinity="1",
    )
    _write_xml(tree, dst_with_box)


def build_retarget_urdf(src: Path, dst: Path) -> None:
    tree = ET.parse(src)  # noqa: S314
    root = tree.getroot()
    root.set("name", "k1_23dof")
    _rewrite_urdf_mesh_paths(root)
    _write_xml(tree, dst)


def copy_meshes(src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for mesh in sorted(src_dir.glob("*.stl")):
        target = dst_dir / mesh.name
        if not target.exists() or target.stat().st_size != mesh.stat().st_size:
            shutil.copy2(mesh, target)
        count += 1
    print(f"copied {count} meshes -> {dst_dir.relative_to(REPO_ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cyclo-root", type=Path, default=REPO_ROOT.parent / "cyclo_lab")
    parser.add_argument("--omni-k1-root", type=Path, default=REPO_ROOT.parent / "omni-k1")
    parser.add_argument("--k1-urdf", type=Path, default=None)
    parser.add_argument("--k1-mjcf", type=Path, default=None)
    parser.add_argument("--k1-meshes", type=Path, default=None)
    parser.add_argument("--k1-retarget-mjcf", type=Path, default=None)
    args = parser.parse_args()

    description = args.cyclo_root / "third_party/ai_sapiens/ai_sapiens_description"
    k1_urdf = args.k1_urdf or description / "urdf/k1_rev1/k1.urdf"
    k1_mjcf = args.k1_mjcf or description / "mujoco/k1/k1.xml"
    k1_meshes = args.k1_meshes or description / "meshes/k1_rev1"
    k1_retarget_mjcf = args.k1_retarget_mjcf or args.omni_k1_root / "assets/robots/k1_rev1/k1_retarget.xml"

    for path in (k1_urdf, k1_mjcf, k1_meshes, k1_retarget_mjcf):
        if not path.exists():
            raise FileNotFoundError(path)

    build_training_urdf(k1_urdf, TRAINING_DIR / "k1_23dof.urdf")
    build_training_mjcf(k1_mjcf, TRAINING_DIR / "k1_23dof.xml")
    copy_meshes(k1_meshes, TRAINING_DIR / "meshes")

    build_retarget_urdf(k1_urdf, RETARGET_DIR / "k1_23dof.urdf")
    build_retarget_mjcf(k1_retarget_mjcf, RETARGET_DIR / "k1_23dof.xml", RETARGET_DIR / "k1_23dof_w_largebox.xml")
    copy_meshes(k1_meshes, RETARGET_DIR / "meshes")


if __name__ == "__main__":
    main()
