#!/usr/bin/env python3
"""Replay a reference against URDF collision primitives and a horizontal plane.

This is kinematic inspection, not a PhysX contact/force simulation. No reset noise,
joint clipping, or root-height correction is applied. Capsule mode assumes the
URDF cylinder length is the capsule's central segment length; it is not a USD
importer verification. Self collisions are not tested.

Run in hsretargeting, from any directory:
    python /home/cai/holosoma/scripts/replay_motion_collisions.py --frame 45
For a dependency-light numeric scan (numpy/scipy only), add --headless.
"""

from __future__ import annotations

import argparse
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MOTION = ROOT / (
    "src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/dance1_subject3_frames_172_412_mj.npz"
)
DEFAULT_URDF = ROOT / "src/holosoma/holosoma/data/robots/g1/g1_29dof.urdf"


@dataclass
class Collider:
    name: str
    kind: str
    size: np.ndarray
    rotation: np.ndarray
    position: np.ndarray
    lowest: np.ndarray


def origin(element: ET.Element) -> tuple[np.ndarray, np.ndarray]:
    node = element.find("origin")
    attributes = {} if node is None else node.attrib
    position = np.fromstring(attributes.get("xyz", "0 0 0"), sep=" ")
    rotation = Rotation.from_euler("xyz", np.fromstring(attributes.get("rpy", "0 0 0"), sep=" ")).as_matrix()
    return rotation, position


def lowest_point(kind, size, rotation, position):
    """Exact minimum-z support point of a transformed primitive (batched)."""
    normal = rotation[:, 2, :]  # World +z expressed in local coordinates.
    if kind == "sphere":
        local = -size[0] * normal
    elif kind == "box":
        local = -np.sign(normal) * size / 2
    elif kind in {"cylinder", "capsule"}:
        radius, length = size
        if kind == "capsule":
            local = -radius * normal
            local[:, 2] -= np.sign(normal[:, 2]) * length / 2
        else:
            local = np.zeros_like(normal)
            norm = np.linalg.norm(normal[:, :2], axis=1, keepdims=True)
            local[:, :2] = -radius * normal[:, :2] / np.maximum(norm, 1e-15)
            local[:, 2] = -np.sign(normal[:, 2]) * length / 2
    else:
        raise ValueError(f"Unsupported collision shape: {kind}")
    return position + np.einsum("nij,nj->ni", rotation, local)


def load_scene(npz_path, urdf_path, cylinder_mode="cylinder"):
    with np.load(npz_path, allow_pickle=False) as data:
        qpos = np.asarray(data["joint_pos"], dtype=np.float64)
        names = data["joint_names"].tolist()
        fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    if qpos.ndim != 2 or qpos.shape != (len(qpos), 7 + len(names)) or len(qpos) < 2:
        raise ValueError("Expected >=2 frames of joint_pos [xyz, quaternion wxyz, named joints].")
    if not np.isfinite(qpos).all() or not np.isfinite(fps) or fps <= 0:
        raise ValueError("Non-finite motion or invalid fps.")
    if len(set(names)) != len(names):
        raise ValueError("Duplicate joint_names in motion.")
    tree = ET.parse(urdf_path).getroot()  # noqa: S314 -- Trusted local robot description.
    joints = list(tree.findall("joint"))
    roots = {link.get("name") for link in tree.findall("link")} - {joint.find("child").get("link") for joint in joints}
    if len(roots) != 1:
        raise ValueError(f"Expected one URDF root, found {roots}.")
    poses = {roots.pop(): (Rotation.from_quat(qpos[:, [4, 5, 6, 3]]).as_matrix(), qpos[:, :3])}
    while joints:
        pending = []
        for joint in joints:
            parent = joint.find("parent").get("link")
            if parent not in poses:
                pending.append(joint)
                continue
            pr, pp = poses[parent]
            jr, jp = origin(joint)
            rotation = pr @ jr
            position = pp + np.einsum("nij,j->ni", pr, jp)
            kind = joint.get("type")
            if kind != "fixed":
                if joint.find("mimic") is not None:
                    raise ValueError("Mimic joints are not supported by this diagnostic.")
                angle = qpos[:, 7 + names.index(joint.get("name"))]
                axis_node = joint.find("axis")
                axis = np.fromstring("1 0 0" if axis_node is None else axis_node.get("xyz"), sep=" ")
                axis = axis / np.linalg.norm(axis)
                if kind in {"revolute", "continuous"}:
                    rotation = rotation @ Rotation.from_rotvec(angle[:, None] * axis).as_matrix()
                elif kind == "prismatic":
                    position = position + np.einsum("nij,nj->ni", rotation, angle[:, None] * axis)
                else:
                    raise ValueError(f"Unsupported joint type: {kind}")
            poses[joint.find("child").get("link")] = rotation, position
        if len(pending) == len(joints):
            raise ValueError("Could not resolve the URDF joint tree.")
        joints = pending

    colliders = []
    for link in tree.findall("link"):
        lr, lp = poses[link.get("name")]
        for index, collision in enumerate(link.findall("collision")):
            cr, cp = origin(collision)
            rotation = lr @ cr
            position = lp + np.einsum("nij,j->ni", lr, cp)
            shape = next(iter(collision.find("geometry")))
            kind = shape.tag
            if kind == "sphere":
                size = np.array([float(shape.get("radius"))])
            elif kind == "cylinder":
                size = np.array([float(shape.get("radius")), float(shape.get("length"))])
                kind = cylinder_mode
            elif kind == "box":
                size = np.fromstring(shape.get("size"), sep=" ")
            else:
                raise ValueError(f"{link.get('name')}: {kind} collider unsupported; use the training primitive URDF.")
            colliders.append(
                Collider(
                    f"{link.get('name')}:{index}",
                    kind,
                    size,
                    rotation,
                    position,
                    lowest_point(kind, size, rotation, position),
                )
            )
    if not colliders:
        raise ValueError("No collision geometry in URDF.")
    return qpos, names, fps, colliders


def primitive_mesh(collider, trimesh):
    if collider.kind == "sphere":
        return trimesh.creation.icosphere(subdivisions=3, radius=collider.size[0])
    if collider.kind == "box":
        return trimesh.creation.box(extents=collider.size)
    if collider.kind == "cylinder":
        return trimesh.creation.cylinder(radius=collider.size[0], height=collider.size[1], sections=48)
    mesh = trimesh.creation.capsule(radius=collider.size[0], height=collider.size[1])
    # trimesh versions differ in whether the capsule is centered at zero.
    mesh.apply_translation(-(mesh.bounds[0] + mesh.bounds[1]) / 2)
    return mesh


def run_viewer(args, qpos, names, fps, colliders, clearances):
    try:
        import trimesh  # noqa: PLC0415 -- Optional viewer dependency; keep --headless lightweight.
        import viser  # noqa: PLC0415
    except ImportError as exc:
        raise SystemExit("Use hsretargeting (requires viser, trimesh). Numeric scan: add --headless.") from exc

    server = viser.ViserServer(host="127.0.0.1", port=args.port)
    server.scene.set_up_direction("+z")
    offset = np.array([qpos[0, 0], qpos[0, 1], 0.0])
    server.scene.add_grid("/grid", width=6, height=6, plane="xy", position=(0, 0, args.ground_z))
    ground = trimesh.creation.box(extents=(6, 6, 0.002))
    server.scene.add_mesh_simple(
        "/ground",
        vertices=ground.vertices,
        faces=ground.faces,
        color=(130, 150, 170),
        opacity=0.2,
        position=(0, 0, args.ground_z - 0.001),
    )
    handles = []
    labels = []
    for i, collider in enumerate(colliders):
        mesh = primitive_mesh(collider, trimesh)
        handles.append(
            server.scene.add_mesh_simple(
                f"/collisions/{i}",
                vertices=mesh.vertices,
                faces=mesh.faces,
                color=(60, 170, 230),
                opacity=0.65,
            )
        )
        labels.append(server.scene.add_label(f"/labels/{i}", text=collider.name, visible=False))
    points = server.scene.add_point_cloud(
        "/lowest_points",
        points=np.zeros((1, 3)),
        colors=np.array([[255, 40, 40]], dtype=np.uint8),
        point_size=0.014,
    )
    depth_lines = server.scene.add_line_segments(
        "/penetration_depth",
        points=np.zeros((1, 2, 3)),
        colors=np.array([[[255, 220, 0], [255, 220, 0]]], dtype=np.uint8),
        line_width=5,
    )
    visual = None
    if args.show_visual:
        import yourdfpy  # noqa: PLC0415 -- Only needed for optional visual meshes.
        from viser.extras import ViserUrdf  # noqa: PLC0415

        visual_root = server.scene.add_frame("/visual", show_axes=False)
        urdf = yourdfpy.URDF.load(str(args.robot_urdf), load_meshes=True, build_scene_graph=True)
        visual = ViserUrdf(server, urdf_or_path=urdf, root_node_name="/visual")
        visual_cols = [7 + names.index(name) for name in visual.get_actuated_joint_limits()]

    with server.gui.add_folder("Playback"):
        playing = server.gui.add_checkbox("Playing", initial_value=False)
        frame = server.gui.add_slider(
            "Frame", min=args.start_frame, max=args.end_frame, step=1, initial_value=args.frame
        )
        speed = server.gui.add_slider("Speed", min=0.05, max=2.0, step=0.05, initial_value=args.speed)
        worst = server.gui.add_button("Jump to deepest penetration")
    with server.gui.add_folder("Display"):
        only_bad = server.gui.add_checkbox("Only penetrating colliders", initial_value=False)
        if visual is not None:
            show_visual = server.gui.add_checkbox("Visual mesh", initial_value=True)
        info = server.gui.add_markdown("")

    @server.on_client_connect
    def connect(client):
        center = qpos[args.frame, :3] - offset
        client.camera.look_at = tuple(center)
        client.camera.position = tuple(center + np.array([1.3, -1.6, 0.7]))

    lock = threading.Lock()

    def update(_event=None):
        with lock, server.atomic():
            f = int(frame.value)
            clearance = clearances[f]
            bad = clearance < -args.tolerance_mm / 1000
            for i, collider in enumerate(colliders):
                handle = handles[i]
                handle.position = collider.position[f] - offset
                handle.wxyz = Rotation.from_matrix(collider.rotation[f]).as_quat()[[3, 0, 1, 2]]
                handle.color = (240, 45, 45) if bad[i] else (60, 170, 230)
                handle.visible = bool(bad[i] or not only_bad.value)
                labels[i].position = collider.lowest[f] - offset + [0, 0, 0.04]
                labels[i].text = f"{collider.name}: {-clearance[i] * 1000:.1f} mm"
                labels[i].visible = bool(bad[i])
            lowest = np.stack([c.lowest[f] for c in colliders]) - offset
            selected = lowest[bad]
            points.visible = depth_lines.visible = bool(bad.any())
            if bad.any():
                points.points = selected
                points.colors = np.tile(np.array([255, 40, 40], dtype=np.uint8), (len(selected), 1))
                surface = selected.copy()
                surface[:, 2] = args.ground_z
                depth_lines.points = np.stack([selected, surface], axis=1)
                depth_lines.colors = np.tile(np.array([255, 220, 0], dtype=np.uint8), (len(selected), 2, 1))
            if visual is not None:
                visual_root.position = qpos[f, :3] - offset
                visual_root.wxyz = qpos[f, 3:7]
                visual.update_cfg(qpos[f, visual_cols])
                visual.show_visual = bool(show_visual.value)
            rows = [f"| {colliders[i].name} | {clearance[i] * 1000:.2f} |" for i in np.argsort(clearance)[:6]]
            info.content = (
                f"Frame **{f}**, source time **{f / fps:.2f}s**; "
                f"minimum clearance **{clearance.min() * 1000:.2f}mm**.\n\n"
                "Red = below plane beyond tolerance. Blue = other colliders. Yellow = penetration depth.\n\n"
                "| Collider | Clearance (mm; negative = below ground) |\n|---|---:|\n"
                + "\n".join(rows)
                + f"\n\nGeometry: **{args.cylinder_mode}**. "
                "Kinematic reference only; no physics, reset noise, or self-collision checks. "
                "Frames/time exclude runtime prepend/append. "
                "Capsule mode assumes central-segment length = URDF cylinder length."
            )

    frame.on_update(update)
    only_bad.on_update(update)
    if visual is not None:
        show_visual.on_update(update)

    @worst.on_click
    def jump(_event):
        playing.value = False
        frame.value = args.start_frame + int(np.argmin(clearances[args.start_frame : args.end_frame + 1].min(axis=1)))

    update()
    print(f"Open http://127.0.0.1:{args.port} (paused). Use Frame, Playing, Speed; Ctrl+C to exit.")
    last = time.monotonic()
    try:
        while True:
            now = time.monotonic()
            if playing.value and now - last >= 1 / (fps * float(speed.value)):
                frame.value = args.start_frame if frame.value >= args.end_frame else frame.value + 1
                last = now
            elif not playing.value:
                last = now
            time.sleep(0.005)
    except KeyboardInterrupt:
        server.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--npz-path", type=Path, default=DEFAULT_MOTION)
    parser.add_argument("--robot-urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--frame", type=int, default=0, help="Initial frame, paused.")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None, help="Last replay frame, inclusive.")
    parser.add_argument("--speed", type=float, default=0.25)
    parser.add_argument("--ground-z", type=float, default=0.0)
    parser.add_argument("--tolerance-mm", type=float, default=1.0, help="Mark red when depth exceeds this threshold.")
    parser.add_argument("--cylinder-mode", choices=("cylinder", "capsule"), default="cylinder")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--show-visual", action="store_true", help="Also load visual meshes (requires yourdfpy).")
    parser.add_argument("--headless", action="store_true", help="Print numeric results without starting a viewer.")
    args = parser.parse_args()
    qpos, names, fps, colliders = load_scene(args.npz_path, args.robot_urdf, args.cylinder_mode)
    args.end_frame = len(qpos) - 1 if args.end_frame is None else args.end_frame
    if not 0 <= args.start_frame <= args.frame <= args.end_frame < len(qpos) or args.start_frame == args.end_frame:
        parser.error("Require 0 <= start-frame <= frame <= end-frame < frame count, with a nonempty playback span.")
    if not 0.05 <= args.speed <= 2 or not np.isfinite(args.tolerance_mm) or args.tolerance_mm < 0:
        parser.error("speed must be 0.05..2.0 and tolerance-mm must be finite and nonnegative.")
    if not np.isfinite(args.ground_z):
        parser.error("ground-z must be finite.")
    clearances = np.stack([c.lowest[:, 2] - args.ground_z for c in colliders], axis=1)
    print(f"Motion: {args.npz_path}\nCollision URDF: {args.robot_urdf}")
    print(f"{len(qpos)} frames @ {fps:g}Hz; {len(colliders)} colliders; cylinder mode={args.cylinder_mode}")
    window = clearances[args.start_frame : args.end_frame + 1]
    f, i = np.unravel_index(window.argmin(), window.shape)
    print(
        f"Deepest in replay range: frame {f + args.start_frame}, "
        f"{colliders[i].name}, clearance {window[f, i] * 1000:.3f}mm"
    )
    print(f"Frame {args.frame} ({args.frame / fps:.2f}s), lowest colliders:")
    for i in np.argsort(clearances[args.frame])[:6]:
        print(f"  {colliders[i].name:34s} {clearances[args.frame, i] * 1000:9.3f} mm")
    print("Negative clearance = below plane. No dynamics/forces, reset noise, or self-collision test.")
    if not args.headless:
        run_viewer(args, qpos, names, fps, colliders, clearances)


if __name__ == "__main__":
    main()
