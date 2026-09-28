"""Convert holosoma whole-body-tracking artifacts into a cyclo_lab (ai_sapiens_sim2real) bundle.

holosoma stays the source of truth (checkpoint, bundled ONNX, motion ``.npz``); this module
derives the three files the ``ai_sapiens_sim2real`` runtime consumes for a mimic mode::

    <bundle>/
    ├── exported/policy.onnx      # single input (observation) -> single output (action)
    ├── params/sim2real.yaml      # joints, gains, action scale/offset, observation layout
    ├── params/<motion>.csv       # root xyz, root quat (xyzw), joints in policy order @ step_dt
    └── manifest.json             # provenance

Conventions verified against ``ai_sapiens_sim2real`` sources:

* ``policy_joints`` is matched to hardware joints by name, so holosoma's URDF joint order is
  used verbatim.
* Observation terms are concatenated in yaml order; ``joint_pos_rel`` / ``last_action`` and
  ``motion_anchor_ori_b`` correspond exactly to holosoma's ``dof_pos`` / ``actions`` /
  ``motion_ref_ori_b`` (same formulas, ``torso_link`` anchor).
* Action target = ``offset + scale * action`` with ``offset`` = default joint angle and
  ``scale`` = holosoma's per-joint action scale.
* The motion CSV is read as ``root_xyz, root_qxqyqzqw, joints``; a header row names the joint
  columns. Joint velocities are forward-differenced by the runtime.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import yaml

from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_types.robot import RobotConfig
from holosoma.utils.inference_helpers import get_control_gains_from_config
from holosoma.utils.path import resolve_data_file_path

# holosoma observation term -> (cyclo term name, dim(num_joints), params)
OBS_TERM_MAP: dict[str, tuple[str, Any, dict[str, str]]] = {
    "motion_command": ("motion_command", lambda n: 2 * n, {"command_name": "reference_trajectory"}),
    "motion_ref_ori_b": ("motion_anchor_ori_b", lambda _n: 6, {"command_name": "reference_trajectory"}),
    "base_ang_vel": ("base_ang_vel", lambda _n: 3, {}),
    "projected_gravity": ("projected_gravity", lambda _n: 3, {}),
    "dof_pos": ("joint_pos_rel", lambda n: n, {}),
    "dof_vel": ("joint_vel_rel", lambda n: n, {}),
    "actions": ("last_action", lambda n: n, {}),
}

ROOT_CSV_COLUMNS = ["root_x", "root_y", "root_z", "root_qx", "root_qy", "root_qz", "root_qw"]


class _FlowList(list):
    """List rendered in YAML flow style ([a, b, c]) like cyclo's exporter."""


class _Dumper(yaml.SafeDumper):
    pass


_Dumper.add_representer(
    _FlowList, lambda dumper, data: dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)
)


def _r(value: float, ndigits: int = 6) -> float:
    return float(round(float(value), ndigits))


# --------------------------------------------------------------------------- robot side
def action_scales(robot_config: RobotConfig, action_term_scale: float = 1.0) -> list[float]:
    """Per-joint action scale in dof order (mirrors JointPositionActionTerm._configure_action_scales)."""
    control = robot_config.control
    kp, _ = get_control_gains_from_config(robot_config)
    scales = []
    for i, effort in enumerate(robot_config.dof_effort_limit_list):
        if control.action_scales_by_effort_limit_over_p_gain:
            scales.append(0.0 if kp[i] == 0.0 else control.action_scale * effort / kp[i])
        else:
            scales.append(control.action_scale)
    return [s * action_term_scale for s in scales]


def step_dt(exp: ExperimentConfig) -> float:
    sim = exp.simulator.config.sim
    return sim.control_decimation / sim.fps


def observation_layout(exp: ExperimentConfig, group: str = "actor_obs") -> list[dict[str, Any]]:
    """Ordered cyclo observation terms derived from the holosoma actor observation group."""
    obs_group = exp.observation.groups[group]
    num_joints = len(exp.robot.dof_names)
    terms = []
    for name, term in obs_group.terms.items():
        func = term.func.split(":")[-1]
        if func not in OBS_TERM_MAP:
            raise ValueError(f"Observation term '{name}' ({term.func}) has no ai_sapiens_sim2real equivalent")
        cyclo_name, dim_fn, params = OBS_TERM_MAP[func]
        dim = dim_fn(num_joints)
        scale = term.scale if isinstance(term.scale, (tuple, list)) else [term.scale] * dim
        if len(scale) != dim:
            raise ValueError(f"Observation term '{name}' scale has {len(scale)} entries, expected {dim}")
        terms.append(
            {
                "name": cyclo_name,
                "holosoma_name": name,
                "dim": dim,
                "params": dict(params),
                "clip": None if term.clip is None else [float(term.clip[0]), float(term.clip[1])],
                "scale": [float(s) for s in scale],
                "history_length": int(obs_group.history_length),
            }
        )
    return terms


def build_sim2real_config(exp: ExperimentConfig, action_term: str = "joint_control") -> dict[str, Any]:
    """sim2real.yaml content for a holosoma WBT experiment."""
    robot = exp.robot
    joints = list(robot.dof_names)
    kp, kd = get_control_gains_from_config(robot)
    term_cfg = exp.action.terms[action_term]
    scales = action_scales(robot, float(term_cfg.scale))
    defaults = [float(robot.init_state.default_joint_angles[j]) for j in joints]

    joint_properties = {
        joint: {
            "default_position": _r(defaults[i]),
            "stiffness": _r(kp[i], 3),
            "damping": _r(kd[i], 3),
            "position_limit": None,
        }
        for i, joint in enumerate(joints)
    }
    observations = {}
    for term in observation_layout(exp):
        observations[term["name"]] = {
            "params": term["params"],
            "clip": None if term["clip"] is None else _FlowList(term["clip"]),
            "scale": _FlowList(_r(s) for s in term["scale"]),
            "history_length": term["history_length"],
        }
    return {
        "policy_joints": joints,
        "step_dt": _r(step_dt(exp)),
        "joint_properties": joint_properties,
        "commands": {},
        "actions": {
            "joint_pos": {
                "clip": None if term_cfg.clip is None else _FlowList(float(v) for v in term_cfg.clip),
                "scale": _FlowList(_r(s) for s in scales),
                "offset": _FlowList(_r(d) for d in defaults),
            }
        },
        "observations": observations,
    }


def dump_sim2real_yaml(config: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.dump(config, f, Dumper=_Dumper, sort_keys=False, default_flow_style=False)


# ---------------------------------------------------------------------------- policy
def extract_policy_onnx(src: Path, dst: Path) -> tuple[int, int]:
    """Write a single-input/single-output actor ONNX (observation -> action).

    Accepts holosoma's bundled WBT export (inputs ``obs``, ``time_step``; outputs ``actions``,
    ``joint_pos``, ...) or a plain export (one input, one output). Returns (obs_dim, action_dim).
    """
    model = onnx.load(str(src))
    inputs = [i.name for i in model.graph.input]
    outputs = [o.name for o in model.graph.output]
    dst.parent.mkdir(parents=True, exist_ok=True)
    if len(inputs) == 1 and len(outputs) == 1:
        shutil.copy2(src, dst)
    elif "obs" in inputs and "actions" in outputs:
        onnx.utils.extract_model(str(src), str(dst), ["obs"], ["actions"], check_model=True)
        extracted = onnx.load(str(dst))
        if not extracted.metadata_props:
            extracted.metadata_props.extend(model.metadata_props)
            onnx.save(extracted, str(dst))
    else:
        raise ValueError(f"Unsupported ONNX interface: inputs={inputs} outputs={outputs}")

    policy = onnx.load(str(dst))
    onnx.checker.check_model(policy)

    def _last_dim(value_info) -> int:
        return int(value_info.type.tensor_type.shape.dim[-1].dim_value)

    return _last_dim(policy.graph.input[0]), _last_dim(policy.graph.output[0])


def run_policy_once(policy_path: Path, obs_dim: int) -> np.ndarray:
    session = ort.InferenceSession(str(policy_path), providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    return session.run(None, {name: np.zeros((1, obs_dim), dtype=np.float32)})[0]


# ---------------------------------------------------------------------------- motion
@dataclass
class MotionFrames:
    root_pos: np.ndarray  # [T, 3]
    root_quat_xyzw: np.ndarray  # [T, 4]
    joint_pos: np.ndarray  # [T, J] in `joint_names` order
    joint_names: list[str]
    fps: float


def load_motion_npz(path: Path, root_body: str = "pelvis") -> MotionFrames:
    with np.load(path) as data:
        body_names = [str(b) for b in data["body_names"]]
        joint_names = [str(j) for j in data["joint_names"]]
        root = body_names.index(root_body)
        quat_wxyz = np.asarray(data["body_quat_w"][:, root], dtype=np.float64)
        return MotionFrames(
            root_pos=np.asarray(data["body_pos_w"][:, root], dtype=np.float64),
            root_quat_xyzw=quat_wxyz[:, [1, 2, 3, 0]],
            joint_pos=np.asarray(data["joint_pos"][:, 7:], dtype=np.float64),
            joint_names=joint_names,
            fps=float(np.asarray(data["fps"]).reshape(-1)[0]),
        )


def _slerp(q0: np.ndarray, q1: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1, dot = -q1, -dot
    alpha = alpha[:, None]
    if dot > 0.9995:
        out = q0 + alpha * (q1 - q0)
    else:
        theta = np.arccos(np.clip(dot, -1.0, 1.0))
        out = np.sin((1 - alpha) * theta) / np.sin(theta) * q0 + np.sin(alpha * theta) / np.sin(theta) * q1
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def _yaw_of_xyzw(q: np.ndarray) -> float:
    x, y, z, w = q
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def default_pose_prepend(
    motion: MotionFrames,
    default_joint_pos: np.ndarray,
    default_root_z: float,
    duration_s: float,
) -> MotionFrames:
    """Prepend an interpolation from the default standing pose to frame 0.

    Mirrors holosoma's ``MotionCommand`` prepend: default joints, root at the clip's frame-0
    x/y and yaw with the configured standing height and zero roll/pitch.
    """
    num_steps = round(duration_s * motion.fps)
    if num_steps <= 1:
        return motion
    yaw = _yaw_of_xyzw(motion.root_quat_xyzw[0])
    default_quat = np.array([0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)])
    default_pos = np.array([motion.root_pos[0, 0], motion.root_pos[0, 1], default_root_z])
    alphas = np.arange(num_steps, dtype=np.float64) / num_steps  # frames strictly before frame 0
    joints = default_joint_pos[None, :] + alphas[:, None] * (motion.joint_pos[0] - default_joint_pos)[None, :]
    pos = default_pos[None, :] + alphas[:, None] * (motion.root_pos[0] - default_pos)[None, :]
    quat = _slerp(default_quat, motion.root_quat_xyzw[0], alphas)
    return MotionFrames(
        root_pos=np.concatenate([pos, motion.root_pos]),
        root_quat_xyzw=np.concatenate([quat, motion.root_quat_xyzw]),
        joint_pos=np.concatenate([joints, motion.joint_pos]),
        joint_names=motion.joint_names,
        fps=motion.fps,
    )


def reorder_joints(motion: MotionFrames, joint_order: list[str]) -> MotionFrames:
    missing = [j for j in joint_order if j not in motion.joint_names]
    if missing:
        raise ValueError(f"Motion lacks joints required by the policy: {missing}")
    index = [motion.joint_names.index(j) for j in joint_order]
    return MotionFrames(
        motion.root_pos, motion.root_quat_xyzw, motion.joint_pos[:, index], list(joint_order), motion.fps
    )


def write_motion_csv(motion: MotionFrames, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = np.concatenate([motion.root_pos, motion.root_quat_xyzw, motion.joint_pos], axis=1)
    header = ",".join(ROOT_CSV_COLUMNS + motion.joint_names)
    np.savetxt(path, rows, delimiter=",", fmt="%.9f", header=header, comments="")


# ---------------------------------------------------------------------------- bundle
def _git_commit(repo: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def motion_config_of(exp: ExperimentConfig):
    return exp.command.setup_terms["motion_command"].params["motion_config"]


def export_cyclo_bundle(
    exp_name: str,
    exp: ExperimentConfig,
    onnx_path: Path,
    out_dir: Path,
    motion_name: str,
    motion_npz: Path | None = None,
    prepend_s: float | None = None,
    checkpoint: Path | None = None,
) -> dict[str, Any]:
    """Write policy.onnx, sim2real.yaml and the motion CSV; returns the manifest."""
    out_dir = Path(out_dir)
    policy_path = out_dir / "exported" / "policy.onnx"
    yaml_path = out_dir / "params" / "sim2real.yaml"
    csv_path = out_dir / "params" / f"{motion_name}.csv"

    obs_dim, action_dim = extract_policy_onnx(Path(onnx_path), policy_path)
    config = build_sim2real_config(exp)
    layout = observation_layout(exp)
    expected_obs = sum(t["dim"] * t["history_length"] for t in layout)
    if expected_obs != obs_dim:
        raise ValueError(f"ONNX expects {obs_dim} observation values but the yaml layout produces {expected_obs}")
    if action_dim != len(config["policy_joints"]):
        raise ValueError(f"ONNX outputs {action_dim} actions but the robot has {len(config['policy_joints'])} joints")
    action = run_policy_once(policy_path, obs_dim)
    if action.shape != (1, action_dim) or not np.isfinite(action).all():
        raise RuntimeError(f"policy.onnx sanity run failed: shape={action.shape}")
    dump_sim2real_yaml(config, yaml_path)

    motion_cfg = motion_config_of(exp)
    npz = Path(motion_npz) if motion_npz else Path(resolve_data_file_path(motion_cfg.motion_file))
    motion = reorder_joints(load_motion_npz(npz), config["policy_joints"])
    target_fps = 1.0 / config["step_dt"]
    if abs(motion.fps - target_fps) > 1e-6:
        raise ValueError(f"Motion fps {motion.fps} differs from policy rate {target_fps}; re-convert the motion")
    if prepend_s is None:
        prepend_s = motion_cfg.default_pose_prepend_duration_s if motion_cfg.enable_default_pose_prepend else 0.0
    num_clip_frames = motion.joint_pos.shape[0]
    if prepend_s > 0.0:
        defaults = np.array([exp.robot.init_state.default_joint_angles[j] for j in config["policy_joints"]])
        motion = default_pose_prepend(motion, defaults, float(exp.robot.init_state.pos[2]), prepend_s)
    write_motion_csv(motion, csv_path)

    repo_root = Path(__file__).resolve().parents[4]
    manifest = {
        "format": "ai_sapiens_sim2real mimic bundle (generated by holosoma)",
        "experiment": exp_name,
        "holosoma_commit": _git_commit(repo_root),
        "source_onnx": str(Path(onnx_path).resolve()),
        "source_checkpoint": None if checkpoint is None else str(Path(checkpoint).resolve()),
        "source_motion_npz": str(npz.resolve()),
        "motion_csv": csv_path.name,
        "motion_frames": int(motion.joint_pos.shape[0]),
        "clip_frames": int(num_clip_frames),
        "prepend_s": float(prepend_s),
        "fps": float(motion.fps),
        "obs_dim": int(obs_dim),
        "action_dim": int(action_dim),
        "observations": [{"name": t["name"], "holosoma": t["holosoma_name"], "dim": t["dim"]} for t in layout],
        "policy_joints": config["policy_joints"],
        "anchor_body": list(motion_cfg.body_name_ref),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
