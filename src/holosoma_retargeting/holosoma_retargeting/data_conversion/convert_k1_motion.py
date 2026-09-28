"""Convert AI Sapiens K1 Rev.1 motions into holosoma whole-body-tracking ``.npz`` files.

The script normalizes any of the K1 sources below into a ``qpos`` trajectory and then
runs holosoma's MuJoCo forward-kinematics exporter (``convert_data_format_mj.py``) on
the K1 training model, so the result is exactly what
``holosoma.managers.command.terms.wbt`` expects (``fps``, ``joint_pos``, ``joint_vel``,
``body_pos_w``, ``body_quat_w`` (wxyz), ``body_lin_vel_w``, ``body_ang_vel_w``,
``joint_names``, ``body_names`` and, for object interaction, ``object_*``).

Supported inputs
----------------
* **omni-k1 bundle** - a directory holding ``motion_raw.npz`` (+ ``manifest.json``) or
  the ``.npz`` itself. ``qpos`` layout ``root_xyz, root_wxyz, 23 joints, object_xyz,
  object_wxyz``; ``fps``, ``joint_names``, ``quat_order`` and ``object_size`` are honoured.
* **cyclo_lab K1 CSV** (``scripts/tools/motion/csv_to_npz.py`` input) - rows of
  ``root_xyz, root_quat, 23 joints`` (optionally followed by ``object_xyz, object_quat``).
  cyclo's CSVs store the root quaternion as ``xyzw`` at 50 Hz; both are the defaults.
* **holosoma_retargeting result** (``examples/robot_retarget.py --robot k1``) - npz with
  ``qpos`` [T, 30] (or [T, 37] with an object) and ``fps``; quaternions are wxyz.
* **generic npz** with a ``qpos`` array (and optional ``fps``/``joint_names``/``quat_order``).

Examples
--------
Result of holosoma_retargeting (run from the ``holosoma_retargeting`` package directory)::

    python data_conversion/convert_k1_motion.py demo_results/k1/robot_only/omomo/sub3_largebox_003.npz

Retargeted OMOMO box-carrying bundle from omni-k1::

    python data_conversion/convert_k1_motion.py ~/omni-k1/outputs/sub3_largebox_003_k1

LAFAN dance retargeted in cyclo_lab (CSV, root quaternion xyzw)::

    python data_conversion/convert_k1_motion.py ~/cyclo_lab/source/cyclo_lab/data/motions/K1_rev1/dance1/dance1.csv

Outputs default to ``src/holosoma/holosoma/data/motions/k1_23dof/whole_body_tracking/<name>_mj[_w_obj].npz``
which can be referenced from a WBT command config as
``holosoma/data/motions/k1_23dof/whole_body_tracking/<name>_mj.npz``.
"""

from __future__ import annotations

import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import tyro
from tyro.conf import Positional

src_root = Path(__file__).resolve().parents[2]
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))
from holosoma_retargeting.config_types.data_conversion import (  # noqa: E402
    K1_JOINT_NAMES,
    DataConversionConfig,
)
from holosoma_retargeting.config_types.data_type import MotionDataConfig  # noqa: E402
from holosoma_retargeting.config_types.robot import RobotConfig  # noqa: E402
from holosoma_retargeting.data_conversion.convert_data_format_mj import run_simulator  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MODEL_XML = REPO_ROOT / "src/holosoma/holosoma/data/robots/k1/k1_23dof.xml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "src/holosoma/holosoma/data/motions/k1_23dof/whole_body_tracking"
DEFAULT_BOX_SIZE = (0.35, 0.30, 0.28)

ROOT_WIDTH = 7
OBJECT_WIDTH = 7
NUM_JOINTS = len(K1_JOINT_NAMES)
ROBOT_WIDTH = ROOT_WIDTH + NUM_JOINTS
ROBOT_OBJECT_WIDTH = ROBOT_WIDTH + OBJECT_WIDTH

QuatOrder = Literal["auto", "wxyz", "xyzw"]


@dataclass(frozen=True)
class K1MotionConversionConfig:
    """CLI configuration (tyro)."""

    input: Positional[str]
    """omni-k1 bundle directory / motion_raw.npz, holosoma_retargeting result npz, cyclo K1 CSV,
    or a generic qpos npz (positional; ``--input`` also works)."""

    output: str | None = None
    """Output npz path. Default: DEFAULT_OUTPUT_DIR/<input name>_mj[_w_obj].npz."""

    input_fps: float | None = None
    """Frame rate of the input. Default: npz ``fps`` key, else 50 for CSV / 30 for npz."""

    output_fps: int = 50
    """Frame rate of the exported motion (holosoma WBT policies run at 50 Hz)."""

    root_quat_order: QuatOrder = "auto"
    """Quaternion order of the root (and object) in the input. auto: npz ``quat_order`` key,
    else xyzw for CSV (cyclo convention) and wxyz for npz (omni-k1 convention)."""

    has_object: bool | None = None
    """Whether the trailing 7 columns are an object pose. Default: inferred from the width."""

    box_size: tuple[float, float, float] | None = None
    """Largebox extents (m) used for the object body. Default: npz ``object_size`` or 0.35 0.30 0.28."""

    box_mass: float = 0.3
    """Largebox mass (kg), informational only for forward kinematics."""

    root_height_offset: float = 0.0
    """Added to the root (and object) z before FK, e.g. to lift feet out of the ground."""

    line_range: tuple[int, int] | None = None
    """Inclusive (start, end) input frame range to convert."""

    model_xml: str = str(DEFAULT_MODEL_XML)
    """K1 MuJoCo model used for forward kinematics (defaults to the holosoma training model)."""

    scratch_dir: str | None = None
    """Directory for intermediate files (default: a temporary directory)."""

    headless: bool = True
    """Run without the MuJoCo viewer."""


@dataclass(frozen=True)
class SourceMotion:
    qpos: np.ndarray  # [T, 30] or [T, 37]; quaternions wxyz
    fps: float
    has_object: bool
    box_size: tuple[float, float, float] | None
    name: str
    description: str


def _normalize_quat(q: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if not np.all(np.isfinite(norm)) or np.any(norm < 1e-6):
        raise ValueError("Encountered a degenerate quaternion in the input motion")
    return q / norm


def _to_wxyz(q: np.ndarray, order: str) -> np.ndarray:
    if order == "wxyz":
        return q
    if order == "xyzw":
        return q[:, [3, 0, 1, 2]]
    raise ValueError(f"Unknown quaternion order: {order}")


def _resolve_input(path: Path) -> Path:
    if path.is_dir():
        bundle = path / "motion_raw.npz"
        if not bundle.is_file():
            raise FileNotFoundError(f"{path} is a directory but holds no motion_raw.npz")
        return bundle
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _reorder_joints(qpos: np.ndarray, joint_names: list[str], has_object: bool) -> np.ndarray:
    if joint_names == list(K1_JOINT_NAMES):
        return qpos
    missing = [name for name in K1_JOINT_NAMES if name not in joint_names]
    if missing:
        raise ValueError(f"Input joint_names lack K1 joints: {missing}")
    index = [joint_names.index(name) for name in K1_JOINT_NAMES]
    joints = qpos[:, ROOT_WIDTH : ROOT_WIDTH + len(joint_names)][:, index]
    tail = qpos[:, -OBJECT_WIDTH:] if has_object else qpos[:, :0]
    return np.concatenate([qpos[:, :ROOT_WIDTH], joints, tail], axis=1)


def _fps_from_value(value: float) -> float:
    # Some files store a timestep (1/30) instead of a rate (30).
    return value if value > 1.0 else 1.0 / value


def load_source(cfg: K1MotionConversionConfig) -> SourceMotion:
    path = _resolve_input(Path(cfg.input).expanduser())
    box_size = cfg.box_size
    quat_order = cfg.root_quat_order
    joint_names: list[str] | None = None
    fps = cfg.input_fps

    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            if "qpos" not in data:
                raise KeyError(f"{path} has no 'qpos' array (keys: {data.files})")
            qpos = np.asarray(data["qpos"], dtype=np.float64)
            if fps is None and "fps" in data:
                fps = _fps_from_value(float(np.asarray(data["fps"]).reshape(-1)[0]))
            if quat_order == "auto":
                quat_order = str(data["quat_order"]) if "quat_order" in data else "wxyz"
            if "joint_names" in data:
                joint_names = [str(name) for name in np.asarray(data["joint_names"]).reshape(-1)]
            if box_size is None and "object_size" in data:
                box_size = tuple(float(v) for v in np.asarray(data["object_size"]).reshape(-1)[:3])
        description = f"npz {path}"
        name = path.parent.name if path.name == "motion_raw.npz" else path.stem
        if fps is None:
            fps = 30.0
    elif path.suffix == ".csv":
        qpos = np.loadtxt(path, delimiter=",", dtype=np.float64, ndmin=2)
        if quat_order == "auto":
            quat_order = "xyzw"  # cyclo_lab csv_to_npz.py default
        if fps is None:
            fps = 50.0
        description = f"csv {path}"
        name = path.stem
    else:
        raise ValueError(f"Unsupported input type: {path.suffix} (expected .npz, .csv or a bundle directory)")

    if qpos.ndim != 2 or qpos.shape[1] not in (ROBOT_WIDTH, ROBOT_OBJECT_WIDTH):
        raise ValueError(
            f"qpos must be [T, {ROBOT_WIDTH}] or [T, {ROBOT_OBJECT_WIDTH}] "
            f"(root 7 + {NUM_JOINTS} joints [+ object 7]); got {qpos.shape}"
        )
    if not np.isfinite(qpos).all():
        raise ValueError("qpos contains NaN/inf values")
    has_object = qpos.shape[1] == ROBOT_OBJECT_WIDTH if cfg.has_object is None else cfg.has_object
    if has_object and qpos.shape[1] != ROBOT_OBJECT_WIDTH:
        raise ValueError("--has-object requires 7 trailing object columns")
    if not has_object and qpos.shape[1] == ROBOT_OBJECT_WIDTH:
        qpos = qpos[:, :ROBOT_WIDTH]
    if qpos.shape[0] < 2:
        raise ValueError("At least two frames are required")

    if joint_names is not None:
        qpos = _reorder_joints(qpos, joint_names, has_object)

    qpos = qpos.copy()
    qpos[:, 3:7] = _normalize_quat(_to_wxyz(qpos[:, 3:7], quat_order))
    qpos[:, 2] += cfg.root_height_offset
    if has_object:
        qpos[:, -4:] = _normalize_quat(_to_wxyz(qpos[:, -4:], quat_order))
        qpos[:, -5] += cfg.root_height_offset  # object z
        if box_size is None:
            box_size = DEFAULT_BOX_SIZE

    return SourceMotion(
        qpos=qpos,
        fps=float(fps),
        has_object=has_object,
        box_size=box_size,
        name=name,
        description=f"{description} ({quat_order} -> wxyz, {qpos.shape[0]} frames @ {fps:g} Hz)",
    )


def build_fk_model(
    model_xml: Path, out_xml: Path, box_size: tuple[float, float, float] | None, box_mass: float
) -> None:
    """Write a copy of ``model_xml`` with an absolute meshdir and, optionally, a free largebox body."""
    tree = ET.parse(model_xml)  # noqa: S314
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    compiler.set("meshdir", str((model_xml.parent / compiler.get("meshdir", ".")).resolve()))
    if box_size is not None:
        sx, sy, sz = box_size
        worldbody = root.find("worldbody")
        box = ET.SubElement(worldbody, "body", name="largebox_link", pos=f"0 0 {sz / 2:g}")
        ET.SubElement(box, "freejoint")  # unnamed, like the robot root
        ET.SubElement(
            box,
            "inertial",
            pos="0 0 0",
            mass=f"{box_mass:g}",
            diaginertia=(
                f"{box_mass * (sy * sy + sz * sz) / 12:g} "
                f"{box_mass * (sx * sx + sz * sz) / 12:g} "
                f"{box_mass * (sx * sx + sy * sy) / 12:g}"
            ),
        )
        ET.SubElement(
            box,
            "geom",
            name="largebox",
            type="box",
            size=f"{sx / 2:g} {sy / 2:g} {sz / 2:g}",
            rgba="0.65 0.45 0.25 0.65",
            contype="1",
            conaffinity="1",
        )
    ET.indent(tree, space="  ")
    out_xml.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out_xml, encoding="utf-8", xml_declaration=True)


def summarize(output: Path) -> None:
    with np.load(output) as data:
        joint_names = [str(name) for name in data["joint_names"]]
        body_names = [str(name) for name in data["body_names"]]
        if joint_names != list(K1_JOINT_NAMES):
            raise RuntimeError(f"Exported joint order differs from K1_JOINT_NAMES: {joint_names}")
        frames = data["joint_pos"].shape[0]
        fps = float(np.asarray(data["fps"]).reshape(-1)[0])
        root_z = data["body_pos_w"][:, body_names.index("pelvis"), 2]
        feet = [name for name in ("left_foot_contact_point", "right_foot_contact_point") if name in body_names]
        foot_z = np.stack([data["body_pos_w"][:, body_names.index(name), 2] for name in feet], axis=1) if feet else None
        print(f"[convert_k1_motion] wrote {output}")
        print(
            f"  frames={frames} fps={fps:g} duration={frames / fps:.2f}s "
            f"joints={len(joint_names)} bodies={len(body_names)}"
        )
        print(f"  pelvis z: min={root_z.min():.3f} max={root_z.max():.3f}")
        if foot_z is not None:
            print(f"  foot contact point z: min={foot_z.min():.3f} max={foot_z.max():.3f}")
        if "object_pos_w" in data:
            obj_z = data["object_pos_w"][:, 2]
            print(f"  object z: min={obj_z.min():.3f} max={obj_z.max():.3f}")


def convert(cfg: K1MotionConversionConfig) -> Path:
    source = load_source(cfg)
    print(f"[convert_k1_motion] input: {source.description}")

    scratch = Path(cfg.scratch_dir).expanduser() if cfg.scratch_dir else Path(tempfile.mkdtemp(prefix="k1_motion_"))
    scratch.mkdir(parents=True, exist_ok=True)

    qpos_file = scratch / f"{source.name}_qpos.npz"
    np.savez(qpos_file, qpos=source.qpos.astype(np.float32), fps=np.asarray([source.fps]))

    # convert_data_format_mj derives the MJCF path from the URDF path
    # (<stem>.xml or <stem>_w_largebox.xml), so only the naming matters here.
    urdf_placeholder = scratch / "k1_23dof.urdf"
    model_out = scratch / ("k1_23dof_w_largebox.xml" if source.has_object else "k1_23dof.xml")
    build_fk_model(
        Path(cfg.model_xml).expanduser(), model_out, source.box_size if source.has_object else None, cfg.box_mass
    )

    if cfg.output is None:
        suffix = "_mj_w_obj.npz" if source.has_object else "_mj.npz"
        output = DEFAULT_OUTPUT_DIR / f"{source.name}{suffix}"
    else:
        output = Path(cfg.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)

    conversion = DataConversionConfig(
        input_file=str(qpos_file),
        robot="k1",
        data_format="smplh",
        object_name="largebox" if source.has_object else "ground",
        input_fps=round(source.fps),
        output_fps=cfg.output_fps,
        line_range=cfg.line_range,
        has_dynamic_object=source.has_object,
        output_name=str(output),
        once=True,
        headless=cfg.headless,
        use_omniretarget_data=False,
        robot_config=RobotConfig(robot_type="k1", robot_urdf_file=str(urdf_placeholder)),
        motion_data_config=MotionDataConfig(data_format="smplh", robot_type="k1"),
        joint_names=list(K1_JOINT_NAMES),
    )
    run_simulator(conversion)

    meta = {
        "source": str(Path(cfg.input).expanduser().resolve()),
        "input_fps": source.fps,
        "output_fps": cfg.output_fps,
        "has_object": source.has_object,
        "box_size": source.box_size,
        "root_height_offset": cfg.root_height_offset,
        "model_xml": str(Path(cfg.model_xml).expanduser().resolve()),
        "joint_names": list(K1_JOINT_NAMES),
    }
    output.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    summarize(output)
    return output


def main() -> None:
    convert(tyro.cli(K1MotionConversionConfig))


if __name__ == "__main__":
    main()
