#!/usr/bin/env python3
"""Export a holosoma WBT policy as a cyclo_lab / ai_sapiens_sim2real mimic bundle.

holosoma artifacts stay untouched; this writes a derived bundle::

    <out>/exported/policy.onnx, <out>/params/sim2real.yaml, <out>/params/<motion>.csv, manifest.json

Example (K1 dance1 policy)::

    python scripts/export_cyclo_sim2real.py \
        --exp k1-23dof-wbt-lafan-dance1 \
        --onnx logs/WholeBodyTracking/<run>/exported/model_30000.onnx \
        --motion-name dance1 \
        --out ~/cyclo_lab/third_party/ai_sapiens/ai_sapiens_sim2real/assets/k1/mimic/dance1_holosoma

Then register the bundle in ``ai_sapiens_sim2real/config/k1_config.yaml`` (``kind: mimic``,
``asset: mimic/dance1_holosoma``, ``motion: dance1.csv``).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import tyro

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "holosoma"))

from holosoma.config_values.experiment import DEFAULTS  # noqa: E402
from holosoma.utils.cyclo_sim2real_export import export_cyclo_bundle  # noqa: E402


@dataclass(frozen=True)
class ExportConfig:
    exp: str
    """Experiment name, e.g. k1-23dof-wbt-lafan-dance1 (or k1_23dof_wbt_lafan_dance1)."""

    onnx: str
    """holosoma exported ONNX (bundled WBT export or plain actor export)."""

    out: str | None = None
    """Bundle directory (default: exports/cyclo/<exp>)."""

    motion_name: str | None = None
    """CSV base name (default: motion file stem without the _mj suffix)."""

    motion_npz: str | None = None
    """Override the motion .npz (default: the experiment's motion_file)."""

    prepend_s: float | None = None
    """Default-pose prepend seconds (default: the experiment's motion config; 0 disables)."""

    checkpoint: str | None = None
    """Optional checkpoint path recorded in the manifest."""


def main(cfg: ExportConfig) -> None:
    key = cfg.exp.replace("-", "_")
    if key not in DEFAULTS:
        known = [k for k in DEFAULTS if k.startswith("k1_")]
        raise SystemExit(f"Unknown experiment '{cfg.exp}'. Known K1 experiments: {known}")
    exp = DEFAULTS[key]
    motion_file = exp.command.setup_terms["motion_command"].params["motion_config"].motion_file
    motion_name = cfg.motion_name or Path(motion_file).stem.removesuffix("_mj")
    out = Path(cfg.out) if cfg.out else REPO_ROOT / "exports" / "cyclo" / key
    manifest = export_cyclo_bundle(
        key,
        exp,
        Path(cfg.onnx),
        out,
        motion_name,
        motion_npz=Path(cfg.motion_npz) if cfg.motion_npz else None,
        prepend_s=cfg.prepend_s,
        checkpoint=Path(cfg.checkpoint) if cfg.checkpoint else None,
    )
    print(f"[export_cyclo_sim2real] bundle written to {out}")
    print(f"  policy: obs_dim={manifest['obs_dim']} action_dim={manifest['action_dim']}")
    print(
        f"  motion: {manifest['motion_csv']} frames={manifest['motion_frames']} "
        f"(clip {manifest['clip_frames']}, prepend {manifest['prepend_s']}s)"
    )
    layout = ", ".join(f"{o['name']}({o['dim']})" for o in manifest["observations"])
    print(f"  observations: {layout}")


if __name__ == "__main__":
    main(tyro.cli(ExportConfig))
