from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts/crop_lafan_motion.py"
SPEC = importlib.util.spec_from_file_location("crop_lafan_motion", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_crop_lafan_motion_uses_end_exclusive_range(tmp_path):
    source = tmp_path / "source.npy"
    output = tmp_path / "clips" / "clip.npy"
    motion = np.arange(20 * 22 * 3, dtype=np.float32).reshape(20, 22, 3)
    np.save(source, motion)

    source_frames, cropped_frames = MODULE.crop_lafan_motion(source, output, 4, 9)

    assert source_frames == 20
    assert cropped_frames == 5
    np.testing.assert_array_equal(np.load(output), motion[4:9])


def test_crop_lafan_motion_rejects_invalid_range(tmp_path):
    source = tmp_path / "source.npy"
    np.save(source, np.zeros((10, 22, 3), dtype=np.float32))

    with pytest.raises(ValueError, match="Invalid frame range"):
        MODULE.crop_lafan_motion(source, tmp_path / "clip.npy", 8, 11)
