from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from scripts import aw_precompute_weights as aw
from scripts import aw_measurement_c


def _write_aw_h5(path: Path, rewards: np.ndarray) -> None:
    num_rows = rewards.shape[0]
    with h5py.File(path, "w") as h5_file:
        h5_file.create_dataset("rewards", data=rewards.astype(np.float32))
        h5_file.create_dataset("motion_phase", data=np.linspace(0.0, 0.99, num_rows, dtype=np.float32))
        dones = np.zeros(num_rows, dtype=np.uint8)
        dones[-1] = 1
        h5_file.create_dataset("dones", data=dones)
        h5_file.create_dataset("truncations", data=np.zeros(num_rows, dtype=np.uint8))
        bad_tracking = np.zeros(num_rows, dtype=np.uint8)
        bad_tracking[-1] = 1
        h5_file.create_dataset("next_done_bad_tracking", data=bad_tracking)


def _write_sidecar(h5_path: Path, rewards: np.ndarray, *, rhash: str | None = None) -> Path:
    sidecar_path = Path(f"{h5_path}.aw_weights.npz")
    stored_rewards = rewards.astype(np.float32).astype(np.float64)
    np.savez_compressed(
        sidecar_path,
        weight=np.ones(rewards.shape[0], dtype=np.float32),
        phase_bin=np.minimum(
            np.arange(rewards.shape[0], dtype=np.int16) * 20 // rewards.shape[0],
            19,
        ),
        beta=np.float64(1.0),
        ess_frac=np.float64(1.0),
        n=rewards.shape[0],
        h5=h5_path.name,
        rhash=rhash or aw.reward_fingerprint(stored_rewards),
    )
    return sidecar_path


def test_verify_accepts_matching_h5_sidecar_pair(tmp_path, capsys):
    h5_path = tmp_path / "dataset.h5"
    rewards = np.linspace(-1.0, 2.0, 2500, dtype=np.float64)
    _write_aw_h5(h5_path, rewards)
    _write_sidecar(h5_path, rewards)

    return_code = aw.main(["--verify", str(h5_path)])

    assert return_code == 0
    assert "PASS: sidecar is paired with this H5" in capsys.readouterr().out


def test_verify_rejects_same_length_sidecar_with_wrong_reward_hash(tmp_path, capsys):
    h5_path = tmp_path / "dataset.h5"
    rewards = np.linspace(-1.0, 2.0, 2500, dtype=np.float64)
    _write_aw_h5(h5_path, rewards)
    _write_sidecar(h5_path, rewards, rhash="deadbeefdeadbeef")

    return_code = aw.main(["--verify", str(h5_path)])

    assert return_code == 1
    assert "FAIL: sidecar/H5 pairing mismatch" in capsys.readouterr().out


def test_report_only_never_writes_sidecar(tmp_path, capsys):
    h5_path = tmp_path / "dataset.h5"
    rewards = np.linspace(0.01, 0.2, 100, dtype=np.float64)
    _write_aw_h5(h5_path, rewards)
    sidecar_path = Path(f"{h5_path}.aw_weights.npz")

    return_code = aw.main(
        [
            "--report-only",
            "--H",
            "10",
            "--n-bins",
            "4",
            str(h5_path),
        ]
    )

    assert return_code in {0, 2}
    assert not sidecar_path.exists()
    assert "[report-only] no sidecar written" in capsys.readouterr().out


def test_measurement_c_reuses_verified_sidecar_pair(tmp_path, capsys):
    h5_path = tmp_path / "dataset.h5"
    rewards = np.linspace(-1.0, 2.0, 2500, dtype=np.float64)
    _write_aw_h5(h5_path, rewards)
    _write_sidecar(h5_path, rewards)

    return_code = aw_measurement_c.main([str(h5_path), "--bins", "0"])

    assert return_code == 0
    output = capsys.readouterr().out
    assert "[pairing] rhash OK" in output
    assert "bin  0:" in output
