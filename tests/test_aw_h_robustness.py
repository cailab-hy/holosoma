from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from scripts import aw_h_robustness as hr
from scripts import aw_precompute_weights as aw


def _write_h5(path: Path, rewards, dones, truncs, phase, bad) -> None:
    with h5py.File(path, "w") as h5_file:
        h5_file.create_dataset("rewards", data=np.asarray(rewards, dtype=np.float32))
        h5_file.create_dataset("motion_phase", data=np.asarray(phase, dtype=np.float32))
        h5_file.create_dataset("dones", data=np.asarray(dones, dtype=np.uint8))
        h5_file.create_dataset("truncations", data=np.asarray(truncs, dtype=np.uint8))
        h5_file.create_dataset("next_done_bad_tracking", data=np.asarray(bad, dtype=np.uint8))


def _episodic_dataset(path: Path, *, n_episodes: int = 120, ep_len: int = 160, seed: int = 0):
    """Episodes whose reward level is constant within an episode.

    G^H is then a monotone function of that level for every H, so the induced weight
    ordering is (almost) H-invariant -- the "robust" reference case.
    """
    rng = np.random.default_rng(seed)
    rewards, dones, truncs, phase, bad = [], [], [], [], []
    for episode in range(n_episodes):
        level = rng.normal(0.0, 1.0)
        rewards.append(np.full(ep_len, level) + rng.normal(0.0, 0.01, ep_len))
        done = np.zeros(ep_len, np.uint8)
        done[-1] = 1
        dones.append(done)
        truncs.append(np.zeros(ep_len, np.uint8))
        phase.append(np.linspace(0.0, 0.99, ep_len))
        bad_flag = np.zeros(ep_len, np.uint8)
        bad_flag[-1] = 1 if level < 0.0 else 0
        bad.append(bad_flag)
    concat = lambda parts: np.concatenate(parts)
    _write_h5(path, concat(rewards), concat(dones), concat(truncs), concat(phase), concat(bad))
    return concat(rewards).astype(np.float32).astype(np.float64)


def test_top_k_overlap_is_exact_on_a_known_ranking():
    x = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    y = np.array([5.0, 4.0, 1.0, 2.0, 3.0])

    overlap, jaccard = hr.top_k_overlap(x, y, 2)

    assert overlap == pytest.approx(1.0)
    assert jaccard == pytest.approx(1.0)
    overlap, jaccard = hr.top_k_overlap(x, y, 3)
    assert overlap == pytest.approx(2 / 3)  # {0,1,2} vs {0,1,4}
    assert jaccard == pytest.approx(0.5)


def test_spearman_is_one_under_a_monotone_reparameterisation():
    x = np.linspace(0.1, 3.0, 500)

    assert hr.spearman_rho(x, np.exp(x)) == pytest.approx(1.0)
    assert hr.spearman_rho(x, -x) == pytest.approx(-1.0)


def test_survivor_mass_matches_the_measurement_c_definition():
    # two episodes of 2 rows; bins [b, b] and [b, b]; episode 0 fails at the wall bin.
    bins = np.array([1, 1, 1, 1])
    ep_id = np.array([0, 0, 1, 1])
    terminal_bad = np.array([True, False])
    terminal_bin = np.array([1, 1])
    weight = np.array([3.0, 1.0, 1.0, 1.0])

    stat = hr.survivor_mass(weight, bins, ep_id, terminal_bad, terminal_bin, 1, 0)

    assert stat["n_fail"] == 2 and stat["n_surv"] == 2
    assert stat["surv_count_share"] == pytest.approx(0.5)
    assert stat["R_surv"] == pytest.approx(2.0 / 6.0)  # SURV mass 2 of total 6


def test_horizon_realised_frac_counts_only_full_windows():
    starts = np.array([0, 5])
    ends = np.array([4, 9])

    # per episode of length 5, rows with >=2 steps left: t=0,1,2 -> 3 of 5
    assert hr.horizon_realised_frac(starts, ends, 10, 2) == pytest.approx(0.6)
    assert hr.horizon_realised_frac(starts, ends, 10, 0) == pytest.approx(1.0)


def test_h0_arm_reproduces_the_shipped_sidecar_and_reports_robust(tmp_path, capsys):
    h5_path = tmp_path / "dataset.h5"
    _episodic_dataset(h5_path)
    assert aw.main([str(h5_path), "--H", "50", "--gamma", "0.99"]) in (0, 2)
    summary_csv = tmp_path / "summary.csv"

    return_code = hr.main([str(h5_path), "--kendall-sample", "20000",
                           "--wall-bins", "5", "13", "--csv", str(summary_csv)])

    out = capsys.readouterr().out
    assert "[reproduction] H0 arm vs shipped sidecar" in out and "-> PASS" in out
    assert return_code == 0, out
    assert "ROBUST:" in out
    assert summary_csv.exists()
    horizons = [int(row.split(",")[0]) for row in summary_csv.read_text().splitlines()[1:]]
    assert horizons == [25, 50, 100]


def test_rejects_a_sidecar_that_is_not_paired_with_the_h5(tmp_path):
    h5_path = tmp_path / "dataset.h5"
    _episodic_dataset(h5_path)
    aw.main([str(h5_path)])
    sidecar_path = Path(f"{h5_path}.aw_weights.npz")
    with np.load(sidecar_path, allow_pickle=False) as sidecar:
        fields = {key: sidecar[key] for key in sidecar.files}
    fields["rhash"] = np.asarray("0" * 16)
    np.savez_compressed(sidecar_path, **fields)

    with pytest.raises(ValueError, match="rhash mismatch"):
        hr.main([str(h5_path)])


def test_flags_a_dataset_whose_weight_ordering_depends_on_h(tmp_path, capsys):
    # reward sign flips halfway through every episode, so a short horizon and a long
    # horizon rank the same transition oppositely.
    rng = np.random.default_rng(1)
    ep_len, n_episodes = 160, 120
    rewards, dones, truncs, phase, bad = [], [], [], [], []
    for _ in range(n_episodes):
        level = rng.normal(0.0, 1.0)
        seg = np.concatenate([np.full(ep_len // 2, level), np.full(ep_len // 2, -4.0 * level)])
        rewards.append(seg + rng.normal(0.0, 0.05, ep_len))
        done = np.zeros(ep_len, np.uint8)
        done[-1] = 1
        dones.append(done)
        truncs.append(np.zeros(ep_len, np.uint8))
        phase.append(np.linspace(0.0, 0.99, ep_len))
        bad_flag = np.zeros(ep_len, np.uint8)
        bad_flag[-1] = 1 if level < 0.0 else 0
        bad.append(bad_flag)
    concat = lambda parts: np.concatenate(parts)
    h5_path = tmp_path / "flip.h5"
    _write_h5(h5_path, concat(rewards), concat(dones), concat(truncs), concat(phase), concat(bad))

    return_code = hr.main([str(h5_path), "--H0", "50", "--kendall-sample", "0",
                           "--wall-bins", "10"])

    out = capsys.readouterr().out
    assert return_code == 2, out
    assert "NOT ROBUST:" in out
