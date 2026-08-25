from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aw_wall_probe import checkpoint_specs_from_directory
from aw_wall_probe import DEFAULT_GRID
from aw_wall_probe import DEFAULT_H5
from aw_wall_probe import DEFAULT_INDEX_CACHE
from aw_wall_probe import _stub_scorer
from aw_wall_probe import discover_checkpoints
from aw_wall_probe import parse_step
from aw_wall_probe import select_rows
from aw_wall_probe import select_checkpoint_grid
from aw_wall_probe import validate_selected_rows


def _touch_checkpoint(directory: Path, step: int) -> Path:
    path = directory / f"model_{step:07d}.pt"
    path.touch()
    return path


def test_step_parser_supports_compact_units() -> None:
    assert parse_step("20k") == 20_000
    assert parse_step("0.3m") == 300_000
    assert parse_step("210_000") == 210_000


def test_grid_selects_exact_steps_and_every_saved_step_in_range(tmp_path: Path) -> None:
    for step in (20_000, 60_000, 170_000, 180_000, 190_000, 210_000, 230_000, 300_000):
        _touch_checkpoint(tmp_path, step)

    selected = select_checkpoint_grid(
        discover_checkpoints(tmp_path),
        "20k,60k,170k,180k:220k,230k,260k,300k",
        label="cell1",
    )

    assert [step for step, _ in selected] == [
        20_000,
        60_000,
        170_000,
        180_000,
        190_000,
        210_000,
        230_000,
        300_000,
    ]


def test_default_cell1_grid_selects_1k_steps_through_last_checkpoint(tmp_path: Path) -> None:
    for step in (500, 1_000, 2_000, 2_500, 3_000):
        _touch_checkpoint(tmp_path, step)

    selected = select_checkpoint_grid(
        discover_checkpoints(tmp_path),
        DEFAULT_GRID,
        label="default-cell1",
    )

    assert DEFAULT_H5.endswith("g1_29dof_wbt_fastsac_episode1m_env256_dataset.h5")
    assert DEFAULT_INDEX_CACHE == "probe_rows_cell1_v3_full.npz"
    assert [step for step, _ in selected] == [1_000, 2_000, 3_000]


def test_directory_specs_use_actual_checkpoint_step(tmp_path: Path) -> None:
    checkpoint = _touch_checkpoint(tmp_path, 140_000)

    specs = checkpoint_specs_from_directory(tmp_path, "140k", "alpha1", "cql")

    assert specs == [f"alpha1,cql,140000,{checkpoint}"]


def test_strict_index_cache_rejects_reward_hash_mismatch(tmp_path: Path) -> None:
    cache = tmp_path / "probe_rows_cell1.npz"
    cells = np.empty(2, dtype=object)
    cells[0] = np.array([0], dtype=np.int64)
    cells[1] = np.array([1], dtype=np.int64)
    np.savez_compressed(
        cache,
        cell_keys=np.array([(4, "SURV"), (4, "FAIL")]),
        cell_idx=cells,
        span_idx=np.array([0, 1], dtype=np.int64),
        rhash="old-hash",
    )

    with pytest.raises(ValueError, match="rhash mismatch"):
        select_rows(
            bins=np.array([4, 4]),
            ep_id=np.array([0, 1]),
            term_bin=np.array([4, 4]),
            term_bad=np.array([False, True]),
            wall_bins=[4],
            per_cell=1,
            span_n=2,
            cache=str(cache),
            rhash="new-hash",
            strict_cache=True,
        )


def test_dry_run_stub_uses_dataset_action_dimension() -> None:
    rng = np.random.default_rng(1)
    _, pi_fn = _stub_scorer(rng, action_dim=29)

    actions = pi_fn(np.zeros((8, 154), dtype=np.float32))

    assert actions.shape == (8, 29)


def test_full_cell_and_full_span_selection_is_deterministic(tmp_path: Path) -> None:
    bins = np.array([4, 4, 4, 4], dtype=np.int64)
    ep_id = np.array([0, 1, 2, 3], dtype=np.int64)
    term_bin = np.array([4, 4, 8, 8], dtype=np.int64)
    term_bad = np.array([True, True, False, False])
    cache = tmp_path / "probe_rows_v3.npz"

    cells, span_idx = select_rows(
        bins=bins,
        ep_id=ep_id,
        term_bin=term_bin,
        term_bad=term_bad,
        wall_bins=[4],
        per_cell=None,
        span_n=100,
        cache=str(cache),
        rhash="same-hash",
    )

    np.testing.assert_array_equal(cells[(4, "FAIL")], np.array([0, 1]))
    np.testing.assert_array_equal(cells[(4, "SURV")], np.array([2, 3]))
    np.testing.assert_array_equal(span_idx, np.arange(4))
    with np.load(cache, allow_pickle=True) as payload:
        assert str(payload["per_cell_mode"]) == "all"
        assert str(payload["span_n_mode"]) == "all"


def test_strict_cache_rejects_selection_mode_mismatch(tmp_path: Path) -> None:
    cache = tmp_path / "probe_rows_v2.npz"
    cells = np.empty(2, dtype=object)
    cells[0] = np.array([0], dtype=np.int64)
    cells[1] = np.array([1], dtype=np.int64)
    np.savez_compressed(
        cache,
        cell_keys=np.array([(4, "SURV"), (4, "FAIL")]),
        cell_idx=cells,
        span_idx=np.array([0, 1], dtype=np.int64),
        rhash="same-hash",
        per_cell_mode="2000",
        span_n_mode="20000",
    )

    with pytest.raises(ValueError, match="selection mode mismatch"):
        select_rows(
            bins=np.array([4, 4]),
            ep_id=np.array([0, 1]),
            term_bin=np.array([4, 4]),
            term_bad=np.array([False, True]),
            wall_bins=[4],
            per_cell=None,
            span_n=2,
            cache=str(cache),
            rhash="same-hash",
            strict_cache=True,
        )


def test_non_strict_cache_reselects_on_selection_mode_mismatch(tmp_path: Path) -> None:
    cache = tmp_path / "probe_rows_v2.npz"
    cached_cells = np.empty(2, dtype=object)
    cached_cells[0] = np.array([0], dtype=np.int64)
    cached_cells[1] = np.array([1], dtype=np.int64)
    np.savez_compressed(
        cache,
        cell_keys=np.array([(4, "SURV"), (4, "FAIL")]),
        cell_idx=cached_cells,
        span_idx=np.array([0, 1], dtype=np.int64),
        rhash="same-hash",
        per_cell_mode="2000",
        span_n_mode="20000",
    )

    cells, span_idx = select_rows(
        bins=np.array([4, 4, 4, 4]),
        ep_id=np.array([0, 1, 2, 3]),
        term_bin=np.array([8, 4, 8, 4]),
        term_bad=np.array([False, True, False, True]),
        wall_bins=[4],
        per_cell=None,
        span_n=4,
        cache=str(cache),
        rhash="same-hash",
    )

    np.testing.assert_array_equal(cells[(4, "SURV")], np.array([0, 2]))
    np.testing.assert_array_equal(cells[(4, "FAIL")], np.array([1, 3]))
    np.testing.assert_array_equal(span_idx, np.arange(4))
    with np.load(cache, allow_pickle=True) as payload:
        assert str(payload["per_cell_mode"]) == "all"
        assert str(payload["span_n_mode"]) == "all"


def test_validate_selected_rows_rejects_swapped_cell_indices() -> None:
    bins = np.array([4, 4, 4, 4], dtype=np.int64)
    ep_id = np.array([0, 1, 2, 3], dtype=np.int64)
    term_bin = np.array([8, 4, 8, 4], dtype=np.int64)
    term_bad = np.array([False, True, False, True])
    swapped_cells = {
        (4, "SURV"): np.array([1, 3], dtype=np.int64),
        (4, "FAIL"): np.array([0, 2], dtype=np.int64),
    }

    with pytest.raises(ValueError, match="does not exactly match"):
        validate_selected_rows(
            bins,
            ep_id,
            term_bin,
            term_bad,
            [4],
            swapped_cells,
            np.arange(4, dtype=np.int64),
            require_full_cells=True,
            require_full_span=True,
        )


def test_validate_selected_rows_accepts_exact_full_cells() -> None:
    bins = np.array([4, 4, 4, 4], dtype=np.int64)
    ep_id = np.array([0, 1, 2, 3], dtype=np.int64)
    term_bin = np.array([8, 4, 8, 4], dtype=np.int64)
    term_bad = np.array([False, True, False, True])
    cells = {
        (4, "SURV"): np.array([0, 2], dtype=np.int64),
        (4, "FAIL"): np.array([1, 3], dtype=np.int64),
    }

    validate_selected_rows(
        bins,
        ep_id,
        term_bin,
        term_bad,
        [4],
        cells,
        np.arange(4, dtype=np.int64),
        require_full_cells=True,
        require_full_span=True,
    )
