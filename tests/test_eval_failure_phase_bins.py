from __future__ import annotations

from holosoma.utils.eval_phase_utils import (
    bad_tracking_phase_bin_counts,
    bad_tracking_phase_metrics,
    phase_bin_summary,
)


def test_bad_tracking_phase_bins_include_right_endpoint():
    results = [
        {"stop_reason": "bad_tracking", "terminal_motion_phase": 0.0},
        {"stop_reason": "bad_tracking", "terminal_motion_phase": 0.249},
        {"stop_reason": "bad_tracking", "terminal_motion_phase": 0.25},
        {"stop_reason": "bad_tracking", "terminal_motion_phase": 1.0},
        {"stop_reason": "motion_ends", "terminal_motion_phase": 0.5},
        {"stop_reason": "bad_tracking"},
    ]

    counts, unresolved = bad_tracking_phase_bin_counts(results, num_bins=4)
    count_summary, percentage_summary = phase_bin_summary(counts, bad_tracking_total=5)
    metrics, metric_unresolved = bad_tracking_phase_metrics(results, num_bins=4)

    assert counts == [2, 1, 0, 1]
    assert unresolved == 1
    assert count_summary == {
        "bin00[0.00,0.25)": 2,
        "bin01[0.25,0.50)": 1,
        "bin03[0.75,1.00]": 1,
    }
    assert percentage_summary["bin00[0.00,0.25)"] == 40.0
    assert metrics == {
        "Bad_tracking/phase_0.00_0.25": 40.0,
        "Bad_tracking/phase_0.25_0.50": 20.0,
        "Bad_tracking/phase_0.50_0.75": 0.0,
        "Bad_tracking/phase_0.75_1.00": 20.0,
    }
    assert metric_unresolved == 1


def test_default_twenty_bins_use_requested_metric_names():
    results = [{"stop_reason": "bad_tracking", "terminal_motion_phase": 0.031}]

    metrics, unresolved = bad_tracking_phase_metrics(results, num_bins=20)

    assert metrics["Bad_tracking/phase_0.00_0.05"] == 100.0
    assert metrics["Bad_tracking/phase_0.05_0.10"] == 0.0
    assert unresolved == 0
