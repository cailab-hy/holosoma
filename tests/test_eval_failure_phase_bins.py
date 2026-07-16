from __future__ import annotations

from holosoma.eval_agent import _bad_tracking_phase_bin_counts, _phase_bin_summary


def test_bad_tracking_phase_bins_include_right_endpoint():
    results = [
        {"stop_reason": "bad_tracking", "terminal_motion_phase": 0.0},
        {"stop_reason": "bad_tracking", "terminal_motion_phase": 0.249},
        {"stop_reason": "bad_tracking", "terminal_motion_phase": 0.25},
        {"stop_reason": "bad_tracking", "terminal_motion_phase": 1.0},
        {"stop_reason": "motion_ends", "terminal_motion_phase": 0.5},
        {"stop_reason": "bad_tracking"},
    ]

    counts, unresolved = _bad_tracking_phase_bin_counts(results, num_bins=4)
    count_summary, percentage_summary = _phase_bin_summary(counts, bad_tracking_total=5)

    assert counts == [2, 1, 0, 1]
    assert unresolved == 1
    assert count_summary == {
        "bin00[0.00,0.25)": 2,
        "bin01[0.25,0.50)": 1,
        "bin03[0.75,1.00]": 1,
    }
    assert percentage_summary["bin00[0.00,0.25)"] == 40.0
