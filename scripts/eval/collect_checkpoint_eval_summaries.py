#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


def _parse_step(text: str) -> int | None:
    match = re.search(r"(\d+)", text)
    return int(match.group(1)) if match else None


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file() or path.stat().st_size == 0:
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _load_text(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text().strip()


def collect_rows(output_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result_dir in sorted([p for p in output_dir.iterdir() if p.is_dir()]):
        summary_path = result_dir / "eval_summary.json"
        console_log = result_dir / "console.log"
        status_path = result_dir / "eval_status.txt"
        invocation_path = result_dir / "eval_invocation.json"
        failure_path = result_dir / "eval_failure.json"

        summary = _load_json(summary_path)
        invocation = _load_json(invocation_path) or {}
        failure = _load_json(failure_path) or {}
        status_txt = _load_text(status_path)

        checkpoint_path = ""
        checkpoint = result_dir.name
        checkpoint_step: int | None = None
        eval_status = "missing"

        if summary is not None:
            checkpoint_path = str(summary.get("checkpoint", invocation.get("checkpoint_path", "")))
            checkpoint = Path(checkpoint_path).name if checkpoint_path else result_dir.name
            checkpoint_step = summary.get("checkpoint_step")
            eval_status = "ok"
        else:
            checkpoint_path = str(invocation.get("checkpoint_path", ""))
            checkpoint = Path(checkpoint_path).name if checkpoint_path else result_dir.name
            checkpoint_step = _parse_step(checkpoint)
            if status_txt:
                eval_status = status_txt
            if eval_status == "ok":
                eval_status = "fail_missing_summary"

        if checkpoint_step is None:
            checkpoint_step = _parse_step(checkpoint)

        row = {
            "checkpoint": checkpoint,
            "checkpoint_path": checkpoint_path,
            "checkpoint_step": checkpoint_step,
            "eval_status": eval_status,
            "num_envs": None,
            "envs_finished": None,
            "envs_unfinished": None,
            "success_count": None,
            "success_rate": None,
            "success_std": None,
            "success_rate_finished": None,
            "failure_count": None,
            "timeout_count": None,
            "bad_tracking_count": None,
            "max_eval_steps_unfinished_count": None,
            "episode_reward_sum_mean": None,
            "episode_reward_sum_std": None,
            "episode_discounted_return_mean": None,
            "episode_discounted_return_std": None,
            "episode_length_mean": None,
            "episode_length_std": None,
            "return_gamma": None,
            "eval_results_dir": str(result_dir),
            "console_log": str(console_log),
            "failure_reason": "",
            "error_excerpt": "",
            "cmd_status": None,
            "attempts": None,
        }
        if summary is not None:
            for key in list(row):
                if key in summary and key != "checkpoint":
                    row[key] = summary[key]
        else:
            # Keep key bookkeeping fields populated even when eval summary is missing.
            row["num_envs"] = invocation.get("num_envs")
            row["checkpoint_step"] = invocation.get("checkpoint_step", checkpoint_step)

        # Attach failure diagnostics when present.
        if failure:
            row["failure_reason"] = failure.get("failure_reason", "")
            row["error_excerpt"] = failure.get("error_excerpt", "")
            row["cmd_status"] = failure.get("cmd_status")
            row["attempts"] = failure.get("attempts")

        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect per-checkpoint eval summaries into batch CSV/JSON.")
    parser.add_argument("output_dir", help="Directory containing per-checkpoint result folders")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_dir():
        raise SystemExit(f"output_dir not found: {output_dir}")

    rows = collect_rows(output_dir)
    csv_path = output_dir / "batch_eval_summary.csv"
    json_path = output_dir / "batch_eval_summary.json"

    fieldnames = [
        "checkpoint",
        "checkpoint_path",
        "checkpoint_step",
        "eval_status",
        "num_envs",
        "envs_finished",
        "envs_unfinished",
        "success_count",
        "success_rate",
        "success_std",
        "success_rate_finished",
        "failure_count",
        "timeout_count",
        "bad_tracking_count",
        "max_eval_steps_unfinished_count",
        "episode_reward_sum_mean",
        "episode_reward_sum_std",
        "episode_discounted_return_mean",
        "episode_discounted_return_std",
        "episode_length_mean",
        "episode_length_std",
        "return_gamma",
        "failure_reason",
        "error_excerpt",
        "cmd_status",
        "attempts",
        "eval_results_dir",
        "console_log",
    ]

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    with json_path.open("w") as f:
        json.dump(rows, f, indent=2)

    print(f"[Batch Eval] wrote {csv_path}")
    print(f"[Batch Eval] wrote {json_path}")


if __name__ == "__main__":
    main()
