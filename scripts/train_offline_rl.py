#!/usr/bin/env python
"""Opt-in offline-RL runner (Step 7).

Thin wrapper around the **legacy** training path
(``src/holosoma/holosoma/train_agent.py`` + ``OfflineCQLAgent``).
The wrapper:

* resolves an algorithm key against the offline-RL algorithm registry;
* resolves a dataset key against the dataset registry;
* loads the recommended YAML config example (or a caller-supplied
  config path) and forwards its ``hyperparameters`` block onto the
  legacy CLI as ``--algo.config.<kebab-case-key> <value>`` flags;
* renders the resulting command in dry-run mode (default), or invokes
  ``subprocess.run`` when the user passes ``--run``.

This script does **not** introduce a new training loop.  It is a
documentation/CI tool — production launches should keep using the
replication shell scripts under ``scripts/train_replication/``.

Usage
-----
::

    # dry-run (default): print the legacy command and exit
    python scripts/train_offline_rl.py \
        --algorithm smqr_sg --dataset wbt_object \
        --seed 1 --run-tag exp_step7_dryrun --num-iters 50

    # invoke the legacy train path
    python scripts/train_offline_rl.py \
        --algorithm cql --dataset wbt_object \
        --seed 1 --run-tag exp_step7_cql_test --num-iters 50 \
        --save-interval 50 --run
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
# Make src/ importable when run from a fresh shell.
sys.path.insert(0, str(REPO_ROOT / "src" / "holosoma"))

from holosoma.agents.offline_rl.algorithms.legacy_adapter import (  # noqa: E402
    build_legacy_adapter,
    render_shell_command,
)
from holosoma.agents.offline_rl.algorithms.registry import (  # noqa: E402
    get_algorithm_entry,
    list_algorithms,
)
from holosoma.agents.offline_rl.datasets.registry import (  # noqa: E402
    get_dataset_entry,
    list_datasets,
)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="train_offline_rl",
        description=(
            "Opt-in runner that resolves the offline-RL registry and "
            "invokes the legacy train_agent.py path.  Defaults to "
            "dry-run; pass --run to actually launch training."
        ),
    )
    p.add_argument(
        "--algorithm",
        required=True,
        help=f"algorithm registry key (one of: {list_algorithms()})",
    )
    p.add_argument(
        "--dataset",
        default=None,
        help=(
            "dataset registry key; defaults to the algorithm's "
            f"default_dataset_key (registered: {list_datasets()})"
        ),
    )
    p.add_argument(
        "--config",
        default=None,
        help=(
            "Optional YAML config path overriding the registered "
            "recommended_config_example.  ``hyperparameters`` keys are "
            "forwarded onto --algo.config.<kebab-key> <value>."
        ),
    )
    p.add_argument("--run-tag", default="exp_offline_rl_runner",
                   help="--training.name forwarded to train_agent.py")
    p.add_argument("--seed", type=int, default=1,
                   help="--training.seed forwarded to train_agent.py")
    p.add_argument("--num-iters", type=int, default=None,
                   help="overrides hyperparameters.num_learning_iterations")
    p.add_argument("--save-interval", type=int, default=None,
                   help="overrides hyperparameters.save_interval")
    p.add_argument("--dataset-path", default="offline_data/fastsac_dataset.h5",
                   help="--algo.config.dataset-path forwarded verbatim")
    p.add_argument("--logger", default="wandb",
                   help="logger:<choice> token (e.g. 'wandb' or 'disabled')")
    p.add_argument("--python-bin", default=sys.executable,
                   help="python interpreter used to invoke train_agent.py")
    p.add_argument(
        "--run",
        action="store_true",
        help="actually launch training (default is dry-run only)",
    )
    return p


def _load_hyperparameters(config_path: Path | None) -> dict[str, Any]:
    if config_path is None:
        return {}
    if not config_path.exists():
        raise FileNotFoundError(f"config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    hp = cfg.get("hyperparameters") or {}
    if not isinstance(hp, dict):
        raise ValueError(
            f"hyperparameters block in {config_path} must be a mapping "
            f"(got {type(hp).__name__})"
        )
    # Drop the two well-known ``--training.*`` knobs that the runner
    # forwards via dedicated flags; everything else goes through
    # --algo.config.<kebab-key>.
    hp = {k: v for k, v in hp.items() if k not in {
        "num_learning_iterations",
        "save_interval",
    }}
    return hp


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    # ── Step 1: resolve registries (validates placeholders early) ──
    algo_entry = get_algorithm_entry(args.algorithm)
    if algo_entry.status == "placeholder":
        print(
            f"[runner] algorithm '{args.algorithm}' is a placeholder "
            f"(no implementation).",
            file=sys.stderr,
        )
        print(
            f"[runner]   description: {algo_entry.description}",
            file=sys.stderr,
        )
        return 2

    dataset_key = args.dataset or algo_entry.default_dataset_key
    if dataset_key is None:
        print(
            f"[runner] algorithm '{args.algorithm}' has no default_dataset_key; "
            f"pass --dataset explicitly.",
            file=sys.stderr,
        )
        return 2
    dataset_entry = get_dataset_entry(dataset_key)
    if dataset_entry.status == "placeholder":
        print(
            f"[runner] dataset '{dataset_key}' is a placeholder "
            f"(no loader available).",
            file=sys.stderr,
        )
        return 2

    # ── Step 2: load recommended (or user-supplied) hyperparameters ─
    config_path: Path | None
    if args.config:
        config_path = Path(args.config).resolve()
    elif algo_entry.recommended_config_example:
        config_path = REPO_ROOT / algo_entry.recommended_config_example
    else:
        config_path = None

    overrides = _load_hyperparameters(config_path)

    # ── Step 3: assemble adapter + final argv ──────────────────────
    adapter = build_legacy_adapter(args.algorithm, dataset_key)

    cfg_path_str = config_path.relative_to(REPO_ROOT) if config_path else None
    hp_yaml = (
        yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        if config_path is not None
        else None
    )
    num_iters = (
        args.num_iters
        if args.num_iters is not None
        else (hp_yaml or {}).get("hyperparameters", {}).get(
            "num_learning_iterations", 50
        )
    )
    save_interval = (
        args.save_interval
        if args.save_interval is not None
        else (hp_yaml or {}).get("hyperparameters", {}).get(
            "save_interval", num_iters
        )
    )

    argv_full = adapter.build_train_command(
        run_tag=args.run_tag,
        seed=args.seed,
        num_iters=num_iters,
        save_interval=save_interval,
        dataset_path=args.dataset_path,
        python_bin=args.python_bin,
        logger=args.logger,
        overrides=overrides,
    )

    # ── Step 4: report ─────────────────────────────────────────────
    print("=" * 70)
    print(f" Offline-RL runner — dry-run={'NO' if args.run else 'YES'}")
    print(f"   algorithm   : {algo_entry.name}  (family={algo_entry.family})")
    print(f"   dataset     : {dataset_entry.name}  (status={dataset_entry.status})")
    print(f"   legacy agent: {adapter.legacy_agent_class_path}")
    print(f"   preset      : {adapter.preset}")
    print(f"   run tag     : {args.run_tag}  (seed={args.seed})")
    print(f"   iters       : {num_iters}  (save_interval={save_interval})")
    print(f"   config      : {cfg_path_str or '<none>'}")
    if algo_entry.train_script_reference:
        print(f"   ref script  : {algo_entry.train_script_reference}")
    if algo_entry.eval_manifest_reference:
        print(f"   eval ref    : {algo_entry.eval_manifest_reference}")
    print("-" * 70)
    print(" command:")
    print(f"  {render_shell_command(argv_full)}")
    print("=" * 70)

    if not args.run:
        return 0

    # Actually invoke train_agent.py.  We CD to the repo root so that
    # the preset and relative paths resolve correctly.
    return subprocess.run(argv_full, cwd=str(REPO_ROOT), check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
