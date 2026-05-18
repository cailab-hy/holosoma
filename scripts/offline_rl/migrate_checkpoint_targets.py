"""Step 8 — copy-on-write migration helper for legacy ``_target_`` strings.

Scan checkpoint ``.pt`` files and / or sibling ``holosoma_config.yaml``
files, detect the legacy ``OfflineCQLAgent`` ``_target_`` string, and
emit (a) a migration manifest describing the recommended direct
target and (b) — optionally — a *copy* of each input file with the
target string updated.

Strict policy
-------------
* **Default = dry-run.**  No filesystem writes other than the manifest.
* **In-place rewrite is forbidden.**  ``--write`` requires
  ``--out-dir <path>`` and the script will refuse if any output path
  would collide with an input path.
* Output files are always *copies*: original checkpoints and configs
  are never modified.
* Unresolved cases (unknown ``critic_penalty_mode`` / ``smqr_lse_mode``,
  missing config fields, etc.) are reported as ``unresolved`` and
  **skipped** — they are never written.

Usage
-----
Dry-run (default)::

    python scripts/offline_rl/migrate_checkpoint_targets.py \
        --input logs/hv-g1-manager/exp_80_perf_cql_seed1_bs4096_300k

Copy-on-write::

    python scripts/offline_rl/migrate_checkpoint_targets.py \
        --input logs/hv-g1-manager/exp_80_perf_cql_seed1_bs4096_300k \
        --out-dir /tmp/migrated_step8 \
        --write
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


# ── Constants (mirror target_compat module to avoid runtime imports) ──

LEGACY_TARGET = (
    "holosoma.agents.offline_cql.offline_cql_agent.OfflineCQLAgent"
)


@dataclass
class MigrationRecord:
    source_path: str
    file_type: str  # "checkpoint" | "yaml"
    old_target: str | None
    new_target: str | None
    reason: str
    status: str  # "dry-run" | "written" | "skipped" | "unresolved" | "no-op"
    output_path: str | None = None
    error: str | None = None


# ── File discovery ───────────────────────────────────────────────────


def _iter_files(input_path: Path) -> list[Path]:
    """Walk *input_path* and return ``.pt`` + ``holosoma_config.yaml`` files."""
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"--input not found: {input_path}")

    out: list[Path] = []
    for root, _dirs, files in os.walk(input_path):
        for f in files:
            p = Path(root) / f
            if p.suffix == ".pt" or p.name == "holosoma_config.yaml":
                out.append(p)
    return sorted(out)


# ── YAML scanning ────────────────────────────────────────────────────

_YAML_ALGO_TARGET_RE = re.compile(
    # Match a `_target_:` line under the `algo:` block.  Greedy enough
    # to find the first instance; we only care about the algo-level
    # target.  Other modules (env, sim, etc.) carry their own _target_
    # and are intentionally left alone.
    r"^(?P<indent>\s+)_target_:\s*(?P<value>\S+)\s*$",
    re.MULTILINE,
)

_YAML_KEY_RE = re.compile(
    r"^(?P<indent>\s*)(?P<key>[\w_]+):\s*(?P<value>\S+)\s*$",
    re.MULTILINE,
)


def _extract_yaml_fields(yaml_text: str) -> dict[str, Any]:
    """Pull ``critic_penalty_mode`` + ``smqr_lse_mode`` out of a config YAML.

    We use a regex pass rather than full YAML parsing to avoid forcing
    a pyyaml dependency at migration time and to be robust against
    Hydra-style references / interpolations.
    """
    fields: dict[str, Any] = {}
    for m in _YAML_KEY_RE.finditer(yaml_text):
        key = m.group("key")
        if key in ("critic_penalty_mode", "smqr_lse_mode") and key not in fields:
            value = m.group("value").strip("'\"")
            fields[key] = value
    return fields


def _find_algo_target_in_yaml(yaml_text: str) -> tuple[str, int, int] | None:
    """Return the first ``_target_`` value under the ``algo:`` block.

    Returns ``(value, match_start, match_end)`` or ``None``.
    We look for the first ``_target_:`` line whose *line* is preceded
    by an ``algo:`` block (cheaply: the first ``_target_:`` that lies
    within ~30 lines of an ``algo:`` line).
    """
    lines = yaml_text.splitlines(keepends=True)
    offsets: list[int] = []
    cur = 0
    for line in lines:
        offsets.append(cur)
        cur += len(line)
    offsets.append(cur)

    algo_line_idx: int | None = None
    for i, line in enumerate(lines):
        if re.match(r"^algo:\s*$", line):
            algo_line_idx = i
            break
    if algo_line_idx is None:
        return None

    for j in range(algo_line_idx + 1, min(algo_line_idx + 50, len(lines))):
        m = re.match(r"^(\s+)_target_:\s*(\S+)\s*$", lines[j])
        if m:
            value = m.group(2)
            start = offsets[j] + m.start(2)
            end = offsets[j] + m.end(2)
            return value, start, end
    return None


def _scan_yaml(path: Path) -> MigrationRecord:
    text = path.read_text(encoding="utf-8")
    hit = _find_algo_target_in_yaml(text)
    if hit is None:
        return MigrationRecord(
            source_path=str(path),
            file_type="yaml",
            old_target=None,
            new_target=None,
            reason="no algo._target_ found",
            status="no-op",
        )
    old_target, _, _ = hit
    if old_target != LEGACY_TARGET:
        return MigrationRecord(
            source_path=str(path),
            file_type="yaml",
            old_target=old_target,
            new_target=old_target,
            reason="already direct or unknown — no rewrite",
            status="no-op",
        )

    cfg_fields = _extract_yaml_fields(text)
    try:
        # Local import keeps migration script importable without torch.
        from holosoma.agents.offline_rl.common.target_compat import (
            migrate_target_string,
        )

        new_target, reason = migrate_target_string(old_target, cfg_fields)
    except Exception as exc:  # pragma: no cover - defensive
        return MigrationRecord(
            source_path=str(path),
            file_type="yaml",
            old_target=old_target,
            new_target=None,
            reason="unable to recommend direct target",
            status="unresolved",
            error=str(exc),
        )

    return MigrationRecord(
        source_path=str(path),
        file_type="yaml",
        old_target=old_target,
        new_target=new_target,
        reason=reason,
        status="dry-run",
    )


def _rewrite_yaml(text: str, new_target: str) -> str:
    """Replace the algo-block ``_target_`` value, leaving everything else."""
    hit = _find_algo_target_in_yaml(text)
    if hit is None:
        return text
    _, start, end = hit
    return text[:start] + new_target + text[end:]


# ── Checkpoint scanning ──────────────────────────────────────────────


def _scan_checkpoint(path: Path) -> MigrationRecord:
    try:
        import torch  # local import — script must run without torch for YAML scan
    except ImportError:
        return MigrationRecord(
            source_path=str(path),
            file_type="checkpoint",
            old_target=None,
            new_target=None,
            reason="torch not installed",
            status="unresolved",
            error="torch import failed",
        )

    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        return MigrationRecord(
            source_path=str(path),
            file_type="checkpoint",
            old_target=None,
            new_target=None,
            reason="could not torch.load checkpoint",
            status="unresolved",
            error=str(exc),
        )

    ec = ck.get("experiment_config") if isinstance(ck, dict) else None
    if not isinstance(ec, dict):
        return MigrationRecord(
            source_path=str(path),
            file_type="checkpoint",
            old_target=None,
            new_target=None,
            reason="no experiment_config in checkpoint",
            status="no-op",
        )
    algo = ec.get("algo")
    if not isinstance(algo, dict):
        return MigrationRecord(
            source_path=str(path),
            file_type="checkpoint",
            old_target=None,
            new_target=None,
            reason="no experiment_config.algo block",
            status="no-op",
        )

    old_target = algo.get("_target_")
    if old_target != LEGACY_TARGET:
        return MigrationRecord(
            source_path=str(path),
            file_type="checkpoint",
            old_target=old_target,
            new_target=old_target,
            reason="already direct or unknown — no rewrite",
            status="no-op",
        )

    cfg = algo.get("config", {})
    try:
        from holosoma.agents.offline_rl.common.target_compat import (
            migrate_target_string,
        )

        new_target, reason = migrate_target_string(old_target, cfg)
    except Exception as exc:
        return MigrationRecord(
            source_path=str(path),
            file_type="checkpoint",
            old_target=old_target,
            new_target=None,
            reason="unable to recommend direct target",
            status="unresolved",
            error=str(exc),
        )

    return MigrationRecord(
        source_path=str(path),
        file_type="checkpoint",
        old_target=old_target,
        new_target=new_target,
        reason=reason,
        status="dry-run",
    )


def _rewrite_checkpoint(src: Path, dst: Path, new_target: str) -> None:
    import torch

    ck = torch.load(src, map_location="cpu", weights_only=False)
    if not isinstance(ck, dict):
        raise ValueError("checkpoint is not a dict — refusing to rewrite")
    ec = ck.get("experiment_config")
    if not isinstance(ec, dict) or "algo" not in ec:
        raise ValueError("checkpoint has no experiment_config.algo")
    ec["algo"]["_target_"] = new_target
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, dst)


# ── Driver ───────────────────────────────────────────────────────────


def _resolve_output_path(
    src: Path, input_root: Path, out_dir: Path
) -> Path:
    """Compose ``out_dir / <relative-path-from-input-root>``."""
    try:
        rel = src.relative_to(input_root)
    except ValueError:
        rel = Path(src.name)
    return out_dir / rel


def run_migration(args: argparse.Namespace) -> tuple[list[MigrationRecord], int]:
    input_path = Path(args.input).resolve()
    files = _iter_files(input_path)

    input_root = input_path if input_path.is_dir() else input_path.parent

    if args.write:
        if not args.out_dir:
            raise SystemExit(
                "--write requires --out-dir <path> "
                "(in-place rewrites are forbidden by policy)"
            )
        out_dir = Path(args.out_dir).resolve()
        # Reject in-place writes: out_dir must not equal or live inside
        # input_root and vice-versa.
        if out_dir == input_root or out_dir in input_root.parents:
            raise SystemExit(
                f"--out-dir {out_dir} would collide with --input {input_root}; "
                "copy-on-write requires a disjoint output directory."
            )
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = None

    records: list[MigrationRecord] = []
    for src in files:
        if src.suffix == ".pt":
            rec = _scan_checkpoint(src)
        else:
            rec = _scan_yaml(src)

        if args.write and rec.status == "dry-run" and rec.new_target:
            try:
                dst = _resolve_output_path(src, input_root, out_dir)  # type: ignore[arg-type]
                if dst.resolve() == src.resolve():
                    raise RuntimeError(
                        "refusing to overwrite source file (in-place rewrite)"
                    )
                if src.name == "holosoma_config.yaml":
                    text = src.read_text(encoding="utf-8")
                    new_text = _rewrite_yaml(text, rec.new_target)
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    dst.write_text(new_text, encoding="utf-8")
                else:
                    _rewrite_checkpoint(src, dst, rec.new_target)
                rec.status = "written"
                rec.output_path = str(dst)
            except Exception as exc:  # pragma: no cover - defensive
                rec.status = "unresolved"
                rec.error = f"write failed: {exc}"

        records.append(rec)

    return records, len(files)


def _write_manifest(
    records: list[MigrationRecord],
    out_dir: Path | None,
    args: argparse.Namespace,
) -> Path:
    ts = _dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    if out_dir is not None:
        manifest_path = out_dir / f"migration_manifest_{ts}.json"
    elif args.manifest:
        manifest_path = Path(args.manifest).resolve()
    else:
        manifest_path = Path.cwd() / f"migration_manifest_{ts}.json"

    payload = {
        "step": "Step 8 — _target_ compatibility migration",
        "input": str(Path(args.input).resolve()),
        "out_dir": str(Path(args.out_dir).resolve()) if args.out_dir else None,
        "mode": "write" if args.write else "dry-run",
        "timestamp": ts,
        "records": [asdict(r) for r in records],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


def _summarise(records: list[MigrationRecord]) -> str:
    counts: dict[str, int] = {}
    for r in records:
        counts[r.status] = counts.get(r.status, 0) + 1
    parts = [f"{k}={v}" for k, v in sorted(counts.items())]
    return "  ".join(parts) if parts else "(no files scanned)"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="migrate_checkpoint_targets",
        description=(
            "Step 8 copy-on-write migration helper for legacy "
            "OfflineCQLAgent _target_ strings."
        ),
    )
    parser.add_argument(
        "--input",
        required=True,
        help="path to a checkpoint .pt, a holosoma_config.yaml, or "
        "a directory containing them",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="output directory for copy-on-write rewrites (required "
        "if --write is set; must be disjoint from --input)",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="actually emit migrated copies (default: dry-run)",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="path to write the migration manifest JSON (default: "
        "<out-dir>/migration_manifest_<ts>.json or CWD)",
    )
    args = parser.parse_args(argv)

    records, n_files = run_migration(args)
    out_dir = Path(args.out_dir).resolve() if args.out_dir else None
    manifest = _write_manifest(records, out_dir, args)

    print(f"Step 8 migration ({'write' if args.write else 'dry-run'} mode)")
    print(f"  input:  {args.input}")
    if out_dir:
        print(f"  out-dir: {out_dir}")
    print(f"  scanned: {n_files} files")
    print(f"  summary: { _summarise(records) }")
    print(f"  manifest: {manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
