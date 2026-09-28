#!/usr/bin/env python3
"""Delete model files (checkpoints / ONNX) from Weights & Biases runs, keeping metrics and configs.

holosoma used to upload every ``model_*.pt`` checkpoint to the run (see
``--logger.upload-model-files``); this removes those files from existing runs so only the
logged metrics (graphs), run config and media stay. Local files are never touched.

Examples::

    # what would be deleted, per project (nothing is removed)
    python scripts/wandb_delete_model_files.py --entity my-entity --project WholeBodyTracking --dry-run

    # delete checkpoints (.pt) from every run of the project
    python scripts/wandb_delete_model_files.py --entity my-entity --project WholeBodyTracking --yes

    # also delete exported .onnx files, across several projects
    python scripts/wandb_delete_model_files.py --entity my-entity --project WholeBodyTracking --project CORL \\
        --include-onnx --yes
"""

from __future__ import annotations

import argparse
import fnmatch
import sys
from dataclasses import dataclass, field

import wandb

DEFAULT_PATTERNS = ["model_*.pt", "**/model_*.pt"]
ONNX_PATTERNS = ["*.onnx", "**/*.onnx"]


@dataclass
class RunReport:
    run_path: str
    run_name: str
    files: list = field(default_factory=list)  # wandb.apis.public.File objects

    @property
    def num_bytes(self) -> int:
        return sum(int(f.size or 0) for f in self.files)


def _matches(name: str, patterns: list[str]) -> bool:
    base = name.rsplit("/", 1)[-1]
    return any(fnmatch.fnmatch(name, p) or fnmatch.fnmatch(base, p) for p in patterns)


def scan_project(api: wandb.Api, entity: str, project: str, patterns: list[str]) -> list[RunReport]:
    reports = []
    for run in api.runs(f"{entity}/{project}", per_page=200):
        matched = [f for f in run.files(per_page=1000) if _matches(f.name, patterns)]
        if matched:
            reports.append(RunReport(run_path="/".join(run.path), run_name=run.name, files=matched))
    return reports


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--entity", default=None, help="wandb entity (default: API default entity)")
    parser.add_argument("--project", action="append", required=True, help="project name (repeatable)")
    parser.add_argument("--pattern", action="append", default=None, help="file glob to delete (default: model_*.pt)")
    parser.add_argument("--include-onnx", action="store_true", help="also delete exported *.onnx files")
    parser.add_argument("--dry-run", action="store_true", help="list matching files without deleting")
    parser.add_argument("--yes", action="store_true", help="delete without interactive confirmation")
    args = parser.parse_args(argv)

    api = wandb.Api()
    entity = args.entity or api.default_entity
    patterns = list(args.pattern or DEFAULT_PATTERNS)
    if args.include_onnx:
        patterns += ONNX_PATTERNS

    total_files = 0
    total_bytes = 0
    all_reports: list[tuple[str, RunReport]] = []
    for project in args.project:
        reports = scan_project(api, entity, project, patterns)
        project_bytes = sum(r.num_bytes for r in reports)
        project_files = sum(len(r.files) for r in reports)
        print(
            f"[{entity}/{project}] runs with model files: {len(reports)}, files: {project_files}, "
            f"size: {project_bytes / 1e9:.2f} GB"
        )
        for r in reports:
            print(f"  {r.run_name:70s} {len(r.files):4d} files {r.num_bytes / 1e9:6.2f} GB")
        total_files += project_files
        total_bytes += project_bytes
        all_reports += [(project, r) for r in reports]
    print(f"TOTAL: {total_files} files, {total_bytes / 1e9:.2f} GB matching {patterns}")

    if args.dry_run or total_files == 0:
        return 0
    if not args.yes:
        answer = input("Delete these files from wandb? Metrics/config/media are kept. [y/N] ")
        if answer.strip().lower() != "y":
            print("aborted")
            return 1

    deleted = 0
    failed = 0
    for project, report in all_reports:
        for f in report.files:
            try:
                f.delete()
                deleted += 1
            except Exception as exc:  # noqa: PERF203  # keep going; report at the end
                failed += 1
                print(f"  failed: {report.run_path}/{f.name}: {exc}", file=sys.stderr)
        print(f"  cleaned {project}/{report.run_name}")
    print(f"deleted {deleted} files ({total_bytes / 1e9:.2f} GB), {failed} failures")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
