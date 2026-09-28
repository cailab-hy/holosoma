#!/usr/bin/env python3
"""Post-hoc fixed-probe Q-level curves from saved checkpoints (no simulator needed).

For every ``model_*.pt`` of each run, the critic, actor and observation normalizers are
rebuilt from the checkpoint alone and evaluated on the fixed probe set of the run's dataset
(``holosoma.agents.modules.q_probe``; the same ``q_probe_seed`` / size that live training
logs as ``Loss/probe/*``). This recovers, for runs trained before the probe metrics existed:

    Q_level(t)   = mean min(Q1, Q2)(s_probe, a_probe)      -> probe/q_level
    LSE_level(t) = mean 0.5 (LSE1 + LSE2)(s_probe)         -> probe/q_lse
    bracket(t)   = mean 0.5 [(LSE1 - Q1) + (LSE2 - Q2)]    -> probe/lse_minus_q_data

Eval success (``Eval/stop_reason_percent/motion_ends``) is joined from the run's TensorBoard
log by step, so Q levels can be reported at the peak-success checkpoint, at the final
checkpoint and as a final-k mean.

Outputs (``--out-dir``): q_level_runs.csv, q_level_summary.csv, q_level_summary.md,
q_level_curves.png. With ``--paired-baseline <method>`` also the paired displacement
DeltaQ(run, t) = Q_level(run, seed s, t) - Q_level(baseline, seed s, t) per checkpoint:
q_level_paired.csv, q_level_paired_summary.{csv,md}, q_level_paired.png.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import summarize_validation_results as S

from holosoma.agents.cql.cql import Actor, DoubleQCritic
from holosoma.agents.cql.cql_utils import EmpiricalNormalization
from holosoma.agents.modules.q_probe import compute_q_probe_stats, load_probe_set, select_probe_indices
from holosoma.utils.safe_torch_import import torch

CKPT_RE = re.compile(r"model_(\d+)\.pt$")
STAT_KEYS = ("probe/q_level", "probe/q_lse", "probe/lse_minus_q_data", "probe/q_pi", "probe/q_curr", "probe/q_rand")
DEFAULT_METHODS = ["g1_29dof_wbt_cql", "g1_29dof_wbt_aw_cql_H50", "g1_29dof_wbt_b_arm"]


def _indices(dim: int) -> dict[str, dict[str, int]]:
    return {"all": {"start": 0, "end": dim, "size": dim}}


def _normalizer(state, device) -> torch.nn.Module:
    if not isinstance(state, dict) or "_mean" not in state:
        return torch.nn.Identity()
    dim = int(state["_mean"].shape[-1])
    norm = EmpiricalNormalization(shape=dim, device=device)
    norm.load_state_dict(state)
    norm.eval()
    return norm


class CheckpointModels:
    """Critic / actor rebuilt from a checkpoint's ``args`` and state-dict shapes."""

    def __init__(self, ckpt: dict, device: torch.device):
        args = ckpt["args"]
        qsd, asd = ckpt["qnet_state_dict"], ckpt["actor_state_dict"]
        n_act = int(asd["fc_mu.0.weight"].shape[0])
        critic_obs_dim = int(qsd["q1.net.0.weight"].shape[1]) - n_act
        actor_obs_dim = int(asd["net.0.weight"].shape[1])
        self.qnet = DoubleQCritic(
            obs_indices=_indices(critic_obs_dim),
            obs_keys=["all"],
            n_act=n_act,
            hidden_dim=int(args["critic_hidden_dim"]),
            use_layer_norm=bool(args["use_layer_norm"]),
            device=device,
        )
        self.actor = Actor(
            obs_indices=_indices(actor_obs_dim),
            obs_keys=["all"],
            n_act=n_act,
            num_envs=1,
            hidden_dim=int(args["actor_hidden_dim"]),
            log_std_max=float(args["log_std_max"]),
            log_std_min=float(args["log_std_min"]),
            use_tanh=bool(args["use_tanh"]),
            use_layer_norm=bool(args["use_layer_norm"]),
            device=device,
        )
        self.device = device
        self.num_action_samples = int(args["cql_num_action_samples"])
        self.temperature = float(args["cql_temperature"])
        self.use_tanh = bool(args["use_tanh"])
        self.dataset_path = Path(args["offline_dataset_path"])
        self.load(ckpt)

    def load(self, ckpt: dict) -> None:
        self.qnet.load_state_dict(ckpt["qnet_state_dict"])
        self.actor.load_state_dict(ckpt["actor_state_dict"])
        self.qnet.eval()
        self.actor.eval()
        self.obs_normalizer = _normalizer(ckpt.get("obs_normalizer_state"), self.device)
        self.critic_obs_normalizer = _normalizer(ckpt.get("critic_obs_normalizer_state"), self.device)

    def stats(self, probe: dict[str, torch.Tensor], seed: int) -> dict[str, float]:
        out = compute_q_probe_stats(
            self.qnet,
            self.actor,
            self.obs_normalizer,
            self.critic_obs_normalizer,
            probe,
            num_action_samples=self.num_action_samples,
            temperature=self.temperature,
            use_tanh=self.use_tanh,
            seed=seed,
        )
        return {k: float(v) for k, v in out.items()}


def checkpoints(run_dir: Path, stride: int) -> list[tuple[int, Path]]:
    found = []
    for p in run_dir.glob("model_*.pt"):
        m = CKPT_RE.search(p.name)
        if m:
            found.append((int(m.group(1)), p))
    found.sort()
    if stride > 1:
        last = found[-1:]
        found = [f for i, f in enumerate(found) if i % stride == 0]
        if last and last[0] not in found:
            found.append(last[0])
    return found


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log-root", default="logs/WholeBodyTracking")
    ap.add_argument("--methods", nargs="+", default=DEFAULT_METHODS, help="run-name prefixes (before _seedN)")
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--final-step", type=int, default=100000, help="only runs holding this checkpoint (0 = any)")
    ap.add_argument("--stride", type=int, default=1, help="evaluate every k-th checkpoint (last one always)")
    ap.add_argument("--probe-size", type=int, default=4096)
    ap.add_argument("--probe-seed", type=int, default=12345)
    ap.add_argument("--h5", default=None, help="override the dataset path stored in the checkpoints")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--final-k", type=int, default=10)
    ap.add_argument("--out-dir", default="validation/results/q_level")
    ap.add_argument(
        "--paired-baseline",
        default=None,
        help="method whose same-seed run is subtracted (e.g. g1_29dof_wbt_aw_cql_H50)",
    )
    ap.add_argument(
        "--paired-from", type=int, default=10000, help="steps >= this enter the trajectory-mean displacement"
    )
    ap.add_argument(
        "--paired-only",
        action="store_true",
        help="skip checkpoint evaluation; re-pair from <out-dir>/q_level_runs.csv (needs --paired-baseline)",
    )
    args = ap.parse_args()
    if args.paired_baseline and args.paired_baseline not in args.methods:
        args.methods = [*args.methods, args.paired_baseline]
    if args.paired_only:
        if not args.paired_baseline:
            raise SystemExit("--paired-only needs --paired-baseline")
        rows = []
        with (Path(args.out_dir) / "q_level_runs.csv").open() as f:
            for raw in csv.DictReader(f):
                renamed = {("probe/q_level" if k == "probe/q_data" else k): v for k, v in raw.items()}  # old CSVs
                r = {k: (v if k in ("run", "method") else float(v)) for k, v in renamed.items()}
                r["seed"], r["step"] = int(r["seed"]), int(r["step"])
                rows.append(r)
        per_run = defaultdict(list)
        for r in rows:
            per_run[(r["method"], r["seed"])].append(r)
        import matplotlib as mpl  # noqa: PLC0415

        mpl.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415

        write_paired(args, Path(args.out_dir), per_run, plt)
        return

    device = torch.device(args.device)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    wanted = set(args.methods)
    run_dirs = [
        d for d in S.discover_runs(Path(args.log_root), args.final_step) if S.RUN_RE.match(d.name)["method"] in wanted
    ]
    if args.seeds:
        run_dirs = [d for d in run_dirs if int(S.RUN_RE.match(d.name)["seed"]) in set(args.seeds)]
    if not run_dirs:
        raise SystemExit("no runs matched")

    probes: dict[Path, dict[str, torch.Tensor]] = {}
    rows: list[dict] = []
    for run_dir in run_dirs:
        m = S.RUN_RE.match(run_dir.name)
        method, seed = m["method"], int(m["seed"])
        tb = S.load_run(run_dir)
        success_by_step = dict(zip(tb.steps, tb.curves["success"])) if tb and "success" in tb.curves else {}
        models: CheckpointModels | None = None
        ckpts = checkpoints(run_dir, args.stride)
        print(f"{run_dir.name}: {len(ckpts)} checkpoints", flush=True)
        for step, path in ckpts:
            ckpt = torch.load(path, map_location=device, weights_only=False)
            if models is None:
                models = CheckpointModels(ckpt, device)
                h5 = Path(args.h5) if args.h5 else models.dataset_path
                if h5 not in probes:
                    with h5py.File(h5, "r") as f:
                        n = int(f.attrs.get("num_samples", f["observations"].shape[0]))
                    idx = select_probe_indices(n, args.probe_size, args.probe_seed)
                    probes[h5] = load_probe_set(h5, idx, device)
                    print(f"probe set: {idx.size} rows of {h5}", flush=True)
                probe = probes[h5]
            else:
                models.load(ckpt)
            stats = models.stats(probe, args.probe_seed)
            rows.append(
                {
                    "run": run_dir.name,
                    "method": method,
                    "seed": seed,
                    "step": step,
                    "success": success_by_step.get(step, math.nan),
                    **stats,
                }
            )
            del ckpt

    fieldnames = ["run", "method", "seed", "step", "success", *STAT_KEYS]
    with (out / "q_level_runs.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # ---- per-run reductions: at peak-success step, at final step, final-k mean ----
    per_run: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r in rows:
        per_run[(r["method"], r["seed"])].append(r)
    reduced: dict[str, list[dict]] = defaultdict(list)
    for (method, seed), rs in per_run.items():
        rs.sort(key=lambda r: r["step"])
        with_success = [r for r in rs if not math.isnan(r["success"])]
        peak = max(with_success, key=lambda r: r["success"]) if with_success else rs[-1]
        tail = rs[-args.final_k :]
        red = {"method": method, "seed": seed, "peak_step": peak["step"], "peak_success": peak["success"]}
        for k in STAT_KEYS:
            red[f"{k}@peak"] = peak[k]
            red[f"{k}@final"] = rs[-1][k]
            red[f"{k}@final{args.final_k}"] = float(np.mean([r[k] for r in tail]))
        reduced[method].append(red)

    summary_rows = []
    for method in args.methods:
        reds = reduced.get(method, [])
        if not reds:
            continue
        row = {"method": method, "label": S.METHOD_INFO.get(method, ("", method, 0))[1], "n_seeds": len(reds)}
        for key in [k for k in reds[0] if k not in ("method", "seed")]:
            mu, sd = S.mean_std([float(r[key]) for r in reds])
            row[f"{key}_mean"] = mu
            row[f"{key}_std"] = sd
        summary_rows.append(row)
    with (out / "q_level_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    def cell(row: dict, key: str, digits: int = 1) -> str:
        return S.fmt(row[f"{key}_mean"], row[f"{key}_std"], digits)

    lines = [
        f"# Fixed-probe Q levels (probe {args.probe_size} rows, seed {args.probe_seed}; mean ± std over seeds)",
        "",
        "Q_level = mean min(Q1,Q2)(s_probe, a_probe); LSE = mean 0.5(LSE1+LSE2)(s_probe); "
        "bracket = LSE - Q_D twin mean.",
        "",
        "| Method | n | peak step | Q_level @peak | Q_level @final | Q_level final-k | LSE @peak | LSE @final "
        "| bracket @peak | bracket @final | Q_pi @final | Q_rand @final |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary_rows:
        lines.append(  # noqa: PERF401
            f"| {row['label']} | {row['n_seeds']} | {cell(row, 'peak_step', 0)} | {cell(row, 'probe/q_level@peak')} | "
            f"{cell(row, 'probe/q_level@final')} | {cell(row, f'probe/q_level@final{args.final_k}')} | "
            f"{cell(row, 'probe/q_lse@peak')} | {cell(row, 'probe/q_lse@final')} | "
            f"{cell(row, 'probe/lse_minus_q_data@peak')} | {cell(row, 'probe/lse_minus_q_data@final')} | "
            f"{cell(row, 'probe/q_pi@final')} | {cell(row, 'probe/q_rand@final')} |"
        )
    (out / "q_level_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    import matplotlib as mpl  # noqa: PLC0415

    mpl.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    panels = [
        ("probe/q_level", "Q_level (dataset actions)"),
        ("probe/q_lse", "LSE level"),
        ("probe/lse_minus_q_data", "LSE - Q_D"),
        ("success", "eval success %"),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.4))
    for ax, (key, title) in zip(axes, panels):
        for method in args.methods:
            runs = [rs for (m_, _), rs in per_run.items() if m_ == method]
            if not runs:
                continue
            steps = sorted({r["step"] for rs in runs for r in rs})
            mat = np.full((len(runs), len(steps)), np.nan)
            for i, rs in enumerate(runs):
                lookup = {r["step"]: r[key] for r in rs}
                mat[i] = [lookup.get(s, np.nan) for s in steps]
            mu = np.nanmean(mat, axis=0)
            sd = np.nanstd(mat, axis=0)
            label = S.METHOD_INFO.get(method, ("", method, 0))[1]
            ax.plot(steps, mu, label=label)
            ax.fill_between(steps, mu - sd, mu + sd, alpha=0.2)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "q_level_curves.png", dpi=150)
    print(f"written: {out}/q_level_runs.csv, q_level_summary.{{csv,md}}, q_level_curves.png")

    if args.paired_baseline:
        write_paired(args, out, per_run, plt)


PAIRED_KEYS = ("probe/q_level", "probe/q_lse", "probe/q_pi")


def write_paired(args, out: Path, per_run: dict[tuple[str, int], list[dict]], plt) -> None:
    """Same-seed displacement of every run relative to the baseline method, per checkpoint."""
    base = {seed: {r["step"]: r for r in rs} for (m, seed), rs in per_run.items() if m == args.paired_baseline}
    if not base:
        raise SystemExit(f"paired baseline {args.paired_baseline} has no runs")
    paired: list[dict] = []
    for (method, seed), rs in sorted(per_run.items()):
        if method == args.paired_baseline or seed not in base:
            continue
        for r in rs:
            b = base[seed].get(r["step"])
            if b is None:
                continue
            row = {"run": r["run"], "method": method, "seed": seed, "step": r["step"]}
            for k in PAIRED_KEYS:
                row["d_" + k.split("/")[1]] = r[k] - b[k]
            paired.append(row)
    if not paired:
        print("paired: no same-seed pairs found")
        return
    fields = list(paired[0].keys())
    with (out / "q_level_paired.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(paired)

    by_run: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r in paired:
        by_run[(r["method"], r["seed"])].append(r)
    reduced: dict[str, list[dict]] = defaultdict(list)
    for (method, seed), rs in by_run.items():
        rs.sort(key=lambda r: r["step"])
        traj = [r for r in rs if r["step"] >= args.paired_from]
        tail = rs[-args.final_k :]
        red = {"method": method, "seed": seed}
        for k in ("d_q_level", "d_q_lse", "d_q_pi"):
            red[f"{k}_traj"] = float(np.mean([r[k] for r in traj])) if traj else math.nan
            red[f"{k}_final{args.final_k}"] = float(np.mean([r[k] for r in tail]))
        reduced[method].append(red)
    summary = []
    for method in args.methods:
        reds = reduced.get(method)
        if not reds:
            continue
        row = {"method": method, "label": S.METHOD_INFO.get(method, ("", method, 0))[1], "n_seeds": len(reds)}
        for key in [k for k in reds[0] if k not in ("method", "seed")]:
            mu, sd = S.mean_std([r[key] for r in reds])
            row[f"{key}_mean"], row[f"{key}_std"] = mu, sd
        summary.append(row)
    with (out / "q_level_paired_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    fk = args.final_k
    lines = [
        f"# Paired Q-level displacement vs {args.paired_baseline} (same seed, same checkpoint step)",
        "",
        f"traj = mean over checkpoints with step >= {args.paired_from}; "
        f"final-{fk} = mean over the last {fk} checkpoints.",
        "",
        f"| Method | n | dQ_level traj | dQ_level final-{fk} | dLSE traj | dLSE final-{fk} | dQ_pi final-{fk} |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary:
        lines.append(  # noqa: PERF401
            f"| {row['label']} | {row['n_seeds']} | {S.fmt(row['d_q_level_traj_mean'], row['d_q_level_traj_std'])} | "
            f"{S.fmt(row[f'd_q_level_final{fk}_mean'], row[f'd_q_level_final{fk}_std'])} | "
            f"{S.fmt(row['d_q_lse_traj_mean'], row['d_q_lse_traj_std'])} | "
            f"{S.fmt(row[f'd_q_lse_final{fk}_mean'], row[f'd_q_lse_final{fk}_std'])} | "
            f"{S.fmt(row[f'd_q_pi_final{fk}_mean'], row[f'd_q_pi_final{fk}_std'])} |"
        )
    (out / "q_level_paired_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    panels = (("d_q_level", "dQ_level (dataset actions)"), ("d_q_lse", "dLSE"), ("d_q_pi", "dQ_pi"))
    base_label = S.METHOD_INFO.get(args.paired_baseline, ("", args.paired_baseline, 0))[1]
    for ax, (key, title) in zip(axes, panels):
        for method in args.methods:
            runs = [rs for (m_, _), rs in by_run.items() if m_ == method]
            if not runs:
                continue
            steps = sorted({r["step"] for rs in runs for r in rs})
            mat = np.full((len(runs), len(steps)), np.nan)
            for i, rs in enumerate(runs):
                lookup = {r["step"]: r[key] for r in rs}
                mat[i] = [lookup.get(st, np.nan) for st in steps]
            mu, sd = np.nanmean(mat, axis=0), np.nanstd(mat, axis=0)
            ax.plot(steps, mu, label=S.METHOD_INFO.get(method, ("", method, 0))[1])
            ax.fill_between(steps, mu - sd, mu + sd, alpha=0.2)
        ax.axhline(0.0, color="k", lw=0.8)
        ax.set_title(f"{title} vs {base_label}", fontsize=9)
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "q_level_paired.png", dpi=150)
    print(f"written: {out}/q_level_paired.csv, q_level_paired_summary.{{csv,md}}, q_level_paired.png")


if __name__ == "__main__":
    main()
