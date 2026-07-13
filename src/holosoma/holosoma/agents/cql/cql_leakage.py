"""Loss-component "leakage balance" for CQL: who moves the dataset-Q, and how hard.

Settles H9's open cell: does the CQL conservative term push Q(s, a_D) DOWN
through shared parameters/features, and does the Bellman term fail to restore
it in the low-coverage run (env32, gain > 1) while succeeding in the
high-coverage run (env128, gain < 1)?

Core quantity — per-component "push" on the probe data-Q
========================================================
For checkpoint t and loss component C, with g_probe = grad_theta of the
probe-set mean min(Q1,Q2)(s, a_D) over the FIXED probe set:

    push_C = <g_probe, -grad_theta L_C>

the first-order predicted change of the probe data-Q if one unit-lr gradient
step were taken on component C alone (negative = C drags data-Q down).
Components, built exactly as the training update does, on K training-config
batches drawn from the run's own dataset/GPU cache:

    L_bellman     both critics' loss to the detached TD target
    L_cql_total   cql_weight x conservative term, all three sources
    L_cql_rand / L_cql_curr / L_cql_next
                  single-source diagnostic variants. NOTE: logsumexp is not
                  additive across sources — these do NOT sum to L_cql_total;
                  the total is reported separately.

Each push is reported raw (who wins at the actual loss scale) and
cosine-normalized (how aligned the directions are), mean +- std over the K
batches. The same K batch seeds are used for every checkpoint and every run,
so env32/env128 time series are measured with comparable draws.

Headline number
===============
    gain_proxy(t) = -push_cql_total / push_bellman     (valid when bellman > 0)

H9 prediction: env32 gain_proxy rises above ~1 in the 30k-90k range (or sits
persistently above env128's at matched steps) while env128 stays below.

Ground truth & controls
=======================
- One-step actual delta: on 2 of the K batches, clone the critic, take ONE
  optimizer step on a single component's grads (plain SGD at the run's lr for
  a clean first-order check; fresh-state AdamW at the run's lr/betas as an
  approximate preconditioned variant — its state is NOT the training state),
  and re-measure the probe data-Q. Compare with lr * push.
- random-direction baseline: <g_probe, random unit vector x ||grad_C||> gives
  the no-alignment scale.
- decomposition check: grads of (bellman + cql_total) match grads of the
  jointly-built combined loss (allclose) — proves the split is faithful.
- fidelity check (first checkpoint): the component losses are recomputed on a
  state-snapshotted agent via the agent's own ``_update_q`` and compared
  value-for-value (loose rtol; training may run bf16 autocast).

Conduit test (H5 salvage, sample level)
=======================================
Per probe sample i:  push_i = <grad_theta minQ(s_i, a_D_i), -grad_theta L_cql_rand>,
computed for ALL probe samples in one forward-mode JVP pass (fallback: looped
grads on a subsample). Correlated (Pearson/Spearman) against the sample's DR3
feature cosine cos(phi(s,a_D), phi(s',a_pi)) and feature norm ||phi(s,a_D)||.
A positive correlation of downward push with cosine revives the conduit story
WITHIN an env even though env-level mean cosines match.

Usage
=====
    python -m holosoma.agents.cql.cql_leakage \\
        --checkpoint logs/WholeBodyTracking/<run> \\
        --step-min 30000 --step-max 90000 --step-stride 10000

Same checkpoint selection / probe-set / two-stage tyro override skeleton as
cql_probe (run it on BOTH runs' checkpoints with their own probe sets).
Outputs merge incrementally into <run>/probe/leakage_scan.{csv,npz} after
every checkpoint, exactly like cql_probe's scan files.

Interpretation table
====================
| observation                                               | verdict |
|-----------------------------------------------------------|---------|
| env32 gain_proxy > env128 at matched steps, rand dominant  | H9 confirmed, downward force = CQL(rand); rand-removal arm becomes the confirmatory intervention |
| pushes similar across envs, bellman push much larger in env128 | H9 confirmed via restoring-force side; coverage prescription (exp B) promoted |
| push_cql ~ 0 or positive on data-Q                         | H9's CQL-origin refuted; hunt other downward forces (target-net dynamics, entropy term in target) |
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tyro
from loguru import logger

from holosoma.agents.cql import cql_probe as probe
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.utils.eval_utils import CheckpointConfig, init_eval_logging, load_checkpoint, load_saved_experiment_config
from holosoma.utils.helpers import get_class
from holosoma.utils.safe_torch_import import F, torch
from holosoma.utils.sim_utils import close_simulation_app, setup_simulation_environment
from holosoma.utils.tyro_utils import TYRO_CONIFG

CQL_SOURCES = ("rand", "curr", "next")
COMPONENTS = ("bellman", "cql_total", "cql_rand", "cql_curr", "cql_next")
ONESTEP_COMPONENTS = ("bellman", "cql_total", "cql_rand")
_BATCH_SEED_STRIDE = 7919  # same batch draws for every checkpoint and both runs


@dataclass(frozen=True)
class LeakageCLI:
    checkpoint: str
    """Checkpoints to scan: model_*.pt path, glob (quote it), run directory, or comma list (wandb:// ok)."""

    steps: str | None = None
    """Explicit comma-separated steps to scan; overrides the stride/range filters."""

    step_min: int | None = None
    """Only scan checkpoints with step >= this."""

    step_max: int | None = None
    """Only scan checkpoints with step <= this."""

    step_stride: int = 0
    """Only scan steps divisible by this (0 = all matched)."""

    probe_set: str | None = None
    """Fixed probe-set npz (shared with cql_probe). Built once if missing.
    Defaults to <checkpoint_dir>/probe/probe_set.npz."""

    probe_size: int = 2048
    """Probe-set size when it has to be built."""

    probe_seed: int = 0
    """Seed for probe-set building and for the per-batch measurement seeds."""

    output_dir: str | None = None
    """Output directory. Defaults to <checkpoint_dir>/probe."""

    num_batches: int = 8
    """K training batches per checkpoint for the component-gradient measurement."""

    batch_size: int = 0
    """Batch size for those draws; 0 = the run's training batch_size."""

    one_step: bool = True
    """Run the one-step actual-delta ground-truth check (SGD + fresh-state AdamW) on 2 batches."""

    conduit: bool = True
    """Run the per-sample conduit test (JVP over the whole probe set; loop fallback)."""

    conduit_fallback_samples: int = 256
    """Probe subsample size when the JVP path is unavailable and the loop fallback runs."""

    fidelity_check: bool = True
    """At the first checkpoint, compare the rebuilt component losses against the agent's own
    _update_q values on a state-snapshotted agent (loose rtol; training may use bf16)."""


# ---------------------------------------------------------------------------
# loss components, built exactly as CQLAgent._update_q builds them
# ---------------------------------------------------------------------------


def _critic_params(algo: Any) -> list[torch.Tensor]:
    return [p for p in algo.qnet.parameters() if p.requires_grad]


def _grads(loss: torch.Tensor, params: list[torch.Tensor]) -> tuple[torch.Tensor, ...]:
    # allow_unused+materialize: per-critic directions (Q1-only/Q2-only) leave the
    # other critic's parameters out of the graph; their gradient is exactly zero.
    return torch.autograd.grad(
        loss, params, retain_graph=False, create_graph=False, allow_unused=True, materialize_grads=True
    )


def bellman_loss(algo: Any, data: Any) -> torch.Tensor:
    """Both critics' loss to the detached TD target, mirroring _update_q."""
    args = algo.config
    observations = data["observations"]
    next_observations = data["next"]["observations"]
    critic_observations = data["critic_observations"]
    next_critic_observations = data["next"]["critic_observations"]
    dataset_actions = probe._to_critic_actions(algo, data["actions"]).detach()
    rewards = float(getattr(algo, "reward_scale", 1.0)) * data["next"]["rewards"]
    bootstrap = (~data["next"]["dones"].bool()).float()
    del observations

    with torch.no_grad():
        discount = float(args.gamma) ** data["next"]["effective_n_steps"]
        rewards_ = rewards.view(-1)
        bootstrap_ = bootstrap.view(-1)
        discount_ = discount.view(-1)
        if bool(getattr(args, "cql_max_target_backup", False)):
            batch_size = next_observations.shape[0]
            num_backup = int(args.cql_max_target_backup_samples)
            expanded_next_obs = next_observations[:, None, :].expand(batch_size, num_backup, -1).reshape(batch_size * num_backup, -1)
            expanded_next_cobs = next_critic_observations[:, None, :].expand(batch_size, num_backup, -1).reshape(batch_size * num_backup, -1)
            next_actions, next_log_probs = algo.actor.get_actions_and_log_probs(expanded_next_obs)
            nq1, nq2 = algo.qnet_target(expanded_next_cobs, next_actions)
            next_min_all = torch.minimum(nq1.view(batch_size, num_backup), nq2.view(batch_size, num_backup))
            next_min, max_idx = next_min_all.max(dim=1)
            next_log_probs = next_log_probs.view(batch_size, num_backup).gather(1, max_idx.unsqueeze(1)).squeeze(1)
        else:
            next_actions, next_log_probs = algo.actor.get_actions_and_log_probs(next_observations)
            nq1, nq2 = algo.qnet_target(next_critic_observations, next_actions)
            next_min = torch.minimum(nq1, nq2).view(-1)
            next_log_probs = next_log_probs.view(-1)
        if bool(getattr(args, "backup_entropy", False)):
            next_v = next_min - algo.log_alpha.exp().detach() * next_log_probs
        else:
            next_v = next_min
        q_target = (rewards_ + discount_ * bootstrap_ * next_v).clamp(min=-10000.0, max=10000.0)

    q1, q2 = algo.qnet(critic_observations, dataset_actions)
    if getattr(args, "bellman_loss_type", "mse") == "huber":
        beta = float(getattr(args, "huber_beta", 1.0))
        return F.smooth_l1_loss(q1, q_target, beta=beta) + F.smooth_l1_loss(q2, q_target, beta=beta)
    return F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)


def cql_conservative_loss(algo: Any, data: Any, sources: tuple[str, ...] = CQL_SOURCES) -> torch.Tensor:
    """cql_weight x conservative term with only ``sources`` inside the logsumexp.

    With all three sources this is the training term (incl. its -Q(s,a_D)
    push-up part and the Lagrange multiplier when configured). Single-source
    variants are diagnostic: logsumexp is not additive, so they do not sum to
    the total.
    """
    args = algo.config
    cql_weight = float(getattr(algo, "_cql_weight", getattr(args, "cql_weight", 0.0)))
    if cql_weight <= 0.0:
        return torch.zeros((), device=data["actions"].device)
    num_repeat = int(getattr(algo, "_num_repeat_actions", getattr(args, "cql_num_action_samples", 10)))
    temperature = float(getattr(algo, "_temperature", getattr(args, "cql_temperature", 1.0)))

    observations = data["observations"]
    next_observations = data["next"]["observations"]
    critic_observations = data["critic_observations"]
    dataset_actions = probe._to_critic_actions(algo, data["actions"]).detach()
    batch = dataset_actions.shape[0]
    device = dataset_actions.device

    expanded_obs = observations[:, None, :].expand(batch, num_repeat, -1).reshape(batch * num_repeat, -1)
    expanded_cobs = critic_observations[:, None, :].expand(batch, num_repeat, -1).reshape(batch * num_repeat, -1)
    expanded_next_obs = next_observations[:, None, :].expand(batch, num_repeat, -1).reshape(batch * num_repeat, -1)

    # RNG order matches _update_q (curr, next, rand) regardless of the source
    # subset so that fixed seeds give identical draws across variants.
    with torch.no_grad():
        curr_actions, curr_logp = algo.actor.get_actions_and_log_probs(expanded_obs)
        next_actions_rep, next_logp = algo.actor.get_actions_and_log_probs(expanded_next_obs)
        action_scale = algo.actor.action_scale.to(device=device, dtype=dataset_actions.dtype)
        action_bias = algo.actor.action_bias.to(device=device, dtype=dataset_actions.dtype)
        rand_actions = torch.empty(batch * num_repeat, dataset_actions.shape[-1], device=device, dtype=dataset_actions.dtype).uniform_(-1.0, 1.0)
        rand_actions = rand_actions * action_scale + action_bias
        if bool(getattr(args, "use_tanh", True)):
            random_density = math.log(0.5) * dataset_actions.shape[-1] - torch.log(action_scale + 1e-6).sum()
        else:
            random_density = math.log(0.5) * dataset_actions.shape[-1]

    q1_data, q2_data = algo.qnet(critic_observations, dataset_actions)
    q1_terms, q2_terms = [], []
    if "rand" in sources:
        q1_rand, q2_rand = algo.qnet(expanded_cobs, rand_actions)
        q1_terms.append((q1_rand - random_density).view(batch, num_repeat))
        q2_terms.append((q2_rand - random_density).view(batch, num_repeat))
    if "curr" in sources:
        q1_curr, q2_curr = algo.qnet(expanded_cobs, curr_actions)
        q1_terms.append(q1_curr.view(batch, num_repeat) - curr_logp.view(batch, num_repeat))
        q2_terms.append(q2_curr.view(batch, num_repeat) - curr_logp.view(batch, num_repeat))
    if "next" in sources:
        q1_next, q2_next = algo.qnet(expanded_cobs, next_actions_rep)
        q1_terms.append(q1_next.view(batch, num_repeat) - next_logp.view(batch, num_repeat))
        q2_terms.append(q2_next.view(batch, num_repeat) - next_logp.view(batch, num_repeat))

    cql1 = (torch.logsumexp(torch.cat(q1_terms, dim=1) / temperature, dim=1) * temperature - q1_data).mean()
    cql2 = (torch.logsumexp(torch.cat(q2_terms, dim=1) / temperature, dim=1) * temperature - q2_data).mean()
    loss = cql_weight * 0.5 * (cql1 + cql2)
    log_cql_alpha = getattr(algo, "log_cql_alpha", None)
    if bool(getattr(args, "use_lagrange", False)) and log_cql_alpha is not None:
        loss = loss * log_cql_alpha.exp().detach().clamp(max=float(getattr(args, "cql_lagrange_max", 1e6)))
    return loss


def component_loss(algo: Any, data: Any, component: str) -> torch.Tensor:
    if component == "bellman":
        return bellman_loss(algo, data)
    if component == "cql_total":
        return cql_conservative_loss(algo, data, CQL_SOURCES)
    if component.startswith("cql_"):
        return cql_conservative_loss(algo, data, (component.removeprefix("cql_"),))
    raise ValueError(f"unknown component {component!r}")


# ---------------------------------------------------------------------------
# probe-side direction and per-sample conduit
# ---------------------------------------------------------------------------


def probe_qdata_grads(algo: Any, tensors: dict[str, torch.Tensor]) -> dict[str, tuple[torch.Tensor, ...]]:
    """grad_theta of the probe-set mean data-Q: min-twin (main) plus per-critic."""
    params = _critic_params(algo)
    cobs = algo.critic_obs_normalizer(tensors["critic_observations"])
    a_data = probe._to_critic_actions(algo, tensors["actions"]).detach()
    out: dict[str, tuple[torch.Tensor, ...]] = {}
    q1, q2 = algo.qnet(cobs, a_data)
    out["min"] = _grads(torch.minimum(q1, q2).mean(), params)
    q1, q2 = algo.qnet(cobs, a_data)
    out["q1"] = _grads(q1.mean(), params)
    q1, q2 = algo.qnet(cobs, a_data)
    out["q2"] = _grads(q2.mean(), params)
    return out


def per_sample_push_jvp(algo: Any, tensors: dict[str, torch.Tensor], direction: tuple[torch.Tensor, ...]) -> np.ndarray:
    """push_i = <grad_theta minQ(s_i, a_D_i), direction> for every probe sample in one JVP pass."""
    from torch.func import functional_call, jvp

    cobs = algo.critic_obs_normalizer(tensors["critic_observations"]).detach()
    a_data = probe._to_critic_actions(algo, tensors["actions"]).detach()
    qnet = algo.qnet
    named = [(name, param) for name, param in qnet.named_parameters() if param.requires_grad]
    assert len(named) == len(direction)
    primals = {name: param.detach() for name, param in named}
    tangents = {name: vec.detach() for (name, _), vec in zip(named, direction)}
    buffers = {name: buf.detach() for name, buf in qnet.named_buffers()}

    def q_min(params: dict[str, torch.Tensor]) -> torch.Tensor:
        q1, q2 = functional_call(qnet, {**params, **buffers}, (cobs, a_data))
        return torch.minimum(q1, q2)

    _, push = jvp(q_min, (primals,), (tangents,))
    return push.detach().float().cpu().numpy()


def per_sample_push_loop(
    algo: Any,
    tensors: dict[str, torch.Tensor],
    direction: tuple[torch.Tensor, ...],
    max_samples: int,
) -> np.ndarray:
    """Loop fallback on a subsample (first max_samples probe rows)."""
    params = _critic_params(algo)
    cobs = algo.critic_obs_normalizer(tensors["critic_observations"]).detach()
    a_data = probe._to_critic_actions(algo, tensors["actions"]).detach()
    count = min(max_samples, cobs.shape[0])
    push = np.zeros(count, dtype=np.float64)
    for i in range(count):
        q1, q2 = algo.qnet(cobs[i : i + 1], a_data[i : i + 1])
        grads = _grads(torch.minimum(q1, q2).squeeze(), params)
        push[i] = probe._flat_dot(grads, direction)
    return push


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 8 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    return _pearson(np.argsort(np.argsort(a)).astype(np.float64), np.argsort(np.argsort(b)).astype(np.float64))


# ---------------------------------------------------------------------------
# per-checkpoint measurement
# ---------------------------------------------------------------------------


def _seeded_batch(algo: Any, batch_size: int, seed: int) -> Any:
    torch.manual_seed(seed)
    return algo._sample_offline_batch(
        batch_size=batch_size,
        normalize_obs=algo.obs_normalizer.forward,
        normalize_critic_obs=algo.critic_obs_normalizer.forward,
    )


def _one_step_delta(
    algo: Any,
    data: Any,
    component: str,
    tensors: dict[str, torch.Tensor],
    optimizer_kind: str,
    seed: int,
) -> float:
    """Actual probe data-Q change after ONE optimizer step on a single component."""
    params = _critic_params(algo)
    snapshot = [p.detach().clone() for p in params]
    args = algo.config
    lr = float(getattr(args, "critic_learning_rate", 3e-4))
    with torch.no_grad():
        cobs = algo.critic_obs_normalizer(tensors["critic_observations"])
        a_data = probe._to_critic_actions(algo, tensors["actions"])
        q1, q2 = algo.qnet(cobs, a_data)
        before = float(torch.minimum(q1, q2).mean().item())
    torch.manual_seed(seed)
    grads = _grads(component_loss(algo, data, component), params)
    try:
        if optimizer_kind == "sgd":
            with torch.no_grad():
                for param, grad in zip(params, grads):
                    param.sub_(lr * grad)
        else:  # fresh-state AdamW: preconditioning approximation, not the training state
            optimizer = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
            for param, grad in zip(params, grads):
                param.grad = grad
            optimizer.step()
            for param in params:
                param.grad = None
        with torch.no_grad():
            q1, q2 = algo.qnet(cobs, a_data)
            after = float(torch.minimum(q1, q2).mean().item())
    finally:
        with torch.no_grad():
            for param, saved in zip(params, snapshot):
                param.copy_(saved)
    return after - before


@torch.enable_grad()
def measure_checkpoint(
    algo: Any,
    tensors: dict[str, torch.Tensor],
    cli: LeakageCLI,
    run_fidelity: bool,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    probe._set_eval_mode(algo)
    args = algo.config
    lr = float(getattr(args, "critic_learning_rate", 3e-4))
    batch_size = cli.batch_size if cli.batch_size > 0 else int(getattr(args, "batch_size", 1024))
    params = _critic_params(algo)
    cql_active = float(getattr(algo, "_cql_weight", getattr(args, "cql_weight", 0.0))) > 0.0
    if int(getattr(algo, "_num_near_actions", 0)) > 0:
        logger.warning("[QLeak] cql_near_action_samples > 0 is not replicated; CQL components omit it.")

    g_probe = probe_qdata_grads(algo, tensors)
    gnorm_probe = probe._flat_dot(g_probe["min"], g_probe["min"]) ** 0.5

    push: dict[str, list[float]] = {c: [] for c in COMPONENTS}
    push_cos: dict[str, list[float]] = {c: [] for c in COMPONENTS}
    push_q1_total: list[float] = []
    push_q2_total: list[float] = []
    push_q1_bellman: list[float] = []
    push_q2_bellman: list[float] = []
    rand_baseline: dict[str, list[float]] = {"bellman": [], "cql_total": []}
    decomposition_err = float("nan")
    fidelity: dict[str, float] = {}
    onestep: dict[str, list[float]] = {}
    conduit_direction: tuple[torch.Tensor, ...] | None = None

    batches = []
    for k in range(cli.num_batches):
        batches.append(_seeded_batch(algo, batch_size, cli.probe_seed + _BATCH_SEED_STRIDE * (k + 1)))

    for k, data in enumerate(batches):
        loss_seed = cli.probe_seed + _BATCH_SEED_STRIDE * (k + 1) + 1
        component_grads: dict[str, tuple[torch.Tensor, ...]] = {}
        for component in COMPONENTS:
            if component != "bellman" and not cql_active:
                continue
            torch.manual_seed(loss_seed)
            component_grads[component] = _grads(component_loss(algo, data, component), params)
        for component, grads in component_grads.items():
            push[component].append(-probe._flat_dot(g_probe["min"], grads))
            push_cos[component].append(-probe._grad_cosine(g_probe["min"], grads))
        push_q1_bellman.append(-probe._flat_dot(g_probe["q1"], component_grads["bellman"]))
        push_q2_bellman.append(-probe._flat_dot(g_probe["q2"], component_grads["bellman"]))
        if cql_active:
            push_q1_total.append(-probe._flat_dot(g_probe["q1"], component_grads["cql_total"]))
            push_q2_total.append(-probe._flat_dot(g_probe["q2"], component_grads["cql_total"]))

        # random-direction baseline at each component's gradient scale
        torch.manual_seed(loss_seed + 2)
        for component in rand_baseline:
            if component not in component_grads:
                continue
            rand_vec = [torch.randn_like(p) for p in params]
            rand_norm = probe._flat_dot(rand_vec, rand_vec) ** 0.5
            comp_norm = probe._flat_dot(component_grads[component], component_grads[component]) ** 0.5
            rand_baseline[component].append(probe._flat_dot(g_probe["min"], rand_vec) * comp_norm / max(rand_norm, 1e-12))

        if k == 0:
            # decomposition control: grad(bellman)+grad(cql_total) == grad(bellman+cql_total)
            if cql_active:
                torch.manual_seed(loss_seed)
                bell = bellman_loss(algo, data)
                cql = cql_conservative_loss(algo, data, CQL_SOURCES)
                joint = _grads(bell + cql, params)
                # replicate the SAME rng stream: bellman consumes its draws first,
                # then cql continues from that state (no reseed in between).
                torch.manual_seed(loss_seed)
                g_b = _grads(bellman_loss(algo, data), params)
                g_c = _grads(cql_conservative_loss(algo, data, CQL_SOURCES), params)
                num = sum(float(((jb - (bb + cc)) ** 2).sum()) for jb, bb, cc in zip(joint, g_b, g_c))
                den = sum(float((jb**2).sum()) for jb in joint)
                decomposition_err = (num / max(den, 1e-20)) ** 0.5
                torch.manual_seed(loss_seed)
                conduit_direction = tuple(-g for g in _grads(cql_conservative_loss(algo, data, ("rand",)), params))
            if run_fidelity:
                fidelity = _fidelity_check(algo, data, loss_seed)

        if cli.one_step and k < 2:
            for component in ONESTEP_COMPONENTS:
                if component != "bellman" and not cql_active:
                    continue
                for kind in ("sgd", "adam"):
                    onestep.setdefault(f"{kind}_{component}", []).append(
                        _one_step_delta(algo, data, component, tensors, kind, loss_seed)
                    )
                onestep.setdefault(f"pred_{component}", []).append(lr * push[component][k])

    row: dict[str, float] = {
        "leak_batches": float(len(batches)),
        "leak_batch_size": float(batch_size),
        "gnorm_probe": gnorm_probe,
        "decomposition_relerr": decomposition_err,
    }
    for component in COMPONENTS:
        values = np.asarray(push[component], dtype=np.float64)
        if values.size == 0:
            row[f"push_{component}_mean"] = float("nan")
            row[f"push_{component}_std"] = float("nan")
            row[f"cospush_{component}_mean"] = float("nan")
            continue
        row[f"push_{component}_mean"] = float(values.mean())
        row[f"push_{component}_std"] = float(values.std())
        row[f"cospush_{component}_mean"] = float(np.mean(push_cos[component]))
    row["push_bellman_q1_mean"] = float(np.mean(push_q1_bellman))
    row["push_bellman_q2_mean"] = float(np.mean(push_q2_bellman))
    row["push_cql_total_q1_mean"] = float(np.mean(push_q1_total)) if push_q1_total else float("nan")
    row["push_cql_total_q2_mean"] = float(np.mean(push_q2_total)) if push_q2_total else float("nan")
    for component, values in rand_baseline.items():
        row[f"randbase_{component}_mean"] = float(np.mean(values)) if values else float("nan")
        row[f"randbase_{component}_std"] = float(np.std(values)) if values else float("nan")

    bellman_push = row["push_bellman_mean"]
    cql_push = row["push_cql_total_mean"]
    row["bellman_push_nonpositive"] = float(bellman_push <= 0.0)
    row["gain_proxy"] = (-cql_push / bellman_push) if (cql_push == cql_push and bellman_push > 0.0) else float("nan")

    for key, values in onestep.items():
        row[f"onestep_dq_{key}_mean"] = float(np.mean(values))
    for key, value in fidelity.items():
        row[key] = value

    arrays: dict[str, np.ndarray] = {
        "push_per_batch": np.asarray([[np.nan if not push[c] or len(push[c]) <= k else push[c][k] for c in COMPONENTS] for k in range(len(batches))]),
    }

    if cli.conduit and cql_active and conduit_direction is not None:
        try:
            sample_push = per_sample_push_jvp(algo, tensors, conduit_direction)
            row["conduit_mode"] = 0.0  # jvp
        except Exception as error:  # torch.func unavailable or op unsupported
            logger.warning("[QLeak] JVP conduit path failed ({}); falling back to a {}-sample loop", error, cli.conduit_fallback_samples)
            sample_push = per_sample_push_loop(algo, tensors, conduit_direction, cli.conduit_fallback_samples)
            row["conduit_mode"] = 1.0  # loop
        count = sample_push.shape[0]
        with torch.no_grad():
            cobs = algo.critic_obs_normalizer(tensors["critic_observations"][:count])
            ncobs = algo.critic_obs_normalizer(tensors["next_critic_observations"][:count])
            nobs = algo.obs_normalizer(tensors["next_observations"][:count])
            a_data = probe._to_critic_actions(algo, tensors["actions"][:count])
            a_next = algo.actor(nobs)[0]
            feat_data = algo.qnet.q1.features(cobs, a_data)
            feat_next = algo.qnet.q1.features(ncobs, a_next)
            dr3_cos = torch.nn.functional.cosine_similarity(feat_data, feat_next, dim=-1).float().cpu().numpy()
            feat_norm = feat_data.norm(dim=-1).float().cpu().numpy()
        row["conduit_push_mean"] = float(sample_push.mean())
        row["conduit_push_vs_dr3cos_pearson"] = _pearson(sample_push, dr3_cos)
        row["conduit_push_vs_dr3cos_spearman"] = _spearman(sample_push, dr3_cos)
        row["conduit_push_vs_featnorm_pearson"] = _pearson(sample_push, feat_norm)
        row["conduit_push_vs_featnorm_spearman"] = _spearman(sample_push, feat_norm)
        arrays["conduit_push"] = sample_push
        arrays["conduit_dr3_cos"] = dr3_cos
        arrays["conduit_feat_norm"] = feat_norm

    return row, arrays


def _fidelity_check(algo: Any, data: Any, loss_seed: int) -> dict[str, float]:
    """Compare rebuilt component losses against the agent's own _update_q on the same batch.

    Snapshots and restores everything _update_q mutates. Training may run
    bf16 autocast while the rebuild runs fp32, so compare with loose rtol.
    """
    state = {
        "qnet": {k: v.detach().clone() for k, v in algo.qnet.state_dict().items()},
        "log_alpha": algo.log_alpha.detach().clone(),
        "q_opt": algo.q_optimizer.state_dict(),
        "alpha_opt": algo.alpha_optimizer.state_dict(),
        "scaler": algo.scaler.state_dict(),
    }
    try:
        torch.manual_seed(loss_seed)
        out = algo._update_q(data)
        agent_conservative = float(out[13])
        agent_bellman = float(out[14])
    finally:
        algo.qnet.load_state_dict(state["qnet"])
        with torch.no_grad():
            algo.log_alpha.copy_(state["log_alpha"])
        algo.q_optimizer.load_state_dict(state["q_opt"])
        algo.alpha_optimizer.load_state_dict(state["alpha_opt"])
        algo.scaler.load_state_dict(state["scaler"])
    torch.manual_seed(loss_seed)
    with torch.no_grad():
        mine_bellman = float(bellman_loss(algo, data))
        mine_conservative = float(cql_conservative_loss(algo, data, CQL_SOURCES))
    bell_err = abs(mine_bellman - agent_bellman) / max(abs(agent_bellman), 1e-8)
    cql_err = abs(mine_conservative - agent_conservative) / max(abs(agent_conservative), 1e-8)
    logger.info(
        "[QLeak] fidelity vs _update_q: bellman {:.6g} vs {:.6g} (rel {:.3%}), cql {:.6g} vs {:.6g} (rel {:.3%})",
        mine_bellman,
        agent_bellman,
        bell_err,
        mine_conservative,
        agent_conservative,
        cql_err,
    )
    if bell_err > 0.05 or cql_err > 0.05:
        logger.warning("[QLeak] fidelity check exceeded 5% — component decomposition may not match training exactly.")
    return {"fidelity_bellman_relerr": bell_err, "fidelity_cql_relerr": cql_err}


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def run_leakage_scan(cli: LeakageCLI, tyro_config: ExperimentConfig) -> None:
    all_references = probe._resolve_checkpoints(cli.checkpoint)
    references = probe._filter_references(
        all_references, probe._parse_steps(cli.steps), cli.step_min, cli.step_max, cli.step_stride
    )
    if len(references) != len(all_references):
        logger.info("[QLeak] step filter selected {}/{} matched checkpoints", len(references), len(all_references))
    output_dir = probe._default_output_dir(cli.output_dir, references[0])
    output_dir.mkdir(parents=True, exist_ok=True)
    probe_set_path = Path(cli.probe_set).expanduser() if cli.probe_set else output_dir / "probe_set.npz"

    checkpoints = [Path(load_checkpoint(reference, str(output_dir))) for reference in references]
    missing = [str(p) for p in checkpoints if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Checkpoint file(s) not found: {missing}")

    logger.info("[QLeak] scanning {} checkpoints; outputs -> {}", len(checkpoints), output_dir)
    env, device, simulation_app = setup_simulation_environment(tyro_config)
    try:
        algo_class = get_class(tyro_config.algo._target_)
        algo = algo_class(device=device, env=env, config=tyro_config.algo.config, log_dir=str(output_dir), multi_gpu_cfg=None)
        algo.setup()

        if probe_set_path.exists():
            probe_set = probe.load_probe_set(probe_set_path)
        else:
            probe_set = probe.build_probe_set(algo, cli.probe_size, cli.probe_seed, probe_set_path)
        tensors = probe._probe_tensors(probe_set, device)

        csv_path = output_dir / "leakage_scan.csv"
        npz_path = output_dir / "leakage_scan.npz"
        existing_rows = probe._read_existing_rows(csv_path)
        if existing_rows:
            logger.info("[QLeak] {} existing rows in {}; merging by step", len(existing_rows), csv_path)

        rows: list[dict[str, float]] = []
        per_step_arrays: dict[str, list[np.ndarray]] = {}
        scanned_steps: list[int] = []
        for index, checkpoint_path in enumerate(checkpoints):
            step = probe._step_from_path(checkpoint_path)
            algo.load(str(checkpoint_path))
            row, arrays = measure_checkpoint(algo, tensors, cli, run_fidelity=cli.fidelity_check and index == 0)
            row = {"step": float(step), **row}
            rows.append(row)
            scanned_steps.append(step)
            for key, value in arrays.items():
                per_step_arrays.setdefault(key, []).append(value)

            total_rows = probe._write_scan_csv(csv_path, existing_rows, rows)
            payload: dict[str, np.ndarray] = {
                "steps": np.asarray(scanned_steps, dtype=np.int64),
                "components": np.asarray(COMPONENTS),
                "probe_set_path": np.asarray(str(probe_set_path)),
            }
            for key, values in per_step_arrays.items():
                payload[key] = np.stack(values)
            probe._write_npz_atomic(npz_path, payload)

            dominant = "n/a"
            source_pushes = {s: row.get(f"push_cql_{s}_mean", float("nan")) for s in CQL_SOURCES}
            if any(v == v for v in source_pushes.values()):
                dominant = min(source_pushes, key=lambda s: source_pushes[s] if source_pushes[s] == source_pushes[s] else float("inf"))
            verdict = ""
            gain = row.get("gain_proxy", float("nan"))
            if gain == gain and gain > 1.0:
                verdict = "  <-- CQL push-down exceeds Bellman restoring force (gain > 1)"
            logger.info(
                "[QLeak] [{}/{}] step={} push_bellman={:+.4g} push_cql_total={:+.4g} gain_proxy={:.3f} "
                "dominant_source={} onestep(sgd) cql actual/pred={:+.4g}/{:+.4g} conduit_r={:.3f} -> {} rows in {}{}",
                index + 1,
                len(checkpoints),
                step,
                row.get("push_bellman_mean", float("nan")),
                row.get("push_cql_total_mean", float("nan")),
                gain,
                dominant,
                row.get("onestep_dq_sgd_cql_total_mean", float("nan")),
                row.get("onestep_dq_pred_cql_total_mean", float("nan")),
                row.get("conduit_push_vs_dr3cos_pearson", float("nan")),
                total_rows,
                csv_path.name,
                verdict,
            )
        logger.info("[QLeak] done: {} and {}", csv_path, npz_path)
    finally:
        if simulation_app:
            close_simulation_app(simulation_app)


def main() -> None:
    init_eval_logging()
    cli, remaining_args = tyro.cli(LeakageCLI, return_unknown_args=True, add_help=False)
    first_checkpoint = probe._resolve_checkpoints(cli.checkpoint)[0]
    saved_cfg, _ = load_saved_experiment_config(CheckpointConfig(checkpoint=str(first_checkpoint)))
    eval_cfg = saved_cfg.get_eval_config()
    tyro_config = tyro.cli(
        ExperimentConfig,
        default=eval_cfg,
        args=remaining_args,
        description="Overrides applied on top of the experiment config embedded in the checkpoint.",
        config=TYRO_CONIFG,
    )
    run_leakage_scan(cli, tyro_config)


if __name__ == "__main__":
    main()
