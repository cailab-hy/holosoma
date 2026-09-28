"""Fixed-probe Q-level diagnostics for the offline CQL family.

A *probe set* is a fixed, seed-determined subset of dataset transitions
``(s, a, s')`` that is identical for every run trained on the same H5 file
(it depends on the dataset size and ``q_probe_seed`` only, never on the training
seed). Evaluating the critic on it at every logging step, or post hoc on every
saved checkpoint, gives Q-level curves that are comparable across methods and seeds:

    probe/q_level         Q_level(t) = mean_i min(Q1, Q2)(s_i, a_i)     (dataset actions)
    probe/q_lse           mean_i 0.5 (LSE1 + LSE2)(s_i)                  (CQL push-down target)
    probe/lse_minus_q_data mean_i 0.5 [(LSE1 - Q1) + (LSE2 - Q2)](s_i, a_i)  (unweighted bracket)
    probe/q_pi            mean_i min(Q1, Q2)(s_i, pi(s_i))               (deterministic policy)
    probe/q_curr          mean over policy samples of 0.5 (Q1 + Q2)
    probe/q_rand          mean over uniform random actions of 0.5 (Q1 + Q2)

``LSE`` is the importance-corrected logsumexp over the same random / current-policy /
next-policy action mixture the CQL bracket uses (same sample count and temperature).
Sampling uses a forked RNG seeded with ``seed`` so the diagnostic never perturbs
training randomness and is reproducible post hoc.
"""

from __future__ import annotations

import inspect
import math
from pathlib import Path

import h5py
import numpy as np

from holosoma.utils.safe_torch_import import torch

PROBE_KEYS = ("observations", "critic_observations", "actions", "next_observations")


def select_probe_indices(num_samples: int, probe_size: int, seed: int) -> np.ndarray:
    """Sorted, unique H5 row indices of the probe set (deterministic in ``seed``)."""
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    size = min(int(probe_size), int(num_samples))
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(int(num_samples), size=size, replace=False))


def load_probe_set(h5_path: str | Path, indices: np.ndarray, device: torch.device | str) -> dict[str, torch.Tensor]:
    """Read the probe transitions (raw, un-normalized) from the H5 file onto ``device``."""
    indices = np.asarray(indices, dtype=np.int64)
    if indices.size and np.any(np.diff(indices) <= 0):
        raise ValueError("probe indices must be strictly increasing (h5py point selection)")
    with h5py.File(str(h5_path), "r") as f:
        return {
            key: torch.as_tensor(np.asarray(f[key][indices]), dtype=torch.float32, device=device) for key in PROBE_KEYS
        }


def _normalize(normalizer, x: torch.Tensor) -> torch.Tensor:
    """Apply an observation normalizer without updating its running statistics."""
    if normalizer is None:
        return x
    forward = getattr(normalizer, "forward", normalizer)
    try:
        accepts_update = "update" in inspect.signature(forward).parameters
    except (TypeError, ValueError):
        accepts_update = False
    return forward(x, update=False) if accepts_update else forward(x)


@torch.no_grad()
def compute_q_probe_stats(
    qnet,
    actor,
    obs_normalizer,
    critic_obs_normalizer,
    probe: dict[str, torch.Tensor],
    *,
    num_action_samples: int,
    temperature: float,
    use_tanh: bool,
    seed: int,
    chunk_size: int = 1024,
) -> dict[str, torch.Tensor]:
    """Q levels of ``qnet`` on the fixed probe set (see module docstring for the keys)."""
    actions_all = probe["actions"]
    device = actions_all.device
    n_total = int(actions_all.shape[0])
    if n_total == 0:
        raise ValueError("empty probe set")
    n_act = int(actions_all.shape[-1])
    num_repeat = max(1, int(num_action_samples))
    temperature = float(temperature)

    obs_all = _normalize(obs_normalizer, probe["observations"])
    next_obs_all = _normalize(obs_normalizer, probe["next_observations"])
    critic_obs_all = _normalize(critic_obs_normalizer, probe["critic_observations"])

    action_scale = actor.action_scale.to(device=device, dtype=actions_all.dtype)
    action_bias = actor.action_bias.to(device=device, dtype=actions_all.dtype)
    if use_tanh:
        random_density = math.log(0.5) * n_act - torch.log(action_scale + 1e-6).sum()
    else:
        random_density = torch.as_tensor(math.log(0.5) * n_act, device=device, dtype=actions_all.dtype)

    sums = {
        "probe/q_level": 0.0,
        "probe/q_lse": 0.0,
        "probe/lse_minus_q_data": 0.0,
        "probe/q_pi": 0.0,
        "probe/q_curr": 0.0,
        "probe/q_rand": 0.0,
    }
    fork_devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(int(seed))
        for start in range(0, n_total, max(1, int(chunk_size))):
            sl = slice(start, start + chunk_size)
            obs, next_obs, critic_obs, actions = obs_all[sl], next_obs_all[sl], critic_obs_all[sl], actions_all[sl]
            b = int(actions.shape[0])

            q1, q2 = qnet(critic_obs, actions)
            q_data = torch.minimum(q1, q2)

            pi_det = actor(obs)[0]
            q1_pi, q2_pi = qnet(critic_obs, pi_det)
            q_pi = torch.minimum(q1_pi, q2_pi)

            exp_obs = obs[:, None, :].expand(b, num_repeat, -1).reshape(b * num_repeat, -1)
            exp_next_obs = next_obs[:, None, :].expand(b, num_repeat, -1).reshape(b * num_repeat, -1)
            exp_critic_obs = critic_obs[:, None, :].expand(b, num_repeat, -1).reshape(b * num_repeat, -1)
            curr_actions, curr_logp = actor.get_actions_and_log_probs(exp_obs)
            next_actions, next_logp = actor.get_actions_and_log_probs(exp_next_obs)
            rand_actions = torch.empty(b * num_repeat, n_act, device=device, dtype=actions.dtype).uniform_(-1.0, 1.0)
            if use_tanh:
                rand_actions = rand_actions * action_scale + action_bias

            q1_rand, q2_rand = qnet(exp_critic_obs, rand_actions)
            q1_curr, q2_curr = qnet(exp_critic_obs, curr_actions)
            q1_next, q2_next = qnet(exp_critic_obs, next_actions)
            q1_rand, q2_rand = q1_rand.view(b, num_repeat), q2_rand.view(b, num_repeat)
            q1_curr, q2_curr = q1_curr.view(b, num_repeat), q2_curr.view(b, num_repeat)
            q1_next, q2_next = q1_next.view(b, num_repeat), q2_next.view(b, num_repeat)
            curr_logp = curr_logp.reshape(b, num_repeat)
            next_logp = next_logp.reshape(b, num_repeat)

            cat_q1 = torch.cat([q1_rand - random_density, q1_curr - curr_logp, q1_next - next_logp], dim=1)
            cat_q2 = torch.cat([q2_rand - random_density, q2_curr - curr_logp, q2_next - next_logp], dim=1)
            lse1 = torch.logsumexp(cat_q1 / temperature, dim=1) * temperature
            lse2 = torch.logsumexp(cat_q2 / temperature, dim=1) * temperature

            sums["probe/q_level"] += float(q_data.sum())
            sums["probe/q_lse"] += float((0.5 * (lse1 + lse2)).sum())
            sums["probe/lse_minus_q_data"] += float((0.5 * ((lse1 - q1) + (lse2 - q2))).sum())
            sums["probe/q_pi"] += float(q_pi.sum())
            sums["probe/q_curr"] += float((0.5 * (q1_curr + q2_curr)).mean(dim=1).sum())
            sums["probe/q_rand"] += float((0.5 * (q1_rand + q2_rand)).mean(dim=1).sum())

    return {key: torch.as_tensor(value / n_total, device=device) for key, value in sums.items()}
