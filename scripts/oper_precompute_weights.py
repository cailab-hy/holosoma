#!/usr/bin/env python3
# ruff: noqa: E501
"""ODPR-A / OPER-A priority weights for a holosoma offline transition dataset.

Re-implementation of the first stage of ODPR ("Decoupled Prioritized Resampling", Yue et al.,
official code github.com/yueyang130/ODPR, files main.py / advantage.py / utils.py), applied to our
H5 replay datasets so the result can be compared with the AW-CQL sidecar (scripts/aw_precompute_weights.py).

Algorithm (defaults = the README command ``--n_step 1 --iter 5 --first_eval_steps 1e6 --bc_eval_steps 1e6``):
  1. states are z-normalised (ReplayBuffer.normalize_states, eps=1e-3);
  2. a behaviour value function V(s) (DoubleValueNet, min of two 256-256 ReLU MLPs) is fitted by TD(0):
       target = r + gamma * (1 - terminal) * min(V'_1, V'_2)(s')      (Polyak target, tau=0.005)
     Adam 3e-4, batch 256, cosine LR decay over the stage; mini-batches are drawn from the *rebalanced*
     behaviour distribution (uniform in stage 1, then proportional to the current priority weights);
  3. after every stage the advantage of every transition is evaluated with the online net,
       adv_i = r_i + gamma * (1 - terminal_i) * V(s'_i) - V(s_i),
     shifted to be non-negative (adv - min adv), normalised to mean one, multiplied into the running
     weight, renormalised to mean one, and used as the sampling distribution of the next stage;
  4. the weight after every stage is stored (ODPR saves ``bc_eval_results[iter]``); the case studies
     use iteration ``--iter`` and scale the weights' std at load time (see ``replace_weights``).

Differences from the official code are limited to data plumbing: terminals come from the H5 ``dones``
column (episode end), exactly as our offline agents treat them, and observations are taken from the key
given by ``--obs-key`` (default: critic_observations, the critic input of our CQL family).

Output npz keys: weight (final iteration, mean one), weights_by_iter [iter, N], advantage, value, td_target
(all from the final evaluation), per-iteration ess/std stats, config, and the same ``rhash`` pairing hash
as the AW sidecar.

Usage:
  python scripts/oper_precompute_weights.py offline_data/<dataset>.h5 --seed 1 --out <dataset>.h5.oper_a.seed1.npz
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import lr_scheduler


# ----------------------------------------------------------------------------- data
def reward_fingerprint(rewards: np.ndarray) -> str:
    r = np.asarray(rewards, dtype=np.float64).reshape(-1)
    return hashlib.sha256(
        np.ascontiguousarray(r[:1000]).tobytes() + np.ascontiguousarray(r[-1000:]).tobytes()
    ).hexdigest()[:16]


def load_h5(path: Path, obs_key: str):
    with h5py.File(path, "r") as f:
        n = int(f.attrs.get("num_samples", f["rewards"].shape[0]))
        next_key = "next_" + obs_key
        if next_key not in f:
            raise KeyError(f"{next_key} missing in {path}; keys: {list(f.keys())}")
        state = f[obs_key][:n].astype(np.float32)
        next_state = f[next_key][:n].astype(np.float32)
        reward = f["rewards"][:n].astype(np.float32)
        dones = f["dones"][:n].astype(np.float32)  # episode end = terminal (how our offline agents treat it)
        phase = f["motion_phase"][:n].astype(np.float32) if "motion_phase" in f else None
        episode_id = f["episode_id"][:n] if "episode_id" in f else None
    return state, next_state, reward, dones, phase, episode_id


# ----------------------------------------------------------------------------- nets (advantage.py)
class ValueNet(nn.Module):
    def __init__(self, state_dim: int):
        super().__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

    def forward(self, s):
        return self.l3(F.relu(self.l2(F.relu(self.l1(s)))))


class DoubleValueNet(nn.Module):
    def __init__(self, state_dim: int):
        super().__init__()
        self.v1 = ValueNet(state_dim)
        self.v2 = ValueNet(state_dim)

    def forward(self, s):
        return self.v1(s), self.v2(s)


def cosine_or_linear(optimizer, schedule: str, total: int):
    def rule(step):
        if schedule == "cosine":
            return 0.5 * (1 + math.cos(step / total * math.pi))
        if schedule == "linear":
            return 1.0 - step / total
        return 1.0

    return lr_scheduler.LambdaLR(optimizer, lr_lambda=rule)


class ValueAdvantage:
    """V_Advantage / DoubleV_Advantage with td_type='nstep', n_step=1, adv_type='nstep'."""

    def __init__(self, state_dim, critic_type, discount, tau, lr, schedule, maxstep, device):
        self.double = critic_type == "doublev"
        self.value = (DoubleValueNet if self.double else ValueNet)(state_dim).to(device)
        self.value_target = copy.deepcopy(self.value)
        self.discount, self.tau, self.lr, self.schedule, self.device = discount, tau, lr, schedule, device
        self.reset_optimizer(maxstep)

    def reset_optimizer(self, maxstep):
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=self.lr)
        self.value_lr_scheduler = cosine_or_linear(self.value_optimizer, self.schedule, maxstep)

    @torch.no_grad()
    def get_value_target(self, s):
        if self.double:
            v1, v2 = self.value_target(s)
            return torch.minimum(v1, v2)
        return self.value_target(s)

    def get_value(self, s):
        if self.double:
            v1, v2 = self.value(s)
            return torch.minimum(v1, v2)
        return self.value(s)

    def train_step(self, s, s2, r, not_done):
        with torch.no_grad():
            v_target = r + not_done * self.discount * self.get_value_target(s2)
            assert v_target.shape == (s.shape[0], 1), v_target.shape
        if self.double:
            v1, v2 = self.value(s)
            loss = F.mse_loss(v1, v_target) + F.mse_loss(v2, v_target)
        else:
            loss = F.mse_loss(self.value(s), v_target)
        self.value_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.value_optimizer.step()
        with torch.no_grad():
            for p, tp in zip(self.value.parameters(), self.value_target.parameters()):
                tp.data.mul_(1 - self.tau).add_(self.tau * p.data)
        self.value_lr_scheduler.step()
        return float(loss)

    @torch.no_grad()
    def evaluate(self, state, next_state, reward, not_done, batch_size=65536):
        """adv = r + gamma (1-terminal) V(s') - V(s) with the ONLINE net (Advantage.eval, adv_type nstep, n=1)."""
        n = state.shape[0]
        values = torch.empty(n, device=self.device)
        next_values = torch.empty(n, device=self.device)
        for i in range(0, n, batch_size):
            sl = slice(i, min(i + batch_size, n))
            values[sl] = self.get_value(state[sl]).squeeze(1)
            next_values[sl] = self.get_value(next_state[sl]).squeeze(1)
        # per-row V in float32 like the reference nets; q/adv in float64 like its numpy arithmetic
        q = reward.squeeze(1).double() + not_done.squeeze(1).double() * self.discount * next_values.double()
        return q - values.double(), q, values.double()


# ----------------------------------------------------------------------------- sampler (utils.PrefetchBalancedSampler)
class PrefetchSampler:
    """Uniform (probs=None) or probs-proportional sampling with replacement, prefetched in blocks."""

    def __init__(self, n, batch_size, device, probs=None, n_prefetch=1000, generator=None):
        self.n, self.bs, self.device, self.gen = n, batch_size, device, generator
        self.n_prefetch = min(n_prefetch, n // batch_size)
        # float64 like numpy.random.choice in the reference: a float32 prefix sum over 2.5M rows quantises
        # per-row probabilities (~1e-7 each) and zeroes the low-weight tail
        self.probs = None if probs is None else (probs.double() / probs.double().sum()).to(device)
        self.cnt = self.n_prefetch - 1
        self.indices = None

    def sample(self):
        self.cnt = (self.cnt + 1) % self.n_prefetch
        if self.cnt == 0:
            m = self.bs * self.n_prefetch
            if self.probs is None:
                self.indices = torch.randint(0, self.n, (m,), device=self.device, generator=self.gen)
            else:
                assert self.probs.dtype == torch.float64
                self.indices = torch.multinomial(self.probs, m, replacement=True, generator=self.gen)
        return self.indices[self.cnt * self.bs : (self.cnt + 1) * self.bs]


def ess_frac(w: np.ndarray) -> float:
    w = np.asarray(w, dtype=np.float64)
    return float(w.sum() ** 2 / (len(w) * (w * w).sum()))


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("h5")
    ap.add_argument("--out", default=None, help="output npz (default: <h5>.oper_a.seed<seed>.npz)")
    ap.add_argument("--obs-key", default="critic_observations", choices=["critic_observations", "observations"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--iter", type=int, default=5, help="number of rebalancing iterations (ODPR --iter)")
    ap.add_argument("--first-eval-steps", type=int, default=1_000_000)
    ap.add_argument("--bc-eval-steps", type=int, default=1_000_000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--discount", type=float, default=0.99)
    ap.add_argument("--tau", type=float, default=0.005)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--critic-type", default="doublev", choices=["v", "doublev"])
    ap.add_argument("--bc-lr-schedule", default="cosine", choices=["cosine", "linear", "none"])
    ap.add_argument("--normalize", type=int, default=1, help="z-normalise states (ODPR --normalize)")
    ap.add_argument(
        "--scale", action="store_true", help="ODPR --scale: rescale weight std to --std after each iteration"
    )
    ap.add_argument("--std", type=float, default=2.0)
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--log-freq", type=int, default=50_000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-bins", type=int, default=20, help="phase bins for the per-bin mass report only")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    gen = torch.Generator(device=a.device)
    gen.manual_seed(a.seed)
    out = Path(a.out) if a.out else Path(a.h5 + f".oper_a.seed{a.seed}.npz")
    t0 = time.time()

    state_np, next_np, reward_np, dones_np, phase, episode_id = load_h5(Path(a.h5), a.obs_key)
    N, D = state_np.shape
    rhash = reward_fingerprint(reward_np)
    print(
        f"[load] N={N:,} obs_key={a.obs_key} dim={D} terminals={int(dones_np.sum()):,} rhash={rhash} ({time.time() - t0:.0f}s)"
    )

    dev = torch.device(a.device)
    state = torch.from_numpy(state_np).to(dev)
    next_state = torch.from_numpy(next_np).to(dev)
    del state_np, next_np
    if a.normalize:  # ReplayBuffer.normalize_states (eps=1e-3), stats from `state` only
        mean = state.mean(0, keepdim=True)
        std = state.std(0, keepdim=True, correction=0) + 1e-3  # numpy population std like the reference
        state = (state - mean) / std
        next_state = (next_state - mean) / std
    reward = torch.from_numpy(reward_np).to(dev).unsqueeze(1)  # [N, 1] like ODPR's ReplayBuffer
    not_done = (1.0 - torch.from_numpy(dones_np).to(dev)).unsqueeze(1)

    adv_model = ValueAdvantage(D, a.critic_type, a.discount, a.tau, a.lr, a.bc_lr_schedule, a.first_eval_steps, dev)
    sampler = PrefetchSampler(N, a.batch_size, dev, probs=None, generator=gen)  # stage 1: uniform
    weight = np.ones(N, dtype=np.float64)
    results = {"config": vars(a), "rhash": rhash, "iters": []}
    weights_by_iter = []
    bins = None if phase is None else np.clip((phase * a.n_bins).astype(int), 0, a.n_bins - 1)

    total_steps = a.bc_eval_steps * (a.iter - 1) + a.first_eval_steps
    loss_acc, loss_n = 0.0, 0
    for t in range(total_steps):
        idx = sampler.sample()
        loss_acc += adv_model.train_step(state[idx], next_state[idx], reward[idx], not_done[idx])
        loss_n += 1
        if (t + 1) % a.log_freq == 0:
            print(
                f"[train] step {t + 1:,}/{total_steps:,} value_loss {loss_acc / loss_n:.5f} lr {adv_model.value_optimizer.param_groups[0]['lr']:.2e} ({time.time() - t0:.0f}s)",
                flush=True,
            )
            loss_acc, loss_n = 0.0, 0
        if (t + 1 - a.first_eval_steps) >= 0 and (t + 1 - a.first_eval_steps) % a.bc_eval_steps == 0:
            curr = int((t + 1 - a.first_eval_steps) / a.bc_eval_steps + 1)
            adv, q, v = adv_model.evaluate(state, next_state, reward, not_done)
            adv_np = adv.cpu().numpy()
            padv = adv_np - adv_np.min()
            current_weight = padv / padv.sum() * N
            weight = weight * current_weight
            weight = weight / weight.sum() * N
            if a.scale:
                scale = a.std / weight.std()
                if scale > 1:
                    weight = np.maximum(scale * (weight - 1) + 1, a.eps)
                    weight = weight / weight.sum() * N
            weights_by_iter.append(weight.astype(np.float32))
            info = {
                "iter": curr,
                "step": t + 1,
                "adv_mean": float(adv_np.mean()),
                "adv_abs_mean": float(np.abs(adv_np).mean()),
                "adv_std": float(adv_np.std()),
                "adv_min": float(adv_np.min()),
                "adv_max": float(adv_np.max()),
                "q_mean": float(q.mean()),
                "v_mean": float(v.mean()),
                "weight_std": float(weight.std()),
                "weight_max": float(weight.max()),
                "ess_frac": ess_frac(weight),
            }
            if bins is not None:
                mass = np.array([weight[bins == k].sum() for k in range(a.n_bins)]) / weight.sum()
                info["bin_weight_mass"] = mass.round(5).tolist()
            results["iters"].append(info)
            print(
                f"[iter {curr}] "
                + " ".join(
                    f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in info.items()
                    if k != "bin_weight_mass"
                ),
                flush=True,
            )
            if bins is not None:
                print(
                    "        bin weight mass %: " + " ".join(f"{k}:{100 * m:.1f}" for k, m in enumerate(mass)),
                    flush=True,
                )
            if curr < a.iter:
                adv_model.reset_optimizer(a.bc_eval_steps)
                sampler = PrefetchSampler(
                    N, a.batch_size, dev, probs=torch.from_numpy(weight), generator=gen
                )  # reset_bc
            final = (adv_np.astype(np.float32), q.float().cpu().numpy(), v.float().cpu().numpy())

    adv_np, q_np, v_np = final
    np.savez(
        out,
        weight=weight.astype(np.float32),
        weights_by_iter=np.stack(weights_by_iter),
        advantage=adv_np,
        td_target=q_np,
        value=v_np,
        phase_bin=(bins.astype(np.int16) if bins is not None else np.zeros(0, np.int16)),
        rhash=rhash,
        n=N,
        obs_key=a.obs_key,
        iter=a.iter,
        discount=a.discount,
        config_json=json.dumps(vars(a)),
        iters_json=json.dumps(results["iters"]),
    )
    print(f"[saved] {out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
