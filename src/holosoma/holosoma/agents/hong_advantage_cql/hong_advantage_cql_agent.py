"""Hong-Advantage CQL: CQL trained on Hong et al. (ICLR 2023) advantage-weighted trajectory sampling.

"Harnessing Mixed Offline Reinforcement Learning Datasets via Trajectory Weighting" re-weights the
*sampling distribution* of the dataset by trajectory-level advantage; the learning objective is left
untouched. This agent reproduces that recipe on top of the unmodified CQL trainer:

    G_i      = sum_t r_{i,t}                              undiscounted episodic return of trajectory i
    V_mu     = argmin_V sum_i (G_i - V(s_{i,0}))^2        LINEAR regression on the initial state (fit once)
    A_i      = G_i - V_mu(s_{i,0})                        trajectory advantage (shared by all its transitions)
    A~_i     = (A_i - min A) / (max A - min A + eps)      dataset-wide max-min normalisation
    q_i      = exp(A~_i / temperature)                    Boltzmann priority, temperature 0.2 (paper default for CQL)
    p_{i,t}  = q_i / sum_j T_j q_j                        every transition of trajectory i gets q_i; normalised over transitions

Minibatches are drawn WITH replacement from p. Nothing else changes: the CQL critic Bellman loss,
conservative regulariser, actor loss, alpha/Lagrange paths and target updates are those of CQLAgent
(this class overrides no loss hook), and the sampled batch carries no weight that any loss could use.

Not Hong et al. and therefore not done here: w * (LSE - Q_D) (PRe/AW-CQL), weighted Bellman or actor
losses, H-step returns, progress-bin baselines, learned per-transition V_H, iterated value fitting.
"""

from __future__ import annotations

import h5py
import numpy as np
from loguru import logger

from holosoma.agents.cql.cql_agent import CQLAgent
from holosoma.config_types.algo import HongAdvantageCQLConfig
from holosoma.envs.base_task.base_task import BaseTask
from holosoma.utils.safe_torch_import import torch


def episode_bounds_from_h5(dones: np.ndarray, truncations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inclusive (start, end) row indices of every episode.

    A row ends an episode when ``dones`` or ``truncations`` is set (the exporter's convention, also used by
    the offline agents and aw_precompute_weights.py); the last row of the file always closes an episode.
    """
    end = (dones.astype(bool) | truncations.astype(bool)).copy()
    end[-1] = True
    ends = np.flatnonzero(end)
    starts = np.concatenate([[0], ends[:-1] + 1])
    return starts, ends


def fit_linear_value(
    x0: np.ndarray, g: np.ndarray, rcond: float = 1e-6, const_std: float = 1e-6
) -> tuple[np.ndarray, float, np.ndarray]:
    """Linear least squares with intercept: returns (coef, intercept, predictions) in the raw feature space.

    Initial states of a tracking dataset are rank-deficient (constant columns, reference-phase-determined
    joints, duplicated history slots; cond ~1e11), where a plain lstsq fits noise directions with huge
    coefficients. So constant columns are dropped, the rest standardised, and the minimum-norm solution is
    taken over singular directions above ``rcond * s_max`` — the same OLS fit restricted to the identifiable
    subspace (sklearn LinearRegression's estimator, made numerically stable).

    The SVD runs in torch (float64, CPU): numpy's bundled OpenBLAS LAPACK in the hssim env returns wrong
    singular values / lstsq solutions even for well-conditioned random matrices.
    """
    x = np.asarray(x0, dtype=np.float64)
    y = np.asarray(g, dtype=np.float64)
    mean_x, std_x = x.mean(axis=0), x.std(axis=0)
    keep = std_x > const_std
    mean_y = y.mean()
    coef = np.zeros(x.shape[1])
    if keep.any():
        z = (x[:, keep] - mean_x[keep]) / std_x[keep]
        u, sv, vt = (t.numpy() for t in torch.linalg.svd(torch.from_numpy(z), full_matrices=False))
        rank = sv > rcond * sv[0]
        coef_z = vt[rank].T @ ((u[:, rank].T @ (y - mean_y)) / sv[rank])
        coef[keep] = coef_z / std_x[keep]
    intercept = float(mean_y - mean_x @ coef)
    return coef, intercept, x @ coef + intercept


def hong_transition_probabilities(
    g: np.ndarray, v0: np.ndarray, lengths: np.ndarray, temperature: float, eps: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Trajectory advantage -> max-min normalised -> Boltzmann priority -> transition probabilities."""
    adv = np.asarray(g, dtype=np.float64) - np.asarray(v0, dtype=np.float64)
    adv_norm = (adv - adv.min()) / (adv.max() - adv.min() + eps)
    priority = np.exp(adv_norm / temperature)
    transition_priority = np.repeat(priority, lengths)  # same q_i for every transition of trajectory i
    prob = transition_priority / transition_priority.sum()
    return adv, adv_norm, priority, prob


class HongAdvantageCQLAgent(CQLAgent):
    """CQLAgent whose offline minibatches are drawn from Hong et al.'s trajectory-advantage distribution."""

    config: HongAdvantageCQLConfig

    def __init__(self, env: BaseTask, config: HongAdvantageCQLConfig, device: str, log_dir: str,
                 multi_gpu_cfg: dict | None = None):
        self._hong_prob: torch.Tensor | None = None
        self._hong_generator: torch.Generator | None = None
        self._hong_stats: dict[str, float] = {}
        super().__init__(env, config, device, log_dir, multi_gpu_cfg)

    def setup(self) -> None:
        super().setup()
        if not self.config.hong_aw_enabled:
            logger.warning("Hong-AW disabled (hong_aw_enabled=False): this run is plain uniform-sampling CQL.")
            return
        if self._offline_gpu_cache is None:
            raise RuntimeError(
                "Hong-AW sampling needs the whole dataset resident (use_gpu_cache=True); the RAM shuffle-buffer "
                "backend streams blocks and cannot sample from a dataset-wide distribution."
            )
        prob, stats = self._compute_hong_probabilities()
        self._hong_prob = torch.as_tensor(prob, dtype=torch.float64, device=self.device)
        self._hong_stats = stats
        # Minibatch RNG: seeded from the training seed (set by seeding() before the agent is built) so that
        # each seed's sampling stream is reproducible; the probabilities themselves are deterministic.
        self._hong_generator = torch.Generator(device=self.device)
        self._hong_generator.manual_seed(int(torch.initial_seed()) % (2**63 - 1))
        self._install_weighted_sampler()

    # ------------------------------------------------------------------ precompute
    def _compute_hong_probabilities(self) -> tuple[np.ndarray, dict[str, float]]:
        cfg = self.config
        with h5py.File(self._offline_dataset_path, "r") as f:
            n = int(f.attrs.get("num_samples", f["observations"].shape[0]))  # same rule as GPUTransitionCache
            rewards = np.asarray(f["rewards"][:n], dtype=np.float64).reshape(-1)
            dones = np.asarray(f["dones"][:n]).reshape(-1)
            truncations = np.asarray(f["truncations"][:n]).reshape(-1) if "truncations" in f else np.zeros(n, bool)
            if cfg.hong_aw_obs_key not in f:
                raise KeyError(f"hong_aw_obs_key='{cfg.hong_aw_obs_key}' not in {self._offline_dataset_path}; keys={list(f.keys())}")
            starts, ends = episode_bounds_from_h5(dones, truncations)
            x0 = np.asarray(f[cfg.hong_aw_obs_key][np.sort(starts)], dtype=np.float32)  # initial state of every episode
        if n != self._offline_num_samples:
            raise RuntimeError(f"H5 has {n} rows but the sampler holds {self._offline_num_samples}")
        lengths = ends - starts + 1
        cumsum = np.concatenate([[0.0], np.cumsum(rewards)])
        g = cumsum[ends + 1] - cumsum[starts]  # undiscounted full-trajectory return

        coef, intercept, v0 = fit_linear_value(x0, g)
        adv, adv_norm, priority, prob = hong_transition_probabilities(g, v0, lengths, cfg.hong_aw_temperature, cfg.hong_aw_eps)
        ess = 1.0 / float(np.square(prob).sum())
        r2 = 1.0 - float(np.square(g - v0).sum() / max(np.square(g - g.mean()).sum(), 1e-12))

        # sanity checks (fail loudly: a wrong distribution would silently change the experiment)
        assert abs(prob.sum() - 1.0) < 1e-6, prob.sum()
        assert prob.shape == (n,)
        check_eps = [0, len(starts) // 2, len(starts) - 1]
        for i in check_eps:
            seg = prob[starts[i] : ends[i] + 1]
            assert np.allclose(seg, seg[0]), "transitions of one episode must share a priority"
        hi, lo = adv >= np.quantile(adv, 0.9), adv <= np.quantile(adv, 0.1)
        assert priority[hi].mean() > priority[lo].mean(), "higher trajectory advantage must give higher priority"

        stats = {
            "num_transitions": float(n), "num_episodes": float(len(starts)),
            "episode_length_mean": float(lengths.mean()), "episode_length_std": float(lengths.std()),
            "episode_length_min": float(lengths.min()), "episode_length_max": float(lengths.max()),
            "return_mean": float(g.mean()), "return_std": float(g.std()), "return_min": float(g.min()), "return_max": float(g.max()),
            "predicted_V0_mean": float(v0.mean()), "predicted_V0_std": float(v0.std()), "V0_fit_r2": r2,
            "advantage_mean": float(adv.mean()), "advantage_std": float(adv.std()), "advantage_min": float(adv.min()), "advantage_max": float(adv.max()),
            "normalized_advantage_mean": float(adv_norm.mean()), "normalized_advantage_std": float(adv_norm.std()),
            "normalized_advantage_min": float(adv_norm.min()), "normalized_advantage_max": float(adv_norm.max()),
            "temperature": float(cfg.hong_aw_temperature),
            "weight_mean": float(priority.mean()), "weight_std": float(priority.std()),
            "transition_prob_min": float(prob.min()), "transition_prob_max": float(prob.max()),
            "max_sampling_prob": float(prob.max()), "uniform_prob": 1.0 / n,
            "ess": ess, "ess_ratio": ess / n,
            "mass_on_top10pct_adv_trajectories": float(prob[np.repeat(hi, lengths)].sum()),
        }
        logger.info(
            "[Hong-AW] state fields for V_mu(s_0): '{}' ({} dims, linear regression with intercept, R2={:.4f})",
            cfg.hong_aw_obs_key, x0.shape[1], r2,
        )
        logger.info("[Hong-AW] num_transitions={num_transitions:.0f} num_episodes={num_episodes:.0f}", **stats)
        logger.info("[Hong-AW] episode_length mean/std/min/max = {episode_length_mean:.1f}/{episode_length_std:.1f}/{episode_length_min:.0f}/{episode_length_max:.0f}", **stats)
        logger.info("[Hong-AW] trajectory_return mean/std/min/max = {return_mean:.3f}/{return_std:.3f}/{return_min:.3f}/{return_max:.3f}", **stats)
        logger.info("[Hong-AW] predicted_V0 mean/std = {predicted_V0_mean:.3f}/{predicted_V0_std:.3f}", **stats)
        logger.info("[Hong-AW] advantage mean/std/min/max = {advantage_mean:.3f}/{advantage_std:.3f}/{advantage_min:.3f}/{advantage_max:.3f}", **stats)
        logger.info("[Hong-AW] normalized_advantage mean/std/min/max = {normalized_advantage_mean:.4f}/{normalized_advantage_std:.4f}/{normalized_advantage_min:.4f}/{normalized_advantage_max:.4f}", **stats)
        logger.info("[Hong-AW] AW_temperature={temperature}  transition_probability min/max = {transition_prob_min:.3e}/{transition_prob_max:.3e} (uniform {uniform_prob:.3e})", **stats)
        logger.info("[Hong-AW] ESS={ess:.1f}  ESS/N={ess_ratio:.4f}  mass on top-10% trajectories={mass_on_top10pct_adv_trajectories:.3f}", **stats)
        logger.info("[Hong-AW] sanity: prob sums to 1, per-episode priorities constant, high-advantage > low-advantage priority; "
                    "no sampling weight enters any CQL loss (this agent overrides no loss hook).")
        return prob, stats

    # ------------------------------------------------------------------ sampler
    def _install_weighted_sampler(self) -> None:
        """Replace the GPU cache's uniform index draw by a draw from the Hong transition distribution.

        Only the index generation changes; batch assembly, normalisation and augmentation stay identical.
        """
        cache = self._offline_gpu_cache
        assert cache is not None and self._hong_prob is not None and self._hong_generator is not None
        prob, gen = self._hong_prob, self._hong_generator
        storage_sampler = cache.sample

        def sample_weighted(batch_size: int):
            from holosoma.data.hdf5_offline_dataset import _index_nested_batch  # noqa: PLC0415

            if cache._storage is None:
                raise RuntimeError("GPUTransitionCache storage is empty.")
            idx = torch.multinomial(prob, batch_size, replacement=True, generator=gen)
            batch = _index_nested_batch(cache._storage, idx, pin_memory=False)
            batch["dataset_index"] = idx.to(torch.long)
            return batch

        sample_weighted.uniform_sample = storage_sampler  # type: ignore[attr-defined]
        cache.sample = sample_weighted  # type: ignore[method-assign]
        logger.info("[Hong-AW] offline sampler: torch.multinomial over the trajectory-advantage distribution (with replacement)")

    # ------------------------------------------------------------------ logging
    def _extra_log_dicts(self) -> dict[str, dict[str, float]]:
        out = super()._extra_log_dicts()
        if self._hong_stats:
            out["hong_aw"] = {
                "return_mean": self._hong_stats["return_mean"],
                "advantage_std": self._hong_stats["advantage_std"],
                "weight_mean": self._hong_stats["weight_mean"],
                "weight_std": self._hong_stats["weight_std"],
                "ess_ratio": self._hong_stats["ess_ratio"],
                "max_sampling_prob": self._hong_stats["max_sampling_prob"],
                "max_prob_over_uniform": self._hong_stats["max_sampling_prob"] / self._hong_stats["uniform_prob"],
            }
        return out
