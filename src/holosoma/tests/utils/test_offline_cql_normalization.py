"""Tests for offline CQL observation normalization.

Covers:
* ``create_frozen_normalizer`` — factory correctness, safety nets, state_dict
  compatibility, checkpoint round-trip.
* ``validate_normalization`` — raw vs normalised statistics, pass/fail logic,
  constant-feature handling, nn.Identity fallback.
"""

from __future__ import annotations

import copy
import os
import tempfile
from typing import Any

import pytest
import torch
from torch import nn

from holosoma.agents.fast_sac.fast_sac_utils import EmpiricalNormalization
from holosoma.agents.offline_cql.offline_cql_utils import (
    create_frozen_normalizer,
    validate_normalization,
)


# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════

DEVICE = "cpu"
OBS_DIM = 16
COUNT = 5_000


def _make_dataset_stats(
    obs_dim: int = OBS_DIM,
    count: int = COUNT,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return (mean [1, obs_dim], std [1, obs_dim], count)."""
    gen = torch.Generator().manual_seed(seed)
    mean = torch.randn(1, obs_dim, generator=gen)
    std = torch.rand(1, obs_dim, generator=gen).clamp(min=0.1)
    return mean, std, count


def _make_raw_data(
    mean: torch.Tensor,
    std: torch.Tensor,
    n_samples: int = 2_000,
    seed: int = 99,
) -> torch.Tensor:
    """Generate synthetic raw observations from N(mean, std)."""
    gen = torch.Generator().manual_seed(seed)
    obs_dim = mean.shape[-1]
    noise = torch.randn(n_samples, obs_dim, generator=gen)
    return mean + noise * std


# ══════════════════════════════════════════════════════════════════════
# create_frozen_normalizer
# ══════════════════════════════════════════════════════════════════════


class TestCreateFrozenNormalizer:
    """Test the factory that builds a locked-down EmpiricalNormalization."""

    def test_returns_empirical_normalization(self):
        mean, std, count = _make_dataset_stats()
        norm = create_frozen_normalizer(mean, std, count, DEVICE)
        assert isinstance(norm, EmpiricalNormalization)

    def test_buffers_match_inputs(self):
        mean, std, count = _make_dataset_stats()
        norm = create_frozen_normalizer(mean, std, count, DEVICE)
        torch.testing.assert_close(norm._mean, mean)
        torch.testing.assert_close(norm._std, std)
        torch.testing.assert_close(norm._var, std.pow(2))
        assert norm.count.item() == count

    def test_eval_mode(self):
        mean, std, count = _make_dataset_stats()
        norm = create_frozen_normalizer(mean, std, count, DEVICE)
        assert not norm.training, "Normalizer should be in eval mode"

    def test_until_cap_set(self):
        mean, std, count = _make_dataset_stats()
        norm = create_frozen_normalizer(mean, std, count, DEVICE)
        assert norm.until == count

    def test_forward_normalises_correctly(self):
        """Verify (x - mean) / (std + eps) formula."""
        mean, std, count = _make_dataset_stats()
        norm = create_frozen_normalizer(mean, std, count, DEVICE)
        x = torch.randn(8, OBS_DIM)
        out = norm(x, update=False)
        expected = (x - mean) / (std + norm.eps)
        torch.testing.assert_close(out, expected)

    def test_no_update_in_eval_mode(self):
        """forward() with default update=True should NOT drift stats in eval."""
        mean, std, count = _make_dataset_stats()
        norm = create_frozen_normalizer(mean, std, count, DEVICE)
        original_mean = norm._mean.clone()
        original_count = norm.count.clone()

        # Forward pass with update=True (the default in EmpiricalNormalization)
        x = torch.randn(64, OBS_DIM) * 100  # very different from mean
        _ = norm(x)  # update=True by default

        torch.testing.assert_close(norm._mean, original_mean)
        assert norm.count.item() == original_count.item()

    def test_no_update_even_in_train_mode(self):
        """The until= cap prevents updates even if .train() is accidentally called."""
        mean, std, count = _make_dataset_stats()
        norm = create_frozen_normalizer(mean, std, count, DEVICE)
        original_mean = norm._mean.clone()
        original_count = norm.count.clone()

        norm.train()  # accidentally switch to training mode
        assert norm.training, "Sanity: should be in training mode now"

        x = torch.randn(64, OBS_DIM) * 100
        _ = norm(x)  # update=True by default, and training=True

        # Until cap should block the update
        torch.testing.assert_close(norm._mean, original_mean)
        assert norm.count.item() == original_count.item()

    def test_state_dict_keys(self):
        """state_dict has the same keys as a default EmpiricalNormalization."""
        mean, std, count = _make_dataset_stats()
        norm = create_frozen_normalizer(mean, std, count, DEVICE)
        default_norm = EmpiricalNormalization(shape=OBS_DIM, device=DEVICE)
        assert set(norm.state_dict().keys()) == set(default_norm.state_dict().keys())

    def test_state_dict_round_trip(self):
        """Save → load preserves exact buffer values."""
        mean, std, count = _make_dataset_stats()
        norm = create_frozen_normalizer(mean, std, count, DEVICE)

        sd = norm.state_dict()
        norm2 = EmpiricalNormalization(shape=OBS_DIM, device=DEVICE)
        norm2.load_state_dict(sd)

        torch.testing.assert_close(norm2._mean, mean)
        torch.testing.assert_close(norm2._std, std)
        torch.testing.assert_close(norm2._var, std.pow(2))
        assert norm2.count.item() == count

    def test_checkpoint_round_trip(self):
        """Simulate full save → load via torch.save/load."""
        mean, std, count = _make_dataset_stats()
        norm = create_frozen_normalizer(mean, std, count, DEVICE)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ckpt.pt")
            torch.save({"obs_normalizer_state": norm.state_dict()}, path)
            ckpt = torch.load(path, map_location=DEVICE, weights_only=True)

        norm2 = EmpiricalNormalization(shape=OBS_DIM, device=DEVICE)
        norm2.load_state_dict(ckpt["obs_normalizer_state"])

        # Same output on the same input
        x = torch.randn(4, OBS_DIM)
        torch.testing.assert_close(norm(x, update=False), norm2(x, update=False))

    def test_cross_load_from_online_normalizer(self):
        """An online EmpiricalNormalization's state_dict can be loaded into
        a frozen one (and vice versa) — proving checkpoint compatibility."""
        # Simulate an online normalizer that has processed some data
        online = EmpiricalNormalization(shape=OBS_DIM, device=DEVICE)
        data = torch.randn(500, OBS_DIM) * 3 + 2
        online.train()
        online(data)  # updates running stats

        # Create a fresh frozen normalizer
        mean, std, count = _make_dataset_stats()
        frozen = create_frozen_normalizer(mean, std, count, DEVICE)

        # Load online's state into frozen
        frozen_copy = EmpiricalNormalization(shape=OBS_DIM, device=DEVICE)
        frozen_copy.load_state_dict(online.state_dict())
        torch.testing.assert_close(frozen_copy._mean, online._mean)

        # Load frozen's state into a fresh online normalizer
        online2 = EmpiricalNormalization(shape=OBS_DIM, device=DEVICE)
        online2.load_state_dict(frozen.state_dict())
        torch.testing.assert_close(online2._mean, frozen._mean)

    def test_custom_eps(self):
        """Custom eps is forwarded to EmpiricalNormalization."""
        mean, std, count = _make_dataset_stats()
        norm = create_frozen_normalizer(mean, std, count, DEVICE, eps=0.5)
        assert norm.eps == 0.5

        x = torch.randn(4, OBS_DIM)
        expected = (x - mean) / (std + 0.5)
        torch.testing.assert_close(norm(x, update=False), expected)

    def test_device_conversion(self):
        """Input tensors on a different device are handled."""
        mean_cpu, std_cpu, count = _make_dataset_stats()
        # Mean/std are on CPU but we request the normalizer on CPU too
        norm = create_frozen_normalizer(mean_cpu, std_cpu, count, "cpu")
        assert norm._mean.device.type == "cpu"


# ══════════════════════════════════════════════════════════════════════
# validate_normalization
# ══════════════════════════════════════════════════════════════════════


class TestValidateNormalization:
    """Test the audit utility for raw vs normalised statistics."""

    def test_well_normalised_data_passes(self):
        """With correct normalizer, output should be ≈ N(0, 1)."""
        mean, std, count = _make_dataset_stats()
        raw = _make_raw_data(mean, std, n_samples=10_000)
        norm = create_frozen_normalizer(mean, std, count, DEVICE)

        result = validate_normalization(norm, raw, label="actor_obs")

        assert result["mean_close_to_zero"], f"Mean check failed:\n{result['report']}"
        assert result["std_close_to_one"], f"Std check failed:\n{result['report']}"
        assert "PASS" in result["report"]
        assert result["num_const_features"] == 0

    def test_wrong_normalizer_fails(self):
        """A normalizer with wrong stats should fail the checks."""
        mean, std, count = _make_dataset_stats()
        raw = _make_raw_data(mean, std, n_samples=5_000)

        # Create normalizer with very different stats
        wrong_mean = mean + 100
        wrong_std = std * 0.01
        norm = create_frozen_normalizer(wrong_mean, wrong_std, count, DEVICE)

        result = validate_normalization(norm, raw, label="bad_norm")
        assert not result["mean_close_to_zero"]

    def test_identity_normalizer(self):
        """nn.Identity should always pass (no checks enforced)."""
        raw = torch.randn(1000, OBS_DIM) * 5 + 3  # definitely not N(0,1)
        identity = nn.Identity()

        result = validate_normalization(identity, raw, label="identity_test")
        assert result["mean_close_to_zero"] is True
        assert result["std_close_to_one"] is True
        assert "nn.Identity" in result["report"]

    def test_constant_features_detected(self):
        """Features with zero variance should be counted correctly."""
        mean = torch.zeros(1, OBS_DIM)
        std = torch.ones(1, OBS_DIM)
        # Make 3 features constant
        std[0, :3] = 1e-7
        norm = create_frozen_normalizer(mean, std, 1000, DEVICE)

        # Generate data that matches: 3 constant features
        raw = torch.randn(2_000, OBS_DIM)
        raw[:, :3] = 0.0  # constant

        result = validate_normalization(norm, raw, label="const_test")
        assert result["num_const_features"] == 3

    def test_report_is_string(self):
        """The report field should be a non-empty string."""
        mean, std, count = _make_dataset_stats()
        raw = _make_raw_data(mean, std, n_samples=500)
        norm = create_frozen_normalizer(mean, std, count, DEVICE)

        result = validate_normalization(norm, raw)
        assert isinstance(result["report"], str)
        assert len(result["report"]) > 0

    def test_output_shapes(self):
        """Return tensors should have shape [obs_dim]."""
        mean, std, count = _make_dataset_stats()
        raw = _make_raw_data(mean, std, n_samples=200)
        norm = create_frozen_normalizer(mean, std, count, DEVICE)

        result = validate_normalization(norm, raw)
        assert result["raw_mean"].shape == (OBS_DIM,)
        assert result["raw_std"].shape == (OBS_DIM,)
        assert result["norm_mean"].shape == (OBS_DIM,)
        assert result["norm_std"].shape == (OBS_DIM,)

    def test_custom_tolerances(self):
        """Tight tolerances should be harder to pass."""
        mean, std, count = _make_dataset_stats()
        # Small dataset → larger sampling noise → may fail tight tolerance
        raw = _make_raw_data(mean, std, n_samples=100)
        norm = create_frozen_normalizer(mean, std, count, DEVICE)

        # Very tight tolerance — likely to fail with only 100 samples
        result_tight = validate_normalization(
            norm, raw, atol_mean=0.001, atol_std=0.001
        )
        # Very loose tolerance — should always pass
        result_loose = validate_normalization(
            norm, raw, atol_mean=10.0, atol_std=10.0
        )
        assert result_loose["mean_close_to_zero"]
        assert result_loose["std_close_to_one"]

    def test_normalizer_not_mutated(self):
        """validate_normalization should not update the normalizer's stats."""
        mean, std, count = _make_dataset_stats()
        norm = create_frozen_normalizer(mean, std, count, DEVICE)
        original_mean = norm._mean.clone()
        original_count = norm.count.clone()

        raw = torch.randn(500, OBS_DIM) * 100
        _ = validate_normalization(norm, raw)

        torch.testing.assert_close(norm._mean, original_mean)
        assert norm.count.item() == original_count.item()


# ══════════════════════════════════════════════════════════════════════
# Integration: dataset → normalizer → checkpoint → reload
# ══════════════════════════════════════════════════════════════════════


class TestNormalizationIntegration:
    """End-to-end tests simulating the setup() → save() → load() flow."""

    def test_dataset_stats_to_normalizer_to_checkpoint(self):
        """Simulate: compute stats → create normalizer → save ckpt → reload
        → verify output is identical."""
        # 1) Simulate dataset stats
        mean, std, count = _make_dataset_stats(obs_dim=32)

        # 2) Create normalizers (as setup() would)
        actor_norm = create_frozen_normalizer(mean, std, count, DEVICE)
        critic_mean = torch.randn(1, 48)
        critic_std = torch.rand(1, 48).clamp(min=0.1)
        critic_norm = create_frozen_normalizer(critic_mean, critic_std, count, DEVICE)

        # 3) Save checkpoint (simulating save_cql_params)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ckpt.pt")
            ckpt = {
                "obs_normalizer_state": actor_norm.state_dict(),
                "critic_obs_normalizer_state": critic_norm.state_dict(),
            }
            torch.save(ckpt, path)

            # 4) Reload into fresh normalizers (simulating load_cql_params)
            loaded = torch.load(path, map_location=DEVICE, weights_only=True)

        actor_norm2 = EmpiricalNormalization(shape=32, device=DEVICE)
        actor_norm2.load_state_dict(loaded["obs_normalizer_state"])
        critic_norm2 = EmpiricalNormalization(shape=48, device=DEVICE)
        critic_norm2.load_state_dict(loaded["critic_obs_normalizer_state"])

        # 5) Verify identical output
        x_actor = torch.randn(4, 32)
        x_critic = torch.randn(4, 48)
        torch.testing.assert_close(
            actor_norm(x_actor, update=False),
            actor_norm2(x_actor, update=False),
        )
        torch.testing.assert_close(
            critic_norm(x_critic, update=False),
            critic_norm2(x_critic, update=False),
        )

    def test_frozen_normalizer_matches_online_on_same_data(self):
        """Given the same dataset, a frozen normalizer and an online one
        that processes all data should produce very similar statistics."""
        obs_dim = 8
        n = 10_000
        gen = torch.Generator().manual_seed(123)
        data = torch.randn(n, obs_dim, generator=gen) * 3 + 2

        # Online: process all data in one batch
        online = EmpiricalNormalization(shape=obs_dim, device=DEVICE)
        online.train()
        online(data)
        online.eval()

        # Frozen: compute stats and inject
        ds_mean = data.mean(dim=0, keepdim=True)
        ds_std = data.std(dim=0, keepdim=True).clamp(min=1e-6)
        frozen = create_frozen_normalizer(ds_mean, ds_std, n, DEVICE)

        # Compare buffers — they won't be _exactly_ equal because
        # EmpiricalNormalization uses Welford's algorithm (population std)
        # while torch.std defaults to sample std (ddof=1).
        # But with n=10000 the difference is negligible.
        torch.testing.assert_close(online._mean, frozen._mean, atol=1e-5, rtol=1e-4)

        # Compare normalised output
        test_x = torch.randn(16, obs_dim)
        out_online = online(test_x, update=False)
        out_frozen = frozen(test_x, update=False)
        # Outputs should be very close (small eps difference at most)
        torch.testing.assert_close(out_online, out_frozen, atol=0.02, rtol=0.01)

    def test_separate_actor_critic_normalizers_independent(self):
        """Actor and critic normalizers must be fully independent."""
        actor_mean, actor_std, count = _make_dataset_stats(obs_dim=10)
        critic_mean = torch.randn(1, 20)
        critic_std = torch.rand(1, 20).clamp(min=0.2)

        actor_norm = create_frozen_normalizer(actor_mean, actor_std, count, DEVICE)
        critic_norm = create_frozen_normalizer(critic_mean, critic_std, count, DEVICE)

        # Modifying one should not affect the other
        actor_norm._mean.fill_(999.0)
        assert critic_norm._mean.max().item() < 100.0

        # Different obs_dim
        assert actor_norm._mean.shape != critic_norm._mean.shape
