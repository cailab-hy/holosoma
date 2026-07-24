from __future__ import annotations

import torch

from holosoma.agents.w_bc.w_bc_agent import weighted_bc_nll
from holosoma.config_values.experiment import DEFAULTS


def test_weighted_bc_nll_applies_one_weight_per_transition():
    log_prob_data = torch.tensor([-2.0, -4.0])
    weights = torch.tensor([0.5, 1.5])

    loss = weighted_bc_nll(log_prob_data, weights)

    expected = torch.tensor((0.5 * 2.0 + 1.5 * 4.0) / 2.0)
    torch.testing.assert_close(loss, expected)


def test_weighted_bc_nll_reduces_to_nll_for_uniform_weights():
    generator = torch.Generator().manual_seed(7)
    log_prob_data = torch.randn(16, generator=generator)

    weighted_loss = weighted_bc_nll(log_prob_data, torch.ones(log_prob_data.shape[0]))
    ordinary_nll = -log_prob_data.mean()

    torch.testing.assert_close(weighted_loss, ordinary_nll)


def test_wbt_bc_and_wbc_are_a_paired_nll_comparison():
    bc_experiment = DEFAULTS["g1_29dof_wbt_bc"]
    wbc_experiment = DEFAULTS["g1_29dof_wbt_w_bc"]

    assert bc_experiment.algo.config.offline_dataset_path == wbc_experiment.algo.config.offline_dataset_path
    assert bc_experiment.algo.config.actor_learning_rate == wbc_experiment.algo.config.actor_learning_rate
    assert bc_experiment.algo.config.batch_size == wbc_experiment.algo.config.batch_size
    assert bc_experiment.algo.config.num_updates == wbc_experiment.algo.config.num_updates
    assert wbc_experiment.algo._target_ == "holosoma.agents.w_bc.w_bc_agent.WBCAgent"
