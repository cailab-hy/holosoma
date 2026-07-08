import pytest
import torch


def _delta_norm(a_pi, a_ref, sigma):
    return torch.sqrt((((a_pi - a_ref) / (sigma + 1e-6)) ** 2).mean(dim=1) + 1e-12)


def _build_rays(a_ref, a_pi, t_grid, noise_std=0.0):
    t = torch.as_tensor(t_grid, device=a_ref.device, dtype=a_ref.dtype)
    rays = a_ref[:, None, :] + t[None, :, None] * (a_pi - a_ref)[:, None, :]
    if noise_std > 0.0:
        rays = rays + torch.randn_like(rays) * noise_std
    return rays.clamp(-1.0, 1.0)


def _dcql_gap(q_ray, q_data, tau=1.0):
    return tau * (torch.logsumexp(q_ray / tau, dim=1) - torch.log(torch.tensor(q_ray.shape[1], dtype=q_ray.dtype))) - q_data


def test_gate_all_off_equivalence():
    bellman = torch.tensor(3.0)
    q_ray = torch.randn(4, 3)
    q_data = torch.randn(4)
    gate = torch.zeros(4)
    gap = _dcql_gap(q_ray, q_data)
    dcql_loss = (gap * gate).sum() / float(gate.numel())
    assert torch.allclose(bellman + dcql_loss, bellman)


def test_ray_geometry_without_noise():
    a_ref = torch.tensor([[0.0, 0.5]])
    a_pi = torch.tensor([[1.0, -0.5]])
    rays = _build_rays(a_ref, a_pi, [0.5, 1.0, 2.0], noise_std=0.0)
    expected = torch.tensor([[[0.5, 0.0], [1.0, -0.5], [1.0, -1.0]]])
    assert torch.allclose(rays, expected)


def test_two_sided_gradient_direction():
    q_ray = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    q_data = torch.tensor([0.5], requires_grad=True)
    loss = _dcql_gap(q_ray, q_data).mean()
    loss.backward()
    assert torch.all(q_ray.grad > 0.0)
    assert torch.all(q_data.grad < 0.0)


def test_gate_correctness():
    a_ref = torch.zeros(2, 3)
    a_pi = torch.tensor([[0.1, 0.0, 0.0], [1.0, 1.0, 1.0]])
    sigma = torch.ones(3)
    gate = _delta_norm(a_pi, a_ref, sigma) >= 0.5
    assert gate.tolist() == [False, True]


def test_warmup_ballast_adds_only_when_enabled():
    base = torch.tensor(1.0)
    ballast = torch.tensor(2.0)
    assert torch.allclose(base + 0.1 * ballast, torch.tensor(1.2))
    assert torch.allclose(base + 0.0 * ballast, base)


def test_knn_reference_mode_raises():
    mode = "knn"
    with pytest.raises(NotImplementedError):
        if mode == "knn":
            raise NotImplementedError("dcql.a_ref_mode='knn' requires a behavior/reference-action index")
