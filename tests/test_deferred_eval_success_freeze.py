from __future__ import annotations

from types import SimpleNamespace

import torch

from holosoma.envs.base_task.base_task import BaseTask
from holosoma.envs.wbt.wbt_manager import WholeBodyTrackingManager


class _FakeSimulator:
    def __init__(self) -> None:
        self.robot_root_states = torch.arange(39, dtype=torch.float32).reshape(3, 13)
        self.dof_pos = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        self.dof_vel = torch.ones(3, 4)
        self.root_write_env_ids: list[torch.Tensor] = []
        self.dof_write_env_ids: list[torch.Tensor] = []

    def set_actor_root_state_tensor_robots(self, env_ids: torch.Tensor) -> None:
        self.root_write_env_ids.append(env_ids.clone())

    def set_dof_state_tensor_robots(self, env_ids: torch.Tensor) -> None:
        self.dof_write_env_ids.append(env_ids.clone())


def test_wbt_restores_only_successful_deferred_eval_envs() -> None:
    manager = WholeBodyTrackingManager.__new__(WholeBodyTrackingManager)
    manager._defer_resets = True
    manager._deferred_eval_success_mask = torch.tensor([False, True, False])
    manager.simulator = _FakeSimulator()
    manager._eval_success_root_states = torch.zeros_like(manager.simulator.robot_root_states)
    manager._eval_success_dof_pos = torch.zeros_like(manager.simulator.dof_pos)

    success_env_ids = torch.tensor([1])
    captured_root_pose = manager.simulator.robot_root_states[1, :7].clone()
    captured_dof_pos = manager.simulator.dof_pos[1].clone()
    manager._capture_deferred_eval_success_states(success_env_ids)

    manager.simulator.robot_root_states += 1000.0
    manager.simulator.dof_pos += 1000.0
    manager.simulator.dof_vel += 1000.0
    untouched_failure_root = manager.simulator.robot_root_states[2].clone()
    untouched_failure_dof_pos = manager.simulator.dof_pos[2].clone()
    untouched_failure_dof_vel = manager.simulator.dof_vel[2].clone()

    manager._restore_deferred_eval_success_states()

    torch.testing.assert_close(manager.simulator.robot_root_states[1, :7], captured_root_pose)
    torch.testing.assert_close(manager.simulator.robot_root_states[1, 7:13], torch.zeros(6))
    torch.testing.assert_close(manager.simulator.dof_pos[1], captured_dof_pos)
    torch.testing.assert_close(manager.simulator.dof_vel[1], torch.zeros(4))
    torch.testing.assert_close(manager.simulator.robot_root_states[2], untouched_failure_root)
    torch.testing.assert_close(manager.simulator.dof_pos[2], untouched_failure_dof_pos)
    torch.testing.assert_close(manager.simulator.dof_vel[2], untouched_failure_dof_vel)
    torch.testing.assert_close(manager.simulator.root_write_env_ids[0], success_env_ids)
    torch.testing.assert_close(manager.simulator.dof_write_env_ids[0], success_env_ids)


def test_base_physics_step_restores_before_render_and_after_each_substep() -> None:
    events: list[str] = []
    task = BaseTask.__new__(BaseTask)
    task.simulator = SimpleNamespace(
        simulator_config=SimpleNamespace(sim=SimpleNamespace(control_decimation=2)),
        simulate_at_each_physics_step=lambda: events.append("simulate"),
    )
    task.render = lambda: events.append("render")
    task._apply_force_in_physics_step = lambda: events.append("force")
    task._restore_deferred_eval_success_states = lambda: events.append("restore")

    BaseTask._physics_step(task)

    assert events == [
        "restore",
        "render",
        "force",
        "simulate",
        "restore",
        "force",
        "simulate",
        "restore",
    ]
