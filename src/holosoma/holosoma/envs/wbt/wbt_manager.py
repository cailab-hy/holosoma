from __future__ import annotations

import time

import torch

from holosoma.envs.base_task.base_task import BaseTask

# from holosoma.envs.legged_base_task.legged_robot_base import LeggedRobotBase
from holosoma.utils.simulator_config import SimulatorType
from holosoma.utils.rotations import quat_error_magnitude


class WholeBodyTrackingManager(BaseTask):
    def __init__(self, tyro_config, *, device):
        super().__init__(tyro_config, device=device)

    def _init_buffers(self):
        """Initialize torch tensors which will contain simulation states and processed quantities"""
        super()._init_buffers()

        # -------------------------------- terms same with locomotion_manager.py [start]--------------------------------
        self.base_quat = self.simulator.base_quat
        self.need_to_refresh_envs = torch.ones(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self._eval_terminal_motion_phase = torch.full(
            (self.num_envs,),
            float("nan"),
            dtype=torch.float32,
            device=self.device,
        )
        self._configure_default_dof_pos()
        self._init_domain_rand_buffers()

    def _configure_default_dof_pos(self):
        self.default_dof_pos_base = torch.zeros(
            self.num_dof, dtype=torch.float, device=self.device, requires_grad=False
        )
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            if name not in self.robot_config.init_state.default_joint_angles:
                raise ValueError(f"Missing default joint angle for DOF '{name}' in robot configuration.")
            angle = self.robot_config.init_state.default_joint_angles[name]
            self.default_dof_pos_base[i] = angle

        self.default_dof_pos_base = self.default_dof_pos_base.unsqueeze(0)  # (1, num_dof)
        self.default_dof_pos = self.default_dof_pos_base.repeat(self.num_envs, 1).clone()  # (num_envs, num_dof)

    def _pre_compute_observations_callback(self):
        self.base_quat[:] = self.simulator.base_quat[:]

    def _reset_buffers_callback(self, env_ids, target_buf=None):
        self.need_to_refresh_envs[env_ids] = True
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # pending_episode_update_mask is only used in curriculum_term::AverageEpisodeLengthTracker.
        self._pending_episode_update_mask[env_ids] = True

    def _get_envs_to_refresh(self):
        return self.need_to_refresh_envs.nonzero(as_tuple=False).flatten()

    def _refresh_envs_after_reset(self, env_ids):
        self.simulator.set_actor_root_state_tensor(env_ids, self.simulator.all_root_states)
        self.simulator.set_dof_state_tensor(env_ids, self.simulator.dof_state)
        self.simulator.clear_contact_forces_history(env_ids)
        self.need_to_refresh_envs[env_ids] = False
        self.simulator.refresh_sim_tensors()
        self._pre_compute_observations_callback()

    def _get_eval_status_marker_positions(self) -> torch.Tensor:
        body_idx = None
        for candidate_name in ("head_link", "head", "logo_link", "torso_link"):
            if candidate_name in self.body_names:
                body_idx = self.body_names.index(candidate_name)
                break

        if body_idx is None:
            positions = self.simulator.robot_root_states[:, :3].clone()
            positions[:, 2] += 1.45
        else:
            positions = self.simulator._rigid_body_pos[:, body_idx, :].clone()
            positions[:, 2] += 0.35
        return positions

    def _ensure_eval_status_markers(self) -> bool:
        if self.simulator.get_simulator_type() != SimulatorType.ISAACSIM:
            return False
        if hasattr(self, "_eval_status_markers"):
            return True

        try:
            from isaaclab.markers import VisualizationMarkers
            from isaaclab.markers.config import RAY_CASTER_MARKER_CFG
        except Exception:
            return False

        success_cfg = RAY_CASTER_MARKER_CFG.replace(prim_path="/Visuals/EvalStatus/success")
        success_cfg.markers["hit"].radius = 0.12
        success_cfg.markers["hit"].visual_material.diffuse_color = (0.0, 1.0, 0.0)

        failure_cfg = RAY_CASTER_MARKER_CFG.replace(prim_path="/Visuals/EvalStatus/failure")
        failure_cfg.markers["hit"].radius = 0.12
        failure_cfg.markers["hit"].visual_material.diffuse_color = (1.0, 0.0, 0.0)

        self._eval_status_markers = {
            "success": VisualizationMarkers(success_cfg),
            "failure": VisualizationMarkers(failure_cfg),
        }
        return True

    def _visualize_eval_status_marker(self, marker_name: str, positions: torch.Tensor) -> None:
        marker = self._eval_status_markers[marker_name]
        if positions.numel() == 0:
            if hasattr(marker, "set_visibility"):
                marker.set_visibility(False)
            else:
                hidden_position = torch.tensor([[0.0, 0.0, -1000.0]], device=self.device)
                marker.visualize(hidden_position)
            return

        if hasattr(marker, "set_visibility"):
            marker.set_visibility(True)
        marker.visualize(positions)

    def _reset_eval_status_visualization(self):
        if not self._ensure_eval_status_markers():
            return
        self._visualize_eval_status_marker("success", torch.empty(0, 3, device=self.device))
        self._visualize_eval_status_marker("failure", torch.empty(0, 3, device=self.device))

    def _update_eval_status_visualization(self):
        if not self._ensure_eval_status_markers():
            return

        positions = self._get_eval_status_marker_positions()
        success_positions = positions[self._deferred_eval_success_mask]
        failure_positions = positions[self._deferred_eval_failure_mask]
        self._visualize_eval_status_marker("success", success_positions)
        self._visualize_eval_status_marker("failure", failure_positions)

    def _get_average_episode_tracker(self):
        tracker = self.curriculum_manager.get_term("average_episode_tracker")
        if tracker is None:
            raise RuntimeError("AverageEpisodeLengthTracker is not registered with the curriculum manager.")
        return tracker

    # -------------------------------- terms same with locomotion_manager.py [end]--------------------------------

    def _update_log_dict(self):
        # _update_log_dict happens before reset_envs_idx
        # -------------------------------- terms same with locomotion_manager.py [start]--------------------------------
        avg = self._get_average_episode_tracker().get_average()
        self.log_dict["average_episode_length"] = avg.detach().cpu()
        # -------------------------------- terms same with locomotion_manager.py [end]--------------------------------
        # Add tracking metrics to log_dict
        motion_command = self.command_manager.get_state("motion_command")
        motion_command.update_metrics()
        self.log_dict.update(motion_command.metrics)
        motion_denominator = max(int(motion_command.motion.time_step_total) - 1, 1)
        motion_phase = motion_command.time_steps.to(torch.float32) / float(motion_denominator)
        self.extras["motion_phase"] = motion_phase.clone()
        terminal_mask = self.reset_buf.to(torch.bool)
        if self._defer_resets:
            terminal_mask &= ~self._deferred_reset_mask
        self._eval_terminal_motion_phase[terminal_mask] = motion_phase[terminal_mask]

    def _collect_tracking_metrics(self) -> dict[str, torch.Tensor]:
        motion_command = self.command_manager.get_state("motion_command")
        body_pos_error = torch.norm(
            motion_command.body_pos_relative_w - motion_command.robot_body_pos_w,
            dim=-1,
        )

        err_object_pos = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        err_object_ori = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        if motion_command.motion.has_object:
            err_object_pos = torch.norm(
                motion_command.object_pos_w - motion_command.simulator_object_pos_w,
                dim=-1,
            ).to(torch.float32)
            err_object_ori = quat_error_magnitude(
                motion_command.object_quat_w,
                motion_command.simulator_object_quat_w,
            ).to(torch.float32)

        return {
            "err_root_pos": torch.norm(
                motion_command.ref_pos_w - motion_command.robot_ref_pos_w,
                dim=-1,
            ).to(torch.float32),
            "err_root_ori": quat_error_magnitude(
                motion_command.ref_quat_w,
                motion_command.robot_ref_quat_w,
            ).to(torch.float32),
            "err_body_pos_max": body_pos_error.max(dim=-1).values.to(torch.float32),
            "err_body_pos_mean": body_pos_error.mean(dim=-1).to(torch.float32),
            "err_object_pos": err_object_pos,
            "err_object_ori": err_object_ori,
        }

    def reset_all(self):
        # If reset_all is called several times, clear buffer in motion_command
        motion_command = self.command_manager.get_state("motion_command")
        motion_command.init_buffers()
        self._eval_terminal_motion_phase.fill_(float("nan"))
        return super().reset_all()

    def set_defer_resets(self, enabled: bool) -> None:
        super().set_defer_resets(enabled)
        self._eval_terminal_motion_phase.fill_(float("nan"))

    def get_eval_terminal_motion_phases(self) -> torch.Tensor:
        return self._eval_terminal_motion_phase.clone()

    def _reset_robot_states_callback(self, env_ids, target_states=None):
        # TODO(jchen): Now,reset robot/object states is implemented in command/terms/wbt.MotionCommand.reset
        # discuss whether to move to here in the future.
        pass

    ########################################################### Push robots #########################################
    # TODO: This should be moved to the randomization manager.
    def _init_domain_rand_buffers(self):
        ######################################### DR related tensors #########################################
        # Action delay buffers are now initialized by randomization manager's setup_action_delay_buffers term

        self.push_robot_vel_buf = torch.zeros(
            self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.record_push_robot_vel_buf = torch.zeros(
            self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False
        )
        self._randomize_push_robots = False
        self._max_push_vel = torch.zeros(6, dtype=torch.float32, device=self.device)

    def _push_robots(self, env_ids):
        """Random pushes the robots. Emulates an impulse by setting a randomized base velocity."""
        if len(env_ids) == 0:
            return
        self.need_to_refresh_envs[env_ids] = True
        max_vel_tensor = self._max_push_vel
        if self.randomization_manager is not None:
            state = self.randomization_manager.get_state("push_randomizer_state")
            if state is not None:
                max_vel_tensor = state.max_push_vel.clone().to(self.device)

        if not isinstance(max_vel_tensor, torch.Tensor) or max_vel_tensor.numel() != 6:
            raise ValueError("WholeBodyTracking push velocity vector must have exactly 6 components.")

        rand = torch.rand(len(env_ids), 6, device=self.device) * 2 - 1
        self.push_robot_vel_buf[env_ids] = rand * max_vel_tensor.unsqueeze(0)
        self.record_push_robot_vel_buf[env_ids] = self.push_robot_vel_buf[env_ids].clone()
        self.simulator.robot_root_states[env_ids, 7:13] = self.push_robot_vel_buf[env_ids]
        # Push impulses only take effect in the simulator once we write the mutated root state tensor back.
        self.simulator.set_actor_root_state_tensor_robots(env_ids, self.simulator.robot_root_states)
        self._max_push_vel = max_vel_tensor.clone()

    #########################################################################################################
    ## Debug visualization
    #########################################################################################################

    def _draw_debug_vis_isaacsim(self):
        motion_command = self.command_manager.get_state("motion_command")
        # torso link
        real_robot_pos_xyz = motion_command.robot_ref_pos_w.clone()
        real_robot_quat_xyzw = motion_command.robot_ref_quat_w.clone()
        real_robot_quat_wxyz = real_robot_quat_xyzw[:, [3, 0, 1, 2]]
        motion_command.visualization_markers["real_robot"].visualize(real_robot_pos_xyz, real_robot_quat_wxyz)

        motion_robot_pos_xyz = motion_command.ref_pos_w.clone()
        motion_robot_quat_xyzw = motion_command.ref_quat_w.clone()
        motion_robot_quat_wxyz = motion_robot_quat_xyzw[:, [3, 0, 1, 2]]
        motion_command.visualization_markers["motion_robot"].visualize(motion_robot_pos_xyz, motion_robot_quat_wxyz)

        for body_idx, body_names in enumerate(motion_command.motion_cfg.body_names_to_track):
            motion_robot_body_pos_xyz = motion_command.body_pos_w[0, body_idx].clone()
            motion_command.visualization_markers[f"motion_{body_names}"].visualize(
                motion_robot_body_pos_xyz.unsqueeze(0)
            )

        # object
        if motion_command.motion.has_object:
            real_object_pos_xyz = motion_command.simulator_object_pos_w.clone()
            real_object_quat_xyzw = motion_command.simulator_object_quat_w.clone()
            real_object_quat_wxyz = real_object_quat_xyzw[:, [3, 0, 1, 2]]
            motion_command.visualization_markers["real_object"].visualize(real_object_pos_xyz, real_object_quat_wxyz)

            motion_object_pos_xyz = motion_command.object_pos_w.clone()
            motion_object_quat_xyzw = motion_command.object_quat_w.clone()
            motion_object_quat_wxyz = motion_object_quat_xyzw[:, [3, 0, 1, 2]]
            motion_command.visualization_markers["motion_object"].visualize(
                motion_object_pos_xyz, motion_object_quat_wxyz
            )

    def _draw_debug_vis_isaacgym(self):
        self.simulator.clear_lines()
        n_bodies = len(self.motion_command.motion_cfg.body_names_to_track)
        for env_id in range(self.num_envs):
            for body_idx in range(n_bodies):
                color = (0.0, 1.0, 0.0)
                self.simulator.draw_sphere(
                    self.motion_command.body_pos_relative_w[env_id, body_idx], 0.03, color, env_id, body_idx
                )

                color = (0.0, 0.0, 1.0)
                self.simulator.draw_sphere(
                    self.motion_command.robot_body_pos_w[env_id, body_idx], 0.03, color, env_id, n_bodies + body_idx
                )

            color = (0.0, 1.0, 0.0)
            self.simulator.draw_sphere(self.motion_command.ref_pos_w[env_id], 0.05, color, env_id, n_bodies * 2 + 0)
            color = (0.0, 0.0, 1.0)
            self.simulator.draw_sphere(
                self.motion_command.robot_ref_pos_w[env_id], 0.05, color, env_id, n_bodies * 2 + 1
            )

    def _draw_debug_vis(self):
        if self.simulator.get_simulator_type() == SimulatorType.ISAACSIM:
            self._draw_debug_vis_isaacsim()
        elif self.simulator.get_simulator_type() == SimulatorType.ISAACGYM:
            self._draw_debug_vis_isaacgym()

    def step_visualize_motion(self, actions):
        motion_command = self.command_manager.get_state("motion_command")
        dt = 1.0 / float(motion_command.motion.fps)
        motion_command.step()
        print("time_steps: ", motion_command.time_steps[0].item())
        self._draw_debug_vis()

        # set root_states_from_motion_command
        root_pos = motion_command.root_pos_w.clone()
        root_ori = motion_command.root_quat_w.clone()  # wxyz
        root_lin_vel = motion_command.body_lin_vel_w[:, 0].clone()
        root_ang_vel = motion_command.body_ang_vel_w[:, 0].clone()

        joint_pos = motion_command.joint_pos.clone()
        joint_vel = motion_command.joint_vel.clone()

        env_ids = torch.arange(self.num_envs, device=self.device)
        self.simulator.dof_pos[env_ids] = joint_pos
        self.simulator.dof_vel[env_ids] = joint_vel

        self.simulator.robot_root_states[env_ids, :3] = root_pos
        self.simulator.robot_root_states[env_ids, 3:7] = root_ori
        self.simulator.robot_root_states[env_ids, 7:10] = root_lin_vel
        self.simulator.robot_root_states[env_ids, 10:13] = root_ang_vel

        self.simulator.set_actor_root_state_tensor(env_ids, self.simulator.all_root_states)
        self.simulator.set_dof_state_tensor(env_ids, self.simulator.dof_state)

        if motion_command.motion.has_object:
            # set object root_states from motion command
            object_pos = motion_command.object_pos_w.clone()
            object_ori = motion_command.object_quat_w.clone()
            object_lin_vel = motion_command.object_lin_vel_w.clone()

            object_states = torch.zeros(len(env_ids), 13, device=self.device)
            object_states[:, :3] = object_pos[:]
            object_states[:, 3:7] = object_ori[:]
            object_states[:, 7:10] = object_lin_vel[:]
            object_states[:, 10:13] = torch.zeros_like(object_lin_vel[:])
            self.simulator.set_actor_states(["object"], env_ids, object_states)

        self.simulator.scene.write_data_to_sim()
        self.simulator.sim.forward()
        self.simulator.sim.render()
        self.simulator.refresh_sim_tensors()

        time.sleep(dt)

        return motion_command.time_steps[0].item() >= motion_command.motion.time_step_total - 2
