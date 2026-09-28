"""Action presets for the AI Sapiens K1 Rev.1 (23-DoF) robot.

The per-joint action scale comes from ``RobotControlConfig`` (``0.25 * effort_limit / kp``
via ``action_scales_by_effort_limit_over_p_gain``), which matches cyclo_lab's
``K1_REV1_INERTIA_TUNED_ACTION_SCALE`` so policies transfer between the two stacks.
"""

from holosoma.config_types.action import ActionManagerCfg, ActionTermCfg

k1_23dof_joint_pos = ActionManagerCfg(
    terms={
        "joint_control": ActionTermCfg(
            func="holosoma.managers.action.terms.joint_control:JointPositionActionTerm",
            params={},
            scale=1.0,
            clip=None,
        ),
    }
)

__all__ = ["k1_23dof_joint_pos"]
