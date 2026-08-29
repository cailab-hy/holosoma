"""Whole-body tracking termination presets for the T1 29-DoF robot."""

from holosoma.config_types.termination import TerminationManagerCfg, TerminationTermCfg
from holosoma.config_values.wbt.t1.command import T1_WBT_BODY_NAMES, T1_WBT_END_EFFECTOR_NAMES

t1_29dof_wbt_termination = TerminationManagerCfg(
    terms={
        "timeout": TerminationTermCfg(
            func="holosoma.managers.termination.terms.common:timeout_exceeded",
            is_timeout=True,
        ),
        "motion_ends": TerminationTermCfg(
            func="holosoma.managers.termination.terms.wbt:motion_ends",
        ),
        "bad_tracking": TerminationTermCfg(
            func="holosoma.managers.termination.terms.wbt:BadTracking",
            params={
                "bad_ref_pos_threshold": 0.5,
                "bad_ref_ori_threshold": 0.8,
                "bad_motion_body_pos_threshold": 0.25,
                "body_names_to_track": T1_WBT_BODY_NAMES,
                "bad_motion_body_pos_body_names": T1_WBT_END_EFFECTOR_NAMES,
                "bad_object_pos_threshold": 0.25,
                "bad_object_ori_threshold": 0.8,
            },
        ),
    }
)

__all__ = ["t1_29dof_wbt_termination"]
