"""Whole Body Tracking termination presets for the K1 robot."""

from holosoma.config_types.termination import TerminationManagerCfg, TerminationTermCfg

k1_23dof_wbt_termination = TerminationManagerCfg(
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
                # robot tracking
                "bad_ref_pos_threshold": 0.5,
                "bad_ref_ori_threshold": 0.8,
                "bad_motion_body_pos_threshold": 0.25,
                # NOTE: body_names_to_track is shared with command_manager
                "body_names_to_track": [
                    "pelvis",
                    "left_hip_roll_link",
                    "left_knee_link",
                    "left_ankle_roll_link",
                    "right_hip_roll_link",
                    "right_knee_link",
                    "right_ankle_roll_link",
                    "torso_link",
                    "left_shoulder_roll_link",
                    "left_elbow_link",
                    "left_wrist_roll_rubber_hand",
                    "right_shoulder_roll_link",
                    "right_elbow_link",
                    "right_wrist_roll_rubber_hand",
                ],
                "bad_motion_body_pos_body_names": [
                    "left_ankle_roll_link",
                    "right_ankle_roll_link",
                    "left_wrist_roll_rubber_hand",
                    "right_wrist_roll_rubber_hand",
                ],
                # object tracking
                # only triggered when has_object=True
                "bad_object_pos_threshold": 0.25,
                "bad_object_ori_threshold": 0.8,
            },
        ),
    }
)

k1_23dof_wbt_termination_offline_collect = TerminationManagerCfg(
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
                # Collection continues after crossing strict evaluation limits
                # (0.5, 0.8, 0.25) and terminates only at these relaxed limits.
                "bad_ref_pos_threshold": 1.0,
                "bad_ref_ori_threshold": 1.6,
                "bad_motion_body_pos_threshold": 0.5,
                "body_names_to_track": [
                    "pelvis",
                    "left_hip_roll_link",
                    "left_knee_link",
                    "left_ankle_roll_link",
                    "right_hip_roll_link",
                    "right_knee_link",
                    "right_ankle_roll_link",
                    "torso_link",
                    "left_shoulder_roll_link",
                    "left_elbow_link",
                    "left_wrist_roll_rubber_hand",
                    "right_shoulder_roll_link",
                    "right_elbow_link",
                    "right_wrist_roll_rubber_hand",
                ],
                "bad_motion_body_pos_body_names": [
                    "left_ankle_roll_link",
                    "right_ankle_roll_link",
                    "left_wrist_roll_rubber_hand",
                    "right_wrist_roll_rubber_hand",
                ],
                "bad_object_pos_threshold": 0.5,
                "bad_object_ori_threshold": 1.6,
            },
        ),
    }
)

__all__ = [
    "k1_23dof_wbt_termination",
    "k1_23dof_wbt_termination_offline_collect",
]
