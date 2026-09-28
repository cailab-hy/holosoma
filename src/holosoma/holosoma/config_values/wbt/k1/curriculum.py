"""Whole-body tracking curriculum presets for the AI Sapiens K1 Rev.1 (23-DoF) robot."""

from holosoma.config_types.curriculum import CurriculumManagerCfg, CurriculumTermCfg

k1_23dof_wbt_curriculum = CurriculumManagerCfg(
    params={"num_compute_average_epl": 1000},
    setup_terms={
        "average_episode_tracker": CurriculumTermCfg(
            func="holosoma.managers.curriculum.terms.locomotion:AverageEpisodeLengthTracker",
            params={},
        ),
    },
    reset_terms={},
    step_terms={},
)

__all__ = ["k1_23dof_wbt_curriculum"]
