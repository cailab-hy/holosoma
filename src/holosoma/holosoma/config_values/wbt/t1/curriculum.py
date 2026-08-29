"""Whole-body tracking curriculum presets for the T1 29-DoF robot."""

from holosoma.config_types.curriculum import CurriculumManagerCfg, CurriculumTermCfg

t1_29dof_wbt_curriculum = CurriculumManagerCfg(
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

__all__ = ["t1_29dof_wbt_curriculum"]
