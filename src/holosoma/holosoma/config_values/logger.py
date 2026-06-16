from holosoma.config_types.logger import DisabledLoggerConfig, WandbLoggerConfig
from holosoma.config_types.video import FixedCameraConfig, VideoConfig

disabled = DisabledLoggerConfig()

wandb = WandbLoggerConfig(mode="online")

wandb_wide = WandbLoggerConfig(
    mode="online",
    video=VideoConfig(
        camera=FixedCameraConfig(
            position=[7.0, -7.0, 5.5],
            target=[3.0, 3.0, 0.8],
        ),
        vertical_fov=60.0,
        show_command_overlay=False,
    ),
)

wandb_offline = WandbLoggerConfig(mode="offline")

DEFAULTS = {
    "disabled": disabled,
    "wandb": wandb,
    "wandb_wide": wandb_wide,
    "wandb_offline": wandb_offline,
}
