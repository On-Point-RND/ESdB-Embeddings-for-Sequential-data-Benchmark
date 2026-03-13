from omegaconf import OmegaConf

OmegaConf.register_new_resolver(
    "inherit",
    lambda base, overrides: OmegaConf.merge(base, overrides)
)

OmegaConf.register_new_resolver("min", lambda *args: min(args))
OmegaConf.register_new_resolver("max", lambda *args: min(args))