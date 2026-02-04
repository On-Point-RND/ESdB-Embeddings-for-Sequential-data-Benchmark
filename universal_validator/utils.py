import sys
import tempfile

import pyrallis
from omegaconf import OmegaConf

from .pipeline.utils import ValidatorConfig


def pop_arg(args, key):
    i = 0
    new_args = []
    value = None
    while i < len(args):
        if args[i] == key:
            value = args[i + 1]
            if key in ["--config_factory"]:
                assert value[0] == "[" and value[-1] == "]", "Wrong factory format"
                value = value[1:-1].split(",")
            i += 2
        else:
            new_args += [args[i]]
            i += 1
    return new_args, value


def run_config_factory(config_path, config_factory):
    if config_factory is not None:
        config_paths = [f"configs/{name}.yaml" for name in config_factory]
    else:
        config_paths = []
    config_paths += [config_path]
    configs = [OmegaConf.load(path) for path in config_paths]
    merged_config = OmegaConf.merge(*configs)
    merged_config["config_factory"] = None
    return merged_config


def run_with_config(func, default_conf="config.yaml"):
    args = sys.argv[1:]

    # 1. config generation
    args, config_factory = pop_arg(args, "--config_factory")
    # 2. overwrite with certain fields
    args, config_path = pop_arg(args, "--config_path")
    # 3. in case we need to overwrite all above
    args, overwrite_factory = pop_arg(args, "--overwrite_factory")
    path = config_path or default_conf

    config_factory = config_factory or OmegaConf.load(path).get("config_factory")
    merged_config = run_config_factory(path, config_factory, overwrite_factory)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as tmpfile:
        OmegaConf.save(config=merged_config, f=tmpfile.name)
        temp_config_path = tmpfile.name
        print(f"Saved temporary config: {temp_config_path}")
        cfg = pyrallis.parse(ValidatorConfig, temp_config_path, args)
        res = func(cfg)
    return res
