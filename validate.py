"""Main execution script with OmegaConf support"""

from collections.abc import Mapping
from dataclasses import replace
from typing import Any, cast

from omegaconf import OmegaConf

from universal_validator.pipeline.universal_validator import UniversalValidator
from universal_validator.pipeline.utils import ValidatorConfig
from universal_validator.utils import run_with_config


def main(cfg: ValidatorConfig):
    validator = UniversalValidator(cfg)

    all_tasks = validator.get_available_tasks(verbose=True)
    if cfg.list_configs:
        return

    if cfg.task_names is None:
        tasks = all_tasks
    else:
        assert set(cfg.task_names) <= set(all_tasks)
        tasks = cfg.task_names

    reports = []
    for task in tasks:
        report = validator.run_pipeline(task_name=task)
        report["task_name"] = task
        print(report)
        reports += [report]
    return reports


def run_with_paths(
    downstream_config: Mapping[str, Any],
    train_path: str,
    test_path: str,
):
    cfg = cast(
        ValidatorConfig,
        OmegaConf.to_object(
            OmegaConf.merge(
                OmegaConf.structured(ValidatorConfig),
                OmegaConf.create(dict(downstream_config)),
            )
        ),
    )

    cfg = replace(
        cfg,
        data_conf=replace(cfg.data_conf, train_path=train_path, test_path=test_path),
    )
    return main(cfg)


if __name__ == "__main__":
    try:
        result = run_with_config(main, "universal_validator")
        print(result)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
