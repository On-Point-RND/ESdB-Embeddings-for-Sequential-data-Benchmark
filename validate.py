"""Main execution script with OmegaConf support"""

import logging
from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime
from typing import Any, cast

import pandas as pd
from omegaconf import OmegaConf

from universal_validator.pipeline.universal_validator import UniversalValidator
from universal_validator.pipeline.utils import ValidatorConfig
from universal_validator.utils import run_with_config


def _suppress_noisy_py4j_logs() -> None:
    # Keep py4j shutdown messages from flooding stdout during training.
    logging.getLogger("py4j.clientserver").setLevel(logging.ERROR)
    logging.getLogger("py4j.java_gateway").setLevel(logging.ERROR)
    logging.getLogger("py4j").setLevel(logging.ERROR)


def main(cfg: ValidatorConfig):
    _suppress_noisy_py4j_logs()
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
        #print(report)
        reports += [report]
    return reports


def run_with_paths(
    downstream_config: Mapping[str, Any],
    train_path: str,
    test_path: str,
):
    raw_config = dict(downstream_config)
    data_conf_overrides = dict(raw_config.pop("data_conf", {}))

    cfg = cast(
        ValidatorConfig,
        OmegaConf.to_object(
            OmegaConf.merge(
                OmegaConf.structured(ValidatorConfig),
                OmegaConf.create(raw_config),
            )
        ),
    )

    cfg = replace(
        cfg,
        data_conf=replace(
            cfg.data_conf,
            **data_conf_overrides,
            train_path=train_path,
            test_path=test_path,
        ),
    )
    return main(cfg)


if __name__ == "__main__":
    try:
        result = run_with_config(main, "universal_validator")
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        pd.DataFrame(result).to_json(f"validator_output__{timestamp}.json",
                                     orient='records', indent=4, date_format='iso')
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
