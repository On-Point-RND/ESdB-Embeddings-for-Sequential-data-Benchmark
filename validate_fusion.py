"""Main execution script with OmegaConf support"""

import logging
from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime
from typing import Any, cast
from copy import deepcopy
from pathlib import Path
import csv
import pandas as pd
import re
from omegaconf import OmegaConf

from universal_validator.pipeline.universal_validator import UniversalValidator
from universal_validator.pipeline.utils import ValidatorConfig
from universal_validator.utils import ensure_validator_logging, run_with_config


def _suppress_noisy_py4j_logs() -> None:
    logging.getLogger("py4j.clientserver").setLevel(logging.ERROR)
    logging.getLogger("py4j.java_gateway").setLevel(logging.ERROR)
    logging.getLogger("py4j").setLevel(logging.ERROR)


def main(cfg: ValidatorConfig):
    ensure_validator_logging()
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
    embedding_report = validator.run_embedding_metrics()
    if embedding_report:
        reports.append(embedding_report)

    for task in tasks:
        report = validator.run_pipeline(task_name=task)
        report["task_name"] = task
        reports.append(report)
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


def extract_downstream_metrics(reports) -> dict[str, float]:
    metrics = {}
    for report in reports:
        if "metrics" in report:
            metrics.update(report["metrics"])
            continue
        if not report:
            continue

        _, metric_names = report["task_name"].rsplit("__", 1)
        best_model = report.get("best_model")
        m = metric_names.split("+")[0]
        if m == "mse":
            m = "neg_mean_squared_error"
        all_results = report["all_results"]
        metrics[report["task_name"]] = float(all_results[best_model][m])
    return metrics


def set_validator_seed(downstream_config: dict, seed: int) -> None:
    for model_config in downstream_config.get("models", {}).values():
        shared_params = model_config.get("shared_params", {})
        if "random_state" in shared_params:
            shared_params["random_state"] = seed


def run_downstream_with_seed(
    downstream_config: dict,
    train_path: str,
    test_path: str,
    seed: int,
) -> tuple[list[dict], dict[str, float]]:
    seeded_config = deepcopy(downstream_config)
    seeded_config.pop("validator_seeds", None)
    if "embedding_metrics" in seeded_config:
        seeded_config["embedding_metrics"]["enabled"] = False
    set_validator_seed(seeded_config, seed)
    reports = run_with_paths(
        downstream_config=seeded_config,
        train_path=train_path,
        test_path=test_path,
    )
    metrics = extract_downstream_metrics(reports)
    return reports, metrics


def save_seed_metrics(path: Path, metrics: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in metrics.items():
            writer.writerow([key, value])


def get_embedding_postfix(path: str) -> str:
    """Extract the folder name starting with 'embeddings_' from the given path."""
    parts = Path(path).parts
    for part in parts:
        if part.startswith("embeddings_"):
            return part
    return "embeddings_unknown"


if __name__ == "__main__":
    try:
        root_dir = './log/full/age/NTP_GRU/tests/forecast/seed_0/'
        all_postfixs = all_postfixs = sorted(set(
            re.search(r'embeddings_(.+)', d.name).group(1)
            for d in Path(root_dir).iterdir()
            if d.is_dir() and d.name.startswith('embeddings_')
        ))
        print(all_postfixs)
        #input('...')
        for postfix in all_postfixs:
            config_path = f'./universal_validator/configs/validator/logreg_3seed_embedding_metrics.yaml'
            train_path = f'{root_dir}/embeddings_{postfix}/train_postproc/'
            test_path = f'{root_dir}/embeddings_{postfix}/test_postproc/'

            config_dict = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
            print(config_dict)

            seeds = config_dict.get("validator_seeds")
            if seeds is None:
                seeds = [None]
            elif isinstance(seeds, (int, float)):
                seeds = [int(seeds)]
            else:
                seeds = list(seeds)

            # Extract embedding postfix from the train path
            postfix = get_embedding_postfix(train_path)

            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            run_dir = Path.cwd() / f"validator_output_{timestamp}_{postfix}"
            run_dir.mkdir(parents=True, exist_ok=True)

            for seed in seeds:
                if seed is not None:
                    seed_reports, seed_metrics = run_downstream_with_seed(
                        config_dict, train_path, test_path, seed
                    )
                    label = f"seed_{seed}"
                else:
                    no_seed_config = deepcopy(config_dict)
                    no_seed_config.pop("validator_seeds", None)
                    if "embedding_metrics" in no_seed_config:
                        no_seed_config["embedding_metrics"]["enabled"] = False
                    seed_reports = run_with_paths(
                        downstream_config=no_seed_config,
                        train_path=train_path,
                        test_path=test_path,
                    )
                    seed_metrics = extract_downstream_metrics(seed_reports)
                    label = "no_seed"

                save_seed_metrics(
                    run_dir / f"downstream_validator_{label}.csv",
                    seed_metrics,
                )
                pd.DataFrame(seed_reports).to_json(
                    run_dir / f"validator_output_{label}.json",
                    orient='records',
                    indent=4,
                    date_format='iso'
                )

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
