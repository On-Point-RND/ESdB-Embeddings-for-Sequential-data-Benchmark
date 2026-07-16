import logging
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..data.utils import build_loaders
from .data_retrieve.downstreams import (
    create_postproc_spark_session,
    list_downstream_tasks,
    normalize_loader_ref,
    prepare_downstream_file,
    run_downstream_validator,
)

logger = logging.getLogger(__name__)


class DownstreamCheckpointEvaluator:
    def __init__(
        self,
        *,
        config: Mapping[str, Any],
        data_config: Mapping[str, Any],
        train_loaders: Mapping[str, Any],
        test_loaders: Mapping[str, Any],
        downstream_config: Mapping[str, Any],
        every_n_epochs: int,
        train_loader_refs: list[str],
        global_train_loader_refs: list[str],
        score_loaders: Mapping[str, list[str]],
        global_score_name: str,
    ):
        if every_n_epochs < 1:
            raise ValueError("downstream_selection.every_n_epochs must be positive")

        self.config = config
        self.data_config = data_config
        self.train_loaders = train_loaders
        self.test_loaders = test_loaders
        self.downstream_config = downstream_config
        self.every_n_epochs = every_n_epochs
        self.train_loader_refs = train_loader_refs
        self.global_train_loader_refs = global_train_loader_refs
        self.score_loader_refs = dict(score_loaders)
        self.global_score_name = global_score_name

    def evaluate(self, trainer, epoch: int) -> dict[str, float]:
        if epoch != 1 and epoch % self.every_n_epochs != 0:
            return {}

        run_dir = Path(self.config["log_dir"]) / self.config["run_name"]
        root = run_dir / "downstream_selection" / f"epoch_{epoch:04d}"
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Epoch %04d: downstream checkpoint evaluation started",
            epoch,
        )
        try:
            metrics = self._evaluate_epoch(trainer=trainer, root=root)
        finally:
            shutil.rmtree(root, ignore_errors=True)

        prefixed = {
            f"downstream/{score_name}/{key}": value
            for score_name, score_metrics in metrics.items()
            for key, value in score_metrics.items()
        }
        logger.info(
            "Epoch %04d: downstream checkpoint evaluation finished",
            epoch,
        )
        return prefixed

    def _evaluate_epoch(
        self,
        *,
        trainer,
        root: Path,
    ) -> dict[str, dict[str, float]]:
        spark = create_postproc_spark_session()
        try:
            local_train_path = prepare_downstream_file(
                trainer=trainer,
                train_loaders=self.train_loaders,
                test_loaders=self.test_loaders,
                config=self.data_config,
                loader_refs=self.train_loader_refs,
                output_path=root / "local_train",
                spark=spark,
            )
            task_names = list_downstream_tasks(
                local_train_path,
                self.downstream_config.get("task_names"),
            )
            global_tasks = [task for task in task_names if "__global__" in task]
            local_tasks = [task for task in task_names if "__local__" in task]

            metrics = {}
            if global_tasks:
                if self.global_train_loader_refs == self.train_loader_refs:
                    global_train_path = local_train_path
                else:
                    global_train_path = prepare_downstream_file(
                        trainer=trainer,
                        train_loaders=self.train_loaders,
                        test_loaders=self.test_loaders,
                        config=self.data_config,
                        loader_refs=self.global_train_loader_refs,
                        output_path=root / "global_train",
                        spark=spark,
                    )
                metrics[self.global_score_name] = run_downstream_validator(
                    dict(self.downstream_config),
                    train_path=global_train_path,
                    test_path=global_train_path,
                    task_names=global_tasks,
                )

            if local_tasks:
                if not self.score_loader_refs:
                    raise ValueError(
                        "downstream_selection.score_loaders is required for "
                        "local downstream tasks"
                    )
                for score_name, score_loader_refs in self.score_loader_refs.items():
                    score_path = prepare_downstream_file(
                        trainer=trainer,
                        train_loaders=self.train_loaders,
                        test_loaders=self.test_loaders,
                        config=self.data_config,
                        loader_refs=score_loader_refs,
                        output_path=root / score_name,
                        spark=spark,
                    )
                    metrics[score_name] = run_downstream_validator(
                        dict(self.downstream_config),
                        train_path=local_train_path,
                        test_path=score_path,
                        task_names=local_tasks,
                    )
            return metrics
        finally:
            spark.stop()


def build_downstream_checkpoint_evaluator(
    *,
    config: Mapping[str, Any],
    train_loaders: Mapping[str, Any],
    test_loaders: Mapping[str, Any],
) -> DownstreamCheckpointEvaluator | None:
    selection_config = config.get("downstream_selection")
    if not selection_config:
        return None

    _check_downstream_track_metric(config)

    downstream_config = config.get("universal_validator")
    if downstream_config is None:
        raise ValueError(
            "`downstream_selection` requires universal validator config. "
            "Pass it with `-dv path/to/validator.yaml`."
        )

    data_config = config
    if "data" in selection_config:
        data_config = {
            **dict(config),
            "data": selection_config["data"],
        }
        train_loaders = build_loaders(**selection_config["data"])
        if "test_data" in selection_config:
            data_config["test_data"] = selection_config["test_data"]
            test_loaders = build_loaders(**selection_config["test_data"])
        else:
            test_loaders = {}

    def as_loader_refs(value) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [normalize_loader_ref(value)]
        return [normalize_loader_ref(ref) for ref in value]

    if "score_loaders" in selection_config:
        score_loaders = {
            name: as_loader_refs(refs)
            for name, refs in selection_config["score_loaders"].items()
        }
    elif selection_config.get("score_loader") is not None:
        score_loader_refs = as_loader_refs(selection_config["score_loader"])
        score_ref = score_loader_refs[0]
        _, score_key = score_ref.split(".", 1)
        score_name = selection_config.get(
            "score_name", score_key.removeprefix("downstream_")
        )
        score_loaders = {score_name: score_loader_refs}
    else:
        score_loaders = {}

    if "train_loaders" in selection_config:
        train_loader_refs = as_loader_refs(selection_config["train_loaders"])
    else:
        train_loader_refs = [normalize_loader_ref(selection_config["fit_loader"])]

    global_train_loader_refs = as_loader_refs(
        selection_config.get("global_train_loaders", train_loader_refs)
    )
    all_refs = train_loader_refs + global_train_loader_refs
    for refs in score_loaders.values():
        all_refs.extend(refs)
    if "data" in selection_config and any(
        ref.startswith("test_data.") for ref in all_refs
    ) and "test_data" not in selection_config:
        raise ValueError(
            "downstream_selection references test_data loaders, but "
            "downstream_selection.test_data is not configured"
        )

    return DownstreamCheckpointEvaluator(
        config=config,
        data_config=data_config,
        train_loaders=train_loaders,
        test_loaders=test_loaders,
        downstream_config=downstream_config,
        every_n_epochs=selection_config["every_n_epochs"],
        train_loader_refs=train_loader_refs,
        global_train_loader_refs=global_train_loader_refs,
        score_loaders=score_loaders,
        global_score_name=selection_config.get("global_score_name", "global"),
    )


def _check_downstream_track_metric(config: Mapping[str, Any]) -> None:
    metric = None
    for trainer_key in ("unsupervised_trainer", "trainer"):
        trainer_config = config.get(trainer_key, {})
        metric = trainer_config.get("ckpt_track_metric")
        if isinstance(metric, str) and metric.startswith("downstream/"):
            break
    else:
        return

    parts = metric.split("/", 2)
    if len(parts) != 3:
        return

    score_name, task_name = parts[1], parts[2]
    if score_name == "global" and "__local__" in task_name:
        raise ValueError(
            f"Invalid downstream checkpoint metric: {metric!r}. "
            "Local tasks are scored under downstream/val/... or "
            "downstream/test/..., not downstream/global/...."
        )
    if score_name in {"val", "test"} and "__global__" in task_name:
        raise ValueError(
            f"Invalid downstream checkpoint metric: {metric!r}. "
            "Global tasks are scored under downstream/global/..., not "
            "downstream/val/... or downstream/test/...."
        )
