from collections.abc import Mapping
from pathlib import Path

from ...data.utils import build_loaders
from ...model import build_model
from ...trainer import Trainer
from ..base_runner import Runner
from ..downstream_selection import build_downstream_checkpoint_evaluator
from ..data_retrieve.downstreams import compute_downstreams
from ..utils import (
    get_loss,
    get_metrics,
    get_optimizer,
    get_scheduler,
    suggest_conf,
)


class BertRunner(Runner):
    def pipeline(self, config: Mapping) -> dict[str, float]:
        loaders = build_loaders(**config["data"])
        test_loaders = build_loaders(**config["test_data"])
        net = build_model(config["model"])
        opt = get_optimizer(net.parameters(), **config["optimizer"])
        lr_scheduler = None
        if "lr_scheduler" in config:
            lr_scheduler = get_scheduler(opt, **config["lr_scheduler"])
        metrics = get_metrics(config["metrics"], "cpu")
        loss = get_loss(**config["main_loss"])
        checkpoint_evaluator = build_downstream_checkpoint_evaluator(
            config=config,
            train_loaders=loaders,
            test_loaders=test_loaders,
        )

        trainer = Trainer(
            model=net,
            loss=loss,
            optimizer=opt,
            lr_scheduler=lr_scheduler,
            metrics=metrics,
            train_loader=loaders["train"],
            val_loader=loaders["train_val"],
            run_name=config["run_name"],
            ckpt_dir=Path(config["log_dir"]) / config["run_name"] / "ckpt",
            device=config["device"],
            checkpoint_evaluator=checkpoint_evaluator,
            **config["trainer"],
        )

        trainer.run()

        ckpt_dir = Path(config["log_dir"]) / config["run_name"] / "ckpt"
        if any(ckpt_dir.glob("*.ckpt")):
            trainer.load_best_model()

        run_type = config["runner"]["run_type"]
        downstream_config = config.get("universal_validator", None)
        downstream_metrics = {}
        if downstream_config or run_type == "simple":
            downstream_metrics = compute_downstreams(
                trainer=trainer,
                train_loaders=loaders,
                test_loaders=test_loaders,
                config=config,
                downstream_config=downstream_config,
            )

        train_metrics = trainer.validate(loaders["full_train"])
        train_val_metrics = trainer.validate(loaders["train_val"])
        test_metrics = trainer.validate(test_loaders["test"])

        train_metrics = {"train_" + k: v for k, v in train_metrics.items()}
        train_val_metrics = {"train_val_" + k: v for k, v in train_val_metrics.items()}
        test_metrics = {"test_" + k: v for k, v in test_metrics.items()}

        return dict(
            **train_metrics, **train_val_metrics, **test_metrics, **downstream_metrics
        )

    def param_grid(self, trial, config):
        suggest_conf(config["optuna"]["suggestions"], config, trial)
        return trial, config
