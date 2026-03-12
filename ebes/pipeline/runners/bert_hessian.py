from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ...data.utils import build_loaders
from ...model import build_model
from ...trainer import Trainer
from ..base_runner import Runner
from ..hessian_probe import compute_hessian_metrics
from ..utils import get_loss, get_metrics, get_optimizer, suggest_conf, get_scheduler


def _resolve_loader(
    loader_name: str,
    train_loaders: Mapping[str, Any],
    test_loaders: Mapping[str, Any],
):
    if loader_name in train_loaders:
        return train_loaders[loader_name]
    if loader_name in test_loaders:
        return test_loaders[loader_name]
    raise ValueError(f"Unknown loss/hessian loader '{loader_name}'")


class BertHessianRunner(Runner):
    """
    Lightweight BERT runner for loss + Hessian probes.
    Does not generate embeddings or post-processing artifacts.
    """

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
            **config["trainer"],
        )

        trainer.run()

        ckpt_dir = Path(config["log_dir"]) / config["run_name"] / "ckpt"
        if any(ckpt_dir.glob("*.ckpt")):
            trainer.load_best_model()

        probe_loader_name = str(config.get("hessian", {}).get("loader", "train_val"))
        probe_loader = _resolve_loader(probe_loader_name, loaders, test_loaders)
        probe_loss = trainer.validate(probe_loader)["loss"]

        hessian_metrics = compute_hessian_metrics(
            config=config,
            trainer=trainer,
            loaders=loaders,
            test_loaders=test_loaders,
            loss_fn=loss,
        )

        return {
            "probe_loss": probe_loss,
            **hessian_metrics,
        }

    def param_grid(self, trial, config):
        suggest_conf(config["optuna"]["suggestions"], config, trial)
        return trial, config

