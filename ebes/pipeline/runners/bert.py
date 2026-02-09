from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from ...data.utils import build_loaders
from ...model import build_model
from ...trainer import Trainer
from ..base_runner import Runner
from ..utils import get_loss, get_metrics, get_optimizer, suggest_conf, get_scheduler


class BertRunner(Runner):
    def pipeline(self, config: Mapping) -> tuple[dict[str, float], pd.DataFrame]:
        breakpoint()
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
        trainer.load_best_model()

        df_train = trainer.bert_emb_gather(loaders["train"])
        df_train_val = trainer.bert_emb_gather(loaders["train_val"])
        df_hpo = trainer.bert_emb_gather(loaders["hpo_val"])
        df_test = trainer.bert_emb_gather(test_loaders["test"])
        df_all = pd.concat([df_train, df_train_val, df_hpo, df_test], ignore_index=True)

        train_metrics = trainer.validate(loaders["full_train"])
        train_val_metrics = trainer.validate(loaders["train_val"])
        hpo_metrics = trainer.validate(loaders["hpo_val"])
        test_metrics = trainer.validate(test_loaders["test"])

        train_metrics = {"train_" + k: v for k, v in train_metrics.items()}
        train_val_metrics = {"train_val_" + k: v for k, v in train_val_metrics.items()}
        test_metrics = {"test_" + k: v for k, v in test_metrics.items()}

        return dict(**hpo_metrics, **train_metrics, **train_val_metrics, **test_metrics), df_all

    def param_grid(self, trial, config):
        suggest_conf(config["optuna"]["suggestions"], config, trial)
        return trial, config
