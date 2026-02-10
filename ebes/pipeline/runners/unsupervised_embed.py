from collections.abc import Mapping
import gc
from pathlib import Path

# from torch import nn
import pandas as pd

import logging
import torch
from tqdm.autonotebook import tqdm

from ..data_retrieve.embeddings_gen import ResultsGetter
from ..data_retrieve.auto_post_processing import post_processing
from ...data.utils import build_loaders
from ...model import build_model
from ...trainer import Trainer
from ..base_runner import Runner
from ..utils import get_loss, get_metrics, get_optimizer, suggest_conf, get_scheduler

logger = logging.getLogger(__name__)


class UnsupervisedEmbedRunner(Runner):
    def pipeline(self, config: Mapping) -> dict[str, float]:
        loaders = build_loaders(**config["data"])
        test_loaders = build_loaders(**config["test_data"])

        net = build_model(config["unsupervised_model"])
        opt = get_optimizer(net.parameters(), **config["optimizer"])
        lr_scheduler = None
        if "lr_scheduler" in config:
            lr_scheduler = get_scheduler(opt, **config["lr_scheduler"])
        loss = get_loss(**config["unsupervised_loss"])
        metrics = get_metrics(config.get("unsupervised_metrics"), "cpu")
        trainer = Trainer(
            model=net,
            loss=loss,
            optimizer=opt,
            lr_scheduler=lr_scheduler,
            train_loader=loaders["unsupervised_train"],
            val_loader=loaders["unsupervised_train_val"],
            run_name=config["run_name"] + "/pretrain",
            ckpt_dir=Path(config["log_dir"]) / config["run_name"] / "pretrain" / "ckpt",
            device=config["device"],
            metrics=metrics,
            **config["unsupervised_trainer"],
        )
        trainer.run()

        # loaders["unsupervised_train"]  # type: ignore
        # loaders["unsupervised_train_val"]  # type: ignore
        gc.collect()

        net.eval()

        trainer.load_best_model()

        run_type = config["runner"]["run_type"]
        if run_type == "simple":
            train_embeddings_getter = ResultsGetter(config, "train")
            keys = {"train", "train_val"}
            subloaders = {k: loaders[k] for k in keys if k in loaders}
            df_train = train_embeddings_getter.df_get(subloaders, trainer)
            embed_train_file = Path(config["log_dir"]) / config["run_name"] / "embeddings" / "train" 
            embed_train_file.parent.mkdir(parents=True, exist_ok=True)
            df_train.to_parquet(embed_train_file, index=False)
            post_processing(config, embed_train_file, "train")

            test_embeddings_getter = ResultsGetter(config, "test")
            df_test = test_embeddings_getter.df_get(test_loaders, trainer)
            embed_test_file = Path(config["log_dir"]) / config["run_name"] / "embeddings" / "test" 
            embed_test_file.parent.mkdir(parents=True, exist_ok=True)
            df_test.to_parquet(embed_test_file, index=False)
            post_processing(config, embed_test_file, "test")

        del loaders["train"]  # type: ignore
        train_metrics = trainer.validate(loaders["unsupervised_train"])
        del loaders["full_train"]  # type: ignore
        train_val_metrics = trainer.validate(loaders["unsupervised_train_val"])
        del loaders["train_val"]  # type: ignore


        train_metrics = {"train_" + k: v for k, v in train_metrics.items()}
        # train_val_metrics = {k: v for k, v in train_val_metrics.items()}
        #test_metrics = {"test_" + k: v for k, v in test_metrics.items()}
        return dict(**train_metrics, **train_val_metrics)

    def param_grid(self, trial, config):
        suggest_conf(config["optuna"]["suggestions"], config, trial)
        return trial, config

