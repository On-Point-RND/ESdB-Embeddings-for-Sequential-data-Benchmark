import gc
import logging
from collections.abc import Mapping
from pathlib import Path

# from torch import nn
import pandas as pd
import torch
from tqdm.autonotebook import tqdm

from validate import run_with_paths

from ...data.utils import build_loaders
from ...model import build_model
from ...trainer import Trainer
from ..base_runner import Runner
from ..data_retrieve.auto_post_processing import post_processing
from ..data_retrieve.embeddings_gen import ResultsGetter
from ..utils import get_loss, get_metrics, get_optimizer, get_scheduler, suggest_conf, extract_downstream_metrics

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
        downstream_config = config.get("universal_validator", None)
        embed_train_file = None
        embed_test_file = None
        if downstream_config:
            train_embeddings_getter = ResultsGetter(config, "train")
            keys = {"gen_train", "gen_train_val"}
            subloaders = {k: loaders[k] for k in keys if k in loaders}
            df_train = train_embeddings_getter.df_get(subloaders, trainer)
            embed_train_file = (
                Path(config["log_dir"]) / config["run_name"] / "embeddings" / "train"
            )
            embed_train_file.parent.mkdir(parents=True, exist_ok=True)
            df_train.to_parquet(embed_train_file, index=False)
            post_processing(config, embed_train_file, "train")

            test_embeddings_getter = ResultsGetter(config, "test")
            keys = {"gen_test"}
            subloaders = {k: test_loaders[k] for k in keys if k in test_loaders}
            df_test = test_embeddings_getter.df_get(subloaders, trainer)
            embed_test_file = (
                Path(config["log_dir"]) / config["run_name"] / "embeddings" / "test"
            )
            embed_test_file.parent.mkdir(parents=True, exist_ok=True)
            df_test.to_parquet(embed_test_file, index=False)
            post_processing(config, embed_test_file, "test")

        del loaders["train"]  # type: ignore
        del loaders["gen_train"]  # type: ignore
        del loaders["train_val"]  # type: ignore
        del loaders["gen_train_val"]  # type: ignore
        del test_loaders  # type: ignore

        downstream_metrics = {}
        if downstream_config:
            reports = run_with_paths(
                downstream_config=downstream_config,
                train_path=str(embed_train_file),
                test_path=str(embed_test_file),
            )
            downstream_metrics = extract_downstream_metrics(reports)
        train_metrics = trainer.validate(loaders["unsupervised_train"])
        train_val_metrics = trainer.validate(loaders["unsupervised_train_val"])

        train_metrics = {"train_" + k: v for k, v in train_metrics.items()}
        # train_val_metrics = {k: v for k, v in train_val_metrics.items()}
        # test_metrics = {"test_" + k: v for k, v in test_metrics.items()}
        return dict(**train_metrics, **train_val_metrics, **downstream_metrics)

    def param_grid(self, trial, config):
        suggest_conf(config["optuna"]["suggestions"], config, trial)
        return trial, config
