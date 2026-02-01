from collections.abc import Mapping
import gc
from pathlib import Path

# from torch import nn
import pandas as pd

import logging
import torch
from tqdm.autonotebook import tqdm

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
            df_train_1 = self.df_getter(loaders["train"], trainer)
            df_train_2 = self.df_getter(loaders["train_val"], trainer)
            df_all = pd.concat([df_train_1, df_train_2], ignore_index=True)
            df_test = self.df_getter(test_loaders["test"], trainer)
            df_all = pd.concat([df_all, df_test], ignore_index=True)
            embed_file = Path(config["log_dir"]) / config["run_name"] / "embeddings_joined"
            df_all.to_parquet(embed_file, index=False)

        del loaders["train"]  # type: ignore
        train_metrics = trainer.validate(loaders["unsupervised_train"])
        del loaders["full_train"]  # type: ignore
        train_val_metrics = trainer.validate(loaders["unsupervised_train_val"])
        train_val_metrics = {k: -v for k, v in test_metrics.items()}
        del loaders["train_val"]  # type: ignore


        train_metrics = {"train_" + k: v for k, v in train_metrics.items()}
        # train_val_metrics = {k: v for k, v in train_val_metrics.items()}
        test_metrics = {"test_" + k: v for k, v in test_metrics.items()}
        return dict(**train_metrics, **train_val_metrics, **test_metrics)

    def param_grid(self, trial, config):
        suggest_conf(config["optuna"]["suggestions"], config, trial)
        return trial, config

    @staticmethod
    def df_getter(loader, trainer):
        assert trainer.model is not None
        embedding_model = trainer.model
        if loader is None:
            raise ValueError("Incorrect loader for embeddings generation")
        logger.info("Embedding generation on one of the loaders started")

        records = []
        for batch in tqdm(loader, disable=not trainer.verbose):
            ###
            # Here should be the function which makes batch much bigger, composed out of all the idx we have in data's ""
            ###
            batch.to(trainer.device)
            with torch.no_grad():
                emb = embedding_model(batch)

            emb_list = emb.cpu().numpy().tolist()
            index_data = batch.extract_indexes_from_batch()

            for i in range(len(emb)):
                record = {
                    "embedding": emb_list[i],
                    "index": index_data[i],
                }
                records.append(record)
        df = pd.DataFrame(records)
        logger.info("Embedding generation finished")
        return df
