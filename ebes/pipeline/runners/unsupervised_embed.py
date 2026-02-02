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
            df_train_1 = self.df_getter(config, loaders["train"], trainer)
            df_train_2 = self.df_getter(config, loaders["train_val"], trainer)
            df_all = pd.concat([df_train_1, df_train_2], ignore_index=True)
            #df_test = self.df_getter(config, test_loaders["test"], trainer)
            #df_all = pd.concat([df_all, df_test], ignore_index=True)
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
    def df_getter(config, loader, trainer):
        assert trainer.model is not None
        embedding_model = trainer.model
        if loader is None:
            raise ValueError("Incorrect loader for embeddings generation")
        logger.info("Embedding generation on one of the loaders started")

        shift_transformer = batch_transformer(config)

        records = []
        for batch_old in tqdm(loader, disable=not trainer.verbose):
            ###
            # Here should be the function which makes batch much bigger, composed out of all the idx we have
            # in data's "shifts" column
            ###
            batch = shift_transformer.transform(batch_old)

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
class batch_transformer():
    def __init__(self, config):
        data_path = Path(config["data"]["dataset"]["parquet_path"])
        index_name=config["data"]["preprocessing"]["index_name"]
        data_df=pd.read_parquet(data_path, columns=[self.index_name, "shifts"])
        self.shifts_by_index = (
            data_df
            .set_index(index_name)["shifts"]
            .to_dict()
        )


    def transform(self, batch):
        device = batch.time.device if isinstance(batch.time, torch.Tensor) else None
        old_len, old_batch = batch.time.shape

        new_num_features = []
        new_cat_features = []
        new_times = []
        new_lengths = []
        new_indices = []
        new_targets = []
        reps=[]
        if batch.emb_features is not None:
            new_emb_features = {k: [] for k in batch.emb_features}

        for b in range(old_batch):
            old_index = batch.index[b]
            shifts = self.shifts_by_index[old_index]
            assert batch.cat_mask is None and batch.num_mask is not None and batch.emb_mask is not None, "mask is not implemented here in batch division"
            for s in shifts:
                s = int(s)

                # ---- time ----
                t = batch.time[:, b]
                new_t = torch.zeros(old_len, device=device)
                new_t[:s] = t[:s]
                new_times.append(new_t)

                # ---- num features ----
                if batch.emb_features is not None:
                    for name, emb in batch.emb_features.items():
                        e = emb[:, :, b]
                        new_e = torch.zeros_like(e)
                        new_e[:, :s] = e[:, :s]
                        new_emb_features[name].append(new_e)

                # ---- num features ----
                if batch.num_features is not None:
                    nf = batch.num_features[:, b, :]
                    new_nf = torch.zeros_like(nf)
                    new_nf[:s] = nf[:s]
                    new_num_features.append(new_nf)

                # ---- cat features ----
                if batch.cat_features is not None:
                    cf = batch.cat_features[:, b, :]
                    new_cf = torch.zeros_like(cf)
                    new_cf[:s] = cf[:s]
                    new_cat_features.append(new_cf)

                # ---- target ----
                if batch.target is not None:
                    new_targets.append(batch.target[b])
                # ---- length ----
                new_lengths.append(s)
                # ---- index ----
                new_indices.append(f"{old_index.item()}__{s}")

        # stack
        if batch.target is not None:
            new_target = torch.stack(new_targets, dim=1)
        new_time = torch.stack(new_times, dim=1)
        new_lengths = torch.tensor(new_lengths, device=device)
        new_index = new_indices

        batch.emb_features = ({
            name: torch.stack(lst, dim=2)
            for name, lst in new_emb_features.items()
        } if batch.emb_features is not None else None)

        new_num_features = (
            torch.stack(new_num_features, dim=1)
            if batch.num_features is not None else None
        )

        new_cat_features = (
            torch.stack(new_cat_features, dim=1)
            if batch.cat_features is not None else None
        )
        return Batch(
            lengths=new_lengths,
            time=new_time,
            index=new_index,
            target=new_target,  # если target уже задублирован по shifts — ок
            num_features=new_num_features,
            cat_features=new_cat_features,
            emb_features=new_emb_features,
            cat_features_names=batch.cat_features_names,
            num_features_names=batch.num_features_names,
            emb_features_names=batch.emb_features_names,
        )


