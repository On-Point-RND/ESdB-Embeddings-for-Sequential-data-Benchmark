from collections.abc import Mapping
from pathlib import Path

from ...data.utils import build_loaders
from ...model import build_model
from ...trainer import Trainer
from ..base_runner import Runner
from ..data_retrieve.auto_post_processing import post_processing
from ..data_retrieve.embeddings_gen import ResultsGetter
from ..utils import get_loss, get_metrics, get_optimizer, suggest_conf, get_scheduler


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

        run_type = config["runner"]["run_type"]
        if run_type == "simple":
            train_embeddings_getter = ResultsGetter(config, "train")
            keys = {"gen_train", "gen_train_val"}
            subloaders = {k: loaders[k] for k in keys if k in loaders}
            df_train = train_embeddings_getter.df_get(subloaders, trainer)
            embed_train_file = Path(config["log_dir"]) / config["run_name"] / "embeddings" / "train"
            embed_train_file.parent.mkdir(parents=True, exist_ok=True)
            df_train.to_parquet(embed_train_file, index=False)
            post_processing(
                config,
                embed_train_file,
                "train",
                transformer_skipped_target_removal=True,
            )

            test_embeddings_getter = ResultsGetter(config, "test")
            keys = {"gen_test"}
            subloaders = {k: test_loaders[k] for k in keys if k in test_loaders}
            df_test = test_embeddings_getter.df_get(subloaders, trainer)
            embed_test_file = Path(config["log_dir"]) / config["run_name"] / "embeddings" / "test"
            embed_test_file.parent.mkdir(parents=True, exist_ok=True)
            df_test.to_parquet(embed_test_file, index=False)
            post_processing(
                config,
                embed_test_file,
                "test",
                transformer_skipped_target_removal=True,
            )

        # intervented here a little: no full_train exists anymore and gen_train is for generation only
        train_metrics = trainer.validate(loaders["train"])
        train_val_metrics = trainer.validate(loaders["train_val"])
        test_metrics = trainer.validate(test_loaders["test"])

        train_metrics = {"train_" + k: v for k, v in train_metrics.items()}
        train_val_metrics = {"train_val_" + k: v for k, v in train_val_metrics.items()}
        test_metrics = {"test_" + k: v for k, v in test_metrics.items()}

        return dict(**train_metrics, **train_val_metrics, **test_metrics)

    def param_grid(self, trial, config):
        suggest_conf(config["optuna"]["suggestions"], config, trial)
        return trial, config
