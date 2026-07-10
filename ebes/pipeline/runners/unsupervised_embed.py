import gc
import logging
from collections.abc import Mapping
from pathlib import Path

import torch

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

logger = logging.getLogger(__name__)


class _EmbeddingCheckpointEvaluator:
    def __init__(self, evaluator, embedding_model, device: str):
        self.evaluator = evaluator
        self.embedding_model = embedding_model
        self.device = device

    def evaluate(self, trainer: Trainer, epoch: int) -> dict[str, float]:
        training_model = trainer.model
        assert training_model is not None

        self.embedding_model.load_state_dict(
            training_model.state_dict(),
            strict=False,
        )
        self.embedding_model.eval().to(self.device)

        trainer._model = self.embedding_model
        try:
            return self.evaluator.evaluate(trainer, epoch)
        finally:
            trainer._model = training_model
            self.embedding_model.to("cpu")
            torch.cuda.empty_cache()


class UnsupervisedEmbedRunner(Runner):

    def pipeline(self, config: Mapping) -> dict[str, float]:
        loaders = build_loaders(**config["data"])
        test_loaders = build_loaders(**config["test_data"])

        net = build_model(config["training_model"])
        opt = get_optimizer(net.parameters(), **config["optimizer"])
        lr_scheduler = None
        if "lr_scheduler" in config:
            lr_scheduler = get_scheduler(opt, **config["lr_scheduler"])
        loss = get_loss(**config["unsupervised_loss"])
        metrics = get_metrics(config.get("unsupervised_metrics"), "cpu")
        checkpoint_evaluator = build_downstream_checkpoint_evaluator(
            config=config,
            train_loaders=loaders,
            test_loaders=test_loaders,
        )
        if checkpoint_evaluator is not None:
            checkpoint_evaluator = _EmbeddingCheckpointEvaluator(
                evaluator=checkpoint_evaluator,
                embedding_model=build_model(config["model"]),
                device=config["device"],
            )
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
            checkpoint_evaluator=checkpoint_evaluator,
            **config["unsupervised_trainer"],
        )
        trainer.run()
        gc.collect()
        net.eval()

        ckpt_dir = Path(config["log_dir"]) / config["run_name"] / "pretrain" / "ckpt"
        if any(ckpt_dir.glob("*.ckpt")):
            ckpt_path = trainer.best_checkpoint()
        else:
            ckpt_path = config["unsupervised_trainer"].get("ckpt_resume")
        if not ckpt_path:
            raise ValueError(
                "No checkpoint found and unsupervised_trainer.ckpt_resume is empty"
            )
        trainer.load_ckpt(ckpt_path)

        # metrics collection on training model
        ###############
        train_metrics = trainer.validate(loaders["unsupervised_train"])
        train_val_metrics = trainer.validate(loaders["unsupervised_train_val"])

        train_metrics = {"train_" + k: v for k, v in train_metrics.items()}
        # train_val_metrics = {k: v for k, v in train_val_metrics.items()}
        # test_metrics = {"test_" + k: v for k, v in test_metrics.items()}
        ###############
        embed_net = build_model(config["model"])
        embed_net.load_state_dict(
            torch.load(ckpt_path, map_location="cpu")["model"],
            strict=False,
        )
        trainer._model = embed_net.eval().to(config["device"])

        # Free training-time state (model with contrastive heads, Adam buffers,
        # optional scheduler) before embedding generation to leave headroom for
        # the gen-phase activations on the same GPU.
        del net, opt
        if lr_scheduler is not None:
            del lr_scheduler
        gc.collect()
        torch.cuda.empty_cache()

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

        del loaders["train"]  # type: ignore
        del loaders["gen_train"]  # type: ignore
        del loaders["train_val"]  # type: ignore
        del loaders["gen_train_val"]  # type: ignore
        if "hpo_val" in loaders.keys():
            del loaders["hpo_val"]  # type: ignore
        del test_loaders  # type: ignore

        return dict(**train_metrics, **train_val_metrics, **downstream_metrics)

    def param_grid(self, trial, config):
        suggest_conf(config["optuna"]["suggestions"], config, trial)
        return trial, config
