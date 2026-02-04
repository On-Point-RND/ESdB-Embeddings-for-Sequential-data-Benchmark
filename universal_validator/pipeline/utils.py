from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Optional

from universal_validator.data.data_types import DataConfig


# @dataclass
# class NRunsConfig:
#     n_runs: int
#     n_workers: int


# @dataclass
# class RunnerConfig:
#     name: str = "GenerationRunner"
#     run_type: str = "simple"
#     seed_keys: list[str] = field(default_factory=lambda: ["common_seed"])
#     params: NRunsConfig = field(default_factory=NRunsConfig)
#     device_list: Optional[list[str]] = None


# @dataclass(frozen=True)
# class OptunaParams:
#     target_metric: str
#     n_trials: int = 50
#     n_startup_trials: int = 3
#     request_list: list[dict] = field(default_factory=list)
#     multivariate: bool = True
#     group: bool = True


# @dataclass(frozen=True)
# class OptunaConfig:
#     suggestions: list
#     params: OptunaParams = field(default_factory=OptunaParams)


@dataclass
class PipelineConfig:
    config_factory: Optional[list[str]] = None
    # log_dir: str = "log"
    # common_seed: int = 0
    # data_conf: DataConf = field(default_factory=DataConfig)
    # # Specific task type
    # task_types: Literal['classification', 'regression', 'anomaly_detection', 'forecast']
    # # Run all configured experiments
    # run_all: bool = False
    # # List available configurations
    # list-configs: bool = False
    # run_name: str = "debug/-"
    # device: str = "cuda:0"

    # evaluator: EvaluatorConfig = field(default_factory=EvaluatorConfig)
    # model: Optional[ModelConfig] = None
    # trainer: TrainConfig = field(default_factory=TrainConfig)
    # optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    # schedulers: Optional[Mapping[str, Mapping[str, Any] | str]] = None
    # loss: LossConfig = field(default_factory=LossConfig)
    # logging: LoginConfig = field(default_factory=LoginConfig)
    # runner: RunnerConfig = field(default_factory=RunnerConfig)
    # optuna: Optional[OptunaConfig] = None
