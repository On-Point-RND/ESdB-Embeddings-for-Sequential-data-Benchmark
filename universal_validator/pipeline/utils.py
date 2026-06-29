from dataclasses import dataclass, field
from typing import Any, Optional

from universal_validator.data.dataset import DataConfig
from universal_validator.tasks.hpo_optimizer import HPOConfig

# @dataclass(frozen=True)
# class OptunaConfig:
#     suggestions: list
#     n_trials: int = 50
#     n_startup_trials: int = 3
#     request_list: list[dict] = field(default_factory=list)
#     multivariate: bool = True
#     group: bool = True


@dataclass
class EmbeddingMetricsConfig:
    enabled: bool = False
    metrics: list[str] = field(default_factory=lambda: ["effective_rank", "anisotropy"])
    sources: Optional[list[str]] = None
    splits: Optional[list[str]] = None
    sample_size: Optional[int] = None
    sample_seed: int = 42
    asmi: dict[str, Any] = field(default_factory=dict)
    total_compression: dict[str, Any] = field(default_factory=dict)
    lidar: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidatorConfig:
    config_factory: Optional[list[str]] = None
    data_conf: DataConfig = field(default_factory=DataConfig)
    # Specific tasks. If None, then run all available
    task_names: Optional[list[str]] = None
    validator_seeds: Optional[list[int]] = None
    # List available configurations
    list_configs: bool = False
    seed: int = 0
    verbose: bool = True

    hpo: HPOConfig = field(default_factory=HPOConfig)
    models: dict = field(default_factory=dict)
    embedding_metrics: EmbeddingMetricsConfig = field(
        default_factory=EmbeddingMetricsConfig
    )
