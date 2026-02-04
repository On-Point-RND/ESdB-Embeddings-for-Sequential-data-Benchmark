from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Optional

from universal_validator.data.data_types import DataConfig


# @dataclass(frozen=True)
# class OptunaConfig:
#     suggestions: list
#     n_trials: int = 50
#     n_startup_trials: int = 3
#     request_list: list[dict] = field(default_factory=list)
#     multivariate: bool = True
#     group: bool = True


@dataclass
class ValidatorConfig:
    config_factory: Optional[list[str]] = None
    data_conf: DataConfig = field(default_factory=DataConfig)
    # Specific tasks. If None, then run all available
    task_names: Optional[list[str]] = None
    # List available configurations
    list_configs: bool = False
    seed: int = 0

    downstream: dict = field(default_factory=dict)