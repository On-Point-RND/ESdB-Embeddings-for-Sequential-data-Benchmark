from dataclasses import dataclass
from typing import Any, Mapping, Optional, Union

@dataclass(frozen=True)
class DataConfig:
    dataset_name: str = "shakespeare"
    train_path: str = ""
    test_path: str = ""
    # batch_size: int = 128
    # num_workers: int = 4
    # val_ratio: float = 0.15
    #
    # max_seq_len: int = 0
    #
    # time_name: str = "Time"
    # cat_cardinalities: Optional[Union[list[list[Any]], Mapping[str, Any]]] = None
    # num_names: Optional[list[str]] = None
    # index_name: Optional[str] = None
    
    # train_transforms: Optional[Mapping[str, Optional[Mapping[str, Any] | str]]] = None
    # val_transforms: Optional[Mapping[str, Optional[Mapping[str, Any] | str]]] = None
    # padding_value: float = 0
    # List of features to focus on in loss and metrics. If None->focus on all
    # @property
    # def seq_cols(self):
    #     seq_cols = [self.time_name]
    #     if self.cat_cardinalities is not None:
    #         seq_cols += list(self.cat_cardinalities)
    #     if self.num_names is not None:
    #         seq_cols += self.num_names
    #     return seq_cols
