"""Something basic with returning output_dim"""

import torch
from torch import nn

from .basemodel import BaseModel
from ..types import Seq


class Linear(BaseModel):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.output_dim = out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)



