from dataclasses import dataclass
from typing import List, Dict
import torch


@dataclass
class GradientsUpdateMessage:
    params: Dict[str, torch.Tensor]
    num_datapoints: int  # If 0 the node is done
    node: str


@dataclass
class DataForPerfectNodeMessage:
    id: int
    iteration: int  # from which iteration is this data
    inputs: torch.Tensor
    targets: torch.Tensor
