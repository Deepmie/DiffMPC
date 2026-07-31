from dataclasses import dataclass, field
from torch import Tensor
import torch.nn as nn
from typing import Tuple, Optional
from helper.system import QuadCost

@dataclass
class MPConfig:
    state_dim: int = 3
    control_dim: int = 4
    T: int = 5
    u_upper: Tuple = (1.0, 1.0)
    u_lower: Tuple = (0.1, 0.1)
    lqr_iters: int = 10
    dynamics: Optional[nn.Module] = None
    cost: Optional[QuadCost] = None

@dataclass
class LQRBaseConfig:
    u_upper: Tensor # (u, 1)
    u_lower: Tensor # (u, 1)
    eps_bound: float = 1e-6
    eps_grad: float = 1e-6
    max_linesearch_iters: int = 5
    linesearch_decay: float = 0.9
    dynamic: Optional[nn.Module] = None
    cost: Optional[QuadCost] = None

    def get(self) -> Tuple[Tensor, Tensor, float, float, nn.Module, QuadCost]:
        return self.u_upper, self.u_lower, self.eps_bound, self.eps_grad, self.dynamic, self.cost
