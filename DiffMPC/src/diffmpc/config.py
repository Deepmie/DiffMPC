from dataclasses import dataclass, field
from torch import Tensor
import torch
import torch.nn as nn
from typing import Tuple, Optional
from .system import QuadCost, BaseDynamic


@dataclass
class BaseConfig:
    state_dim: int                  = 3
    control_dim: int                = 4
    T: int                          = 5
    batch_size: int                 = 2
    u_upper: Optional[Tensor]       = None  # (T, b, u)
    u_lower: Optional[Tensor]       = None  # (T, b, u)
    u_zero_I: Optional[Tensor]      = None # (T, b, u)
    lqr_iters: int                  = 10
    eps_bound: float                = 1e-6
    eps_grad: float                 = 1e-6
    eps_best_cost: float            = 1e-4
    # lqr forward linesearch config #
    max_linesearch_iters: int       = 5
    linesearch_decay: float         = 0.9
    no_op_forward: bool             = False
    dynamics: Optional[BaseDynamic] = None
    cost: Optional[QuadCost]        = None
    dtype: torch.dtype              = torch.float32
    
    def get_controlconfig(self) -> Tuple[int, int, int, int]:
        return self.state_dim, self.control_dim, self.T, self.batch_size

    def get_lqrconfig(self) -> Tuple[Tensor, Tensor, Tensor, float, float, BaseDynamic, QuadCost]:
        return self.u_upper, self.u_lower, self.u_zero_I, self.eps_bound, self.eps_grad, self.dynamics, self.cost
