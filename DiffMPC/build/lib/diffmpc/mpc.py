import torch
from torch import Tensor
from .config import MPConfig, LQRBaseConfig
from .lqr import LQRStepFunc
from .system import AffineDynamics, BaseDynamic, QuadCost
import numpy as np
from typing import Tuple

class MPC:
    def __init__(self, config: MPConfig):
        self._state_dim: int               = config.state_dim
        self._control_dim: int             = config.control_dim
        self._T: int                       = config.T
        self._dynamics: AffineDynamics     = config.dynamics
        self._cost: QuadCost               = config.cost
        self._u_upper, self._u_lower       = config.u_upper, config.u_lower
        self._lqrbaseconfig: LQRBaseConfig = LQRBaseConfig(self._u_upper, self._u_lower)
        self.config: MPConfig              = config

    
    def forward(self, x_init: Tensor): # (b, s)
        batch_size: int = x_init.shape[0]

        # init u
        u: Tensor = torch.rand(self.T, batch_size, self.control_dim) # (T, b, u)
        
        for i in range(self.config.lqr_iters):
            u = u.detach().requires_grad_(True) # (T, b, u)
            x = self._dynamics.get_traj(u, x_init) # (T, b, s)
            F, f = self._dynamics.get_linear_params(x, u) # (T-1, b, s, c), (T-1, b, s)
            C, c = self._cost.get_linear_params() # (T, b, c, c), (T, b, c)
            self._solve_lqr_subproblem(x_init, C, c, F, f, x, u)
    
    def _solve_lqr_subproblem(
            self, x_init: Tensor,
            C: Tensor, c: Tensor,
            F: Tensor, f: Tensor,
            x: Tensor, u: Tensor,
        ):
        LQRStepFunc.apply(x_init, C, c, F, f, x, u, self._lqrbaseconfig)
