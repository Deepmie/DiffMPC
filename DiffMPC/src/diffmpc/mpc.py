import torch
from torch import Tensor
import torch.nn as nn
from .config import BaseConfig
from .lqr import LQRStepFunc
from .system import AffineDynamics, BaseDynamic, QuadCost
import numpy as np
from typing import Tuple

class MPC(nn.Module):
    def __init__(self, config: BaseConfig):
        super(MPC, self).__init__()
        self._state_dim: int               = config.state_dim
        self._control_dim: int             = config.control_dim
        self._T: int                       = config.T
        self._dynamics: AffineDynamics     = config.dynamics
        self._cost: QuadCost               = config.cost
        self._u_upper, self._u_lower       = config.u_upper, config.u_lower
        self.config: BaseConfig            = config

    
    def forward(self, x_init: Tensor) -> Tuple[Tensor, Tensor, Tensor]: # (b, s)
        batch_size: int = x_init.shape[0]
        
        # init u
        u: Tensor = torch.zeros(self.config.T, batch_size, self.config.control_dim).to(self.config.dtype) # (T, b, u)

        best = None
        n_not_improved: int = 0
        for i in range(self.config.lqr_iters):
            u = u.detach().requires_grad_(True) # (T, b, u)
            x = self._dynamics.get_traj(u, x_init) # (T, b, s)
            F, f = self._dynamics.get_linear_params(x, u) # (T-1, b, s, c), (T-1, b, s)
            C, c = self._cost.get_linear_params() # (T, b, c, c), (T, b, c)
            x, u, costs = self._solve_lqr_subproblem(x_init, C, c, F, f, x, u)
            n_not_improved += 1

            if not best:
                best = {
                    'x': x.clone(), # (T, b, s)
                    'u': u.clone(), # (T, b, u)
                    'costs': costs, # (b, )
                }
            else:
                for j in range(batch_size):
                    if costs[j] <= best['costs'][j] + self.config.eps_best_cost:
                        n_not_improved = 0
                        best['x'][:, j] = x[:, j]
                        best['u'][:, j] = u[:, j]
                        best['costs'][j] = costs[j]

        x, u, costs = best['x'], best['u'], best['costs']
        F, f = self._dynamics.get_linear_params(x, u)
        C, c = self._cost.get_linear_params()
        x, u, _ = self._solve_lqr_subproblem(x_init, C, c, F, f, x, u, no_op_forward=True)
        return x, u, costs # (T, b, s), (T, b, u), (b, )
    
    def _solve_lqr_subproblem(
            self, x_init: Tensor,
            C: Tensor, c: Tensor,
            F: Tensor, f: Tensor,
            x: Tensor, u: Tensor,
            no_op_forward: bool=False,
        ) -> Tuple[Tensor, Tensor]:
        x_new, u_new, costs = LQRStepFunc.apply(x_init, C, c, F, f, x, u, self.config, no_op_forward)
        return x_new, u_new, costs # (T, b, s), (T, b, u), (b, )
