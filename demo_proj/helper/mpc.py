import torch
from torch import Tensor
from helper.type import MPConfig, LQRBaseConfig
from helper.lqr import LQRStepFunc
from helper.system import AffineDynamics, BaseDynamic, QuadCost
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
        self._u_upper = torch.tensor(self._u_upper).unsqueeze(dim=-1)
        self._u_lower = torch.tensor(self._u_lower).unsqueeze(dim=-1)
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


if __name__ == '__main__':
    # ======== PARAM SETUP ========= #
    state_dim: int = 3
    control_dim: int = 4
    T: int = 5
    batch_size: int = 2
    u_upper: Tuple = (1.0, 1.0)
    u_lower: Tuple = (0.1, 0.1)
    alpha: float = 0.2
    tau_dim: int = state_dim + control_dim

    # 
    np.random.seed(1)
    C = np.random.randn(T, batch_size, tau_dim, tau_dim) # (T, b, c, c)
    C = np.matmul(C.transpose(0, 1, 3, 2), C) # (T, b, c, c)
    c = np.random.randn(T, batch_size, tau_dim) # (T, b, c)
    R = np.tile(np.eye(state_dim) + alpha * np.random.randn(state_dim, state_dim), reps=(T, batch_size, 1, 1))
    S = np.tile(np.random.randn(state_dim, control_dim), reps=(T, batch_size, 1, 1))
    F = np.concatenate((R, S), axis=3)
    f = np.tile(np.random.randn(state_dim), reps=(T, batch_size, 1))
    x_init = np.random.randn(batch_size, state_dim)
    u_lower = -1e4 * np.ones((T, batch_size, control_dim))
    u_upper = 1e4 * np.ones((T, batch_size, control_dim))

    C, c, R, S, F, f, x_init, u_lower, u_upper = [
        torch.from_numpy(x).double() if x is not None else None
        for x in [C, c, R, S, F, f, x_init, u_lower, u_upper]
    ]
    dynamics = AffineDynamics(R[0, 0], S[0, 0], f[0, 0])
    cost = QuadCost(C, c)

    # set mpc config
    mpconfig = MPConfig(
        state_dim,
        control_dim,
        T,
        u_upper=u_upper,
        u_lower=u_lower,
        dynamics=dynamics,
        cost=cost,
    )
    mpc = MPC(mpconfig)