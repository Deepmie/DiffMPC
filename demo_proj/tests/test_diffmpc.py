from diffmpc import MPC
from diffmpc.system import AffineDynamics, LinDynamics, QuadCost
from diffmpc.config import BaseConfig
from typing import Tuple, cast
import numpy as np
from numpy import ndarray
import torch
from torch import Tensor


def test():
    np.random.seed(0)
    torch.manual_seed(0)
    # ======== PARAM SETUP ========= #
    batch_size, state_dim, control_dim, T = 2, 3, 4, 5
    hidden_sizes = [10, 10]
    tau_dim = state_dim + control_dim

    C: ndarray = 10 * np.random.randn(T, batch_size, tau_dim, tau_dim).astype(np.float64) # (T, b, c, c)
    C = C.transpose(0, 1, 3, 2) @ C # (T, b, c, c)
    c = 10.*np.random.randn(T, batch_size, tau_dim).astype(np.float64) # (T, b, c)

    x_init: ndarray = np.random.randn(batch_size, state_dim).astype(np.float64) # (b, s)
    beta: int = 100.
    u_lower: ndarray = -beta * np.ones((T, batch_size, control_dim), dtype=np.float64) # (T, b, u)
    u_upper: ndarray = beta * np.ones((T, batch_size, control_dim), dtype=np.float64) # (T, b, u)

    F = np.random.randn(T-1, batch_size, state_dim, tau_dim) # (T-1, b, s, c)
    f = np.random.randn(T-1, batch_size, state_dim) # (T-1, b, s)

    _C, _c, _x_init, _u_lower, _u_upper, _F, _f = [torch.from_numpy(x).requires_grad_() for x in [C, c, x_init, u_lower, u_upper, F, f]]
    
    u_init = None
    config = BaseConfig(
        state_dim            = state_dim,
        control_dim          = control_dim,
        T                    = T,
        batch_size           = batch_size,
        u_upper              = _u_upper,
        u_lower              = _u_lower,
        lqr_iters            = 40,
        max_linesearch_iters = 2,
        dynamics             = LinDynamics(_F, _f),
        cost                 = QuadCost(_C, _c),
        dtype                = torch.float64,
    )
    mpc = MPC(config)
    x, u, costs = mpc(_x_init)
    x, u, costs = [cast(Tensor, e) for e in [x, u, costs]]

    # Test Grad
    u = u.reshape(-1)
    dl_dC = []
    dl_dc = []
    dl_dF = []
    dl_df = []
    for i in range(u.shape[0]):
        li = u[i]
        gradi = torch.autograd.grad(li, (_C, _c, _F, _f), retain_graph=True)
        dli_dC, dli_dc, dli_dF, dli_df  = [x.reshape(-1) for x in gradi]
        dl_dC.append(dli_dC); dl_dc.append(dli_dc); dl_dF.append(dli_dF); dl_df.append(dli_df)
    dl_dC = torch.stack(dl_dC).cpu().numpy()
    dl_dc = torch.stack(dl_dc).cpu().numpy()
    dl_dF = torch.stack(dl_dF).cpu().numpy()
    dl_df = torch.stack(dl_df).cpu().numpy()
    print('Finished...')




if __name__ == '__main__':
    test()