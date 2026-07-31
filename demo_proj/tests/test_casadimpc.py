from diffmpc.casadi import MPC, BaseConfig
import numpy as np
from numpy import ndarray
import torch

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

    config: BaseConfig = BaseConfig(
        state_dim   = state_dim,
        control_dim = control_dim,
        T           = T,
        C           = C,
        c           = c,
        F           = F, 
        f           = f,
    )

    mpc: MPC = MPC(config)
    tau = mpc.solve(x_init)
    print(tau)


if __name__ == '__main__':
    test()