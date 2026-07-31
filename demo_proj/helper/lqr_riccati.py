import torch
from torch import Tensor
from typing import List, Tuple
from qpth.qp import QPFunction
from torch.autograd import Variable
from helper.system import Cost, Dynamic


def lqr_simplest():
    # make visual data
    state_dim: int = 4
    control_dim: int = 2
    T: int = 5

    Q: Tensor = torch.rand([T, state_dim, state_dim])
    R: Tensor = torch.rand([T, control_dim, control_dim])
    A: Tensor = torch.rand([T, state_dim, state_dim])
    B: Tensor = torch.rand([T, state_dim, control_dim])

    # construct C & F
    C: Tensor = torch.concat([
        torch.concat([Q, torch.zeros([T, state_dim, control_dim])], dim=2),
        torch.concat([torch.zeros([T, control_dim, state_dim]), R], dim=2),
    ], dim=1) # (T, c, c)

    F: Tensor = torch.concat([A, B], dim=2) # (T, s, c)
    P: Tensor = torch.zeros([state_dim, state_dim])

    Ps: List[Tensor] = list()
    Ks: List[Tensor] = list()

    for t in range(T-1, -1, -1):
        print(f'execute time {t+1}...')
        Ft = F[t, ...] # (s, c)
        H = C[t, ...] + Ft.T @ P @ Ft # (c, c)
        # split block
        H_xx = H[:state_dim, :state_dim] # (s, s)
        H_xu = H[:state_dim, state_dim:] # (s, u)
        H_ux = H[state_dim:, :state_dim] # (u, s)
        H_uu = H[state_dim:, state_dim:] # (u, u)

        # minimize this problem:
        # J = 1/2 * tau^T H tau
        # \partial J / \partial u = 0 ==> u = Kx = -H_uu^-1 * H_ux
        K = - H_uu.inverse() @ H_ux

        # update P
        P = H_xx + H_xu @ K + K.T @ H_ux + K.T @ H_uu @ K

        Ps.append(P)
        Ks.append(K)

    print(len(Ps))
    print(len(Ks))


def lqr_extra():
    # make visual data
    state_dim: int = 4
    control_dim: int = 2
    T: int = 5

    Q: Tensor = torch.rand([T, state_dim, state_dim])
    R: Tensor = torch.rand([T, control_dim, control_dim])
    q: Tensor = torch.rand([T, state_dim, 1])
    r: Tensor = torch.rand([T, control_dim, 1])
    A: Tensor = torch.rand([T, state_dim, state_dim])
    B: Tensor = torch.rand([T, state_dim, control_dim])

    # construct C & c & F
    C: Tensor = torch.concat([
        torch.concat([Q, torch.zeros([T, state_dim, control_dim])], dim=2),
        torch.concat([torch.zeros([T, control_dim, state_dim]), R], dim=2),
    ], dim=1) # (T, c, c)
    c: Tensor = torch.concat([q, r], dim=1) # (T, c, 1)
    F: Tensor = torch.concat([A, B], dim=2) # (T, s, c)
    f: Tensor = torch.rand([T, state_dim, 1])
    P: Tensor = torch.zeros([state_dim, state_dim])
    p: Tensor = torch.zeros([state_dim, 1])

    Ps: List[Tensor] = list()
    Ks: List[Tensor] = list()
    ks: List[Tensor] = list()

    for t in range(T-1, -1, -1):
        print(f'execute time {t+1}...')

        # Q-function
        Ft = F[t, ...] # (s, c)
        # quadratic
        H = C[t, ...] + Ft.T @ P @ Ft # (c, c)
        # split block
        H_xx = H[:state_dim, :state_dim] # (s, s)
        H_xu = H[:state_dim, state_dim:] # (s, u)
        H_ux = H[state_dim:, :state_dim] # (u, s)
        H_uu = H[state_dim:, state_dim:] # (u, u)

        # linear
        h = c[t, ...] + Ft.T @ (P @ f[t, ...] + p) # (c, 1)
        # split block
        h_x = h[:state_dim, :]
        h_u = h[state_dim:, :]

        # minimize this problem:
        # J = 1/2 * tau^T H tau + h^T * tau
        # \partial J / \partial u = 0 ==> u = Kx + k = -H_uu^-1 * H_ux * x + -H_uu^-1 * h_u 

        # common coding:
        # K = - H_uu.inverse() @ H_ux
        # k = - H_uu.inverse() @ h_u
        # use LU's trick:
        H_uu_lu, H_uu_pivots = torch.linalg.lu_factor(H_uu)
        K = -torch.linalg.lu_solve(H_uu_lu, H_uu_pivots, H_ux)
        k = -torch.linalg.lu_solve(H_uu_lu, H_uu_pivots, h_u)

        # update P
        P = H_xx + H_xu @ K + K.T @ H_ux + K.T @ H_uu @ K
        p = h_x + H_xu @ k + K.T @ h_u + K.T @ H_uu @ k

        Ps.append(P)
        Ks.append(K)
        ks.append(k)

    print(len(Ps))
    print(len(Ks))
    print(len(ks))


def lqr_extra_with_constraints():
    # make visual data
    state_dim: int = 4
    control_dim: int = 2
    T: int = 5

    Q: Tensor = torch.rand([T, state_dim, state_dim])
    R: Tensor = torch.rand([T, control_dim, control_dim])
    q: Tensor = torch.rand([T, state_dim, 1])
    r: Tensor = torch.rand([T, control_dim, 1])
    A: Tensor = torch.rand([T, state_dim, state_dim])
    B: Tensor = torch.rand([T, state_dim, control_dim])

    # construct C & c & F
    C: Tensor = torch.concat([
        torch.concat([Q, torch.zeros([T, state_dim, control_dim])], dim=2),
        torch.concat([torch.zeros([T, control_dim, state_dim]), R], dim=2),
    ], dim=1) # (T, c, c)
    c: Tensor = torch.concat([q, r], dim=1) # (T, c, 1)
    F: Tensor = torch.concat([A, B], dim=2) # (T, s, c)
    f: Tensor = torch.rand([T, state_dim, 1])
    P: Tensor = torch.zeros([state_dim, state_dim])
    p: Tensor = torch.zeros([state_dim, 1])
    u_lower: Tensor = torch.tensor([0.1, 0.1], dtype=torch.float32).reshape(-1, 1)
    u_upper: Tensor = torch.tensor([5.0, 5.0], dtype=torch.float32).reshape(-1, 1)

    Ps: List[Tensor] = list()
    Ks: List[Tensor] = list() # (T, u, s)
    ks: List[Tensor] = list() # (T, u, 1)

    for t in range(T-1, -1, -1):
        print(f'execute time {t+1}...')

        # Q-function
        Ft = F[t, ...] # (s, c)
        # quadratic
        H = C[t, ...] + Ft.T @ P @ Ft # (c, c)
        # split block
        H_xx = H[:state_dim, :state_dim] # (s, s)
        H_xu = H[:state_dim, state_dim:] # (s, u)
        H_ux = H[state_dim:, :state_dim] # (u, s)
        H_uu = H[state_dim:, state_dim:] # (u, u)

        # linear
        h = c[t, ...] + Ft.T @ (P @ f[t, ...] + p) # (c, 1)
        # split block
        h_x = h[:state_dim, :]
        h_u = h[state_dim:, :]
        
        # minimize this problem:
        # J = 1/2 * tau^T H_uu tau + h_u^T * tau, s.t. <constraints>
        # \partial J / \partial u = 0 ==> u = k 
        # box-constraint
        G = torch.concat([torch.eye(control_dim), -torch.eye(control_dim)], dim=0)
        h = torch.concat([u_upper, u_lower], dim=0)
        # eq-constraint
        e = Variable(torch.Tensor())
        k: Tensor = QPFunction(verbose=False)(H_uu, h_u.flatten(), G, h.flatten(), e, e)
        k = k.reshape(-1, 1) # (u, 1)

        # build free variable indicator
        eps_bound = 1e-6
        eps_grad  = 1e-6
        g: Tensor = H_uu @ k + h_u # (u, 1)
        Ic: Tensor = ((k <= u_lower + eps_bound) & (g > eps_grad)) | ((k >= u_upper - eps_bound) & (g < -eps_grad)) # (u, 1)
        If: Tensor = 1.0 - Ic.float() # (u, 1)
        notIff: Tensor = 1 - If @ If.T # (u, u)
        H_uu_free = H_uu.clone()
        H_uu_free[notIff.bool()] = 0.0

        H_ux_free = H_ux.clone()
        H_ux_free[If.expand(-1, state_dim).bool()] = 0.0

        # solve K
        # common coding:
        # K = - H_uu_free.inverse() @ H_ux_free (u, s)
        # use LU's trick:
        print(H_uu_free.shape)
        H_uu_free_lu, H_uu_free_pivots = torch.linalg.lu_factor(H_uu_free)
        K = -torch.linalg.lu_solve(H_uu_free_lu, H_uu_free_pivots, H_ux_free)

        # update P
        P = H_xx + H_xu @ K + K.T @ H_ux + K.T @ H_uu @ K
        p = h_x + H_xu @ k + K.T @ h_u + K.T @ H_uu @ k

        Ps.append(P)
        Ks.append(K)
        ks.append(k)

    print(len(Ps))
    print(len(Ks))
    print(len(ks))


def lqr_forward(Ks: List[Tensor], ks: List[Tensor]): # (u, s), (u, 1)
    # make visual data
    state_dim: int = 4
    control_dim: int = 2
    T: int = 5
    max_linesearch_iters: int = 10
    linesearch_decay: float = 0.8

    x: Tensor = torch.tensor([T+1, state_dim])
    u: Tensor = torch.tensor([T, control_dim])
    x_init: Tensor = torch.tensor([state_dim, 1])
    u_lower: Tensor = torch.tensor([0.1, 0.1], dtype=torch.float32).reshape(-1, 1)
    u_upper: Tensor = torch.tensor([5.0, 5.0], dtype=torch.float32).reshape(-1, 1)


    cost = Cost(state_dim, control_dim)
    dynamic = Dynamic(state_dim, control_dim)
    old_cost = cost.get_obj(x, u)
    current_cost = None
    alpha = torch.ones(1)

    i = 0
    while (current_cost is None or (old_cost is not None and current_cost > old_cost)) and i < max_linesearch_iters:
        u_new = []
        x_new = [x_init]
        dx = [torch.zeros_like(x_init)]
        objs = []

        for t in range(T):
            Kt = Ks[t]
            kt = ks[t]
            xt_new = x_new[t]
            xt = x[t]
            ut = u[t]
            dxt = dx[t]

            # key iteration formulation
            ut_new = Kt @ dxt + ut + kt # (u, 1)
            ut_new = torch.clamp(ut_new, u_lower, u_upper)
            u_new.append(ut_new)
            
            if t < T-1:
                xstept = dynamic.step(xt_new, ut_new)
                x_new.append(xstept)
                dx.append(xstept - x[t+1])

            obj = cost.step(xt_new, ut_new)
            objs.append(obj)

        objs = torch.stack(objs)
        current_cost = objs.sum(dim=0)

        if current_cost > old_cost: alpha *= linesearch_decay
        i += 1
        



if __name__ == '__main__':
    # lqr_simplest()
    # lqr_extra()
    lqr_extra_with_constraints()