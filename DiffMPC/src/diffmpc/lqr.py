from torch.autograd.function import Function, BackwardCFunction
from torch.autograd import Variable
from torch import Tensor
import torch
from typing import Tuple, Optional, cast
from dataclasses import dataclass
from .config import BaseConfig
from .system import LinDynamics, QuadCost
from qpth.qp import QPFunction

class LQRStepFunc(Function):
    @staticmethod
    def forward(
        ctx: BackwardCFunction,
        x_init: Tensor, # (b, s)
        C: Tensor, c: Tensor, # (T, b, c, c), (T, b, c)
        F: Tensor, f: Tensor, # (T-1, b, s, c), (T-1, b, s)
        # ==========no grad=========== #
        x_curr: Tensor, u_curr: Tensor, # (T, b, s), (T, b, u)
        config: BaseConfig,
        no_op_forward: bool=False,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        '''
        * LQR Step Problem:
        min J = min sum_t 1/2*taut^T*Ct*taut + ct^T*taut
        s.t. { x1 = x_init, xt+1 = Ft*taut + ft   // taut = [xt, ut]
        x_init, Ct, ct, Ft, ft --> LQRStep --> x, u (tau)

        * Input Dimensions:
        LQRParam:
        x_init: (b, s)
        C: (T, b, c, c),   c: (T, b, c)
        F: (T-1, b, s, c), f: (T-1, b, s)

        x_curr: (T, b, s), u_curr: (T, b, u)

        BaseConfig:
        u_upper: (u, 1), u_lower: (u, 1)

        * Return:
        x_new: (T, b, s), u_new: (T, b, u), costs: (b, )
        '''
        if no_op_forward:
            ctx.save_for_backward(x_init, C, c, F, f, x_curr, u_curr)
            ctx.config = config
            return x_curr, u_curr, None

        # pre process input control
        tau_curr: Tensor = torch.concat([x_curr, u_curr], dim=-1) # (T, b, c)
        c_back: Tensor = (C @ tau_curr.unsqueeze(dim=-1)).squeeze() + c # (T, b, c)
        f_back = None
        x_new, u_new, costs = LQRStepFunc._solve_lqr_by_riccati(
            ctx, 
            x_init, 
            C, c, 
            F, f, 
            c_back, f_back,
            x_curr, u_curr,
            config,
        )
        ctx.save_for_backward(x_init, C, c, F, f, x_new, u_new)
        ctx.config = config
        return x_new, u_new, costs
    
    @staticmethod
    def backward(
        ctx: BackwardCFunction,
        dl_dx: Tensor, dl_du: Tensor, # (T, b, s), (T, b, u)
        temp=None, # recieved costs grad
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        from . import MPC
        from .config import BaseConfig
        
        x_init, C, c, F, f, x_new, u_new = ctx.saved_tensors
        x_init, C, c, F, f, x_new, u_new = [cast(Tensor, v) for v in [x_init, C, c, F, f, x_new, u_new]]
        config: BaseConfig = ctx.config
        dtype: torch.dtype = config.dtype
        u_upper, u_lower, u_zero_I, eps_bound, eps_grad, dynamics, cost = config.get_lqrconfig()
        state_dim, control_dim, T, batch_size = config.get_controlconfig()
        r = torch.concat([dl_dx, dl_du], dim=-1) # (T, b, c)
        I = None if (u_lower is None or u_upper is None) else (torch.abs(u_new-u_lower) <= 1e-8) | (torch.abs(u_upper-u_new) <= 1e-8)
        dx_init: Tensor = torch.zeros_like(x_init) # (b, s)

        # construct a MPC
        cost_back: QuadCost = QuadCost(C, -r)
        dynamics_back: LinDynamics = LinDynamics(F, f=None)
        mpconfig: BaseConfig = BaseConfig(
            u_zero_I = I,
            dynamics = dynamics_back,
            cost     = cost_back,
            dtype    = config.dtype,
        )
        mpc: MPC = MPC(mpconfig)
        dx, du, _ = mpc(dx_init) # (T, b, s), (T, b, u)
        dtau = torch.concat([dx, du], dim=-1) # (T, b, c)
        tau  = torch.concat([x_new, u_new], dim=-1) # (T, b, c)

        # ========== solve grad ========== #
        def get_lambdas(C: Tensor, c: Tensor, x: Tensor, u: Tensor) -> Tensor:
            '''
            * Inputs:
            C: (T, b, c, c), c: (T, b, c), x: (T, b, s), u: (T, b, u)

            * Outputs:
            lambdas: (T, b, s)
            '''
            lambdas: Tensor = torch.zeros(T, batch_size, state_dim, dtype=dtype) # (T, b, s)
            for t in range(T-1, -1, -1):
                Ct_xx      = C[t, :, :state_dim, :state_dim] # (b, s, s)
                Ct_xu      = C[t, :, :state_dim, state_dim:] # (b, s, u)
                ct_x       = c[t, :, :state_dim]             # (b, s)
                xt         = x[t] # (b, s)
                ut         = u[t] # (b, u)
                lambdas[t] = (Ct_xx @ xt.unsqueeze(dim=-1)).squeeze() + (Ct_xu @ ut.unsqueeze(dim=-1)).squeeze() + ct_x # (b, s)
                if t < T-1:
                    Fxt = F[t, :, :, :state_dim] # (b, s, s)
                    lambdas[t] += (Fxt @ lambdas[t+1].unsqueeze(dim=-1)).squeeze()
            return lambdas # (T, b, s)

        lambdas: Tensor  = get_lambdas(C, c, x_new, u_new) # (T, b, s)
        dlambdas: Tensor = get_lambdas(C, -r, dx, du) # (T, b, s)

        dl_dC = -0.5 * (dtau.unsqueeze(dim=-1) @ tau.unsqueeze(dim=-2) + tau.unsqueeze(dim=-1) @ dtau.unsqueeze(dim=-2)) # (T, b, c, c)
        dl_dc = -dtau # (T, b, c)
        dl_dF = -(dlambdas[1:].unsqueeze(dim=-1) @ tau[:-1].unsqueeze(dim=-2) + lambdas[1:].unsqueeze(dim=-1) @ dtau[:-1].unsqueeze(dim=-2)) # (T-1, b, s, c)
        dl_df = -dlambdas[1:] # (T-1, b, s)
        dl_dx_init = -dlambdas[0] # (b, s)
        return dl_dx_init, dl_dC, dl_dc, dl_dF, dl_df, None, None, None, None

    @staticmethod
    def _solve_lqr_by_riccati(
        ctx: BackwardCFunction,
        x_init: Tensor, # (b, s)
        C: Tensor, c: Tensor, # (T, b, c, c), (T, b, c)
        F: Tensor, f: Tensor, # (T-1, b, s, c), (T-1, b, s)
        c_back: Tensor, f_back: Optional[Tensor], # (T, b, c), (T-1, b, s)
        x_curr: Tensor, u_curr: Tensor, # (T, b, s), (T, b, u)
        config: BaseConfig,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        T, batch_size, state_dim, control_dim = LQRStepFunc._get_param(x_init, C)
        u_upper, u_lower, u_zero_I, eps_bound, eps_grad, dynamics, cost = config.get_lqrconfig()
        dtype: torch.dtype = config.dtype

        P: Tensor = torch.zeros([batch_size, state_dim, state_dim], dtype=dtype) # (b, s, s)
        p: Tensor = torch.zeros([batch_size, state_dim, 1], dtype=dtype) # (b, s)
        Ks: Tensor = torch.zeros([T, batch_size, control_dim, state_dim], dtype=dtype) # (T, b, u, s)
        ks: Tensor = torch.zeros([T, batch_size, control_dim], dtype=dtype) # (T, b, u)
        alphas: Tensor = torch.ones([batch_size], dtype=dtype) # (b, )
        costs_prev: Tensor = cost.get_costs(x_curr, u_curr) # (b, )
        costs_curr: Tensor = float('inf') * torch.ones_like(costs_prev) # (b, )

        # =========== backward =========== #
        for t in range(T-1, -1, -1):
            # solve Q-function
            if t == T-1:
                Qt = C[t] # (b, c, c)
                qt = c_back[t] # (b, c)
            else:
                Ft = F[t] # (b, s, c)
                Qt = C[t] + Ft.permute(0, 2, 1) @ P @ Ft # (b, c, c)
                # qt = c_back[t] + (Ft.permute(0, 2, 1) @ (P @ f_back[t].unsqueeze(dim=-1) + p)).squeeze() # (b, c)
                qt = c_back[t] + (Ft.permute(0, 2, 1) @ p.unsqueeze(dim=-1)).squeeze() # (b, c)
                
            Qt_xx = Qt[:, :state_dim, :state_dim] # (b, s, s)
            Qt_xu = Qt[:, :state_dim, state_dim:] # (b, s, u)
            Qt_ux = Qt[:, state_dim:, :state_dim] # (b, u, s)
            Qt_uu = Qt[:, state_dim:, state_dim:] # (b, u, u)

            qt_x = qt[:, :state_dim] # (b, s)
            qt_u = qt[:, state_dim:] # (b, u)

            def _get_free_tensor(Qt_uu: Tensor, Qt_ux: Tensor, If: Tensor) -> Tuple[Tensor, Tensor]:
                # Qt_uu: (b, u, u), Qt_ux: (b, u, s), If: (b, u)
                notIff: Tensor = 1.0 - If.unsqueeze(dim=-1) @ If.unsqueeze(dim=1) # (b, u, u)
                Qt_uu_free = Qt_uu.clone() # (b, u, u)
                Qt_uu_free[notIff.bool()] = 0.0

                Qt_ux_free = Qt_ux.clone() # (b, u, s)
                Qt_ux_free[(1-If).unsqueeze(dim=-1).repeat(1, 1, state_dim).bool()] = 0.0
                return Qt_uu_free, Qt_ux_free # (b, u, u), (b, u, s)


            if u_upper is not None and u_lower is not None:
                # minimize a problem
                G = torch.concat([torch.eye(control_dim, dtype=dtype), -torch.eye(control_dim, dtype=dtype)], dim=0).repeat(batch_size, 1, 1) # (b, 2u, u)
                h = torch.concat([u_upper[t], -u_lower[t]], dim=-1) # (b, 2u)
                e = torch.tensor([], dtype=dtype)
                Qt_uu = Qt_uu + 1e-11 * torch.eye(control_dim, dtype=dtype)
                k: Tensor = QPFunction(verbose=False)(Qt_uu, qt_u, G, h, e, e) # (b, u)

                # build free variable indicator
                g: Tensor = (Qt_uu @ k.unsqueeze(dim=-1)).squeeze() + qt_u # (b, u)
                Ic: Tensor = ((k <= u_lower[t] + eps_bound) & (g > eps_grad)) | ((k >= u_upper[t] - eps_bound) & (g < -eps_grad)) # (b, u)
                If: Tensor = 1.0 - Ic.float() # (b, u)
                Qt_uu_free, Qt_ux_free = _get_free_tensor(Qt_uu, Qt_ux, If)

                # solve K
                Qt_uu_free_lu, Qt_uu_free_pivots = torch.linalg.lu_factor(Qt_uu_free)
                K: Tensor = -torch.linalg.lu_solve(Qt_uu_free_lu, Qt_uu_free_pivots, Qt_ux_free) # (b, u, s)
            else: # no constraints case
                if u_zero_I is not None:
                    Ic: Tensor = u_zero_I[t].clone().bool() # (b, u)
                    If: Tensor = 1.0 - Ic.float() # (b, u)
                    qt_u_free = qt_u.clone()
                    qt_u_free[Ic.bool()] = 0.0 # (b, u)
                    Qt_uu_free, Qt_ux_free = _get_free_tensor(Qt_uu, Qt_ux, If)
                else:
                    Qt_uu_free, Qt_ux_free, qt_u_free = Qt_uu.clone(), Qt_ux.clone(), qt_u.clone()
                
                # solve K, k
                Qt_uu_free_lu, Qt_uu_free_pivots = torch.linalg.lu_factor(Qt_uu_free)
                K: Tensor = -torch.linalg.lu_solve(Qt_uu_free_lu, Qt_uu_free_pivots, Qt_ux_free) # (b, u, s)
                k: Tensor = -torch.linalg.lu_solve(Qt_uu_free_lu, Qt_uu_free_pivots, qt_u_free.unsqueeze(dim=-1)).squeeze() # (b, u)

            # update P: (b, s, s), p: (b, s)
            P = Qt_xx + Qt_xu @ K + K.permute(0, 2, 1) @ Qt_ux + K.permute(0, 2, 1) @ Qt_uu @ K # (b, s, s)
            p = qt_x + (Qt_xu @ k.unsqueeze(dim=-1)).squeeze() + (K.permute(0, 2, 1) @ qt_u.unsqueeze(dim=-1)).squeeze() + (K.permute(0, 2, 1) @ Qt_uu @ k.unsqueeze(dim=-1)).squeeze() # (b, s)

            # record K, k
            Ks[t] = K
            ks[t] = k
        
        # ============ forward =========== #
        def is_curr_better(costs_curr: Tensor, costs_prev: Tensor) -> bool: # (b, ), (b, )
            return torch.any(costs_curr > costs_prev).cpu().item()

        i = 0
        while (costs_prev is not None and is_curr_better(costs_curr, costs_prev)) and i < config.max_linesearch_iters:
            u_new = torch.zeros(T, batch_size, control_dim, dtype=dtype) # (T, b, u)
            x_new = torch.zeros(T, batch_size, state_dim, dtype=dtype)   # (T, b, s)
            dx    = torch.zeros(T, batch_size, state_dim, dtype=dtype)   # (T, b, s)

            x_new[0] = x_init
            # key iteration formulation
            for t in range(T):
                ut_new = (Ks[t] @ dx[t].unsqueeze(dim=-1)).squeeze() + u_curr[t] + ks[t] # (b, u)
                if u_zero_I is not None: ut_new[u_zero_I[t]] = 0.0 # (b, u)
                if u_upper is not None and u_lower is not None: ut_new = torch.clamp(ut_new, u_lower[t], u_upper[t])
                
                if t < T-1:
                    x_new[t+1] = dynamics(x_new[t], ut_new, t) # Dynamic Recursive
                    dx[t+1] = x_new[t+1] - x_curr[t+1] # (b, s)
                u_new[t] = ut_new
            
            # recompute cost
            objs_curr = cost.get_objs(x_new, u_new) # (T, b)
            costs_curr = objs_curr.sum(dim=0) # (b, )

            # update alphas, i
            alphas[costs_curr > costs_prev] *= config.linesearch_decay # (b, )
            i += 1
        
        return x_new, u_new, costs_curr, # (T, b, s), (T, b, u), (b, )


    @staticmethod
    def _get_param(x_init, C) -> Tuple[int, int, int, int]:
        '''
        Return T, batch_size, state_dim, control_dim
        '''
        tau_dim: int = C.shape[-1]
        state_dim: int = x_init.shape[-1]
        control_dim: int = tau_dim - state_dim
        return C.shape[0], x_init.shape[0], state_dim, control_dim