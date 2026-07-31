from torch.autograd.function import Function, BackwardCFunction
from torch.autograd import Variable
from torch import Tensor
import torch
from typing import Tuple, Optional, cast
from dataclasses import dataclass
from helper.type import LQRBaseConfig
from helper.mpc import MPC
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
        lqrbaseconfig: LQRBaseConfig,
    ) -> Tuple[Tensor]:
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

        LQRBaseConfig:
        u_upper: (u, 1), u_lower: (u, 1)
        '''
        # pre process input control
        if u_upper.ndimension() == 1: u_upper = u_upper.unsqueeze(dim=-1)
        if u_lower.ndimension() == 1: u_lower = u_lower.unsqueeze(dim=-1)
        
        tau_curr: Tensor = torch.concat([x_curr, u_curr], dim=-1) # (T, b, c)
        c_back: Tensor = (C @ tau_curr.unsqueeze(dim=-1)).squeeze() + c # (T, b, c)
        f_back = None
        x_new, u_new = LQRStepFunc._solve_lqr_by_riccati(
            ctx, 
            x_init, 
            C, c, 
            F, f, 
            c_back, f_back,
            x_curr, u_curr,
            lqrbaseconfig,
        )
        ctx.save_for_backward(x_init, C, c, F, f, x_new, u_new)
        return x_new, u_new
    
    @staticmethod
    def backward(
        ctx: BackwardCFunction,
        dl_dx: Tensor, dl_du: Tensor, # (T, b, s), (T, b, u)
    ) -> Tuple[Tensor]:
        x_init, C, c, F, f, x_new, u_new = ctx.saved_tensors
        x_init, C, c, F, f, x_new, u_new = [cast(Tensor, v) for v in [x_init, C, c, F, f, x_new, u_new]]
        r = torch.concat([dl_dx, dl_du], dim=-1) # (T, b, c)
        

        dl_dx_init, dl_dC, dl_dc, dl_dF, dl_df = ()
        return dl_dx_init, dl_dC, dl_dc, dl_dF, dl_df

    @staticmethod
    def _solve_lqr_by_riccati(
        ctx: BackwardCFunction, 
        x_init: Tensor, # (b, s)
        C: Tensor, c: Tensor, # (T, b, c, c), (T, b, c)
        F: Tensor, f: Tensor, # (T-1, b, s, c), (T-1, b, s)
        c_back: Tensor, f_back: Optional[Tensor], # (T, b, c), (T-1, b, s)
        x_curr: Tensor, u_curr: Tensor, # (T, b, s), (T, b, u)
        lqrbaseconfig: LQRBaseConfig,
    ) -> Tuple[Tensor]:
        T, batch_size, state_dim, control_dim = LQRStepFunc._get_param(x_init, C)
        u_upper, u_lower, eps_bound, eps_grad, dynamic, cost = lqrbaseconfig.get()
        P: Tensor = torch.zeros([batch_size, state_dim, state_dim]) # (b, s, s)
        p: Tensor = torch.zeros([batch_size, state_dim, 1]) # (b, s, 1)
        Ks: Tensor = torch.zeros([T, batch_size, control_dim, state_dim]) # (T, b, u, s)
        ks: Tensor = torch.zeros([T, batch_size, control_dim]) # (T, b, u)
        alphas: Tensor = torch.ones([batch_size]) # (b, )
        cost_prev: Tensor = cost.get_obj(x_curr, u_curr) # (b, )
        cost_curr: Optional[Tensor] = None

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
                qt = c_back[t] + (Ft.permute(0, 2, 1) @ p).squeeze() # (b, c)
                
            Qt_xx = Qt[:, :state_dim, :state_dim] # (b, s, s)
            Qt_xu = Qt[:, :state_dim, state_dim:] # (b, s, u)
            Qt_ux = Qt[:, state_dim:, :state_dim] # (b, u, s)
            Qt_uu = Qt[:, state_dim:, state_dim:] # (b, u, u)

            qt_x = qt[:, :state_dim] # (b, s)
            qt_u = qt[:, state_dim:] # (b, u)

            # minimize a problem
            G = torch.concat([torch.eye(control_dim), -torch.eye(control_dim)], dim=0).repeat(batch_size, 1, 1) # (b, 2u, u)
            h = torch.concat([u_upper, u_lower], dim=0).flatten().repeat(batch_size, 1) # (b, 2u)
            e = Variable(torch.Tensor())
            k: Tensor = QPFunction(verbose=False)(Qt_uu, qt_u, G, h, e, e) # (b, u)

            # build free variable indicator
            g: Tensor = (Qt_uu @ k.unsqueeze(dim=-1)).squeeze() + qt_u # (b, u)
            Ic: Tensor = ((k <= u_lower + eps_bound) & (g > eps_grad)) | ((k >= u_upper - eps_bound) & (g < -eps_grad)) # (b, u)
            If: Tensor = 1.0 - Ic.float() # (b, u)
            notIff: Tensor = 1.0 - If.unsqueeze(dim=-1) @ If.unsqueeze(dim=1) # (b, u, u)
            Qt_uu_free = Qt_uu.clone() # (b, u, u)
            Qt_uu_free[notIff.bool()] = 0.0

            Qt_ux_free = Qt_ux.clone() # (b, u, s)
            Qt_ux_free[If.repeat(1, 1, state_dim).bool()] = 0.0

            # solve K
            Qt_uu_free_lu, Qt_uu_free_pivots = torch.linalg.lu_factor(Qt_uu_free)
            K: Tensor = -torch.linalg.lu_solve(Qt_uu_free_lu, Qt_uu_free_pivots, Qt_ux_free) # (b, u, s)

            # update P: (b, s, s), p: (b, s)
            P = Qt_xx + Qt_xu @ K + K.permute(0, 2, 1) @ Qt_ux + K.permute(0, 2, 1) @ Qt_uu @ K # (b, s, s)
            p = qt_x + (Qt_xu @ k.unsqueeze(dim=-1)).squeeze() + (K.permute(0, 2, 1) @ qt_u.unsqueeze(dim=-1)).squeeze() + (K.permute(0, 2, 1) @ Qt_uu @ k.unsqueeze(dim=-1)).squeeze() # (b, s)

            # record K, k
            Ks[T-1-t] = K
            ks[T-1-t] = k
        
        # ============ forward =========== #
        i = 0
        while (cost_curr is None or (cost_prev is not None and cost_curr > cost_prev) and i < lqrbaseconfig.max_linesearch_iters):
            u_new = torch.zeros(T, batch_size, control_dim) # (T, b, u)
            x_new = torch.zeros(T, batch_size, state_dim)   # (T, b, s)
            dx    = torch.zeros(T, batch_size, state_dim)   # (T, b, s)

            x_new[0] = x_init
            # key iteration formulation
            for t in range(T):
                u_new[t] = (Ks[t] @ dx[t].unsqueeze(dim=-1)).squeeze() + u_curr[t] + ks[t] # (b, u)
                if t < T-1:
                    x_new[t+1] = dynamic(x_new[t], u_new[t]) # Dynamic Recursive
                    dx[t+1] = x_new[t+1] - x_curr[t+1] # (b, s)
            
            # recompute cost
            cost_curr = cost.get_obj(x_new, u_new) # (b, )

            # update alphas, i
            alphas[cost_curr > cost_prev] *= lqrbaseconfig.linesearch_decay # (b, )
            i += 1
        
        return x_new, u_new # (T, b, s), (T, b, u)


    @staticmethod
    def _get_param(x_init, C) -> Tuple[int, int, int, int]:
        '''
        Return T, batch_size, state_dim, control_dim
        '''
        tau_dim: int = C.shape[-1]
        state_dim: int = x_init.shape[-1]
        control_dim: int = tau_dim - state_dim
        return C.shape[0], x_init.shape[0], state_dim, control_dim