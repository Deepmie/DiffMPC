import casadi as ca
from .config import BaseConfig

class MPC:
    def __init__(self, config: BaseConfig):
        self.config = config
        self.state_dim, self.control_dim, self.T = self.config.state_dim, self.config.control_dim, self.config.T
        self.tau_dim = self.state_dim + self.control_dim
        self.Cn, self.cn, self.Fn, self.fn = self.config.get_parameters()
        self._build_solver()

    def _build_solver(self):
        self.opti = ca.Opti()
        self.Tau  = self.opti.variable(self.tau_dim * self.T)
        self.x_init = self.opti.parameter(self.state_dim)
        C = []; c = []; F = []; f = []
        
        for t in range(self.T):
            C.append(self.opti.parameter(self.tau_dim, self.tau_dim))
            c.append(self.opti.parameter(self.tau_dim))
            F.append(self.opti.parameter(self.state_dim, self.tau_dim))
            f.append(self.opti.parameter(self.state_dim))
        self.C = C; self.c = c; self.F = F; self.f = f

        cost = 0
        for t in range(self.T):
            tau = self.Tau[t * self.tau_dim: (t+1) * self.tau_dim]
            cost += 0.5 * ca.mtimes([tau.T, C[t], tau]) + ca.dot(c[t], tau)
        self.opti.minimize(cost)

        x = self.x_init
        self.X = []

        for t in range(self.T):
            tau = self.Tau[t * self.tau_dim: (t+1) * self.tau_dim]
            x_next = F[t] @ tau + f[t]
            self.X.append(x_next)
            x = x_next

        opts = {'print_time': False, 'ipopt.print_level': 0}
        self.opti.solver('ipopt', opts)
    
    def solve(self, x_init):
        self.opti.set_value(self.x_init, x_init)

        for t in range(self.T):
            self.opti.set_value(self.C[t], self.Cn[t])
            self.opti.set_value(self.c[t], self.cn[t])
            self.opti.set_value(self.F[t], self.Fn[t])
            self.opti.set_value(self.f[t], self.fn[t])

        sol = self.opti.solve()
        tau_opt = sol.value(self.Tau)
        return tau_opt
            
        