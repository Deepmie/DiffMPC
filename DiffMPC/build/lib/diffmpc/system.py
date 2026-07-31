from torch import Tensor
import torch
import torch.nn as nn
from typing import Tuple, List
from abc import ABC, abstractmethod

class BaseDynamic(ABC, nn.Module):
    def __init__(self):
        super(BaseDynamic, self).__init__()
    
    @abstractmethod
    def forward(self, x: Tensor, u: Tensor) -> Tensor: # (b, s), (b, u)
        return torch.rand(x.shape) # (b, s)
    
    def step(self, u: Tensor, x_init: Tensor) -> Tensor: # (b, u), (b, s)
        x_next: Tensor = self.forward(x_init, u) # (b, s)
        return x_next

    def get_traj(self, u: Tensor, x_init: Tensor): # (T, b, u), (b, s)
        T = u.shape[0]; batch_size = u.shape[1]; state_dim = x_init.shape[1]
        x: Tensor = torch.zeros(T+1, batch_size, state_dim)
        x[0] = x_init
        for t in range(T-1): x[t+1] = self.step(u, x[t])
        return x # (T, b, u)
    
    @abstractmethod
    def grad_input(self, x: Tensor, u: Tensor):
        ...
    
    @abstractmethod
    def get_linear_params(self, x: Tensor, u: Tensor):
        ...


class AffineDynamics(BaseDynamic):
    def __init__(self, A: Tensor, B: Tensor, c: Tensor | None = None): # (s, s), (s, u), (1, s)
        super(AffineDynamics, self).__init__()
        self.A = A; self.B = B
        if c is None: c = 0.0
        if c.ndimension() == 1: c = c.unsqueeze(dim=0)
        self.c = c
    
    def forward(self, x: Tensor, u: Tensor) -> Tensor: # (b, s), (b, u)
        if x.ndimension() == 1: x = x.unsqueeze(dim=-1)
        if u.ndimension() == 1: u = u.unsqueeze(dim=-1)
        z: Tensor = x @ self.A.T + u @ self.B.T + self.c # (b, s)
        return z # (b, s)

    def grad_input(self, x: Tensor, u: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size: int = x.shape[0]
        A_, B_ = self.A.clone(), self.B.clone() # (b, s, s), (b, s, u)
        A_ = A_.unsqueeze(dim=0).repeat(batch_size, 1, 1) # (b, s, s)
        B_ = B_.unsqueeze(dim=0).repeat(batch_size, 1, 1) # (b, s, u)
        return A_, B_
    
    def get_linear_params(self, x: Tensor, u: Tensor) -> Tuple[Tensor, Tensor]: # (T, b, s), (T, b, u)
        T, batch_size, state_dim = x.shape
        control_dim = u.shape[-1]

        _x = x[:-1].reshape(-1, state_dim); _u = u[:-1].reshape(-1, control_dim) # (T-1*b, s), (T-1*b, u)
        _x_new: Tensor = self.forward(_x, _u) # (T-1*b, s)
        
        R, S = self.grad_input(_x, _u) # (T-1*b, s, s), (T-1*b, s, u)
        f: Tensor = _x_new - (R @ _x.unsqueeze(dim=-1)).squeeze() - (S @ _u.unsqueeze(dim=-1)).squeeze() # (T-1*b, s)
        f = f.reshape(T-1, batch_size, state_dim) # (T-1, b, s)

        R = R.reshape(T-1, batch_size, state_dim, state_dim) # (T-1, b, s, s)
        S = S.reshape(T-1, batch_size, state_dim, control_dim) # (T-1, b, s, u)
        F = torch.concat([R, S], dim=-1) # (T-1, b, s, c)
        return F, f # (T-1, b, s, c), (T-1, b, s)


class QuadCost:
    def __init__(self, C: Tensor, c: Tensor): # (T, b, c, c), (T, b, c)
        self.C = C; self.c = c
    
    def get_linear_params(self) -> Tuple[Tensor, Tensor]:
        return self.C, self.c

    def get_obj(self, x: Tensor, u: Tensor) -> Tensor: # (T, b, s), (T, b, u)
        tau: Tensor = torch.concat([x, u], dim=-1) # (T, b, c)
        objs: Tensor = 0.5 * (tau.unsqueeze(dim=2) @ self.C @ tau.unsqueeze(dim=-1)).squeeze() + (self.c.unsqueeze(dim=2) @ tau.squeeze(dim=-1)).squeeze() # (T, b)
        return objs.sum(dim=0) # (b, )


if __name__ == '__main__':
    batch_size: int = 2
    state_dim: int = 3
    control_dim: int = 4
    T = 5
    dynamics = BaseDynamic()

    x_init = torch.rand(batch_size, state_dim)
    u = torch.rand(T, batch_size, control_dim)

    x = dynamics.get_traj(u, x_init)
    print(x.shape)