import warnings
warnings.filterwarnings('ignore')

from qpth.qp import QPFunction
import numpy as np
from numpy import ndarray
import torch
from torch import Tensor
from torch.autograd import Variable
from qpsolvers import solve_qp

Q = np.array([[2.0, 0.0],
              [0.0, 2.0]])

p = np.array([-2.0, -5.0])

G = np.array([[1.0, 2.0],
              [-1.0, 2.0],
              [-1.0, -2.0],
              [1.0, -2.0]])

h = np.array([3.0, 2.0, -2.0, 2.0])


def solve_normal(Q: ndarray, p: ndarray, G: ndarray, h: ndarray):
    x = solve_qp(Q, p, G, h, solver="osqp")
    print(x)

def solve_byqpth(Q: ndarray, p: ndarray, G: ndarray, h: ndarray):
    batch_size: int = 10
    # convert matrix from ndarray to Tensor
    Q: Tensor = torch.from_numpy(Q).float().requires_grad_(True)
    p: Tensor = torch.from_numpy(p).float().requires_grad_(True)
    G: Tensor = torch.from_numpy(G).float().requires_grad_(True)
    h: Tensor = torch.from_numpy(h).float().requires_grad_(True)
    e: Tensor = Variable(torch.Tensor())

    Q = Q.repeat(batch_size, 1, 1)
    p = p.repeat(batch_size, 1)
    G = G.repeat(batch_size, 1, 1)
    h = h.repeat(batch_size, 1)
    
    # print(Q.shape, p.shape, G.shape, h.shape)
    x = QPFunction(verbose=False)(Q, p, G, h, e, e)
    print(x.shape)
    print(x.grad_fn)

def main():
    print('normal solve qp:')
    solve_normal(Q, p, G, h)

    print('solve qp by qpth')
    solve_byqpth(Q, p, G, h)


if __name__ == '__main__':
    main()

