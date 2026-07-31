import torch
from torch import Tensor
from torch.autograd.function import Function, BackwardCFunction
from typing import Tuple

class SquareFunction(Function):
    @staticmethod
    def forward(ctx: BackwardCFunction, x: Tensor):
        ctx.save_for_backward(x)
        return x ** 2

    @staticmethod
    def backward(ctx: BackwardCFunction, grad_output: Tensor):
        print(grad_output)
        x, = ctx.saved_tensors
        return grad_output * 2 * x

class MyFunction2D(Function):
    @staticmethod
    def forward(ctx: BackwardCFunction, x: Tensor) -> Tuple[Tensor]:
        ctx.save_for_backward(x)
        y1, y2 = x ** 2, x.exp()
        return y1, y2

    @staticmethod
    def backward(ctx: BackwardCFunction, grad_y1: Tensor, grad_y2: Tensor) -> Tensor:
        x, = ctx.saved_tensors
        return grad_y1 * 2 * x + grad_y2 * x.exp()

class MyFunction4D(Function):
    @staticmethod
    def forward(ctx: BackwardCFunction, x1: Tensor, x2: Tensor) -> Tuple[Tensor]:
        ctx.save_for_backward(x1, x2)
        y1 = x1 ** 2
        y2 = x2.exp()
        return y1, y2
    
    @staticmethod
    def backward(ctx: BackwardCFunction, dl_dy1: Tensor, dl_dy2: Tensor):
        x1, x2, = ctx.saved_tensors
        dl_dx1 = dl_dy1 * (2 * x1)
        dl_dx2 = dl_dy2 * (x2.exp())
        return dl_dx1, dl_dx2


def main():
    x1 = torch.tensor(3.0, requires_grad=True)
    x2 = torch.tensor(5.0, requires_grad=True)
    # y = SquareFunction.apply(x)
    y1, y2 = MyFunction4D.apply(x1, x2)
    loss = y1.sum() + y2.sum()
    loss.backward()
    
    print(x1.grad, x2.grad)

if __name__ == '__main__':
    main()