import torch

""" diagonal by Bryn Gillcash
implements a diagonal layer in PyTorch,
f(x) = d * x

where d is of the same size as the number of nodes n, and has elements:
d(i) = f(a(i)) / f(a(i-1)), 0 <= i <= n-1

where a for 0 <= i <= n-1 are randomly initialized trainable parameters
and f(a(-1)) = f(a(n-1)) 

and f is defined as
f = M * s(x/M) + M

where s a sigmoid function
and M is a constant

as input, takes the number of nodes and buffer M
This layer contains one tensor of trainable parameters

assumes data is of form *xL where L is a list of size nodes, * is any number of additional dimensions
"""


def sigmoid(x):
    return 1 / (1 + (-x).exp())


def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


class Diagonal(torch.nn.Module):
    def __init__(self, nodes, M=1):
        super().__init__()
        self.a = torch.nn.Parameter(M * torch.randn(nodes))
        self.register_buffer('M', torch.tensor([M]).float())

    def forward(self, inp):
        return diagonal()(inp, self.a, self.M)
    
    def get_diag(self):
        f = self.M * sigmoid(self.a / self.M) + self.M
        return f / f.roll(-1)
    
    def init_ident(self):
        with torch.no_grad():
            self.a.zero_()
        return self


class diagonal(torch.autograd.Function):
    def forward(ctx, inp, a, M):

        # computing diagonal elements from parameters
        f = M * sigmoid(a / M) + M
        # roll -1 is a(i+1), roll 1 is a(i-1)
        d = f / f.roll(-1)

        # function implementation
        outp = d * inp

        ctx.save_for_backward(inp, a, M, d)
        return outp

    def backward(ctx, grad_L_y):
        inp, a, M, d = ctx.saved_tensors

        # w.r.t. input
        grad_y_inp = d
        grad_L_inp = grad_L_y * grad_y_inp

        # w.r.t. parameters
        f = M * sigmoid(a / M) + M
        df = d_sigmoid(a / M)

        d1 = grad_L_y * inp * df / f.roll(-1)
        d2 = grad_L_y * inp.roll(1,dims=-1) * f.roll(1) / (f ** 2) * df
        grad_L_al = d1 - d2
        grad_L_a = grad_L_al.sum(dim=0)

        return grad_L_inp, grad_L_a, None, None
