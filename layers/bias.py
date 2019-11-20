import torch

""" bias layer by Bryn Gillcash
implements a bias layer in PyTorch, elementwise
f(x) = x + b

as input, takes the number of nodes.
This layer contains one tensor of trainable parameters

applies bias to innermost dimension
"""


class Bias(torch.nn.Module):
    def __init__(self, nodes):
        super().__init__()
        self.bias = torch.randn(nodes)
        self.bias = torch.nn.Parameter(
              self.bias / torch.sqrt(torch.tensor([nodes], dtype=torch.float))
        )
        self.bias.requires_grad_()

    def forward(self, inp):
        return bias()(inp, self.bias)

    def init_ident(self):
        with torch.no_grad():
            self.bias.zero_()
        return self

class bias(torch.autograd.Function):
    def forward(ctx, inp, bias):
        return inp + bias

    def backward(ctx, grad_L_y):
        # sum over all but dim -1
        if grad_L_y.dim() != 1:  # sum has undesirable behaviour for dim=()
            grad_L_bias = grad_L_y.sum(dim=tuple(range(grad_L_y.dim()-1)))
        else:
            grad_L_bias = grad_L_y
        return grad_L_y, grad_L_bias
