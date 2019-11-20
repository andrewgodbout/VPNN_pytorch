import torch

""" chebyshev_M_param by Bryn Gillcash.

see chebyshev_linked for full details

uses M as a trainable parameter

This layer contains one trainable parameter.
As input takes the number of input nodes and M for the chebyshev function on initialization.
requires even number of layers

assumes data is of form NxL where each N is a seperate peice of data, L is a list
"""


class Chebyshev_t(torch.nn.Module):
    def __init__(self, inp_size, M=2):
        super().__init__()
        self.M = torch.nn.Parameter(torch.tensor([float(M)]).repeat(int(inp_size/2)))

    def forward(self, inp):
        inp = Non_zero()(inp)
        return chebyshev_t()(inp, self.M)

    def init_ident(self):
        with torch.no_grad():
            self.M = torch.nn.Parameter(torch.ones_like(self.M))
        return self

# prevents nan
# 159 µs ± 2.27 µs for 10000 elements, find speedup
class Non_zero(torch.autograd.Function):
    def forward(ctx, inp):
        # if  0 add 1e-7 to it
        offset = (inp == 0).float()*(1e-7)
        return inp + offset

    def backward(ctx, outp):
        return outp


class chebyshev_t(torch.autograd.Function):
    def forward(ctx, inp, M):
        
        indices = torch.tensor(range(inp.size()[1])).reshape(-1,2).t()
        
        outp = torch.empty_like(inp)
        
        #reused indexing/computations
        xi = inp[:, indices[0]]  
        xj = inp[:, indices[1]]  
        x_norm = torch.sqrt(xi**2 + xj**2) 

        # trig form, clamp input to acos to prevent edge case with floats
        angle = torch.acos((xi / x_norm).clamp(min=-1.,max=1.))
        chebyt_outp = torch.cos(M * angle)
        chebyu_outp = torch.sin(M * angle)
        
        # function implementation
        outp[:,indices[0]] = x_norm / torch.sqrt(M) * chebyt_outp
        outp[:,indices[1]] = xj.sign() * x_norm / torch.sqrt(M) * chebyu_outp
        
        ctx.save_for_backward(xi, xj, x_norm ** 2, M, indices, outp, angle)
        
        return outp
     
    def backward(ctx, grad_L_y):
        xi, xj, x2_norm, M, indices, outp, angle = ctx.saved_tensors
        #read grad_a_b as the derivitive of a w.r.t b
        
        # split function output
        yi = outp[:, indices[0]]
        yj = outp[:, indices[1]]
        
        # function gradient computation w.r.t. inputs
        grad_yi_xi = (xi * yi + M * xj * yj) / x2_norm
        grad_yj_xi = (-M * xj * yi + xi * yj) / x2_norm
        grad_yi_xj = (xj * yi + -M * xi * yj) / x2_norm
        grad_yj_xj = (M * xi * yi + xj * yj) / x2_norm

        # given gradients
        grad_L_yi = grad_L_y[:, indices[0]]
        grad_L_yj = grad_L_y[:, indices[1]]

        # chain rule
        grad_L_xi = grad_L_yi * grad_yi_xi + grad_L_yj * grad_yj_xi
        grad_L_xj = grad_L_yi * grad_yi_xj + grad_L_yj * grad_yj_xj
        
        # splice gradients together
        grad_L_x = torch.empty_like(grad_L_y) 
        grad_L_x[:, indices[0]] = grad_L_xi
        grad_L_x[:, indices[1]] = grad_L_xj
        
        # w.r.t M
        grad_yi_Ml = -yi / (2 * M) + -yj * yj.sign() * angle
        grad_yj_Ml = yi * yj.sign() * angle + -yj / (2 * M)
        
        # chain rule
        grad_L_Ml = grad_L_yi * grad_yi_Ml + grad_L_yj * grad_yj_Ml 
        
        grad_L_M = (grad_L_Ml).sum(dim=0)
        return grad_L_x, grad_L_M
