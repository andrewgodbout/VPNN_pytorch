import torch

""" chebyshev by Bryn Gillcash.
implements the following coupled activation function sequencially pairwise.
f[r, theta] = [r/sqrt(M), M*theta] in polar coordinates.
Which is equivilent to

f[xi, xj] = [
sqrt((xi^2+xj^2)/M)*cos(M * acos(x/sqrt(xi^2+xj^2))),
sgn(xj) * sqrt((xi^2+xj^2)/M)*cos(M * asin(x/sqrt(xi^2+xj^2)))
]

cos(M * acos(x)) = T(M, x)
sin(M * acos(x)) = U(M-1, x)*sqrt(1-x**2)
T(M, x) is the Chebyshev polynomial of the 1st kind.
U(M, x) is the Chebyshev polynomial of the 2st kind.

see the following for reference:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.eval_chebyt.html
https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.eval_chebyu.html

This layer contains no trainable parameters.
As input, it takes M for the chebyshev function on initialization.
requires even number of layers

assumes data is of form *xL where L is a list of size nodes, * is any number of additional dimensions
"""


class Chebyshev(torch.nn.Module):
    def __init__(self, M=2):
        super().__init__()
        self.register_buffer('M', torch.tensor([float(M)]))

    def forward(self, inp):
        inp = Non_zero()(inp)
        return chebyshev()(inp, self.M)

    def init_ident(self):
        with torch.no_grad():
            self.M = torch.ones_like(self.M)
        return self

# prevents nan
class Non_zero(torch.autograd.Function):
    def forward(ctx, inp):
        # if  0 add 1e-7 to it
        offset = (inp == 0).float()*(1e-7)
        return inp + offset

    def backward(ctx, outp):
        return outp


class chebyshev(torch.autograd.Function):
    def forward(ctx, inp, M):
        
        indices = torch.tensor(range(inp.size()[-1])).reshape(-1,2).t()
        
        outp = torch.empty_like(inp)
        
        #reused indexing/computations
        xi = inp[..., indices[0]]  
        xj = inp[..., indices[1]]
        
        
        x_norm = torch.sqrt(xi**2 + xj**2) 

        # trig form, clamp input to acos to prevent edge case with floats
        M_angle = M * torch.acos((xi / x_norm).clamp(min=-1.,max=1.))
        chebyt_outp = torch.cos(M_angle)
        chebyu_outp = torch.sin(M_angle)
        
        # function implementation
        outp[...,indices[0]] = x_norm / torch.sqrt(M) * chebyt_outp
        outp[...,indices[1]] = xj.sign() * x_norm / torch.sqrt(M) * chebyu_outp
        
        ctx.save_for_backward(xi, xj, x_norm ** 2, M, indices, outp)
        return outp
     
    def backward(ctx, grad_L_y):
        xi, xj, x2_norm, M, indices, outp = ctx.saved_tensors
        #read grad_a_b as the derivitive of a w.r.t b

        # split function output
        yi = outp[..., indices[0]]
        yj = outp[..., indices[1]]
        
        # function gradient computation w.r.t. inputs
        grad_yi_xi = (xi * yi + M * xj * yj) / x2_norm
        grad_yj_xi = (-M * xj * yi + xi * yj) / x2_norm
        grad_yi_xj = (xj * yi + -M * xi * yj) / x2_norm
        grad_yj_xj = (M * xi * yi + xj * yj) / x2_norm
        
        # given gradients
        grad_L_yi = grad_L_y[..., indices[0]]
        grad_L_yj = grad_L_y[..., indices[1]]

        # chain rule
        grad_L_xi = grad_L_yi * grad_yi_xi + grad_L_yj * grad_yj_xi
        grad_L_xj = grad_L_yi * grad_yi_xj + grad_L_yj * grad_yj_xj
        
        # splice gradients together
        grad_L_x = torch.empty_like(grad_L_y) 
        
        grad_L_x[..., indices[0]] = grad_L_xi
        grad_L_x[..., indices[1]] = grad_L_xj
        return grad_L_x, None
