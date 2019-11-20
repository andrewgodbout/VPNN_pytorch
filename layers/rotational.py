import torch


""" Rotational layer Module for PyTorch by Bryn Gillcash
implements [yi, yj] = f[xi, xj] = [xi*cos(a) - xj*sin(a), xj*cos(a) + xi*sin(a)]
applies to pairs of inputs with trainable parameter a
randomly pairs up all indices of the data, nonoverlaping
must have even number of nodes

assumes data is of form *xL where L is a list of size nodes, * is any number of additional dimensions
"""

class Rotational(torch.nn.Module):
    def __init__(self, nodes):
        super().__init__()
        # number of nodes in layer
        self.nodes = nodes
        # trainable angle parameters
        self.angles = torch.nn.Parameter(torch.randn(nodes//2, requires_grad=True))
        # indices to pair off inputs
        self.register_buffer('pairs', torch.randperm(nodes).reshape(-1, 2))
        self.register_buffer('outp_pairs', torch.randperm(nodes).reshape(-1, 2))    
 

    def forward(self, inp):
        return rotate()(inp, self.angles, self.pairs, 
                     self.outp_pairs)
    
    # set all angles to zero, returns same Rotational layer
    def init_ident(self):
        with torch.no_grad():
            self.angles.zero_()
        return self

# rotate function for Rotational layer
# implements function and backpropogation
class rotate(torch.autograd.Function):
    def forward(ctx, inp, angles, pairs, outp_pairs):
        c = torch.cos(angles)
        s = torch.sin(angles)
        outp = inp.clone()
        
        # spliting input into two
        xi = inp[..., pairs[:, 0]]
        xj = inp[..., pairs[:, 1]]
              
        # function implementation
        yi = c * xi - s * xj
        yj = c * xj + s * xi
        
        # splicing together outputs
        outp[..., outp_pairs[:, 0]] = yi
        outp[..., outp_pairs[:, 1]] = yj

        
        ctx.save_for_backward(xi, xj, pairs, outp_pairs, c, s)
        return outp

    def backward(ctx, grad_L_y):
        xi, xj, pairs, outp_pairs, c, s = ctx.saved_tensors

        # read: grad_a_b as the partial derivative of a with respect to b
        grad_L_yi = grad_L_y[..., outp_pairs[:, 0]]
        grad_L_yj = grad_L_y[..., outp_pairs[:, 1]]

        # computes gradients of input data w.r.t. output
        grad_L_x = torch.empty_like(grad_L_y)

        grad_yi_xi = c
        grad_yi_xj = -s
        grad_yj_xi = s
        grad_yj_xj = c
        
        # chain rule
        grad_L_xi = grad_L_yi * grad_yi_xi + grad_L_yj * grad_yj_xi
        grad_L_xj = grad_L_yi * grad_yi_xj + grad_L_yj * grad_yj_xj
        
        # splicing together
        grad_L_x[..., pairs[:, 0]] = grad_L_xi
        grad_L_x[..., pairs[:, 1]] = grad_L_xj
        
        # computes gradients of angles w.r.t. output
        # grad_L_al is the partial derivative of the loss w.r.t the angle from
        # datum l
        grad_L_al = torch.empty(*grad_L_y.shape[:-1], grad_L_y.shape[-1] // 2)
        #grad_L_al = torch.empty(grad_L_y.size()[0], grad_L_y.size()[1] // 2)
        
        grad_yi_al = -s * xi - c * xj
        grad_yj_al = -s * xj + c * xi
        
        # chain rule
        grad_L_al = grad_L_yi * grad_yi_al + grad_L_yj * grad_yj_al
        # keep dimension -1
        if grad_L_al.dim() != 1:  # sum doesn't give desired behaviour in case dim=()
            grad_L_a = grad_L_al.sum(dim=tuple(range(grad_L_al.dim()-1)))
        else:
            grad_L_a = grad_L_al
        # pairs doesn't need gradient
        return grad_L_x, grad_L_a, None, None, None
