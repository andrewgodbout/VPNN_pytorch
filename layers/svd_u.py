import torch
""" svd_u layer by Bryn Gillcash
implements a fixed matrix multiply by a submatrix of a hadamard matrix in PyTorch,
f(x) = x.mm(U)

where U is the u result of svd on some random input_siz x outp_size matrix

intended to act as a fixed downsizer without having trainable parameters

as input, takes inp_size, outp_size
This layer contains no trainable parameters
input_size must be greater then outp_size
assumes data is of form nxl
"""

class Svd_u(torch.nn.Module):
    def __init__(self, inp_size, outp_size):
        super().__init__()
        x=torch.randn(inp_size, outp_size)
        U=x.svd()[0]
        self.register_buffer('U', U) 
        
    def forward(self, inp):
        return inp.mm(self.U)
