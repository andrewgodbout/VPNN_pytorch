import torch
from layers.diagonal import Diagonal
from layers.chebyshev import Chebyshev
from layers.svd_u import Svd_u

class Mixed1(torch.nn.Module):
    def __init__(self, inp_size, outp_size, hidden_layers, chebyshev_M=2, svd_u=False):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        
        for i in range(hidden_layers):
            self.layers.append(torch.nn.Linear(inp_size, inp_size))
            self.layers.append(Chebyshev(chebyshev_M))
            
        # output layer
        if not svd_u:
            self.layers.append(torch.nn.Linear(inp_size, outp_size))
        else:
            self.layers.append(Svd_u(inp_size, outp_size))
            
        
    def forward(self, inp):
        for layer in self.layers:
            inp = layer(inp)
        return inp
