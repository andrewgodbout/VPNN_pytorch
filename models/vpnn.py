import torch
from layers.diagonal import Diagonal
from layers.chebyshev import Chebyshev
from layers.rotational import Rotational
from layers.bias import Bias
from layers.svd_u import Svd_u

class Vpnn(torch.nn.Module):
    def __init__(self, inp_size, outp_size, hidden_layers, diagonal_M=0.01, chebyshev_M=2, 
                 rotations=3, svd_u=False):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        
        # hidden layers
        for i in range(hidden_layers):
            for j in range(rotations):
                 self.layers.append(Rotational(inp_size))
            self.layers.append(Diagonal(inp_size, diagonal_M))
            for j in range(rotations):
                 self.layers.append(Rotational(inp_size))
            self.layers.append(Bias(inp_size))
            self.layers.append(Chebyshev(chebyshev_M))
            
            
        # output layer
        if not svd_u:  # default fc
            self.layers.append(torch.nn.Linear(inp_size, outp_size))
        else:
            self.layers.append(Svd_u(inp_size, outp_size))        
    

    def forward(self, inp):
        for layer in self.layers:
            inp = layer(inp)
        return inp
