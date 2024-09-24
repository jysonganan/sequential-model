import torch
import torch.nn as nn

import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F


from torchdiffeq import odeint_adjoint as odeint

import math


class GRUODEfunc(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        self.bias = bias
        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias = bias)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias = bias)
        self.nfe = 0
        self.hidden_size = hidden_size

    def forward(self, t, h, x):
        self.nfe += 1
        gate_x = self.x2h(x)
        gate_h = self.h2h(h)
        gates = gate_x + gate_h

        h_r, h_i, h_n = gate_h.chunk(3,1)
        resetgate = F.sigmoid(h_r)
        inputgate = F.sigmoid(h_i)
        newgate = F.tanh(resetgate * h_n)
        #hy = (1 - inputgate) * (newgate - hidden) 
        hy = (1 - update_gate) * hidden + update_gate * new_gate 
        return hy  
    

class GRUODECell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUODECell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.odefunc = GRUODEfunc(input_size, hidden_size, bias)
        self.reset_parameters()
        self.integration_time = torch.tensor([0, 1]).float()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, h, x):      
        #allowing the dynamics of h to jointly depend on x, h (which governed by ode)
        # instead of combined/concatenate to state vector, x influence the dynamics as args in ODE function
        # (reason: x doesn't change over time hence not necessary intergrate; less compute)
        self.integration_time = self.integration_time.type_as(x)
        out = odeint(self.odefunc, h, self.integration_time, args=(x,))
        return out[1][:, self.input_size:]  
        ## return the updated hidden state (at time 1)
  
    
    @property
    def nfe(self):
        return self.odefunc.nfe
    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value
  

class GRUODE(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUODE, self).__init__()
        self.hidden_size = hidden_size
        self.gruode_cell = GRUODECell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        batch_size, seq_len, _ = input.size()
        h = torch.zeros(batch_size, self.hidden_size).to(input.device)
        output = []
        for t in range(seq_len):
            x_t = input[:,t,:]
            h = self.gruode_cell(h, x_t)
            output.append(self.fc(h))
        return torch.cat(output, dim=1)
        
