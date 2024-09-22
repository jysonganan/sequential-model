from torchdiffeq import odeint_adjoint as odeint

import numpy as np

import torch
import torch.nn as nn


class ode_simple(nn.Module):
    """
    function dy/dt = t + y/5.
    """
    def forward(self, t, y):
        return t + y/5
    
    
class ode_spiral(nn.Module):
    def __init__(self, A):
        super(ode_spiral, self).__init__()
        self.A = A

    def forward(self, t, y):
        return torch.mm(y**3, self.A)
    

class ode_spiral_complex(nn.Module):
    def __init__(self, A, B, y0):
        super(ode_spiral_complex, self).__init__()
        # A, B, y0 are layers
        self.A = nn.Linear(2, 2, bias=False)
        self.A.weight = nn.Parameter(A)
        self.B = nn.Linear(2, 2, bias=False)
        self.B.weight = nn.Parameter(B)
        self.y0 = nn.Parameter(y0)
        

    def forward(self, t, y):
        yTy0= torch.sum(y*self.y0, dim = 1)
        dydt = torch.sigmoid(yTy0) * self.A(y - self.y0) + torch.sigmoid(-yTy0) * self.B(y + self.y0)
        return dydt    
    

if __name__ == "__main__":

    ### simulated 1d time series following simple ode:  dy/dt = t + y/5.
    y0 = torch.tensor(-3, dtype=torch.float64)
    t = torch.linspace(0., 4., 101)
    with torch.no_grad():
        y = odeint(ode_simple(), y0, t, method='dopri5')

    ### simulated 2d time series following spiral dynamics
    true_A = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]])

    true_y0 = torch.tensor([[2., 0.]])
    t = torch.linspace(0., 25., 1000)
    with torch.no_grad():
        true_y = odeint(ode_spiral(true_A), true_y0, t, method='dopri5')

    
    ### simulated 2d time series following more complex spiral dynamics
    true_y0 = torch.tensor([[-1., 0.]])
    t = torch.linspace(0., 25., 1000)
    
    true_A = torch.tensor([[-0.1, -0.5], [0.5, -0.1]])
    true_B = torch.tensor([[0.2, 1.], [-1, 0.2]])

    with torch.no_grad():
        true_y = odeint(ode_spiral_complex(true_A, true_B, true_y0), 
                        true_y0, t, method='dopri5')
     
