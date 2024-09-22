import torch
import torch.nn as nn

import numpy as np
from torchdiffeq import odeint_adjoint as odeint

import sequential_model.simulation_dynamics as sim_dyn


class simulation_data:
    """
    Simulate sequential data following different patterns. 
    The generated sequential data have shapes as (n_samples, n_timesteps, n_inputs), 
    which correponds to number of samples/sequences, time steps and input feature dimensions.
    """
    def __init__(self, n_samples, n_timesteps):
        self.n_samples = n_samples
        self.n_timesteps = n_timesteps


    def generate_sin_data(self, noise=0.4):
        t = np.arange(self.n_timesteps)
        data = []
        for _ in range(self.n_samples):
            trajectory = 10 * np.sin(0.2 * t) + noise*np.random.normal(0, 1, self.n_timesteps)
            data.append(trajectory)
        data = np.stack(data, axis=0)
        return torch.tensor(t, dtype=torch.float32), torch.tensor(data, dtype=torch.float32)
    
    
    def generate_spiral_data(self, noise=0.01, a=0.1, b=0.2):
        t = np.linspace(0, 4 * np.pi, self.n_timesteps)
        data = []
        for _ in range(self.n_samples):
            r = a + b * t
            x = r * np.sin(t)
            y = r * np.cos(t)
            trajectory = np.stack([x, y], axis=1)
            trajectory += noise * np.random.randn(self.n_timesteps, 2)
            data.append(trajectory)
        data = np.stack(data, axis=0)
        return torch.tensor(t, dtype=torch.float32), torch.tensotorch.tensor(data, dtype=torch.float32)


    def generate_brownian_process_data(self, mean=0, stdev=np.sqrt(1/500)):
        distances = np.cumsum(np.random.normal(mean, stdev, (self.n_samples, self.n_timesteps)), axis=1)
        distances = distances.reshape(*distances.shape, 1)
        return torch.tensor(distances, dtype=torch.float32)
    

    
    def generate_from_ode_simple(self, y0 = torch.tensor(-3, dtype=torch.float64)):
        t = torch.linspace(0., 4., self.n_timesteps)
        data = []
        for _ in range(self.n_samples):
            y = odeint(sim_dyn.ode_simple(), y0, t, method='dopri5')
            y = y.reshape(1, self.n_timesteps, 1)
            data.append(y)
        data = (torch.stack(data, axis=0)).squeeze(1)
        return torch.tensor(t, dtype=torch.float32), data
    

    
    def generate_from_ode_spiral(self, true_A = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]]), 
                                 true_y0 = torch.tensor([[2., 0.]])):
        t = torch.linspace(0., 25., 1000)
        data = []
        for _ in range(self.n_samples):
            true_y = odeint(sim_dyn.ode_spiral(true_A), true_y0, t, method='dopri5')
            true_y = true_y.permute(1,0,2)
            data.append(true_y)
        data = (torch.stack(data, axis=0)).squeeze(1)
        return torch.tensor(t, dtype=torch.float32), data
    


    def generate_from_ode_spiral_complex(self, true_y0 = torch.tensor([[-1., 0.]]),
                                         true_A = torch.tensor([[-0.1, -0.5], [0.5, -0.1]]),
                                         true_B = torch.tensor([[0.2, 1.], [-1, 0.2]])):
        t = torch.linspace(0., 25., 1000)
        data = []
        for _ in range(self.n_samples):
            with torch.no_grad():
                true_y = odeint(sim_dyn.ode_spiral_complex(true_A, true_B, true_y0), 
                                true_y0, t, method='dopri5')
            true_y = true_y.permute(1,0,2)
            data.append(true_y)
        data = (torch.stack(data, axis=0)).squeeze(1) 
        return torch.tensor(t, dtype=torch.float32), data




 
 




    

    








    
        


    



    
