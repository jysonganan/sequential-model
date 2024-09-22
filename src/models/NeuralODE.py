import torch
import torch.nn as nn
#from torchdiffeq import odeint
from torchdiffeq import odeint_adjoint as odeint



class ODEFunc(nn.Module):
    def __init__(self, input_dim=2):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 50),
            nn.ReLU(),
            nn.Linear(50, input_dim),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, x):
        dxdt = self.net(x)
        return dxdt
    


class NeuralODE(nn.Module):
    def __init__(self, odefunc, input_dim):
        super(NeuralODE, self).__init__()
        self.odefunc = odefunc(input_dim)

    def forward(self, x0, t):
        return odeint(self.odefunc, x0, t)
    


class LatentNeuralODE(nn.Module):
    def __init__(self, odefunc, input_dim, latent_dim):
        super(LatentNeuralODE, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, latent_dim),
                                     nn.ReLU(),
                                     nn.Linear(latent_dim, latent_dim))
        self.odefunc = odefunc(latent_dim)
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 50),
                                     nn.ReLU(),
                                     nn.Linear(50, input_dim))
        
    def forward(self, x0, t):
        z0 = self.encoder(x0)
        z_t = odeint(self.odefunc, z0, t)
        x_t = self.decoder(z_t)
        return x_t



##### VAE-NeuralODE

class RNNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(RNNEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.rnn = nn.GRU(input_dim+1, hidden_dim)
        self.hid2lat = nn.Linear(hidden_dim, 2 * latent_dim)

    def forward(self, x, t):
        t = t.clone()
        t[1:] = t[:-1] - t[1:]
        t[0] = 0
        xt = torch.cat((x, t), dim=-1)
        _, h0 = self.rnn(xt.flip((0,)))
        # last hidden state h0 is used to compute the latent
        z0 = self.hid2lat(h0[0])
        z0_mean = z0[:, :self.latent_dim]
        z0_logvar = z0[:, self.latent_dim:]
        return z0_mean, z0_logvar
    

class VAE_NeuralODE(nn.Module):
    def __init__(self, odefunc, input_dim, hidden_dim, latent_dim):
        super(VAE_NeuralODE, self).__init__()
        self.encoder = RNNEncoder(input_dim, hidden_dim, latent_dim)
        self.odefunc = odefunc(latent_dim)
        self.decoder = nn.Sequential(
            nn.linear(latent_dim, hidden_dim),
            nn.Linear(hidden_dim, input_dim)
        )

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mean)
        return mean + eps * std

    def forward(self, x, t):
        z_mean, z_logvar = self.encoder(x, t)
        z0 = self.reparameterize(z_mean, z_logvar)
        z_t = odeint(self.odefunc, z0, t)
        x_recon_t = self.decoder(z_t)
        return x_recon_t, z0, z_mean, z_logvar










### to be validated -- NeuralODE for MNIST classification (replace feature extraction from ResNet to neuralODE: self.feature=ode)

def add_time(in_tensor, t):
    bs, c, w, h = in_tensor.shape
    return torch.cat((in_tensor, t.expand(bs, 1, w, h)), dim=1)

class ConvODEF(nn.Module):
    def __init__(self, input_dim):
        super(ConvODEF, self).__init__()
        #self.conv1 = conv3*3(dim+1, dim)
        #self.norm1 = norm(dim)
        #self.conv2 = conv3*3(dim+1, dim)
        #self.norm2 = norm(dim)
    def forward(self, t, x):
        xt = add_time(x, t)
        h = self.norm1(torch.relu(self.conv1(xt)))
        ht = add_time(h, t)
        dxdt = self.norm2(torch.relu(self.conv2(ht)))
        return dxdt
    






