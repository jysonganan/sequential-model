import torch
import torch.nn as nn

import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F


class KalmanVAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(nn.Linear(input_dim, 256),
                                     nn.ReLU(),
                                     nn.Linear(256, 2 * latent_dim)
                                    )
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 256),
                                     nn.ReLU(),
                                     nn.Linear(256, input_dim),
                                     nn.sigmoid()
                                    )
        
        self.transition_matrix = nn.Parameter(torch.eye(latent_dim))
        self.process_covariance = nn.Parameter(torch.eye(latent_dim))
        self.measurement_matrix = nn.Parameter(torch.eye(latent_dim))
        self.measurement_covariance = nn.Parameter(torch.eye(latent_dim))

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        z_means, z_logvars, z_preds, z_updateds = [], [], [], []
        z_prev = torch.zeros(batch_size, self.latent_dim).to(x.device)
        for t in range(seq_len):
            x_t = x[:,t,:]
            z_mean, z_logvar = torch.chunk(self.encoder(x_t), 2, dim=-1)
            z_means.append(z_mean)
            z_logvars.append(z_logvar)
            z = self.reparameterize(z_mean, z_logvar)

            # Kalman Filtering
            z_pred = torch.matmul(self.transition_matrix, z_prev.t()).t() + z
            z_preds.append(z_pred)

            z_pred_cov = torch.matmul(self.transition_matrix, self.process_covariance)
            innovation = x_t - torch.matmul(self.measurement_matrix, z_pred.t()).t()
            innovation_cov = torch.matmul(self.measurement_matrix, z_pred_cov) + self.measurement_covariance
            kalman_gain = torch.matmul(z_pred_cov, torch.inverse(innovation_cov))
            z_updated = z_pred + torch.matmul(kalman_gain, innovation.t()).t()
            z_updateds.append(z_updated)

            z_prev = z_updated

        z_means = torch.stack(z_means, dim=1)
        z_logvars = torch.stack(z_logvars, dim=1)
        z_updateds = torch.stack(z_updated, dim=1)

        x_recon = self.decoder(z_updateds.view(-1, self.latent_dim)).view(batch_size, seq_len, -1)
        z_means = z_means.view(-1, self.latent_dim)
        z_logvars = z_logvars.view(-1, self.latent_dim)
        return x_recon, z_means, z_logvars


            

if __name__ == "__main__":
    def vae_loss(recon_x, x, mean, logvar):
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        kld_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        return recon_loss + kld_loss


