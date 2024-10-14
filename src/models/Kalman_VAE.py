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
                                     nn.Linear(256, input_dim)
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
        mean, logvar, z_preds, z_updateds = [], [], [], []
        z_prev = torch.zeros(batch_size, self.latent_dim).to(x.device)
        for t in range(seq_len):
            x_t = x[:,t,:]
            mean_t, logvar_t = torch.chunk(self.encoder(x_t), 2, dim=-1)
            mean.append(mean_t)
            logvar.append(logvar_t)
            
            z_t = self.reparameterize(mean_t, logvar_t)

            # Kalman Filtering
            z_pred_t = torch.matmul(self.transition_matrix, z_prev.t()).t() + z_t
            z_preds.append(z_pred_t)

            z_pred_cov = torch.matmul(self.transition_matrix, self.process_covariance)
            innovation = x_t - torch.matmul(self.measurement_matrix, z_pred_t.t()).t()
            innovation_cov = torch.matmul(self.measurement_matrix, z_pred_cov) + self.measurement_covariance
            kalman_gain = torch.matmul(z_pred_cov, torch.inverse(innovation_cov))
            z_updated = z_pred_t + torch.matmul(kalman_gain, innovation.t()).t()
            z_updateds.append(z_updated)

            z_prev = z_updated

        mean = torch.stack(mean, dim=1)
        logvar = torch.stack(logvar, dim=1)

        z_updateds = torch.stack(z_updated, dim=1)
        x_recon = self.decoder(z_updateds.view(-1, self.latent_dim)).view(batch_size, seq_len, -1)

        return x_recon, mean, logvar


            

if __name__ == "__main__":
    def vae_loss(recon_x, x, mean, logvar):
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        kld_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        return recon_loss + kld_loss


