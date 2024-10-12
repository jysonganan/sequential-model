import torch
import torch.nn as nn

import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F


class RNN_Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(RNN_Encoder, self).__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, h):
        out, h = self.rnn(x, h)
        mean, logvar = self.fc_mean(out), self.fc_logvar(out)
        return mean, logvar, h
    
class RNN_Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(RNN_Decoder, self).__init__()
        self.rnn = nn.GRU(latent_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, z, h):
        out, h = self.rnn(z, h)
        out = self.fc(out)
        return out, h

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.sequential(nn.Linear(input_dim, 128), nn.ReLU(),
                                     nn.Linear(128, 2 * latent_dim))
    def forward(self, x):
        mean, logvar = torch.chunk(self.encoder(x), 2, dim=-1)
        return mean, logvar

class VAE_RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, encoder_version = "RNN"):
        super(VAE_RNN, self).__init__()
        if encoder_version == "RNN":
            self.encoder = RNN_Encoder(input_dim, hidden_dim, latent_dim)
        else:
            self.encoder = Encoder(input_dim, latent_dim)
        
        self.encoder_version = encoder_version
        self.decoder = RNN_Decoder(latent_dim, hidden_dim, input_dim)
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):

        batch_size = x.size(0)
        h = torch.zeros(1, batch_size, self.hidden_dim).to(x.device)

        if self.encoder_version == "RNN":
            mean, logvar, h = self.encoder(x, h)
        
        ### here, h is passed from encoder to decoder (stateful)
        ### we can also set them as stateless, h init in encoder/decoder class 
        # h = torch.zeros(1, x.size(0), hidden_dim).to(x.device)
        # and not pass h to decoder

        else:
            mean, logvar = self.encoder(x)
            
            
        z = self.reparameterize(mean, logvar)
        x_recon, _ = self.decoder(z, h)
        return x_recon, mean, logvar
        




if __name__ == "__main__":
    def generate_synthetic_data(seq_len, num_sequences):
        data = [] 
        for _ in range(num_sequences):
            base = np.random.randn()
            trend = np.random.randn()
            noise = np.random.randn(seq_len) * 0.1
            sequence = base + trend * np.arange(seq_len) + noise
            data.append(sequence)
        return np.array(data)
    
    seq_len = 30
    num_sequences = 1000
    data = generate_synthetic_data(seq_len, num_sequences)
    data = data[:, :, np.newaxis]
    train_data = torch.tensor(data, dtype=torch.float(32))



    input_dim = 1
    hidden_dim = 20
    latent_dim = 5
    output_dim = 1
    batch_size = 16

    train_loader = DataLoader(TensorDataset(train_data, train_data), 
                              batch_size=batch_size, shuffle=True)
    
    def vae_loss(recon_x, x, mean, logvar):
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        kld_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        return recon_loss + kld_loss
    

    model = VAE_RNN(input_dim, hidden_dim, latent_dim, output_dim)
    ###
    # omitted
    ###
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            x, _ = batch
            optimizer.zero_grad()
            x_recon, mean, logvar = model(x)
            loss = vae_loss(x_recon, x, mean, logvar)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader.dataset)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss}")



