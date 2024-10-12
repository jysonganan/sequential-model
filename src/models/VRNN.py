import torch
import torch.nn as nn

class VRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
       
        self.encoder_x = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                       nn.ReLU())
        self.encoder_z = nn.Sequential(nn.Linear(latent_dim, hidden_dim),
                                       nn.ReLU())
        
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        self.decoder = nn.Linear(hidden_dim, input_dim)

        self.rnn = nn.GRU(hidden_dim * 2, hidden_dim)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        h = torch.zeros(1, batch_size, self.hidden_dim).to(x.device)
        x_recon, mean, logvar = [], [], []
        for t in range(seq_len):
            encoded_x_t = self.encoder_x(x[:,t,:])
           
            mean_t, logvar_t = self.fc_mean(encoded_x_t), self.fc_logvar(encoded_x_t)
            z_t = self.reparameterize(mean_t, logvar_t)

            encoded_z_t = self.encoder_z(z_t)
            x_recon_t = self.decoder(encoded_z_t)

            rnn_input_t = torch.cat([encoded_x_t, encoded_z_t], dim=1).unsqueeze(0)
            _, h = self.rnn(rnn_input_t, h)
            
            x_recon.append(x_recon_t)
            mean.append(mean_t)
            logvar.append(logvar_t)

        x_recon = torch.stack(x_recon, dim=1)
        mean = torch.stack(mean, dim=1)
        logvar = torch.stack(logvar, dim=1)

        return x_recon, mean, logvar
            

