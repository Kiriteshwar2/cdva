"""
model.py — CrystalVAE with conditioning.
The score shifts the latent vector before decoding,
biasing generation toward exotic (positive) or stable (negative) structures.
"""

import torch
import torch.nn as nn


class CrystalVAE(nn.Module):
    def __init__(self, input_dim: int = 8, latent_dim: int = 4):
        super().__init__()

        # ── Encoder ──────────────────────────────────────────────────────────
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.fc_mu     = nn.Linear(64, latent_dim)   # mean
        self.fc_logvar = nn.Linear(64, latent_dim)   # log-variance

        # ── Decoder ──────────────────────────────────────────────────────────
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
        )

    # ── Reparameterization trick ──────────────────────────────────────────────
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # ── Forward pass ─────────────────────────────────────────────────────────
    def forward(self, x):
        h      = self.encoder(x)
        mu     = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z      = self.reparameterize(mu, logvar)
        recon  = self.decoder(z)
        return recon, mu, logvar

    # ── Conditioned generation ────────────────────────────────────────────────
    def generate(self, n: int = 1, antigravity_score: float = 0.0):
        """
        Sample `n` latent vectors and bias them by antigravity_score.
          +2 → high energy / exotic / unstable structures
           0 → neutral sampling
          -2 → denser / conventional structures
        """
        z = torch.randn(n, self.fc_mu.out_features)
        # Antigravity shift: add a scaled constant offset to all latent dims
        z = z + antigravity_score * torch.ones_like(z)
        with torch.no_grad():
            params = self.decoder(z)
        return params


# ── VAE loss (reconstruction + KL divergence) ────────────────────────────────
def vae_loss(recon_x, x, mu, logvar, beta: float = 1.0):
    recon_loss = nn.MSELoss()(recon_x, x)
    kl_loss    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss, recon_loss.item(), kl_loss.item()
