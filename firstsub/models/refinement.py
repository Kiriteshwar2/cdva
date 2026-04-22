from __future__ import annotations

import torch
import torch.nn as nn

from models.encoder import MLP


class CoordinateRefinementNetwork(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        context_dim: int,
        num_species: int,
        hidden_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        input_dim = 3 + num_species + latent_dim + context_dim
        self.network = MLP(
            input_dim=input_dim,
            hidden_dims=[hidden_dim, hidden_dim],
            output_dim=hidden_dim,
            dropout=dropout,
        )
        self.delta_head = nn.Linear(hidden_dim, 3)

    def forward(
        self,
        noisy_frac_coords: torch.Tensor,
        atom_type_logits: torch.Tensor,
        latent: torch.Tensor,
        global_context: torch.Tensor,
    ) -> torch.Tensor:
        atom_probabilities = atom_type_logits.softmax(dim=-1)
        latent_features = latent.unsqueeze(1).expand(-1, noisy_frac_coords.size(1), -1)
        context_features = global_context.unsqueeze(1).expand(-1, noisy_frac_coords.size(1), -1)
        hidden = self.network(
            torch.cat([noisy_frac_coords, atom_probabilities, latent_features, context_features], dim=-1)
        )
        delta_logits = self.delta_head(hidden)
        coord_logits = torch.logit(noisy_frac_coords.clamp(1e-4, 1.0 - 1e-4)) + delta_logits
        return torch.sigmoid(coord_logits)
