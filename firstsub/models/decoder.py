from __future__ import annotations

import torch
import torch.nn as nn

from models.encoder import MLP


class CDVAEDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        max_atoms: int,
        num_species: int,
        context_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.max_atoms = max_atoms
        self.num_species = num_species
        self.context_dim = context_dim or hidden_dim
        self.context_from_latent = MLP(latent_dim, [hidden_dim], self.context_dim, dropout=dropout)
        decoder_input_dim = latent_dim + self.context_dim
        self.global_decoder = MLP(decoder_input_dim, [hidden_dim * 2, hidden_dim], hidden_dim, dropout=dropout)
        self.count_head = nn.Linear(hidden_dim, max_atoms)
        self.lattice_head = nn.Linear(hidden_dim, 6)
        self.slot_embedding = nn.Embedding(max_atoms, hidden_dim)
        self.anchor_logits = nn.Embedding(max_atoms, 3)
        nn.init.uniform_(self.anchor_logits.weight, -1.5, 1.5)
        self.node_decoder = MLP(decoder_input_dim + hidden_dim, [hidden_dim, hidden_dim], hidden_dim, dropout=dropout)
        self.atom_type_head = nn.Linear(hidden_dim, num_species)
        self.coord_head = nn.Linear(hidden_dim, 3)

    def forward(
        self,
        z: torch.Tensor,
        global_context: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch_size = z.size(0)
        if global_context is None:
            global_context = self.context_from_latent(z)

        decoder_condition = torch.cat([z, global_context], dim=-1)
        global_hidden = self.global_decoder(decoder_condition)
        count_logits = self.count_head(global_hidden)
        lattice_raw = self.lattice_head(global_hidden)

        slot_ids = torch.arange(self.max_atoms, device=z.device)
        slot_features = self.slot_embedding(slot_ids).unsqueeze(0).expand(batch_size, -1, -1)
        condition_features = decoder_condition.unsqueeze(1).expand(-1, self.max_atoms, -1)
        node_hidden = self.node_decoder(torch.cat([condition_features, slot_features], dim=-1))
        atom_type_logits = self.atom_type_head(node_hidden)
        anchor_logits = self.anchor_logits(slot_ids).unsqueeze(0).expand(batch_size, -1, -1)
        coord_logits = anchor_logits + self.coord_head(node_hidden)
        frac_coords = torch.sigmoid(coord_logits)

        return {
            "count_logits": count_logits,
            "lattice_raw": lattice_raw,
            "atom_type_logits": atom_type_logits,
            "frac_coords": frac_coords,
            "decoder_global_context": global_context,
            "decoder_condition": decoder_condition,
            "anchor_frac_coords": torch.sigmoid(anchor_logits),
        }
