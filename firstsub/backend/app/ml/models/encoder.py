from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn import o3
from e3nn.nn import BatchNorm, NormActivation
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        dropout: float = 0.0,
        activate_last: bool = False,
    ) -> None:
        super().__init__()
        dims = [input_dim, *hidden_dims, output_dim]
        layers: list[nn.Module] = []
        for index in range(len(dims) - 1):
            layers.append(nn.Linear(dims[index], dims[index + 1]))
            is_last = index == len(dims) - 2
            if not is_last or activate_last:
                layers.append(nn.SiLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class RadialBasisExpansion(nn.Module):
    def __init__(self, num_rbf: int, cutoff: float) -> None:
        super().__init__()
        centers = torch.linspace(0.0, cutoff, num_rbf)
        gamma = 1.0 / max((centers[1] - centers[0]).item() ** 2, 1e-6) if num_rbf > 1 else 1.0
        self.register_buffer("centers", centers)
        self.gamma = gamma

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        distances = distances.view(-1, 1)
        return torch.exp(-self.gamma * (distances - self.centers) ** 2)


class EquivariantInteractionBlock(nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        num_rbf: int,
        cutoff: float,
        lmax: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax)
        self.rbf = RadialBasisExpansion(num_rbf=num_rbf, cutoff=cutoff)
        self.tp = o3.FullyConnectedTensorProduct(
            self.irreps_in,
            self.sh_irreps,
            self.irreps_out,
            shared_weights=False,
        )
        self.edge_mlp = MLP(
            input_dim=num_rbf,
            hidden_dims=[max(num_rbf * 2, 32), max(self.irreps_out.dim, 32)],
            output_dim=self.tp.weight_numel,
            dropout=dropout,
        )
        self.self_connection = o3.Linear(self.irreps_in, self.irreps_out)
        self.norm = BatchNorm(self.irreps_out)
        self.activation = NormActivation(self.irreps_out, scalar_nonlinearity=F.silu, normalize=True)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vec: torch.Tensor,
        edge_distance: torch.Tensor,
    ) -> torch.Tensor:
        src, dst = edge_index
        valid_edges = edge_distance > 1e-8

        aggregated = x.new_zeros(x.size(0), self.irreps_out.dim)
        if valid_edges.any():
            filtered_src = src[valid_edges]
            filtered_dst = dst[valid_edges]
            filtered_vec = edge_vec[valid_edges]
            filtered_distance = edge_distance[valid_edges]

            sh = o3.spherical_harmonics(
                self.sh_irreps,
                filtered_vec,
                normalize=True,
                normalization="component",
            )
            weights = self.edge_mlp(self.rbf(filtered_distance))
            messages = self.tp(x[filtered_src], sh, weights)
            aggregated.index_add_(0, filtered_dst, messages)

        updated = self.self_connection(x) + aggregated
        updated = self.norm(updated)
        return self.activation(updated)


class CDVAEEncoder(nn.Module):
    def __init__(
        self,
        num_species: int,
        hidden_dim: int,
        latent_dim: int,
        num_layers: int,
        num_rbf: int,
        cutoff: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        scalar_channels = max(8, hidden_dim // 2)
        vector_channels = max(4, hidden_dim // 6)
        self.input_irreps = o3.Irreps(f"{hidden_dim}x0e")
        self.hidden_irreps = o3.Irreps(f"{scalar_channels}x0e + {vector_channels}x1o")
        self.readout_irreps = o3.Irreps(f"{hidden_dim}x0e")

        self.atom_embedding = nn.Embedding(num_species, hidden_dim)
        interaction_blocks: list[nn.Module] = [
            EquivariantInteractionBlock(
                irreps_in=self.input_irreps,
                irreps_out=self.hidden_irreps,
                num_rbf=num_rbf,
                cutoff=cutoff,
                dropout=dropout,
            )
        ]
        interaction_blocks.extend(
            EquivariantInteractionBlock(
                irreps_in=self.hidden_irreps,
                irreps_out=self.hidden_irreps,
                num_rbf=num_rbf,
                cutoff=cutoff,
                dropout=dropout,
            )
            for _ in range(max(num_layers - 1, 0))
        )
        self.interaction_blocks = nn.ModuleList(interaction_blocks)
        self.scalar_readout = o3.Linear(self.hidden_irreps, self.readout_irreps)
        self.lattice_encoder = MLP(6, [hidden_dim], hidden_dim, dropout=dropout)
        self.readout = MLP(hidden_dim * 4, [hidden_dim * 2, hidden_dim], hidden_dim, dropout=dropout)
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

    def forward(self, batch, return_context: bool = False):
        x = self.atom_embedding(batch.atom_types)
        edge_distance = batch.edge_attr.view(-1)
        edge_vec = batch.edge_vec
        for interaction in self.interaction_blocks:
            x = interaction(x, batch.edge_index, edge_vec, edge_distance)

        scalar_features = self.scalar_readout(x)
        pooled_mean = global_mean_pool(scalar_features, batch.batch)
        pooled_add = global_add_pool(scalar_features, batch.batch)
        pooled_max = global_max_pool(scalar_features, batch.batch)
        lattice_features = self.lattice_encoder(batch.lattice_params.view(-1, 6))
        crystal_features = torch.cat([pooled_mean, pooled_add, pooled_max, lattice_features], dim=-1)
        hidden = self.readout(crystal_features)
        mu = self.mu_head(hidden)
        logvar = self.logvar_head(hidden)
        if return_context:
            return mu, logvar, hidden
        return mu, logvar
