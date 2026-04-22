from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch

from models.decoder import CDVAEDecoder
from models.encoder import CDVAEEncoder
from models.refinement import CoordinateRefinementNetwork
from utils.losses import compute_permutation_invariant_loss
from utils.preprocessing import lattice_matrix_to_params_torch, lattice_volume_torch


@dataclass(frozen=True)
class CDVAEModelConfig:
    num_species: int
    max_atoms: int
    hidden_dim: int = 192
    latent_dim: int = 128
    num_encoder_layers: int = 4
    num_rbf: int = 32
    cutoff: float = 5.0
    dropout: float = 0.1
    kl_beta: float = 1e-4
    min_distance_angstrom: float = 1.2
    ignore_index: int = -100
    assignment_atom_type_penalty: float = 1.0
    assignment_unmatched_cost: float = 2.0
    refinement_hidden_dim: int = 128
    refinement_noise_std: float = 0.05
    refinement_steps: int = 5
    energy_proxy_sigma: float = 1.5
    energy_proxy_epsilon: float = 0.1
    min_lattice_volume: float = 10.0
    max_lattice_volume: float = 2500.0


class CDVAE(nn.Module):
    def __init__(self, config: CDVAEModelConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = CDVAEEncoder(
            num_species=config.num_species,
            hidden_dim=config.hidden_dim,
            latent_dim=config.latent_dim,
            num_layers=config.num_encoder_layers,
            num_rbf=config.num_rbf,
            cutoff=config.cutoff,
            dropout=config.dropout,
        )
        self.decoder = CDVAEDecoder(
            latent_dim=config.latent_dim,
            hidden_dim=config.hidden_dim,
            max_atoms=config.max_atoms,
            num_species=config.num_species,
            context_dim=config.hidden_dim,
            dropout=config.dropout,
        )
        self.refinement_network = CoordinateRefinementNetwork(
            latent_dim=config.latent_dim,
            context_dim=config.hidden_dim,
            num_species=config.num_species,
            hidden_dim=config.refinement_hidden_dim,
            dropout=config.dropout,
        )
        self.register_buffer("lattice_min", torch.tensor([1.0, 1.0, 1.0, 30.0, 30.0, 30.0], dtype=torch.float32))
        self.register_buffer("lattice_max", torch.tensor([20.0, 20.0, 20.0, 150.0, 150.0, 150.0], dtype=torch.float32))
        self.register_buffer("volume_min", torch.tensor(float(config.min_lattice_volume), dtype=torch.float32))
        self.register_buffer("volume_max", torch.tensor(float(config.max_lattice_volume), dtype=torch.float32))

    def set_lattice_bounds(self, minima: torch.Tensor, maxima: torch.Tensor) -> None:
        self.lattice_min.copy_(minima.to(self.lattice_min.device))
        self.lattice_max.copy_(maxima.to(self.lattice_max.device))

    def set_volume_bounds(self, min_volume: float, max_volume: float) -> None:
        self.volume_min.copy_(torch.tensor(float(min_volume), device=self.volume_min.device))
        self.volume_max.copy_(torch.tensor(float(max_volume), device=self.volume_max.device))

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode_lattice_matrix(self, lattice_raw: torch.Tensor) -> torch.Tensor:
        diagonal = F.softplus(lattice_raw[:, [0, 2, 5]]) + 1e-3
        lattice_matrix = lattice_raw.new_zeros(lattice_raw.size(0), 3, 3)
        lattice_matrix[:, 0, 0] = diagonal[:, 0]
        lattice_matrix[:, 1, 0] = lattice_raw[:, 1]
        lattice_matrix[:, 1, 1] = diagonal[:, 1]
        lattice_matrix[:, 2, 0] = lattice_raw[:, 3]
        lattice_matrix[:, 2, 1] = lattice_raw[:, 4]
        lattice_matrix[:, 2, 2] = diagonal[:, 2]
        return lattice_matrix

    def encode(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(batch)

    def encode_with_context(self, batch):
        return self.encoder(batch, return_context=True)

    def decode(self, z: torch.Tensor, global_context: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        decoded = self.decoder(z, global_context=global_context)
        decoded["frac_coords"] = torch.clamp(decoded["frac_coords"], 0.0, 1.0)
        decoded["lattice_matrix"] = self.decode_lattice_matrix(decoded["lattice_raw"])
        decoded["lattice_matrix"] = torch.clamp(decoded["lattice_matrix"], -20.0, 20.0)
        decoded["lattice_params"] = lattice_matrix_to_params_torch(decoded["lattice_matrix"])
        decoded["pred_num_atoms"] = decoded["count_logits"].argmax(dim=-1) + 1
        decoded["pred_atom_types"] = decoded["atom_type_logits"].argmax(dim=-1)
        return decoded

    def forward(self, batch) -> dict[str, torch.Tensor]:
        mu, logvar, graph_context = self.encode_with_context(batch)
        z = self.reparameterize(mu, logvar)
        decoded = self.decode(z, global_context=graph_context)
        decoded["mu"] = mu
        decoded["logvar"] = logvar
        decoded["latent"] = z
        decoded["graph_context"] = graph_context
        return decoded

    def sample(self, num_samples: int, device: torch.device | str) -> dict[str, torch.Tensor]:
        z = torch.randn(num_samples, self.config.latent_dim, device=device)
        decoded = self.decode(z)
        decoded["latent"] = z
        decoded["graph_context"] = decoded["decoder_global_context"]
        return decoded

    def compute_losses(
        self,
        batch,
        outputs: dict[str, torch.Tensor],
        loss_weights: dict[str, float],
        kl_beta: float | None = None,
    ) -> dict[str, torch.Tensor]:
        atom_types = batch.atom_types.long()
        target_atom_types, atom_mask = to_dense_batch(
            atom_types,
            batch.batch,
            max_num_nodes=self.config.max_atoms,
            fill_value=self.config.ignore_index,
        )
        target_frac_coords, _ = to_dense_batch(
            batch.frac_coords,
            batch.batch,
            max_num_nodes=self.config.max_atoms,
            fill_value=0.0,
        )
        target_num_atoms = batch.num_atoms.view(-1).long()
        target_lattice_matrix = batch.lattice_matrix.view(-1, 3, 3)

        count_loss = F.cross_entropy(outputs["count_logits"], target_num_atoms - 1)
        permutation_losses = compute_permutation_invariant_loss(
            pred_atom_type_logits=outputs["atom_type_logits"],
            pred_frac_coords=outputs["frac_coords"],
            pred_num_atoms=outputs["pred_num_atoms"],
            target_atom_types=target_atom_types,
            target_frac_coords=target_frac_coords,
            target_num_atoms=target_num_atoms,
            lattice_matrix=target_lattice_matrix,
            atom_type_penalty=self.config.assignment_atom_type_penalty,
            unmatched_cost=self.config.assignment_unmatched_cost,
            ignore_index=self.config.ignore_index,
        )
        atom_type_loss = permutation_losses["atom_type_loss"]
        coord_loss = permutation_losses["coord_loss"]
        unmatched_penalty = permutation_losses["unmatched_penalty"]
        lattice_loss = F.mse_loss(outputs["lattice_matrix"], target_lattice_matrix)
        logvar = torch.clamp(outputs["logvar"], -10.0, 10.0)
        kl_loss = -0.5 * torch.mean(
            torch.sum(1 + logvar - outputs["mu"].pow(2) - logvar.exp(), dim=-1)
        )
        distance_consistency_loss = self.distance_consistency_loss(
            pred_frac_coords=outputs["frac_coords"],
            target_frac_coords=target_frac_coords,
            pred_lattice_matrix=outputs["lattice_matrix"],
            target_lattice_matrix=target_lattice_matrix,
            matched_pred_indices=permutation_losses["matched_pred_indices"],
            matched_target_indices=permutation_losses["matched_target_indices"],
        )
        symmetry_consistency_loss = distance_consistency_loss
        refinement_loss = self.refinement_loss(
            initial_frac_coords=outputs["frac_coords"],
            atom_type_logits=outputs["atom_type_logits"],
            latent=outputs["latent"],
            global_context=outputs["graph_context"],
            target_frac_coords=target_frac_coords,
            matched_pred_indices=permutation_losses["matched_pred_indices"],
            matched_target_indices=permutation_losses["matched_target_indices"],
            atom_mask=permutation_losses["pred_atom_mask"],
            noise_std=self.config.refinement_noise_std,
        )
        energy_proxy_loss = self.energy_proxy_loss(
            frac_coords=outputs["frac_coords"],
            lattice_matrix=outputs["lattice_matrix"],
            atom_mask=permutation_losses["pred_atom_mask"],
            sigma=self.config.energy_proxy_sigma,
            epsilon=self.config.energy_proxy_epsilon,
        )
        volume_loss = self.volume_constraint_loss(outputs["lattice_matrix"])
        min_separation_loss = self.minimum_separation_penalty(
            outputs["frac_coords"],
            outputs["lattice_matrix"],
            permutation_losses["pred_atom_mask"],
            self.config.min_distance_angstrom,
        )

        total_loss = (
            loss_weights["count"] * count_loss
            + loss_weights["atom_type"] * atom_type_loss
            + loss_weights["coord"] * coord_loss
            + loss_weights["lattice"] * lattice_loss
            + loss_weights["kl"] * (self.config.kl_beta if kl_beta is None else kl_beta) * kl_loss
            + loss_weights.get("distance_consistency", 0.0) * distance_consistency_loss
            + loss_weights.get("symmetry_consistency", 0.0) * symmetry_consistency_loss
            + loss_weights.get("refinement", 0.25) * refinement_loss
            + loss_weights.get("energy_proxy", 0.05) * energy_proxy_loss
            + loss_weights.get("volume", 0.05) * volume_loss
            + loss_weights.get("matching_unmatched", 1.0) * unmatched_penalty
            + loss_weights.get("min_separation", 0.0) * min_separation_loss
        )

        with torch.no_grad():
            atom_accuracy = permutation_losses["matched_atom_type_accuracy"]
            count_accuracy = (outputs["pred_num_atoms"] == target_num_atoms).float().mean()

        return {
            "loss": total_loss,
            "count_loss": count_loss.detach(),
            "atom_type_loss": atom_type_loss.detach(),
            "coord_loss": coord_loss.detach(),
            "lattice_loss": lattice_loss.detach(),
            "kl_loss": kl_loss.detach(),
            "distance_consistency_loss": distance_consistency_loss.detach(),
            "symmetry_consistency_loss": symmetry_consistency_loss.detach(),
            "refinement_loss": refinement_loss.detach(),
            "energy_proxy_loss": energy_proxy_loss.detach(),
            "volume_loss": volume_loss.detach(),
            "matching_unmatched_loss": unmatched_penalty.detach(),
            "matched_fraction": permutation_losses["matched_fraction"].detach(),
            "matched_fractional_distance": permutation_losses["matched_fractional_distance"].detach(),
            "min_separation_loss": min_separation_loss.detach(),
            "count_accuracy": count_accuracy.detach(),
            "atom_type_accuracy": atom_accuracy.detach(),
        }

    def refine_coordinates(
        self,
        frac_coords: torch.Tensor,
        atom_type_logits: torch.Tensor,
        latent: torch.Tensor,
        global_context: torch.Tensor,
        lattice_matrix: torch.Tensor,
        atom_mask: torch.Tensor,
        *,
        num_steps: int | None = None,
        noise_std: float | None = None,
    ) -> torch.Tensor:
        steps = num_steps or self.config.refinement_steps
        active_noise_std = self.config.refinement_noise_std if noise_std is None else noise_std
        refined = frac_coords
        if active_noise_std > 0:
            refined = (refined + torch.randn_like(refined) * active_noise_std).clamp(0.0, 1.0)

        for _ in range(max(steps, 1)):
            refined = self.refinement_network(refined, atom_type_logits, latent, global_context)
            refined = self._apply_repulsion(refined, lattice_matrix, atom_mask, self.config.min_distance_angstrom)
            refined = refined.clamp(0.0, 1.0)

        return torch.where(atom_mask.unsqueeze(-1), refined, frac_coords)

    @staticmethod
    def minimum_separation_penalty(
        frac_coords: torch.Tensor,
        lattice_matrix: torch.Tensor,
        atom_mask: torch.Tensor,
        minimum_distance: float,
    ) -> torch.Tensor:
        penalties: list[torch.Tensor] = []
        for index in range(frac_coords.size(0)):
            valid = atom_mask[index]
            if valid.sum() < 2:
                continue
            coords = frac_coords[index][valid]
            lattice = lattice_matrix[index]
            deltas = coords.unsqueeze(1) - coords.unsqueeze(0)
            deltas = deltas - torch.round(deltas)
            cartesian = deltas @ lattice
            distances = torch.linalg.norm(cartesian, dim=-1)
            upper_triangle = torch.triu(torch.ones_like(distances, dtype=torch.bool), diagonal=1)
            pair_distances = distances[upper_triangle]
            if pair_distances.numel() == 0:
                continue
            violations = F.relu(minimum_distance - pair_distances)
            penalties.append(violations.pow(2).mean())

        if not penalties:
            return torch.tensor(0.0, device=frac_coords.device)
        return torch.stack(penalties).mean()

    @staticmethod
    def distance_consistency_loss(
        pred_frac_coords: torch.Tensor,
        target_frac_coords: torch.Tensor,
        pred_lattice_matrix: torch.Tensor,
        target_lattice_matrix: torch.Tensor,
        matched_pred_indices: list[torch.Tensor],
        matched_target_indices: list[torch.Tensor],
    ) -> torch.Tensor:
        losses: list[torch.Tensor] = []
        for batch_index, (pred_indices, target_indices) in enumerate(zip(matched_pred_indices, matched_target_indices)):
            if pred_indices.numel() < 2 or target_indices.numel() < 2:
                continue

            pred_coords = pred_frac_coords[batch_index, pred_indices]
            gt_coords = target_frac_coords[batch_index, target_indices]

            pred_distances = CDVAE._pairwise_cartesian_distances(pred_coords, pred_lattice_matrix[batch_index])
            gt_distances = CDVAE._pairwise_cartesian_distances(gt_coords, target_lattice_matrix[batch_index])
            upper_triangle = torch.triu(torch.ones_like(pred_distances, dtype=torch.bool), diagonal=1)
            losses.append(F.mse_loss(pred_distances[upper_triangle], gt_distances[upper_triangle]))

        if not losses:
            return pred_frac_coords.new_zeros(())
        return torch.stack(losses).mean()

    def refinement_loss(
        self,
        initial_frac_coords: torch.Tensor,
        atom_type_logits: torch.Tensor,
        latent: torch.Tensor,
        global_context: torch.Tensor,
        target_frac_coords: torch.Tensor,
        matched_pred_indices: list[torch.Tensor],
        matched_target_indices: list[torch.Tensor],
        atom_mask: torch.Tensor,
        noise_std: float,
    ) -> torch.Tensor:
        noisy_coords = (initial_frac_coords.detach() + torch.randn_like(initial_frac_coords) * noise_std).clamp(0.0, 1.0)
        refined_coords = self.refinement_network(
            noisy_frac_coords=noisy_coords,
            atom_type_logits=atom_type_logits.detach(),
            latent=latent,
            global_context=global_context,
        )
        refined_coords = torch.where(atom_mask.unsqueeze(-1), refined_coords, initial_frac_coords.detach())

        losses: list[torch.Tensor] = []
        for batch_index, (pred_indices, target_indices) in enumerate(zip(matched_pred_indices, matched_target_indices)):
            if pred_indices.numel() == 0:
                continue
            pred_coords = refined_coords[batch_index, pred_indices]
            gt_coords = target_frac_coords[batch_index, target_indices]
            coord_delta = pred_coords - gt_coords
            coord_delta = coord_delta - torch.round(coord_delta)
            losses.append(coord_delta.pow(2).mean())

        if not losses:
            return initial_frac_coords.new_zeros(())
        return torch.stack(losses).mean()

    @staticmethod
    def _pairwise_cartesian_distances(frac_coords: torch.Tensor, lattice_matrix: torch.Tensor) -> torch.Tensor:
        deltas = frac_coords.unsqueeze(1) - frac_coords.unsqueeze(0)
        deltas = deltas - torch.round(deltas)
        cartesian = deltas @ lattice_matrix
        return torch.linalg.norm(cartesian, dim=-1)

    def energy_proxy_loss(
        self,
        frac_coords: torch.Tensor,
        lattice_matrix: torch.Tensor,
        atom_mask: torch.Tensor,
        sigma: float,
        epsilon: float,
    ) -> torch.Tensor:
        energies: list[torch.Tensor] = []
        for batch_index in range(frac_coords.size(0)):
            valid = atom_mask[batch_index]
            if valid.sum() < 2:
                continue
            coords = frac_coords[batch_index, valid]
            distances = self._pairwise_cartesian_distances(coords, lattice_matrix[batch_index])
            upper_triangle = torch.triu(torch.ones_like(distances, dtype=torch.bool), diagonal=1)
            pair_distances = distances[upper_triangle].clamp_min(1e-3)
            ratios = (sigma / pair_distances).clamp_max(50.0)
            lj_repulsion = epsilon * ratios.pow(12)
            energies.append(lj_repulsion.mean())

        if not energies:
            return frac_coords.new_zeros(())
        return torch.stack(energies).mean()

    def volume_constraint_loss(self, lattice_matrix: torch.Tensor) -> torch.Tensor:
        volumes = lattice_volume_torch(lattice_matrix)
        low_violation = F.relu(self.volume_min - volumes)
        high_violation = F.relu(volumes - self.volume_max)
        return (low_violation.pow(2) + high_violation.pow(2)).mean()

    @staticmethod
    def _apply_repulsion(
        frac_coords: torch.Tensor,
        lattice_matrix: torch.Tensor,
        atom_mask: torch.Tensor,
        minimum_distance: float,
        strength: float = 0.05,
    ) -> torch.Tensor:
        adjusted = frac_coords.clone()
        for batch_index in range(frac_coords.size(0)):
            valid = atom_mask[batch_index]
            if valid.sum() < 2:
                continue

            coords = adjusted[batch_index, valid]
            lattice = lattice_matrix[batch_index]
            deltas = coords.unsqueeze(1) - coords.unsqueeze(0)
            deltas = deltas - torch.round(deltas)
            cartesian = deltas @ lattice
            distances = torch.linalg.norm(cartesian, dim=-1).clamp_min(1e-6)
            pair_mask = torch.triu(torch.ones_like(distances, dtype=torch.bool), diagonal=1)
            violating_pairs = pair_mask & (distances < minimum_distance)
            if not violating_pairs.any():
                continue

            displacement = torch.zeros_like(coords)
            violating_rows, violating_cols = violating_pairs.nonzero(as_tuple=True)
            for row, col in zip(violating_rows.tolist(), violating_cols.tolist()):
                direction = deltas[row, col]
                norm = torch.linalg.norm(direction).clamp_min(1e-6)
                direction = direction / norm
                magnitude = strength * (minimum_distance - distances[row, col]) / minimum_distance
                displacement[row] = displacement[row] + direction * magnitude
                displacement[col] = displacement[col] - direction * magnitude

            coords = (coords + displacement).remainder(1.0)
            adjusted[batch_index, valid] = coords

        return adjusted
