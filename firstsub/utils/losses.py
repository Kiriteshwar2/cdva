from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def build_predicted_atom_mask(
    pred_atom_type_logits: torch.Tensor,
    pred_num_atoms: torch.Tensor,
) -> torch.Tensor:
    """
    Select the active predicted atom slots for each crystal.

    Because the decoder emits a fixed number of slots, we keep the top-k slots by
    atom-type confidence, where k is the predicted atom count.
    """

    batch_size, max_atoms, _ = pred_atom_type_logits.shape
    confidence = pred_atom_type_logits.softmax(dim=-1).amax(dim=-1)
    mask = torch.zeros(batch_size, max_atoms, dtype=torch.bool, device=pred_atom_type_logits.device)

    for batch_index in range(batch_size):
        num_atoms = int(pred_num_atoms[batch_index].item())
        num_atoms = max(0, min(num_atoms, max_atoms))
        if num_atoms == 0:
            continue
        selected = torch.topk(confidence[batch_index], k=num_atoms, largest=True).indices
        mask[batch_index, selected] = True

    return mask


def pairwise_periodic_distance(
    pred_frac_coords: torch.Tensor,
    target_frac_coords: torch.Tensor,
) -> torch.Tensor:
    deltas = pred_frac_coords[:, None, :] - target_frac_coords[None, :, :]
    deltas = deltas - torch.round(deltas)
    return torch.linalg.norm(deltas, dim=-1)


def pairwise_lattice_aware_distance(
    pred_frac_coords: torch.Tensor,
    target_frac_coords: torch.Tensor,
    lattice_matrix: torch.Tensor,
) -> torch.Tensor:
    deltas = pred_frac_coords[:, None, :] - target_frac_coords[None, :, :]
    deltas = deltas - torch.round(deltas)
    cartesian = deltas @ lattice_matrix
    return torch.linalg.norm(cartesian, dim=-1)


def compute_permutation_invariant_loss(
    pred_atom_type_logits: torch.Tensor,
    pred_frac_coords: torch.Tensor,
    pred_num_atoms: torch.Tensor,
    target_atom_types: torch.Tensor,
    target_frac_coords: torch.Tensor,
    target_num_atoms: torch.Tensor,
    lattice_matrix: torch.Tensor | None = None,
    *,
    atom_type_penalty: float = 1.0,
    unmatched_cost: float = 2.0,
    ignore_index: int = -100,
) -> dict[str, torch.Tensor]:
    """
    Compute permutation-invariant reconstruction losses using Hungarian matching.

    Matching is solved independently for each crystal in the batch. The cost matrix
    combines minimum-image, lattice-aware Cartesian distance with an atom-type
    mismatch penalty. Extra predicted or missing atoms are handled by dummy
    assignments with an unmatched penalty.
    """

    batch_size = pred_atom_type_logits.size(0)
    pred_mask = build_predicted_atom_mask(pred_atom_type_logits, pred_num_atoms)

    coord_losses: list[torch.Tensor] = []
    atom_type_losses: list[torch.Tensor] = []
    unmatched_penalties: list[torch.Tensor] = []
    matched_accuracies: list[torch.Tensor] = []
    mean_distances: list[torch.Tensor] = []
    matched_fractions: list[torch.Tensor] = []
    matched_pred_indices: list[torch.Tensor] = []
    matched_target_indices: list[torch.Tensor] = []

    for batch_index in range(batch_size):
        gt_count = int(target_num_atoms[batch_index].item())
        pred_indices = pred_mask[batch_index].nonzero(as_tuple=False).view(-1)
        gt_indices = (target_atom_types[batch_index] != ignore_index).nonzero(as_tuple=False).view(-1)[:gt_count]

        pred_count = int(pred_indices.numel())
        gt_count = int(gt_indices.numel())

        if pred_count == 0 and gt_count == 0:
            zero = pred_frac_coords.new_zeros(())
            coord_losses.append(zero)
            atom_type_losses.append(zero)
            unmatched_penalties.append(zero)
            matched_accuracies.append(zero)
            mean_distances.append(zero)
            matched_fractions.append(torch.ones_like(zero))
            matched_pred_indices.append(torch.empty(0, dtype=torch.long, device=pred_frac_coords.device))
            matched_target_indices.append(torch.empty(0, dtype=torch.long, device=pred_frac_coords.device))
            continue

        pred_coords = pred_frac_coords[batch_index, pred_indices] if pred_count > 0 else pred_frac_coords.new_zeros((0, 3))
        pred_logits = (
            pred_atom_type_logits[batch_index, pred_indices]
            if pred_count > 0
            else pred_atom_type_logits.new_zeros((0, pred_atom_type_logits.size(-1)))
        )
        gt_coords = target_frac_coords[batch_index, gt_indices] if gt_count > 0 else target_frac_coords.new_zeros((0, 3))
        gt_types = target_atom_types[batch_index, gt_indices] if gt_count > 0 else target_atom_types.new_zeros((0,), dtype=torch.long)

        square_size = max(pred_count, gt_count, 1)
        augmented_cost = pred_frac_coords.new_full((square_size, square_size), fill_value=float(unmatched_cost))

        if pred_count > 0 and gt_count > 0:
            with torch.no_grad():
                if lattice_matrix is None:
                    coord_cost = pairwise_periodic_distance(pred_coords.detach(), gt_coords.detach())
                else:
                    coord_cost = pairwise_lattice_aware_distance(
                        pred_coords.detach(),
                        gt_coords.detach(),
                        lattice_matrix[batch_index].detach(),
                    )
                pred_type_ids = pred_logits.detach().argmax(dim=-1)
                type_cost = (pred_type_ids[:, None] != gt_types.detach()[None, :]).float() * float(atom_type_penalty)
                augmented_cost[:pred_count, :gt_count] = coord_cost + type_cost


        # Replace NaN/Inf with large safe values
        augmented_cost = torch.nan_to_num(
            augmented_cost,
            nan=1e6,
            posinf=1e6,
            neginf=-1e6
        )

        row_ind_np, col_ind_np = linear_sum_assignment(augmented_cost.detach().cpu().numpy())
        row_ind = torch.as_tensor(row_ind_np, device=pred_frac_coords.device, dtype=torch.long)
        col_ind = torch.as_tensor(col_ind_np, device=pred_frac_coords.device, dtype=torch.long)

        matched_mask = (row_ind < pred_count) & (col_ind < gt_count)
        matched_rows = row_ind[matched_mask]
        matched_cols = col_ind[matched_mask]
        matched_count = int(matched_mask.sum().item())
        normalizer = max(pred_count, gt_count, 1)
        matched_pred_indices.append(pred_indices[matched_rows] if matched_count > 0 else pred_indices.new_empty(0))
        matched_target_indices.append(gt_indices[matched_cols] if matched_count > 0 else gt_indices.new_empty(0))

        unmatched_total = (
            ((row_ind < pred_count) & (col_ind >= gt_count)).sum()
            + ((row_ind >= pred_count) & (col_ind < gt_count)).sum()
        )
        unmatched_penalties.append(unmatched_total.float() / normalizer)
        matched_fractions.append(pred_frac_coords.new_tensor(matched_count / normalizer))

        if matched_count == 0:
            zero = pred_frac_coords.new_zeros(())
            coord_losses.append(zero)
            atom_type_losses.append(zero)
            matched_accuracies.append(zero)
            mean_distances.append(zero)
            continue

        matched_pred_coords = pred_coords[matched_rows]
        matched_gt_coords = gt_coords[matched_cols]
        coord_delta = matched_pred_coords - matched_gt_coords
        coord_delta = coord_delta - torch.round(coord_delta)
        coord_losses.append(coord_delta.pow(2).mean())
        if lattice_matrix is None:
            mean_distances.append(torch.linalg.norm(coord_delta.detach(), dim=-1).mean())
        else:
            cartesian_delta = coord_delta.detach() @ lattice_matrix[batch_index].detach()
            mean_distances.append(torch.linalg.norm(cartesian_delta, dim=-1).mean())

        matched_pred_logits = pred_logits[matched_rows]
        matched_gt_types = gt_types[matched_cols]
        atom_type_losses.append(F.cross_entropy(matched_pred_logits, matched_gt_types, reduction="mean"))

        matched_type_predictions = matched_pred_logits.argmax(dim=-1)
        matched_accuracies.append((matched_type_predictions == matched_gt_types).float().mean())

    def _mean_or_zero(values: list[torch.Tensor], reference: torch.Tensor) -> torch.Tensor:
        if not values:
            return reference.new_zeros(())
        return torch.stack(values).mean()

    reference = pred_frac_coords
    return {
        "coord_loss": _mean_or_zero(coord_losses, reference),
        "atom_type_loss": _mean_or_zero(atom_type_losses, reference),
        "unmatched_penalty": _mean_or_zero(unmatched_penalties, reference),
        "matched_atom_type_accuracy": _mean_or_zero(matched_accuracies, reference),
        "matched_fractional_distance": _mean_or_zero(mean_distances, reference),
        "matched_fraction": _mean_or_zero(matched_fractions, reference),
        "matched_pred_indices": matched_pred_indices,
        "matched_target_indices": matched_target_indices,
        "pred_atom_mask": pred_mask,
    }
