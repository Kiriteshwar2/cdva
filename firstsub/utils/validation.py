from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from pymatgen.core import Structure
from pymatgen.io.cif import CifParser, CifWriter

from utils.preprocessing import lattice_volume_torch


def validate_fractional_coordinates(frac_coords: torch.Tensor, tolerance: float = 1e-5) -> dict[str, Any]:
    frac_coords = frac_coords.detach().cpu()
    min_value = float(frac_coords.min().item()) if frac_coords.numel() else 0.0
    max_value = float(frac_coords.max().item()) if frac_coords.numel() else 0.0
    return {
        "valid": min_value >= -tolerance and max_value <= 1.0 + tolerance,
        "min_value": min_value,
        "max_value": max_value,
    }


def validate_lattice_matrix(
    lattice_matrix: torch.Tensor,
    min_volume: float = 0.1,
    max_volume: float = 5000.0,
) -> dict[str, Any]:
    lattice_matrix = lattice_matrix.detach().cpu().float()
    determinant = float(torch.det(lattice_matrix).item())
    volume = float(lattice_volume_torch(lattice_matrix.unsqueeze(0)).item())
    finite = bool(torch.isfinite(lattice_matrix).all().item())
    valid = finite and determinant > 0.0 and min_volume <= volume <= max_volume
    return {
        "valid": valid,
        "determinant": determinant,
        "volume": volume,
        "finite": finite,
    }


def validate_graph_data(graph, distance_tolerance: float = 5e-3) -> dict[str, Any]:
    edge_count = int(graph.edge_index.size(1))
    edge_distance = graph.edge_attr.view(-1).detach().cpu()
    edge_vector_norm = torch.linalg.norm(graph.edge_vec.detach().cpu(), dim=-1)
    distance_error = float((edge_distance - edge_vector_norm).abs().max().item()) if edge_count > 0 else 0.0
    valid = (
        hasattr(graph, "edge_vec")
        and hasattr(graph, "cart_coords")
        and edge_count == int(graph.edge_attr.size(0))
        and edge_count == int(graph.edge_vec.size(0))
        and distance_error <= distance_tolerance
    )
    return {
        "valid": valid,
        "edge_count": edge_count,
        "num_nodes": int(graph.num_nodes),
        "max_edge_distance_error": distance_error,
    }


def validate_structure_integrity(
    structure: Structure,
    *,
    min_distance: float = 0.5,
    min_volume: float = 0.1,
    max_volume: float = 5000.0,
    roundtrip_dir: str | Path | None = None,
) -> dict[str, Any]:
    frac_coords = torch.tensor(structure.frac_coords, dtype=torch.float32)
    frac_validation = validate_fractional_coordinates(frac_coords)
    lattice_validation = validate_lattice_matrix(torch.tensor(structure.lattice.matrix, dtype=torch.float32), min_volume, max_volume)

    distance_matrix = structure.distance_matrix
    if len(structure) > 1:
        nonzero_distances = distance_matrix[distance_matrix > 1e-8]
        min_pair_distance = float(nonzero_distances.min()) if nonzero_distances.size > 0 else float("inf")
    else:
        min_pair_distance = float("inf")
    overlap_ok = min_pair_distance >= min_distance or len(structure) <= 1

    roundtrip_ok = True
    roundtrip_error = ""
    if roundtrip_dir is not None:
        roundtrip_dir = Path(roundtrip_dir)
        roundtrip_dir.mkdir(parents=True, exist_ok=True)
        cif_path = roundtrip_dir / "roundtrip_validation.cif"
        CifWriter(structure).write_file(str(cif_path))
        try:
            parsed = CifParser(str(cif_path)).parse_structures(primitive=False)
            roundtrip_ok = len(parsed) > 0 and len(parsed[0]) == len(structure)
        except Exception as exc:  # pragma: no cover
            roundtrip_ok = False
            roundtrip_error = str(exc)

    return {
        "valid": frac_validation["valid"] and lattice_validation["valid"] and overlap_ok and roundtrip_ok,
        "fractional_coordinates": frac_validation,
        "lattice": lattice_validation,
        "minimum_pair_distance": min_pair_distance,
        "overlap_ok": overlap_ok,
        "roundtrip_ok": roundtrip_ok,
        "roundtrip_error": roundtrip_error,
    }


def write_validation_report(report: dict[str, Any], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output_path
