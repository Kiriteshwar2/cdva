from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from pymatgen.core import Lattice, Structure


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_indices(dataset_size: int, val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    generator = random.Random(seed)
    indices = list(range(dataset_size))
    generator.shuffle(indices)
    val_size = max(1, int(round(dataset_size * val_ratio))) if dataset_size > 1 else 0
    val_indices = indices[:val_size]
    train_indices = indices[val_size:] if val_size < dataset_size else indices
    return train_indices, val_indices


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def read_jsonl_record_at_offset(path: str | Path, offset: int) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        handle.seek(offset)
        line = handle.readline()
    return json.loads(line)


def is_valid_lattice(structure):
    lattice = structure.lattice

    a, b, c = lattice.a, lattice.b, lattice.c
    alpha, beta, gamma = lattice.alpha, lattice.beta, lattice.gamma

    return (
        2 <= a <= 20 and
        2 <= b <= 20 and
        2 <= c <= 20 and
        60 <= alpha <= 120 and
        60 <= beta <= 120 and
        60 <= gamma <= 120
    )

def record_to_structure(record: dict[str, Any]) -> Structure | None:
    if "structure" in record:
        structure = Structure.from_dict(record["structure"])
        if structure is None:
            continue
    else:
        lattice = record["lattice"]
        structure = Structure(
            lattice=Lattice.from_parameters(
                a=float(lattice["a"]),
                b=float(lattice["b"]),
                c=float(lattice["c"]),
                alpha=float(lattice["alpha"]),
                beta=float(lattice["beta"]),
                gamma=float(lattice["gamma"]),
            ),
            species=record["species"],
            coords=record["frac_coords"],
            coords_are_cartesian=False,
        )

    # 🚨 ADD THIS FILTER
    if not is_valid_lattice(structure):
        return None

    return structure

def canonicalize_structure(structure: Structure) -> Structure:
    ordered_sites = sorted(
        structure.sites,
        key=lambda site: (
            site.specie.Z,
            round(float(site.frac_coords[0]) % 1.0, 8),
            round(float(site.frac_coords[1]) % 1.0, 8),
            round(float(site.frac_coords[2]) % 1.0, 8),
        ),
    )
    return Structure(
        lattice=structure.lattice,
        species=[site.specie.symbol for site in ordered_sites],
        coords=[site.frac_coords for site in ordered_sites],
        coords_are_cartesian=False,
    )


def extract_lattice_parameters(structure: Structure) -> list[float]:
    lattice = structure.lattice
    return [
        float(lattice.a),
        float(lattice.b),
        float(lattice.c),
        float(lattice.alpha),
        float(lattice.beta),
        float(lattice.gamma),
    ]


def lattice_params_to_matrix_torch(lattice_params: torch.Tensor) -> torch.Tensor:
    a = lattice_params[:, 0]
    b = lattice_params[:, 1]
    c = lattice_params[:, 2]
    alpha = torch.deg2rad(lattice_params[:, 3])
    beta = torch.deg2rad(lattice_params[:, 4])
    gamma = torch.deg2rad(lattice_params[:, 5])

    cos_alpha = torch.cos(alpha)
    cos_beta = torch.cos(beta)
    cos_gamma = torch.cos(gamma)
    sin_gamma = torch.sin(gamma).clamp_min(1e-6)

    vector_a = torch.stack([a, torch.zeros_like(a), torch.zeros_like(a)], dim=-1)
    vector_b = torch.stack([b * cos_gamma, b * sin_gamma, torch.zeros_like(b)], dim=-1)
    c_x = c * cos_beta
    c_y = c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma
    c_z = torch.sqrt(torch.clamp(c.pow(2) - c_x.pow(2) - c_y.pow(2), min=1e-8))
    vector_c = torch.stack([c_x, c_y, c_z], dim=-1)
    return torch.stack([vector_a, vector_b, vector_c], dim=1)


def lattice_matrix_to_params_torch(lattice_matrix: torch.Tensor) -> torch.Tensor:
    a_vec = lattice_matrix[:, 0]
    b_vec = lattice_matrix[:, 1]
    c_vec = lattice_matrix[:, 2]

    a = torch.linalg.norm(a_vec, dim=-1).clamp_min(1e-8)
    b = torch.linalg.norm(b_vec, dim=-1).clamp_min(1e-8)
    c = torch.linalg.norm(c_vec, dim=-1).clamp_min(1e-8)

    alpha = torch.rad2deg(torch.acos(torch.clamp((b_vec * c_vec).sum(dim=-1) / (b * c), -1.0, 1.0)))
    beta = torch.rad2deg(torch.acos(torch.clamp((a_vec * c_vec).sum(dim=-1) / (a * c), -1.0, 1.0)))
    gamma = torch.rad2deg(torch.acos(torch.clamp((a_vec * b_vec).sum(dim=-1) / (a * b), -1.0, 1.0)))

    return torch.stack([a, b, c, alpha, beta, gamma], dim=-1)


def lattice_volume_torch(lattice_matrix: torch.Tensor) -> torch.Tensor:
    return torch.abs(torch.det(lattice_matrix))


def structure_from_prediction(
    atom_type_ids: torch.Tensor,
    frac_coords: torch.Tensor,
    lattice: torch.Tensor,
    idx_to_symbol: dict[int, str],
) -> Structure:
    atom_type_ids = atom_type_ids.detach().cpu().long()
    frac_coords = frac_coords.detach().cpu().float() % 1.0
    species = [idx_to_symbol[int(index)] for index in atom_type_ids.tolist()]
    lattice = lattice.detach().cpu().float()

    if lattice.ndim == 2 and lattice.shape == (3, 3):
        lattice_obj = Lattice(lattice.numpy())
    else:
        lattice_obj = Lattice.from_parameters(
            a=float(np.clip(lattice[0].item(), 2.0, 20.0)),
            b=float(np.clip(lattice[1].item(), 2.0, 20.0)),
            c=float(np.clip(lattice[2].item(), 2.0, 20.0)),
            alpha=float(np.clip(lattice[3].item(), 60.0, 120.0)),
            beta=float(np.clip(lattice[4].item(), 60.0, 120.0)),
            gamma=float(np.clip(lattice[5].item(), 60.0, 120.0)),
        )

    return Structure(lattice=lattice_obj, species=species, coords=frac_coords.numpy(), coords_are_cartesian=False)


def tensor_to_python(metrics: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            result[key] = float(value.detach().cpu().item())
        else:
            result[key] = value
    return result
