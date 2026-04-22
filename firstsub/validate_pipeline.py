from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch_geometric.loader import DataLoader

from data.graph_builder import CrystalGraphDataset, GraphBuildConfig
from models import CDVAE, CDVAEModelConfig
from train import load_dataset_and_splits, resolve_dataset_paths
from utils import (
    ensure_directory,
    load_yaml_config,
    structure_from_prediction,
    validate_graph_data,
    validate_structure_integrity,
    write_validation_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate CDVAE graphs, model forward path, and CIF export integrity.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the YAML config file.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional checkpoint to validate model/generation.")
    parser.add_argument("--num-graphs", type=int, default=8, help="Number of cached graphs to validate.")
    parser.add_argument("--output-dir", type=str, default="runs/cdvae/validation", help="Directory for validation artifacts.")
    parser.add_argument("--device", type=str, default=None, help="Override device, e.g. cuda or cpu.")
    return parser.parse_args()


def load_model(checkpoint_path: str | Path, device: torch.device) -> CDVAE:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = CDVAE(CDVAEModelConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=False)
    if "volume_min" in checkpoint and "volume_max" in checkpoint:
        model.set_volume_bounds(float(checkpoint["volume_min"]), float(checkpoint["volume_max"]))
    return model


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    output_dir = ensure_directory(args.output_dir)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    dataset, train_indices, val_indices = load_dataset_and_splits(config)
    inspect_indices = (train_indices + val_indices)[: args.num_graphs]
    graph_reports = []
    for index in inspect_indices:
        graph_reports.append({"dataset_index": index, **validate_graph_data(dataset[index])})

    report: dict[str, Any] = {
        "device": str(device),
        "graphs_valid": all(item["valid"] for item in graph_reports),
        "graph_reports": graph_reports,
    }

    if args.checkpoint:
        model = load_model(args.checkpoint, device=device)
        sample_indices = inspect_indices[: min(2, len(inspect_indices))]
        loader = DataLoader([dataset[index] for index in sample_indices], batch_size=len(sample_indices), shuffle=False)
        batch = next(iter(loader)).to(device)
        outputs = model(batch)
        report["forward_validation"] = {
            "mu_shape": list(outputs["mu"].shape),
            "logvar_shape": list(outputs["logvar"].shape),
            "frac_coords_finite": bool(torch.isfinite(outputs["frac_coords"]).all().item()),
            "lattice_matrix_finite": bool(torch.isfinite(outputs["lattice_matrix"]).all().item()),
            "output_device": str(outputs["mu"].device),
        }

        with torch.no_grad():
            samples = model.sample(1, device=device)
        species_vocab = torch.load(args.checkpoint, map_location="cpu")["species_vocab"]
        idx_to_symbol = {index: symbol for symbol, index in species_vocab.items()}
        num_atoms = int(samples["pred_num_atoms"][0].item())
        structure = structure_from_prediction(
            samples["pred_atom_types"][0, :num_atoms],
            samples["frac_coords"][0, :num_atoms],
            samples["lattice_matrix"][0],
            idx_to_symbol,
        )
        report["generation_validation"] = validate_structure_integrity(
            structure,
            min_distance=float(model.config.min_distance_angstrom),
            min_volume=float(model.volume_min.item()),
            max_volume=float(model.volume_max.item()),
            roundtrip_dir=output_dir / "cif_roundtrip",
        )

    report_path = write_validation_report(report, output_dir / "validation_report.json")
    print(f"Validation report written to {report_path}")


if __name__ == "__main__":
    main()
