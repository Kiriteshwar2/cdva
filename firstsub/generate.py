from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from pymatgen.io.cif import CifWriter

from models import CDVAE, CDVAEModelConfig
from utils.losses import build_predicted_atom_mask
from utils.preprocessing import ensure_directory, structure_from_prediction
from utils.validation import validate_structure_integrity
from utils.visualization import plot_structure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate crystal structures from a trained CDVAE checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a trained checkpoint.")
    parser.add_argument("--num-samples", type=int, default=8, help="Number of crystals to generate.")
    parser.add_argument("--output-dir", type=str, default="generated", help="Directory for CIF files and metadata.")
    parser.add_argument("--device", type=str, default=None, help="Override device, e.g. cuda or cpu.")
    parser.add_argument("--visualize", action="store_true", help="Render a 3D PNG for each generated structure.")
    parser.add_argument("--refinement-steps", type=int, default=None, help="Number of coordinate refinement steps.")
    parser.add_argument("--refinement-noise-std", type=float, default=None, help="Noise scale used before refinement.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(
            f"Checkpoint not found at `{args.checkpoint}`. Train a model first or pass a valid checkpoint path."
        )
    checkpoint = torch.load(args.checkpoint, map_location=device)

    model_config = CDVAEModelConfig(**checkpoint["model_config"])
    model = CDVAE(model_config).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=False)
    model.set_lattice_bounds(checkpoint["lattice_min"].to(device), checkpoint["lattice_max"].to(device))
    if "volume_min" in checkpoint and "volume_max" in checkpoint:
        model.set_volume_bounds(float(checkpoint["volume_min"]), float(checkpoint["volume_max"]))
    model.eval()

    output_dir = ensure_directory(args.output_dir)
    species_vocab = checkpoint["species_vocab"]
    idx_to_symbol = {index: symbol for symbol, index in species_vocab.items()}

    generated_metadata: list[dict[str, object]] = []
    with torch.no_grad():
        samples = model.sample(args.num_samples, device=device)
        pred_atom_mask = build_predicted_atom_mask(samples["atom_type_logits"], samples["pred_num_atoms"])
        refinement_steps = args.refinement_steps if args.refinement_steps is not None else model.config.refinement_steps
        refinement_noise_std = (
            args.refinement_noise_std if args.refinement_noise_std is not None else model.config.refinement_noise_std
        )
        if refinement_steps > 0:
            samples["refined_frac_coords"] = model.refine_coordinates(
                frac_coords=samples["frac_coords"],
                atom_type_logits=samples["atom_type_logits"],
                latent=samples["latent"],
                global_context=samples["graph_context"],
                lattice_matrix=samples["lattice_matrix"],
                atom_mask=pred_atom_mask,
                num_steps=refinement_steps,
                noise_std=refinement_noise_std,
            )
        else:
            samples["refined_frac_coords"] = samples["frac_coords"]

    for index in range(args.num_samples):
        num_atoms = int(samples["pred_num_atoms"][index].item())
        atom_type_ids = samples["pred_atom_types"][index, :num_atoms]
        frac_coords = samples["refined_frac_coords"][index, :num_atoms]
        lattice_matrix = samples["lattice_matrix"][index]
        structure = structure_from_prediction(atom_type_ids, frac_coords, lattice_matrix, idx_to_symbol)

        cif_path = output_dir / f"generated_{index:03d}.cif"
        CifWriter(structure).write_file(str(cif_path))

        metadata = {
            "sample_index": index,
            "num_atoms": num_atoms,
            "species": [site.specie.symbol for site in structure.sites],
            "lattice": {
                "a": float(structure.lattice.a),
                "b": float(structure.lattice.b),
                "c": float(structure.lattice.c),
                "alpha": float(structure.lattice.alpha),
                "beta": float(structure.lattice.beta),
                "gamma": float(structure.lattice.gamma),
            },
            "cif_path": str(cif_path),
            "refinement_steps": refinement_steps,
            "refinement_noise_std": refinement_noise_std,
        }
        metadata["validation"] = validate_structure_integrity(
            structure,
            min_distance=float(model.config.min_distance_angstrom),
            min_volume=float(model.volume_min.item()),
            max_volume=float(model.volume_max.item()),
            roundtrip_dir=output_dir / f"validation_{index:03d}",
        )

        if args.visualize:
            image_path = output_dir / f"generated_{index:03d}.png"
            plot_structure(structure, image_path, title=f"Generated Crystal {index}")
            metadata["image_path"] = str(image_path)

        generated_metadata.append(metadata)

    metadata_path = output_dir / "generated_metadata.json"
    metadata_path.write_text(json.dumps(generated_metadata, indent=2), encoding="utf-8")
    print(f"Saved {len(generated_metadata)} generated structures to {output_dir}")


if __name__ == "__main__":
    main()
