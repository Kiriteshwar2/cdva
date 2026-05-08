from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from backend.app.ml.utils.preprocessing import ensure_directory
from backend.app.ml.utils.visualization import plot_structure

from backend.app.schemas.generation import GenerationRequest, LatticeConstraint, TargetPropertiesConstraint
from backend.app.services.inference import CDVAEGenerationService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate constrained crystal structures from a CDVAE checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path or registry name of the checkpoint to use.")
    parser.add_argument("--num-samples", type=int, default=4, help="Number of accepted structures to write.")
    parser.add_argument("--output-dir", type=str, default="generated", help="Directory for CIF files and metadata.")
    parser.add_argument("--visualize", action="store_true", help="Render a 3D PNG for each accepted structure.")
    parser.add_argument("--elements", type=str, required=True, help="Comma-separated allowed elements, e.g. C,Si,O.")
    parser.add_argument("--num-atoms", type=int, required=True, help="Exact number of atoms to enforce.")
    parser.add_argument("--lattice-a", type=float, default=None)
    parser.add_argument("--lattice-b", type=float, default=None)
    parser.add_argument("--lattice-c", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--energy-min", type=float, default=None)
    parser.add_argument("--energy-max", type=float, default=None)
    parser.add_argument("--density-min", type=float, default=None)
    parser.add_argument("--density-max", type=float, default=None)
    parser.add_argument("--min-interatomic-distance", type=float, default=1.2)
    parser.add_argument("--candidate-pool-size", type=int, default=24)
    parser.add_argument("--max-attempts", type=int, default=96)
    parser.add_argument("--refinement-steps", type=int, default=None)
    parser.add_argument("--refinement-noise-std", type=float, default=None)
    return parser.parse_args()


async def generate_many(args: argparse.Namespace) -> list[dict[str, object]]:
    service = CDVAEGenerationService()
    await service.load_model(args.checkpoint)
    output_dir = ensure_directory(args.output_dir)
    accepted: list[dict[str, object]] = []

    for index in range(args.num_samples):
        request = GenerationRequest(
            checkpoint_name=args.checkpoint,
            elements=[item.strip() for item in args.elements.split(",") if item.strip()],
            num_atoms=args.num_atoms,
            lattice=LatticeConstraint(
                a=args.lattice_a,
                b=args.lattice_b,
                c=args.lattice_c,
                alpha=args.alpha,
                beta=args.beta,
                gamma=args.gamma,
            ),
            target_properties=TargetPropertiesConstraint(
                energy_min=args.energy_min,
                energy_max=args.energy_max,
                density_min=args.density_min,
                density_max=args.density_max,
            ),
            min_interatomic_distance=args.min_interatomic_distance,
            candidate_pool_size=args.candidate_pool_size,
            max_attempts=args.max_attempts,
            refinement_steps=args.refinement_steps,
            refinement_noise_std=args.refinement_noise_std,
        )
        candidate = await service.generate(request)
        cif_path = output_dir / f"generated_{index:03d}.cif"
        cif_path.write_text(candidate.cif_string, encoding="utf-8")

        record = {
            "sample_index": index,
            "checkpoint_name": service.status["model_id"],
            "cif_path": str(cif_path),
            "metadata": candidate.metadata,
        }
        if args.visualize:
            image_path = output_dir / f"generated_{index:03d}.png"
            plot_structure(candidate.structure, image_path, title=f"Generated Crystal {index}")
            record["image_path"] = str(image_path)
        accepted.append(record)
    return accepted


def main() -> None:
    args = parse_args()
    records = asyncio.run(generate_many(args))
    output_dir = ensure_directory(args.output_dir)
    metadata_path = output_dir / "generated_metadata.json"
    metadata_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Accepted structures: {len(records)}")
    print(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()
