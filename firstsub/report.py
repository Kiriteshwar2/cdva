from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from utils.preprocessing import load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a PDF report for the upgraded CDVAE project.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the YAML config file.")
    parser.add_argument("--history", type=str, default="runs/cdvae/history.jsonl", help="Training history JSONL path.")
    parser.add_argument(
        "--generated-metadata",
        type=str,
        default="refine_smoke_generated_validated/generated_metadata.json",
        help="Generated metadata JSON path.",
    )
    parser.add_argument(
        "--validation-report",
        type=str,
        default="runs/cdvae/validation/validation_report.json",
        help="Validation report JSON path.",
    )
    parser.add_argument("--output", type=str, default="runs/cdvae/cdvae_project_report.pdf", help="Output PDF path.")
    return parser.parse_args()


def load_json_if_exists(path: str | Path) -> Any:
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_history(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            records.append(json.loads(line))
    return records


def wrapped_lines(text: str, width: int = 96) -> str:
    return "\n".join(textwrap.fill(paragraph, width=width) for paragraph in text.split("\n"))


def add_text_page(pdf: PdfPages, title: str, body: str) -> None:
    figure = plt.figure(figsize=(8.27, 11.69))
    axis = figure.add_axes([0.08, 0.05, 0.84, 0.9])
    axis.axis("off")
    axis.text(0.0, 1.0, title, fontsize=18, fontweight="bold", va="top")
    axis.text(0.0, 0.96, wrapped_lines(body), fontsize=10.5, va="top", family="monospace")
    pdf.savefig(figure)
    plt.close(figure)


def add_history_plot(pdf: PdfPages, history: list[dict[str, Any]]) -> None:
    if not history:
        return
    epochs = [record["epoch"] for record in history]
    train_loss = [record["train_loss"] for record in history]
    val_loss = [record["val_loss"] for record in history]
    figure, axis = plt.subplots(figsize=(8.27, 4.5))
    axis.plot(epochs, train_loss, label="Train Loss")
    axis.plot(epochs, val_loss, label="Validation Loss")
    axis.set_title("Training Curve")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.legend()
    axis.grid(alpha=0.3)
    figure.tight_layout()
    pdf.savefig(figure)
    plt.close(figure)


def summarize_results(history: list[dict[str, Any]], generated_metadata: list[dict[str, Any]] | None) -> str:
    parts: list[str] = []
    if history:
        final_record = history[-1]
        parts.append(
            f"Final recorded epoch: {final_record['epoch']} | train_loss={final_record.get('train_loss', 'n/a')} | "
            f"val_loss={final_record.get('val_loss', 'n/a')} | val_atom_type_accuracy={final_record.get('val_atom_type_accuracy', 'n/a')}."
        )
    else:
        parts.append("No full training history was found in the workspace, so quantitative convergence claims are not included in this report.")

    if generated_metadata:
        total = len(generated_metadata)
        valid = sum(1 for item in generated_metadata if item.get("validation", {}).get("valid"))
        min_distances = [
            item.get("validation", {}).get("minimum_pair_distance")
            for item in generated_metadata
            if item.get("validation", {}).get("minimum_pair_distance") is not None
        ]
        volumes = [
            item.get("validation", {}).get("lattice", {}).get("volume")
            for item in generated_metadata
            if item.get("validation", {}).get("lattice", {}).get("volume") is not None
        ]
        avg_min_distance = sum(min_distances) / len(min_distances) if min_distances else None
        avg_volume = sum(volumes) / len(volumes) if volumes else None
        parts.append(
            f"Generated structures available: {total}. Validation-passing structures: {valid}/{total}. "
            f"Average minimum pair distance: {avg_min_distance if avg_min_distance is not None else 'n/a'}. "
            f"Average lattice volume: {avg_volume if avg_volume is not None else 'n/a'}."
        )
    else:
        parts.append("No generated structure metadata was found, so the report focuses on architecture and implementation status.")

    return "\n\n".join(parts)


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    history = load_history(args.history)
    generated_metadata = load_json_if_exists(args.generated_metadata)
    validation_report = load_json_if_exists(args.validation_report)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset_cfg = config["dataset"]
    model_cfg = config["model"]
    training_cfg = config["training"]

    intro = (
        "Crystal Diffusion Variational Autoencoder (CDVAE) is a generative model for periodic crystal structures. "
        "This project targets Materials Project structures and aims to generate atom types, fractional coordinates, "
        "and valid lattices while respecting permutation symmetry, periodic boundary conditions, and geometric constraints.\n\n"
        "The upgraded system in this workspace uses a cached mp-api ingestion pipeline, PyTorch Geometric graph construction, "
        "an e3nn-based SE(3)-equivariant encoder, a hybrid latent/global-context decoder, permutation-invariant Hungarian matching, "
        "and an iterative coordinate refiner for diffusion-style cleanup during generation."
    )

    dataset_section = (
        f"Materials Project preprocessing is configured for up to {dataset_cfg.get('max_materials')} materials with "
        f"{dataset_cfg.get('max_elements')} maximum unique elements and {dataset_cfg.get('max_atoms')} maximum atoms per crystal. "
        f"Graphs use a cutoff radius of {dataset_cfg.get('cutoff_angstrom')} A and keep up to {dataset_cfg.get('max_neighbors')} neighbors per atom.\n\n"
        "Each cached record stores material_id, species, fractional coordinates, lattice parameters, and graph tensors. "
        "The graph builder now keeps periodic edge vectors, interatomic distances, Cartesian coordinates, and lattice matrices so "
        "both equivariant message passing and lattice-aware matching operate in physically meaningful coordinates."
    )

    architecture_section = (
        f"Encoder: e3nn SE(3)-equivariant message passing with hidden_dim={model_cfg.get('hidden_dim')} and latent_dim={model_cfg.get('latent_dim')}. "
        "The encoder consumes atom embeddings, periodic edge vectors, and radial distance bases, then pools invariant scalar features into latent mean/logvar.\n\n"
        "Decoder: hybrid latent/global-context decoder with learned slot embeddings and anchor-based residual coordinate prediction. "
        "The decoder predicts atom count, atom types, fractional coordinates, and a constrained lower-triangular 3x3 lattice matrix with positive determinant.\n\n"
        "Refinement: a coordinate refinement network conditions on noisy coordinates, atom-type probabilities, latent vectors, and global context to iteratively denoise generated structures."
    )

    loss_section = (
        "The reconstruction objective is permutation invariant. Hungarian matching is performed with a lattice-aware, minimum-image cost in Cartesian space plus an atom-type mismatch penalty.\n\n"
        "Loss terms currently include atom-count cross entropy, matched atom-type cross entropy, matched coordinate MSE, lattice-matrix MSE, KL divergence, distance consistency, symmetry-consistency aliasing, "
        "refinement denoising loss, minimum separation penalty, unmatched-slot penalty, repulsive energy proxy, and lattice-volume constraint.\n\n"
        f"Configured weights: {json.dumps(training_cfg.get('loss_weights', {}), indent=2)}"
    )

    training_section = (
        f"Optimization uses AdamW with learning_rate={training_cfg.get('learning_rate')} and weight_decay={training_cfg.get('weight_decay')}. "
        f"Gradient clipping is enabled with max norm {training_cfg.get('grad_clip_norm')}, and ReduceLROnPlateau remains active with factor {training_cfg.get('lr_decay_factor')} and patience {training_cfg.get('lr_patience')}.\n\n"
        f"KL annealing configuration: {json.dumps(training_cfg.get('kl_annealing', {}), indent=2)}\n\n"
        f"Curriculum schedule: {json.dumps(training_cfg.get('curriculum_stages', []), indent=2)}"
    )

    results_section = summarize_results(history, generated_metadata)

    limitations_section = (
        "The implementation is substantially closer to a research-grade CDVAE, but several limitations remain. "
        "The decoder is still only partially equivariant, the refinement network is diffusion-inspired rather than a full denoising diffusion probabilistic model, "
        "and current generated smoke-test samples still reveal unrealistic lattice volumes when the model is untrained.\n\n"
        "In addition, final scientific evaluation depends on running the full curriculum on a large cached Materials Project dataset and measuring novelty, validity, and stability against baselines."
    )

    future_work_section = (
        "Natural next steps are a fully equivariant decoder, stronger diffusion refinement over both coordinates and lattice, symmetry-aware augmentation, richer chemistry priors, "
        "and post-generation ranking using surrogate formation-energy or stability predictors.\n\n"
        "Scaling beyond the current setup would also benefit from distributed dataloading, larger datasets (20k to 50k+ structures), and systematic ablation studies for matching, constraints, and refinement depth."
    )

    validation_section = (
        f"Validation report available: {'yes' if validation_report else 'no'}.\n\n"
        f"{json.dumps(validation_report, indent=2) if validation_report else 'No standalone validation JSON was found; generation metadata validation was used instead.'}"
    )

    with PdfPages(output_path) as pdf:
        add_text_page(pdf, "1. Introduction", intro)
        add_text_page(pdf, "2. Dataset", dataset_section)
        add_text_page(pdf, "3. Model Architecture", architecture_section)
        add_text_page(pdf, "4. Loss Functions", loss_section)
        add_text_page(pdf, "5. Training", training_section)
        add_history_plot(pdf, history)
        add_text_page(pdf, "6. Results", results_section)
        add_text_page(pdf, "7. Validation", validation_section)
        add_text_page(pdf, "8. Limitations", limitations_section)
        add_text_page(pdf, "9. Future Work", future_work_section)

    print(f"PDF report written to {output_path}")


if __name__ == "__main__":
    main()
