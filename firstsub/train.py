from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from data.graph_builder import CrystalGraphDataset, GraphBuildConfig
from models import CDVAE, CDVAEModelConfig
from utils.preprocessing import ensure_directory, load_yaml_config, seed_everything, split_indices, tensor_to_python
from utils.visualization import plot_training_curves

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable=None, **kwargs):  # type: ignore[override]
        return iterable if iterable is not None else []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a graph-based CDVAE on cached Materials Project crystals.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the YAML config file.")
    parser.add_argument("--device", type=str, default=None, help="Override device, e.g. cuda or cpu.")
    parser.add_argument("--resume", type=str, default=None, help="Optional checkpoint to resume from.")
    return parser.parse_args()


def resolve_dataset_paths(config: dict[str, Any]) -> tuple[Path, Path]:
    dataset_cfg = config["dataset"]
    processed_jsonl = dataset_cfg.get("processed_jsonl")
    species_vocab = dataset_cfg.get("species_vocab")
    if processed_jsonl and species_vocab:
        processed_path = Path(processed_jsonl)
        species_path = Path(species_vocab)
        if not processed_path.exists() or not species_path.exists():
            raise FileNotFoundError(
                "Configured dataset artifacts were not found. Expected "
                f"`{processed_path}` and `{species_path}`. "
                "Either build the dataset first with `python -m data.mp_dataset ...` "
                "or update `config.yaml` to point at the correct cache paths."
            )
        return processed_path, species_path

    from data.mp_dataset import MPDownloadConfig, MaterialsProjectDatasetBuilder

    try:
        builder = MaterialsProjectDatasetBuilder(
            MPDownloadConfig(
                api_key=dataset_cfg.get("api_key"),
                cache_root=Path(dataset_cfg.get("cache_root", "cache/mp")),
                dataset_name=dataset_cfg.get("dataset_name", "mp_structures"),
                max_materials=dataset_cfg.get("max_materials", 10_000),
                max_elements=dataset_cfg.get("max_elements", 3),
                max_atoms=dataset_cfg.get("max_atoms", 50),
                chunk_size=dataset_cfg.get("chunk_size", 500),
                include_gnome=dataset_cfg.get("include_gnome", False),
                include_deprecated=dataset_cfg.get("include_deprecated", False),
                allow_disordered=dataset_cfg.get("allow_disordered", False),
                timeout_seconds=dataset_cfg.get("timeout_seconds", 60),
            )
        )
        artifacts = builder.build(force_refresh=False)
    except Exception as exc:
        raise RuntimeError(
            "Dataset resolution failed. Provide existing `processed_jsonl` and `species_vocab` paths in "
            "`config.yaml`, or set `MP_API_KEY` and let the downloader rebuild the cache."
        ) from exc
    return artifacts.processed_jsonl_path, artifacts.species_vocab_path


def load_dataset_and_splits(config: dict[str, Any]) -> tuple[CrystalGraphDataset, list[int], list[int]]:
    processed_jsonl_path, species_vocab_path = resolve_dataset_paths(config)
    dataset_cfg = config["dataset"]
    graph_dataset = CrystalGraphDataset(
        processed_jsonl_path=processed_jsonl_path,
        species_vocab_path=species_vocab_path,
        graph_config=GraphBuildConfig(
            cutoff_angstrom=dataset_cfg["cutoff_angstrom"],
            max_neighbors=dataset_cfg["max_neighbors"],
            graph_cache_dir=Path(dataset_cfg["graph_cache_dir"]) if dataset_cfg.get("graph_cache_dir") else None,
            force_rebuild=dataset_cfg.get("force_rebuild_graphs", False),
        ),
    )

    train_indices, val_indices = split_indices(
        dataset_size=len(graph_dataset),
        val_ratio=dataset_cfg["val_ratio"],
        seed=config["seed"],
    )

    if dataset_cfg.get("precompute_graphs", True):
        graph_dataset.precompute_graphs(indices=train_indices + val_indices)

    return graph_dataset, train_indices, val_indices


def build_dataloader(
    dataset: CrystalGraphDataset,
    indices: list[int],
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    subset = Subset(dataset, indices)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def compute_volume_bounds(
    dataset: CrystalGraphDataset,
    indices: list[int],
    margin_fraction: float = 0.1,
) -> tuple[float, float]:
    volumes: list[float] = []
    iterator = tqdm(indices, desc="Computing volume bounds", unit="graph")
    for index in iterator:
        graph = dataset[index]
        volume = float(torch.abs(torch.det(graph.lattice_matrix.squeeze(0))).item())
        volumes.append(volume)

    min_volume = min(volumes)
    max_volume = max(volumes)
    span = max(max_volume - min_volume, 1e-3)
    min_volume = max(0.1, min_volume - margin_fraction * span)
    max_volume = max_volume + margin_fraction * span
    return min_volume, max_volume


def expand_curriculum_plan(training_cfg: dict[str, Any], total_train_samples: int) -> list[dict[str, Any]]:
    curriculum_stages = training_cfg.get("curriculum_stages") or []
    if not curriculum_stages:
        return [
            {
                "name": "full",
                "max_samples": total_train_samples,
                "epochs": int(training_cfg["epochs"]),
            }
        ]

    plan: list[dict[str, Any]] = []
    for stage_index, stage in enumerate(curriculum_stages, start=1):
        stage_epochs = int(stage["epochs"])
        plan.extend(
            [
                {
                    "name": stage.get("name", f"stage_{stage_index}"),
                    "max_samples": min(int(stage.get("max_samples", total_train_samples)), total_train_samples),
                }
            ]
            * stage_epochs
        )
    return plan


def compute_kl_beta(training_cfg: dict[str, Any], epoch: int, total_epochs: int) -> float:
    annealing_cfg = training_cfg.get("kl_annealing", {})
    start_beta = float(annealing_cfg.get("start_beta", 0.0))
    end_beta = float(annealing_cfg.get("end_beta", training_cfg["kl_beta"]))
    warmup_epochs = int(annealing_cfg.get("warmup_epochs", max(1, total_epochs // 5)))
    if warmup_epochs <= 1:
        return end_beta
    progress = min(max((epoch - 1) / (warmup_epochs - 1), 0.0), 1.0)
    return start_beta + progress * (end_beta - start_beta)


def run_epoch(
    model: CDVAE,
    loader: DataLoader,
    optimizer: AdamW | None,
    device: torch.device,
    loss_weights: dict[str, float],
    grad_clip_norm: float,
    kl_beta: float,
) -> dict[str, float]:
    is_training = optimizer is not None
    model.train(is_training)
    aggregated: dict[str, float] = {}
    total_graphs = 0

    iterator = tqdm(loader, desc="Train" if is_training else "Validate", unit="batch")
    for batch in iterator:
        batch = batch.to(device)
        if is_training:
            optimizer.zero_grad(set_to_none=True)

        outputs = model(batch)
        loss_dict = model.compute_losses(batch, outputs, loss_weights=loss_weights, kl_beta=kl_beta)
        loss = loss_dict["loss"]

        if is_training:
            loss.backward()
            clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        batch_graphs = int(batch.num_graphs)
        total_graphs += batch_graphs
        for key, value in tensor_to_python(loss_dict).items():
            aggregated[key] = aggregated.get(key, 0.0) + float(value) * batch_graphs

        running_loss = aggregated["loss"] / max(total_graphs, 1)
        iterator.set_postfix(loss=f"{running_loss:.4f}")

    return {key: value / max(total_graphs, 1) for key, value in aggregated.items()}


def save_checkpoint(
    path: Path,
    model: CDVAE,
    optimizer: AdamW,
    scheduler: ReduceLROnPlateau,
    epoch: int,
    best_val_loss: float,
    config: dict[str, Any],
    species_vocab: dict[str, int],
    volume_min: float,
    volume_max: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "config": config,
            "model_config": model.config.__dict__,
            "species_vocab": species_vocab,
            "lattice_min": model.lattice_min.detach().cpu(),
            "lattice_max": model.lattice_max.detach().cpu(),
            "volume_min": volume_min,
            "volume_max": volume_max,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    seed_everything(config["seed"])

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    run_dir = ensure_directory(config["paths"]["run_dir"])
    checkpoint_dir = ensure_directory(run_dir / "checkpoints")
    history_path = run_dir / "history.jsonl"

    graph_dataset, train_indices, val_indices = load_dataset_and_splits(config)
    val_loader = build_dataloader(
        dataset=graph_dataset,
        indices=val_indices,
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
        shuffle=False,
    )
    lattice_min, lattice_max = graph_dataset.compute_lattice_bounds(train_indices)
    volume_min, volume_max = compute_volume_bounds(graph_dataset, train_indices)

    model_config = CDVAEModelConfig(
        num_species=graph_dataset.num_species,
        max_atoms=config["dataset"]["max_atoms"],
        hidden_dim=config["model"]["hidden_dim"],
        latent_dim=config["model"]["latent_dim"],
        num_encoder_layers=config["model"]["num_encoder_layers"],
        num_rbf=config["model"]["num_rbf"],
        cutoff=config["dataset"]["cutoff_angstrom"],
        dropout=config["model"]["dropout"],
        kl_beta=config["training"]["kl_beta"],
        min_distance_angstrom=config["training"]["min_distance_angstrom"],
    )
    model = CDVAE(model_config).to(device)
    model.set_lattice_bounds(lattice_min.to(device), lattice_max.to(device))
    model.set_volume_bounds(volume_min, volume_max)

    optimizer = AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(config["training"]["lr_decay_factor"]),
        patience=int(config["training"]["lr_patience"]),
    )
    start_epoch = 1
    best_val_loss = float("inf")

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val_loss = float(checkpoint["best_val_loss"])
        if "volume_min" in checkpoint and "volume_max" in checkpoint:
            volume_min = float(checkpoint["volume_min"])
            volume_max = float(checkpoint["volume_max"])
            model.set_volume_bounds(volume_min, volume_max)

    (run_dir / "config_resolved.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (run_dir / "splits.json").write_text(
        json.dumps({"train_indices": train_indices, "val_indices": val_indices}, indent=2),
        encoding="utf-8",
    )

    if history_path.exists() and start_epoch == 1:
        history_path.unlink()

    curriculum_plan = expand_curriculum_plan(config["training"], len(train_indices))
    total_epochs = len(curriculum_plan)
    current_stage_name: str | None = None
    current_stage_samples: int | None = None
    train_loader: DataLoader | None = None

    for epoch in range(start_epoch, total_epochs + 1):
        stage = curriculum_plan[epoch - 1]
        stage_name = stage["name"]
        stage_samples = min(int(stage["max_samples"]), len(train_indices))
        if train_loader is None or stage_name != current_stage_name or stage_samples != current_stage_samples:
            stage_train_indices = train_indices[:stage_samples]
            train_loader = build_dataloader(
                dataset=graph_dataset,
                indices=stage_train_indices,
                batch_size=config["training"]["batch_size"],
                num_workers=config["training"]["num_workers"],
                shuffle=True,
            )
            current_stage_name = stage_name
            current_stage_samples = stage_samples

        active_kl_beta = compute_kl_beta(config["training"], epoch, total_epochs)
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            loss_weights=config["training"]["loss_weights"],
            grad_clip_norm=float(config["training"]["grad_clip_norm"]),
            kl_beta=active_kl_beta,
        )
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            device=device,
            loss_weights=config["training"]["loss_weights"],
            grad_clip_norm=float(config["training"]["grad_clip_norm"]),
            kl_beta=active_kl_beta,
        )
        scheduler.step(val_metrics["loss"])

        summary: dict[str, Any] = {
            "epoch": epoch,
            "stage": stage_name,
            "stage_samples": stage_samples,
            "kl_beta": active_kl_beta,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        for key, value in train_metrics.items():
            summary[f"train_{key}"] = value
        for key, value in val_metrics.items():
            summary[f"val_{key}"] = value
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(summary) + "\n")

        save_checkpoint(
            path=checkpoint_dir / "last_model.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_val_loss=best_val_loss,
            config=config,
            species_vocab=graph_dataset.species_vocab,
            volume_min=volume_min,
            volume_max=volume_max,
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = float(val_metrics["loss"])
            save_checkpoint(
                path=checkpoint_dir / "best_model.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_val_loss=best_val_loss,
                config=config,
                species_vocab=graph_dataset.species_vocab,
                volume_min=volume_min,
                volume_max=volume_max,
            )

        print(
            f"Epoch {epoch:03d}/{total_epochs:03d} | "
            f"stage={stage_name} ({stage_samples} samples) | "
            f"kl_beta={active_kl_beta:.6f} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_count_acc={val_metrics['count_accuracy']:.4f} | "
            f"val_atom_acc={val_metrics['atom_type_accuracy']:.4f}"
        )

    plot_training_curves(history_path, run_dir / "training_curves.png")


if __name__ == "__main__":
    main()
