from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from pymatgen.core import Structure


def plot_structure(structure: Structure, output_path: str | Path, title: str | None = None) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cart_coords = structure.cart_coords
    species = [site.specie.symbol for site in structure.sites]
    unique_species = sorted(set(species))
    cmap = plt.get_cmap("tab20", max(len(unique_species), 1))
    color_map = {symbol: cmap(index) for index, symbol in enumerate(unique_species)}

    figure = plt.figure(figsize=(6, 5))
    axis = figure.add_subplot(111, projection="3d")
    for symbol in unique_species:
        mask = [site_symbol == symbol for site_symbol in species]
        coords = cart_coords[mask]
        axis.scatter(coords[:, 0], coords[:, 1], coords[:, 2], label=symbol, s=50, color=color_map[symbol])

    axis.set_xlabel("x (A)")
    axis.set_ylabel("y (A)")
    axis.set_zlabel("z (A)")
    axis.set_title(title or "Crystal Structure")
    axis.legend(loc="upper right", fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    return output_path


def plot_training_curves(history_path: str | Path, output_path: str | Path) -> Path:
    history_path = Path(history_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    epochs: list[int] = []
    train_losses: list[float] = []
    val_losses: list[float] = []

    with history_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            epochs.append(int(record["epoch"]))
            train_losses.append(float(record["train_loss"]))
            val_losses.append(float(record["val_loss"]))

    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(epochs, train_losses, label="Train")
    axis.plot(epochs, val_losses, label="Validation")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.set_title("CDVAE Training Curves")
    axis.legend()
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    return output_path
