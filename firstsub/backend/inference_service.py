from __future__ import annotations

import os
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from pymatgen.io.cif import CifWriter

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models import CDVAE, CDVAEModelConfig  # noqa: E402
from utils.losses import build_predicted_atom_mask  # noqa: E402
from utils.preprocessing import structure_from_prediction  # noqa: E402
from utils.validation import validate_structure_integrity  # noqa: E402


@dataclass(frozen=True)
class ServiceStatus:
    loaded: bool
    checkpoint_path: str | None
    device: str


class CDVAEInferenceService:
    def __init__(self, checkpoint_path: str | Path | None = None, device: str | None = None) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._checkpoint_path = self._resolve_checkpoint_path(checkpoint_path)
        self._checkpoint: dict[str, Any] | None = None
        self._model: CDVAE | None = None
        self._idx_to_symbol: dict[int, str] | None = None
        if self._checkpoint_path is not None:
            self._load_checkpoint(self._checkpoint_path)

    @property
    def status(self) -> ServiceStatus:
        return ServiceStatus(
            loaded=self._model is not None,
            checkpoint_path=str(self._checkpoint_path) if self._checkpoint_path else None,
            device=str(self.device),
        )

    def generate_structures(self, num_samples: int) -> list[dict[str, Any]]:
        if self._model is None or self._checkpoint is None or self._idx_to_symbol is None:
            raise RuntimeError(
                "CDVAE checkpoint is not loaded. Set CDVAE_CHECKPOINT_PATH or place a checkpoint in runs/cdvae/checkpoints."
            )

        self._model.eval()
        with torch.no_grad():
            samples = self._model.sample(num_samples, device=self.device)
            pred_atom_mask = build_predicted_atom_mask(samples["atom_type_logits"], samples["pred_num_atoms"])
            generation_cfg = self._checkpoint.get("config", {}).get("generation", {})
            refinement_steps = int(generation_cfg.get("refinement_steps", self._model.config.refinement_steps))
            refinement_noise_std = float(
                generation_cfg.get("refinement_noise_std", self._model.config.refinement_noise_std)
            )
            refined_frac_coords = self._model.refine_coordinates(
                frac_coords=samples["frac_coords"],
                atom_type_logits=samples["atom_type_logits"],
                latent=samples["latent"],
                global_context=samples["graph_context"],
                lattice_matrix=samples["lattice_matrix"],
                atom_mask=pred_atom_mask,
                num_steps=refinement_steps,
                noise_std=refinement_noise_std,
            )

        responses: list[dict[str, Any]] = []
        for index in range(num_samples):
            predicted_count = int(samples["pred_num_atoms"][index].item())
            predicted_count = max(1, min(predicted_count, samples["pred_atom_types"].size(1)))
            atom_type_ids = samples["pred_atom_types"][index, :predicted_count]
            frac_coords = refined_frac_coords[index, :predicted_count]
            lattice_matrix = samples["lattice_matrix"][index]
            structure = structure_from_prediction(atom_type_ids, frac_coords, lattice_matrix, self._idx_to_symbol)
            validation = validate_structure_integrity(
                structure=structure,
                min_distance=float(self._model.config.min_distance_angstrom),
                min_volume=float(self._model.volume_min.item()),
                max_volume=float(self._model.volume_max.item()),
            )
            cif_string = str(CifWriter(structure))
            responses.append(
                {
                    "id": f"crystal-{uuid.uuid4().hex[:12]}",
                    "atom_types": [site.specie.symbol for site in structure.sites],
                    "frac_coords": [[float(value) for value in coords] for coords in structure.frac_coords.tolist()],
                    "lattice": [[float(value) for value in row] for row in structure.lattice.matrix.tolist()],
                    "cif_string": cif_string,
                    "valid": bool(validation["valid"]),
                    "metadata": {
                        "num_atoms": int(len(structure)),
                        "formula": structure.composition.reduced_formula,
                        "volume": float(structure.volume),
                        "density": float(structure.density),
                        "minimum_pair_distance": float(validation["minimum_pair_distance"]),
                        "lattice_validation": validation["lattice"],
                        "fractional_validation": validation["fractional_coordinates"],
                        "roundtrip_ok": bool(validation["roundtrip_ok"]),
                        "refinement_steps": refinement_steps,
                        "refinement_noise_std": refinement_noise_std,
                    },
                }
            )
        return responses

    def _load_checkpoint(self, checkpoint_path: Path) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model = CDVAE(CDVAEModelConfig(**checkpoint["model_config"])).to(self.device)
        model.load_state_dict(checkpoint["model_state"], strict=False)
        if "lattice_min" in checkpoint and "lattice_max" in checkpoint:
            model.set_lattice_bounds(checkpoint["lattice_min"].to(self.device), checkpoint["lattice_max"].to(self.device))
        if "volume_min" in checkpoint and "volume_max" in checkpoint:
            model.set_volume_bounds(float(checkpoint["volume_min"]), float(checkpoint["volume_max"]))
        model.eval()

        species_vocab = checkpoint["species_vocab"]
        idx_to_symbol = {index: symbol for symbol, index in species_vocab.items()}

        self._checkpoint = checkpoint
        self._model = model
        self._idx_to_symbol = idx_to_symbol

    @staticmethod
    def _resolve_checkpoint_path(checkpoint_path: str | Path | None) -> Path | None:
        candidates: list[Path] = []
        if checkpoint_path:
            candidates.append(Path(checkpoint_path))

        env_path = os.getenv("CDVAE_CHECKPOINT_PATH")
        if env_path:
            candidates.append(Path(env_path))

        candidates.extend(
            [
                ROOT_DIR / "runs" / "cdvae" / "checkpoints" / "best_model.pt",
                ROOT_DIR / "runs" / "cdvae" / "checkpoints" / "last_model.pt",
                ROOT_DIR / "backend" / "checkpoints" / "best_model.pt",
            ]
        )

        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        return None
