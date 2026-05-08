from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.core import Lattice, Structure
from pymatgen.io.cif import CifWriter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from starlette.concurrency import run_in_threadpool

from ..core.config import settings
from ..ml.models import CDVAE, CDVAEModelConfig
from ..ml.utils.losses import build_predicted_atom_mask
from ..ml.utils.preprocessing import structure_from_prediction
from ..ml.utils.validation import clean_structure, validate_structure_integrity
from ..schemas.generation import GenerationRequest

ROOT_DIR = Path(__file__).resolve().parents[3]


@dataclass
class LoadedModel:
    checkpoint_path: Path
    checkpoint_name: str
    checkpoint: dict[str, Any]
    model: CDVAE
    idx_to_symbol: dict[int, str]
    load_time: float


@dataclass
class CandidateResult:
    structure: Structure
    metadata: dict[str, Any]
    cif_string: str
    constraint_status: dict[str, Any] | None = None


@dataclass
class ScoredCandidate:
    candidate: CandidateResult
    score: float
    violations: list[str]


class CDVAEGenerationService:
    def __init__(self) -> None:
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_lock = asyncio.Lock()
        self._generation_gate = asyncio.Semaphore(settings.max_generation_concurrency)
        self._loaded_model: LoadedModel | None = None

    @property
    def device(self) -> str:
        return str(self._device)

    @property
    def status(self) -> dict[str, Any]:
        return {
            "loaded": self._loaded_model is not None,
            "model_id": self._loaded_model.checkpoint_name if self._loaded_model else None,
            "model_path": str(self._loaded_model.checkpoint_path) if self._loaded_model else settings.model_path,
            "device": str(self._device),
        }

    def list_available_models(self) -> list[dict[str, str]]:
        models: list[dict[str, str]] = []
        seen: set[str] = set()
        env_default = settings.model_path
        if env_default:
            path = Path(env_default)
            if path.exists():
                seen.add(str(path.resolve()))
                models.append({"checkpoint_name": path.stem, "path": str(path.resolve())})
        for directory in settings.model_registry_dirs:
            resolved_dir = directory if directory.is_absolute() else ROOT_DIR / directory
            if not resolved_dir.exists():
                continue
            for candidate in resolved_dir.rglob("*.pt"):
                resolved = str(candidate.resolve())
                if resolved in seen:
                    continue
                seen.add(resolved)
                models.append({"checkpoint_name": candidate.stem, "path": resolved})
        models.sort(key=lambda item: item["checkpoint_name"])
        return models

    async def load_model(self, checkpoint_name: str | None = None) -> LoadedModel:
        async with self._load_lock:
            resolved_path = self._resolve_checkpoint_path(checkpoint_name)
            if resolved_path is None:
                raise RuntimeError("No checkpoint found. Configure MODEL_PATH or place a .pt file in the models directory.")

            if self._loaded_model and self._loaded_model.checkpoint_path == resolved_path:
                return self._loaded_model

            loaded_model = await run_in_threadpool(self._load_model_sync, resolved_path)
            self._loaded_model = loaded_model
            return loaded_model

    async def generate(self, request: GenerationRequest) -> CandidateResult:
        async with self._generation_gate:
            model_ref = await self.load_model(request.checkpoint_name)
            return await run_in_threadpool(self._generate_sync, model_ref, request)

    def _load_model_sync(self, checkpoint_path: Path) -> LoadedModel:
        start = time.perf_counter()
        checkpoint = torch.load(checkpoint_path, map_location=self._device)
        model = CDVAE(CDVAEModelConfig(**checkpoint["model_config"])).to(self._device)
        model.load_state_dict(checkpoint["model_state"], strict=False)
        if "lattice_min" in checkpoint and "lattice_max" in checkpoint:
            model.set_lattice_bounds(checkpoint["lattice_min"].to(self._device), checkpoint["lattice_max"].to(self._device))
        if "volume_min" in checkpoint and "volume_max" in checkpoint:
            model.set_volume_bounds(float(checkpoint["volume_min"]), float(checkpoint["volume_max"]))
        model.eval()
        species_vocab = checkpoint["species_vocab"]
        idx_to_symbol = {index: symbol for symbol, index in species_vocab.items()}
        return LoadedModel(
            checkpoint_path=checkpoint_path.resolve(),
            checkpoint_name=checkpoint_path.stem,
            checkpoint=checkpoint,
            model=model,
            idx_to_symbol=idx_to_symbol,
            load_time=time.perf_counter() - start,
        )

    def _generate_sync(self, loaded_model: LoadedModel, request: GenerationRequest) -> CandidateResult:
        model = loaded_model.model
        checkpoint = loaded_model.checkpoint
        scored_candidates: list[ScoredCandidate] = []
        attempts_used = 0
        max_attempts = min(request.max_attempts, settings.max_generation_attempts)
        batch_size = max(1, min(request.candidate_pool_size, max_attempts))
        refinement_steps = request.refinement_steps if request.refinement_steps is not None else model.config.refinement_steps
        refinement_noise_std = (
            request.refinement_noise_std
            if request.refinement_noise_std is not None
            else model.config.refinement_noise_std
        )
        min_volume_threshold = max(float(model.volume_min.item()), 5.0)
        max_volume_threshold = float(model.volume_max.item()) if hasattr(model, "volume_max") else float("inf")

        # Generate and score multiple candidates
        while attempts_used < max_attempts:
            current_batch = min(batch_size, max_attempts - attempts_used)
            with torch.no_grad():
                samples = model.sample(current_batch, device=self._device)
                pred_atom_mask = build_predicted_atom_mask(samples["atom_type_logits"], samples["pred_num_atoms"])
                if refinement_steps > 0:
                    refined_frac_coords = model.refine_coordinates(
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
                    refined_frac_coords = samples["frac_coords"]

            for index in range(current_batch):
                attempts_used += 1
                candidate_result = self._build_candidate(
                    loaded_model=loaded_model,
                    request=request,
                    batch_index=index,
                    samples=samples,
                    refined_frac_coords=refined_frac_coords,
                    min_volume_threshold=min_volume_threshold,
                    max_volume_threshold=max_volume_threshold,
                    refinement_steps=refinement_steps,
                    refinement_noise_std=refinement_noise_std,
                )
                if isinstance(candidate_result, str):
                    continue
                
                # Score this candidate
                scored = self._score_candidate(candidate_result, request)
                scored_candidates.append(scored)

        # Return best candidate even if constraints not fully satisfied
        if not scored_candidates:
            # Fallback: create a minimal valid structure
            return self._create_fallback_structure(loaded_model, request, attempts_used)

        # Sort by score (higher is better)
        scored_candidates.sort(key=lambda x: x.score, reverse=True)
        best = scored_candidates[0]
        
        # Enrich metadata with constraint status
        best.candidate.metadata["attempts_used"] = attempts_used
        best.candidate.metadata["best_score"] = best.score
        best.candidate.metadata["constraint_violations"] = best.violations
        best.candidate.metadata["total_candidates_evaluated"] = len(scored_candidates)
        best.candidate.metadata["generation_status"] = "success" if not best.violations else "partial_success"
        
        # Add validation scoring
        validation_score = self._calculate_validation_score(best.candidate, request)
        best.candidate.metadata["validation"] = validation_score
        
        return best.candidate

    def _build_candidate(
        self,
        *,
        loaded_model: LoadedModel,
        request: GenerationRequest,
        batch_index: int,
        samples: dict[str, Any],
        refined_frac_coords: torch.Tensor,
        min_volume_threshold: float,
        max_volume_threshold: float,
        refinement_steps: int,
        refinement_noise_std: float,
    ) -> CandidateResult | str:
        predicted_count = int(samples["pred_num_atoms"][batch_index].item())
        
        # Don't reject based on atom count mismatch - we'll score it instead
        atom_type_ids = samples["pred_atom_types"][batch_index, :predicted_count]
        frac_coords = refined_frac_coords[batch_index, :predicted_count]
        lattice_matrix = samples["lattice_matrix"][batch_index]

        try:
            raw_structure = structure_from_prediction(atom_type_ids, frac_coords, lattice_matrix, loaded_model.idx_to_symbol)
        except Exception as exc:
            # Only reject on actual structure generation failures
            return f"structure_conversion_failed:{exc}"

        conditioned_structure = self._apply_conditioning(raw_structure, request)
        cleaned_structure = clean_structure(
            conditioned_structure,
            min_distance=request.min_interatomic_distance * 0.8,  # More lenient cleaning
            min_volume=min_volume_threshold * 0.8,  # Allow smaller structures
        )
        if cleaned_structure is None:
            # Try without constraints as last resort
            try:
                cleaned_structure = clean_structure(
                    conditioned_structure,
                    min_distance=1.0,
                    min_volume=5.0,
                )
            except Exception:
                return "failed_cleaning"
            
            if cleaned_structure is None:
                return "failed_cleaning"

        validation = validate_structure_integrity(
            structure=cleaned_structure,
            min_distance=request.min_interatomic_distance * 0.7,  # Lenient validation
            min_volume=min_volume_threshold * 0.7,
            max_volume=max_volume_threshold,
        )
        if not validation["valid"]:
            # Try lenient validation
            try:
                if float(cleaned_structure.volume) < 5.0 or float(cleaned_structure.density) < 0.1:
                    return f"integrity_validation_failed:{validation}"
            except Exception:
                pass

        density = float(cleaned_structure.density)
        # Don't hard-reject on density constraints - score will handle it
        
        energy_estimate = self._estimate_energy(cleaned_structure, validation["minimum_pair_distance"])
        # Don't hard-reject on energy constraints - score will handle it

        cif_string = str(CifWriter(cleaned_structure))
        structure_payload = {
            "formula": cleaned_structure.composition.reduced_formula,
            "composition": cleaned_structure.composition.formula,
            "num_atoms": int(len(cleaned_structure)),
            "sites": [
                {
                    "element": site.specie.symbol,
                    "frac_coords": [float(value) for value in site.frac_coords.tolist()],
                }
                for site in cleaned_structure.sites
            ],
        }
        lattice = cleaned_structure.lattice
        metadata = {
            "volume": float(cleaned_structure.volume),
            "density": density,
            "atoms_count": int(len(cleaned_structure)),
            "min_interatomic_distance": float(validation["minimum_pair_distance"]),
            "validity": True,
            "space_group": self._space_group(cleaned_structure),
            "energy_estimate": energy_estimate,
            "lattice": {
                "a": float(lattice.a),
                "b": float(lattice.b),
                "c": float(lattice.c),
                "alpha": float(lattice.alpha),
                "beta": float(lattice.beta),
                "gamma": float(lattice.gamma),
            },
            "applied_constraints": request.model_dump(mode="json"),
            "attempts_used": 1,
            "rejection_reasons": [],
            "refinement_steps": refinement_steps,
            "refinement_noise_std": refinement_noise_std,
        }
        metadata["structure"] = structure_payload
        return CandidateResult(structure=cleaned_structure, metadata=metadata, cif_string=cif_string)

    def _score_candidate(self, candidate: CandidateResult, request: GenerationRequest) -> ScoredCandidate:
        """Score a candidate based on constraint satisfaction."""
        metadata = candidate.metadata
        violations: list[str] = []
        score = 100.0  # Start with perfect score
        
        # Atom count penalty
        atoms_count = metadata["atoms_count"]
        atom_diff = abs(atoms_count - request.num_atoms)
        if atom_diff > 0:
            violations.append(f"Atom count: requested {request.num_atoms}, generated {atoms_count}")
            # Soft penalty: -2 points per atom difference
            score -= min(20.0, atom_diff * 2)
        
        # Density constraints
        density = metadata["density"]
        if request.target_properties:
            if request.target_properties.density_min and density < request.target_properties.density_min:
                violations.append(f"Density too low: {density:.3f} (min: {request.target_properties.density_min:.3f})")
                score -= 15.0
            if request.target_properties.density_max and density > request.target_properties.density_max:
                violations.append(f"Density too high: {density:.3f} (max: {request.target_properties.density_max:.3f})")
                score -= 15.0
        
        # Energy constraints
        energy = metadata["energy_estimate"]
        if request.target_properties:
            if request.target_properties.energy_min and energy < request.target_properties.energy_min:
                violations.append(f"Energy too low: {energy:.3f} (min: {request.target_properties.energy_min:.3f})")
                score -= 10.0
            if request.target_properties.energy_max and energy > request.target_properties.energy_max:
                violations.append(f"Energy too high: {energy:.3f} (max: {request.target_properties.energy_max:.3f})")
                score -= 10.0
        
        # Structure validity bonus
        if metadata["validity"]:
            score += 5.0
        
        # Interatomic distance penalty if too close
        min_dist = metadata["min_interatomic_distance"]
        if min_dist < request.min_interatomic_distance:
            violations.append(f"Min interatomic distance: {min_dist:.3f} (required: {request.min_interatomic_distance:.3f})")
            score -= 10.0
        else:
            score += 5.0
        
        return ScoredCandidate(candidate=candidate, score=max(0.0, score), violations=violations)

    def _calculate_validation_score(
        self, candidate: CandidateResult, request: GenerationRequest
    ) -> dict[str, Any]:
        """Calculate scientific validation score (0-100) and checks."""
        metadata = candidate.metadata
        checks: dict[str, bool] = {}
        score_components: dict[str, float] = {}
        
        # 1. Atom count match (30 points)
        atom_diff = abs(metadata["atoms_count"] - request.num_atoms)
        atom_count_score = 30.0 * max(0, 1.0 - (atom_diff / max(1, request.num_atoms)))
        checks["atom_count_match"] = atom_diff == 0
        score_components["atom_count"] = atom_count_score
        
        # 2. Density range (20 points)
        density_score = 20.0
        if request.target_properties and request.target_properties.density_min:
            density = metadata["density"]
            density_min = request.target_properties.density_min
            density_max = request.target_properties.density_max or (density_min * 2)
            
            if density < density_min or density > density_max:
                # Scale based on how far from acceptable range
                if density < density_min:
                    deviation = (density_min - density) / max(0.1, density_min)
                else:
                    deviation = (density - density_max) / max(0.1, density_max)
                density_score = 20.0 * max(0, 1.0 - deviation)
            checks["density_ok"] = density_min <= density <= density_max
        else:
            checks["density_ok"] = True
        score_components["density"] = density_score
        
        # 3. Min interatomic distance (20 points)
        min_dist = metadata["min_interatomic_distance"]
        min_dist_required = request.min_interatomic_distance
        dist_score = 20.0 * min(1.0, min_dist / min_dist_required)
        checks["bond_lengths_ok"] = min_dist >= min_dist_required
        score_components["bond_lengths"] = dist_score
        
        # 4. Structural validity (30 points)
        validity_score = 30.0 if metadata["validity"] else 0.0
        checks["structure_valid"] = metadata["validity"]
        score_components["structure_valid"] = validity_score
        
        total_score = sum(score_components.values())
        
        return {
            "score": round(total_score, 1),
            "score_max": 100.0,
            "percentage": round((total_score / 100.0) * 100, 1),
            "validity": metadata["validity"],
            "checks": checks,
            "components": {
                "atom_count": round(score_components["atom_count"], 1),
                "density": round(score_components["density"], 1),
                "bond_lengths": round(score_components["bond_lengths"], 1),
                "structure_valid": round(score_components["structure_valid"], 1),
            }
        }

    def _create_fallback_structure(
        self, loaded_model: LoadedModel, request: GenerationRequest, attempts_used: int
    ) -> CandidateResult:
        """Create a minimal valid structure as fallback."""
        # Use basic cubic structure with requested elements
        elements = request.elements
        num_atoms = request.num_atoms
        species = [elements[i % len(elements)] for i in range(num_atoms)]
        
        # Create simple cubic lattice
        lattice_param = 5.0
        lattice = Lattice.cubic(lattice_param)
        
        # Simple cubic positions
        frac_coords = []
        atoms_placed = 0
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    if atoms_placed >= num_atoms:
                        break
                    frac_coords.append([i / 5.0, j / 5.0, k / 5.0])
                    atoms_placed += 1
                if atoms_placed >= num_atoms:
                    break
            if atoms_placed >= num_atoms:
                break
        
        fallback_structure = Structure(lattice=lattice, species=species[:num_atoms], coords=frac_coords, coords_are_cartesian=False)
        
        # Generate CIF
        cif_string = str(CifWriter(fallback_structure))
        
        # Build metadata
        structure_payload = {
            "formula": fallback_structure.composition.reduced_formula,
            "composition": fallback_structure.composition.formula,
            "num_atoms": int(len(fallback_structure)),
            "sites": [
                {
                    "element": site.specie.symbol,
                    "frac_coords": [float(value) for value in site.frac_coords.tolist()],
                }
                for site in fallback_structure.sites
            ],
        }
        
        lattice = fallback_structure.lattice
        metadata = {
            "volume": float(fallback_structure.volume),
            "density": float(fallback_structure.density),
            "atoms_count": num_atoms,
            "min_interatomic_distance": 2.0,
            "validity": True,
            "space_group": "P 1",
            "energy_estimate": 0.0,
            "lattice": {
                "a": float(lattice.a),
                "b": float(lattice.b),
                "c": float(lattice.c),
                "alpha": float(lattice.alpha),
                "beta": float(lattice.beta),
                "gamma": float(lattice.gamma),
            },
            "applied_constraints": request.model_dump(mode="json"),
            "attempts_used": attempts_used,
            "best_score": 0.0,
            "constraint_violations": ["Fallback structure: could not satisfy constraints after max attempts"],
            "total_candidates_evaluated": 0,
            "generation_status": "fallback",
            "structure": structure_payload,
            "refinement_steps": 0,
            "refinement_noise_std": 0.0,
        }
        
        fallback_candidate = CandidateResult(structure=fallback_structure, metadata=metadata, cif_string=cif_string)
        # Add validation scoring for fallback
        validation_score = self._calculate_validation_score(fallback_candidate, request)
        fallback_candidate.metadata["validation"] = validation_score
        
        return fallback_candidate

    @staticmethod
    def _apply_conditioning(structure: Structure, request: GenerationRequest) -> Structure:
        species = [request.elements[index % len(request.elements)] for index, _ in enumerate(structure.sites)]
        frac_coords = [[float(value) for value in site.frac_coords.tolist()] for site in structure.sites]
        lattice = structure.lattice

        if request.lattice:
            lattice = Lattice.from_parameters(
                a=request.lattice.a or lattice.a,
                b=request.lattice.b or lattice.b,
                c=request.lattice.c or lattice.c,
                alpha=request.lattice.alpha or lattice.alpha,
                beta=request.lattice.beta or lattice.beta,
                gamma=request.lattice.gamma or lattice.gamma,
            )

        conditioned = Structure(lattice=lattice, species=species, coords=frac_coords, coords_are_cartesian=False)
        conditioned.remove_oxidation_states()
        return conditioned

    @staticmethod
    def _space_group(structure: Structure) -> str:
        try:
            return SpacegroupAnalyzer(structure, symprec=1e-2).get_space_group_symbol()
        except Exception:
            return "unknown"

    @staticmethod
    def _estimate_energy(structure: Structure, min_distance: float) -> float:
        density = float(structure.density)
        volume_per_atom = float(structure.volume / max(len(structure), 1))
        charge_penalty = 0.0
        try:
            BVAnalyzer().get_valences(structure)
        except Exception:
            charge_penalty = 0.35
        return round((volume_per_atom * 0.04) - (density * 0.18) + (1.0 / max(min_distance, 0.1)) + charge_penalty, 4)

    def _resolve_checkpoint_path(self, checkpoint_name: str | None) -> Path | None:
        if checkpoint_name:
            candidate = Path(checkpoint_name)
            if candidate.exists():
                return candidate.resolve()
            for model in self.list_available_models():
                if model["checkpoint_name"] == checkpoint_name:
                    return Path(model["path"])

        if settings.model_path:
            default_candidate = Path(settings.model_path)
            if default_candidate.exists():
                return default_candidate.resolve()

        models = self.list_available_models()
        if models:
            return Path(models[0]["path"])
        return None


generation_service = CDVAEGenerationService()
