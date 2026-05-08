from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class LatticeConstraint(BaseModel):
    a: float | None = Field(default=None, gt=0)
    b: float | None = Field(default=None, gt=0)
    c: float | None = Field(default=None, gt=0)
    alpha: float | None = Field(default=None, gt=0, lt=180)
    beta: float | None = Field(default=None, gt=0, lt=180)
    gamma: float | None = Field(default=None, gt=0, lt=180)


class TargetPropertiesConstraint(BaseModel):
    energy_min: float | None = None
    energy_max: float | None = None
    density_min: float | None = Field(default=None, gt=0)
    density_max: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_ranges(self) -> "TargetPropertiesConstraint":
        if self.energy_min is not None and self.energy_max is not None and self.energy_min > self.energy_max:
            raise ValueError("energy_min must be less than or equal to energy_max.")
        if self.density_min is not None and self.density_max is not None and self.density_min > self.density_max:
            raise ValueError("density_min must be less than or equal to density_max.")
        return self


class GenerationRequest(BaseModel):
    checkpoint_name: str | None = None
    elements: list[str] = Field(min_length=1, max_length=8)
    num_atoms: int = Field(ge=1, le=128)
    lattice: LatticeConstraint | None = None
    target_properties: TargetPropertiesConstraint | None = None
    min_interatomic_distance: float = Field(default=1.2, gt=0.2, le=10.0)
    candidate_pool_size: int = Field(default=24, ge=1, le=128)
    max_attempts: int = Field(default=64, ge=1, le=256)
    refinement_steps: int | None = Field(default=None, ge=0, le=500)
    refinement_noise_std: float | None = Field(default=None, ge=0, le=5.0)

    @field_validator("elements")
    @classmethod
    def normalize_elements(cls, value: list[str]) -> list[str]:
        normalized = []
        for element in value:
            token = element.strip()
            if not token:
                continue
            normalized.append(token[0].upper() + token[1:].lower())
        if not normalized:
            raise ValueError("At least one valid chemical element is required.")
        return normalized


class StructureSite(BaseModel):
    element: str
    frac_coords: list[float]


class StructurePayload(BaseModel):
    formula: str
    composition: str
    num_atoms: int
    sites: list[StructureSite]


class GenerationMetadata(BaseModel):
    volume: float
    density: float
    atoms_count: int
    min_interatomic_distance: float
    validity: bool
    space_group: str
    energy_estimate: float
    lattice: dict[str, float]
    applied_constraints: dict[str, Any]
    attempts_used: int
    rejection_reasons: list[str]
    refinement_steps: int
    refinement_noise_std: float
    best_score: float | None = None
    constraint_violations: list[str] | None = None
    total_candidates_evaluated: int | None = None
    generation_status: str | None = None
    validation: dict[str, Any] | None = None


class GenerationResponse(BaseModel):
    id: str
    user_id: str
    checkpoint_name: str
    output_cif: str
    structure: StructurePayload
    metadata: GenerationMetadata
    input_parameters: dict[str, Any]
    created_at: datetime


class GenerationListResponse(BaseModel):
    items: list[GenerationResponse]
    total: int


class ModelStatusResponse(BaseModel):
    checkpoint_name: str
    path: str
    loaded_status: bool
    last_loaded_at: datetime | None = None
