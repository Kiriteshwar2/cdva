from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    checkpoint_path: str | None = None
    device: str


class GenerateRequest(BaseModel):
    num_samples: int = Field(default=4, ge=1, le=32)


class GeneratedStructureResponse(BaseModel):
    id: str
    atom_types: list[str]
    frac_coords: list[list[float]]
    lattice: list[list[float]]
    cif_string: str
    valid: bool
    metadata: dict[str, Any]


class GenerateResponse(BaseModel):
    structures: list[GeneratedStructureResponse]
