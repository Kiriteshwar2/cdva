from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class APIMessage(BaseModel):
    message: str


class MongoModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True)

    id: str = Field(alias="_id")


class HealthResponse(BaseModel):
    status: str
    database: str
    model_loaded: bool
    model_path: str | None = None
    device: str
    version: str


class PaginationMeta(BaseModel):
    total: int
    limit: int


class TimestampedResponse(BaseModel):
    created_at: datetime


JSONDict = dict[str, Any]
