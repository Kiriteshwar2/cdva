from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "CDVAE Crystal Platform"
    api_prefix: str = ""
    env: Literal["development", "staging", "production"] = "development"
    jwt_secret: str = Field(default="change-me-in-production", min_length=16)
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 12

    mongo_uri: str = "mongodb://localhost:27017"
    mongodb_database: str = "cdvae_platform"

    cors_allowed_origins: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ]

    model_path: str | None = None
    max_generation_attempts: int = 96
    max_generation_concurrency: int = 1

    model_registry_dirs: list[Path] = [
        Path("runs/cdvae/checkpoints"),
        Path("backend/checkpoints"),
        Path("models"),
    ]

    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[2] / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )


settings = Settings()
