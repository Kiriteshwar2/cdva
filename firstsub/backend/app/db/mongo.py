from __future__ import annotations

from typing import Any

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo.errors import PyMongoError

from ..core.config import settings

_client: AsyncIOMotorClient[Any] | None = None
_mongo_state: dict[str, str | None] = {"status": "unknown", "error": None}


def get_client() -> AsyncIOMotorClient[Any]:
    global _client
    if _client is None:
        _client = AsyncIOMotorClient(
            settings.mongo_uri,
            uuidRepresentation="standard",
            serverSelectionTimeoutMS=2500,
        )
    return _client


def get_database() -> AsyncIOMotorDatabase[Any]:
    return get_client()[settings.mongodb_database]


async def ping_database() -> bool:
    try:
        result = await get_database().command("ping")
        ok = bool(result.get("ok"))
        _mongo_state.update(status="ok" if ok else "down", error=None if ok else "Mongo ping failed")
        return ok
    except PyMongoError as exc:
        _mongo_state.update(status="down", error=str(exc))
        return False


def mongo_state() -> dict[str, str | None]:
    return dict(_mongo_state)


async def ensure_indexes() -> None:
    if not await ping_database():
        return
    db = get_database()
    try:
        await db.users.create_index("email", unique=True, name="uq_users_email")
        await db.generations.create_index([("user_id", 1), ("created_at", -1)], name="ix_generations_user_created")
        await db.models.create_index("checkpoint_name", unique=True, name="uq_models_checkpoint")
    except PyMongoError as exc:
        _mongo_state.update(status="down", error=str(exc))


async def close_database() -> None:
    global _client
    if _client is not None:
        _client.close()
        _client = None
