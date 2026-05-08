from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, Query, status
from motor.motor_asyncio import AsyncIOMotorDatabase

from ..deps import get_current_user, get_db
from ...schemas.generation import (
    GenerationListResponse,
    GenerationRequest,
    GenerationResponse,
    ModelStatusResponse,
)
from ...services.inference import generation_service

router = APIRouter(tags=["generation"])


def _serialize_generation(document: dict[str, Any]) -> GenerationResponse:
    return GenerationResponse(
        id=str(document["_id"]),
        user_id=str(document["user_id"]),
        checkpoint_name=document["checkpoint_name"],
        output_cif=document["output_cif"],
        structure=document["structure"],
        metadata=document["metadata"],
        input_parameters=document["input_parameters"],
        created_at=document["created_at"],
    )


@router.post("/generate", response_model=GenerationResponse, status_code=status.HTTP_201_CREATED)
async def generate_crystal(
    payload: GenerationRequest,
    db: AsyncIOMotorDatabase = Depends(get_db),
    current_user: dict = Depends(get_current_user),
) -> GenerationResponse:
    candidate = await generation_service.generate(payload)
    now = datetime.now(timezone.utc)
    checkpoint_name = generation_service.status["model_id"] or payload.checkpoint_name or "unknown"
    
    # DEBUG: Verify CIF before storing
    print(f"\n{'='*70}")
    print(f"[BACKEND /generate] CIF Generation Debug")
    print(f"{'='*70}")
    print(f"CIF String Type: {type(candidate.cif_string)}")
    print(f"CIF String Length: {len(candidate.cif_string)} characters")
    print(f"CIF Valid (starts with 'data_'): {str(candidate.cif_string).startswith('data_')}")
    print(f"CIF First 150 chars:\n{candidate.cif_string[:150]}")
    print(f"{'='*70}\n")
    
    generation_doc = {
        "user_id": current_user["_id"],
        "checkpoint_name": checkpoint_name,
        "input_parameters": payload.model_dump(mode="json"),
        "output_cif": candidate.cif_string,
        "structure": candidate.metadata.pop("structure"),
        "metadata": candidate.metadata,
        "created_at": now,
    }
    result = await db.generations.insert_one(generation_doc)
    await db.models.update_one(
        {"checkpoint_name": checkpoint_name},
        {
            "$set": {
                "checkpoint_name": checkpoint_name,
                "path": generation_service.status["model_path"],
                "loaded_status": True,
                "last_loaded_at": now,
            }
        },
        upsert=True,
    )
    created = await db.generations.find_one({"_id": result.inserted_id})
    assert created is not None
    return _serialize_generation(created)


@router.get("/history", response_model=GenerationListResponse)
async def generation_history(
    limit: int = Query(default=20, ge=1, le=100),
    db: AsyncIOMotorDatabase = Depends(get_db),
    current_user: dict = Depends(get_current_user),
) -> GenerationListResponse:
    cursor = db.generations.find({"user_id": current_user["_id"]}).sort("created_at", -1).limit(limit)
    items = [_serialize_generation(item) async for item in cursor]
    total = await db.generations.count_documents({"user_id": current_user["_id"]})
    return GenerationListResponse(items=items, total=total)


@router.get("/generation/{generation_id}", response_model=GenerationResponse)
async def generation_detail(
    generation_id: str,
    db: AsyncIOMotorDatabase = Depends(get_db),
    current_user: dict = Depends(get_current_user),
) -> GenerationResponse:
    try:
        object_id = ObjectId(generation_id)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid generation id.") from exc

    item = await db.generations.find_one({"_id": object_id, "user_id": current_user["_id"]})
    if item is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Generation not found.")
    return _serialize_generation(item)


@router.get("/models", response_model=list[ModelStatusResponse])
async def list_models(db: AsyncIOMotorDatabase = Depends(get_db)) -> list[ModelStatusResponse]:
    model_docs = {doc["checkpoint_name"]: doc async for doc in db.models.find({})}
    results: list[ModelStatusResponse] = []
    for entry in generation_service.list_available_models():
        persisted = model_docs.get(entry["checkpoint_name"])
        results.append(
            ModelStatusResponse(
                checkpoint_name=entry["checkpoint_name"],
                path=entry["path"],
                loaded_status=bool(persisted["loaded_status"]) if persisted else False,
                last_loaded_at=persisted.get("last_loaded_at") if persisted else None,
            )
        )
    return results
