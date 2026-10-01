from __future__ import annotations

from fastapi import APIRouter

from app.schemas.ollama_storage import OllamaStorageStatus
from app.services.ollama_migration import get_migration_status

router = APIRouter(prefix="/api/ollama", tags=["ollama-storage"])


@router.get("/storage", response_model=OllamaStorageStatus)
def storage_status() -> dict:
    return get_migration_status()

