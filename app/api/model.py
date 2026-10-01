from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.schemas.model import ModelAcquisitionStatus
from app.services.model_acquisition import acquire_snapshot, get_status, verify_snapshot

router = APIRouter(prefix="/api/model/qwen36", tags=["model-acquisition"])


@router.get("/status", response_model=ModelAcquisitionStatus)
def status() -> dict:
    return get_status()


@router.post("/download", response_model=ModelAcquisitionStatus)
def download() -> dict:
    try:
        return acquire_snapshot()
    except Exception as exc:
        raise HTTPException(status_code=409 if "already running" in str(exc) else 502, detail=str(exc)) from exc


@router.post("/verify", response_model=ModelAcquisitionStatus)
def verify() -> dict:
    try:
        return verify_snapshot()
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

