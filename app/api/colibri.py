from fastapi import APIRouter, HTTPException

from app.services.colibri_installer import download_release, extract_release
from app.services.colibri_smoke import smoke_test
from app.services.github_release import fetch_latest_release
from app.services.state_store import load_state, update_state

router = APIRouter(prefix="/api/colibri", tags=["colibri"])


@router.get("/release")
def release() -> dict:
    result = fetch_latest_release()
    update_state("colibri_release", result)
    return result


@router.post("/download")
def download() -> dict:
    state = load_state()
    metadata = state.get("colibri_release")
    if not metadata or metadata.get("status") != "PASS":
        raise HTTPException(status_code=409, detail="Run a successful official release lookup first")
    try:
        result = download_release(metadata)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    update_state("colibri_download", result)
    return result


@router.post("/extract")
def extract() -> dict:
    result = extract_release(load_state().get("colibri_download") or {})
    update_state("colibri_installation", result)
    return result


@router.post("/smoke")
def smoke() -> dict:
    result = smoke_test(load_state().get("colibri_installation") or {})
    update_state("smoke_test", result)
    return result
