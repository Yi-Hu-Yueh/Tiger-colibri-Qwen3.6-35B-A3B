from fastapi import APIRouter, HTTPException

from app.schemas.phase import BenchmarkRequest, ModelStorageRequest
from app.services.disk_benchmark import benchmark_storage
from app.services.state_store import load_state, save_state, update_state
from app.services.storage_probe import derive_model_root, probe_storage, validate_model_storage_root
from app.services.space_analysis import analyze_space

router = APIRouter(prefix="/api/storage", tags=["storage"])


@router.get("")
def storage() -> dict:
    state = load_state()
    result = probe_storage(download_root=state.get("download_root"),
                           model_runtime_root=state.get("model_runtime_root"), perform_write_test=False)
    update_state("storage", result)
    return result


@router.get("/space-analysis")
def get_space_analysis() -> dict:
    result = load_state().get("space_analysis")
    return result or {"status": "NOT RUN", "advisory": "Read-only analysis. No files are deleted automatically."}


@router.post("/analyze-space")
def run_space_analysis() -> dict:
    try:
        result = analyze_space()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    update_state("space_analysis", result)
    return result


@router.post("/model-root")
def select_model_root(request: ModelStorageRequest) -> dict:
    state = load_state()
    inventory = probe_storage()
    letter = request.drive_letter.strip().rstrip(":").upper()
    if not any(str(v.get("drive_letter", "")).upper() == letter for v in inventory.get("volumes", [])):
        raise HTTPException(status_code=422, detail="Drive must be selected from the detected local-volume inventory")
    try:
        root = derive_model_root(letter)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    validation = validate_model_storage_root(root, inventory, perform_write_test=True)
    inventory["model_storage_root"] = root
    inventory["model_storage_validation"] = validation
    state["model_runtime_root"] = root
    state["model_runtime_validation"] = validation
    state["model_storage_root"] = root  # backward-compatible state alias
    state["model_storage_validation"] = validation
    state["storage"] = inventory
    save_state(state)
    return validation


@router.post("/benchmark")
def benchmark(request: BenchmarkRequest) -> dict:
    state = load_state()
    root = state.get("model_runtime_root")
    validation = state.get("model_runtime_validation") or {}
    if not root or not validation.get("writable"):
        raise HTTPException(status_code=409, detail="Select and validate MODEL_STORAGE_ROOT before benchmarking")
    try:
        from pathlib import Path
        result = benchmark_storage(request.size_mib, Path(root))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    update_state("disk_benchmark", result)
    return result
