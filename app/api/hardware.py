from fastapi import APIRouter

from app.services.hardware_probe import probe_hardware
from app.services.ollama_probe import probe_ollama
from app.services.state_store import load_state, update_state

router = APIRouter(prefix="/api", tags=["hardware"])


@router.get("/hardware")
def hardware() -> dict:
    result = probe_hardware()
    update_state("hardware", result)
    return result


@router.get("/ollama")
def ollama() -> dict:
    result = probe_ollama()
    state = load_state()
    if result["detected"] or "ollama_baseline_present" not in state:
        state["ollama_baseline_present"] = result["detected"]
        from app.services.state_store import save_state
        save_state(state)
    update_state("ollama", result)
    return result
