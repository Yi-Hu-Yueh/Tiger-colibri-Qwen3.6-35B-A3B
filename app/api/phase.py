from fastapi import APIRouter

from app.services.phase_gate import evaluate_phase1
from app.services.state_store import load_state, update_state

router = APIRouter(prefix="/api/phase", tags=["phase"])


@router.get("/1")
def phase1() -> dict:
    result = evaluate_phase1(load_state())
    update_state("phase1_gate", result)
    return result
