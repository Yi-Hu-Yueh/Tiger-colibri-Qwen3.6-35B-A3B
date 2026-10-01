from __future__ import annotations

from pathlib import Path
from typing import Any

from app.config import VENDOR_DIR
from app.services.command_runner import run_command


def classify_smoke(result: dict[str, Any]) -> str:
    combined = f"{result.get('stdout', '')}\n{result.get('stderr', '')}".lower()
    if result.get("timed_out"):
        return "TIMEOUT"
    if result.get("error"):
        return "EXECUTION_FAILURE"
    if any(term in combined for term in ("dll was not found", "missing dll", "0xc0000135", "unable to load dll")):
        return "DLL_LOAD_FAILURE"
    if any(term in combined for term in ("usage:", "options:", "--help", "commands:")):
        return "USAGE_DISPLAYED"
    if any(term in combined for term in (
        "model required", "missing model", "container required", "no model", "without a model",
        "needs a model", "model directory", "set snap", "snap environment variable",
    )):
        return "MODEL_REQUIRED"
    if result.get("exit_code") == 0:
        return "BINARY_LAUNCHED"
    return "EXECUTION_FAILURE"


def smoke_test(installation: dict[str, Any]) -> dict[str, Any]:
    raw = installation.get("qwen36_path")
    if not raw:
        return {"status": "FAIL", "classification": "EXECUTION_FAILURE", "error": "qwen36.exe is not installed"}
    executable = Path(raw).resolve()
    vendor = VENDOR_DIR.resolve()
    if vendor not in executable.parents or executable.name.lower() != "qwen36.exe" or not executable.is_file():
        return {"status": "FAIL", "classification": "EXECUTION_FAILURE", "error": "Executable path is not whitelisted"}
    attempts = []
    best = "EXECUTION_FAILURE"
    for flag in ("--help", "-h"):
        command_result = run_command([str(executable), flag], timeout=15, cwd=executable.parent)
        classification = classify_smoke(command_result)
        command_result["classification"] = classification
        attempts.append(command_result)
        if classification in {"BINARY_LAUNCHED", "USAGE_DISPLAYED", "MODEL_REQUIRED"}:
            best = classification
            break
        if classification == "DLL_LOAD_FAILURE":
            best = classification
            break
        if classification == "TIMEOUT":
            best = classification
    passed = best in {"BINARY_LAUNCHED", "USAGE_DISPLAYED", "MODEL_REQUIRED"}
    return {"status": "PASS" if passed else "FAIL", "classification": best, "attempts": attempts,
            "note": "This validates binary launch only; no model was downloaded or loaded."}
