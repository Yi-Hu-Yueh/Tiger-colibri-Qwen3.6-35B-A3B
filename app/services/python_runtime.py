from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from app.services.command_runner import run_command


def validate_python_interpreter(executable: str, runner: Callable[..., dict[str, Any]] = run_command) -> dict[str, Any]:
    path = Path(executable)
    if not path.is_file():
        return {"valid": False, "executable": executable, "version": None, "error": "Interpreter does not exist"}
    result = runner([str(path), "-c", "import platform; print(platform.python_version())"], timeout=8)
    version = result.get("stdout", "").strip()
    valid = result.get("exit_code") == 0 and version.startswith("3.11.")
    return {"valid": valid, "executable": str(path.resolve()), "version": version or None,
            "error": None if valid else (result.get("error") or result.get("stderr") or f"Python 3.11.x required; found {version or 'unknown'}")}
