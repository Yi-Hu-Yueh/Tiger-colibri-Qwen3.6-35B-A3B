from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Sequence


def run_command(command: Sequence[str], timeout: float = 15, cwd: Path | None = None) -> dict:
    safe_command = [str(part) for part in command]
    try:
        result = subprocess.run(
            safe_command,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
            shell=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return {
            "command": safe_command,
            "exit_code": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "timed_out": False,
            "error": None,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": safe_command,
            "exit_code": None,
            "stdout": (exc.stdout or "") if isinstance(exc.stdout, str) else "",
            "stderr": (exc.stderr or "") if isinstance(exc.stderr, str) else "",
            "timed_out": True,
            "error": f"Command timed out after {timeout} seconds",
        }
    except (OSError, ValueError) as exc:
        return {
            "command": safe_command,
            "exit_code": None,
            "stdout": "",
            "stderr": "",
            "timed_out": False,
            "error": str(exc),
        }
