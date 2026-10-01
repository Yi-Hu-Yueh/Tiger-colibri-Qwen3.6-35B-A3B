from __future__ import annotations

import json
import os
import re
import shutil
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from app.services.command_runner import run_command


def _valid_executable(path: str | None) -> str | None:
    if not path:
        return None
    candidate = Path(path.strip().strip('"'))
    try:
        return str(candidate.resolve()) if candidate.is_file() and candidate.name.lower() == "ollama.exe" else None
    except OSError:
        return None


def probe_ollama_api(timeout: float = 1.5, opener: Callable[..., Any] = urllib.request.urlopen) -> dict[str, Any]:
    request = urllib.request.Request("http://127.0.0.1:11434/api/version", headers={"User-Agent": "Tiger-Colibri-Phase1R/1.0"})
    try:
        with opener(request, timeout=timeout) as response:
            payload = json.load(response)
        return {"reachable": True, "version": payload.get("version"), "error": None}
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        return {"reachable": False, "version": None, "error": str(exc)}


def _process_evidence() -> tuple[bool, list[dict[str, Any]]]:
    script = "Get-Process -ErrorAction SilentlyContinue | Where-Object {$_.ProcessName -like '*ollama*'} | ForEach-Object {[pscustomobject]@{name=$_.ProcessName;id=$_.Id;path=$_.Path}} | ConvertTo-Json -Compress"
    result = run_command(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script], timeout=8)
    if result["exit_code"] != 0 or not result["stdout"]:
        return False, []
    try:
        parsed = json.loads(result["stdout"])
        values = parsed if isinstance(parsed, list) else [parsed]
        return bool(values), [v for v in values if isinstance(v, dict)]
    except json.JSONDecodeError:
        return False, []


def _find_executables() -> tuple[list[str], list[str]]:
    found: list[str] = []
    methods: list[str] = []
    where = run_command(["where.exe", "ollama"], timeout=5)
    if where["exit_code"] == 0:
        for line in where["stdout"].splitlines():
            if valid := _valid_executable(line):
                found.append(valid)
                methods.append("where.exe")
    command = run_command(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
                           "(Get-Command ollama -ErrorAction SilentlyContinue).Source"], timeout=5)
    if command["exit_code"] == 0:
        if valid := _valid_executable(command["stdout"]):
            found.append(valid)
            methods.append("PowerShell Get-Command")
    if valid := _valid_executable(shutil.which("ollama")):
        found.append(valid)
        methods.append("Python PATH lookup")
    local = os.environ.get("LOCALAPPDATA")
    if local:
        common = [Path(local) / "Programs" / "Ollama" / "ollama.exe", Path(local) / "Ollama" / "ollama.exe",
                  Path(local) / "Programs" / "Ollama" / "app" / "ollama.exe"]
        for path in common:
            if valid := _valid_executable(str(path)):
                found.append(valid)
                methods.append(f"common location: {path.parent}")
    unique = list(dict.fromkeys(found))
    return unique, list(dict.fromkeys(methods))


def _parse_models(output: str) -> list[dict[str, Any]]:
    models = []
    for line in output.splitlines()[1:]:
        columns = re.split(r"\s{2,}", line.strip())
        if columns and columns[0]:
            models.append({"name": columns[0], "id": columns[1] if len(columns) > 1 else None,
                           "size": columns[2] if len(columns) > 2 else None,
                           "modified": columns[3] if len(columns) > 3 else None})
    return models


def probe_ollama() -> dict[str, Any]:
    executables, methods = _find_executables()
    process_detected, processes = _process_evidence()
    if process_detected:
        methods.append("running process")
        for process in processes:
            if valid := _valid_executable(process.get("path")):
                executables.append(valid)
    executables = list(dict.fromkeys(executables))
    api = probe_ollama_api()
    if api["reachable"]:
        methods.append("local API /api/version")
    executable = executables[0] if executables else None
    cli_version = None
    installed_models: list[dict[str, Any]] = []
    errors: list[str] = []
    if executable:
        version = run_command([executable, "--version"], timeout=10)
        if version["exit_code"] == 0:
            cli_version = version["stdout"] or version["stderr"]
            methods.append("ollama --version")
        else:
            errors.append(version["error"] or version["stderr"] or "ollama --version failed")
        listing = run_command([executable, "list"], timeout=15)
        if listing["exit_code"] == 0:
            installed_models = _parse_models(listing["stdout"])
            methods.append("ollama list")
        else:
            errors.append(listing["error"] or listing["stderr"] or "ollama list failed")
    detected = bool(executable or process_detected or api["reachable"])
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(), "detected": detected,
        "executable_found": bool(executable), "executable_path": executable,
        "all_executable_paths": executables, "process_detected": process_detected, "processes": processes,
        "api_reachable": api["reachable"], "api_version": api["version"], "api_error": api["error"],
        "cli_version": cli_version, "installed_models": installed_models, "detection_methods": list(dict.fromkeys(methods)),
        "errors": errors, "protection": "Ollama is baseline only and is not modified by this project.",
    }
