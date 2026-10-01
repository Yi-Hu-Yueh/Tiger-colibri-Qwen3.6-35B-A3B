from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from app.config import OLLAMA_MODEL_DESTINATION, OLLAMA_MODEL_SOURCE
from app.services.state_store import load_state, update_state

FILE_ATTRIBUTE_REPARSE_POINT = 0x400
ALLOWED_TOP_LEVEL = {"blobs", "manifests"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_reparse(path: Path) -> bool:
    try:
        return bool(getattr(path.stat(follow_symlinks=False), "st_file_attributes", 0) & FILE_ATTRIBUTE_REPARSE_POINT)
    except OSError:
        return False


def validate_source(path: Path = OLLAMA_MODEL_SOURCE) -> Path:
    resolved = path.resolve(strict=False)
    expected = OLLAMA_MODEL_SOURCE.resolve(strict=False)
    if resolved != expected or resolved.name.lower() != "models" or resolved.parent.name.lower() != ".ollama":
        raise ValueError(f"Source must be exactly {OLLAMA_MODEL_SOURCE}")
    if not resolved.is_dir() or _is_reparse(resolved):
        raise ValueError("Source model directory is missing or is a reparse point")
    return resolved


def validate_destination(path: Path = OLLAMA_MODEL_DESTINATION, source: Path = OLLAMA_MODEL_SOURCE) -> Path:
    resolved = path.resolve(strict=False)
    expected = OLLAMA_MODEL_DESTINATION.resolve(strict=False)
    if resolved != expected or resolved.drive.upper() != "D:":
        raise ValueError(f"Destination must be exactly {OLLAMA_MODEL_DESTINATION}")
    if resolved.exists() and _is_reparse(resolved):
        raise ValueError("Destination must not be a reparse point")
    if resolved.exists():
        source_paths = set(inventory(source)["files_by_path"])
        for child in resolved.iterdir():
            if child.name not in ALLOWED_TOP_LEVEL:
                raise ValueError(f"Destination contains unrelated entry: {child.name}")
        for file in _walk_files(resolved):
            if file.relative_to(resolved).as_posix() not in source_paths:
                raise ValueError(f"Destination contains unrelated file: {file.relative_to(resolved)}")
    resolved.mkdir(parents=True, exist_ok=True)
    probe = resolved / ".tiger-write-probe"
    probe.write_bytes(b"ok")
    probe.unlink()
    return resolved


def _walk_files(root: Path) -> list[Path]:
    result: list[Path] = []
    for current, directories, files in os.walk(root, followlinks=False):
        current_path = Path(current)
        directories[:] = [name for name in directories if not _is_reparse(current_path / name)]
        for name in files:
            path = current_path / name
            if _is_reparse(path):
                raise ValueError(f"Unexpected file reparse point: {path}")
            result.append(path)
    return result


def inventory(root: Path) -> dict[str, Any]:
    files = _walk_files(root) if root.is_dir() else []
    by_path = {path.relative_to(root).as_posix(): path.stat().st_size for path in files}
    blobs = {name: size for name, size in by_path.items() if name.startswith("blobs/")}
    manifests = {name: size for name, size in by_path.items() if name.startswith("manifests/")}
    return {
        "path": str(root),
        "file_count": len(by_path),
        "total_bytes": sum(by_path.values()),
        "blob_count": len(blobs),
        "blob_bytes": sum(blobs.values()),
        "manifest_count": len(manifests),
        "files_by_path": by_path,
    }


def compare_inventories(source: dict[str, Any], destination: dict[str, Any]) -> dict[str, Any]:
    source_files = source["files_by_path"]
    destination_files = destination["files_by_path"]
    missing = sorted(set(source_files) - set(destination_files))
    extra = sorted(set(destination_files) - set(source_files))
    size_mismatches = sorted(name for name in set(source_files) & set(destination_files) if source_files[name] != destination_files[name])
    return {
        "status": "PASS" if not missing and not extra and not size_mismatches else "FAIL",
        "missing": missing,
        "extra": extra,
        "size_mismatches": size_mismatches,
    }


def copy_models(
    source: Path = OLLAMA_MODEL_SOURCE,
    destination: Path = OLLAMA_MODEL_DESTINATION,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    validate_source(source)
    validate_destination(destination, source)
    command = ["robocopy", str(source), str(destination), "/E", "/Z", "/COPY:DAT", "/DCOPY:DAT", "/R:3", "/W:2", "/XJ", "/SL", "/NP", "/NFL", "/NDL"]
    result = runner(command, capture_output=True, text=True, errors="replace", check=False)
    if result.returncode >= 8:
        raise RuntimeError(f"Robocopy failed with exit code {result.returncode}: {result.stderr or result.stdout}")
    return {"status": "PASS", "robocopy_exit_code": result.returncode, "restartable": True}


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_blob_hashes(root: Path = OLLAMA_MODEL_DESTINATION) -> dict[str, Any]:
    blobs = root / "blobs"
    failures: list[dict[str, str]] = []
    count = 0
    total = 0
    for path in sorted(_walk_files(blobs)):
        name = path.name
        if not name.startswith("sha256-") or len(name) != 71:
            failures.append({"path": str(path), "error": "Blob filename does not encode a SHA-256 digest"})
            continue
        count += 1
        total += path.stat().st_size
        actual = sha256_file(path)
        expected = name.removeprefix("sha256-").lower()
        if actual != expected:
            failures.append({"path": str(path), "expected": expected, "actual": actual})
    return {"status": "PASS" if not failures else "FAIL", "hashed_blob_count": count, "hashed_bytes": total, "failures": failures}


def verify_manifests(root: Path = OLLAMA_MODEL_DESTINATION) -> dict[str, Any]:
    manifest_root = root / "manifests"
    errors: list[str] = []
    files = _walk_files(manifest_root) if manifest_root.is_dir() else []
    for path in files:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                errors.append(f"Manifest is not an object: {path}")
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            errors.append(f"Unreadable manifest {path}: {exc}")
    if not files:
        errors.append("No manifests found")
    return {"status": "PASS" if not errors else "FAIL", "manifest_count": len(files), "errors": errors}


def get_user_ollama_models() -> str | None:
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            return winreg.QueryValueEx(key, "OLLAMA_MODELS")[0]
    except FileNotFoundError:
        return None


def set_user_ollama_models(value: str | None) -> None:
    import winreg

    with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, "Environment", 0, winreg.KEY_SET_VALUE) as key:
        if value is None:
            try:
                winreg.DeleteValue(key, "OLLAMA_MODELS")
            except FileNotFoundError:
                pass
        else:
            winreg.SetValueEx(key, "OLLAMA_MODELS", 0, winreg.REG_SZ, value)
    try:
        import ctypes
        from ctypes import wintypes

        send = ctypes.windll.user32.SendMessageTimeoutW
        send.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPCWSTR,
                         wintypes.UINT, wintypes.UINT, ctypes.POINTER(ctypes.c_size_t)]
        send.restype = wintypes.LPARAM
        result = ctypes.c_size_t()
        send(0xFFFF, 0x001A, 0, "Environment", 0x0002, 5000, ctypes.byref(result))
    except (AttributeError, OSError):
        pass


def rollback_user_environment(previous: str | None) -> dict[str, Any]:
    set_user_ollama_models(previous)
    return {"status": "PASS" if get_user_ollama_models() == previous else "FAIL", "restored_value": previous}


def compare_model_names(before: list[str], after: list[str]) -> dict[str, Any]:
    missing = sorted(set(before) - set(after))
    extra = sorted(set(after) - set(before))
    return {"status": "PASS" if not missing and not extra and len(before) == len(after) else "FAIL", "missing": missing, "extra": extra}


def deletion_gate(evidence: dict[str, Any]) -> bool:
    required = ("copy_verification", "blob_verification", "environment", "restart", "api", "inventory", "smoke")
    return all(evidence.get(key, {}).get("status") == "PASS" for key in required)


def delete_old_model_store(evidence: dict[str, Any], source: Path = OLLAMA_MODEL_SOURCE) -> dict[str, Any]:
    validated = validate_source(source)
    if not deletion_gate(evidence):
        raise PermissionError("Deletion gate is not fully PASS")
    if validated == validated.parent or validated.name.lower() != "models":
        raise ValueError("Refusing to delete the .ollama root or a non-model path")
    shutil.rmtree(validated)
    return {"status": "PASS", "deleted_path": str(validated), "source_exists_after": validated.exists()}


def get_migration_status() -> dict[str, Any]:
    saved = load_state().get("ollama_storage_migration")
    return saved if isinstance(saved, dict) else {
        "status": "NOT_RUN",
        "source": str(OLLAMA_MODEL_SOURCE),
        "destination": str(OLLAMA_MODEL_DESTINATION),
        "warning": "D: is HDD-backed; Ollama cold-load time may be slower.",
    }


def persist_migration(evidence: dict[str, Any]) -> dict[str, Any]:
    evidence["updated_at"] = utc_now()
    update_state("ollama_storage_migration", evidence)
    return evidence
