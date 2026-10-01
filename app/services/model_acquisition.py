from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import urllib.request
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

from huggingface_hub import HfApi, snapshot_download

from app.config import (
    COLIBRI_QWEN36_DOC_URL,
    DOWNLOAD_ROOT,
    HF_CACHE_DIR,
    HF_XET_CACHE_DIR,
    MODEL_DESTINATION,
    MODEL_REPO_ID,
)
from app.services.state_store import load_state, update_state

GIB = 1024**3
DOWNLOAD_STATUSES = {"NOT_DOWNLOADED", "DOWNLOADING", "PARTIAL", "COMPLETE", "VERIFIED", "FAILED"}
INTEGRITY_STATUSES = {"NOT_VERIFIED", "VERIFIED_METADATA_AND_HASHES", "VERIFIED_METADATA", "PARTIAL_VERIFICATION", "FAILED"}
REQUIRED_FILES = ("config.json", "tokenizer.json", "qwen36_meta.json")
GLOBAL_MANIFEST_NAMES = {"sha256sums", "sha256sums.txt", "checksums.txt", "checksum.sha256"}
_download_lock = threading.Lock()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_destination(destination: Path | None = None) -> Path:
    destination = MODEL_DESTINATION if destination is None else destination
    root = DOWNLOAD_ROOT.resolve(strict=False)
    resolved = destination.resolve(strict=False)
    if resolved != MODEL_DESTINATION.resolve(strict=False):
        raise ValueError(f"Phase 2 destination is fixed to {MODEL_DESTINATION}")
    if resolved.drive.upper() != "D:" or root not in resolved.parents:
        raise ValueError("Model payload destination must be below the D: DOWNLOAD_ROOT")
    return resolved


def _base_state() -> dict[str, Any]:
    return {
        "repo_id": MODEL_REPO_ID,
        "revision": None,
        "destination": str(MODEL_DESTINATION),
        "remote_file_count": None,
        "remote_expected_bytes": None,
        "remote_last_modified": None,
        "remote_files": [],
        "local_file_count": 0,
        "local_bytes": 0,
        "download_status": "NOT_DOWNLOADED",
        "integrity_status": "NOT_VERIFIED",
        "download_started_at": None,
        "download_completed_at": None,
        "verification_timestamp": None,
        "verification_errors": [],
        "free_space_before_bytes": None,
        "free_space_after_bytes": None,
        "resume_supported": True,
        "resume_events": [],
        "global_checksum_manifest": None,
        "warnings": [
            "Acquisition storage PASS. Runtime inference storage NOT VALIDATED. This model currently resides on an HDD.",
            "RAM WARN: approximately 15.96 GiB is below the documented ~30 GB comfortable expert-cache configuration.",
            "MODEL_RUNTIME_ROOT remains UNSET.",
        ],
    }


def get_status() -> dict[str, Any]:
    data = dict(_base_state())
    saved = load_state().get("phase2_acquisition")
    if isinstance(saved, dict):
        data.update(saved)
    count, size = local_inventory(MODEL_DESTINATION)
    data["local_file_count"] = count
    data["local_bytes"] = size
    return data


def persist(status: dict[str, Any]) -> dict[str, Any]:
    if status.get("download_status") not in DOWNLOAD_STATUSES:
        raise ValueError("Invalid Phase 2 download status")
    if status.get("integrity_status") not in INTEGRITY_STATUSES:
        raise ValueError("Invalid Phase 2 integrity status")
    update_state("phase2_acquisition", status)
    return status


def revalidate_official_recommendation(opener: Callable[..., Any] = urllib.request.urlopen) -> dict[str, Any]:
    request = urllib.request.Request(COLIBRI_QWEN36_DOC_URL, headers={"User-Agent": "Tiger-Colibri-Phase2"})
    with opener(request, timeout=30) as response:
        text = response.read().decode("utf-8")
    lower = text.lower()
    repo_present = MODEL_REPO_ID.lower() in lower
    gs64_present = "gs64" in lower and "int4" in lower
    recommended = repo_present and gs64_present and "recommended" in lower
    result = {
        "status": "PASS" if recommended else "FAIL",
        "checked_at": utc_now(),
        "source": COLIBRI_QWEN36_DOC_URL,
        "repo_present": repo_present,
        "gs64_int4_present": gs64_present,
        "recommended": recommended,
    }
    if not recommended:
        raise RuntimeError("Current official Colibri Qwen3.6 documentation does not confirm the fixed gs64 int4 repository")
    return result


def resolve_metadata(api: HfApi | None = None) -> dict[str, Any]:
    info = (api or HfApi()).model_info(MODEL_REPO_ID, files_metadata=True)
    files: list[dict[str, Any]] = []
    reliable = True
    for sibling in info.siblings:
        size = getattr(sibling, "size", None)
        reliable = reliable and isinstance(size, int)
        lfs = getattr(sibling, "lfs", None)
        sha256 = getattr(lfs, "sha256", None) if lfs else None
        files.append({"path": sibling.rfilename, "size": size, "sha256": sha256})
    return {
        "repo_id": MODEL_REPO_ID,
        "revision": info.sha,
        "remote_last_modified": info.last_modified.isoformat() if info.last_modified else None,
        "remote_file_count": len(files),
        "remote_expected_bytes": sum(item["size"] for item in files) if reliable else None,
        "remote_files": files,
    }


def free_space_precheck(expected_bytes: int | None, destination: Path | None = None) -> dict[str, Any]:
    destination = MODEL_DESTINATION if destination is None else destination
    validate_destination(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(destination.parent).free
    required = 40 * GIB if expected_bytes is None else expected_bytes + max(5 * GIB, expected_bytes // 10)
    return {"status": "PASS" if free >= required else "FAIL", "free_bytes": free, "required_bytes": required}


def local_inventory(destination: Path) -> tuple[int, int]:
    if not destination.exists():
        return 0, 0
    files = [p for p in destination.rglob("*") if p.is_file() and ".cache" not in p.relative_to(destination).parts]
    return len(files), sum(p.stat().st_size for p in files)


@contextmanager
def _d_drive_hf_environment() -> Iterator[None]:
    values = {"HF_HOME": str(HF_CACHE_DIR), "HF_HUB_CACHE": str(HF_CACHE_DIR / "hub"), "HF_XET_CACHE": str(HF_XET_CACHE_DIR)}
    old = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def acquire_snapshot(
    *,
    api: HfApi | None = None,
    downloader: Callable[..., str] = snapshot_download,
    docs_opener: Callable[..., Any] = urllib.request.urlopen,
) -> dict[str, Any]:
    if not _download_lock.acquire(blocking=False):
        raise RuntimeError("A Qwen3.6 container download is already running")
    try:
        destination = validate_destination()
        state = get_status()
        state["official_recommendation"] = revalidate_official_recommendation(docs_opener)
        state.update(resolve_metadata(api))
        if (
            state.get("download_status") == "VERIFIED"
            and state.get("local_file_count") == state.get("remote_file_count")
            and state.get("local_bytes") == state.get("remote_expected_bytes")
        ):
            persist(state)
            return verify_snapshot(api=api)
        check = free_space_precheck(state["remote_expected_bytes"])
        state["free_space_before_bytes"] = check["free_bytes"]
        if check["status"] != "PASS":
            state.update(download_status="FAILED", verification_errors=["Insufficient D: free space"])
            return persist(state)
        state.update(
            download_status="DOWNLOADING",
            integrity_status="NOT_VERIFIED",
            download_started_at=state.get("download_started_at") or utc_now(),
            download_completed_at=None,
            verification_errors=[],
        )
        persist(state)
        destination.mkdir(parents=True, exist_ok=True)
        try:
            with _d_drive_hf_environment():
                downloader(
                    repo_id=MODEL_REPO_ID,
                    revision=state["revision"],
                    local_dir=str(destination),
                    cache_dir=str(HF_CACHE_DIR / "hub"),
                    max_workers=int(os.environ.get("TIGER_HF_MAX_WORKERS", "4")),
                )
        except BaseException as exc:
            count, size = local_inventory(destination)
            state.update(local_file_count=count, local_bytes=size, download_status="PARTIAL" if size else "FAILED", verification_errors=[f"{type(exc).__name__}: {exc}"])
            state["resume_events"] = [*state.get("resume_events", []), {"timestamp": utc_now(), "result": "INTERRUPTED", "local_bytes": size}]
            persist(state)
            raise
        count, size = local_inventory(destination)
        state.update(local_file_count=count, local_bytes=size, download_status="COMPLETE", download_completed_at=utc_now(), free_space_after_bytes=shutil.disk_usage(destination.parent).free)
        state["resume_events"] = [*state.get("resume_events", []), {"timestamp": utc_now(), "result": "COMPLETED", "local_bytes": size}]
        persist(state)
        return verify_snapshot(api=api)
    finally:
        _download_lock.release()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_recursive(data: Any, key: str) -> Any:
    if isinstance(data, dict):
        if key in data:
            return data[key]
        for value in data.values():
            found = _find_recursive(value, key)
            if found is not None:
                return found
    elif isinstance(data, list):
        for value in data:
            found = _find_recursive(value, key)
            if found is not None:
                return found
    return None


def validate_structure(destination: Path | None = None) -> dict[str, Any]:
    destination = MODEL_DESTINATION if destination is None else destination
    errors: list[str] = []
    for name in REQUIRED_FILES:
        if not (destination / name).is_file():
            errors.append(f"Missing required file: {name}")
    if errors:
        return {"status": "FAIL", "errors": errors}
    try:
        config = json.loads((destination / "config.json").read_text(encoding="utf-8"))
        tokenizer = json.loads((destination / "tokenizer.json").read_text(encoding="utf-8"))
        meta = json.loads((destination / "qwen36_meta.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": "FAIL", "errors": [f"Metadata parse error: {exc}"]}
    expert_gs = _find_recursive(meta, "expert_gs")
    model_type = config.get("model_type")
    text_config = config.get("text_config") if isinstance(config.get("text_config"), dict) else config
    geometry = {
        "model_type": model_type,
        "text_model_type": text_config.get("model_type"),
        "hidden_size": text_config.get("hidden_size"),
        "num_hidden_layers": text_config.get("num_hidden_layers"),
        "num_experts": text_config.get("num_experts"),
        "num_experts_per_tok": text_config.get("num_experts_per_tok"),
    }
    family_ok = model_type in {"qwen3_5_moe", "qwen3_5_moe_text"} or geometry["text_model_type"] in {"qwen3_5_moe", "qwen3_5_moe_text"}
    expected_geometry = geometry["hidden_size"] == 2048 and geometry["num_hidden_layers"] == 40 and geometry["num_experts"] == 256 and geometry["num_experts_per_tok"] == 8
    if expert_gs != 64:
        errors.append(f"expert_gs expected 64, got {expert_gs!r}")
    if not family_ok:
        errors.append(f"Unexpected Qwen3.6 model type: {model_type!r}/{geometry['text_model_type']!r}")
    if not expected_geometry:
        errors.append(f"Unexpected Qwen3.6 geometry: {geometry}")
    if not isinstance(tokenizer, dict) or not tokenizer:
        errors.append("tokenizer.json is not a populated JSON object")
    return {"status": "PASS" if not errors else "FAIL", "errors": errors, "expert_gs": expert_gs, "geometry": geometry, "tokenizer_valid": isinstance(tokenizer, dict) and bool(tokenizer)}


def compare_repository_files(destination: Path, remote_files: list[dict[str, Any]], *, verify_hashes: bool = True) -> dict[str, Any]:
    expected = {item["path"]: item for item in remote_files}
    actual = {p.relative_to(destination).as_posix(): p for p in destination.rglob("*") if p.is_file() and ".cache" not in p.relative_to(destination).parts}
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    size_mismatches = []
    hash_mismatches = []
    hashed = 0
    hash_bytes = 0
    for name, metadata in expected.items():
        path = actual.get(name)
        if path is None:
            continue
        expected_size = metadata.get("size")
        if expected_size is not None and path.stat().st_size != expected_size:
            size_mismatches.append(name)
            continue
        expected_hash = metadata.get("sha256")
        if verify_hashes and expected_hash:
            hashed += 1
            hash_bytes += path.stat().st_size
            if sha256_file(path) != expected_hash:
                hash_mismatches.append(name)
    return {
        "missing": missing,
        "extra": extra,
        "size_mismatches": size_mismatches,
        "hash_mismatches": hash_mismatches,
        "hash_files_verified": hashed,
        "hash_bytes_verified": hash_bytes,
        "hash_files_available": sum(1 for item in remote_files if item.get("sha256")),
        "file_list_match": not missing and not extra,
        "file_sizes_match": not size_mismatches,
        "hashes_match": not hash_mismatches,
    }


def verify_snapshot(*, api: HfApi | None = None, verify_hashes: bool = True) -> dict[str, Any]:
    destination = validate_destination()
    state = get_status()
    metadata = resolve_metadata(api)
    if state.get("revision") and state["revision"] != metadata["revision"]:
        metadata = {k: state[k] for k in ("repo_id", "revision", "remote_last_modified", "remote_file_count", "remote_expected_bytes", "remote_files")}
    comparison = compare_repository_files(destination, metadata["remote_files"], verify_hashes=verify_hashes)
    structure = validate_structure(destination)
    manifest = next((item["path"] for item in metadata["remote_files"] if Path(item["path"]).name.lower() in GLOBAL_MANIFEST_NAMES), None)
    errors = []
    if not comparison["file_list_match"]:
        errors.append(f"File list mismatch: missing={comparison['missing']}; extra={comparison['extra']}")
    if not comparison["file_sizes_match"]:
        errors.append(f"File size mismatches: {comparison['size_mismatches']}")
    if not comparison["hashes_match"]:
        errors.append(f"Hash mismatches: {comparison['hash_mismatches']}")
    errors.extend(structure["errors"])
    available = comparison["hash_files_available"]
    checked = comparison["hash_files_verified"]
    if errors:
        integrity = "FAILED"
        download_status = "FAILED"
    elif available and checked == available:
        integrity = "VERIFIED_METADATA_AND_HASHES"
        download_status = "VERIFIED"
    elif comparison["file_list_match"] and comparison["file_sizes_match"]:
        integrity = "PARTIAL_VERIFICATION" if available else "VERIFIED_METADATA"
        download_status = "VERIFIED"
    else:
        integrity = "FAILED"
        download_status = "FAILED"
    count, size = local_inventory(destination)
    state.update(metadata)
    state.update(
        local_file_count=count,
        local_bytes=size,
        download_status=download_status,
        integrity_status=integrity,
        verification_timestamp=utc_now(),
        verification_errors=errors,
        verification=comparison,
        structure_validation=structure,
        global_checksum_manifest=manifest,
        free_space_after_bytes=shutil.disk_usage(destination.parent).free,
    )
    return persist(state)
