from __future__ import annotations

import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RISK_LABELS = {"USER_REVIEW", "KNOWN_CACHE", "APPLICATION_DATA", "SYSTEM_MANAGED", "UNKNOWN"}
LARGE_FILE_THRESHOLD = 500 * 2**20
MAX_LARGE_FILES = 100
MAX_LARGE_DIRECTORIES = 100
DEFAULT_TIMEOUT_SECONDS = 180
DEFAULT_MAX_ENTRIES = 2_000_000

_scan_lock = threading.Lock()


def _error(path: Path, exc: BaseException) -> dict[str, str]:
    return {"path": str(path), "error": f"{type(exc).__name__}: {exc}"}


def scan_tree(root: Path, *, threshold_bytes: int = LARGE_FILE_THRESHOLD,
              deadline: float | None = None, max_entries: int = DEFAULT_MAX_ENTRIES) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(root), "size_bytes": 0, "file_count": 0, "directory_count": 0,
                              "large_files": [], "errors": [], "skipped_paths": [], "entries_seen": 0,
                              "complete": True}
    if not root.exists():
        result["skipped_paths"].append({"path": str(root), "reason": "not found"})
        return result
    stack = [root]
    while stack:
        if deadline is not None and time.monotonic() >= deadline:
            result["complete"] = False
            result["skipped_paths"].append({"path": str(stack[-1]), "reason": "scan time limit reached"})
            break
        current = stack.pop()
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    result["entries_seen"] += 1
                    if result["entries_seen"] > max_entries:
                        result["complete"] = False
                        result["skipped_paths"].append({"path": str(current), "reason": "entry limit reached"})
                        stack.clear()
                        break
                    try:
                        if entry.is_symlink():
                            result["skipped_paths"].append({"path": entry.path, "reason": "link/reparse point not followed"})
                        elif entry.is_dir(follow_symlinks=False):
                            result["directory_count"] += 1
                            stack.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False):
                            stat = entry.stat(follow_symlinks=False)
                            size = stat.st_size
                            result["size_bytes"] += size
                            result["file_count"] += 1
                            if size >= threshold_bytes:
                                result["large_files"].append({
                                    "path": entry.path, "size_bytes": size, "parent": str(Path(entry.path).parent),
                                    "last_modified": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                                    "risk_label": "USER_REVIEW",
                                })
                    except (OSError, PermissionError) as exc:
                        result["errors"].append(_error(Path(entry.path), exc))
        except (OSError, PermissionError) as exc:
            result["errors"].append(_error(current, exc))
    result["large_files"].sort(key=lambda item: item["size_bytes"], reverse=True)
    result["large_files"] = result["large_files"][:MAX_LARGE_FILES]
    return result


def _directory_item(path: Path, label: str, scan: dict[str, Any], category: str) -> dict[str, Any]:
    if label not in RISK_LABELS:
        raise ValueError(f"Invalid risk label: {label}")
    return {"path": str(path), "name": path.name or str(path), "size_bytes": scan["size_bytes"],
            "file_count": scan["file_count"], "risk_label": label, "category": category,
            "complete": scan["complete"]}


def _scan_immediate_children(root: Path, label: str, category: str, deadline: float,
                             remaining_entries: list[int]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list, list]:
    directories: list[dict[str, Any]] = []
    large_files: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []
    if not root.exists():
        skipped.append({"path": str(root), "reason": "not found"})
        return directories, large_files, errors, skipped
    try:
        entries = list(os.scandir(root))
    except (OSError, PermissionError) as exc:
        errors.append(_error(root, exc))
        return directories, large_files, errors, skipped
    root_files_size = 0
    for entry in entries:
        if time.monotonic() >= deadline or remaining_entries[0] <= 0:
            skipped.append({"path": str(root), "reason": "global scan bound reached"})
            break
        path = Path(entry.path)
        try:
            if entry.is_symlink():
                skipped.append({"path": str(path), "reason": "link/reparse point not followed"})
            elif entry.is_dir(follow_symlinks=False):
                item_label = label
                if category == "AppData Local" and entry.name.lower() in {"pip", "npm-cache", "temp"}:
                    item_label = "KNOWN_CACHE"
                scan = scan_tree(path, deadline=deadline, max_entries=remaining_entries[0])
                remaining_entries[0] -= scan["entries_seen"]
                directories.append(_directory_item(path, item_label, scan, category))
                for item in scan["large_files"]:
                    item["risk_label"] = item_label
                large_files.extend(scan["large_files"])
                errors.extend(scan["errors"])
                skipped.extend(scan["skipped_paths"])
            elif entry.is_file(follow_symlinks=False):
                stat = entry.stat(follow_symlinks=False)
                root_files_size += stat.st_size
                if stat.st_size >= LARGE_FILE_THRESHOLD:
                    large_files.append({"path": str(path), "size_bytes": stat.st_size, "parent": str(root),
                                        "last_modified": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                                        "risk_label": label})
        except (OSError, PermissionError) as exc:
            errors.append(_error(path, exc))
    if root_files_size:
        directories.append({"path": str(root), "name": "files directly in root", "size_bytes": root_files_size,
                            "file_count": None, "risk_label": label, "category": category, "complete": True})
    return directories, large_files, errors, skipped


def _measure_known_path(path: Path, label: str, category: str, deadline: float,
                        remaining_entries: list[int]) -> tuple[dict[str, Any], dict[str, Any]]:
    scan = scan_tree(path, deadline=deadline, max_entries=max(remaining_entries[0], 1))
    remaining_entries[0] -= scan["entries_seen"]
    for item in scan["large_files"]:
        item["risk_label"] = label
    return _directory_item(path, label, scan, category), scan


def _system_files() -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    files, errors = [], []
    for name in ("pagefile.sys", "hiberfil.sys", "swapfile.sys"):
        path = Path("C:/") / name
        try:
            stat = path.stat()
            files.append({"path": str(path), "size_bytes": stat.st_size, "risk_label": "SYSTEM_MANAGED",
                          "advisory": "SYSTEM MANAGED — DO NOT MODIFY AUTOMATICALLY"})
        except FileNotFoundError:
            continue
        except (OSError, PermissionError) as exc:
            errors.append(_error(path, exc))
    return files, errors


def _ollama_roots(user_profile: Path) -> list[Path]:
    candidates = [user_profile / ".ollama"]
    configured = os.environ.get("OLLAMA_MODELS")
    if configured:
        candidates.append(Path(configured).expanduser())
    return list(dict.fromkeys(path.resolve() for path in candidates))


def analyze_space(*, timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
                  max_entries: int = DEFAULT_MAX_ENTRIES) -> dict[str, Any]:
    if not _scan_lock.acquire(blocking=False):
        raise RuntimeError("A C: space analysis is already running")
    try:
        started = time.monotonic()
        deadline = started + timeout_seconds
        remaining = [max_entries]
        user_profile = Path(os.environ.get("USERPROFILE", str(Path.home()))).resolve()
        local = Path(os.environ.get("LOCALAPPDATA", user_profile / "AppData" / "Local")).resolve()
        roaming = Path(os.environ.get("APPDATA", user_profile / "AppData" / "Roaming")).resolve()
        total, used, free = __import__("shutil").disk_usage("C:/")
        major_directories: list[dict[str, Any]] = []
        child_directories: list[dict[str, Any]] = []
        large_files: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []
        skipped: list[dict[str, str]] = []

        for name in ("Desktop", "Documents", "Downloads", "Videos", "Pictures", "Music"):
            path = user_profile / name
            item, scan = _measure_known_path(path, "USER_REVIEW", "user directory", deadline, remaining)
            major_directories.append(item)
            large_files.extend(scan["large_files"]); errors.extend(scan["errors"]); skipped.extend(scan["skipped_paths"])

        for root, category in ((local, "AppData Local"), (roaming, "AppData Roaming"), (Path("C:/ProgramData"), "ProgramData")):
            dirs, files, errs, skips = _scan_immediate_children(root, "APPLICATION_DATA", category, deadline, remaining)
            child_directories.extend(dirs); large_files.extend(files); errors.extend(errs); skipped.extend(skips)
            major_directories.append({"path": str(root), "name": category,
                                      "size_bytes": sum(item["size_bytes"] for item in dirs),
                                      "file_count": None, "risk_label": "APPLICATION_DATA", "category": category,
                                      "complete": all(item["complete"] for item in dirs)})

        ollama_items = []
        for root in _ollama_roots(user_profile):
            item, scan = _measure_known_path(root, "APPLICATION_DATA", "Ollama baseline model storage", deadline, remaining)
            item["advisory"] = "Baseline model storage — manual decision only."
            blobs = root / "models" / "blobs"
            manifests = root / "models" / "manifests"
            blob_item, _ = _measure_known_path(blobs, "APPLICATION_DATA", "Ollama blobs", deadline, remaining)
            manifest_item, _ = _measure_known_path(manifests, "APPLICATION_DATA", "Ollama manifests", deadline, remaining)
            item["model_blobs_bytes"] = blob_item["size_bytes"]
            item["manifests_bytes"] = manifest_item["size_bytes"]
            ollama_items.append(item)
            large_files.extend(scan["large_files"]); errors.extend(scan["errors"]); skipped.extend(scan["skipped_paths"])

        cache_paths = [
            (local / "pip" / "Cache", "pip cache"),
            (user_profile / ".cache" / "huggingface", "Hugging Face cache"),
            (user_profile / ".cache" / "torch", "torch cache"),
            (roaming / "Code" / "Cache", "VS Code cache"),
            (roaming / "Code" / "CachedData", "VS Code cached data"),
        ]
        cache_items = []
        for path, category in cache_paths:
            item, scan = _measure_known_path(path, "KNOWN_CACHE", category, deadline, remaining)
            cache_items.append(item)
            errors.extend(scan["errors"]); skipped.extend(scan["skipped_paths"])
        temp_item, temp_scan = _measure_known_path(local / "Temp", "KNOWN_CACHE", "Windows user temp", deadline, remaining)
        errors.extend(temp_scan["errors"]); skipped.extend(temp_scan["skipped_paths"])

        system_files, system_errors = _system_files()
        errors.extend(system_errors)
        unique_files = {os.path.normcase(item["path"]): item for item in large_files}
        largest_files = sorted(unique_files.values(), key=lambda item: item["size_bytes"], reverse=True)[:MAX_LARGE_FILES]
        largest_directories = sorted(child_directories + major_directories, key=lambda item: item["size_bytes"], reverse=True)[:MAX_LARGE_DIRECTORIES]
        return {
            "status": "PASS", "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(time.monotonic() - started, 2), "complete": time.monotonic() < deadline and remaining[0] > 0,
            "bounds": {"timeout_seconds": timeout_seconds, "max_entries": max_entries,
                       "large_file_threshold_bytes": LARGE_FILE_THRESHOLD, "largest_file_limit": MAX_LARGE_FILES},
            "drive": {"letter": "C:", "total_bytes": total, "used_bytes": used, "free_bytes": free,
                      "free_gib": round(free / 2**30, 2), "minimum_target_gib": 30, "preferred_target_gib": [45, 50],
                      "additional_for_30_gib": round(max(0, 30 - free / 2**30), 2),
                      "additional_for_45_gib": round(max(0, 45 - free / 2**30), 2),
                      "additional_for_50_gib": round(max(0, 50 - free / 2**30), 2)},
            "major_directories": major_directories, "largest_directories": largest_directories,
            "largest_files": largest_files, "ollama_storage": ollama_items, "development_caches": cache_items,
            "temp_storage": temp_item, "system_managed_files": system_files,
            "errors": errors[:200], "skipped_paths": skipped[:200],
            "error_count": len(errors), "skipped_count": len(skipped),
            "risk_labels": sorted(RISK_LABELS),
            "advisory": "Read-only analysis. No files are deleted automatically.",
        }
    finally:
        _scan_lock.release()
