from __future__ import annotations

import hashlib
import os
import re
import shutil
import tempfile
import threading
import urllib.error
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from app.config import GITHUB_REPOSITORY, VENDOR_DIR

_download_lock = threading.Lock()


def parse_checksums(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        match = re.match(r"^([A-Fa-f0-9]{64})\s+\*?(.+?)\s*$", line)
        if match:
            values[Path(match.group(2)).name] = match.group(1).lower()
    return values


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_checksum(zip_path: Path, checksum_path: Path | None) -> dict[str, Any]:
    actual = sha256_file(zip_path)
    if not checksum_path or not checksum_path.exists():
        return {"status": "UNAVAILABLE", "actual": actual, "expected": None,
                "message": "Official SHA256SUMS.txt was not published."}
    checksums = parse_checksums(checksum_path.read_text(encoding="utf-8", errors="replace"))
    expected = checksums.get(zip_path.name)
    if expected is None:
        return {"status": "UNAVAILABLE", "actual": actual, "expected": None,
                "message": f"No checksum entry found for {zip_path.name}."}
    matched = actual == expected
    return {"status": "PASS" if matched else "FAIL", "actual": actual, "expected": expected,
            "message": "SHA-256 matches the official checksum." if matched else "SHA-256 mismatch; extraction is refused."}


def _official_url(url: str, tag: str) -> bool:
    return url.startswith(f"https://github.com/{GITHUB_REPOSITORY}/releases/download/{tag}/")


def _download(url: str, destination: Path, expected_size: int | None, timeout: float = 120) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "Tiger-Colibri-Phase1/1.0"})
    temp = destination.with_suffix(destination.suffix + ".part")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response, temp.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        if expected_size is not None and temp.stat().st_size != expected_size:
            raise IOError(f"Download size mismatch: expected {expected_size}, received {temp.stat().st_size}")
        os.replace(temp, destination)
    finally:
        if temp.exists():
            temp.unlink()


def download_release(release: dict[str, Any]) -> dict[str, Any]:
    if not _download_lock.acquire(blocking=False):
        raise RuntimeError("A Colibri download is already running")
    try:
        tag = str(release.get("tag", ""))
        asset = release.get("windows_asset") or {}
        checksum_asset = release.get("checksum_asset") or {}
        if not tag or not asset.get("url") or not _official_url(asset["url"], tag):
            raise ValueError("Release does not contain a whitelisted official Windows asset")
        download_dir = (VENDOR_DIR / tag / "downloads").resolve()
        if VENDOR_DIR.resolve() not in download_dir.parents:
            raise ValueError("Invalid release path")
        download_dir.mkdir(parents=True, exist_ok=True)
        zip_path = download_dir / Path(asset["name"]).name
        _download(asset["url"], zip_path, asset.get("size"))
        checksum_path = None
        if checksum_asset.get("url"):
            if not _official_url(checksum_asset["url"], tag):
                raise ValueError("Checksum URL is not from the official release")
            checksum_path = download_dir / Path(checksum_asset["name"]).name
            _download(checksum_asset["url"], checksum_path, checksum_asset.get("size"))
        verification = verify_checksum(zip_path, checksum_path)
        return {"status": "PASS" if verification["status"] != "FAIL" else "FAIL", "tag": tag,
                "zip_path": str(zip_path), "checksum_path": str(checksum_path) if checksum_path else None,
                "downloaded_bytes": zip_path.stat().st_size, "checksum": verification}
    except (OSError, urllib.error.URLError, urllib.error.HTTPError, ValueError) as exc:
        return {"status": "FAIL", "error": str(exc)}
    finally:
        _download_lock.release()


def validate_zip_members(names: list[str], destination: Path) -> None:
    base = destination.resolve()
    for name in names:
        normalized = name.replace("\\", "/")
        pure = PurePosixPath(normalized)
        if pure.is_absolute() or ".." in pure.parts or re.match(r"^[A-Za-z]:", normalized):
            raise ValueError(f"Unsafe ZIP member path: {name}")
        resolved = (base / Path(*pure.parts)).resolve()
        if resolved != base and base not in resolved.parents:
            raise ValueError(f"ZIP member escapes runtime directory: {name}")


def extract_release(download: dict[str, Any]) -> dict[str, Any]:
    if download.get("status") != "PASS":
        return {"status": "FAIL", "error": "A successful download is required before extraction"}
    checksum = download.get("checksum", {})
    if checksum.get("status") == "FAIL":
        return {"status": "FAIL", "error": "Checksum mismatch; extraction refused"}
    zip_path = Path(str(download.get("zip_path", ""))).resolve()
    tag = str(download.get("tag", ""))
    allowed_download = (VENDOR_DIR / tag / "downloads").resolve()
    runtime = (VENDOR_DIR / tag / "runtime").resolve()
    if allowed_download not in zip_path.parents or VENDOR_DIR.resolve() not in runtime.parents:
        return {"status": "FAIL", "error": "Installation path validation failed"}
    if not zip_path.exists():
        return {"status": "FAIL", "error": "Downloaded ZIP does not exist"}
    try:
        runtime.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as archive:
            validate_zip_members(archive.namelist(), runtime)
            archive.extractall(runtime)
        files = [str(p.relative_to(runtime)) for p in runtime.rglob("*") if p.is_file()]
        executables = [name for name in files if Path(name).suffix.lower() in {".exe", ".cmd", ".bat"}]
        qwen = next((str(p) for p in runtime.rglob("qwen36.exe") if p.is_file()), None)
        launchers = [str(p) for p in runtime.rglob("*") if p.is_file() and p.name.lower() in {"coli.cmd", "colibri.cmd", "coli.exe"}]
        return {"status": "PASS" if qwen else "FAIL", "runtime_path": str(runtime), "files": files,
                "executables": executables, "qwen36_path": qwen, "launchers": launchers,
                "error": None if qwen else "qwen36.exe was not found after extraction"}
    except (OSError, zipfile.BadZipFile, ValueError) as exc:
        return {"status": "FAIL", "error": str(exc), "runtime_path": str(runtime)}
