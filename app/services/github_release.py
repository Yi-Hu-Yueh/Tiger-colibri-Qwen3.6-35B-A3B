from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any

from app.config import GITHUB_RELEASE_API, GITHUB_REPOSITORY

WINDOWS_ASSET = re.compile(r"^colibri-v.*-windows-x86_64\.zip$", re.IGNORECASE)


def select_release_assets(assets: list[dict[str, Any]]) -> dict[str, Any]:
    windows = None
    checksum = None
    for asset in assets:
        name = str(asset.get("name", ""))
        url = str(asset.get("browser_download_url", ""))
        if not url.startswith(f"https://github.com/{GITHUB_REPOSITORY}/releases/download/"):
            continue
        clean = {"name": name, "url": url, "size": asset.get("size"), "content_type": asset.get("content_type")}
        if WINDOWS_ASSET.fullmatch(name):
            windows = clean
        elif name.upper() == "SHA256SUMS.TXT":
            checksum = clean
    return {"windows_asset": windows, "checksum_asset": checksum}


def fetch_latest_release(timeout: float = 20) -> dict[str, Any]:
    request = urllib.request.Request(
        GITHUB_RELEASE_API,
        headers={"Accept": "application/vnd.github+json", "User-Agent": "Tiger-Colibri-Phase1/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
        return {"status": "FAIL", "error": str(exc), "repository": GITHUB_REPOSITORY,
                "checked_at": datetime.now(timezone.utc).isoformat()}
    if payload.get("draft") or not payload.get("tag_name"):
        return {"status": "FAIL", "error": "Latest GitHub response is not a published release", "repository": GITHUB_REPOSITORY}
    selected = select_release_assets(payload.get("assets", []))
    status = "PASS" if selected["windows_asset"] else "FAIL"
    return {
        "status": status, "error": None if status == "PASS" else "No official Windows x86_64 ZIP asset found",
        "repository": GITHUB_REPOSITORY, "tag": payload.get("tag_name"), "published_at": payload.get("published_at"),
        "release_url": payload.get("html_url"), "checked_at": datetime.now(timezone.utc).isoformat(), **selected,
    }
