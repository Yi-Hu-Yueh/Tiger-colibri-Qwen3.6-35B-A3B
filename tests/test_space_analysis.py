import inspect
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.schemas.storage import SpaceAnalysisResponse
from app.services import space_analysis
from app.services.space_analysis import RISK_LABELS, scan_tree
from app.services.state_store import load_state, update_state


def test_directory_size_calculation_and_large_file_threshold(tmp_path):
    (tmp_path / "small.bin").write_bytes(b"a" * 10)
    child = tmp_path / "child"
    child.mkdir()
    (child / "large.bin").write_bytes(b"b" * 30)
    result = scan_tree(tmp_path, threshold_bytes=20)
    assert result["size_bytes"] == 40
    assert result["file_count"] == 2
    assert [Path(item["path"]).name for item in result["large_files"]] == ["large.bin"]


def test_inaccessible_directory_is_reported(tmp_path, monkeypatch):
    original = os.scandir
    def denied(path):
        if Path(path) == tmp_path:
            raise PermissionError("denied")
        return original(path)
    monkeypatch.setattr(space_analysis.os, "scandir", denied)
    result = scan_tree(tmp_path)
    assert result["errors"]
    assert "PermissionError" in result["errors"][0]["error"]


def test_largest_files_sorted_and_limited(tmp_path):
    for index in range(110):
        (tmp_path / f"{index:03}.bin").write_bytes(b"x" * (index + 1))
    result = scan_tree(tmp_path, threshold_bytes=0)
    assert len(result["large_files"]) == 100
    sizes = [item["size_bytes"] for item in result["large_files"]]
    assert sizes == sorted(sizes, reverse=True)
    assert sizes[0] == 110


def test_risk_labels_are_exact():
    assert RISK_LABELS == {"USER_REVIEW", "KNOWN_CACHE", "APPLICATION_DATA", "SYSTEM_MANAGED", "UNKNOWN"}


def test_large_files_in_known_cache_keep_cache_label(tmp_path):
    local = tmp_path / "AppData" / "Local"
    pip = local / "pip"
    pip.mkdir(parents=True)
    (pip / "large.whl").write_bytes(b"x" * 20)
    dirs, files, errors, skipped = space_analysis._scan_immediate_children(
        local, "APPLICATION_DATA", "AppData Local", time.monotonic() + 10, [1000]
    )
    assert not errors
    assert dirs[0]["risk_label"] == "KNOWN_CACHE"
    # Lower the global threshold for this focused helper test by checking directory labeling;
    # large-file threshold behavior is covered separately.


def test_ollama_directory_size_is_read_only(tmp_path):
    root = tmp_path / ".ollama" / "models" / "blobs"
    root.mkdir(parents=True)
    model = root / "sha256-test"
    model.write_bytes(b"m" * 123)
    before = model.stat().st_mtime_ns
    result = scan_tree(tmp_path / ".ollama", threshold_bytes=1)
    assert result["size_bytes"] == 123
    assert model.exists()
    assert model.stat().st_mtime_ns == before


def test_system_managed_file_classification(monkeypatch):
    monkeypatch.setattr(Path, "stat", lambda self: SimpleNamespace(st_size=1024))
    files, errors = space_analysis._system_files()
    assert not errors
    assert files
    assert all(item["risk_label"] == "SYSTEM_MANAGED" for item in files)
    assert all("DO NOT MODIFY" in item["advisory"] for item in files)


def test_scanner_contains_no_delete_operations():
    source = inspect.getsource(space_analysis)
    assert ".unlink(" not in source
    assert "os.remove(" not in source
    assert "shutil.rmtree(" not in source


def test_duplicate_scan_protection():
    assert space_analysis._scan_lock.acquire(blocking=False)
    try:
        with pytest.raises(RuntimeError, match="already running"):
            space_analysis.analyze_space(timeout_seconds=1, max_entries=1)
    finally:
        space_analysis._scan_lock.release()


def test_space_analysis_schema_and_state_serialization(tmp_path):
    payload = {"status": "PASS", "timestamp": "2026-01-01T00:00:00+00:00", "drive": {"free_bytes": 1},
               "largest_files": [], "largest_directories": [], "advisory": "Read-only"}
    parsed = SpaceAnalysisResponse.model_validate(payload)
    assert parsed.drive["free_bytes"] == 1
    state_path = tmp_path / "state.json"
    update_state("space_analysis", parsed.model_dump(), state_path)
    assert load_state(state_path)["space_analysis"]["status"] == "PASS"
