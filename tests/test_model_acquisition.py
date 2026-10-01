from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.config import DOWNLOAD_ROOT, MODEL_DESTINATION, MODEL_REPO_ID
from app.main import app
from app.schemas.model import ModelAcquisitionStatus
from app.services import model_acquisition as acquisition
from app.services.state_store import load_state, update_state


class FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self):
        return f"recommended int4 gs64 hf download {MODEL_REPO_ID}".encode()


def fake_opener(*_args, **_kwargs):
    return FakeResponse()


def remote_files():
    payloads = {
        "config.json": json.dumps({"model_type": "qwen3_5_moe", "hidden_size": 2048, "num_hidden_layers": 40, "num_experts": 256, "num_experts_per_tok": 8}).encode(),
        "tokenizer.json": b'{"model":{"type":"BPE"}}',
        "qwen36_meta.json": b'{"expert_gs":64}',
        "model.safetensors": b"weights",
    }
    return payloads, [{"path": name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest() if name.endswith("safetensors") else None} for name, data in payloads.items()]


def fake_api(files):
    siblings = []
    for item in files:
        lfs = SimpleNamespace(sha256=item["sha256"]) if item["sha256"] else None
        siblings.append(SimpleNamespace(rfilename=item["path"], size=item["size"], lfs=lfs))
    info = SimpleNamespace(sha="a" * 40, last_modified=None, siblings=siblings)
    return SimpleNamespace(model_info=lambda *_args, **_kwargs: info)


@pytest.fixture
def isolated(monkeypatch):
    test_root = Path(tempfile.mkdtemp(prefix="phase2-test-", dir=Path.cwd() / "data"))
    root = test_root / "downloads"
    destination = root / "qwen36-35b-a3b-colibri-i4-gs64"
    monkeypatch.setattr(acquisition, "DOWNLOAD_ROOT", root)
    monkeypatch.setattr(acquisition, "MODEL_DESTINATION", destination)
    monkeypatch.setattr(acquisition, "HF_CACHE_DIR", root / ".cache")
    monkeypatch.setattr(acquisition, "HF_XET_CACHE_DIR", root / ".xet")
    memory = {}
    monkeypatch.setattr(acquisition, "load_state", lambda: {"phase2_acquisition": memory.copy()} if memory else {})
    monkeypatch.setattr(acquisition, "update_state", lambda _key, value: memory.update(value))
    yield root, destination, memory
    shutil.rmtree(test_root)


def test_fixed_repo_and_destination():
    assert MODEL_REPO_ID == "Kreuzzelg/qwen36-35b-a3b-colibri-i4-gs64"
    assert MODEL_DESTINATION == DOWNLOAD_ROOT / "qwen36-35b-a3b-colibri-i4-gs64"
    assert MODEL_DESTINATION.drive == "D:"


def test_c_drive_destination_rejected(monkeypatch):
    with pytest.raises(ValueError, match="D:"):
        acquisition.validate_destination(Path(r"C:\model"))


def test_revision_metadata_and_free_space(isolated, monkeypatch):
    _root, destination, _memory = isolated
    _payloads, files = remote_files()
    metadata = acquisition.resolve_metadata(fake_api(files))
    assert metadata["revision"] == "a" * 40
    assert metadata["remote_expected_bytes"] == sum(x["size"] for x in files)
    monkeypatch.setattr(acquisition.shutil, "disk_usage", lambda _p: SimpleNamespace(free=100 * acquisition.GIB))
    assert acquisition.free_space_precheck(metadata["remote_expected_bytes"], destination)["status"] == "PASS"
    monkeypatch.setattr(acquisition.shutil, "disk_usage", lambda _p: SimpleNamespace(free=1))
    assert acquisition.free_space_precheck(None, destination)["status"] == "FAIL"


def test_complete_transition_and_persistence(isolated, monkeypatch):
    _root, destination, memory = isolated
    payloads, files = remote_files()
    monkeypatch.setattr(acquisition.shutil, "disk_usage", lambda _p: SimpleNamespace(free=100 * acquisition.GIB))

    def downloader(**kwargs):
        assert kwargs["repo_id"] == MODEL_REPO_ID
        assert kwargs["revision"] == "a" * 40
        for name, data in payloads.items():
            path = Path(kwargs["local_dir"]) / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        return kwargs["local_dir"]

    state = acquisition.acquire_snapshot(api=fake_api(files), downloader=downloader, docs_opener=fake_opener)
    assert state["download_status"] == "VERIFIED"
    assert state["integrity_status"] == "VERIFIED_METADATA_AND_HASHES"
    assert memory["revision"] == "a" * 40
    assert memory["download_started_at"] and memory["download_completed_at"]


def test_partial_state_is_preserved_for_resume(isolated, monkeypatch):
    _root, destination, memory = isolated
    _payloads, files = remote_files()
    monkeypatch.setattr(acquisition.shutil, "disk_usage", lambda _p: SimpleNamespace(free=100 * acquisition.GIB))

    def interrupted(**kwargs):
        path = Path(kwargs["local_dir"]) / "model.safetensors"
        path.write_bytes(b"partial")
        raise ConnectionError("network interruption")

    with pytest.raises(ConnectionError):
        acquisition.acquire_snapshot(api=fake_api(files), downloader=interrupted, docs_opener=fake_opener)
    assert memory["download_status"] == "PARTIAL"
    assert (destination / "model.safetensors").read_bytes() == b"partial"
    assert memory["resume_supported"] is True


def test_duplicate_download_lock():
    acquisition._download_lock.acquire()
    try:
        with pytest.raises(RuntimeError, match="already running"):
            acquisition.acquire_snapshot()
    finally:
        acquisition._download_lock.release()


def test_file_size_and_hash_comparison(tmp_path):
    payload = b"verified payload"
    (tmp_path / "file.bin").write_bytes(payload)
    remote = [{"path": "file.bin", "size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}]
    result = acquisition.compare_repository_files(tmp_path, remote)
    assert result["file_list_match"] and result["file_sizes_match"] and result["hashes_match"]
    assert result["hash_files_verified"] == 1


def test_required_structure_expert_group_and_family(tmp_path):
    payloads, _files = remote_files()
    for name, data in payloads.items():
        (tmp_path / name).write_bytes(data)
    result = acquisition.validate_structure(tmp_path)
    assert result["status"] == "PASS"
    assert result["expert_gs"] == 64
    assert result["geometry"]["model_type"] == "qwen3_5_moe"
    (tmp_path / "qwen36_meta.json").write_text('{"expert_gs":32}')
    assert acquisition.validate_structure(tmp_path)["status"] == "FAIL"


def test_atomic_state_serialization_preserves_phase1(tmp_path):
    path = tmp_path / "state.json"
    update_state("phase1_gate", {"verdict": "PASS"}, path)
    update_state("phase2_acquisition", {"download_status": "PARTIAL", "revision": "abc"}, path)
    data = load_state(path)
    assert data["phase1_gate"]["verdict"] == "PASS"
    assert data["phase2_acquisition"]["revision"] == "abc"


def test_api_schema_and_no_arbitrary_repository_input(monkeypatch):
    state = acquisition._base_state()
    state["revision"] = "abc"
    ModelAcquisitionStatus.model_validate(state)
    monkeypatch.setattr("app.api.model.get_status", lambda: state)
    client = TestClient(app)
    response = client.get("/api/model/qwen36/status")
    assert response.status_code == 200
    assert response.json()["repo_id"] == MODEL_REPO_ID
    operation = app.openapi()["paths"]["/api/model/qwen36/download"]["post"]
    assert "requestBody" not in operation
