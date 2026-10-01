from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from app.config import OLLAMA_MODEL_DESTINATION, OLLAMA_MODEL_SOURCE
from app.services import ollama_migration as migration
from app.services.state_store import load_state, update_state


def make_store(root: Path) -> bytes:
    payload = b"verified ollama blob"
    digest = hashlib.sha256(payload).hexdigest()
    blob = root / "blobs" / f"sha256-{digest}"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(payload)
    manifest = root / "manifests" / "registry.ollama.ai" / "library" / "gemma3" / "1b"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"schemaVersion": 2}), encoding="utf-8")
    return payload


def test_exact_allowed_paths():
    assert str(OLLAMA_MODEL_SOURCE) == r"C:\Users\user\.ollama\models"
    assert str(OLLAMA_MODEL_DESTINATION) == r"D:\TigerModels\Ollama"


def test_source_rejects_ollama_root(monkeypatch, tmp_path):
    monkeypatch.setattr(migration, "OLLAMA_MODEL_SOURCE", tmp_path / ".ollama" / "models")
    (tmp_path / ".ollama" / "models").mkdir(parents=True)
    with pytest.raises(ValueError):
        migration.validate_source(tmp_path / ".ollama")


def test_destination_rejects_wrong_path(monkeypatch, tmp_path):
    monkeypatch.setattr(migration, "OLLAMA_MODEL_DESTINATION", tmp_path / "Ollama")
    with pytest.raises(ValueError):
        migration.validate_destination(Path(r"C:\Users\user\.ollama\models"), tmp_path)


def test_inventory_comparison(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "destination"
    make_store(source)
    make_store(destination)
    assert migration.compare_inventories(migration.inventory(source), migration.inventory(destination))["status"] == "PASS"
    next(destination.rglob("sha256-*")).write_bytes(b"wrong")
    assert migration.compare_inventories(migration.inventory(source), migration.inventory(destination))["status"] == "FAIL"


def test_blob_sha256_filename_verification(tmp_path):
    make_store(tmp_path)
    result = migration.verify_blob_hashes(tmp_path)
    assert result["status"] == "PASS" and result["hashed_blob_count"] == 1
    next(tmp_path.rglob("sha256-*")).write_bytes(b"tampered")
    assert migration.verify_blob_hashes(tmp_path)["status"] == "FAIL"


def test_manifest_preservation(tmp_path):
    make_store(tmp_path)
    result = migration.verify_manifests(tmp_path)
    assert result == {"status": "PASS", "manifest_count": 1, "errors": []}


def test_user_scope_environment_configuration(monkeypatch):
    values = {"value": None}
    monkeypatch.setattr(migration, "set_user_ollama_models", lambda value: values.update(value=value))
    monkeypatch.setattr(migration, "get_user_ollama_models", lambda: values["value"])
    migration.set_user_ollama_models(r"D:\TigerModels\Ollama")
    assert migration.get_user_ollama_models() == r"D:\TigerModels\Ollama"


def test_rollback_handling(monkeypatch):
    values = {"value": r"D:\TigerModels\Ollama"}
    monkeypatch.setattr(migration, "set_user_ollama_models", lambda value: values.update(value=value))
    monkeypatch.setattr(migration, "get_user_ollama_models", lambda: values["value"])
    assert migration.rollback_user_environment(None)["status"] == "PASS"


def test_model_inventory_comparison():
    assert migration.compare_model_names(["a", "b"], ["b", "a"])["status"] == "PASS"
    assert migration.compare_model_names(["a"], ["b"])["status"] == "FAIL"


def test_deletion_gate_and_key_protection(monkeypatch, tmp_path):
    models = tmp_path / ".ollama" / "models"
    make_store(models)
    monkeypatch.setattr(migration, "OLLAMA_MODEL_SOURCE", models)
    evidence = {key: {"status": "PASS"} for key in ("copy_verification", "blob_verification", "environment", "restart", "api", "inventory", "smoke")}
    evidence["smoke"] = {"status": "FAIL"}
    with pytest.raises(PermissionError):
        migration.delete_old_model_store(evidence, models)
    key = tmp_path / ".ollama" / "id_ed25519"
    key.write_text("untouched")
    evidence["smoke"] = {"status": "PASS"}
    assert migration.delete_old_model_store(evidence, models)["status"] == "PASS"
    assert key.read_text() == "untouched"


def test_state_persistence_preserves_phase1(tmp_path):
    state = tmp_path / "state.json"
    update_state("phase1_gate", {"verdict": "PASS"}, state)
    update_state("ollama_storage_migration", {"status": "PASS"}, state)
    assert load_state(state)["phase1_gate"]["verdict"] == "PASS"

