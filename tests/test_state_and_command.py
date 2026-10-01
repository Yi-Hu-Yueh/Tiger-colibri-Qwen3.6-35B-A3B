import json
import sys

from app.services.command_runner import run_command
from app.services.colibri_smoke import classify_smoke
from app.services.state_store import load_state, save_state, update_state
from app.services.python_runtime import validate_python_interpreter


def test_state_serialization_round_trip(tmp_path):
    path = tmp_path / "state" / "phase1.json"
    save_state({"phase": 1, "value": "ok"}, path)
    assert load_state(path)["value"] == "ok"
    update_state("gate", {"verdict": "PASS"}, path)
    parsed = json.loads(path.read_text(encoding="utf-8"))
    assert parsed["gate"]["verdict"] == "PASS"
    assert "updated_at" in parsed


def test_selected_model_root_persistence(tmp_path):
    path = tmp_path / "phase1.json"
    root = r"C:\TigerModels\Tiger-colibri-Qwen3.6-35B-A3B"
    update_state("model_storage_root", root, path)
    assert load_state(path)["model_storage_root"] == root


def test_download_and_runtime_roots_persist_separately(tmp_path):
    path = tmp_path / "phase1.json"
    download = r"D:\TigerModels\Tiger-colibri-Qwen3.6-35B-A3B\downloads"
    update_state("download_root", download, path)
    update_state("model_runtime_root", None, path)
    state = load_state(path)
    assert state["download_root"] == download
    assert state["model_runtime_root"] is None


def test_command_timeout_is_reported():
    result = run_command([sys.executable, "-c", "import time; time.sleep(2)"], timeout=0.05)
    assert result["timed_out"] is True
    assert result["exit_code"] is None


def test_command_error_is_reported():
    result = run_command(["this-command-does-not-exist-phase1"], timeout=1)
    assert result["error"]
    assert result["timed_out"] is False


def test_nonzero_missing_model_response_proves_binary_launched():
    result = {"exit_code": 1, "stdout": "", "stderr": "engine was started without a model", "timed_out": False, "error": None}
    assert classify_smoke(result) == "MODEL_REQUIRED"


def test_python_interpreter_requires_311(tmp_path):
    executable = tmp_path / "python.exe"
    executable.touch()
    good = lambda command, timeout: {"exit_code": 0, "stdout": "3.11.3", "stderr": "", "error": None}
    bad = lambda command, timeout: {"exit_code": 0, "stdout": "3.14.7", "stderr": "", "error": None}
    assert validate_python_interpreter(str(executable), good)["valid"] is True
    assert validate_python_interpreter(str(executable), bad)["valid"] is False
