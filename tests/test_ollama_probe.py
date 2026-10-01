import io
import json
import urllib.error

from app.services import ollama_probe


class ApiResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def test_ollama_api_success_and_failure():
    success = lambda request, timeout: ApiResponse(json.dumps({"version": "1.2.3"}).encode())
    assert ollama_probe.probe_ollama_api(opener=success)["version"] == "1.2.3"
    def failure(request, timeout):
        raise urllib.error.URLError("offline")
    result = ollama_probe.probe_ollama_api(opener=failure)
    assert result["reachable"] is False
    assert "offline" in result["error"]
    def timeout(request, timeout):
        raise TimeoutError("timed out")
    timed_out = ollama_probe.probe_ollama_api(opener=timeout)
    assert timed_out["reachable"] is False
    assert "timed out" in timed_out["error"]


def test_ollama_one_successful_method_establishes_presence(monkeypatch):
    monkeypatch.setattr(ollama_probe, "_find_executables", lambda: ([], []))
    monkeypatch.setattr(ollama_probe, "_process_evidence", lambda: (True, [{"name": "ollama", "id": 7, "path": None}]))
    monkeypatch.setattr(ollama_probe, "probe_ollama_api", lambda: {"reachable": False, "version": None, "error": "offline"})
    result = ollama_probe.probe_ollama()
    assert result["detected"] is True
    assert result["process_detected"] is True
    assert "running process" in result["detection_methods"]
