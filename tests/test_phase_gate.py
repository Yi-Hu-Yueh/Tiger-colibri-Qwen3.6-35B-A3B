from app.services.phase_gate import evaluate_phase1, ram_check, storage_type_check


def complete_state(**overrides):
    state = {
        "hardware": {"windows": {"detected": True}, "python": {"meets_minimum": True, "version": "3.11.9"},
                     "ram": {"total_gib": 64}, "gpu": {"devices": []}},
        "storage": {"project_storage": {"disk_number": 0}, "physical_disks": [
            {"disk_number": 0, "classification": {"classification": "HDD"}}
        ]},
        "model_storage_root": "C:\\TigerModels\\Tiger-colibri-Qwen3.6-35B-A3B",
        "model_storage_validation": {"classification": "NVME", "free_gib": 100, "writable": True,
                                     "local_filesystem": True, "status": "PASS"},
        "download_root": r"D:\TigerModels\Tiger-colibri-Qwen3.6-35B-A3B\downloads",
        "download_root_validation": {"status": "PASS", "storage_class": "HDD", "free_gib": 100,
                                     "free_bytes": 100 * 2**30, "writable": True, "local_filesystem": True},
        "model_runtime_root": None,
        "model_runtime_validation": None,
        "colibri_release": {"status": "PASS"},
        "colibri_download": {"status": "PASS", "checksum": {"status": "PASS", "message": "matches"}},
        "colibri_installation": {"status": "PASS", "qwen36_path": "qwen36.exe"},
        "smoke_test": {"status": "PASS", "classification": "USAGE_DISPLAYED"},
        "ollama_baseline_present": True,
        "ollama": {"detected": True},
    }
    state.update(overrides)
    return state


def test_complete_gate_has_no_required_failures():
    result = evaluate_phase1(complete_state())
    assert result["verdict"] == "CONDITIONAL PASS"  # informational GPU warning
    assert not any(c["required"] and c["status"] == "FAIL" for c in result["checks"])
    assert result["download_storage_gate"]["status"] == "PASS"
    assert result["runtime_storage_gate"]["status"] == "NOT READY"
    assert result["acquisition_readiness"] == "PASS"
    assert result["inference_readiness"] == "NOT READY"


def test_download_free_space_under_threshold_fails_acquisition_gate():
    state = complete_state()
    state["download_root_validation"]["free_gib"] = 29.99
    state["download_root_validation"]["status"] = "FAIL"
    assert evaluate_phase1(state)["verdict"] == "FAIL"


def test_ram_16_gib_warns_not_fails():
    result = ram_check(16)
    assert result["status"] == "WARN"
    assert "Phase 3" in result["message"]


def test_storage_gate_mapping():
    assert storage_type_check("NVMe")["status"] == "PASS"
    assert storage_type_check("SATA SSD")["status"] == "WARN"
    assert storage_type_check("HDD")["status"] == "FAIL"
    assert storage_type_check("Unknown")["status"] == "FAIL"
