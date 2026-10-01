from __future__ import annotations

from typing import Any


def ram_check(total_gib: float | None) -> dict[str, str]:
    if total_gib is None:
        return {"status": "WARN", "message": "RAM total could not be determined."}
    if total_gib < 20:
        return {"status": "WARN", "message": "16 GB RAM is below the currently documented comfortable/recommended Qwen3.6 configuration. CPU streaming feasibility must be tested in Phase 3."}
    return {"status": "PASS", "message": f"{total_gib:.1f} GiB physical RAM detected."}


def storage_type_check(classification: str | None) -> dict[str, str]:
    mapping = {
        "NVME": ("PASS", "NVMe storage detected."),
        "SATA_SSD": ("WARN", "SATA SSD detected; performance may be lower than NVMe."),
        "SSD_UNKNOWN_BUS": ("WARN", "SSD detected but its underlying bus is hidden or inconclusive."),
        "HDD": ("FAIL", "HDD storage is unsuitable for the intended Qwen3.6 workload."),
        "UNKNOWN": ("FAIL", "Storage type is unknown; Phase 2 remains blocked pending clear hardware evidence."),
    }
    aliases = {"NVMe": "NVME", "SATA SSD": "SATA_SSD", "Unknown": "UNKNOWN"}
    key = aliases.get(classification or "UNKNOWN", classification or "UNKNOWN")
    status, message = mapping.get(key, mapping["UNKNOWN"])
    return {"status": status, "message": message}


def _check(name: str, status: str, message: str, required: bool = True) -> dict[str, Any]:
    return {"name": name, "status": status, "message": message, "required": required}


def evaluate_phase1(state: dict[str, Any]) -> dict[str, Any]:
    hardware = state.get("hardware") or {}
    storage = state.get("storage") or {}
    download_validation = state.get("download_root_validation") or storage.get("download_root_validation") or {}
    runtime_validation = state.get("model_runtime_validation") or storage.get("model_runtime_validation") or {}
    release = state.get("colibri_release") or {}
    download = state.get("colibri_download") or {}
    extraction = state.get("colibri_installation") or {}
    smoke = state.get("smoke_test") or {}
    ollama = state.get("ollama") or {}
    checks: list[dict[str, Any]] = []

    windows = hardware.get("windows", {})
    checks.append(_check("Windows", "PASS" if windows.get("detected") else "FAIL", "Windows detected." if windows.get("detected") else "Supported Windows was not detected."))
    python = hardware.get("python", {})
    python_version = str(python.get("version", "unknown"))
    checks.append(_check("Python runtime", "PASS" if python_version.startswith("3.11.") else "FAIL", f"Python {python_version}; Phase 1-R requires Python 3.11.x."))
    project = storage.get("project_storage") or {}
    project_disk = next((d for d in storage.get("physical_disks", []) if d.get("disk_number") == project.get("disk_number")), {})
    project_class = (project_disk.get("classification") or {}).get("classification", "UNKNOWN")
    checks.append(_check("PROJECT_ROOT", "PASS",
                         f"Project code remains on {project_class} storage at {storage.get('project_root') or project.get('path')}; project media is not the acquisition gate.", required=False))
    configured_download = state.get("download_root") or storage.get("download_root")
    download_status = "PASS" if download_validation.get("status") == "PASS" else "FAIL"
    download_message = (f"{configured_download or 'DOWNLOAD_ROOT unset'}; class={download_validation.get('storage_class', 'UNKNOWN')}; "
                        f"free={download_validation.get('free_gib', 'unknown')} GiB; writable={bool(download_validation.get('writable'))}. "
                        "HDD is allowed for acquisition/archive but is not accepted as practical inference storage.")
    checks.append(_check("DOWNLOAD_STORAGE_GATE", download_status, download_message))
    runtime_root = state.get("model_runtime_root") or storage.get("model_runtime_root")
    if not runtime_root:
        runtime_status = "NOT READY"
        runtime_message = "MODEL_RUNTIME_ROOT is unset. Actual Colibrì inference/expert-streaming storage is not ready."
    else:
        classification = runtime_validation.get("classification", "UNKNOWN")
        runtime_eval = storage_type_check(classification)
        runtime_status = runtime_eval["status"]
        if not runtime_validation.get("writable") or not runtime_validation.get("local_filesystem"):
            runtime_status = "FAIL"
        runtime_message = f"{runtime_root}; {runtime_eval['message']} free={runtime_validation.get('free_gib', 'unknown')} GiB."
    checks.append(_check("RUNTIME_STORAGE_GATE", runtime_status, runtime_message, required=False))
    ram_eval = ram_check(hardware.get("ram", {}).get("total_gib"))
    checks.append(_check("System RAM", ram_eval["status"], ram_eval["message"]))
    devices = hardware.get("gpu", {}).get("devices", [])
    gpu_msg = "GPU detection is informational. CUDA is not being validated in Phase 1; GTX 1070 / Pascal CUDA evaluation is deferred to Phase 4."
    checks.append(_check("GPU", "WARN" if devices else "WARN", gpu_msg, required=False))
    checks.append(_check("Colibri release metadata", "PASS" if release.get("status") == "PASS" else "FAIL", release.get("error") or "Official release metadata and Windows asset found."))
    checks.append(_check("Prebuilt download", "PASS" if download.get("status") == "PASS" else "FAIL", download.get("error") or "Official Windows prebuilt downloaded."))
    checksum = download.get("checksum", {})
    checksum_status = checksum.get("status", "FAIL")
    if checksum_status == "UNAVAILABLE":
        checks.append(_check("SHA-256", "WARN", checksum.get("message", "Official checksum unavailable.")))
    else:
        checks.append(_check("SHA-256", "PASS" if checksum_status == "PASS" else "FAIL", checksum.get("message", "Checksum has not passed.")))
    checks.append(_check("Extraction", "PASS" if extraction.get("status") == "PASS" else "FAIL", extraction.get("error") or "Local extraction completed."))
    checks.append(_check("qwen36.exe", "PASS" if extraction.get("qwen36_path") else "FAIL", "qwen36.exe found." if extraction.get("qwen36_path") else "qwen36.exe not found."))
    checks.append(_check("Smoke launch", "PASS" if smoke.get("status") == "PASS" else "FAIL", smoke.get("classification") or smoke.get("error") or "Not run."))
    ollama_ok = ollama.get("detected") is True
    ollama_message = "Ollama is detectable through read-only evidence and was not modified." if ollama_ok else "Ollama could not be detected by PATH, common locations, processes, or its local API."
    checks.append(_check("Ollama baseline", "PASS" if ollama_ok else "FAIL", ollama_message))

    required_fail = any(c["status"] == "FAIL" and c["required"] for c in checks)
    warnings = any(c["status"] == "WARN" for c in checks)
    verdict = "FAIL" if required_fail else ("CONDITIONAL PASS" if warnings else "PASS")
    inference_readiness = "PASS" if runtime_status == "PASS" else ("CONDITIONAL" if runtime_status == "WARN" else "NOT READY")
    return {"phase": 1, "verdict": verdict,
            "phase1_status": "ACQUISITION READY / RUNTIME NOT READY" if not required_fail and inference_readiness == "NOT READY" else verdict,
            "acquisition_readiness": "PASS" if not required_fail else "FAIL",
            "inference_readiness": inference_readiness,
            "download_storage_gate": {"status": download_status, "root": configured_download,
                                      "storage_class": download_validation.get("storage_class"),
                                      "free_bytes": download_validation.get("free_bytes")},
            "runtime_storage_gate": {"status": runtime_status, "root": runtime_root,
                                     "storage_class": runtime_validation.get("classification") if runtime_root else None},
            "checks": checks, "model_download_locked": True,
            "message": "Model acquisition remains locked until separately authorized Phase 2; inference storage is evaluated independently."}
