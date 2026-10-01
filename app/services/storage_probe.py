from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any

from app.config import DOWNLOAD_ROOT, MIN_FREE_GIB, PROJECT_ROOT
from app.services.command_runner import run_command

MODEL_DIRECTORY_NAME = r"TigerModels\Tiger-colibri-Qwen3.6-35B-A3B"
DOWNLOAD_DIRECTORY_NAME = MODEL_DIRECTORY_NAME + r"\downloads"


def classify_device(media_type: str | None, bus_type: str | None, friendly_name: str | None = None,
                    model: str | None = None, interface_type: str | None = None) -> dict[str, Any]:
    media = (media_type or "").strip().lower()
    bus = (bus_type or "").strip().lower()
    name = " ".join((friendly_name or "", model or "")).lower()
    interface = (interface_type or "").strip().lower()
    evidence: list[str] = []
    if media == "hdd" or "hard disk" in media:
        evidence.append(f"Get-PhysicalDisk MediaType={media_type}")
        if bus:
            evidence.append(f"BusType={bus_type} (does not override HDD media evidence)")
        return {"classification": "HDD", "confidence": "high", "evidence": evidence}
    ssd_signal = media == "ssd" or "solid state" in media
    nvme_signals = []
    if bus == "nvme":
        nvme_signals.append(f"Get-PhysicalDisk BusType={bus_type}")
    if interface == "nvme":
        nvme_signals.append(f"Win32_DiskDrive InterfaceType={interface_type}")
    if re.search(r"\bnvme\b", name):
        nvme_signals.append("Physical-disk name/model contains NVMe")
    if nvme_signals and (ssd_signal or len(nvme_signals) >= 2):
        evidence.extend(nvme_signals)
        if ssd_signal:
            evidence.append(f"Get-PhysicalDisk MediaType={media_type}")
        return {"classification": "NVME", "confidence": "high" if ssd_signal else "medium", "evidence": evidence}
    sata_signal = bus in {"sata", "ata"} or interface in {"sata", "ata", "ide"}
    if ssd_signal and sata_signal:
        evidence.extend([f"Get-PhysicalDisk MediaType={media_type}",
                         f"Storage interface evidence BusType={bus_type or 'UNKNOWN'}, InterfaceType={interface_type or 'UNKNOWN'}"])
        return {"classification": "SATA_SSD", "confidence": "high", "evidence": evidence}
    if ssd_signal:
        evidence.extend([f"Get-PhysicalDisk MediaType={media_type}",
                         f"Bus/interface hidden or inconclusive ({bus_type or 'UNKNOWN'}/{interface_type or 'UNKNOWN'})"])
        return {"classification": "SSD_UNKNOWN_BUS", "confidence": "medium", "evidence": evidence}
    if ("ssd" in name or "solid state" in name) and sata_signal:
        evidence.extend(["Physical-disk name/model indicates SSD", f"Storage interface indicates {bus_type or interface_type}"])
        return {"classification": "SATA_SSD", "confidence": "medium", "evidence": evidence}
    if bus == "raid":
        evidence.append("BusType=RAID; controller may hide the underlying media/interface")
    if media and media != "unspecified":
        evidence.append(f"MediaType={media_type}")
    if not evidence:
        evidence.append("Windows did not expose decisive media and bus evidence")
    return {"classification": "UNKNOWN", "confidence": "low", "evidence": evidence}


def classify_storage(media_type: str | None, bus_type: str | None, friendly_name: str | None = None) -> str:
    labels = {"NVME": "NVMe", "SATA_SSD": "SATA SSD", "SSD_UNKNOWN_BUS": "Unknown", "HDD": "HDD", "UNKNOWN": "Unknown"}
    return labels[classify_device(media_type, bus_type, friendly_name)["classification"]]


def free_space_status(free_bytes: int | None, threshold_gib: float = MIN_FREE_GIB) -> str:
    return "PASS" if free_bytes is not None and free_bytes / 2**30 >= threshold_gib else "FAIL"


def _as_list(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return [value] if isinstance(value, dict) else []


def map_storage_records(raw: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    physical = {str(x.get("device_id")): x for x in _as_list(raw.get("physical_disks"))}
    cim = {str(x.get("index")): x for x in _as_list(raw.get("cim_disks"))}
    partitions = _as_list(raw.get("partitions"))
    volumes = _as_list(raw.get("volumes"))
    def clean_letter(value: Any) -> str:
        return str(value or "").replace("\x00", "").strip().upper()
    volume_by_letter = {clean_letter(v.get("drive_letter")): v for v in volumes if clean_letter(v.get("drive_letter"))}
    mapped_volumes: list[dict[str, Any]] = []
    disks: list[dict[str, Any]] = []
    disk_rows = _as_list(raw.get("disks"))
    known_numbers = {str(row.get("disk_number")) for row in disk_rows}
    for number in sorted(set(known_numbers) | set(physical) | set(cim), key=lambda x: int(x) if x.isdigit() else 9999):
        disk = next((d for d in disk_rows if str(d.get("disk_number")) == number), {})
        pd, cd = physical.get(number, {}), cim.get(number, {})
        friendly = pd.get("friendly_name") or disk.get("friendly_name") or cd.get("model")
        model = cd.get("model") or disk.get("model") or friendly
        media, bus, interface = pd.get("media_type") or disk.get("media_type"), pd.get("bus_type") or disk.get("bus_type"), cd.get("interface_type")
        classification = classify_device(media, bus, friendly, model, interface)
        disk_partitions = []
        for part in partitions:
            if str(part.get("disk_number")) != number:
                continue
            letter = clean_letter(part.get("drive_letter"))
            volume = dict(volume_by_letter.get(letter, {})) if letter else {}
            record = {"disk_number": int(number) if number.isdigit() else number,
                      "partition_number": part.get("partition_number"), "partition_size_bytes": part.get("size_bytes"),
                      "drive_letter": letter or None, "filesystem": volume.get("filesystem"), "volume_label": volume.get("label"),
                      "volume_size_bytes": volume.get("size_bytes"), "free_bytes": volume.get("free_bytes"),
                      "percent_free": volume.get("percent_free"), "drive_type": volume.get("drive_type"), "path": volume.get("path")}
            disk_partitions.append(record)
            if letter:
                mapped_volumes.append(record)
        disks.append({"disk_number": int(number) if number.isdigit() else number, "friendly_name": friendly or "UNKNOWN",
                      "model": model or "UNKNOWN", "serial_number": pd.get("serial_number") or disk.get("serial_number") or cd.get("serial_number") or "UNKNOWN",
                      "media_type": media or "UNKNOWN", "bus_type": bus or "UNKNOWN", "interface_type": interface or "UNKNOWN",
                      "health_status": pd.get("health_status") or "UNKNOWN",
                      "operational_status": pd.get("operational_status") or disk.get("operational_status") or "UNKNOWN",
                      "size_bytes": pd.get("size_bytes") or disk.get("size_bytes") or cd.get("size_bytes"),
                      "classification": classification, "partitions": disk_partitions})
    return disks, mapped_volumes


def _windows_inventory() -> tuple[dict[str, Any], list[str]]:
    script = r'''
$ErrorActionPreference='SilentlyContinue'
$pd=@(Get-PhysicalDisk | ForEach-Object {[pscustomobject]@{device_id=[string]$_.DeviceId;friendly_name=$_.FriendlyName;serial_number=$_.SerialNumber;media_type=[string]$_.MediaType;bus_type=[string]$_.BusType;health_status=[string]$_.HealthStatus;operational_status=[string]$_.OperationalStatus;size_bytes=$_.Size}})
$gd=@(Get-Disk | ForEach-Object {[pscustomobject]@{disk_number=$_.Number;friendly_name=$_.FriendlyName;serial_number=$_.SerialNumber;model=$_.Model;media_type=[string]$_.MediaType;bus_type=[string]$_.BusType;operational_status=[string]$_.OperationalStatus;size_bytes=$_.Size}})
$cd=@(Get-CimInstance Win32_DiskDrive | ForEach-Object {[pscustomobject]@{index=$_.Index;model=$_.Model;serial_number=$_.SerialNumber;interface_type=$_.InterfaceType;media_type=$_.MediaType;size_bytes=[uint64]$_.Size}})
$parts=@(Get-Partition | ForEach-Object {[pscustomobject]@{disk_number=$_.DiskNumber;partition_number=$_.PartitionNumber;drive_letter=[string]$_.DriveLetter;size_bytes=$_.Size;type=[string]$_.Type}})
$vol=@(Get-Volume | ForEach-Object {$free=[uint64]$_.SizeRemaining;$size=[uint64]$_.Size;[pscustomobject]@{drive_letter=[string]$_.DriveLetter;path=$_.Path;filesystem=$_.FileSystem;label=$_.FileSystemLabel;size_bytes=$size;free_bytes=$free;percent_free=$(if($size){[math]::Round($free*100/$size,2)}else{$null});drive_type=[string]$_.DriveType;health_status=[string]$_.HealthStatus;operational_status=[string]$_.OperationalStatus}})
[pscustomobject]@{physical_disks=$pd;disks=$gd;cim_disks=$cd;partitions=$parts;volumes=$vol}|ConvertTo-Json -Depth 7 -Compress
'''
    result = run_command(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script], timeout=30)
    errors: list[str] = []
    if result["exit_code"] != 0 or not result["stdout"]:
        errors.append(result["error"] or result["stderr"] or "Windows storage inventory returned no data")
        return {}, errors
    try:
        return json.loads(result["stdout"]), errors
    except json.JSONDecodeError as exc:
        errors.append(f"Unable to parse Windows storage inventory: {exc}")
        return {}, errors


def _is_network_path(path: str) -> bool:
    value = path.strip()
    return value.startswith("\\\\") or value.startswith("//")


def validate_model_storage_root(root: str, inventory: dict[str, Any], perform_write_test: bool = True) -> dict[str, Any]:
    result: dict[str, Any] = {"root": root, "status": "FAIL", "exists": False, "local_filesystem": False,
                              "writable": False, "network_share": _is_network_path(root), "removable": False,
                              "free_bytes": None, "free_gib": None, "classification": "UNKNOWN",
                              "classification_detail": None, "disk_number": None, "errors": []}
    if result["network_share"]:
        result["errors"].append("Network paths are not allowed for MODEL_STORAGE_ROOT")
        return result
    pure = PureWindowsPath(root)
    if not pure.drive or not re.fullmatch(r"[A-Za-z]:", pure.drive):
        result["errors"].append("MODEL_STORAGE_ROOT must be on a detected local drive")
        return result
    letter = pure.drive[0].upper()
    expected = PureWindowsPath(f"{letter}:\\") / PureWindowsPath(MODEL_DIRECTORY_NAME)
    if str(pure).rstrip("\\").lower() != str(expected).rstrip("\\").lower():
        result["errors"].append("MODEL_STORAGE_ROOT must use the project-specific derived directory")
        return result
    volume = next((v for v in inventory.get("volumes", []) if str(v.get("drive_letter", "")).upper() == letter), None)
    if not volume:
        result["errors"].append("Selected drive is not present in the detected local-volume inventory")
        return result
    drive_type = str(volume.get("drive_type") or "UNKNOWN").lower()
    result["removable"] = drive_type in {"removable", "2"}
    if drive_type in {"network", "4"}:
        result["network_share"] = True
        result["errors"].append("Selected volume is a network drive")
        return result
    if result["removable"]:
        result["errors"].append("Removable storage is not supported for MODEL_STORAGE_ROOT")
        return result
    result["local_filesystem"] = drive_type in {"fixed", "3", "unknown"}
    result["free_bytes"] = volume.get("free_bytes")
    result["free_gib"] = round(result["free_bytes"] / 2**30, 2) if isinstance(result["free_bytes"], int) else None
    result["disk_number"] = volume.get("disk_number")
    disk = next((d for d in inventory.get("physical_disks", []) if d.get("disk_number") == result["disk_number"]), None)
    if disk:
        detail = disk.get("classification") or {}
        result.update(classification_detail=detail, classification=detail.get("classification", "UNKNOWN"), physical_device=disk.get("friendly_name"))
    path = Path(root)
    try:
        path.mkdir(parents=True, exist_ok=True)
        result["exists"] = path.is_dir()
        if perform_write_test:
            handle, temp_name = tempfile.mkstemp(prefix=".phase1r-write-test-", suffix=".tmp", dir=path)
            try:
                os.write(handle, b"phase1r")
            finally:
                os.close(handle)
                Path(temp_name).unlink(missing_ok=True)
            result["writable"] = True
        else:
            result["writable"] = os.access(path, os.W_OK)
    except OSError as exc:
        result["errors"].append(f"Write validation failed: {exc}")
    enough_space = isinstance(result["free_gib"], (int, float)) and result["free_gib"] >= MIN_FREE_GIB
    classification = result["classification"]
    if not result["writable"] or not result["local_filesystem"] or not enough_space:
        if not enough_space:
            result["errors"].append(f"At least {MIN_FREE_GIB:.0f} GiB free is required")
        result["status"] = "FAIL"
    elif classification == "NVME":
        result["status"] = "PASS"
    elif classification in {"SATA_SSD", "SSD_UNKNOWN_BUS"}:
        result["status"] = "WARN"
    elif classification == "HDD":
        result["status"] = "FAIL"
        result["errors"].append("HDD is unsuitable for the intended model workload")
    else:
        result["status"] = "WARN"
        result["errors"].append("Storage classification is UNKNOWN and requires clear manual hardware evidence")
    return result


def validate_download_root(root: str, inventory: dict[str, Any], perform_write_test: bool = True) -> dict[str, Any]:
    result: dict[str, Any] = {
        "root": root, "purpose": "model acquisition/archive", "status": "FAIL", "exists": False,
        "local_filesystem": False, "network_share": _is_network_path(root), "removable": False,
        "writable": False, "write_test_read_back": False, "temporary_file_deleted": False,
        "free_bytes": None, "free_gib": None, "storage_class": "UNKNOWN", "physical_device": None,
        "media_type": "UNKNOWN", "disk_number": None, "path_contained_on_d": False, "errors": [],
        "warning": "Download storage is HDD-backed. It is suitable for acquisition/archive capacity but has not been validated as practical Colibrì inference storage.",
    }
    if result["network_share"]:
        result["errors"].append("Network paths are not allowed for DOWNLOAD_ROOT")
        return result
    pure = PureWindowsPath(root)
    expected = PureWindowsPath(str(DOWNLOAD_ROOT))
    if str(pure).rstrip("\\").lower() != str(expected).rstrip("\\").lower():
        result["errors"].append("DOWNLOAD_ROOT must match the fixed project acquisition path on D:")
        return result
    if pure.drive.upper() != "D:":
        result["errors"].append("DOWNLOAD_ROOT must remain under D:")
        return result
    result["path_contained_on_d"] = True
    volume = next((v for v in inventory.get("volumes", []) if str(v.get("drive_letter", "")).upper() == "D"), None)
    if not volume:
        result["errors"].append("D: is not present in the detected local-volume inventory")
        return result
    drive_type = str(volume.get("drive_type") or "UNKNOWN").lower()
    result["removable"] = drive_type in {"removable", "2"}
    if drive_type in {"network", "4"}:
        result["network_share"] = True
        result["errors"].append("D: is a network drive")
        return result
    if result["removable"]:
        result["errors"].append("Removable storage is not supported for DOWNLOAD_ROOT")
        return result
    result["local_filesystem"] = drive_type in {"fixed", "3", "unknown"}
    result["free_bytes"] = volume.get("free_bytes")
    result["free_gib"] = round(result["free_bytes"] / 2**30, 2) if isinstance(result["free_bytes"], int) else None
    result["disk_number"] = volume.get("disk_number")
    disk = next((d for d in inventory.get("physical_disks", []) if d.get("disk_number") == result["disk_number"]), None)
    if disk:
        result["storage_class"] = (disk.get("classification") or {}).get("classification", "UNKNOWN")
        result["physical_device"] = disk.get("friendly_name")
        result["media_type"] = disk.get("media_type", "UNKNOWN")
    path = Path(root)
    temp_path: Path | None = None
    try:
        if perform_write_test:
            path.mkdir(parents=True, exist_ok=True)
        resolved = path.resolve()
        d_root = Path("D:/").resolve()
        if resolved != d_root and d_root not in resolved.parents:
            result["errors"].append("Resolved DOWNLOAD_ROOT escaped D:")
            return result
        result["exists"] = path.is_dir()
        if perform_write_test:
            handle, temp_name = tempfile.mkstemp(prefix=".phase1r4-write-test-", suffix=".tmp", dir=path)
            temp_path = Path(temp_name)
            try:
                payload = b"phase1-r4-download-root"
                os.write(handle, payload)
                os.lseek(handle, 0, os.SEEK_SET)
                result["write_test_read_back"] = os.read(handle, len(payload)) == payload
            finally:
                os.close(handle)
            result["writable"] = result["write_test_read_back"]
        else:
            result["writable"] = result["exists"] and os.access(path, os.W_OK)
            result["write_test_read_back"] = result["writable"]
    except OSError as exc:
        result["errors"].append(f"Download-root validation failed: {exc}")
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
                result["temporary_file_deleted"] = not temp_path.exists()
            except OSError as exc:
                result["errors"].append(f"Temporary validation-file cleanup failed: {exc}")
    enough_space = isinstance(result["free_gib"], (int, float)) and result["free_gib"] >= MIN_FREE_GIB
    if not enough_space:
        result["errors"].append(f"At least {MIN_FREE_GIB:.0f} GiB free is required for DOWNLOAD_ROOT")
    if result["local_filesystem"] and result["path_contained_on_d"] and result["writable"] and enough_space:
        result["status"] = "PASS"
    return result


def derive_model_root(drive_letter: str) -> str:
    letter = drive_letter.strip().rstrip(":\\/").upper()
    if not re.fullmatch(r"[A-Z]", letter):
        raise ValueError("A single detected drive letter is required")
    return f"{letter}:\\{MODEL_DIRECTORY_NAME}"


def probe_storage(project_root: Path = PROJECT_ROOT, selected_model_root: str | None = None,
                  perform_write_test: bool = False, download_root: str | None = None,
                  model_runtime_root: str | None = None) -> dict[str, Any]:
    resolved = project_root.resolve()
    usage = shutil.disk_usage(resolved)
    raw, errors = _windows_inventory() if os.name == "nt" else ({}, ["Windows storage inventory is only available on Windows"])
    disks, volumes = map_storage_records(raw)
    drive = resolved.drive or resolved.anchor
    project_volume = next((v for v in volumes if str(v.get("drive_letter", "")).upper() == drive.rstrip(":").upper()), None)
    inventory: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(), "project_path": str(resolved), "project_drive": drive,
        "project_storage": {"path": str(resolved), "drive_letter": drive.rstrip(":"), "total_bytes": usage.total,
                            "free_bytes": usage.free, "free_gib": round(usage.free / 2**30, 2),
                            "disk_number": project_volume.get("disk_number") if project_volume else None},
        "physical_disks": disks, "volumes": volumes, "probe_errors": errors,
    }
    inventory["model_storage_candidates"] = [
        {**volume,
         "classification": (next((d.get("classification") for d in disks if d.get("disk_number") == volume.get("disk_number")), {}) or {}).get("classification", "UNKNOWN"),
         "classification_detail": next((d.get("classification") for d in disks if d.get("disk_number") == volume.get("disk_number")), {}) or {},
         "eligible_free_space": free_space_status(volume.get("free_bytes")) == "PASS"}
        for volume in volumes if str(volume.get("drive_letter") or "").replace("\x00", "").strip()
    ]
    runtime_root = model_runtime_root if model_runtime_root is not None else selected_model_root
    configured_download = download_root or str(DOWNLOAD_ROOT)
    download_validation = validate_download_root(configured_download, inventory, perform_write_test)
    runtime_validation = validate_model_storage_root(runtime_root, inventory, perform_write_test) if runtime_root else None
    inventory["model_storage_root"] = runtime_root
    inventory["model_storage_validation"] = runtime_validation
    inventory["project_root"] = str(project_root.resolve())
    inventory["download_root"] = configured_download
    inventory["model_runtime_root"] = runtime_root
    inventory["download_root_validation"] = download_validation
    inventory["model_runtime_validation"] = runtime_validation
    inventory["download_root_status"] = download_validation.get("status")
    inventory["runtime_root_status"] = runtime_validation.get("status") if runtime_validation else "NOT CONFIGURED"
    inventory["download_storage_class"] = download_validation.get("storage_class")
    inventory["runtime_storage_class"] = runtime_validation.get("classification") if runtime_validation else None
    inventory["download_free_bytes"] = download_validation.get("free_bytes")
    return inventory
