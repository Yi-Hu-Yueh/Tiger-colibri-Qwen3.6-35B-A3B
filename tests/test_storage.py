from pathlib import Path

from app.services.storage_probe import (
    classify_device, classify_storage, derive_model_root, free_space_status,
    map_storage_records, validate_download_root, validate_model_storage_root,
)


def test_classification_does_not_assume_all_ssds_are_nvme():
    assert classify_storage("SSD", "SATA", "Samsung SSD") == "SATA SSD"
    assert classify_storage("SSD", "NVMe", "Disk") == "NVMe"
    assert classify_storage("SSD", "USB", "Disk") == "Unknown"
    assert classify_storage("HDD", "SATA", "Disk") == "HDD"


def test_free_space_exact_threshold_passes():
    assert free_space_status(30 * 2**30) == "PASS"
    assert free_space_status(30 * 2**30 - 1) == "FAIL"


def test_phase1r_storage_classifications():
    assert classify_device("SSD", "NVMe", "Disk", "Model", "SCSI")["classification"] == "NVME"
    assert classify_device("SSD", "SATA", "Disk", "Model", "SATA")["classification"] == "SATA_SSD"
    assert classify_device("HDD", "RAID", "Disk", "Model", "SCSI")["classification"] == "HDD"
    raid = classify_device("Unspecified", "RAID", "Generic", "Generic", "SCSI")
    assert raid["classification"] == "UNKNOWN"
    assert any("RAID" in item for item in raid["evidence"])


def test_physical_disk_partition_volume_mapping():
    raw = {
        "physical_disks": [{"device_id": "1", "friendly_name": "NVMe Test", "media_type": "SSD", "bus_type": "RAID", "size_bytes": 100}],
        "disks": [{"disk_number": 1, "friendly_name": "NVMe Test", "size_bytes": 100}],
        "cim_disks": [{"index": 1, "model": "NVMe Test", "interface_type": "SCSI"}],
        "partitions": [{"disk_number": 1, "partition_number": 3, "drive_letter": "C", "size_bytes": 90}],
        "volumes": [{"drive_letter": "C", "filesystem": "NTFS", "size_bytes": 90, "free_bytes": 40,
                     "percent_free": 44.4, "drive_type": "Fixed", "path": "C:\\"}],
    }
    disks, volumes = map_storage_records(raw)
    assert disks[0]["classification"]["classification"] == "NVME"
    assert disks[0]["partitions"][0]["drive_letter"] == "C"
    assert volumes[0]["disk_number"] == 1


def test_network_model_root_rejected():
    result = validate_model_storage_root(r"\\server\models", {"volumes": [], "physical_disks": []})
    assert result["status"] == "FAIL"
    assert result["network_share"] is True


def test_model_root_validation_and_free_space(monkeypatch):
    root = derive_model_root("C")
    inventory = {"volumes": [{"drive_letter": "C", "drive_type": "Fixed", "disk_number": 1,
                              "free_bytes": 40 * 2**30}],
                 "physical_disks": [{"disk_number": 1, "friendly_name": "NVMe Test",
                                      "classification": {"classification": "NVME", "confidence": "high", "evidence": ["test"]}}]}
    monkeypatch.setattr(Path, "mkdir", lambda self, **kwargs: None)
    monkeypatch.setattr(Path, "is_dir", lambda self: True)
    monkeypatch.setattr("app.services.storage_probe.os.access", lambda path, mode: True)
    result = validate_model_storage_root(root, inventory, perform_write_test=False)
    assert result["status"] == "PASS"
    assert result["writable"] is True
    inventory["volumes"][0]["free_bytes"] = 29 * 2**30
    assert validate_model_storage_root(root, inventory, perform_write_test=False)["status"] == "FAIL"


def download_inventory(free_gib=100):
    return {"volumes": [{"drive_letter": "D", "drive_type": "Fixed", "disk_number": 0,
                         "free_bytes": free_gib * 2**30}],
            "physical_disks": [{"disk_number": 0, "friendly_name": "HDD Test", "media_type": "HDD",
                                 "classification": {"classification": "HDD", "confidence": "high", "evidence": ["MediaType=HDD"]}}]}


def test_hdd_allowed_for_download_but_not_runtime(monkeypatch):
    root = r"D:\TigerModels\Tiger-colibri-Qwen3.6-35B-A3B\downloads"
    monkeypatch.setattr(Path, "resolve", lambda self: self)
    monkeypatch.setattr(Path, "is_dir", lambda self: True)
    monkeypatch.setattr("app.services.storage_probe.os.access", lambda path, mode: True)
    download = validate_download_root(root, download_inventory(), perform_write_test=False)
    assert download["status"] == "PASS"
    assert download["storage_class"] == "HDD"
    runtime_root = derive_model_root("D")
    runtime = validate_model_storage_root(runtime_root, download_inventory(), perform_write_test=False)
    assert runtime["status"] == "FAIL"
    assert runtime["classification"] == "HDD"


def test_download_capacity_network_and_containment(monkeypatch):
    root = r"D:\TigerModels\Tiger-colibri-Qwen3.6-35B-A3B\downloads"
    monkeypatch.setattr(Path, "resolve", lambda self: self)
    monkeypatch.setattr(Path, "is_dir", lambda self: True)
    monkeypatch.setattr("app.services.storage_probe.os.access", lambda path, mode: True)
    assert validate_download_root(root, download_inventory(29), False)["status"] == "FAIL"
    assert validate_download_root(r"\\server\models", download_inventory(), False)["network_share"] is True
    outside = validate_download_root(r"D:\Other\downloads", download_inventory(), False)
    assert outside["status"] == "FAIL"
    assert outside["path_contained_on_d"] is False
