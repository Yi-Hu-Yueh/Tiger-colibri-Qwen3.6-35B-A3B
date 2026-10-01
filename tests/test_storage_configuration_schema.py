from app.schemas.storage import StorageConfigurationResponse


def test_storage_configuration_api_serialization():
    payload = {
        "project_root": r"D:\project",
        "download_root": r"D:\TigerModels\Tiger-colibri-Qwen3.6-35B-A3B\downloads",
        "model_runtime_root": None,
        "download_root_status": "PASS",
        "runtime_root_status": "NOT CONFIGURED",
        "download_storage_class": "HDD",
        "runtime_storage_class": None,
        "download_free_bytes": 100 * 2**30,
    }
    encoded = StorageConfigurationResponse.model_validate(payload).model_dump()
    assert encoded["download_root"] != encoded["model_runtime_root"]
    assert encoded["download_storage_class"] == "HDD"
