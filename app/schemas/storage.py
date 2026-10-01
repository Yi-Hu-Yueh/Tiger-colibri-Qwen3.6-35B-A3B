from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SpaceAnalysisResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    status: str
    timestamp: str | None = None
    drive: dict[str, Any] | None = None
    largest_files: list[dict[str, Any]] = Field(default_factory=list)
    largest_directories: list[dict[str, Any]] = Field(default_factory=list)
    advisory: str


class StorageConfigurationResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    project_root: str
    download_root: str
    model_runtime_root: str | None
    download_root_status: str
    runtime_root_status: str
    download_storage_class: str | None
    runtime_storage_class: str | None
    download_free_bytes: int | None
