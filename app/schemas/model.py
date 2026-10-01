from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ModelAcquisitionStatus(BaseModel):
    model_config = ConfigDict(extra="allow")

    repo_id: str
    revision: str | None = None
    destination: str
    remote_file_count: int | None = None
    remote_expected_bytes: int | None = None
    local_file_count: int = 0
    local_bytes: int = 0
    download_status: str
    integrity_status: str
    warnings: list[str] = Field(default_factory=list)
    verification_errors: list[str] = Field(default_factory=list)
    remote_files: list[dict[str, Any]] = Field(default_factory=list)
