from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class OllamaStorageStatus(BaseModel):
    model_config = ConfigDict(extra="allow")

    status: str
    source: str
    destination: str
    warning: str | None = None

