from typing import Any

from pydantic import BaseModel, ConfigDict


class HardwareResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    timestamp: str
    cpu: dict[str, Any]
    ram: dict[str, Any]
    windows: dict[str, Any]
    python: dict[str, Any]
    gpu: dict[str, Any]
