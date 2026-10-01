from typing import Any

from pydantic import BaseModel, ConfigDict


class ColibriResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    status: str

