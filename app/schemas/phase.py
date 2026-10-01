from pydantic import BaseModel, Field


class BenchmarkRequest(BaseModel):
    size_mib: int = Field(default=256)


class ModelStorageRequest(BaseModel):
    drive_letter: str = Field(min_length=1, max_length=3)


class PhaseResponse(BaseModel):
    phase: int
    verdict: str
    checks: list[dict]
    model_download_locked: bool
    message: str
