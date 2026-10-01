from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Iterator
from typing import Literal

import anyio
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from app.services.qwen_chat import ChatBusyError, ChatRuntimeError, chat_runtime

router = APIRouter(prefix="/api/chat", tags=["chat"])
LOGGER = logging.getLogger(__name__)
_STREAM_DONE = object()


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=20000)


class CompletionRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1, max_length=100)
    max_tokens: int = Field(default=4096, ge=1, le=4096)


class CancelRequest(BaseModel):
    generation_id: int | None = Field(default=None, ge=1)


@router.get("/status")
def status() -> dict:
    return chat_runtime.status()


@router.post("/start", response_model=None)
def start():
    try:
        return chat_runtime.start()
    except ChatRuntimeError as error:
        status_code = 409 if error.code in {"COLIBRI_PORT_IN_USE", "INSUFFICIENT_RAM", "MODEL_RUNTIME_ERROR"} else 503
        return JSONResponse(status_code=status_code, content={"error": str(error), "code": error.code})
    except Exception as error:
        LOGGER.exception("Unexpected Qwen3.6 startup failure")
        return JSONResponse(
            status_code=500,
            content={"error": f"Unexpected model startup failure: {error}", "code": "MODEL_START_FAILED"},
        )


@router.post("/stop")
def stop() -> dict:
    return chat_runtime.stop()


@router.post("/cancel")
def cancel(request: CancelRequest | None = None) -> dict:
    try:
        return chat_runtime.cancel_generation(request.generation_id if request is not None else None)
    except ChatRuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


def _next_stream_item(stream: Iterator[bytes]) -> bytes | object:
    try:
        return next(stream)
    except StopIteration:
        return _STREAM_DONE


async def _disconnect_safe_stream(stream: Iterator[bytes], generation_id: int) -> AsyncIterator[bytes]:
    try:
        while True:
            item = await anyio.to_thread.run_sync(
                _next_stream_item,
                stream,
                abandon_on_cancel=True,
            )
            if item is _STREAM_DONE:
                return
            yield item
    finally:
        with anyio.CancelScope(shield=True):
            await anyio.to_thread.run_sync(
                chat_runtime.cancel_generation,
                generation_id,
                abandon_on_cancel=True,
            )


@router.post("/completions")
def completions(request: CompletionRequest) -> StreamingResponse:
    try:
        generation_id, stream = chat_runtime.stream_with_generation(
            [message.model_dump() for message in request.messages],
            request.max_tokens,
        )
    except ChatBusyError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except ChatRuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return StreamingResponse(
        _disconnect_safe_stream(stream, generation_id),
        media_type="application/x-ndjson",
        headers={"X-Generation-ID": str(generation_id), "Cache-Control": "no-store"},
    )
