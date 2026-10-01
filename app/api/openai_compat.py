from __future__ import annotations

from typing import Any, Literal, Self

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, model_validator

from app.config import QWEN36_MODEL_ID
from app.services.qwen_chat import ChatBusyError, ChatRuntimeError, chat_runtime
from app.services.tool_compat import (
    ToolCompatibilityError,
    prepare_tool_request,
    translate_tool_completion,
)

router = APIRouter(prefix="/v1", tags=["openai-compatible"])


class OpenAIFunctionCall(BaseModel):
    name: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
    arguments: str = Field(min_length=2, max_length=20000)


class OpenAIToolCall(BaseModel):
    id: str = Field(min_length=1, max_length=200)
    type: Literal["function"]
    function: OpenAIFunctionCall


class OpenAIMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = Field(default=None, max_length=20000)
    tool_call_id: str | None = Field(default=None, min_length=1, max_length=200)
    tool_calls: list[OpenAIToolCall] | None = Field(default=None, min_length=1, max_length=1)

    @model_validator(mode="after")
    def validate_role_fields(self) -> Self:
        if self.role == "tool":
            if self.tool_call_id is None or not self.content:
                raise ValueError("tool messages require tool_call_id and non-empty content")
            if self.tool_calls is not None:
                raise ValueError("tool messages cannot contain tool_calls")
            return self
        if self.tool_call_id is not None:
            raise ValueError("tool_call_id is valid only for role='tool'")
        if self.role == "assistant" and self.tool_calls is not None:
            return self
        if not self.content:
            raise ValueError(f"{self.role} messages require non-empty content")
        if self.tool_calls is not None:
            raise ValueError("tool_calls are valid only for role='assistant'")
        return self


class OpenAIFunctionDefinition(BaseModel):
    name: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
    description: str | None = Field(default=None, max_length=2000)
    parameters: dict[str, Any]


class OpenAITool(BaseModel):
    type: Literal["function"]
    function: OpenAIFunctionDefinition


class OpenAIExplicitFunction(BaseModel):
    name: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")


class OpenAIExplicitToolChoice(BaseModel):
    type: Literal["function"]
    function: OpenAIExplicitFunction


class OpenAIChatCompletionRequest(BaseModel):
    model: str
    messages: list[OpenAIMessage] = Field(min_length=1, max_length=100)
    stream: bool = False
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    tools: list[OpenAITool] | None = Field(default=None, min_length=1, max_length=64)
    tool_choice: Literal["auto", "none"] | OpenAIExplicitToolChoice = "auto"


def openai_error(status_code: int, message: str, error_type: str, code: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": error_type, "code": code}},
    )


@router.get("/models")
def models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": QWEN36_MODEL_ID,
                "object": "model",
                "created": 0,
                "owned_by": "local",
            }
        ],
    }


@router.post("/chat/completions")
def chat_completions(request: OpenAIChatCompletionRequest):
    if request.model != QWEN36_MODEL_ID:
        return openai_error(
            400,
            f"Unsupported model '{request.model}'. The only available model is '{QWEN36_MODEL_ID}'.",
            "invalid_request_error",
            "model_not_found",
        )

    messages = [message.model_dump(exclude_none=True) for message in request.messages]
    try:
        if request.tools:
            if request.stream:
                return openai_error(
                    400,
                    "Streaming tool calls are not supported. Use stream=false when tools are supplied.",
                    "invalid_request_error",
                    "tool_streaming_unsupported",
                )
            tools = [tool.model_dump(exclude_none=True) for tool in request.tools]
            choice = (
                request.tool_choice
                if isinstance(request.tool_choice, str)
                else request.tool_choice.model_dump(exclude_none=True)
            )
            prepared = prepare_tool_request(messages, tools, choice)
            completion = chat_runtime.complete_openai(
                prepared.messages,
                request.max_tokens,
                request.temperature,
            )
            if prepared.expect_tool_action:
                completion = translate_tool_completion(
                    completion,
                    tools,
                    selected_tool=prepared.selected_tool,
                )
            return completion
        if any(message.role == "tool" or message.tool_calls for message in request.messages):
            return openai_error(
                400,
                "Tool-result messages require the corresponding tools definition.",
                "invalid_request_error",
                "tools_required",
            )
        if request.stream:
            stream = chat_runtime.stream_openai(messages, request.max_tokens, request.temperature)
            return StreamingResponse(
                stream,
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        return chat_runtime.complete_openai(messages, request.max_tokens, request.temperature)
    except ToolCompatibilityError as error:
        return openai_error(
            400 if error.client_error else 502,
            str(error),
            "invalid_request_error" if error.client_error else "backend_error",
            error.code,
        )
    except ChatBusyError:
        return openai_error(
            429,
            "Qwen3.6 is busy with another generation. Try again after it finishes.",
            "rate_limit_error",
            "generation_busy",
        )
    except ChatRuntimeError as error:
        if "not ready" in str(error).lower():
            return openai_error(
                503,
                "Qwen3.6 is not running. Start it with the existing Start Model control or POST /api/chat/start.",
                "server_error",
                "model_not_ready",
            )
        return openai_error(502, str(error), "backend_error", "backend_failure")
