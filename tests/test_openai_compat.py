from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api import openai_compat
from app.main import app
from app.services.qwen_chat import ChatBusyError, ChatRuntimeError
from app.services.tool_compat import parse_structured_action

client = TestClient(app)


def request_body(**overrides):
    body = {
        "model": "qwen36",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": False,
        "max_tokens": 64,
        "temperature": 0.5,
    }
    body.update(overrides)
    return body


def calculator_tool():
    return {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Perform deterministic arithmetic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {"type": "string", "enum": ["add", "subtract", "multiply", "divide"]},
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["operation", "a", "b"],
                "additionalProperties": False,
            },
        },
    }


def completion(content: str, finish_reason: str = "stop"):
    return {
        "id": "chatcmpl-local",
        "object": "chat.completion",
        "created": 123,
        "model": "qwen36",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def test_models_lists_qwen36():
    response = client.get("/v1/models")
    assert response.status_code == 200
    assert response.json() == {
        "object": "list",
        "data": [{"id": "qwen36", "object": "model", "created": 0, "owned_by": "local"}],
    }


def test_unsupported_model_uses_openai_error(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(openai_compat.chat_runtime, "complete_openai", lambda *_args: pytest.fail("backend called"))
    response = client.post("/v1/chat/completions", json=request_body(model="other"))
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "model_not_found"


def test_stopped_server_returns_503(monkeypatch: pytest.MonkeyPatch):
    def stopped(*_args):
        raise ChatRuntimeError("The Qwen3.6 server is not ready.")

    monkeypatch.setattr(openai_compat.chat_runtime, "complete_openai", stopped)
    response = client.post("/v1/chat/completions", json=request_body())
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "model_not_ready"


def test_non_stream_response_shape(monkeypatch: pytest.MonkeyPatch):
    completion = {
        "id": "chatcmpl-local",
        "object": "chat.completion",
        "created": 123,
        "model": "qwen36",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
    }
    monkeypatch.setattr(openai_compat.chat_runtime, "complete_openai", lambda *_args: completion)
    response = client.post("/v1/chat/completions", json=request_body())
    assert response.status_code == 200
    assert set(response.json()) >= {"id", "object", "created", "model", "choices", "usage"}
    assert response.json()["usage"] == completion["usage"]


def test_streaming_sse_shape(monkeypatch: pytest.MonkeyPatch):
    def stream(*_args):
        yield b'data: {"id":"chatcmpl-local","object":"chat.completion.chunk","choices":[{"delta":{"content":"Hi"}}]}\n\n'
        yield b"data: [DONE]\n\n"

    monkeypatch.setattr(openai_compat.chat_runtime, "stream_openai", stream)
    response = client.post("/v1/chat/completions", json=request_body(stream=True))
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert '"object":"chat.completion.chunk"' in response.text
    assert response.text.endswith("data: [DONE]\n\n")


@pytest.mark.parametrize("max_tokens", [0, 4097])
def test_max_tokens_validation_uses_openai_error(max_tokens: int):
    response = client.post("/v1/chat/completions", json=request_body(max_tokens=max_tokens))
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_max_tokens_4096_is_accepted(monkeypatch: pytest.MonkeyPatch):
    observed = {}

    def complete(_messages, max_tokens, _temperature):
        observed["max_tokens"] = max_tokens
        return completion("Long-answer budget accepted")

    monkeypatch.setattr(openai_compat.chat_runtime, "complete_openai", complete)
    response = client.post("/v1/chat/completions", json=request_body(max_tokens=4096))
    assert response.status_code == 200
    assert observed["max_tokens"] == 4096


def test_busy_rejection_returns_429(monkeypatch: pytest.MonkeyPatch):
    def busy(*_args):
        raise ChatBusyError("BUSY")

    monkeypatch.setattr(openai_compat.chat_runtime, "complete_openai", busy)
    response = client.post("/v1/chat/completions", json=request_body())
    assert response.status_code == 429
    assert response.json()["error"]["code"] == "generation_busy"


def test_function_tool_schema_is_accepted_and_translated(monkeypatch: pytest.MonkeyPatch):
    action = '{"name":"calculator","arguments":{"operation":"multiply","a":37,"b":19}}'
    monkeypatch.setattr(openai_compat.chat_runtime, "complete_openai", lambda *_args: completion(action))
    response = client.post(
        "/v1/chat/completions",
        json=request_body(tools=[calculator_tool()]),
    )
    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] is None
    call = choice["message"]["tool_calls"][0]
    assert call["id"].startswith("call_")
    assert call["type"] == "function"
    assert call["function"]["name"] == "calculator"
    assert call["function"]["arguments"] == '{"a":37,"b":19,"operation":"multiply"}'


def test_tool_choice_none_uses_normal_completion(monkeypatch: pytest.MonkeyPatch):
    observed = {}

    def complete(messages, *_args):
        observed["messages"] = messages
        return completion("Normal answer")

    monkeypatch.setattr(openai_compat.chat_runtime, "complete_openai", complete)
    response = client.post(
        "/v1/chat/completions",
        json=request_body(tools=[calculator_tool()], tool_choice="none"),
    )
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "Normal answer"
    assert all("Available functions" not in message["content"] for message in observed["messages"])


def test_structured_action_parser_accepts_known_function():
    parsed = parse_structured_action(
        '{"name":"calculator","arguments":{"operation":"add","a":2,"b":3}}',
        [calculator_tool()],
    )
    assert parsed == ("calculator", {"operation": "add", "a": 2, "b": 3})


def test_unknown_model_tool_is_rejected(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        openai_compat.chat_runtime,
        "complete_openai",
        lambda *_args: completion('{"name":"shell","arguments":{}}'),
    )
    response = client.post("/v1/chat/completions", json=request_body(tools=[calculator_tool()]))
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "unknown_tool"


def test_role_tool_follow_up_is_translated_for_backend(monkeypatch: pytest.MonkeyPatch):
    observed = {}

    def complete_final(messages, *_args):
        observed["messages"] = messages
        return completion("There are 703 components.")

    monkeypatch.setattr(openai_compat.chat_runtime, "complete_openai", complete_final)
    messages = [
        {"role": "user", "content": "How many components?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_test",
                    "type": "function",
                    "function": {
                        "name": "calculator",
                        "arguments": '{"operation":"multiply","a":37,"b":19}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_test", "content": "703"},
    ]
    response = client.post(
        "/v1/chat/completions",
        json=request_body(messages=messages, tools=[calculator_tool()]),
    )
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "There are 703 components."
    assert all(message["role"] != "tool" for message in observed["messages"])
    assert any("703" in message["content"] for message in observed["messages"])


def test_malformed_tool_action_returns_openai_error(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        openai_compat.chat_runtime,
        "complete_openai",
        lambda *_args: completion('{"name":"calculator","arguments":'),
    )
    response = client.post("/v1/chat/completions", json=request_body(tools=[calculator_tool()]))
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "tool_protocol_error"


def test_tools_with_stream_returns_clear_limitation():
    response = client.post(
        "/v1/chat/completions",
        json=request_body(tools=[calculator_tool()], stream=True),
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "tool_streaming_unsupported"
