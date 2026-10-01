from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.api import chat as chat_api
from app.main import app
from app.services import qwen_chat

client = TestClient(app)


class FakeSocket:
    def __init__(self, closed: threading.Event) -> None:
        self.closed = closed
        self.shutdown_called = False
        self.close_called = False

    def shutdown(self, _how: int) -> None:
        self.shutdown_called = True
        self.closed.set()

    def close(self) -> None:
        self.close_called = True
        self.closed.set()


class BlockingResponse:
    status = 200

    def __init__(self, closed: threading.Event) -> None:
        self.closed = closed
        self.read_started = threading.Event()
        self.close_called = False
        self.raw_socket = FakeSocket(closed)
        self.fp = SimpleNamespace(raw=SimpleNamespace(_sock=self.raw_socket))

    def readline(self) -> bytes:
        self.read_started.set()
        self.closed.wait(2)
        raise OSError("stream closed")

    def close(self) -> None:
        self.close_called = True
        self.closed.set()


class SequenceResponse:
    status = 200

    def __init__(self) -> None:
        self.lines = iter(
            [
                b'data: {"choices":[{"delta":{"content":"OK"}}]}\n',
                b'data: {"usage":{"completion_tokens":1},"choices":[]}\n',
                b"data: [DONE]\n",
            ]
        )
        self.close_called = False

    def readline(self) -> bytes:
        return next(self.lines, b"")

    def close(self) -> None:
        self.close_called = True


class FakeConnection:
    def __init__(self, response: BlockingResponse | SequenceResponse) -> None:
        self.response = response
        self.closed = response.closed if isinstance(response, BlockingResponse) else threading.Event()
        self.sock = FakeSocket(self.closed)
        self.close_called = False
        self.requests = []

    def request(self, *args, **kwargs) -> None:
        self.requests.append((args, kwargs))
        return None

    def getresponse(self):
        return self.response

    def close(self) -> None:
        self.close_called = True
        self.closed.set()


class FakeProcess:
    def __init__(self, pid: int = 7001) -> None:
        self.pid = pid
        self.alive = True
        self.return_code = 0

    def poll(self):
        return None if self.alive else self.return_code

    def wait(self, timeout=None):
        if self.alive:
            raise qwen_chat.subprocess.TimeoutExpired("fake", timeout)
        return self.return_code

    def terminate(self):
        self.alive = False

    def kill(self):
        self.alive = False


class ImmediateThread:
    def __init__(self, *, target, args=(), **_kwargs) -> None:
        self.target = target
        self.args = args

    def start(self) -> None:
        self.target(*self.args)


def configure_successful_start(runtime: qwen_chat.QwenChatRuntime, monkeypatch: pytest.MonkeyPatch) -> FakeProcess:
    process = FakeProcess()

    def launch():
        runtime._process = process
        return process

    monkeypatch.setattr(runtime, "_launch_managed_process", launch)
    monkeypatch.setattr(runtime, "_ready_probe", lambda: True)
    monkeypatch.setattr(runtime, "_refresh_managed_processes", lambda: None)
    monkeypatch.setattr(qwen_chat.threading, "Thread", ImmediateThread)
    return process


@pytest.fixture
def runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> qwen_chat.QwenChatRuntime:
    launcher = tmp_path / "coli.cmd"
    model = tmp_path / "model"
    launcher.write_text("@echo off\n", encoding="utf-8")
    model.mkdir()
    monkeypatch.setattr(qwen_chat, "available_physical_memory", lambda: 8 * 1024**3)
    monkeypatch.setattr(qwen_chat, "_port_open", lambda *_args, **_kwargs: False)
    return qwen_chat.QwenChatRuntime(coli_cmd=launcher, runtime_dir=tmp_path, model_root=model)


def test_fixed_cpu_only_command(runtime: qwen_chat.QwenChatRuntime):
    command = runtime.build_start_command()
    assert command[3].endswith("coli.cmd")
    assert command[4] == "serve"
    assert command[command.index("--gpu") + 1] == "none"
    assert command[command.index("--cap") + 1] == "16"
    assert command[command.index("--ngen") + 1] == "4096"
    assert "--no-think" in command
    assert runtime._child_environment()["OMP_NUM_THREADS"] == "4"
    assert all(name not in runtime._child_environment() for name in qwen_chat.CUDA_ENVIRONMENT_VARIABLES)


def test_duplicate_model_start_is_rejected(runtime: qwen_chat.QwenChatRuntime, monkeypatch: pytest.MonkeyPatch):
    process = FakeProcess()
    monkeypatch.setattr(runtime, "_launch_managed_process", lambda: process)
    monkeypatch.setattr(qwen_chat.threading.Thread, "start", lambda _self: None)
    assert runtime.start()["status"] == "STARTING"
    with pytest.raises(qwen_chat.ChatRuntimeError, match="already"):
        runtime.start()


def test_low_ram_start_is_rejected(runtime: qwen_chat.QwenChatRuntime, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(qwen_chat, "available_physical_memory", lambda: 4 * 1024**3 - 1)
    with pytest.raises(qwen_chat.ChatRuntimeError, match="4 GiB"):
        runtime.start()
    assert runtime.status()["status"] == "STOPPED"


def test_start_from_stopped_reaches_ready(runtime: qwen_chat.QwenChatRuntime, monkeypatch: pytest.MonkeyPatch):
    configure_successful_start(runtime, monkeypatch)
    result = runtime.start()
    assert result["status"] == "READY"
    assert runtime._process is not None


def test_start_from_error_cleans_stale_state_then_reaches_ready(
    runtime: qwen_chat.QwenChatRuntime,
    monkeypatch: pytest.MonkeyPatch,
):
    runtime._state = "ERROR"
    runtime._error = "previous failure"
    cleanup = []
    monkeypatch.setattr(runtime, "_reset_stale_generation", lambda: cleanup.append("generation"))

    def terminate():
        cleanup.append("processes")
        runtime._process = None
        runtime._managed_processes.clear()

    monkeypatch.setattr(runtime, "_terminate_managed_processes", terminate)
    configure_successful_start(runtime, monkeypatch)
    result = runtime.start()
    assert cleanup == ["generation", "processes"]
    assert result["status"] == "READY"
    assert result["error"] is None


def test_startup_exception_returns_controlled_http_response(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        chat_api.chat_runtime,
        "start",
        lambda: (_ for _ in ()).throw(qwen_chat.ChatRuntimeError("spawn failed", code="MODEL_START_FAILED")),
    )
    response = client.post("/api/chat/start")
    assert response.status_code == 503
    assert response.json() == {"error": "spawn failed", "code": "MODEL_START_FAILED"}


def test_unexpected_startup_exception_returns_controlled_500(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(chat_api.chat_runtime, "start", lambda: (_ for _ in ()).throw(OSError("unexpected spawn error")))
    response = client.post("/api/chat/start")
    assert response.status_code == 500
    assert response.json() == {
        "error": "Unexpected model startup failure: unexpected spawn error",
        "code": "MODEL_START_FAILED",
    }


def test_stale_managed_process_cleanup_runs_before_restart(runtime: qwen_chat.QwenChatRuntime, monkeypatch: pytest.MonkeyPatch):
    stale = FakeProcess(7002)
    runtime._process = stale
    runtime._managed_processes[stale.pid] = qwen_chat._ManagedProcess(stale.pid, 10.0, "launcher")
    cleaned = []

    def terminate():
        cleaned.extend(runtime._managed_processes)
        runtime._managed_processes.clear()
        runtime._process = None

    monkeypatch.setattr(runtime, "_terminate_managed_processes", terminate)
    runtime._recover_for_start()
    assert cleaned == [7002]
    assert runtime._process is None


def test_unrelated_port_owner_is_never_terminated(runtime: qwen_chat.QwenChatRuntime, monkeypatch: pytest.MonkeyPatch):
    cleanup_calls = []
    monkeypatch.setattr(runtime, "_terminate_managed_processes", lambda: cleanup_calls.append("managed-only"))
    monkeypatch.setattr(runtime, "_port_owner_pid", lambda: 99001)
    monkeypatch.setattr(qwen_chat, "_port_open", lambda *_args, **_kwargs: True)
    with pytest.raises(qwen_chat.ChatRuntimeError) as captured:
        runtime._recover_for_start()
    assert captured.value.code == "COLIBRI_PORT_IN_USE"
    assert "PID 99001" in str(captured.value)
    assert cleanup_calls == ["managed-only"]


def test_fastapi_process_cannot_be_selected_for_cleanup(runtime: qwen_chat.QwenChatRuntime):
    current_pid = os.getpid()
    runtime._managed_processes[current_pid] = qwen_chat._ManagedProcess(
        current_pid,
        qwen_chat.psutil.Process(current_pid).create_time(),
        "incorrect",
    )
    runtime._terminate_managed_processes()
    assert qwen_chat.psutil.pid_exists(current_pid)
    assert runtime._managed_processes == {}


def test_failed_start_leaves_no_process_reference_and_second_start_succeeds(
    runtime: qwen_chat.QwenChatRuntime,
    monkeypatch: pytest.MonkeyPatch,
):
    attempts = 0
    process = FakeProcess(7003)

    def launch():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("controlled failure")
        runtime._process = process
        return process

    def cleanup():
        runtime._process = None
        runtime._managed_processes.clear()

    monkeypatch.setattr(runtime, "_launch_managed_process", launch)
    monkeypatch.setattr(runtime, "_terminate_managed_processes", cleanup)
    monkeypatch.setattr(runtime, "_ready_probe", lambda: True)
    monkeypatch.setattr(runtime, "_refresh_managed_processes", lambda: None)
    monkeypatch.setattr(qwen_chat.threading, "Thread", ImmediateThread)

    with pytest.raises(qwen_chat.ChatRuntimeError, match="controlled failure"):
        runtime.start()
    assert runtime.status()["status"] == "ERROR"
    assert runtime._process is None
    assert runtime._managed_processes == {}

    assert runtime.start()["status"] == "READY"
    assert attempts == 2


def test_single_generation_busy_behavior(runtime: qwen_chat.QwenChatRuntime):
    runtime._state = "READY"
    runtime._process = SimpleNamespace(pid=7, poll=lambda: None)
    assert runtime._generation_lock.acquire(blocking=False)
    with pytest.raises(qwen_chat.ChatBusyError, match="BUSY"):
        runtime.stream([{"role": "user", "content": "hello"}], 64)
    runtime._generation_lock.release()


def make_ready(runtime: qwen_chat.QwenChatRuntime) -> None:
    runtime._state = "READY"
    runtime._process = SimpleNamespace(pid=7, poll=lambda: None)


def test_normal_completion_returns_to_ready(
    runtime: qwen_chat.QwenChatRuntime,
    monkeypatch: pytest.MonkeyPatch,
):
    make_ready(runtime)
    connection = FakeConnection(SequenceResponse())
    monkeypatch.setattr(qwen_chat.http.client, "HTTPConnection", lambda *_args, **_kwargs: connection)

    events = [json.loads(event) for event in runtime.stream([{"role": "user", "content": "hello"}], 64)]

    assert events[-1]["type"] == "done"
    assert runtime.status()["status"] == "READY"
    assert runtime._active_generation is None
    assert connection.close_called


def test_browser_history_is_bounded_before_qwen(runtime: qwen_chat.QwenChatRuntime, monkeypatch: pytest.MonkeyPatch):
    make_ready(runtime)
    connection = FakeConnection(SequenceResponse())
    monkeypatch.setattr(qwen_chat.http.client, "HTTPConnection", lambda *_args, **_kwargs: connection)
    messages = []
    for index in range(5):
        messages.extend(
            [
                {"role": "user", "content": f"user-{index}-" + ("u" * 700)},
                {"role": "assistant", "content": f"assistant-{index}-" + ("a" * 700)},
            ]
        )
    current = "Reply with exactly: HISTORY OK"
    messages.append({"role": "user", "content": current})

    events = [json.loads(event) for event in runtime.stream(messages, 64)]
    body = json.loads(connection.requests[0][1]["body"].decode("utf-8"))
    sent = body["messages"]
    historical = sent[1:-1]

    assert events[-1]["type"] == "done"
    assert sent[0] == {"role": "system", "content": qwen_chat.SYSTEM_PROMPT}
    assert sent[-1] == {"role": "user", "content": current}
    assert len(historical) == 4
    assert sum(len(message["content"]) for message in historical) <= 3500
    assert runtime.status()["history_pairs_sent"] == 2
    assert runtime.status()["history_characters_sent"] <= 3500


def test_browser_history_keeps_at_most_three_complete_pairs():
    messages = []
    for index in range(5):
        messages.extend(
            [
                {"role": "user", "content": f"question {index}"},
                {"role": "assistant", "content": f"answer {index}"},
            ]
        )
    messages.append({"role": "user", "content": "current"})

    bounded, stats = qwen_chat.bound_browser_messages(messages)

    assert bounded == messages[-7:]
    assert stats == {
        "history_pairs": 3,
        "history_characters": sum(len(message["content"]) for message in messages[-7:-1]),
    }


def test_cancellation_closes_upstream_clears_state_and_allows_second_request(
    runtime: qwen_chat.QwenChatRuntime,
    monkeypatch: pytest.MonkeyPatch,
):
    make_ready(runtime)
    blocking_response = BlockingResponse(threading.Event())
    blocking_connection = FakeConnection(blocking_response)
    blocking_connection.sock = None
    next_connection = FakeConnection(SequenceResponse())
    connections = iter([blocking_connection, next_connection])
    monkeypatch.setattr(qwen_chat.http.client, "HTTPConnection", lambda *_args, **_kwargs: next(connections))

    stream = runtime.stream([{"role": "user", "content": "long answer"}], 4096)
    session = runtime._active_generation
    worker = threading.Thread(target=lambda: list(stream))
    worker.start()
    assert blocking_response.read_started.wait(1)

    result = runtime.cancel_generation(wait_seconds=2)
    worker.join(timeout=2)

    assert result["cancelled"] is True
    assert result["status"] == "READY"
    assert not worker.is_alive()
    assert runtime._active_generation is None
    assert session is not None and not session.cancel_event.is_set()
    assert blocking_response.raw_socket.shutdown_called
    assert blocking_response.raw_socket.close_called
    assert blocking_connection.close_called
    assert blocking_response.close_called

    second_events = list(runtime.stream([{"role": "user", "content": "second"}], 64))
    assert any(json.loads(event)["type"] == "done" for event in second_events)
    assert runtime.status()["status"] == "READY"


@pytest.mark.parametrize(
    "message",
    [
        "local question",
        "weather question\n\nCurrent weather evidence",
        "web question\n\nTemporary web-search evidence\n[1] official source",
    ],
)
def test_cancellation_closes_stream_suspended_after_delta_for_all_qwen_routes(
    runtime: qwen_chat.QwenChatRuntime,
    monkeypatch: pytest.MonkeyPatch,
    message: str,
):
    make_ready(runtime)
    first_connection = FakeConnection(SequenceResponse())
    second_connection = FakeConnection(SequenceResponse())
    connections = iter([first_connection, second_connection])
    monkeypatch.setattr(qwen_chat.http.client, "HTTPConnection", lambda *_args, **_kwargs: next(connections))

    generation_id, stream = runtime.stream_with_generation([{"role": "user", "content": message}], 4096)
    first_event = json.loads(next(stream))

    assert first_event == {"type": "delta", "content": "OK"}
    assert runtime.status()["active_generation_id"] == generation_id
    result = runtime.cancel_generation(generation_id, wait_seconds=0.5)

    assert result["cancelled"] is True
    assert result["status"] == "READY"
    assert result["active_generation_id"] is None
    assert first_connection.close_called
    assert first_connection.response.close_called
    assert runtime._active_generation is None

    second_events = list(runtime.stream([{"role": "user", "content": "second"}], 64))
    assert any(json.loads(event)["type"] == "done" for event in second_events)


def test_stale_generation_cleanup_cannot_overwrite_new_generation(runtime: qwen_chat.QwenChatRuntime):
    make_ready(runtime)
    old_session = runtime._begin_generation()
    runtime._finish_generation(old_session)
    new_session = runtime._begin_generation()

    runtime._finish_generation(old_session)

    assert runtime._active_generation is new_session
    assert runtime.status()["status"] == "GENERATING"
    runtime._finish_generation(new_session)
    assert runtime.status()["status"] == "READY"


def test_web_search_context_uses_same_cancellation_path(
    runtime: qwen_chat.QwenChatRuntime,
    monkeypatch: pytest.MonkeyPatch,
):
    make_ready(runtime)
    response = BlockingResponse(threading.Event())
    connection = FakeConnection(response)
    monkeypatch.setattr(qwen_chat.http.client, "HTTPConnection", lambda *_args, **_kwargs: connection)
    search_message = "question\n\nTemporary web-search evidence\n[1] official source"
    stream = runtime.stream([{"role": "user", "content": search_message}], 4096)
    worker = threading.Thread(target=lambda: list(stream))
    worker.start()
    assert response.read_started.wait(1)

    result = runtime.cancel_generation(wait_seconds=2)
    worker.join(timeout=2)

    assert result["status"] == "READY"
    assert runtime._active_generation is None
    assert connection.close_called


@pytest.mark.parametrize("max_tokens", [0, 4097, True])
def test_generation_token_range_is_enforced(runtime: qwen_chat.QwenChatRuntime, max_tokens: int):
    with pytest.raises(qwen_chat.ChatRuntimeError, match="between 1 and 4096"):
        runtime.stream([{"role": "user", "content": "hello"}], max_tokens)


def test_clean_model_stop(runtime: qwen_chat.QwenChatRuntime, monkeypatch: pytest.MonkeyPatch):
    fake = SimpleNamespace(pid=11, poll=lambda: None)
    runtime._process = fake
    runtime._state = "READY"
    stopped = []
    monkeypatch.setattr(runtime, "_terminate_managed_processes", lambda: (stopped.append(fake), setattr(runtime, "_process", None)))
    result = runtime.stop()
    assert stopped == [fake]
    assert result["status"] == "STOPPED"
    assert result["pid"] is None


def test_stream_event_is_ndjson():
    event = qwen_chat.QwenChatRuntime._event("delta", content="Hi")
    assert event.endswith(b"\n")
    assert json.loads(event) == {"type": "delta", "content": "Hi"}


def test_cancel_api_forwards_generation_id(monkeypatch: pytest.MonkeyPatch):
    observed = []

    def cancel_generation(generation_id=None):
        observed.append(generation_id)
        return {"status": "READY", "cancelled": True, "generation_id": generation_id}

    monkeypatch.setattr(chat_api.chat_runtime, "cancel_generation", cancel_generation)

    response = client.post("/api/chat/cancel", json={"generation_id": 73})

    assert response.status_code == 200
    assert response.json()["generation_id"] == 73
    assert observed == [73]
