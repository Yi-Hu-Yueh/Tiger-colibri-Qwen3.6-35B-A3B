from __future__ import annotations

import ctypes
import http.client
import json
import logging
import os
import socket
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import psutil

from app.config import (
    COLIBRI_CMD,
    COLIBRI_RUNTIME_DIR,
    DATA_DIR,
    QWEN36_CAP,
    QWEN36_HOST,
    QWEN36_MAX_TOKENS,
    QWEN36_MIN_AVAILABLE_RAM_BYTES,
    QWEN36_MODEL_ID,
    QWEN36_OMP_THREADS,
    QWEN36_PORT,
    QWEN36_RUNTIME_ROOT,
)

SYSTEM_PROMPT = "You are a helpful local AI assistant."
MAX_BROWSER_HISTORY_PAIRS = 3
MAX_BROWSER_HISTORY_CHARACTERS = 3500
LOGGER = logging.getLogger(__name__)
CUDA_ENVIRONMENT_VARIABLES = (
    "COLI_CUDA",
    "CUDA_DENSE",
    "CUDA_EXPERT_GB",
    "COLI_GPU",
    "COLI_GPUS",
    "COLI_CUDA_PIPE",
)


class ChatRuntimeError(RuntimeError):
    def __init__(self, message: str, *, code: str = "MODEL_RUNTIME_ERROR") -> None:
        super().__init__(message)
        self.code = code


class ChatBusyError(ChatRuntimeError):
    pass


@dataclass
class _GenerationSession:
    generation_id: int
    cancel_event: threading.Event = field(default_factory=threading.Event)
    done_event: threading.Event = field(default_factory=threading.Event)
    connection: http.client.HTTPConnection | None = None
    response: http.client.HTTPResponse | None = None
    stream: Iterator[bytes] | None = None
    require_backend_cancel_ack: bool = False


@dataclass(frozen=True)
class _ManagedProcess:
    pid: int
    create_time: float
    role: str


class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def available_physical_memory() -> int:
    status = _MemoryStatusEx()
    status.dwLength = ctypes.sizeof(status)
    if os.name != "nt" or not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise ChatRuntimeError("Available physical RAM could not be measured.")
    return int(status.ullAvailPhys)


def _port_open(host: str, port: int, timeout: float = 0.25) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def bound_browser_messages(messages: list[dict[str, str]]) -> tuple[list[dict[str, str]], dict[str, int]]:
    if not messages or messages[-1].get("role") != "user":
        raise ChatRuntimeError("The current browser chat message must be the final user message.")
    current = dict(messages[-1])
    pairs: list[tuple[dict[str, str], dict[str, str]]] = []
    pending_user: dict[str, str] | None = None
    for message in messages[:-1]:
        role = message.get("role")
        if role == "user":
            pending_user = dict(message)
        elif role == "assistant" and pending_user is not None:
            pairs.append((pending_user, dict(message)))
            pending_user = None
        else:
            pending_user = None
    pairs = pairs[-MAX_BROWSER_HISTORY_PAIRS:]

    def pair_characters(pair: tuple[dict[str, str], dict[str, str]]) -> int:
        return len(str(pair[0].get("content", ""))) + len(str(pair[1].get("content", "")))

    history_characters = sum(pair_characters(pair) for pair in pairs)
    while pairs and history_characters > MAX_BROWSER_HISTORY_CHARACTERS:
        history_characters -= pair_characters(pairs.pop(0))
    bounded = [message for pair in pairs for message in pair]
    bounded.append(current)
    return bounded, {"history_pairs": len(pairs), "history_characters": history_characters}


class QwenChatRuntime:
    def __init__(
        self,
        *,
        coli_cmd: Path = COLIBRI_CMD,
        runtime_dir: Path = COLIBRI_RUNTIME_DIR,
        model_root: Path = QWEN36_RUNTIME_ROOT,
        host: str = QWEN36_HOST,
        port: int = QWEN36_PORT,
    ) -> None:
        self.coli_cmd = Path(coli_cmd)
        self.runtime_dir = Path(runtime_dir)
        self.model_root = Path(model_root)
        self.host = host
        self.port = port
        self._lock = threading.RLock()
        self._generation_lock = threading.Lock()
        self._process: subprocess.Popen[str] | None = None
        self._state = "STOPPED"
        self._error: str | None = None
        self._error_code: str | None = None
        self._started_at: float | None = None
        self._startup_seconds: float | None = None
        self._available_ram_before_start: int | None = None
        self._stop_requested = threading.Event()
        self._stdout_handle: Any = None
        self._stderr_handle: Any = None
        self._next_generation_id = 0
        self._active_generation: _GenerationSession | None = None
        self._managed_processes: dict[int, _ManagedProcess] = {}
        self._last_browser_prompt_stats = {"history_pairs": 0, "history_characters": 0}

    def build_start_command(self) -> list[str]:
        return [
            "cmd.exe",
            "/d",
            "/c",
            str(self.coli_cmd),
            "serve",
            "--model",
            str(self.model_root),
            "--gpu",
            "none",
            "--cap",
            str(QWEN36_CAP),
            "--ngen",
            str(QWEN36_MAX_TOKENS),
            "--no-think",
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--model-id",
            QWEN36_MODEL_ID,
        ]

    def _child_environment(self) -> dict[str, str]:
        env = os.environ.copy()
        for name in CUDA_ENVIRONMENT_VARIABLES:
            env.pop(name, None)
        env["OMP_NUM_THREADS"] = str(QWEN36_OMP_THREADS)
        env["PY_PYTHON"] = "3.11"
        env["PYTHONUNBUFFERED"] = "1"
        return env

    def status(self) -> dict[str, Any]:
        with self._lock:
            if self._process is not None and self._state not in {"STOPPED", "ERROR"}:
                return_code = self._process.poll()
                if return_code is not None:
                    self._state = "ERROR"
                    self._error = f"Colibrì exited unexpectedly with code {return_code}."
                    self._error_code = "MODEL_PROCESS_EXITED"
            return {
                "status": self._state,
                "error": self._error,
                "error_code": self._error_code,
                "pid": self._process.pid if self._process is not None else None,
                "history_pairs_sent": self._last_browser_prompt_stats["history_pairs"],
                "history_characters_sent": self._last_browser_prompt_stats["history_characters"],
                "active_generation_id": (
                    self._active_generation.generation_id if self._active_generation is not None else None
                ),
                "managed_pids": sorted(self._managed_processes),
                "port_owner_pid": self._port_owner_pid(),
                "host": self.host,
                "port": self.port,
                "model": "Qwen3.6-35B-A3B",
                "mode": "CPU",
                "startup_seconds": self._startup_seconds,
                "available_ram_before_start_bytes": self._available_ram_before_start,
            }

    def start(self) -> dict[str, Any]:
        with self._lock:
            if self._state in {"STARTING", "READY", "GENERATING"}:
                raise ChatRuntimeError("The Qwen3.6 server is already running or starting.")
        self._recover_for_start()
        available = available_physical_memory()
        if available < QWEN36_MIN_AVAILABLE_RAM_BYTES:
            raise ChatRuntimeError(
                f"At least 4 GiB available system RAM is required; only {available / 1024**3:.2f} GiB is available.",
                code="INSUFFICIENT_RAM",
            )
        if not self.coli_cmd.is_file():
            raise ChatRuntimeError(f"Colibrì launcher not found: {self.coli_cmd}", code="MODEL_START_FAILED")
        if not self.model_root.is_dir():
            raise ChatRuntimeError(f"Qwen3.6 model root not found: {self.model_root}", code="MODEL_START_FAILED")
        with self._lock:
            self._state = "STARTING"
            self._error = None
            self._error_code = None
            self._started_at = time.monotonic()
            self._startup_seconds = None
            self._available_ram_before_start = available
            self._stop_requested.clear()
        try:
            process = self._launch_managed_process()
            worker = threading.Thread(
                target=self._startup_worker,
                args=(process,),
                name="qwen36-startup",
                daemon=True,
            )
            worker.start()
        except Exception as error:
            self._record_startup_failure(error)
            if isinstance(error, ChatRuntimeError):
                raise
            raise ChatRuntimeError(f"Model startup failed: {error}", code="MODEL_START_FAILED") from error
        return self.status()

    def _launch_managed_process(self) -> subprocess.Popen[str]:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._stdout_handle = (DATA_DIR / "qwen36-chat.out.log").open("w", encoding="utf-8")
        self._stderr_handle = (DATA_DIR / "qwen36-chat.err.log").open("w", encoding="utf-8")
        creationflags = 0
        if os.name == "nt":
            creationflags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
        process = subprocess.Popen(
            self.build_start_command(),
            cwd=self.runtime_dir,
            env=self._child_environment(),
            stdin=subprocess.DEVNULL,
            stdout=self._stdout_handle,
            stderr=self._stderr_handle,
            text=True,
            creationflags=creationflags,
        )
        with self._lock:
            self._process = process
        self._register_managed_pid(process.pid, "launcher")
        return process

    def _startup_worker(self, process: subprocess.Popen[str]) -> None:
        try:
            deadline = time.monotonic() + 180.0
            while time.monotonic() < deadline:
                if self._stop_requested.is_set():
                    return
                return_code = process.poll()
                if return_code is not None:
                    raise ChatRuntimeError(
                        f"Colibrì exited while loading with code {return_code}.",
                        code="MODEL_START_FAILED",
                    )
                self._refresh_managed_processes()
                if self._ready_probe():
                    self._refresh_managed_processes()
                    with self._lock:
                        self._startup_seconds = round(time.monotonic() - (self._started_at or time.monotonic()), 3)
                        self._state = "READY"
                        self._error_code = None
                    return
                time.sleep(0.5)
            raise ChatRuntimeError("Qwen3.6 did not become ready within 180 seconds.", code="MODEL_START_TIMEOUT")
        except Exception as error:
            if not self._stop_requested.is_set():
                LOGGER.exception("Qwen3.6 startup failed")
                self._record_startup_failure(error)

    def _ready_probe(self) -> bool:
        try:
            with urllib.request.urlopen(f"http://{self.host}:{self.port}/health", timeout=1.0) as response:
                if response.status != 200:
                    return False
            with urllib.request.urlopen(f"http://{self.host}:{self.port}/v1/models", timeout=1.0) as response:
                body = json.loads(response.read().decode("utf-8"))
            return any(item.get("id") == QWEN36_MODEL_ID for item in body.get("data", []))
        except (OSError, ValueError, urllib.error.URLError):
            return False

    def _port_owner_pid(self) -> int | None:
        try:
            for connection in psutil.net_connections(kind="tcp"):
                if connection.status != psutil.CONN_LISTEN or not connection.laddr:
                    continue
                local_port = getattr(connection.laddr, "port", connection.laddr[1])
                local_address = getattr(connection.laddr, "ip", connection.laddr[0])
                if local_port == self.port and local_address in {self.host, "0.0.0.0", "::", "::1"}:
                    return connection.pid
        except (psutil.AccessDenied, psutil.Error, OSError):
            return None
        return None

    @staticmethod
    def _process_create_time(pid: int) -> float:
        return float(psutil.Process(pid).create_time())

    def _register_managed_pid(self, pid: int, role: str) -> None:
        if pid == os.getpid() or pid <= 0:
            raise ChatRuntimeError("Refusing to register the FastAPI process as a model child.", code="PROCESS_OWNERSHIP_ERROR")
        try:
            record = _ManagedProcess(pid=pid, create_time=self._process_create_time(pid), role=role)
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError) as error:
            raise ChatRuntimeError(f"Could not track started Colibrì PID {pid}.", code="PROCESS_OWNERSHIP_ERROR") from error
        with self._lock:
            self._managed_processes[pid] = record

    def _identity_matches(self, record: _ManagedProcess) -> bool:
        if record.pid == os.getpid():
            return False
        try:
            return abs(self._process_create_time(record.pid) - record.create_time) < 0.01
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            return False

    def _refresh_managed_processes(self) -> None:
        with self._lock:
            process = self._process
            root_record = self._managed_processes.get(process.pid) if process is not None else None
        if process is None or root_record is None or not self._identity_matches(root_record):
            return
        try:
            descendants = psutil.Process(process.pid).children(recursive=True)
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            return
        for child in descendants:
            if child.pid == os.getpid():
                continue
            try:
                name = child.name().lower()
                role = "engine" if name == "qwen36.exe" else "server"
                record = _ManagedProcess(child.pid, float(child.create_time()), role)
            except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
                continue
            with self._lock:
                self._managed_processes[child.pid] = record

    def _reset_stale_generation(self) -> None:
        with self._lock:
            session = self._active_generation
            if session is not None:
                session.cancel_event.set()
        if session is not None:
            self._close_generation_io(session)
            session.done_event.wait(1.0)
        with self._lock:
            if session is not None and self._active_generation is session:
                self._active_generation = None
                session.connection = None
                session.response = None
                session.cancel_event.clear()
                session.done_event.set()
                if self._generation_lock.locked():
                    try:
                        self._generation_lock.release()
                    except RuntimeError:
                        pass

    def _remove_owned_pidfile(self, records: dict[int, _ManagedProcess]) -> None:
        pidfile = Path(tempfile.gettempdir()) / f"coli-serve-{self.port}.pid"
        try:
            pid = int(pidfile.read_text(encoding="utf-8").split()[0])
        except (OSError, ValueError, IndexError):
            return
        if pid in records:
            try:
                pidfile.unlink()
            except OSError:
                pass

    def _terminate_managed_processes(self) -> None:
        self._refresh_managed_processes()
        with self._lock:
            records = dict(self._managed_processes)
            process = self._process
        protected = {os.getpid()}
        try:
            protected.update(parent.pid for parent in psutil.Process(os.getpid()).parents())
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            pass
        targets: list[psutil.Process] = []
        for record in records.values():
            if record.pid in protected or not self._identity_matches(record):
                continue
            try:
                targets.append(psutil.Process(record.pid))
            except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
                continue
        if process is not None and process.pid not in records and process.pid != os.getpid():
            try:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=5)
            except (subprocess.TimeoutExpired, OSError):
                try:
                    process.kill()
                except OSError:
                    pass
        for target in reversed(targets):
            try:
                target.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
                pass
        _gone, alive = psutil.wait_procs(targets, timeout=10.0)
        for target in alive:
            try:
                target.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
                pass
        if alive:
            psutil.wait_procs(alive, timeout=5.0)
        if process is not None:
            try:
                process.wait(timeout=1)
            except (subprocess.TimeoutExpired, OSError):
                pass
        self._remove_owned_pidfile(records)
        with self._lock:
            self._managed_processes.clear()
            self._process = None

    def _recover_for_start(self) -> None:
        self._reset_stale_generation()
        self._terminate_managed_processes()
        owner_pid = self._port_owner_pid()
        if _port_open(self.host, self.port):
            owner = f"PID {owner_pid}" if owner_pid is not None else "an unidentified process"
            raise ChatRuntimeError(
                f"COLIBRI_PORT_IN_USE: {self.host}:{self.port} is occupied by {owner}; it is not a process owned by this model manager.",
                code="COLIBRI_PORT_IN_USE",
            )
        with self._lock:
            self._process = None
            self._managed_processes.clear()
            self._close_logs()

    def _record_startup_failure(self, error: Exception) -> None:
        self._terminate_managed_processes()
        with self._lock:
            self._process = None
            self._close_logs()
            self._state = "ERROR"
            self._error = str(error)
            self._error_code = error.code if isinstance(error, ChatRuntimeError) else "MODEL_START_FAILED"

    def stop(self) -> dict[str, Any]:
        with self._lock:
            if self._state == "STOPPED" and self._process is None and not self._managed_processes:
                return self.status()
            self._stop_requested.set()
            active_generation = self._active_generation
            if active_generation is not None:
                active_generation.cancel_event.set()
        if active_generation is not None:
            self._close_generation_io(active_generation)
        self._terminate_managed_processes()
        deadline = time.monotonic() + 10.0
        while _port_open(self.host, self.port) and time.monotonic() < deadline:
            time.sleep(0.2)
        port_closed = not _port_open(self.host, self.port)
        with self._lock:
            self._process = None
            self._managed_processes.clear()
            self._close_logs()
            if port_closed:
                self._state = "STOPPED"
                self._error = None
                self._error_code = None
            else:
                self._state = "ERROR"
                self._error = f"Colibrì stopped but internal port {self.port} remained open."
                self._error_code = "COLIBRI_PORT_IN_USE"
            return self.status()

    def _close_logs(self) -> None:
        for handle_name in ("_stdout_handle", "_stderr_handle"):
            handle = getattr(self, handle_name)
            if handle is not None:
                try:
                    handle.close()
                except OSError:
                    pass
                setattr(self, handle_name, None)

    def stream(self, messages: list[dict[str, str]], max_tokens: int) -> Iterator[bytes]:
        _generation_id, stream = self.stream_with_generation(messages, max_tokens)
        return stream

    def stream_with_generation(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
    ) -> tuple[int, Iterator[bytes]]:
        if isinstance(max_tokens, bool) or not 1 <= max_tokens <= QWEN36_MAX_TOKENS:
            raise ChatRuntimeError(f"max_tokens must be between 1 and {QWEN36_MAX_TOKENS}.")
        bounded_messages, prompt_stats = bound_browser_messages(messages)
        with self._lock:
            self._last_browser_prompt_stats = prompt_stats
        session = self._begin_generation()
        stream = self._stream_locked(bounded_messages, max_tokens, session)
        with self._lock:
            if self._active_generation is session:
                session.stream = stream
        return session.generation_id, stream

    def _begin_generation(self) -> _GenerationSession:
        if not self._generation_lock.acquire(blocking=False):
            raise ChatBusyError("BUSY: another generation is already running.")
        with self._lock:
            if self._state != "READY" or self._process is None or self._process.poll() is not None:
                self._generation_lock.release()
                raise ChatRuntimeError("The Qwen3.6 server is not ready.")
            self._next_generation_id += 1
            session = _GenerationSession(self._next_generation_id)
            self._active_generation = session
            self._state = "GENERATING"
            self._error = None
            return session

    def _finish_generation(self, session: _GenerationSession) -> None:
        cancelled = session.cancel_event.is_set()
        backend_idle = True
        if cancelled and session.require_backend_cancel_ack:
            backend_idle = self._wait_for_backend_idle(8.0)
        release_generation_lock = False
        with self._lock:
            if self._active_generation is not session:
                session.connection = None
                session.response = None
                session.stream = None
                session.cancel_event.clear()
                session.done_event.set()
                return
            self._active_generation = None
            session.connection = None
            session.response = None
            session.stream = None
            if not backend_idle and self._state != "STOPPED":
                self._state = "ERROR"
                self._error = "Colibrì did not acknowledge generation cancellation within 8 seconds."
                self._error_code = "GENERATION_CANCEL_TIMEOUT"
            elif self._process is not None and self._process.poll() is None and self._state != "STOPPED":
                self._state = "READY"
                self._error = None
                self._error_code = None
            elif self._state != "STOPPED":
                self._state = "ERROR"
                self._error = "Colibrì exited during generation."
                self._error_code = "MODEL_PROCESS_EXITED"
            session.cancel_event.clear()
            session.done_event.set()
            release_generation_lock = True
        if release_generation_lock:
            self._generation_lock.release()

    def _register_generation_io(
        self,
        session: _GenerationSession,
        *,
        connection: http.client.HTTPConnection | None = None,
        response: http.client.HTTPResponse | None = None,
    ) -> bool:
        with self._lock:
            if self._active_generation is not session or session.cancel_event.is_set():
                return False
            if connection is not None:
                session.connection = connection
            if response is not None:
                session.response = response
                session.require_backend_cancel_ack = isinstance(response, http.client.HTTPResponse)
            return True

    def _wait_for_backend_idle(self, wait_seconds: float) -> bool:
        deadline = time.monotonic() + wait_seconds
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(f"http://{self.host}:{self.port}/health", timeout=0.5) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                scheduler = payload.get("scheduler") or {}
                if scheduler.get("active") == 0:
                    return True
            except (OSError, ValueError, urllib.error.URLError):
                with self._lock:
                    if self._process is None or self._process.poll() is not None:
                        return True
            time.sleep(0.05)
        return False

    @staticmethod
    def _close_generation_io(session: _GenerationSession) -> None:
        connection = session.connection
        response = session.response
        sockets: list[Any] = []
        if connection is not None and getattr(connection, "sock", None) is not None:
            sockets.append(connection.sock)
        if response is not None:
            response_fp = getattr(response, "fp", None)
            response_raw = getattr(response_fp, "raw", None)
            for candidate in (
                getattr(response_fp, "_sock", None),
                getattr(response_raw, "_sock", None),
            ):
                if candidate is not None:
                    sockets.append(candidate)
        seen_sockets: set[int] = set()
        for sock in sockets:
            if id(sock) in seen_sockets:
                continue
            seen_sockets.add(id(sock))
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except (OSError, ValueError):
                pass
            try:
                sock.close()
            except (AttributeError, OSError, ValueError):
                pass
        if response is not None:
            try:
                response.close()
            except (OSError, ValueError):
                pass
        if connection is not None:
            try:
                connection.close()
            except (OSError, ValueError):
                pass

    def cancel_generation(
        self,
        generation_id: int | None = None,
        *,
        wait_seconds: float = 10.0,
    ) -> dict[str, Any]:
        with self._lock:
            session = self._active_generation
            if session is None or (generation_id is not None and session.generation_id != generation_id):
                result = self.status()
                result.update({"cancelled": False, "generation_id": generation_id})
                return result
            session.cancel_event.set()
        self._close_generation_io(session)
        stream = session.stream
        if stream is not None:
            try:
                stream.close()  # type: ignore[attr-defined]
            except (AttributeError, RuntimeError, ValueError):
                # An executing generator cannot be closed from another thread.
                # Closing its socket above wakes the blocking read; its finally
                # block then performs the same authoritative cleanup.
                pass
        if not session.done_event.wait(wait_seconds):
            raise ChatRuntimeError(f"The active generation did not stop within {wait_seconds:g} seconds.")
        result = self.status()
        result.update({"cancelled": True, "generation_id": session.generation_id})
        return result

    @staticmethod
    def _backend_payload(
        messages: list[dict[str, str]],
        max_tokens: int,
        *,
        stream: bool,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": QWEN36_MODEL_ID,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": stream,
            "reasoning_effort": "none",
            "enable_thinking": False,
        }
        if stream:
            payload["stream_options"] = {"include_usage": True}
        if temperature is not None:
            payload["temperature"] = temperature
        return payload

    def _stream_locked(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        session: _GenerationSession,
    ) -> Iterator[bytes]:
        connection: http.client.HTTPConnection | None = None
        started = time.monotonic()
        first_token_at: float | None = None
        completion_tokens: int | None = None
        content_parts: list[str] = []
        try:
            payload = self._backend_payload(
                [{"role": "system", "content": SYSTEM_PROMPT}, *messages],
                max_tokens,
                stream=True,
            )
            connection = http.client.HTTPConnection(self.host, self.port, timeout=None)
            if not self._register_generation_io(session, connection=connection):
                return
            connection.request(
                "POST",
                "/v1/chat/completions",
                body=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            if session.cancel_event.is_set():
                return
            response = connection.getresponse()
            if not self._register_generation_io(session, response=response):
                return
            if response.status != 200:
                detail = response.read().decode("utf-8", "replace")
                raise ChatRuntimeError(f"Colibrì returned HTTP {response.status}: {detail[:500]}")
            while True:
                raw_line = response.readline()
                if not raw_line:
                    break
                line = raw_line.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                event = json.loads(data)
                usage = event.get("usage")
                if isinstance(usage, dict) and isinstance(usage.get("completion_tokens"), int):
                    completion_tokens = usage["completion_tokens"]
                choices = event.get("choices") or []
                if choices:
                    content = (choices[0].get("delta") or {}).get("content")
                    if content:
                        if first_token_at is None:
                            first_token_at = time.monotonic()
                        content_parts.append(content)
                        yield self._event("delta", content=content)
            finished = time.monotonic()
            ttft = None if first_token_at is None else first_token_at - started
            decode_seconds = None if first_token_at is None else max(0.0, finished - first_token_at)
            decode_tps = None
            if completion_tokens is not None and decode_seconds and decode_seconds > 0:
                decode_tps = completion_tokens / decode_seconds
            yield self._event(
                "metrics",
                ttft_seconds=None if ttft is None else round(ttft, 3),
                completion_tokens=completion_tokens,
                decode_tokens_per_second=None if decode_tps is None else round(decode_tps, 3),
                total_response_seconds=round(finished - started, 3),
            )
            yield self._event("done", content="".join(content_parts))
        except GeneratorExit:
            raise
        except Exception as error:
            if not session.cancel_event.is_set():
                yield self._event("error", message=str(error))
        finally:
            self._close_generation_io(session)
            self._finish_generation(session)

    def complete_openai(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> dict[str, Any]:
        session = self._begin_generation()
        connection: http.client.HTTPConnection | None = None
        try:
            payload = self._backend_payload(
                messages,
                max_tokens,
                stream=False,
                temperature=temperature,
            )
            connection = http.client.HTTPConnection(self.host, self.port, timeout=None)
            if not self._register_generation_io(session, connection=connection):
                raise ChatRuntimeError("The generation was cancelled.")
            connection.request(
                "POST",
                "/v1/chat/completions",
                body=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            if not self._register_generation_io(session, response=response):
                raise ChatRuntimeError("The generation was cancelled.")
            raw_body = response.read().decode("utf-8", "replace")
            if response.status != 200:
                raise ChatRuntimeError(f"Colibrì returned HTTP {response.status}: {raw_body[:500]}")
            body = json.loads(raw_body)
            if not isinstance(body, dict):
                raise ChatRuntimeError("Colibrì returned an invalid chat completion response.")
            body.setdefault("usage", None)
            return body
        except ChatRuntimeError:
            raise
        except Exception as error:
            raise ChatRuntimeError(f"Colibrì request failed: {error}") from error
        finally:
            self._close_generation_io(session)
            self._finish_generation(session)

    def stream_openai(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> Iterator[bytes]:
        session = self._begin_generation()
        connection: http.client.HTTPConnection | None = None
        try:
            payload = self._backend_payload(
                messages,
                max_tokens,
                stream=True,
                temperature=temperature,
            )
            connection = http.client.HTTPConnection(self.host, self.port, timeout=None)
            if not self._register_generation_io(session, connection=connection):
                raise ChatRuntimeError("The generation was cancelled.")
            connection.request(
                "POST",
                "/v1/chat/completions",
                body=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            if not self._register_generation_io(session, response=response):
                raise ChatRuntimeError("The generation was cancelled.")
            if response.status != 200:
                detail = response.read().decode("utf-8", "replace")
                raise ChatRuntimeError(f"Colibrì returned HTTP {response.status}: {detail[:500]}")
            stream = self._relay_openai_stream(connection, response, session)
            with self._lock:
                if self._active_generation is session:
                    session.stream = stream
            return stream
        except ChatRuntimeError:
            self._close_generation_io(session)
            self._finish_generation(session)
            raise
        except Exception as error:
            self._close_generation_io(session)
            self._finish_generation(session)
            raise ChatRuntimeError(f"Colibrì request failed: {error}") from error

    def _relay_openai_stream(
        self,
        connection: http.client.HTTPConnection,
        response: http.client.HTTPResponse,
        session: _GenerationSession,
    ) -> Iterator[bytes]:
        done = False
        try:
            while True:
                raw_line = response.readline()
                if not raw_line:
                    break
                line = raw_line.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                yield (line + "\n\n").encode("utf-8")
                if line[5:].strip() == "[DONE]":
                    done = True
                    break
            if not done and not session.cancel_event.is_set():
                error = {
                    "error": {
                        "message": "Colibrì stream ended before the completion marker.",
                        "type": "backend_error",
                        "code": "backend_stream_incomplete",
                    }
                }
                yield ("data: " + json.dumps(error) + "\n\n").encode("utf-8")
                yield b"data: [DONE]\n\n"
        except GeneratorExit:
            raise
        except Exception as error:
            if not session.cancel_event.is_set():
                body = {
                    "error": {
                        "message": f"Colibrì stream failed: {error}",
                        "type": "backend_error",
                        "code": "backend_stream_failed",
                    }
                }
                yield ("data: " + json.dumps(body) + "\n\n").encode("utf-8")
                yield b"data: [DONE]\n\n"
        finally:
            self._close_generation_io(session)
            self._finish_generation(session)

    @staticmethod
    def _event(event_type: str, **payload: Any) -> bytes:
        return (json.dumps({"type": event_type, **payload}, ensure_ascii=False) + "\n").encode("utf-8")


chat_runtime = QwenChatRuntime()
