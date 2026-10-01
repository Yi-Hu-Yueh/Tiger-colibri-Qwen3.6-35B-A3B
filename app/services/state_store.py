from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import STATE_FILE

_lock = threading.RLock()


def _default_state() -> dict[str, Any]:
    return {"phase": 1, "updated_at": None}


def load_state(path: Path = STATE_FILE) -> dict[str, Any]:
    with _lock:
        if not path.exists():
            return _default_state()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else _default_state()
        except (OSError, json.JSONDecodeError):
            return _default_state()


def save_state(data: dict[str, Any], path: Path = STATE_FILE) -> None:
    with _lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(data)
        payload["updated_at"] = datetime.now(timezone.utc).isoformat()
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(temp, path)


def update_state(key: str, value: Any, path: Path = STATE_FILE) -> dict[str, Any]:
    with _lock:
        state = load_state(path)
        state[key] = value
        save_state(state, path)
        return state
