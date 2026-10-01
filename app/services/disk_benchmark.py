from __future__ import annotations

import os
import random
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from app.config import BENCHMARK_DIR, BENCHMARK_SIZES_MIB, PROJECT_ROOT

_benchmark_lock = threading.Lock()


def benchmark_storage(size_mib: int, target: Path = PROJECT_ROOT) -> dict:
    if size_mib not in BENCHMARK_SIZES_MIB:
        raise ValueError(f"Benchmark size must be one of {sorted(BENCHMARK_SIZES_MIB)} MiB")
    if not _benchmark_lock.acquire(blocking=False):
        raise RuntimeError("A storage benchmark is already running")
    temp_path: Path | None = None
    timestamp = datetime.now(timezone.utc).isoformat()
    result = {
        "status": "FAIL", "timestamp": timestamp, "target": str(target.resolve()), "test_size_mib": size_mib,
        "write_elapsed_seconds": None, "write_mbps": None, "read_elapsed_seconds": None, "read_mbps": None,
        "random_read_elapsed_seconds": None, "random_read_mbps": None,
        "note": "Indicative local benchmark; Windows caching may affect results.", "errors": [],
    }
    try:
        target.mkdir(parents=True, exist_ok=True)
        fd, raw_path = tempfile.mkstemp(prefix=".colibri-benchmark-", suffix=".tmp", dir=target)
        os.close(fd)
        temp_path = Path(raw_path)
        total_bytes = size_mib * 2**20
        block = b"\xA5" * (4 * 2**20)
        start = time.perf_counter()
        with temp_path.open("wb", buffering=0) as handle:
            remaining = total_bytes
            while remaining:
                chunk = block[: min(len(block), remaining)]
                handle.write(chunk)
                remaining -= len(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        elapsed = max(time.perf_counter() - start, 1e-9)
        result["write_elapsed_seconds"] = round(elapsed, 4)
        result["write_mbps"] = round(size_mib / elapsed, 2)

        start = time.perf_counter()
        read_bytes = 0
        with temp_path.open("rb", buffering=0) as handle:
            while chunk := handle.read(4 * 2**20):
                read_bytes += len(chunk)
        elapsed = max(time.perf_counter() - start, 1e-9)
        result["read_elapsed_seconds"] = round(elapsed, 4)
        result["read_mbps"] = round((read_bytes / 2**20) / elapsed, 2)

        random_block = 256 * 1024
        samples = min(256, max(1, total_bytes // random_block))
        offsets = [random.randrange(0, total_bytes - random_block + 1) for _ in range(samples)]
        start = time.perf_counter()
        with temp_path.open("rb", buffering=0) as handle:
            for offset in offsets:
                handle.seek(offset)
                handle.read(random_block)
        elapsed = max(time.perf_counter() - start, 1e-9)
        result["random_read_elapsed_seconds"] = round(elapsed, 4)
        result["random_read_mbps"] = round((samples * random_block / 2**20) / elapsed, 2)
        result["random_read_block_kib"] = 256
        result["random_read_samples"] = samples
        result["status"] = "PASS"
    except Exception as exc:
        result["errors"].append(str(exc))
    finally:
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError as exc:
                result["errors"].append(f"Temporary file cleanup failed: {exc}")
                result["status"] = "FAIL"
        _benchmark_lock.release()
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    safe_timestamp = timestamp.replace(":", "-")
    output = BENCHMARK_DIR / f"benchmark-{safe_timestamp}.json"
    import json
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["result_file"] = str(output)
    return result
