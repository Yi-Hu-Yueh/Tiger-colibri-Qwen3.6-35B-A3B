# Tiger Colibri Qwen3.6 — Phase 1-R

A local FastAPI dashboard for inspecting this Windows machine and validating the official Colibrì Windows prebuilt before any model work begins. Phase 1-R repairs storage, Ollama, and Python runtime validation; it does **not** claim practical Qwen3.6 performance.

## Scope and hardware target

The dashboard probes Windows, CPU, RAM, NVIDIA GPU/VRAM/driver information, every physical disk and mapped local volume, Python, and the existing Ollama baseline. Storage has three separate roles:

- `PROJECT_ROOT`: source code and application files in this repository.
- `DOWNLOAD_ROOT`: `D:\TigerModels\Tiger-colibri-Qwen3.6-35B-A3B\downloads`, reserved for future model acquisition, archive, checksum, and staging work.
- `MODEL_RUNTIME_ROOT`: separately selected practical Colibrì inference/expert-streaming storage; initially unset.

The HDD-backed download root can pass the acquisition-capacity gate without being accepted as practical runtime storage. A runtime benchmark is available only after `MODEL_RUNTIME_ROOT` is configured.

The download root requires a writable fixed local filesystem and at least 30 GiB free; HDD is permitted for this acquisition/archive role. Runtime storage is evaluated separately: NVMe passes, SATA SSD or SSD with a hidden bus warns, HDD fails, and an unset root is not ready. Project-drive media does not itself fail either gate. A 16 GiB machine receives a warning rather than an automatic failure. GPU findings are informational in Phase 1-R.

## Start the web UI

Python 3.11.x is mandatory. Install the small dependency set with that interpreter:

```powershell
D:\python3.11.3\python.exe -m pip install -r requirements.txt
./scripts/start.ps1
```

The start script prefers `COLIBRI_PYTHON`, then validates `D:\python3.11.3\python.exe`, then searches safely for another Python 3.11 interpreter. It refuses to fall back to Python 3.14. It never changes PATH or Python associations.

Open <http://127.0.0.1:18081>. If PowerShell script execution is blocked, no policy change is required:

```powershell
D:\python3.11.3\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 18081
```

## Safety boundaries

- **Model download lock:** model download is locked until Phase 1 validation passes. This application contains no model downloader and never downloads Qwen3.6 weights.
- **CUDA lock:** Phase 1 does not install CUDA, compile CUDA libraries, build Colibrì from source, modify CUDA variables, or install build toolchains. CUDA work belongs to Phase 4.
- **Ollama protection:** Ollama is baseline only and is not modified. The app only locates the executable and runs `--version` and `list`; it does not pull/delete models or change the service, config, or environment.
- **Model-storage path safety:** the browser selects a detected local drive letter. The server derives `<drive>:\TigerModels\Tiger-colibri-Qwen3.6-35B-A3B`; arbitrary paths, network shares, and removable media are rejected.
- Downloads and extraction occur only under `vendor/colibri/<version>/`. PATH and Windows system folders are never changed.
- Official assets must come from `https://github.com/JustVugg/colibri/releases/download/...`.
- ZIP paths are checked before extraction. A published checksum mismatch is a hard failure.
- Windows security controls are never bypassed.

## Status semantics

- **PASS:** the check met its Phase 1 requirement using recorded evidence.
- **WARN:** non-blocking limitation or manual review is required.
- **FAIL:** a required Phase 1 condition is missing or unsafe.
- **NOT RUN:** no evidence has been collected yet.

The overall result is `PASS`, `CONDITIONAL PASS`, or `FAIL`. A required failure always makes the gate fail. Warnings produce a conditional pass once all required checks pass.

## Project layout

```text
app/                    FastAPI app, routes, services, schemas, UI
data/benchmarks/        Timestamped benchmark results
data/state/phase1.json  Atomic non-sensitive Phase 1 state
scripts/                Start and environment checks
tests/                  Deterministic unit tests
vendor/colibri/         Versioned downloads and extracted runtimes
```

## Tests

```powershell
python -m pytest
```

The tests cover gate logic, free-space thresholds, the RAM warning, storage classification, official asset filtering, checksum verification, ZIP-slip prevention, atomic state serialization, and subprocess timeout/error behavior.

## Explicitly out of scope

Phase 1 does not download model weights, run inference, evaluate CPU streaming, install or validate CUDA, compile Colibrì, modify Ollama, or begin Phase 2. Those actions require separate authorization.
