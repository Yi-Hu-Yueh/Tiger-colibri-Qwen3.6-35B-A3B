from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
BENCHMARK_DIR = DATA_DIR / "benchmarks"
STATE_DIR = DATA_DIR / "state"
STATE_FILE = STATE_DIR / "phase1.json"
VENDOR_DIR = PROJECT_ROOT / "vendor" / "colibri"
DOWNLOAD_ROOT = Path(r"D:\TigerModels\Tiger-colibri-Qwen3.6-35B-A3B\downloads")
MODEL_REPO_ID = "Kreuzzelg/qwen36-35b-a3b-colibri-i4-gs64"
MODEL_DESTINATION = DOWNLOAD_ROOT / "qwen36-35b-a3b-colibri-i4-gs64"
HF_CACHE_DIR = DOWNLOAD_ROOT / ".huggingface-cache"
HF_XET_CACHE_DIR = DOWNLOAD_ROOT / ".huggingface-xet"
COLIBRI_QWEN36_DOC_URL = "https://raw.githubusercontent.com/JustVugg/colibri/main/docs/qwen36.md"
OLLAMA_MODEL_SOURCE = Path(r"C:\Users\user\.ollama\models")
OLLAMA_MODEL_DESTINATION = Path(r"D:\TigerModels\Ollama")
OLLAMA_EXECUTABLE = Path(r"C:\Users\user\AppData\Local\Programs\Ollama\ollama.exe")
OLLAMA_APP_EXECUTABLE = Path(r"C:\Users\user\AppData\Local\Programs\Ollama\ollama app.exe")
TEMPLATES_DIR = PROJECT_ROOT / "app" / "templates"
STATIC_DIR = PROJECT_ROOT / "app" / "static"
COLIBRI_RUNTIME_DIR = VENDOR_DIR / "v1.12.1" / "runtime"
COLIBRI_CMD = COLIBRI_RUNTIME_DIR / "coli.cmd"
QWEN36_RUNTIME_ROOT = Path(
    r"C:\TigerModels\Tiger-colibri-Qwen3.6-35B-A3B\runtime\qwen36-35b-a3b-colibri-i4-gs64"
)
QWEN36_HOST = "127.0.0.1"
QWEN36_PORT = 18150
QWEN36_MODEL_ID = "qwen36"
QWEN36_CAP = 16
QWEN36_OMP_THREADS = 4
QWEN36_MAX_TOKENS = 4096
QWEN36_MIN_AVAILABLE_RAM_BYTES = 4 * 1024**3
GITHUB_REPOSITORY = "JustVugg/colibri"
GITHUB_RELEASE_API = f"https://api.github.com/repos/{GITHUB_REPOSITORY}/releases/latest"
MIN_FREE_GIB = 30.0
BENCHMARK_SIZES_MIB = {256, 512}


def ensure_directories() -> None:
    for path in (BENCHMARK_DIR, STATE_DIR, VENDOR_DIR):
        path.mkdir(parents=True, exist_ok=True)
