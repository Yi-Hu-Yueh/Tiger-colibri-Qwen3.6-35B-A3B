from __future__ import annotations

import platform
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.api import chat, colibri, hardware, model, ollama_storage, openai_compat, phase, storage, web_search
from app.config import STATIC_DIR, TEMPLATES_DIR, ensure_directories
from app.services.qwen_chat import chat_runtime

ensure_directories()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield
    chat_runtime.stop()


app = FastAPI(title="Tiger Colibri Phase 7", version="1.1.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

app.include_router(hardware.router)
app.include_router(storage.router)
app.include_router(colibri.router)
app.include_router(phase.router)
app.include_router(model.router)
app.include_router(ollama_storage.router)
app.include_router(chat.router)
app.include_router(openai_compat.router)
app.include_router(web_search.router)


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, error: RequestValidationError):
    if request.url.path.startswith("/v1/"):
        details = error.errors()
        message = details[0].get("msg", "Invalid request.") if details else "Invalid request."
        return openai_compat.openai_error(422, message, "invalid_request_error", "validation_error")
    return await request_validation_exception_handler(request, error)


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={})


@app.get("/api/health")
def health() -> dict:
    runtime_ok = sys.version_info[:2] == (3, 11)
    return {"status": "PASS" if runtime_ok else "FAIL", "service": "Tiger Colibri Phase 7", "phase": "7",
            "python_version": platform.python_version(), "python_executable": sys.executable,
            "required_python": "3.11.x", "runtime_valid": runtime_ok}
