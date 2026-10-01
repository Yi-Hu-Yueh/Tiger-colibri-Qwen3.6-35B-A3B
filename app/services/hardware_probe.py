from __future__ import annotations

import ctypes
import json
import os
import platform
import sys
from datetime import datetime, timezone
from typing import Any

from app.services.command_runner import run_command


def _registry_system_info() -> dict[str, Any]:
    if os.name != "nt":
        return {}
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            cpu = winreg.QueryValueEx(key, "ProcessorNameString")[0]
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows NT\CurrentVersion") as key:
            product = winreg.QueryValueEx(key, "ProductName")[0]
            display = winreg.QueryValueEx(key, "DisplayVersion")[0]
            build = winreg.QueryValueEx(key, "CurrentBuildNumber")[0]
        return {"cpu": cpu, "product": product, "display": display, "build": build}
    except OSError:
        return {}


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def _powershell_json(script: str, timeout: float = 15) -> Any:
    result = run_command(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script], timeout)
    if result["exit_code"] != 0 or not result["stdout"]:
        return None
    try:
        return json.loads(result["stdout"])
    except json.JSONDecodeError:
        return None


def _memory() -> dict[str, Any]:
    if os.name == "nt":
        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            total, available = status.ullTotalPhys, status.ullAvailPhys
            used = total - available
            return {
                "total_bytes": total, "available_bytes": available, "used_bytes": used,
                "percent_used": round(used * 100 / total, 1) if total else 0.0,
                "total_gib": round(total / 2**30, 2), "available_gib": round(available / 2**30, 2),
                "used_gib": round(used / 2**30, 2),
            }
    return {"total_bytes": None, "available_bytes": None, "used_bytes": None, "percent_used": None,
            "total_gib": None, "available_gib": None, "used_gib": None}


def _gpu() -> dict[str, Any]:
    query = run_command([
        "nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"
    ], timeout=10)
    gpus: list[dict[str, Any]] = []
    if query["exit_code"] == 0:
        for line in query["stdout"].splitlines():
            fields = [x.strip() for x in line.split(",")]
            if len(fields) >= 3:
                try:
                    memory_mib = int(fields[1])
                except ValueError:
                    memory_mib = None
                gpus.append({"name": fields[0], "vram_mib": memory_mib, "driver_version": fields[2], "source": "nvidia-smi"})
    if not gpus and os.name == "nt":
        cim = _powershell_json("Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM,DriverVersion | ConvertTo-Json -Compress")
        items = cim if isinstance(cim, list) else ([cim] if isinstance(cim, dict) else [])
        for item in items:
            vram = item.get("AdapterRAM")
            gpus.append({"name": item.get("Name"), "vram_mib": round(vram / 2**20) if isinstance(vram, int) else None,
                         "driver_version": item.get("DriverVersion"), "source": "Windows CIM"})
    cuda = run_command(["nvidia-smi"], timeout=10) if query["exit_code"] == 0 else None
    cuda_version = None
    if cuda:
        import re
        match = re.search(r"CUDA Version:\s*([\d.]+)", cuda["stdout"])
        cuda_version = match.group(1) if match else None
    return {"nvidia_smi_available": query["exit_code"] == 0, "cuda_runtime_reported": cuda_version,
            "cuda_note": "CUDA installation and validation are deferred to Phase 4.", "devices": gpus,
            "probe_error": query["error"] or (query["stderr"] if query["exit_code"] not in (0, None) else None)}


def probe_hardware() -> dict[str, Any]:
    computer = _powershell_json(
        "$cpu=Get-CimInstance Win32_Processor|Select-Object -First 1 Name,NumberOfCores,NumberOfLogicalProcessors;"
        "$os=Get-CimInstance Win32_OperatingSystem|Select-Object Caption,Version,BuildNumber,OSArchitecture;"
        "[pscustomobject]@{cpu=$cpu;os=$os}|ConvertTo-Json -Depth 4 -Compress"
    ) if os.name == "nt" else None
    cpu_info = (computer or {}).get("cpu", {}) if isinstance(computer, dict) else {}
    os_info = (computer or {}).get("os", {}) if isinstance(computer, dict) else {}
    registry = _registry_system_info()
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cpu": {
            "model": (cpu_info.get("Name") or registry.get("cpu") or platform.processor() or "Unknown").strip(),
            "physical_cores": cpu_info.get("NumberOfCores"),
            "logical_processors": cpu_info.get("NumberOfLogicalProcessors") or os.cpu_count(),
            "architecture": platform.machine(),
        },
        "ram": _memory(),
        "windows": {
            "detected": platform.system() == "Windows", "edition": os_info.get("Caption") or registry.get("product") or platform.system(),
            "version": os_info.get("Version") or registry.get("display") or platform.version(), "build": os_info.get("BuildNumber") or registry.get("build") or platform.release(),
            "architecture": os_info.get("OSArchitecture") or platform.machine(),
        },
        "python": {"version": platform.python_version(), "executable": sys.executable,
                   "implementation": platform.python_implementation(), "meets_minimum": sys.version_info >= (3, 10)},
        "gpu": _gpu(),
    }
