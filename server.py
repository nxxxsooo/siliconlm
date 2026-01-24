#!/usr/bin/env python3
"""Localboard - Local Services Dashboard"""

import os
import signal
import socket
import subprocess
import uuid
from pathlib import Path
from typing import Optional

import psutil
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI(title="Localboard")

# Configuration
MODELS_DIR = Path.home() / ".lmstudio" / "models"
DASHBOARD_DIR = Path(__file__).parent
DOWNLOADS: dict = {}  # {task_id: {repo, progress, speed, status}}

# Service definitions with start commands
SERVICES = {
    "lmstudio": {
        "display": "LMStudio Server",
        "port": 1234,
        "check": "port",
        "start_cmd": None,  # Started via LMStudio app
        "note": "Start from LMStudio app"
    },
    "opencode": {
        "display": "OpenCode",
        "process": "opencode",
        "check": "process",
        "start_cmd": ["opencode"],
        "start_in_terminal": True
    },
    "claude": {
        "display": "Claude Code",
        "process": "claude",
        "check": "process",
        "start_cmd": ["claude"],
        "start_in_terminal": True
    }
}

# Recommended models for download
DOWNLOAD_PRESETS = [
    {"repo": "mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit", "desc": "Fast MoE model (~17GB)"},
    {"repo": "mlx-community/Qwen2.5-72B-Instruct-4bit", "desc": "Main workhorse (~41GB)"},
    {"repo": "mlx-community/DeepSeek-R1-Distill-Llama-70B-4bit", "desc": "Deep thinking (~40GB)"},
    {"repo": "mlx-community/mxbai-embed-large-v1", "desc": "Embedding model (~670MB)"},
    {"repo": "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ", "desc": "Fast embedding (~350MB)"},
    {"repo": "mlx-community/e5-mistral-7b-instruct-mlx", "desc": "Large embedding (~2.8GB)"},
]


def check_port(port: int) -> bool:
    """Check if a port is in use"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


def check_process(name: str) -> Optional[int]:
    """Check if a process is running, return PID or None"""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info.get('cmdline') or []
            if name in proc.info.get('name', '') or any(name in arg for arg in cmdline):
                return proc.info['pid']
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return None


def get_service_status(name: str) -> dict:
    """Get status of a service"""
    service = SERVICES.get(name, {})
    status = {
        "name": name,
        "display": service.get("display", name),
        "running": False,
        "pid": None,
        "can_start": service.get("start_cmd") is not None,
        "note": service.get("note")
    }

    if service.get("check") == "port":
        port = service.get("port")
        status["running"] = check_port(port)
        status["port"] = port
    elif service.get("check") == "process":
        pid = check_process(service.get("process", name))
        status["running"] = pid is not None
        status["pid"] = pid

    return status


def get_system_stats() -> dict:
    """Get system memory and disk stats"""
    mem = psutil.virtual_memory()

    if MODELS_DIR.exists():
        disk = psutil.disk_usage(str(MODELS_DIR))
        models_size = sum(f.stat().st_size for f in MODELS_DIR.rglob('*') if f.is_file())
    else:
        disk = psutil.disk_usage('/')
        models_size = 0

    return {
        "memory": {"total": mem.total, "used": mem.used, "percent": mem.percent},
        "disk": {"total": disk.total, "used": disk.used, "free": disk.free, "percent": disk.percent},
        "models_size": models_size
    }


def get_downloaded_models() -> list:
    """List downloaded models (only existing ones)"""
    models = []
    if MODELS_DIR.exists():
        for org_dir in MODELS_DIR.iterdir():
            if org_dir.is_dir():
                for model_dir in org_dir.iterdir():
                    if model_dir.is_dir() and model_dir.exists():
                        try:
                            size = sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file())
                            models.append({
                                "name": f"{org_dir.name}/{model_dir.name}",
                                "size": size,
                                "path": str(model_dir)
                            })
                        except (OSError, FileNotFoundError):
                            pass  # Skip if deleted
    return sorted(models, key=lambda x: x['size'], reverse=True)


# API Routes
@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the dashboard HTML"""
    html_path = DASHBOARD_DIR / "templates" / "index.html"
    return html_path.read_text()


@app.get("/api/status")
async def get_status():
    """Get all service statuses and system stats"""
    services = [get_service_status(name) for name in SERVICES]
    return {
        "services": services,
        "system": get_system_stats(),
        "models": get_downloaded_models(),
        "downloads": list(DOWNLOADS.values()),
        "presets": DOWNLOAD_PRESETS
    }


@app.post("/api/service/{name}/start")
async def start_service(name: str):
    """Start a service"""
    service = SERVICES.get(name)
    if not service:
        return {"success": False, "message": "Unknown service"}

    if not service.get("start_cmd"):
        return {"success": False, "message": service.get("note", "Cannot start this service")}

    try:
        if service.get("start_in_terminal"):
            # Open in new Terminal window
            cmd = service["start_cmd"][0]
            subprocess.Popen([
                "osascript", "-e",
                f'tell application "Terminal" to do script "{cmd}"'
            ])
        else:
            subprocess.Popen(service["start_cmd"], start_new_session=True)
        return {"success": True, "message": f"Starting {name}"}
    except Exception as e:
        return {"success": False, "message": str(e)}


@app.post("/api/service/{name}/stop")
async def stop_service(name: str):
    """Stop a service"""
    status = get_service_status(name)
    if status["pid"]:
        try:
            os.kill(status["pid"], signal.SIGTERM)
            return {"success": True, "message": f"Sent SIGTERM to {name}"}
        except Exception as e:
            return {"success": False, "message": str(e)}
    return {"success": False, "message": "Service not running or PID not found"}


@app.post("/api/service/{name}/restart")
async def restart_service(name: str):
    """Restart a service"""
    await stop_service(name)
    import asyncio
    await asyncio.sleep(1)
    return await start_service(name)


@app.post("/api/model/reveal")
async def reveal_model(path: str):
    """Reveal model in Finder"""
    model_path = Path(path)
    if model_path.exists():
        subprocess.run(["open", "-R", str(model_path)])
        return {"success": True}
    return {"success": False, "message": "Path not found"}


class DownloadRequest(BaseModel):
    repo: str


@app.post("/api/download")
async def start_download(request: DownloadRequest, background_tasks: BackgroundTasks):
    """Start a HuggingFace model download"""
    repo = request.repo
    task_id = str(uuid.uuid4())[:8]

    DOWNLOADS[task_id] = {
        "id": task_id,
        "repo": repo,
        "progress": 0,
        "status": "starting",
        "speed": "0 MB/s"
    }

    background_tasks.add_task(download_model, task_id, repo)
    return {"success": True, "task_id": task_id}


def download_model(task_id: str, repo: str):
    """Background task to download a model"""
    try:
        DOWNLOADS[task_id]["status"] = "downloading"

        from huggingface_hub import snapshot_download

        local_dir = MODELS_DIR / repo

        snapshot_download(
            repo,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False
        )

        DOWNLOADS[task_id]["status"] = "completed"
        DOWNLOADS[task_id]["progress"] = 100

    except Exception as e:
        DOWNLOADS[task_id]["status"] = "error"
        DOWNLOADS[task_id]["error"] = str(e)


@app.delete("/api/download/{task_id}")
async def cancel_download(task_id: str):
    """Cancel/remove a download from list"""
    if task_id in DOWNLOADS:
        del DOWNLOADS[task_id]
        return {"success": True}
    return {"success": False, "message": "Task not found"}


if __name__ == "__main__":
    import uvicorn
    print("🖥️  Localboard starting at http://localhost:8765")
    uvicorn.run(app, host="0.0.0.0", port=8765)
