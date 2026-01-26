#!/usr/bin/env python3
"""SiliconLM - Apple Silicon LLM Dashboard"""

import json
import socket
import subprocess
import time
from pathlib import Path
from typing import Optional

import httpx
import psutil
from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from download_manager import download_manager, PRESET_MODELS

app = FastAPI(title="SiliconLM")

SETTINGS_FILE = Path(__file__).parent / "settings.json"
DEFAULT_SETTINGS = {
    "models_dir": "~/.lmstudio/models",
    "services": {
        "mlx_embeddings": {"enabled": True, "port": 8766},
        "lmstudio": {"enabled": True, "port": 1234},
        "opencode": {"enabled": True}
    },
    "embedding": {"max_batch_size": 32, "max_input_chars": 8192}
}

def load_settings():
    if SETTINGS_FILE.exists():
        with open(SETTINGS_FILE) as f:
            return json.load(f)
    return DEFAULT_SETTINGS.copy()

def save_settings(settings):
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f, indent=2)

_settings = load_settings()

PROXY_TARGETS = {
    "embeddings": "http://localhost:8766",
    "lmstudio": "http://localhost:1234",
}

EMBEDDING_MODELS = {"embed", "gte-", "bge-", "e5-", "mxbai-embed", "nomic-embed"}


def _is_embedding_request(path: str, body: dict = None) -> bool:
    if "/embeddings" in path:
        return True
    if body and body.get("model"):
        model = body["model"].lower()
        return any(p in model for p in EMBEDDING_MODELS)
    return False


async def _proxy_request(request: Request, target_url: str) -> Response:
    async with httpx.AsyncClient(timeout=300.0, trust_env=False) as client:
        url = f"{target_url}{request.url.path}"
        if request.url.query:
            url = f"{url}?{request.url.query}"
        
        headers = dict(request.headers)
        headers.pop("host", None)
        
        body = await request.body()
        
        response = await client.request(
            method=request.method,
            url=url,
            headers=headers,
            content=body,
        )
        
        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=dict(response.headers),
        )

# Cache for expensive computations
_cache = {
    "models": {"data": [], "total_size": 0, "timestamp": 0},
}
CACHE_TTL = 30

# LMStudio request tracking
_lmstudio_stats = {
    "requests": 0,
    "tokens": 0,
    "start_time": time.time(),
}

# Combined API activity log (both embeddings and LMStudio)
from collections import deque
_activity_log = deque(maxlen=50)

# Start download manager on app startup
@app.on_event("startup")
async def startup_event():
    download_manager.start()

@app.on_event("shutdown")
async def shutdown_event():
    download_manager.stop()

# Static files
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Configuration
def get_models_dir():
    models_path = _settings.get("models_dir", "~/.lmstudio/models")
    return Path(models_path).expanduser()

MODELS_DIR = get_models_dir()
DASHBOARD_DIR = Path(__file__).parent

# Service definitions
SERVICES = {
    "mlx_embeddings": {
        "display": "MLX Embeddings",
        "port": 8766,
        "check": "port",
        "process": "embedding_server",
        "start_cmd": [str(DASHBOARD_DIR / ".venv" / "bin" / "python"), str(DASHBOARD_DIR / "embedding_server.py")],
        "metrics_url": "http://localhost:8766/api/metrics"
    },
    "lmstudio": {
        "display": "LMStudio",
        "port": 1234,
        "check": "port",
        "start_cmd": None,
        "note": "Start from LMStudio app"
    },
    "opencode": {
        "display": "OpenCode",
        "process": "opencode",
        "check": "process",
        "start_cmd": ["opencode"],
        "start_in_terminal": True
    }
}


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
    service_settings = _settings.get("services", {}).get(name, {})
    enabled = service_settings.get("enabled", True)
    status = {
        "name": name,
        "display": service.get("display", name),
        "running": False,
        "pid": None,
        "enabled": enabled,
        "can_start": service.get("start_cmd") is not None and enabled,
        "note": service.get("note") if enabled else "Disabled in settings"
    }

    if service.get("check") == "port":
        port = service.get("port")
        status["running"] = check_port(port)
        status["port"] = port
        # Also check process for PID
        if service.get("process"):
            status["pid"] = check_process(service.get("process"))
    elif service.get("check") == "process":
        pid = check_process(service.get("process", name))
        status["running"] = pid is not None
        status["pid"] = pid

    # Fetch metrics if available and running
    if status["running"]:
        if service.get("metrics_url"):
            try:
                import urllib.request
                with urllib.request.urlopen(service["metrics_url"], timeout=1) as resp:
                    status["metrics"] = json.loads(resp.read().decode())
            except Exception:
                pass
        
        # LMStudio metrics - fetch model info + proxy stats
        if name == "lmstudio":
            try:
                import urllib.request
                with urllib.request.urlopen("http://localhost:1234/v1/models", timeout=1) as resp:
                    models_data = json.loads(resp.read().decode())
                    models = models_data.get("data", [])
                    status["metrics"] = {
                        "models_loaded": len(models),
                        "total_requests": _lmstudio_stats["requests"],
                        "total_tokens": _lmstudio_stats["tokens"],
                        "models": [m["id"][:25] for m in models[:5]]
                    }
            except Exception:
                status["metrics"] = {
                    "models_loaded": 0,
                    "total_requests": _lmstudio_stats["requests"],
                    "total_tokens": _lmstudio_stats["tokens"],
                }
        
        # OpenCode metrics - process info
        if name == "opencode" and status["pid"]:
            try:
                proc = psutil.Process(status["pid"])
                mem = proc.memory_info()
                create_time = proc.create_time()
                uptime = time.time() - create_time
                status["metrics"] = {
                    "memory_mb": round(mem.rss / 1024 / 1024),
                    "uptime_seconds": round(uptime),
                    "cpu_percent": round(proc.cpu_percent(interval=0.1), 1)
                }
            except Exception:
                pass

    return status


def get_machine_info() -> dict:
    """Get machine hardware info (cached)"""
    if "machine" not in _cache:
        try:
            chip = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                capture_output=True, text=True).stdout.strip()
            
            ram_bytes = int(subprocess.run(['sysctl', '-n', 'hw.memsize'], 
                                          capture_output=True, text=True).stdout.strip())
            ram_gb = ram_bytes // (1024**3)
            
            macos = subprocess.run(['sw_vers', '-productVersion'], 
                                  capture_output=True, text=True).stdout.strip()
            
            cpu_cores = int(subprocess.run(['sysctl', '-n', 'hw.ncpu'], 
                                          capture_output=True, text=True).stdout.strip())
            
            perf_cores = int(subprocess.run(['sysctl', '-n', 'hw.perflevel0.logicalcpu'], 
                                           capture_output=True, text=True).stdout.strip() or 0)
            eff_cores = int(subprocess.run(['sysctl', '-n', 'hw.perflevel1.logicalcpu'], 
                                          capture_output=True, text=True).stdout.strip() or 0)
            
            gpu_cores = subprocess.run(['system_profiler', 'SPDisplaysDataType'], 
                                       capture_output=True, text=True).stdout
            gpu_core_count = ""
            for line in gpu_cores.split('\n'):
                if 'Total Number of Cores' in line:
                    gpu_core_count = line.split(':')[-1].strip()
                    break
            
            neural_cores = ""
            for line in gpu_cores.split('\n'):
                if 'Neural Engine' in line.lower():
                    neural_cores = "16-core"
                    break
            
            _cache["machine"] = {
                "chip": chip,
                "ram_gb": ram_gb,
                "macos": macos,
                "cpu_cores": cpu_cores,
                "perf_cores": perf_cores,
                "eff_cores": eff_cores,
                "gpu_cores": gpu_core_count,
                "neural_engine": neural_cores or "16-core",
            }
        except Exception:
            _cache["machine"] = {
                "chip": "Unknown",
                "ram_gb": 0,
                "macos": "Unknown",
                "cpu_cores": 0,
                "perf_cores": 0,
                "eff_cores": 0,
                "gpu_cores": "",
                "neural_engine": "",
            }
    return _cache["machine"]


def get_system_stats() -> dict:
    """Get system memory and disk stats"""
    mem = psutil.virtual_memory()

    if MODELS_DIR.exists():
        disk = psutil.disk_usage(str(MODELS_DIR))
    else:
        disk = psutil.disk_usage('/')

    # Use cached models_size
    _refresh_models_cache_if_needed()
    models_size = _cache["models"]["total_size"]

    return {
        "memory": {"total": mem.total, "used": mem.used, "percent": mem.percent},
        "disk": {"total": disk.total, "used": disk.used, "free": disk.free, "percent": disk.percent},
        "models_size": models_size,
        "machine": get_machine_info()
    }


def _refresh_models_cache_if_needed(force: bool = False):
    """Refresh models cache if TTL expired or forced"""
    now = time.time()
    if not force and now - _cache["models"]["timestamp"] < CACHE_TTL:
        return
    
    models = []
    total_size = 0
    if MODELS_DIR.exists():
        for org_dir in MODELS_DIR.iterdir():
            if org_dir.is_dir() and not org_dir.name.startswith('.'):
                for model_dir in org_dir.iterdir():
                    if model_dir.is_dir() and not model_dir.name.startswith('.'):
                        try:
                            size = sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file())
                            total_size += size
                            models.append({
                                "name": f"{org_dir.name}/{model_dir.name}",
                                "size": size,
                                "path": str(model_dir)
                            })
                        except (OSError, FileNotFoundError):
                            pass
    
    _cache["models"]["data"] = sorted(models, key=lambda x: x['size'], reverse=True)
    _cache["models"]["total_size"] = total_size
    _cache["models"]["timestamp"] = now


def invalidate_models_cache():
    """Force cache refresh on next request"""
    _cache["models"]["timestamp"] = 0


def get_downloaded_models() -> list:
    """List downloaded models (cached)"""
    _refresh_models_cache_if_needed()
    return _cache["models"]["data"]


# Track download speeds
DOWNLOAD_HISTORY: dict = {}  # {repo: [(time, size), ...]}


def detect_active_downloads() -> list:
    """Detect external downloads (not managed by DownloadManager)"""
    import time
    import json
    downloads = []

    if not MODELS_DIR.exists():
        return downloads

    # Get repos managed by download_manager to exclude them
    managed_repos = set()
    dm_status = download_manager.get_status()
    for t in dm_status.get("active", []):
        managed_repos.add(t["repo_id"])
    for t in dm_status.get("queue", []):
        managed_repos.add(t["repo_id"])

    for org_dir in MODELS_DIR.iterdir():
        if not org_dir.is_dir() or org_dir.name.startswith('.'):
            continue
        for model_dir in org_dir.iterdir():
            if not model_dir.is_dir() or model_dir.name.startswith('.'):
                continue
            if '_archived_' in model_dir.name:
                continue

            repo = f"{org_dir.name}/{model_dir.name}"
            
            # Skip if managed by DownloadManager
            if repo in managed_repos:
                continue

            # Only check models with .incomplete files (active HF downloads)
            incomplete_files = list(model_dir.rglob('*.incomplete'))
            if not incomplete_files:
                # Clean up history for completed downloads
                if repo in DOWNLOAD_HISTORY:
                    del DOWNLOAD_HISTORY[repo]
                continue

            try:
                # Calculate sizes - actual disk usage for speed, apparent for progress
                files = [f for f in model_dir.rglob('*') if f.is_file()]
                actual_size = sum(f.stat().st_blocks * 512 for f in files)  # Actual disk usage
                apparent_size = sum(f.stat().st_size for f in files)  # For progress calc
                current_size = actual_size  # Use actual for speed tracking
                current_time = time.time()

                # Initialize history
                if repo not in DOWNLOAD_HISTORY:
                    DOWNLOAD_HISTORY[repo] = []

                history = DOWNLOAD_HISTORY[repo]
                history.append((current_time, current_size))

                # Keep only last 15 seconds of history
                history = [(t, s) for t, s in history if current_time - t < 15]
                DOWNLOAD_HISTORY[repo] = history

                # Calculate speed - only if size changed
                speed = 0
                is_active = False
                if len(history) >= 2:
                    oldest = history[0]
                    time_diff = current_time - oldest[0]
                    size_diff = current_size - oldest[1]
                    if time_diff > 0 and size_diff > 0:
                        speed = size_diff / time_diff
                        is_active = True

                # Skip if not actively downloading (no size change in 15s)
                if not is_active and len(history) >= 3:
                    continue

                # Get expected total from index file
                total_size = 0
                index_file = model_dir / "model.safetensors.index.json"
                if index_file.exists():
                    with open(index_file) as f:
                        index = json.load(f)
                    weight_map = index.get("weight_map", {})
                    num_shards = len(set(weight_map.values()))
                    # Estimate ~4.5GB per shard for 4bit models
                    total_size = num_shards * 4.5 * 1024 * 1024 * 1024

                progress = 0
                if total_size > 0:
                    progress = min(99, int((apparent_size / total_size) * 100))

                downloads.append({
                    "repo": repo,
                    "current_size": apparent_size,  # Show apparent size to user
                    "total_size": total_size if total_size > 0 else None,
                    "progress": progress,
                    "speed": speed,
                    "path": str(model_dir)
                })
            except Exception:
                pass

    return downloads


# API Routes
@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the dashboard HTML"""
    html_path = DASHBOARD_DIR / "templates" / "index.html"
    return html_path.read_text()


@app.get("/api/status")
async def get_status():
    services = [get_service_status(name) for name in SERVICES]
    return {
        "services": services,
        "system": get_system_stats(),
        "models": get_downloaded_models(),
        "downloads": detect_active_downloads(),
    }


@app.get("/api/activity")
async def get_activity():
    return {"activity": list(_activity_log)}


@app.get("/api/settings")
async def get_settings():
    return _settings


@app.put("/api/settings")
async def update_settings(request: Request):
    global _settings
    try:
        new_settings = await request.json()
        _settings.update(new_settings)
        save_settings(_settings)
        return {"success": True, "settings": _settings}
    except Exception as e:
        return {"success": False, "message": str(e)}


@app.post("/api/service/{name}/start")
async def start_service(name: str):
    """Start a service in new Terminal window"""
    service = SERVICES.get(name)
    if not service:
        return {"success": False, "message": "Unknown service"}

    if not service.get("start_cmd"):
        return {"success": False, "message": service.get("note", "Cannot start this service")}

    try:
        if service.get("start_in_terminal"):
            cmd = " ".join(service["start_cmd"])
            script = f'''
            tell application "Terminal"
                activate
                do script "{cmd}"
            end tell
            '''
            subprocess.run(["osascript", "-e", script], check=True)
            return {"success": True, "message": f"Starting {name} in Terminal"}
        else:
            log_file = Path(f"/tmp/{name}.log")
            with open(log_file, "a") as f:
                subprocess.Popen(
                    service["start_cmd"],
                    stdout=f,
                    stderr=f,
                    start_new_session=True
                )
            return {"success": True, "message": f"Started {name} (log: {log_file})"}
    except Exception as e:
        return {"success": False, "message": str(e)}


@app.post("/api/service/{name}/stop")
async def stop_service(name: str):
    """Stop a service by killing all matching processes"""
    service = SERVICES.get(name)
    if not service:
        return {"success": False, "message": "Unknown service"}

    process_name = service.get("process", name)
    procs_to_kill = []

    # Collect matching processes
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info.get('cmdline') or []
            proc_name = proc.info.get('name', '')
            if process_name in proc_name or any(process_name in arg for arg in cmdline):
                procs_to_kill.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    if not procs_to_kill:
        return {"success": False, "message": "No matching process found"}

    # SIGTERM first (graceful)
    killed = []
    for proc in procs_to_kill:
        try:
            proc.terminate()
            killed.append(proc.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    # Wait up to 3s for graceful shutdown
    gone, alive = psutil.wait_procs(procs_to_kill, timeout=3)

    # SIGKILL stubborn processes
    force_killed = []
    for proc in alive:
        try:
            proc.kill()
            force_killed.append(proc.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    msg = f"Stopped PIDs: {killed}"
    if force_killed:
        msg += f" (force killed: {force_killed})"
    return {"success": True, "message": msg}


@app.post("/api/service/{name}/restart")
async def restart_service(name: str):
    """Restart a service"""
    stop_result = await stop_service(name)
    import asyncio
    await asyncio.sleep(2)
    start_result = await start_service(name)
    return {
        "success": start_result.get("success", False),
        "message": f"Stop: {stop_result.get('message')} | Start: {start_result.get('message')}"
    }


@app.post("/api/model/reveal")
async def reveal_model(path: str):
    """Reveal model in Finder"""
    model_path = Path(path)
    if model_path.exists():
        subprocess.run(["open", "-R", str(model_path)])
        return {"success": True}
    return {"success": False, "message": "Path not found"}


@app.delete("/api/model")
async def delete_model(path: str):
    """Delete a downloaded model"""
    import shutil
    model_path = Path(path)
    
    # Security: ensure path is under MODELS_DIR
    try:
        model_path.resolve().relative_to(MODELS_DIR.resolve())
    except ValueError:
        return {"success": False, "message": "Invalid path"}
    
    if not model_path.exists():
        return {"success": False, "message": "Model not found"}
    
    try:
        shutil.rmtree(model_path)
        invalidate_models_cache()
        return {"success": True, "message": f"Deleted {model_path.name}"}
    except Exception as e:
        return {"success": False, "message": str(e)}


# Download Management API
class DownloadRequest(BaseModel):
    repo_id: str


@app.get("/api/downloads")
async def get_downloads():
    """Get download queue status and presets"""
    return download_manager.get_status()


@app.post("/api/download/start")
async def start_download(req: DownloadRequest):
    """Add a model to download queue"""
    task = download_manager.add_download(req.repo_id)
    return {"success": True, "task": task.to_dict()}


@app.post("/api/download/pause")
async def pause_download(req: DownloadRequest):
    """Pause current download"""
    success = download_manager.pause_download(req.repo_id)
    return {"success": success}


@app.post("/api/download/resume")
async def resume_download(req: DownloadRequest):
    """Resume paused download"""
    success = download_manager.resume_download(req.repo_id)
    return {"success": success}


@app.post("/api/download/cancel")
async def cancel_download(req: DownloadRequest):
    """Cancel download and optionally delete files"""
    success = download_manager.remove_download(req.repo_id, delete_files=False)
    return {"success": success}


@app.post("/api/download/delete")
async def delete_download(req: DownloadRequest):
    """Cancel download and delete files"""
    success = download_manager.remove_download(req.repo_id, delete_files=True)
    return {"success": success}


class SearchRequest(BaseModel):
    query: str
    filter: str = "embedding"  # "embedding", "llm", "all"


@app.post("/api/search/huggingface")
async def search_huggingface(req: SearchRequest):
    """Search HuggingFace for models"""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        
        query = req.query
        
        # Search for models
        results = api.list_models(
            search=query,
            limit=20,
            sort="downloads",
            direction=-1
        )
        
        models = []
        for model in results:
            # Get model info
            try:
                info = api.model_info(model.id)
                size_bytes = sum(s.size for s in (info.siblings or []) if s.size)
            except Exception:
                size_bytes = 0
            
            models.append({
                "id": model.id,
                "name": model.id.split("/")[-1],
                "downloads": model.downloads or 0,
                "likes": model.likes or 0,
                "size_bytes": size_bytes,
                "tags": model.tags[:5] if model.tags else [],
            })
        
        return {"models": models}
    except Exception as e:
        return {"models": [], "error": str(e)}


# ============================================================================
# OpenAI-compatible Proxy Routes (/v1/*)
# Automatically routes embeddings to MLX (8766), others to LMStudio (1234)
# ============================================================================

@app.api_route("/v1/embeddings", methods=["GET", "POST"])
async def proxy_embeddings(request: Request):
    body = {}
    if request.method == "POST":
        try:
            body = await request.json()
        except Exception:
            pass
    
    start_time = time.time()
    response = await _proxy_request(request, PROXY_TARGETS["embeddings"])
    latency_ms = (time.time() - start_time) * 1000
    
    try:
        resp_data = json.loads(response.body)
        usage = resp_data.get("usage", {})
        tokens = usage.get("total_tokens", 0)
        dimensions = usage.get("dimensions", 0)
        model = body.get("model", resp_data.get("model", "unknown"))
        input_text = body.get("input", "")
        if isinstance(input_text, list):
            input_text = input_text[0] if input_text else ""
        preview = (input_text[:40] + "...") if len(input_text) > 40 else input_text
        
        _activity_log.append({
            "time": time.strftime("%H:%M:%S"),
            "type": "embed",
            "model": model.split("/")[-1][:25],
            "latency_ms": round(latency_ms, 1),
            "tokens": tokens,
            "dimensions": dimensions,
            "preview": preview,
        })
    except Exception:
        pass
    
    return response


@app.api_route("/v1/models", methods=["GET"])
async def proxy_models(request: Request):
    async with httpx.AsyncClient(timeout=10.0, trust_env=False) as client:
        results = {"object": "list", "data": []}
        
        # Get embedding models from MLX server
        try:
            r = await client.get(f"{PROXY_TARGETS['embeddings']}/v1/models")
            if r.status_code == 200:
                data = r.json().get("data", [])
                for m in data:
                    m["type"] = "embedding"
                results["data"].extend(data)
        except Exception:
            pass
        
        # Get chat models from LMStudio (filter out embedding models)
        try:
            r = await client.get(f"{PROXY_TARGETS['lmstudio']}/v1/models")
            if r.status_code == 200:
                data = r.json().get("data", [])
                for m in data:
                    model_id = m.get("id", "").lower()
                    # Skip embedding models - they're served by MLX
                    if any(p in model_id for p in EMBEDDING_MODELS):
                        continue
                    m["type"] = "chat"
                    results["data"].append(m)
        except Exception:
            pass
        
        return results


@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_v1(request: Request, path: str):
    body = {}
    if request.method in ["POST", "PUT"]:
        try:
            body = await request.json()
        except Exception:
            pass
    
    start_time = time.time()
    is_embedding = _is_embedding_request(path, body)
    
    if is_embedding:
        target = PROXY_TARGETS["embeddings"]
    else:
        target = PROXY_TARGETS["lmstudio"]
    
    response = await _proxy_request(request, target)
    latency_ms = (time.time() - start_time) * 1000
    
    # Log activity
    try:
        resp_data = json.loads(response.body)
        tokens = resp_data.get("usage", {}).get("total_tokens", 0)
        model = body.get("model", resp_data.get("model", "unknown"))
        
        if is_embedding:
            input_text = body.get("input", "")
            if isinstance(input_text, list):
                input_text = input_text[0] if input_text else ""
            preview = (input_text[:40] + "...") if len(input_text) > 40 else input_text
            req_type = "embed"
        else:
            _lmstudio_stats["requests"] += 1
            _lmstudio_stats["tokens"] += tokens
            messages = body.get("messages", [])
            last_msg = messages[-1]["content"] if messages else ""
            preview = (last_msg[:40] + "...") if len(last_msg) > 40 else last_msg
            req_type = "chat"
        
        _activity_log.append({
            "time": time.strftime("%H:%M:%S"),
            "type": req_type,
            "model": model.split("/")[-1][:25],
            "latency_ms": round(latency_ms, 1),
            "tokens": tokens,
            "preview": preview,
        })
    except Exception:
        pass
    
    return response


if __name__ == "__main__":
    import uvicorn
    print("🍎 SiliconLM starting at http://localhost:8765")
    uvicorn.run(app, host="0.0.0.0", port=8765)
