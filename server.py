#!/usr/bin/env python3
"""SiliconLM - Apple Silicon LLM Dashboard"""

import json
import os
import socket
import subprocess
import shutil
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional
import asyncio
import httpx
import psutil
from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from download_manager import download_manager


@asynccontextmanager
async def lifespan(app):
    download_manager.start()
    asyncio.create_task(_autostart_services())
    yield
    download_manager.stop()


async def _autostart_services():
    """Auto-start dashboard-managed services that were explicitly started before last shutdown."""
    await asyncio.sleep(1)
    started = _load_started_services()
    for name in ("opencode",):
        if name in started:
            try:
                await start_service(name)
            except Exception:
                pass


app = FastAPI(title="SiliconLM", lifespan=lifespan)

SETTINGS_FILE = Path(__file__).parent / "settings.json"
DEFAULT_SERVICES: dict[str, dict[str, Any]] = {
    "ollama": {"enabled": True, "port": 11434, "readonly": True},
    "opencode": {"enabled": True},
}
DEFAULT_PROXY: dict[str, Any] = {"enabled": False, "host": "127.0.0.1", "port": 7890}
DEFAULT_SETTINGS: dict[str, Any] = {
    "chat_backend": "none",
    "models_dir": "~/Models",
    "services": DEFAULT_SERVICES,
    "proxy": DEFAULT_PROXY,
}


def load_settings() -> dict[str, Any]:
    settings: dict[str, Any] = {
        "chat_backend": DEFAULT_SETTINGS["chat_backend"],
        "models_dir": DEFAULT_SETTINGS["models_dir"],
        "services": {name: cfg.copy() for name, cfg in DEFAULT_SERVICES.items()},
        "proxy": DEFAULT_PROXY.copy(),
    }

    if SETTINGS_FILE.exists():
        with open(SETTINGS_FILE) as f:
            loaded = json.load(f)
        settings.update({k: v for k, v in loaded.items() if k not in {"services", "embedding"}})
        settings["chat_backend"] = loaded.get("chat_backend", settings["chat_backend"])
        settings["models_dir"] = loaded.get("models_dir", settings["models_dir"])
        loaded_services = loaded.get("services", {})
        settings["services"] = {
            "ollama": {
                **DEFAULT_SERVICES["ollama"],
                **loaded_services.get("ollama", {}),
            },
            "opencode": {
                **DEFAULT_SERVICES["opencode"],
                **loaded_services.get("opencode", {}),
            },
        }
        settings["proxy"] = {**DEFAULT_PROXY, **loaded.get("proxy", {})}
    return settings


def _resolve_opencode_agents(data: dict) -> dict:
    agents = data.get("agents", {})
    categories = data.get("categories", {})
    resolved = {}

    for name, agent in agents.items():
        if not isinstance(agent, dict):
            resolved[name] = agent
            continue

        agent_data = dict(agent)
        category_name = agent_data.get("category")
        category = categories.get(category_name) if category_name else None

        if isinstance(category, dict):
            agent_data.setdefault("model", category.get("model"))
            agent_data.setdefault("variant", category.get("variant"))

        resolved[name] = agent_data

    return resolved


def get_opencode_profiles():
    config_dir = Path.home() / ".config" / "opencode"
    active_config = config_dir / "oh-my-openagent.json"

    profiles = [
        {
            "id": "opus",
            "file": "oh-my-openagent-opus.json",
            "name": "Opus · current mixed routing",
        },
        {
            "id": "gpt",
            "file": "oh-my-openagent-gpt.json",
            "name": "GPT · GPT-only",
        },
        {
            "id": "qwen",
            "file": "oh-my-openagent-qwen.json",
            "name": "Qwen3-Max + GLM-5",
        },
    ]

    active_id = "custom"
    if not active_config.exists():
        active_id = "none"
    else:
        try:
            active_data = json.loads(active_config.read_text())
            for p in profiles:
                p_file = config_dir / p["file"]
                if p_file.exists():
                    p_data = json.loads(p_file.read_text())
                    if active_data == p_data:
                        active_id = p["id"]
                        break
        except Exception:
            pass

    results = []
    for p in profiles:
        p_file = config_dir / p["file"]
        agents = {}
        if p_file.exists():
            try:
                data = json.loads(p_file.read_text())
                agents = _resolve_opencode_agents(data)
            except Exception:
                pass
        results.append(
            {
                "id": p["id"],
                "name": p["name"],
                "agents": agents,
                "isActive": active_id == p["id"],
            }
        )

    return {"active": active_id, "profiles": results}


def switch_opencode_profile(profile_id: str):
    config_dir = Path.home() / ".config" / "opencode"
    active_config = config_dir / "oh-my-openagent.json"

    profiles = {
        "opus": "oh-my-openagent-opus.json",
        "gpt": "oh-my-openagent-gpt.json",
        "qwen": "oh-my-openagent-qwen.json",
        "sub2api": "oh-my-openagent-opus.json",
        "relay": "oh-my-openagent-opus.json",
        "me": "oh-my-openagent-opus.json",
        "openai": "oh-my-openagent-gpt.json",
        "oai": "oh-my-openagent-gpt.json",
    }

    if profile_id not in profiles:
        return False, "Profile not found"

    target_file = config_dir / profiles[profile_id]
    if not target_file.exists():
        return False, f"Config file not found: {target_file.name}"

    try:
        shutil.copy2(target_file, active_config)
        return True, f"Switched to {profile_id}"
    except Exception as e:
        return False, str(e)


def save_settings(settings):
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f, indent=2)


_STOPPED_SERVICES_FILE = (
    Path.home() / ".local" / "share" / "siliconlm" / "stopped_services.json"
)
_STARTED_SERVICES_FILE = (
    Path.home() / ".local" / "share" / "siliconlm" / "started_services.json"
)


def _load_stopped_services() -> set:
    try:
        if _STOPPED_SERVICES_FILE.exists():
            return set(json.loads(_STOPPED_SERVICES_FILE.read_text()))
    except Exception:
        pass
    return set()


def _save_stopped_services(stopped: set):
    _STOPPED_SERVICES_FILE.parent.mkdir(parents=True, exist_ok=True)
    _STOPPED_SERVICES_FILE.write_text(json.dumps(list(stopped)))


def _load_started_services() -> set:
    try:
        if _STARTED_SERVICES_FILE.exists():
            return set(json.loads(_STARTED_SERVICES_FILE.read_text()))
    except Exception:
        pass
    return set()


def _save_started_services(started: set):
    _STARTED_SERVICES_FILE.parent.mkdir(parents=True, exist_ok=True)
    _STARTED_SERVICES_FILE.write_text(json.dumps(list(started)))


def _get_chat_backend() -> str:
    """Return active chat backend name. SiliconLM is dashboard-only."""
    return _settings.get("chat_backend", "none")


def _get_chat_backend_port() -> Optional[int]:
    """Dashboard mode has no managed chat backend port."""
    return None


def _get_proxy_targets():
    """SiliconLM no longer proxies inference requests."""
    return {}


_settings = load_settings()


async def _proxy_request(request: Request, target_url: str) -> Response:
    try:
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
    except (httpx.ConnectError, httpx.ConnectTimeout):
        return Response(
            content=json.dumps(
                {
                    "error": {
                        "message": f"Backend unavailable at {target_url}",
                        "type": "proxy_error",
                    }
                }
            ),
            status_code=502,
            headers={"content-type": "application/json"},
        )
    except httpx.TimeoutException:
        return Response(
            content=json.dumps(
                {
                    "error": {
                        "message": f"Backend timeout ({target_url})",
                        "type": "proxy_error",
                    }
                }
            ),
            status_code=504,
            headers={"content-type": "application/json"},
        )


# Cache for expensive computations
_cache = {
    "models": {"data": [], "total_size": 0, "timestamp": 0},
}
CACHE_TTL = 30

# Chat request tracking
_chat_stats = {
    "requests": 0,
    "tokens": 0,
    "start_time": time.time(),
}

# Combined API activity log
from collections import deque

_activity_log = deque(maxlen=50)


# Static files
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# Configuration
def get_models_dir():
    models_path = _settings.get("models_dir", "~/Models")
    return Path(models_path).expanduser()


MODELS_DIR = get_models_dir()
DASHBOARD_DIR = Path(__file__).parent

# Service definitions
SERVICES = {
    "ollama": {
        "display": "Ollama",
        "port": 11434,
        "check": "ollama_api",
        "readonly": True,
        "note": "Read-only status from local Ollama API",
    },
    "opencode": {
        "display": "OpenCode",
        "port": 4096,
        "check": "port",
        "process": "opencode serve",
        "plist": str(
            Path.home() / "Library" / "LaunchAgents" / "ai.opencode.server.plist"
        ),
    },
}


def check_port(port: int) -> bool:
    """Check if a port is in use"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def check_process(name: str) -> Optional[int]:
    """Check if a process is running, return PID or None"""
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmdline = proc.info.get("cmdline") or []
            if name == "opencode serve":
                has_opencode = any("opencode" in arg for arg in cmdline)
                has_serve = any(arg == "serve" for arg in cmdline)
                if has_opencode and has_serve:
                    return proc.info["pid"]

            if name in proc.info.get("name", "") or any(name in arg for arg in cmdline):
                return proc.info["pid"]
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return None


def _ollama_status(port: int = 11434) -> dict:
    """Check Ollama status using the read-only tags API."""
    try:
        import urllib.request

        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/api/tags", timeout=2
        ) as resp:
            data = json.loads(resp.read().decode())
        models = data.get("models", [])
        return {
            "running": True,
            "port": port,
            "models": models,
            "metrics": {"models_available": len(models)},
        }
    except Exception:
        return {"running": False, "port": port, "models": [], "metrics": {"models_available": 0}}


def get_service_status(name: str) -> dict:
    service = SERVICES.get(name, {})
    service_settings = _settings.get("services", {}).get(name, {})
    enabled = service_settings.get("enabled", True)
    status = {
        "name": name,
        "display": service.get("display", name),
        "running": False,
        "pid": None,
        "enabled": enabled,
        "can_start": enabled and not service.get("readonly", False),
        "readonly": service.get("readonly", False),
        "note": service.get("note") if enabled else "Disabled in settings",
    }

    if service.get("check") == "ollama_api":
        port = service_settings.get("port", service.get("port", 11434))
        ollama = _ollama_status(port)
        status["running"] = ollama["running"]
        status["port"] = ollama["port"]
        status["metrics"] = ollama["metrics"]
        status["models"] = ollama["models"]
        if status["running"]:
            status["note"] = "Ollama API reachable"
    elif service.get("check") == "port":
        port = service.get("port")
        status["running"] = check_port(port)
        status["port"] = port
        if service.get("process"):
            status["pid"] = check_process(service.get("process"))
    elif service.get("check") == "process":
        pid = check_process(service.get("process", name))
        status["running"] = pid is not None
        status["pid"] = pid

    if status["running"]:
        if service.get("metrics_url"):
            try:
                import urllib.request

                with urllib.request.urlopen(service["metrics_url"], timeout=1) as resp:
                    status["metrics"] = json.loads(resp.read().decode())
            except Exception:
                pass

        if name == "opencode" and status["pid"]:
            try:
                proc = psutil.Process(status["pid"])
                mem = proc.memory_info()
                create_time = proc.create_time()
                uptime = time.time() - create_time
                status["metrics"] = {
                    "memory_mb": round(mem.rss / 1024 / 1024),
                    "uptime_seconds": round(uptime),
                    "cpu_percent": round(proc.cpu_percent(interval=0.1), 1),
                }
            except Exception:
                pass

    return status


def get_machine_info() -> dict:
    """Get machine hardware info (cached)"""
    if "machine" not in _cache:
        try:
            chip = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
            ).stdout.strip()

            ram_bytes = int(
                subprocess.run(
                    ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True
                ).stdout.strip()
            )
            ram_gb = ram_bytes // (1024**3)

            macos = subprocess.run(
                ["sw_vers", "-productVersion"], capture_output=True, text=True
            ).stdout.strip()

            cpu_cores = int(
                subprocess.run(
                    ["sysctl", "-n", "hw.ncpu"], capture_output=True, text=True
                ).stdout.strip()
            )

            perf_cores = int(
                subprocess.run(
                    ["sysctl", "-n", "hw.perflevel0.logicalcpu"],
                    capture_output=True,
                    text=True,
                ).stdout.strip()
                or 0
            )
            eff_cores = int(
                subprocess.run(
                    ["sysctl", "-n", "hw.perflevel1.logicalcpu"],
                    capture_output=True,
                    text=True,
                ).stdout.strip()
                or 0
            )

            gpu_cores = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"],
                capture_output=True,
                text=True,
            ).stdout
            gpu_core_count = ""
            for line in gpu_cores.split("\n"):
                if "Total Number of Cores" in line:
                    gpu_core_count = line.split(":")[-1].strip()
                    break

            neural_cores = ""
            for line in gpu_cores.split("\n"):
                if "Neural Engine" in line.lower():
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
        disk = psutil.disk_usage("/")

    # Use cached models_size
    _refresh_models_cache_if_needed()
    models_size = _cache["models"]["total_size"]

    return {
        "memory": {"total": mem.total, "used": mem.used, "percent": mem.percent},
        "disk": {
            "total": disk.total,
            "used": disk.used,
            "free": disk.free,
            "percent": disk.percent,
        },
        "models_size": models_size,
        "machine": get_machine_info(),
    }


async def _fetch_latest_github_release(repo: str) -> Optional[str]:
    """Fetch latest release tag from GitHub."""
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"https://api.github.com/repos/{repo}/releases/latest")
            if r.status_code == 200:
                return r.json().get("tag_name", "").lstrip("v")
    except Exception:
        pass
    return None


async def _fetch_latest_pypi_version(package: str) -> Optional[str]:
    """Fetch latest version from PyPI."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"https://pypi.org/pypi/{package}/json")
            if r.status_code == 200:
                return r.json().get("info", {}).get("version")
    except Exception:
        pass
    return None


async def _check_opencode() -> dict:
    """Check OpenCode installation status."""
    name = "opencode"
    category = "opencode"
    install_cmd = "brew install opencode"
    update_cmd = "brew upgrade opencode"
    try:
        # Check common install locations since launchd PATH is minimal
        opencode_path = shutil.which("opencode")
        if not opencode_path:
            for p in [
                Path.home() / ".local" / "bin" / "opencode",
                Path("/opt/homebrew/bin/opencode"),
            ]:
                if p.exists():
                    opencode_path = str(p)
                    break
        if not opencode_path:
            return {
                "name": name,
                "category": category,
                "installed": False,
                "version": None,
                "latest": None,
                "status": "missing",
                "install_cmd": install_cmd,
                "update_cmd": update_cmd,
            }
        version_result = await asyncio.to_thread(
            subprocess.run,
            [opencode_path, "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        version = (
            version_result.stdout.strip().split()[-1]
            if version_result.returncode == 0
            else None
        )
        # Use brew info to get latest version (opencode is in homebrew-core)
        latest = None
        try:
            brew_result = await asyncio.to_thread(
                subprocess.run,
                ["/opt/homebrew/bin/brew", "info", "--json=v2", "opencode"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if brew_result.returncode == 0:
                brew_data = json.loads(brew_result.stdout)
                formulae = brew_data.get("formulae", [])
                if formulae:
                    latest = formulae[0].get("versions", {}).get("stable")
        except Exception:
            pass
        status = "current"
        if version and latest:
            try:
                from packaging.version import Version

                if Version(version) < Version(latest):
                    status = "outdated"
            except Exception:
                # Fallback to string comparison
                if version != latest:
                    status = "outdated"
        elif not latest:
            status = "current"  # Can't determine latest, assume current
        return {
            "name": name,
            "category": category,
            "installed": True,
            "version": version,
            "latest": latest,
            "status": status,
            "install_cmd": install_cmd,
            "update_cmd": update_cmd,
        }
    except Exception:
        return {
            "name": name,
            "category": category,
            "installed": False,
            "version": None,
            "latest": None,
            "status": "unknown",
            "install_cmd": install_cmd,
            "update_cmd": update_cmd,
        }


async def _check_uv_tools() -> list[dict]:
    """Check MLX tools (mlx, mlx-lm, mlx-embeddings) in external venv."""
    tools = [
        {"name": "mlx", "package": "mlx"},
        {"name": "mlx-lm", "package": "mlx-lm"},
        {"name": "mlx-embeddings", "package": "mlx-embeddings"},
    ]
    category = "mlx_tools"
    install_tpl = "cd ~/.local/share/siliconlm && source venv/bin/activate && pip install <package>"
    update_tpl = "cd ~/.local/share/siliconlm && source venv/bin/activate && pip install --upgrade <package>"
    venv_pip = str(
        Path.home() / ".local" / "share" / "siliconlm" / "venv" / "bin" / "pip"
    )

    async def _check_one(tool: dict) -> dict:
        install_cmd = install_tpl.replace("<package>", tool["package"])
        update_cmd = update_tpl.replace("<package>", tool["package"])
        try:
            pip_result = await asyncio.to_thread(
                subprocess.run,
                [venv_pip, "show", tool["package"]],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if pip_result.returncode != 0:
                latest = await _fetch_latest_pypi_version(tool["package"])
                return {
                    "name": tool["name"],
                    "category": category,
                    "installed": False,
                    "version": None,
                    "latest": latest,
                    "status": "missing",
                    "install_cmd": install_cmd,
                    "update_cmd": update_cmd,
                }
            version = None
            for line in pip_result.stdout.split("\n"):
                if line.startswith("Version:"):
                    version = line.split(":", 1)[1].strip()
                    break
            latest = await _fetch_latest_pypi_version(tool["package"])
            status = "current"
            if version and latest:
                try:
                    from packaging.version import Version

                    if Version(version) < Version(latest):
                        status = "outdated"
                except Exception:
                    if version != latest:
                        status = "outdated"
            else:
                status = "unknown"
            return {
                "name": tool["name"],
                "category": category,
                "installed": True,
                "version": version,
                "latest": latest,
                "status": status,
                "install_cmd": install_cmd,
                "update_cmd": update_cmd,
            }
        except Exception:
            latest = await _fetch_latest_pypi_version(tool["package"])
            return {
                "name": tool["name"],
                "category": category,
                "installed": False,
                "version": None,
                "latest": latest,
                "status": "unknown",
                "install_cmd": install_cmd,
                "update_cmd": update_cmd,
            }

    return list(await asyncio.gather(*[_check_one(t) for t in tools]))


async def _check_brew_packages() -> list[dict]:
    """Check Homebrew packages (python3)."""
    packages = [{"name": "python3", "brew_name": "python3"}]
    category = "homebrew"
    install_cmd = "brew install <package>"
    update_cmd = "brew upgrade <package>"
    results = []
    for pkg in packages:
        try:
            brew_result = await asyncio.to_thread(
                subprocess.run,
                ["brew", "list", "--versions", pkg["brew_name"]],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if brew_result.returncode != 0:
                results.append(
                    {
                        "name": pkg["name"],
                        "category": category,
                        "installed": False,
                        "version": None,
                        "latest": None,
                        "status": "missing",
                        "install_cmd": install_cmd.replace(
                            "<package>", pkg["brew_name"]
                        ),
                        "update_cmd": update_cmd.replace("<package>", pkg["brew_name"]),
                    }
                )
                continue
            version = (
                brew_result.stdout.strip().split()[-1]
                if brew_result.stdout.strip()
                else None
            )
            results.append(
                {
                    "name": pkg["name"],
                    "category": category,
                    "installed": True,
                    "version": version,
                    "latest": None,
                    "status": "current",
                    "install_cmd": install_cmd.replace("<package>", pkg["brew_name"]),
                    "update_cmd": update_cmd.replace("<package>", pkg["brew_name"]),
                }
            )
        except Exception:
            results.append(
                {
                    "name": pkg["name"],
                    "category": category,
                    "installed": False,
                    "version": None,
                    "latest": None,
                    "status": "unknown",
                    "install_cmd": install_cmd.replace("<package>", pkg["brew_name"]),
                    "update_cmd": update_cmd.replace("<package>", pkg["brew_name"]),
                }
            )
    return results




def _version_status(version: Optional[str], latest: Optional[str]) -> str:
    if not version:
        return "missing"
    if not latest:
        return "current"
    try:
        from packaging.version import Version
        return "outdated" if Version(version) < Version(latest) else "current"
    except Exception:
        return "outdated" if version != latest else "current"


def _tool_result(name: str, category: str, version=None, latest=None, status=None, update_cmd=None, install_cmd=None, notes=None, updatable=True):
    return {
        "name": name,
        "id": name,
        "category": category,
        "installed": bool(version),
        "version": version,
        "latest": latest,
        "status": status or _version_status(version, latest),
        "update_cmd": update_cmd,
        "install_cmd": install_cmd,
        "notes": notes,
        "updatable": updatable and bool(update_cmd),
    }


async def _run_capture(cmd: list[str], timeout: int = 20, cwd: Optional[str] = None) -> tuple[int, str, str]:
    result = await asyncio.to_thread(
        subprocess.run,
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result.returncode, result.stdout.strip(), result.stderr.strip()


async def _fetch_latest_npm_version(package: str) -> Optional[str]:
    try:
        code, stdout, _ = await _run_capture(["/opt/homebrew/bin/npm", "view", package, "version"], timeout=20)
        return stdout.splitlines()[-1].strip() if code == 0 and stdout else None
    except Exception:
        return None


async def _check_opencode() -> dict:
    opencode_bin = Path.home() / ".opencode" / "bin" / "opencode"
    version = None
    if opencode_bin.exists():
        try:
            code, stdout, _ = await _run_capture([str(opencode_bin), "--version"], timeout=10)
            if code == 0 and stdout:
                version = stdout.split()[-1]
        except Exception:
            pass
    latest = await _fetch_latest_github_release("sst/opencode")
    return _tool_result(
        "opencode",
        "opencode",
        version,
        latest,
        update_cmd=f"{opencode_bin} upgrade",
        install_cmd="curl -fsSL https://opencode.ai/install | bash",
    )


def _read_package_version(path: Path) -> Optional[str]:
    try:
        return json.loads(path.read_text()).get("version")
    except Exception:
        return None


async def _check_oh_my_openagent() -> dict:
    cache_version = _read_package_version(Path.home() / ".cache" / "opencode" / "packages" / "oh-my-openagent@latest" / "node_modules" / "oh-my-openagent" / "package.json")
    lock_version = _read_package_version(Path.home() / ".local" / "share" / "opencode" / "node_modules" / "oh-my-openagent" / "package.json")
    latest = await _fetch_latest_npm_version("oh-my-openagent")
    cache_drift = bool(cache_version and lock_version and cache_version != lock_version)
    status = "cache-drift" if cache_drift else _version_status(lock_version or cache_version, latest)
    notes = f"loaded={cache_version or 'unknown'} lockfile={lock_version or 'unknown'}"
    return _tool_result(
        "oh-my-openagent",
        "opencode-plugin",
        lock_version or cache_version,
        latest,
        status=status,
        update_cmd="cd ~/.local/share/opencode && bun update oh-my-openagent && rm -rf ~/.cache/opencode/packages/oh-my-openagent@latest && ~/.bin/oc-restart",
        notes=notes,
    )


async def _check_ollama() -> dict:
    port = _settings.get("services", {}).get("ollama", {}).get("port", 11434)
    status = _ollama_status(port)
    notes = f"127.0.0.1:{port}"
    if status["running"]:
        notes = f"{notes} ({status['metrics']['models_available']} tags)"
    return _tool_result(
        "ollama",
        "backend",
        "reachable" if status["running"] else None,
        None,
        status="current" if status["running"] else "missing",
        update_cmd=None,
        install_cmd="brew install ollama",
        notes=notes,
        updatable=False,
    )


async def _check_sub2api() -> dict:
    provider_url = "http://123.57.81.93:8081/v1"
    try:
        config = json.loads((Path.home() / ".config" / "opencode" / "opencode.json").read_text())
        provider = config.get("provider", {}).get("sub2api", {})
        provider_url = provider.get("options", {}).get("baseURL") or provider.get("baseURL") or provider_url
    except Exception:
        pass
    health_url = provider_url.rsplit("/v1", 1)[0]
    status = "unknown"
    notes = provider_url
    try:
        async with httpx.AsyncClient(timeout=5.0, trust_env=False) as client:
            response = await client.get(health_url)
            status = "current" if response.status_code < 500 else "unknown"
            notes = f"{provider_url} ({response.status_code})"
    except Exception as e:
        notes = f"{provider_url} ({e})"
    return _tool_result("sub2api", "proxy", "reachable" if status == "current" else None, None, status=status, update_cmd=None, notes=notes, updatable=False)


async def _check_uv_tools() -> list[dict]:
    tools = []
    try:
        code, stdout, _ = await _run_capture(["/opt/homebrew/bin/uv", "tool", "list"], timeout=30)
        if code != 0:
            return []
        for line in stdout.splitlines():
            if " v" not in line or line.startswith("-"):
                continue
            name, version = line.split(" v", 1)
            name = name.strip()
            version = version.split()[0].strip()
            latest = await _fetch_latest_pypi_version(name)
            tools.append(_tool_result(name, "uv-tool", version, latest, update_cmd=f"uv tool upgrade {name}"))
    except Exception:
        return []
    return tools


async def _check_global_npm() -> list[dict]:
    package_names = ["@larksuite/cli", "@mermaid-js/mermaid-cli", "@anthropic-ai/claude-code"]
    installed = {}
    try:
        code, stdout, _ = await _run_capture(["/opt/homebrew/bin/npm", "ls", "-g", "--depth=0", "--json"], timeout=30)
        if code == 0 and stdout:
            deps = json.loads(stdout).get("dependencies", {})
            installed = {name: data.get("version") for name, data in deps.items()}
    except Exception:
        pass
    results = []
    for name in package_names:
        version = installed.get(name)
        latest = await _fetch_latest_npm_version(name)
        results.append(_tool_result(name, "global-npm", version, latest, update_cmd=f"npm update -g {name}", install_cmd=f"npm install -g {name}"))
    return results
def _refresh_models_cache_if_needed(force: bool = False):
    """Refresh models cache if TTL expired or forced"""
    now = time.time()
    if not force and now - _cache["models"]["timestamp"] < CACHE_TTL:
        return

    models = []
    total_size = 0
    if MODELS_DIR.exists():
        for org_dir in MODELS_DIR.iterdir():
            if org_dir.is_dir() and not org_dir.name.startswith("."):
                for model_dir in org_dir.iterdir():
                    if model_dir.is_dir() and not model_dir.name.startswith("."):
                        try:
                            size = sum(
                                f.stat().st_size
                                for f in model_dir.rglob("*")
                                if f.is_file()
                            )
                            total_size += size
                            models.append(
                                {
                                    "name": f"{org_dir.name}/{model_dir.name}",
                                    "size": size,
                                    "path": str(model_dir),
                                }
                            )
                        except (OSError, FileNotFoundError):
                            pass

    _cache["models"]["data"] = sorted(models, key=lambda x: x["size"], reverse=True)
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
        if not org_dir.is_dir() or org_dir.name.startswith("."):
            continue
        for model_dir in org_dir.iterdir():
            if not model_dir.is_dir() or model_dir.name.startswith("."):
                continue
            if "_archived_" in model_dir.name:
                continue

            repo = f"{org_dir.name}/{model_dir.name}"

            # Skip if managed by DownloadManager
            if repo in managed_repos:
                continue

            # Only check models with .incomplete files (active HF downloads)
            incomplete_files = list(model_dir.rglob("*.incomplete"))
            if not incomplete_files:
                # Clean up history for completed downloads
                if repo in DOWNLOAD_HISTORY:
                    del DOWNLOAD_HISTORY[repo]
                continue

            try:
                # Calculate sizes - actual disk usage for speed, apparent for progress
                files = [f for f in model_dir.rglob("*") if f.is_file()]
                actual_size = sum(
                    f.stat().st_blocks * 512 for f in files
                )  # Actual disk usage
                apparent_size = sum(
                    f.stat().st_size for f in files
                )  # For progress calc
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

                downloads.append(
                    {
                        "repo": repo,
                        "current_size": apparent_size,  # Show apparent size to user
                        "total_size": total_size if total_size > 0 else None,
                        "progress": progress,
                        "speed": speed,
                        "path": str(model_dir),
                    }
                )
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


@app.get("/api/settings/chat-backend")
async def get_chat_backend_api():
    return {"backend": _get_chat_backend()}


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


@app.get("/api/opencode/profiles")
async def api_get_opencode_profiles():
    return get_opencode_profiles()


class ProfileSwitchRequest(BaseModel):
    profile_id: str


@app.post("/api/opencode/profile/switch")
async def api_switch_opencode_profile(req: ProfileSwitchRequest):
    success, message = switch_opencode_profile(req.profile_id)
    return {"success": success, "message": message}


@app.get("/api/cli-agents")
async def get_cli_agents():
    """Return status of update-skill-aligned CLI tools."""
    checks = await asyncio.gather(
        _check_opencode(),
        _check_oh_my_openagent(),
        _check_ollama(),
        _check_sub2api(),
        _check_uv_tools(),
        _check_global_npm(),
    )
    agents = []
    for item in checks:
        if isinstance(item, list):
            agents.extend(item)
        else:
            agents.append(item)
    return {"agents": agents}


async def _run_shell_command(cmd: str, timeout: int = 60) -> tuple[bool, str]:
    """Run a shell command with timeout. Returns (success, message)."""
    try:
        result = await asyncio.to_thread(
            subprocess.run,
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            return True, result.stdout.strip() or "Command completed successfully"
        else:
            return (
                False,
                result.stderr.strip() or result.stdout.strip() or "Command failed",
            )
    except subprocess.TimeoutExpired:
        return False, f"Command timed out after {timeout} seconds"
    except Exception as e:
        return False, str(e)


@app.post("/api/cli-agents/install")
async def install_cli_agents(request: Request):
    """Install missing CLI tools."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    requested_tools = body.get("tool", body.get("tools", "all"))
    agents = (await get_cli_agents())["agents"]
    target_tools = None if requested_tools == "all" else (requested_tools if isinstance(requested_tools, list) else [requested_tools])
    target_agents = agents if target_tools is None else [a for a in agents if a["id"] in target_tools or a["name"] in target_tools]
    results = []
    for agent in target_agents:
        if agent.get("installed"):
            results.append({"name": agent["name"], "success": True, "message": f"{agent['name']} is already installed"})
            continue
        install_cmd = agent.get("install_cmd")
        if not install_cmd:
            results.append({"name": agent["name"], "success": False, "message": "No install command"})
            continue
        success, message = await _run_shell_command(install_cmd, timeout=120)
        results.append({"name": agent["name"], "success": success, "message": message})
    return {"results": results}


@app.post("/api/cli-agents/update")
async def update_cli_agents(request: Request):
    """Update selected CLI tools."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    requested_tools = body.get("tool", body.get("tools", "all"))
    agents = (await get_cli_agents())["agents"]
    target_tools = None if requested_tools == "all" else (requested_tools if isinstance(requested_tools, list) else [requested_tools])
    target_agents = agents if target_tools is None else [a for a in agents if a["id"] in target_tools or a["name"] in target_tools]
    results = []
    for agent in target_agents:
        update_cmd = agent.get("update_cmd")
        if not update_cmd:
            results.append({"name": agent["name"], "success": False, "message": "No update command"})
            continue
        success, message = await _run_shell_command(update_cmd, timeout=300)
        results.append({"name": agent["name"], "success": success, "message": message})
    return {"results": results}


@app.post("/api/cli-agents/run-update-all")
async def run_update_all():
    log_dir = Path.home() / "Library" / "Logs" / "dev.opencode.autoupdate"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "siliconlm-update-all.log"
    with open(log_file, "a") as f:
        subprocess.Popen([str(Path.home() / ".local" / "bin" / "update-all")], stdout=f, stderr=f, start_new_session=True)
    return {"success": True, "log_path": str(log_file)}


@app.get("/api/cli-agents/run-update-all/log")
async def get_update_all_log():
    log_file = Path.home() / "Library" / "Logs" / "dev.opencode.autoupdate" / "siliconlm-update-all.log"
    if not log_file.exists():
        return {"log": ""}
    lines = log_file.read_text(errors="ignore").splitlines()[-120:]
    return {"log": "\n".join(lines), "log_path": str(log_file)}


@app.post("/api/service/{name}/start")
async def start_service(name: str):
    service = SERVICES.get(name)
    if not service:
        return {"success": False, "message": "Unknown service"}

    if service.get("readonly"):
        return {"success": False, "message": service.get("note", "Read-only service")}

    # Track that this service was explicitly started
    started = _load_started_services()
    started.add(name)
    _save_started_services(started)
    # Remove from stopped services when starting
    stopped = _load_stopped_services()
    stopped.discard(name)
    _save_stopped_services(stopped)

    if name == "opencode":
        return _opencode_start()

    if not service.get("start_cmd"):
        return {
            "success": False,
            "message": service.get("note", "Cannot start this service"),
        }

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
                    service["start_cmd"], stdout=f, stderr=f, start_new_session=True
                )
            return {"success": True, "message": f"Started {name} (log: {log_file})"}
    except Exception as e:
        return {"success": False, "message": str(e)}



def _opencode_stop() -> dict:
    uid = os.getuid()
    plist = SERVICES["opencode"].get("plist", "")
    subprocess.run(
        ["launchctl", "bootout", f"gui/{uid}", plist],
        capture_output=True,
        timeout=5,
    )
    time.sleep(0.5)
    subprocess.run(["pkill", "-f", "opencode serve"], capture_output=True)
    for _ in range(10):
        if not check_port(4096):
            return {"success": True, "message": "OpenCode server stopped"}
        time.sleep(0.5)
    return {
        "success": True,
        "message": "OpenCode stopped (port may still be releasing)",
    }


def _opencode_start() -> dict:
    uid = os.getuid()
    plist = SERVICES["opencode"].get("plist", "")
    subprocess.run(
        ["launchctl", "bootstrap", f"gui/{uid}", plist],
        capture_output=True,
        text=True,
        timeout=5,
    )
    for _ in range(20):
        if check_port(4096):
            pid = check_process("opencode serve")
            return {"success": True, "message": f"OpenCode running (PID: {pid})"}
        time.sleep(0.5)
    return {"success": False, "message": "OpenCode failed to start. Check server logs."}


@app.post("/api/service/{name}/stop")
async def stop_service(name: str):
    service = SERVICES.get(name)
    if not service:
        return {"success": False, "message": "Unknown service"}

    if service.get("readonly"):
        return {"success": False, "message": service.get("note", "Read-only service")}

    # Remove from started services (so it won't auto-start next time)
    started = _load_started_services()
    started.discard(name)
    _save_started_services(started)
    # Persist stopped state BEFORE blocking kill (so restart script sees it)
    stopped = _load_stopped_services()
    stopped.add(name)
    _save_stopped_services(stopped)

    if name == "opencode":
        return await asyncio.to_thread(_opencode_stop)

    # Generic process stop — run in thread to avoid blocking event loop
    return await asyncio.to_thread(_generic_stop, service)


def _generic_stop(service: dict) -> dict:
    """Stop a service by process name (blocking, run via asyncio.to_thread)."""
    process_name = service.get("process", "")
    procs_to_kill = []

    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmdline = proc.info.get("cmdline") or []
            proc_name = proc.info.get("name", "")
            if process_name in proc_name or any(process_name in arg for arg in cmdline):
                procs_to_kill.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    if not procs_to_kill:
        return {"success": False, "message": "No matching process found"}

    killed = []
    for proc in procs_to_kill:
        try:
            proc.terminate()
            killed.append(proc.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    gone, alive = psutil.wait_procs(procs_to_kill, timeout=3)

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
    if SERVICES.get(name, {}).get("readonly"):
        return {"success": False, "message": SERVICES[name].get("note", "Read-only service")}
    stop_result = await stop_service(name)
    if name not in ("opencode",):
        import asyncio

        await asyncio.sleep(2)
    start_result = await start_service(name)
    return {
        "success": start_result.get("success", False),
        "message": f"Stop: {stop_result.get('message')} | Start: {start_result.get('message')}",
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
    filter: str = "llm"  # "llm", "all"


@app.post("/api/search/huggingface")
async def search_huggingface(req: SearchRequest):
    """Search HuggingFace for models"""
    from huggingface_hub import HfApi

    # Apply proxy if configured via environment variables (huggingface_hub uses httpx/requests internally)
    settings = load_settings()
    proxy_cfg = settings.get("proxy", {})
    old_http_proxy = os.environ.get("HTTP_PROXY")
    old_https_proxy = os.environ.get("HTTPS_PROXY")
    if proxy_cfg.get("enabled"):
        proxy_url = (
            f"http://{proxy_cfg.get('host', '127.0.0.1')}:{proxy_cfg.get('port', 7890)}"
        )
        os.environ["HTTP_PROXY"] = proxy_url
        os.environ["HTTPS_PROXY"] = proxy_url

    try:
        api = HfApi()

        query = req.query

        # Search for models
        results = api.list_models(
            search=query, limit=20, sort="downloads", direction=-1
        )

        models = []
        for model in results:
            # Get model info
            try:
                info = api.model_info(model.id)
                size_bytes = sum(s.size for s in (info.siblings or []) if s.size)
            except Exception:
                size_bytes = 0

            models.append(
                {
                    "id": model.id,
                    "name": model.id.split("/")[-1],
                    "downloads": model.downloads or 0,
                    "likes": model.likes or 0,
                    "size_bytes": size_bytes,
                    "tags": model.tags[:5] if model.tags else [],
                }
            )

        return {"models": models}
    except Exception as e:
        return {"models": [], "error": str(e)}
    finally:
        # Restore proxy env vars to their original state
        if old_http_proxy is None:
            os.environ.pop("HTTP_PROXY", None)
        else:
            os.environ["HTTP_PROXY"] = old_http_proxy
        if old_https_proxy is None:
            os.environ.pop("HTTPS_PROXY", None)
        else:
            os.environ["HTTPS_PROXY"] = old_https_proxy


# ============================================================================
# OpenAI-compatible Proxy Routes (/v1/*)
# SiliconLM is dashboard-only; inference proxying is intentionally disabled.
# ============================================================================


@app.api_route("/v1/models", methods=["GET"])
async def proxy_models(request: Request):
    return {"object": "list", "data": []}


@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_v1(request: Request, path: str):
    return Response(
        status_code=404,
        content=json.dumps(
            {
                "error": {
                    "message": "Inference proxying is not provided by SiliconLM dashboard mode",
                    "type": "not_found",
                }
            }
        ),
        headers={"content-type": "application/json"},
    )


if __name__ == "__main__":
    import uvicorn

    print("🍎 SiliconLM starting at http://localhost:1234")
    uvicorn.run(app, host="0.0.0.0", port=1234)
