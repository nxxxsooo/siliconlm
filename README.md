# SiliconLM

Local LLM dashboard for Apple Silicon Macs. Manage models, services, and downloads.

![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-M1%2FM2%2FM3-black?logo=apple)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## Features

- **Machine Info** - Chip, GPU cores, Neural Engine, RAM, disk at a glance
- **Service Management** - Start/stop LMStudio, OpenCode, Claude Code
- **Model Downloads** - HuggingFace models with aria2 acceleration for large files
- **Smart Presets** - RAM-aware model recommendations (Coding/General/Embedding)
- **Download Queue** - Pause, resume, cancel, auto-retry on failure
- **Model Browser** - View installed models, reveal in Finder, delete

## Quick Start

```bash
cd ~/Documents/sync/GitHub/siliconlm

# Setup
python3 -m venv .venv
.venv/bin/pip install fastapi uvicorn psutil huggingface_hub pydantic

# Optional: aria2 for large file downloads (>1.5GB)
brew install aria2

# Run
.venv/bin/python server.py
open http://localhost:8765
```

## Shell Alias

Add to `~/.zshrc`:

```bash
alias slm='cd ~/Documents/sync/GitHub/siliconlm && nohup .venv/bin/python server.py > /tmp/siliconlm.log 2>&1 & sleep 1 && open http://localhost:8765'
```

## aria2 Integration

Large model files (>1.5GB) automatically use aria2 for:
- Resume interrupted downloads
- Auto-retry on failures (max 10 retries)
- 60s stall detection with restart
- HuggingFace token auth for gated models

Fallback to huggingface_hub if aria2 not installed.

## Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | FastAPI + uvicorn |
| Frontend | TailwindCSS + Vanilla JS |
| Downloads | huggingface_hub + aria2 |
| Refresh | 3s polling |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/status` | GET | System info, services, models |
| `/api/downloads` | GET | Active downloads, queue, presets |
| `/api/download/start` | POST | Start model download |
| `/api/download/{id}/pause` | POST | Pause download |
| `/api/download/{id}/resume` | POST | Resume download |
| `/api/download/{id}/cancel` | POST | Cancel download |
| `/api/service/{name}/{action}` | POST | Service start/stop/restart |
| `/api/model` | DELETE | Delete model |
| `/api/model/reveal` | POST | Show in Finder |

## License

MIT
