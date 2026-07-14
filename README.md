# SiliconLM

[中文文档](README_CN.md) | [Website](https://mjshao.fun/siliconlm/)

> [!NOTE]
> SiliconLM was retired on July 14, 2026. The repository and website are preserved as historical source; the local dashboard is no longer maintained or deployed.

Local LLM dashboard and status layer for Apple Silicon Macs. SiliconLM shows machine status, Ollama read-only status, OpenCode profile/lifecycle visibility, HuggingFace downloads, and local tool/update visibility from one FastAPI dashboard.

![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-M%20series-black?logo=apple)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## Features

- **Machine Info** - Chip, GPU cores, Neural Engine, RAM, and disk at a glance
- **Ollama Status** - Read-only reachability and model-tag visibility from the local Ollama API
- **OpenCode Control** - Track and manage the local OpenCode server lifecycle
- **Model Downloads** - HuggingFace search plus queued downloads into the configured models directory
- **Update Visibility** - Dashboard checks aligned with the local `/update` workflow for OpenCode, oh-my-openagent, Ollama reachability, sub2api, uv tools, and global npm packages

## Architecture

```text
Browser / local operator
        │
        ▼
http://localhost:1234  (SiliconLM dashboard)
        │
        ▼
Read-only local status APIs, including Ollama on :11434 when available
```

SiliconLM is now dashboard-only:

- Local LLM backends are read-only status sources, not managed runtimes.
- SiliconLM does not start, stop, restart, watch, or proxy Ollama or any other inference backend.
- `/v1/*` no longer proxies inference traffic; use the runtime backend directly for chat/completions.
- Ollama status is optional and read through `http://127.0.0.1:11434/api/tags` when available.

## Current Local Models

The dashboard defaults to `/Users/mingjian/Models` for local model inventory. Ollama model tags are shown only when Ollama is already running locally; SiliconLM never pulls or creates tags.

## Quick Start

```bash
cd ~/Documents/sync/GitHub/siliconlm

python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python server.py

open http://localhost:1234
```

## AI Agent Setup Prompt

Copy-paste this into your AI assistant:

```text
Install and set up SiliconLM on my Mac.

Repository: https://github.com/nxxxsooo/siliconlm

Steps:
1. Clone the repo to ~/Documents/sync/GitHub/siliconlm, or ask me where to put it.
2. Create a Python venv and install requirements.txt.
3. Start the dashboard with server.py on port 1234.
4. Confirm the dashboard starts, OpenCode is visible, and Ollama status appears if Ollama is already running.
5. Add shell aliases to my ~/.zshrc for easy startup.

Requirements:
- macOS 14.0+ with Apple Silicon (M series)
- Python 3.10+
- No local LLM runtime is required for the dashboard to boot
- No API keys or secrets needed

After setup, open http://localhost:1234 and verify /api/status reports dashboard status.
```

## Shell Alias

Add to `~/.zshrc`:

```bash
alias slm='cd ~/Documents/sync/GitHub/siliconlm && nohup .venv/bin/python server.py > /tmp/siliconlm.log 2>&1 & sleep 2 && open http://localhost:1234'
```

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/status` | GET | System info, services, models |
| `/api/settings` | GET/PUT | Dashboard settings |
| `/api/downloads` | GET | Active downloads, queue, presets |
| `/api/download/start` | POST | Start model download |
| `/api/search/huggingface` | POST | Search HuggingFace models |
| `/api/opencode/profiles` | GET | List available OpenCode profiles |
| `/api/opencode/profiles/{profile_id}` | POST | Switch active OpenCode profile |
| `/api/update/run` | POST | Spawn the local update workflow |
| `/v1/models` | GET | Dashboard-mode empty model list |
| `/v1/{path}` | any | Returns 404 because SiliconLM does not proxy inference |

## Chat API Usage

SiliconLM no longer provides chat proxying. Send inference requests directly to your chosen runtime backend, such as Ollama, instead of routing them through the dashboard.

## Tech Stack

| Component | Technology |
|---|---|
| Backend | FastAPI + uvicorn |
| Frontend | TailwindCSS + vanilla JS |
| Local status | Ollama tags API, OpenCode status |
| Downloads | huggingface_hub |
| HTTP client | httpx async |

## License

MIT
