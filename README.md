# Localboard

Local services dashboard for macOS - manage LLM models and dev services.

## Features

- **Service Management**: Start/Stop/Restart local services (LMStudio, OpenCode, Claude Code)
- **Model Downloads**: Download HuggingFace models with preset list
- **Model Management**: View downloaded models, show in Finder
- **System Monitor**: Memory & disk usage

## Quick Start

```bash
# Install dependencies
cd ~/Documents/sync/GitHub/localboard
python3 -m venv .venv
.venv/bin/pip install fastapi uvicorn psutil huggingface_hub pydantic

# Run
.venv/bin/python server.py
# Open http://localhost:8765
```

## Alias

Add to `~/.zshrc`:
```bash
alias dash='cd ~/Documents/sync/GitHub/localboard && .venv/bin/python server.py'
```

## Tech Stack

- Backend: FastAPI + uvicorn
- Frontend: TailwindCSS CDN + Vanilla JS
- Auto-refresh: 3 seconds polling

## License

MIT
