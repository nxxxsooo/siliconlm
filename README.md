# Localboard

Local services dashboard for macOS - manage LLM models and dev services.

## Features

- **Service Management**: Start/Stop/Restart local services (LMStudio, OpenCode, Claude Code)
- **Model Downloads**: Download HuggingFace models with preset list, stop/resume support
- **Download Management**: Show in Finder, stop, resume, remove
- **Model Browser**: View downloaded models, reveal in Finder
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

## Alias (recommended)

Add to `~/.zshrc`:
```bash
alias dash='cd ~/Documents/sync/GitHub/localboard && nohup .venv/bin/python server.py > /tmp/localboard.log 2>&1 & sleep 1 && open http://localhost:8765'
```

Then run `dash` to start the dashboard in background and open browser.

## Tech Stack

- Backend: FastAPI + uvicorn
- Frontend: TailwindCSS CDN + Vanilla JS
- Auto-refresh: 3 seconds polling

## License

MIT
