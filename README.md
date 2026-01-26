# SiliconLM

[中文文档](README_CN.md)

Local LLM dashboard for Apple Silicon Macs. Manage models, services, embeddings, and downloads.

![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-M1%2FM2%2FM3-black?logo=apple)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## Features

- **Machine Info** - Chip, GPU cores, Neural Engine, RAM, disk at a glance
- **MLX Embeddings Server** - OpenAI-compatible `/v1/embeddings` API on port 8766
- **Multi-Backend Support** - MLX, mlx-lm (decoder models), sentence-transformers
- **Service Management** - Start/stop LMStudio, MLX Embeddings, OpenCode
- **Smart Proxy** - Routes `/v1/embeddings` to MLX, `/v1/chat` to LMStudio
- **Model Downloads** - HuggingFace search + aria2 acceleration for large files
- **Settings Panel** - Configure models directory, default embedding model

## Architecture

```
CherryStudio / Client
        │
        ▼
http://localhost:8765/v1/*  (SiliconLM Proxy)
        │
   ┌────┴────┐
   ▼         ▼
/v1/embeddings   /v1/chat/*
   │              │
   ▼              ▼
:8766 (MLX)    :1234 (LMStudio)
   │
   ├─► MLX (bert, roberta)
   ├─► mlx-lm (Qwen3, gte-Qwen2)
   └─► sentence-transformers (bge-m3)
```

## Supported Embedding Models

| Model | Backend | Dimensions | Speed |
|-------|---------|------------|-------|
| mixedbread-ai/mxbai-embed-large-v1 | MLX | 1024 | Fast |
| BAAI/bge-m3 | sentence-transformers | 1024 | Medium |
| mlx-community/Qwen3-Embedding-0.6B-4bit | mlx-lm | 1024 | Fast |
| mlx-community/Qwen3-Embedding-8B-4bit | mlx-lm | 4096 | Medium |
| mlx-community/gte-Qwen2-7B-instruct-4bit | mlx-lm | 3584 | Medium |

## Quick Start

```bash
cd ~/Documents/sync/GitHub/siliconlm

# Setup
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Or manual install
.venv/bin/pip install fastapi uvicorn psutil huggingface_hub pydantic httpx \
    mlx mlx-embeddings mlx-lm sentence-transformers

# Optional: aria2 for large file downloads (>1.5GB)
brew install aria2

# Run dashboard (port 8765)
.venv/bin/python server.py

# Run embedding server (port 8766)
.venv/bin/python embedding_server.py

# Open dashboard
open http://localhost:8765
```

## Shell Alias

Add to `~/.zshrc`:

```bash
# Start SiliconLM dashboard + embedding server
alias slm='cd ~/Documents/sync/GitHub/siliconlm && \
    nohup .venv/bin/python server.py > /tmp/siliconlm.log 2>&1 & \
    nohup .venv/bin/python embedding_server.py > /tmp/mlx_embeddings.log 2>&1 & \
    sleep 2 && open http://localhost:8765'
```

## API Endpoints

### Dashboard (port 8765)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/status` | GET | System info, services, models |
| `/api/settings` | GET/PUT | Dashboard settings |
| `/api/downloads` | GET | Active downloads, queue, presets |
| `/api/download/start` | POST | Start model download |
| `/api/search/huggingface` | POST | Search HuggingFace models |
| `/v1/embeddings` | POST | Proxy to MLX Embeddings |
| `/v1/chat/completions` | POST | Proxy to LMStudio |

### MLX Embeddings (port 8766)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/embeddings` | POST | Generate embeddings (OpenAI-compatible) |
| `/v1/models` | GET | List available embedding models |
| `/api/metrics` | GET | Request stats, latency, activity |
| `/health` | GET | Health check |

## Embedding API Usage

```bash
# Generate embeddings
curl -X POST http://localhost:8766/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mixedbread-ai/mxbai-embed-large-v1",
    "input": "Hello, world!"
  }'

# Batch embeddings
curl -X POST http://localhost:8766/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ",
    "input": ["text 1", "text 2", "text 3"]
  }'
```

## Concurrent Request Handling

- **GPU models** (MLX, mlx-lm): Serialized to prevent Metal crashes
- **CPU models** (sentence-transformers): Can run parallel with GPU
- **Mixed workloads**: GPU and CPU requests run concurrently

## Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | FastAPI + uvicorn |
| Frontend | TailwindCSS + Vanilla JS |
| Embeddings | MLX + mlx-lm + sentence-transformers |
| Downloads | huggingface_hub + aria2 |
| Proxy | httpx async |

## License

MIT
