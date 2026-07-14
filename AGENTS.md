# SiliconLM

## Overview
Dashboard and status layer for local LLM tooling on Apple Silicon Macs. Provides a web UI on port 1234 for machine status, Ollama read-only status, OpenCode profile/lifecycle visibility, HuggingFace downloads, and CLI/tool update visibility. Written in Python with FastAPI and a Tailwind/vanilla JS frontend.

## Status

Not locally deployed as of 2026-07-14. The public project, source, and landing page remain active, while the author's local launchers, aliases, runtime state, and mectl resource were removed. Ollama models and `/Users/mingjian/Models/` are independent assets and were intentionally preserved.

## Architecture
```text
Client / Browser
         |
http://localhost:1234  (SiliconLM Dashboard)
         |
         v
Read-only local status APIs such as Ollama :11434
```

- **server.py** - FastAPI app: dashboard APIs, read-only local status, OpenCode profile/service visibility, update endpoint, download queue integration
- **templates/index.html** - Dashboard UI (Tailwind)
- **settings.json** - Runtime config for services, proxy, models directory, and CLI agent checker state
- **download_manager.py** - HuggingFace model download queue using the configured models directory
- Operational launchers live outside the repo: `~/.local/bin/start-siliconlm` is the LaunchAgent target; `~/.bin/siliconlm-start`, `~/.bin/siliconlm-stop`, and `~/.bin/siliconlm-restart` are manual helpers.

### Dashboard-Only Simplification (2026-05-24)
SiliconLM is now a pure dashboard/management/status layer:
- Removed local inference runtime lifecycle control from SiliconLM.
- It does not start, stop, restart, watch, or proxy any local LLM backend.
- Ollama is treated as an optional read-only status source via its local tags API.
- OpenCode profile/dashboard management and HuggingFace download visibility remain.
- `/v1/*` routes no longer proxy inference and return dashboard-mode responses.

### Update Skill-Aligned Monitoring (2026-05-03)
Legacy CLI Agents monitoring was replaced with checker rows aligned to the local `/update` workflow:
- OpenCode binary
- oh-my-openagent plugin, including the 3-layer cache/runtime reality
- Ollama read-only reachability
- sub2api
- uv tools
- global npm packages

The UI also exposes a "Run /update" action that spawns `update-all`.

## Key Files
- `server.py` - Main FastAPI server, dashboard APIs, local status checks, CLI/update checkers, API endpoints
- `templates/index.html` - Dashboard UI
- `settings.json` - Runtime config (services/proxy/models/update checker settings)
- `download_manager.py` - HuggingFace download queue
- `scripts/` - Install/setup scripts
- `docs/` - Landing page (deployed to portfolio)

## Patterns & Conventions
- Service management: read-only for local LLM backends; launchctl control remains for OpenCode.
- Models directory defaults to `/Users/mingjian/Models`.
- Settings persist via `settings.json`; started-service state lives in `~/.local/share/siliconlm/`.
- Logs live under `~/Library/Logs/SiliconLM/` for the LaunchAgent-managed dashboard.
- CLI/tool health rows should follow the same operational truth as the `update` skill and `update-all` script rather than maintaining a separate CLI Agents schema.

## Resolved Issues
- Switched from aria2 to `huggingface_hub.snapshot_download`.
- Fixed service management: launchctl bootstrap/bootout instead of process kill.
- Historical: oMLX robustness work existed while SiliconLM managed that runtime (2026-03-23); this is no longer an active responsibility.
- Removed LMStudio and `llmster` as legacy backends; later replaced runtime management with dashboard-only status mode (2026-05-24).
- Removed MLX Embeddings server and embedding settings/routes; `/v1/embeddings` is not provided by dashboard mode.
- Cleaned stale local launch/helper scripts and dashboard activity paths after earlier migrations: no LMStudio startup block, no active `embedding_server.py` launcher/route, no `mlx_embeddings` runtime dependency, and no active dependency on port `8766` (2026-05-21).
- Replaced legacy CLI Agents monitoring with update skill-aligned checkers and a `Run /update` endpoint (2026-05-03).
