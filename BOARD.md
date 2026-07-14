# SiliconLM — Changelog

<!-- Append-only. Newest date first. -->
<!-- Prefixes: feat / fix / refactor / chore / docs / deploy -->

## Changelog

### 2026-07-14
- **chore**: Retired SiliconLM after its dashboard had remained disabled and unused; preserved the final dashboard-only source and public landing page as historical material.
- **chore**: Removed local launch aliases, helper scripts, mectl registration, runtime venv/state, and logs while preserving all Ollama tags and `/Users/mingjian/Models/` source folders.

### 2026-05-21
- **fix**: Removed stale LM Studio and removed MLX Embeddings launch references from the SiliconLM LaunchAgent/manual helper scripts so they only manage the dashboard on `:1234`.
- **fix**: Cleaned dashboard/proxy activity paths to use oMLX chat and `/api/activity` instead of the removed `:8766` embedding service; verified live services are `omlx` and `opencode`.

### 2026-05-05
- **docs**: Updated SiliconLM public docs, landing page copy, and local model notes to reflect the oMLX-only architecture and current OpenCode `omlx/*` model IDs.

### 2026-05-03
- **refactor**: Removed LMStudio, llmster, and MLX Embeddings backends so SiliconLM is now focused on oMLX and OpenCode service management.
- **feat**: Added update skill-aligned checker rows for OpenCode, oh-my-openagent plugin cache/runtime state, oMLX, sub2api, uv tools, and global npm packages.
- **feat**: Added a dashboard endpoint/action to spawn `update-all` via the "Run /update" control.
- **chore**: Cleaned settings, requirements, download manager defaults, and frontend controls around the new `/Users/mingjian/Models` model directory and removed embedding/backend selector UI.
- **docs**: Captured verified behavior: services return `omlx` and `opencode`, `/v1/embeddings` returns 404, and the new CLI agents schema populates.
