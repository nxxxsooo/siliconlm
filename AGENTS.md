# SiliconLM

## Overview
CLI tool + LaunchAgent for managing local LLM models on Apple Silicon Macs. Downloads models from HuggingFace, manages LMStudio server lifecycle, provides model switching. Written in Python.

## Architecture
- Python CLI (`siliconlm/`)
- LaunchAgent for auto-start (`launchctl bootstrap/bootout`)
- HuggingFace Hub for model downloads (snapshot_download, no aria2)
- Landing page at mjshao.fun/siliconlm/

## Key Files
- `siliconlm/` — CLI source
- `scripts/` — Install/setup scripts
- `docs/` — Landing page (deployed to portfolio)
- `README.md` + `README_CN.md` — Bilingual docs

## Patterns & Conventions
- Service management via launchctl (not process kill)
- Models stored in LMStudio default path
- "For AI Agent" setup prompt in README and landing page

## Current Status
- ✅ Published on GitHub
- ✅ Landing page on portfolio
- ✅ launchctl-based service management

## Next Steps
- Model catalog updates as new models release

## Resolved Issues
- Switched from aria2 to huggingface_hub snapshot_download
- Fixed service management: launchctl bootstrap/bootout instead of process kill
