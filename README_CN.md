# SiliconLM

[English](README.md) | [网站](https://mjshao.fun/siliconlm/)

> [!NOTE]
> SiliconLM 作为公开项目继续保留，但目前不再部署于作者的 Mac。本地控制面板运行环境和启动入口已于 2026 年 7 月 14 日移除。

Apple Silicon Mac 本地 LLM 运维面板。SiliconLM 现在专注展示机器状态、Ollama 只读状态、OpenCode 配置/服务可见性、HuggingFace 模型下载，以及本地工具更新状态。

![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-M%20series-black?logo=apple)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## 功能特性

- **系统信息** - 芯片、GPU 核心、神经引擎、内存、磁盘一览
- **Ollama 状态** - 只读查看本地 Ollama API 可达性和模型 tag
- **OpenCode 控制** - 查看并管理本地 OpenCode server 生命周期
- **模型下载** - HuggingFace 搜索 + 队列下载到配置的模型目录
- **更新状态** - 对齐本地 `/update` 工作流，检查 OpenCode、oh-my-openagent、Ollama 可达性、sub2api、uv tools、全局 npm 包

## 架构

```text
浏览器 / 本机操作者
        │
        ▼
http://localhost:1234  (SiliconLM 面板)
        │
        ▼
只读本地状态 API，包括可选的 Ollama :11434
```

SiliconLM 现在是纯面板模式：

- 本地 LLM 后端只是只读状态来源，不再由 SiliconLM 管理生命周期。
- SiliconLM 不启动、停止、重启、watch 或代理 Ollama / 其他推理后端。
- `/v1/*` 不再代理推理流量；聊天补全请直接访问实际运行时后端。
- Ollama 状态是可选信息，仅在本机 `http://127.0.0.1:11434/api/tags` 可达时展示。

## 当前本地模型

面板默认模型目录是 `/Users/mingjian/Models`，用于本地模型盘点。Ollama 模型 tag 只在 Ollama 已经本地运行时展示；SiliconLM 不拉取、不创建任何 tag。

## 快速开始

```bash
cd ~/Documents/sync/GitHub/siliconlm

python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python server.py

open http://localhost:1234
```

## 让 AI 帮你安装

复制下面这段给 AI 助手：

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

## Shell 别名

添加到 `~/.zshrc`：

```bash
alias slm='cd ~/Documents/sync/GitHub/siliconlm && nohup .venv/bin/python server.py > /tmp/siliconlm.log 2>&1 & sleep 2 && open http://localhost:1234'
```

## API 接口

| 接口 | 方法 | 描述 |
|---|---|---|
| `/api/status` | GET | 系统信息、服务、模型 |
| `/api/settings` | GET/PUT | 面板设置 |
| `/api/downloads` | GET | 下载进度、队列、预设 |
| `/api/download/start` | POST | 开始下载模型 |
| `/api/search/huggingface` | POST | 搜索 HuggingFace 模型 |
| `/api/opencode/profiles` | GET | 列出 OpenCode profiles |
| `/api/opencode/profiles/{profile_id}` | POST | 切换当前 OpenCode profile |
| `/api/update/run` | POST | 启动本地 update 工作流 |
| `/v1/models` | GET | 纯面板模式下返回空模型列表 |
| `/v1/{path}` | any | 返回 404；SiliconLM 不代理推理 |

## Chat API 示例

SiliconLM 不再提供聊天代理。推理请求请直接发送给实际运行时后端，例如 Ollama，而不是经过 dashboard。

## 技术栈

| 组件 | 技术 |
|---|---|
| 后端 | FastAPI + uvicorn |
| 前端 | TailwindCSS + 原生 JS |
| 本地状态 | Ollama tags API、OpenCode 状态 |
| 下载 | huggingface_hub |
| HTTP 客户端 | httpx async |

## 许可证

MIT
