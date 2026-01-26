# SiliconLM

Apple Silicon Mac 本地 LLM 管理面板。管理模型、服务、嵌入向量和下载。

![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-M1%2FM2%2FM3-black?logo=apple)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## 功能特性

- **系统信息** - 芯片、GPU核心、神经引擎、内存、磁盘一览
- **MLX 嵌入服务** - OpenAI 兼容的 `/v1/embeddings` API（端口 8766）
- **多后端支持** - MLX、mlx-lm（解码器模型）、sentence-transformers
- **服务管理** - 启动/停止 LMStudio、MLX Embeddings、OpenCode
- **智能代理** - `/v1/embeddings` 路由到 MLX，`/v1/chat` 路由到 LMStudio
- **模型下载** - HuggingFace 搜索 + aria2 大文件加速
- **设置面板** - 配置模型目录、默认嵌入模型

## 架构

```
CherryStudio / 客户端
        │
        ▼
http://localhost:8765/v1/*  (SiliconLM 代理)
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

## 支持的嵌入模型

| 模型 | 后端 | 维度 | 速度 |
|------|------|------|------|
| mixedbread-ai/mxbai-embed-large-v1 | MLX | 1024 | 快 |
| BAAI/bge-m3 | sentence-transformers | 1024 | 中 |
| mlx-community/Qwen3-Embedding-0.6B-4bit | mlx-lm | 1024 | 快 |
| mlx-community/Qwen3-Embedding-8B-4bit | mlx-lm | 4096 | 中 |
| mlx-community/gte-Qwen2-7B-instruct-4bit | mlx-lm | 3584 | 中 |

## 快速开始

```bash
cd ~/Documents/sync/GitHub/siliconlm

# 安装
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 或手动安装
.venv/bin/pip install fastapi uvicorn psutil huggingface_hub pydantic httpx \
    mlx mlx-embeddings mlx-lm sentence-transformers

# 可选：aria2 用于大文件下载 (>1.5GB)
brew install aria2

# 启动面板（端口 8765）
.venv/bin/python server.py

# 启动嵌入服务（端口 8766）
.venv/bin/python embedding_server.py

# 打开面板
open http://localhost:8765
```

## Shell 别名

添加到 `~/.zshrc`：

```bash
# 启动 SiliconLM 面板 + 嵌入服务
alias slm='cd ~/Documents/sync/GitHub/siliconlm && \
    nohup .venv/bin/python server.py > /tmp/siliconlm.log 2>&1 & \
    nohup .venv/bin/python embedding_server.py > /tmp/mlx_embeddings.log 2>&1 & \
    sleep 2 && open http://localhost:8765'
```

## API 接口

### 面板服务（端口 8765）

| 接口 | 方法 | 描述 |
|------|------|------|
| `/api/status` | GET | 系统信息、服务、模型 |
| `/api/settings` | GET/PUT | 面板设置 |
| `/api/downloads` | GET | 下载进度、队列、预设 |
| `/api/download/start` | POST | 开始下载模型 |
| `/api/search/huggingface` | POST | 搜索 HuggingFace 模型 |
| `/v1/embeddings` | POST | 代理到 MLX Embeddings |
| `/v1/chat/completions` | POST | 代理到 LMStudio |

### MLX 嵌入服务（端口 8766）

| 接口 | 方法 | 描述 |
|------|------|------|
| `/v1/embeddings` | POST | 生成嵌入向量（OpenAI 兼容） |
| `/v1/models` | GET | 列出可用嵌入模型 |
| `/api/metrics` | GET | 请求统计、延迟、活动 |
| `/health` | GET | 健康检查 |

## 嵌入 API 使用示例

```bash
# 生成嵌入向量
curl -X POST http://localhost:8766/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mixedbread-ai/mxbai-embed-large-v1",
    "input": "你好，世界！"
  }'

# 批量嵌入
curl -X POST http://localhost:8766/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ",
    "input": ["文本1", "文本2", "文本3"]
  }'
```

## 并发请求处理

- **GPU 模型**（MLX、mlx-lm）：串行执行，防止 Metal 崩溃
- **CPU 模型**（sentence-transformers）：可与 GPU 并行运行
- **混合负载**：GPU 和 CPU 请求可同时执行

## 技术栈

| 组件 | 技术 |
|------|------|
| 后端 | FastAPI + uvicorn |
| 前端 | TailwindCSS + 原生 JS |
| 嵌入 | MLX + mlx-lm + sentence-transformers |
| 下载 | huggingface_hub + aria2 |
| 代理 | httpx async |

## 许可证

MIT
