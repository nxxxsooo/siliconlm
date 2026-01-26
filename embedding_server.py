#!/usr/bin/env python3
"""
MLX Embeddings Server - OpenAI-compatible embedding API
Runs on port 8766, managed by SiliconLM dashboard
"""

import time
import json
import threading
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Union, List
from collections import deque

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Lazy load mlx_embeddings to speed up startup
_model_cache: dict = {}
_model_lock = threading.Lock()

MODELS_DIR = Path.home() / ".lmstudio" / "models"
EMBEDDING_PORT = 8766

# Metrics tracking
@dataclass
class Metrics:
    total_requests: int = 0
    total_tokens: int = 0
    total_latency_ms: float = 0.0
    recent_requests: deque = field(default_factory=lambda: deque(maxlen=50))
    start_time: float = field(default_factory=time.time)
    current_model: Optional[str] = None
    embedding_dim: int = 0
    
    def record_request(self, model: str, tokens: int, latency_ms: float, input_preview: str, backend: str = ""):
        self.total_requests += 1
        self.total_tokens += tokens
        self.total_latency_ms += latency_ms
        self.recent_requests.append({
            "time": time.strftime("%H:%M:%S"),
            "model": model.split("/")[-1][:20],
            "tokens": tokens,
            "latency_ms": round(latency_ms, 1),
            "preview": input_preview[:50] + "..." if len(input_preview) > 50 else input_preview,
            "backend": backend
        })
    
    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.total_requests if self.total_requests > 0 else 0
    
    @property
    def uptime_seconds(self) -> float:
        return time.time() - self.start_time
    
    def to_dict(self) -> dict:
        return {
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "uptime_seconds": round(self.uptime_seconds),
            "current_model": self.current_model,
            "embedding_dim": self.embedding_dim,
            "recent_requests": list(self.recent_requests)
        }

metrics = Metrics()

app = FastAPI(title="MLX Embeddings Server", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Model type to backend mapping
MLX_SUPPORTED_TYPES = {"bert", "xlm-roberta", "roberta"}
# sentence-transformers supported architectures (encoder models)
ST_SUPPORTED_TYPES = {"bert", "xlm-roberta", "roberta", "mpnet", "distilbert"}
# mlx-lm supported architectures (decoder models with mean pooling)
MLX_LM_SUPPORTED_TYPES = {"qwen3", "qwen2", "llama", "mistral", "gemma", "phi"}

def _get_model_info(model_path: Path) -> dict:
    """Determine which backend can handle this model, or None if unsupported."""
    config_path = model_path / "config.json"
    if not config_path.exists():
        return None
    try:
        with open(config_path) as f:
            config = json.load(f)
        model_type = config.get("model_type", "").lower()
        has_safetensors = any(model_path.glob("*.safetensors"))
        has_pytorch = any(model_path.glob("*.bin")) or any(model_path.glob("*.pt"))
        
        # MLX backend: safetensors + supported encoder architecture
        if has_safetensors and model_type in MLX_SUPPORTED_TYPES:
            return {"model_type": model_type, "backend": "mlx"}
        
        # mlx-lm backend: MLX-quantized decoder models (safetensors only, no pytorch)
        if has_safetensors and not has_pytorch and model_type in MLX_LM_SUPPORTED_TYPES:
            return {"model_type": model_type, "backend": "mlx-lm"}
        
        # sentence-transformers: pytorch weights + supported encoder architecture
        if has_pytorch and model_type in ST_SUPPORTED_TYPES:
            return {"model_type": model_type, "backend": "sentence-transformers"}
        
        # Fallback: try sentence-transformers if it has pytorch weights (may fail)
        if has_pytorch:
            return {"model_type": model_type, "backend": "sentence-transformers"}
        
        return None
    except Exception:
        return None


def discover_embedding_models() -> List[dict]:
    models = []
    if not MODELS_DIR.exists():
        return models
    
    embedding_patterns = ["embed", "gte-", "bge-", "e5-", "mxbai-embed"]
    
    for org_dir in MODELS_DIR.iterdir():
        if not org_dir.is_dir():
            continue
        for model_dir in org_dir.iterdir():
            if not model_dir.is_dir():
                continue
            
            model_name = model_dir.name.lower()
            is_embedding = any(p in model_name for p in embedding_patterns)
            
            if is_embedding:
                info = _get_model_info(model_dir)
                if not info:
                    continue
                
                repo_id = f"{org_dir.name}/{model_dir.name}"
                size = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
                models.append({
                    "id": repo_id,
                    "name": model_dir.name,
                    "size_bytes": size,
                    "path": str(model_dir),
                    "backend": info["backend"]
                })
    
    return models


_st_models = {}
_mlx_lm_models = {}

def get_model(model_id: str):
    """Load model with appropriate backend. Returns (model_data, backend_type)."""
    global _model_cache, _st_models, _mlx_lm_models
    
    with _model_lock:
        # Check caches first
        if model_id in _model_cache:
            return _model_cache[model_id], "mlx"
        if model_id in _st_models:
            return _st_models[model_id], "st"
        if model_id in _mlx_lm_models:
            return _mlx_lm_models[model_id], "mlx-lm"
        
        local_path = MODELS_DIR / model_id
        info = _get_model_info(local_path) if local_path.exists() else None
        backend = info["backend"] if info else "mlx"
        
        try:
            if backend == "mlx":
                from mlx_embeddings import load
                if local_path.exists():
                    model, tokenizer = load(str(local_path))
                else:
                    model, tokenizer = load(model_id)
                _model_cache[model_id] = (model, tokenizer)
                metrics.current_model = model_id
                return (model, tokenizer), "mlx"
            
            elif backend == "mlx-lm":
                from mlx_lm import load as mlx_lm_load
                model, tokenizer = mlx_lm_load(str(local_path) if local_path.exists() else model_id)
                _mlx_lm_models[model_id] = (model, tokenizer)
                metrics.current_model = model_id
                return (model, tokenizer), "mlx-lm"
            
            else:  # sentence-transformers
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer(str(local_path) if local_path.exists() else model_id)
                _st_models[model_id] = model
                metrics.current_model = model_id
                return model, "st"
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


# Request/Response models
class EmbeddingRequest(BaseModel):
    model: str
    input: Union[str, List[str]]
    encoding_format: Optional[str] = "float"


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: list
    model: str
    usage: dict


@app.get("/")
def root():
    return {"status": "ok", "service": "MLX Embeddings Server", "port": EMBEDDING_PORT}


@app.get("/v1/models")
def list_models():
    """List available embedding models"""
    models = discover_embedding_models()
    return {
        "object": "list",
        "data": [{"id": m["id"], "object": "model", "owned_by": "local"} for m in models]
    }


@app.get("/models")
def list_models_alt():
    """Alias for /v1/models"""
    return list_models()


MAX_BATCH_SIZE = 32
MAX_INPUT_CHARS = 8192
_inference_lock = threading.Lock()

@app.post("/v1/embeddings")
def create_embeddings(request: EmbeddingRequest):
    start_time = time.time()
    inputs = request.input if isinstance(request.input, list) else [request.input]
    
    if len(inputs) > MAX_BATCH_SIZE:
        raise HTTPException(status_code=400, detail=f"Batch size {len(inputs)} exceeds max {MAX_BATCH_SIZE}")
    
    inputs = [t[:MAX_INPUT_CHARS] for t in inputs]
    
    backend = ""
    with _inference_lock:
        try:
            import gc
            model_data, backend = get_model(request.model)
            
            if backend == "mlx":
                from mlx_embeddings import generate
                import mlx.core as mx
                
                model, tokenizer = model_data
                embeddings = generate(model, tokenizer, inputs)
                mx.eval(embeddings)
                
                if embeddings.ndim == 3:
                    embeddings = mx.mean(embeddings, axis=1)
                elif embeddings.ndim == 2 and len(inputs) == 1 and embeddings.shape[0] > 1:
                    embeddings = mx.mean(embeddings, axis=0, keepdims=True)
                
                mx.eval(embeddings)
                embeddings_list = embeddings.tolist()
                del embeddings
            
            elif backend == "mlx-lm":
                # Decoder model embedding via last hidden state + mean pooling
                import mlx.core as mx
                import mlx.nn as nn
                
                model, tokenizer = model_data
                all_embeddings = []
                
                for text in inputs:
                    input_ids = mx.array([tokenizer.encode(text)])
                    
                    # Get token embeddings
                    h = model.model.embed_tokens(input_ids)
                    
                    # Create causal mask
                    T = h.shape[1]
                    mask = nn.MultiHeadAttention.create_additive_causal_mask(T)
                    mask = mask.astype(h.dtype)
                    
                    # Pass through transformer layers
                    for layer in model.model.layers:
                        h = layer(h, mask=mask)
                    
                    # Final layer norm + mean pooling
                    h = model.model.norm(h)
                    embedding = mx.mean(h, axis=1)
                    all_embeddings.append(embedding)
                
                embeddings = mx.concatenate(all_embeddings, axis=0)
                mx.eval(embeddings)
                embeddings_list = embeddings.tolist()
                del embeddings, all_embeddings
            
            else:  # sentence-transformers
                embeddings = model_data.encode(inputs, convert_to_numpy=True)
                embeddings_list = embeddings.tolist()
                del embeddings
            
            dim = len(embeddings_list[0]) if embeddings_list else 0
            gc.collect()
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))
    
    total_chars = sum(len(t) for t in inputs)
    estimated_tokens = max(1, total_chars // 4)
    latency_ms = (time.time() - start_time) * 1000
    
    preview = inputs[0] if inputs else ""
    metrics.record_request(request.model, estimated_tokens, latency_ms, preview, backend)
    metrics.embedding_dim = dim
    
    return {
        "object": "list",
        "data": [
            {"object": "embedding", "embedding": emb, "index": i}
            for i, emb in enumerate(embeddings_list)
        ],
        "model": request.model,
        "usage": {"prompt_tokens": estimated_tokens, "total_tokens": estimated_tokens, "dimensions": dim}
    }


@app.post("/embeddings")
def create_embeddings_alt(request: EmbeddingRequest):
    """Alias for /v1/embeddings"""
    return create_embeddings(request)


@app.get("/api/metrics")
def get_metrics():
    """Get server metrics for dashboard"""
    return metrics.to_dict()


@app.get("/api/models")
def get_local_models():
    """Get detailed info about local embedding models"""
    return {"models": discover_embedding_models()}


@app.get("/health")
def health():
    return {"status": "healthy", "uptime": metrics.uptime_seconds}


if __name__ == "__main__":
    import uvicorn
    print(f"Starting MLX Embeddings Server on port {EMBEDDING_PORT}...")
    print(f"Models directory: {MODELS_DIR}")
    print(f"Discovered models: {[m['id'] for m in discover_embedding_models()]}")
    uvicorn.run(app, host="0.0.0.0", port=EMBEDDING_PORT, log_level="info")
