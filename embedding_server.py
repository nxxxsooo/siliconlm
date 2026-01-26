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
    
    def record_request(self, model: str, tokens: int, latency_ms: float, input_preview: str):
        self.total_requests += 1
        self.total_tokens += tokens
        self.total_latency_ms += latency_ms
        self.recent_requests.append({
            "time": time.strftime("%H:%M:%S"),
            "model": model.split("/")[-1][:20],
            "tokens": tokens,
            "latency_ms": round(latency_ms, 1),
            "preview": input_preview[:50] + "..." if len(input_preview) > 50 else input_preview
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


def discover_embedding_models() -> List[dict]:
    """Find embedding models in the models directory"""
    models = []
    if not MODELS_DIR.exists():
        return models
    
    # Look for models with 'embed' in name or specific known patterns
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
                # Check for model files
                has_safetensors = any(model_dir.glob("*.safetensors"))
                has_config = (model_dir / "config.json").exists()
                
                if has_safetensors or has_config:
                    repo_id = f"{org_dir.name}/{model_dir.name}"
                    size = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
                    models.append({
                        "id": repo_id,
                        "name": model_dir.name,
                        "size_bytes": size,
                        "path": str(model_dir)
                    })
    
    return models


def get_model(model_id: str):
    """Load model with caching"""
    global _model_cache
    
    with _model_lock:
        if model_id in _model_cache:
            return _model_cache[model_id]
        
        # Try to load model
        try:
            from mlx_embeddings import load_model
            
            # Check if it's a local path or HF repo
            local_path = MODELS_DIR / model_id
            if local_path.exists():
                model = load_model(str(local_path))
            else:
                model = load_model(model_id)
            
            _model_cache[model_id] = model
            metrics.current_model = model_id
            return model
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")


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


@app.post("/v1/embeddings")
def create_embeddings(request: EmbeddingRequest):
    """Create embeddings - OpenAI compatible"""
    start_time = time.time()
    
    # Normalize input to list
    inputs = request.input if isinstance(request.input, list) else [request.input]
    
    try:
        from mlx_embeddings import embed
        
        model = get_model(request.model)
        embeddings = embed(model, inputs)
        
        # Convert to list format
        if hasattr(embeddings, 'tolist'):
            embeddings_list = embeddings.tolist()
        else:
            embeddings_list = [e.tolist() if hasattr(e, 'tolist') else list(e) for e in embeddings]
        
        # Calculate tokens (rough estimate: 1 token per 4 chars)
        total_chars = sum(len(t) for t in inputs)
        estimated_tokens = total_chars // 4
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Record metrics
        preview = inputs[0] if inputs else ""
        metrics.record_request(request.model, estimated_tokens, latency_ms, preview)
        
        return {
            "object": "list",
            "data": [
                {"object": "embedding", "embedding": emb, "index": i}
                for i, emb in enumerate(embeddings_list)
            ],
            "model": request.model,
            "usage": {
                "prompt_tokens": estimated_tokens,
                "total_tokens": estimated_tokens
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
