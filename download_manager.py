"""SiliconLM Download Manager - Queue-based HuggingFace model downloads"""

import json
import multiprocessing
import os
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional
import shutil

QUEUE_FILE = Path(__file__).parent / ".download_queue.json"
HF_TOKEN = os.environ.get("HF_TOKEN")

# Patterns to ignore during download
IGNORE_PATTERNS = [
    "*.gguf", "*.onnx", "*.onnx_data", "onnx/*", "openvino/*",
    "*.msgpack", "*.h5", "*.tflite", "*.tar.gz", "*.zip",
    "coreml/*", "flax_model*", "tf_model*", "rust_model*",
]


def _should_ignore(filename: str) -> bool:
    """Check if file matches ignore patterns"""
    import fnmatch
    for pattern in IGNORE_PATTERNS:
        if fnmatch.fnmatch(filename, pattern):
            return True
        # Also check if file is in ignored directory
        if "/" in pattern and fnmatch.fnmatch(filename, pattern.replace("/*", "/**")):
            return True
    return False


# Embedding model detection patterns
EMBEDDING_PATTERNS = ["embed", "gte-", "bge-", "e5-", "mxbai-embed", "nomic-embed"]


def _is_embedding_model(repo_id: str) -> bool:
    """Check if repo is an embedding model based on name"""
    name = repo_id.lower()
    return any(p in name for p in EMBEDDING_PATTERNS)


def _search_gguf_version(repo_id: str) -> Optional[str]:
    """Search HuggingFace for GGUF version of a model"""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        
        # Extract model name from repo_id (e.g., "mlx-community/Qwen3-Embedding-8B-4bit-DWQ" -> "Qwen3-Embedding-8B")
        model_name = repo_id.split("/")[-1]
        # Remove common suffixes
        for suffix in ["-4bit-DWQ", "-4bit", "-8bit", "-DWQ", "-mlx", "-MLX"]:
            model_name = model_name.replace(suffix, "")
        
        # Search for GGUF versions
        search_query = f"{model_name} GGUF"
        results = api.list_models(search=search_query, limit=10, sort="downloads", direction=-1)
        
        for model in results:
            model_id = model.id.lower()
            # Check if it's a GGUF version (contains gguf in name or org)
            if "gguf" in model_id and any(p in model_id for p in EMBEDDING_PATTERNS):
                # Prefer certain orgs
                preferred_orgs = ["lmstudio-community", "bartowski", "nomic-ai", "sentence-transformers"]
                org = model.id.split("/")[0].lower()
                if org in preferred_orgs or "gguf" in model.id.lower():
                    return model.id
        
        # If no preferred org found, return first GGUF result
        for model in results:
            if "gguf" in model.id.lower():
                return model.id
        
        return None
    except Exception as e:
        print(f"GGUF search failed: {e}")
        return None


def _download_worker(repo_id: str, local_dir: str, status_queue: multiprocessing.Queue, token: Optional[str] = None):
    import os
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    
    from huggingface_hub import snapshot_download
    
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            token=token,
            ignore_patterns=IGNORE_PATTERNS,
        )
        status_queue.put(("completed", None))
    except Exception as e:
        status_queue.put(("failed", str(e)[:200]))


class DownloadStatus(str, Enum):
    PENDING = "pending"
    DOWNLOADING = "downloading"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class DownloadTask:
    repo_id: str
    status: DownloadStatus = DownloadStatus.PENDING
    progress: int = 0
    current_size: int = 0
    total_size: Optional[int] = None
    speed: float = 0.0
    error: Optional[str] = None
    _process: Optional[multiprocessing.Process] = field(default=None, repr=False)
    _status_queue: Optional[multiprocessing.Queue] = field(default=None, repr=False)
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None  # When download actually started
    _restart_count: int = field(default=0, repr=False)  # Track auto-restarts
    
    @property
    def model_name(self) -> str:
        return self.repo_id.split("/")[-1] if "/" in self.repo_id else self.repo_id
    
    @property
    def local_path(self) -> Path:
        models_dir = Path.home() / ".lmstudio" / "models"
        return models_dir / self.repo_id.replace("/", "/")
    
    @property
    def elapsed_seconds(self) -> Optional[float]:
        """Time since download started"""
        if self.started_at and self.status == DownloadStatus.DOWNLOADING:
            return time.time() - self.started_at
        return None
    
    @property
    def avg_speed(self) -> Optional[float]:
        """Average speed since start (bytes/sec)"""
        elapsed = self.elapsed_seconds
        if elapsed and elapsed > 0 and self.current_size > 0:
            return self.current_size / elapsed
        return None
    
    @property
    def eta_seconds(self) -> Optional[float]:
        """Estimated seconds remaining"""
        if self.total_size and self.speed and self.speed > 0:
            remaining = self.total_size - self.current_size
            if remaining > 0:
                return remaining / self.speed
        return None
    
    def to_dict(self) -> dict:
        return {
            "repo_id": self.repo_id,
            "model_name": self.model_name,
            "status": self.status.value,
            "progress": self.progress,
            "current_size": self.current_size,
            "total_size": self.total_size,
            "speed": self.speed,
            "avg_speed": self.avg_speed,
            "elapsed": self.elapsed_seconds,
            "eta": self.eta_seconds,
            "error": self.error,
            "path": str(self.local_path),
        }


# Get system RAM to determine appropriate model presets
def _get_system_ram_gb() -> int:
    """Get system RAM in GB"""
    try:
        import subprocess
        result = subprocess.run(['sysctl', '-n', 'hw.memsize'], capture_output=True, text=True)
        return int(result.stdout.strip()) // (1024**3)
    except Exception:
        return 16  # Default assumption

SYSTEM_RAM_GB = _get_system_ram_gb()

# Model presets organized by category - filtered by system RAM
_ALL_PRESETS = {
    "coding": [
        {"repo": "lmstudio-community/Qwen3-Coder-Next-MLX-4bit", "name": "Qwen3 Coder Next 80B", "size": "42GB", "ram": 48},
        {"repo": "mlx-community/Codestral-22B-v0.1-4bit", "name": "Codestral 22B", "size": "13GB", "ram": 20},
    ],
    "general": [
        {"repo": "mlx-community/Qwen3.5-35B-A3B-4bit", "name": "Qwen3.5 35B A3B 4bit", "size": "~7GB", "ram": 12},
        {"repo": "mlx-community/Qwen3.5-35B-A3B-8bit", "name": "Qwen3.5 35B A3B 8bit", "size": "~10GB", "ram": 16},
        {"repo": "mlx-community/Qwen3.5-122B-A10B-4bit", "name": "Qwen3.5 122B A10B", "size": "~20GB", "ram": 24},
    ],
    "reasoning": [
        {"repo": "mlx-community/DeepSeek-R1-Distill-Llama-70B-4bit", "name": "DeepSeek R1 70B", "size": "40GB", "ram": 48},
    ],
    "embedding": [
        {"repo": "mlx-community/Qwen3-Embedding-8B-4bit-DWQ", "name": "Qwen3 Embed 8B", "size": "4.5GB", "ram": 8},
        {"repo": "mlx-community/gte-Qwen2-7B-instruct-4bit-DWQ", "name": "GTE Qwen2 7B", "size": "4.5GB", "ram": 8},
        {"repo": "mixedbread-ai/mxbai-embed-large-v1", "name": "MixedBread Embed", "size": "1.3GB", "ram": 4},
    ],
}

def get_preset_models() -> list:
    """Get model presets filtered by system RAM"""
    presets = []
    for category, models in _ALL_PRESETS.items():
        for model in models:
            if model["ram"] <= SYSTEM_RAM_GB * 0.8:  # Allow models up to 80% of RAM
                presets.append({
                    "repo": model["repo"],
                    "name": model["name"],
                    "size": model["size"],
                    "category": category,
                })
    return presets

# For backward compatibility
PRESET_MODELS = get_preset_models()


class DownloadManager:
    """Manages download queue and background downloads"""
    
    MAX_CONCURRENT = 3  # Maximum parallel downloads
    
    def __init__(self, models_dir: Optional[Path] = None):
        self.models_dir = models_dir or Path.home() / ".lmstudio" / "models"
        self.queue: list[DownloadTask] = []
        self.active_tasks: list[DownloadTask] = []  # Currently downloading (up to MAX_CONCURRENT)
        self._lock = threading.Lock()
        self._monitor_thread: Optional[threading.Thread] = None
        self._running = False
        self._speed_history: dict = {}  # repo_id -> {history: [...], last_speed: float, last_change: float}
    
    def start(self):
        """Start the download manager background thread"""
        if self._running:
            return
        self._load_queue()  # Restore queue from disk
        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
    
    def stop(self):
        """Stop the download manager"""
        self._running = False
        for task in self.active_tasks:
            if task._process:
                task._process.terminate()
    
    def add_download(self, repo_id: str) -> DownloadTask:
        """Add a model to the download queue"""
        with self._lock:
            # Check if already in queue or downloading
            for task in self.queue:
                if task.repo_id == repo_id:
                    return task
            for task in self.active_tasks:
                if task.repo_id == repo_id:
                    return task
            
            # Check if already downloaded
            model_path = self.models_dir / repo_id.replace("/", "/")
            if model_path.exists() and not list(model_path.rglob("*.incomplete")):
                task = DownloadTask(repo_id=repo_id, status=DownloadStatus.COMPLETED)
                return task
            
            task = DownloadTask(repo_id=repo_id)
            self.queue.append(task)
            self._save_queue()
            return task
    
    def remove_download(self, repo_id: str, delete_files: bool = False) -> bool:
        """Remove a download from queue or cancel active download"""
        with self._lock:
            # Check queue
            for i, task in enumerate(self.queue):
                if task.repo_id == repo_id:
                    self.queue.pop(i)
                    if delete_files:
                        self._delete_model_files(repo_id)
                    self._save_queue()
                    return True
            
            # Check active tasks
            for i, task in enumerate(self.active_tasks):
                if task.repo_id == repo_id:
                    if task._process:
                        task._process.terminate()
                    task.status = DownloadStatus.CANCELLED
                    if delete_files:
                        self._delete_model_files(repo_id)
                    self.active_tasks.pop(i)
                    self._save_queue()
                    return True
            
            return False
    
    def pause_download(self, repo_id: str) -> bool:
        """Pause active download (terminates process, HF will resume from checkpoint)"""
        with self._lock:
            for i, task in enumerate(self.active_tasks):
                if task.repo_id == repo_id:
                    # Terminate the download process
                    if task._process:
                        task._process.terminate()
                    task.status = DownloadStatus.PAUSED
                    task._process = None
                    task._status_queue = None
                    # Move to front of queue
                    self.active_tasks.pop(i)
                    self.queue.insert(0, task)
                    self._save_queue()
                    return True
            return False
    
    def resume_download(self, repo_id: str) -> bool:
        """Resume a paused download (moves to front of queue)"""
        with self._lock:
            for i, task in enumerate(self.queue):
                if task.repo_id == repo_id and task.status == DownloadStatus.PAUSED:
                    task.status = DownloadStatus.PENDING
                    # Move to front
                    self.queue.pop(i)
                    self.queue.insert(0, task)
                    self._save_queue()
                    return True
            return False
    
    def get_status(self) -> dict:
        """Get download status: active downloads and queue"""
        with self._lock:
            active = [t.to_dict() for t in self.active_tasks]
            queue = [t.to_dict() for t in self.queue]
            return {
                "active": active,
                "current": active[0] if active else None,  # Backward compat
                "queue": queue,
                "presets": PRESET_MODELS,
            }
    
    def _delete_model_files(self, repo_id: str):
        """Delete model files including incomplete downloads"""
        model_path = self.models_dir / repo_id.replace("/", "/")
        if model_path.exists():
            try:
                shutil.rmtree(model_path)
            except Exception:
                pass
    
    def _save_queue(self):
        """Persist queue to disk"""
        try:
            data = {
                "queue": [
                    {"repo_id": t.repo_id, "status": t.status.value}
                    for t in self.queue
                ],
                "active": [t.repo_id for t in self.active_tasks]
            }
            QUEUE_FILE.write_text(json.dumps(data, indent=2))
        except Exception:
            pass
    
    def _load_queue(self):
        """Restore queue from disk"""
        if not QUEUE_FILE.exists():
            return
        try:
            data = json.loads(QUEUE_FILE.read_text())
            # Restore queue items
            for item in data.get("queue", []):
                repo_id = item.get("repo_id")
                if repo_id:
                    task = DownloadTask(repo_id=repo_id)
                    # If was paused, keep paused status
                    if item.get("status") == "paused":
                        task.status = DownloadStatus.PAUSED
                    self.queue.append(task)
            # If there were active downloads, add them back to front of queue
            for repo_id in data.get("active", []):
                if repo_id and repo_id not in [t.repo_id for t in self.queue]:
                    self.queue.insert(0, DownloadTask(repo_id=repo_id))
            # Backward compat: handle old "current" field
            current = data.get("current")
            if current and current not in [t.repo_id for t in self.queue]:
                self.queue.insert(0, DownloadTask(repo_id=current))
        except Exception:
            pass
    
    def _start_download(self, task: DownloadTask):
        """Start downloading a model in a separate process (can be terminated)"""
        task.status = DownloadStatus.DOWNLOADING
        task.error = None
        if task.started_at is None:
            task.started_at = time.time()  # Only set on first start, not resume
        
        # Create a queue for the process to report status back
        task._status_queue = multiprocessing.Queue()
        task._process = multiprocessing.Process(
            target=_download_worker,
            args=(task.repo_id, str(task.local_path), task._status_queue, HF_TOKEN),
            daemon=True
        )
        task._process.start()
    
    def _update_progress(self, task: DownloadTask):
        """Update download progress by checking file sizes"""
        import os
        if not task.local_path.exists():
            return
        
        try:
            completed_size = 0
            incomplete_size = 0
            incomplete_files = []
            
            for f in task.local_path.rglob("*"):
                if f.is_file():
                    if f.suffix in (".aria2", ):
                        continue
                    
                    if ".cache" in f.parts and f.suffix == ".incomplete":
                        try:
                            fd = os.open(str(f), os.O_RDONLY)
                            stat = os.fstat(fd)
                            os.close(fd)
                            size = stat.st_size
                        except OSError:
                            size = f.stat().st_size
                        incomplete_files.append((f, size))
                        incomplete_size += size
                    else:
                        completed_size += f.stat().st_size
            
            task.current_size = completed_size + incomplete_size
            
            # Track speed using history
            current_time = time.time()
            repo = task.repo_id
            if repo not in self._speed_history:
                self._speed_history[repo] = {"history": [], "last_speed": 0.0, "last_change": current_time}
            
            speed_data = self._speed_history[repo]
            history = speed_data["history"]
            history.append((current_time, task.current_size))
            
            # Keep last 15 seconds of history
            history = [(t, s) for t, s in history if current_time - t < 15]
            speed_data["history"] = history
            
            # Calculate speed from history
            if len(history) >= 2:
                # Use oldest and newest for smoother average
                time_diff = history[-1][0] - history[0][0]
                size_diff = history[-1][1] - history[0][1]
                
                if time_diff > 0 and size_diff > 0:
                    # Real progress detected
                    speed_data["last_speed"] = size_diff / time_diff
                    speed_data["last_change"] = current_time
                    task.speed = speed_data["last_speed"]
                elif current_time - speed_data["last_change"] < 30:
                    # No change but recent progress - keep last known speed (stat caching)
                    task.speed = speed_data["last_speed"]
                else:
                    # No progress for 30+ seconds - likely stalled
                    task.speed = 0.0
            
            # Estimate total from index file or HF API
            if not task.total_size:
                index_file = task.local_path / "model.safetensors.index.json"
                if index_file.exists():
                    with open(index_file) as f:
                        index = json.load(f)
                    weight_map = index.get("weight_map", {})
                    num_shards = len(set(weight_map.values()))
                    # Estimate based on 4-bit quantization (~4.5GB per shard)
                    task.total_size = int(num_shards * 4.5 * 1024 * 1024 * 1024)
                elif incomplete_files:
                    # For small models, try to get total from Content-Length if available
                    # Fallback: estimate as 2x current incomplete size when >50% likely done
                    pass
            
            # Calculate progress
            if task.total_size and task.total_size > 0:
                task.progress = min(99, int((task.current_size / task.total_size) * 100))
            elif task.current_size > 0:
                # No total size - show indeterminate progress based on file count
                task.progress = min(50, len(incomplete_files) * 10) if incomplete_files else 0
            
            # Check if process is done
            if task._process and not task._process.is_alive():
                # Check status from the queue
                try:
                    if task._status_queue and not task._status_queue.empty():
                        status, error = task._status_queue.get_nowait()
                        if status == "completed":
                            task.status = DownloadStatus.COMPLETED
                            task.progress = 100
                        elif status == "failed":
                            task.status = DownloadStatus.FAILED
                            task.error = error
                except Exception:
                    # Process died without reporting - check if files exist
                    incomplete = list(task.local_path.rglob("*.incomplete"))
                    if not incomplete and task.local_path.exists():
                        task.status = DownloadStatus.COMPLETED
                        task.progress = 100
                    else:
                        task.status = DownloadStatus.FAILED
                        task.error = "Download process terminated unexpectedly"
                    
        except Exception:
            pass
    
    def _monitor_loop(self):
        """Background loop to manage parallel downloads"""
        while self._running:
            with self._lock:
                # Update progress for all active downloads
                for task in self.active_tasks[:]:  # Copy list to allow modification
                    self._update_progress(task)
                    
                    # Detect stalled downloads and auto-restart
                    if (task.status == DownloadStatus.DOWNLOADING and 
                        task._process and task._process.is_alive()):
                        speed_data = self._speed_history.get(task.repo_id, {})
                        last_change = speed_data.get("last_change", time.time())
                        stall_duration = time.time() - last_change
                        
                        # If stalled for >60s and haven't restarted too many times
                        if stall_duration > 60 and task._restart_count < 10:
                            print(f"⚠️ Download stalled for {stall_duration:.0f}s, restarting: {task.model_name}")
                            task._process.terminate()
                            task._process.join(timeout=5)
                            task._process = None
                            task._status_queue = None
                            task._restart_count += 1
                            # Clear speed history for fresh start
                            self._speed_history.pop(task.repo_id, None)
                            # Restart download
                            self._start_download(task)
                    
                    # Check if task is done
                    if task.status in (
                        DownloadStatus.COMPLETED,
                        DownloadStatus.FAILED,
                        DownloadStatus.CANCELLED,
                    ):
                        # Clean up
                        self._speed_history.pop(task.repo_id, None)
                        self.active_tasks.remove(task)
                        self._save_queue()
                
                # Start more downloads if under limit
                while len(self.active_tasks) < self.MAX_CONCURRENT and self.queue:
                    # Find first non-paused task
                    started = False
                    for i, task in enumerate(self.queue):
                        if task.status != DownloadStatus.PAUSED:
                            task = self.queue.pop(i)
                            self.active_tasks.append(task)
                            self._start_download(task)
                            self._save_queue()
                            started = True
                            break
                    if not started:
                        break  # Only paused tasks remain
            
            time.sleep(1)


# Global instance
download_manager = DownloadManager()
