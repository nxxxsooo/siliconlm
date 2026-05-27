"""Hot Model Scanner — dynamic HuggingFace MLX model discovery"""

import json
import os
import re
import time
from pathlib import Path
from typing import Optional


try:
    from huggingface_hub import HfApi
except ImportError:
    HfApi = None  # Graceful degradation if huggingface_hub not installed


# Model families to scan: (author, search_pattern, category)
MODEL_FAMILIES = [
    ("mlx-community", "Qwen3.6-35B-A3B", "general"),
    ("mlx-community", "Qwen3-Coder-Next", "coding"),
    ("mlx-community", "Qwen3.6-122B", "reasoning"),
    ("mlx-community", "Qwen3-Embedding", "embedding"),
    ("mlx-community", "gemma-4-31b", "general"),
    ("mlx-community", "gemma-4-26b-a4b", "general"),
    ("mlx-community", "DeepSeek-R1", "reasoning"),
]

# Quant type extraction patterns (order matters — more specific first)
QUANT_PATTERNS = [
    (r"mxfp4", "mxfp4"),
    (r"nvfp4", "nvfp4"),
    (r"(\d+)bit", lambda m: f"{m.group(1)}bit"),
    (r"bf16", "bf16"),
]

STATE_DIR = Path(os.path.expanduser("~/.local/share/siliconlm"))
STATE_FILE = STATE_DIR / "hot_models_state.json"


def _extract_quant_type(model_id: str) -> str:
    """Extract quantization type from model ID (e.g. '8bit', 'bf16', 'mxfp4')."""
    name = model_id.lower().split("/")[-1]
    for pattern, replacement in QUANT_PATTERNS:
        m = re.search(pattern, name)
        if m:
            if callable(replacement):
                return replacement(m)
            return replacement
    return "unknown"


def _extract_family_name(repo_id: str, search: str) -> str:
    """Extract a display-friendly family name from repo_id."""
    name = repo_id.split("/")[-1]
    for p, _ in QUANT_PATTERNS:
        name = re.sub(p, "", name, flags=re.IGNORECASE)
    # Clean up trailing dashes/underscores
    name = re.sub(r"[-_]+$", "", name)
    return name or search


def _count_safetensors(siblings: Optional[list]) -> int:
    """Count .safetensors files in model siblings."""
    if not siblings:
        return 0
    return sum(1 for s in siblings if s.get("filename", "").endswith(".safetensors"))


class HotModelScanner:
    """Scans HuggingFace for MLX model variants and tracks new releases."""

    def __init__(self, state_path: Optional[str] = None):
        self.state_path = Path(state_path) if state_path else STATE_FILE
        self._known_ids: set = set()
        self._last_scan: float = 0.0
        self._load_state()

    def _load_state(self):
        """Load known model IDs and last scan timestamp from disk."""
        try:
            if self.state_path.exists():
                data = json.loads(self.state_path.read_text())
                self._known_ids = set(data.get("known_ids", []))
                self._last_scan = data.get("last_scan", 0.0)
        except Exception:
            self._known_ids = set()
            self._last_scan = 0.0

    def _save_state(self):
        """Persist known IDs and timestamp to disk."""
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(
                json.dumps(
                    {
                        "known_ids": sorted(self._known_ids),
                        "last_scan": self._last_scan,
                    },
                    indent=2,
                )
            )
        except Exception:
            pass

    def scan(self, families: Optional[list] = None) -> list:
        """Scan HF for model families, return sorted list of model dicts.

        Returns list of dicts:
            {family, repo_id, name, quant, downloads, file_count, is_new, url, category}
        """
        if HfApi is None:
            return []

        families = families or MODEL_FAMILIES
        results = []
        now = time.time()

        try:
            api = HfApi()
            for author, search, category in families:
                try:
                    models = api.list_models(
                        author=author,
                        search=search,
                        sort="last_modified",
                        direction=-1,
                        expand=["lastModified", "siblings", "tags", "downloads"],
                    )
                    for model in models:
                        repo_id = getattr(model, "id", "")
                        if not repo_id:
                            continue

                        # Filter: must be mlx (check tags or safetensors files)
                        tags = getattr(model, "tags", []) or []
                        siblings = getattr(model, "siblings", []) or []
                        is_mlx = any("mlx" in str(t).lower() for t in tags)
                        has_safetensors = _count_safetensors(siblings) > 0

                        if not is_mlx and not has_safetensors:
                            continue

                        # Skip GGUF-only models
                        sf_list = [
                            s.get("filename", "").lower() for s in (siblings or [])
                        ]
                        if all(f.endswith(".gguf") for f in sf_list if f):
                            continue

                        repo_id_lower = repo_id.lower()
                        if "gguf" in repo_id_lower:
                            continue

                        downloads = getattr(model, "downloads", 0) or 0
                        quant = _extract_quant_type(repo_id)
                        family_name = _extract_family_name(repo_id, search)
                        is_new = repo_id not in self._known_ids

                        results.append(
                            {
                                "family": family_name,
                                "repo_id": repo_id,
                                "name": repo_id.split("/")[-1],
                                "quant": quant,
                                "downloads": downloads,
                                "file_count": len(siblings) if siblings else 0,
                                "is_new": is_new,
                                "url": f"https://huggingface.co/{repo_id}",
                                "category": category,
                                "last_modified": getattr(model, "lastModified", None),
                            }
                        )
                except Exception:
                    # Continue scanning other families even if one fails
                    continue

            # Mark all discovered IDs as known
            for r in results:
                self._known_ids.add(r["repo_id"])
            self._last_scan = now
            self._save_state()

            # Sort: new first, then by downloads descending
            results.sort(key=lambda x: (not x["is_new"], -x["downloads"]))
            return results

        except Exception:
            return []


# Global instance
hot_scanner = HotModelScanner()


if __name__ == "__main__":
    import sys

    print("Scanning HuggingFace for MLX models...")
    results = hot_scanner.scan()
    if not results:
        print("No models found (HF may be unreachable)")
        sys.exit(0)

    # Group by category
    by_cat = {}
    for r in results:
        cat = r["category"]
        if cat not in by_cat:
            by_cat[cat] = []
        by_cat[cat].append(r)

    for cat, models in by_cat.items():
        print(f"\n{'=' * 40}")
        print(f"  [{cat.upper()}] — {len(models)} models")
        print(f"{'=' * 40}")
        for m in models:
            badge = " 🔥 NEW" if m["is_new"] else ""
            print(
                f"  [{m['quant']:8s}] {m['name']:45s}  downloads={m['downloads']:,}{badge}"
            )

    print(f"\nTotal: {len(results)} models discovered")
    print(f"State saved to: {hot_scanner.state_path}")
