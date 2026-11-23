"""Inference caching and resumption mechanism for HybridCascade.

Timestamp: 2026-11-09 13:55 (Asia/Shanghai)

Provides disk-persisted, deterministic SHA-256 request hashing to support
instant resumption, eliminate redundant API calls, and desensitize entries.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from pydantic import BaseModel

from src.clients.base import Prediction
from src.infra.paths import get_runs_dir


def compute_request_fingerprint(
    dataset_id: str,
    role: str,
    sample_id: str,
    model_id: str,
    arm_id: str,
    prompt_payload: Any,
    generation_params: Optional[Dict[str, Any]] = None,
    execution_mode: str = "live",
) -> str:
    """Compute deterministic SHA-256 fingerprint representing the exact request."""
    norm_params = json.dumps(generation_params or {}, sort_keys=True)
    norm_prompt = json.dumps(prompt_payload, sort_keys=True, ensure_ascii=False) if isinstance(prompt_payload, (dict, list)) else str(prompt_payload).strip()
    
    components = [
        f"dataset:{dataset_id}",
        f"role:{role}",
        f"sample:{sample_id}",
        f"model:{model_id}",
        f"arm:{arm_id}",
        f"mode:{execution_mode}",
        f"params:{norm_params}",
        f"payload:{norm_prompt}",
    ]
    raw_str = "|".join(components)
    return hashlib.sha256(raw_str.encode("utf-8")).hexdigest()


class InferenceCache:
    """Thread-safe disk-backed prediction cache for inference resumption."""

    def __init__(self, cache_dir: Optional[Path] = None):
        if cache_dir is None:
            cache_dir = get_runs_dir() / ".cache"
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.cache_dir / "index.jsonl"
        self._memory_cache: Dict[str, Prediction] = {}
        self._sample_cache: Dict[Tuple[str, str, str, str], Prediction] = {}
        self._load_cache()
        self._load_historical_runs()

    def _load_cache(self) -> None:
        """Load existing cached predictions from disk."""
        if not self.index_file.exists():
            return
        with open(self.index_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    fp = entry.get("fingerprint")
                    pred_data = entry.get("prediction")
                    if fp and pred_data:
                        pred = Prediction(**pred_data)
                        self._memory_cache[fp] = pred
                        if pred.status == "valid":
                            is_mock = pred.raw_response.get("mock", False) if isinstance(pred.raw_response, dict) else False
                            mode = "mock" if is_mock else "live"
                            self._sample_cache[(pred.sample_id, pred.model_id, pred.arm_id, mode)] = pred
                except Exception:
                    continue

    def _load_historical_runs(self) -> None:
        """Prime sample cache with valid predictions from completed runs."""
        runs_dir = get_runs_dir()
        for pred_file in runs_dir.glob("B02/*/predictions.json"):
            audit_file = pred_file.parent / "audit.json"
            run_mode = "live"
            if audit_file.exists():
                try:
                    with open(audit_file, "r", encoding="utf-8") as af:
                        audit = json.load(af)
                    run_mode = audit.get("mode", "live")
                except Exception:
                    pass
            try:
                with open(pred_file, "r", encoding="utf-8") as f:
                    preds = json.load(f)
                for p_dict in preds:
                    if p_dict.get("status") == "valid":
                        pred = Prediction(**p_dict)
                        is_mock = (run_mode == "mock") or (pred.raw_response.get("mock", False) if isinstance(pred.raw_response, dict) else False)
                        mode = "mock" if is_mock else "live"
                        key = (pred.sample_id, pred.model_id, pred.arm_id, mode)
                        if key not in self._sample_cache:
                            self._sample_cache[key] = pred
            except Exception:
                continue

    def get(self, fingerprint: str) -> Optional[Prediction]:
        """Look up a prediction by its exact request fingerprint."""
        return self._memory_cache.get(fingerprint)

    def get_by_sample(self, sample_id: str, model_id: str, arm_id: str, mode: str = "live") -> Optional[Prediction]:
        """Look up an existing valid prediction by sample_id, model_id, arm_id, and mode."""
        return self._sample_cache.get((sample_id, model_id, arm_id, mode))

    def put(self, fingerprint: str, prediction: Prediction, mode: str = "live") -> None:
        """Store a prediction in memory and append to disk cache."""
        self._memory_cache[fingerprint] = prediction
        if prediction.status == "valid":
            is_mock = (mode == "mock") or (prediction.raw_response.get("mock", False) if isinstance(prediction.raw_response, dict) else False)
            actual_mode = "mock" if is_mock else "live"
            self._sample_cache[(prediction.sample_id, prediction.model_id, prediction.arm_id, actual_mode)] = prediction
        entry = {
            "fingerprint": fingerprint,
            "prediction": prediction.model_dump(),
        }
        with open(self.index_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def __len__(self) -> int:
        return len(self._memory_cache)

