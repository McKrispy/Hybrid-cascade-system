"""Base definitions and contract for API clients.

Timestamp: 2026-11-09 11:05 (Asia/Shanghai)

Defines the unified Prediction schema, Handoff schema, and abstract interfaces.
Enforces request/response desensitization and ensures API secrets never leak.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class Handoff(BaseModel):
    """Handoff specification passed from System-1 (or baseline) to System-2."""
    arm_id: str  # "F0", "F1", "C0", "C1", or "FR"
    allowed_labels: List[str]
    prior_payload: Optional[Dict[str, Any]] = None  # Jev Top-2 rank, label, p
    reject_allowed: bool = False


class Prediction(BaseModel):
    """Unified prediction record across all models and arms."""
    model_config = {"protected_namespaces": ()}
    sample_id: str
    dataset_id: Optional[str] = None
    request_id: str
    role: str  # "synthetic", "calibration", "selection", "holdout"
    model_id: str  # Requested model ID
    returned_model: str  # Actual returned model from API response
    arm_id: str  # "JEV", "F0", "F1", "C0", "C1", "FR"
    template_version: str = "v1.0"
    decision: Optional[str] = None
    probabilities: Dict[str, float] = Field(default_factory=dict)
    usage: Dict[str, int] = Field(default_factory=dict)  # prompt_tokens, completion_tokens, total_tokens
    cost: float = 0.0
    latency_ms: float = 0.0
    attempts: int = 1
    status: str = "valid"  # "valid", "rejected", "failed"
    error_code: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None  # Desensitized response summary


def desensitize_dict(d: Any) -> Any:
    """Recursively clean sensitive auth headers or keys from dictionaries."""
    if isinstance(d, dict):
        clean = {}
        for k, v in d.items():
            if k.lower() in ("authorization", "api-key", "api_key", "typesafe_api_key", "token", "password"):
                clean[k] = "[REDACTED]"
            else:
                clean[k] = desensitize_dict(v)
        return clean
    elif isinstance(d, list):
        return [desensitize_dict(item) for item in d]
    return d
