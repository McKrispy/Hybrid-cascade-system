"""Configuration management and validation for HybridCascade.

Timestamp: 2026-11-09 11:05 (Asia/Shanghai)

Ensures parameters are loaded strictly from JSON configurations, validates schemas,
and strictly enforces that secrets and API keys never leak into configs or snapshots.
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from src.infra.paths import get_configs_dir


class RateLimitConfig(BaseModel):
    max_concurrency: int = Field(default=5, ge=1)
    timeout_seconds: float = Field(default=30.0, gt=0)
    max_retries: int = Field(default=3, ge=0)
    backoff_factor: float = Field(default=1.5, ge=1.0)


class BudgetLimitConfig(BaseModel):
    max_cost_usd: float = Field(default=10.0, ge=0)
    max_requests: int = Field(default=50000, ge=1)


class ModelPricingConfig(BaseModel):
    input_cost_per_million: float = Field(default=0.0, ge=0)
    output_cost_per_million: float = Field(default=0.0, ge=0)


class ModelSpec(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_id: str
    provider: str
    endpoint: str
    env_key_name: str
    pricing: ModelPricingConfig
    parameters: Dict[str, Any] = Field(default_factory=dict)


class SyntheticSample(BaseModel):
    sample_id: str
    task_type: str  # "choice" or "noul"
    question: str
    options: Optional[List[str]] = None
    criteria: Optional[Dict[str, str]] = None
    gold_label: Optional[str] = None


class BatchConfig(BaseModel):
    batch_id: str
    schema_version: str = "1.0"
    description: str
    seed: int = 42
    rate_limits: RateLimitConfig = Field(default_factory=RateLimitConfig)
    budget_limits: BudgetLimitConfig = Field(default_factory=BudgetLimitConfig)
    models: Dict[str, ModelSpec] = Field(default_factory=dict)
    synthetic_samples: List[SyntheticSample] = Field(default_factory=list)
    custom_params: Dict[str, Any] = Field(default_factory=dict)


class BatchRegistration(BaseModel):
    batch_id: str
    name: str
    type: str  # "research" (Bxx) or "engineering" (Pxx)
    max_new_configs: int
    prerequisites: List[str]
    status: str  # "not_started", "in_progress", "completed", "failed", "skipped"
    config_file: Optional[str] = None
    record_file: Optional[str] = None


class ProjectConfig(BaseModel):
    schema_version: str = "1.0"
    project_name: str
    dev_root: str = "dev"
    batches: Dict[str, BatchRegistration]


SECRET_ENV_KEYS = [
    "TYPESAFE_API_KEY",
    "OPENAI_API_KEY",
    "DEEPSEEK_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
]


def get_env_var(name: str, default: Optional[str] = None) -> Optional[str]:
    """Retrieve environment variable from os.environ or Windows user/machine registry if missing."""
    val = os.environ.get(name)
    if not val and os.name == "nt":
        try:
            import winreg
            for root in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
                try:
                    sub = (
                        r"Environment"
                        if root == winreg.HKEY_CURRENT_USER
                        else r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"
                    )
                    with winreg.OpenKey(root, sub) as key:
                        v, _ = winreg.QueryValueEx(key, name)
                        if v:
                            return str(v)
                except OSError:
                    pass
        except ImportError:
            pass
    return val if val is not None else default


def assert_no_secrets(obj: Any) -> None:
    """Recursively verify that no secrets or API key values are embedded in obj."""
    env_secrets = [get_env_var(k) for k in SECRET_ENV_KEYS if get_env_var(k)]

    def _check(val: Any):
        if isinstance(val, str):
            for secret in env_secrets:
                if secret and len(secret) > 8 and secret in val:
                    raise ValueError("Security violation: environment secret found in object data!")
            if re.search(r"sk-[a-zA-Z0-9]{20,}", val):
                raise ValueError("Security violation: potential API key pattern matched in object data!")
        elif isinstance(val, dict):
            for k, v in val.items():
                _check(k)
                _check(v)
        elif isinstance(val, (list, tuple, set)):
            for item in val:
                _check(item)

    _check(obj)


def load_project_config() -> ProjectConfig:
    """Load project.json from dev/configs/."""
    path = get_configs_dir() / "project.json"
    if not path.exists():
        raise FileNotFoundError(f"Project config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert_no_secrets(data)
    return ProjectConfig(**data)


def load_batch_config(batch_id: str) -> BatchConfig:
    """Load <batch-id>.json from dev/configs/."""
    path = get_configs_dir() / f"{batch_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Batch config not found for {batch_id}: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert_no_secrets(data)
    return BatchConfig(**data)
