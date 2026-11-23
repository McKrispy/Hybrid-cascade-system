"""Telemetry and progress tracking for execution runs.

Timestamp: 2026-11-09 11:05 (Asia/Shanghai)

Provides atomic progress.json updates and metrics.jsonl appending
to support live monitoring and Panel observability.
"""

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

from src.infra.config import assert_no_secrets


class ProgressData(BaseModel):
    batch_id: str
    run_id: str
    phase: str
    status: str = "not_started"  # not_started, running, completed, failed, paused, skipped
    planned: int = 0
    completed: int = 0
    valid: int = 0
    failed: int = 0
    attempts: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    updated_at: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    config_version: str = "1.0"


class TelemetryTracker:
    """Manages telemetry progress and metric logging for a single run."""

    def __init__(self, run_dir: Path, batch_id: str, run_id: str, config_version: str = "1.0"):
        self.run_dir = run_dir
        self.batch_id = batch_id
        self.run_id = run_id
        self.config_version = config_version
        self.progress_file = run_dir / "progress.json"
        self.metrics_file = run_dir / "metrics.jsonl"
        self.current_progress = ProgressData(
            batch_id=batch_id,
            run_id=run_id,
            phase="initialization",
            status="not_started",
            config_version=config_version,
        )
        self.save_progress()

    def update_progress(
        self,
        phase: Optional[str] = None,
        status: Optional[str] = None,
        planned: Optional[int] = None,
        completed: Optional[int] = None,
        valid: Optional[int] = None,
        failed: Optional[int] = None,
        attempts: Optional[int] = None,
        total_tokens: Optional[int] = None,
        total_cost_usd: Optional[float] = None,
    ) -> None:
        """Update progress fields and atomically write to progress.json."""
        if phase is not None:
            self.current_progress.phase = phase
        if status is not None:
            self.current_progress.status = status
        if planned is not None:
            self.current_progress.planned = planned
        if completed is not None:
            self.current_progress.completed = completed
        if valid is not None:
            self.current_progress.valid = valid
        if failed is not None:
            self.current_progress.failed = failed
        if attempts is not None:
            self.current_progress.attempts = attempts
        if total_tokens is not None:
            self.current_progress.total_tokens = total_tokens
        if total_cost_usd is not None:
            self.current_progress.total_cost_usd = total_cost_usd

        self.current_progress.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.save_progress()

    def save_progress(self) -> None:
        """Atomically write current_progress to progress.json using replace."""
        data = self.current_progress.model_dump()
        assert_no_secrets(data)
        
        # Write to temp file in same directory first to ensure atomic rename
        temp_fd, temp_path = tempfile.mkstemp(dir=self.run_dir, prefix="prog_", suffix=".tmp")
        try:
            with os.fdopen(temp_fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(temp_path, self.progress_file)
        except Exception:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise

    def log_metric(self, event_type: str, data: Dict[str, Any]) -> None:
        """Append an event or metric record to metrics.jsonl."""
        assert_no_secrets(data)
        record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "event_type": event_type,
            **data,
        }
        with open(self.metrics_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

