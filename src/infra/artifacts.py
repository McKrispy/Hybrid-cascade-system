"""Run artifacts and execution output management.

Timestamp: 2026-11-09 11:05 (Asia/Shanghai)

Handles creation of unique non-overwriting run directories:
dev/runs/<batch-id>/<run-id>/
Saves config_snapshot.json, results.json, console.log, audit.json,
cost_ledger.json, and request_index.json with strict secret desensitization.
"""

import json
import logging
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from src.infra.config import assert_no_secrets
from src.infra.paths import get_runs_dir


def allocate_run_dir(batch_id: str, run_id: Optional[str] = None) -> Path:
    """Create and return a new non-overwriting run directory.
    
    If run_id is omitted or already exists, allocates the next candidate ID (e.g. B00-C01, B00-C02).
    """
    batch_dir = get_runs_dir() / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)

    if run_id:
        target_dir = batch_dir / run_id
        if not target_dir.exists():
            target_dir.mkdir(parents=True, exist_ok=False)
            return target_dir
        # If exists, append numerical suffix to prevent overwriting
        idx = 2
        while True:
            candidate = batch_dir / f"{run_id}_v{idx}"
            if not candidate.exists():
                candidate.mkdir(parents=True, exist_ok=False)
                return candidate
            idx += 1
    else:
        # Default candidate numbering: C01, C02, ...
        idx = 1
        while True:
            candidate = batch_dir / f"{batch_id}-C{idx:02d}"
            if not candidate.exists():
                candidate.mkdir(parents=True, exist_ok=False)
                return candidate
            idx += 1


class RunArtifacts:
    """Manages files and logs for a single run execution."""

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.console_log_path = run_dir / "console.log"
        self._setup_logger()

    def _setup_logger(self) -> None:
        self.logger = logging.getLogger(f"run_{self.run_dir.name}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        
        # Avoid duplicate handlers if re-instantiated
        if not self.logger.handlers:
            fh = logging.FileHandler(self.console_log_path, encoding="utf-8")
            fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
            fh.setFormatter(fmt)
            self.logger.addHandler(fh)

            ch = logging.StreamHandler(sys.stdout)
            ch.setFormatter(fmt)
            self.logger.addHandler(ch)

    def log(self, message: str, level: int = logging.INFO) -> None:
        self.logger.log(level, message)

    def save_config_snapshot(self, config_dict: Dict[str, Any]) -> Path:
        """Save immutable config snapshot for the run."""
        assert_no_secrets(config_dict)
        path = self.run_dir / "config_snapshot.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        return path

    def save_results(self, results_dict: Dict[str, Any]) -> Path:
        """Save results.json for the run."""
        assert_no_secrets(results_dict)
        path = self.run_dir / "results.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)
        return path

    def save_audit(self, audit_dict: Dict[str, Any]) -> Path:
        """Save audit.json including system platform, python version, and deterministic settings."""
        assert_no_secrets(audit_dict)
        system_info = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "os": platform.platform(),
            "python_version": sys.version,
            "machine": platform.machine(),
            "processor": platform.processor(),
        }
        full_audit = {**system_info, **audit_dict}
        path = self.run_dir / "audit.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(full_audit, f, indent=2, ensure_ascii=False)
        return path

    def save_cost_ledger(self, ledger_dict: Dict[str, Any]) -> Path:
        """Save cost_ledger.json distinguishing research calls, cached calls, and alias reuse."""
        assert_no_secrets(ledger_dict)
        path = self.run_dir / "cost_ledger.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(ledger_dict, f, indent=2, ensure_ascii=False)
        return path

    def save_request_index(self, index_dict: Dict[str, Any]) -> Path:
        """Save request_index.json mapping sample requests to predictions or aliases."""
        assert_no_secrets(index_dict)
        path = self.run_dir / "request_index.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(index_dict, f, indent=2, ensure_ascii=False)
        return path

    def save_predictions(self, predictions_list: list) -> Path:
        """Save predictions.json."""
        assert_no_secrets(predictions_list)
        path = self.run_dir / "predictions.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(predictions_list, f, indent=2, ensure_ascii=False)
        return path

