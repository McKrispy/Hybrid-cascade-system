"""Input loading and strict role boundary enforcement for HybridCascade.

Timestamp: 2026-11-09 13:18 (Asia/Shanghai)

Enforces that development batches (B01–B06) can ONLY load calibration or selection roles.
Attempting to load holdout data via the standard loader raises PermissionError.
"""

import json
from pathlib import Path
from typing import Generator, List, Optional, Union
from pydantic import ValidationError

from src.infra.paths import get_data_dir, get_dev_root
from src.input.splits import GoldRecord, SampleInput


ALLOWED_DEV_ROLES = {"calibration", "selection"}


def load_inputs(
    manifest_path: Optional[Union[str, Path]] = None,
    allowed_role: str = "calibration",
    dataset_id: Optional[str] = None,
    danger_enable_holdout_for_b07: bool = False,
) -> Generator[SampleInput, None, None]:
    """Load SampleInput instances for a specified data role.
    
    Strict development guard:
    - Raises PermissionError if allowed_role == 'holdout' and danger_enable_holdout_for_b07 is False.
    - Raises ValueError if allowed_role is unrecognized.
    """
    if allowed_role == "holdout" and not danger_enable_holdout_for_b07:
        raise PermissionError(
            "Access to holdout data is strictly forbidden during development (B01–B06). "
            "Only 'calibration' and 'selection' roles are permitted."
        )

    if allowed_role not in ALLOWED_DEV_ROLES and allowed_role != "holdout":
        raise ValueError(f"Invalid data role '{allowed_role}'. Expected one of {ALLOWED_DEV_ROLES} or 'holdout'")

    if manifest_path is None:
        manifest_path = get_data_dir() / "manifest.json"
    manifest_path = Path(manifest_path)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found at {manifest_path}. Has Batch B01 been executed?")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    dev_root = get_dev_root()
    datasets_meta = manifest.get("datasets", {})

    target_datasets = [dataset_id] if dataset_id else sorted(datasets_meta.keys())

    for ds_id in target_datasets:
        if ds_id not in datasets_meta:
            raise KeyError(f"Dataset '{ds_id}' not found in manifest.")
        role_info = datasets_meta[ds_id]["roles"].get(allowed_role)
        if not role_info:
            continue

        rel_path = role_info["file"]
        # rel_path is stored relative to dev root parent (workspace root) or dev root
        full_path = dev_root.parent / rel_path
        if not full_path.exists():
            full_path = dev_root / rel_path

        if not full_path.exists():
            raise FileNotFoundError(f"Sample file not found at {full_path}")

        with open(full_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    sample = SampleInput(**data)
                    yield sample
                except (json.JSONDecodeError, ValidationError) as e:
                    raise ValueError(f"Corrupt sample in {full_path}:{line_no}: {e}") from e


def load_gold_records(
    manifest_path: Optional[Union[str, Path]] = None,
    role: str = "calibration",
    dataset_id: Optional[str] = None,
    danger_enable_holdout_for_b07: bool = False,
) -> List[GoldRecord]:
    """Load GoldRecord instances for evaluation purposes."""
    if role == "holdout" and not danger_enable_holdout_for_b07:
        raise PermissionError(
            "Access to holdout gold records is strictly forbidden during development (B01–B06)."
        )

    if manifest_path is None:
        manifest_path = get_data_dir() / "manifest.json"
    manifest_path = Path(manifest_path)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    dev_root = get_dev_root()
    datasets_meta = manifest.get("datasets", {})
    target_datasets = [dataset_id] if dataset_id else sorted(datasets_meta.keys())

    records: List[GoldRecord] = []
    for ds_id in target_datasets:
        role_info = datasets_meta[ds_id]["roles"].get(role)
        if not role_info:
            continue
        rel_gold_path = role_info["gold_file"]
        full_gold_path = dev_root.parent / rel_gold_path
        if not full_gold_path.exists():
            full_gold_path = dev_root / rel_gold_path

        with open(full_gold_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(GoldRecord(**json.loads(line)))

    return records

