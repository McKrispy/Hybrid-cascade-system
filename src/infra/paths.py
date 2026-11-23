"""Path resolution utilities for HybridCascade.

Timestamp: 2026-11-09 11:05 (Asia/Shanghai)

Provides fixed parent-traversal based path resolutions independent of working directory (CWD).
"""

from pathlib import Path


def get_dev_root() -> Path:
    """Return the dev root directory (dev/).
    
    dev/src/infra/paths.py -> parents[2] is dev/
    """
    return Path(__file__).resolve().parents[2]


def get_project_root() -> Path:
    """Return the project root directory.
    
    dev/src/infra/paths.py -> parents[2] is project root
    """
    return Path(__file__).resolve().parents[2]


def get_configs_dir() -> Path:
    """Return dev/configs/ directory."""
    return get_dev_root() / "configs"


def get_runs_dir() -> Path:
    """Return dev/runs/ directory."""
    return get_dev_root() / "runs"


def get_docs_dir() -> Path:
    """Return dev/docs/ directory."""
    return get_dev_root() / "docs"


def get_data_dir() -> Path:
    """Return dev/data/ directory."""
    return get_dev_root() / "data"


def get_panel_dir() -> Path:
    """Return dev/panel/ directory for frontend static assets."""
    return get_dev_root() / "panel"


def get_dot_panel_dir() -> Path:
    """Return dev/.panel/ directory for local panel states/snapshots."""
    dot_panel = get_dev_root() / ".panel"
    dot_panel.mkdir(parents=True, exist_ok=True)
    return dot_panel


def get_designated_python_executable() -> str:
    """Return the designated python executable path for the project from project.json."""
    import json, sys
    project_cfg = get_configs_dir() / "project.json"
    if project_cfg.exists():
        try:
            with open(project_cfg, "r", encoding="utf-8") as f:
                data = json.load(f)
                py_path = data.get("environment", {}).get("python_executable")
                if py_path and Path(py_path).exists():
                    return py_path
        except Exception:
            pass
    return sys.executable

