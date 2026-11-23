"""Handoff module for System-1 to System-2 prior injection and candidate pruning.

Timestamp: 2026-11-10 13:25 (Asia/Shanghai)
"""

from src.handoff.builders import build_handoff
from src.handoff.priors import extract_top2_prior

__all__ = ["build_handoff", "extract_top2_prior"]

