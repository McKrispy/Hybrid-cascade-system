"""Routing package for HybridCascade.

Timestamp: 2026-11-10 13:03 (Asia/Shanghai)
"""

from src.routing.signals import (
    compute_dataset_routing_signals,
    compute_sample_routing_signals,
)
from src.routing.masks import (
    build_nested_masks_for_signal,
    build_oracle_nested_masks,
    build_random_nested_masks,
    build_route_mask,
    evaluate_cascade,
)

__all__ = [
    "compute_sample_routing_signals",
    "compute_dataset_routing_signals",
    "build_route_mask",
    "build_nested_masks_for_signal",
    "build_random_nested_masks",
    "build_oracle_nested_masks",
    "evaluate_cascade",
]

