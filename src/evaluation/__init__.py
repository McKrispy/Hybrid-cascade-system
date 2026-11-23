"""Evaluation metrics, calibration analysis, and handoff bootstrap testing for HybridCascade.

Timestamp: 2026-11-10 13:28 (Asia/Shanghai)
"""

from src.evaluation.bootstrap import (
    compute_ci_and_pvalue,
    paired_group_bootstrap_multicondition,
    paired_group_bootstrap_series,
)
from src.evaluation.fr_control import evaluate_fr_control_batch
from src.evaluation.handoff_effects import (
    analyze_error_following,
    analyze_gold_in_top2_coverage,
    check_sample_correctness,
    evaluate_handoff_batch,
)
from src.evaluation.metrics import (
    compute_accuracy,
    compute_category_brier,
    compute_ece_toplabel,
    compute_error_detection_auroc_aurc,
    compute_nll,
    compute_normalized_entropy,
    search_clinc_oos_threshold,
)

__all__ = [
    "analyze_error_following",
    "analyze_gold_in_top2_coverage",
    "check_sample_correctness",
    "compute_accuracy",
    "compute_category_brier",
    "compute_ci_and_pvalue",
    "compute_ece_toplabel",
    "compute_error_detection_auroc_aurc",
    "compute_nll",
    "compute_normalized_entropy",
    "evaluate_fr_control_batch",
    "evaluate_handoff_batch",
    "paired_group_bootstrap_multicondition",
    "paired_group_bootstrap_series",
    "search_clinc_oos_threshold",
]
