"""Feature and target extraction for calibration methods in HybridCascade.

Timestamp: 2026-11-09 18:34 (Asia/Shanghai)

Extracts risk features from Jev System-1 probability outputs:
1. 1 - margin: 1D monotonic risk feature for IsotonicRegression
2. [p1, margin, H_norm]: 3D feature vector for regularized LogisticRegression
3. Application error target definitions across standard datasets and CLINC150 mixed pool
"""

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.evaluation.metrics import compute_normalized_entropy


def extract_sample_risk_features(
    probabilities: Dict[str, float],
    num_classes: int,
) -> Dict[str, Any]:
    """Extract calibration features from a single Jev probability distribution.
    
    Args:
        probabilities: Dictionary mapping class labels to probabilities.
        num_classes: Expected number of classes (K) for entropy normalization.
        
    Returns:
        Dictionary with:
            - p1: Top-1 probability
            - p2: Top-2 probability
            - margin: p1 - p2
            - risk_margin: 1.0 - margin (feature for Isotonic)
            - h_norm: Normalized Shannon entropy
            - feature_vector: [p1, margin, h_norm] (feature for Logistic)
    """
    if not probabilities:
        return {
            "p1": 0.0,
            "p2": 0.0,
            "margin": 0.0,
            "risk_margin": 1.0,
            "h_norm": 1.0,
            "feature_vector": [0.0, 0.0, 1.0],
        }

    sorted_probs = sorted(probabilities.values(), reverse=True)
    p1 = float(sorted_probs[0])
    p2 = float(sorted_probs[1]) if len(sorted_probs) > 1 else 0.0
    margin = float(p1 - p2)
    risk_margin = float(max(0.0, min(1.0, 1.0 - margin)))
    h_norm = float(compute_normalized_entropy(probabilities, num_classes=num_classes))

    return {
        "p1": p1,
        "p2": p2,
        "margin": margin,
        "risk_margin": risk_margin,
        "h_norm": h_norm,
        "feature_vector": [p1, margin, h_norm],
    }


def compute_application_error_target(
    decision: Optional[str],
    gold_label: str,
    dataset_id: str,
    probabilities: Optional[Dict[str, float]] = None,
    is_oos: bool = False,
    clinc_tau: Optional[float] = None,
    num_classes: Optional[int] = None,
) -> int:
    """Compute binary application error target e in {0, 1}.
    
    Target:
        e = 0: Correct application output
        e = 1: Application error
        
    For Banking77, ANLI, StrategyQA:
        Standard classification error: e = 0 if decision == gold_label else 1
        
    For CLINC150 (mixed pool application error):
        Uses fixed rejection threshold tau*: Reject if H_norm >= tau*
        - ID (not is_oos): correct if NOT rejected and decision == gold_label
        - OOS (is_oos): correct if rejected
    """
    if dataset_id == "clinc150":
        if clinc_tau is None:
            raise ValueError("clinc_tau must be provided for CLINC150 application error computation.")
        K = num_classes if num_classes is not None else 150
        h_norm = compute_normalized_entropy(probabilities or {}, num_classes=K)
        is_rejected = bool(h_norm >= clinc_tau)

        if not is_oos:
            # In-Domain sample: correct if accepted and correctly classified
            is_correct = (not is_rejected) and (decision is not None and decision == gold_label)
        else:
            # Out-of-Scope sample: correct if properly rejected
            is_correct = is_rejected

        return 0 if is_correct else 1
    else:
        # Standard benchmark classification error
        is_correct = bool(decision is not None and decision == gold_label)
        return 0 if is_correct else 1


def prepare_calibration_dataset(
    jev_predictions: List[Dict[str, Any]],
    gold_records: Dict[str, Any],
    dataset_id: str,
    num_classes: int,
    clinc_tau: Optional[float] = None,
) -> Dict[str, Any]:
    """Prepare structured features and targets for calibration fitting or evaluation.
    
    Returns:
        Dictionary containing sample_ids, X_isotonic, X_logistic, y_error,
        raw_p1, raw_margins, is_oos_list, and raw_probabilities.
    """
    sample_ids: List[str] = []
    x_iso_list: List[float] = []
    x_log_list: List[List[float]] = []
    y_err_list: List[int] = []
    raw_p1_list: List[float] = []
    raw_margins_list: List[float] = []
    is_oos_list: List[bool] = []
    probabilities_list: List[Dict[str, float]] = []
    gold_labels_list: List[str] = []

    for pred in jev_predictions:
        sid = pred["sample_id"]
        gold = gold_records[sid]
        probs = pred.get("probabilities") or {}
        decision = pred.get("decision")
        is_oos = bool(getattr(gold, "is_oos", False) if hasattr(gold, "is_oos") else gold.get("is_oos", False))
        gold_label = str(getattr(gold, "gold_label", "") if hasattr(gold, "gold_label") else gold.get("gold_label", ""))

        feats = extract_sample_risk_features(probs, num_classes=num_classes)
        err = compute_application_error_target(
            decision=decision,
            gold_label=gold_label,
            dataset_id=dataset_id,
            probabilities=probs,
            is_oos=is_oos,
            clinc_tau=clinc_tau,
            num_classes=num_classes,
        )

        sample_ids.append(sid)
        x_iso_list.append(feats["risk_margin"])
        x_log_list.append(feats["feature_vector"])
        y_err_list.append(err)
        raw_p1_list.append(feats["p1"])
        raw_margins_list.append(feats["margin"])
        is_oos_list.append(is_oos)
        probabilities_list.append(probs)
        gold_labels_list.append(gold_label)

    return {
        "sample_ids": sample_ids,
        "X_isotonic": np.array(x_iso_list, dtype=np.float64),
        "X_logistic": np.array(x_log_list, dtype=np.float64),
        "y_error": np.array(y_err_list, dtype=np.int32),
        "raw_p1": np.array(raw_p1_list, dtype=np.float64),
        "raw_margins": np.array(raw_margins_list, dtype=np.float64),
        "is_oos_list": is_oos_list,
        "probabilities_list": probabilities_list,
        "gold_labels_list": gold_labels_list,
        "N": len(sample_ids),
    }

