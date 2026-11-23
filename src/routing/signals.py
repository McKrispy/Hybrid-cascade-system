"""Routing signal calculation for HybridCascade.

Timestamp: 2026-11-10 13:02 (Asia/Shanghai)

Extracts four risk signals from Jev System-1 outputs:
1. 1 - p1: Primary signal for main handoff cascades
2. 1 - margin: 1 - (p1 - p2)
3. Hnorm: Normalized Shannon entropy -sum(p * log(p)) / log(K)
4. logistic: Calibrated error risk e_hat from B03 fitted LogisticCalibrator
"""

from typing import Any, Dict, List, Optional
import numpy as np

from src.calibration.features import extract_sample_risk_features
from src.calibration.fit import LogisticCalibrator
from src.evaluation.metrics import compute_normalized_entropy


def compute_sample_routing_signals(
    prediction: Dict[str, Any],
    dataset_id: str,
    num_classes: int,
    logistic_calibrator: Optional[LogisticCalibrator] = None,
) -> Dict[str, float]:
    """Compute the 4 candidate risk signals for a single Jev prediction.
    
    If the prediction is invalid or missing, assigns +infinity risk
    to prioritize escalation into the System-2 routing mask.
    """
    status = prediction.get("status", "valid")
    probs = prediction.get("probabilities") or {}

    if status != "valid" or not probs:
        return {
            "1-p1": float("inf"),
            "1-margin": float("inf"),
            "Hnorm": float("inf"),
            "logistic": float("inf"),
        }

    feats = extract_sample_risk_features(probs, num_classes=num_classes)
    p1 = feats["p1"]
    risk_p1 = float(max(0.0, min(1.0, 1.0 - p1)))
    risk_margin = feats["risk_margin"]
    h_norm = feats["h_norm"]

    if logistic_calibrator is not None and logistic_calibrator.is_fitted:
        feat_vec = np.array([feats["feature_vector"]], dtype=np.float64)
        risk_logistic = float(logistic_calibrator.predict(feat_vec)[0])
    else:
        risk_logistic = risk_p1

    return {
        "1-p1": risk_p1,
        "1-margin": risk_margin,
        "Hnorm": h_norm,
        "logistic": risk_logistic,
    }


def compute_dataset_routing_signals(
    jev_predictions: List[Dict[str, Any]],
    dataset_id: str,
    num_classes: int,
    logistic_calibrator: Optional[LogisticCalibrator] = None,
) -> Dict[str, Dict[str, float]]:
    """Compute all 4 routing signals for a list of Jev predictions in a dataset.
    
    Returns:
        Dictionary mapping sample_id -> { "1-p1": r1, "1-margin": r2, "Hnorm": r3, "logistic": r4 }
    """
    signals_by_sample: Dict[str, Dict[str, float]] = {}
    for pred in jev_predictions:
        sid = pred["sample_id"]
        signals_by_sample[sid] = compute_sample_routing_signals(
            prediction=pred,
            dataset_id=dataset_id,
            num_classes=num_classes,
            logistic_calibrator=logistic_calibrator,
        )
    return signals_by_sample

