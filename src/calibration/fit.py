"""Calibration fitting, inference, and model selection for HybridCascade.

Timestamp: 2026-11-09 18:35 (Asia/Shanghai)

Implements:
1. IsotonicCalibrator: 1D monotonic mapping (1 - margin) -> e_hat
2. LogisticCalibrator: Regularized logistic regression [p1, margin, H_norm] -> e_hat
3. TemperatureCalibrator: Multiclass temperature scaling p -> p' minimizing NLL on calibration
4. Model Selection Protocol: Macro unweighted selection error Brier comparison against raw 1 - p1
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.evaluation.metrics import (
    compute_category_brier,
    compute_ece_toplabel,
    compute_nll,
    compute_normalized_entropy,
)


class IsotonicCalibrator:
    """1D monotonic error risk calibrator: (1 - margin) -> error probability e_hat."""

    def __init__(self):
        self.is_fitted = False
        self.is_fallback = False
        self.fallback_risk = 0.0
        self.x_points: List[float] = []
        self.y_points: List[float] = []
        self.x_min = 0.0
        self.x_max = 1.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "IsotonicCalibrator":
        """Fit isotonic regression on calibration data.
        
        Args:
            X: 1D array of risk features (1 - margin), values in [0, 1].
            y: 1D array of binary error targets in {0, 1}.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        unique_y = np.unique(y)
        unique_x = np.unique(X)

        # Fallback condition: single class or constant feature
        if len(unique_y) <= 1 or len(unique_x) <= 1:
            self.is_fitted = True
            self.is_fallback = True
            self.fallback_risk = float(np.mean(y)) if len(y) > 0 else 0.5
            return self

        iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
        iso.fit(X, y)

        self.is_fitted = True
        self.is_fallback = False
        self.fallback_risk = float(np.mean(y))
        self.x_points = [float(x) for x in iso.f_.x]
        self.y_points = [float(y_val) for y_val in iso.f_.y]
        self.x_min = float(iso.X_min_)
        self.x_max = float(iso.X_max_)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict calibrated error risk e_hat in [0, 1]."""
        if not self.is_fitted:
            raise RuntimeError("IsotonicCalibrator must be fitted before predict.")

        X = np.asarray(X, dtype=np.float64)
        if self.is_fallback:
            return np.full_like(X, self.fallback_risk)

        # Piecewise linear interpolation with out-of-bounds clipping
        # Equivalent to scikit-learn out_of_bounds='clip'
        X_clipped = np.clip(X, self.x_min, self.x_max)
        preds = np.interp(X_clipped, self.x_points, self.y_points)
        return np.clip(preds, 0.0, 1.0)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize parameters for JSON storage and offline reproduction."""
        return {
            "method": "isotonic",
            "is_fitted": self.is_fitted,
            "is_fallback": self.is_fallback,
            "fallback_risk": round(self.fallback_risk, 6),
            "x_min": round(self.x_min, 6),
            "x_max": round(self.x_max, 6),
            "num_knots": len(self.x_points),
            "x_points": [round(x, 6) for x in self.x_points],
            "y_points": [round(y, 6) for y in self.y_points],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IsotonicCalibrator":
        """Instantiate calibrator from serialized JSON dictionary."""
        cal = cls()
        cal.is_fitted = data.get("is_fitted", False)
        cal.is_fallback = data.get("is_fallback", False)
        cal.fallback_risk = float(data.get("fallback_risk", 0.0))
        cal.x_min = float(data.get("x_min", 0.0))
        cal.x_max = float(data.get("x_max", 1.0))
        cal.x_points = [float(x) for x in data.get("x_points", [])]
        cal.y_points = [float(y) for y in data.get("y_points", [])]
        return cal


class LogisticCalibrator:
    """Fixed-regularization logistic error risk calibrator: [p1, margin, H_norm] -> e_hat."""

    def __init__(self, C: float = 1.0):
        self.C = C
        self.is_fitted = False
        self.is_fallback = False
        self.fallback_risk = 0.0
        self.weights: List[float] = [0.0, 0.0, 0.0]
        self.intercept: float = 0.0
        self.scaler_mean: List[float] = [0.0, 0.0, 0.0]
        self.scaler_scale: List[float] = [1.0, 1.0, 1.0]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticCalibrator":
        """Fit standardized logistic regression on calibration data.
        
        Args:
            X: 2D array of shape (N, 3) containing [p1, margin, H_norm].
            y: 1D array of binary error targets in {0, 1}.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        unique_y = np.unique(y)
        if len(unique_y) <= 1:
            self.is_fitted = True
            self.is_fallback = True
            self.fallback_risk = float(np.mean(y)) if len(y) > 0 else 0.5
            return self

        # Preprocessing fit strictly on calibration split
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        clf = LogisticRegression(C=self.C, solver="lbfgs", random_state=42)
        clf.fit(X_scaled, y)

        self.is_fitted = True
        self.is_fallback = False
        self.fallback_risk = float(np.mean(y))
        self.weights = [float(w) for w in clf.coef_[0]]
        self.intercept = float(clf.intercept_[0])
        self.scaler_mean = [float(m) for m in scaler.mean_]
        # Avoid division by zero for zero-variance features
        self.scaler_scale = [float(s) if s > 1e-12 else 1.0 for s in scaler.scale_]

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict calibrated error risk e_hat in [0, 1]."""
        if not self.is_fitted:
            raise RuntimeError("LogisticCalibrator must be fitted before predict.")

        X = np.asarray(X, dtype=np.float64)
        if self.is_fallback:
            return np.full((X.shape[0],), self.fallback_risk)

        # Standardize using calibration-fitted scalers
        mean = np.array(self.scaler_mean, dtype=np.float64)
        scale = np.array(self.scaler_scale, dtype=np.float64)
        X_norm = (X - mean) / scale

        # Logit z = b + sum_j w_j * x_norm_j
        weights = np.array(self.weights, dtype=np.float64)
        logits = self.intercept + np.dot(X_norm, weights)

        # Sigmoid: 1 / (1 + exp(-z)) with numerical clipping
        logits_clipped = np.clip(logits, -30.0, 30.0)
        e_hat = 1.0 / (1.0 + np.exp(-logits_clipped))
        return np.clip(e_hat, 0.0, 1.0)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize parameters for JSON storage and offline reproduction."""
        return {
            "method": "logistic",
            "C": self.C,
            "is_fitted": self.is_fitted,
            "is_fallback": self.is_fallback,
            "fallback_risk": round(self.fallback_risk, 6),
            "intercept": round(self.intercept, 6),
            "weights": [round(w, 6) for w in self.weights],
            "scaler_mean": [round(m, 6) for m in self.scaler_mean],
            "scaler_scale": [round(s, 6) for s in self.scaler_scale],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LogisticCalibrator":
        """Instantiate calibrator from serialized JSON dictionary."""
        cal = cls(C=float(data.get("C", 1.0)))
        cal.is_fitted = data.get("is_fitted", False)
        cal.is_fallback = data.get("is_fallback", False)
        cal.fallback_risk = float(data.get("fallback_risk", 0.0))
        cal.intercept = float(data.get("intercept", 0.0))
        cal.weights = [float(w) for w in data.get("weights", [0.0, 0.0, 0.0])]
        cal.scaler_mean = [float(m) for m in data.get("scaler_mean", [0.0, 0.0, 0.0])]
        cal.scaler_scale = [float(s) for s in data.get("scaler_scale", [1.0, 1.0, 1.0])]
        return cal


class TemperatureCalibrator:
    """Multiclass temperature scaling calibrator: p -> p' minimizing NLL on calibration."""

    def __init__(self, epsilon: float = 1e-12):
        self.epsilon = epsilon
        self.is_fitted = False
        self.temperature: float = 1.0
        self.calibration_nll: Optional[float] = None
        self.converged: bool = False

    def fit(
        self,
        probabilities_list: List[Dict[str, float]],
        gold_labels: List[str],
        bounds: Tuple[float, float] = (0.05, 10.0),
    ) -> "TemperatureCalibrator":
        """Fit scalar temperature T by minimizing Negative Log-Likelihood on calibration.
        
        Args:
            probabilities_list: List of category probability distributions.
            gold_labels: List of gold class labels corresponding to samples.
            bounds: Search bounds for scalar temperature T.
        """
        N = len(probabilities_list)
        if N == 0 or len(gold_labels) != N:
            raise ValueError("probabilities_list and gold_labels must be non-empty and have same length.")

        labels = sorted(list(probabilities_list[0].keys()))
        label_to_idx = {l: i for i, l in enumerate(labels)}

        # Build matrix of log-probabilities
        matrix = np.zeros((N, len(labels)), dtype=np.float64)
        gold_indices = np.zeros(N, dtype=np.int32)

        for i, (probs, gold) in enumerate(zip(probabilities_list, gold_labels)):
            for l, idx in label_to_idx.items():
                p = probs.get(l, 0.0)
                matrix[i, idx] = max(p, self.epsilon)
            gold_indices[i] = label_to_idx.get(gold, 0)

        log_p_matrix = np.log(matrix)

        def nll_objective(T: float) -> float:
            if T <= 0.0:
                return 1e9
            scaled_logits = log_p_matrix / T
            # Log-sum-exp trick for numerical stability
            max_logits = np.max(scaled_logits, axis=1, keepdims=True)
            exp_logits = np.exp(scaled_logits - max_logits)
            denom = np.sum(exp_logits, axis=1, keepdims=True)
            norm_probs = exp_logits / denom

            correct_probs = norm_probs[np.arange(N), gold_indices]
            clipped_probs = np.clip(correct_probs, self.epsilon, 1.0)
            return float(-np.mean(np.log(clipped_probs)))

        res = minimize_scalar(nll_objective, bounds=bounds, method="bounded")
        self.is_fitted = True
        self.temperature = float(res.x)
        self.calibration_nll = float(res.fun)
        self.converged = bool(res.success)

        return self

    def recalibrate(self, probabilities: Dict[str, float]) -> Dict[str, float]:
        """Apply temperature scaling to a single probability distribution."""
        if not self.is_fitted or abs(self.temperature - 1.0) < 1e-9:
            return dict(probabilities)

        T = self.temperature
        labels = list(probabilities.keys())
        p_vals = np.array([max(probabilities.get(l, 0.0), self.epsilon) for l in labels], dtype=np.float64)
        
        logits = np.log(p_vals) / T
        logits -= np.max(logits)
        exp_logits = np.exp(logits)
        denom = np.sum(exp_logits)
        scaled_p = exp_logits / denom

        return {labels[i]: float(scaled_p[i]) for i in range(len(labels))}

    def recalibrate_batch(self, probabilities_list: List[Dict[str, float]]) -> List[Dict[str, float]]:
        """Recalibrate a list of probability distributions."""
        return [self.recalibrate(p) for p in probabilities_list]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize temperature parameters for JSON storage and offline reproduction."""
        return {
            "method": "temperature",
            "temperature": round(self.temperature, 6),
            "calibration_nll": round(self.calibration_nll, 6) if self.calibration_nll is not None else None,
            "converged": self.converged,
            "epsilon": self.epsilon,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TemperatureCalibrator":
        """Instantiate temperature calibrator from serialized JSON dictionary."""
        cal = cls(epsilon=float(data.get("epsilon", 1e-12)))
        cal.is_fitted = True
        cal.temperature = float(data.get("temperature", 1.0))
        cal.calibration_nll = data.get("calibration_nll")
        cal.converged = bool(data.get("converged", True))
        return cal


def evaluate_and_select_calibrator_family(
    calibrators_by_dataset: Dict[str, Dict[str, Any]],
    selection_features_by_dataset: Dict[str, Dict[str, Any]],
    ece_bins: int = 15,
) -> Dict[str, Any]:
    """Execute model selection protocol between isotonic and logistic across 4 datasets.
    
    Protocol rules:
    1. On selection, compare macro unweighted (1/4 each) Error Brier score. Lower score is strictly better.
    2. If logistic strictly lower than isotonic, candidate method family is logistic.
    3. If isotonic strictly lower than logistic, candidate method family is isotonic.
    4. If logistic and isotonic tie exactly, select isotonic.
    5. The winner MUST be strictly lower than the raw 1 - p1 baseline. Otherwise, select null.
    """
    dataset_ids = sorted(list(calibrators_by_dataset.keys()))
    num_datasets = len(dataset_ids)

    brier_scores_raw: Dict[str, float] = {}
    brier_scores_isotonic: Dict[str, float] = {}
    brier_scores_logistic: Dict[str, float] = {}

    temperature_evals: Dict[str, Dict[str, Any]] = {}

    for ds_id in dataset_ids:
        cals = calibrators_by_dataset[ds_id]
        sel_data = selection_features_by_dataset[ds_id]

        y_true = sel_data["y_error"]
        raw_p1 = sel_data["raw_p1"]

        # Raw 1 - p1 baseline error Brier
        raw_risk = 1.0 - raw_p1
        brier_raw = float(np.mean((raw_risk - y_true) ** 2))
        brier_scores_raw[ds_id] = round(brier_raw, 6)

        # Isotonic prediction and Brier
        iso_cal: IsotonicCalibrator = cals["isotonic"]
        iso_preds = iso_cal.predict(sel_data["X_isotonic"])
        brier_iso = float(np.mean((iso_preds - y_true) ** 2))
        brier_scores_isotonic[ds_id] = round(brier_iso, 6)

        # Logistic prediction and Brier
        log_cal: LogisticCalibrator = cals["logistic"]
        log_preds = log_cal.predict(sel_data["X_logistic"])
        brier_log = float(np.mean((log_preds - y_true) ** 2))
        brier_scores_logistic[ds_id] = round(brier_log, 6)

        # Temperature scaling evaluation on selection
        temp_cal: TemperatureCalibrator = cals["temperature"]
        probs_list = sel_data["probabilities_list"]
        gold_labels = sel_data["gold_labels_list"]
        is_oos_list = sel_data.get("is_oos_list", [False] * len(probs_list))

        # Filter ID only for CLINC150 temperature evaluation
        if ds_id == "clinc150":
            eval_indices = [i for i, oos in enumerate(is_oos_list) if not oos]
        else:
            eval_indices = list(range(len(probs_list)))

        probs_eval = [probs_list[i] for i in eval_indices]
        golds_eval = [gold_labels[i] for i in eval_indices]

        # Raw category calibration
        raw_ece_res = compute_ece_toplabel(probs_eval, golds_eval, n_bins=ece_bins)
        raw_nll = compute_nll(probs_eval, golds_eval)
        raw_cat_brier = compute_category_brier(probs_eval, golds_eval, allowed_labels=list(probs_eval[0].keys()))

        # Recalibrated category calibration
        recal_probs_eval = temp_cal.recalibrate_batch(probs_eval)
        scaled_ece_res = compute_ece_toplabel(recal_probs_eval, golds_eval, n_bins=ece_bins)
        scaled_nll = compute_nll(recal_probs_eval, golds_eval)
        scaled_cat_brier = compute_category_brier(recal_probs_eval, golds_eval, allowed_labels=list(recal_probs_eval[0].keys()))

        temperature_evals[ds_id] = {
            "temperature": temp_cal.temperature,
            "eval_n": len(eval_indices),
            "raw": {
                "ece": raw_ece_res["ece"],
                "mce": raw_ece_res["mce"],
                "nll": raw_nll,
                "category_brier": raw_cat_brier,
            },
            "scaled": {
                "ece": scaled_ece_res["ece"],
                "mce": scaled_ece_res["mce"],
                "nll": scaled_nll,
                "category_brier": scaled_cat_brier,
            },
        }

    # Macro unweighted averages (1/4 weight each)
    macro_raw = float(np.mean([brier_scores_raw[d] for d in dataset_ids]))
    macro_iso = float(np.mean([brier_scores_isotonic[d] for d in dataset_ids]))
    macro_log = float(np.mean([brier_scores_logistic[d] for d in dataset_ids]))

    # Model selection decision logic
    # Rule 1 & 2 & 3: Compare candidate method families
    if macro_log < macro_iso:
        candidate_winner = "logistic"
        candidate_winner_brier = macro_log
    elif macro_iso < macro_log:
        candidate_winner = "isotonic"
        candidate_winner_brier = macro_iso
    else:
        # Exact tie: prioritize isotonic
        candidate_winner = "isotonic"
        candidate_winner_brier = macro_iso

    # Rule 4: Compare with raw 1 - p1 baseline
    if candidate_winner_brier < macro_raw:
        selected_method_family = candidate_winner
        selection_reason = (
            f"{candidate_winner} strictly beat alternative ({candidate_winner} Brier={candidate_winner_brier:.6f} "
            f"vs {'logistic' if candidate_winner == 'isotonic' else 'isotonic'} Brier={macro_log if candidate_winner == 'isotonic' else macro_iso:.6f}) "
            f"and strictly beat raw 1-p1 baseline (Brier={macro_raw:.6f})."
        )
    else:
        selected_method_family = None
        selection_reason = (
            f"Candidate winner {candidate_winner} (Brier={candidate_winner_brier:.6f}) failed to strictly beat "
            f"raw 1-p1 baseline (Brier={macro_raw:.6f}). No calibrated routing candidate is retained."
        )

    return {
        "selected_method_family": selected_method_family,
        "selection_reason": selection_reason,
        "macro_selection_error_brier": {
            "raw_1_minus_p1": round(macro_raw, 6),
            "isotonic": round(macro_iso, 6),
            "logistic": round(macro_log, 6),
        },
        "by_dataset_selection_error_brier": {
            "raw_1_minus_p1": brier_scores_raw,
            "isotonic": brier_scores_isotonic,
            "logistic": brier_scores_logistic,
        },
        "temperature_category_calibration": temperature_evals,
    }

