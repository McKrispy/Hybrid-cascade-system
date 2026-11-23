"""Evaluation metrics and OOS policy search for HybridCascade.

Timestamp: 2026-11-09 13:56 (Asia/Shanghai)

Implements:
1. Accuracy (fraction of correct predictions)
2. 15-bin equal-width ECE (top-label) & MCE
3. Category Brier Score (sum_k (p_k - y_k)^2 averaged over samples)
4. Negative Log-Likelihood (NLL) with epsilon clipping
5. Error detection AUROC & AURC (using 1 - p1 risk)
6. Normalized entropy calculation H_norm
7. CLINC selection-based OOS optimal threshold search with higher tau tie-break
"""

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np


def compute_accuracy(predictions: Sequence[Optional[str]], golds: Sequence[str]) -> float:
    """Compute top-1 classification accuracy over total N."""
    if not golds:
        return 0.0
    correct = sum(1 for p, g in zip(predictions, golds) if p is not None and p == g)
    return float(correct) / len(golds)


def compute_ece_toplabel(
    probabilities_list: Sequence[Dict[str, float]],
    golds: Sequence[str],
    n_bins: int = 15,
) -> Dict[str, Any]:
    """Compute 15-bin equal-width top-label ECE and MCE.
    
    Formula:
        c_i = max_k p_ik
        z_i = 1 if argmax_k p_ik == y_i else 0
        ECE = sum_b (n_b / N) * |mean(z_b) - mean(c_b)|
    """
    N = len(golds)
    if N == 0 or len(probabilities_list) != N:
        return {"ece": None, "mce": None, "n_bins": n_bins, "valid_count": 0}

    confs: List[float] = []
    corrects: List[int] = []

    for probs, gold in zip(probabilities_list, golds):
        if not probs:
            continue
        pred_label = max(probs, key=probs.get)
        c = float(probs[pred_label])
        z = 1 if pred_label == gold else 0
        confs.append(c)
        corrects.append(z)

    n_valid = len(confs)
    if n_valid == 0:
        return {"ece": None, "mce": None, "n_bins": n_bins, "valid_count": 0}

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    bin_details = []

    for b in range(n_bins):
        low = bin_boundaries[b]
        high = bin_boundaries[b + 1]

        # Samples falling into [low, high) or [low, 1.0] for final bin
        if b == n_bins - 1:
            indices = [i for i, c in enumerate(confs) if low <= c <= high]
        else:
            indices = [i for i, c in enumerate(confs) if low <= c < high]

        n_b = len(indices)
        if n_b > 0:
            bin_acc = sum(corrects[i] for i in indices) / float(n_b)
            bin_conf = sum(confs[i] for i in indices) / float(n_b)
            gap = abs(bin_acc - bin_conf)
            ece += (n_b / float(n_valid)) * gap
            if gap > mce:
                mce = gap
            bin_details.append({"bin": b, "count": n_b, "acc": bin_acc, "conf": bin_conf, "gap": gap})
        else:
            bin_details.append({"bin": b, "count": 0, "acc": None, "conf": None, "gap": None})

    return {
        "ece": round(float(ece), 6),
        "mce": round(float(mce), 6),
        "n_bins": n_bins,
        "valid_count": n_valid,
        "bins": bin_details,
    }


def compute_category_brier(
    probabilities_list: Sequence[Dict[str, float]],
    golds: Sequence[str],
    allowed_labels: Sequence[str],
) -> Optional[float]:
    """Compute category Brier score: mean_i sum_k (p_ik - y_ik)^2 (without dividing by K)."""
    if not golds or len(probabilities_list) != len(golds):
        return None

    squared_errors: List[float] = []
    labels_set = set(allowed_labels)

    for probs, gold in zip(probabilities_list, golds):
        if not probs:
            continue
        sample_err = 0.0
        for opt in labels_set:
            p = float(probs.get(opt, 0.0))
            y = 1.0 if opt == gold else 0.0
            sample_err += (p - y) ** 2
        squared_errors.append(sample_err)

    if not squared_errors:
        return None
    return round(float(np.mean(squared_errors)), 6)


def compute_nll(
    probabilities_list: Sequence[Dict[str, float]],
    golds: Sequence[str],
    epsilon: float = 1e-12,
) -> Optional[float]:
    """Compute Negative Log-Likelihood: -mean_i log(clip(p_i,y_i, eps, 1.0))."""
    if not golds or len(probabilities_list) != len(golds):
        return None

    losses: List[float] = []
    for probs, gold in zip(probabilities_list, golds):
        if not probs or gold not in probs:
            p = epsilon
        else:
            p = max(float(probs[gold]), epsilon)
        losses.append(-math.log(p))

    if not losses:
        return None
    return round(float(np.mean(losses)), 6)


def compute_error_detection_auroc_aurc(
    probabilities_list: Sequence[Dict[str, float]],
    golds: Sequence[str],
) -> Dict[str, Optional[float]]:
    """Compute AUROC and AURC for error detection where positive class is classification error.
    
    Risk signal: r_i = 1 - max_k p_ik.
    """
    if not golds or len(probabilities_list) != len(golds):
        return {"auroc": None, "aurc": None}

    risks: List[float] = []
    errors: List[int] = []

    for probs, gold in zip(probabilities_list, golds):
        if not probs:
            continue
        pred_label = max(probs, key=probs.get)
        c = float(probs[pred_label])
        r = 1.0 - c
        is_error = 1 if pred_label != gold else 0
        risks.append(r)
        errors.append(is_error)

    N = len(risks)
    n_errors = sum(errors)
    n_corrects = N - n_errors

    if N == 0 or n_errors == 0 or n_corrects == 0:
        # AUROC undefined if only one class exists
        return {"auroc": None, "aurc": None}

    # 1. AUROC computation (Wilcoxon-Mann-Whitney rank sum)
    # Higher risk score -> more likely to be error (positive class)
    ranks = np.argsort(np.argsort(risks)) + 1  # 1-based ranks
    sum_ranks_pos = sum(ranks[i] for i in range(N) if errors[i] == 1)
    u_stat = sum_ranks_pos - (n_errors * (n_errors + 1)) / 2.0
    auroc = float(u_stat) / (n_errors * n_corrects)

    # 2. AURC computation
    # Sort samples from lowest risk to highest risk
    sort_order = np.argsort(risks)
    sorted_errors = [errors[idx] for idx in sort_order]
    
    cum_errors = np.cumsum(sorted_errors)
    coverages = np.arange(1, N + 1)
    risk_at_coverage = cum_errors / coverages
    aurc = float(np.mean(risk_at_coverage))

    return {
        "auroc": round(auroc, 6),
        "aurc": round(aurc, 6),
    }


def compute_normalized_entropy(probabilities: Dict[str, float], num_classes: Optional[int] = None) -> float:
    """Compute normalized Shannon entropy: H_norm = -sum(p * log(p)) / log(K).
    
    Uses natural logarithm; 0 * log(0) = 0.
    """
    K = num_classes if num_classes is not None else len(probabilities)
    if K <= 1:
        return 0.0

    entropy = 0.0
    for p in probabilities.values():
        if p > 0.0:
            entropy -= p * math.log(p)

    h_norm = entropy / math.log(K)
    # Numerical clip to [0.0, 1.0]
    return float(max(0.0, min(1.0, h_norm)))


def search_clinc_oos_threshold(
    selection_probs: Sequence[Dict[str, float]],
    selection_is_oos: Sequence[bool],
    num_classes: int = 150,
) -> Dict[str, Any]:
    """Search optimal OOS rejection threshold on CLINC selection Jev entropy.
    
    Decision rule: Reject if H_norm >= tau.
    Balanced Accuracy = (TPR + TNR) / 2
        TPR = ID correctly classified as ID (H_norm < tau)
        TNR = OOS correctly rejected (H_norm >= tau)
    Tie-breaker: choose higher tau if balanced accuracy ties.
    """
    N = len(selection_is_oos)
    if N == 0 or len(selection_probs) != N:
        raise ValueError("selection_probs and selection_is_oos must have same non-zero length.")

    entropies = [compute_normalized_entropy(p, num_classes=num_classes) for p in selection_probs]
    
    # Candidate thresholds: all observed H_norm values + 0.0 and 1.0 boundary points
    candidates = sorted(list(set(entropies) | {0.0, 1.0}))

    best_tau = 1.0
    best_bal_acc = -1.0
    best_tpr = 0.0
    best_tnr = 0.0

    n_id = sum(1 for oos in selection_is_oos if not oos)
    n_oos = sum(1 for oos in selection_is_oos if oos)

    if n_id == 0 or n_oos == 0:
        raise ValueError(f"CLINC selection must contain both ID and OOS samples (got id={n_id}, oos={n_oos})")

    for tau in candidates:
        # Predict REJECT if H_norm >= tau
        correct_id = sum(1 for h, oos in zip(entropies, selection_is_oos) if not oos and h < tau)
        correct_oos = sum(1 for h, oos in zip(entropies, selection_is_oos) if oos and h >= tau)

        tpr = correct_id / float(n_id)
        tnr = correct_oos / float(n_oos)
        bal_acc = (tpr + tnr) / 2.0

        # Maximize balanced accuracy; on ties, prefer HIGHER tau (conservative rejection)
        if bal_acc > best_bal_acc or (abs(bal_acc - best_bal_acc) < 1e-9 and tau > best_tau):
            best_bal_acc = bal_acc
            best_tau = tau
            best_tpr = tpr
            best_tnr = tnr

    return {
        "optimal_tau": round(float(best_tau), 6),
        "balanced_accuracy": round(float(best_bal_acc), 6),
        "id_tpr": round(float(best_tpr), 6),
        "oos_tnr": round(float(best_tnr), 6),
        "n_id": n_id,
        "n_oos": n_oos,
        "tie_break_rule": "higher_tau",
    }

