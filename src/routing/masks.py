"""Routing mask generation, nesting validation, and cascade evaluation for HybridCascade.

Timestamp: 2026-11-10 13:03 (Asia/Shanghai)

Implements:
1. Deterministic sorting: Risk descending, sample_id ascending
2. Multi-q nested masks: q in {0.15, 0.25, 0.35} with ceil(q * N_total)
3. Strict nesting verification: M_15 subset M_25 subset M_35
4. Primary 1-p1 mask extraction and persistence (975 dev pool samples)
5. Cascade simulation, transfer matrix (corrected/damaged), and deployment cost
6. Reference baselines: Random permutation and Oracle upper bound
"""

import math
import random
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import numpy as np

from src.evaluation.metrics import compute_normalized_entropy


def build_route_mask(
    sample_ids: Sequence[str],
    risk_scores: Sequence[float],
    q: float,
) -> List[str]:
    """Build a deterministic escalation mask for budget fraction q.
    
    Sorting rules:
    - Primary key: risk score descending (-risk)
    - Tie-breaker: sample_id ascending (alphabetical)
    - Capacity: N_routed = ceil(q * N_total)
    """
    N_total = len(sample_ids)
    if N_total == 0 or q <= 0.0:
        return []

    target_n = min(N_total, math.ceil(q * N_total))

    # Pair each sample with its risk score
    paired = [
        {"sample_id": sid, "risk": float(r)}
        for sid, r in zip(sample_ids, risk_scores)
    ]

    # Deterministic sort: highest risk first, lowest sample_id first on ties
    paired.sort(key=lambda x: (-x["risk"], x["sample_id"]))

    return [item["sample_id"] for item in paired[:target_n]]


def build_nested_masks_for_signal(
    sample_ids: Sequence[str],
    risk_scores: Sequence[float],
    q_rates: Sequence[float] = (0.15, 0.25, 0.35),
) -> Dict[str, Any]:
    """Generate and verify strictly nested masks across budget points q.
    
    Returns dictionary with:
        - masks: { "0.15": [...], "0.25": [...], "0.35": [...] }
        - union_35: [...]
        - counts: { "0.15": n1, "0.25": n2, "0.35": n3, "union_35": n3 }
        - ranked_order: Full deterministic list of sample_ids in descending risk order
    """
    N_total = len(sample_ids)
    paired = [
        {"sample_id": sid, "risk": float(r)}
        for sid, r in zip(sample_ids, risk_scores)
    ]
    paired.sort(key=lambda x: (-x["risk"], x["sample_id"]))
    ranked_sids = [x["sample_id"] for x in paired]

    masks: Dict[str, List[str]] = {}
    counts: Dict[str, int] = {}

    sorted_q = sorted(q_rates)
    for q in sorted_q:
        q_key = f"{q:.2f}"
        target_n = min(N_total, math.ceil(q * N_total))
        mask_q = ranked_sids[:target_n]
        masks[q_key] = mask_q
        counts[q_key] = len(mask_q)

    # Verify nesting: M_q1 subset M_q2 for q1 < q2
    for i in range(len(sorted_q) - 1):
        q_curr = f"{sorted_q[i]:.2f}"
        q_next = f"{sorted_q[i + 1]:.2f}"
        set_curr = set(masks[q_curr])
        set_next = set(masks[q_next])
        if not set_curr.issubset(set_next):
            raise AssertionError(f"Mask nesting violated between q={q_curr} and q={q_next}!")

    max_q_key = f"{sorted_q[-1]:.2f}"
    union_sids = list(masks[max_q_key])
    counts["union"] = len(union_sids)

    return {
        "masks": masks,
        "union_sids": union_sids,
        "counts": counts,
        "ranked_order": ranked_sids,
    }


def build_random_nested_masks(
    sample_ids: Sequence[str],
    q_rates: Sequence[float] = (0.15, 0.25, 0.35),
    seed: int = 42,
) -> Dict[str, Any]:
    """Generate random baseline nested masks using deterministic random permutation."""
    rng = random.Random(seed)
    shuffled_sids = list(sample_ids)
    rng.shuffle(shuffled_sids)

    N_total = len(sample_ids)
    masks: Dict[str, List[str]] = {}
    counts: Dict[str, int] = {}

    sorted_q = sorted(q_rates)
    for q in sorted_q:
        q_key = f"{q:.2f}"
        target_n = min(N_total, math.ceil(q * N_total))
        masks[q_key] = shuffled_sids[:target_n]
        counts[q_key] = target_n

    max_q_key = f"{sorted_q[-1]:.2f}"
    counts["union"] = len(masks[max_q_key])

    return {
        "masks": masks,
        "union_sids": masks[max_q_key],
        "counts": counts,
        "ranked_order": shuffled_sids,
    }


def build_oracle_nested_masks(
    sample_ids: Sequence[str],
    jev_correct_flags: Sequence[bool],
    s2_correct_flags: Sequence[bool],
    q_rates: Sequence[float] = (0.15, 0.25, 0.35),
) -> Dict[str, Any]:
    """Generate theoretical upper-bound oracle masks using gold labels.
    
    Priority levels:
    - Tier 1 (weight 3): Jev Wrong & S2 Correct (Positive transfer: Correction)
    - Tier 2 (weight 2): Jev Wrong & S2 Wrong (Neutral: already incorrect)
    - Tier 3 (weight 1): Jev Correct & S2 Correct (Neutral: preserves correctness)
    - Tier 4 (weight 0): Jev Correct & S2 Wrong (Negative transfer: Damage)
    """
    paired = []
    for sid, j_corr, s2_corr in zip(sample_ids, jev_correct_flags, s2_correct_flags):
        if not j_corr and s2_corr:
            tier = 3
        elif not j_corr and not s2_corr:
            tier = 2
        elif j_corr and s2_corr:
            tier = 1
        else:
            tier = 0
        paired.append({"sample_id": sid, "tier": tier})

    # Sort tier descending, then sample_id ascending
    paired.sort(key=lambda x: (-x["tier"], x["sample_id"]))
    ranked_sids = [x["sample_id"] for x in paired]

    N_total = len(sample_ids)
    masks: Dict[str, List[str]] = {}
    counts: Dict[str, int] = {}

    sorted_q = sorted(q_rates)
    for q in sorted_q:
        q_key = f"{q:.2f}"
        target_n = min(N_total, math.ceil(q * N_total))
        masks[q_key] = ranked_sids[:target_n]
        counts[q_key] = target_n

    max_q_key = f"{sorted_q[-1]:.2f}"
    counts["union"] = len(masks[max_q_key])

    return {
        "masks": masks,
        "union_sids": masks[max_q_key],
        "counts": counts,
        "ranked_order": ranked_sids,
    }


def evaluate_cascade(
    sample_ids: Sequence[str],
    jev_predictions_map: Dict[str, Dict[str, Any]],
    s2_predictions_map: Dict[str, Dict[str, Any]],
    gold_records_map: Dict[str, Any],
    mask: Sequence[str],
    dataset_id: str,
    clinc_tau: Optional[float] = None,
    num_classes: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate a single cascaded configuration at a given escalation mask.
    
    Computes:
    - Overall cascaded accuracy over N_total
    - Upgraded subset accuracy over N_routed
    - Transfer matrix: corrected, damaged, both_correct, both_wrong, net_gain
    - Deployment costs: total cost, routed cost, jev cost
    """
    mask_set: Set[str] = set(mask)
    N_total = len(sample_ids)
    N_routed = len(mask_set)

    if N_total == 0:
        return {
            "N_total": 0,
            "N_routed": 0,
            "acc_cascade": 0.0,
            "acc_routed": None,
            "transfers": {
                "corrected": 0,
                "damaged": 0,
                "both_correct": 0,
                "both_wrong": 0,
                "net_gain": 0,
            },
            "costs": {"total_cost_usd": 0.0, "routed_cost_usd": 0.0, "jev_cost_usd": 0.0},
        }

    def check_application_correctness(pred_record: Dict[str, Any], gold: Any) -> bool:
        """Helper to determine application correctness for Jev or S2 F0."""
        is_valid = bool(pred_record.get("status") == "valid")
        if not is_valid:
            return False

        decision = pred_record.get("decision")
        gold_label = getattr(gold, "gold_label", "") if hasattr(gold, "gold_label") else gold.get("gold_label", "")
        is_oos = bool(getattr(gold, "is_oos", False) if hasattr(gold, "is_oos") else gold.get("is_oos", False))

        if dataset_id == "clinc150":
            model_id = pred_record.get("model_id", "")
            if model_id == "jev-1.14.0":
                # Jev uses entropy threshold tau* to reject
                probs = pred_record.get("probabilities") or {}
                K = num_classes if num_classes is not None else 150
                h_norm = compute_normalized_entropy(probs, num_classes=K)
                is_rejected = bool(h_norm >= (clinc_tau or 0.159485))
                if not is_oos:
                    return (not is_rejected) and (decision == gold_label)
                else:
                    return is_rejected
            else:
                # System-2 F0 decision has explicit REJECT option
                is_rejected = bool(decision == "REJECT")
                if not is_oos:
                    return (not is_rejected) and (decision == gold_label)
                else:
                    return is_rejected
        else:
            return bool(decision == gold_label)

    cascade_correct_count = 0
    routed_correct_count = 0
    jev_correct_count = 0

    corrected_count = 0
    damaged_count = 0
    both_correct_count = 0
    both_wrong_count = 0

    jev_cost_sum = 0.0
    s2_routed_cost_sum = 0.0

    for sid in sample_ids:
        j_pred = jev_predictions_map[sid]
        s2_pred = s2_predictions_map[sid]
        gold = gold_records_map[sid]

        j_correct = check_application_correctness(j_pred, gold)
        s2_correct = check_application_correctness(s2_pred, gold)

        j_cost = float(j_pred.get("cost", 0.0))
        s2_cost = float(s2_pred.get("cost", 0.0))
        jev_cost_sum += j_cost

        if j_correct:
            jev_correct_count += 1

        is_routed = bool(sid in mask_set)

        if is_routed:
            s2_routed_cost_sum += s2_cost
            c_correct = s2_correct
            if s2_correct:
                routed_correct_count += 1

            # Transfer categorization on routed samples
            if not j_correct and s2_correct:
                corrected_count += 1
            elif j_correct and not s2_correct:
                damaged_count += 1
            elif j_correct and s2_correct:
                both_correct_count += 1
            else:
                both_wrong_count += 1
        else:
            c_correct = j_correct

        if c_correct:
            cascade_correct_count += 1

    acc_cascade = float(cascade_correct_count) / float(N_total)
    acc_routed = float(routed_correct_count) / float(N_routed) if N_routed > 0 else None
    net_gain = corrected_count - damaged_count

    # Exact identity assertion: Net gain must equal cascade_correct - jev_correct
    expected_net_gain = cascade_correct_count - jev_correct_count
    assert net_gain == expected_net_gain, (
        f"Net gain identity violated: {net_gain} != {expected_net_gain} "
        f"(corrected={corrected_count}, damaged={damaged_count})"
    )

    return {
        "N_total": N_total,
        "N_routed": N_routed,
        "acc_cascade": round(acc_cascade, 6),
        "acc_routed": round(acc_routed, 6) if acc_routed is not None else None,
        "acc_jev_baseline": round(float(jev_correct_count) / float(N_total), 6),
        "transfers": {
            "corrected": corrected_count,
            "damaged": damaged_count,
            "both_correct": both_correct_count,
            "both_wrong": both_wrong_count,
            "net_gain": net_gain,
            "net_accuracy_gain_pp": round((acc_cascade - (float(jev_correct_count) / float(N_total))) * 100.0, 4),
        },
        "costs": {
            "total_deploy_cost_usd": round(jev_cost_sum + s2_routed_cost_sum, 6),
            "routed_s2_cost_usd": round(s2_routed_cost_sum, 6),
            "jev_cost_usd": round(jev_cost_sum, 6),
        },
    }

