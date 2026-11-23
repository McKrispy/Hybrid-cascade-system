"""Handoff effects, cascade evaluation, and error-following analysis for HybridCascade.

Timestamp: 2026-11-10 13:27 (Asia/Shanghai)

Evaluates:
1. Four handoff conditions: F0, F1, C0, C1 on development pool masks.
2. Main comparison: F1 - F0 accuracy difference at primary q=25% (4-dataset macro average).
3. Multiclass comparisons: Pruning (C0 - F0), Prior (C1 - C0), Interaction on 3 multiclass datasets.
4. Gold-in-Top-2 coverage: Diagnostics for candidate pruning availability.
5. Error-following / anchoring: Probability that System-2 selects Jev's incorrect Top-1 label (F1 vs F0).
6. Strict separation of Calibration role vs Selection role.
"""

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import numpy as np

from src.evaluation.bootstrap import paired_group_bootstrap_multicondition
from src.evaluation.metrics import compute_normalized_entropy
from src.input.splits import GoldRecord, SampleInput


def check_sample_correctness(
    prediction: Dict[str, Any],
    gold: Any,
    dataset_id: str,
    clinc_tau: Optional[float] = None,
    num_classes: Optional[int] = None,
) -> bool:
    """Determine application correctness for a prediction record."""
    if not prediction or prediction.get("status") != "valid":
        return False

    decision = prediction.get("decision")
    gold_label = getattr(gold, "gold_label", "") if hasattr(gold, "gold_label") else gold.get("gold_label", "")
    is_oos = bool(getattr(gold, "is_oos", False) if hasattr(gold, "is_oos") else gold.get("is_oos", False))

    if dataset_id == "clinc150":
        model_id = prediction.get("model_id", "")
        if model_id == "jev-1.14.0":
            probs = prediction.get("probabilities") or {}
            K = num_classes if num_classes is not None else 150
            h_norm = compute_normalized_entropy(probs, num_classes=K)
            is_rejected = bool(h_norm >= (clinc_tau or 0.159485))
            if not is_oos:
                return (not is_rejected) and (decision == gold_label)
            else:
                return is_rejected
        else:
            is_rejected = bool(decision == "REJECT")
            if not is_oos:
                return (not is_rejected) and (decision == gold_label)
            else:
                return is_rejected
    else:
        return bool(decision == gold_label)


def analyze_gold_in_top2_coverage(
    samples: Sequence[SampleInput],
    jev_preds_map: Dict[str, Dict[str, Any]],
    golds_map: Dict[str, Any],
    label_schemas: Dict[str, Any],
) -> Dict[str, Any]:
    """Calculate the percentage of samples where the gold label is present in Jev Top-2."""
    by_dataset: Dict[str, Dict[str, Any]] = {}
    from src.handoff.priors import extract_top2_prior

    dataset_samples: Dict[str, List[SampleInput]] = {}
    for s in samples:
        dataset_samples.setdefault(s.dataset_id, []).append(s)

    for ds_id, s_list in dataset_samples.items():
        total = len(s_list)
        gold_in_top2_count = 0
        schema = label_schemas.get(ds_id)
        allowed_labels = schema.allowed_labels if hasattr(schema, "allowed_labels") else schema.get("allowed_labels", [])

        for s in s_list:
            gold = golds_map.get(s.sample_id)
            gold_label = getattr(gold, "gold_label", "") if hasattr(gold, "gold_label") else gold.get("gold_label", "")
            is_oos = bool(getattr(gold, "is_oos", False) if hasattr(gold, "is_oos") else gold.get("is_oos", False))

            if ds_id == "clinc150" and is_oos:
                # OOS gold is REJECT, which is always retained in C0/C1
                gold_in_top2_count += 1
                continue

            j_pred = jev_preds_map.get(s.sample_id, {})
            probs = j_pred.get("probabilities", {})
            top2_labels, _ = extract_top2_prior(probs, allowed_labels)
            if gold_label in top2_labels:
                gold_in_top2_count += 1

        coverage_rate = float(gold_in_top2_count) / float(total) if total > 0 else 0.0
        by_dataset[ds_id] = {
            "total_samples": total,
            "gold_in_top2_count": gold_in_top2_count,
            "gold_in_top2_coverage": round(coverage_rate, 6),
        }

    return by_dataset


def analyze_error_following(
    sample_ids: Sequence[str],
    jev_preds_map: Dict[str, Dict[str, Any]],
    f0_preds_map: Dict[str, Dict[str, Any]],
    f1_preds_map: Dict[str, Dict[str, Any]],
    golds_map: Dict[str, Any],
    dataset_id: str,
    clinc_tau: Optional[float] = None,
    num_classes: Optional[int] = None,
) -> Dict[str, Any]:
    """Analyze anchoring bias: When Jev is wrong, how often does S2 follow Jev's incorrect Top-1?"""
    n_jev_errors = 0
    f0_followed_count = 0
    f1_followed_count = 0

    for sid in sample_ids:
        j_pred = jev_preds_map.get(sid, {})
        gold = golds_map.get(sid)
        j_correct = check_sample_correctness(j_pred, gold, dataset_id, clinc_tau, num_classes)

        if not j_correct:
            n_jev_errors += 1
            j_top1 = j_pred.get("decision")

            f0_decision = f0_preds_map.get(sid, {}).get("decision")
            f1_decision = f1_preds_map.get(sid, {}).get("decision")

            if f0_decision and f0_decision == j_top1:
                f0_followed_count += 1
            if f1_decision and f1_decision == j_top1:
                f1_followed_count += 1

    f0_rate = float(f0_followed_count) / float(n_jev_errors) if n_jev_errors > 0 else 0.0
    f1_rate = float(f1_followed_count) / float(n_jev_errors) if n_jev_errors > 0 else 0.0
    delta_rate = f1_rate - f0_rate

    return {
        "dataset_id": dataset_id,
        "n_jev_errors": n_jev_errors,
        "f0_followed_count": f0_followed_count,
        "f0_error_following_rate": round(f0_rate, 6),
        "f1_followed_count": f1_followed_count,
        "f1_error_following_rate": round(f1_rate, 6),
        "delta_error_following_rate": round(delta_rate, 6),
    }


def evaluate_handoff_batch(
    dataset_ids: Sequence[str],
    samples_by_dataset: Dict[str, List[SampleInput]],
    jev_preds_map: Dict[str, Dict[str, Any]],
    s2_all_preds_map: Dict[str, Dict[str, Dict[str, Any]]],  # model_id -> arm_id -> sid -> pred
    golds_map: Dict[str, Any],
    primary_masks: Dict[str, Any],  # key: f"{ds_id}:{role}" -> dict with 'masks'
    role: str,
    model_ids: Sequence[str] = ("gpt-6.1-sol", "deepseek-v4.1-flash"),
    q_rates: Sequence[float] = (0.15, 0.25, 0.35),
    primary_q: float = 0.25,
    num_bootstrap: int = 2000,
    seed: int = 42,
    clinc_tau: float = 0.159485,
    num_classes_map: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """Evaluate full handoff experimental battery across models, arms, and budgets.
    
    Returns structured results for results.json.
    """
    K_map = num_classes_map or {"banking77": 77, "clinc150": 150, "anli_r2r3": 3, "strategyqa": 2}
    results_by_model: Dict[str, Any] = {}

    for model_id in model_ids:
        model_arms = s2_all_preds_map[model_id]
        f0_map = model_arms["F0"]
        f1_map = model_arms["F1"]
        c0_map = model_arms["C0"]
        c1_map = model_arms["C1"]

        # Evaluate at each q rate
        eval_by_q: Dict[str, Any] = {}
        for q in q_rates:
            q_key = f"{q:.2f}"
            dataset_records_for_boot: Dict[str, List[Dict[str, Any]]] = {}
            error_following_by_ds: Dict[str, Any] = {}

            for ds_id in dataset_ids:
                s_list = samples_by_dataset[ds_id]
                mask_info = primary_masks["by_dataset_role"].get(f"{ds_id}:{role}", {})
                q_mask_set = set(mask_info.get("masks", {}).get(q_key, []))

                ds_records = []
                for s in s_list:
                    sid = s.sample_id
                    gold = golds_map[sid]
                    K = K_map[ds_id]

                    j_corr = float(check_sample_correctness(jev_preds_map[sid], gold, ds_id, clinc_tau, K))

                    if sid in q_mask_set:
                        f0_corr = float(check_sample_correctness(f0_map[sid], gold, ds_id, clinc_tau, K))
                        f1_corr = float(check_sample_correctness(f1_map[sid], gold, ds_id, clinc_tau, K))
                        c0_corr = float(check_sample_correctness(c0_map[sid], gold, ds_id, clinc_tau, K))
                        c1_corr = float(check_sample_correctness(c1_map[sid], gold, ds_id, clinc_tau, K))
                    else:
                        # Below escalation threshold: retains Jev prediction
                        f0_corr = j_corr
                        f1_corr = j_corr
                        c0_corr = j_corr
                        c1_corr = j_corr

                    ds_records.append({
                        "sample_id": sid,
                        "group_id": s.group_id,
                        "f0_correct": f0_corr,
                        "f1_correct": f1_corr,
                        "c0_correct": c0_corr,
                        "c1_correct": c1_corr,
                    })

                dataset_records_for_boot[ds_id] = ds_records

                # Error-following analysis on escalated samples
                escalated_sids = [s.sample_id for s in s_list if s.sample_id in q_mask_set]
                error_following_by_ds[ds_id] = analyze_error_following(
                    sample_ids=escalated_sids,
                    jev_preds_map=jev_preds_map,
                    f0_preds_map=f0_map,
                    f1_preds_map=f1_map,
                    golds_map=golds_map,
                    dataset_id=ds_id,
                    clinc_tau=clinc_tau,
                    num_classes=K_map[ds_id],
                )

            # Run paired group bootstrap across all datasets
            boot_res = paired_group_bootstrap_multicondition(
                dataset_records=dataset_records_for_boot,
                num_bootstrap=num_bootstrap,
                seed=seed,
                alpha=0.05,
            )

            eval_by_q[q_key] = {
                "q": q,
                "bootstrap_evaluation": boot_res,
                "error_following": error_following_by_ds,
            }

        # Also evaluate on the upgraded subset directly at primary_q
        pri_q_key = f"{primary_q:.2f}"
        subset_records_for_boot: Dict[str, List[Dict[str, Any]]] = {}
        for ds_id in dataset_ids:
            s_list = samples_by_dataset[ds_id]
            mask_info = primary_masks["by_dataset_role"].get(f"{ds_id}:{role}", {})
            pri_mask_set = set(mask_info.get("masks", {}).get(pri_q_key, []))

            sub_records = []
            for s in s_list:
                sid = s.sample_id
                if sid not in pri_mask_set:
                    continue
                gold = golds_map[sid]
                K = K_map[ds_id]

                f0_corr = float(check_sample_correctness(f0_map[sid], gold, ds_id, clinc_tau, K))
                f1_corr = float(check_sample_correctness(f1_map[sid], gold, ds_id, clinc_tau, K))
                c0_corr = float(check_sample_correctness(c0_map[sid], gold, ds_id, clinc_tau, K))
                c1_corr = float(check_sample_correctness(c1_map[sid], gold, ds_id, clinc_tau, K))

                sub_records.append({
                    "sample_id": sid,
                    "group_id": s.group_id,
                    "f0_correct": f0_corr,
                    "f1_correct": f1_corr,
                    "c0_correct": c0_corr,
                    "c1_correct": c1_corr,
                })
            subset_records_for_boot[ds_id] = sub_records

        upgraded_subset_boot = paired_group_bootstrap_multicondition(
            dataset_records=subset_records_for_boot,
            num_bootstrap=num_bootstrap,
            seed=seed,
            alpha=0.05,
        )

        results_by_model[model_id] = {
            "model_id": model_id,
            "role": role,
            "by_q": eval_by_q,
            "upgraded_subset_primary_q": upgraded_subset_boot,
        }

    return results_by_model

