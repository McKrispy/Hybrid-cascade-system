"""Statistical evaluation for Batch B06: FR numerical prior control.

Timestamp: 2026-11-10 16:10 (Asia/Shanghai)

Disentangles the anchoring bias into:
1. Identity / Ranking Effect: FR - F0
2. Numerical Probability Effect: F1 - FR
3. Total Prior Effect on B06 subset: F1 - F0

Computes 4-dataset equal-weight macro averages with paired group bootstrap (default 2,000 resamples).
"""

from collections import defaultdict
from typing import Any, Dict, List, Sequence
import numpy as np

from src.evaluation.bootstrap import compute_ci_and_pvalue


def evaluate_fr_control_batch(
    dataset_ids: Sequence[str],
    samples_by_dataset: Dict[str, List[Any]],
    golds_map: Dict[str, Any],
    predictions_by_model: Dict[str, Dict[str, Dict[str, Any]]],
    role: str,
    num_bootstrap: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Evaluate F0, FR, and F1 comparisons on the B06 sampled subset.
    
    Args:
        dataset_ids: List of 4 datasets.
        samples_by_dataset: Map from dataset_id to list of SampleInput objects.
        golds_map: Map from sample_id to GoldRecord.
        predictions_by_model: model_id -> arm_id ("F0", "FR", "F1") -> sample_id -> prediction dict.
        role: "selection" or "calibration".
        num_bootstrap: Number of bootstrap iterations.
        seed: Random seed.
        alpha: Alpha level for CI.
        
    Returns:
        Structured evaluation dictionary per model and macro average.
    """
    rng = np.random.default_rng(seed)
    eval_by_model: Dict[str, Any] = {}

    for model_id, arms_map in predictions_by_model.items():
        f0_preds = arms_map.get("F0", {})
        fr_preds = arms_map.get("FR", {})
        f1_preds = arms_map.get("F1", {})

        dataset_records: Dict[str, List[Dict[str, Any]]] = {}
        by_dataset_metrics: Dict[str, Any] = {}

        for ds_id in dataset_ids:
            samples = samples_by_dataset.get(ds_id, [])
            records: List[Dict[str, Any]] = []

            for s in samples:
                sid = s.sample_id
                gold_rec = golds_map[sid]
                gold = gold_rec.gold_label

                p_f0 = f0_preds.get(sid, {})
                p_fr = fr_preds.get(sid, {})
                p_f1 = f1_preds.get(sid, {})

                dec_f0 = p_f0.get("decision")
                dec_fr = p_fr.get("decision")
                dec_f1 = p_f1.get("decision")

                f0_corr = 1.0 if (dec_f0 is not None and dec_f0 == gold) else 0.0
                fr_corr = 1.0 if (dec_fr is not None and dec_fr == gold) else 0.0
                f1_corr = 1.0 if (dec_f1 is not None and dec_f1 == gold) else 0.0

                records.append({
                    "sample_id": sid,
                    "group_id": s.group_id,
                    "gold": gold,
                    "f0_decision": dec_f0,
                    "fr_decision": dec_fr,
                    "f1_decision": dec_f1,
                    "f0_correct": f0_corr,
                    "fr_correct": fr_corr,
                    "f1_correct": f1_corr,
                })

            dataset_records[ds_id] = records

            # Dataset-level observed accuracies
            n_samples = len(records)
            acc_f0 = float(np.mean([r["f0_correct"] for r in records])) if n_samples > 0 else 0.0
            acc_fr = float(np.mean([r["fr_correct"] for r in records])) if n_samples > 0 else 0.0
            acc_f1 = float(np.mean([r["f1_correct"] for r in records])) if n_samples > 0 else 0.0

            by_dataset_metrics[ds_id] = {
                "num_samples": n_samples,
                "f0_accuracy": round(acc_f0, 6),
                "fr_accuracy": round(acc_fr, 6),
                "f1_accuracy": round(acc_f1, 6),
                "diff_fr_minus_f0": round(acc_fr - acc_f0, 6),
                "diff_f1_minus_fr": round(acc_f1 - acc_fr, 6),
                "diff_f1_minus_f0": round(acc_f1 - acc_f0, 6),
            }

        # Multi-dataset Stratified Paired Group Bootstrap
        # 1. Pre-aggregate by group per dataset
        ds_group_aggregates: Dict[str, List[Dict[str, float]]] = {}
        for ds_id, records in dataset_records.items():
            g_map = defaultdict(lambda: {"n": 0.0, "f0": 0.0, "fr": 0.0, "f1": 0.0})
            for r in records:
                gid = r["group_id"]
                g_map[gid]["n"] += 1.0
                g_map[gid]["f0"] += r["f0_correct"]
                g_map[gid]["fr"] += r["fr_correct"]
                g_map[gid]["f1"] += r["f1_correct"]
            ds_group_aggregates[ds_id] = list(g_map.values())

        # Bootstrap draws
        boot_macro_fr_f0 = np.empty(num_bootstrap, dtype=float)
        boot_macro_f1_fr = np.empty(num_bootstrap, dtype=float)
        boot_macro_f1_f0 = np.empty(num_bootstrap, dtype=float)

        boot_ds_fr_f0: Dict[str, np.ndarray] = {ds_id: np.empty(num_bootstrap, dtype=float) for ds_id in dataset_ids}
        boot_ds_f1_fr: Dict[str, np.ndarray] = {ds_id: np.empty(num_bootstrap, dtype=float) for ds_id in dataset_ids}
        boot_ds_f1_f0: Dict[str, np.ndarray] = {ds_id: np.empty(num_bootstrap, dtype=float) for ds_id in dataset_ids}

        for b in range(num_bootstrap):
            b_ds_diff_fr_f0 = []
            b_ds_diff_f1_fr = []
            b_ds_diff_f1_f0 = []

            for ds_id in dataset_ids:
                groups = ds_group_aggregates[ds_id]
                m_groups = len(groups)
                if m_groups == 0:
                    continue
                sampled_indices = rng.integers(0, m_groups, size=m_groups)

                tot_n = sum(groups[i]["n"] for i in sampled_indices)
                tot_f0 = sum(groups[i]["f0"] for i in sampled_indices)
                tot_fr = sum(groups[i]["fr"] for i in sampled_indices)
                tot_f1 = sum(groups[i]["f1"] for i in sampled_indices)

                d_fr_f0 = (tot_fr - tot_f0) / tot_n if tot_n > 0 else 0.0
                d_f1_fr = (tot_f1 - tot_fr) / tot_n if tot_n > 0 else 0.0
                d_f1_f0 = (tot_f1 - tot_f0) / tot_n if tot_n > 0 else 0.0

                boot_ds_fr_f0[ds_id][b] = d_fr_f0
                boot_ds_f1_fr[ds_id][b] = d_f1_fr
                boot_ds_f1_f0[ds_id][b] = d_f1_f0

                b_ds_diff_fr_f0.append(d_fr_f0)
                b_ds_diff_f1_fr.append(d_f1_fr)
                b_ds_diff_f1_f0.append(d_f1_f0)

            # Equal-weight 4-dataset macro average
            boot_macro_fr_f0[b] = np.mean(b_ds_diff_fr_f0)
            boot_macro_f1_fr[b] = np.mean(b_ds_diff_f1_fr)
            boot_macro_f1_f0[b] = np.mean(b_ds_diff_f1_f0)

        # Macro observed statistics
        obs_macro_f0 = float(np.mean([by_dataset_metrics[d]["f0_accuracy"] for d in dataset_ids]))
        obs_macro_fr = float(np.mean([by_dataset_metrics[d]["fr_accuracy"] for d in dataset_ids]))
        obs_macro_f1 = float(np.mean([by_dataset_metrics[d]["f1_accuracy"] for d in dataset_ids]))

        obs_macro_fr_f0 = obs_macro_fr - obs_macro_f0
        obs_macro_f1_fr = obs_macro_f1 - obs_macro_fr
        obs_macro_f1_f0 = obs_macro_f1 - obs_macro_f0

        # Dataset-level bootstrap CIs
        for ds_id in dataset_ids:
            obs = by_dataset_metrics[ds_id]
            by_dataset_metrics[ds_id]["bootstrap_fr_minus_f0"] = compute_ci_and_pvalue(
                boot_ds_fr_f0[ds_id], obs["diff_fr_minus_f0"], alpha=alpha
            )
            by_dataset_metrics[ds_id]["bootstrap_f1_minus_fr"] = compute_ci_and_pvalue(
                boot_ds_f1_fr[ds_id], obs["diff_f1_minus_fr"], alpha=alpha
            )
            by_dataset_metrics[ds_id]["bootstrap_f1_minus_f0"] = compute_ci_and_pvalue(
                boot_ds_f1_f0[ds_id], obs["diff_f1_minus_f0"], alpha=alpha
            )

        # Macro bootstrap CIs
        macro_eval = {
            "num_datasets": len(dataset_ids),
            "macro_f0_accuracy": round(obs_macro_f0, 6),
            "macro_fr_accuracy": round(obs_macro_fr, 6),
            "macro_f1_accuracy": round(obs_macro_f1, 6),
            "identity_rank_effect_fr_minus_f0": compute_ci_and_pvalue(
                boot_macro_fr_f0, obs_macro_fr_f0, alpha=alpha
            ),
            "numerical_probability_effect_f1_minus_fr": compute_ci_and_pvalue(
                boot_macro_f1_fr, obs_macro_f1_fr, alpha=alpha
            ),
            "total_effect_f1_minus_f0": compute_ci_and_pvalue(
                boot_macro_f1_f0, obs_macro_f1_f0, alpha=alpha
            ),
        }

        eval_by_model[model_id] = {
            "model_id": model_id,
            "role": role,
            "total_samples": sum(len(recs) for recs in dataset_records.values()),
            "by_dataset": by_dataset_metrics,
            "macro_4_datasets": macro_eval,
        }

    return eval_by_model

