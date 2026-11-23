"""Paired group bootstrap evaluation for clustered samples and macro metrics.

Timestamp: 2026-11-10 13:26 (Asia/Shanghai)

Enforces:
1. Resampling whole groups (group_id) with replacement, preserving premise clusters (e.g. ANLI).
2. Stratified resampling within each dataset to maintain exact dataset balance.
3. Paired evaluation: conditions A and B evaluated on identical resampled replicates.
4. Percentile confidence intervals (default 95%) and two-tailed empirical p-values.
"""

from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np


def compute_ci_and_pvalue(
    bootstrap_diffs: np.ndarray,
    observed_diff: float,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Calculate percentile CI, standard error, and two-tailed p-value from bootstrap draws."""
    lower_pct = (alpha / 2.0) * 100.0
    upper_pct = (1.0 - (alpha / 2.0)) * 100.0

    ci_lower = float(np.percentile(bootstrap_diffs, lower_pct))
    ci_upper = float(np.percentile(bootstrap_diffs, upper_pct))
    se = float(np.std(bootstrap_diffs, ddof=1)) if len(bootstrap_diffs) > 1 else 0.0

    # Empirical two-tailed p-value against null hypothesis Diff = 0
    p_le = np.mean(bootstrap_diffs <= 0.0)
    p_ge = np.mean(bootstrap_diffs >= 0.0)
    p_val = float(min(1.0, 2.0 * min(p_le, p_ge)))

    is_significant = bool((ci_lower > 0.0 and ci_upper > 0.0) or (ci_lower < 0.0 and ci_upper < 0.0))

    return {
        "observed_diff": round(float(observed_diff), 6),
        "mean_diff": round(float(np.mean(bootstrap_diffs)), 6),
        "std_error": round(se, 6),
        "ci_lower": round(ci_lower, 6),
        "ci_upper": round(ci_upper, 6),
        "confidence_level": round(1.0 - alpha, 2),
        "p_value": round(p_val, 6),
        "significant": is_significant,
    }


def paired_group_bootstrap_series(
    group_ids: Sequence[str],
    values_a: Sequence[float],
    values_b: Sequence[float],
    num_bootstrap: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Single-dataset paired group bootstrap comparing condition A vs condition B.
    
    Args:
        group_ids: Sequence of group identifiers for each sample.
        values_a: Outcomes for condition A (e.g. 1.0 for correct, 0.0 for incorrect).
        values_b: Outcomes for condition B.
        num_bootstrap: Number of bootstrap iterations.
        seed: Random seed for reproducibility.
        alpha: Significance level (default 0.05 for 95% CI).
    """
    N = len(group_ids)
    if N == 0 or len(values_a) != N or len(values_b) != N:
        raise ValueError("group_ids, values_a, and values_b must have identical non-zero lengths.")

    # Cluster indices by group_id
    group_to_indices = defaultdict(list)
    for idx, gid in enumerate(group_ids):
        group_to_indices[gid].append(idx)

    unique_groups = sorted(list(group_to_indices.keys()))
    M = len(unique_groups)

    va = np.array(values_a, dtype=np.float64)
    vb = np.array(values_b, dtype=np.float64)

    obs_a = float(np.mean(va))
    obs_b = float(np.mean(vb))
    obs_diff = obs_a - obs_b

    rng = np.random.default_rng(seed)
    boot_diffs = np.empty(num_bootstrap, dtype=np.float64)

    # Pre-aggregate group sums and counts for high-speed vector bootstrap
    group_sums_a = np.array([np.sum(va[group_to_indices[g]]) for g in unique_groups])
    group_sums_b = np.array([np.sum(vb[group_to_indices[g]]) for g in unique_groups])
    group_counts = np.array([len(group_to_indices[g]) for g in unique_groups], dtype=np.float64)

    for b in range(num_bootstrap):
        sampled_group_indices = rng.integers(0, M, size=M)
        total_a = np.sum(group_sums_a[sampled_group_indices])
        total_b = np.sum(group_sums_b[sampled_group_indices])
        total_n = np.sum(group_counts[sampled_group_indices])
        boot_diffs[b] = (total_a - total_b) / total_n

    res = compute_ci_and_pvalue(boot_diffs, obs_diff, alpha=alpha)
    res["mean_a"] = round(obs_a, 6)
    res["mean_b"] = round(obs_b, 6)
    res["num_groups"] = M
    res["num_samples"] = N
    return res


def paired_group_bootstrap_multicondition(
    dataset_records: Dict[str, List[Dict[str, Any]]],
    num_bootstrap: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Stratified paired group bootstrap across multiple datasets and conditions.
    
    Args:
        dataset_records: Mapping from dataset_id to list of sample records.
            Each sample record must contain:
            - 'sample_id': str
            - 'group_id': str
            - 'f0_correct': float (1.0 or 0.0)
            - 'f1_correct': float (1.0 or 0.0)
            - 'c0_correct': float (1.0 or 0.0)
            - 'c1_correct': float (1.0 or 0.0)
            
    Returns:
        Dictionary of evaluated paired contrasts with 95% CIs:
        - By dataset:
            - F1 - F0
            - C0 - F0
            - C1 - C0
            - Interaction: (C1 - C0) - (F1 - F0)
        - 4-Dataset Macro Unweighted Average:
            - F1 - F0
        - 3-Dataset Multiclass Macro Unweighted Average (banking77, clinc150, anli_r2r3):
            - C0 - F0
            - C1 - C0
            - Interaction: (C1 - C0) - (F1 - F0)
    """
    rng = np.random.default_rng(seed)

    # Prepare group summaries per dataset
    ds_prepared: Dict[str, Any] = {}
    for ds_id, records in dataset_records.items():
        group_to_records = defaultdict(list)
        for r in records:
            group_to_records[r["group_id"]].append(r)

        unique_groups = sorted(list(group_to_records.keys()))
        M = len(unique_groups)
        N = len(records)

        g_f0 = np.array([sum(r["f0_correct"] for r in group_to_records[g]) for g in unique_groups], dtype=np.float64)
        g_f1 = np.array([sum(r["f1_correct"] for r in group_to_records[g]) for g in unique_groups], dtype=np.float64)
        g_c0 = np.array([sum(r["c0_correct"] for r in group_to_records[g]) for g in unique_groups], dtype=np.float64)
        g_c1 = np.array([sum(r["c1_correct"] for r in group_to_records[g]) for g in unique_groups], dtype=np.float64)
        g_counts = np.array([len(group_to_records[g]) for g in unique_groups], dtype=np.float64)

        obs_f0 = float(sum(r["f0_correct"] for r in records)) / float(N)
        obs_f1 = float(sum(r["f1_correct"] for r in records)) / float(N)
        obs_c0 = float(sum(r["c0_correct"] for r in records)) / float(N)
        obs_c1 = float(sum(r["c1_correct"] for r in records)) / float(N)

        ds_prepared[ds_id] = {
            "M": M,
            "N": N,
            "unique_groups": unique_groups,
            "g_f0": g_f0,
            "g_f1": g_f1,
            "g_c0": g_c0,
            "g_c1": g_c1,
            "g_counts": g_counts,
            "obs": {
                "f0": obs_f0,
                "f1": obs_f1,
                "c0": obs_c0,
                "c1": obs_c1,
                "f1_minus_f0": obs_f1 - obs_f0,
                "c0_minus_f0": obs_c0 - obs_f0,
                "c1_minus_c0": obs_c1 - obs_c0,
                "interaction": (obs_c1 - obs_c0) - (obs_f1 - obs_f0),
            },
        }

    # Pre-allocate bootstrap arrays per dataset
    boot_ds_diffs = {
        ds_id: {
            "f1_minus_f0": np.empty(num_bootstrap, dtype=np.float64),
            "c0_minus_f0": np.empty(num_bootstrap, dtype=np.float64),
            "c1_minus_c0": np.empty(num_bootstrap, dtype=np.float64),
            "interaction": np.empty(num_bootstrap, dtype=np.float64),
        }
        for ds_id in dataset_records
    }

    # Run bootstrap draws across all datasets
    for ds_id, prep in ds_prepared.items():
        M = prep["M"]
        g_f0 = prep["g_f0"]
        g_f1 = prep["g_f1"]
        g_c0 = prep["g_c0"]
        g_c1 = prep["g_c1"]
        g_counts = prep["g_counts"]

        for b in range(num_bootstrap):
            sampled_idx = rng.integers(0, M, size=M)
            total_n = np.sum(g_counts[sampled_idx])
            mean_f0 = np.sum(g_f0[sampled_idx]) / total_n
            mean_f1 = np.sum(g_f1[sampled_idx]) / total_n
            mean_c0 = np.sum(g_c0[sampled_idx]) / total_n
            mean_c1 = np.sum(g_c1[sampled_idx]) / total_n

            d_f1_f0 = mean_f1 - mean_f0
            d_c0_f0 = mean_c0 - mean_f0
            d_c1_c0 = mean_c1 - mean_c0
            d_inter = d_c1_c0 - d_f1_f0

            boot_ds_diffs[ds_id]["f1_minus_f0"][b] = d_f1_f0
            boot_ds_diffs[ds_id]["c0_minus_f0"][b] = d_c0_f0
            boot_ds_diffs[ds_id]["c1_minus_c0"][b] = d_c1_c0
            boot_ds_diffs[ds_id]["interaction"][b] = d_inter

    # Calculate per-dataset CI and summary
    by_dataset: Dict[str, Any] = {}
    for ds_id, prep in ds_prepared.items():
        obs = prep["obs"]
        by_dataset[ds_id] = {
            "N": prep["N"],
            "num_groups": prep["M"],
            "accuracies": {
                "F0": round(obs["f0"], 6),
                "F1": round(obs["f1"], 6),
                "C0": round(obs["c0"], 6),
                "C1": round(obs["c1"], 6),
            },
            "contrasts": {
                "f1_minus_f0": compute_ci_and_pvalue(boot_ds_diffs[ds_id]["f1_minus_f0"], obs["f1_minus_f0"], alpha),
                "c0_minus_f0": compute_ci_and_pvalue(boot_ds_diffs[ds_id]["c0_minus_f0"], obs["c0_minus_f0"], alpha),
                "c1_minus_c0": compute_ci_and_pvalue(boot_ds_diffs[ds_id]["c1_minus_c0"], obs["c1_minus_c0"], alpha),
                "interaction": compute_ci_and_pvalue(boot_ds_diffs[ds_id]["interaction"], obs["interaction"], alpha),
            },
        }

    # 4-Dataset Macro Unweighted Average for F1 - F0
    all_ds_keys = sorted(list(dataset_records.keys()))
    boot_macro_f1_f0 = np.mean([boot_ds_diffs[ds]["f1_minus_f0"] for ds in all_ds_keys], axis=0)
    obs_macro_f1_f0 = float(np.mean([ds_prepared[ds]["obs"]["f1_minus_f0"] for ds in all_ds_keys]))
    macro_4_f1_f0 = compute_ci_and_pvalue(boot_macro_f1_f0, obs_macro_f1_f0, alpha)

    # 3-Dataset Multiclass Macro Unweighted Average (exclude strategyqa)
    multiclass_keys = [ds for ds in all_ds_keys if ds != "strategyqa"]
    macro_3_results = {}
    if multiclass_keys:
        for contrast_name in ["c0_minus_f0", "c1_minus_c0", "interaction"]:
            boot_arr = np.mean([boot_ds_diffs[ds][contrast_name] for ds in multiclass_keys], axis=0)
            obs_val = float(np.mean([ds_prepared[ds]["obs"][contrast_name] for ds in multiclass_keys]))
            macro_3_results[contrast_name] = compute_ci_and_pvalue(boot_arr, obs_val, alpha)

    return {
        "by_dataset": by_dataset,
        "macro_4_datasets": {
            "datasets": all_ds_keys,
            "f1_minus_f0": macro_4_f1_f0,
        },
        "macro_3_multiclass": {
            "datasets": multiclass_keys,
            "contrasts": macro_3_results,
        },
    }

