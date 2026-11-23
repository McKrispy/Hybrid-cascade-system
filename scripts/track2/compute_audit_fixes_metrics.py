"""Compute offline audit fix metrics for Q2, Q3, Q4, and Q6 from archived runs."""

import json
import math
import sys
from collections import Counter, deque
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SANDBOX_ROOT = PROJECT_ROOT / "sandbox"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SANDBOX_ROOT))

from src.calibration.features import compute_application_error_target  # noqa: E402
from src.evaluation.metrics import compute_normalized_entropy  # noqa: E402
from src.input.loader import load_gold_records  # noqa: E402
from sandbox.daemon.gatekeeper import DynamicGatekeeper  # noqa: E402


def compute_q2_clinc_entropy() -> Dict[str, Any]:
    tau_old = 0.159485
    tau_151 = tau_old * (math.log(150.0) / math.log(151.0))

    b02_preds = json.loads((PROJECT_ROOT / "dev/runs/B02/B02-C05/predictions.json").read_text(encoding="utf-8"))
    b07_preds = json.loads((PROJECT_ROOT / "dev/runs/B07/B07-C01/predictions.json").read_text(encoding="utf-8"))

    jev_b02 = {p["sample_id"]: p for p in b02_preds if p.get("model_id") == "jev-1.14.0"}
    jev_b07 = {p["sample_id"]: p for p in b07_preds if p.get("model_id") == "jev-1.14.0"}

    def eval_split(role_name: str, jev_map: Dict[str, Any], danger_holdout: bool = False) -> Dict[str, Any]:
        golds = [
            g for g in load_gold_records(role=role_name, danger_enable_holdout_for_b07=danger_holdout)
            if g.dataset_id == "clinc150"
        ]
        n_id, n_oos = 0, 0
        id_ok_old, oos_ok_old = 0, 0
        id_ok_new, oos_ok_new = 0, 0
        max_h150, max_h151 = 0.0, 0.0
        for g in golds:
            jp = jev_map[g.sample_id]
            probs = jp.get("probabilities") or {}
            h150 = compute_normalized_entropy(probs, num_classes=150)
            h151 = compute_normalized_entropy(probs, num_classes=151)
            max_h150 = max(max_h150, h150)
            max_h151 = max(max_h151, h151)
            rej_old = (h150 >= tau_old)
            rej_new = (h151 >= tau_151)
            if g.is_oos:
                n_oos += 1
                oos_ok_old += int(rej_old)
                oos_ok_new += int(rej_new)
            else:
                n_id += 1
                id_ok_old += int((not rej_old) and jp.get("decision") == g.gold_label)
                id_ok_new += int((not rej_new) and jp.get("decision") == g.gold_label)
        # Balanced accuracy uses rejection TPR/TNR on OOS vs ID
        id_tpr = (n_id - sum(1 for g in golds if (not g.is_oos) and compute_normalized_entropy(jev_map[g.sample_id].get("probabilities") or {}, 151) >= tau_151)) / max(n_id, 1)
        oos_tnr = oos_ok_new / max(n_oos, 1)
        return {
            "n_total": len(golds),
            "n_id": n_id,
            "n_oos": n_oos,
            "tau_old_150": tau_old,
            "tau_151": round(tau_151, 6),
            "tau_151_4dp": round(tau_151, 4),
            "identical_decisions": (id_ok_old == id_ok_new and oos_ok_old == oos_ok_new),
            "app_acc_pct": round(100.0 * (id_ok_new + oos_ok_new) / max(len(golds), 1), 2),
            "id_app_acc_pct": round(100.0 * id_ok_new / max(n_id, 1), 2),
            "oos_reject_tnr_pct": round(100.0 * oos_tnr, 2),
            "id_accept_tpr_pct": round(100.0 * id_tpr, 2),
            "balanced_acc_pct": round(100.0 * 0.5 * (id_tpr + oos_tnr), 2),
            "max_observed_h150": round(max_h150, 6),
            "max_observed_h151": round(max_h151, 6),
        }

    return {
        "scale_factor_ln150_over_ln151": round(math.log(150.0) / math.log(151.0), 6),
        "calibration": eval_split("calibration", jev_b02),
        "selection": eval_split("selection", jev_b02),
        "holdout": eval_split("holdout", jev_b07, danger_holdout=True),
    }


def compute_q3_per_asset_queue() -> Dict[str, Any]:
    recs = [
        json.loads(ln)
        for ln in (PROJECT_ROOT / "archives" / "track2_alpaca_real" / "records.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    stream_by_id = {
        json.loads(ln)["tick_id"]: json.loads(ln)
        for ln in (PROJECT_ROOT / "archives" / "track2_alpaca_real" / "alpaca_real_30m_stream.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    }

    # Verify pooled gatekeeper matches recorded escalations, and simulate per-asset W=96 isolated queues
    # Sharing the same 60-min cooldown (by symbol) and global token bucket (cap=4, refill=1/1800s) OR per-asset quantile only
    import time as _time
    per_asset_windows: Dict[str, deque] = {s: deque(maxlen=96) for s in ("NVDA", "TSLA", "BTC/USD", "ETH/USD")}
    # Track quantile gate only (comparing PR_pooled >= 0.75 vs PR_per_asset >= 0.75) and full 3-layer with per-asset quantile
    gk_per_asset_full = DynamicGatekeeper(rolling_window_size=96, quantile_threshold=0.75, cooldown_seconds=3600.0)
    # Create per-asset DynamicGatekeeper instances ONLY for quantile + cooldown, plus shared token bucket
    from sandbox.daemon.gatekeeper import TokenBucket
    shared_bucket = TokenBucket(capacity=4.0, refill_interval_seconds=1800.0, refill_tokens=1.0)
    per_sym_gk = {
        s: DynamicGatekeeper(rolling_window_size=96, quantile_threshold=0.75, cooldown_seconds=3600.0)
        for s in ("NVDA", "TSLA", "BTC/USD", "ETH/USD")
    }
    for gk in per_sym_gk.values():
        gk.token_bucket = shared_bucket

    pooled_esc_ids = set()
    per_asset_esc_ids = set()
    pooled_quant_pass = set()
    per_asset_quant_pass = set()
    equity_open_pooled = []
    equity_open_per_asset = []

    pooled_hist = deque(maxlen=384)
    for r in recs:
        tid = r["tick_id"]
        sym = r["symbol"]
        u = float(r["jev_uncertainty"])
        st = stream_by_id[tid]
        news_ids = [n["news_id"] for n in st["news_stream"]]
        ts_val = _time.mktime(_time.strptime(r["timestamp"], "%Y-%m-%d %H:%M:%S"))

        if r["was_escalated"]:
            pooled_esc_ids.add(tid)
        if r["gate_reason"] != "below_quantile_threshold":
            pooled_quant_pass.add(tid)

        # Quantile-only counterfactual
        w_pool = list(pooled_hist) + [u]
        pr_pool = sum(1 for x in w_pool if x <= u) / len(w_pool)
        pooled_hist.append(u)

        w_pa = list(per_asset_windows[sym]) + [u]
        pr_pa = sum(1 for x in w_pa if x <= u) / len(w_pa)
        per_asset_windows[sym].append(u)
        if pr_pa >= 0.75:
            per_asset_quant_pass.add(tid)

        # Full 3-layer counterfactual with per-asset quantile W=96 + same cooldown + shared bucket
        esc_pa, reason_pa, _ = per_sym_gk[sym].evaluate_tick(sym, u, news_ids, now_ts=ts_val)
        if esc_pa:
            per_asset_esc_ids.add(tid)

        # Check equity ticks during Mon-Fri US open (13:30-15:00 UTC) after crypto enters (2026-11-16+)
        if sym in ("NVDA", "TSLA") and r["timestamp"] >= "2026-11-16" and r["timestamp"][11:16] in ("13:30", "14:00", "14:30", "15:00"):
            equity_open_pooled.append(int(r["was_escalated"]))
            equity_open_per_asset.append(int(esc_pa))

    inter = pooled_esc_ids & per_asset_esc_ids
    union = pooled_esc_ids | per_asset_esc_ids
    q_inter = pooled_quant_pass & per_asset_quant_pass
    q_union = pooled_quant_pass | per_asset_quant_pass
    return {
        "total_ticks": len(recs),
        "pooled_escalated_count": len(pooled_esc_ids),
        "pooled_escalation_rate_pct": round(100.0 * len(pooled_esc_ids) / len(recs), 2),
        "per_asset_escalated_count": len(per_asset_esc_ids),
        "per_asset_escalation_rate_pct": round(100.0 * len(per_asset_esc_ids) / len(recs), 2),
        "escalated_overlap_count": len(inter),
        "escalated_jaccard_pct": round(100.0 * len(inter) / max(len(union), 1), 2),
        "quantile_stage_pooled_pass": len(pooled_quant_pass),
        "quantile_stage_per_asset_pass": len(per_asset_quant_pass),
        "quantile_stage_jaccard_pct": round(100.0 * len(q_inter) / max(len(q_union), 1), 2),
        "post_crypto_equity_open_ticks": len(equity_open_pooled),
        "post_crypto_equity_open_pooled_esc": sum(equity_open_pooled),
        "post_crypto_equity_open_per_asset_esc": sum(equity_open_per_asset),
    }


def compute_q4_holm6() -> Dict[str, Any]:
    res = json.loads((PROJECT_ROOT / "dev/runs/B07/B07-C01/results.json").read_text(encoding="utf-8"))
    ev = res["evaluation"]["holdout_cascade"]
    tests = []
    for m in ("gpt-6.1-sol", "deepseek-v4.1-flash"):
        for q in ("0.15", "0.25", "0.35"):
            f1_f0 = ev[m]["by_q"][q]["bootstrap_evaluation"]["macro_4_datasets"]["f1_minus_f0"]
            tests.append({
                "model": m,
                "q": q,
                "diff_pp": round(f1_f0["observed_diff"] * 100.0, 4),
                "raw_p": float(f1_f0["p_value"]),
            })
    # Sort ascending by raw_p for Holm-Bonferroni across all 6 tests
    sorted_idx = sorted(range(len(tests)), key=lambda i: tests[i]["raw_p"])
    m_tests = len(tests)
    prev_adj = 0.0
    for rank, idx in enumerate(sorted_idx):
        adj = min(1.0, max(prev_adj, (m_tests - rank) * tests[idx]["raw_p"]))
        tests[idx]["holm_6_p"] = round(adj, 4)
        tests[idx]["rank_in_6"] = rank + 1
        prev_adj = adj
    return {"family_6_tests": tests}


def compute_q6_monte_carlo_routing() -> Dict[str, Any]:
    b02_preds = json.loads((PROJECT_ROOT / "dev/runs/B02/B02-C05/predictions.json").read_text(encoding="utf-8"))
    golds = {g.sample_id: g for g in load_gold_records(role="selection")}
    tau_old = 0.159485

    # Match predictions strictly by sample_id & model_id (role metadata may vary)
    jev_map: Dict[str, Any] = {}
    gpt_map: Dict[str, Any] = {}
    for p in b02_preds:
        sid = p.get("sample_id")
        if sid not in golds:
            continue
        if p.get("model_id") == "jev-1.14.0":
            jev_map[sid] = p
        elif p.get("model_id") == "gpt-6.1-sol":
            gpt_map[sid] = p

    by_ds: Dict[str, List[Any]] = {}
    for sid, g in golds.items():
        by_ds.setdefault(g.dataset_id, []).append(g)

    jev_ok, gpt_ok = {}, {}
    per_ds_summary = {}
    for ds, items in sorted(by_ds.items()):
        j_vec, g_vec = [], []
        for g in items:
            jp = jev_map[g.sample_id]
            gp = gpt_map[g.sample_id]
            j_err = compute_application_error_target(
                jp.get("decision"), g.gold_label, ds, jp.get("probabilities"), g.is_oos, clinc_tau=tau_old
            )
            gpt_target = "REJECT" if g.is_oos else g.gold_label
            g_err = 0 if (gp.get("decision") == gpt_target) else 1
            j_vec.append(1 - j_err)
            g_vec.append(1 - g_err)
        jev_ok[ds] = np.array(j_vec, dtype=float)
        gpt_ok[ds] = np.array(g_vec, dtype=float)
        per_ds_summary[ds] = {
            "n": len(items),
            "jev_app_pct": round(float(np.mean(j_vec)) * 100.0, 2),
            "gpt_app_pct": round(float(np.mean(g_vec)) * 100.0, 2),
        }

    exact_rand = float(np.mean([0.75 * np.mean(jev_ok[ds]) + 0.25 * np.mean(gpt_ok[ds]) for ds in by_ds]))
    rng = np.random.default_rng(2026)
    mc_vals = []
    for _ in range(10000):
        ds_accs = []
        for ds in by_ds:
            n = len(jev_ok[ds])
            k = int(round(0.25 * n))
            idx = rng.choice(n, size=k, replace=False)
            arr = jev_ok[ds].copy()
            arr[idx] = gpt_ok[ds][idx]
            ds_accs.append(float(np.mean(arr)))
        mc_vals.append(float(np.mean(ds_accs)))
    mc_arr = np.array(mc_vals)

    # Also read exact 1-p1 selection F0 accuracy from B05 results.json
    b05 = json.loads((PROJECT_ROOT / "dev/runs/B05/B05-C01/results.json").read_text(encoding="utf-8"))
    by_ds_f0 = b05["selection_evaluation"]["gpt-6.1-sol"]["by_q"]["0.25"]["bootstrap_evaluation"]["by_dataset"]
    f0_acc = float(np.mean([v["accuracies"]["F0"] for v in by_ds_f0.values()]))
    return {
        "per_dataset": per_ds_summary,
        "jev_macro_app_pct": round(float(np.mean([np.mean(v) for v in jev_ok.values()])) * 100.0, 4),
        "gpt_macro_app_pct": round(float(np.mean([np.mean(v) for v in gpt_ok.values()])) * 100.0, 4),
        "analytical_random_q25_pct": round(exact_rand * 100.0, 4),
        "mc_10000_mean_pct": round(float(np.mean(mc_arr)) * 100.0, 4),
        "mc_10000_std_pct": round(float(np.std(mc_arr, ddof=1)) * 100.0, 4),
        "mc_10000_ci95_pct": [round(float(np.percentile(mc_arr, 2.5)) * 100.0, 4), round(float(np.percentile(mc_arr, 97.5)) * 100.0, 4)],
        "p1_route_q25_pct": round(f0_acc * 100.0, 4),
        "gain_over_analytical_pp": round((f0_acc - exact_rand) * 100.0, 4),
        "mc_empirical_p_value": round(float(np.mean(mc_arr >= f0_acc)), 6),
    }


def main() -> None:
    out = {
        "q2_clinc_entropy": compute_q2_clinc_entropy(),
        "q3_per_asset_queue": compute_q3_per_asset_queue(),
        "q4_holm6": compute_q4_holm6(),
        "q6_monte_carlo_routing": compute_q6_monte_carlo_routing(),
    }
    out_path = PROJECT_ROOT / "archives" / "track2_alpaca_real" / "audit_metrics_summary.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
