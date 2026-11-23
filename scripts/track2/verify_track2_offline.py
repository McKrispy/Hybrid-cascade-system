"""Offline zero-API verifier for Track 2 streaming sandbox results and ablations.

Reads the archived real Alpaca market/news stream and API replay ledgers in
`archives/track2_alpaca_real/` and recomputes gatekeeper funnel counts,
Tier-1 continuous trading metrics, and Tier-2 independent four-bucket NAV
performance (unscaled 5 bps one-way transaction fee, calendar-day Sharpe).
"""

import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SANDBOX_ROOT = PROJECT_ROOT / "sandbox"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SANDBOX_ROOT))

from daemon.settlement import PortfolioSettlementEngine, Tier1MetricsTracker
from scripts.track2.run_alpaca_real_replay import compute_dual_backend_summary


def main() -> None:
    archive_dir = PROJECT_ROOT / "archives" / "track2_alpaca_real"
    records_path = archive_dir / "records.jsonl"
    config = json.loads((SANDBOX_ROOT / "config.json").read_text(encoding="utf-8"))
    recs = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    port_cfg = config["portfolio"]
    init_cap = float(port_cfg.get("initial_capital_usd", 100000.0))
    fee_bps = float(port_cfg.get("transaction_cost_bps", 5.0))
    scale = float(port_cfg.get("position_scale", 0.03125))

    gpt_port = PortfolioSettlementEngine(initial_capital=init_cap, transaction_cost_bps=fee_bps, position_scale=scale)
    ds_port = PortfolioSettlementEngine(initial_capital=init_cap, transaction_cost_bps=fee_bps, position_scale=scale)
    gpt_t1 = Tier1MetricsTracker()
    ds_t1 = Tier1MetricsTracker()

    for r in recs:
        gpt_t1.record_settled_tick(
            tick_id=r["tick_id"],
            symbol=r["symbol"],
            gold_label=r["gold_label"],
            jev_decision=r["jev_decision"],
            was_escalated=r["was_escalated"],
            s2_decisions=r["s2_decisions_gpt"],
            signal_conflict=r.get("signal_conflict_flag", False),
        )
        ds_t1.record_settled_tick(
            tick_id=r["tick_id"],
            symbol=r["symbol"],
            gold_label=r["gold_label"],
            jev_decision=r["jev_decision"],
            was_escalated=r["was_escalated"],
            s2_decisions=r.get("s2_decisions_deepseek", {}),
            signal_conflict=r.get("signal_conflict_flag", False),
        )
        gpt_port.settle_tick(
            symbol=r["symbol"],
            timestamp=r["timestamp"],
            actual_return_4h=r["actual_return_4h"],
            jev_decision=r["jev_decision"],
            was_escalated=r["was_escalated"],
            s2_decisions=r["s2_decisions_gpt"],
        )
        ds_port.settle_tick(
            symbol=r["symbol"],
            timestamp=r["timestamp"],
            actual_return_4h=r["actual_return_4h"],
            jev_decision=r["jev_decision"],
            was_escalated=r["was_escalated"],
            s2_decisions=r.get("s2_decisions_deepseek", {}),
        )

    gate_counts = Counter(r["gate_reason"] for r in recs)
    dual_t1 = compute_dual_backend_summary(records_path)
    cot_ablation = json.loads((archive_dir / "cot_ablation_200.json").read_text(encoding="utf-8"))
    clean_ablation = json.loads((archive_dir / "f0_clean_ablation_200.json").read_text(encoding="utf-8"))

    def strip_nav(summary_dict: dict) -> dict:
        return {k: {kk: vv for kk, vv in v.items() if kk != "nav_history"} for k, v in summary_dict.items()}

    out = {
        "total_settled_ticks": len(recs),
        "gatekeeper_counts": dict(gate_counts),
        "dual_backend_tier1": dual_t1,
        "tier2_portfolio_summary": {
            "gpt": strip_nav(gpt_port.get_summary()),
            "deepseek": strip_nav(ds_port.get_summary()),
        },
        "paired_tier1_gpt": gpt_t1.compute_summary_metrics(),
        "paired_tier1_deepseek": ds_t1.compute_summary_metrics(),
        "short_direct_excess_anchoring_pp": cot_ablation.get("short_direct_excess_anchoring_pp"),
        "cot_4field_excess_anchoring_pp": cot_ablation.get("cot_4field_excess_anchoring_pp"),
        "f0_clean_excess_or_metrics": clean_ablation,
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
