"""Rebuild Regime-B canonical real Alpaca archive artifacts and sync directories."""

import hashlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SANDBOX_ROOT = PROJECT_ROOT / "sandbox"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SANDBOX_ROOT) not in sys.path:
    sys.path.insert(0, str(SANDBOX_ROOT))

from sandbox.daemon.settlement import PortfolioSettlementEngine, Tier1MetricsTracker
from scripts.track2.run_alpaca_real_replay import compute_dual_backend_summary


def main() -> None:
    real_dir = PROJECT_ROOT / "archives" / "track2_alpaca_real"
    final_dir = PROJECT_ROOT / "archives" / "track2_alpaca_real"
    data_dir = SANDBOX_ROOT / "data"
    for d in (real_dir, final_dir, data_dir):
        d.mkdir(parents=True, exist_ok=True)

    config = json.loads((SANDBOX_ROOT / "config.json").read_text(encoding="utf-8"))
    recs = [
        json.loads(line)
        for line in (real_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    old_ckpt = json.loads((real_dir / "checkpoints.json").read_text(encoding="utf-8"))
    old_telem = json.loads((real_dir / "telemetry-alpaca-real.json").read_text(encoding="utf-8"))

    port_cfg = config["portfolio"]
    scale = float(port_cfg.get("position_scale", 1.0 / 32.0))
    init_cap = float(port_cfg.get("initial_capital_usd", 100000.0))
    fee_bps = float(port_cfg.get("transaction_cost_bps", 5.0))

    gpt_port = PortfolioSettlementEngine(
        initial_capital=init_cap,
        transaction_cost_bps=fee_bps,
        position_scale=scale,
    )
    ds_port = PortfolioSettlementEngine(
        initial_capital=init_cap,
        transaction_cost_bps=fee_bps,
        position_scale=scale,
    )
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

    gpt_port_summary = gpt_port.get_summary()
    ds_port_summary = ds_port.get_summary()
    gpt_paired = gpt_t1.compute_summary_metrics()
    ds_paired = ds_t1.compute_summary_metrics()
    dual = compute_dual_backend_summary(real_dir / "records.jsonl")

    gate_counts = Counter(r["gate_reason"] for r in recs)
    n_total = len(recs)
    n_esc = gate_counts["escalate_high_uncertainty"]

    gk_state = dict(old_ckpt["gatekeeper"])
    gk_state["total_evaluated"] = n_total
    gk_state["total_escalated"] = n_esc
    gk_state["escalation_rate"] = round(n_esc / max(n_total, 1), 4)
    gk_state["total_suppressed_cooldown"] = gate_counts["cooldown_suppressed"]
    gk_state["total_suppressed_bucket"] = gate_counts["token_bucket_exhausted"]

    ckpt_new = {
        "timestamp": old_ckpt["timestamp"],
        "processed_ticks_count": n_total,
        "escalated_ticks_count": n_esc,
        "total_spent_usd": old_ckpt["total_spent_usd"],
        "gatekeeper": gk_state,
        "budget": old_ckpt["budget"],
        "portfolio": gpt_port_summary,
        "portfolio_deepseek": ds_port_summary,
    }

    overview = dict(old_telem.get("overview", {}))
    overview["total_ticks_processed"] = n_total
    overview["total_ticks_escalated"] = n_esc
    overview["escalation_rate_pct"] = round(n_esc / max(n_total, 1) * 100.0, 2)

    reasoning_stream = old_telem.get("reasoning_stream", [])
    telem_new = {
        "overview": overview,
        "portfolio": gpt_port_summary,
        "portfolio_deepseek": ds_port_summary,
        "paired_metrics": gpt_paired,
        "paired_metrics_deepseek": ds_paired,
        "dual_backend_summary": dual,
        "reasoning_stream": reasoning_stream,
    }

    # Update alpaca_real_data
    (real_dir / "checkpoints.json").write_text(
        json.dumps(ckpt_new, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (real_dir / "telemetry-alpaca-real.json").write_text(
        json.dumps(telem_new, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Sync canonical archive into sandbox_final_data
    for fname in ("records.jsonl", "ledger.jsonl", "checkpoints.json"):
        shutil.copy2(real_dir / fname, final_dir / fname)
        shutil.copy2(real_dir / fname, data_dir / fname)
    shutil.copy2(SANDBOX_ROOT / "config.json", final_dir / "config.json")
    (final_dir / "telemetry-pre-stop.json").write_text(
        json.dumps(telem_new, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    state_txt = (
        "archive_type=100pct_real_alpaca_30m_stream_replay\n"
        f"settled_ticks={n_total}\n"
        f"escalated_ticks={n_esc}\n"
        f"below_quantile_threshold={gate_counts['below_quantile_threshold']}\n"
        f"cooldown_suppressed={gate_counts['cooldown_suppressed']}\n"
        f"token_bucket_exhausted={gate_counts['token_bucket_exhausted']}\n"
        f"total_spent_usd={old_ckpt['total_spent_usd']}\n"
        f"position_scale={scale}\n"
        "deepseek_fallback_ticks=0\n"
        f"first_timestamp={recs[0]['timestamp']}\n"
        f"last_timestamp={recs[-1]['timestamp']}\n"
    )
    (final_dir / "container-state.txt").write_text(state_txt, encoding="utf-8")

    archive_files = [
        "SHA256SUMS.txt",  # excluded below
        "records.jsonl",
        "ledger.jsonl",
        "checkpoints.json",
        "config.json",
        "telemetry-pre-stop.json",
        "container-state.txt",
    ]
    sha_lines = []
    for name in sorted(archive_files[1:]):
        digest = hashlib.sha256((final_dir / name).read_bytes()).hexdigest()
        sha_lines.append(f"{digest}  {name}")
    (final_dir / "SHA256SUMS.txt").write_text("\n".join(sha_lines) + "\n", encoding="utf-8")

    print("Rebuilt Regime-B real Alpaca canonical archive successfully.")
    print("Gate counts:", dict(gate_counts))
    print("GPT Portfolio:", json.dumps({k: {m: v[m] for m in ('nav', 'cumulative_return_pct', 'sharpe_ratio', 'max_drawdown_pct', 'total_trades', 'turnover')} for k, v in gpt_port_summary.items()}, indent=2))
    print("DeepSeek Portfolio:", json.dumps({k: {m: v[m] for m in ('nav', 'cumulative_return_pct', 'sharpe_ratio', 'max_drawdown_pct', 'total_trades', 'turnover')} for k, v in ds_port_summary.items()}, indent=2))


if __name__ == "__main__":
    main()
