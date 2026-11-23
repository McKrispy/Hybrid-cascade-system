"""Run local 100% real Alpaca 30-min replay stream through B08 sandbox up to 200 escalated ticks."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SANDBOX_ROOT = PROJECT_ROOT / "sandbox"
sys.path.insert(0, str(SANDBOX_ROOT))

from daemon.engine import StreamingEngine  # noqa: E402
from daemon.settlement import Tier1MetricsTracker  # noqa: E402


def load_sandbox_env() -> None:
    env_file = SANDBOX_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def compute_dual_backend_summary(records_path: Path) -> dict:
    rows = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    gpt_t1 = Tier1MetricsTracker()
    ds_t1 = Tier1MetricsTracker()
    top2_hits = 0
    esc_rows = [r for r in rows if r.get("was_escalated")]
    ds_fallback_events = 0

    for r in rows:
        gpt_t1.record_settled_tick(
            tick_id=r["tick_id"],
            symbol=r["symbol"],
            gold_label=r["gold_label"],
            jev_decision=r["jev_decision"],
            was_escalated=r["was_escalated"],
            s2_decisions=r.get("s2_decisions_gpt", {}),
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
        if r.get("was_escalated"):
            top2_labels = {c["label"] for c in r.get("jev_top2", [])}
            if r["gold_label"] in top2_labels:
                top2_hits += 1
            if r.get("deepseek_fallback_arms"):
                ds_fallback_events += 1

    def _rescue_harm(backend_key: str) -> dict:
        wrong_jev = [r for r in esc_rows if r["jev_decision"] != r["gold_label"]]
        ok_jev = [r for r in esc_rows if r["jev_decision"] == r["gold_label"]]
        res = {}
        for arm in ("F0", "F1", "C0", "C1"):
            n_err = len(wrong_jev)
            n_ok = len(ok_jev)
            ef = sum(1 for r in wrong_jev if r.get(backend_key, {}).get(arm) == r["jev_decision"])
            rescue = sum(1 for r in wrong_jev if r.get(backend_key, {}).get(arm) == r["gold_label"])
            harm = sum(1 for r in ok_jev if r.get(backend_key, {}).get(arm) != r["gold_label"])
            acc = sum(1 for r in esc_rows if r.get(backend_key, {}).get(arm) == r["gold_label"])
            res[arm] = {
                "escalated_acc_pct": round(100.0 * acc / max(len(esc_rows), 1), 2),
                "escalated_correct": acc,
                "error_follow_pct": round(100.0 * ef / max(n_err, 1), 2),
                "error_follow_count": ef,
                "rescue_pct": round(100.0 * rescue / max(n_err, 1), 2),
                "rescue_count": rescue,
                "harm_pct": round(100.0 * harm / max(n_ok, 1), 2),
                "harm_count": harm,
            }
        res["n_escalated"] = len(esc_rows)
        res["n_wrong_jev"] = len(wrong_jev)
        res["n_ok_jev"] = len(ok_jev)
        return res

    return {
        "total_ticks": len(rows),
        "escalated_ticks": len(esc_rows),
        "gold_in_top2_count": top2_hits,
        "gold_in_top2_pct": round(100.0 * top2_hits / max(len(esc_rows), 1), 2),
        "deepseek_fallback_ticks": ds_fallback_events,
        "gpt_tier1": gpt_t1.compute_summary_metrics(),
        "deepseek_tier1": ds_t1.compute_summary_metrics(),
        "gpt_decomposition": _rescue_harm("s2_decisions_gpt"),
        "deepseek_decomposition": _rescue_harm("s2_decisions_deepseek"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-escalated", type=int, default=200)
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    load_sandbox_env()
    out_dir = PROJECT_ROOT / "archives" / "track2_alpaca_real"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.fresh:
        for fname in ("records.jsonl", "ledger.jsonl", "checkpoints.json", "telemetry-alpaca-real.json"):
            p = out_dir / fname
            if p.exists():
                p.unlink()

    config = json.loads((SANDBOX_ROOT / "config.json").read_text(encoding="utf-8"))
    # Allow enough daily room so historical bursts in accelerated replay aren't throttled by daily cap;
    # keep global hard cap at $16.00 (200 ticks expected ~$5-$6).
    config["budget_killswitch"]["daily_budget_cap_usd"] = 10.0

    replay_path = PROJECT_ROOT / "archives" / "track2_alpaca_real" / "alpaca_real_30m_stream.jsonl"
    engine = StreamingEngine(
        sandbox_root=SANDBOX_ROOT,
        config=config,
        data_dir_override=out_dir,
        replay_path_override=replay_path,
        ignore_checkpoint=True,
    )

    print(
        f"Starting local 100% real Alpaca 30m replay -> target {args.target_escalated} escalated ticks",
        flush=True,
    )
    print(
        f"Keys: Jev={bool(engine.jev_client.api_key)}, GPT={bool(engine.gpt_client.api_key)}, DS={bool(engine.deepseek_client.api_key)}",
        flush=True,
    )

    t0 = time.time()

    def on_progress(idx: int, esc_count: int, entry: dict) -> None:
        if entry.get("was_escalated") or idx % 25 == 0:
            spent = engine.budget_killswitch.total_spent_usd
            ds_fb = entry.get("deepseek_fallback_arms", [])
            fb_tag = f" [DS_FB:{ds_fb}]" if ds_fb else ""
            print(
                f"[{time.time()-t0:6.1f}s] tick={idx:4d} {entry['timestamp']} sym={entry['symbol']:7s} "
                f"esc={esc_count:3d}/{args.target_escalated} reason={entry['gate_reason']:24s} "
                f"gold={entry['gold_label']:15s} spent=${spent:.4f}{fb_tag}",
                flush=True,
            )

    # Re-hydrate state cleanly from existing valid records if resuming (non-fresh)
    records_file = out_dir / "records.jsonl"
    if not args.fresh and records_file.exists():
        # Filter out any tick that had DeepSeek SSL fallback so it gets cleanly retried, or replay state up to valid prefix
        existing = [json.loads(l) for l in records_file.read_text(encoding="utf-8").splitlines() if l.strip()]
        # Rebuild exact prefix without mock fallback if possible, keeping all valid records
        from daemon.fetcher import StreamTick
        stream_by_id = {
            json.loads(l)["tick_id"]: StreamTick(**json.loads(l))
            for l in replay_path.read_text(encoding="utf-8").splitlines()
            if l.strip()
        }
        import time as _time
        cleaned = [r for r in existing if not r.get("deepseek_fallback_arms")]
        records_file.write_text("".join(json.dumps(r) + "\n" for r in cleaned), encoding="utf-8")
        # Replay state trackers from cleaned records without making API calls
        for r in cleaned:
            t_obj = stream_by_id[r["tick_id"]]
            try:
                t_ts = _time.mktime(_time.strptime(t_obj.timestamp, "%Y-%m-%d %H:%M:%S"))
            except Exception:
                t_ts = _time.time()
            engine.processed_ticks_count += 1
            engine.gatekeeper.evaluate_tick(
                symbol=t_obj.symbol,
                uncertainty=r["jev_uncertainty"],
                current_news_ids=[n.news_id for n in t_obj.news_stream],
                now_ts=t_ts,
            )
            if r["was_escalated"]:
                engine.escalated_ticks_count += 1
                engine.recent_reasoning_stream.append({
                    "tick_id": t_obj.tick_id,
                    "symbol": t_obj.symbol,
                    "timestamp": t_obj.timestamp,
                    "signal_conflict": t_obj.signal_conflict_flag,
                    "jev_prediction": {
                        "decision": r["jev_decision"],
                        "uncertainty": r["jev_uncertainty"],
                        "probabilities": r["jev_probabilities"],
                        "top2_candidates": r["jev_top2"],
                    },
                    "gpt_decisions": r["s2_decisions_gpt"],
                    "deepseek_decisions": r["s2_decisions_deepseek"],
                    "gpt_f1_reasoning": r.get("s2_f1_reasoning_gpt") or {},
                    "deepseek_f1_reasoning": r.get("s2_f1_reasoning_deepseek") or {},
                })
                if len(engine.recent_reasoning_stream) > 50:
                    engine.recent_reasoning_stream.pop(0)
            engine.budget_killswitch.record_spend(r["tick_cost_usd"], r["tick_id"], now_ts=t_ts)
            engine.tier1_tracker.record_settled_tick(
                tick_id=r["tick_id"],
                symbol=r["symbol"],
                gold_label=r["gold_label"],
                jev_decision=r["jev_decision"],
                was_escalated=r["was_escalated"],
                s2_decisions=r["s2_decisions_gpt"],
                signal_conflict=r["signal_conflict_flag"],
            )
            engine.portfolio_engine.settle_tick(
                symbol=r["symbol"],
                timestamp=r["timestamp"],
                actual_return_4h=r["actual_return_4h"],
                jev_decision=r["jev_decision"],
                was_escalated=r["was_escalated"],
                s2_decisions=r["s2_decisions_gpt"],
            )
        # Rebuild ledger cleanly to match kept records
        ledger_file = out_dir / "ledger.jsonl"
        ledger_file.write_text(
            "".join(
                json.dumps({
                    "timestamp": r["timestamp"],
                    "tick_id": r["tick_id"],
                    "cost_usd": r["tick_cost_usd"],
                    "total_spent_usd": 0.0,
                    "was_escalated": r["was_escalated"],
                }) + "\n"
                for r in cleaned
            ),
            encoding="utf-8",
        )
        print(
            f"Resumed cleanly from {len(cleaned)} valid records (dropped {len(existing)-len(cleaned)} fallback rows): "
            f"escalated={engine.escalated_ticks_count}, spent=${engine.budget_killswitch.total_spent_usd:.4f}",
            flush=True,
        )

    summary = engine.replay_historical_events(
        max_escalated_ticks=args.target_escalated,
        progress_callback=on_progress,
        resume=not args.fresh,
    )
    dual = compute_dual_backend_summary(out_dir / "records.jsonl")
    snap = engine.get_telemetry_snapshot()
    snap["replay_summary"] = summary
    snap["dual_backend_summary"] = dual
    snap["full_reasoning_stream"] = engine.recent_reasoning_stream
    (out_dir / "telemetry-alpaca-real.json").write_text(
        json.dumps(snap, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\n=== COMPLETED REAL ALPACA REPLAY ===")
    print("Gatekeeper:", json.dumps(summary.get("gatekeeper_state"), ensure_ascii=False))
    print("Budget:", json.dumps(summary.get("budget_state"), ensure_ascii=False))
    print("Portfolio:", json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'nav_history'} for k, v in summary.get("portfolio_summary", {}).items()}, indent=2, ensure_ascii=False))
    print("Dual Backend Tier-1:", json.dumps(dual, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
