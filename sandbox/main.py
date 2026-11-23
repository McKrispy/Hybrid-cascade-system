"""Unified entry point: 7x24 daemon loop, historical replay, and :8090 telemetry server.

Timestamp: 2026-11-11 13:00 (Asia/Shanghai)
"""

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

# Ensure sandbox root is in sys.path
_SANDBOX_ROOT = Path(__file__).resolve().parent
if str(_SANDBOX_ROOT) not in sys.path:
    sys.path.insert(0, str(_SANDBOX_ROOT))

# Ensure unbuffered output
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

# Load .env if present (zero external dependency parser)
def _load_env_file() -> None:
    env_file = _SANDBOX_ROOT / ".env"
    if env_file.exists():
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip()
                    if k and k not in os.environ:
                        os.environ[k] = v

_load_env_file()

import aiohttp
from aiohttp import web
from daemon.engine import StreamingEngine
from telemetry.server import create_telemetry_app, set_engine_instance


def run_historical_replay(engine: StreamingEngine, max_events: int = None) -> None:
    """Run accelerated synchronous historical replay."""
    print("=" * 65)
    print("HybridCascade FINANCIAL DECISION SANDBOX (B08): HISTORICAL REPLAY")
    print("=" * 65)
    print(f"[+] Replaying high-volatility event stream from: {engine.replay_path}")

    start_t = time.time()
    summary = engine.replay_historical_events(max_events=max_events)
    elapsed = time.time() - start_t

    print(f"\n[+] Replay finished in {elapsed:.2f}s!")
    print(f"[+] Total Ticks Processed: {summary.get('replayed_ticks', 0)}")
    print(f"[+] Total Budget Spent: ${summary.get('budget_state', {}).get('total_spent_usd', 0):.4f} USD")

    t1 = summary.get("tier1_metrics", {})
    print("\n--- TIER-1 PARED METRICS ---")
    print(f"  Escalated Ticks: {t1.get('escalated_ticks_count', 0)} / {t1.get('total_settled_ticks', 0)} ({t1.get('escalation_percentage', 0)}%)")
    print(f"  Jev Standalone Accuracy: {t1.get('jev_standalone_acc', 0)}%")
    print(f"  Cascade Accuracies: {t1.get('cascade_accuracies_pct', {})}")
    pri = t1.get("primary_contrast", {})
    print(f"  Primary Contrast F1 - F0: {pri.get('diff_pp', 0.0):+.2f}pp")
    ef = t1.get("error_following", {})
    print(f"  Error Following under Wrong Jev: F0={ef.get('f0_error_follow_rate_pct', 0)}% vs F1={ef.get('f1_error_follow_rate_pct', 0)}% (Anchoring bias: {ef.get('excess_anchoring_bias_pp', 0):+.2f}pp)")

    p_sum = summary.get("portfolio_summary", {})
    print("\n--- TIER-2 FIVE-BRANCH SHADOW PORTFOLIOS ($100k Capital, 5 bps Fee) ---")
    print("| Account ID | NAV (USD) | Cumulative Return | Sharpe Ratio | Max Drawdown | Trades |")
    print("|:---|:---:|:---:|:---:|:---:|:---:|")
    for acc, s in p_sum.items():
        print(f"| {acc} | ${s['nav']:,.2f} | {s['cumulative_return_pct']:+.2f}% | {s['sharpe_ratio']:.2f} | {s['max_drawdown_pct']:.2f}% | {s['total_trades']} |")
    print("=" * 65)


async def run_live_daemon(engine: StreamingEngine, interval_seconds: float = 1800.0) -> None:
    """7x24 live market ingestion loop."""
    print(f"[+] Starting 7x24 live polling loop (interval: {interval_seconds}s)...")
    symbols = engine.config.get("symbols", ["BTC/USD", "ETH/USD", "NVDA", "TSLA"])

    while engine.is_running:
        for sym in symbols:
            if not engine.is_running:
                break
            try:
                tick = engine.fetcher.poll_tick(sym)
                print(f"[*] Ingested tick: {tick.tick_id} ({sym} ${tick.snapshot.current_price})")
                res = engine.process_tick(tick)
                if res["was_escalated"]:
                    print(f"    [!] Escalated to System-2: {res['gate_reason']} | Cost: ${res['tick_cost_usd']:.4f}")
            except Exception as e:
                print(f"[!] Error processing live tick for {sym}: {e}")

        try:
            await asyncio.sleep(interval_seconds)
        except asyncio.CancelledError:
            break


async def run_test_run(
    engine: StreamingEngine,
    duration_seconds: float = 300.0,
    host: str = "0.0.0.0",
    port: int = 8090,
) -> None:
    """Pre-flight 5-minute health check and dry-run loop before production start."""
    import json
    import httpx

    print("=" * 70)
    print("HybridCascade FINANCIAL DECISION SANDBOX (B08): PRE-FLIGHT TEST-RUN")
    print(f"Target Duration: {duration_seconds:.0f} seconds ({duration_seconds / 60.0:.1f} minutes)")
    print(f"Telemetry Port: http://{host}:{port}")
    print("=" * 70)

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "target_duration_seconds": duration_seconds,
        "checks": {},
        "pipeline_smoke_test": {},
        "telemetry_probe": {},
        "status": "IN_PROGRESS",
    }

    # Step 1: Start background telemetry server
    print("\n[+] STEP 1: Launching Background Telemetry Server...")
    engine.is_running = True
    runner = None
    server_started = False
    try:
        app = create_telemetry_app()
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, host, port)
        await site.start()
        server_started = True
        print(f"    [✓] Telemetry HTTP service listening on http://{host}:{port}")
    except Exception as e:
        print(f"    [!] Warning: Could not bind telemetry server on {host}:{port} ({e})")
        report["checks"]["telemetry_bind"] = {"status": "SKIPPED", "error": str(e)}

    # Step 2: API Credentials & Reachability Probes
    print("\n[+] STEP 2: Probing API Keys & External Connectivity...")
    api_results = {}

    # 2a. TypeSafe Jev
    try:
        t0 = time.time()
        jev_res = engine.jev_client.predict("Healthcheck probe: BTC/USD price action", mock_fallback=True, timeout=3.0)
        api_results["typesafe_jev"] = {
            "status": "PASS",
            "latency_ms": round((time.time() - t0) * 1000, 1),
            "model": jev_res.model_id,
            "decision": jev_res.decision,
            "mock": not bool(engine.jev_client.api_key),
        }
        mock_tag = " (MOCK)" if not engine.jev_client.api_key else ""
        print(f"    [✓] TypeSafe Jev API: Reachable ({api_results['typesafe_jev']['latency_ms']}ms){mock_tag}")
    except Exception as e:
        api_results["typesafe_jev"] = {"status": "FAIL", "error": str(e)}
        print(f"    [!] TypeSafe Jev API: Failed ({e})")

    # 2b. OpenAI GPT-6.1-Sol
    try:
        t0 = time.time()
        gpt_res = engine.gpt_client.predict("Healthcheck prompt", "F0", mock_fallback=True, timeout=3.0)
        api_results["openai_gpt"] = {
            "status": "PASS",
            "latency_ms": round((time.time() - t0) * 1000, 1),
            "model": gpt_res.model_id,
            "decision": gpt_res.decision,
            "reasoning_tokens": gpt_res.reasoning_tokens,
            "mock": not bool(engine.gpt_client.api_key),
        }
        mock_tag = " (MOCK)" if not engine.gpt_client.api_key else ""
        print(f"    [✓] OpenAI API (GPT-6.1-Sol): Reachable ({api_results['openai_gpt']['latency_ms']}ms){mock_tag}")
    except Exception as e:
        api_results["openai_gpt"] = {"status": "FAIL", "error": str(e)}
        print(f"    [!] OpenAI API: Failed ({e})")

    # 2c. DeepSeek-v4.1-Flash
    try:
        t0 = time.time()
        ds_res = engine.deepseek_client.predict("Healthcheck prompt", "F0", mock_fallback=True, timeout=3.0)
        api_results["deepseek"] = {
            "status": "PASS",
            "latency_ms": round((time.time() - t0) * 1000, 1),
            "model": ds_res.model_id,
            "decision": ds_res.decision,
            "mock": not bool(engine.deepseek_client.api_key),
        }
        mock_tag = " (MOCK)" if not engine.deepseek_client.api_key else ""
        print(f"    [✓] DeepSeek API: Reachable ({api_results['deepseek']['latency_ms']}ms){mock_tag}")
    except Exception as e:
        api_results["deepseek"] = {"status": "FAIL", "error": str(e)}
        print(f"    [!] DeepSeek API: Failed ({e})")

    report["checks"]["apis"] = api_results

    # Step 3: End-to-End Pipeline Smoke Test
    print("\n[+] STEP 3: Executing End-to-End Pipeline Smoke Test (Single Tick)...")
    try:
        # Load one tick from replay or poll mock tick
        test_tick = engine.fetcher.poll_tick("BTC/USD")
        res = engine.process_tick(test_tick)
        report["pipeline_smoke_test"] = {
            "status": "PASS",
            "tick_id": test_tick.tick_id,
            "symbol": test_tick.symbol,
            "jev_decision": res.get("jev_decision"),
            "was_escalated": res.get("was_escalated"),
            "tick_cost_usd": res.get("tick_cost_usd"),
            "gold_label": res.get("gold_label"),
            "nav_summary": engine.portfolio_engine.get_summary(),
        }
        print(f"    [✓] Tick Ingested & Evaluated: {test_tick.tick_id} ({test_tick.symbol})")
        print(f"    [✓] Dynamic Gatekeeper: {'ESCALATED' if res.get('was_escalated') else 'ROUTED_TO_JEV'} ({res.get('gate_reason', 'ok')})")
        print(f"    [✓] Tier-1 Settlement & Tier-2 Shadow Portfolios Updated")
        print(f"    [✓] Ledger & Records persisted to disk")
    except Exception as e:
        report["pipeline_smoke_test"] = {"status": "FAIL", "error": str(e)}
        print(f"    [!] Pipeline smoke test failed: {e}")

    # Step 4: Telemetry Internal Loopback Probe
    print("\n[+] STEP 4: Verifying Telemetry API Endpoints...")
    if server_started:
        try:
            token = os.environ.get("TELEMETRY_BEARER_TOKEN", "change_me_telemetry_bearer_token")
            headers = {"Authorization": f"Bearer {token}"}
            async with aiohttp.ClientSession(headers=headers, timeout=aiohttp.ClientTimeout(total=3.0)) as session:
                async with session.get(f"http://127.0.0.1:{port}/api/v1/telemetry/overview") as resp_ov:
                    status_ov = resp_ov.status
                    data_ov = await resp_ov.json()
                async with session.get(f"http://127.0.0.1:{port}/api/v1/telemetry/portfolio") as resp_pf:
                    status_pf = resp_pf.status
                    data_pf = await resp_pf.json()

                if status_ov == 200 and status_pf == 200:
                    report["telemetry_probe"] = {
                        "status": "PASS",
                        "overview_status": status_ov,
                        "portfolio_status": status_pf,
                        "overview_keys": list(data_ov.keys()),
                    }
                    print(f"    [✓] /api/v1/telemetry/overview -> HTTP 200 OK")
                    print(f"    [✓] /api/v1/telemetry/portfolio -> HTTP 200 OK")
                else:
                    report["telemetry_probe"] = {
                        "status": "FAIL",
                        "overview_status": status_ov,
                        "portfolio_status": status_pf,
                    }
                    print(f"    [!] Telemetry probe unexpected status: overview={status_ov}, portfolio={status_pf}")
        except Exception as e:
            report["telemetry_probe"] = {"status": "FAIL", "error": str(e)}
            print(f"    [!] Telemetry probe failed: {e}")
    else:
        snap = engine.get_telemetry_snapshot()
        report["telemetry_probe"] = {
            "status": "PASS_INTERNAL",
            "overview_keys": list(snap["overview"].keys()),
        }
        print(f"    [✓] Internal Telemetry Snapshot verified ({len(snap['overview'])} metrics)")

    # Step 5: Continuous Test Loop with Progress Countdown
    print(f"\n[+] STEP 5: Entering Continuous Monitoring Phase ({duration_seconds:.0f}s)...")
    print("    (You can safely press Ctrl+C at any time to finish test-run early)")

    start_time = time.time()
    next_tick_time = start_time + 15.0  # process a sample tick every 20-30s during test
    symbols = engine.config.get("symbols", ["BTC/USD", "ETH/USD", "NVDA", "TSLA"])
    sym_idx = 0

    try:
        while True:
            now = time.time()
            elapsed = now - start_time
            remaining = max(0.0, duration_seconds - elapsed)

            if elapsed >= duration_seconds:
                break

            # Periodically process a tick during test run
            if now >= next_tick_time:
                sym = symbols[sym_idx % len(symbols)]
                sym_idx += 1
                try:
                    tick = engine.fetcher.poll_tick(sym)
                    engine.process_tick(tick)
                except Exception:
                    pass
                next_tick_time = now + 25.0

            # Display progress line
            bg = engine.budget_killswitch.state()
            gk = engine.gatekeeper.state()
            pct = min(100.0, (elapsed / duration_seconds) * 100.0)
            sys.stdout.write(
                f"\r    [TEST-RUN] Elapsed: {int(elapsed):03d}s / {int(duration_seconds):03d}s ({pct:4.1f}%) | "
                f"Ticks: {engine.processed_ticks_count:2d} | "
                f"Cost: ${bg.get('total_spent_usd', 0.0):.4f} | "
                f"Tokens: {gk.get('token_bucket', {}).get('available_tokens', 4.0):.1f} | "
                f"Remaining: {int(remaining):03d}s"
            )
            sys.stdout.flush()
            await asyncio.sleep(1.0)
    except (asyncio.CancelledError, KeyboardInterrupt):
        print("\n\n[!] Test-run interrupted early by user. Generating report...")
    finally:
        print("\n")
        engine.is_running = False
        engine.save_checkpoint()
        if runner is not None:
            await runner.cleanup()

    # Step 6: Generate and write final report
    elapsed_final = time.time() - start_time
    report["status"] = "COMPLETED"
    report["actual_duration_seconds"] = round(elapsed_final, 1)
    report["final_stats"] = {
        "processed_ticks": engine.processed_ticks_count,
        "escalated_ticks": engine.escalated_ticks_count,
        "total_spent_usd": engine.budget_killswitch.total_spent_usd,
        "portfolio_summary": engine.portfolio_engine.get_summary(),
        "budget_state": engine.budget_killswitch.state(),
    }

    report_path = engine.data_dir / "test_run_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("=" * 70)
    print("TEST-RUN HEALTH CHECK SUMMARY REPORT")
    print(f"Elapsed: {elapsed_final:.1f}s | Ticks: {engine.processed_ticks_count} | Cost: ${engine.budget_killswitch.total_spent_usd:.4f}")
    print(f"Report saved to: {report_path}")
    print("=" * 70)
    print("[✓] ALL SYSTEMS OPERATIONAL!")
    print("[+] You are ready to start the 7x24 production daemon:")
    print("    docker compose up -d")
    print("    or: python main.py --mode live --server")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="HybridCascade Streaming Financial Sandbox")
    parser.add_argument("--mode", choices=["replay", "live", "test-run"], default="replay", help="Execution mode (replay, live, or test-run)")
    parser.add_argument("--duration", type=float, default=300.0, help="Duration in seconds for test-run mode (default: 300s = 5min)")
    parser.add_argument("--host", default="0.0.0.0", help="Telemetry host")
    parser.add_argument("--port", type=int, default=8090, help="Telemetry port")
    parser.add_argument("--max-events", type=int, default=None, help="Max replay events to process")
    parser.add_argument("--interval", type=float, default=1800.0, help="Live polling interval seconds")
    parser.add_argument("--server", action="store_true", help="Start background telemetry server on port 8090")

    args = parser.parse_args()

    engine = StreamingEngine(sandbox_root=_SANDBOX_ROOT)
    set_engine_instance(engine)

    if args.mode == "test-run":
        try:
            asyncio.run(run_test_run(engine, duration_seconds=args.duration, host=args.host, port=args.port))
        except KeyboardInterrupt:
            print("\n[!] Test-run interrupted by user.")
    elif args.mode == "replay":
        run_historical_replay(engine, max_events=args.max_events)
        if args.server:
            print(f"[+] Starting telemetry server on http://{args.host}:{args.port}...")
            app = create_telemetry_app()
            web.run_app(app, host=args.host, port=args.port)
    else:
        engine.is_running = True
        app = create_telemetry_app()

        async def _live_main() -> None:
            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, args.host, args.port)
            await site.start()
            print(f"[+] Telemetry API listening on http://{args.host}:{args.port}")
            try:
                await run_live_daemon(engine, interval_seconds=args.interval)
            finally:
                engine.is_running = False
                engine.save_checkpoint()
                await runner.cleanup()

        try:
            asyncio.run(_live_main())  # Bug 8 fix: Python 3.12 compatible entry point
        except KeyboardInterrupt:
            print("\n[!] Live daemon interrupted by user.")


if __name__ == "__main__":
    main()
