"""Unit and compliance test suite for Batch B08: Streaming Financial Sandbox.

Timestamp: 2026-11-11 13:00 (Asia/Shanghai)
"""

import ast
import json
import os
import time
from pathlib import Path

import pytest
from aiohttp.test_utils import AioHTTPTestCase, unittest_run_loop

from sandbox.daemon.clients import DeepSeekClient, JevClient, OpenAIClient
from sandbox.daemon.engine import StreamingEngine
from sandbox.daemon.fetcher import MarketDataFetcher, compute_macd, compute_rsi, detect_signal_conflict
from sandbox.daemon.gatekeeper import BudgetKillSwitch, DynamicGatekeeper, TokenBucket
from sandbox.daemon.handoff import FinancialHandoffBuilder
from sandbox.daemon.settlement import PortfolioSettlementEngine, Tier1MetricsTracker
from sandbox.telemetry.server import create_telemetry_app, set_engine_instance


def test_sandbox_zero_external_imports():
    """Verify that dev/sandbox is 100% physically self-contained with 0 imports from src/."""
    sandbox_root = Path(__file__).resolve().parent.parent / "sandbox"
    assert sandbox_root.exists(), "Sandbox directory does not exist"

    violations = []
    py_files = list(sandbox_root.rglob("*.py"))
    assert len(py_files) >= 5, f"Expected sandbox python files, found {len(py_files)}"

    for py_file in py_files:
        with open(py_file, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=str(py_file))

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.name
                    if name.startswith("src") or name.startswith("src") or name == "dev":
                        violations.append((str(py_file.relative_to(sandbox_root)), name))
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if mod.startswith("src") or mod.startswith("src") or mod == "dev":
                    violations.append((str(py_file.relative_to(sandbox_root)), mod))

    assert len(violations) == 0, f"Found forbidden external imports in sandbox: {violations}"


def test_technical_indicators_calculation():
    """Verify mathematical correctness of RSI, MACD, and signal conflict detector."""
    import numpy as np

    # Monotonically increasing prices -> RSI should be high (>70)
    up_prices = np.linspace(100, 150, 30)
    rsi_up = compute_rsi(up_prices, 14)
    assert rsi_up > 70.0, f"Expected overbought RSI, got {rsi_up}"

    # Monotonically decreasing prices -> RSI should be low (<30)
    down_prices = np.linspace(150, 100, 30)
    rsi_down = compute_rsi(down_prices, 14)
    assert rsi_down < 30.0, f"Expected oversold RSI, got {rsi_down}"

    # MACD test
    val, sig, hist = compute_macd(up_prices)
    assert isinstance(val, float) and isinstance(hist, float)

    # Signal conflict detection
    fetcher = MarketDataFetcher()
    snap = fetcher.fetch_market_snapshot("BTC/USD")
    snap.indicators.rsi_14 = 78.0
    snap.indicators.macd_hist = -0.5

    news = fetcher.fetch_latest_news("BTC/USD")
    for n in news:
        n.sentiment_score = 0.8  # Strong bullish news vs bearish technicals

    has_conflict, reason = detect_signal_conflict(snap.indicators, news)
    assert has_conflict is True, "Expected signal conflict to be detected"
    assert "divergent" in reason.lower() or "vs" in reason.lower()


def test_dynamic_quantile_and_token_bucket():
    """Verify dynamic rolling quantile gating, asset cool-down, and token bucket limiting."""
    bucket = TokenBucket(capacity=4.0, refill_interval_seconds=1800.0, refill_tokens=1.0)
    assert bucket.can_consume(1.0) is True
    assert bucket.consume(1.0) is True
    assert bucket.consume(3.0) is True
    assert bucket.consume(1.0) is False  # Empty!

    gk = DynamicGatekeeper(rolling_window_size=100, quantile_threshold=0.75, cooldown_seconds=3600.0)

    # Prime with normal uncertainties around 0.3
    t_start = 1730000000.0
    for i in range(50):
        gk.evaluate_tick("BTC/USD", uncertainty=0.30, now_ts=t_start + i * 10)

    # Low uncertainty -> should not escalate
    esc, reason, _ = gk.evaluate_tick("BTC/USD", uncertainty=0.20, now_ts=t_start + 1000)
    assert esc is False
    assert reason == "below_quantile_threshold"

    # Extreme high uncertainty -> should escalate
    esc, reason, _ = gk.evaluate_tick("BTC/USD", uncertainty=0.85, current_news_ids=["n1"], now_ts=t_start + 2000)
    assert esc is True
    assert reason == "escalate_high_uncertainty"

    # Immediate repeat within 60 min with same news -> cooldown suppressed
    esc, reason, _ = gk.evaluate_tick("BTC/USD", uncertainty=0.90, current_news_ids=["n1"], now_ts=t_start + 2500)
    assert esc is False
    assert reason == "cooldown_suppressed"

    # Same repeat within 60 min with NEW news -> cooldown bypassed!
    esc, reason, _ = gk.evaluate_tick("BTC/USD", uncertainty=0.92, current_news_ids=["n1", "n2_breaking"], now_ts=t_start + 2600)
    assert esc is True


def test_three_tier_budget_killswitch():
    """Verify Level-1 alert, Level-2 rolling 24h cap, and Level-3 global hard cap."""
    ks = BudgetKillSwitch(
        single_tick_alert_usd=0.06,
        daily_budget_cap_usd=2.50,
        total_budget_cap_usd=16.00,
    )
    t0 = 1730000000.0

    # Under budget
    can_disp, _ = ks.can_dispatch_system2(now_ts=t0)
    assert can_disp is True

    # Record small spend
    ks.record_spend(0.02, "tick_1", now_ts=t0)
    assert len(ks.alert_log) == 0

    # Level-1 alert: spend 0.08 in single tick
    ks.record_spend(0.08, "tick_2", now_ts=t0 + 10)
    assert len(ks.alert_log) == 1
    assert ks.alert_log[0]["level"] == "LEVEL_1_ALERT"

    # Level-2 rolling cap: push rolling 24h spend over $2.50
    ks.record_spend(2.45, "tick_3", now_ts=t0 + 20)
    assert ks.mode == "JEV_ONLY_DEGRADED"
    can_disp, reason = ks.can_dispatch_system2(now_ts=t0 + 30)
    assert can_disp is False
    assert reason == "daily_budget_cap_exceeded"

    # Level-3 hard cap: push total spend over $16.00
    ks.record_spend(14.0, "tick_4", now_ts=t0 + 100000)  # Next day
    assert ks.mode == "HARD_CAPPED"
    can_disp, reason = ks.can_dispatch_system2(now_ts=t0 + 100010)
    assert can_disp is False
    assert reason == "hard_budget_cap_exceeded"


def test_handoff_prompts_and_schema():
    """Verify prompt generation across F0, F1, C0, C1 conditions."""
    fetcher = MarketDataFetcher()
    tick = fetcher.poll_tick("NVDA")
    jev_client = JevClient()
    jev_pred = jev_client._mock_predict("NVDA tech state")

    # F0: Full candidates, No prior block
    prompt_f0 = FinancialHandoffBuilder.build_prompt(tick, "F0", jev_pred)
    assert "Full Candidate Set" in prompt_f0
    assert "System-1 Fast Prior Information" not in prompt_f0
    assert "N/A (No numerical prior provided)" in prompt_f0

    # F1: Full candidates, WITH prior block
    prompt_f1 = FinancialHandoffBuilder.build_prompt(tick, "F1", jev_pred)
    assert "Full Candidate Set" in prompt_f1
    assert "System-1 Fast Prior Information" in prompt_f1
    assert "Rank 1:" in prompt_f1

    # C0: Top-2 pruned, No prior block
    prompt_c0 = FinancialHandoffBuilder.build_prompt(tick, "C0", jev_pred)
    assert "Top-2 Pruned Set" in prompt_c0
    assert "System-1 Fast Prior Information" not in prompt_c0

    # C1: Top-2 pruned, WITH prior block
    prompt_c1 = FinancialHandoffBuilder.build_prompt(tick, "C1", jev_pred)
    assert "Top-2 Pruned Set" in prompt_c1
    assert "System-1 Fast Prior Information" in prompt_c1


def test_two_tier_settlement_and_portfolio():
    """Verify Tier-1 metrics tracker and Tier-2 5-branch shadow accounts."""
    t1 = Tier1MetricsTracker()
    port = PortfolioSettlementEngine(initial_capital=100000.0, transaction_cost_bps=5.0)

    # Tick 1: Market went up +2% (Return = +0.02)
    t1.record_settled_tick(
        tick_id="t1",
        symbol="BTC/USD",
        gold_label="LONG_ACCUMULATE",
        jev_decision="LONG_ACCUMULATE",
        was_escalated=True,
        s2_decisions={"F0": "LONG_ACCUMULATE", "F1": "NEUTRAL_HOLD", "C0": "LONG_ACCUMULATE", "C1": "NEUTRAL_HOLD"},
    )
    port.settle_tick(
        symbol="BTC/USD",
        timestamp="2026-11-11 12:00:00",
        actual_return_4h=0.02,
        jev_decision="LONG_ACCUMULATE",
        was_escalated=True,
        s2_decisions={"F0": "LONG_ACCUMULATE", "F1": "NEUTRAL_HOLD", "C0": "LONG_ACCUMULATE", "C1": "NEUTRAL_HOLD"},
    )

    summary_t1 = t1.compute_summary_metrics()
    assert summary_t1["total_settled_ticks"] == 1
    assert summary_t1["cascade_accuracies_pct"]["F0"] == 100.0
    assert summary_t1["cascade_accuracies_pct"]["F1"] == 0.0

    p_summary = port.get_summary()
    assert len(p_summary) == 5
    # F0 was long (+0.02 minus 5 bps fee) -> NAV > 100,000
    assert p_summary["Cascade_F0"]["nav"] > 100000.0
    # F1 was neutral (0 return minus fee) -> NAV <= 100,000
    assert p_summary["Cascade_F1"]["nav"] <= 100000.0


def test_historical_replay_execution(tmp_path):
    """Verify execution of historical replay on sample of events in an isolated temp dir."""
    import shutil
    src_sandbox = Path(__file__).resolve().parent.parent / "sandbox"
    tmp_sb = tmp_path / "sandbox"
    tmp_sb.mkdir()
    shutil.copy2(src_sandbox / "config.json", tmp_sb / "config.json")
    cfg = json.loads((tmp_sb / "config.json").read_text(encoding="utf-8"))
    cfg["data_paths"]["replay_file"] = str(
        (Path(__file__).resolve().parent.parent / "archives" / "track2_alpaca_real" / "alpaca_real_30m_stream.jsonl").resolve()
    )
    (tmp_sb / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    engine = StreamingEngine(sandbox_root=tmp_sb)
    engine.jev_client.predict = lambda snap, **kw: engine.jev_client._mock_predict(snap)
    engine.gpt_client.predict = lambda p, arm, **kw: engine.gpt_client._mock_predict(p, arm)
    engine.deepseek_client.predict = lambda p, arm, **kw: engine.deepseek_client._mock_predict(p, arm)

    res = engine.replay_historical_events(max_events=10)
    assert res["status"] == "completed"
    assert res["replayed_ticks"] == 10
    assert res["budget_state"]["total_spent_usd"] > 0.0
    assert (tmp_sb / "data" / "checkpoints.json").exists()


def test_telemetry_handlers_in_memory():
    """Verify telemetry REST API endpoints and Bearer auth enforcement in-memory without sockets."""
    from aiohttp.test_utils import make_mocked_request
    from aiohttp.web_exceptions import HTTPUnauthorized
    from sandbox.telemetry.server import (
        handle_health,
        handle_overview,
        handle_portfolio,
        handle_paired_metrics,
        handle_reasoning_stream,
    )

    def run_coro(coro):
        try:
            coro.send(None)
        except StopIteration as e:
            return e.value

    sandbox_root = Path(__file__).resolve().parent.parent / "sandbox"
    engine = StreamingEngine(sandbox_root=sandbox_root)
    set_engine_instance(engine)

    # 1. /health public endpoint
    req = make_mocked_request("GET", "/health")
    resp = run_coro(handle_health(req))
    assert resp.status == 200

    # 2. Unauthorized without auth header -> HTTP 401
    req_no_auth = make_mocked_request("GET", "/api/v1/telemetry/overview")
    with pytest.raises(HTTPUnauthorized):
        run_coro(handle_overview(req_no_auth))

    # 3. Unauthorized with invalid token -> HTTP 401
    req_bad = make_mocked_request("GET", "/api/v1/telemetry/overview", headers={"Authorization": "Bearer bad_token"})
    with pytest.raises(HTTPUnauthorized):
        run_coro(handle_overview(req_bad))

    # 4. Authorized with valid token
    token = os.environ.get("TELEMETRY_BEARER_TOKEN", "change_me_telemetry_bearer_token")
    req_auth = make_mocked_request("GET", "/api/v1/telemetry/overview", headers={"Authorization": f"Bearer {token}"})
    resp = run_coro(handle_overview(req_auth))
    assert resp.status == 200

    # 5. /portfolio
    req_port = make_mocked_request("GET", "/api/v1/telemetry/portfolio", headers={"Authorization": f"Bearer {token}"})
    resp = run_coro(handle_portfolio(req_port))
    assert resp.status == 200

    # 6. /paired_metrics
    req_pm = make_mocked_request("GET", "/api/v1/telemetry/paired_metrics", headers={"Authorization": f"Bearer {token}"})
    resp = run_coro(handle_paired_metrics(req_pm))
    assert resp.status == 200

    # 7. /reasoning_stream
    req_rs = make_mocked_request("GET", "/api/v1/telemetry/reasoning_stream?limit=5", headers={"Authorization": f"Bearer {token}"})
    resp = run_coro(handle_reasoning_stream(req_rs))
    assert resp.status == 200
