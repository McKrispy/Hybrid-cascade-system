"""Streaming daemon and historical replay orchestrator for financial sandbox.

Timestamp: 2026-11-11 13:00 (Asia/Shanghai)
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .clients import DeepSeekClient, JevClient, OpenAIClient
from .fetcher import MarketDataFetcher, StreamTick
from .gatekeeper import BudgetKillSwitch, DynamicGatekeeper
from .handoff import FinancialHandoffBuilder
from .settlement import PortfolioSettlementEngine, Tier1MetricsTracker


class StreamingEngine:
    """Core asynchronous orchestrator managing tick ingestion, gating, handoff, and settlement."""

    def __init__(
        self,
        sandbox_root: Optional[Path] = None,
        config: Optional[Dict[str, Any]] = None,
        data_dir_override: Optional[Path] = None,
        replay_path_override: Optional[Path] = None,
        ignore_checkpoint: bool = False,
    ):
        self.root = sandbox_root or Path(__file__).resolve().parent.parent
        self.config = config or self._load_config()

        # Data directories & files
        self.data_dir = Path(data_dir_override) if data_dir_override else (self.root / "data")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        if data_dir_override:
            self.records_path = self.data_dir / "records.jsonl"
            self.ledger_path = self.data_dir / "ledger.jsonl"
            self.checkpoints_path = self.data_dir / "checkpoints.json"
        else:
            self.records_path = self.root / self.config.get("data_paths", {}).get("records_file", "data/records.jsonl")
            self.ledger_path = self.root / self.config.get("data_paths", {}).get("ledger_file", "data/ledger.jsonl")
            self.checkpoints_path = self.root / self.config.get("data_paths", {}).get("checkpoints_file", "data/checkpoints.json")
        self.replay_path = (
            Path(replay_path_override)
            if replay_path_override
            else (self.root / self.config.get("data_paths", {}).get("replay_file", "replay_data/high_volatility_events.jsonl"))
        )

        # Initialize sub-components
        self.fetcher = MarketDataFetcher(self.config)
        tb_cfg = self.config.get("gatekeeper", {}).get("token_bucket", {})
        from .gatekeeper import TokenBucket
        token_bucket = TokenBucket(
            capacity=tb_cfg.get("capacity", 4),
            refill_interval_seconds=tb_cfg.get("refill_interval_minutes", 30) * 60.0,
            refill_tokens=tb_cfg.get("refill_tokens", 1),
        )
        self.gatekeeper = DynamicGatekeeper(
            quantile_threshold=self.config.get("gatekeeper", {}).get("quantile_threshold", 0.75),
            cooldown_seconds=self.config.get("gatekeeper", {}).get("cooldown_minutes", 60) * 60.0,
            token_bucket=token_bucket,
        )
        self.budget_killswitch = BudgetKillSwitch(
            single_tick_alert_usd=self.config.get("budget_killswitch", {}).get("single_tick_alert_usd", 0.06),
            daily_budget_cap_usd=self.config.get("budget_killswitch", {}).get("daily_budget_cap_usd", 2.50),
            total_budget_cap_usd=self.config.get("budget_killswitch", {}).get("total_budget_cap_usd", 16.00),
        )

        self.jev_client = JevClient()
        self.gpt_client = OpenAIClient()
        self.deepseek_client = DeepSeekClient()

        self.tier1_tracker = Tier1MetricsTracker()
        self.portfolio_engine = PortfolioSettlementEngine(
            initial_capital=self.config.get("portfolio", {}).get("initial_capital_usd", 100000.0),
            transaction_cost_bps=self.config.get("portfolio", {}).get("transaction_cost_bps", 5.0),
            position_scale=self.config.get("portfolio", {}).get("position_scale", 1.0 / 32.0),
        )

        # Runtime state
        self.is_running = False
        self.processed_ticks_count = 0
        self.escalated_ticks_count = 0
        self.recent_reasoning_stream: List[Dict[str, Any]] = []
        if not ignore_checkpoint:
            self._load_checkpoint()

    def _load_config(self) -> Dict[str, Any]:
        cfg_file = self.root / "config.json"
        if cfg_file.exists():
            with open(cfg_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _load_checkpoint(self) -> None:
        """Load state from previous checkpoint if present."""
        if self.checkpoints_path.exists():
            try:
                with open(self.checkpoints_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.processed_ticks_count = data.get("processed_ticks_count", 0)
                self.escalated_ticks_count = data.get("escalated_ticks_count", 0)
                self.budget_killswitch.total_spent_usd = data.get("total_spent_usd", 0.0)
            except Exception:
                pass

    def save_checkpoint(self) -> None:
        """Persist current state to checkpoints.json."""
        state = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
            "processed_ticks_count": self.processed_ticks_count,
            "escalated_ticks_count": self.escalated_ticks_count,
            "total_spent_usd": round(self.budget_killswitch.total_spent_usd, 6),
            "gatekeeper": self.gatekeeper.state(),
            "budget": self.budget_killswitch.state(),
            "portfolio": self.portfolio_engine.get_summary(),
        }
        with open(self.checkpoints_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

    def process_tick(self, tick: StreamTick, mock_fallback: bool = False, timeout: float = 45.0) -> Dict[str, Any]:
        """Execute full evaluation pipeline for a single tick (synchronous)."""
        self.processed_ticks_count += 1
        tick_cost = 0.0

        try:
            tick_ts = time.mktime(time.strptime(tick.timestamp, "%Y-%m-%d %H:%M:%S"))
        except Exception:
            tick_ts = time.time()

        # 1. System-1 Jev Fast Prior
        snap_text = (
            f"Asset: {tick.symbol}, Price: {tick.snapshot.current_price}, "
            f"24h_change: {tick.snapshot.price_change_24h_pct}%, "
            f"RSI: {tick.snapshot.indicators.rsi_14}, MACD_hist: {tick.snapshot.indicators.macd_hist}, "
            f"Trend: {tick.snapshot.indicators.trend_regime}. News: "
            + "; ".join(n.headline for n in tick.news_stream[:2])
        )
        jev_res = self.jev_client.predict(snap_text, mock_fallback=mock_fallback, timeout=timeout)
        tick_cost += jev_res.cost_usd

        # 2. Gatekeeper Gating Check
        news_ids = [n.news_id for n in tick.news_stream]
        should_escalate, gate_reason, gate_meta = self.gatekeeper.evaluate_tick(
            symbol=tick.symbol,
            uncertainty=jev_res.uncertainty,
            current_news_ids=news_ids,
            now_ts=tick_ts,
        )

        # 3. Budget Kill-Switch Check
        can_dispatch, budget_reason = self.budget_killswitch.can_dispatch_system2(now_ts=tick_ts)
        if not can_dispatch:
            should_escalate = False
            gate_meta["budget_override"] = budget_reason

        # 4. System-2 Execution (if escalated)
        # Bug 6 fix: dispatch BOTH GPT-6.1-Sol and DeepSeek-v4.1-Flash for all 4 arms (8 total)
        s2_decisions: Dict[str, str] = {}       # primary (GPT) decisions for Tier-1/Tier-2
        s2_records_gpt: Dict[str, Any] = {}
        s2_records_ds: Dict[str, Any] = {}
        s2_decisions_ds: Dict[str, str] = {}  # DeepSeek decisions for heterogeneous contrast

        if should_escalate:
            self.escalated_ticks_count += 1
            arms = ["F0", "F1", "C0", "C1"]
            prompts = {arm: FinancialHandoffBuilder.build_prompt(tick, arm, jev_res) for arm in arms}

            def _call_gpt(arm: str) -> Dict[str, Any]:
                res = self.gpt_client.predict(prompts[arm], arm, mock_fallback=mock_fallback, timeout=timeout)
                return res.dict()

            def _call_ds(arm: str) -> Dict[str, Any]:
                try:
                    res_ds = self.deepseek_client.predict(prompts[arm], arm, mock_fallback=mock_fallback, timeout=timeout)
                    return res_ds.dict()
                except Exception as e:
                    print(f"[!] DeepSeek dispatch failed for arm {arm}: {e}")
                    res_ds = self.deepseek_client._mock_predict(prompts[arm], arm)
                    d = res_ds.dict()
                    d["is_mock_fallback"] = True
                    d["fallback_error"] = str(e)
                    return d

            with ThreadPoolExecutor(max_workers=8) as pool:
                gpt_futs = {arm: pool.submit(_call_gpt, arm) for arm in arms}
                ds_futs = {arm: pool.submit(_call_ds, arm) for arm in arms}
                for arm in arms:
                    gpt_rec = gpt_futs[arm].result()
                    s2_records_gpt[arm] = gpt_rec
                    s2_decisions[arm] = gpt_rec["decision"]
                    tick_cost += float(gpt_rec.get("cost_usd", 0.0))

                    ds_rec = ds_futs[arm].result()
                    s2_records_ds[arm] = ds_rec
                    s2_decisions_ds[arm] = ds_rec["decision"]
                    if not ds_rec.get("is_mock_fallback"):
                        tick_cost += float(ds_rec.get("cost_usd", 0.0))

            # If GPT failed? Note GPT exception propagates before writing record;
            # Also save full tick reasoning cleanly.

            # Record in recent reasoning stream for telemetry (last 50 entries)
            self.recent_reasoning_stream.append({
                "tick_id": tick.tick_id,
                "symbol": tick.symbol,
                "timestamp": tick.timestamp,
                "signal_conflict": tick.signal_conflict_flag,
                "jev_prediction": jev_res.dict(),
                "gpt_decisions": {k: v.get("decision") for k, v in s2_records_gpt.items()},
                "deepseek_decisions": s2_decisions_ds,
                "gpt_f1_reasoning": s2_records_gpt.get("F1", {}),
                "deepseek_f1_reasoning": s2_records_ds.get("F1", {}),
            })
            if len(self.recent_reasoning_stream) > 50:
                self.recent_reasoning_stream.pop(0)

        # Record spend in budget ledger
        self.budget_killswitch.record_spend(tick_cost, tick.tick_id, now_ts=tick_ts)

        # 5. Settlement (Tier-1 and Tier-2)
        # Determine 4h future return and gold label
        if tick.future_price_4h is not None:
            r_4h = (tick.future_price_4h - tick.snapshot.current_price) / max(tick.snapshot.current_price, 1e-4)
        elif tick.future_return_4h is not None:
            r_4h = tick.future_return_4h
        else:
            # Simulated 4h return based on technical indicators and noise
            noise = float(np.random.normal(0, tick.snapshot.indicators.volatility_24h * 0.5))
            r_4h = noise

        theta_m = self.config.get("volatility_threshold_factor", 0.5) * tick.snapshot.indicators.volatility_24h
        if tick.gold_label:
            gold = tick.gold_label
        else:
            if r_4h > theta_m:
                gold = "LONG_ACCUMULATE"
            elif r_4h < -theta_m:
                gold = "SHORT_LIQUIDATE"
            else:
                gold = "NEUTRAL_HOLD"

        # Record Tier-1
        self.tier1_tracker.record_settled_tick(
            tick_id=tick.tick_id,
            symbol=tick.symbol,
            gold_label=gold,
            jev_decision=jev_res.decision,
            was_escalated=should_escalate,
            s2_decisions=s2_decisions,
            signal_conflict=tick.signal_conflict_flag,
        )

        # Settle Tier-2 Portfolio
        portfolio_summary = self.portfolio_engine.settle_tick(
            symbol=tick.symbol,
            timestamp=tick.timestamp,
            actual_return_4h=r_4h,
            jev_decision=jev_res.decision,
            was_escalated=should_escalate,
            s2_decisions=s2_decisions,
        )

        # 6. Append to log files
        record_entry = {
            "tick_id": tick.tick_id,
            "symbol": tick.symbol,
            "timestamp": tick.timestamp,
            "current_price": tick.snapshot.current_price,
            "signal_conflict_flag": tick.signal_conflict_flag,
            "jev_decision": jev_res.decision,
            "jev_uncertainty": jev_res.uncertainty,
            "jev_probabilities": jev_res.probabilities,
            "jev_top2": jev_res.top2_candidates,
            "was_escalated": should_escalate,
            "gate_reason": gate_reason,
            "gold_label": gold,
            "actual_return_4h": round(r_4h, 5),
            "s2_decisions_gpt": s2_decisions,            # GPT-6.1-Sol (primary, drives portfolio)
            "s2_decisions_deepseek": s2_decisions_ds if should_escalate else {},
            "s2_f1_reasoning_gpt": s2_records_gpt.get("F1") if should_escalate else None,
            "s2_f1_reasoning_deepseek": s2_records_ds.get("F1") if should_escalate else None,
            "deepseek_fallback_arms": [
                arm for arm, rec in s2_records_ds.items() if rec.get("is_mock_fallback")
            ] if should_escalate else [],
            "tick_cost_usd": round(tick_cost, 6),
        }
        with open(self.records_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record_entry) + "\n")

        ledger_entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
            "tick_id": tick.tick_id,
            "cost_usd": round(tick_cost, 6),
            "total_spent_usd": round(self.budget_killswitch.total_spent_usd, 6),
            "was_escalated": should_escalate,
        }
        with open(self.ledger_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(ledger_entry) + "\n")

        self.save_checkpoint()
        return record_entry

    def replay_historical_events(
        self,
        max_events: Optional[int] = None,
        max_escalated_ticks: Optional[int] = None,
        progress_callback: Optional[Any] = None,
        resume: bool = False,
    ) -> Dict[str, Any]:
        """Replay all historical high-volatility event ticks (synchronous)."""
        if not self.replay_path.exists():
            return {"error": f"Replay file not found: {self.replay_path}"}

        ticks: List[StreamTick] = []
        with open(self.replay_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    ticks.append(StreamTick(**data))

        if max_events is not None:
            ticks = ticks[:max_events]

        done_ids: set = set()
        if resume and self.records_path.exists():
            with open(self.records_path, "r", encoding="utf-8") as rf:
                for line in rf:
                    if line.strip():
                        try:
                            rec = json.loads(line)
                            done_ids.add(rec["tick_id"])
                        except Exception:
                            pass

        processed_in_run = len(done_ids) if resume else 0
        for tick in ticks:
            if resume and tick.tick_id in done_ids:
                # Re-hydrate engine state cleanly if resuming via separate resume helper
                continue
            entry = self.process_tick(tick)
            processed_in_run += 1
            if progress_callback is not None:
                progress_callback(processed_in_run, self.escalated_ticks_count, entry)
            if max_escalated_ticks is not None and self.escalated_ticks_count >= max_escalated_ticks:
                break

        return {
            "status": "completed",
            "replayed_ticks": processed_in_run,
            "escalated_ticks": self.escalated_ticks_count,
            "tier1_metrics": self.tier1_tracker.compute_summary_metrics(),
            "portfolio_summary": self.portfolio_engine.get_summary(),
            "budget_state": self.budget_killswitch.state(),
            "gatekeeper_state": self.gatekeeper.state(),
        }

    async def replay_historical_events_async(self, max_events: Optional[int] = None) -> Dict[str, Any]:
        """Replay all historical high-volatility event ticks (asynchronous)."""
        return self.replay_historical_events(max_events=max_events)

    def get_telemetry_snapshot(self) -> Dict[str, Any]:
        """Generate structured snapshot for the REST telemetry server."""
        tier1 = self.tier1_tracker.compute_summary_metrics()
        port = self.portfolio_engine.get_summary()
        bg = self.budget_killswitch.state()
        gk = self.gatekeeper.state()

        return {
            "overview": {
                "sandbox_id": self.config.get("sandbox_id", "B08-SANDBOX-v2.0"),
                "status": "RUNNING" if self.is_running else "READY",
                "mode": bg.get("mode", "NORMAL"),
                "total_ticks_processed": self.processed_ticks_count,
                "total_ticks_escalated": self.escalated_ticks_count,
                "escalation_rate_pct": round((self.escalated_ticks_count / max(self.processed_ticks_count, 1)) * 100.0, 2),
                "total_spent_usd": bg.get("total_spent_usd", 0.0),
                "daily_spent_usd": bg.get("rolling_24h_spent_usd", 0.0),
                "budget_cap_usd": bg.get("total_budget_cap_usd", 16.0),
                "budget_remaining_pct": bg.get("budget_remaining_pct", 100.0),
                "token_bucket_available": gk.get("token_bucket", {}).get("available_tokens", 4.0),
                "current_quantile_75": gk.get("current_quantile_75", 0.50),
            },
            "portfolio": port,
            "paired_metrics": tier1,
            "reasoning_stream": self.recent_reasoning_stream[-20:],
        }
