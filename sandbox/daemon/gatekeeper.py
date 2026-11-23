"""Dynamic rolling quantile gatekeeper, token bucket rate limiter, and three-tier kill-switch.

Timestamp: 2026-11-11 13:00 (Asia/Shanghai)
"""

import collections
import time
from typing import Any, Deque, Dict, List, Optional, Set, Tuple


class TokenBucket:
    """Token bucket rate limiter ensuring smooth hourly budget expenditure."""

    def __init__(self, capacity: float = 4.0, refill_interval_seconds: float = 1800.0, refill_tokens: float = 1.0):
        self.capacity = float(capacity)
        self.refill_interval = float(refill_interval_seconds)
        self.refill_tokens = float(refill_tokens)
        self.tokens = float(capacity)
        self.last_refill: Optional[float] = None

    def _refill(self, now_ts: Optional[float] = None) -> None:
        now = now_ts if now_ts is not None else time.time()
        if self.last_refill is None:
            self.last_refill = now
            return
        elapsed = now - self.last_refill
        if elapsed > 0:
            added = (elapsed / self.refill_interval) * self.refill_tokens
            self.tokens = min(self.capacity, self.tokens + added)
            self.last_refill = now
        elif elapsed < -3600.0:  # Reset if timestamp jumped backwards (e.g. simulation start)
            self.last_refill = now
            self.tokens = self.capacity

    def can_consume(self, amount: float = 1.0, now_ts: Optional[float] = None) -> bool:
        self._refill(now_ts=now_ts)
        return self.tokens >= amount

    def consume(self, amount: float = 1.0, now_ts: Optional[float] = None) -> bool:
        self._refill(now_ts=now_ts)
        if self.tokens >= amount:
            self.tokens -= amount
            return True
        return False

    def state(self, now_ts: Optional[float] = None) -> Dict[str, Any]:
        if now_ts is not None or self.last_refill is None:
            self._refill(now_ts=now_ts)
        return {
            "capacity": self.capacity,
            "available_tokens": round(self.tokens, 3),
            "refill_interval_seconds": self.refill_interval,
            "refill_tokens": self.refill_tokens,
        }


class DynamicGatekeeper:
    """Three-in-one dynamic gatekeeper: rolling quantile + debounce cooldown + token bucket."""

    def __init__(
        self,
        rolling_window_size: int = 384,
        quantile_threshold: float = 0.75,
        cooldown_seconds: float = 3600.0,
        token_bucket: Optional[TokenBucket] = None,
    ):
        self.rolling_window_size = rolling_window_size
        self.quantile_threshold = quantile_threshold
        self.cooldown_seconds = cooldown_seconds
        self.token_bucket = token_bucket or TokenBucket()

        # Rolling uncertainty window
        self.uncertainty_window: Deque[float] = collections.deque(maxlen=rolling_window_size)

        # Asset-level cooldown tracking: symbol -> last_timestamp
        self.last_escalation_time: Dict[str, float] = {}
        # Asset-level news IDs tracking: symbol -> set of news_id
        self.last_escalation_news: Dict[str, Set[str]] = {}

        # Gating statistics
        self.total_evaluated = 0
        self.total_escalated = 0
        self.total_suppressed_cooldown = 0
        self.total_suppressed_bucket = 0

    def compute_percentile_rank(self, uncertainty: float) -> float:
        """Calculate empirical percentile rank PR(u) in rolling window."""
        if len(self.uncertainty_window) < 10:
            # Cold-start heuristic: assume uniform [0, 1]
            return float(uncertainty)

        arr = list(self.uncertainty_window)
        count_le = sum(1 for val in arr if val <= uncertainty)
        return float(count_le / len(arr))

    def evaluate_tick(
        self,
        symbol: str,
        uncertainty: float,
        current_news_ids: Optional[List[str]] = None,
        now_ts: Optional[float] = None,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Evaluate whether a stream tick should escalate to System-2.

        Returns:
            (should_escalate, reason, metadata)
        """
        now = now_ts if now_ts is not None else time.time()
        self.total_evaluated += 1

        # 1. Update rolling window and compute quantile
        pr = self.compute_percentile_rank(uncertainty)
        self.uncertainty_window.append(uncertainty)

        current_news_set = set(current_news_ids or [])
        meta = {
            "symbol": symbol,
            "uncertainty": round(uncertainty, 4),
            "percentile_rank": round(pr, 4),
            "quantile_threshold": self.quantile_threshold,
            "rolling_window_count": len(self.uncertainty_window),
            "available_tokens": self.token_bucket.state(now_ts=now)["available_tokens"],
        }

        # Check Level 1: Dynamic Quantile Gating
        if pr < self.quantile_threshold:
            meta["gating_verdict"] = "BELOW_QUANTILE"
            return False, "below_quantile_threshold", meta

        # Check Level 2: Asset-level 60-minute Debounce & Cool-down
        last_t = self.last_escalation_time.get(symbol, 0.0)
        last_news = self.last_escalation_news.get(symbol, set())

        if (now - last_t) < self.cooldown_seconds:
            # If within cooldown, check if there are genuine new breaking news IDs
            has_new_news = len(current_news_set - last_news) > 0
            if not has_new_news:
                self.total_suppressed_cooldown += 1
                meta["gating_verdict"] = "COOLDOWN_SUPPRESSED"
                meta["elapsed_cooldown_seconds"] = round(now - last_t, 1)
                return False, "cooldown_suppressed", meta

        # Check Level 3: Token Bucket Rate Limiter
        if not self.token_bucket.consume(1.0, now_ts=now):
            self.total_suppressed_bucket += 1
            meta["gating_verdict"] = "TOKEN_BUCKET_EXHAUSTED"
            return False, "token_bucket_exhausted", meta

        # Escalation approved!
        self.total_escalated += 1
        self.last_escalation_time[symbol] = now
        self.last_escalation_news[symbol] = current_news_set

        meta["gating_verdict"] = "ESCALATE_TO_SYSTEM2"
        return True, "escalate_high_uncertainty", meta

    def state(self) -> Dict[str, Any]:
        """Return summary of gatekeeper state."""
        return {
            "total_evaluated": self.total_evaluated,
            "total_escalated": self.total_escalated,
            "escalation_rate": round(self.total_escalated / max(self.total_evaluated, 1), 4),
            "total_suppressed_cooldown": self.total_suppressed_cooldown,
            "total_suppressed_bucket": self.total_suppressed_bucket,
            "rolling_window_count": len(self.uncertainty_window),
            "current_quantile_75": (
                round(float(sorted(self.uncertainty_window)[int(len(self.uncertainty_window) * 0.75)]), 4)
                if len(self.uncertainty_window) >= 10
                else 0.50
            ),
            "token_bucket": self.token_bucket.state(),
        }


class BudgetKillSwitch:
    """Three-tier budget enforcement: Single-tick alert, 24h rolling pause, Global hard cap."""

    def __init__(
        self,
        single_tick_alert_usd: float = 0.06,
        daily_budget_cap_usd: float = 2.50,
        total_budget_cap_usd: float = 16.00,
    ):
        self.single_tick_alert_usd = single_tick_alert_usd
        self.daily_budget_cap_usd = daily_budget_cap_usd
        self.total_budget_cap_usd = total_budget_cap_usd

        # History of expenses: (timestamp, amount)
        self.expense_history: List[Tuple[float, float]] = []
        self.total_spent_usd: float = 0.0
        self.mode: str = "NORMAL"  # NORMAL, JEV_ONLY_DEGRADED, HARD_CAPPED
        self.alert_log: List[Dict[str, Any]] = []
        self.last_now_ts: Optional[float] = None

    def get_rolling_24h_spent(self, now_ts: Optional[float] = None) -> float:
        if now_ts is not None:
            self.last_now_ts = now_ts
        now = now_ts if now_ts is not None else (self.last_now_ts if self.last_now_ts is not None else time.time())
        cutoff = now - 86400.0
        # Clean older records
        self.expense_history = [e for e in self.expense_history if e[0] >= cutoff]
        return float(sum(e[1] for e in self.expense_history))

    def record_spend(self, amount_usd: float, tick_id: str, now_ts: Optional[float] = None) -> None:
        """Record an expense and evaluate budget thresholds."""
        if now_ts is not None:
            self.last_now_ts = now_ts
        now = now_ts if now_ts is not None else (self.last_now_ts if self.last_now_ts is not None else time.time())
        self.expense_history.append((now, amount_usd))
        self.total_spent_usd += amount_usd

        # Level 1 check: Single-tick alert
        if amount_usd > self.single_tick_alert_usd:
            self.alert_log.append({
                "timestamp": now,
                "level": "LEVEL_1_ALERT",
                "tick_id": tick_id,
                "amount_usd": round(amount_usd, 5),
                "threshold_usd": self.single_tick_alert_usd,
                "message": f"Single tick expense ${amount_usd:.4f} exceeded alert threshold ${self.single_tick_alert_usd:.4f}",
            })

        # Level 2 check: Rolling 24h cap
        r24 = self.get_rolling_24h_spent(now)
        if r24 >= self.daily_budget_cap_usd:
            self.mode = "JEV_ONLY_DEGRADED"
            self.alert_log.append({
                "timestamp": now,
                "level": "LEVEL_2_ROLLING_CAP",
                "rolling_24h_usd": round(r24, 4),
                "cap_usd": self.daily_budget_cap_usd,
                "message": f"Rolling 24h spend ${r24:.4f} exceeded daily cap ${self.daily_budget_cap_usd:.4f}. Entering JEV_ONLY mode.",
            })

        # Level 3 check: Global hard cap
        if self.total_spent_usd >= self.total_budget_cap_usd:
            self.mode = "HARD_CAPPED"
            self.alert_log.append({
                "timestamp": now,
                "level": "LEVEL_3_HARD_CAP",
                "total_spent_usd": round(self.total_spent_usd, 4),
                "total_cap_usd": self.total_budget_cap_usd,
                "message": f"Total project spend ${self.total_spent_usd:.4f} hit hard limit ${self.total_budget_cap_usd:.4f}. System permanently locked.",
            })

    def can_dispatch_system2(self, now_ts: Optional[float] = None) -> Tuple[bool, str]:
        """Check if paid System-2 dispatch is allowed under current budget constraints."""
        if now_ts is not None:
            self.last_now_ts = now_ts
        now = now_ts if now_ts is not None else (self.last_now_ts if self.last_now_ts is not None else time.time())

        if self.total_spent_usd >= self.total_budget_cap_usd:
            self.mode = "HARD_CAPPED"
            return False, "hard_budget_cap_exceeded"

        r24 = self.get_rolling_24h_spent(now)
        if r24 >= self.daily_budget_cap_usd:
            self.mode = "JEV_ONLY_DEGRADED"
            return False, "daily_budget_cap_exceeded"

        self.mode = "NORMAL"
        return True, "budget_ok"

    def state(self, now_ts: Optional[float] = None) -> Dict[str, Any]:
        """Return budget state summary."""
        if now_ts is not None:
            self.last_now_ts = now_ts
        now = now_ts if now_ts is not None else (self.last_now_ts if self.last_now_ts is not None else time.time())
        r24 = self.get_rolling_24h_spent(now)
        return {
            "mode": self.mode,
            "total_spent_usd": round(self.total_spent_usd, 4),
            "total_budget_cap_usd": self.total_budget_cap_usd,
            "budget_remaining_pct": round(max(0.0, 1.0 - (self.total_spent_usd / self.total_budget_cap_usd)) * 100.0, 2),
            "rolling_24h_spent_usd": round(r24, 4),
            "daily_budget_cap_usd": self.daily_budget_cap_usd,
            "alerts_count": len(self.alert_log),
        }
