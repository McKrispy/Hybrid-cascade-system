"""Two-Tier evaluation and portfolio settlement engine.

Timestamp: 2026-11-11 13:00 (Asia/Shanghai)
"""

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from pydantic import BaseModel, Field


class PortfolioState(BaseModel):
    """Financial performance metrics of an individual shadow account."""
    account_id: str
    current_nav: float
    initial_capital: float = 100000.0
    cumulative_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    total_trades: int
    turnover_rate: float
    current_position: float = 0.0  # +1, 0, -1
    nav_history: List[Dict[str, Any]] = Field(default_factory=list)


class Tier1MetricsTracker:
    """Tracks stateless classification accuracies, paired contrasts, and error-following."""

    def __init__(self):
        self.records: List[Dict[str, Any]] = []

    def record_settled_tick(
        self,
        tick_id: str,
        symbol: str,
        gold_label: str,
        jev_decision: str,
        was_escalated: bool,
        s2_decisions: Dict[str, str],  # arm_id -> decision, e.g. {"F0": "...", "F1": "..."}
        signal_conflict: bool = False,
    ) -> None:
        """Record settled tick decisions across all arms."""
        rec = {
            "tick_id": tick_id,
            "symbol": symbol,
            "gold_label": gold_label,
            "jev_decision": jev_decision,
            "was_escalated": was_escalated,
            "s2_decisions": s2_decisions,
            "signal_conflict": signal_conflict,
        }
        self.records.append(rec)

    def compute_summary_metrics(self) -> Dict[str, Any]:
        """Compute comprehensive Tier-1 accuracy and contrast metrics."""
        if not self.records:
            return {
                "total_settled_ticks": 0,
                "overall_accuracies": {},
                "primary_contrast": {"f1_minus_f0_pp": 0.0},
                "conflict_subset": {},
            }

        total = len(self.records)
        escalated_records = [r for r in self.records if r["was_escalated"]]
        n_esc = len(escalated_records)

        # 1. Jev Standalone Accuracy
        jev_correct = sum(1 for r in self.records if r["jev_decision"] == r["gold_label"])
        jev_acc = jev_correct / total

        # 2. Cascade Accuracies (Full Stream)
        # If not escalated, all cascades use Jev's decision
        cascade_accs: Dict[str, float] = {}
        for arm in ["F0", "F1", "C0", "C1"]:
            correct_count = 0
            for r in self.records:
                if r["was_escalated"] and arm in r["s2_decisions"]:
                    ans = r["s2_decisions"][arm]
                else:
                    ans = r["jev_decision"]
                if ans == r["gold_label"]:
                    correct_count += 1
            cascade_accs[arm] = correct_count / total

        # 3. Escalated Subset Accuracies
        s2_subset_accs: Dict[str, float] = {}
        if n_esc > 0:
            for arm in ["F0", "F1", "C0", "C1"]:
                s2_c = sum(1 for r in escalated_records if r["s2_decisions"].get(arm) == r["gold_label"])
                s2_subset_accs[arm] = s2_c / n_esc

        # 4. Primary Contrast F1 - F0
        diff_pp = (cascade_accs.get("F1", 0.0) - cascade_accs.get("F0", 0.0)) * 100.0

        # 5. Error-Following Rate on Wrong Jev instances
        wrong_jev_records = [r for r in escalated_records if r["jev_decision"] != r["gold_label"]]
        n_wrong_jev = len(wrong_jev_records)
        err_follow_f0 = 0.0
        err_follow_f1 = 0.0
        if n_wrong_jev > 0:
            f0_follow = sum(1 for r in wrong_jev_records if r["s2_decisions"].get("F0") == r["jev_decision"])
            f1_follow = sum(1 for r in wrong_jev_records if r["s2_decisions"].get("F1") == r["jev_decision"])
            err_follow_f0 = f0_follow / n_wrong_jev
            err_follow_f1 = f1_follow / n_wrong_jev

        # 6. Conflict Subset Metrics
        conflict_records = [r for r in self.records if r["signal_conflict"]]
        conflict_accs: Dict[str, float] = {}
        if conflict_records:
            n_conf = len(conflict_records)
            for arm in ["F0", "F1", "C0", "C1"]:
                c_cnt = 0
                for r in conflict_records:
                    ans = r["s2_decisions"].get(arm) if r["was_escalated"] and arm in r["s2_decisions"] else r["jev_decision"]
                    if ans == r["gold_label"]:
                        c_cnt += 1
                conflict_accs[arm] = round(c_cnt / n_conf, 4)

        return {
            "total_settled_ticks": total,
            "escalated_ticks_count": n_esc,
            "escalation_percentage": round((n_esc / total) * 100.0, 2),
            "jev_standalone_acc": round(jev_acc * 100.0, 2),
            "cascade_accuracies_pct": {k: round(v * 100.0, 2) for k, v in cascade_accs.items()},
            "escalated_subset_accuracies_pct": {k: round(v * 100.0, 2) for k, v in s2_subset_accs.items()},
            "primary_contrast": {
                "metric": "Cascade Acc(F1) - Acc(F0)",
                "diff_pp": round(diff_pp, 4),
                "is_prior_degrading": diff_pp < 0.0,
            },
            "error_following": {
                "n_wrong_jev_escalated": n_wrong_jev,
                "f0_error_follow_rate_pct": round(err_follow_f0 * 100.0, 2),
                "f1_error_follow_rate_pct": round(err_follow_f1 * 100.0, 2),
                "excess_anchoring_bias_pp": round((err_follow_f1 - err_follow_f0) * 100.0, 2),
            },
            "conflict_subset": {
                "total_conflict_ticks": len(conflict_records),
                "accuracies_pct": conflict_accs,
            },
        }


class PortfolioSettlementEngine:
    """Tier-2 Five-branch shadow portfolio NAV evolver with 5 bps turnover transaction fee."""

    ACTION_WEIGHTS = {
        "LONG_ACCUMULATE": 1.0,
        "NEUTRAL_HOLD": 0.0,
        "SHORT_LIQUIDATE": -1.0,
    }

    def __init__(
        self,
        initial_capital: float = 100000.0,
        transaction_cost_bps: float = 5.0,
        position_scale: float = 1.0 / 32.0,
        horizon_scale: float = 1.0 / 8.0,
        num_assets: int = 4,
    ):
        self.initial_capital = float(initial_capital)
        self.transaction_cost = (transaction_cost_bps / 10000.0)  # 5 bps = 0.0005 unscaled
        # Time-horizon deduplication ratio Delta_t / H = 0.5h / 4h = 1/8 (applied ONLY to overlapping 4h return)
        self.horizon_scale = float(horizon_scale)
        self.num_assets = int(num_assets)
        self.per_asset_capital = self.initial_capital / float(self.num_assets)
        # Retained for backwards config logging compatibility
        self.position_scale = float(position_scale)

        self.accounts = [
            "Jev_Standalone",
            "Cascade_F0",
            "Cascade_F1",
            "Cascade_C0",
            "Cascade_C1",
        ]

        self.asset_buckets: Dict[str, Dict[str, float]] = {a: {} for a in self.accounts}
        self.navs: Dict[str, float] = {a: self.initial_capital for a in self.accounts}
        self.prev_weights: Dict[str, Dict[str, float]] = {a: {} for a in self.accounts}
        self.peak_navs: Dict[str, float] = {a: self.initial_capital for a in self.accounts}
        self.max_drawdowns: Dict[str, float] = {a: 0.0 for a in self.accounts}
        self.returns_history: Dict[str, List[float]] = {a: [] for a in self.accounts}
        self.daily_closing_navs: Dict[str, Dict[str, float]] = {a: {} for a in self.accounts}
        self.nav_history: Dict[str, List[Dict[str, Any]]] = {a: [] for a in self.accounts}
        self.total_trades: Dict[str, int] = {a: 0 for a in self.accounts}
        self.turnover: Dict[str, float] = {a: 0.0 for a in self.accounts}

    def settle_tick(
        self,
        symbol: str,
        timestamp: str,
        actual_return_4h: float,
        jev_decision: str,
        was_escalated: bool,
        s2_decisions: Dict[str, str],  # arm_id -> decision
    ) -> Dict[str, Any]:
        """Advance independent per-asset sub-portfolio NAVs with unscaled 5 bps turnover fee."""
        jev_w = self.ACTION_WEIGHTS.get(jev_decision, 0.0)

        weights_current: Dict[str, float] = {
            "Jev_Standalone": jev_w,
        }

        # For cascade accounts, if not escalated, execute Jev's weight
        for arm in ["F0", "F1", "C0", "C1"]:
            acc_id = f"Cascade_{arm}"
            if was_escalated and arm in s2_decisions:
                w = self.ACTION_WEIGHTS.get(s2_decisions[arm], 0.0)
            else:
                w = jev_w
            weights_current[acc_id] = w

        # Update independent asset sub-portfolio and aggregate total NAV
        day_key = timestamp[:10]
        for acc in self.accounts:
            if symbol not in self.asset_buckets[acc]:
                self.asset_buckets[acc][symbol] = self.per_asset_capital

            w_curr = weights_current[acc]
            w_prev = self.prev_weights[acc].get(symbol, 0.0)

            # Unscaled instantaneous 5 bps transaction fee on asset weight change
            delta_w = abs(w_curr - w_prev)
            fee = delta_w * self.transaction_cost
            if delta_w > 1e-4:
                self.total_trades[acc] += 1
                self.turnover[acc] += delta_w

            # Scale ONLY the overlapping 4h market return by (Delta_t / H = 1/8); deduct full fee directly
            asset_return = (self.horizon_scale * (w_curr * actual_return_4h)) - fee
            self.asset_buckets[acc][symbol] = max(self.asset_buckets[acc][symbol] * (1.0 + asset_return), 1.0)

            prev_nav = self.navs[acc]
            unallocated = max(0, self.num_assets - len(self.asset_buckets[acc])) * self.per_asset_capital
            new_nav = sum(self.asset_buckets[acc].values()) + unallocated
            self.navs[acc] = new_nav
            self.returns_history[acc].append((new_nav - prev_nav) / prev_nav if prev_nav > 0 else 0.0)
            self.daily_closing_navs[acc][day_key] = new_nav

            self.peak_navs[acc] = max(self.peak_navs[acc], new_nav)
            cur_dd = (
                ((self.peak_navs[acc] - new_nav) / self.peak_navs[acc]) * 100.0
                if self.peak_navs[acc] > 0
                else 0.0
            )
            self.max_drawdowns[acc] = max(self.max_drawdowns[acc], cur_dd)
            self.prev_weights[acc][symbol] = w_curr

            # Record history point
            self.nav_history[acc].append({
                "timestamp": timestamp,
                "symbol": symbol,
                "nav": round(new_nav, 2),
                "return_pct": round(((new_nav - self.initial_capital) / self.initial_capital) * 100.0, 3),
                "position": w_curr,
            })

        return self.get_summary()

    def get_summary(self) -> Dict[str, Any]:
        """Compute NAV, cumulative return, calendar-day Sharpe (sqrt 252), and drawdown."""
        summary = {}
        for acc in self.accounts:
            nav = self.navs[acc]
            cum_ret = ((nav - self.initial_capital) / self.initial_capital) * 100.0
            mdd = self.max_drawdowns[acc]

            days = sorted(self.daily_closing_navs[acc].keys())
            if len(days) >= 2:
                eq_curve = [self.initial_capital] + [self.daily_closing_navs[acc][d] for d in days]
                d_rets = [
                    (eq_curve[i + 1] - eq_curve[i]) / eq_curve[i]
                    for i in range(len(days))
                    if eq_curve[i] > 0
                ]
                std_d = float(np.std(d_rets, ddof=1)) if len(d_rets) >= 2 else 0.0
                sharpe = float((np.mean(d_rets) / std_d) * math.sqrt(252)) if std_d > 1e-9 else 0.0
            else:
                sharpe = 0.0

            summary[acc] = {
                "account_id": acc,
                "nav": round(nav, 2),
                "cumulative_return_pct": round(cum_ret, 3),
                "sharpe_ratio": round(sharpe, 3),
                "max_drawdown_pct": round(mdd, 3),
                "total_trades": self.total_trades[acc],
                "turnover": round(self.turnover[acc], 2),
                "history_points": len(self.nav_history[acc]),
                "nav_history": self.nav_history[acc][-100:],
            }
        return summary

