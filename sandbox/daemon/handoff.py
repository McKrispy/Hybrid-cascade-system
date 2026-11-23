"""Financial handoff prompt builder for F0, F1, C0, C1 conditions.

Timestamp: 2026-11-11 13:00 (Asia/Shanghai)
"""

from typing import Any, Dict, List, Optional
from .fetcher import StreamTick
from .clients import JevPrediction


class FinancialHandoffBuilder:
    """Builds controlled prompts across the 4 handoff conditions."""

    ALL_ACTIONS = [
        ("LONG_ACCUMULATE", "Expect asset price to rise by > 0.5 sigma over next 4 hours (Weight: +1.0)"),
        ("NEUTRAL_HOLD", "Expect asset price to stay within +/- 0.5 sigma over next 4 hours (Weight: 0.0)"),
        ("SHORT_LIQUIDATE", "Expect asset price to fall by > 0.5 sigma over next 4 hours (Weight: -1.0)"),
    ]

    @classmethod
    def build_prompt(
        cls,
        tick: StreamTick,
        arm_id: str,
        jev_pred: Optional[JevPrediction] = None,
    ) -> str:
        """Construct prompt for a specific handoff arm (F0, F1, C0, C1)."""
        snap = tick.snapshot
        ind = snap.indicators

        lines = [
            "# FINANCIAL DECISION DISPATCH: MULTI-FACTOR REASONING",
            "",
            "## 1. Asset & Market Snapshot",
            f"- Asset Symbol: {snap.symbol}",
            f"- Snapshot Timestamp: {snap.timestamp}",
            f"- Current Price: ${snap.current_price:,.2f}",
            f"- 24h Price Change: {snap.price_change_24h_pct:+.2f}% (High: ${snap.high_24h:,.2f}, Low: ${snap.low_24h:,.2f})",
            f"- 24h Realized Volatility: {ind.volatility_24h:.4f}",
            f"- Technical Indicators: RSI(14)={ind.rsi_14:.1f}, MACD_hist={ind.macd_hist:+.4f}, Bollinger_%B={ind.bollinger_pct_b:.2f}, Volume_Ratio={ind.volume_ratio_20:.2f}x",
            f"- Technical Regime: {ind.trend_regime}",
            "",
            "## 2. Breaking News Stream & Sentiment",
        ]

        if tick.news_stream:
            for n in tick.news_stream:
                lines.append(f"- [{n.published_at}] {n.headline} (Sentiment score: {n.sentiment_score:+.2f})")
        else:
            lines.append("- No breaking news events recorded in current monitoring window.")

        lines.append("")

        # Candidate Space Definition (Full vs Top-2 Pruned)
        if arm_id in ["C0", "C1"]:
            if jev_pred and jev_pred.top2_candidates:
                top2_labels = [c["label"] for c in jev_pred.top2_candidates]
                candidates = [a for a in cls.ALL_ACTIONS if a[0] in top2_labels]
            else:
                candidates = cls.ALL_ACTIONS[:2]
            lines.append("## 3. Allowed Candidate Actions (Top-2 Pruned Set)")
        else:
            candidates = cls.ALL_ACTIONS
            lines.append("## 3. Allowed Candidate Actions (Full Candidate Set)")

        for act, desc in candidates:
            lines.append(f"- `{act}`: {desc}")

        lines.append("")

        # Prior Injection Block (F1, C1 only)
        if arm_id in ["F1", "C1"]:
            lines.append("## 4. System-1 Fast Prior Information")
            if jev_pred:
                lines.append("A specialized fast pattern classifier (System-1 Jev) analyzed this tick and provided the following prior:")
                for c in jev_pred.top2_candidates:
                    lines.append(f"  - Rank {c['rank']}: `{c['label']}` (Unnormalized probability: {c['probability']:.4f})")
            else:
                lines.append("Fast Prior: LONG_ACCUMULATE (Rank 1, p=0.6500), NEUTRAL_HOLD (Rank 2, p=0.2500)")

            lines.append("")
            lines.append("> **Important Guidance**: The System-1 Fast Prior is provided for reference only. It relies heavily on surface lexical cues and fast momentum. You have full authority to override it if fundamental, technical, or causal reasoning suggests a different outcome.")
            lines.append("")

        # Output Schema Instructions
        lines.append("## 5. Decision Output Schema")
        lines.append("You must output strict JSON with the following 4 fields:")
        lines.append("```json")
        lines.append("{")
        lines.append('  "quant_vs_news_synthesis": "1-2 sentences synthesizing technical indicators against breaking news headlines, explicitly identifying any divergence.",')
        if arm_id in ["F0", "C0"]:
            lines.append('  "prior_stance_evaluation": "N/A (No numerical prior provided)",')
        else:
            lines.append('  "prior_stance_evaluation": "Explicitly evaluate whether the System-1 fast prior is justified or misled by surface cues.",')
        lines.append('  "reasoning_chain": "Step-by-step causal multi-factor deduction weighing momentum sustainability and risk-reward asymmetry.",')
        lines.append('  "decision": "<ONE_OF_ALLOWED_CANDIDATE_ACTIONS>"')
        lines.append("}")
        lines.append("```")

        return "\n".join(lines)

