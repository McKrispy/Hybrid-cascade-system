"""Fetch 100% real 30-min historical bars and news from Alpaca Market Data API and build replay stream."""

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SANDBOX_ROOT = PROJECT_ROOT / "sandbox"
sys.path.insert(0, str(SANDBOX_ROOT))

from daemon.fetcher import (  # noqa: E402
    MarketDataFetcher,
    MarketSnapshot,
    NewsArticle,
    StreamTick,
    TechnicalIndicators,
    compute_bollinger,
    compute_macd,
    compute_rsi,
    compute_volatility_24h,
    compute_volume_ratio,
    detect_signal_conflict,
)


BULLISH_WORDS = (
    "surge", "surges", "rally", "rallies", "gain", "gains", "beat", "beats",
    "bullish", "upgrade", "upgrades", "record high", "outperform", "growth",
    "inflow", "inflows", "breakout", "boost", "soar", "soars", "boom", "optimis",
)
BEARISH_WORDS = (
    "drop", "drops", "fall", "falls", "plunge", "plunges", "miss", "misses",
    "bearish", "downgrade", "downgrades", "lawsuit", "probe", "warning",
    "outflow", "outflows", "slump", "crash", "regulatory", "selloff", "loss", "risk",
)


def load_sandbox_env() -> None:
    env_file = SANDBOX_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def parse_iso_utc(ts_str: str) -> datetime:
    if ts_str.endswith("Z"):
        ts_str = ts_str[:-1] + "+00:00"
    return datetime.fromisoformat(ts_str).astimezone(timezone.utc)


def score_sentiment(headline: str, summary: str) -> float:
    text = f"{headline} {summary}".lower()
    pos = sum(1 for w in BULLISH_WORDS if w in text)
    neg = sum(1 for w in BEARISH_WORDS if w in text)
    if pos == 0 and neg == 0:
        return 0.0
    score = (pos - neg) / max(pos + neg, 1)
    return round(float(np.clip(score * 0.85, -0.95, 0.95)), 2)


def build_ticks() -> List[StreamTick]:
    load_sandbox_env()
    config = json.loads((SANDBOX_ROOT / "config.json").read_text(encoding="utf-8"))
    fetcher = MarketDataFetcher(config)

    end_dt = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
    start_dt = end_dt - timedelta(days=30)
    start_iso = start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_iso = end_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    symbols = config.get("symbols", ["BTC/USD", "ETH/USD", "NVDA", "TSLA"])
    news_query_symbols = [
        "NVDA", "TSLA", "COIN", "MSTR", "IBIT", "ETHA", "MARA", "RIOT", "CRYPTO:BTC", "CRYPTO:ETH"
    ]
    all_news_raw = fetcher.fetch_alpaca_news(news_query_symbols, start_iso=start_iso, end_iso=end_iso, limit=50, max_pages=60)

    news_by_sym: Dict[str, List[Dict[str, Any]]] = {s: [] for s in symbols}
    btc_keywords = ("bitcoin", "btc", "crypto", "coinbase", "microstrategy", "strategy", "ibit", "mara", "riot")
    eth_keywords = ("ethereum", "eth", "ether", "etha", "crypto", "coinbase")
    for art in all_news_raw:
        art_syms = set(art.get("symbols") or [])
        text_low = f"{art.get('headline', '')} {art.get('summary', '')}".lower()
        if "NVDA" in art_syms:
            news_by_sym["NVDA"].append(art)
        if "TSLA" in art_syms:
            news_by_sym["TSLA"].append(art)
        if (
            "BTCUSD" in art_syms or "BTC/USD" in art_syms or "CRYPTO:BTC" in art_syms
            or any(s in art_syms for s in ("IBIT", "MSTR", "MARA", "RIOT", "COIN"))
            or any(k in text_low for k in btc_keywords)
        ):
            news_by_sym["BTC/USD"].append(art)
        if (
            "ETHUSD" in art_syms or "ETH/USD" in art_syms or "CRYPTO:ETH" in art_syms
            or any(s in art_syms for s in ("ETHA", "COIN"))
            or any(k in text_low for k in eth_keywords)
        ):
            news_by_sym["ETH/USD"].append(art)
    for s in symbols:
        # Deduplicate by news id
        seen_ids = set()
        deduped = []
        for a in sorted(news_by_sym[s], key=lambda x: x["created_at"]):
            if a["id"] not in seen_ids:
                seen_ids.add(a["id"])
                deduped.append(a)
        news_by_sym[s] = deduped

    ticks: List[StreamTick] = []
    lookback = 48  # 24h of 30-min bars
    horizon_bars = 8  # 4h future settlement (8 * 30m)
    vol_factor = float(config.get("volatility_threshold_factor", 0.5))

    summary_stats: Dict[str, Any] = {}
    for sym in symbols:
        bars = fetcher.fetch_alpaca_bars(sym, start_iso=start_iso, end_iso=end_iso, timeframe="30Min", limit=1000)
        bars.sort(key=lambda b: b["t"])
        closes = np.array([float(b["c"]) for b in bars], dtype=float)
        highs = np.array([float(b["h"]) for b in bars], dtype=float)
        lows = np.array([float(b["l"]) for b in bars], dtype=float)
        vols = np.array([float(b["v"]) for b in bars], dtype=float)
        times = [parse_iso_utc(b["t"]) for b in bars]

        sym_ticks = 0
        # Target up to 300 latest 30m bars per symbol (~1,000-1,200 total ticks across 4 assets)
        valid_indices = list(range(lookback, len(bars) - horizon_bars))
        if len(valid_indices) > 300:
            valid_indices = valid_indices[-300:]

        sym_news = news_by_sym[sym]
        for idx in valid_indices:
            t_dt = times[idx]
            win_c = closes[idx - lookback + 1 : idx + 1]
            win_h = highs[idx - lookback + 1 : idx + 1]
            win_l = lows[idx - lookback + 1 : idx + 1]
            win_v = vols[idx - lookback + 1 : idx + 1]

            cur_p = float(closes[idx])
            p_24h = float(closes[idx - lookback + 1])
            chg_pct = ((cur_p - p_24h) / max(p_24h, 1e-8)) * 100.0

            rsi = compute_rsi(win_c, 14)
            m_val, m_sig, m_hist = compute_macd(win_c)
            pct_b, _, _, _ = compute_bollinger(win_c, 20)
            vol_24h = compute_volatility_24h(win_c)
            vol_ratio = compute_volume_ratio(win_v, 20)
            trend = "BULLISH" if (rsi > 60 and m_hist > 0) else ("BEARISH" if (rsi < 40 and m_hist < 0) else "NEUTRAL")

            indicators = TechnicalIndicators(
                rsi_14=round(rsi, 2),
                macd_val=round(m_val, 4),
                macd_signal=round(m_sig, 4),
                macd_hist=round(m_hist, 4),
                bollinger_pct_b=round(pct_b, 4),
                volatility_24h=round(vol_24h, 5),
                volume_ratio_20=round(vol_ratio, 2),
                trend_regime=trend,
            )
            snapshot = MarketSnapshot(
                symbol=sym,
                timestamp=t_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                current_price=round(cur_p, 2 if cur_p > 10 else 4),
                price_change_24h_pct=round(chg_pct, 2),
                high_24h=round(float(np.max(win_h)), 2),
                low_24h=round(float(np.min(win_l)), 2),
                volume_24h=round(float(np.sum(win_v)), 2),
                indicators=indicators,
            )

            # Match up to 2 most recent real Alpaca news articles published <= t_dt (within 96h)
            cutoff_news = t_dt - timedelta(hours=96)
            matched = [
                a for a in sym_news
                if cutoff_news <= parse_iso_utc(a["created_at"]) <= t_dt
            ][-2:]
            news_objs = [
                NewsArticle(
                    news_id=f"alpaca_{a['id']}",
                    headline=a.get("headline", "").strip(),
                    summary=(a.get("summary") or a.get("headline") or "").strip()[:300],
                    source=a.get("source", "alpaca"),
                    published_at=a.get("created_at", ""),
                    sentiment_score=score_sentiment(a.get("headline", ""), a.get("summary", "")),
                    symbols=[sym],
                )
                for a in reversed(matched)
            ]

            has_conflict, reason = detect_signal_conflict(indicators, news_objs)
            fut_p = float(closes[idx + horizon_bars])
            fut_ret = (fut_p - cur_p) / max(cur_p, 1e-8)
            theta_m = vol_factor * indicators.volatility_24h
            if fut_ret > theta_m:
                gold = "LONG_ACCUMULATE"
            elif fut_ret < -theta_m:
                gold = "SHORT_LIQUIDATE"
            else:
                gold = "NEUTRAL_HOLD"

            ts_epoch = int(t_dt.timestamp())
            ticks.append(
                StreamTick(
                    tick_id=f"alpaca_{sym.replace('/', '_')}_{ts_epoch}",
                    symbol=sym,
                    timestamp=t_dt.strftime("%Y-%m-%d %H:%M:%S"),
                    snapshot=snapshot,
                    news_stream=news_objs,
                    signal_conflict_flag=has_conflict,
                    conflict_reason=reason,
                    future_price_4h=round(fut_p, 4),
                    future_return_4h=round(fut_ret, 6),
                    gold_label=gold,
                )
            )
            sym_ticks += 1

        summary_stats[sym] = {
            "total_bars_fetched": len(bars),
            "news_articles_fetched": len(sym_news),
            "ticks_emitted": sym_ticks,
            "price_min": round(float(np.min(closes)), 2) if len(closes) else None,
            "price_max": round(float(np.max(closes)), 2) if len(closes) else None,
        }

    ticks.sort(key=lambda t: (t.timestamp, t.symbol))
    out_path = PROJECT_ROOT / "archives" / "track2_alpaca_real" / "alpaca_real_30m_stream.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for t in ticks:
            f.write(t.model_dump_json() + "\n")

    print(f"Wrote {len(ticks)} real 30-min Alpaca ticks to {out_path}")
    print("Per-symbol stats:", json.dumps(summary_stats, indent=2))
    unique_news = {n.news_id for t in ticks for n in t.news_stream}
    conflicts = sum(1 for t in ticks if t.signal_conflict_flag)
    gold_dist = {
        lbl: sum(1 for t in ticks if t.gold_label == lbl)
        for lbl in ("LONG_ACCUMULATE", "NEUTRAL_HOLD", "SHORT_LIQUIDATE")
    }
    print(f"Unique real Alpaca news articles matched: {len(unique_news)} | Signal conflicts: {conflicts}")
    print(f"Gold distribution: {gold_dist}")
    return ticks


if __name__ == "__main__":
    build_ticks()
