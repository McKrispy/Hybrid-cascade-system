"""Market data, news stream fetcher and technical indicator engine.

Timestamp: 2026-11-11 13:00 (Asia/Shanghai)
"""

import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np
from pydantic import BaseModel, Field


class NewsArticle(BaseModel):
    """Financial news headline and metadata."""
    news_id: str
    headline: str
    summary: str
    source: str
    published_at: str
    sentiment_score: float = Field(default=0.0, description="[-1.0 bearish to +1.0 bullish]")
    symbols: List[str] = Field(default_factory=list)


class TechnicalIndicators(BaseModel):
    """Standardized technical momentum and volatility indicators."""
    rsi_14: float
    macd_val: float
    macd_signal: float
    macd_hist: float
    bollinger_pct_b: float
    volatility_24h: float
    volume_ratio_20: float
    trend_regime: str = Field(default="NEUTRAL", description="BULLISH, BEARISH, RANGE_BOUND")


class MarketSnapshot(BaseModel):
    """Instantaneous price, volume and technical snapshot for an asset."""
    symbol: str
    timestamp: str
    current_price: float
    price_change_24h_pct: float
    high_24h: float
    low_24h: float
    volume_24h: float
    indicators: TechnicalIndicators


class StreamTick(BaseModel):
    """Complete evaluation tick containing market data, news, and conflict flag."""
    tick_id: str
    symbol: str
    timestamp: str
    snapshot: MarketSnapshot
    news_stream: List[NewsArticle] = Field(default_factory=list)
    signal_conflict_flag: bool = False
    conflict_reason: Optional[str] = None
    future_price_4h: Optional[float] = None
    future_return_4h: Optional[float] = None
    gold_label: Optional[str] = None


def compute_rsi(prices: np.ndarray, period: int = 14) -> float:
    """Compute Relative Strength Index (RSI)."""
    if len(prices) < period + 1:
        return 50.0
    deltas = np.diff(prices)
    gains = np.maximum(deltas, 0.0)
    losses = np.maximum(-deltas, 0.0)

    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])

    if avg_loss == 0.0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return float(np.clip(rsi, 0.0, 100.0))


def compute_macd(prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float, float]:
    """Compute Moving Average Convergence Divergence (MACD)."""
    if len(prices) < slow + signal:
        return 0.0, 0.0, 0.0

    def ema(series: np.ndarray, span: int) -> np.ndarray:
        alpha = 2.0 / (span + 1.0)
        weights = (1.0 - alpha) ** np.arange(len(series))[::-1]
        weights /= weights.sum()
        return np.convolve(series, weights, mode="valid")

    ema_fast = ema(prices, fast)
    ema_slow = ema(prices, slow)
    min_len = min(len(ema_fast), len(ema_slow))
    macd_line = ema_fast[-min_len:] - ema_slow[-min_len:]

    if len(macd_line) < signal:
        return float(macd_line[-1]), 0.0, float(macd_line[-1])

    signal_line = ema(macd_line, signal)
    val = float(macd_line[-1])
    sig = float(signal_line[-1])
    hist = val - sig
    return val, sig, hist


def compute_bollinger(prices: np.ndarray, period: int = 20, num_std: float = 2.0) -> Tuple[float, float, float, float]:
    """Compute Bollinger Bands (%B, upper, mid, lower)."""
    if len(prices) < period:
        p = float(prices[-1]) if len(prices) > 0 else 100.0
        return 0.5, p * 1.05, p, p * 0.95

    window = prices[-period:]
    mid = float(np.mean(window))
    std = float(np.std(window))
    upper = mid + num_std * std
    lower = mid - num_std * std
    cur = float(prices[-1])

    denom = upper - lower
    pct_b = (cur - lower) / denom if denom > 1e-8 else 0.5
    return float(pct_b), upper, mid, lower


def compute_volatility_24h(prices: np.ndarray) -> float:
    """Compute 24-hour log return volatility."""
    if len(prices) < 2:
        return 0.015
    log_returns = np.diff(np.log(np.maximum(prices, 1e-8)))
    std = float(np.std(log_returns))
    return max(std, 0.005)


def compute_volume_ratio(volumes: np.ndarray, period: int = 20) -> float:
    """Compute volume ratio relative to 20-period moving average."""
    if len(volumes) < period:
        return 1.0
    avg_vol = float(np.mean(volumes[-period:]))
    cur_vol = float(volumes[-1])
    return cur_vol / max(avg_vol, 1.0)


def detect_signal_conflict(indicators: TechnicalIndicators, news: List[NewsArticle]) -> Tuple[bool, Optional[str]]:
    """Detect conflict between technical momentum and news sentiment."""
    if not news:
        return False, None

    avg_sentiment = float(np.mean([n.sentiment_score for n in news]))
    rsi = indicators.rsi_14
    hist = indicators.macd_hist

    # Bearish technicals vs Bullish news
    tech_bearish = (rsi > 70.0 and hist < 0.0) or (rsi > 75.0)
    news_bullish = avg_sentiment > 0.4
    if tech_bearish and news_bullish:
        return True, f"Bearish technicals (RSI={rsi:.1f}, MACD_hist={hist:.2f}) vs Bullish News (Sentiment={avg_sentiment:+.2f})"

    # Bullish technicals vs Bearish news
    tech_bullish = (rsi < 30.0 and hist > 0.0) or (rsi < 25.0)
    news_bearish = avg_sentiment < -0.4
    if tech_bullish and news_bearish:
        return True, f"Bullish technicals (RSI={rsi:.1f}, MACD_hist={hist:.2f}) vs Bearish News (Sentiment={avg_sentiment:+.2f})"

    return False, None


class MarketDataFetcher:
    """Fetches or simulates live financial ticks, order book momentum, and news feeds."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.alpaca_key = os.environ.get("APCA_API_KEY_ID", "")
        self.alpaca_secret = os.environ.get("APCA_API_SECRET_KEY", "")
        self.alpaca_data_base = "https://data.alpaca.markets"
        self._price_history: Dict[str, List[float]] = {}
        self._volume_history: Dict[str, List[float]] = {}
        self._init_mock_histories()

    def _alpaca_headers(self) -> Dict[str, str]:
        return {
            "APCA-API-KEY-ID": self.alpaca_key.strip(),
            "APCA-API-SECRET-KEY": self.alpaca_secret.strip(),
        }

    def fetch_alpaca_bars(
        self,
        symbol: str,
        start_iso: str,
        end_iso: str,
        timeframe: str = "30Min",
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """Fetch historical OHLCV bars from Alpaca Market Data API."""
        if not self.alpaca_key or not self.alpaca_secret:
            raise RuntimeError("Missing APCA_API_KEY_ID or APCA_API_SECRET_KEY for Alpaca bars.")

        is_crypto = "/" in symbol
        if is_crypto:
            url = f"{self.alpaca_data_base}/v1beta3/crypto/us/bars"
            params: Dict[str, Any] = {
                "symbols": symbol,
                "timeframe": timeframe,
                "start": start_iso,
                "end": end_iso,
                "limit": limit,
            }
        else:
            url = f"{self.alpaca_data_base}/v2/stocks/bars"
            params = {
                "symbols": symbol,
                "timeframe": timeframe,
                "start": start_iso,
                "end": end_iso,
                "limit": limit,
                "feed": "iex",
                "adjustment": "raw",
            }

        bars: List[Dict[str, Any]] = []
        page_token: Optional[str] = None
        with httpx.Client(timeout=20.0, headers=self._alpaca_headers()) as client:
            while True:
                req_params = dict(params)
                if page_token:
                    req_params["page_token"] = page_token
                resp = client.get(url, params=req_params)
                resp.raise_for_status()
                payload = resp.json()
                bars.extend(payload.get("bars", {}).get(symbol, []))
                page_token = payload.get("next_page_token")
                if not page_token or len(bars) >= 5000:
                    break
        return bars

    def fetch_alpaca_news(
        self,
        symbols: List[str],
        start_iso: str,
        end_iso: str,
        limit: int = 50,
        max_pages: int = 20,
    ) -> List[Dict[str, Any]]:
        """Fetch historical breaking financial news articles from Alpaca News API."""
        if not self.alpaca_key or not self.alpaca_secret:
            raise RuntimeError("Missing APCA_API_KEY_ID or APCA_API_SECRET_KEY for Alpaca news.")

        api_syms = ",".join(s.replace("/", "") for s in symbols)
        url = f"{self.alpaca_data_base}/v1beta1/news"
        articles: List[Dict[str, Any]] = []
        page_token: Optional[str] = None
        with httpx.Client(timeout=20.0, headers=self._alpaca_headers()) as client:
            for _ in range(max_pages):
                params: Dict[str, Any] = {
                    "symbols": api_syms,
                    "start": start_iso,
                    "end": end_iso,
                    "limit": limit,
                    "sort": "asc",
                    "include_content": "false",
                }
                if page_token:
                    params["page_token"] = page_token
                resp = client.get(url, params=params)
                resp.raise_for_status()
                payload = resp.json()
                articles.extend(payload.get("news", []))
                page_token = payload.get("next_page_token")
                if not page_token:
                    break
        return articles

    def _init_mock_histories(self) -> None:
        """Initialize rolling price histories for supported symbols."""
        base_prices = {
            "BTC/USD": 68500.0,
            "ETH/USD": 3550.0,
            "NVDA": 138.5,
            "TSLA": 252.0,
        }
        np.random.seed(42)
        for sym, p in base_prices.items():
            noise = np.random.normal(0, 0.005, size=60)
            cum_returns = np.cumprod(1 + noise)
            self._price_history[sym] = list(p * cum_returns)
            self._volume_history[sym] = list(np.random.uniform(1000, 50000, size=60))

    def compute_indicators(self, symbol: str) -> TechnicalIndicators:
        """Calculate technical indicators from rolling price and volume arrays."""
        prices = np.array(self._price_history.get(symbol, [100.0] * 30), dtype=float)
        volumes = np.array(self._volume_history.get(symbol, [1000.0] * 30), dtype=float)

        rsi = compute_rsi(prices, 14)
        macd_val, macd_sig, macd_hist = compute_macd(prices)
        pct_b, _, _, _ = compute_bollinger(prices, 20)
        vol_24h = compute_volatility_24h(prices)
        vol_ratio = compute_volume_ratio(volumes, 20)

        trend = "NEUTRAL"
        if rsi > 60 and macd_hist > 0:
            trend = "BULLISH"
        elif rsi < 40 and macd_hist < 0:
            trend = "BEARISH"

        return TechnicalIndicators(
            rsi_14=round(rsi, 2),
            macd_val=round(macd_val, 4),
            macd_signal=round(macd_sig, 4),
            macd_hist=round(macd_hist, 4),
            bollinger_pct_b=round(pct_b, 4),
            volatility_24h=round(vol_24h, 5),
            volume_ratio_20=round(vol_ratio, 2),
            trend_regime=trend,
        )

    def fetch_market_snapshot(self, symbol: str) -> MarketSnapshot:
        """Generate current market snapshot for an asset."""
        prices = self._price_history.get(symbol, [100.0])
        volumes = self._volume_history.get(symbol, [1000.0])
        cur_price = prices[-1]
        p_24h_ago = prices[0] if len(prices) > 48 else prices[0]
        change_pct = ((cur_price - p_24h_ago) / p_24h_ago) * 100.0

        indicators = self.compute_indicators(symbol)
        now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        return MarketSnapshot(
            symbol=symbol,
            timestamp=now_iso,
            current_price=round(cur_price, 2 if cur_price > 10 else 4),
            price_change_24h_pct=round(change_pct, 2),
            high_24h=round(float(np.max(prices[-48:])), 2),
            low_24h=round(float(np.min(prices[-48:])), 2),
            volume_24h=round(float(np.sum(volumes[-48:])), 2),
            indicators=indicators,
        )

    def fetch_latest_news(self, symbol: str) -> List[NewsArticle]:
        """Fetch or mock breaking financial news headlines."""
        now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        news_templates = {
            "BTC/USD": [
                ("ETF inflows accelerate as institutional asset managers expand allocations", 0.65),
                ("Derivatives open interest hits new record high amid heightened leverage warnings", -0.20),
            ],
            "ETH/USD": [
                ("Layer-2 transaction throughput surges 40% following protocol optimization", 0.55),
                ("Validator staking yield stabilizes as deflationary burn rate increases", 0.35),
            ],
            "NVDA": [
                ("Datacenter compute demand outpaces supply forecasts ahead of next-gen architecture rollout", 0.80),
                ("Supply chain lead times remain tight for cutting-edge packaging capacity", 0.10),
            ],
            "TSLA": [
                ("Autonomous driving compute milestone achieved in latest fleet telemetry deployment", 0.70),
                ("Regulatory review opens on driver assist feature performance in adverse weather", -0.45),
            ],
        }

        articles = []
        templates = news_templates.get(symbol, [("Market participants await key macroeconomic data releases", 0.0)])
        for i, (hd, sent) in enumerate(templates):
            n_id = f"news_{symbol.replace('/', '_')}_{int(time.time())}_{i}"
            articles.append(
                NewsArticle(
                    news_id=n_id,
                    headline=hd,
                    summary=f"Automated wire feed summary for {symbol}: {hd}.",
                    source="FinancialWire",
                    published_at=now_iso,
                    sentiment_score=sent,
                    symbols=[symbol],
                )
            )
        return articles

    def poll_tick(self, symbol: str) -> StreamTick:
        """Poll a complete stream tick containing market snapshot, news, and conflict check."""
        # Advance mock price with small delta
        cur_p = self._price_history[symbol][-1]
        delta = cur_p * np.random.normal(0, 0.003)
        new_p = max(cur_p + delta, 0.01)
        self._price_history[symbol].append(new_p)
        if len(self._price_history[symbol]) > 200:
            self._price_history[symbol].pop(0)

        vol = np.random.uniform(5000, 30000)
        self._volume_history[symbol].append(vol)
        if len(self._volume_history[symbol]) > 200:
            self._volume_history[symbol].pop(0)

        snapshot = self.fetch_market_snapshot(symbol)
        news = self.fetch_latest_news(symbol)
        has_conflict, reason = detect_signal_conflict(snapshot.indicators, news)

        now_str = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        tick_id = f"tick_{symbol.replace('/', '_')}_{int(time.time())}"

        return StreamTick(
            tick_id=tick_id,
            symbol=symbol,
            timestamp=now_str,
            snapshot=snapshot,
            news_stream=news,
            signal_conflict_flag=has_conflict,
            conflict_reason=reason,
        )

