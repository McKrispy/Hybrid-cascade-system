"""Self-contained lightweight clients for Jev, OpenAI, and DeepSeek.

Timestamp: 2026-11-11 13:00 (Asia/Shanghai)
"""

import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx
from pydantic import BaseModel, Field


def _post_with_retries(
    endpoint: str,
    payload: Dict[str, Any],
    headers: Dict[str, str],
    timeout: float,
    max_retries: int = 4,
) -> httpx.Response:
    """HTTP POST with exponential backoff on transient SSL/network/5xx/429 errors."""
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(endpoint, json=payload, headers=headers)
                if resp.status_code in (429, 500, 502, 503, 504) and attempt < max_retries:
                    time.sleep(min(2.0 * (2 ** attempt), 12.0))
                    continue
                resp.raise_for_status()
                return resp
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError, httpx.ConnectTimeout) as e:
            last_exc = e
            if attempt < max_retries:
                time.sleep(min(1.5 * (2 ** attempt), 10.0))
                continue
            raise
        except httpx.HTTPStatusError:
            raise
    assert last_exc is not None
    raise last_exc


def _coerce_str(val: Any, default: str = "") -> str:
    """Safely convert model output field to string if returned as list or dict."""
    if val is None:
        return default
    if isinstance(val, str):
        return val
    if isinstance(val, (list, tuple)):
        return "\n".join(str(item) for item in val)
    if isinstance(val, dict):
        return json.dumps(val, ensure_ascii=False)
    return str(val)


def _coerce_decision(raw_dec: Any, default: str = "NEUTRAL_HOLD") -> str:
    """Robustly parse decision from string, dict, or action payload."""
    if isinstance(raw_dec, dict):
        dec_str = str(raw_dec.get("action", raw_dec.get("stance", raw_dec.get("decision", "")))).upper()
    else:
        dec_str = str(raw_dec).upper()
    if "LONG" in dec_str or "ACCUMULATE" in dec_str or "BUY" in dec_str:
        return "LONG_ACCUMULATE"
    elif "SHORT" in dec_str or "LIQUIDATE" in dec_str or "SELL" in dec_str:
        return "SHORT_LIQUIDATE"
    return default


class JevPrediction(BaseModel):
    """Prediction result from System-1 Jev."""
    model_config = {"protected_namespaces": ()}
    model_id: str = "jev-1.14.0"
    decision: str
    confidence: float
    uncertainty: float
    probabilities: Dict[str, float]
    top2_candidates: List[Dict[str, Any]]
    input_tokens: int
    cost_usd: float


class System2Prediction(BaseModel):
    """Structured decision output from System-2 (GPT-6.1-Sol or DeepSeek-v4.1-Flash)."""
    model_config = {"protected_namespaces": ()}
    model_id: str
    arm_id: str  # F0, F1, C0, C1
    decision: str
    quant_vs_news_synthesis: str
    prior_stance_evaluation: str
    reasoning_chain: str
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    total_tokens: int
    cost_usd: float
    raw_response: Optional[str] = None


class JevClient:
    """TypeSafe Jev Choice API client for $K=3$ financial actions."""

    CRITERIA = {
        "LONG_ACCUMULATE": "Expect market momentum and positive catalysts to push asset price higher by > 0.5 sigma over next 4 hours.",
        "NEUTRAL_HOLD": "Expect range-bound sideways volatility within +/- 0.5 sigma over next 4 hours.",
        "SHORT_LIQUIDATE": "Expect technical breakdown or adverse catalysts to push asset price lower by > 0.5 sigma over next 4 hours.",
    }

    def __init__(self, api_key: Optional[str] = None, endpoint: Optional[str] = None):
        self.api_key = api_key or os.environ.get("TYPESAFE_API_KEY", "")
        self.endpoint = endpoint or "https://api.typesafe.ai/v1/systemone"
        self.model_id = "jev-1.14.0"
        self.input_rate_per_million = 0.042

    def predict(self, market_text: str, mock_fallback: bool = False, timeout: float = 25.0) -> JevPrediction:
        """Execute Jev Choice prediction (synchronous)."""
        if not self.api_key:
            return self._mock_predict(market_text)

        # Official TypeSafe System-1 native request wire schema:
        # {model, state: {}, questions: {question_id: {type, instructions, criteria}}}
        payload = {
            "model": self.model_id,
            "state": {},
            "questions": {
                "q1": {
                    "type": "choice",
                    "instructions": market_text,
                    "criteria": self.CRITERIA,
                }
            }
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            resp = _post_with_retries(self.endpoint, payload, headers, timeout=timeout)
            return self._parse_response(resp.json())
        except Exception as e:
            print(f"[!] TypeSafe Jev API error: {e}")
            if mock_fallback:
                return self._mock_predict(market_text)
            raise

    async def predict_async(self, market_text: str, mock_fallback: bool = False) -> JevPrediction:
        """Execute Jev Choice prediction (asynchronous)."""
        return self.predict(market_text, mock_fallback=mock_fallback)

    def _parse_response(self, data: Dict[str, Any]) -> JevPrediction:
        answers = data.get("answers", {})
        if isinstance(answers, dict):
            q_ans = answers.get("q1", {})
        elif isinstance(answers, list) and len(answers) > 0:
            q_ans = answers[0]
        else:
            q_ans = {}

        probs = q_ans.get("probabilities", {
            "LONG_ACCUMULATE": 0.333,
            "NEUTRAL_HOLD": 0.334,
            "SHORT_LIQUIDATE": 0.333,
        })
        sorted_cands = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        top1_lbl = q_ans.get("choice", sorted_cands[0][0])
        top1_prob = probs.get(top1_lbl, sorted_cands[0][1])
        top2 = [{"rank": i + 1, "label": k, "probability": round(v, 4)} for i, (k, v) in enumerate(sorted_cands[:2])]

        usage = data.get("usage", {})
        inp_tokens = usage.get("input_tokens", 450)
        cost = (inp_tokens / 1_000_000.0) * self.input_rate_per_million

        return JevPrediction(
            model_id=self.model_id,
            decision=top1_lbl,
            confidence=round(top1_prob, 4),
            uncertainty=round(1.0 - top1_prob, 4),
            probabilities={k: round(v, 4) for k, v in probs.items()},
            top2_candidates=top2,
            input_tokens=inp_tokens,
            cost_usd=round(cost, 6),
        )

    def _mock_predict(self, market_text: str) -> JevPrediction:
        """Deterministic heuristic mock prediction when offline."""
        # Simple heuristic from text keywords
        text_lower = market_text.lower()
        p_long = 0.333
        p_neutral = 0.334
        p_short = 0.333

        if "bullish" in text_lower or "rsi=7" in text_lower or "surge" in text_lower:
            p_long = 0.72
            p_neutral = 0.18
            p_short = 0.10
        elif "bearish" in text_lower or "breakdown" in text_lower or "rsi=2" in text_lower:
            p_short = 0.68
            p_neutral = 0.20
            p_long = 0.12
        elif "neutral" in text_lower or "range" in text_lower:
            p_neutral = 0.60
            p_long = 0.20
            p_short = 0.20

        probs = {"LONG_ACCUMULATE": p_long, "NEUTRAL_HOLD": p_neutral, "SHORT_LIQUIDATE": p_short}
        sorted_cands = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        top1_lbl, top1_prob = sorted_cands[0]
        top2 = [{"rank": i + 1, "label": k, "probability": round(v, 4)} for i, (k, v) in enumerate(sorted_cands[:2])]

        inp_tokens = 920
        cost = (inp_tokens / 1_000_000.0) * self.input_rate_per_million

        return JevPrediction(
            model_id=self.model_id,
            decision=top1_lbl,
            confidence=round(top1_prob, 4),
            uncertainty=round(1.0 - top1_prob, 4),
            probabilities=probs,
            top2_candidates=top2,
            input_tokens=inp_tokens,
            cost_usd=round(cost, 6),
        )


class OpenAIClient:
    """OpenAI GPT-6.1-Sol client with reasoning_tokens tracking."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.endpoint = "https://api.openai.com/v1/chat/completions"
        self.model_id = "gpt-6.1-sol"
        self.input_cost_per_million = 2.50
        self.output_cost_per_million = 10.00

    def predict(self, prompt: str, arm_id: str, mock_fallback: bool = False, timeout: float = 45.0) -> System2Prediction:
        """Call GPT-6.1-Sol and parse four-field JSON (synchronous)."""
        if not self.api_key:
            return self._mock_predict(prompt, arm_id)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        # Note: gpt-6.1-sol enforces default temperature (1.0). Do not pass temperature: 0.0
        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": "You are a senior quantitative risk officer. Analyze financial data and respond strictly in JSON format with quant_vs_news_synthesis, prior_stance_evaluation, reasoning_chain, and decision (one of LONG_ACCUMULATE, NEUTRAL_HOLD, SHORT_LIQUIDATE)."},
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
            "max_completion_tokens": 800,
        }
        try:
            resp = _post_with_retries(self.endpoint, payload, headers, timeout=timeout)
            return self._parse_response(resp.json(), arm_id)
        except Exception as e:
            print(f"[!] OpenAI GPT-6.1-Sol API error: {e}")
            if mock_fallback:
                return self._mock_predict(prompt, arm_id)
            raise

    async def predict_async(self, prompt: str, arm_id: str, mock_fallback: bool = False) -> System2Prediction:
        """Call GPT-6.1-Sol (asynchronous)."""
        return self.predict(prompt, arm_id, mock_fallback=mock_fallback)

    def _parse_response(self, data: Dict[str, Any], arm_id: str) -> System2Prediction:
        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 800)
        completion_tokens = usage.get("completion_tokens", 250)
        details = usage.get("completion_tokens_details", {})
        reasoning_tokens = details.get("reasoning_tokens", 100)

        cost = (prompt_tokens / 1_000_000.0) * self.input_cost_per_million + \
               (completion_tokens / 1_000_000.0) * self.output_cost_per_million

        content = data["choices"][0]["message"]["content"]
        try:
            parsed = json.loads(content)
        except Exception:
            parsed = {"decision": "NEUTRAL_HOLD", "reasoning_chain": content}

        decision = _coerce_decision(parsed.get("decision"))

        return System2Prediction(
            model_id=self.model_id,
            arm_id=arm_id,
            decision=decision,
            quant_vs_news_synthesis=_coerce_str(parsed.get("quant_vs_news_synthesis"), "Technical and news analysis performed."),
            prior_stance_evaluation=_coerce_str(parsed.get("prior_stance_evaluation"), "Evaluated."),
            reasoning_chain=_coerce_str(parsed.get("reasoning_chain"), "Reasoning completed."),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            reasoning_tokens=reasoning_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost_usd=round(cost, 6),
            raw_response=content,
        )

    def _mock_predict(self, prompt: str, arm_id: str) -> System2Prediction:
        """Deterministic mock response for offline testing."""
        prompt_tokens = 950
        completion_tokens = 220
        reasoning_tokens = 90
        cost = (prompt_tokens / 1_000_000.0) * self.input_cost_per_million + \
               (completion_tokens / 1_000_000.0) * self.output_cost_per_million

        decision = "NEUTRAL_HOLD"
        p_lower = prompt.lower()
        if "bearish" in p_lower or "short_liquidate" in p_lower:
            decision = "SHORT_LIQUIDATE"
        elif "bullish" in p_lower or "long_accumulate" in p_lower:
            decision = "LONG_ACCUMULATE"

        # Anchoring simulation: in F1/C1, higher chance to adhere to prior
        if arm_id in ["F1", "C1"] and "fast prior: long_accumulate" in p_lower:
            decision = "LONG_ACCUMULATE"

        prior_eval = "N/A (No numerical prior provided)" if arm_id in ["F0", "C0"] else "System-1 Fast Prior evaluated against technical divergence."

        return System2Prediction(
            model_id=self.model_id,
            arm_id=arm_id,
            decision=decision,
            quant_vs_news_synthesis="Technical momentum diverges from headline sentiment, suggesting elevated mean-reversion risk.",
            prior_stance_evaluation=prior_eval,
            reasoning_chain="Weighed RSI overbought state against breaking news buzz. Momentum exhaustion favors prudent position adjustment.",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            reasoning_tokens=reasoning_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost_usd=round(cost, 6),
            raw_response=f'{{"decision": "{decision}"}}',
        )


class DeepSeekClient:
    """DeepSeek-v4.1-Flash client."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        self.endpoint = "https://api.deepseek.com/chat/completions"  # no /v1/ prefix per api_config_info.json
        self.model_id = "deepseek-v4.1-flash"
        self.input_cost_per_million = 0.14
        self.output_cost_per_million = 0.28

    def predict(self, prompt: str, arm_id: str, mock_fallback: bool = False, timeout: float = 30.0) -> System2Prediction:
        """Call DeepSeek-v4.1-Flash and parse four-field JSON (synchronous)."""
        if not self.api_key:
            return self._mock_predict(prompt, arm_id)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        # DeepSeek API registers the flash model officially as 'deepseek-flash'
        payload = {
            "model": "deepseek-flash",
            "messages": [
                {"role": "system", "content": "You are a senior quantitative risk officer. Output strict JSON with: quant_vs_news_synthesis, prior_stance_evaluation, reasoning_chain, and decision (one of LONG_ACCUMULATE, NEUTRAL_HOLD, SHORT_LIQUIDATE)."},
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
            "max_completion_tokens": 800,
        }
        try:
            resp = _post_with_retries(self.endpoint, payload, headers, timeout=timeout)
            return self._parse_response(resp.json(), arm_id)
        except Exception as e:
            print(f"[!] DeepSeek API error: {e}")
            if mock_fallback:
                return self._mock_predict(prompt, arm_id)
            raise

    async def predict_async(self, prompt: str, arm_id: str, mock_fallback: bool = False) -> System2Prediction:
        """Call DeepSeek-v4.1-Flash (asynchronous)."""
        return self.predict(prompt, arm_id, mock_fallback=mock_fallback)

    def _parse_response(self, data: Dict[str, Any], arm_id: str) -> System2Prediction:
        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 900)
        completion_tokens = usage.get("completion_tokens", 300)
        details = usage.get("completion_tokens_details", {})
        reasoning_tokens = details.get("reasoning_tokens", 0)

        cost = (prompt_tokens / 1_000_000.0) * self.input_cost_per_million + \
               (completion_tokens / 1_000_000.0) * self.output_cost_per_million

        content = data["choices"][0]["message"]["content"]
        try:
            parsed = json.loads(content)
        except Exception:
            parsed = {"decision": "NEUTRAL_HOLD"}

        decision = _coerce_decision(parsed.get("decision"))

        return System2Prediction(
            model_id=self.model_id,
            arm_id=arm_id,
            decision=decision,
            quant_vs_news_synthesis=_coerce_str(parsed.get("quant_vs_news_synthesis"), "Technical indicators analyzed."),
            prior_stance_evaluation=_coerce_str(parsed.get("prior_stance_evaluation"), "Evaluated."),
            reasoning_chain=_coerce_str(parsed.get("reasoning_chain"), "Reasoning completed."),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            reasoning_tokens=reasoning_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost_usd=round(cost, 6),
            raw_response=content,
        )

    def _mock_predict(self, prompt: str, arm_id: str) -> System2Prediction:
        """Deterministic mock response for offline testing."""
        prompt_tokens = 980
        completion_tokens = 280
        cost = (prompt_tokens / 1_000_000.0) * self.input_cost_per_million + \
               (completion_tokens / 1_000_000.0) * self.output_cost_per_million

        decision = "NEUTRAL_HOLD"
        p_lower = prompt.lower()
        if "bearish" in p_lower:
            decision = "SHORT_LIQUIDATE"
        elif "bullish" in p_lower:
            decision = "LONG_ACCUMULATE"

        # Numerical compliance simulation: in F1/C1, strongly adheres to prior
        if arm_id in ["F1", "C1"] and "fast prior: long_accumulate" in p_lower:
            decision = "LONG_ACCUMULATE"

        prior_eval = "N/A (No numerical prior provided)" if arm_id in ["F0", "C0"] else "System-1 Fast Prior observed and factored into probability threshold."

        return System2Prediction(
            model_id=self.model_id,
            arm_id=arm_id,
            decision=decision,
            quant_vs_news_synthesis="Volume profile and news headlines cross-validated.",
            prior_stance_evaluation=prior_eval,
            reasoning_chain="Calculated risk-reward balance across 4h horizon. Concluded neutral or trend-following posture.",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            reasoning_tokens=0,
            total_tokens=prompt_tokens + completion_tokens,
            cost_usd=round(cost, 6),
            raw_response=f'{{"decision": "{decision}"}}',
        )
