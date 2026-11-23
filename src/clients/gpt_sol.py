"""Client implementation for gpt-6.1-sol backend.

Timestamp: 2026-11-09 11:32 (Asia/Shanghai)

Implements System-2 client for gpt-6.1-sol via standard OpenAI-compatible API.
Reads key strictly from OPENAI_API_KEY environment variable.
Enforces strict JSON decision parsing, empty probabilities, and generation parameters pass-through.
"""

import hashlib
import json
import os
import time
from typing import Any, Dict, Optional
import httpx

from src.clients.base import Handoff, Prediction, desensitize_dict
from src.clients.system2 import System2Client, build_system2_prompt, parse_system2_decision
from src.infra.config import get_env_var


class GptSolClient(System2Client):
    """Client for gpt-6.1-sol System-2 model."""

    DEFAULT_MODEL = "gpt-6.1-sol"
    DEFAULT_ENDPOINT = "https://api.openai.com/v1/chat/completions"
    DEFAULT_INPUT_COST = 2.50   # USD per 1M tokens (OpenAI platform billing rate)
    DEFAULT_OUTPUT_COST = 10.00  # USD per 1M tokens (OpenAI platform billing rate, includes reasoning tokens)

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        model_id: str = DEFAULT_MODEL,
        input_cost_per_million: float = DEFAULT_INPUT_COST,
        output_cost_per_million: float = DEFAULT_OUTPUT_COST,
        timeout: float = 30.0,
        max_retries: int = 3,
        backoff_factor: float = 1.5,
        parameters: Optional[Dict[str, Any]] = None,
        mock_mode: bool = False,
    ):
        resolved_endpoint = endpoint or get_env_var("OPENAI_BASE_URL") or self.DEFAULT_ENDPOINT
        if resolved_endpoint.endswith("/v1") or resolved_endpoint.endswith("/v1/"):
            resolved_endpoint = resolved_endpoint.rstrip("/") + "/chat/completions"

        super().__init__(
            model_id=model_id,
            env_key_name="OPENAI_API_KEY",
            endpoint=resolved_endpoint,
            input_cost_per_million=input_cost_per_million,
            output_cost_per_million=output_cost_per_million,
            timeout=timeout,
            max_retries=max_retries,
            backoff_factor=backoff_factor,
            parameters=parameters,
            mock_mode=mock_mode,
        )
        if api_key:
            self.api_key = api_key

    def _generate_mock_prediction(
        self,
        sample_id: str,
        request_id: str,
        role: str,
        handoff: Handoff,
        question: str,
        dataset_id: Optional[str] = None,
    ) -> Prediction:
        """Deterministic mock response for offline testing."""
        h = int(hashlib.md5(f"{sample_id}:{self.model_id}:{handoff.arm_id}".encode("utf-8")).hexdigest(), 16)
        opts = list(handoff.allowed_labels)
        if handoff.reject_allowed:
            opts.append("REJECT")
        decision = opts[h % len(opts)]

        simulated_prompt_tokens = len(question.split()) * 2 + len(opts) * 5 + 40
        simulated_completion_tokens = 15
        simulated_cost = self.calculate_cost(simulated_prompt_tokens, simulated_completion_tokens)

        return Prediction(
            sample_id=sample_id,
            dataset_id=dataset_id,
            request_id=request_id,
            role=role,
            model_id=self.model_id,
            returned_model=f"{self.model_id}-mock",
            arm_id=handoff.arm_id,
            template_version="v1.0",
            decision=decision,
            probabilities={},  # System-2 has no real probability distribution
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "simulated_total_tokens": simulated_prompt_tokens + simulated_completion_tokens},
            cost=0.0,
            latency_ms=85.0 + (h % 50),
            attempts=1,
            status="valid",
            error_code=None,
            raw_response={"mock": True, "execution_mode": "mock", "simulated_cost": simulated_cost},
        )

    def decide(
        self,
        sample_id: str,
        request_id: str,
        role: str,
        question: str,
        handoff: Handoff,
        label_descriptions: Optional[Dict[str, str]] = None,
        dataset_id: Optional[str] = None,
    ) -> Prediction:
        if self.mock_mode:
            return self._generate_mock_prediction(sample_id, request_id, role, handoff, question, dataset_id=dataset_id)

        if not self.api_key:
            return Prediction(
                sample_id=sample_id,
                dataset_id=dataset_id,
                request_id=request_id,
                role=role,
                model_id=self.model_id,
                returned_model="none",
                arm_id=handoff.arm_id,
                template_version="v1.0",
                decision=None,
                probabilities={},
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                cost=0.0,
                latency_ms=0.0,
                attempts=0,
                status="failed",
                error_code="MISSING_API_KEY",
                raw_response={"error": "OPENAI_API_KEY environment variable is not set"},
            )

        prompt = build_system2_prompt(question, handoff, label_descriptions)
        max_tokens = int(self.parameters.get("max_tokens", 200))

        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": "You are a precise decision classification agent. You must output only JSON."},
                {"role": "user", "content": prompt},
            ],
            "max_completion_tokens": max_tokens,
        }
        # gpt-6.1-sol enforces default temperature (1.0). Only pass temperature if explicitly > 0.0 and != 1.0
        temp = self.parameters.get("temperature")
        if temp is not None and float(temp) > 0.0 and float(temp) != 1.0:
            payload["temperature"] = float(temp)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        attempts = 0
        backoff = 1.0
        start_time = time.perf_counter()

        while attempts <= self.max_retries:
            attempts += 1
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.post(self.endpoint, json=payload, headers=headers)
                    latency_ms = (time.perf_counter() - start_time) * 1000.0

                    if resp.status_code == 200:
                        data = resp.json()
                        returned_model = data.get("model", self.model_id)
                        usage = data.get("usage", {})
                        prompt_tokens = int(usage.get("prompt_tokens", 0))
                        completion_tokens = int(usage.get("completion_tokens", 0))
                        total_tokens = prompt_tokens + completion_tokens
                        cost = self.calculate_cost(prompt_tokens, completion_tokens)

                        content = ""
                        choices = data.get("choices", [])
                        if choices:
                            content = choices[0].get("message", {}).get("content", "")

                        decision = parse_system2_decision(
                            content,
                            handoff.allowed_labels,
                            handoff.reject_allowed,
                        )
                        status = "valid" if decision is not None else "failed"
                        error_code = None if decision is not None else "INVALID_OUTPUT_LABEL"

                        return Prediction(
                            sample_id=sample_id,
                            dataset_id=dataset_id,
                            request_id=request_id,
                            role=role,
                            model_id=self.model_id,
                            returned_model=returned_model,
                            arm_id=handoff.arm_id,
                            template_version="v1.0",
                            decision=decision,
                            probabilities={},  # System-2 has no real probability distribution
                            usage={"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": total_tokens},
                            cost=cost,
                            latency_ms=latency_ms,
                            attempts=attempts,
                            status=status,
                            error_code=error_code,
                            raw_response=desensitize_dict(data),
                        )
                    elif resp.status_code in (429, 500, 502, 503, 504):
                        if attempts > self.max_retries:
                            break
                        retry_after = resp.headers.get("Retry-After")
                        sleep_time = float(retry_after) if retry_after and retry_after.isdigit() else backoff
                        time.sleep(sleep_time)
                        backoff *= self.backoff_factor
                        continue
                    else:
                        latency_ms = (time.perf_counter() - start_time) * 1000.0
                        return Prediction(
                            sample_id=sample_id,
                            dataset_id=dataset_id,
                            request_id=request_id,
                            role=role,
                            model_id=self.model_id,
                            returned_model="none",
                            arm_id=handoff.arm_id,
                            template_version="v1.0",
                            decision=None,
                            probabilities={},
                            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                            cost=0.0,
                            latency_ms=latency_ms,
                            attempts=attempts,
                            status="failed",
                            error_code=f"HTTP_{resp.status_code}",
                            raw_response={"error": resp.text[:500]},
                        )
            except Exception as e:
                if attempts > self.max_retries:
                    latency_ms = (time.perf_counter() - start_time) * 1000.0
                    return Prediction(
                        sample_id=sample_id,
                        dataset_id=dataset_id,
                        request_id=request_id,
                        role=role,
                        model_id=self.model_id,
                        returned_model="none",
                        arm_id=handoff.arm_id,
                        template_version="v1.0",
                        decision=None,
                        probabilities={},
                        usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                        cost=0.0,
                        latency_ms=latency_ms,
                        attempts=attempts,
                        status="failed",
                        error_code=type(e).__name__,
                        raw_response={"error": str(e)},
                    )
                time.sleep(backoff)
                backoff *= self.backoff_factor

        return Prediction(
            sample_id=sample_id,
            dataset_id=dataset_id,
            request_id=request_id,
            role=role,
            model_id=self.model_id,
            returned_model="none",
            arm_id=handoff.arm_id,
            template_version="v1.0",
            decision=None,
            probabilities={},
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            cost=0.0,
            latency_ms=0.0,
            attempts=attempts,
            status="failed",
            error_code="MAX_RETRIES_EXCEEDED",
            raw_response=None,
        )
