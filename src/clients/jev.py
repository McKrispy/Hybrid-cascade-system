"""Jev API client for System-1 probability decision models.

Timestamp: 2026-11-09 11:32 (Asia/Shanghai)

Supports jev-1.14.0 via POST https://api.typesafe.ai/v1/systemone with Bearer authentication.
Adheres strictly to the official TypeSafe System-1 wire schema:
Request: {model, state, questions: {question_id: {type, instructions, criteria}}}
Response: {model, answers: {question_id: {choice/noul, probabilities}}, usage: {input_tokens, output_tokens}}
Enforces probability legality, model version consistency, exact usage parsing, and zero guessing.
"""

import hashlib
import json
import math
import os
import time
from typing import Any, Dict, List, Optional
import httpx

from src.clients.base import Prediction, desensitize_dict
from src.infra.config import get_env_var


class JevClient:
    """Client for TypeSafe Jev System-1 native HTTP API."""

    DEFAULT_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
    DEFAULT_MODEL = "jev-1.14.0"
    INPUT_COST_PER_MILLION = 0.042  # $0.042 / 1M input tokens
    OUTPUT_COST_PER_MILLION = 0.0   # Output tokens free

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        model_id: str = DEFAULT_MODEL,
        timeout: float = 30.0,
        max_retries: int = 3,
        backoff_factor: float = 1.5,
        mock_mode: bool = False,
    ):
        self.api_key = api_key or get_env_var("TYPESAFE_API_KEY", "")
        self.endpoint = endpoint or get_env_var("TYPESAFE_BASE_URL", self.DEFAULT_ENDPOINT)
        self.model_id = model_id
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.mock_mode = mock_mode

    def calculate_cost(self, input_tokens: int, output_tokens: int = 0) -> float:
        """Calculate USD cost strictly based on input token usage ($0.042/1M)."""
        return (input_tokens / 1_000_000.0) * self.INPUT_COST_PER_MILLION

    def parse_native_response(
        self,
        data: Dict[str, Any],
        sample_id: str,
        task_type: str,
        options: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Validate and parse TypeSafe native API response payload.
        
        Returns a dictionary with parsed fields and validation status.
        """
        returned_model = data.get("model", "")
        if not returned_model or not returned_model.startswith(self.model_id):
            return {
                "status": "failed",
                "error_code": "UNEXPECTED_MODEL_VERSION",
                "returned_model": returned_model or "none",
                "decision": None,
                "probabilities": {},
                "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                "cost": 0.0,
                "reason": f"Expected model '{self.model_id}', got '{returned_model}'",
            }

        usage_raw = data.get("usage", {})
        input_tokens = usage_raw.get("input_tokens", usage_raw.get("prompt_tokens"))
        output_tokens = usage_raw.get("output_tokens", usage_raw.get("completion_tokens", 0))

        if input_tokens is None:
            # Native usage is missing - do not guess by word count
            input_tokens = 0
            has_valid_usage = False
        else:
            input_tokens = int(input_tokens)
            output_tokens = int(output_tokens) if output_tokens is not None else 0
            has_valid_usage = True

        total_tokens = input_tokens + output_tokens
        cost = self.calculate_cost(input_tokens, output_tokens)

        answers = data.get("answers", {})
        if not isinstance(answers, dict) or sample_id not in answers:
            return {
                "status": "failed",
                "error_code": "MISSING_SAMPLE_ANSWER",
                "returned_model": returned_model,
                "decision": None,
                "probabilities": {},
                "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": total_tokens},
                "cost": cost,
                "reason": f"Sample ID '{sample_id}' not found in answers mapping",
            }

        ans_item = answers[sample_id]

        if task_type == "noul":
            # Extract p_yes
            p_val = None
            if isinstance(ans_item, (int, float)):
                p_val = float(ans_item)
            elif isinstance(ans_item, dict):
                if "noul" in ans_item:
                    noul_content = ans_item["noul"]
                    if isinstance(noul_content, (int, float)):
                        p_val = float(noul_content)
                    elif isinstance(noul_content, dict):
                        p_val = noul_content.get("p_yes", noul_content.get("probability"))
                elif "p_yes" in ans_item:
                    p_val = ans_item["p_yes"]
                elif "probability" in ans_item:
                    p_val = ans_item["probability"]

            if p_val is None or not isinstance(p_val, (int, float)) or math.isnan(p_val):
                return {
                    "status": "failed",
                    "error_code": "INVALID_NOUL_PROBABILITY",
                    "returned_model": returned_model,
                    "decision": None,
                    "probabilities": {},
                    "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": total_tokens},
                    "cost": cost,
                    "reason": f"Cannot parse valid float probability for noul from {ans_item}",
                }

            p_yes = float(p_val)
            if p_yes < 0.0 or p_yes > 1.0:
                return {
                    "status": "failed",
                    "error_code": "PROBABILITY_OUT_OF_BOUNDS",
                    "returned_model": returned_model,
                    "decision": None,
                    "probabilities": {"raw_p_yes": p_yes},
                    "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": total_tokens},
                    "cost": cost,
                    "reason": f"p_yes={p_yes} is outside [0.0, 1.0]",
                }

            probs = {"no": round(1.0 - p_yes, 6), "yes": round(p_yes, 6)}
            decision = "yes" if p_yes >= 0.5 else "no"

            return {
                "status": "valid",
                "error_code": None,
                "returned_model": returned_model,
                "decision": decision,
                "probabilities": probs,
                "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": total_tokens},
                "cost": cost,
                "has_valid_usage": has_valid_usage,
            }

        else:
            # Choice primitive
            probs = {}
            decision = None
            if isinstance(ans_item, dict):
                if "choice" in ans_item:
                    choice_field = ans_item["choice"]
                    if isinstance(choice_field, dict):
                        probs = choice_field.get("probabilities", {})
                        decision = choice_field.get("decision")
                    elif isinstance(choice_field, str):
                        decision = choice_field
                        probs = ans_item.get("probabilities", {})
                if not probs and "probabilities" in ans_item:
                    probs = ans_item.get("probabilities", {})
                if not decision and "decision" in ans_item:
                    decision = ans_item.get("decision")

            if not probs or not isinstance(probs, dict):
                return {
                    "status": "failed",
                    "error_code": "MISSING_CHOICE_PROBABILITIES",
                    "returned_model": returned_model,
                    "decision": None,
                    "probabilities": {},
                    "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": total_tokens},
                    "cost": cost,
                    "reason": f"Missing probabilities mapping in choice answer: {ans_item}",
                }

            # Validate each probability and sum
            sum_p = 0.0
            cleaned_probs = {}
            for opt, p in probs.items():
                if not isinstance(p, (int, float)) or math.isnan(p) or p < 0.0 or p > 1.0:
                    return {
                        "status": "failed",
                        "error_code": "PROBABILITY_OUT_OF_BOUNDS",
                        "returned_model": returned_model,
                        "decision": None,
                        "probabilities": probs,
                        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": total_tokens},
                        "cost": cost,
                        "reason": f"Probability for option '{opt}' is invalid: {p}",
                    }
                cleaned_probs[opt] = float(p)
                sum_p += float(p)

            # Check sum of probabilities (~1.0 within numerical tolerance)
            if abs(sum_p - 1.0) > 0.05:
                return {
                    "status": "failed",
                    "error_code": "PROBABILITIES_SUM_INVALID",
                    "returned_model": returned_model,
                    "decision": None,
                    "probabilities": cleaned_probs,
                    "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": total_tokens},
                    "cost": cost,
                    "reason": f"Probabilities sum to {sum_p:.4f}, expected ~1.0",
                }

            if not decision:
                decision = max(cleaned_probs, key=cleaned_probs.get)

            return {
                "status": "valid",
                "error_code": None,
                "returned_model": returned_model,
                "decision": decision,
                "probabilities": cleaned_probs,
                "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": total_tokens},
                "cost": cost,
                "has_valid_usage": has_valid_usage,
            }

    def _generate_mock_prediction(
        self,
        sample_id: str,
        request_id: str,
        role: str,
        task_type: str,
        question: str,
        options: Optional[List[str]] = None,
        dataset_id: Optional[str] = None,
    ) -> Prediction:
        """Deterministic mock response following native TypeSafe System-1 format."""
        h = int(hashlib.md5(f"{sample_id}:{question}".encode("utf-8")).hexdigest(), 16)
        latency_ms = 45.0 + (h % 30)

        if task_type == "noul":
            p_yes = round(0.1 + (h % 800) / 1000.0, 4)
            p_no = round(1.0 - p_yes, 4)
            probs = {"no": p_no, "yes": p_yes}
            decision = "yes" if p_yes >= 0.5 else "no"
            mock_native_answer = {"noul": p_yes}
        else:
            opts = options or ["option_A", "option_B"]
            K = len(opts)
            raw_weights = [1.0 / (i + 1) ** 1.2 for i in range(K)]
            shift = h % K
            raw_weights = raw_weights[shift:] + raw_weights[:shift]
            total_w = sum(raw_weights)
            probs = {opt: round(w / total_w, 4) for opt, w in zip(opts, raw_weights)}
            sum_p = sum(probs.values())
            first_key = list(probs.keys())[0]
            probs[first_key] = round(probs[first_key] + (1.0 - sum_p), 4)
            decision = max(probs, key=probs.get)
            mock_native_answer = {"choice": {"decision": decision, "probabilities": probs}}

        simulated_input_tokens = max(10, len(question.split()) * 2 + (len(options or []) * 10))
        simulated_cost = self.calculate_cost(simulated_input_tokens, 0)

        return Prediction(
            sample_id=sample_id,
            dataset_id=dataset_id,
            request_id=request_id,
            role=role,
            model_id=self.model_id,
            returned_model=f"{self.model_id}-mock",
            arm_id="JEV",
            template_version="v1.0",
            decision=decision,
            probabilities=probs,
            # In mock mode, actual tokens and cost are 0, mock simulation recorded separately
            usage={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "simulated_input_tokens": simulated_input_tokens},
            cost=0.0,
            latency_ms=latency_ms,
            attempts=1,
            status="valid",
            error_code=None,
            raw_response={
                "mock": True,
                "execution_mode": "mock",
                "simulated_cost": simulated_cost,
                "native_payload": {
                    "model": f"{self.model_id}-mock",
                    "answers": {sample_id: mock_native_answer},
                    "usage": {"input_tokens": simulated_input_tokens, "output_tokens": 0},
                },
            },
        )

    def decide(
        self,
        sample_id: str,
        request_id: str,
        role: str,
        task_type: str,
        question: str,
        options: Optional[List[str]] = None,
        criteria: Optional[Dict[str, str]] = None,
        dataset_id: Optional[str] = None,
    ) -> Prediction:
        """Execute request to Jev System-1 API following official native schema."""
        if self.mock_mode:
            return self._generate_mock_prediction(sample_id, request_id, role, task_type, question, options, dataset_id=dataset_id)

        if not self.api_key:
            return Prediction(
                sample_id=sample_id,
                dataset_id=dataset_id,
                request_id=request_id,
                role=role,
                model_id=self.model_id,
                returned_model="none",
                arm_id="JEV",
                template_version="v1.0",
                decision=None,
                probabilities={},
                usage={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                cost=0.0,
                latency_ms=0.0,
                attempts=0,
                status="failed",
                error_code="MISSING_API_KEY",
                raw_response={"error": "TYPESAFE_API_KEY environment variable is not set"},
            )

        # Official TypeSafe System-1 native request wire schema:
        # {model, state: {}, questions: {question_id: {type, instructions, criteria}}}
        q_spec: Dict[str, Any] = {
            "type": task_type,
            "instructions": question,
        }

        if task_type == "choice":
            # Criteria mapping: option label -> criteria description
            crit_dict = {}
            if criteria:
                crit_dict = dict(criteria)
            elif options:
                crit_dict = {opt: f"Classification category: {opt}" for opt in options}
            q_spec["criteria"] = crit_dict
        elif task_type == "noul" and criteria:
            q_spec["criteria"] = criteria

        payload: Dict[str, Any] = {
            "model": self.model_id,
            "state": {},
            "questions": {
                sample_id: q_spec,
            },
        }

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
                        parsed = self.parse_native_response(data, sample_id, task_type, options)

                        return Prediction(
                            sample_id=sample_id,
                            dataset_id=dataset_id,
                            request_id=request_id,
                            role=role,
                            model_id=self.model_id,
                            returned_model=parsed["returned_model"],
                            arm_id="JEV",
                            template_version="v1.0",
                            decision=parsed["decision"],
                            probabilities=parsed["probabilities"],
                            usage=parsed["usage"],
                            cost=parsed["cost"],
                            latency_ms=latency_ms,
                            attempts=attempts,
                            status=parsed["status"],
                            error_code=parsed.get("error_code"),
                            raw_response=desensitize_dict(data),
                        )
                    elif resp.status_code in (429, 500, 502, 503, 504):
                        if attempts > self.max_retries:
                            break
                        # Respect Retry-After if provided
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
                            arm_id="JEV",
                            template_version="v1.0",
                            decision=None,
                            probabilities={},
                            usage={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
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
                        arm_id="JEV",
                        template_version="v1.0",
                        decision=None,
                        probabilities={},
                        usage={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
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
            arm_id="JEV",
            template_version="v1.0",
            decision=None,
            probabilities={},
            usage={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            cost=0.0,
            latency_ms=0.0,
            attempts=attempts,
            status="failed",
            error_code="MAX_RETRIES_EXCEEDED",
            raw_response=None,
        )
