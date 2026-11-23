"""Batch inference runner with concurrency, rate limiting, and caching for HybridCascade.

Timestamp: 2026-11-09 13:58 (Asia/Shanghai)

Coordinates execution across datasets and roles, enforcing budget limits,
rate limits, disk-backed cache resumption, and strict desensitization.
"""

import concurrent.futures
import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from tqdm import tqdm

from src.clients.base import Handoff, Prediction
from src.clients.deepseek_flash import DeepSeekFlashClient
from src.clients.gpt_sol import GptSolClient
from src.clients.jev import JevClient
from src.handoff.builders import build_handoff
from src.inference.cache import InferenceCache, compute_request_fingerprint
from src.infra.artifacts import RunArtifacts
from src.infra.config import BatchConfig
from src.infra.telemetry import TelemetryTracker
from src.input.splits import SampleInput
from src.input.tasks import TaskLabelSchema


class BatchInferenceRunner:
    """Manages batch API execution with thread-pool concurrency and caching."""

    def __init__(
        self,
        batch_config: BatchConfig,
        run_artifacts: RunArtifacts,
        telemetry: TelemetryTracker,
        cache: Optional[InferenceCache] = None,
        mode: str = "live",
    ):
        self.batch_config = batch_config
        self.artifacts = run_artifacts
        self.telemetry = telemetry
        self.cache = cache or InferenceCache()
        self.mode = mode

        self.max_concurrency = batch_config.rate_limits.max_concurrency
        self.timeout = batch_config.rate_limits.timeout_seconds
        self.max_retries = batch_config.rate_limits.max_retries
        self.backoff_factor = batch_config.rate_limits.backoff_factor

        self.budget_max_cost = batch_config.budget_limits.max_cost_usd
        self.budget_max_requests = batch_config.budget_limits.max_requests

        self.actual_requests_dispatched = 0
        self.actual_cost_total = 0.0
        self.cache_hits = 0
        self.halt_reason: Optional[str] = None

    def _check_budget(self) -> bool:
        """Check if request or cost budget limits have been reached."""
        if self.actual_requests_dispatched >= self.budget_max_requests:
            self.halt_reason = "MAX_REQUESTS_REACHED"
            return False
        if self.actual_cost_total >= self.budget_max_cost:
            self.halt_reason = "MAX_COST_REACHED"
            return False
        return True

    def execute_jev_samples(
        self,
        samples: List[SampleInput],
        label_schemas: Dict[str, TaskLabelSchema],
        role: str,
    ) -> List[Prediction]:
        """Execute Jev System-1 inference over a list of samples."""
        jev_cfg = self.batch_config.models.get("jev-1.14.0")
        client = JevClient(
            endpoint=jev_cfg.endpoint if jev_cfg else None,
            timeout=self.timeout,
            max_retries=self.max_retries,
            backoff_factor=self.backoff_factor,
            mock_mode=(self.mode == "mock"),
        )

        predictions: List[Prediction] = []

        def worker(sample: SampleInput) -> Prediction:
            schema = label_schemas[sample.label_schema_id]
            task_type = sample.task_type
            
            # Format question text
            if sample.dataset_id in ("banking77", "clinc150"):
                q_text = sample.text_fields.get("text", "")
            elif sample.dataset_id == "anli_r2r3":
                premise = sample.text_fields.get("premise", "")
                hypo = sample.text_fields.get("hypothesis", "")
                q_text = f"Premise: {premise}\nHypothesis: {hypo}"
            elif sample.dataset_id == "strategyqa":
                q_text = sample.text_fields.get("question", "")
            else:
                q_text = str(sample.text_fields)

            options = schema.allowed_labels
            criteria = schema.label_descriptions if task_type == "choice" else None

            # Check cache
            fp = compute_request_fingerprint(
                dataset_id=sample.dataset_id,
                role=role,
                sample_id=sample.sample_id,
                model_id="jev-1.14.0",
                arm_id="JEV",
                prompt_payload={"q": q_text, "opts": options, "crit": criteria},
                generation_params={"mode": "default_text"},
                execution_mode=self.mode,
            )

            cached_pred = self.cache.get(fp) or self.cache.get_by_sample(sample.sample_id, "jev-1.14.0", "JEV", mode=self.mode)
            if cached_pred and cached_pred.status == "valid":
                self.cache_hits += 1
                return cached_pred, False

            # Budget gate check
            if not self._check_budget():
                return Prediction(
                    sample_id=sample.sample_id,
                    dataset_id=sample.dataset_id,
                    request_id=f"req_jev_{sample.sample_id}",
                    role=role,
                    model_id="jev-1.14.0",
                    returned_model="none",
                    arm_id="JEV",
                    status="failed",
                    error_code="BUDGET_HALTED",
                    raw_response={"reason": self.halt_reason},
                ), False

            req_id = f"req_jev_{sample.sample_id}_{int(time.time()*1000)}"
            pred = client.decide(
                sample_id=sample.sample_id,
                request_id=req_id,
                role=role,
                task_type=task_type,
                question=q_text,
                options=options,
                criteria=criteria,
                dataset_id=sample.dataset_id,
            )

            if pred.status == "valid":
                self.cache.put(fp, pred, mode=self.mode)
            return pred, True

        # Concurrency pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_concurrency) as executor:
            futures = [executor.submit(worker, s) for s in samples]
            for f in concurrent.futures.as_completed(futures):
                pred, is_fresh = f.result()
                if is_fresh:
                    self.actual_requests_dispatched += 1
                    self.actual_cost_total += pred.cost
                predictions.append(pred)

        predictions.sort(key=lambda p: p.sample_id)
        return predictions

    def execute_system2_f0_samples(
        self,
        samples: List[SampleInput],
        label_schemas: Dict[str, TaskLabelSchema],
        model_id: str,  # "gpt-6.1-sol" or "deepseek-v4.1-flash"
        role: str,
    ) -> List[Prediction]:
        """Execute System-2 F0 (Hard baseline, full candidate options, no prior)."""
        model_cfg = self.batch_config.models.get(model_id)
        if model_id == "gpt-6.1-sol":
            client = GptSolClient(
                endpoint=model_cfg.endpoint if model_cfg else None,
                timeout=self.timeout,
                max_retries=self.max_retries,
                backoff_factor=self.backoff_factor,
                parameters=model_cfg.parameters if model_cfg else None,
                mock_mode=(self.mode == "mock"),
            )
        elif model_id == "deepseek-v4.1-flash":
            client = DeepSeekFlashClient(
                endpoint=model_cfg.endpoint if model_cfg else None,
                timeout=self.timeout,
                max_retries=self.max_retries,
                backoff_factor=self.backoff_factor,
                parameters=model_cfg.parameters if model_cfg else None,
                mock_mode=(self.mode == "mock"),
            )
        else:
            raise ValueError(f"Unsupported System-2 model: {model_id}")

        predictions: List[Prediction] = []

        def worker(sample: SampleInput) -> Prediction:
            schema = label_schemas[sample.label_schema_id]

            if sample.dataset_id in ("banking77", "clinc150"):
                q_text = sample.text_fields.get("text", "")
            elif sample.dataset_id == "anli_r2r3":
                premise = sample.text_fields.get("premise", "")
                hypo = sample.text_fields.get("hypothesis", "")
                q_text = f"Premise: {premise}\nHypothesis: {hypo}"
            elif sample.dataset_id == "strategyqa":
                q_text = sample.text_fields.get("question", "")
            else:
                q_text = str(sample.text_fields)

            # Construct F0 Handoff: Full candidate options, no prior package
            handoff = Handoff(
                arm_id="F0",
                allowed_labels=schema.allowed_labels,
                prior_payload=None,
                reject_allowed=schema.reject_allowed,
            )

            # Check cache
            fp = compute_request_fingerprint(
                dataset_id=sample.dataset_id,
                role=role,
                sample_id=sample.sample_id,
                model_id=model_id,
                arm_id="F0",
                prompt_payload={"q": q_text, "arm": "F0", "opts": schema.allowed_labels},
                generation_params=model_cfg.parameters if model_cfg else {},
                execution_mode=self.mode,
            )

            cached_pred = self.cache.get(fp) or self.cache.get_by_sample(sample.sample_id, model_id, "F0", mode=self.mode)
            if cached_pred and cached_pred.status == "valid":
                self.cache_hits += 1
                return cached_pred, False

            if not self._check_budget():
                return Prediction(
                    sample_id=sample.sample_id,
                    dataset_id=sample.dataset_id,
                    request_id=f"req_{model_id}_{sample.sample_id}",
                    role=role,
                    model_id=model_id,
                    returned_model="none",
                    arm_id="F0",
                    status="failed",
                    error_code="BUDGET_HALTED",
                    raw_response={"reason": self.halt_reason},
                ), False

            req_id = f"req_{model_id}_{sample.sample_id}_{int(time.time()*1000)}"
            pred = client.decide(
                sample_id=sample.sample_id,
                request_id=req_id,
                role=role,
                question=q_text,
                handoff=handoff,
                label_descriptions=schema.label_descriptions,
                dataset_id=sample.dataset_id,
            )

            if pred.status == "valid":
                self.cache.put(fp, pred, mode=self.mode)
            return pred, True

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_concurrency) as executor:
            futures = [executor.submit(worker, s) for s in samples]
            for f in concurrent.futures.as_completed(futures):
                pred, is_fresh = f.result()
                if is_fresh:
                    self.actual_requests_dispatched += 1
                    self.actual_cost_total += pred.cost
                predictions.append(pred)

        predictions.sort(key=lambda p: p.sample_id)
        return predictions

    def execute_system2_handoff_samples(
        self,
        samples: List[SampleInput],
        jev_preds_map: Dict[str, Dict[str, Any]],
        label_schemas: Dict[str, TaskLabelSchema],
        model_id: str,
        arm_id: str,
        role: str,
        f0_preds_map: Optional[Dict[str, Dict[str, Any]]] = None,
        f1_preds_map: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> List[Prediction]:
        """Execute System-2 inference for experimental handoff arms (F1, C0, C1).
        
        Handles StrategyQA structural aliases (C0 -> F0, C1 -> F1) with zero API calls.
        Dispatches remaining requests through thread-pool concurrency and caching.
        """
        model_cfg = self.batch_config.models.get(model_id)
        if model_id == "gpt-6.1-sol":
            client = GptSolClient(
                endpoint=model_cfg.endpoint if model_cfg else None,
                timeout=self.timeout,
                max_retries=self.max_retries,
                backoff_factor=self.backoff_factor,
                parameters=model_cfg.parameters if model_cfg else None,
                mock_mode=(self.mode == "mock"),
            )
        elif model_id == "deepseek-v4.1-flash":
            client = DeepSeekFlashClient(
                endpoint=model_cfg.endpoint if model_cfg else None,
                timeout=self.timeout,
                max_retries=self.max_retries,
                backoff_factor=self.backoff_factor,
                parameters=model_cfg.parameters if model_cfg else None,
                mock_mode=(self.mode == "mock"),
            )
        else:
            raise ValueError(f"Unsupported System-2 model: {model_id}")

        predictions: List[Prediction] = []

        def worker(sample: SampleInput) -> Tuple[Prediction, bool]:
            schema = label_schemas[sample.label_schema_id]
            jev_pred = jev_preds_map.get(sample.sample_id, {})

            if sample.dataset_id in ("banking77", "clinc150"):
                q_text = sample.text_fields.get("text", "")
            elif sample.dataset_id == "anli_r2r3":
                premise = sample.text_fields.get("premise", "")
                hypo = sample.text_fields.get("hypothesis", "")
                q_text = f"Premise: {premise}\nHypothesis: {hypo}"
            elif sample.dataset_id == "strategyqa":
                q_text = sample.text_fields.get("question", "")
            else:
                q_text = str(sample.text_fields)

            # Build arm handoff and detect structural alias
            handoff, alias_to = build_handoff(sample, jev_pred, schema, arm_id)

            if alias_to == "F0":
                src_map = f0_preds_map or {}
                source_pred = src_map.get(sample.sample_id, {})
                alias_pred = Prediction(
                    sample_id=sample.sample_id,
                    dataset_id=sample.dataset_id,
                    request_id=f"alias_{model_id}_{arm_id}_from_F0_{sample.sample_id}",
                    role=role,
                    model_id=model_id,
                    returned_model=source_pred.get("returned_model", model_id),
                    arm_id=arm_id,
                    template_version=source_pred.get("template_version", "v1.0"),
                    decision=source_pred.get("decision"),
                    probabilities={},
                    usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    cost=0.0,
                    latency_ms=0.0,
                    attempts=1,
                    status=source_pred.get("status", "valid"),
                    error_code=source_pred.get("error_code"),
                    raw_response={
                        "alias_to": "F0",
                        "source_request_id": source_pred.get("request_id"),
                        "reason": "StrategyQA K=2 C0 is structurally identical to F0",
                    },
                )
                return alias_pred, False

            elif alias_to == "F1":
                src_map = f1_preds_map or {}
                source_pred = src_map.get(sample.sample_id, {})
                alias_pred = Prediction(
                    sample_id=sample.sample_id,
                    dataset_id=sample.dataset_id,
                    request_id=f"alias_{model_id}_{arm_id}_from_F1_{sample.sample_id}",
                    role=role,
                    model_id=model_id,
                    returned_model=source_pred.get("returned_model", model_id),
                    arm_id=arm_id,
                    template_version=source_pred.get("template_version", "v1.0"),
                    decision=source_pred.get("decision"),
                    probabilities={},
                    usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    cost=0.0,
                    latency_ms=0.0,
                    attempts=1,
                    status=source_pred.get("status", "valid"),
                    error_code=source_pred.get("error_code"),
                    raw_response={
                        "alias_to": "F1",
                        "source_request_id": source_pred.get("request_id"),
                        "reason": "StrategyQA K=2 C1 is structurally identical to F1",
                    },
                )
                return alias_pred, False

            # Normal execution for physical requests
            fp = compute_request_fingerprint(
                dataset_id=sample.dataset_id,
                role=role,
                sample_id=sample.sample_id,
                model_id=model_id,
                arm_id=arm_id,
                prompt_payload={
                    "q": q_text,
                    "arm": arm_id,
                    "opts": handoff.allowed_labels,
                    "prior": handoff.prior_payload,
                },
                generation_params=model_cfg.parameters if model_cfg else {},
                execution_mode=self.mode,
            )

            cached_pred = self.cache.get(fp) or self.cache.get_by_sample(sample.sample_id, model_id, arm_id, mode=self.mode)
            if cached_pred and cached_pred.status == "valid":
                self.cache_hits += 1
                return cached_pred, False

            if not self._check_budget():
                return Prediction(
                    sample_id=sample.sample_id,
                    dataset_id=sample.dataset_id,
                    request_id=f"req_{model_id}_{arm_id}_{sample.sample_id}",
                    role=role,
                    model_id=model_id,
                    returned_model="none",
                    arm_id=arm_id,
                    status="failed",
                    error_code="BUDGET_HALTED",
                    raw_response={"reason": self.halt_reason},
                ), False

            req_id = f"req_{model_id}_{arm_id}_{sample.sample_id}_{int(time.time()*1000)}"
            pred = client.decide(
                sample_id=sample.sample_id,
                request_id=req_id,
                role=role,
                question=q_text,
                handoff=handoff,
                label_descriptions=schema.label_descriptions,
                dataset_id=sample.dataset_id,
            )

            if pred.status == "valid":
                self.cache.put(fp, pred, mode=self.mode)
            return pred, True

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_concurrency) as executor:
            futures = [executor.submit(worker, s) for s in samples]
            for f in concurrent.futures.as_completed(futures):
                pred, is_fresh = f.result()
                if is_fresh:
                    self.actual_requests_dispatched += 1
                    self.actual_cost_total += pred.cost
                predictions.append(pred)

        predictions.sort(key=lambda p: p.sample_id)
        return predictions

