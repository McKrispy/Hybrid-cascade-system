"""System-2 abstract base client and handoff prompt formatting.

Timestamp: 2026-11-09 11:32 (Asia/Shanghai)

Defines the contract for System-2 LLM clients (gpt-6.1-sol and deepseek-v4.1-flash).
Formats F0, F1, C0, and C1 prompts with real label criteria and prior packages.
Enforces strict JSON decision parsing without heuristic regex guessing on conversational text.
"""

import abc
import json
import re
from typing import Any, Dict, List, Optional

from src.clients.base import Handoff, Prediction
from src.infra.config import get_env_var


def build_system2_prompt(
    question: str,
    handoff: Handoff,
    label_descriptions: Optional[Dict[str, str]] = None,
) -> str:
    """Format prompt for System-2 including label descriptions and exact prior package."""
    arm = handoff.arm_id
    allowed_labels = list(handoff.allowed_labels)
    descriptions = label_descriptions or {}

    lines = [
        "You are evaluating a precise decision classification task.",
        f"Input Text / Query:\n{question}\n",
        "Available Options:",
    ]

    for opt in allowed_labels:
        desc = descriptions.get(opt, "")
        if desc:
            lines.append(f"- {opt}: {desc}")
        else:
            lines.append(f"- {opt}")

    if handoff.reject_allowed and "REJECT" not in allowed_labels:
        allowed_labels.append("REJECT")
        reject_desc = descriptions.get("REJECT", "Select this if the input query does not match any of the above categories (Out-Of-Scope).")
        lines.append(f"- REJECT: {reject_desc}")

    # Prior payload for F1 and C1 (with probability numbers) and FR (identity/ranking only)
    if arm in ("F1", "C1") and handoff.prior_payload:
        priors = handoff.prior_payload.get("candidates", [])
        lines.append("\n[Prior Screening Information]")
        lines.append("Disclaimer: The initial fast-screening model may be incorrect or overconfident. Base your final judgment strictly on the input text evidence.")
        for p_item in priors:
            rank = p_item.get("rank")
            label = p_item.get("label")
            p = p_item.get("probability", 0.0)
            lines.append(f"- Top-{rank} Prior: {label} (Probability: {p:.4f})")
        if handoff.reject_allowed:
            lines.append("Note: The screening model does not assign probability to REJECT.")
    elif arm == "FR" and handoff.prior_payload:
        priors = handoff.prior_payload.get("candidates", [])
        lines.append("\n[Prior Screening Information]")
        lines.append("Disclaimer: The initial fast-screening model may be incorrect or overconfident. Base your final judgment strictly on the input text evidence.")
        for p_item in priors:
            rank = p_item.get("rank")
            label = p_item.get("label")
            lines.append(f"- Top-{rank} Prior: {label}")
        if handoff.reject_allowed:
            lines.append("Note: The screening model does not assign probability to REJECT.")

    lines.append("\nOutput Instructions:")
    lines.append("You MUST select exactly ONE option from the Available Options.")
    lines.append("Respond ONLY with a valid JSON object matching this schema:")
    lines.append('{"decision": "<one_exact_allowed_label>"}')

    return "\n".join(lines)


def parse_system2_decision(
    response_text: str,
    allowed_labels: List[str],
    reject_allowed: bool = False,
) -> Optional[str]:
    """Strictly parse JSON decision without guessing from conversational text.
    
    Returns valid label if strictly present in allowed_labels; otherwise returns None.
    """
    valid_set = set(allowed_labels)
    if reject_allowed:
        valid_set.add("REJECT")

    cleaned = response_text.strip()
    # Remove markdown code fences if present (```json ... ```)
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    try:
        data = json.loads(cleaned)
        if isinstance(data, dict) and "decision" in data:
            decision = str(data["decision"]).strip()
            if decision in valid_set:
                return decision
    except Exception:
        pass

    # Direct JSON object extraction fallback if surrounding text exists
    json_match = re.search(r"\{[^{}]*\"decision\"\s*:\s*\"([^\"]+)\"[^{}]*\}", response_text)
    if json_match:
        decision = json_match.group(1).strip()
        if decision in valid_set:
            return decision

    # Never guess from arbitrary text mentions
    return None


class System2Client(abc.ABC):
    """Abstract contract for System-2 models."""

    def __init__(
        self,
        model_id: str,
        env_key_name: str,
        endpoint: str,
        input_cost_per_million: float,
        output_cost_per_million: float,
        timeout: float = 30.0,
        max_retries: int = 3,
        backoff_factor: float = 1.5,
        parameters: Optional[Dict[str, Any]] = None,
        mock_mode: bool = False,
    ):
        self.model_id = model_id
        self.env_key_name = env_key_name
        self.api_key = get_env_var(env_key_name, "")
        self.endpoint = endpoint
        self.input_cost_per_million = input_cost_per_million
        self.output_cost_per_million = output_cost_per_million
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.parameters = parameters or {"temperature": 0.0, "max_tokens": 100}
        self.mock_mode = mock_mode

    def calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate USD cost based on token pricing."""
        return (prompt_tokens / 1_000_000.0) * self.input_cost_per_million + (
            completion_tokens / 1_000_000.0
        ) * self.output_cost_per_million

    @abc.abstractmethod
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
        """Request decision from System-2 model."""
        pass
