"""Handoff prompt and condition builders for HybridCascade.

Timestamp: 2026-11-10 13:25 (Asia/Shanghai)

Builds handoff specifications for the 4 conditions:
- F0: Full candidate options, no prior package, canonical order
- F1: Full candidate options, Top-2 prior package (raw p), canonical order
- C0: Top-2 candidates in relative canonical order, no prior package
      StrategyQA structural alias -> F0 (alias_to='F0')
- C1: Top-2 candidates in relative canonical order, Top-2 prior package (raw p)
      StrategyQA structural alias -> F1 (alias_to='F1')
CLINC150 retains REJECT in all 4 conditions.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

from src.clients.base import Handoff, Prediction
from src.handoff.priors import extract_top2_prior
from src.input.splits import SampleInput
from src.input.tasks import TaskLabelSchema


def build_handoff(
    sample: SampleInput,
    jev_prediction: Union[Prediction, Dict[str, Any]],
    task_schema: TaskLabelSchema,
    arm_id: str,
) -> Tuple[Handoff, Optional[str]]:
    """Build Handoff specification and determine alias target if applicable.
    
    Args:
        sample: Input sample specification.
        jev_prediction: Upstream Jev System-1 prediction (object or dict).
        task_schema: Canonical TaskLabelSchema for the dataset.
        arm_id: Target experimental arm ("F0", "F1", "C0", "C1").
        
    Returns:
        Tuple of (Handoff object, alias_to: Optional[str]).
        If alias_to is not None (e.g. "F0" or "F1"), this request is structurally
        identical to that arm and should be aliased with 0 API cost.
    """
    arm_id = arm_id.upper()
    if arm_id not in ("F0", "F1", "C0", "C1", "FR"):
        raise ValueError(f"Unknown arm_id '{arm_id}'. Must be one of F0, F1, C0, C1, FR.")

    if isinstance(jev_prediction, dict):
        probs = jev_prediction.get("probabilities") or {}
    else:
        probs = getattr(jev_prediction, "probabilities", {}) or {}

    is_strategyqa = (sample.dataset_id == "strategyqa" or len(task_schema.allowed_labels) <= 2)

    if arm_id == "F0":
        handoff = Handoff(
            arm_id="F0",
            allowed_labels=list(task_schema.allowed_labels),
            prior_payload=None,
            reject_allowed=task_schema.reject_allowed,
        )
        return handoff, None

    elif arm_id == "F1":
        top2_labels, prior_payload = extract_top2_prior(probs, task_schema.allowed_labels)
        handoff = Handoff(
            arm_id="F1",
            allowed_labels=list(task_schema.allowed_labels),
            prior_payload=prior_payload,
            reject_allowed=task_schema.reject_allowed,
        )
        return handoff, None

    elif arm_id == "FR":
        top2_labels, prior_payload = extract_top2_prior(probs, task_schema.allowed_labels)
        handoff = Handoff(
            arm_id="FR",
            allowed_labels=list(task_schema.allowed_labels),
            prior_payload=prior_payload,
            reject_allowed=task_schema.reject_allowed,
        )
        return handoff, None

    elif arm_id == "C0":
        if is_strategyqa:
            # StrategyQA K=2: Top-2 is the full set, no prior -> structural alias to F0
            handoff = Handoff(
                arm_id="C0",
                allowed_labels=list(task_schema.allowed_labels),
                prior_payload=None,
                reject_allowed=task_schema.reject_allowed,
            )
            return handoff, "F0"

        # Multiclass: extract top2, sort into relative canonical order
        top2_labels, _ = extract_top2_prior(probs, task_schema.allowed_labels)
        top2_set = set(top2_labels)
        pruned_canonical = [lbl for lbl in task_schema.allowed_labels if lbl in top2_set]

        handoff = Handoff(
            arm_id="C0",
            allowed_labels=pruned_canonical,
            prior_payload=None,
            reject_allowed=task_schema.reject_allowed,
        )
        return handoff, None

    elif arm_id == "C1":
        top2_labels, prior_payload = extract_top2_prior(probs, task_schema.allowed_labels)
        if is_strategyqa:
            # StrategyQA K=2: Top-2 is the full set with prior -> structural alias to F1
            handoff = Handoff(
                arm_id="C1",
                allowed_labels=list(task_schema.allowed_labels),
                prior_payload=prior_payload,
                reject_allowed=task_schema.reject_allowed,
            )
            return handoff, "F1"

        top2_set = set(top2_labels)
        pruned_canonical = [lbl for lbl in task_schema.allowed_labels if lbl in top2_set]

        handoff = Handoff(
            arm_id="C1",
            allowed_labels=pruned_canonical,
            prior_payload=prior_payload,
            reject_allowed=task_schema.reject_allowed,
        )
        return handoff, None

    raise ValueError(f"Unhandled arm_id: {arm_id}")

