"""Unit tests for Batch B05 handoff conditions, prior formatting, aliases, and bootstrap.

Timestamp: 2026-11-10 13:28 (Asia/Shanghai)
"""

import pytest
import numpy as np
from src.clients.base import Handoff, Prediction
from src.clients.system2 import build_system2_prompt, parse_system2_decision
from src.evaluation.bootstrap import (
    compute_ci_and_pvalue,
    paired_group_bootstrap_series,
    paired_group_bootstrap_multicondition,
)
from src.handoff.builders import build_handoff
from src.handoff.priors import extract_top2_prior
from src.input.splits import SampleInput
from src.input.tasks import TaskLabelSchema


def test_top2_prior_extraction_and_raw_probabilities():
    """Verify Top-2 prior extraction preserves raw probabilities and sorts descending with canonical tie-break."""
    allowed = ["apple", "banana", "cherry", "date"]
    probs = {"apple": 0.20, "banana": 0.45, "cherry": 0.20, "date": 0.15}

    top2_labels, payload = extract_top2_prior(probs, allowed)
    # banana is top-1 (0.45). For top-2, apple and cherry tie at 0.20. Apple appears first canonically.
    assert top2_labels == ["banana", "apple"]

    candidates = payload["candidates"]
    assert len(candidates) == 2
    assert candidates[0]["rank"] == 1
    assert candidates[0]["label"] == "banana"
    assert abs(candidates[0]["probability"] - 0.45) < 1e-6

    assert candidates[1]["rank"] == 2
    assert candidates[1]["label"] == "apple"
    assert abs(candidates[1]["probability"] - 0.20) < 1e-6

    # Verify no re-normalization (sum is 0.65 != 1.0)
    raw_sum = candidates[0]["probability"] + candidates[1]["probability"]
    assert abs(raw_sum - 0.65) < 1e-6
    # Ensure error risk e_hat is NOT in payload
    assert "e_hat" not in payload
    assert "error_risk" not in payload


def test_handoff_builders_all_four_arms_banking():
    """Test F0, F1, C0, C1 builders for multiclass dataset (Banking77)."""
    schema = TaskLabelSchema(
        label_schema_id="banking77",
        task_type="choice",
        allowed_labels=["card_lost", "card_damaged", "pin_blocked", "top_up_failed"],
        label_descriptions={"card_lost": "Lost card", "card_damaged": "Damaged card"},
        reject_allowed=False,
    )
    sample = SampleInput(
        sample_id="b77_001",
        group_id="b77_001",
        dataset_id="banking77",
        source_split="train",
        task_type="choice",
        text_fields={"text": "I lost my credit card."},
        label_schema_id="banking77",
    )
    # Jev prediction puts pin_blocked at 0.6, card_lost at 0.3
    jev_pred = {
        "probabilities": {"card_lost": 0.3, "card_damaged": 0.05, "pin_blocked": 0.6, "top_up_failed": 0.05}
    }

    # 1. F0
    h_f0, alias_f0 = build_handoff(sample, jev_pred, schema, "F0")
    assert alias_f0 is None
    assert h_f0.arm_id == "F0"
    assert h_f0.allowed_labels == schema.allowed_labels
    assert h_f0.prior_payload is None
    assert not h_f0.reject_allowed

    # 2. F1
    h_f1, alias_f1 = build_handoff(sample, jev_pred, schema, "F1")
    assert alias_f1 is None
    assert h_f1.arm_id == "F1"
    assert h_f1.allowed_labels == schema.allowed_labels
    assert h_f1.prior_payload is not None
    assert len(h_f1.prior_payload["candidates"]) == 2

    # 3. C0: Top-2 candidates in relative CANONICAL order
    h_c0, alias_c0 = build_handoff(sample, jev_pred, schema, "C0")
    assert alias_c0 is None
    assert h_c0.arm_id == "C0"
    # Canonical order of schema is [card_lost, card_damaged, pin_blocked, top_up_failed]
    # Top-2 are pin_blocked (0.6) and card_lost (0.3).
    # In canonical relative order, card_lost must come BEFORE pin_blocked!
    assert h_c0.allowed_labels == ["card_lost", "pin_blocked"]
    assert h_c0.prior_payload is None

    # 4. C1: Top-2 in relative canonical order, WITH prior payload
    h_c1, alias_c1 = build_handoff(sample, jev_pred, schema, "C1")
    assert alias_c1 is None
    assert h_c1.arm_id == "C1"
    assert h_c1.allowed_labels == ["card_lost", "pin_blocked"]
    assert h_c1.prior_payload is not None


def test_strategyqa_structural_alias_detection():
    """Verify StrategyQA (K=2) triggers structural alias: C0 -> F0, C1 -> F1."""
    schema = TaskLabelSchema(
        label_schema_id="strategyqa",
        task_type="noul",
        allowed_labels=["yes", "no"],
        label_descriptions={"yes": "Affirmative", "no": "Negative"},
        reject_allowed=False,
    )
    sample = SampleInput(
        sample_id="sqa_001",
        group_id="sqa_001",
        dataset_id="strategyqa",
        source_split="train",
        task_type="noul",
        text_fields={"question": "Can dogs breathe underwater?"},
        label_schema_id="strategyqa",
    )
    jev_pred = {"probabilities": {"yes": 0.1, "no": 0.9}}

    # C0 on StrategyQA must return alias_to="F0"
    h_c0, alias_c0 = build_handoff(sample, jev_pred, schema, "C0")
    assert alias_c0 == "F0"

    # C1 on StrategyQA must return alias_to="F1"
    h_c1, alias_c1 = build_handoff(sample, jev_pred, schema, "C1")
    assert alias_c1 == "F1"


def test_clinc150_reject_retention_in_all_arms():
    """Verify CLINC150 retains REJECT in all 4 conditions and prompt formats note."""
    schema = TaskLabelSchema(
        label_schema_id="clinc150",
        task_type="choice",
        allowed_labels=["balance", "transactions", "pay_bill"],
        label_descriptions={"balance": "Check balance", "REJECT": "Out of scope"},
        reject_allowed=True,
        reject_label="REJECT",
    )
    sample = SampleInput(
        sample_id="clinc_001",
        group_id="clinc_001",
        dataset_id="clinc150",
        source_split="train",
        task_type="choice",
        text_fields={"text": "Can you do backflips?"},
        label_schema_id="clinc150",
    )
    jev_pred = {"probabilities": {"balance": 0.5, "transactions": 0.4, "pay_bill": 0.1}}

    for arm in ["F0", "F1", "C0", "C1"]:
        handoff, _ = build_handoff(sample, jev_pred, schema, arm)
        assert handoff.reject_allowed is True

        prompt = build_system2_prompt("Can you do backflips?", handoff, schema.label_descriptions)
        # REJECT must be in prompt options
        assert "- REJECT:" in prompt
        # arm_id must NOT appear anywhere in the prompt text
        assert f"arm_id" not in prompt
        assert f"arm:" not in prompt
        assert f"[{arm}]" not in prompt

        if arm in ("F1", "C1"):
            assert "The screening model does not assign probability to REJECT." in prompt


def test_paired_group_bootstrap_exact_group_resampling():
    """Test paired group bootstrap clusters samples by group_id and computes paired contrast."""
    # Group g1 has 2 samples, g2 has 2 samples
    groups = ["g1", "g1", "g2", "g2"]
    # In g1, A is 1, B is 0; In g2, A is 1, B is 1
    va = [1.0, 1.0, 1.0, 1.0]  # mean 1.0
    vb = [0.0, 0.0, 1.0, 1.0]  # mean 0.5
    # diff = 0.5

    res = paired_group_bootstrap_series(groups, va, vb, num_bootstrap=500, seed=42)
    assert abs(res["observed_diff"] - 0.5) < 1e-6
    assert res["ci_lower"] >= 0.0
    assert res["ci_upper"] <= 1.0
    assert res["num_groups"] == 2
    assert res["num_samples"] == 4

