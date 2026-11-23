"""Unit and integration tests for Batch B04 Routing and Escalation Masks.

Timestamp: 2026-11-10 13:05 (Asia/Shanghai)

Tests:
1. Deterministic sorting and sample_id tie-breaking
2. Strict mask nesting (M_15 subset M_25 subset M_35) and union == M_35
3. Cascade boundary identities: q=0 equals Jev, q=100% equals System-2 F0
4. Net gain transfer identity: Corrected - Damaged == Cascaded Correct - Jev Correct
5. Primary mask invariance across System-2 backends
6. CLINC mixed pool application correctness logic in cascades
7. Holdout physical isolation guard
"""

import math
import numpy as np
import pytest

from src.input.loader import load_inputs
from src.routing.masks import (
    build_nested_masks_for_signal,
    build_oracle_nested_masks,
    build_random_nested_masks,
    build_route_mask,
    evaluate_cascade,
)
from src.routing.signals import compute_sample_routing_signals


def test_deterministic_sorting_and_tie_breaking():
    """Verify that sample sorting places higher risk first and breaks ties with sample_id ascending."""
    sample_ids = ["samp_C", "samp_A", "samp_B", "samp_D"]
    # samp_A and samp_C tie on risk 0.8; samp_B has 0.9; samp_D has 0.1
    risk_scores = [0.8, 0.8, 0.9, 0.1]

    # q=0.5 -> ceil(4 * 0.5) = 2 samples
    mask = build_route_mask(sample_ids, risk_scores, q=0.5)
    # Expected top 2: samp_B (0.9), then between samp_A and samp_C, samp_A wins on alphabetical tie-break
    assert mask == ["samp_B", "samp_A"]

    # Test +infinity risk for invalid upstream prediction
    sample_ids_inf = ["samp_X", "samp_Y"]
    risk_inf = [float("inf"), 0.99]
    mask_inf = build_route_mask(sample_ids_inf, risk_inf, q=0.5)
    assert mask_inf == ["samp_X"]


def test_mask_nesting_and_union():
    """Verify that masks are strictly nested across q rates and union equals M_35."""
    np.random.seed(42)
    N = 100
    sample_ids = [f"sample_{i:03d}" for i in range(N)]
    risks = np.random.uniform(0.0, 1.0, size=N).tolist()

    q_rates = [0.15, 0.25, 0.35]
    res = build_nested_masks_for_signal(sample_ids, risks, q_rates=q_rates)

    m15 = set(res["masks"]["0.15"])
    m25 = set(res["masks"]["0.25"])
    m35 = set(res["masks"]["0.35"])

    assert len(m15) == math.ceil(100 * 0.15) == 15
    assert len(m25) == math.ceil(100 * 0.25) == 25
    assert len(m35) == math.ceil(100 * 0.35) == 35

    assert m15.issubset(m25), "M_15 is not a subset of M_25!"
    assert m25.issubset(m35), "M_25 is not a subset of M_35!"
    assert m15 | m25 | m35 == m35, "Union of masks does not equal M_35!"
    assert set(res["union_sids"]) == m35


def test_cascade_boundary_identities():
    """Verify that at q=0% cascade equals Jev alone, and at q=100% cascade equals S2 alone."""
    sample_ids = ["s1", "s2", "s3", "s4"]
    golds = {
        "s1": {"gold_label": "A", "is_oos": False},
        "s2": {"gold_label": "B", "is_oos": False},
        "s3": {"gold_label": "A", "is_oos": False},
        "s4": {"gold_label": "B", "is_oos": False},
    }
    jev_preds = {
        "s1": {"sample_id": "s1", "decision": "A", "status": "valid", "cost": 0.001, "model_id": "jev-1.14.0"},
        "s2": {"sample_id": "s2", "decision": "A", "status": "valid", "cost": 0.001, "model_id": "jev-1.14.0"}, # Wrong
        "s3": {"sample_id": "s3", "decision": "B", "status": "valid", "cost": 0.001, "model_id": "jev-1.14.0"}, # Wrong
        "s4": {"sample_id": "s4", "decision": "B", "status": "valid", "cost": 0.001, "model_id": "jev-1.14.0"},
    }
    s2_preds = {
        "s1": {"sample_id": "s1", "decision": "A", "status": "valid", "cost": 0.010, "model_id": "gpt-6.1-sol"},
        "s2": {"sample_id": "s2", "decision": "B", "status": "valid", "cost": 0.010, "model_id": "gpt-6.1-sol"}, # Corrected
        "s3": {"sample_id": "s3", "decision": "A", "status": "valid", "cost": 0.010, "model_id": "gpt-6.1-sol"}, # Corrected
        "s4": {"sample_id": "s4", "decision": "A", "status": "valid", "cost": 0.010, "model_id": "gpt-6.1-sol"}, # Damaged
    }

    # 1. Boundary: q = 0% (mask empty)
    res_q0 = evaluate_cascade(
        sample_ids=sample_ids,
        jev_predictions_map=jev_preds,
        s2_predictions_map=s2_preds,
        gold_records_map=golds,
        mask=[],
        dataset_id="banking77",
    )
    # Jev gets s1 and s4 right -> 2/4 = 0.50
    assert res_q0["acc_cascade"] == 0.50
    assert res_q0["transfers"]["corrected"] == 0
    assert res_q0["transfers"]["damaged"] == 0
    assert abs(res_q0["costs"]["total_deploy_cost_usd"] - 0.004) < 1e-6

    # 2. Boundary: q = 100% (all samples routed)
    res_q100 = evaluate_cascade(
        sample_ids=sample_ids,
        jev_predictions_map=jev_preds,
        s2_predictions_map=s2_preds,
        gold_records_map=golds,
        mask=sample_ids,
        dataset_id="banking77",
    )
    # S2 gets s1, s2, s3 right -> 3/4 = 0.75
    assert res_q100["acc_cascade"] == 0.75
    assert res_q100["transfers"]["corrected"] == 2 # s2, s3
    assert res_q100["transfers"]["damaged"] == 1   # s4
    assert res_q100["transfers"]["net_gain"] == 1
    # Cost = 4 * 0.001 (Jev) + 4 * 0.010 (S2) = 0.044
    assert abs(res_q100["costs"]["total_deploy_cost_usd"] - 0.044) < 1e-6


def test_transfer_net_gain_identity():
    """Verify exact net gain identity: Corrected - Damaged == Cascaded Correct - Jev Correct."""
    sample_ids = ["s1", "s2", "s3", "s4"]
    golds = {s: {"gold_label": "yes", "is_oos": False} for s in sample_ids}

    # Jev gets s1, s2 correct; s3, s4 wrong
    jev_preds = {
        "s1": {"sample_id": "s1", "decision": "yes", "status": "valid", "cost": 0.001, "model_id": "jev-1.14.0"},
        "s2": {"sample_id": "s2", "decision": "yes", "status": "valid", "cost": 0.001, "model_id": "jev-1.14.0"},
        "s3": {"sample_id": "s3", "decision": "no", "status": "valid", "cost": 0.001, "model_id": "jev-1.14.0"},
        "s4": {"sample_id": "s4", "decision": "no", "status": "valid", "cost": 0.001, "model_id": "jev-1.14.0"},
    }
    # S2 gets s2, s3, s4 correct; s1 wrong
    s2_preds = {
        "s1": {"sample_id": "s1", "decision": "no", "status": "valid", "cost": 0.01, "model_id": "gpt-6.1-sol"},
        "s2": {"sample_id": "s2", "decision": "yes", "status": "valid", "cost": 0.01, "model_id": "gpt-6.1-sol"},
        "s3": {"sample_id": "s3", "decision": "yes", "status": "valid", "cost": 0.01, "model_id": "gpt-6.1-sol"},
        "s4": {"sample_id": "s4", "decision": "yes", "status": "valid", "cost": 0.01, "model_id": "gpt-6.1-sol"},
    }

    # Route only s1 and s3
    mask = ["s1", "s3"]
    res = evaluate_cascade(sample_ids, jev_preds, s2_preds, golds, mask, "strategyqa")

    # s1 was Jev correct -> S2 damaged (damaged=1)
    # s3 was Jev wrong -> S2 corrected (corrected=1)
    # Net gain = 1 - 1 = 0
    # Jev correct: 2 (s1, s2). Cascade correct: s2 (from jev) + s3 (from s2) = 2.
    assert res["transfers"]["corrected"] == 1
    assert res["transfers"]["damaged"] == 1
    assert res["transfers"]["net_gain"] == 0
    assert res["acc_cascade"] == 0.50


def test_clinc_application_correctness_in_cascade():
    """Verify CLINC mixed application correctness in cascade evaluation."""
    clinc_tau = 0.159485
    sample_ids = ["id_samp", "oos_samp"]
    golds = {
        "id_samp": {"gold_label": "balance", "is_oos": False},
        "oos_samp": {"gold_label": "oos", "is_oos": True},
    }
    # Jev: id_samp has low entropy (accepted, correct); oos_samp has low entropy (accepted, wrong)
    jev_preds = {
        "id_samp": {"sample_id": "id_samp", "decision": "balance", "probabilities": {"balance": 0.99, "other": 0.01}, "status": "valid", "cost": 0.001, "model_id": "jev-1.14.0"},
        "oos_samp": {"sample_id": "oos_samp", "decision": "balance", "probabilities": {"balance": 0.99, "other": 0.01}, "status": "valid", "cost": 0.001, "model_id": "jev-1.14.0"},
    }
    # S2: id_samp rejected (damaged!); oos_samp rejected (corrected!)
    s2_preds = {
        "id_samp": {"sample_id": "id_samp", "decision": "REJECT", "status": "valid", "cost": 0.01, "model_id": "gpt-6.1-sol"},
        "oos_samp": {"sample_id": "oos_samp", "decision": "REJECT", "status": "valid", "cost": 0.01, "model_id": "gpt-6.1-sol"},
    }

    # Route oos_samp only
    res = evaluate_cascade(sample_ids, jev_preds, s2_preds, golds, ["oos_samp"], "clinc150", clinc_tau=clinc_tau, num_classes=150)
    # id_samp unrouted: Jev accepted and correct -> 1
    # oos_samp routed: S2 REJECT and is_oos -> 1
    # Overall cascade acc = 2/2 = 1.0
    assert res["acc_cascade"] == 1.0
    assert res["transfers"]["corrected"] == 1
    assert res["transfers"]["damaged"] == 0


def test_holdout_isolation_in_b04():
    """Verify that development loader strictly blocks holdout access during B04."""
    with pytest.raises((PermissionError, ValueError), match="holdout"):
        _ = list(load_inputs(allowed_role="holdout"))

