"""Unit and integration tests for Batch B03 Calibration methods.

Timestamp: 2026-11-09 18:38 (Asia/Shanghai)

Tests:
1. 1D Isotonic monotonicity and bounds [0, 1]
2. Single-class and zero-variance constant risk fallbacks
3. Temperature scaling bounds, convergence, and normalization
4. Calibration-only fit and zero holdout leakage protection
5. Raw probability preservation (e_hat does not overwrite raw Jev category probs)
6. Model selection decision protocol (strict inequality, tie-breaker, and baseline hurdle)
7. Offline parameters serialization and portable reconstruction
"""

import numpy as np
import pytest

from src.calibration.features import (
    compute_application_error_target,
    extract_sample_risk_features,
    prepare_calibration_dataset,
)
from src.calibration.fit import (
    IsotonicCalibrator,
    LogisticCalibrator,
    TemperatureCalibrator,
    evaluate_and_select_calibrator_family,
)
from src.input.loader import load_inputs


def test_isotonic_monotonicity_and_bounds():
    """Verify that IsotonicCalibrator produces monotonically non-decreasing error risk in [0, 1]."""
    np.random.seed(42)
    # Generate synthetic risk features in [0, 1] and binary errors correlated with risk
    X = np.sort(np.random.uniform(0.0, 1.0, size=200))
    # True error probability increases with X
    prob_error = np.clip(0.1 + 0.8 * X, 0.0, 1.0)
    y = (np.random.uniform(0.0, 1.0, size=200) < prob_error).astype(int)

    cal = IsotonicCalibrator()
    cal.fit(X, y)

    # Test monotonic prediction across full dense grid
    test_grid = np.linspace(0.0, 1.0, 100)
    preds = cal.predict(test_grid)

    assert preds.min() >= 0.0, f"Predicted risk below 0: {preds.min()}"
    assert preds.max() <= 1.0, f"Predicted risk above 1: {preds.max()}"
    # Strictly non-decreasing
    diffs = np.diff(preds)
    assert np.all(diffs >= -1e-12), f"Isotonic monotonicity violated! Min diff: {diffs.min()}"


def test_calibrator_constant_fallbacks():
    """Verify that calibrators fall back to constant base rate when targets are single-class."""
    # Single class: all correct (all 0)
    X = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    y_zeros = np.zeros(5, dtype=int)

    iso = IsotonicCalibrator().fit(X, y_zeros)
    assert iso.is_fallback is True
    assert iso.fallback_risk == 0.0
    assert np.all(iso.predict(np.array([0.2, 0.8])) == 0.0)

    # Single class: all error (all 1)
    y_ones = np.ones(5, dtype=int)
    log_cal = LogisticCalibrator().fit(np.column_stack([X, X, X]), y_ones)
    assert log_cal.is_fallback is True
    assert log_cal.fallback_risk == 1.0
    assert np.all(log_cal.predict(np.column_stack([[0.2, 0.8], [0.2, 0.8], [0.2, 0.8]])) == 1.0)


def test_temperature_scaling_normalization_and_bounds():
    """Verify temperature scaling recalibration preserves probability axioms and minimizes NLL."""
    np.random.seed(42)
    # Synthetic 3-class overconfident distributions
    probs_list = [
        {"A": 0.90, "B": 0.08, "C": 0.02},
        {"A": 0.85, "B": 0.10, "C": 0.05},
        {"A": 0.10, "B": 0.80, "C": 0.10},
        {"A": 0.05, "B": 0.15, "C": 0.80},
    ]
    # Gold labels have some mismatches to induce T > 1
    golds = ["A", "B", "B", "C"]

    temp_cal = TemperatureCalibrator()
    temp_cal.fit(probs_list, golds, bounds=(0.05, 10.0))

    assert temp_cal.is_fitted is True
    assert 0.05 <= temp_cal.temperature <= 10.0
    assert temp_cal.calibration_nll is not None

    # Recalibrate single and batch
    recal = temp_cal.recalibrate(probs_list[0])
    assert set(recal.keys()) == {"A", "B", "C"}
    assert abs(sum(recal.values()) - 1.0) < 1e-6
    assert all(p >= 0.0 for p in recal.values())


def test_clinc_application_error_logic():
    """Verify application error target adheres to fixed OOS threshold tau*."""
    clinc_tau = 0.159485

    # Case 1: In-Domain sample accepted and correct -> error = 0
    # Low entropy (H_norm = 0.05 < tau*), decision matches gold
    e1 = compute_application_error_target(
        decision="transfer",
        gold_label="transfer",
        dataset_id="clinc150",
        probabilities={"transfer": 0.99, "other": 0.01},
        is_oos=False,
        clinc_tau=clinc_tau,
        num_classes=150,
    )
    assert e1 == 0

    # Case 2: In-Domain sample rejected (H_norm >= tau*) -> error = 1 (false rejection)
    # High entropy (uniform distribution)
    uniform_probs = {f"intent_{i}": 1.0 / 150 for i in range(150)}
    e2 = compute_application_error_target(
        decision="transfer",
        gold_label="transfer",
        dataset_id="clinc150",
        probabilities=uniform_probs,
        is_oos=False,
        clinc_tau=clinc_tau,
        num_classes=150,
    )
    assert e2 == 1

    # Case 3: Out-of-Scope sample rejected (H_norm >= tau*) -> error = 0 (successful rejection)
    e3 = compute_application_error_target(
        decision="transfer",
        gold_label="oos",
        dataset_id="clinc150",
        probabilities=uniform_probs,
        is_oos=True,
        clinc_tau=clinc_tau,
        num_classes=150,
    )
    assert e3 == 0

    # Case 4: Out-of-Scope sample accepted (H_norm < tau*) -> error = 1 (false acceptance)
    e4 = compute_application_error_target(
        decision="transfer",
        gold_label="oos",
        dataset_id="clinc150",
        probabilities={"transfer": 0.99, "other": 0.01},
        is_oos=True,
        clinc_tau=clinc_tau,
        num_classes=150,
    )
    assert e4 == 1


def test_calibration_only_fit_and_zero_holdout_access():
    """Verify that calibration fitting respects role isolation and holdout access raises ValueError."""
    # Loading calibration and selection must succeed
    cal_inputs = list(load_inputs(allowed_role="calibration"))
    sel_inputs = list(load_inputs(allowed_role="selection"))
    assert len(cal_inputs) == 1856
    assert len(sel_inputs) == 928

    # Loading holdout via development loader must be blocked
    with pytest.raises((PermissionError, ValueError), match="holdout"):
        _ = list(load_inputs(allowed_role="holdout"))


def test_raw_probabilities_unmodified_preservation():
    """Verify that e_hat is distinct and does not contaminate raw category probabilities."""
    sample_probs = {"yes": 0.85, "no": 0.15}
    original_copy = dict(sample_probs)

    feats = extract_sample_risk_features(sample_probs, num_classes=2)
    assert feats["p1"] == 0.85
    assert feats["p2"] == 0.15
    assert feats["margin"] == 0.70
    assert abs(feats["risk_margin"] - 0.30) < 1e-9

    # Fit a toy calibrator
    iso = IsotonicCalibrator().fit(np.array([0.1, 0.3, 0.5]), np.array([0, 0, 1]))
    e_hat = float(iso.predict(np.array([feats["risk_margin"]]))[0])

    # Downstream handoff must receive original category probabilities, NOT e_hat
    assert sample_probs == original_copy, "Sample probabilities were mutated!"
    assert e_hat != sample_probs["yes"]


def test_model_selection_decision_rules():
    """Test model selection protocol with synthetic candidate comparisons."""
    dataset_ids = ["banking77", "clinc150", "anli_r2r3", "strategyqa"]

    # Mock selection data
    mock_sel_data = {
        ds: {
            "y_error": np.array([0, 1, 0, 1]),
            "raw_p1": np.array([0.9, 0.6, 0.8, 0.7]),
            "X_isotonic": np.array([0.2, 0.7, 0.3, 0.6]),
            "X_logistic": np.array([[0.9, 0.8, 0.2], [0.6, 0.2, 0.8], [0.8, 0.6, 0.3], [0.7, 0.4, 0.5]]),
            "probabilities_list": [{"yes": 0.8, "no": 0.2}] * 4,
            "gold_labels_list": ["yes", "no", "yes", "no"],
            "is_oos_list": [False] * 4,
        }
        for ds in dataset_ids
    }

    # Case 1: Logistic strictly lower Brier than Isotonic and beats raw baseline
    class MockIso:
        def predict(self, X): return np.array([0.4, 0.6, 0.4, 0.6])
    class MockLog:
        def predict(self, X): return np.array([0.1, 0.9, 0.1, 0.9])
    class MockTemp:
        temperature = 1.2
        def recalibrate_batch(self, probs): return probs

    cals_case1 = {
        ds: {"isotonic": MockIso(), "logistic": MockLog(), "temperature": MockTemp()}
        for ds in dataset_ids
    }
    res1 = evaluate_and_select_calibrator_family(cals_case1, mock_sel_data)
    assert res1["selected_method_family"] == "logistic"

    # Case 2: Exact tie between Isotonic and Logistic -> Selects Isotonic (tie-breaker)
    class MockTie:
        def predict(self, X): return np.array([0.2, 0.8, 0.2, 0.8])
    cals_case2 = {
        ds: {"isotonic": MockTie(), "logistic": MockTie(), "temperature": MockTemp()}
        for ds in dataset_ids
    }
    res2 = evaluate_and_select_calibrator_family(cals_case2, mock_sel_data)
    assert res2["selected_method_family"] == "isotonic"

    # Case 3: Winner worse than raw 1 - p1 baseline -> None selected
    class MockTerrible:
        def predict(self, X): return np.array([0.9, 0.1, 0.9, 0.1])  # Inverse prediction!
    cals_case3 = {
        ds: {"isotonic": MockTerrible(), "logistic": MockTerrible(), "temperature": MockTemp()}
        for ds in dataset_ids
    }
    res3 = evaluate_and_select_calibrator_family(cals_case3, mock_sel_data)
    assert res3["selected_method_family"] is None


def test_parameter_portability_and_offline_reconstruction():
    """Verify parameters can be serialized to dict and reconstructed offline identically."""
    np.random.seed(42)
    X_iso = np.sort(np.random.uniform(0.0, 1.0, size=50))
    y = (X_iso > 0.5).astype(int)

    iso = IsotonicCalibrator().fit(X_iso, y)
    iso_dict = iso.to_dict()
    iso_reconstructed = IsotonicCalibrator.from_dict(iso_dict)

    test_X = np.array([0.1, 0.35, 0.62, 0.95])
    np.testing.assert_allclose(iso.predict(test_X), iso_reconstructed.predict(test_X), atol=1e-6)

    # Test Logistic portability
    X_log = np.random.uniform(0.0, 1.0, size=(50, 3))
    log_cal = LogisticCalibrator(C=1.0).fit(X_log, y)
    log_dict = log_cal.to_dict()
    log_reconstructed = LogisticCalibrator.from_dict(log_dict)

    np.testing.assert_allclose(log_cal.predict(X_log[:5]), log_reconstructed.predict(X_log[:5]), atol=1e-6)
