"""Unit and contract tests for HybridCascade Batch B02: Single-model development baselines.

Timestamp: 2026-11-09 14:00 (Asia/Shanghai)
"""

import json
import math
import tempfile
import unittest
from pathlib import Path

from src.clients.base import Handoff, Prediction
from src.evaluation.metrics import (
    compute_accuracy,
    compute_category_brier,
    compute_ece_toplabel,
    compute_error_detection_auroc_aurc,
    compute_nll,
    compute_normalized_entropy,
    search_clinc_oos_threshold,
)
from src.inference.cache import InferenceCache, compute_request_fingerprint
from src.input.splits import SampleInput


class TestBatchB02Baseline(unittest.TestCase):

    def test_01_hand_calculated_metric_verifications(self):
        """Verify ECE, Brier, NLL, and AUROC against exact hand-calculated toy examples."""
        # 4 toy samples across 3 classes: ["A", "B", "C"]
        probs = [
            {"A": 0.8, "B": 0.1, "C": 0.1},  # pred A (conf 0.8), gold A -> Correct (z=1)
            {"A": 0.7, "B": 0.2, "C": 0.1},  # pred A (conf 0.7), gold B -> Wrong (z=0)
            {"A": 0.1, "B": 0.6, "C": 0.3},  # pred B (conf 0.6), gold B -> Correct (z=1)
            {"A": 0.2, "B": 0.3, "C": 0.5},  # pred C (conf 0.5), gold A -> Wrong (z=0)
        ]
        golds = ["A", "B", "B", "A"]
        preds = ["A", "A", "B", "C"]

        # 1. Accuracy: 2 / 4 = 0.50
        acc = compute_accuracy(preds, golds)
        self.assertAlmostEqual(acc, 0.5, places=5)

        # 2. Category Brier Score: mean_i sum_k (p_ik - y_ik)^2
        # Sample 1: (0.8-1)^2 + (0.1-0)^2 + (0.1-0)^2 = 0.04 + 0.01 + 0.01 = 0.06
        # Sample 2: (0.7-0)^2 + (0.2-1)^2 + (0.1-0)^2 = 0.49 + 0.64 + 0.01 = 1.14
        # Sample 3: (0.1-0)^2 + (0.6-1)^2 + (0.3-0)^2 = 0.01 + 0.16 + 0.09 = 0.26
        # Sample 4: (0.2-1)^2 + (0.3-0)^2 + (0.5-0)^2 = 0.64 + 0.09 + 0.25 = 0.98
        # Expected Mean: (0.06 + 1.14 + 0.26 + 0.98) / 4 = 2.44 / 4 = 0.61
        brier = compute_category_brier(probs, golds, ["A", "B", "C"])
        self.assertAlmostEqual(brier, 0.61, places=5)

        # 3. NLL: -mean_i log(p_i, gold)
        # - (log(0.8) + log(0.2) + log(0.6) + log(0.2)) / 4
        expected_nll = -(math.log(0.8) + math.log(0.2) + math.log(0.6) + math.log(0.2)) / 4.0
        nll = compute_nll(probs, golds)
        self.assertAlmostEqual(nll, expected_nll, places=5)

        # 4. Error detection AUROC:
        # Errors: [0, 1, 0, 1] (samples 2 and 4 are errors)
        # Risks (1 - c):
        # Sample 1 (Correct): 1 - 0.8 = 0.2
        # Sample 2 (Error):   1 - 0.7 = 0.3
        # Sample 3 (Correct): 1 - 0.6 = 0.4
        # Sample 4 (Error):   1 - 0.5 = 0.5
        # Pairs:
        # Error 2 (0.3) vs Correct 1 (0.2): 0.3 > 0.2 -> score = 1
        # Error 2 (0.3) vs Correct 3 (0.4): 0.3 < 0.4 -> score = 0
        # Error 4 (0.5) vs Correct 1 (0.2): 0.5 > 0.2 -> score = 1
        # Error 4 (0.5) vs Correct 3 (0.4): 0.5 > 0.4 -> score = 1
        # Expected AUROC = (1 + 0 + 1 + 1) / (2 * 2) = 3 / 4 = 0.75
        auroc_res = compute_error_detection_auroc_aurc(probs, golds)
        self.assertAlmostEqual(auroc_res["auroc"], 0.75, places=5)

    def test_02_ece_calculation(self):
        """Verify 15-bin equal-width ECE calculation."""
        probs = [
            {"A": 0.95, "B": 0.05},  # z=1, conf=0.95
            {"A": 0.95, "B": 0.05},  # z=0, conf=0.95
        ]
        golds = ["A", "B"]
        res = compute_ece_toplabel(probs, golds, n_bins=15)
        # Both fall in same bin. acc = 0.5, conf = 0.95 -> gap = 0.45. ECE = 0.45
        self.assertAlmostEqual(res["ece"], 0.45, places=5)
        self.assertAlmostEqual(res["mce"], 0.45, places=5)

    def test_03_normalized_entropy_and_boundary_conditions(self):
        """Verify H_norm is 1.0 for uniform distribution and 0.0 for one-hot distribution."""
        # 1. One-hot distribution -> Entropy must be exactly 0.0
        one_hot = {"A": 1.0, "B": 0.0, "C": 0.0}
        self.assertAlmostEqual(compute_normalized_entropy(one_hot), 0.0, places=6)

        # 2. Uniform distribution over 4 classes -> Entropy must be exactly 1.0
        uniform = {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}
        self.assertAlmostEqual(compute_normalized_entropy(uniform), 1.0, places=6)

        # 3. K = 1 edge case
        single = {"A": 1.0}
        self.assertAlmostEqual(compute_normalized_entropy(single), 0.0, places=6)

    def test_04_clinc_oos_threshold_search_and_tie_breaking(self):
        """Verify OOS threshold search on selection entropy and higher-tau tie-breaking."""
        # 2 ID samples and 2 OOS samples
        # Let K=150
        # ID 1: sharp (H_norm ~ 0.1)
        # ID 2: moderate (H_norm ~ 0.3)
        # OOS 1: high (H_norm ~ 0.7)
        # OOS 2: high (H_norm ~ 0.9)
        selection_probs = [
            {"intent1": 0.9, "intent2": 0.1},
            {"intent1": 0.6, "intent2": 0.4},
            {"intent1": 0.3, "intent2": 0.3, "intent3": 0.4},
            {"intent1": 0.25, "intent2": 0.25, "intent3": 0.25, "intent4": 0.25},
        ]
        selection_is_oos = [False, False, True, True]

        policy = search_clinc_oos_threshold(selection_probs, selection_is_oos, num_classes=150)
        self.assertIn("optimal_tau", policy)
        self.assertEqual(policy["balanced_accuracy"], 1.0)
        self.assertEqual(policy["id_tpr"], 1.0)
        self.assertEqual(policy["oos_tnr"], 1.0)

        # Test tie-breaking rule: if tau1 and tau2 give same balanced acc, pick higher tau
        entropies = [compute_normalized_entropy(p, 150) for p in selection_probs]
        # Any tau between max(ID)=entropies[1] and min(OOS)=entropies[2] achieves balanced_acc=1.0.
        # The algorithm must pick min(OOS) = entropies[2], the higher tau.
        self.assertAlmostEqual(policy["optimal_tau"], entropies[2], places=5)
        self.assertEqual(policy["tie_break_rule"], "higher_tau")

    def test_05_inference_cache_resumption(self):
        """Verify InferenceCache stores, resumes, and prevents duplicate requests."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache = InferenceCache(cache_dir=Path(tmp_dir))
            self.assertEqual(len(cache), 0)

            fp = compute_request_fingerprint(
                dataset_id="banking77",
                role="calibration",
                sample_id="banking77_00001",
                model_id="jev-1.14.0",
                arm_id="JEV",
                prompt_payload="Sample test question",
                generation_params={"mode": "default"},
                execution_mode="live",
            )

            self.assertIsNone(cache.get(fp))

            pred = Prediction(
                sample_id="banking77_00001",
                dataset_id="banking77",
                request_id="req_001",
                role="calibration",
                model_id="jev-1.14.0",
                returned_model="jev-1.14.0",
                arm_id="JEV",
                decision="card_arrival",
                probabilities={"card_arrival": 0.95, "card_linking": 0.05},
                usage={"input_tokens": 120, "output_tokens": 0, "total_tokens": 120},
                cost=0.000005,
                latency_ms=120.0,
                status="valid",
            )

            cache.put(fp, pred)
            self.assertEqual(len(cache), 1)

            # Re-read with a fresh cache instance pointing to same directory
            cache2 = InferenceCache(cache_dir=Path(tmp_dir))
            cached_pred = cache2.get(fp)
            self.assertIsNotNone(cached_pred)
            self.assertEqual(cached_pred.decision, "card_arrival")
            self.assertEqual(cached_pred.sample_id, "banking77_00001")


if __name__ == "__main__":
    unittest.main()

