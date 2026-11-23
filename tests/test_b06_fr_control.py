"""Automated verification and regression tests for Batch B06: FR numerical prior control.

Timestamp: 2026-11-10 16:38 (Asia/Shanghai)
"""

import json
from pathlib import Path
import unittest

from src.clients.base import Handoff
from src.clients.system2 import build_system2_prompt
from src.infra.paths import get_configs_dir, get_data_dir, get_project_root, get_runs_dir


class TestB06FrControl(unittest.TestCase):
    """Test suite verifying B06 sampling, prompt compliance, and paired statistical results."""

    @classmethod
    def setUpClass(cls):
        cls.project_root = get_project_root()
        cls.config_path = get_configs_dir() / "B06.json"
        cls.run_dir = get_runs_dir() / "B06" / "B06-C01"

        with open(cls.config_path, "r", encoding="utf-8") as f:
            cls.config = json.load(f)

        with open(cls.run_dir / "sampled_manifest.json", "r", encoding="utf-8") as f:
            cls.sampled_manifest = json.load(f)

        with open(cls.run_dir / "results.json", "r", encoding="utf-8") as f:
            cls.results = json.load(f)

        with open(cls.run_dir / "cost_ledger.json", "r", encoding="utf-8") as f:
            cls.cost_ledger = json.load(f)

        with open(cls.run_dir / "audit.json", "r", encoding="utf-8") as f:
            cls.audit = json.load(f)

        with open(get_data_dir() / "primary_masks.json", "r", encoding="utf-8") as f:
            cls.primary_masks = json.load(f)["by_dataset_role"]

    def test_01_config_and_constraints(self):
        """Verify B06 configuration conforms strictly to plan.md constraints."""
        self.assertEqual(self.config["batch_id"], "B06")
        self.assertIn("gpt-6.1-sol", self.config["models"])
        self.assertIn("deepseek-v4.1-flash", self.config["models"])
        self.assertFalse(self.config["roles"]["holdout"]["allowed"])
        self.assertEqual(self.config["sampling_quotas"]["calibration"], 50)
        self.assertEqual(self.config["sampling_quotas"]["selection"], 25)
        self.assertEqual(self.config["sampling_quotas"]["holdout"], 175)

    def test_02_sampled_manifest_and_quota_adherence(self):
        """Verify sampling quotas: 50 cal, 25 sel per dataset = 300 dev pool samples within 35% masks."""
        self.assertEqual(self.sampled_manifest["total_calibration_samples"], 200)
        self.assertEqual(self.sampled_manifest["total_selection_samples"], 100)
        self.assertEqual(self.sampled_manifest["total_dev_samples"], 300)

        cal_sids = set()
        sel_sids = set()

        for ds_id in ["banking77", "clinc150", "anli_r2r3", "strategyqa"]:
            cal_info = self.sampled_manifest["by_dataset_role"][f"{ds_id}:calibration"]
            sel_info = self.sampled_manifest["by_dataset_role"][f"{ds_id}:selection"]

            self.assertEqual(cal_info["sampled_count"], 50)
            self.assertEqual(sel_info["sampled_count"], 25)

            # Check that sampled IDs are strictly within the primary 35% escalation mask
            cal_mask35 = set(self.primary_masks[f"{ds_id}:calibration"]["masks"]["0.35"])
            sel_mask35 = set(self.primary_masks[f"{ds_id}:selection"]["masks"]["0.35"])

            self.assertTrue(set(cal_info["sample_ids"]).issubset(cal_mask35))
            self.assertTrue(set(sel_info["sample_ids"]).issubset(sel_mask35))

            cal_sids.update(cal_info["sample_ids"])
            sel_sids.update(sel_info["sample_ids"])

        # Zero overlap between roles
        self.assertEqual(len(cal_sids & sel_sids), 0)
        self.assertEqual(len(cal_sids | sel_sids), 300)

    def test_03_fr_prompt_formatting_zero_probability_leakage(self):
        """Verify FR prompt contains Top-2 identity/ranking but ZERO probability numbers."""
        prior_payload = {
            "candidates": [
                {"rank": 1, "label": "card_arrival", "probability": 0.8124},
                {"rank": 2, "label": "transfer_fee", "probability": 0.1250},
            ]
        }
        allowed = ["card_arrival", "transfer_fee", "lost_card"]

        # 1. Arm FR
        handoff_fr = Handoff(arm_id="FR", allowed_labels=allowed, prior_payload=prior_payload)
        prompt_fr = build_system2_prompt("I lost my card", handoff_fr)

        self.assertIn("[Prior Screening Information]", prompt_fr)
        self.assertIn("- Top-1 Prior: card_arrival", prompt_fr)
        self.assertIn("- Top-2 Prior: transfer_fee", prompt_fr)
        self.assertNotIn("Probability:", prompt_fr)
        self.assertNotIn("0.8124", prompt_fr)
        self.assertNotIn("0.1250", prompt_fr)

        # 2. Arm F1
        handoff_f1 = Handoff(arm_id="F1", allowed_labels=allowed, prior_payload=prior_payload)
        prompt_f1 = build_system2_prompt("I lost my card", handoff_f1)

        self.assertIn("Probability: 0.8124", prompt_f1)
        self.assertIn("Probability: 0.1250", prompt_f1)

        # 3. Instruction and candidate list consistency
        self.assertIn("Available Options:\n- card_arrival\n- transfer_fee\n- lost_card", prompt_fr)
        self.assertIn("Available Options:\n- card_arrival\n- transfer_fee\n- lost_card", prompt_f1)

    def test_04_execution_and_billing_ledger(self):
        """Verify exactly 600 live requests dispatched with platform billing tracking."""
        self.assertEqual(self.cost_ledger["total_requests"], 600)
        self.assertEqual(self.audit["actual_http_requests"], 600)
        self.assertEqual(self.audit["holdout_samples_accessed"], 0)
        self.assertEqual(self.audit["zero_redundant_calls_check"], "passed")

        gpt_cost = self.cost_ledger["by_model"]["gpt-6.1-sol"]["total_cost_usd"]
        ds_cost = self.cost_ledger["by_model"]["deepseek-v4.1-flash"]["total_cost_usd"]

        self.assertGreater(gpt_cost, 0.5)
        self.assertGreater(ds_cost, 0.05)
        self.assertAlmostEqual(self.cost_ledger["total_cost_usd"], gpt_cost + ds_cost, places=5)

    def test_05_statistical_evaluation_and_effect_decomposition(self):
        """Verify statistical evaluation on Selection role and mathematical decomposition identity."""
        sel_eval = self.results["evaluation"]["selection"]

        for m in ["gpt-6.1-sol", "deepseek-v4.1-flash"]:
            macro_stat = sel_eval[m]["macro_4_datasets"]
            f0_acc = macro_stat["macro_f0_accuracy"]
            fr_acc = macro_stat["macro_fr_accuracy"]
            f1_acc = macro_stat["macro_f1_accuracy"]

            diff_rank = macro_stat["identity_rank_effect_fr_minus_f0"]["observed_diff"]
            diff_num = macro_stat["numerical_probability_effect_f1_minus_fr"]["observed_diff"]
            diff_tot = macro_stat["total_effect_f1_minus_f0"]["observed_diff"]

            self.assertAlmostEqual(diff_rank, fr_acc - f0_acc, places=5)
            self.assertAlmostEqual(diff_num, f1_acc - fr_acc, places=5)
            self.assertAlmostEqual(diff_tot, f1_acc - f0_acc, places=5)
            # Mathematical identity: (FR - F0) + (F1 - FR) == (F1 - F0)
            self.assertAlmostEqual(diff_rank + diff_num, diff_tot, places=5)

            # Bootstrap CIs bounded
            ci_rank = macro_stat["identity_rank_effect_fr_minus_f0"]
            self.assertLessEqual(ci_rank["ci_lower"], ci_rank["ci_upper"])
            ci_num = macro_stat["numerical_probability_effect_f1_minus_fr"]
            self.assertLessEqual(ci_num["ci_lower"], ci_num["ci_upper"])


if __name__ == "__main__":
    unittest.main()

