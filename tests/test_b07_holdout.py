"""Automated verification and regression tests for Batch B07: Final Holdout Evaluation.

Timestamp: 2026-11-10 23:15 (Asia/Shanghai)
"""

import json
from pathlib import Path
import unittest

from src.infra.paths import get_configs_dir, get_data_dir, get_project_root, get_runs_dir


class TestB07Holdout(unittest.TestCase):
    """Test suite verifying B07 pre-registration, holdout masks, cost ledger, and statistical results."""

    @classmethod
    def setUpClass(cls):
        cls.project_root = get_project_root()
        cls.config_path = get_configs_dir() / "B07.json"
        cls.run_dir = get_runs_dir() / "B07" / "B07-C01"

        with open(cls.config_path, "r", encoding="utf-8") as f:
            cls.config = json.load(f)

        with open(cls.run_dir / "final_protocol.json", "r", encoding="utf-8") as f:
            cls.protocol = json.load(f)

        with open(cls.run_dir / "holdout_primary_masks.json", "r", encoding="utf-8") as f:
            cls.masks = json.load(f)

        with open(cls.run_dir / "results.json", "r", encoding="utf-8") as f:
            cls.results = json.load(f)

        with open(cls.run_dir / "cost_ledger.json", "r", encoding="utf-8") as f:
            cls.cost_ledger = json.load(f)

        with open(cls.run_dir / "audit.json", "r", encoding="utf-8") as f:
            cls.audit = json.load(f)

    def test_01_config_and_budget_constraints(self):
        """Verify B07 configuration conforms strictly to plan.md constraints."""
        self.assertEqual(self.config["batch_id"], "B07")
        self.assertEqual(self.config["seed"], 42)
        self.assertIn("jev-1.14.0", self.config["models"])
        self.assertIn("gpt-6.1-sol", self.config["models"])
        self.assertIn("deepseek-v4.1-flash", self.config["models"])
        self.assertLessEqual(self.config["budget_limits"]["max_cost_usd"], 30.0)

    def test_02_pre_registration_protocol(self):
        """Verify pre-registered final_protocol.json exists and specifies primary contrasts."""
        self.assertEqual(self.protocol["batch_id"], "B07")
        self.assertEqual(self.protocol["total_holdout_samples"], 6496)
        self.assertEqual(self.protocol["frozen_parameters"]["confidence_level"], 0.95)
        self.assertEqual(self.protocol["frozen_parameters"]["bootstrap_seed"], 42)
        self.assertIn("F1 - F0", self.protocol["hypotheses_and_contrasts"]["primary_contrast"]["name"])

    def test_03_holdout_masks_structure(self):
        """Verify holdout 35% union has exactly 2,274 samples and B06 quota has 700 samples."""
        total_union = set()
        total_quota = set()
        for ds_id in ["banking77", "clinc150", "anli_r2r3", "strategyqa"]:
            entry = self.masks["by_dataset_role"][f"{ds_id}:holdout"]
            total_union.update(entry["union_35"])
            total_quota.update(entry["b06_sampled_sids"])

        self.assertEqual(len(total_union), 2274)
        self.assertEqual(len(total_quota), 700)

    def test_04_primary_comparisons_statistical_significance(self):
        """Verify primary contrast behavior after full HTTP_402 DeepSeek repair on Holdout."""
        primary_comp = self.results["summary"]["primary_comparisons"]

        # GPT-6.1-Sol: raw negative degradation (95% CI strictly < 0, raw p = 0.047)
        gpt_comp = primary_comp["gpt-6.1-sol"]
        self.assertLess(gpt_comp["macro_diff_pp"], 0.0)
        self.assertLess(gpt_comp["ci_95_pp"][1], 0.0)
        self.assertLess(gpt_comp["raw_p_value"], 0.05)

        # DeepSeek-v4.1-Flash: mild neutral-to-negative contrast crossing zero (CI crosses 0)
        ds_comp = primary_comp["deepseek-v4.1-flash"]
        self.assertLess(ds_comp["macro_diff_pp"], 0.0)
        self.assertGreater(ds_comp["ci_95_pp"][1], 0.0)
        self.assertFalse(ds_comp["significant_holm"])

    def test_05_cost_ledger_and_audit(self):
        """Verify cost ledger adheres to actual platform billing and stays within $30 budget."""
        self.assertEqual(self.cost_ledger["batch_id"], "B07")
        self.assertEqual(self.cost_ledger["total_requests"], 24128)
        self.assertEqual(self.cost_ledger["structural_aliases"], 1960)
        self.assertLess(self.cost_ledger["total_cost_usd"], 30.0)
        self.assertGreater(self.cost_ledger["total_cost_usd"], 20.0)
        self.assertEqual(self.audit["status"], "completed")
        self.assertTrue(self.audit["pre_registration_verified"])


if __name__ == "__main__":
    unittest.main()
