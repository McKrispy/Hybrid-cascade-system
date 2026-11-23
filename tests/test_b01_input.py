"""Tests for HybridCascade Batch B01: Input data ingestion, grouping, role splitting, and loader boundaries.

Timestamp: 2026-11-09 13:22 (Asia/Shanghai)
"""

import json
import unittest
from pathlib import Path

from src.infra.paths import get_data_dir, get_dev_root
from src.input.loader import load_gold_records, load_inputs
from src.input.splits import GoldRecord, SampleInput
from src.input.tasks import load_label_schemas


class TestBatchB01Input(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_dir = get_data_dir()
        cls.manifest_path = cls.data_dir / "manifest.json"
        cls.schemas_path = cls.data_dir / "label_schemas.json"

        if not cls.manifest_path.exists():
            raise unittest.SkipTest("manifest.json not found. Run dev/run.py --batch B01 first.")

        with open(cls.manifest_path, "r", encoding="utf-8") as f:
            cls.manifest = json.load(f)

        cls.label_schemas = load_label_schemas(cls.schemas_path)

    def test_01_total_sample_counts_and_role_quotas(self):
        """Verify total 9,280 questions and exact role quotas per dataset."""
        self.assertEqual(self.manifest["total_unique_questions"], 9280)
        self.assertEqual(self.manifest["role_counts"]["calibration"], 1856)
        self.assertEqual(self.manifest["role_counts"]["selection"], 928)
        self.assertEqual(self.manifest["role_counts"]["holdout"], 6496)

        datasets = self.manifest["datasets"]
        self.assertEqual(datasets["banking77"]["total_questions"], 3080)
        self.assertEqual(datasets["clinc150"]["total_questions"], 2000)
        self.assertEqual(datasets["anli_r2r3"]["total_questions"], 2200)
        self.assertEqual(datasets["strategyqa"]["total_questions"], 2000)

        # Banking77 20/10/70
        self.assertEqual(datasets["banking77"]["roles"]["calibration"]["count"], 616)
        self.assertEqual(datasets["banking77"]["roles"]["selection"]["count"], 308)
        self.assertEqual(datasets["banking77"]["roles"]["holdout"]["count"], 2156)

        # CLINC 20/10/70
        self.assertEqual(datasets["clinc150"]["roles"]["calibration"]["count"], 400)
        self.assertEqual(datasets["clinc150"]["roles"]["selection"]["count"], 200)
        self.assertEqual(datasets["clinc150"]["roles"]["holdout"]["count"], 1400)

        # ANLI 20/10/70
        self.assertEqual(datasets["anli_r2r3"]["roles"]["calibration"]["count"], 440)
        self.assertEqual(datasets["anli_r2r3"]["roles"]["selection"]["count"], 220)
        self.assertEqual(datasets["anli_r2r3"]["roles"]["holdout"]["count"], 1540)

        # StrategyQA 20/10/70
        self.assertEqual(datasets["strategyqa"]["roles"]["calibration"]["count"], 400)
        self.assertEqual(datasets["strategyqa"]["roles"]["selection"]["count"], 200)
        self.assertEqual(datasets["strategyqa"]["roles"]["holdout"]["count"], 1400)

    def test_02_strict_role_and_group_disjointness(self):
        """Verify that sample_ids and group_ids NEVER leak across calibration, selection, or holdout."""
        role_sample_ids = {"calibration": set(), "selection": set(), "holdout": set()}
        role_group_ids = {"calibration": set(), "selection": set(), "holdout": set()}

        for role in ["calibration", "selection", "holdout"]:
            role_dir = self.data_dir / role
            for jsonl_file in role_dir.glob("*.jsonl"):
                with open(jsonl_file, "r", encoding="utf-8") as f:
                    for line in f:
                        data = json.loads(line)
                        sample = SampleInput(**data)
                        role_sample_ids[role].add(sample.sample_id)
                        role_group_ids[role].add(sample.group_id)

        # 1. Sample ID pairwise disjointness
        self.assertTrue(role_sample_ids["calibration"].isdisjoint(role_sample_ids["selection"]))
        self.assertTrue(role_sample_ids["calibration"].isdisjoint(role_sample_ids["holdout"]))
        self.assertTrue(role_sample_ids["selection"].isdisjoint(role_sample_ids["holdout"]))

        # 2. Group ID pairwise disjointness (Premise family strictly preserved within role)
        self.assertTrue(
            role_group_ids["calibration"].isdisjoint(role_group_ids["selection"]),
            "Group leakage detected between calibration and selection!",
        )
        self.assertTrue(
            role_group_ids["calibration"].isdisjoint(role_group_ids["holdout"]),
            "Group leakage detected between calibration and holdout!",
        )
        self.assertTrue(
            role_group_ids["selection"].isdisjoint(role_group_ids["holdout"]),
            "Group leakage detected between selection and holdout!",
        )

    def test_03_sample_input_schema_and_zero_gold_leakage(self):
        """Verify that SampleInput records contain NO gold labels, OOS flags, or model answers."""
        forbidden_keys = {"gold_label", "is_oos", "answer", "reason", "facts", "decomposition"}

        for role in ["calibration", "selection", "holdout"]:
            role_dir = self.data_dir / role
            for jsonl_file in role_dir.glob("*.jsonl"):
                with open(jsonl_file, "r", encoding="utf-8") as f:
                    for line in f:
                        raw_data = json.loads(line)
                        for k in forbidden_keys:
                            self.assertNotIn(
                                k,
                                raw_data,
                                f"Forbidden gold/reasoning field '{k}' leaked into {jsonl_file.name}",
                            )
                        # Verify Pydantic validation passes
                        sample = SampleInput(**raw_data)
                        if sample.dataset_id in ["banking77", "clinc150"]:
                            self.assertIn("text", sample.text_fields)
                        elif sample.dataset_id == "anli_r2r3":
                            self.assertIn("premise", sample.text_fields)
                            self.assertIn("hypothesis", sample.text_fields)
                        elif sample.dataset_id == "strategyqa":
                            self.assertIn("question", sample.text_fields)

    def test_04_gold_record_alignment_and_validity(self):
        """Verify GoldRecord files exist in gold/ directory and match SampleInputs 1:1."""
        gold_dir = self.data_dir / "gold"
        self.assertTrue(gold_dir.exists())

        for ds_id in ["banking77", "clinc150", "anli_r2r3", "strategyqa"]:
            schema = self.label_schemas[ds_id]
            valid_labels = set(schema.allowed_labels)
            if schema.reject_allowed and schema.reject_label:
                valid_labels.add(schema.reject_label)

            for role in ["calibration", "selection", "holdout"]:
                gold_file = gold_dir / f"{ds_id}_{role}.jsonl"
                sample_file = self.data_dir / role / f"{ds_id}.jsonl"
                self.assertTrue(gold_file.exists(), f"Missing gold file {gold_file}")

                with open(sample_file, "r", encoding="utf-8") as fs, open(gold_file, "r", encoding="utf-8") as fg:
                    sample_lines = [json.loads(l) for l in fs]
                    gold_lines = [json.loads(l) for l in fg]

                self.assertEqual(
                    len(sample_lines),
                    len(gold_lines),
                    f"Count mismatch between samples and gold in {ds_id} {role}",
                )

                for s, g in zip(sample_lines, gold_lines):
                    self.assertEqual(s["sample_id"], g["sample_id"])
                    self.assertIn(
                        g["gold_label"],
                        valid_labels,
                        f"Invalid gold label '{g['gold_label']}' in {gold_file.name}",
                    )
                    if ds_id == "clinc150" and g["gold_label"] == "REJECT":
                        self.assertTrue(g["is_oos"])

    def test_05_label_schemas_exact_classes(self):
        """Verify label_schemas.json conforms to canonical task definitions."""
        # Banking77: 77 classes
        b77 = self.label_schemas["banking77"]
        self.assertEqual(b77.task_type, "choice")
        self.assertEqual(len(b77.allowed_labels), 77)
        self.assertFalse(b77.reject_allowed)

        # CLINC150: 150 classes + REJECT
        clinc = self.label_schemas["clinc150"]
        self.assertEqual(clinc.task_type, "choice")
        self.assertEqual(len(clinc.allowed_labels), 150)
        self.assertTrue(clinc.reject_allowed)
        self.assertEqual(clinc.reject_label, "REJECT")
        self.assertIn("REJECT", clinc.label_descriptions)

        # ANLI R2/R3: 3 classes
        anli = self.label_schemas["anli_r2r3"]
        self.assertEqual(anli.task_type, "choice")
        self.assertEqual(anli.allowed_labels, ["entailment", "neutral", "contradiction"])

        # StrategyQA: 2 classes
        sq = self.label_schemas["strategyqa"]
        self.assertEqual(sq.task_type, "noul")
        self.assertEqual(sq.allowed_labels, ["yes", "no"])

    def test_06_dev_loader_holdout_access_denied(self):
        """Verify load_inputs strictly blocks holdout access during development."""
        # Allowed dev roles load properly
        cal_samples = list(load_inputs(self.manifest_path, allowed_role="calibration"))
        self.assertEqual(len(cal_samples), 1856)

        sel_samples = list(load_inputs(self.manifest_path, allowed_role="selection"))
        self.assertEqual(len(sel_samples), 928)

        # Holdout attempt MUST raise PermissionError
        with self.assertRaises(PermissionError):
            list(load_inputs(self.manifest_path, allowed_role="holdout"))

        # Explicit b07 override allows loading
        hld_samples = list(
            load_inputs(
                self.manifest_path,
                allowed_role="holdout",
                danger_enable_holdout_for_b07=True,
            )
        )
        self.assertEqual(len(hld_samples), 6496)

        # Invalid role raises ValueError
        with self.assertRaises(ValueError):
            list(load_inputs(self.manifest_path, allowed_role="synthetic"))

    def test_07_b06_sampling_registration(self):
        """Verify B06 sampling quotas (50 cal, 25 sel, 175 hld = 250 per dataset, 1000 total)."""
        b06_reg = self.manifest["b06_sampling_registration"]
        self.assertEqual(len(b06_reg), 4)

        total_b06 = 0
        for ds_id, role_candidates in b06_reg.items():
            self.assertEqual(len(role_candidates["calibration"]), 50)
            self.assertEqual(len(role_candidates["selection"]), 25)
            self.assertEqual(len(role_candidates["holdout"]), 175)
            dataset_total = sum(len(c) for c in role_candidates.values())
            self.assertEqual(dataset_total, 250)
            total_b06 += dataset_total

        self.assertEqual(total_b06, 1000)


if __name__ == "__main__":
    unittest.main()
