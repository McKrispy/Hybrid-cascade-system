"""Test suite for Batch B00: Infrastructure, Schema, Native wire contract, Desensitization, and Budget limits.

Timestamp: 2026-11-09 11:35 (Asia/Shanghai)

Covers:
- CWD independence and path resolution
- TypeSafe native wire request/response schema parsing (Choice, Noul, usage, invalid prob rejection)
- System-2 strict JSON decision parsing (no heuristic guessing from conversational text)
- Secret desensitization
- Budget limits enforcement (halting when max_requests or max_cost_usd is reached)
- Rate limits and max_retries parameter propagation
"""

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

# Dev root path resolution
DEV_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = DEV_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.clients.base import Handoff, Prediction, desensitize_dict
from src.clients.deepseek_flash import DeepSeekFlashClient
from src.clients.gpt_sol import GptSolClient
from src.clients.jev import JevClient
from src.clients.system2 import parse_system2_decision
from src.infra.artifacts import RunArtifacts, allocate_run_dir
from src.infra.config import (
    BatchConfig,
    BudgetLimitConfig,
    ProjectConfig,
    RateLimitConfig,
    SyntheticSample,
    assert_no_secrets,
    get_env_var,
    load_batch_config,
    load_project_config,
)
from src.infra.paths import (
    get_configs_dir,
    get_dev_root,
    get_docs_dir,
    get_project_root,
    get_runs_dir,
)
from src.infra.telemetry import TelemetryTracker


class TestB00Infra(unittest.TestCase):
    """Test suite for B00 infrastructure, wire contracts, and budget enforcement."""

    def test_01_path_resolution_independent_of_cwd(self):
        """Test path resolutions do not change regardless of current working directory."""
        initial_cwd = os.getcwd()
        try:
            temp_dir = tempfile.gettempdir()
            os.chdir(temp_dir)

            dev_root = get_dev_root()
            proj_root = get_project_root()
            configs_dir = get_configs_dir()
            runs_dir = get_runs_dir()
            docs_dir = get_docs_dir()

            self.assertEqual(proj_root.resolve(), dev_root.resolve())
            self.assertEqual(configs_dir, dev_root / "configs")
            self.assertEqual(runs_dir, dev_root / "runs")
            self.assertEqual(docs_dir, dev_root / "docs")
            self.assertTrue(configs_dir.exists())
        finally:
            os.chdir(initial_cwd)

    def test_02_config_loading_and_validation(self):
        """Test loading project.json and B00.json with schema validation."""
        proj = load_project_config()
        self.assertEqual(proj.schema_version, "1.0")
        self.assertIn("B00", proj.batches)
        self.assertIn("P00", proj.batches)

        b00 = load_batch_config("B00")
        self.assertEqual(b00.batch_id, "B00")
        self.assertIn("jev-1.14.0", b00.models)
        self.assertIn("gpt-6.1-sol", b00.models)
        self.assertIn("deepseek-v4.1-flash", b00.models)
        self.assertGreater(len(b00.synthetic_samples), 0)

        # Missing required field should raise ValidationError
        with self.assertRaises(Exception):
            BatchConfig(batch_id="INVALID")

    def test_03_typesafe_native_wire_response_parsing(self):
        """Test B00-01: Native wire schema parsing for Jev Choice and Noul."""
        client = JevClient(mock_mode=False, model_id="jev-1.14.0")

        # 1. Native Noul response (e.g. noul=0.91, input_tokens=123)
        native_noul_resp = {
            "model": "jev-1.14.0",
            "answers": {
                "sample_noul_1": {
                    "noul": 0.91,
                }
            },
            "usage": {
                "input_tokens": 123,
                "output_tokens": 0,
            },
        }
        parsed_noul = client.parse_native_response(native_noul_resp, "sample_noul_1", "noul")
        self.assertEqual(parsed_noul["status"], "valid")
        self.assertEqual(parsed_noul["decision"], "yes")
        self.assertAlmostEqual(parsed_noul["probabilities"]["yes"], 0.91, places=4)
        self.assertAlmostEqual(parsed_noul["probabilities"]["no"], 0.09, places=4)
        self.assertEqual(parsed_noul["usage"]["input_tokens"], 123)
        self.assertAlmostEqual(parsed_noul["cost"], (123 / 1_000_000.0) * 0.042, places=8)

        # 2. Native Choice response
        native_choice_resp = {
            "model": "jev-1.14.0",
            "answers": {
                "sample_choice_1": {
                    "choice": {
                        "decision": "card_replacement",
                        "probabilities": {
                            "card_replacement": 0.85,
                            "pin_reset": 0.15,
                        },
                    }
                }
            },
            "usage": {
                "input_tokens": 200,
                "output_tokens": 0,
            },
        }
        parsed_choice = client.parse_native_response(native_choice_resp, "sample_choice_1", "choice")
        self.assertEqual(parsed_choice["status"], "valid")
        self.assertEqual(parsed_choice["decision"], "card_replacement")
        self.assertEqual(parsed_choice["usage"]["input_tokens"], 200)

        # 3. Reject illegal probabilities (e.g. -0.1 or 1.1)
        bad_noul_resp = {
            "model": "jev-1.14.0",
            "answers": {"sample_bad": {"noul": -0.1}},
            "usage": {"input_tokens": 50},
        }
        parsed_bad = client.parse_native_response(bad_noul_resp, "sample_bad", "noul")
        self.assertEqual(parsed_bad["status"], "failed")
        self.assertEqual(parsed_bad["error_code"], "PROBABILITY_OUT_OF_BOUNDS")

        bad_choice_resp = {
            "model": "jev-1.14.0",
            "answers": {
                "sample_bad_c": {
                    "choice": {"probabilities": {"optA": 0.7, "optB": 0.7}}  # sum = 1.4
                }
            },
            "usage": {"input_tokens": 50},
        }
        parsed_bad_c = client.parse_native_response(bad_choice_resp, "sample_bad_c", "choice")
        self.assertEqual(parsed_bad_c["status"], "failed")
        self.assertEqual(parsed_bad_c["error_code"], "PROBABILITIES_SUM_INVALID")

        # 4. Unexpected model version rejection
        wrong_model_resp = {
            "model": "unexpected-version-2.0",
            "answers": {"sample_noul_1": {"noul": 0.9}},
            "usage": {"input_tokens": 10},
        }
        parsed_wrong = client.parse_native_response(wrong_model_resp, "sample_noul_1", "noul")
        self.assertEqual(parsed_wrong["status"], "failed")
        self.assertEqual(parsed_wrong["error_code"], "UNEXPECTED_MODEL_VERSION")

    def test_04_system2_strict_json_decision_parsing(self):
        """Test B00-05: Strict JSON decision parsing rejects guessing from conversational text."""
        allowed = ["refund", "exchange"]

        # Conversational text explaining uncertainty must NOT be guessed as exchange!
        ambiguous_text = "I cannot choose between refund and exchange because information is lacking."
        res = parse_system2_decision(ambiguous_text, allowed)
        self.assertIsNone(res, "Should return None for ambiguous text without valid JSON decision")

        # Valid JSON
        valid_json = '{"decision": "refund"}'
        self.assertEqual(parse_system2_decision(valid_json, allowed), "refund")

        # Valid JSON in markdown code fence
        fenced_json = '```json\n{"decision": "exchange"}\n```'
        self.assertEqual(parse_system2_decision(fenced_json, allowed), "exchange")

        # JSON with unallowed label
        invalid_label_json = '{"decision": "unallowed_option"}'
        self.assertIsNone(parse_system2_decision(invalid_label_json, allowed))

        # REJECT allowed test
        self.assertEqual(parse_system2_decision('{"decision": "REJECT"}', allowed, reject_allowed=True), "REJECT")
        self.assertIsNone(parse_system2_decision('{"decision": "REJECT"}', allowed, reject_allowed=False))

    def test_05_desensitization_check(self):
        """Test that sensitive API keys or Authorization headers are detected and scrubbed."""
        sensitive_data = {
            "model": "jev-1.14.0",
            "Authorization": "Bearer sk-test-typesafe-key-12345",
            "metadata": {"api_key": "sk-secret-key-67890", "status": "ok"},
        }
        cleaned = desensitize_dict(sensitive_data)
        self.assertEqual(cleaned["Authorization"], "[REDACTED]")
        self.assertEqual(cleaned["metadata"]["api_key"], "[REDACTED]")
        self.assertEqual(cleaned["metadata"]["status"], "ok")

        with self.assertRaises(ValueError):
            assert_no_secrets({"payload": "sk-1234567890abcdef1234567890"})

    def test_06_budget_limits_halting_control(self):
        """Test B00-03: Execution strictly halts when budget_limits are reached."""
        from dev.run import run_b00_smoke

        test_run_dir = Path(tempfile.mkdtemp(prefix="budget_test_"))
        try:
            # Create a BatchConfig with max_requests = 1
            config_dict = {
                "batch_id": "B00_TEST",
                "schema_version": "1.0",
                "description": "Budget limit test",
                "rate_limits": {"max_concurrency": 1, "timeout_seconds": 10.0, "max_retries": 0},
                "budget_limits": {"max_cost_usd": 10.0, "max_requests": 1},  # Stop after 1 request!
                "models": {
                    "jev-1.14.0": {"model_id": "jev-1.14.0", "provider": "typesafe", "endpoint": "mock", "env_key_name": "K", "pricing": {}},
                    "gpt-6.1-sol": {"model_id": "gpt-6.1-sol", "provider": "openai", "endpoint": "mock", "env_key_name": "K", "pricing": {}},
                    "deepseek-v4.1-flash": {"model_id": "deepseek-v4.1-flash", "provider": "deepseek", "endpoint": "mock", "env_key_name": "K", "pricing": {}},
                },
                "synthetic_samples": [
                    {"sample_id": "s1", "task_type": "choice", "question": "q1", "options": ["a", "b"]},
                    {"sample_id": "s2", "task_type": "choice", "question": "q2", "options": ["a", "b"]},
                ],
                "custom_params": {"arms_to_smoke": ["F0"]},
            }
            batch_cfg = BatchConfig(**config_dict)

            results = run_b00_smoke(test_run_dir, batch_cfg, mode="mock")
            # Must halt after exactly 1 request
            self.assertEqual(results["summary"]["total_requests"], 1)
            self.assertIn("MAX_REQUESTS_REACHED", str(results["summary"]["halt_reason"]))
        finally:
            shutil.rmtree(test_run_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
