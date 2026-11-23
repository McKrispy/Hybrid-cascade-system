"""Unit test suite for Batch P02: Offline reproduction and table formatting.

Timestamp: 2026-11-11 11:45 (Asia/Shanghai)
"""

import json
from pathlib import Path

import pytest

from src.infra.paths import get_project_root
from src.reporting.replayer import OfflineBatchReplayer
from src.reporting.report_generator import generate_reproduction_report
from src.reporting.tables import ResultsTableFormatter


@pytest.fixture
def replayer():
    return OfflineBatchReplayer()


@pytest.fixture
def formatter():
    return ResultsTableFormatter()


def test_offline_replayer_b01_data_and_leakage(replayer):
    """Test Batch B01 offline replay: split counts and strict zero leakage."""
    res = replayer.replay_b01()
    assert res["status"] == "reproduced_successfully"
    assert res["checks_failed"] == 0
    assert res["metrics"]["total_unique_questions"] == 9280
    assert res["metrics"]["zero_leakage_verified"] is True


def test_offline_replayer_b02_baselines(replayer):
    """Test Batch B02 offline replay: single-model accuracies and CLINC tau."""
    res = replayer.replay_b02()
    assert res["status"] == "reproduced_successfully"
    assert res["checks_failed"] == 0
    assert abs(res["metrics"]["clinc_optimal_tau"] - 0.159485) < 1e-4


def test_offline_replayer_b03_calibration(replayer):
    """Test Batch B03 offline replay: logistic strictly outperforms isotonic/raw."""
    res = replayer.replay_b03()
    assert res["status"] == "reproduced_successfully"
    assert res["checks_failed"] == 0
    assert res["metrics"]["selected_method"] == "logistic"


def test_offline_replayer_b04_nested_masks(replayer):
    """Test Batch B04 offline replay: strict subset nestedness and cascade gains."""
    res = replayer.replay_b04()
    assert res["status"] == "reproduced_successfully"
    assert res["checks_failed"] == 0
    assert res["metrics"]["primary_35_union_total"] == 975
    assert abs(res["metrics"]["gpt_p1_q25_net_gain_pp"] - 7.7727) < 1e-3


def test_offline_replayer_b05_dev_pool(replayer):
    """Test Batch B05 offline replay: F1-F0 degradation and candidate coverage."""
    res = replayer.replay_b05()
    assert res["status"] == "reproduced_successfully"
    assert res["checks_failed"] == 0
    pri = res["metrics"]["primary_comparisons"]
    assert pri["gpt-6.1-sol"]["macro_diff_pp"] < 0.0
    assert pri["gpt-6.1-sol"]["significant_holm"] is True


def test_offline_replayer_b06_fr_control(replayer):
    """Test Batch B06 offline replay: effect decomposition on 300 subset."""
    res = replayer.replay_b06()
    assert res["status"] == "reproduced_successfully"
    assert res["checks_failed"] == 0
    m = res["metrics"]["gpt_selection_macro"]
    assert abs(m["macro_f0_accuracy"] - 0.81) < 1e-3
    assert abs(m["macro_fr_accuracy"] - 0.79) < 1e-3
    assert abs(m["macro_f1_accuracy"] - 0.77) < 1e-3


def test_offline_replayer_b07_holdout(replayer):
    """Test Batch B07 offline replay: repaired Holdout contrasts verified."""
    res = replayer.replay_b07()
    assert res["status"] == "reproduced_successfully"
    assert res["checks_failed"] == 0
    pri = res["metrics"]["primary_comparisons"]
    assert pri["gpt-6.1-sol"]["macro_diff_pp"] < 0.0
    assert pri["gpt-6.1-sol"]["raw_p_value"] < 0.05
    assert pri["deepseek-v4.1-flash"]["macro_diff_pp"] < 0.0
    assert pri["deepseek-v4.1-flash"]["significant_holm"] is False


def test_offline_cost_ledger_reconciliation(replayer):
    """Test cost ledger reconciliation: total requests and budget cap."""
    res = replayer.replay_cost_ledger()
    assert res["status"] == "reproduced_successfully"
    assert res["total_project_requests"] > 35000
    assert res["total_project_cost_usd"] < 40.0


def test_offline_replayer_full_suite(replayer):
    """Test full replay_all aggregation across all batches."""
    res = replayer.replay_all()
    assert res["overall_status"] == "reproduced_successfully"
    assert res["checks_failed"] == 0
    assert res["checks_passed"] >= 90


def test_results_table_formatter(formatter):
    """Test formatting of all 8 markdown tables."""
    all_tables = formatter.generate_all_tables_markdown()
    assert "Table 1: Dataset and Group-Disjoint Split Statistics" in all_tables
    assert "Table 2: Single-Model Baseline Metrics" in all_tables
    assert "Table 3: Confidence Calibration Method Comparison" in all_tables
    assert "Table 4: Routing Policies & Cascade Performance" in all_tables
    assert "Table 5: Paired Handoff Contrasts on Development Pool" in all_tables
    assert "Table 6: FR Numerical Prior Control" in all_tables
    assert "Table 7: Holdout Replication Results" in all_tables
    assert "Table 8: Project Cost Accounting" in all_tables


def test_generate_reproduction_report_artifacts(tmp_path):
    """Test generation of reproduction report and json artifacts in output directory."""
    res = generate_reproduction_report(output_dir=tmp_path)
    assert res["status"] == "reproduced_successfully"
    assert res["checks_passed"] >= 90

    report_file = tmp_path / "reproduction_report.md"
    metrics_file = tmp_path / "reproduced_metrics.json"
    audit_file = tmp_path / "reproduction_audit.json"

    assert report_file.exists() and report_file.stat().st_size > 1000
    assert metrics_file.exists() and metrics_file.stat().st_size > 1000
    assert audit_file.exists() and audit_file.stat().st_size > 500

    with open(audit_file, "r", encoding="utf-8") as f:
        audit_data = json.load(f)
    assert audit_data["marginal_cost_usd"] == 0.0
    assert audit_data["external_network_calls"] == 0
    assert audit_data["overall_status"] == "reproduced_successfully"

