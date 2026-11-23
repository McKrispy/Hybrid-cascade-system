"""Check the frozen B08 inputs and render the three streaming-track figures."""

import hashlib
import json
from dataclasses import dataclass
from decimal import Decimal
from importlib import import_module
from pathlib import Path

from src.reporting.figures import DEV, read_json

ARCHIVE = DEV / "archives" / "track2_alpaca_real"
ARMS = ("F0", "F1", "C0", "C1")
BACKENDS = {
    "s2_decisions_gpt": "GPT-6.1-Sol",
    "s2_decisions_deepseek": "DeepSeek-v4.1-Flash",
}
SANDBOX_RENDERERS = {
    "fig1_portfolio_nav": "render_nav",
    "fig2_anchoring_pruning": "render_matrix",
    "fig3_gatekeeper_funnel": "render_funnel",
}


@dataclass(frozen=True)
class SandboxArchive:
    records: list[dict]
    ledger: list[dict]
    checkpoint: dict
    config: dict
    telemetry: dict


def _read_jsonl(path):
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def load_sandbox_archive():
    """Verify the transfer checksum, tick alignment, and checkpoint totals."""
    files = {
        "records.jsonl", "ledger.jsonl", "checkpoints.json",
        "config.json", "telemetry-pre-stop.json", "container-state.txt",
    }
    digests = {}
    for line in (ARCHIVE / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        digests[name] = digest.lower()
    if set(digests) != files:
        raise ValueError("The frozen B08 checksum list is incomplete or unexpected")
    for name, expected in digests.items():
        if hashlib.sha256((ARCHIVE / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen B08 checksum mismatch for {name}")

    records = _read_jsonl(ARCHIVE / "records.jsonl")
    ledger = _read_jsonl(ARCHIVE / "ledger.jsonl")
    checkpoint = read_json(ARCHIVE / "checkpoints.json")
    config = read_json(ARCHIVE / "config.json")
    telemetry = read_json(ARCHIVE / "telemetry-pre-stop.json")
    seen = set()
    for tick, charge in zip(records, ledger, strict=True):
        sid = tick["tick_id"]
        if sid in seen or sid != charge["tick_id"]:
            raise ValueError(f"Duplicate or unmatched tick_id {sid}")
        seen.add(sid)
        if tick["was_escalated"] != charge["was_escalated"] or tick["tick_cost_usd"] != charge["cost_usd"]:
            raise ValueError(f"Ledger disagrees with tick {sid}")
        expected_arms = set(ARMS) if tick["was_escalated"] else set()
        if any(set(tick[key]) != expected_arms for key in BACKENDS):
            raise ValueError(f"Unexpected handoff arms at {sid}")

    if len(records) != checkpoint["processed_ticks_count"]:
        raise ValueError("B08 checkpoint and records disagree on total ticks")
    if sum(row["was_escalated"] for row in records) != checkpoint["escalated_ticks_count"]:
        raise ValueError("B08 checkpoint and records disagree on escalated ticks")
    cost = sum(Decimal(str(row["cost_usd"])) for row in ledger)
    if cost != Decimal(str(checkpoint["total_spent_usd"])):
        raise ValueError("B08 checkpoint and ledger disagree on cumulative spending")
    return SandboxArchive(records, ledger, checkpoint, config, telemetry)


def render_sandbox_figures(output_dir):
    """Rebuild all three B08 figures from the frozen local archive."""
    for folder, script in SANDBOX_RENDERERS.items():
        module = import_module(f"figures.{folder}.{script}")
        module.render_figure(Path(output_dir) / folder)
