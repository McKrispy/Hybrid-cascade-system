"""CLI entry point for offline reproduction of HybridCascade experimental batches.

Timestamp: 2026-11-11 11:45 (Asia/Shanghai)

Usage:
    python dev/reproduce.py --batch all
    python dev/reproduce.py --batch B07
    python dev/reproduce.py --batch cost
    python dev/reproduce.py --batch figures
    python dev/reproduce.py --batch sandbox-figures
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.reporting.replayer import OfflineBatchReplayer
from src.reporting.report_generator import generate_reproduction_report
from src.reporting.tables import ResultsTableFormatter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline reproduction runner for HybridCascade empirical batches."
    )
    parser.add_argument(
        "--batch",
        choices=["all", "B01", "B02", "B03", "B04", "B05", "B06", "B07", "cost", "figures", "sandbox-figures"],
        default="all",
        help="Target batch to replay offline (default: all).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_ROOT / "dev" / "reproduced",
        help="Directory to save generated report and artifacts.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed console table output.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch == "figures":
        from src.reporting.figures import render_paper_figures

        render_paper_figures(args.output_dir)
        return 0

    if args.batch == "sandbox-figures":
        from src.reporting.sandbox_figures import render_sandbox_figures

        render_sandbox_figures(args.output_dir)
        return 0

    replayer = OfflineBatchReplayer()
    formatter = ResultsTableFormatter()

    print("=" * 70)
    print("HybridCascade: OFFLINE RESEARCH BATCH REPLAYER (P02)")
    print("Timestamp: 2026-11-11 (Asia/Shanghai)")
    print("Network Mode: Strictly Offline ($0.00 marginal cost, 0 API calls)")
    print("=" * 70)

    if args.batch == "all":
        print("\n[+] Replaying all batches (B01 through B07) + Cost Ledger...")
        summary = generate_reproduction_report(
            output_dir=args.output_dir,
            replayer=replayer,
            formatter=formatter,
        )

        print(f"\n[+] Status: {summary['status'].upper()}")
        print(f"[+] Total Verification Checks: {summary['total_checks']}")
        print(f"[+] Passed: {summary['checks_passed']} | Failed: {summary['total_checks'] - summary['checks_passed']}")
        print(f"[+] Report Markdown: {summary['report_markdown']}")
        print(f"[+] Metrics JSON:    {summary['metrics_json']}")
        print(f"[+] Audit JSON:      {summary['audit_json']}")

        if not args.quiet:
            print("\n" + "=" * 70)
            print("SUMMARY OF EMPIRICAL TABLES")
            print("=" * 70)
            print(formatter.generate_all_tables_markdown())

        return 0 if summary["status"] == "reproduced_successfully" else 1

    elif args.batch == "cost":
        print("\n[+] Replaying project billing ledger...")
        res = replayer.replay_cost_ledger()
        print(f"Total Requests: {res['total_project_requests']:,}")
        print(f"Total Actual Platform Cost: ${res['total_project_cost_usd']:.4f} USD")
        if not args.quiet:
            print("\n" + formatter.format_table_8_cost_ledger())
        return 0

    else:
        # Replay specific batch
        replay_func = getattr(replayer, f"replay_{args.batch.lower()}")
        print(f"\n[+] Replaying Batch {args.batch}...")
        res = replay_func()
        print(f"Batch: {res['batch_id']}")
        print(f"Status: {res['status']}")
        print(f"Passed: {res['checks_passed']} / {res['total_checks']}")
        if res["discrepancies"]:
            print(f"Discrepancies: {res['discrepancies']}")
            return 1
        return 0


if __name__ == "__main__":
    sys.exit(main())

