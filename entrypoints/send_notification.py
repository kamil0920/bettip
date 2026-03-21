#!/usr/bin/env python3
"""CLI dispatcher for sending Telegram notifications.

Designed to be called from GitHub Actions workflow YAML:

    uv run python entrypoints/send_notification.py match_day_digest \
        --rec-file data/05-recommendations/rec_20260321_sniper.csv \
        --schedule-file data/06-prematch/today_schedule.json

    uv run python entrypoints/send_notification.py pre_kickoff \
        --deltas-file data/05-recommendations/pre_kickoff_deltas.json

    uv run python entrypoints/send_notification.py weekly_report \
        --performance-file validation/live_performance.json \
        --drift-file validation/drift_report.json \
        --weekend-file data/weekend/reports/weekend_report.json

    uv run python entrypoints/send_notification.py post_optimization \
        --report-file validation/comparison_report.json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path for src imports
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.notifications import (
    TelegramNotifier,
    format_match_day_digest,
    format_post_optimization,
    format_pre_kickoff_update,
    format_weekly_report,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _load_json(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        logger.warning("File not found: %s", p)
        return {}
    try:
        with open(p) as f:
            return json.load(f)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("Invalid JSON in %s: %s", p, exc)
        return {}


def _load_csv_recs(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        logger.warning("Rec file not found: %s", p)
        return []
    with open(p) as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def cmd_match_day_digest(args: argparse.Namespace) -> int:
    recs: list[dict] = []
    for rec_path in args.rec_file:
        recs.extend(_load_csv_recs(rec_path))

    schedule = _load_json(args.schedule_file) if args.schedule_file else {}
    total_matches = schedule.get("total_matches", 0)

    drift_signals: list[dict] = []
    if args.drift_file:
        drift_data = _load_json(args.drift_file)
        drift_signals = drift_data.get("signals", drift_data.get("markets", []))

    parts = format_match_day_digest(
        recs, drift_signals=drift_signals, total_matches=total_matches
    )

    notifier = TelegramNotifier()
    notifier.send_parts(parts)
    return 0


def cmd_pre_kickoff(args: argparse.Namespace) -> int:
    data = _load_json(args.deltas_file)
    deltas = data if isinstance(data, list) else data.get("deltas", [])

    parts = format_pre_kickoff_update(deltas)
    if not parts:
        logger.info("No significant deltas — skipping notification")
        return 0

    notifier = TelegramNotifier()
    notifier.send_parts(parts)
    return 0


def cmd_weekly_report(args: argparse.Namespace) -> int:
    performance = _load_json(args.performance_file) if args.performance_file else None
    drift = _load_json(args.drift_file) if args.drift_file else None
    weekend = _load_json(args.weekend_file) if args.weekend_file else None

    parts = format_weekly_report(
        performance=performance, drift=drift, weekend=weekend
    )

    notifier = TelegramNotifier()
    notifier.send_parts(parts)
    return 0


def cmd_post_optimization(args: argparse.Namespace) -> int:
    report = _load_json(args.report_file)

    parts = format_post_optimization(report)

    notifier = TelegramNotifier()
    notifier.send_parts(parts)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Send formatted Telegram notifications for bettip pipeline"
    )
    sub = parser.add_subparsers(dest="command")

    # match_day_digest
    p_digest = sub.add_parser("match_day_digest", help="Morning match-day predictions")
    p_digest.add_argument(
        "--rec-file",
        nargs="+",
        required=True,
        help="Path(s) to recommendation CSV(s)",
    )
    p_digest.add_argument("--schedule-file", help="Path to today_schedule.json")
    p_digest.add_argument("--drift-file", help="Path to drift signals JSON")

    # pre_kickoff
    p_pre = sub.add_parser("pre_kickoff", help="Pre-kickoff probability delta updates")
    p_pre.add_argument(
        "--deltas-file",
        required=True,
        help="Path to deltas JSON (list or {deltas: [...]})",
    )

    # weekly_report
    p_weekly = sub.add_parser("weekly_report", help="Consolidated weekly report")
    p_weekly.add_argument("--performance-file", help="Path to live_performance.json")
    p_weekly.add_argument("--drift-file", help="Path to drift_report.json")
    p_weekly.add_argument("--weekend-file", help="Path to weekend report JSON")

    # post_optimization
    p_post = sub.add_parser("post_optimization", help="Post-optimization validation")
    p_post.add_argument(
        "--report-file",
        required=True,
        help="Path to comparison_report.json",
    )

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 1

    dispatch = {
        "match_day_digest": cmd_match_day_digest,
        "pre_kickoff": cmd_pre_kickoff,
        "weekly_report": cmd_weekly_report,
        "post_optimization": cmd_post_optimization,
    }
    return dispatch[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
