#!/usr/bin/env python3
"""
CLV Workflow - Import recommendations and manage CLV tracking.

Called by prematch-intelligence GitHub Actions workflow.

Usage:
    # Import recommendations CSV into CLV tracker
    python experiments/clv_workflow.py import --file data/05-recommendations/rec_20260202_001.csv

    # Show CLV summary
    python experiments/clv_workflow.py summary
"""
import argparse
import logging
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd

from src.ml.clv_tracker import CLVTracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CLV_OUTPUT_DIR = str(project_root / "data" / "04-predictions" / "clv_tracking")


def import_recommendations(filepath: str) -> None:
    """Import a recommendations CSV into the CLV tracker."""
    path = Path(filepath)
    if not path.exists():
        logger.error(f"File not found: {filepath}")
        sys.exit(1)

    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} recommendations from {path.name}")

    tracker = CLVTracker(output_dir=CLV_OUTPUT_DIR)
    imported = 0

    for _, row in df.iterrows():
        fixture_id = str(row.get("fixture_id", ""))
        market = str(row.get("market", "")).lower().replace(".", "").replace("_", "")
        # Normalize market names: OVER_2.5 -> over25, HOME_WIN -> home_win, etc.
        market_map = {
            "over25": "over25",
            "under25": "under25",
            "homewin": "home_win",
            "awaywin": "away_win",
            "btts": "btts",
            "corners": "corners",
            "fouls": "fouls",
            "shots": "shots",
            "cards": "cards",
        }
        bet_type = market_map.get(market, market)

        key = f"{fixture_id}_{bet_type}"
        if key in tracker.predictions:
            continue

        odds = row.get("odds", 0)
        our_prob = row.get("our_prob", 0)

        if not fixture_id or odds <= 0:
            continue

        tracker.record_prediction(
            match_id=fixture_id,
            home_team=str(row.get("home_team", "")),
            away_team=str(row.get("away_team", "")),
            match_date=str(row.get("start_time", "")),
            league="",
            bet_type=bet_type,
            our_probability=float(our_prob),
            our_odds=float(odds),
            market_odds=float(odds),
            prediction_time=str(row.get("created_at", "")),
        )
        imported += 1

    tracker.save_history()
    logger.info(f"Imported {imported} new predictions ({len(tracker.predictions)} total tracked)")


def show_summary() -> None:
    """Display CLV summary statistics."""
    tracker = CLVTracker(output_dir=CLV_OUTPUT_DIR)
    summary = tracker.get_clv_summary()

    print("\n" + "=" * 50)
    print("CLV TRACKING SUMMARY")
    print("=" * 50)

    print(f"Total predictions: {summary.get('total_predictions', 0)}")
    print(f"With closing odds: {summary.get('with_closing_odds', 0)}")
    print(f"Settled: {summary.get('settled', 0)}")

    if summary.get("with_closing_odds", 0) > 0:
        print(f"\nAvg CLV: {summary.get('avg_clv', 0):+.2f}%")
        print(f"Median CLV: {summary.get('median_clv', 0):+.2f}%")
        print(f"Positive CLV rate: {summary.get('positive_clv_rate', 0):.1f}%")

    if summary.get("settled", 0) > 0:
        print(f"\nWin rate: {summary.get('win_rate', 0):.1f}%")
        print(f"Total profit: {summary.get('total_profit', 0):+.2f} units")
        print(f"ROI: {summary.get('roi', 0):+.1f}%")

    print("=" * 50)


def main() -> None:
    parser = argparse.ArgumentParser(description="CLV workflow management")
    subparsers = parser.add_subparsers(dest="command", required=True)

    import_parser = subparsers.add_parser("import", help="Import recommendations into CLV tracker")
    import_parser.add_argument("--file", required=True, help="Path to recommendations CSV")

    subparsers.add_parser("summary", help="Show CLV summary")

    args = parser.parse_args()

    if args.command == "import":
        import_recommendations(args.file)
    elif args.command == "summary":
        show_summary()


if __name__ == "__main__":
    main()
