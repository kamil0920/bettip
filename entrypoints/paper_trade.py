#!/usr/bin/env python
"""
Paper Trading CLI with CLV Tracking

Daily workflow for validating your betting model with real predictions.

Commands:
    dashboard    Show current status and CLV performance
    add          Add today's predictions to tracking
    close        Record closing odds for pending predictions
    result       Record match results
    report       Generate CLV analysis report

Usage:
    # Morning: Add today's predictions
    python entrypoints/paper_trade.py add

    # Before kickoff: Record closing odds
    python entrypoints/paper_trade.py close

    # After matches: Record results
    python entrypoints/paper_trade.py result

    # Anytime: View dashboard
    python entrypoints/paper_trade.py dashboard

    # Weekly: Generate report
    python entrypoints/paper_trade.py report
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
from datetime import datetime
import pandas as pd

from src.ml.live_tracker import LiveTracker
from src.ml.clv_tracker import CLVTracker


def cmd_dashboard(args):
    """Show the live tracking dashboard."""
    tracker = LiveTracker()
    tracker.show_dashboard()


def cmd_add(args):
    """Add predictions from file."""
    tracker = LiveTracker()

    predictions_file = args.file or "experiments/outputs/next_round_predictions.json"

    if not Path(predictions_file).exists():
        print(f"Error: Predictions file not found: {predictions_file}")
        print("\nRun predictions first:")
        print("  python experiments/predict_next_round.py")
        return

    count = tracker.add_predictions_from_file(
        predictions_file,
        only_threshold_met=args.threshold_only
    )

    print(f"\nAdded {count} predictions to tracking")
    tracker.show_dashboard()


def cmd_close(args):
    """Record closing odds for pending predictions."""
    tracker = LiveTracker()

    # Get pending predictions
    pending = tracker.get_pending_predictions()

    if pending.empty:
        print("No pending predictions found.")
        return

    print("\nPending predictions needing closing odds:")
    print("-" * 60)

    for idx, row in pending.iterrows():
        key = f"{row['home_team']}_{row['away_team']}_{row.get('bet_type', '')}_{str(row['match_date'])[:10]}"
        print(f"\n{row['home_team']} vs {row['away_team']}")
        print(f"  Bet type: {row.get('bet_type', 'N/A')}")
        print(f"  Opening odds: {row['market_odds_at_prediction']:.2f}")

        if args.interactive:
            closing = input("  Enter closing odds (or skip): ").strip()
            if closing and closing.lower() != 'skip':
                try:
                    closing_odds = float(closing)
                    tracker.record_closing_odds(key, closing_odds)
                except ValueError:
                    print("  Invalid odds, skipping...")

    if not args.interactive:
        print("\nRun with --interactive to enter closing odds manually:")
        print("  python entrypoints/paper_trade.py close --interactive")


def cmd_result(args):
    """Record match results."""
    tracker = LiveTracker()

    # Get predictions with closing odds but no result
    needs_result = [
        p for p in tracker.predictions.values()
        if p['status'] in ('pending', 'has_closing') and p.get('closing_odds')
    ]

    if not needs_result:
        print("No predictions awaiting results.")
        return

    print("\nMatches awaiting results:")
    print("-" * 60)

    for pred in needs_result:
        key = pred['key']
        print(f"\n{pred['home_team']} vs {pred['away_team']}")
        print(f"  Bet type: {pred['bet_type']}")
        print(f"  Our prob: {pred['our_probability']:.1%}")
        print(f"  CLV: {pred.get('clv', 0) * 100:+.2f}%")

        if args.interactive:
            result = input("  Did bet win? (y/n/skip): ").strip().lower()
            if result == 'y':
                tracker.record_result(key, won=True)
            elif result == 'n':
                tracker.record_result(key, won=False)
            else:
                print("  Skipped")

    if not args.interactive:
        print("\nRun with --interactive to enter results:")
        print("  python entrypoints/paper_trade.py result --interactive")

    tracker.show_dashboard()


def cmd_report(args):
    """Generate CLV analysis report."""
    clv_tracker = CLVTracker()
    clv_tracker.print_clv_report()

    # Also show by bet type and league
    print("\n" + "=" * 60)
    print("CLV BY BET TYPE")
    print("=" * 60)
    by_type = clv_tracker.get_clv_by_bet_type()
    if not by_type.empty:
        print(by_type.to_string())
    else:
        print("No CLV data by bet type yet.")

    print("\n" + "=" * 60)
    print("CLV BY LEAGUE")
    print("=" * 60)
    by_league = clv_tracker.get_clv_by_league()
    if not by_league.empty:
        print(by_league.to_string())
    else:
        print("No CLV data by league yet.")

    # Export to CSV
    if args.export:
        output = clv_tracker.export_to_csv()
        print(f"\nExported to: {output}")


def cmd_quick_add(args):
    """Quick add a single prediction."""
    tracker = LiveTracker()

    print("Add a prediction manually:")
    home = input("Home team: ").strip()
    away = input("Away team: ").strip()
    bet_type = input("Bet type (away_win/btts/home_win): ").strip()
    prob = float(input("Your probability (0-1): ").strip())
    odds = float(input("Market odds: ").strip())
    league = input("League: ").strip()
    date = input("Match date (YYYY-MM-DD HH:MM): ").strip()

    tracker.add_prediction(
        match_id=f"{home}_{away}",
        home_team=home,
        away_team=away,
        match_date=date,
        league=league,
        bet_type=bet_type,
        our_probability=prob,
        market_odds=odds,
        threshold=0.6,
        meets_threshold=prob > 0.6,
        edge=prob - (1/odds)
    )

    print(f"\nAdded prediction: {home} vs {away} - {bet_type}")
    tracker.show_dashboard()


def main():
    parser = argparse.ArgumentParser(
        description="Paper Trading CLI with CLV Tracking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python entrypoints/paper_trade.py dashboard
  python entrypoints/paper_trade.py add
  python entrypoints/paper_trade.py close --interactive
  python entrypoints/paper_trade.py result --interactive
  python entrypoints/paper_trade.py report --export
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Dashboard
    dash_parser = subparsers.add_parser('dashboard', help='Show tracking dashboard')

    # Add predictions
    add_parser = subparsers.add_parser('add', help='Add predictions from file')
    add_parser.add_argument('--file', '-f', help='Predictions JSON file')
    add_parser.add_argument('--threshold-only', '-t', action='store_true',
                           help='Only add predictions that meet threshold')

    # Record closing odds
    close_parser = subparsers.add_parser('close', help='Record closing odds')
    close_parser.add_argument('--interactive', '-i', action='store_true',
                             help='Interactively enter closing odds')

    # Record results
    result_parser = subparsers.add_parser('result', help='Record match results')
    result_parser.add_argument('--interactive', '-i', action='store_true',
                              help='Interactively enter results')

    # Generate report
    report_parser = subparsers.add_parser('report', help='Generate CLV report')
    report_parser.add_argument('--export', '-e', action='store_true',
                              help='Export to CSV')

    # Quick add single prediction
    quick_parser = subparsers.add_parser('quick', help='Quickly add a single prediction')

    args = parser.parse_args()

    if not args.command:
        # Default to dashboard
        cmd_dashboard(args)
        return

    commands = {
        'dashboard': cmd_dashboard,
        'add': cmd_add,
        'close': cmd_close,
        'result': cmd_result,
        'report': cmd_report,
        'quick': cmd_quick_add,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
