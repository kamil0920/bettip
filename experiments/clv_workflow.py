#!/usr/bin/env python
"""
CLV Tracking Workflow

Unified script to manage CLV tracking:
1. Import recommendations into tracker
2. Record closing odds before matches
3. Record results after matches
4. Generate CLV reports

Usage:
    # Step 1: Import recommendations
    python experiments/clv_workflow.py import --file experiments/outputs/recommendations_jan14_19.csv

    # Step 2: Before matches - record closing odds
    python experiments/clv_workflow.py closing-odds

    # Step 3: After matches - record results
    python experiments/clv_workflow.py results

    # View dashboard anytime
    python experiments/clv_workflow.py dashboard

    # Generate CLV report
    python experiments/clv_workflow.py report
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from datetime import datetime, timedelta
import pandas as pd

from src.ml.live_tracker import LiveTracker
from src.ml.clv_tracker import CLVTracker


def import_recommendations(tracker: LiveTracker, filepath: str) -> int:
    """Import recommendations from CSV into tracker."""
    df = pd.read_csv(filepath)

    added = 0
    skipped = 0

    for _, row in df.iterrows():
        # Create unique key
        key = f"{row['home_team']}_{row['away_team']}_{row['bet_type']}_{row['line']}_{str(row['date'])[:10]}"

        # Skip if already exists
        if key in tracker.predictions:
            skipped += 1
            continue

        # Format bet_type with line
        bet_type_full = f"{row['market']} {row['bet_type']} {row['line']}"

        tracker.add_prediction(
            match_id=str(row.get('fixture_id', key)),
            home_team=row['home_team'],
            away_team=row['away_team'],
            match_date=str(row['date']),
            league=row['league'],
            bet_type=bet_type_full,
            our_probability=row['probability'],
            market_odds=row['odds'],
            threshold=0.5,  # All recommendations already meet threshold
            meets_threshold=True,
            edge=row['edge'] / 100  # Convert from percentage
        )
        added += 1

    print(f"Imported {added} predictions, skipped {skipped} duplicates")
    return added


def record_closing_odds_interactive(tracker: LiveTracker):
    """Interactive session to record closing odds."""
    # Get predictions needing closing odds
    pending = [
        p for p in tracker.predictions.values()
        if p['status'] == 'pending' and p.get('closing_odds') is None
    ]

    # Filter to today and tomorrow
    now = datetime.now()
    upcoming = []
    for p in pending:
        try:
            match_date = pd.to_datetime(p['match_date']).replace(tzinfo=None)
            if match_date.date() <= (now + timedelta(days=1)).date():
                upcoming.append(p)
        except:
            continue

    if not upcoming:
        print("\nNo upcoming predictions need closing odds.")
        print("All predictions either already have closing odds or are further in the future.")
        return

    print("\n" + "=" * 70)
    print("RECORD CLOSING ODDS")
    print("=" * 70)
    print(f"\nMatches needing odds: {len(upcoming)}")
    print("\nEnter current odds from your bookmaker (e.g., Bet365, Pinnacle)")
    print("Press Enter to skip, 'q' to quit.\n")

    for pred in sorted(upcoming, key=lambda x: x['match_date']):
        print("-" * 70)
        print(f"{pred['home_team']} vs {pred['away_team']}")
        print(f"Date: {pred['match_date']}")
        print(f"Bet: {pred['bet_type']}")
        print(f"Opening odds: {pred['market_odds_at_prediction']:.2f}")
        print(f"Our probability: {pred['our_probability']:.1%}")

        try:
            odds_input = input(f"Current odds: ").strip()

            if odds_input.lower() == 'q':
                print("\nQuitting...")
                break

            if odds_input:
                closing_odds = float(odds_input)
                if closing_odds > 1:
                    tracker.record_closing_odds(pred['key'], closing_odds)
                else:
                    print("  Invalid odds (must be > 1)")
        except ValueError:
            print("  Invalid input, skipping...")
        except KeyboardInterrupt:
            print("\n\nInterrupted. Progress saved.")
            break

    # Show CLV summary
    print_clv_summary(tracker)


def record_results_interactive(tracker: LiveTracker):
    """Interactive session to record match results."""
    # Get predictions needing results
    needs_result = [
        p for p in tracker.predictions.values()
        if p['status'] in ('pending', 'has_closing') and p.get('won') is None
    ]

    # Filter to past matches
    now = datetime.now()
    past = []
    for p in needs_result:
        try:
            match_date = pd.to_datetime(p['match_date']).replace(tzinfo=None)
            if match_date < now - timedelta(hours=2):  # 2 hours after kickoff
                past.append(p)
        except:
            continue

    if not past:
        print("\nNo past matches need results recorded.")
        return

    print("\n" + "=" * 70)
    print("RECORD MATCH RESULTS")
    print("=" * 70)
    print(f"\nMatches to update: {len(past)}")
    print("\nEnter 'w' for won, 'l' for lost, Enter to skip, 'q' to quit.\n")

    for pred in sorted(past, key=lambda x: x['match_date']):
        print("-" * 70)
        print(f"{pred['home_team']} vs {pred['away_team']}")
        print(f"Date: {pred['match_date']}")
        print(f"Bet: {pred['bet_type']}")
        print(f"Our prob: {pred['our_probability']:.1%}")
        if pred.get('clv'):
            print(f"CLV: {pred['clv']*100:+.2f}%")

        try:
            result_input = input("Result (w/l): ").strip().lower()

            if result_input == 'q':
                print("\nQuitting...")
                break

            if result_input == 'w':
                tracker.record_result(pred['key'], won=True)
            elif result_input == 'l':
                tracker.record_result(pred['key'], won=False)
            else:
                print("  Skipped")

        except KeyboardInterrupt:
            print("\n\nInterrupted. Progress saved.")
            break

    # Show summary
    tracker.show_dashboard()


def print_clv_summary(tracker: LiveTracker):
    """Print CLV summary statistics."""
    stats = tracker.get_summary_stats()

    print("\n" + "=" * 70)
    print("CLV SUMMARY")
    print("=" * 70)

    if 'avg_clv' in stats:
        print(f"\nPredictions with closing odds: {stats['with_closing_odds']}")
        print(f"Average CLV:                   {stats['avg_clv']:+.2f}%")
        print(f"Median CLV:                    {stats['median_clv']:+.2f}%")
        print(f"Positive CLV rate:             {stats['positive_clv_rate']:.1f}%")

        # Interpretation
        if stats['avg_clv'] > 3:
            print("\n>>> EXCELLENT: Strong edge! You're consistently getting better odds than closing.")
        elif stats['avg_clv'] > 0:
            print("\n>>> GOOD: Positive CLV suggests real edge. Keep tracking.")
        elif stats['avg_clv'] > -2:
            print("\n>>> NEUTRAL: CLV near zero. More data needed.")
        else:
            print("\n>>> CONCERN: Negative CLV. Market may be more accurate than your model.")
    else:
        print("\nNo CLV data yet. Record closing odds to calculate CLV.")


def generate_clv_report(tracker: LiveTracker):
    """Generate detailed CLV report."""
    tracker.clv_tracker.print_clv_report()

    # Export to CSV
    csv_path = tracker.clv_tracker.export_to_csv()
    print(f"\nDetailed data exported to: {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="CLV Tracking Workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Import recommendations
  python experiments/clv_workflow.py import --file experiments/outputs/recommendations_jan14_19.csv

  # Record closing odds (before matches)
  python experiments/clv_workflow.py closing-odds

  # Record results (after matches)
  python experiments/clv_workflow.py results

  # View dashboard
  python experiments/clv_workflow.py dashboard

  # Generate report
  python experiments/clv_workflow.py report
        """
    )

    parser.add_argument(
        'command',
        choices=['import', 'closing-odds', 'results', 'dashboard', 'report'],
        help="Command to run"
    )
    parser.add_argument(
        '--file',
        default="experiments/outputs/recommendations_jan14_19.csv",
        help="CSV file to import (for 'import' command)"
    )

    args = parser.parse_args()

    # Initialize tracker
    tracker = LiveTracker()

    if args.command == 'import':
        import_recommendations(tracker, args.file)
        tracker.show_dashboard()

    elif args.command == 'closing-odds':
        record_closing_odds_interactive(tracker)

    elif args.command == 'results':
        record_results_interactive(tracker)

    elif args.command == 'dashboard':
        tracker.show_dashboard()

    elif args.command == 'report':
        generate_clv_report(tracker)


if __name__ == "__main__":
    main()
