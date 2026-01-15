#!/usr/bin/env python
"""
Quick script to record closing odds for today's matches.

Run this ~30 minutes before kickoff to capture closing odds.
Check odds on: Oddschecker, Bet365, or any exchange.

Usage:
    python experiments/record_closing_odds.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ml.live_tracker import LiveTracker
from datetime import datetime
import pandas as pd


def main():
    tracker = LiveTracker()

    # Get today's pending predictions
    today = datetime.now().date()
    todays = [
        p for p in tracker.predictions.values()
        if pd.to_datetime(p['match_date']).date() == today
        and p['status'] == 'pending'
    ]

    if not todays:
        print("No pending predictions for today!")
        return

    print("=" * 60)
    print("RECORD CLOSING ODDS")
    print("=" * 60)
    print(f"\nToday: {today}")
    print(f"Matches to update: {len(todays)}")
    print("\nEnter current odds from your bookmaker.")
    print("Press Enter to skip, 'q' to quit.\n")

    for pred in sorted(todays, key=lambda x: x['match_date']):
        print("-" * 60)
        print(f"{pred['home_team']} vs {pred['away_team']}")
        print(f"Bet: {pred['bet_type']}")
        print(f"Kickoff: {pred['match_date']}")
        print(f"Opening odds: {pred['market_odds_at_prediction']:.2f}")

        try:
            odds_input = input(f"Current/Closing odds [{pred['bet_type']}]: ").strip()

            if odds_input.lower() == 'q':
                print("\nQuitting...")
                break

            if odds_input:
                closing_odds = float(odds_input)
                tracker.record_closing_odds(pred['key'], closing_odds)
        except ValueError:
            print("  Invalid input, skipping...")
        except KeyboardInterrupt:
            print("\n\nInterrupted. Progress saved.")
            break

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Show CLV for updated predictions
    updated = [p for p in tracker.predictions.values() if p.get('clv') is not None]
    if updated:
        print(f"\nPredictions with CLV: {len(updated)}")
        clvs = [p['clv'] * 100 for p in updated]
        print(f"Average CLV: {sum(clvs)/len(clvs):+.2f}%")

        print("\nDetails:")
        for p in updated:
            clv = p['clv'] * 100
            symbol = "+" if clv > 0 else ""
            print(f"  {p['home_team']} vs {p['away_team']} ({p['bet_type']}): {symbol}{clv:.2f}%")
    else:
        print("\nNo closing odds recorded yet.")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
