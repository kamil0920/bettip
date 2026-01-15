#!/usr/bin/env python
"""
Quick script to record match results after games finish.

Run this after matches are complete to calculate actual P&L.

Usage:
    python experiments/record_results.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ml.live_tracker import LiveTracker
from datetime import datetime
import pandas as pd


def determine_bet_result(bet_type: str, home_goals: int, away_goals: int) -> bool:
    """Determine if bet won based on result."""
    bet_type_lower = bet_type.lower()

    if 'home win' in bet_type_lower or bet_type_lower == 'home_win':
        return home_goals > away_goals

    if 'away win' in bet_type_lower or bet_type_lower == 'away_win':
        return away_goals > home_goals

    if 'draw' in bet_type_lower:
        return home_goals == away_goals

    if 'btts' in bet_type_lower:
        return home_goals > 0 and away_goals > 0

    if 'over' in bet_type_lower and '2.5' in bet_type_lower:
        return (home_goals + away_goals) > 2.5

    if 'under' in bet_type_lower and '2.5' in bet_type_lower:
        return (home_goals + away_goals) < 2.5

    if '-0.5' in bet_type_lower:
        # Asian handicap -0.5 (home team)
        if 'home' in bet_type_lower:
            return home_goals > away_goals
        else:
            return away_goals > home_goals

    # Default: ask user
    return None


def main():
    tracker = LiveTracker()

    # Get predictions with closing odds that need results
    needs_result = [
        p for p in tracker.predictions.values()
        if p['status'] in ('pending', 'has_closing')
    ]

    # Filter to past matches only
    now = datetime.now()
    past_matches = [
        p for p in needs_result
        if pd.to_datetime(p['match_date']).replace(tzinfo=None) < now
    ]

    if not past_matches:
        print("No past matches awaiting results!")
        print("\nAll pending predictions are for future matches.")
        return

    print("=" * 60)
    print("RECORD MATCH RESULTS")
    print("=" * 60)
    print(f"\nMatches to update: {len(past_matches)}")
    print("\nEnter final scores. Press Enter to skip, 'q' to quit.\n")

    results_recorded = 0

    for pred in sorted(past_matches, key=lambda x: x['match_date']):
        print("-" * 60)
        print(f"{pred['home_team']} vs {pred['away_team']}")
        print(f"Bet: {pred['bet_type']}")
        print(f"Our prob: {pred['our_probability']:.1%}")
        if pred.get('clv'):
            print(f"CLV: {pred['clv']*100:+.2f}%")

        try:
            score_input = input("Final score (e.g., '2-1' or skip): ").strip()

            if score_input.lower() == 'q':
                print("\nQuitting...")
                break

            if score_input and '-' in score_input:
                parts = score_input.split('-')
                home_goals = int(parts[0].strip())
                away_goals = int(parts[1].strip())

                # Determine if bet won
                won = determine_bet_result(pred['bet_type'], home_goals, away_goals)

                if won is None:
                    won_input = input(f"  Did {pred['bet_type']} win? (y/n): ").strip().lower()
                    won = won_input == 'y'

                tracker.record_result(
                    pred['key'],
                    won=won,
                    home_goals=home_goals,
                    away_goals=away_goals
                )
                results_recorded += 1

                result_str = "WON" if won else "LOST"
                print(f"  Result: {result_str}")

        except (ValueError, IndexError):
            print("  Invalid input, skipping...")
        except KeyboardInterrupt:
            print("\n\nInterrupted. Progress saved.")
            break

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nResults recorded: {results_recorded}")

    # Show updated stats
    tracker.show_dashboard()


if __name__ == "__main__":
    main()
