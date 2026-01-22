#!/usr/bin/env python3
"""
Paper Trading Tracker for Betting Predictions.

Track predictions, results, and P&L for paper trading validation.

Usage:
    # Add predictions from latest run
    python experiments/paper_trading_tracker.py add

    # Update results for completed matches
    python experiments/paper_trading_tracker.py update

    # Show current status and P&L
    python experiments/paper_trading_tracker.py status

    # Export to CSV
    python experiments/paper_trading_tracker.py export
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

TRACKER_FILE = project_root / 'data/paper_trading/paper_trades.csv'
PREDICTIONS_FILE = project_root / 'experiments/outputs/next_round_predictions.json'


def ensure_tracker_exists():
    """Create tracker file if it doesn't exist."""
    TRACKER_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not TRACKER_FILE.exists():
        df = pd.DataFrame(columns=[
            'id', 'date', 'added_at', 'league', 'home_team', 'away_team',
            'bet_type', 'prediction', 'model_prob', 'market_prob', 'edge',
            'odds', 'stake', 'result', 'home_goals', 'away_goals',
            'profit', 'status'
        ])
        df.to_csv(TRACKER_FILE, index=False)
        print(f"Created tracker file: {TRACKER_FILE}")


def load_tracker() -> pd.DataFrame:
    """Load the tracker DataFrame."""
    ensure_tracker_exists()
    return pd.read_csv(TRACKER_FILE)


def save_tracker(df: pd.DataFrame):
    """Save the tracker DataFrame."""
    df.to_csv(TRACKER_FILE, index=False)


def add_predictions(min_edge: float = 0.15, min_prob: float = 0.55):
    """Add predictions from latest prediction run.

    Args:
        min_edge: Minimum edge required (default 15%)
        min_prob: Minimum model probability (default 55%)
    """
    if not PREDICTIONS_FILE.exists():
        print(f"No predictions file found: {PREDICTIONS_FILE}")
        print("Run: python experiments/predict_next_round.py first")
        return

    with open(PREDICTIONS_FILE) as f:
        data = json.load(f)

    # Get predictions from JSON (handles both formats)
    predictions = data.get('value_bets', data.get('predictions', []))
    if not predictions:
        print("No predictions found")
        return

    tracker = load_tracker()
    added = 0
    skipped = 0
    filtered = 0

    for pred in predictions:
        # Parse match name if needed
        if 'match' in pred and 'home_team' not in pred:
            parts = pred['match'].split(' vs ')
            if len(parts) == 2:
                pred['home_team'] = parts[0].strip()
                pred['away_team'] = parts[1].strip()

        # Get values with fallbacks
        edge = pred.get('edge', 0)
        prob = pred.get('probability', pred.get('model_prob', pred.get('our_prob', 0)))

        # Filter by edge and probability
        if edge < min_edge or prob < min_prob:
            filtered += 1
            continue

        # Create unique ID
        home = pred.get('home_team', 'UNK')[:3]
        away = pred.get('away_team', 'UNK')[:3]
        date_str = str(pred.get('date', ''))[:10]
        pred_id = f"{date_str}_{home}_{away}_{pred.get('bet_type', 'UNK')}"

        # Skip if already exists
        if pred_id in tracker['id'].values:
            skipped += 1
            continue

        # Calculate stake (Kelly with fractional)
        kelly_fraction = 0.25
        if edge > 0 and pred.get('market_odds', 0) > 1:
            odds = pred.get('market_odds', pred.get('odds', 2.0))
            kelly = (prob * odds - 1) / (odds - 1)
            stake = max(0, min(50, 1000 * kelly * kelly_fraction))  # Cap at 5%
        else:
            stake = pred.get('stake', 25)

        new_row = {
            'id': pred_id,
            'date': date_str,
            'added_at': datetime.now().isoformat(),
            'league': pred.get('league', ''),
            'home_team': pred.get('home_team', ''),
            'away_team': pred.get('away_team', ''),
            'bet_type': pred.get('bet_type', ''),
            'prediction': pred.get('prediction', pred.get('bet_type', '')),
            'model_prob': round(prob * 100, 1),
            'market_prob': round(pred.get('implied_prob', pred.get('market_prob', 0)) * 100, 1),
            'edge': round(edge * 100, 1),
            'odds': pred.get('market_odds', pred.get('odds', 0)),
            'stake': round(stake, 2),
            'result': None,
            'home_goals': None,
            'away_goals': None,
            'profit': None,
            'status': 'pending'
        }
        tracker = pd.concat([tracker, pd.DataFrame([new_row])], ignore_index=True)
        added += 1

    save_tracker(tracker)
    print(f"Added {added} new predictions")
    print(f"  Skipped {skipped} duplicates, filtered {filtered} below threshold")
    print(f"  (min_edge={min_edge*100:.0f}%, min_prob={min_prob*100:.0f}%)")
    print(f"Total pending bets: {len(tracker[tracker['status'] == 'pending'])}")


def update_results():
    """Interactively update results for pending bets."""
    tracker = load_tracker()
    pending = tracker[tracker['status'] == 'pending'].copy()

    if len(pending) == 0:
        print("No pending bets to update")
        return

    print(f"\n{len(pending)} pending bets:")
    print("-" * 80)

    for idx, row in pending.iterrows():
        print(f"\n[{row['date']}] {row['home_team']} vs {row['away_team']}")
        print(f"  Bet: {row['bet_type']} @ {row['odds']:.2f} (${row['stake']:.2f})")

        result = input("  Result (w/l/v/skip): ").strip().lower()

        if result == 'skip':
            continue
        elif result == 'v':  # Void
            tracker.loc[idx, 'status'] = 'void'
            tracker.loc[idx, 'profit'] = 0
        elif result == 'w':  # Win
            tracker.loc[idx, 'status'] = 'won'
            tracker.loc[idx, 'result'] = 'win'
            profit = row['stake'] * (row['odds'] - 1)
            tracker.loc[idx, 'profit'] = profit
            print(f"  ✓ Won: +${profit:.2f}")
        elif result == 'l':  # Loss
            tracker.loc[idx, 'status'] = 'lost'
            tracker.loc[idx, 'result'] = 'loss'
            tracker.loc[idx, 'profit'] = -row['stake']
            print(f"  ✗ Lost: -${row['stake']:.2f}")

        # Optional: enter score
        score = input("  Score (e.g. 2-1, or skip): ").strip()
        if score and '-' in score:
            try:
                home, away = score.split('-')
                tracker.loc[idx, 'home_goals'] = int(home)
                tracker.loc[idx, 'away_goals'] = int(away)
            except:
                pass

    save_tracker(tracker)
    print("\nResults updated!")


def show_status():
    """Show current paper trading status and P&L."""
    tracker = load_tracker()

    if len(tracker) == 0:
        print("No trades recorded yet")
        return

    print("\n" + "=" * 70)
    print("PAPER TRADING STATUS")
    print("=" * 70)

    # Summary stats
    total = len(tracker)
    pending = len(tracker[tracker['status'] == 'pending'])
    completed = len(tracker[tracker['status'].isin(['won', 'lost'])])
    won = len(tracker[tracker['status'] == 'won'])
    lost = len(tracker[tracker['status'] == 'lost'])

    print(f"\nTotal bets: {total}")
    print(f"  Pending: {pending}")
    print(f"  Completed: {completed}")
    if completed > 0:
        print(f"    Won: {won} ({won/completed*100:.1f}%)")
        print(f"    Lost: {lost} ({lost/completed*100:.1f}%)")

    # P&L
    completed_df = tracker[tracker['status'].isin(['won', 'lost'])]
    if len(completed_df) > 0:
        total_stake = completed_df['stake'].sum()
        total_profit = completed_df['profit'].sum()
        roi = (total_profit / total_stake * 100) if total_stake > 0 else 0

        print(f"\n--- P&L Summary ---")
        print(f"Total staked: ${total_stake:.2f}")
        print(f"Total profit: ${total_profit:+.2f}")
        print(f"ROI: {roi:+.1f}%")

        # By bet type
        print(f"\n--- By Bet Type ---")
        by_type = completed_df.groupby('bet_type').agg({
            'stake': 'sum',
            'profit': 'sum',
            'status': lambda x: (x == 'won').sum()
        }).rename(columns={'status': 'wins'})
        by_type['bets'] = completed_df.groupby('bet_type').size()
        by_type['roi'] = by_type['profit'] / by_type['stake'] * 100
        by_type['win_rate'] = by_type['wins'] / by_type['bets'] * 100

        for bet_type, row in by_type.iterrows():
            print(f"  {bet_type}: {row['wins']:.0f}/{row['bets']:.0f} wins "
                  f"({row['win_rate']:.0f}%), ROI: {row['roi']:+.1f}%")

    # Recent bets
    print(f"\n--- Recent Bets ---")
    recent = tracker.tail(10).iloc[::-1]
    for _, row in recent.iterrows():
        status_icon = {'pending': '⏳', 'won': '✓', 'lost': '✗', 'void': '○'}.get(row['status'], '?')
        profit_str = f"${row['profit']:+.2f}" if pd.notna(row['profit']) else ""
        print(f"  {status_icon} [{row['date']}] {row['home_team'][:12]:12} vs {row['away_team'][:12]:12} "
              f"| {row['bet_type']:12} @ {row['odds']:.2f} {profit_str}")


def export_csv():
    """Export tracker to a clean CSV."""
    tracker = load_tracker()
    export_path = project_root / f'data/paper_trading/export_{datetime.now().strftime("%Y%m%d")}.csv'
    tracker.to_csv(export_path, index=False)
    print(f"Exported to: {export_path}")


def main():
    parser = argparse.ArgumentParser(description='Paper Trading Tracker')
    parser.add_argument('command', choices=['add', 'update', 'status', 'export'],
                        help='Command to run')

    args = parser.parse_args()

    if args.command == 'add':
        add_predictions()
    elif args.command == 'update':
        update_results()
    elif args.command == 'status':
        show_status()
    elif args.command == 'export':
        export_csv()


if __name__ == '__main__':
    main()
