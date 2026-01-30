#!/usr/bin/env python3
"""
Unified Daily Recommendation Generator

Generates predictions across all markets and outputs to standardized format:
- data/05-recommendations/rec_YYYYMMDD_NNN.csv

Integrates:
- Main markets: home_win, away_win, asian_handicap, btts
- Niche markets: corners, cards, shots, fouls

Usage:
    python experiments/generate_daily_recommendations.py
    python experiments/generate_daily_recommendations.py --min-edge 15
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def get_next_rec_number(date_str: str) -> int:
    """Get next recommendation file number for today."""
    rec_dir = project_root / 'data/05-recommendations'
    existing = list(rec_dir.glob(f'rec_{date_str}_*.csv'))
    if not existing:
        return 1
    numbers = []
    for f in existing:
        try:
            num = int(f.stem.split('_')[-1])
            numbers.append(num)
        except ValueError:
            pass
    return max(numbers) + 1 if numbers else 1


def load_main_predictions() -> List[Dict]:
    """Load predictions from main prediction script."""
    predictions = []
    main_file = project_root / 'experiments/outputs/next_round_predictions.json'

    if main_file.exists():
        with open(main_file) as f:
            data = json.load(f)

        for p in data.get('predictions', []):
            match = p.get('match', '')
            parts = match.split(' vs ')
            home_team = parts[0].strip() if len(parts) > 0 else ''
            away_team = parts[1].strip() if len(parts) > 1 else ''

            bet_type = p.get('bet_type', '')
            market = 'MATCH_RESULT'
            side = 'HOME_WIN'
            line = 0.0

            if 'away' in bet_type.lower():
                side = 'AWAY_WIN'
            elif 'btts' in bet_type.lower():
                market = 'BTTS'
                side = 'YES'
            elif 'asian' in bet_type.lower() or 'home -' in bet_type.lower():
                market = 'ASIAN_HANDICAP'
                side = 'HOME'
                line = -0.5

            predictions.append({
                'date': str(p.get('date', ''))[:10],
                'home_team': home_team,
                'away_team': away_team,
                'league': p.get('league', ''),
                'market': market,
                'bet_type': side,
                'line': line,
                'odds': p.get('market_odds', p.get('odds', 1.9)),
                'probability': p.get('our_prob', p.get('probability', 0)),
                'edge': p.get('edge', 0) * 100 if p.get('edge', 0) < 1 else p.get('edge', 0),
                'referee': '',
                'fixture_id': '',
                'result': '',
                'actual': ''
            })

    return predictions


def load_niche_predictions(market: str, tracker_file: str) -> List[Dict]:
    """Load predictions from niche market tracker."""
    predictions = []
    tracker_path = project_root / f'experiments/outputs/{tracker_file}'

    if tracker_path.exists():
        with open(tracker_path) as f:
            data = json.load(f)

        for bet in data.get('bets', []):
            if bet.get('status') != 'pending':
                continue

            predictions.append({
                'date': str(bet.get('match_date', ''))[:10],
                'home_team': bet.get('home_team', ''),
                'away_team': bet.get('away_team', ''),
                'league': bet.get('league', ''),
                'market': market.upper(),
                'bet_type': bet.get('bet_type', 'OVER'),
                'line': bet.get('line', 0),
                'odds': bet.get('our_odds', 1.9),
                'probability': bet.get('our_probability', 0),
                'edge': bet.get('edge', 0),
                'referee': bet.get('referee', ''),
                'fixture_id': bet.get('fixture_id', ''),
                'result': '',
                'actual': ''
            })

    return predictions


def run_prediction_scripts(min_edge: float = 10.0) -> None:
    """Run all prediction scripts to generate fresh predictions."""
    print("Running prediction scripts...")

    # Run main predictions
    main_script = project_root / 'experiments/predict_next_round.py'
    if main_script.exists():
        print("  Running main market predictions...")
        try:
            subprocess.run(
                ['python', str(main_script), '--show-all'],
                cwd=str(project_root),
                capture_output=True,
                timeout=300
            )
        except Exception as e:
            print(f"    Warning: Main predictions failed: {e}")

    # Run niche market predictions
    niche_scripts = [
        ('corners', 'corners_paper_trade.py'),
        ('shots', 'shots_paper_trade.py'),
        ('fouls', 'fouls_paper_trade.py'),
    ]

    for market, script in niche_scripts:
        script_path = project_root / 'experiments' / script
        if script_path.exists():
            print(f"  Running {market} predictions...")
            try:
                subprocess.run(
                    ['python', str(script_path), 'predict', str(min_edge)],
                    cwd=str(project_root),
                    capture_output=True,
                    timeout=300
                )
            except Exception as e:
                print(f"    Warning: {market} predictions failed: {e}")


def consolidate_predictions(min_edge: float = 10.0) -> pd.DataFrame:
    """Consolidate all predictions into a single DataFrame."""
    all_predictions = []

    # Load main market predictions
    main_preds = load_main_predictions()
    all_predictions.extend(main_preds)
    print(f"  Main markets: {len(main_preds)} predictions")

    # Load niche market predictions
    niche_trackers = [
        ('CORNERS', 'corners_tracking_v3.json'),
        ('SHOTS', 'shots_tracking.json'),
        ('FOULS', 'fouls_tracking.json'),
    ]

    for market, tracker_file in niche_trackers:
        preds = load_niche_predictions(market, tracker_file)
        all_predictions.extend(preds)
        print(f"  {market}: {len(preds)} predictions")

    if not all_predictions:
        return pd.DataFrame()

    df = pd.DataFrame(all_predictions)

    # Filter by date - only include today and future matches
    today = datetime.now().strftime('%Y-%m-%d')
    df = df[df['date'] >= today]

    # Filter by edge
    df = df[df['edge'] >= min_edge]

    # Sort by date and edge
    df = df.sort_values(['date', 'edge'], ascending=[True, False])

    # Remove duplicates (same match, market, bet_type, line)
    df = df.drop_duplicates(subset=['home_team', 'away_team', 'market', 'bet_type', 'line'])

    return df


def save_recommendations(df: pd.DataFrame) -> str:
    """Save recommendations to standardized format."""
    if df.empty:
        print("No recommendations to save")
        return ""

    rec_dir = project_root / 'data/05-recommendations'
    rec_dir.mkdir(parents=True, exist_ok=True)

    date_str = datetime.now().strftime('%Y%m%d')
    rec_num = get_next_rec_number(date_str)
    filename = f'rec_{date_str}_{rec_num:03d}.csv'
    filepath = rec_dir / filename

    # Ensure columns are in correct order
    columns = [
        'date', 'home_team', 'away_team', 'league', 'market', 'bet_type',
        'line', 'odds', 'probability', 'edge', 'referee', 'fixture_id',
        'result', 'actual'
    ]

    for col in columns:
        if col not in df.columns:
            df[col] = ''

    df = df[columns]
    df.to_csv(filepath, index=False)

    print(f"\nSaved {len(df)} recommendations to: {filepath}")
    return str(filepath)


def update_readme_index(filepath: str, count: int) -> None:
    """Update README with new file."""
    readme_path = project_root / 'data/05-recommendations/README.md'
    if not readme_path.exists():
        return

    content = readme_path.read_text()

    filename = Path(filepath).name
    date_str = filename.split('_')[1]
    date_formatted = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"

    new_entry = f"| {filename} | {date_formatted} | {count} | Generated |\n"

    # Find index section and add new entry
    if '| File | Date |' in content:
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if '| File | Date |' in line:
                # Insert after header and separator
                insert_idx = i + 2
                lines.insert(insert_idx, new_entry.strip())
                break
        content = '\n'.join(lines)
        readme_path.write_text(content)


def print_summary(df: pd.DataFrame) -> None:
    """Print summary of recommendations."""
    if df.empty:
        print("\nNo recommendations generated")
        return

    print("\n" + "=" * 70)
    print("DAILY RECOMMENDATIONS SUMMARY")
    print("=" * 70)

    print(f"\nTotal recommendations: {len(df)}")

    # By market
    print("\nBy Market:")
    for market in df['market'].unique():
        count = len(df[df['market'] == market])
        avg_edge = df[df['market'] == market]['edge'].mean()
        print(f"  {market}: {count} bets, avg edge {avg_edge:.1f}%")

    # By date
    print("\nBy Date:")
    for date in sorted(df['date'].unique()):
        count = len(df[df['date'] == date])
        print(f"  {date}: {count} bets")

    # Top 10 by edge
    print("\nTop 10 by Edge:")
    print("-" * 70)
    top10 = df.nlargest(10, 'edge')
    for _, row in top10.iterrows():
        match = f"{row['home_team'][:15]} vs {row['away_team'][:15]}"
        bet = f"{row['market']} {row['bet_type']}"
        if row['line']:
            bet += f" {row['line']}"
        print(f"  {row['date']} | {match:<32} | {bet:<20} | +{row['edge']:.1f}%")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Generate daily recommendations')
    parser.add_argument('--min-edge', type=float, default=10.0,
                        help='Minimum edge percentage (default: 10)')
    parser.add_argument('--skip-run', action='store_true',
                        help='Skip running prediction scripts, just consolidate')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print summary but don\'t save file')
    args = parser.parse_args()

    print("=" * 70)
    print("DAILY RECOMMENDATION GENERATOR")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Minimum Edge: {args.min_edge}%")
    print("=" * 70)

    # Run prediction scripts
    if not args.skip_run:
        run_prediction_scripts(args.min_edge)

    # Consolidate predictions
    print("\nConsolidating predictions...")
    df = consolidate_predictions(args.min_edge)

    # Print summary
    print_summary(df)

    # Save recommendations
    if not args.dry_run and not df.empty:
        filepath = save_recommendations(df)
        if filepath:
            update_readme_index(filepath, len(df))
            print(f"\nRecommendations file: {filepath}")

    return 0 if not df.empty else 1


if __name__ == '__main__':
    sys.exit(main())
