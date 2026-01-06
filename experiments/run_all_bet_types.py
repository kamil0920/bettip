#!/usr/bin/env python3
"""
Run training pipeline for all bet types.

This orchestrator runs the full optimization pipeline for:
- Asian Handicap (regression - predicts goal margin)
- BTTS (classification - predicts both teams score)
- Away Win (classification - predicts away team wins)

Usage:
    python experiments/run_all_bet_types.py
    python experiments/run_all_bet_types.py --n-trials 50 --revalidate-features
    python experiments/run_all_bet_types.py --bet-types btts away_win
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from experiments.run_full_optimization_pipeline import run_pipeline


DEFAULT_BET_TYPES = ['asian_handicap', 'btts', 'away_win']


def main():
    parser = argparse.ArgumentParser(
        description='Run training pipeline for all bet types'
    )
    parser.add_argument(
        '--bet-types', nargs='+',
        default=DEFAULT_BET_TYPES,
        help='Bet types to train (default: asian_handicap btts away_win)'
    )
    parser.add_argument(
        '--n-trials', type=int, default=80,
        help='Number of Optuna trials per model (default: 80)'
    )
    parser.add_argument(
        '--revalidate-features', action='store_true',
        help='Two-pass feature selection: re-select features with tuned params'
    )
    parser.add_argument(
        '--walkforward', action='store_true',
        help='Run walk-forward validation after training'
    )
    parser.add_argument(
        '--output-dir', type=str, default='outputs/training',
        help='Output directory for results'
    )
    args = parser.parse_args()

    print("=" * 70)
    print("MULTI-BET TRAINING PIPELINE")
    print("=" * 70)
    print(f"\nBet types: {args.bet_types}")
    print(f"Optuna trials: {args.n_trials}")
    print(f"Revalidate features: {args.revalidate_features}")
    print(f"Walk-forward: {args.walkforward}")

    results = {}
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    for bet_type in args.bet_types:
        print(f"\n{'=' * 70}")
        print(f"TRAINING: {bet_type.upper()}")
        print(f"{'=' * 70}\n")

        try:
            result = run_pipeline(
                bet_type=bet_type,
                n_trials=args.n_trials,
                revalidate_features=args.revalidate_features,
                walkforward=args.walkforward
            )
            results[bet_type] = result
        except Exception as e:
            print(f"ERROR training {bet_type}: {e}")
            results[bet_type] = {'error': str(e)}

    # Summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"\n{'Bet Type':<20} {'ROI':>10} {'P(profit)':>12} {'Bets':>8} {'Strategy':<25}")
    print("-" * 75)

    for bet_type, r in results.items():
        if 'error' in r:
            print(f"{bet_type:<20} ERROR: {r['error']}")
        elif r.get('best_roi') is not None:
            print(f"{bet_type:<20} {r['best_roi']:>+9.1f}% {r['best_p_profit']:>11.0%} "
                  f"{r['best_bets']:>8} {r['best_strategy']:<25}")
        else:
            print(f"{bet_type:<20} No profitable strategy found")

    # Save combined results
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        'timestamp': timestamp,
        'config': {
            'bet_types': args.bet_types,
            'n_trials': args.n_trials,
            'revalidate_features': args.revalidate_features,
            'walkforward': args.walkforward
        },
        'results': results,
        'production_ready': [
            bt for bt, r in results.items()
            if r.get('best_p_profit', 0) >= 0.7 and r.get('best_roi', 0) > 5
        ]
    }

    summary_path = output_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n\nResults saved to: {output_dir}")
    print(f"Production ready strategies: {summary['production_ready']}")

    return results


if __name__ == '__main__':
    main()
