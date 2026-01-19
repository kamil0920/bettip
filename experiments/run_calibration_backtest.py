#!/usr/bin/env python
"""
Calibration Backtest - Validate calibration factors on historical data.

This script runs walk-forward validation to verify that calibration
factors improve prediction accuracy across different markets.

Usage:
    python experiments/run_calibration_backtest.py
    python experiments/run_calibration_backtest.py --market FOULS
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple

from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from src.calibration.market_calibrator import MarketCalibrator

import warnings
warnings.filterwarnings('ignore')


def load_features():
    """Load the main features file."""
    path = Path('data/03-features/features_all_5leagues_with_odds.csv')
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    return df.sort_values('date').reset_index(drop=True)


def prepare_btts_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Prepare BTTS data for backtesting."""
    df = df[df['home_goals'].notna() & df['away_goals'].notna()].copy()
    df['btts'] = ((df['home_goals'] > 0) & (df['away_goals'] > 0)).astype(int)

    exclude_cols = [
        'fixture_id', 'date', 'home_team_name', 'away_team_name',
        'home_team_id', 'away_team_id', 'round', 'home_goals', 'away_goals',
        'total_goals', 'goal_difference', 'result', 'match_result', 'btts',
        'home_win', 'away_win', 'draw', 'league',
    ]

    feature_cols = [c for c in df.columns if c not in exclude_cols
                    and df[c].dtype in ['int64', 'float64', 'int32', 'float32']]

    return df, feature_cols[:50]  # Limit features


def prepare_fouls_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], Dict]:
    """Prepare FOULS data for backtesting."""
    # Load fouls stats
    all_stats = []
    for league in ['premier_league', 'la_liga', 'serie_a']:
        for season in ['2023', '2024', '2025']:
            stats_file = Path(f'data/01-raw/{league}/{season}/match_stats.parquet')
            if stats_file.exists():
                stats = pd.read_parquet(stats_file)
                all_stats.append(stats)

    if not all_stats:
        print("No fouls stats found")
        return None, None, None

    stats_df = pd.concat(all_stats, ignore_index=True)
    stats_df['total_fouls'] = stats_df['home_fouls'] + stats_df['away_fouls']

    # Merge with features
    merged = df.merge(stats_df[['fixture_id', 'total_fouls']], on='fixture_id', how='inner')
    merged = merged[merged['total_fouls'].notna()]

    # Create targets for different lines
    merged['over_22_5'] = (merged['total_fouls'] > 22.5).astype(int)
    merged['over_24_5'] = (merged['total_fouls'] > 24.5).astype(int)
    merged['over_26_5'] = (merged['total_fouls'] > 26.5).astype(int)

    exclude_cols = [
        'fixture_id', 'date', 'home_team_name', 'away_team_name',
        'home_team_id', 'away_team_id', 'round', 'home_goals', 'away_goals',
        'total_goals', 'goal_difference', 'result', 'match_result',
        'home_win', 'away_win', 'draw', 'league', 'total_fouls',
        'over_22_5', 'over_24_5', 'over_26_5',
    ]

    feature_cols = [c for c in merged.columns if c not in exclude_cols
                    and merged[c].dtype in ['int64', 'float64', 'int32', 'float32']]

    return merged, feature_cols[:50], {'lines': [22.5, 24.5, 26.5]}


def prepare_shots_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], Dict]:
    """Prepare SHOTS data for backtesting."""
    all_stats = []
    for league in ['premier_league', 'la_liga', 'serie_a']:
        for season in ['2023', '2024', '2025']:
            stats_file = Path(f'data/01-raw/{league}/{season}/match_stats.parquet')
            if stats_file.exists():
                stats = pd.read_parquet(stats_file)
                all_stats.append(stats)

    if not all_stats:
        print("No shots stats found")
        return None, None, None

    stats_df = pd.concat(all_stats, ignore_index=True)

    # Check for shots columns
    if 'home_shots_total' in stats_df.columns:
        stats_df['total_shots'] = stats_df['home_shots_total'] + stats_df['away_shots_total']
    elif 'home_shots' in stats_df.columns:
        stats_df['total_shots'] = stats_df['home_shots'] + stats_df['away_shots']
    else:
        print("No shots columns found")
        return None, None, None

    merged = df.merge(stats_df[['fixture_id', 'total_shots']], on='fixture_id', how='inner')
    merged = merged[merged['total_shots'].notna()]

    merged['over_24_5'] = (merged['total_shots'] > 24.5).astype(int)

    exclude_cols = [
        'fixture_id', 'date', 'home_team_name', 'away_team_name',
        'home_team_id', 'away_team_id', 'round', 'home_goals', 'away_goals',
        'total_goals', 'goal_difference', 'result', 'match_result',
        'home_win', 'away_win', 'draw', 'league', 'total_shots', 'over_24_5',
    ]

    feature_cols = [c for c in merged.columns if c not in exclude_cols
                    and merged[c].dtype in ['int64', 'float64', 'int32', 'float32']]

    return merged, feature_cols[:50], {'line': 24.5}


def prepare_corners_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], Dict]:
    """Prepare CORNERS data for backtesting."""
    all_stats = []
    for league in ['premier_league', 'la_liga', 'serie_a']:
        for season in ['2023', '2024', '2025']:
            stats_file = Path(f'data/01-raw/{league}/{season}/match_stats.parquet')
            if stats_file.exists():
                stats = pd.read_parquet(stats_file)
                all_stats.append(stats)

    if not all_stats:
        print("No corners stats found")
        return None, None, None

    stats_df = pd.concat(all_stats, ignore_index=True)

    if 'home_corners' in stats_df.columns:
        stats_df['total_corners'] = stats_df['home_corners'] + stats_df['away_corners']
    else:
        print("No corners columns found")
        return None, None, None

    merged = df.merge(stats_df[['fixture_id', 'total_corners']], on='fixture_id', how='inner')
    merged = merged[merged['total_corners'].notna()]

    merged['over_10_5'] = (merged['total_corners'] > 10.5).astype(int)

    exclude_cols = [
        'fixture_id', 'date', 'home_team_name', 'away_team_name',
        'home_team_id', 'away_team_id', 'round', 'home_goals', 'away_goals',
        'total_goals', 'goal_difference', 'result', 'match_result',
        'home_win', 'away_win', 'draw', 'league', 'total_corners', 'over_10_5',
    ]

    feature_cols = [c for c in merged.columns if c not in exclude_cols
                    and merged[c].dtype in ['int64', 'float64', 'int32', 'float32']]

    return merged, feature_cols[:50], {'line': 10.5}


def prepare_away_win_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], Dict]:
    """Prepare AWAY_WIN data for backtesting."""
    df = df[df['home_goals'].notna() & df['away_goals'].notna()].copy()
    df['away_win'] = (df['away_goals'] > df['home_goals']).astype(int)

    exclude_cols = [
        'fixture_id', 'date', 'home_team_name', 'away_team_name',
        'home_team_id', 'away_team_id', 'round', 'home_goals', 'away_goals',
        'total_goals', 'goal_difference', 'result', 'match_result',
        'home_win', 'away_win', 'draw', 'league',
    ]

    feature_cols = [c for c in df.columns if c not in exclude_cols
                    and df[c].dtype in ['int64', 'float64', 'int32', 'float32']]

    return df, feature_cols[:50], {}


def walk_forward_backtest(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    market: str,
    calibrator: MarketCalibrator,
    n_splits: int = 5,
    test_ratio: float = 0.15
) -> Dict:
    """
    Run walk-forward validation with and without calibration.

    Returns dict with performance metrics.
    """
    n = len(df)
    test_size = int(n * test_ratio)
    train_size = n - (n_splits * test_size)

    if train_size < 500:
        print(f"  Warning: small training size ({train_size})")

    results_raw = []
    results_calibrated = []

    for fold in range(n_splits):
        train_end = train_size + fold * test_size
        test_end = train_end + test_size

        if test_end > n:
            break

        train_df = df.iloc[:train_end]
        test_df = df.iloc[train_end:test_end]

        X_train = train_df[feature_cols].fillna(0)
        y_train = train_df[target_col].values
        X_test = test_df[feature_cols].fillna(0)
        y_test = test_df[target_col].values

        # Train model
        model = CatBoostClassifier(
            iterations=150, depth=4, learning_rate=0.05,
            l2_leaf_reg=5, random_state=42, verbose=0
        )
        model.fit(X_train, y_train)

        # Get raw probabilities
        raw_probs = model.predict_proba(X_test)[:, 1]

        # Apply calibration
        cal_probs = calibrator.calibrate(market, raw_probs)

        # Calculate metrics at threshold 0.65
        threshold = 0.65

        # Raw performance
        raw_preds = raw_probs >= threshold
        if raw_preds.sum() > 0:
            raw_hit_rate = y_test[raw_preds].mean()
            raw_avg_prob = raw_probs[raw_preds].mean()
            results_raw.append({
                'fold': fold,
                'n_bets': int(raw_preds.sum()),
                'hit_rate': raw_hit_rate,
                'avg_prob': raw_avg_prob,
                'calibration_gap': raw_hit_rate - raw_avg_prob
            })

        # Calibrated performance
        cal_preds = cal_probs >= threshold
        if cal_preds.sum() > 0:
            cal_hit_rate = y_test[cal_preds].mean()
            cal_avg_prob = cal_probs[cal_preds].mean()
            results_calibrated.append({
                'fold': fold,
                'n_bets': int(cal_preds.sum()),
                'hit_rate': cal_hit_rate,
                'avg_prob': cal_avg_prob,
                'calibration_gap': cal_hit_rate - cal_avg_prob
            })

    return {
        'raw': results_raw,
        'calibrated': results_calibrated
    }


def main():
    parser = argparse.ArgumentParser(description='Run calibration backtest')
    parser.add_argument('--market', type=str, default='ALL',
                        help='Market to backtest (FOULS, BTTS, ALL)')
    args = parser.parse_args()

    print("=" * 70)
    print("CALIBRATION BACKTEST")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    # Initialize calibrator
    calibrator = MarketCalibrator()
    calibrator.load_config('config/strategies.yaml')

    # Load features
    print("\nLoading data...")
    df = load_features()
    print(f"Loaded {len(df)} matches")

    markets_to_test = ['FOULS', 'BTTS', 'SHOTS', 'CORNERS', 'AWAY_WIN'] if args.market == 'ALL' else [args.market.upper()]

    for market in markets_to_test:
        print(f"\n{'='*60}")
        print(f"MARKET: {market}")
        print(f"Calibration factor: {calibrator.get_calibration_factor(market):.2f}")
        print(f"{'='*60}")

        if market == 'BTTS':
            data, feature_cols = prepare_btts_data(df)
            target_col = 'btts'
        elif market == 'FOULS':
            data, feature_cols, config = prepare_fouls_data(df)
            if data is None:
                print("Skipping FOULS - no data")
                continue
            target_col = 'over_24_5'  # Use 24.5 line for backtest
        elif market == 'SHOTS':
            data, feature_cols, config = prepare_shots_data(df)
            if data is None:
                print("Skipping SHOTS - no data")
                continue
            target_col = 'over_24_5'
        elif market == 'CORNERS':
            data, feature_cols, config = prepare_corners_data(df)
            if data is None:
                print("Skipping CORNERS - no data")
                continue
            target_col = 'over_10_5'
        elif market == 'AWAY_WIN':
            data, feature_cols, config = prepare_away_win_data(df)
            target_col = 'away_win'
        else:
            print(f"Unknown market: {market}")
            continue

        print(f"Data: {len(data)} matches")
        print(f"Features: {len(feature_cols)}")
        print(f"Base rate: {data[target_col].mean():.1%}")

        # Run backtest
        results = walk_forward_backtest(
            data, feature_cols, target_col, market, calibrator, n_splits=5
        )

        # Print results
        print(f"\n--- RAW (uncalibrated) ---")
        if results['raw']:
            raw_df = pd.DataFrame(results['raw'])
            print(f"  Avg bets/fold: {raw_df['n_bets'].mean():.0f}")
            print(f"  Avg hit rate:  {raw_df['hit_rate'].mean():.1%}")
            print(f"  Avg pred prob: {raw_df['avg_prob'].mean():.1%}")
            print(f"  Calibration gap: {raw_df['calibration_gap'].mean():+.1%}")

        print(f"\n--- CALIBRATED ---")
        if results['calibrated']:
            cal_df = pd.DataFrame(results['calibrated'])
            print(f"  Avg bets/fold: {cal_df['n_bets'].mean():.0f}")
            print(f"  Avg hit rate:  {cal_df['hit_rate'].mean():.1%}")
            print(f"  Avg pred prob: {cal_df['avg_prob'].mean():.1%}")
            print(f"  Calibration gap: {cal_df['calibration_gap'].mean():+.1%}")

        # Compare
        if results['raw'] and results['calibrated']:
            raw_gap = pd.DataFrame(results['raw'])['calibration_gap'].mean()
            cal_gap = pd.DataFrame(results['calibrated'])['calibration_gap'].mean()
            print(f"\n--- IMPROVEMENT ---")
            print(f"  Raw calibration gap:  {raw_gap:+.1%}")
            print(f"  Cal calibration gap:  {cal_gap:+.1%}")
            print(f"  Gap reduction: {abs(raw_gap) - abs(cal_gap):.1%} better")

    print("\n" + "=" * 70)
    print("BACKTEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
