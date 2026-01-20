#!/usr/bin/env python
"""
Corner Betting Optimization V3 - Full Feature Set with Boruta

Key improvement over V2:
- Uses ALL available features from main features file (~253 features)
- Same approach as cards V2 which achieved +51.5% ROI
- Boruta selects most predictive features from comprehensive set

This includes ELO, form, H2H, match importance, referee patterns,
and many other features that may correlate with corner patterns.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from boruta import BorutaPy

print("=" * 70)
print("CORNER BETTING OPTIMIZATION V3 - FULL FEATURE SET")
print("=" * 70)

# Typical corner odds from bookmakers
CORNER_ODDS = {
    'over_8_5': 1.65, 'under_8_5': 2.20,
    'over_9_5': 1.90, 'under_9_5': 1.90,
    'over_10_5': 2.10, 'under_10_5': 1.72,
    'over_11_5': 2.50, 'under_11_5': 1.55,
}


def load_corner_data():
    """Load corner stats from match_stats parquet files."""
    print("\nLoading corner data from match_stats...")

    all_data = []

    for league in ['premier_league', 'la_liga', 'serie_a']:
        league_path = Path(f'data/01-raw/{league}')
        if not league_path.exists():
            continue

        for season_dir in league_path.iterdir():
            if not season_dir.is_dir():
                continue

            stats_file = season_dir / 'match_stats.parquet'
            matches_file = season_dir / 'matches.parquet'

            if not stats_file.exists() or not matches_file.exists():
                continue

            stats = pd.read_parquet(stats_file)
            matches = pd.read_parquet(matches_file)

            # Get referee info
            matches_slim = matches[[
                'fixture.id', 'fixture.referee'
            ]].rename(columns={
                'fixture.id': 'fixture_id',
                'fixture.referee': 'referee'
            })

            merged = stats.merge(matches_slim, on='fixture_id', how='left')
            merged['league'] = league
            merged['season'] = season_dir.name

            all_data.append(merged)

    df = pd.concat(all_data, ignore_index=True)
    df['total_corners'] = df['home_corners'] + df['away_corners']

    print(f"Corner data: {len(df)} matches")
    return df


def load_main_features():
    """Load the main features file with ~253 features."""
    features_path = Path('data/03-features/features_all_5leagues_with_odds.csv')

    if not features_path.exists():
        raise FileNotFoundError(f"Main features file not found: {features_path}")

    df = pd.read_csv(features_path)
    print(f"Main features: {len(df)} matches, {len(df.columns)} columns")
    return df


def merge_corner_targets(main_df: pd.DataFrame, corner_df: pd.DataFrame) -> pd.DataFrame:
    """Merge corner targets with main features."""
    print("\nMerging datasets...")

    # Standardize team names for matching
    corner_slim = corner_df[[
        'fixture_id', 'home_team', 'away_team', 'total_corners',
        'home_corners', 'away_corners', 'referee', 'date'
    ]].copy()

    # Try to match on fixture_id first
    if 'fixture_id' in main_df.columns:
        merged = main_df.merge(
            corner_slim[['fixture_id', 'total_corners', 'home_corners', 'away_corners', 'referee']],
            on='fixture_id',
            how='inner'
        )
    else:
        # Match on team names and approximate date
        main_df['match_date'] = pd.to_datetime(main_df['date']).dt.date
        corner_slim['match_date'] = pd.to_datetime(corner_slim['date']).dt.date

        merged = main_df.merge(
            corner_slim,
            left_on=['home_team', 'away_team', 'match_date'],
            right_on=['home_team', 'away_team', 'match_date'],
            how='inner',
            suffixes=('', '_corner')
        )

    print(f"Merged matches: {len(merged)}")

    # Create corner targets
    merged['over_8_5'] = (merged['total_corners'] > 8.5).astype(int)
    merged['over_9_5'] = (merged['total_corners'] > 9.5).astype(int)
    merged['over_10_5'] = (merged['total_corners'] > 10.5).astype(int)
    merged['over_11_5'] = (merged['total_corners'] > 11.5).astype(int)

    return merged


def calculate_referee_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Add referee corner statistics as features."""
    # Calculate referee stats on training data
    referee_stats = df.groupby('referee').agg({
        'total_corners': ['mean', 'std', 'count']
    }).reset_index()
    referee_stats.columns = ['referee', 'ref_corner_avg', 'ref_corner_std', 'ref_corner_matches']

    df = df.merge(referee_stats, on='referee', how='left')

    # Fill missing
    df['ref_corner_avg'] = df['ref_corner_avg'].fillna(df['total_corners'].mean())
    df['ref_corner_std'] = df['ref_corner_std'].fillna(df['total_corners'].std())
    df['ref_corner_matches'] = df['ref_corner_matches'].fillna(0)

    return df


def get_numeric_features(df: pd.DataFrame, exclude_cols: list) -> list:
    """Get numeric feature columns, filtering out non-numeric."""
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    numeric_features = []
    for col in feature_cols:
        if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
            numeric_features.append(col)
        elif df[col].dtype == 'object':
            # Try to convert
            try:
                converted = pd.to_numeric(df[col], errors='coerce')
                if converted.notna().sum() > len(df) * 0.5:
                    numeric_features.append(col)
            except:
                pass

    return numeric_features


def run_boruta_selection(X_train, y_train, feature_cols, max_iter=100):
    """Run Boruta feature selection."""
    print("\n" + "-" * 50)
    print("Running Boruta Feature Selection...")
    print("-" * 50)

    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        n_jobs=-1,
        random_state=42
    )

    boruta = BorutaPy(
        rf,
        n_estimators='auto',
        verbose=0,
        random_state=42,
        max_iter=max_iter,
        perc=100
    )

    boruta.fit(X_train.values, y_train)

    # Get results
    confirmed = [f for f, s in zip(feature_cols, boruta.support_) if s]
    tentative = [f for f, s in zip(feature_cols, boruta.support_weak_) if s]
    selected = confirmed + tentative

    print(f"Confirmed features: {len(confirmed)}")
    print(f"Tentative features: {len(tentative)}")
    print(f"Total selected: {len(selected)}")

    # Feature rankings
    ranks = pd.DataFrame({
        'feature': feature_cols,
        'rank': boruta.ranking_,
        'confirmed': boruta.support_,
        'tentative': boruta.support_weak_
    }).sort_values('rank')

    print("\nTop 20 features by Boruta rank:")
    for _, row in ranks.head(20).iterrows():
        status = "CONFIRMED" if row['confirmed'] else ("tentative" if row['tentative'] else "rejected")
        print(f"  {row['feature']:<40} rank={row['rank']:>2} ({status})")

    # If too few selected, use top ranked
    if len(selected) < 15:
        print(f"\nWarning: Only {len(selected)} features, using top 25 by rank")
        selected = ranks.head(25)['feature'].tolist()

    return selected, ranks


def simulate_betting(proba, y_test, odds_over, odds_under, n_bootstrap=1000):
    """Simulate betting strategies with bootstrap confidence intervals."""
    results = []

    # Test OVER strategies
    for thresh in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
        bet_mask = proba >= thresh
        n_bets = bet_mask.sum()

        if n_bets < 20:
            continue

        precision = y_test[bet_mask].mean()

        rois = []
        for _ in range(n_bootstrap):
            idx = np.random.choice(len(proba), len(proba), replace=True)
            p, y = proba[idx], y_test[idx]
            mask = p >= thresh
            if mask.sum() > 0:
                wins = y[mask] == 1
                profit = (wins * (odds_over - 1) - (~wins) * 1).sum()
                rois.append(profit / mask.sum() * 100)

        if not rois:
            continue

        results.append({
            'strategy': f'OVER >= {thresh}',
            'direction': 'over',
            'threshold': thresh,
            'bets': n_bets,
            'precision': precision,
            'roi': np.mean(rois),
            'ci_low': np.percentile(rois, 2.5),
            'ci_high': np.percentile(rois, 97.5),
            'p_profit': (np.array(rois) > 0).mean(),
            'odds': odds_over
        })

    # Test UNDER strategies
    for thresh in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
        under_proba = 1 - proba
        bet_mask = under_proba >= thresh
        n_bets = bet_mask.sum()

        if n_bets < 20:
            continue

        y_under = 1 - y_test
        precision = y_under[bet_mask].mean()

        rois = []
        for _ in range(n_bootstrap):
            idx = np.random.choice(len(proba), len(proba), replace=True)
            p, y = 1 - proba[idx], 1 - y_test[idx]
            mask = p >= thresh
            if mask.sum() > 0:
                wins = y[mask] == 1
                profit = (wins * (odds_under - 1) - (~wins) * 1).sum()
                rois.append(profit / mask.sum() * 100)

        if not rois:
            continue

        results.append({
            'strategy': f'UNDER >= {thresh}',
            'direction': 'under',
            'threshold': thresh,
            'bets': n_bets,
            'precision': precision,
            'roi': np.mean(rois),
            'ci_low': np.percentile(rois, 2.5),
            'ci_high': np.percentile(rois, 97.5),
            'p_profit': (np.array(rois) > 0).mean(),
            'odds': odds_under
        })

    return pd.DataFrame(results)


def main():
    # Load data
    corner_df = load_corner_data()
    main_df = load_main_features()

    # Merge
    df = merge_corner_targets(main_df, corner_df)

    if len(df) < 1000:
        print(f"Warning: Only {len(df)} matches after merge")

    # Add referee stats
    df = calculate_referee_stats(df)

    # Sort by date for temporal split
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # Temporal split
    n = len(df)
    train_end = int(0.6 * n)
    val_end = int(0.8 * n)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    print(f"\nTrain: {len(train_df)} ({train_df['date'].min().date()} to {train_df['date'].max().date()})")
    print(f"Val: {len(val_df)} ({val_df['date'].min().date()} to {val_df['date'].max().date()})")
    print(f"Test: {len(test_df)} ({test_df['date'].min().date()} to {test_df['date'].max().date()})")

    # Define excluded columns
    exclude_cols = [
        'fixture_id', 'date', 'match_date', 'home_team', 'away_team',
        'total_corners', 'home_corners', 'away_corners', 'referee',
        'over_8_5', 'over_9_5', 'over_10_5', 'over_11_5',
        'home_score', 'away_score', 'result', 'btts',
        # Exclude direct odds that encode targets
        'home_win_odds', 'away_win_odds', 'draw_odds',
        'over25_odds', 'under25_odds', 'btts_yes_odds', 'btts_no_odds',
        # Other non-feature columns
        'season', 'league', 'round', 'date_corner'
    ]

    # Get numeric features
    numeric_features = get_numeric_features(df, exclude_cols)
    print(f"\nTotal features: {len(df.columns)}")
    print(f"Numeric features for Boruta: {len(numeric_features)}")

    # Prepare data
    X_train = train_df[numeric_features].fillna(0).astype(float)
    X_val = val_df[numeric_features].fillna(0).astype(float)
    X_test = test_df[numeric_features].fillna(0).astype(float)

    # Run Boruta for main target (over_10_5)
    print("\n" + "=" * 70)
    print("BORUTA FEATURE SELECTION (target: over_10_5)")
    print("=" * 70)

    y_train_10_5 = train_df['over_10_5'].values
    selected_features, feature_ranks = run_boruta_selection(
        X_train, y_train_10_5, numeric_features, max_iter=50
    )

    # Train and evaluate callibration
    print("\n" + "=" * 70)
    print("MODEL TRAINING WITH SELECTED FEATURES")
    print("=" * 70)

    targets = [
        ('over_9_5', CORNER_ODDS['over_9_5'], CORNER_ODDS['under_9_5']),
        ('over_10_5', CORNER_ODDS['over_10_5'], CORNER_ODDS['under_10_5']),
        ('over_11_5', CORNER_ODDS['over_11_5'], CORNER_ODDS['under_11_5']),
    ]

    all_results = []

    for target_name, odds_over, odds_under in targets:
        print(f"\n--- {target_name.upper()} ---")

        y_train = train_df[target_name].values
        y_val = val_df[target_name].values
        y_test = test_df[target_name].values

        print(f"Train positive rate: {y_train.mean():.1%}")
        print(f"Test positive rate: {y_test.mean():.1%}")

        # Train callibration
        xgb = XGBClassifier(
            n_estimators=200, max_depth=4, min_child_weight=15,
            reg_lambda=5.0, learning_rate=0.05, subsample=0.8,
            random_state=42, verbosity=0
        )
        xgb.fit(X_train[selected_features], y_train)
        xgb_cal = CalibratedClassifierCV(xgb, method='sigmoid', cv='prefit')
        xgb_cal.fit(X_val[selected_features], y_val)

        lgbm = LGBMClassifier(
            n_estimators=200, max_depth=4, min_child_samples=50,
            reg_lambda=5.0, learning_rate=0.05, subsample=0.8,
            random_state=42, verbose=-1
        )
        lgbm.fit(X_train[selected_features], y_train)
        lgbm_cal = CalibratedClassifierCV(lgbm, method='sigmoid', cv='prefit')
        lgbm_cal.fit(X_val[selected_features], y_val)

        cat = CatBoostClassifier(
            iterations=200, depth=4, l2_leaf_reg=10,
            learning_rate=0.05, random_state=42, verbose=0
        )
        cat.fit(X_train[selected_features], y_train)
        cat_cal = CalibratedClassifierCV(cat, method='sigmoid', cv='prefit')
        cat_cal.fit(X_val[selected_features], y_val)

        # Ensemble predictions
        xgb_proba = xgb_cal.predict_proba(X_test[selected_features])[:, 1]
        lgbm_proba = lgbm_cal.predict_proba(X_test[selected_features])[:, 1]
        cat_proba = cat_cal.predict_proba(X_test[selected_features])[:, 1]
        avg_proba = (xgb_proba + lgbm_proba + cat_proba) / 3

        # Metrics
        brier = brier_score_loss(y_test, avg_proba)
        accuracy = ((avg_proba >= 0.5) == y_test).mean()
        print(f"Ensemble - Accuracy: {accuracy:.3f}, Brier: {brier:.4f}")

        # Betting simulation
        betting_df = simulate_betting(avg_proba, y_test, odds_over, odds_under)

        if len(betting_df) > 0:
            betting_df['target'] = target_name
            all_results.append(betting_df)

            print(f"\n{'Strategy':<16} {'Bets':>6} {'Prec':>8} {'ROI':>10} {'P(profit)':>10}")
            print("-" * 55)

            for _, row in betting_df.sort_values('roi', ascending=False).iterrows():
                print(f"{row['strategy']:<16} {row['bets']:>6} {row['precision']:>7.1%} "
                      f"{row['roi']:>9.1f}% {row['p_profit']:>9.0%}")

    # Summary
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY - V3 FULL FEATURE SET")
    print("=" * 70)

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined = combined.sort_values('roi', ascending=False)

        print("\nTop 10 strategies:")
        print(f"\n{'Target':<12} {'Strategy':<16} {'Bets':>6} {'ROI':>10} {'P(profit)':>10}")
        print("-" * 60)

        for _, row in combined.head(10).iterrows():
            print(f"{row['target']:<12} {row['strategy']:<16} {row['bets']:>6} "
                  f"{row['roi']:>9.1f}% {row['p_profit']:>9.0%}")

        # Viable strategies
        viable = combined[(combined['roi'] > 0) & (combined['p_profit'] > 0.70)]
        print(f"\nViable strategies (ROI > 0, P > 70%): {len(viable)}")

        # Compare with V2
        print("\n" + "=" * 70)
        print("COMPARISON: V3 (Full Features) vs V2 (Custom 30 Features)")
        print("=" * 70)

        v2_best = {
            'over_9_5': 41.3,  # Under >= 0.6
            'over_10_5': 31.9,  # Under >= 0.7
            'over_11_5': 19.5,  # Under >= 0.7
        }

        for target in ['over_9_5', 'over_10_5', 'over_11_5']:
            target_results = combined[combined['target'] == target]
            if len(target_results) > 0:
                best_v3 = target_results.iloc[0]['roi']
                best_v2 = v2_best.get(target, 0)
                diff = best_v3 - best_v2
                print(f"{target}: V3={best_v3:.1f}% vs V2={best_v2:.1f}% (diff: {diff:+.1f}%)")

        # Save results
        output = {
            'version': 'v3_full_features',
            'matches_analyzed': len(df),
            'test_matches': len(test_df),
            'total_features': len(numeric_features),
            'selected_features': selected_features,
            'feature_rankings': feature_ranks.to_dict('records'),
            'strategies': combined.to_dict('records')
        }

        output_path = Path('experiments/outputs/corners_optimization_v3.json')
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
