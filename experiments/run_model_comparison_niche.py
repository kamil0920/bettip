#!/usr/bin/env python
"""
Model Comparison for Niche Bet Types (Corners, Cards)

Compares different model architectures:
1. Individual callibration: XGBoost, LightGBM, CatBoost, RandomForest
2. Simple average ensemble
3. Stacking with LogisticRegression meta-learner
4. Stacking with XGBoost meta-learner
5. Weighted voting based on CV scores

This helps identify the best model architecture for each niche bet type.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

print("=" * 70)
print("MODEL COMPARISON FOR NICHE BET TYPES")
print("=" * 70)


def load_corners_data():
    """Load and prepare corners data with features."""
    # Load corner data
    corner_data = []
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
            matches_slim = matches[['fixture.id', 'fixture.referee']].rename(
                columns={'fixture.id': 'fixture_id', 'fixture.referee': 'referee'})
            merged = stats.merge(matches_slim, on='fixture_id', how='left')
            merged['league'] = league
            merged['total_corners'] = merged['home_corners'] + merged['away_corners']
            corner_data.append(merged)

    corner_df = pd.concat(corner_data, ignore_index=True)

    # Load main features
    from src.utils.data_io import load_features
    main_df = load_features('data/03-features/features_all_5leagues_with_odds.parquet')

    # Merge
    df = main_df.merge(
        corner_df[['fixture_id', 'total_corners', 'referee']],
        on='fixture_id', how='inner'
    )

    # Create targets
    df['over_9_5'] = (df['total_corners'] > 9.5).astype(int)
    df['over_10_5'] = (df['total_corners'] > 10.5).astype(int)
    df['over_11_5'] = (df['total_corners'] > 11.5).astype(int)

    # Add referee stats
    ref_stats = df.groupby('referee')['total_corners'].agg(['mean', 'std']).reset_index()
    ref_stats.columns = ['referee', 'ref_corner_avg', 'ref_corner_std']
    df = df.merge(ref_stats, on='referee', how='left')
    df['ref_corner_avg'] = df['ref_corner_avg'].fillna(df['total_corners'].mean())
    df['ref_corner_std'] = df['ref_corner_std'].fillna(df['total_corners'].std())

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    exclude_cols = [
        'fixture_id', 'date', 'home_team_name', 'away_team_name',
        'total_corners', 'referee',
        'over_9_5', 'over_10_5', 'over_11_5',
        'home_score', 'away_score', 'result', 'btts',
        'home_win_odds', 'away_win_odds', 'draw_odds',
        'over25_odds', 'under25_odds', 'btts_yes_odds', 'btts_no_odds',
    ]

    numeric_features = []
    for col in df.columns:
        if col not in exclude_cols and df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
            numeric_features.append(col)

    return df, numeric_features[:25]  # Top 25 features


def load_cards_data():
    """Load and prepare cards data with features."""
    features_path = Path('data/03-features/features_all_5leagues_with_odds.csv')
    df = pd.read_csv(features_path)

    # Load events for cards targets
    cards_data = []
    for league in ['premier_league', 'la_liga', 'serie_a']:
        events_file = Path(f'data/01-raw/{league}/2025/events.parquet')
        matches_file = Path(f'data/01-raw/{league}/2025/matches.parquet')
        if events_file.exists() and matches_file.exists():
            events = pd.read_parquet(events_file)
            matches = pd.read_parquet(matches_file)

            # Count yellow cards per match
            yellows = events[events['type'] == 'Card']
            if 'detail' in yellows.columns:
                yellows = yellows[yellows['detail'] == 'Yellow Card']

            card_counts = yellows.groupby('fixture_id').size().reset_index(name='total_yellows')

            matches_slim = matches[['fixture.id']].rename(columns={'fixture.id': 'fixture_id'})
            cards = matches_slim.merge(card_counts, on='fixture_id', how='left')
            cards['total_yellows'] = cards['total_yellows'].fillna(0)
            cards['league'] = league
            cards_data.append(cards)

    if cards_data:
        cards_df = pd.concat(cards_data, ignore_index=True)
        df = df.merge(cards_df[['fixture_id', 'total_yellows']], on='fixture_id', how='inner')

        # Create targets
        df['over_3_5_cards'] = (df['total_yellows'] > 3.5).astype(int)
        df['over_4_5_cards'] = (df['total_yellows'] > 4.5).astype(int)
        df['over_5_5_cards'] = (df['total_yellows'] > 5.5).astype(int)

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # Get numeric features
    exclude_cols = [
        'fixture_id', 'date', 'home_team_name', 'away_team_name',
        'home_score', 'away_score', 'result', 'btts',
        'total_yellows', 'over_3_5_cards', 'over_4_5_cards', 'over_5_5_cards',
        'home_win_odds', 'away_win_odds', 'draw_odds',
        'over25_odds', 'under25_odds', 'btts_yes_odds', 'btts_no_odds',
    ]

    numeric_features = []
    for col in df.columns:
        if col not in exclude_cols and df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
            numeric_features.append(col)

    return df, numeric_features[:30]  # Top 30 features


def create_models():
    """Create all model configurations to compare."""
    models = {}

    # Individual callibration
    models['xgboost'] = XGBClassifier(
        n_estimators=200, max_depth=4, min_child_weight=15,
        reg_lambda=5.0, learning_rate=0.05, subsample=0.8,
        random_state=42, verbosity=0
    )

    models['lightgbm'] = LGBMClassifier(
        n_estimators=200, max_depth=4, min_child_samples=50,
        reg_lambda=5.0, learning_rate=0.05, subsample=0.8,
        random_state=42, verbose=-1
    )

    models['catboost'] = CatBoostClassifier(
        iterations=200, depth=4, l2_leaf_reg=10,
        learning_rate=0.05, random_state=42, verbose=0
    )

    models['random_forest'] = RandomForestClassifier(
        n_estimators=200, max_depth=6, min_samples_split=10,
        min_samples_leaf=5, random_state=42, n_jobs=-1
    )

    # Voting ensemble (simple average)
    models['voting_simple'] = VotingClassifier(
        estimators=[
            ('xgb', XGBClassifier(n_estimators=200, max_depth=4, min_child_weight=15,
                                  reg_lambda=5.0, learning_rate=0.05, random_state=42, verbosity=0)),
            ('lgbm', LGBMClassifier(n_estimators=200, max_depth=4, min_child_samples=50,
                                    reg_lambda=5.0, learning_rate=0.05, random_state=42, verbose=-1)),
            ('cat', CatBoostClassifier(iterations=200, depth=4, l2_leaf_reg=10,
                                       learning_rate=0.05, random_state=42, verbose=0)),
        ],
        voting='soft',
        n_jobs=-1
    )

    # Stacking with LogisticRegression meta-learner
    models['stacking_lr'] = StackingClassifier(
        estimators=[
            ('xgb', XGBClassifier(n_estimators=200, max_depth=4, min_child_weight=15,
                                  reg_lambda=5.0, learning_rate=0.05, random_state=42, verbosity=0)),
            ('lgbm', LGBMClassifier(n_estimators=200, max_depth=4, min_child_samples=50,
                                    reg_lambda=5.0, learning_rate=0.05, random_state=42, verbose=-1)),
            ('cat', CatBoostClassifier(iterations=200, depth=4, l2_leaf_reg=10,
                                       learning_rate=0.05, random_state=42, verbose=0)),
        ],
        final_estimator=LogisticRegression(max_iter=1000, random_state=42),
        cv=3,
        stack_method='predict_proba',
        n_jobs=-1
    )

    # Stacking with XGBoost meta-learner
    models['stacking_xgb'] = StackingClassifier(
        estimators=[
            ('xgb', XGBClassifier(n_estimators=200, max_depth=4, min_child_weight=15,
                                  reg_lambda=5.0, learning_rate=0.05, random_state=42, verbosity=0)),
            ('lgbm', LGBMClassifier(n_estimators=200, max_depth=4, min_child_samples=50,
                                    reg_lambda=5.0, learning_rate=0.05, random_state=42, verbose=-1)),
            ('cat', CatBoostClassifier(iterations=200, depth=4, l2_leaf_reg=10,
                                       learning_rate=0.05, random_state=42, verbose=0)),
        ],
        final_estimator=XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1,
                                      random_state=42, verbosity=0),
        cv=3,
        stack_method='predict_proba',
        n_jobs=-1
    )

    return models


def simulate_betting_roi(proba, y_test, odds, threshold=0.55, direction='over'):
    """Calculate ROI for a betting strategy."""
    if direction == 'over':
        bet_mask = proba >= threshold
        wins = y_test[bet_mask] == 1
    else:  # under
        bet_mask = (1 - proba) >= threshold
        wins = y_test[bet_mask] == 0

    n_bets = bet_mask.sum()
    if n_bets < 20:
        return None, 0

    profit = (wins * (odds - 1) - (~wins) * 1).sum()
    roi = (profit / n_bets) * 100
    return roi, n_bets


def evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test,
                   calibrate=True, odds_over=1.90, odds_under=1.90):
    """Evaluate a model with multiple metrics including betting ROI."""

    # Fit model
    model.fit(X_train, y_train)

    # Calibrate if requested
    if calibrate:
        try:
            cal_model = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
            cal_model.fit(X_val, y_val)
            proba = cal_model.predict_proba(X_test)[:, 1]
        except:
            proba = model.predict_proba(X_test)[:, 1]
    else:
        proba = model.predict_proba(X_test)[:, 1]

    # Standard metrics
    brier = brier_score_loss(y_test, proba)
    try:
        auc = roc_auc_score(y_test, proba)
    except:
        auc = 0.5

    accuracy = ((proba >= 0.5) == y_test).mean()

    # Betting ROI at different thresholds
    best_over_roi = -100
    best_under_roi = -100

    for thresh in [0.55, 0.60, 0.65, 0.70]:
        over_roi, over_bets = simulate_betting_roi(proba, y_test, odds_over, thresh, 'over')
        under_roi, under_bets = simulate_betting_roi(proba, y_test, odds_under, thresh, 'under')

        if over_roi is not None and over_roi > best_over_roi:
            best_over_roi = over_roi
        if under_roi is not None and under_roi > best_under_roi:
            best_under_roi = under_roi

    return {
        'brier': brier,
        'auc': auc,
        'accuracy': accuracy,
        'best_over_roi': best_over_roi,
        'best_under_roi': best_under_roi,
        'best_roi': max(best_over_roi, best_under_roi),
    }


def run_comparison(bet_type: str):
    """Run model comparison for a bet type."""
    print(f"\n{'='*70}")
    print(f"COMPARING MODELS FOR: {bet_type.upper()}")
    print(f"{'='*70}")

    # Load data
    if bet_type == 'corners':
        df, features = load_corners_data()
        target = 'over_10_5'
        odds_over, odds_under = 2.10, 1.72
    else:  # cards
        df, features = load_cards_data()
        target = 'over_3_5_cards'
        odds_over, odds_under = 1.85, 1.95

    if target not in df.columns:
        print(f"Target {target} not found in data")
        return None

    print(f"Data: {len(df)} matches")
    print(f"Features: {len(features)}")
    print(f"Target: {target}")

    # Temporal split
    n = len(df)
    train_end = int(0.6 * n)
    val_end = int(0.8 * n)

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Prepare data
    X_train = train_df[features].fillna(0).astype(float)
    X_val = val_df[features].fillna(0).astype(float)
    X_test = test_df[features].fillna(0).astype(float)

    y_train = train_df[target].values
    y_val = val_df[target].values
    y_test = test_df[target].values

    print(f"\nPositive rate - Train: {y_train.mean():.1%}, Test: {y_test.mean():.1%}")

    # Create callibration
    models = create_models()

    # Evaluate each model
    results = {}
    print(f"\n{'Model':<20} {'Brier':>8} {'AUC':>8} {'Acc':>8} {'Best ROI':>10}")
    print("-" * 60)

    for name, model in models.items():
        try:
            metrics = evaluate_model(
                model, X_train, y_train, X_val, y_val, X_test, y_test,
                calibrate=True, odds_over=odds_over, odds_under=odds_under
            )
            results[name] = metrics

            print(f"{name:<20} {metrics['brier']:>8.4f} {metrics['auc']:>8.3f} "
                  f"{metrics['accuracy']:>7.1%} {metrics['best_roi']:>9.1f}%")
        except Exception as e:
            print(f"{name:<20} ERROR: {str(e)[:40]}")
            results[name] = {'error': str(e)}

    # Find best model
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    if valid_results:
        best_by_roi = max(valid_results.items(), key=lambda x: x[1]['best_roi'])
        best_by_brier = min(valid_results.items(), key=lambda x: x[1]['brier'])

        print(f"\n{'='*60}")
        print(f"BEST MODEL BY ROI: {best_by_roi[0]} ({best_by_roi[1]['best_roi']:.1f}%)")
        print(f"BEST MODEL BY BRIER: {best_by_brier[0]} ({best_by_brier[1]['brier']:.4f})")

    return results


def main():
    all_results = {}

    # Compare for corners
    corners_results = run_comparison('corners')
    if corners_results:
        all_results['corners'] = corners_results

    # Compare for cards
    cards_results = run_comparison('cards')
    if cards_results:
        all_results['cards'] = cards_results

    # Summary
    print("\n" + "=" * 70)
    print("OVERALL RECOMMENDATIONS")
    print("=" * 70)

    for bet_type, results in all_results.items():
        valid = {k: v for k, v in results.items() if 'error' not in v}
        if valid:
            best = max(valid.items(), key=lambda x: x[1]['best_roi'])
            print(f"\n{bet_type.upper()}:")
            print(f"  Best model: {best[0]}")
            print(f"  Best ROI: {best[1]['best_roi']:.1f}%")
            print(f"  Brier score: {best[1]['brier']:.4f}")
            print(f"  AUC: {best[1]['auc']:.3f}")

    # Save results
    output_path = Path('experiments/outputs/model_comparison_niche.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to serializable format
    serializable = {}
    for bet_type, results in all_results.items():
        serializable[bet_type] = {}
        for model, metrics in results.items():
            if 'error' not in metrics:
                serializable[bet_type][model] = {
                    k: float(v) if isinstance(v, (np.floating, float)) else v
                    for k, v in metrics.items()
                }

    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
