#!/usr/bin/env python
"""
Validate xG Feature Improvements

Tests whether adding xG features improves model predictions for goals-based markets.

Usage:
    python experiments/validate_xg_improvement.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')


def load_features():
    """Load features file with xG data."""
    features_path = Path("data/03-features/features_all_5leagues_with_odds.csv")
    df = pd.read_csv(features_path)
    return df


def prepare_data_for_market(df: pd.DataFrame, market: str):
    """Prepare X, y for a specific market."""
    # Define target
    if market == 'btts':
        df['target'] = ((df['home_goals'] > 0) & (df['away_goals'] > 0)).astype(int)
    elif market == 'over25':
        df['target'] = ((df['home_goals'] + df['away_goals']) > 2.5).astype(int)
    elif market == 'under25':
        df['target'] = ((df['home_goals'] + df['away_goals']) <= 2.5).astype(int)
    elif market == 'home_win':
        df['target'] = (df['home_goals'] > df['away_goals']).astype(int)
    elif market == 'away_win':
        df['target'] = (df['away_goals'] > df['home_goals']).astype(int)
    else:
        raise ValueError(f"Unknown market: {market}")

    # Drop rows with missing target
    df = df.dropna(subset=['target', 'home_goals', 'away_goals'])

    return df


def get_feature_sets(df: pd.DataFrame):
    """Define feature sets with and without xG."""
    # Base features (excluding targets, identifiers, and xG)
    exclude_cols = [
        'fixture_id', 'date', 'fixture_date', 'season', 'league',
        'home_team_id', 'away_team_id', 'home_team_name', 'away_team_name',
        'home_goals', 'away_goals', 'target',
        'home_win', 'away_win', 'draw', 'btts', 'over25', 'under25',
        'total_goals', 'goal_margin'
    ]

    # xG columns
    xg_cols = [c for c in df.columns if c.startswith('xg_')]

    # All available numeric features
    all_features = [c for c in df.columns
                   if c not in exclude_cols
                   and df[c].dtype in ['int64', 'float64']
                   and df[c].notna().sum() > len(df) * 0.5]

    # Features without xG
    base_features = [c for c in all_features if c not in xg_cols]

    # Features with xG
    full_features = all_features

    return base_features, full_features, xg_cols


def evaluate_model(X_train, X_test, y_train, y_test, feature_set_name: str):
    """Train and evaluate model."""
    # Handle missing values
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    # Train model
    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        verbosity=0,
        use_label_encoder=False,
        eval_metric='logloss'
    )

    model.fit(X_train, y_train)

    # Predict
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Metrics
    metrics = {
        'feature_set': feature_set_name,
        'log_loss': log_loss(y_test, y_pred_proba),
        'brier_score': brier_score_loss(y_test, y_pred_proba),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'mean_pred': y_pred_proba.mean(),
        'actual_rate': y_test.mean()
    }

    return metrics, model


def run_validation():
    """Run full validation comparing with/without xG features."""
    print("=" * 70)
    print("VALIDATING xG FEATURE IMPROVEMENTS")
    print("=" * 70)

    # Load data
    df = load_features()
    print(f"Loaded {len(df)} matches")

    # Check xG columns exist
    xg_cols = [c for c in df.columns if c.startswith('xg_')]
    print(f"xG columns: {xg_cols}")

    if not xg_cols:
        print("\nERROR: No xG columns found! Run integrate_xg_data.py merge first.")
        return

    # Sort by date for time-series split
    if 'date' in df.columns:
        df['date_parsed'] = pd.to_datetime(df['date'])
        df = df.sort_values('date_parsed').reset_index(drop=True)
    elif 'fixture_date' in df.columns:
        df['date_parsed'] = pd.to_datetime(df['fixture_date'])
        df = df.sort_values('date_parsed').reset_index(drop=True)

    # Markets to test (goals-based)
    markets = ['btts', 'over25', 'under25', 'home_win', 'away_win']

    results = []

    for market in markets:
        print(f"\n{'=' * 70}")
        print(f"MARKET: {market.upper()}")
        print("=" * 70)

        # Prepare data
        market_df = prepare_data_for_market(df.copy(), market)
        base_features, full_features, xg_cols = get_feature_sets(market_df)

        print(f"Base features: {len(base_features)}")
        print(f"Full features (with xG): {len(full_features)}")
        print(f"xG features: {len(xg_cols)}")
        print(f"Target rate: {market_df['target'].mean():.3f}")

        # Time series split
        tscv = TimeSeriesSplit(n_splits=3)

        base_metrics_list = []
        xg_metrics_list = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(market_df)):
            train_df = market_df.iloc[train_idx]
            test_df = market_df.iloc[test_idx]

            y_train = train_df['target']
            y_test = test_df['target']

            # Test base features
            X_train_base = train_df[base_features]
            X_test_base = test_df[base_features]
            base_result, _ = evaluate_model(X_train_base, X_test_base, y_train, y_test, 'base')
            base_metrics_list.append(base_result)

            # Test with xG features
            X_train_xg = train_df[full_features]
            X_test_xg = test_df[full_features]
            xg_result, _ = evaluate_model(X_train_xg, X_test_xg, y_train, y_test, 'with_xg')
            xg_metrics_list.append(xg_result)

        # Average metrics
        base_avg = {
            'log_loss': np.mean([m['log_loss'] for m in base_metrics_list]),
            'brier_score': np.mean([m['brier_score'] for m in base_metrics_list]),
            'roc_auc': np.mean([m['roc_auc'] for m in base_metrics_list]),
        }
        xg_avg = {
            'log_loss': np.mean([m['log_loss'] for m in xg_metrics_list]),
            'brier_score': np.mean([m['brier_score'] for m in xg_metrics_list]),
            'roc_auc': np.mean([m['roc_auc'] for m in xg_metrics_list]),
        }

        # Calculate improvement
        log_loss_improvement = (base_avg['log_loss'] - xg_avg['log_loss']) / base_avg['log_loss'] * 100
        brier_improvement = (base_avg['brier_score'] - xg_avg['brier_score']) / base_avg['brier_score'] * 100
        auc_improvement = (xg_avg['roc_auc'] - base_avg['roc_auc']) / base_avg['roc_auc'] * 100

        print(f"\n--- Results for {market.upper()} ---")
        print(f"{'Metric':<15} {'Base':>12} {'With xG':>12} {'Change':>12}")
        print("-" * 55)
        print(f"{'Log Loss':<15} {base_avg['log_loss']:>12.4f} {xg_avg['log_loss']:>12.4f} {log_loss_improvement:>+11.2f}%")
        print(f"{'Brier Score':<15} {base_avg['brier_score']:>12.4f} {xg_avg['brier_score']:>12.4f} {brier_improvement:>+11.2f}%")
        print(f"{'ROC AUC':<15} {base_avg['roc_auc']:>12.4f} {xg_avg['roc_auc']:>12.4f} {auc_improvement:>+11.2f}%")

        # Store results
        results.append({
            'market': market,
            'base_log_loss': base_avg['log_loss'],
            'xg_log_loss': xg_avg['log_loss'],
            'log_loss_improvement_%': log_loss_improvement,
            'base_brier': base_avg['brier_score'],
            'xg_brier': xg_avg['brier_score'],
            'brier_improvement_%': brier_improvement,
            'base_auc': base_avg['roc_auc'],
            'xg_auc': xg_avg['roc_auc'],
            'auc_improvement_%': auc_improvement,
        })

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: xG Feature Impact")
    print("=" * 70)
    print(f"{'Market':<12} {'Log Loss Δ':>12} {'Brier Δ':>12} {'AUC Δ':>12} {'Verdict':>15}")
    print("-" * 65)

    for r in results:
        # Positive log_loss improvement means lower log_loss (better)
        # Positive brier improvement means lower brier (better)
        # Positive AUC improvement means higher AUC (better)
        verdict = "✓ IMPROVED" if (r['log_loss_improvement_%'] > 0.5 or r['auc_improvement_%'] > 0.5) else "~ No change"

        print(f"{r['market']:<12} {r['log_loss_improvement_%']:>+11.2f}% {r['brier_improvement_%']:>+11.2f}% {r['auc_improvement_%']:>+11.2f}% {verdict:>15}")

    # Save results
    results_df = pd.DataFrame(results)
    output_path = Path("experiments/outputs/xg_validation_results.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    run_validation()
