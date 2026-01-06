#!/usr/bin/env python3
"""
Demo: Generate betting predictions using trained models.

This script demonstrates how to:
1. Load trained model configurations from the full optimization outputs
2. Retrain models with optimal parameters
3. Make predictions on test data (simulating new matches)
4. Generate betting recommendations with Kelly stakes

Usage:
    python experiments/demo_predictions.py
    python experiments/demo_predictions.py --bankroll 1000 --kelly-fraction 0.25
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression, Ridge
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def load_optimization_results(bet_type: str) -> dict:
    """Load optimization results from JSON."""
    path = project_root / f'experiments/outputs/{bet_type}_full_optimization.json'
    if not path.exists():
        raise FileNotFoundError(f"No optimization results found for {bet_type}. Run training first.")
    with open(path) as f:
        return json.load(f)


def load_data():
    """Load feature data."""
    path = project_root / 'data/03-features/features_all_5leagues_with_odds.csv'
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    return df


def get_numeric_features(df, exclude_cols):
    """Get numeric feature columns."""
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    feature_cols = [c for c in feature_cols if 'b365' not in c.lower()
                    and not (c.startswith('avg_') and any(x in c for x in ['home', 'away', 'draw', 'over', 'under', 'ah']))
                    and not (c.startswith('max_') and any(x in c for x in ['home', 'away', 'draw', 'over', 'under', 'ah']))
                    and 'pinnacle' not in c.lower()
                    and 'btts' not in c.lower()]
    numeric_cols = df[feature_cols].select_dtypes(include=['number']).columns.tolist()
    return numeric_cols


def calculate_kelly_stake(prob, odds, kelly_fraction=0.25, max_stake=0.05):
    """Calculate Kelly criterion stake."""
    if odds <= 1:
        return 0
    full_kelly = (prob * odds - 1) / (odds - 1)
    kelly = kelly_fraction * full_kelly
    return max(0, min(kelly, max_stake))


def demo_btts(df, config, bankroll=1000, kelly_fraction=0.25):
    """Demo BTTS predictions."""
    print("\n" + "=" * 70)
    print("BTTS (Both Teams To Score) - Demo")
    print("=" * 70)

    # Prepare data
    df_btts = df[df['btts_yes_avg'].notna()].copy()
    df_btts['target'] = ((df_btts['home_goals'] > 0) & (df_btts['away_goals'] > 0)).astype(int)

    exclude_cols = [
        'fixture_id', 'date', 'home_team_id', 'home_team_name', 'away_team_id',
        'away_team_name', 'round', 'match_result', 'home_win', 'draw', 'away_win',
        'total_goals', 'goal_difference', 'league', 'target', 'ah_result',
        'home_goals', 'away_goals', 'season', 'round_num', 'result',
        'btts_yes_avg', 'btts_no_avg', 'btts_yes_max', 'btts_no_max'
    ]

    feature_cols = get_numeric_features(df_btts, exclude_cols)

    # Time split - use last 20% as "new" matches for demo
    dates = pd.to_datetime(df_btts['date'])
    sorted_indices = dates.argsort()
    n = len(df_btts)
    train_val_idx = sorted_indices[:int(0.8*n)]
    demo_idx = sorted_indices[int(0.8*n):]

    X_train = df_btts.iloc[train_val_idx][feature_cols].copy()
    y_train = df_btts.iloc[train_val_idx]['target'].values
    X_demo = df_btts.iloc[demo_idx][feature_cols].copy()
    y_demo = df_btts.iloc[demo_idx]['target'].values
    odds_demo = df_btts.iloc[demo_idx]['btts_yes_avg'].values
    demo_df = df_btts.iloc[demo_idx].copy()

    # Fill NaN
    for col in X_train.columns:
        median = X_train[col].median()
        X_train[col] = X_train[col].fillna(median)
        X_demo[col] = X_demo[col].fillna(median)

    # Get best params from config
    best_params = config.get('best_params', {})
    threshold = 0.65  # Best threshold from training

    # Train stacking ensemble (simplified)
    models = {}

    # XGBoost
    xgb_params = {**best_params.get('XGBoost', {}), 'random_state': 42, 'verbosity': 0}
    xgb = XGBClassifier(**xgb_params)
    xgb.fit(X_train, y_train)
    models['XGBoost'] = xgb

    # LightGBM
    lgbm_params = {**best_params.get('LightGBM', {}), 'random_state': 42, 'verbose': -1}
    lgbm = LGBMClassifier(**lgbm_params)
    lgbm.fit(X_train, y_train)
    models['LightGBM'] = lgbm

    # CatBoost
    cat_params = {**best_params.get('CatBoost', {}), 'random_state': 42, 'verbose': 0}
    cat = CatBoostClassifier(**cat_params)
    cat.fit(X_train, y_train)
    models['CatBoost'] = cat

    # Get predictions
    preds = {}
    for name, model in models.items():
        preds[name] = model.predict_proba(X_demo)[:, 1]

    # Ensemble average
    ensemble_proba = np.mean([preds[n] for n in preds], axis=0)

    # Generate recommendations
    recommendations = []
    for i in range(len(X_demo)):
        prob = ensemble_proba[i]
        odds = odds_demo[i]

        if prob >= threshold:
            stake_pct = calculate_kelly_stake(prob, odds, kelly_fraction)
            stake = bankroll * stake_pct
            expected_value = prob * odds - 1

            recommendations.append({
                'date': demo_df.iloc[i]['date'],
                'home_team': demo_df.iloc[i]['home_team_name'],
                'away_team': demo_df.iloc[i]['away_team_name'],
                'prediction': 'BTTS Yes',
                'probability': prob,
                'odds': odds,
                'expected_value': expected_value,
                'kelly_stake_pct': stake_pct * 100,
                'stake': stake,
                'actual': 'WIN' if y_demo[i] == 1 else 'LOSS'
            })

    # Print top recommendations
    print(f"\nModel: Ensemble (XGBoost + LightGBM + CatBoost)")
    print(f"Threshold: {threshold}")
    print(f"Bankroll: ${bankroll:,.0f}")
    print(f"Kelly Fraction: {kelly_fraction}")

    if recommendations:
        rec_df = pd.DataFrame(recommendations).sort_values('expected_value', ascending=False)
        print(f"\nTop 10 Recommendations (showing {len(rec_df)} total bets):")
        print("-" * 100)
        print(f"{'Date':<12} {'Match':<35} {'Prob':>6} {'Odds':>6} {'EV':>7} {'Stake':>8} {'Result':>7}")
        print("-" * 100)

        for _, row in rec_df.head(10).iterrows():
            match = f"{row['home_team'][:15]} vs {row['away_team'][:15]}"
            print(f"{str(row['date'])[:10]:<12} {match:<35} {row['probability']:>5.1%} "
                  f"{row['odds']:>6.2f} {row['expected_value']:>+6.1%} ${row['stake']:>7.0f} {row['actual']:>7}")

        # Summary
        wins = sum(1 for r in recommendations if r['actual'] == 'WIN')
        total_staked = sum(r['stake'] for r in recommendations)
        profit = sum(r['stake'] * (r['odds'] - 1) if r['actual'] == 'WIN' else -r['stake'] for r in recommendations)
        roi = (profit / total_staked * 100) if total_staked > 0 else 0

        print("-" * 100)
        print(f"\nDemo Results: {wins}/{len(recommendations)} wins ({wins/len(recommendations)*100:.1f}%)")
        print(f"Total Staked: ${total_staked:,.0f}")
        print(f"Profit/Loss: ${profit:+,.0f}")
        print(f"ROI: {roi:+.1f}%")
    else:
        print("\nNo recommendations above threshold.")

    return recommendations


def demo_away_win(df, config, bankroll=1000, kelly_fraction=0.25):
    """Demo Away Win predictions."""
    print("\n" + "=" * 70)
    print("AWAY WIN - Demo")
    print("=" * 70)

    # Prepare data
    df_away = df[df['avg_away_open'].notna()].copy()
    df_away['target'] = df_away['away_win'].astype(int)

    exclude_cols = [
        'fixture_id', 'date', 'home_team_id', 'home_team_name', 'away_team_id',
        'away_team_name', 'round', 'match_result', 'home_win', 'draw', 'away_win',
        'total_goals', 'goal_difference', 'league', 'target', 'ah_result',
        'home_goals', 'away_goals', 'season', 'round_num', 'result',
        'avg_home_open', 'avg_away_open', 'avg_draw_open'
    ]

    feature_cols = get_numeric_features(df_away, exclude_cols)

    # Time split
    dates = pd.to_datetime(df_away['date'])
    sorted_indices = dates.argsort()
    n = len(df_away)
    train_val_idx = sorted_indices[:int(0.8*n)]
    demo_idx = sorted_indices[int(0.8*n):]

    X_train = df_away.iloc[train_val_idx][feature_cols].copy()
    y_train = df_away.iloc[train_val_idx]['target'].values
    X_demo = df_away.iloc[demo_idx][feature_cols].copy()
    y_demo = df_away.iloc[demo_idx]['target'].values
    odds_demo = df_away.iloc[demo_idx]['avg_away_open'].values
    demo_df = df_away.iloc[demo_idx].copy()

    # Fill NaN
    for col in X_train.columns:
        median = X_train[col].median()
        X_train[col] = X_train[col].fillna(median)
        X_demo[col] = X_demo[col].fillna(median)

    # Best params
    best_params = config.get('best_params', {})
    threshold = 0.7  # Best threshold from training

    # Train LightGBM (best model)
    lgbm_params = {**best_params.get('LightGBM', {}), 'random_state': 42, 'verbose': -1}
    lgbm = LGBMClassifier(**lgbm_params)
    lgbm.fit(X_train, y_train)

    # Get predictions
    proba = lgbm.predict_proba(X_demo)[:, 1]

    # Generate recommendations
    recommendations = []
    for i in range(len(X_demo)):
        prob = proba[i]
        odds = odds_demo[i]

        if prob >= threshold:
            stake_pct = calculate_kelly_stake(prob, odds, kelly_fraction)
            stake = bankroll * stake_pct
            expected_value = prob * odds - 1

            recommendations.append({
                'date': demo_df.iloc[i]['date'],
                'home_team': demo_df.iloc[i]['home_team_name'],
                'away_team': demo_df.iloc[i]['away_team_name'],
                'prediction': 'Away Win',
                'probability': prob,
                'odds': odds,
                'expected_value': expected_value,
                'kelly_stake_pct': stake_pct * 100,
                'stake': stake,
                'actual': 'WIN' if y_demo[i] == 1 else 'LOSS'
            })

    print(f"\nModel: LightGBM")
    print(f"Threshold: {threshold}")
    print(f"Bankroll: ${bankroll:,.0f}")

    if recommendations:
        rec_df = pd.DataFrame(recommendations).sort_values('expected_value', ascending=False)
        print(f"\nTop 10 Recommendations (showing {len(rec_df)} total bets):")
        print("-" * 100)
        print(f"{'Date':<12} {'Match':<35} {'Prob':>6} {'Odds':>6} {'EV':>7} {'Stake':>8} {'Result':>7}")
        print("-" * 100)

        for _, row in rec_df.head(10).iterrows():
            match = f"{row['home_team'][:15]} vs {row['away_team'][:15]}"
            print(f"{str(row['date'])[:10]:<12} {match:<35} {row['probability']:>5.1%} "
                  f"{row['odds']:>6.2f} {row['expected_value']:>+6.1%} ${row['stake']:>7.0f} {row['actual']:>7}")

        wins = sum(1 for r in recommendations if r['actual'] == 'WIN')
        total_staked = sum(r['stake'] for r in recommendations)
        profit = sum(r['stake'] * (r['odds'] - 1) if r['actual'] == 'WIN' else -r['stake'] for r in recommendations)
        roi = (profit / total_staked * 100) if total_staked > 0 else 0

        print("-" * 100)
        print(f"\nDemo Results: {wins}/{len(recommendations)} wins ({wins/len(recommendations)*100:.1f}%)")
        print(f"Total Staked: ${total_staked:,.0f}")
        print(f"Profit/Loss: ${profit:+,.0f}")
        print(f"ROI: {roi:+.1f}%")

    return recommendations


def demo_asian_handicap(df, config, bankroll=1000, kelly_fraction=0.25):
    """Demo Asian Handicap predictions."""
    print("\n" + "=" * 70)
    print("ASIAN HANDICAP (Home Covers) - Demo")
    print("=" * 70)

    # Prepare data
    df_ah = df[df['ah_line'].notna() & df['avg_ah_home'].notna()].copy()
    df_ah['target'] = df_ah['goal_difference'].astype(float)
    df_ah['home_covers'] = (df_ah['goal_difference'] + df_ah['ah_line'] > 0).astype(int)

    exclude_cols = [
        'fixture_id', 'date', 'home_team_id', 'home_team_name', 'away_team_id',
        'away_team_name', 'round', 'match_result', 'home_win', 'draw', 'away_win',
        'total_goals', 'goal_difference', 'league', 'target', 'ah_result',
        'home_goals', 'away_goals', 'season', 'round_num', 'result', 'home_covers',
        'ah_line', 'avg_ah_home', 'avg_ah_away'
    ]

    feature_cols = get_numeric_features(df_ah, exclude_cols)

    # Time split
    dates = pd.to_datetime(df_ah['date'])
    sorted_indices = dates.argsort()
    n = len(df_ah)
    train_val_idx = sorted_indices[:int(0.8*n)]
    demo_idx = sorted_indices[int(0.8*n):]

    X_train = df_ah.iloc[train_val_idx][feature_cols].copy()
    y_train = df_ah.iloc[train_val_idx]['target'].values
    X_demo = df_ah.iloc[demo_idx][feature_cols].copy()
    y_demo = df_ah.iloc[demo_idx]['target'].values
    ah_line_demo = df_ah.iloc[demo_idx]['ah_line'].values
    odds_demo = df_ah.iloc[demo_idx]['avg_ah_home'].values
    home_covers_demo = df_ah.iloc[demo_idx]['home_covers'].values
    demo_df = df_ah.iloc[demo_idx].copy()

    # Fill NaN
    for col in X_train.columns:
        median = X_train[col].median()
        X_train[col] = X_train[col].fillna(median)
        X_demo[col] = X_demo[col].fillna(median)

    # Best params
    best_params = config.get('best_params', {})
    margin_buffer = 0.75  # Best buffer from training

    # Train CatBoost regressor (best model)
    cat_params = {**best_params.get('CatBoost', {}), 'random_state': 42, 'verbose': 0}
    cat = CatBoostRegressor(**cat_params)
    cat.fit(X_train, y_train)

    # Get predictions
    pred_margin = cat.predict(X_demo)

    # Generate recommendations
    recommendations = []
    for i in range(len(X_demo)):
        # Bet home covers when predicted margin > -ah_line + buffer
        if pred_margin[i] > -ah_line_demo[i] + margin_buffer:
            odds = odds_demo[i]
            # Estimate probability from historical data
            prob = 0.60  # Conservative estimate based on precision from training
            stake_pct = calculate_kelly_stake(prob, odds, kelly_fraction)
            stake = bankroll * stake_pct
            expected_value = prob * odds - 1

            recommendations.append({
                'date': demo_df.iloc[i]['date'],
                'home_team': demo_df.iloc[i]['home_team_name'],
                'away_team': demo_df.iloc[i]['away_team_name'],
                'prediction': f'Home covers (line: {ah_line_demo[i]:.1f})',
                'pred_margin': pred_margin[i],
                'ah_line': ah_line_demo[i],
                'odds': odds,
                'expected_value': expected_value,
                'kelly_stake_pct': stake_pct * 100,
                'stake': stake,
                'actual': 'WIN' if home_covers_demo[i] == 1 else 'LOSS'
            })

    print(f"\nModel: CatBoost Regressor")
    print(f"Margin Buffer: {margin_buffer}")
    print(f"Bankroll: ${bankroll:,.0f}")

    if recommendations:
        rec_df = pd.DataFrame(recommendations).sort_values('pred_margin', ascending=False)
        print(f"\nTop 10 Recommendations (showing {len(rec_df)} total bets):")
        print("-" * 110)
        print(f"{'Date':<12} {'Match':<30} {'Pred':>6} {'Line':>6} {'Odds':>6} {'Stake':>8} {'Result':>7}")
        print("-" * 110)

        for _, row in rec_df.head(10).iterrows():
            match = f"{row['home_team'][:13]} vs {row['away_team'][:13]}"
            print(f"{str(row['date'])[:10]:<12} {match:<30} {row['pred_margin']:>+5.2f} "
                  f"{row['ah_line']:>+5.1f} {row['odds']:>6.2f} ${row['stake']:>7.0f} {row['actual']:>7}")

        wins = sum(1 for r in recommendations if r['actual'] == 'WIN')
        total_staked = sum(r['stake'] for r in recommendations)
        profit = sum(r['stake'] * (r['odds'] - 1) if r['actual'] == 'WIN' else -r['stake'] for r in recommendations)
        roi = (profit / total_staked * 100) if total_staked > 0 else 0

        print("-" * 110)
        print(f"\nDemo Results: {wins}/{len(recommendations)} wins ({wins/len(recommendations)*100:.1f}%)")
        print(f"Total Staked: ${total_staked:,.0f}")
        print(f"Profit/Loss: ${profit:+,.0f}")
        print(f"ROI: {roi:+.1f}%")

    return recommendations


def main():
    parser = argparse.ArgumentParser(description='Demo betting predictions')
    parser.add_argument('--bankroll', type=float, default=1000, help='Starting bankroll')
    parser.add_argument('--kelly-fraction', type=float, default=0.25, help='Kelly fraction (0.25 = quarter Kelly)')
    args = parser.parse_args()

    print("=" * 70)
    print("BETTING PREDICTIONS DEMO")
    print("=" * 70)
    print(f"\nBankroll: ${args.bankroll:,.0f}")
    print(f"Kelly Fraction: {args.kelly_fraction} (quarter Kelly)")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data
    print("\nLoading data...")
    df = load_data()
    print(f"Loaded {len(df)} matches")

    # Load configs and run demos
    all_recommendations = {}

    try:
        btts_config = load_optimization_results('btts')
        all_recommendations['btts'] = demo_btts(df, btts_config, args.bankroll, args.kelly_fraction)
    except FileNotFoundError as e:
        print(f"\nSkipping BTTS: {e}")

    try:
        away_config = load_optimization_results('away_win')
        all_recommendations['away_win'] = demo_away_win(df, away_config, args.bankroll, args.kelly_fraction)
    except FileNotFoundError as e:
        print(f"\nSkipping Away Win: {e}")

    try:
        ah_config = load_optimization_results('asian_handicap')
        all_recommendations['asian_handicap'] = demo_asian_handicap(df, ah_config, args.bankroll, args.kelly_fraction)
    except FileNotFoundError as e:
        print(f"\nSkipping Asian Handicap: {e}")

    # Overall summary
    print("\n" + "=" * 70)
    print("OVERALL DEMO SUMMARY")
    print("=" * 70)

    total_bets = 0
    total_wins = 0
    total_staked = 0
    total_profit = 0

    for bet_type, recs in all_recommendations.items():
        if recs:
            wins = sum(1 for r in recs if r['actual'] == 'WIN')
            staked = sum(r['stake'] for r in recs)
            profit = sum(r['stake'] * (r['odds'] - 1) if r['actual'] == 'WIN' else -r['stake'] for r in recs)

            total_bets += len(recs)
            total_wins += wins
            total_staked += staked
            total_profit += profit

    if total_bets > 0:
        print(f"\nTotal Bets: {total_bets}")
        print(f"Win Rate: {total_wins}/{total_bets} ({total_wins/total_bets*100:.1f}%)")
        print(f"Total Staked: ${total_staked:,.0f}")
        print(f"Total Profit: ${total_profit:+,.0f}")
        print(f"Overall ROI: {total_profit/total_staked*100:+.1f}%")
        print(f"Final Bankroll: ${args.bankroll + total_profit:,.0f}")


if __name__ == '__main__':
    main()
