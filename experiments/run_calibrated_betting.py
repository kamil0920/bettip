"""
Calibration-focused betting analysis.

Research shows calibration optimization leads to 69.86% higher returns
than accuracy optimization (Walsh and Joshi, 2024).

Key insight: Model should output probabilities that match actual win rates.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import json
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path("outputs/calibrated_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTCOME_COLS = ['draw', 'home_win', 'away_win', 'match_result', 'total_goals', 'goal_difference', 'winner']
ID_COLS = ['date', 'home_team_name', 'away_team_name', 'home_team_id', 'away_team_id',
           'league', 'season', 'fixture_id', 'round_number']


def load_and_prepare_data(bet_type='away_win'):
    """Load data and prepare for specific bet type."""
    df = pd.read_csv("../data/03-features/features_all_with_odds.csv")
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    if bet_type == 'home_win':
        df['target'] = df['home_win'].astype(int)
        df['odds'] = df['avg_home_close']
        df['actual'] = df['home_win']
        df = df.dropna(subset=['avg_home_close'])
    elif bet_type == 'away_win':
        df['target'] = (df['match_result'] == -1).astype(int)
        df['odds'] = df['avg_away_close']
        df['actual'] = (df['match_result'] == -1).astype(int)
        df = df.dropna(subset=['avg_away_close'])

    return df


def get_clean_features(df):
    """Get clean feature columns."""
    exclude = set(OUTCOME_COLS + ID_COLS + ['target', 'odds', 'actual', 'week'])
    odds_cols = [c for c in df.columns if 'close' in c.lower()]
    exclude.update(odds_cols)
    feature_cols = [c for c in df.columns if c not in exclude]
    feature_cols = [c for c in feature_cols if df[c].dtype in ['int64', 'float64']]
    return feature_cols


def calculate_expected_value(prob, odds):
    """Calculate expected value of a bet."""
    # EV = (prob * (odds - 1)) - ((1 - prob) * 1)
    # EV = prob * odds - 1
    return prob * odds - 1


def find_value_bets(df_test, min_ev=0.05):
    """Find bets with positive expected value."""
    # Value bet = model probability implies better odds than bookmaker offers
    # If our prob > implied prob (1/odds), we have value

    df_test = df_test.copy()
    df_test['implied_prob'] = 1 / df_test['odds']
    df_test['ev'] = calculate_expected_value(df_test['prob'], df_test['odds'])
    df_test['edge'] = df_test['prob'] - df_test['implied_prob']

    # Only bet when we have positive EV above threshold
    value_bets = df_test[df_test['ev'] >= min_ev]

    return value_bets


def simulate_value_betting(df_test, min_ev=0.05, n_bootstrap=1000):
    """Simulate value betting strategy with bootstrap."""
    value_bets = find_value_bets(df_test, min_ev)

    if len(value_bets) < 10:
        return None

    profits = value_bets.apply(
        lambda r: r['odds'] - 1 if r['actual'] == 1 else -1,
        axis=1
    ).values

    actual_roi = profits.mean() * 100

    bootstrap_rois = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(profits, size=len(profits), replace=True)
        bootstrap_rois.append(sample.mean() * 100)

    bootstrap_rois = np.array(bootstrap_rois)

    return {
        'n_bets': len(value_bets),
        'min_ev': min_ev,
        'avg_edge': round(value_bets['edge'].mean() * 100, 2),
        'avg_ev': round(value_bets['ev'].mean() * 100, 2),
        'precision': round(value_bets['actual'].mean() * 100, 2),
        'avg_odds': round(value_bets['odds'].mean(), 2),
        'actual_roi': round(actual_roi, 2),
        'roi_5th_pct': round(np.percentile(bootstrap_rois, 5), 2),
        'roi_median': round(np.percentile(bootstrap_rois, 50), 2),
        'roi_95th_pct': round(np.percentile(bootstrap_rois, 95), 2),
        'prob_profitable': round((bootstrap_rois > 0).mean() * 100, 1),
    }


def main():
    print("="*70)
    print("CALIBRATION-FOCUSED VALUE BETTING ANALYSIS")
    print("="*70)
    print()
    print("Strategy: Only bet when expected value (EV) is positive")
    print("EV = (probability × odds) - 1")
    print()

    bet_types = ['away_win', 'home_win']

    all_results = []

    for bet_type in bet_types:
        print(f"\n{'='*60}")
        print(f"Analyzing: {bet_type.upper()}")
        print(f"{'='*60}")

        df = load_and_prepare_data(bet_type)
        feature_cols = get_clean_features(df)

        print(f"Using {len(feature_cols)} clean features")

        # Train/test split
        split_idx = int(len(df) * 0.8)
        X = df[feature_cols].fillna(df[feature_cols].median())
        y = df['target'].values

        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        df_test = df.iloc[split_idx:].copy()

        print(f"Train: {len(X_train)}, Test: {len(X_test)}")

        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train base model
        base_model = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0,
        )
        base_model.fit(X_train_scaled, y_train)

        # Get base probabilities
        base_probs = base_model.predict_proba(X_test_scaled)[:, 1]

        # Calibrate with isotonic regression
        # Use last 20% of training for calibration
        cal_split = int(len(X_train) * 0.8)
        X_cal = X_train_scaled[cal_split:]
        y_cal = y_train[cal_split:]
        cal_probs = base_model.predict_proba(X_cal)[:, 1]

        # Fit isotonic regression for calibration
        ir = IsotonicRegression(out_of_bounds='clip')
        ir.fit(cal_probs, y_cal)

        # Apply calibration to test set
        calibrated_probs = ir.transform(base_probs)

        df_test['prob_base'] = base_probs
        df_test['prob'] = calibrated_probs

        # Check calibration
        print("\n--- Calibration Check ---")
        for low, high in [(0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8)]:
            mask = (df_test['prob'] >= low) & (df_test['prob'] < high)
            if mask.sum() > 10:
                actual = df_test.loc[mask, 'actual'].mean()
                predicted = df_test.loc[mask, 'prob'].mean()
                print(f"  Prob [{low:.1f}-{high:.1f}]: predicted={predicted:.3f}, actual={actual:.3f}, n={mask.sum()}")

        # Test different EV thresholds
        print("\n--- Value Betting Results ---")
        ev_thresholds = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20]

        for min_ev in ev_thresholds:
            result = simulate_value_betting(df_test, min_ev)
            if result:
                result['bet_type'] = bet_type
                all_results.append(result)
                print(f"Min EV {min_ev:.0%}: {result['n_bets']} bets, "
                      f"avg edge {result['avg_edge']:.1f}%, "
                      f"precision {result['precision']:.1f}%, "
                      f"ROI: {result['actual_roi']:.1f}% "
                      f"[{result['roi_5th_pct']:.1f}% to {result['roi_95th_pct']:.1f}%] | "
                      f"P(profit): {result['prob_profitable']:.0f}%")

    # Save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / "value_betting_results.csv", index=False)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY: Best Value Betting Strategies")
    print("="*70)

    high_conf = results_df[results_df['prob_profitable'] >= 85].sort_values('roi_5th_pct', ascending=False)

    if len(high_conf) > 0:
        print("\n--- Strategies with >= 85% P(profit) ---")
        for _, row in high_conf.iterrows():
            print(f"  {row['bet_type']} @ min_ev={row['min_ev']:.0%}: "
                  f"ROI {row['actual_roi']:.1f}% [{row['roi_5th_pct']:.1f}% to {row['roi_95th_pct']:.1f}%], "
                  f"bets: {row['n_bets']:.0f}")
    else:
        print("\nNo strategies found with >= 85% P(profit)")

    print(f"\nResults saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
