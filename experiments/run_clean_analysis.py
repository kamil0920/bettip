"""
Clean betting analysis with proper feature exclusion.
CRITICAL: Exclude all outcome-leaking features.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import json
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path("outputs/clean_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# CRITICAL: These are outcome columns that must be excluded
OUTCOME_COLS = [
    'draw', 'home_win', 'away_win', 'match_result',
    'total_goals', 'goal_difference', 'winner'
]

# Identifier columns
ID_COLS = [
    'date', 'home_team_name', 'away_team_name', 'home_team_id', 'away_team_id',
    'league', 'season', 'fixture_id', 'round_number'
]


def load_and_prepare_data(bet_type='home_win'):
    """Load data and prepare for specific bet type."""
    df = pd.read_csv("../data/03-features/features_all_with_odds.csv")
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    if bet_type == 'home_win':
        df['target'] = ((df['home_win'] == 1) & (df['avg_home_close'] > 1.0)).astype(int)
        df['odds'] = df['avg_home_close']
        df['actual'] = df['home_win']
        df = df.dropna(subset=['avg_home_close'])
    elif bet_type == 'away_win':
        df['target'] = ((df['match_result'] == -1) & (df['avg_away_close'] > 1.0)).astype(int)
        df['odds'] = df['avg_away_close']
        df['actual'] = (df['match_result'] == -1).astype(int)
        df = df.dropna(subset=['avg_away_close'])
    elif bet_type == 'under_2.5':
        df['target'] = ((df['total_goals'] <= 2.5) & (df['avg_under25_close'] > 1.0)).astype(int)
        df['odds'] = df['avg_under25_close']
        df['actual'] = (df['total_goals'] <= 2.5).astype(int)
        df = df.dropna(subset=['avg_under25_close', 'total_goals'])
    elif bet_type == 'over_2.5':
        df['target'] = ((df['total_goals'] > 2.5) & (df['avg_over25_close'] > 1.0)).astype(int)
        df['odds'] = df['avg_over25_close']
        df['actual'] = (df['total_goals'] > 2.5).astype(int)
        df = df.dropna(subset=['avg_over25_close', 'total_goals'])

    return df


def get_clean_features(df):
    """Get feature columns excluding ALL outcome-leaking columns."""
    exclude = set(OUTCOME_COLS + ID_COLS + ['target', 'odds', 'actual', 'week'])

    # Also exclude any odds columns that might leak info
    # Closing odds are set closer to match time and may partially reflect outcomes
    # Keep only opening odds which are set well before the match
    odds_cols = [c for c in df.columns if 'close' in c.lower()]
    exclude.update(odds_cols)

    feature_cols = [c for c in df.columns if c not in exclude]

    # Only numeric columns
    feature_cols = [c for c in feature_cols if df[c].dtype in ['int64', 'float64']]

    return feature_cols


def bootstrap_single_bet_roi(df_test, threshold, n_bootstrap=1000):
    """Bootstrap single bet ROI."""
    confident = df_test[df_test['prob'] >= threshold].copy()

    if len(confident) < 10:
        return None

    profits = confident.apply(
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
        'n_bets': len(confident),
        'precision': round(confident['actual'].mean() * 100, 2),
        'avg_odds': round(confident['odds'].mean(), 2),
        'actual_roi': round(actual_roi, 2),
        'roi_5th_pct': round(np.percentile(bootstrap_rois, 5), 2),
        'roi_median': round(np.percentile(bootstrap_rois, 50), 2),
        'roi_95th_pct': round(np.percentile(bootstrap_rois, 95), 2),
        'prob_profitable': round((bootstrap_rois > 0).mean() * 100, 1),
    }


def bootstrap_accumulator_roi(df_test, threshold, acc_size, n_bootstrap=1000):
    """Bootstrap accumulator ROI."""
    confident = df_test[df_test['prob'] >= threshold].copy()
    confident['week'] = confident['date'].dt.isocalendar().week.astype(str) + '-' + confident['date'].dt.year.astype(str)

    accumulators = []
    for week, week_matches in confident.groupby('week'):
        if len(week_matches) < acc_size:
            continue
        top_n = week_matches.nlargest(acc_size, 'prob')
        combined_odds = top_n['odds'].prod()
        all_won = (top_n['actual'] == 1).all()
        accumulators.append({'profit': combined_odds - 1 if all_won else -1, 'won': all_won, 'odds': combined_odds})

    if len(accumulators) < 10:
        return None

    acc_df = pd.DataFrame(accumulators)
    profits = acc_df['profit'].values
    actual_roi = profits.mean() * 100

    bootstrap_rois = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(profits, size=len(profits), replace=True)
        bootstrap_rois.append(sample.mean() * 100)

    bootstrap_rois = np.array(bootstrap_rois)

    return {
        'n_accs': len(acc_df),
        'win_rate': round(acc_df['won'].mean() * 100, 2),
        'avg_odds': round(acc_df['odds'].mean(), 2),
        'actual_roi': round(actual_roi, 2),
        'roi_5th_pct': round(np.percentile(bootstrap_rois, 5), 2),
        'roi_median': round(np.percentile(bootstrap_rois, 50), 2),
        'roi_95th_pct': round(np.percentile(bootstrap_rois, 95), 2),
        'prob_profitable': round((bootstrap_rois > 0).mean() * 100, 1),
    }


def main():
    print("="*70)
    print("CLEAN BETTING ANALYSIS (No Data Leakage)")
    print("="*70)
    print()
    print("CRITICAL FIX: Excluding outcome columns: draw, home_win, away_win,")
    print("              match_result, total_goals, goal_difference")
    print()

    bet_types = ['home_win', 'away_win', 'under_2.5', 'over_2.5']
    thresholds = [0.5, 0.55, 0.6, 0.65, 0.7]
    acc_sizes = [2, 3, 4]

    all_results = []

    for bet_type in bet_types:
        print(f"\n{'='*60}")
        print(f"Analyzing: {bet_type.upper()}")
        print(f"{'='*60}")

        df = load_and_prepare_data(bet_type)
        feature_cols = get_clean_features(df)

        print(f"Using {len(feature_cols)} CLEAN features (no outcome leakage)")

        # Train/test split (80/20)
        split_idx = int(len(df) * 0.8)
        X = df[feature_cols].fillna(df[feature_cols].median())
        y = df['target'].values

        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        df_test = df.iloc[split_idx:].copy()

        print(f"Train: {len(X_train)}, Test: {len(X_test)}")
        print(f"Test period: {df_test['date'].min().date()} to {df_test['date'].max().date()}")
        print(f"Target rate in test: {df_test['target'].mean():.1%}")

        # Scale and train
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0,
            scale_pos_weight=len(y_train[y_train==0]) / max(len(y_train[y_train==1]), 1)
        )
        model.fit(X_train_scaled, y_train)

        # Get predictions
        df_test['prob'] = model.predict_proba(X_test_scaled)[:, 1]

        # Top features
        importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        print(f"\nTop 5 features: {importance.head(5)['feature'].tolist()}")

        # Single bets
        print("\n--- Single Bets (Bootstrap 95% CI) ---")
        for thresh in thresholds:
            result = bootstrap_single_bet_roi(df_test, thresh)
            if result:
                result['bet_type'] = bet_type
                result['strategy'] = 'single'
                result['threshold'] = thresh
                all_results.append(result)
                print(f"Threshold {thresh}: {result['n_bets']} bets, "
                      f"precision {result['precision']:.1f}%, "
                      f"ROI: {result['actual_roi']:.1f}% "
                      f"[{result['roi_5th_pct']:.1f}% to {result['roi_95th_pct']:.1f}%] | "
                      f"P(profit): {result['prob_profitable']:.0f}%")

        # Accumulators
        for acc_size in acc_sizes:
            print(f"\n--- {acc_size}-Match Accumulators ---")
            for thresh in thresholds:
                result = bootstrap_accumulator_roi(df_test, thresh, acc_size)
                if result:
                    result['bet_type'] = bet_type
                    result['strategy'] = f'acc_{acc_size}'
                    result['threshold'] = thresh
                    all_results.append(result)
                    print(f"Threshold {thresh}: {result['n_accs']} accs, "
                          f"win rate {result['win_rate']:.1f}%, "
                          f"ROI: {result['actual_roi']:.1f}% "
                          f"[{result['roi_5th_pct']:.1f}% to {result['roi_95th_pct']:.1f}%] | "
                          f"P(profit): {result['prob_profitable']:.0f}%")

    # Save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / "clean_results.csv", index=False)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY: Best Strategies with Clean Features")
    print("="*70)

    # High confidence strategies
    high_conf = results_df[results_df['prob_profitable'] >= 90].sort_values('roi_5th_pct', ascending=False)

    if len(high_conf) > 0:
        print("\n--- Strategies with >= 90% P(profit) ---")
        for _, row in high_conf.head(10).iterrows():
            vol = row.get('n_bets', row.get('n_accs', 0))
            print(f"  {row['bet_type']} {row['strategy']} @ {row['threshold']}: "
                  f"ROI {row['actual_roi']:.1f}% [{row['roi_5th_pct']:.1f}% to {row['roi_95th_pct']:.1f}%], "
                  f"volume: {vol:.0f}")
    else:
        print("\nNo strategies found with >= 90% probability of profit.")
        print("This may indicate the edge is smaller or non-existent with clean features.")

    # Best per bet type
    print("\n--- Best Strategy per Bet Type ---")
    for bet_type in bet_types:
        bt_results = results_df[results_df['bet_type'] == bet_type].sort_values('actual_roi', ascending=False)
        if len(bt_results) > 0:
            best = bt_results.iloc[0]
            print(f"  {bet_type}: {best['strategy']} @ {best['threshold']} -> "
                  f"ROI {best['actual_roi']:.1f}% [{best['roi_5th_pct']:.1f}% to {best['roi_95th_pct']:.1f}%], "
                  f"P(profit): {best['prob_profitable']:.0f}%")

    # Save summary
    with open(OUTPUT_DIR / "clean_summary.json", 'w') as f:
        json.dump({
            'analysis_date': pd.Timestamp.now().isoformat(),
            'bug_fixed': 'Removed outcome leakage columns',
            'excluded_columns': OUTCOME_COLS,
            'n_strategies_tested': len(results_df),
            'high_confidence_strategies': len(high_conf),
        }, f, indent=2)

    print(f"\nResults saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
