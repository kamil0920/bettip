"""
Bootstrap validation to get confidence intervals on ROI estimates.
This tells us the realistic range of expected ROI, not just a lucky point estimate.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import json
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path("outputs/bootstrap_validation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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


def get_feature_columns(df):
    """Get feature columns excluding targets and leaky columns."""
    exclude = [
        'date', 'home_team_name', 'away_team_name', 'target', 'odds', 'actual',
        'week', 'home_team_id', 'away_team_id', 'match_result', 'home_win', 'away_win',
        'total_goals', 'goal_difference', 'league', 'season', 'fixture_id', 'round_number'
    ]
    # Exclude odds columns to prevent leakage
    feature_cols = [c for c in df.columns if c not in exclude]
    feature_cols = [c for c in feature_cols if 'b365' not in c and 'avg_' not in c and 'max_' not in c]
    feature_cols = [c for c in feature_cols if df[c].dtype in ['int64', 'float64']]
    return feature_cols


def bootstrap_single_bet_roi(df_test, threshold, n_bootstrap=1000):
    """Bootstrap single bet ROI to get confidence intervals."""
    confident = df_test[df_test['prob'] >= threshold].copy()

    if len(confident) < 10:
        return None

    # Calculate actual ROI
    profits = confident.apply(
        lambda r: r['odds'] - 1 if r['actual'] == 1 else -1,
        axis=1
    ).values

    actual_roi = profits.mean() * 100

    # Bootstrap
    bootstrap_rois = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(profits, size=len(profits), replace=True)
        bootstrap_rois.append(sample.mean() * 100)

    bootstrap_rois = np.array(bootstrap_rois)

    return {
        'n_bets': len(confident),
        'precision': confident['actual'].mean() * 100,
        'actual_roi': round(actual_roi, 2),
        'roi_5th_pct': round(np.percentile(bootstrap_rois, 5), 2),
        'roi_median': round(np.percentile(bootstrap_rois, 50), 2),
        'roi_95th_pct': round(np.percentile(bootstrap_rois, 95), 2),
        'prob_profitable': round((bootstrap_rois > 0).mean() * 100, 1),
    }


def bootstrap_accumulator_roi(df_test, threshold, acc_size, n_bootstrap=1000):
    """Bootstrap accumulator ROI to get confidence intervals."""
    confident = df_test[df_test['prob'] >= threshold].copy()
    confident['week'] = confident['date'].dt.isocalendar().week.astype(str) + '-' + confident['date'].dt.year.astype(str)

    # Build accumulators
    accumulators = []
    for week, week_matches in confident.groupby('week'):
        if len(week_matches) < acc_size:
            continue
        top_n = week_matches.nlargest(acc_size, 'prob')
        combined_odds = top_n['odds'].prod()
        all_won = (top_n['actual'] == 1).all()
        profit = combined_odds - 1 if all_won else -1
        accumulators.append({'profit': profit, 'won': all_won, 'odds': combined_odds})

    if len(accumulators) < 10:
        return None

    acc_df = pd.DataFrame(accumulators)
    profits = acc_df['profit'].values
    actual_roi = profits.mean() * 100

    # Bootstrap
    bootstrap_rois = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(profits, size=len(profits), replace=True)
        bootstrap_rois.append(sample.mean() * 100)

    bootstrap_rois = np.array(bootstrap_rois)

    return {
        'n_accumulators': len(acc_df),
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
    print("BOOTSTRAP VALIDATION - CONFIDENCE INTERVALS ON ROI")
    print("="*70)
    print()

    bet_types = ['home_win', 'away_win', 'under_2.5', 'over_2.5']
    thresholds = [0.5, 0.6, 0.7, 0.8]
    acc_sizes = [2, 3, 4]

    all_results = []

    for bet_type in bet_types:
        print(f"\n{'='*60}")
        print(f"Analyzing: {bet_type.upper()}")
        print(f"{'='*60}")

        # Load and prepare data
        df = load_and_prepare_data(bet_type)
        feature_cols = get_feature_columns(df)

        # Train/test split
        split_idx = int(len(df) * 0.8)
        X = df[feature_cols].fillna(df[feature_cols].median())
        y = df['target'].values

        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        df_test = df.iloc[split_idx:].copy()

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
        df_test['prob'] = model.predict_proba(X_test_scaled)[:, 1]

        print(f"\nTest set: {len(df_test)} matches")
        print(f"Target rate: {df_test['target'].mean():.1%}")

        # Single bets
        print("\n--- Single Bets (Bootstrap 95% CI) ---")
        for thresh in thresholds:
            result = bootstrap_single_bet_roi(df_test, thresh)
            if result:
                result['bet_type'] = bet_type
                result['strategy'] = 'single'
                result['threshold'] = thresh
                all_results.append(result)
                print(f"Threshold {thresh}: {result['n_bets']} bets | "
                      f"ROI: {result['actual_roi']:.1f}% "
                      f"[{result['roi_5th_pct']:.1f}% - {result['roi_95th_pct']:.1f}%] | "
                      f"P(profit): {result['prob_profitable']:.0f}%")

        # Accumulators
        for acc_size in acc_sizes:
            print(f"\n--- {acc_size}-Match Accumulators (Bootstrap 95% CI) ---")
            for thresh in thresholds:
                result = bootstrap_accumulator_roi(df_test, thresh, acc_size)
                if result:
                    result['bet_type'] = bet_type
                    result['strategy'] = f'acc_{acc_size}'
                    result['threshold'] = thresh
                    all_results.append(result)
                    print(f"Threshold {thresh}: {result['n_accumulators']} accs | "
                          f"ROI: {result['actual_roi']:.1f}% "
                          f"[{result['roi_5th_pct']:.1f}% - {result['roi_95th_pct']:.1f}%] | "
                          f"P(profit): {result['prob_profitable']:.0f}%")

    # Save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / "bootstrap_results.csv", index=False)

    # Summary: Best strategies with high confidence of profitability
    print("\n" + "="*70)
    print("BEST STRATEGIES (>= 90% probability of profit)")
    print("="*70)

    high_conf = results_df[results_df['prob_profitable'] >= 90].sort_values('roi_5th_pct', ascending=False)
    if len(high_conf) > 0:
        print("\nThese strategies have 90%+ chance of being profitable:")
        for _, row in high_conf.head(10).iterrows():
            print(f"  {row['bet_type']} {row['strategy']} @ {row['threshold']}: "
                  f"ROI {row['actual_roi']:.1f}% [{row['roi_5th_pct']:.1f}% - {row['roi_95th_pct']:.1f}%]")

    # Save summary
    summary = {
        'analysis_date': pd.Timestamp.now().isoformat(),
        'n_bootstrap': 1000,
        'test_period': f"{df_test['date'].min().date()} to {df_test['date'].max().date()}",
        'high_confidence_strategies': high_conf.to_dict('records') if len(high_conf) > 0 else []
    }

    with open(OUTPUT_DIR / "bootstrap_summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nResults saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
