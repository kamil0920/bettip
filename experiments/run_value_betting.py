#!/usr/bin/env python3
"""
Value Betting Analysis - Compare model probability vs market odds.

Instead of fixed thresholds (e.g., bet when >80% confident),
we bet when our probability exceeds the market's implied probability.

Value = Our Probability - Implied Probability
If Value > 0, we have an edge.
"""
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier, VotingRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def odds_to_probability(odds: float) -> float:
    """Convert decimal odds to implied probability."""
    if odds <= 1 or pd.isna(odds):
        return np.nan
    return 1 / odds


def calculate_expected_value(our_prob: float, odds: float) -> float:
    """Calculate expected value of a bet.

    EV = (probability * profit) - (1 - probability) * stake
    For a $1 stake: EV = prob * (odds - 1) - (1 - prob) * 1
    """
    if pd.isna(odds) or odds <= 1:
        return np.nan
    return our_prob * (odds - 1) - (1 - our_prob)


def kelly_stake(our_prob: float, odds: float, fraction: float = 0.25) -> float:
    """Calculate Kelly criterion stake.

    Kelly = (bp - q) / b
    where b = odds - 1, p = our probability, q = 1 - p
    """
    if pd.isna(odds) or odds <= 1 or our_prob <= 0:
        return 0
    b = odds - 1
    q = 1 - our_prob
    kelly = (b * our_prob - q) / b
    return max(0, kelly * fraction)  # Fractional Kelly


def load_data():
    """Load data with odds."""
    path = project_root / 'data/03-features/features_with_real_xg.csv'
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    return df


def get_clean_features(df: pd.DataFrame) -> List[str]:
    """Get features without data leakage."""
    exclude_cols = {
        'fixture_id', 'date', 'home_team_id', 'home_team_name', 'away_team_id',
        'away_team_name', 'round', 'match_result', 'home_win', 'draw', 'away_win',
        'total_goals', 'goal_difference', 'league', 'target', 'ah_result',
        'home_goals', 'away_goals', 'season', 'round_num', 'result',
        'home_team', 'away_team', 'time',
        # Exclude ALL odds to prevent leakage
        'ah_line', 'ah_line_close',
        'avg_over25', 'avg_under25', 'avg_over25_close', 'avg_under25_close',
        'avg_ah_home', 'avg_ah_away', 'avg_ah_home_close', 'avg_ah_away_close',
        'max_ah_home', 'max_ah_away', 'max_ah_home_close', 'max_ah_away_close',
        'b365_ah_home', 'b365_ah_away', 'b365_ah_home_close', 'b365_ah_away_close',
        'pinnacle_ah_home', 'pinnacle_ah_away', 'pinnacle_ah_home_close', 'pinnacle_ah_away_close',
        'b365_over25', 'b365_under25', 'b365_over25_close', 'b365_under25_close',
        'odds_home_prob', 'odds_draw_prob', 'odds_away_prob', 'odds_overround',
        'odds_move_home', 'odds_move_draw', 'odds_move_away',
        'odds_prob_move_home', 'odds_prob_move_away',
        'odds_move_home_pct', 'odds_move_away_pct',
        'odds_steam_home', 'odds_steam_away', 'odds_home_favorite',
        'odds_prob_diff', 'odds_prob_max', 'odds_entropy',
        'odds_upset_potential', 'odds_draw_relative',
        'odds_over25_prob', 'odds_under25_prob', 'odds_goals_expectation',
        'b365_home_open', 'b365_draw_open', 'b365_away_open',
        'b365_home_close', 'b365_draw_close', 'b365_away_close',
        'avg_home_open', 'avg_draw_open', 'avg_away_open',
        'avg_home_close', 'avg_draw_close', 'avg_away_close',
        'max_home_open', 'max_draw_open', 'max_away_open',
        'max_home_close', 'max_draw_close', 'max_away_close',
        'btts_yes_avg', 'btts_no_avg', 'btts_yes_max', 'btts_no_max',
        'home_xg_overperform', 'away_xg_overperform',
    }

    feature_cols = [c for c in df.columns if c not in exclude_cols]
    numeric_df = df[feature_cols].select_dtypes(include=['number'])
    return numeric_df.columns.tolist()


def create_ensemble_classifier(params: dict = None):
    """Create ensemble classifier."""
    if params is None:
        params = {
            'xgb': {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.05},
            'lgbm': {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.05},
            'cat': {'iterations': 200, 'depth': 5, 'learning_rate': 0.05},
        }

    xgb = XGBClassifier(**params.get('xgb', {}), random_state=42, verbosity=0, eval_metric='logloss')
    lgbm = LGBMClassifier(**params.get('lgbm', {}), random_state=42, verbose=-1)
    cat = CatBoostClassifier(**params.get('cat', {}), random_state=42, verbose=0)

    return VotingClassifier([('xgb', xgb), ('lgbm', lgbm), ('cat', cat)], voting='soft')


def run_value_betting_backtest(
    df: pd.DataFrame,
    feature_cols: List[str],
    bet_type: str,
    odds_col: str,
    min_edge: float = 0.05,  # Minimum 5% edge required
    start_season: int = 2024,
) -> Dict:
    """
    Run value betting backtest.

    Instead of fixed probability threshold, we bet when:
    our_probability > implied_probability + min_edge
    """
    print(f"\n{'='*60}")
    print(f"VALUE BETTING BACKTEST: {bet_type.upper()}")
    print(f"{'='*60}")

    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # Filter to matches with odds
    df = df[df[odds_col].notna() & (df[odds_col] > 1)].copy()

    # Add season
    df['season'] = df['date'].apply(lambda x: x.year if x.month >= 8 else x.year - 1)
    if 'round_num' not in df.columns:
        df['round_num'] = df['round'].str.extract(r'(\d+)').astype(float).fillna(1).astype(int)

    # Prepare target
    if bet_type == 'home_win':
        df['target'] = df['home_win'].astype(int)
    elif bet_type == 'away_win':
        df['target'] = df['away_win'].astype(int)
    elif bet_type == 'btts':
        df['target'] = ((df['home_goals'] > 0) & (df['away_goals'] > 0)).astype(int)
    else:
        raise ValueError(f"Unknown bet type: {bet_type}")

    # Calculate implied probability from odds
    df['implied_prob'] = df[odds_col].apply(odds_to_probability)

    print(f"Matches with odds: {len(df)}")
    print(f"Odds column: {odds_col}")
    print(f"Minimum edge required: {min_edge:.0%}")

    # Get matchdays for walk-forward
    matchdays = []
    for season in sorted(df['season'].unique()):
        if season < start_season:
            continue
        season_df = df[df['season'] == season]
        for round_num in sorted(season_df['round_num'].unique()):
            matchdays.append((season, round_num))

    print(f"Evaluating {len(matchdays)} matchdays from season {start_season}")

    # Results
    results = []
    total_bets = 0
    total_wins = 0
    total_profit = 0.0
    total_staked = 0.0
    model = None

    # Load best params from ensemble training
    params_path = project_root / 'experiments/outputs/ensemble_models.json'
    best_params = {}
    if params_path.exists():
        with open(params_path) as f:
            config = json.load(f)
            if bet_type in config:
                best_params = config[bet_type].get('best_params', {})

    for i, (season, round_num) in enumerate(matchdays):
        # Train on data before this matchday
        train_mask = (
            (df['season'] < season) |
            ((df['season'] == season) & (df['round_num'] < round_num))
        )
        test_mask = (df['season'] == season) & (df['round_num'] == round_num)

        train_df = df[train_mask]
        test_df = df[test_mask]

        if len(test_df) == 0 or len(train_df) < 100:
            continue

        X_train = train_df[feature_cols].fillna(0)
        y_train = train_df['target']
        X_test = test_df[feature_cols].fillna(0)
        y_test = test_df['target']
        test_odds = test_df[odds_col].values
        test_implied = test_df['implied_prob'].values

        # Retrain every 5 rounds
        if model is None or i % 5 == 0:
            base_model = create_ensemble_classifier(best_params)
            model = CalibratedClassifierCV(base_model, method='isotonic', cv=3)
            model.fit(X_train, y_train)

        # Predict
        our_probs = model.predict_proba(X_test)[:, 1]

        # Find value bets
        for j in range(len(test_df)):
            our_prob = our_probs[j]
            implied_prob = test_implied[j]
            odds = test_odds[j]
            actual = y_test.iloc[j]

            if pd.isna(implied_prob) or pd.isna(odds):
                continue

            edge = our_prob - implied_prob

            # Value bet: our probability exceeds market by min_edge
            if edge >= min_edge:
                # Calculate Kelly stake
                stake = kelly_stake(our_prob, odds, fraction=0.25)
                if stake <= 0:
                    continue

                # Cap stake at 5% of bankroll
                stake = min(stake, 0.05)

                # Calculate profit
                if actual == 1:
                    profit = stake * (odds - 1)
                    total_wins += 1
                else:
                    profit = -stake

                total_bets += 1
                total_profit += profit
                total_staked += stake

                results.append({
                    'season': season,
                    'round': round_num,
                    'our_prob': our_prob,
                    'implied_prob': implied_prob,
                    'edge': edge,
                    'odds': odds,
                    'stake': stake,
                    'won': actual == 1,
                    'profit': profit,
                })

        # Progress
        if (i + 1) % 20 == 0:
            roi = total_profit / total_staked if total_staked > 0 else 0
            hit_rate = total_wins / total_bets if total_bets > 0 else 0
            print(f"  [{i+1}/{len(matchdays)}] Bets: {total_bets}, Wins: {total_wins} ({hit_rate:.0%}), ROI: {roi:+.1%}")

    # Summary
    roi = total_profit / total_staked if total_staked > 0 else 0
    hit_rate = total_wins / total_bets if total_bets > 0 else 0

    print(f"\n{'='*60}")
    print(f"VALUE BETTING RESULTS: {bet_type.upper()}")
    print(f"{'='*60}")
    print(f"Total bets: {total_bets}")
    print(f"Total wins: {total_wins} ({hit_rate:.1%})")
    print(f"Total staked: {total_staked:.2f} units")
    print(f"Total profit: {total_profit:+.2f} units")
    print(f"ROI: {roi:+.1%}")

    if results:
        results_df = pd.DataFrame(results)
        avg_edge = results_df['edge'].mean()
        avg_odds = results_df['odds'].mean()
        print(f"Average edge: {avg_edge:.1%}")
        print(f"Average odds: {avg_odds:.2f}")

        # By edge bucket
        print("\nPerformance by edge size:")
        for low, high in [(0.05, 0.10), (0.10, 0.15), (0.15, 0.20), (0.20, 1.0)]:
            bucket = results_df[(results_df['edge'] >= low) & (results_df['edge'] < high)]
            if len(bucket) > 0:
                b_roi = bucket['profit'].sum() / bucket['stake'].sum()
                b_hit = bucket['won'].mean()
                print(f"  Edge {low:.0%}-{high:.0%}: {len(bucket)} bets, {b_hit:.0%} hit rate, {b_roi:+.1%} ROI")

    return {
        'total_bets': total_bets,
        'total_wins': total_wins,
        'total_profit': total_profit,
        'total_staked': total_staked,
        'roi': roi,
        'hit_rate': hit_rate,
        'results': results,
    }


def compare_approaches(df: pd.DataFrame, feature_cols: List[str]):
    """Compare fixed threshold vs value betting approaches."""
    print("\n" + "=" * 70)
    print("COMPARISON: FIXED THRESHOLD vs VALUE BETTING")
    print("=" * 70)

    bet_configs = [
        ('home_win', 'avg_home_close'),
        ('away_win', 'avg_away_close'),
        ('btts', 'btts_yes_avg'),
    ]

    all_results = {}

    for bet_type, odds_col in bet_configs:
        print(f"\n{'#'*60}")
        print(f"# {bet_type.upper()}")
        print(f"{'#'*60}")

        # Test different minimum edges
        for min_edge in [0.05, 0.10, 0.15]:
            result = run_value_betting_backtest(
                df=df,
                feature_cols=feature_cols,
                bet_type=bet_type,
                odds_col=odds_col,
                min_edge=min_edge,
                start_season=2024,
            )
            all_results[f"{bet_type}_edge{int(min_edge*100)}"] = result

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY: VALUE BETTING RESULTS")
    print("=" * 70)
    print(f"{'Bet Type':<25} {'Min Edge':<10} {'Bets':<8} {'Hit Rate':<10} {'ROI':<10}")
    print("-" * 70)

    for key, result in all_results.items():
        parts = key.rsplit('_edge', 1)
        bet_type = parts[0]
        edge = int(parts[1]) if len(parts) > 1 else 0
        print(f"{bet_type:<25} {edge}%{'':<8} {result['total_bets']:<8} {result['hit_rate']:.1%}{'':<6} {result['roi']:+.1%}")

    return all_results


def main():
    print("=" * 70)
    print("VALUE BETTING ANALYSIS")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    df = load_data()
    feature_cols = get_clean_features(df)

    print(f"Total matches: {len(df)}")
    print(f"Matches with closing odds: {df['avg_home_close'].notna().sum()}")
    print(f"Features: {len(feature_cols)}")

    # Run comparison
    results = compare_approaches(df, feature_cols)

    # Save results
    output_path = project_root / 'experiments/outputs/value_betting_results.json'

    serializable = {}
    for key, result in results.items():
        serializable[key] = {
            'total_bets': result['total_bets'],
            'total_wins': result['total_wins'],
            'roi': result['roi'],
            'hit_rate': result['hit_rate'],
        }

    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
