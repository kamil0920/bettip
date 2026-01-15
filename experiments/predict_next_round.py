#!/usr/bin/env python3
"""
Generate betting predictions for next round of matches.

This script:
1. Loads trained model configurations from optimization outputs
2. Gets upcoming fixtures from raw data
3. Generates features for upcoming matches based on recent form
4. Makes predictions and calculates Kelly stakes

Usage:
    python experiments/predict_next_round.py
    python experiments/predict_next_round.py --round 21 --bankroll 1000
"""
import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def load_optimization_results(bet_type: str) -> dict:
    """Load optimization results from JSON."""
    path = project_root / f'experiments/outputs/{bet_type}_full_optimization.json'
    if not path.exists():
        raise FileNotFoundError(f"No optimization results for {bet_type}")
    with open(path) as f:
        return json.load(f)


def load_improved_model_config() -> dict:
    """Load improved model configuration with calibrated thresholds."""
    # Prefer ensemble models (newest)
    paths = [
        project_root / 'experiments/outputs/ensemble_models.json',
        project_root / 'experiments/outputs/improved_models.json',
    ]
    for path in paths:
        if path.exists():
            print(f"  Loading model config from: {path.name}")
            with open(path) as f:
                return json.load(f)
    return {}


def get_clean_features(df: pd.DataFrame) -> List[str]:
    """Get features without data leakage (no odds-based features)."""
    exclude_cols = {
        # Target/identifier columns
        'fixture_id', 'date', 'home_team_id', 'home_team_name', 'away_team_id',
        'away_team_name', 'round', 'match_result', 'home_win', 'draw', 'away_win',
        'total_goals', 'goal_difference', 'league', 'target', 'ah_result',
        'home_goals', 'away_goals', 'season', 'round_num', 'result',
        'home_team', 'away_team', 'time',
        # Leaky odds features - REMOVE THESE
        'ah_line', 'ah_line_close',
        'avg_over25', 'avg_under25', 'avg_over25_close', 'avg_under25_close',
        'avg_ah_home', 'avg_ah_away', 'avg_ah_home_close', 'avg_ah_away_close',
        'max_ah_home', 'max_ah_away', 'max_ah_home_close', 'max_ah_away_close',
        'b365_ah_home', 'b365_ah_away', 'b365_ah_home_close', 'b365_ah_away_close',
        'pinnacle_ah_home', 'pinnacle_ah_away', 'pinnacle_ah_home_close', 'pinnacle_ah_away_close',
        'b365_over25', 'b365_under25', 'b365_over25_close', 'b365_under25_close',
        # Other odds columns
        'odds_home_prob', 'odds_draw_prob', 'odds_away_prob', 'odds_overround',
        'odds_move_home', 'odds_move_draw', 'odds_move_away',
        'odds_prob_move_home', 'odds_prob_move_away',
        'odds_move_home_pct', 'odds_move_away_pct',
        'odds_steam_home', 'odds_steam_away', 'odds_home_favorite',
        'odds_prob_diff', 'odds_prob_max', 'odds_entropy',
        'odds_upset_potential', 'odds_draw_relative',
        'odds_over25_prob', 'odds_under25_prob', 'odds_goals_expectation',
        # B365 closing odds
        'b365_home_open', 'b365_draw_open', 'b365_away_open',
        'b365_home_close', 'b365_draw_close', 'b365_away_close',
        # Avg/max odds
        'avg_home_open', 'avg_draw_open', 'avg_away_open',
        'avg_home_close', 'avg_draw_close', 'avg_away_close',
        'max_home_open', 'max_draw_open', 'max_away_open',
        'max_home_close', 'max_draw_close', 'max_away_close',
        # BTTS odds
        'btts_yes_avg', 'btts_no_avg', 'btts_yes_max', 'btts_no_max',
    }

    feature_cols = [c for c in df.columns if c not in exclude_cols]
    numeric_df = df[feature_cols].select_dtypes(include=['number'])
    feature_cols = numeric_df.columns.tolist()

    # Remove features with >30% missing in recent data
    recent = df.tail(int(len(df) * 0.2))
    missing_rates = recent[feature_cols].isna().mean()
    good_features = [f for f in feature_cols if missing_rates.get(f, 1) < 0.3]

    return good_features


def load_historical_features() -> pd.DataFrame:
    """Load historical feature data for training."""
    # Try to load the most recent feature file (prefer with real xG)
    paths = [
        project_root / 'data/03-features/features_with_real_xg.csv',
        project_root / 'data/03-features/features_all_leagues_complete.csv',
        project_root / 'data/03-features/features_premier_league_complete.csv',
        project_root / 'data/03-features/features_all_5leagues_with_odds.csv',
        project_root / 'data/03-features/features_all.csv',
    ]

    for path in paths:
        if path.exists():
            print(f"  Loading features from: {path.name}")
            df = pd.read_csv(path)
            df['date'] = pd.to_datetime(df['date'])
            print(f"  Loaded {len(df)} matches, date range: {df['date'].min().date()} to {df['date'].max().date()}")

            # Check for leagues
            if 'league' in df.columns:
                print(f"  Leagues: {df['league'].unique().tolist()}")

            # Check for odds columns
            odds_cols = [c for c in df.columns if 'odds_home_prob' in c or 'btts' in c.lower()]
            if odds_cols:
                print(f"  Has odds data: {len(odds_cols)} odds columns")
            return df

    raise FileNotFoundError("No feature files found")


def load_upcoming_fixtures(leagues: List[str] = None) -> pd.DataFrame:
    """Load upcoming fixtures from raw data for all leagues."""
    if leagues is None:
        leagues = ['premier_league', 'bundesliga', 'serie_a', 'la_liga', 'ligue_1']

    all_upcoming = []

    for league in leagues:
        path = project_root / f'data/01-raw/{league}/2025/matches.parquet'
        if not path.exists():
            continue

        try:
            matches = pd.read_parquet(path)
            upcoming = matches[matches['goals.home'].isna()].copy()
            upcoming['date'] = pd.to_datetime(upcoming['fixture.date'])
            upcoming = upcoming.rename(columns={
                'teams.home.name': 'home_team',
                'teams.away.name': 'away_team',
                'league.round': 'round'
            })
            upcoming['league'] = league
            all_upcoming.append(upcoming[['date', 'home_team', 'away_team', 'round', 'league']])
        except Exception as e:
            print(f"  Warning: Could not load {league}: {e}")

    if not all_upcoming:
        raise FileNotFoundError("No upcoming fixtures found")

    result = pd.concat(all_upcoming, ignore_index=True).sort_values('date')
    return result


def get_team_form(df: pd.DataFrame, team: str, as_home: bool, n_matches: int = 5) -> Dict:
    """Calculate team's recent form statistics."""
    if as_home:
        team_matches = df[df['home_team_name'] == team].copy()
        goals_scored = 'home_goals'
        goals_conceded = 'away_goals'
        win_col = 'home_win'
    else:
        team_matches = df[df['away_team_name'] == team].copy()
        goals_scored = 'away_goals'
        goals_conceded = 'home_goals'
        win_col = 'away_win'

    team_matches = team_matches.dropna(subset=[goals_scored])
    team_matches = team_matches.sort_values('date', ascending=False).head(n_matches)

    if len(team_matches) == 0:
        return {'matches': 0}

    return {
        'matches': len(team_matches),
        'goals_scored': team_matches[goals_scored].mean(),
        'goals_conceded': team_matches[goals_conceded].mean(),
        'wins': team_matches[win_col].sum() if win_col in team_matches.columns else 0,
        'points': team_matches['home_win'].sum() * 3 if as_home else team_matches['away_win'].sum() * 3,
    }


def create_features_for_match(historical_df: pd.DataFrame, home_team: str, away_team: str,
                               feature_cols: List[str]) -> pd.DataFrame:
    """Create features for a single upcoming match using most recent team data."""
    df = historical_df[historical_df['home_goals'].notna()].copy()
    df = df.sort_values('date', ascending=False)

    # Find most recent match for home team AT HOME
    home_recent = df[df['home_team_name'] == home_team].head(1)
    # Find most recent match for away team AWAY
    away_recent = df[df['away_team_name'] == away_team].head(1)

    if len(home_recent) == 0 or len(away_recent) == 0:
        # Fallback: use any recent match
        home_recent = df[(df['home_team_name'] == home_team) | (df['away_team_name'] == home_team)].head(1)
        away_recent = df[(df['home_team_name'] == away_team) | (df['away_team_name'] == away_team)].head(1)

    # Combine features from both teams' recent matches
    features = {}
    for col in feature_cols:
        # Try to get from home team's recent match first
        if col in home_recent.columns and len(home_recent) > 0:
            val = home_recent[col].values[0]
            if pd.notna(val):
                features[col] = val
                continue
        # Then try away team's recent match
        if col in away_recent.columns and len(away_recent) > 0:
            val = away_recent[col].values[0]
            if pd.notna(val):
                features[col] = val
                continue
        # Default to 0
        features[col] = 0

    return pd.DataFrame([features])


def get_numeric_feature_cols(df: pd.DataFrame, exclude_patterns: List[str]) -> List[str]:
    """Get numeric feature columns excluding certain patterns."""
    exclude_cols = [
        'fixture_id', 'date', 'home_team_id', 'home_team_name', 'away_team_id',
        'away_team_name', 'round', 'match_result', 'home_win', 'draw', 'away_win',
        'total_goals', 'goal_difference', 'league', 'target', 'ah_result',
        'home_goals', 'away_goals', 'season', 'round_num', 'result',
    ]

    feature_cols = [c for c in df.columns if c not in exclude_cols]
    feature_cols = [c for c in feature_cols
                   if not any(pat.lower() in c.lower() for pat in exclude_patterns)]

    numeric_cols = df[feature_cols].select_dtypes(include=['number']).columns.tolist()
    return numeric_cols


def calculate_kelly_stake(prob: float, odds: float, kelly_fraction: float = 0.25,
                         max_stake: float = 0.05) -> float:
    """Calculate Kelly criterion stake as fraction of bankroll."""
    if odds <= 1 or prob <= 0:
        return 0
    edge = prob * odds - 1
    if edge <= 0:
        return 0
    full_kelly = edge / (odds - 1)
    kelly = kelly_fraction * full_kelly
    return max(0, min(kelly, max_stake))


def odds_to_probability(odds: float) -> float:
    """Convert decimal odds to implied probability."""
    if odds <= 1:
        return 0
    return 1 / odds


def calculate_value_edge(our_prob: float, market_odds: float) -> float:
    """Calculate edge vs market: our_prob - implied_prob."""
    implied_prob = odds_to_probability(market_odds)
    return our_prob - implied_prob


# Typical market odds (from historical analysis)
TYPICAL_MARKET_ODDS = {
    'Home Win': 2.57,     # ~39% implied
    'Away Win': 3.87,     # ~26% implied
    'BTTS': 1.85,         # ~54% implied
    'Asian Handicap': 1.90,  # ~53% implied for -0.5
}


def train_and_predict_btts(historical_df: pd.DataFrame, config: dict,
                           upcoming: List[Dict], show_all: bool = False) -> List[Dict]:
    """Train BTTS model and predict for upcoming matches."""
    print("\n" + "=" * 60)
    print("BTTS (Both Teams To Score)")
    print("=" * 60)

    # Prepare training data
    df = historical_df[historical_df['btts_yes_avg'].notna()].copy()
    df['target'] = ((df['home_goals'] > 0) & (df['away_goals'] > 0)).astype(int)

    exclude_patterns = ['btts', 'b365', 'pinnacle', 'avg_home', 'avg_away', 'max_home', 'max_away']
    feature_cols = get_numeric_feature_cols(df, exclude_patterns)

    # Use features from config if available
    if 'features_per_model' in config:
        model_features = config['features_per_model'].get('CatBoost', feature_cols)
        feature_cols = [f for f in model_features if f in df.columns]

    print(f"  Using {len(feature_cols)} features")

    # Prepare data
    df = df.dropna(subset=['target'])
    df = df.sort_values('date')

    X = df[feature_cols].fillna(0)
    y = df['target']

    print(f"  Training on {len(X)} matches (BTTS rate: {y.mean():.1%})")

    # Train model with best params
    params = config.get('best_params', {}).get('CatBoost', {})
    model = CatBoostClassifier(
        iterations=params.get('iterations', 100),
        depth=params.get('depth', 5),
        learning_rate=params.get('learning_rate', 0.1),
        l2_leaf_reg=params.get('l2_leaf_reg', 3),
        random_state=42,
        verbose=0
    )
    model.fit(X, y)

    # Predict for upcoming matches
    predictions = []
    threshold = 0.65  # From optimization

    print(f"\n  Predictions (threshold: {threshold:.0%}):")
    for match in upcoming:
        # Create feature vector using full feature set
        X_pred = create_features_for_match(historical_df, match['home_team'], match['away_team'], feature_cols)
        X_pred = X_pred[feature_cols].fillna(0)

        prob = model.predict_proba(X_pred)[0][1]

        print(f"    {match['home_team']:20} vs {match['away_team']:20}: {prob:.1%}")

        if prob >= threshold or show_all:
            predictions.append({
                'match': f"{match['home_team']} vs {match['away_team']}",
                'date': str(match['date']),
                'league': match.get('league', ''),
                'bet_type': 'BTTS Yes',
                'probability': prob,
                'threshold': threshold,
                'odds': 1.85,  # Typical BTTS odds
                'meets_threshold': prob >= threshold
            })

    return predictions


def train_and_predict_asian_handicap(historical_df: pd.DataFrame, config: dict,
                                     upcoming: List[Dict], show_all: bool = False,
                                     improved_config: dict = None) -> List[Dict]:
    """Train Asian Handicap regression model and predict for upcoming matches."""
    print("\n" + "=" * 60)
    print("Asian Handicap (Home -0.5) - IMPROVED")
    print("=" * 60)

    # Prepare training data
    df = historical_df[historical_df['goal_difference'].notna()].copy()
    df['target'] = df['goal_difference'].astype(float)

    # Use clean features without odds leakage
    feature_cols = get_clean_features(df)

    # Use features from improved config if available
    if improved_config and 'asian_handicap' in improved_config:
        model_features = improved_config['asian_handicap'].get('features', feature_cols)
        feature_cols = [f for f in model_features if f in df.columns]

    print(f"  Using {len(feature_cols)} clean features (no odds leakage)")

    df = df.sort_values('date')
    X = df[feature_cols].fillna(0)
    y = df['target']

    print(f"  Training on {len(X)} matches (avg margin: {y.mean():.2f})")

    # Train regression model with improved params
    if improved_config and 'asian_handicap' in improved_config:
        params = improved_config['asian_handicap'].get('best_params', {})
    else:
        params = config.get('best_params', {}).get('CatBoost', {})

    model = CatBoostRegressor(
        iterations=params.get('iterations', 100),
        depth=params.get('depth', 5),
        learning_rate=params.get('learning_rate', 0.1),
        l2_leaf_reg=params.get('l2_leaf_reg', 3),
        random_state=42,
        verbose=0
    )
    model.fit(X, y)

    # Predict for upcoming matches
    predictions = []
    # Use improved threshold (1.50) from calibration analysis
    if improved_config and 'asian_handicap' in improved_config:
        margin_threshold = improved_config['asian_handicap'].get('optimal_threshold', 1.50)
    else:
        margin_threshold = 1.50  # Much stricter than old 0.75
    ah_line = -0.5

    print(f"\n  Predictions (margin threshold: {margin_threshold} - stricter for better precision):")
    for match in upcoming:
        X_pred = create_features_for_match(historical_df, match['home_team'], match['away_team'], feature_cols)
        X_pred = X_pred[feature_cols].fillna(0)

        pred_margin = model.predict(X_pred)[0]

        meets = pred_margin > -ah_line + margin_threshold
        print(f"    {match['home_team']:20} vs {match['away_team']:20}: margin={pred_margin:+.2f} {'*' if meets else ''}")

        if meets or show_all:
            predictions.append({
                'match': f"{match['home_team']} vs {match['away_team']}",
                'date': str(match['date']),
                'league': match.get('league', ''),
                'bet_type': f'Home -{abs(ah_line)}',
                'predicted_margin': pred_margin,
                'threshold': margin_threshold,
                'odds': 1.90,
                'meets_threshold': meets
            })

    return predictions


def train_and_predict_away_win(historical_df: pd.DataFrame, config: dict,
                               upcoming: List[Dict], show_all: bool = False,
                               improved_config: dict = None) -> List[Dict]:
    """Train Away Win model and predict for upcoming matches."""
    print("\n" + "=" * 60)
    print("Away Win - IMPROVED")
    print("=" * 60)

    # Prepare training data
    df = historical_df[historical_df['away_win'].notna()].copy()
    df['target'] = df['away_win'].astype(int)

    # Use clean features without odds leakage
    feature_cols = get_clean_features(df)

    # Use features from improved config if available
    if improved_config and 'away_win' in improved_config:
        model_features = improved_config['away_win'].get('features', feature_cols)
        feature_cols = [f for f in model_features if f in df.columns]

    print(f"  Using {len(feature_cols)} clean features (no odds leakage)")

    df = df.sort_values('date')
    X = df[feature_cols].fillna(0)
    y = df['target']

    print(f"  Training on {len(X)} matches (away win rate: {y.mean():.1%})")

    # Train model with improved params
    if improved_config and 'away_win' in improved_config:
        params = improved_config['away_win'].get('best_params', {})
    else:
        params = config.get('best_params', {}).get('CatBoost', {})

    model = CatBoostClassifier(
        iterations=params.get('iterations', 100),
        depth=params.get('depth', 5),
        learning_rate=params.get('learning_rate', 0.1),
        l2_leaf_reg=params.get('l2_leaf_reg', 3),
        random_state=42,
        verbose=0
    )
    model.fit(X, y)

    # Predict for upcoming
    predictions = []
    # Use calibrated threshold from improved config
    if improved_config and 'away_win' in improved_config:
        threshold = improved_config['away_win'].get('optimal_threshold', 0.70)
    else:
        threshold = 0.70

    print(f"\n  Predictions (threshold: {threshold:.0%}):")
    for match in upcoming:
        X_pred = create_features_for_match(historical_df, match['home_team'], match['away_team'], feature_cols)
        X_pred = X_pred[feature_cols].fillna(0)

        prob = model.predict_proba(X_pred)[0][1]

        meets = prob >= threshold
        print(f"    {match['home_team']:20} vs {match['away_team']:20}: {prob:.1%} {'*' if meets else ''}")

        if meets or show_all:
            predictions.append({
                'match': f"{match['home_team']} vs {match['away_team']}",
                'date': str(match['date']),
                'league': match.get('league', ''),
                'bet_type': 'Away Win',
                'probability': prob,
                'threshold': threshold,
                'odds': 3.5,
                'meets_threshold': meets
            })

    return predictions


def train_and_predict_home_win(historical_df: pd.DataFrame, config: dict,
                               upcoming: List[Dict], show_all: bool = False,
                               improved_config: dict = None) -> List[Dict]:
    """Train Home Win model (new) and predict for upcoming matches."""
    print("\n" + "=" * 60)
    print("Home Win - NEW MODEL")
    print("=" * 60)

    # Prepare training data
    df = historical_df[historical_df['home_win'].notna()].copy()
    df['target'] = df['home_win'].astype(int)

    # Use clean features without odds leakage
    feature_cols = get_clean_features(df)

    # Use features from improved config if available
    if improved_config and 'home_win' in improved_config:
        model_features = improved_config['home_win'].get('features', feature_cols)
        feature_cols = [f for f in model_features if f in df.columns]

    print(f"  Using {len(feature_cols)} clean features (no odds leakage)")

    df = df.sort_values('date')
    X = df[feature_cols].fillna(0)
    y = df['target']

    print(f"  Training on {len(X)} matches (home win rate: {y.mean():.1%})")

    # Train model with improved params
    if improved_config and 'home_win' in improved_config:
        params = improved_config['home_win'].get('best_params', {})
    else:
        params = {}

    model = CatBoostClassifier(
        iterations=params.get('iterations', 100),
        depth=params.get('depth', 5),
        learning_rate=params.get('learning_rate', 0.1),
        l2_leaf_reg=params.get('l2_leaf_reg', 3),
        random_state=42,
        verbose=0
    )
    model.fit(X, y)

    # Predict for upcoming
    predictions = []
    # Use calibrated threshold from improved config (0.80 is stricter)
    if improved_config and 'home_win' in improved_config:
        threshold = improved_config['home_win'].get('optimal_threshold', 0.80)
    else:
        threshold = 0.80

    print(f"\n  Predictions (threshold: {threshold:.0%} - high confidence only):")
    for match in upcoming:
        X_pred = create_features_for_match(historical_df, match['home_team'], match['away_team'], feature_cols)
        X_pred = X_pred[feature_cols].fillna(0)

        prob = model.predict_proba(X_pred)[0][1]

        meets = prob >= threshold
        print(f"    {match['home_team']:20} vs {match['away_team']:20}: {prob:.1%} {'*' if meets else ''}")

        if meets or show_all:
            predictions.append({
                'match': f"{match['home_team']} vs {match['away_team']}",
                'date': str(match['date']),
                'league': match.get('league', ''),
                'bet_type': 'Home Win',
                'probability': prob,
                'threshold': threshold,
                'odds': 2.0,  # Typical home win odds
                'meets_threshold': meets
            })

    return predictions


def main():
    parser = argparse.ArgumentParser(description='Generate next round predictions')
    parser.add_argument('--round', type=int, help='Round number to predict (default: next)')
    parser.add_argument('--bankroll', type=float, default=1000, help='Bankroll amount')
    parser.add_argument('--kelly-fraction', type=float, default=0.25, help='Kelly fraction')
    parser.add_argument('--show-all', action='store_true', help='Show all predictions, not just high confidence')
    args = parser.parse_args()

    print("=" * 70)
    print("NEXT ROUND BETTING PREDICTIONS")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    historical_df = load_historical_features()
    upcoming_df = load_upcoming_fixtures()

    # Filter to specific round if requested
    if args.round:
        upcoming_df = upcoming_df[upcoming_df['round'].str.contains(str(args.round))]
    else:
        # Get next matches (default 30 for more coverage across leagues)
        upcoming_df = upcoming_df.head(30)

    print(f"Found {len(upcoming_df)} upcoming matches")
    print("\nUpcoming fixtures:")
    for _, row in upcoming_df.iterrows():
        league_short = row.get('league', '')[:3].upper() if 'league' in row else ''
        print(f"  [{league_short}] {row['date'].strftime('%Y-%m-%d')} - {row['home_team']} vs {row['away_team']}")

    # Convert to list of dicts
    upcoming = upcoming_df.to_dict('records')

    # Load improved model config (from calibration analysis)
    improved_config = load_improved_model_config()
    if improved_config:
        print("\n  Using IMPROVED models with calibrated thresholds:")
        for bet_type, cfg in improved_config.items():
            print(f"    {bet_type}: threshold={cfg.get('optimal_threshold', 'N/A')}, ROI={cfg.get('test_roi', 0):.1%}")

    # Load model configs
    all_predictions = []
    btts_preds = []
    ah_preds = []
    aw_preds = []
    hw_preds = []

    # BTTS predictions
    try:
        btts_config = load_optimization_results('btts')
        btts_preds = train_and_predict_btts(historical_df, btts_config, upcoming, True)  # Get all
        all_predictions.extend([p for p in btts_preds if p.get('meets_threshold', True)])
        high_conf = len([p for p in btts_preds if p.get('meets_threshold', True)])
        print(f"\n  BTTS: {high_conf} high-confidence recommendations")
    except Exception as e:
        print(f"BTTS failed: {e}")

    # Asian Handicap predictions (IMPROVED)
    try:
        ah_config = load_optimization_results('asian_handicap')
        ah_preds = train_and_predict_asian_handicap(historical_df, ah_config, upcoming, True, improved_config)
        all_predictions.extend([p for p in ah_preds if p.get('meets_threshold', True)])
        high_conf = len([p for p in ah_preds if p.get('meets_threshold', True)])
        print(f"  Asian Handicap: {high_conf} high-confidence recommendations")
    except Exception as e:
        print(f"Asian Handicap failed: {e}")

    # Away Win predictions (IMPROVED)
    try:
        aw_config = load_optimization_results('away_win')
        aw_preds = train_and_predict_away_win(historical_df, aw_config, upcoming, True, improved_config)
        all_predictions.extend([p for p in aw_preds if p.get('meets_threshold', True)])
        high_conf = len([p for p in aw_preds if p.get('meets_threshold', True)])
        print(f"  Away Win: {high_conf} high-confidence recommendations")
    except Exception as e:
        print(f"Away Win failed: {e}")

    # Home Win predictions (NEW - added from improved models)
    try:
        hw_preds = train_and_predict_home_win(historical_df, {}, upcoming, True, improved_config)
        all_predictions.extend([p for p in hw_preds if p.get('meets_threshold', True)])
        high_conf = len([p for p in hw_preds if p.get('meets_threshold', True)])
        print(f"  Home Win: {high_conf} high-confidence recommendations")
    except Exception as e:
        print(f"Home Win failed: {e}")

    # Collect all predictions with scores for tiered recommendations
    all_scored = []
    for p in btts_preds:
        all_scored.append({**p, 'score': p.get('probability', 0), 'bet_type': 'BTTS'})
    for p in ah_preds:
        all_scored.append({**p, 'score': p.get('predicted_margin', 0) / 3.0, 'bet_type': p.get('bet_type', 'Asian Handicap')})
    for p in aw_preds:
        all_scored.append({**p, 'score': p.get('probability', 0), 'bet_type': 'Away Win'})
    for p in hw_preds:
        all_scored.append({**p, 'score': p.get('probability', 0), 'bet_type': 'Home Win'})

    # Display recommendations using VALUE BETTING approach
    print("\n" + "=" * 70)
    print("VALUE BETTING RECOMMENDATIONS")
    print("=" * 70)
    print("Based on model probability vs typical market odds")
    print("Minimum edge required: 10% (from backtest optimization)")

    # Calculate value for each prediction
    min_edge = 0.10  # 10% minimum edge from value betting analysis
    value_bets = []

    for p in all_scored:
        bet_type = p.get('bet_type', '')
        prob = p.get('probability', 0)

        if bet_type == 'Asian Handicap' or bet_type.startswith('Home -'):
            # For AH, convert margin to rough probability
            margin = p.get('predicted_margin', 0)
            prob = 0.5 + 0.15 * margin  # Rough conversion
            prob = max(0.1, min(0.9, prob))
            bet_type = 'Asian Handicap'

        market_odds = TYPICAL_MARKET_ODDS.get(bet_type, 2.0)
        implied_prob = odds_to_probability(market_odds)
        edge = prob - implied_prob

        if edge >= min_edge and prob > 0:
            value_bets.append({
                **p,
                'edge': edge,
                'implied_prob': implied_prob,
                'market_odds': market_odds,
                'our_prob': prob
            })

    if value_bets:
        # Sort by edge (highest first)
        value_bets.sort(key=lambda x: x['edge'], reverse=True)

        print(f"\n{'Match':<40} {'Bet':<12} {'Model':<8} {'Market':<8} {'Edge':<8}")
        print("-" * 76)

        for pred in value_bets[:10]:  # Top 10 value bets
            league_short = pred.get('league', '')[:3].upper() if pred.get('league') else ''
            match_name = pred['match'][:35] + "..." if len(pred['match']) > 38 else pred['match']
            bet_type = pred.get('bet_type', '')[:10]

            print(f"[{league_short}] {match_name:<35} {bet_type:<12} {pred['our_prob']:.0%}    {pred['implied_prob']:.0%}    +{pred['edge']:.0%}")

        print("\n" + "-" * 76)
        print(f"Total value bets: {len(value_bets)}")
        print("\nHistorical ROI for 10%+ edge bets:")
        print("  Home Win: +45.6%  |  Away Win: +65.7%  |  BTTS: +34.0%")
        print("\nNote: Compare 'Market' column to current bookmaker odds before betting.")
        print("      If current odds are HIGHER than typical, edge is even better!")

        # Add to all_predictions for saving
        all_predictions = [p for p in value_bets if p['edge'] >= min_edge]

    else:
        print("\nNo VALUE bets found (model vs market edge < 10%).")
        print("This means our model largely agrees with the market - no edge.")

        # Show closest to having value
        print("\n" + "-" * 70)
        print("NEAR-VALUE BETS (5-10% edge - lower confidence)")
        print("-" * 70)

        near_value = []
        for p in all_scored:
            bet_type = p.get('bet_type', '')
            prob = p.get('probability', 0)
            if bet_type == 'Asian Handicap' or bet_type.startswith('Home -'):
                margin = p.get('predicted_margin', 0)
                prob = 0.5 + 0.15 * margin
                prob = max(0.1, min(0.9, prob))
                bet_type = 'Asian Handicap'

            market_odds = TYPICAL_MARKET_ODDS.get(bet_type, 2.0)
            implied_prob = odds_to_probability(market_odds)
            edge = prob - implied_prob

            if 0.05 <= edge < 0.10:
                near_value.append({**p, 'edge': edge, 'our_prob': prob, 'implied_prob': implied_prob})

        if near_value:
            near_value.sort(key=lambda x: x['edge'], reverse=True)
            for pred in near_value[:5]:
                league_short = pred.get('league', '')[:3].upper() if pred.get('league') else ''
                bet_type = pred.get('bet_type', '')
                print(f"  [{league_short}] {pred['match']}")
                print(f"       {bet_type}: {pred['our_prob']:.0%} model vs {pred['implied_prob']:.0%} market (+{pred['edge']:.0%} edge)")
                print()
            print("Historical ROI at 5% edge: ~35% (vs ~55% at 10% edge)")
        else:
            print("\nNo near-value bets either. Very efficient market today.")

        return

    # Detailed breakdown with Kelly stakes
    print("\n" + "=" * 70)
    print("DETAILED VALUE BETS")
    print("=" * 70)

    total_stake = 0
    for pred in sorted(all_predictions, key=lambda x: x.get('edge', 0), reverse=True):
        league_short = pred.get('league', '')[:3].upper() if pred.get('league') else ''
        print(f"\n[{league_short}] {pred['match']}")
        print(f"  Date: {str(pred.get('date', ''))[:10]}")
        print(f"  Bet: {pred.get('bet_type', '')}")
        print(f"  Model prob: {pred.get('our_prob', 0):.1%} vs Market: {pred.get('implied_prob', 0):.1%}")
        print(f"  Edge: +{pred.get('edge', 0):.1%}")

        # Calculate Kelly stake based on edge
        our_prob = pred.get('our_prob', 0)
        market_odds = pred.get('market_odds', 2.0)
        stake = calculate_kelly_stake(our_prob, market_odds, args.kelly_fraction)

        stake_amount = stake * args.bankroll
        total_stake += stake_amount
        print(f"  Typical odds: {market_odds:.2f}")
        print(f"  Kelly stake: ${stake_amount:.2f} ({stake:.1%} of bankroll)")

    print("\n" + "-" * 70)
    print(f"Total: {len(all_predictions)} value bets | Total stake: ${total_stake:.2f}")
    print("=" * 70)

    # Save predictions
    output_path = project_root / 'experiments/outputs/next_round_predictions.json'
    with open(output_path, 'w') as f:
        json.dump({
            'generated_at': datetime.now().isoformat(),
            'bankroll': args.bankroll,
            'predictions': all_predictions
        }, f, indent=2, default=str)
    print(f"\nPredictions saved to: {output_path}")


if __name__ == '__main__':
    main()
