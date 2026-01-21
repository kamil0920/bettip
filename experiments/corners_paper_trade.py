#!/usr/bin/env python
"""
Corner Betting Paper Trading V3 - Stacking with XGBoost Meta-Learner

This script validates corner betting edge using the V3 model:
1. Uses ALL available features from main features file (~253 features)
2. Boruta-selected features for prediction
3. Stacking ensemble with XGBoost meta-learner (+35.7% ROI vs 25.5% simple voting)

Model comparison showed stacking_xgb outperforms:
- Simple voting: 25.5% ROI
- Stacking with LR: 32.3% ROI
- Stacking with XGB: 35.7% ROI (+10.2% improvement)

Usage:
    python experiments/corners_paper_trade.py predict    # Generate predictions
    python experiments/corners_paper_trade.py close      # Record closing odds
    python experiments/corners_paper_trade.py result     # Record results
    python experiments/corners_paper_trade.py settle     # Auto-settle from data
    python experiments/corners_paper_trade.py status     # View dashboard
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from scipy import stats

from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import warnings
warnings.filterwarnings('ignore')

# Default corner odds from Superbet
DEFAULT_CORNER_ODDS = {
    'over_8_5': 1.65, 'under_8_5': 2.20,
    'over_9_5': 1.90, 'under_9_5': 1.90,
    'over_10_5': 2.10, 'under_10_5': 1.72,
    'over_11_5': 2.50, 'under_11_5': 1.55,
}

# V3 Boruta-selected features (confirmed + tentative)
BORUTA_FEATURES = [
    'away_points_ema',
    'odds_under25_prob',
    'odds_goals_expectation',
    'ref_corner_avg',
    'odds_over25_prob',
    'home_team_id',
    'away_first_half_rate',
    'odds_overround',
    'ref_corner_std',
    # Top ranked additional features
    'avg_under25_close',
    'away_season_gd',
    'avg_under25',
    'away_team_id',
    'avg_over25_close',
    'avg_over25',
    'b365_under25_close',
    'odds_draw_relative',
    'home_pts_to_leader',
    'odds_draw_prob',
    'ref_draw_pct',
    'home_elo',
    'away_elo',
    'elo_diff',
    'home_season_gd',
]

# V3 best strategies from backtest
V3_STRATEGIES = {
    'over_10_5': {'direction': 'under', 'threshold': 0.70, 'roi': 34.5},
    'over_9_5': {'direction': 'over', 'threshold': 0.65, 'roi': 32.2},
    'over_11_5': {'direction': 'under', 'threshold': 0.75, 'roi': 24.5},
}


class CornersTrackerV3:
    """Track corner betting predictions with CLV analysis - V3 with full features."""

    def __init__(self, output_path: str = "experiments/outputs/corners_tracking_v3.json"):
        self.output_path = Path(output_path)
        self.predictions = self._load_data()
        self.models = {}
        self.feature_cols = []

    def _load_data(self) -> Dict:
        """Load existing tracking data."""
        if self.output_path.exists():
            with open(self.output_path, 'r') as f:
                return json.load(f)
        return {"bets": [], "summary": {}, "version": "v3"}

    def _save_data(self):
        """Save tracking data."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, 'w') as f:
            json.dump(self.predictions, f, indent=2, default=str)

    def add_prediction(
        self,
        fixture_id: int,
        match_date: str,
        home_team: str,
        away_team: str,
        league: str,
        referee: str,
        predicted_corners: float,
        bet_type: str,
        line: float,
        our_odds: float,
        our_probability: float,
        edge: float,
        ref_avg_corners: float = None,
        goals_expectation: float = None,
    ):
        """Add a new corner prediction."""
        key = f"{fixture_id}_{bet_type}_{line}"

        bet = {
            "key": key,
            "fixture_id": fixture_id,
            "match_date": match_date,
            "home_team": home_team,
            "away_team": away_team,
            "league": league,
            "referee": referee,
            "ref_avg_corners": ref_avg_corners,
            "goals_expectation": goals_expectation,
            "predicted_corners": predicted_corners,
            "bet_type": bet_type,
            "line": line,
            "our_odds": our_odds,
            "our_probability": our_probability,
            "edge": edge,
            "closing_odds": None,
            "clv": None,
            "actual_corners": None,
            "won": None,
            "created_at": datetime.now().isoformat(),
            "status": "pending"
        }

        # Check if already exists
        existing = [b for b in self.predictions["bets"] if b["key"] == key]
        if existing:
            print(f"  [EXISTS] {home_team} vs {away_team} ({bet_type} {line})")
            return

        self.predictions["bets"].append(bet)
        self._save_data()

        ref_info = f" [Ref: {referee[:15] if referee else 'None'}={ref_avg_corners:.1f}]" if ref_avg_corners else ""
        goals_info = f" [xG: {goals_expectation:.2f}]" if goals_expectation else ""
        print(f"  [NEW] {home_team} vs {away_team} - {bet_type} {line} @ {our_odds:.2f} (+{edge:.1f}%){ref_info}{goals_info}")

    def record_closing_odds(self, key: str, closing_odds: float):
        """Record closing odds for a bet."""
        for bet in self.predictions["bets"]:
            if bet["key"] == key:
                bet["closing_odds"] = closing_odds
                if bet["our_odds"] and closing_odds:
                    bet["clv"] = ((bet["our_odds"] / closing_odds) - 1) * 100
                bet["status"] = "closed"
                self._save_data()
                print(f"Recorded closing odds: {closing_odds:.2f} (CLV: {bet['clv']:+.1f}%)")
                return
        print(f"Bet not found: {key}")

    def record_result(self, fixture_id: int, actual_corners: int):
        """Record actual corners for a match."""
        updated = 0
        for bet in self.predictions["bets"]:
            if bet["fixture_id"] == fixture_id:
                bet["actual_corners"] = actual_corners
                if bet["bet_type"] == "OVER":
                    bet["won"] = actual_corners > bet["line"]
                else:
                    bet["won"] = actual_corners < bet["line"]
                bet["status"] = "settled"
                updated += 1

        if updated > 0:
            self._save_data()
            print(f"Recorded {actual_corners} corners for fixture {fixture_id} ({updated} bets)")
        else:
            print(f"No bets found for fixture {fixture_id}")

    def get_status(self) -> Dict:
        """Get current tracking status."""
        bets = self.predictions["bets"]
        if not bets:
            return {"total_bets": 0}

        pending = [b for b in bets if b["status"] == "pending"]
        closed = [b for b in bets if b["status"] == "closed"]
        settled = [b for b in bets if b["status"] == "settled"]

        summary = {
            "total_bets": len(bets),
            "pending": len(pending),
            "closed": len(closed),
            "settled": len(settled),
        }

        clv_bets = [b for b in bets if b.get("clv") is not None]
        if clv_bets:
            clvs = [b["clv"] for b in clv_bets]
            summary["avg_clv"] = np.mean(clvs)
            summary["clv_positive_rate"] = sum(1 for c in clvs if c > 0) / len(clvs)

        if settled:
            wins = sum(1 for b in settled if b["won"])
            summary["wins"] = wins
            summary["losses"] = len(settled) - wins
            summary["win_rate"] = wins / len(settled)
            profit = sum((b["our_odds"] - 1) if b["won"] else -1 for b in settled)
            summary["roi"] = (profit / len(settled)) * 100
            summary["avg_edge"] = np.mean([b["edge"] for b in settled])

        return summary

    def print_dashboard(self):
        """Print dashboard."""
        status = self.get_status()

        print("\n" + "=" * 70)
        print("CORNER BETTING PAPER TRADE V3 - DASHBOARD (FULL FEATURES)")
        print("=" * 70)

        print(f"\nTotal bets tracked: {status.get('total_bets', 0)}")
        print(f"  Pending: {status.get('pending', 0)}")
        print(f"  Closed: {status.get('closed', 0)}")
        print(f"  Settled: {status.get('settled', 0)}")

        if status.get('avg_clv') is not None:
            print(f"\nCLV Analysis:")
            print(f"  Average CLV: {status['avg_clv']:+.2f}%")
            print(f"  CLV positive rate: {status['clv_positive_rate']:.1%}")

        if status.get('settled', 0) > 0:
            print(f"\nResults:")
            print(f"  Wins: {status['wins']}, Losses: {status['losses']}")
            print(f"  Win rate: {status['win_rate']:.1%}")
            print(f"  ROI: {status['roi']:+.1f}%")
            print(f"  Average edge: {status['avg_edge']:.1f}%")

        print("\n" + "-" * 70)
        print("Recent Bets:")
        print("-" * 70)

        bets = self.predictions["bets"][-15:]
        for bet in bets:
            match = f"{bet['home_team'][:12]}v{bet['away_team'][:12]}"
            date = bet['match_date'][:10] if bet['match_date'] else 'N/A'
            bet_desc = f"{bet['bet_type']} {bet['line']}"
            ref = bet.get('referee', '')[:10] if bet.get('referee') else 'None'

            status_str = bet['status'].upper()
            if bet['status'] == 'settled':
                result = "WON" if bet['won'] else "LOST"
                status_str = f"{result} ({bet['actual_corners']})"

            print(f"  {date} | {match:<26} | {bet_desc:<12} | {ref:<10} | {status_str}")

        print("=" * 70)


def load_corner_data():
    """Load corner stats from match_stats parquet files."""
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

            matches_slim = matches[[
                'fixture.id', 'fixture.referee'
            ]].rename(columns={
                'fixture.id': 'fixture_id',
                'fixture.referee': 'referee',
            })

            merged = stats.merge(matches_slim, on='fixture_id', how='left')
            merged['league'] = league
            merged['season'] = season_dir.name
            merged['total_corners'] = merged['home_corners'] + merged['away_corners']
            all_data.append(merged)

    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()


def load_main_features():
    """Load the main features file (prefers SportMonks odds if available)."""
    candidates = [
        Path('data/03-features/features_with_sportmonks_odds.csv'),
        Path('data/03-features/features_all_5leagues_with_odds.csv'),
    ]
    for features_path in candidates:
        if features_path.exists():
            return pd.read_csv(features_path)
    raise FileNotFoundError("Main features file not found")


def merge_corner_targets(main_df: pd.DataFrame, corner_df: pd.DataFrame) -> pd.DataFrame:
    """Merge corner targets with main features."""
    corner_slim = corner_df[[
        'fixture_id', 'home_team', 'away_team', 'total_corners',
        'home_corners', 'away_corners', 'referee', 'date'
    ]].copy()

    if 'fixture_id' in main_df.columns:
        merged = main_df.merge(
            corner_slim[['fixture_id', 'total_corners', 'home_corners', 'away_corners', 'referee']],
            on='fixture_id',
            how='inner'
        )
    else:
        main_df['match_date'] = pd.to_datetime(main_df['date']).dt.date
        corner_slim['match_date'] = pd.to_datetime(corner_slim['date']).dt.date

        merged = main_df.merge(
            corner_slim,
            left_on=['home_team', 'away_team', 'match_date'],
            right_on=['home_team', 'away_team', 'match_date'],
            how='inner',
            suffixes=('', '_corner')
        )

    merged['over_8_5'] = (merged['total_corners'] > 8.5).astype(int)
    merged['over_9_5'] = (merged['total_corners'] > 9.5).astype(int)
    merged['over_10_5'] = (merged['total_corners'] > 10.5).astype(int)
    merged['over_11_5'] = (merged['total_corners'] > 11.5).astype(int)

    return merged


def calculate_referee_stats(df: pd.DataFrame) -> tuple:
    """Add referee corner statistics as features."""
    referee_stats = df.groupby('referee').agg({
        'total_corners': ['mean', 'std', 'count']
    }).reset_index()
    referee_stats.columns = ['referee', 'ref_corner_avg', 'ref_corner_std', 'ref_corner_matches']

    df = df.merge(referee_stats, on='referee', how='left')
    df['ref_corner_avg'] = df['ref_corner_avg'].fillna(df['total_corners'].mean())
    df['ref_corner_std'] = df['ref_corner_std'].fillna(df['total_corners'].std())
    df['ref_corner_matches'] = df['ref_corner_matches'].fillna(0)

    return df, referee_stats.set_index('referee').to_dict('index')


def get_numeric_features(df: pd.DataFrame, exclude_cols: list) -> list:
    """Get numeric feature columns."""
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    numeric_features = []
    for col in feature_cols:
        if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
            numeric_features.append(col)
        elif df[col].dtype == 'object':
            try:
                converted = pd.to_numeric(df[col], errors='coerce')
                if converted.notna().sum() > len(df) * 0.5:
                    numeric_features.append(col)
            except:
                pass

    return numeric_features


def train_corner_models(min_edge: float = 10.0):
    """Train corner prediction callibration using stacking with XGBoost meta-learner."""
    print("\nLoading data...")

    corner_df = load_corner_data()
    main_df = load_main_features()
    print(f"Corner data: {len(corner_df)}")
    print(f"Main features: {len(main_df)}")

    df = merge_corner_targets(main_df, corner_df)
    print(f"Merged: {len(df)}")

    df, referee_lookup = calculate_referee_stats(df)

    # Sort by date
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # Split
    n = len(df)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()

    # Excluded columns
    exclude_cols = [
        'fixture_id', 'date', 'match_date', 'home_team', 'away_team',
        'total_corners', 'home_corners', 'away_corners', 'referee',
        'over_8_5', 'over_9_5', 'over_10_5', 'over_11_5',
        'home_score', 'away_score', 'result', 'btts',
        'home_win_odds', 'away_win_odds', 'draw_odds',
        'over25_odds', 'under25_odds', 'btts_yes_odds', 'btts_no_odds',
        'season', 'league', 'round', 'date_corner'
    ]

    numeric_features = get_numeric_features(df, exclude_cols)

    # Use available Boruta features
    available_features = [f for f in BORUTA_FEATURES if f in numeric_features]
    if len(available_features) < 10:
        # Fall back to top features if Boruta features not all available
        available_features = numeric_features[:25]

    print(f"Using {len(available_features)} features")

    X_train = train_df[available_features].fillna(0).astype(float)
    X_val = val_df[available_features].fillna(0).astype(float)

    # Combine train and val for stacking (it does internal CV)
    X_train_full = pd.concat([X_train, X_val], ignore_index=True)

    models = {}

    for target in ['over_9_5', 'over_10_5', 'over_11_5']:
        y_train = train_df[target].values
        y_val = val_df[target].values
        y_train_full = np.concatenate([y_train, y_val])

        # Create stacking classifier with XGBoost meta-learner
        # This achieved 35.7% ROI vs 25.5% for simple voting
        stacking = StackingClassifier(
            estimators=[
                ('xgb', XGBClassifier(
                    n_estimators=200, max_depth=4, min_child_weight=15,
                    reg_lambda=5.0, learning_rate=0.05, subsample=0.8,
                    random_state=42, verbosity=0
                )),
                ('lgbm', LGBMClassifier(
                    n_estimators=200, max_depth=4, min_child_samples=50,
                    reg_lambda=5.0, learning_rate=0.05, subsample=0.8,
                    random_state=42, verbose=-1
                )),
                ('cat', CatBoostClassifier(
                    iterations=200, depth=4, l2_leaf_reg=10,
                    learning_rate=0.05, random_state=42, verbose=0
                )),
            ],
            final_estimator=XGBClassifier(
                n_estimators=50, max_depth=3, learning_rate=0.1,
                random_state=42, verbosity=0
            ),
            cv=3,
            stack_method='predict_proba',
            n_jobs=-1
        )

        # Fit stacking model
        stacking.fit(X_train_full, y_train_full)

        # Calibrate the stacking model
        stacking_cal = CalibratedClassifierCV(stacking, method='sigmoid', cv='prefit')
        stacking_cal.fit(X_val, y_val)

        models[target] = {'stacking': stacking_cal}

    return models, available_features, referee_lookup, df


def generate_predictions(tracker: CornersTrackerV3, min_edge: float = 10.0):
    """Generate corner predictions for upcoming matches using stacking model."""
    print("\n" + "=" * 70)
    print("GENERATING CORNER PREDICTIONS (V3 - STACKING + XGB META-LEARNER)")
    print("=" * 70)

    # Train callibration
    models, feature_cols, referee_lookup, historical_df = train_corner_models(min_edge)
    print(f"\nModels trained on {len(historical_df)} matches")
    print(f"Referee patterns: {len(referee_lookup)}")

    # Load main features for feature computation
    main_features = load_main_features()

    # Load upcoming fixtures
    print("\nLoading upcoming fixtures...")
    upcoming = []

    for league in ['premier_league', 'la_liga', 'serie_a']:
        matches_file = Path(f'data/01-raw/{league}/2025/matches.parquet')
        if matches_file.exists():
            df = pd.read_parquet(matches_file)
            df['league'] = league
            not_finished = df[df['fixture.status.short'] != 'FT'].copy()
            not_finished = not_finished.rename(columns={
                'fixture.id': 'fixture_id',
                'fixture.date': 'date',
                'teams.home.name': 'home_team',
                'teams.away.name': 'away_team',
                'fixture.referee': 'referee',
            })
            upcoming.append(not_finished)

    if not upcoming:
        print('No upcoming matches found')
        return

    upcoming_df = pd.concat(upcoming, ignore_index=True)
    upcoming_df['date'] = pd.to_datetime(upcoming_df['date']).dt.tz_localize(None)
    upcoming_df = upcoming_df.sort_values('date')

    # Filter to next 7 days
    today = datetime.now()
    next_week = today + timedelta(days=7)
    upcoming_df = upcoming_df[(upcoming_df['date'] >= today) & (upcoming_df['date'] <= next_week)]

    print(f"Upcoming matches: {len(upcoming_df)}")

    # Find matching features for upcoming matches
    new_bets = 0
    print("\nValue bets found:")

    for _, row in upcoming_df.iterrows():
        fixture_id = int(row['fixture_id'])
        match_date = str(row['date'])
        home_team = row['home_team']
        away_team = row['away_team']
        league = row['league']
        referee = row.get('referee', '')

        # Try to find features for this match
        match_features = main_features[
            (main_features['home_team_name'] == home_team) &
            (main_features['away_team_name'] == away_team)
        ]

        if len(match_features) == 0:
            # Use most recent features for each team as approximation
            home_matches = main_features[main_features['home_team_name'] == home_team].tail(1)
            away_matches = main_features[main_features['away_team_name'] == away_team].tail(1)

            if len(home_matches) > 0 and len(away_matches) > 0:
                # Combine features
                match_features = home_matches.copy()
            else:
                continue

        # Get feature vector
        feature_row = match_features.iloc[-1:].copy()

        # Add referee features
        if referee and referee in referee_lookup:
            ref_stats = referee_lookup[referee]
            feature_row['ref_corner_avg'] = ref_stats['ref_corner_avg']
            feature_row['ref_corner_std'] = ref_stats['ref_corner_std']
        else:
            feature_row['ref_corner_avg'] = historical_df['total_corners'].mean()
            feature_row['ref_corner_std'] = historical_df['total_corners'].std()

        # Get available features
        available_features = [f for f in feature_cols if f in feature_row.columns]
        if len(available_features) < 5:
            continue

        X = feature_row[available_features].fillna(0).astype(float)

        # Get predictions from stacking model
        predictions = {}
        for target in ['over_9_5', 'over_10_5', 'over_11_5']:
            model_set = models[target]
            stacking_prob = model_set['stacking'].predict_proba(X)[:, 1][0]
            predictions[target] = stacking_prob

        # Get additional info
        ref_avg = feature_row['ref_corner_avg'].iloc[0] if 'ref_corner_avg' in feature_row.columns else None
        goals_exp = feature_row['odds_goals_expectation'].iloc[0] if 'odds_goals_expectation' in feature_row.columns else None

        # Predicted corners (weighted average using model probabilities)
        predicted_corners = (
            9.5 + (predictions['over_9_5'] - 0.5) * 2 +
            (predictions['over_10_5'] - 0.5) * 2 +
            (predictions['over_11_5'] - 0.5) * 2
        )

        # Check for value bets using V3 thresholds
        best_bet = None
        best_edge = 0

        # V3 strategies (from backtest results)
        lines = [
            # UNDER bets (strongest in V3)
            ('UNDER', 10.5, 1 - predictions['over_10_5'], DEFAULT_CORNER_ODDS['under_10_5'], 0.70),  # +34.5% ROI
            ('UNDER', 11.5, 1 - predictions['over_11_5'], DEFAULT_CORNER_ODDS['under_11_5'], 0.75),  # +24.5% ROI
            ('UNDER', 9.5, 1 - predictions['over_9_5'], DEFAULT_CORNER_ODDS['under_9_5'], 0.65),     # +21% ROI
            # OVER bets (some value in V3)
            ('OVER', 9.5, predictions['over_9_5'], DEFAULT_CORNER_ODDS['over_9_5'], 0.65),          # +32% ROI
            ('OVER', 10.5, predictions['over_10_5'], DEFAULT_CORNER_ODDS['over_10_5'], 0.55),       # +17% ROI
        ]

        for bet_type, line, prob, odds, threshold in lines:
            edge = (odds * prob - 1) * 100
            if prob >= threshold and edge > min_edge and edge > best_edge:
                best_bet = (bet_type, line, prob, odds, edge)
                best_edge = edge

        if best_bet:
            bet_type, line, prob, odds, edge = best_bet
            tracker.add_prediction(
                fixture_id=fixture_id,
                match_date=match_date,
                home_team=home_team,
                away_team=away_team,
                league=league,
                referee=referee,
                predicted_corners=predicted_corners,
                bet_type=bet_type,
                line=line,
                our_odds=odds,
                our_probability=prob,
                edge=edge,
                ref_avg_corners=ref_avg,
                goals_expectation=goals_exp
            )
            new_bets += 1

    print(f"\nAdded {new_bets} new predictions")


def record_results_from_api(tracker: CornersTrackerV3):
    """Record results for settled matches."""
    print("\n" + "=" * 70)
    print("RECORDING RESULTS FROM MATCH DATA")
    print("=" * 70)

    pending_bets = [
        b for b in tracker.predictions["bets"]
        if b["status"] in ["pending", "closed"]
    ]

    if not pending_bets:
        print("No pending bets to check")
        return

    all_stats = []
    for league in ['premier_league', 'la_liga', 'serie_a']:
        stats_file = Path(f'data/01-raw/{league}/2025/match_stats.parquet')
        if stats_file.exists():
            df = pd.read_parquet(stats_file)
            all_stats.append(df)

    if not all_stats:
        print("No match stats found")
        return

    stats_df = pd.concat(all_stats, ignore_index=True)
    stats_df['total_corners'] = stats_df['home_corners'] + stats_df['away_corners']

    updated = 0
    for bet in pending_bets:
        fixture_id = bet['fixture_id']
        match_stats = stats_df[stats_df['fixture_id'] == fixture_id]
        if len(match_stats) > 0:
            actual_corners = int(match_stats.iloc[0]['total_corners'])
            tracker.record_result(fixture_id, actual_corners)
            updated += 1

    print(f"Updated {updated} bets with results")


def main():
    tracker = CornersTrackerV3()

    if len(sys.argv) < 2:
        print("Usage: python corners_paper_trade.py [predict|close|result|settle|status]")
        tracker.print_dashboard()
        return

    command = sys.argv[1].lower()

    if command == "predict":
        min_edge = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
        generate_predictions(tracker, min_edge=min_edge)
        tracker.print_dashboard()

    elif command == "close":
        if len(sys.argv) < 4:
            print("Usage: python corners_paper_trade.py close <key> <closing_odds>")
            print("\nPending bets:")
            for bet in tracker.predictions["bets"]:
                if bet["status"] == "pending":
                    print(f"  {bet['key']}: {bet['home_team']} vs {bet['away_team']}")
            return
        tracker.record_closing_odds(sys.argv[2], float(sys.argv[3]))

    elif command == "result":
        if len(sys.argv) < 4:
            print("Usage: python corners_paper_trade.py result <fixture_id> <actual_corners>")
            return
        tracker.record_result(int(sys.argv[2]), int(sys.argv[3]))
        tracker.print_dashboard()

    elif command == "settle":
        record_results_from_api(tracker)
        tracker.print_dashboard()

    elif command == "status":
        tracker.print_dashboard()

    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
