#!/usr/bin/env python
"""
Fouls Betting Paper Trading

This script validates total fouls betting edge using REALISTIC BOOKMAKER LINES.

Bookmakers typically offer: 26.5, 27.5, 28.5 (not 22.5, 24.5)

Strategy:
1. Uses features from main features file
2. CatBoost models for each line
3. Referee stats calculated from training data only (leakage-free)

Lines available at bookmaker:
- 26.5: OVER/UNDER
- 27.5: OVER/UNDER
- 28.5: OVER/UNDER

Usage:
    python experiments/fouls_paper_trade.py predict    # Generate predictions
    python experiments/fouls_paper_trade.py settle     # Auto-settle from data
    python experiments/fouls_paper_trade.py status     # View dashboard
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

import warnings
warnings.filterwarnings('ignore')

# Default fouls odds (realistic bookmaker lines)
# Bookmakers typically offer: 26.5, 27.5, 28.5
DEFAULT_FOULS_ODDS = {
    'over_26_5': 1.85, 'under_26_5': 1.95,
    'over_27_5': 2.00, 'under_27_5': 1.80,
    'over_28_5': 2.20, 'under_28_5': 1.65,
}

# Strategies for realistic bookmaker lines
STRATEGIES = {
    'over_26_5': {'direction': 'OVER', 'threshold': 0.60, 'roi': 52.9, 'model': 'catboost'},
    'under_26_5': {'direction': 'UNDER', 'threshold': 0.55, 'roi': 25.0, 'model': 'catboost'},
    'over_27_5': {'direction': 'OVER', 'threshold': 0.60, 'roi': 40.0, 'model': 'catboost'},
    'under_27_5': {'direction': 'UNDER', 'threshold': 0.55, 'roi': 30.0, 'model': 'catboost'},
    'over_28_5': {'direction': 'OVER', 'threshold': 0.65, 'roi': 35.0, 'model': 'catboost'},
    'under_28_5': {'direction': 'UNDER', 'threshold': 0.55, 'roi': 35.0, 'model': 'catboost'},
}


class FoulsTracker:
    """Track fouls betting predictions with CLV analysis."""

    def __init__(self, output_path: str = "experiments/outputs/fouls_tracking.json"):
        self.output_path = Path(output_path)
        self.predictions = self._load_data()

    def _load_data(self) -> Dict:
        if self.output_path.exists():
            with open(self.output_path, 'r') as f:
                return json.load(f)
        return {"bets": [], "summary": {}, "version": "v1"}

    def _save_data(self):
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
        predicted_fouls: float,
        bet_type: str,
        line: float,
        our_odds: float,
        our_probability: float,
        edge: float,
        ref_avg_fouls: float = None,
    ):
        key = f"{fixture_id}_{bet_type}_{line}"

        bet = {
            "key": key,
            "fixture_id": fixture_id,
            "match_date": match_date,
            "home_team": home_team,
            "away_team": away_team,
            "league": league,
            "referee": referee,
            "ref_avg_fouls": ref_avg_fouls,
            "predicted_fouls": predicted_fouls,
            "bet_type": bet_type,
            "line": line,
            "our_odds": our_odds,
            "our_probability": our_probability,
            "edge": edge,
            "closing_odds": None,
            "clv": None,
            "actual_fouls": None,
            "won": None,
            "created_at": datetime.now().isoformat(),
            "status": "pending"
        }

        existing = [b for b in self.predictions["bets"] if b["key"] == key]
        if existing:
            print(f"  [EXISTS] {home_team} vs {away_team} ({bet_type} {line})")
            return

        self.predictions["bets"].append(bet)
        self._save_data()

        ref_info = f" [Ref: {referee[:15] if referee else 'None'}={ref_avg_fouls:.1f}]" if ref_avg_fouls else ""
        print(f"  [NEW] {home_team} vs {away_team} - {bet_type} {line} @ {our_odds:.2f} (+{edge:.1f}%){ref_info}")

    def record_result(self, fixture_id: int, actual_fouls: int):
        updated = 0
        for bet in self.predictions["bets"]:
            if bet["fixture_id"] == fixture_id:
                bet["actual_fouls"] = actual_fouls
                if bet["bet_type"] == "OVER":
                    bet["won"] = actual_fouls > bet["line"]
                else:
                    bet["won"] = actual_fouls < bet["line"]
                bet["status"] = "settled"
                updated += 1

        if updated > 0:
            self._save_data()
            print(f"Recorded {actual_fouls} fouls for fixture {fixture_id} ({updated} bets)")

    def get_status(self) -> Dict:
        bets = self.predictions["bets"]
        if not bets:
            return {"total_bets": 0}

        pending = [b for b in bets if b["status"] == "pending"]
        settled = [b for b in bets if b["status"] == "settled"]

        summary = {
            "total_bets": len(bets),
            "pending": len(pending),
            "settled": len(settled),
        }

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
        status = self.get_status()

        print("\n" + "=" * 70)
        print("FOULS BETTING PAPER TRADE - DASHBOARD")
        print("=" * 70)

        print(f"\nTotal bets tracked: {status.get('total_bets', 0)}")
        print(f"  Pending: {status.get('pending', 0)}")
        print(f"  Settled: {status.get('settled', 0)}")

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
                status_str = f"{result} ({bet['actual_fouls']})"

            print(f"  {date} | {match:<26} | {bet_desc:<12} | {ref:<10} | {status_str}")

        print("=" * 70)


def load_fouls_data():
    """Load fouls stats from match_stats parquet files."""
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
            merged['total_fouls'] = merged['home_fouls'] + merged['away_fouls']
            all_data.append(merged)

    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()


def load_main_features():
    """Load the main features file."""
    features_path = Path('data/03-features/features_all_5leagues_with_odds.csv')
    if not features_path.exists():
        raise FileNotFoundError(f"Main features file not found: {features_path}")
    return pd.read_csv(features_path)


def train_fouls_models(min_edge: float = 10.0):
    """Train fouls prediction models using best performers per target."""
    print("\nLoading data...")

    fouls_df = load_fouls_data()
    main_df = load_main_features()
    print(f"Fouls data: {len(fouls_df)}")
    print(f"Main features: {len(main_df)}")

    # Merge
    merged = main_df.merge(
        fouls_df[['fixture_id', 'total_fouls', 'home_fouls', 'away_fouls', 'referee']],
        on='fixture_id',
        how='inner'
    )
    print(f"Merged: {len(merged)}")

    # Sort by date
    merged['date'] = pd.to_datetime(merged['date'])
    df = merged.sort_values('date').reset_index(drop=True)

    # Create targets for realistic bookmaker lines (26.5, 27.5, 28.5)
    df['over_26_5'] = (df['total_fouls'] > 26.5).astype(int)
    df['over_27_5'] = (df['total_fouls'] > 27.5).astype(int)
    df['over_28_5'] = (df['total_fouls'] > 28.5).astype(int)
    df['under_26_5'] = (df['total_fouls'] < 26.5).astype(int)
    df['under_27_5'] = (df['total_fouls'] < 27.5).astype(int)
    df['under_28_5'] = (df['total_fouls'] < 28.5).astype(int)

    # Temporal split
    n = len(df)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()

    # Calculate referee stats from training data ONLY (leakage-free)
    ref_stats = train_df.groupby('referee')['total_fouls'].agg(['mean', 'std']).reset_index()
    ref_stats.columns = ['referee', 'ref_fouls_avg', 'ref_fouls_std']
    referee_lookup = ref_stats.set_index('referee').to_dict('index')

    global_mean = train_df['total_fouls'].mean()
    global_std = train_df['total_fouls'].std()

    # Apply referee stats
    for df_split in [train_df, val_df]:
        df_split = df_split.merge(ref_stats, on='referee', how='left')
        df_split['ref_fouls_avg'] = df_split['ref_fouls_avg'].fillna(global_mean)
        df_split['ref_fouls_std'] = df_split['ref_fouls_std'].fillna(global_std)

    # Feature columns
    exclude_cols = [
        'fixture_id', 'date', 'home_team_name', 'away_team_name',
        'home_team_id', 'away_team_id', 'round', 'referee',
        'total_fouls', 'home_fouls', 'away_fouls',
        'over_26_5', 'over_27_5', 'over_28_5',
        'under_26_5', 'under_27_5', 'under_28_5',
        'home_score', 'away_score', 'result', 'btts',
    ]

    feature_cols = [c for c in train_df.columns if c not in exclude_cols
                    and train_df[c].dtype in ['int64', 'float64', 'int32', 'float32']]

    print(f"Using {len(feature_cols)} features")

    X_train = train_df[feature_cols].fillna(0).astype(float)
    X_val = val_df[feature_cols].fillna(0).astype(float)
    X_train_full = pd.concat([X_train, X_val], ignore_index=True)

    models = {}

    # Train models for all targets (26.5, 27.5, 28.5 - OVER and UNDER)
    targets = ['over_26_5', 'over_27_5', 'over_28_5', 'under_26_5', 'under_27_5', 'under_28_5']

    for target in targets:
        if target not in train_df.columns:
            continue

        y_train = train_df[target].values
        y_val = val_df[target].values
        y_train_full = np.concatenate([y_train, y_val])

        # Use CatBoost for all (best performer)
        model = CatBoostClassifier(
            iterations=200, depth=4, l2_leaf_reg=10,
            learning_rate=0.05, random_state=42, verbose=0
        )

        model.fit(X_train_full, y_train_full)

        # Calibrate
        model_cal = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
        model_cal.fit(X_val, y_val)

        models[target] = model_cal

    return models, feature_cols, referee_lookup, df, global_mean, global_std


def generate_predictions(tracker: FoulsTracker, min_edge: float = 10.0):
    """Generate fouls predictions for upcoming matches."""
    print("\n" + "=" * 70)
    print("GENERATING FOULS PREDICTIONS (MULTI-MODEL)")
    print("=" * 70)

    models, feature_cols, referee_lookup, historical_df, global_mean, global_std = train_fouls_models(min_edge)
    print(f"\nModels trained on {len(historical_df)} matches")
    print(f"Referee patterns: {len(referee_lookup)}")

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

    new_bets = 0
    print("\nValue bets found:")

    for _, row in upcoming_df.iterrows():
        fixture_id = int(row['fixture_id'])
        match_date = str(row['date'])
        home_team = row['home_team']
        away_team = row['away_team']
        league = row['league']
        referee = row.get('referee', '')

        # Find features
        match_features = main_features[
            (main_features['home_team_name'] == home_team) &
            (main_features['away_team_name'] == away_team)
        ]

        if len(match_features) == 0:
            home_matches = main_features[main_features['home_team_name'] == home_team].tail(1)
            if len(home_matches) > 0:
                match_features = home_matches
            else:
                continue

        feature_row = match_features.iloc[-1:].copy()

        # Add referee features (from training data)
        if referee and referee in referee_lookup:
            ref_stats = referee_lookup[referee]
            feature_row['ref_fouls_avg'] = ref_stats['ref_fouls_avg']
            feature_row['ref_fouls_std'] = ref_stats['ref_fouls_std']
        else:
            feature_row['ref_fouls_avg'] = global_mean
            feature_row['ref_fouls_std'] = global_std

        available_features = [f for f in feature_cols if f in feature_row.columns]
        if len(available_features) < 5:
            continue

        X = feature_row[available_features].fillna(0).astype(float)

        # Get predictions for all targets
        predictions = {}
        for target in models.keys():
            prob = models[target].predict_proba(X)[:, 1][0]
            predictions[target] = prob

        ref_avg = feature_row['ref_fouls_avg'].iloc[0] if 'ref_fouls_avg' in feature_row.columns else None

        # Predicted fouls based on probabilities
        predicted_fouls = (
            26.5 + (predictions.get('over_26_5', 0.5) - 0.5) * 4 +
            (predictions.get('over_27_5', 0.5) - 0.5) * 4 +
            (predictions.get('over_28_5', 0.5) - 0.5) * 4
        )

        # Check for value bets - realistic bookmaker lines (26.5, 27.5, 28.5)
        lines = [
            # Over 26.5
            ('OVER', 26.5, predictions.get('over_26_5', 0), DEFAULT_FOULS_ODDS['over_26_5'], 0.60),
            # Under 26.5
            ('UNDER', 26.5, predictions.get('under_26_5', 0), DEFAULT_FOULS_ODDS['under_26_5'], 0.55),
            # Over 27.5
            ('OVER', 27.5, predictions.get('over_27_5', 0), DEFAULT_FOULS_ODDS['over_27_5'], 0.60),
            # Under 27.5
            ('UNDER', 27.5, predictions.get('under_27_5', 0), DEFAULT_FOULS_ODDS['under_27_5'], 0.55),
            # Over 28.5
            ('OVER', 28.5, predictions.get('over_28_5', 0), DEFAULT_FOULS_ODDS['over_28_5'], 0.65),
            # Under 28.5
            ('UNDER', 28.5, predictions.get('under_28_5', 0), DEFAULT_FOULS_ODDS['under_28_5'], 0.55),
        ]

        for bet_type, line, prob, odds, threshold in lines:
            edge = (odds * prob - 1) * 100
            if prob >= threshold and edge > min_edge:
                tracker.add_prediction(
                    fixture_id=fixture_id,
                    match_date=match_date,
                    home_team=home_team,
                    away_team=away_team,
                    league=league,
                    referee=referee,
                    predicted_fouls=predicted_fouls,
                    bet_type=bet_type,
                    line=line,
                    our_odds=odds,
                    our_probability=prob,
                    edge=edge,
                    ref_avg_fouls=ref_avg
                )
                new_bets += 1

    print(f"\nAdded {new_bets} new predictions")


def record_results_from_api(tracker: FoulsTracker):
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
    stats_df['total_fouls'] = stats_df['home_fouls'] + stats_df['away_fouls']

    updated = 0
    for bet in pending_bets:
        fixture_id = bet['fixture_id']
        match_stats = stats_df[stats_df['fixture_id'] == fixture_id]
        if len(match_stats) > 0:
            actual_fouls = int(match_stats.iloc[0]['total_fouls'])
            tracker.record_result(fixture_id, actual_fouls)
            updated += 1

    print(f"Updated {updated} bets with results")


def main():
    tracker = FoulsTracker()

    if len(sys.argv) < 2:
        print("Usage: python fouls_paper_trade.py [predict|settle|status]")
        tracker.print_dashboard()
        return

    command = sys.argv[1].lower()

    if command == "predict":
        min_edge = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
        generate_predictions(tracker, min_edge=min_edge)
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
