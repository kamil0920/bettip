#!/usr/bin/env python
"""
Away Win Betting Paper Trading

This script validates away win betting predictions using the OPTIMIZED model
from optimization-results-11 (walk-forward validated: +14.1% ROI avg).

Best strategy from optimization: CatBoost >= 0.4
- Walk-forward ROI: +14.1% (σ = 4.3%)
- P(Profit): 97.6%
- 535 bets across 3 temporal folds

Usage:
    python experiments/away_win_paper_trade.py predict    # Generate predictions
    python experiments/away_win_paper_trade.py settle     # Auto-settle from data
    python experiments/away_win_paper_trade.py status     # View dashboard
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from sklearn.calibration import CalibratedClassifierCV
from catboost import CatBoostClassifier

import warnings
warnings.filterwarnings('ignore')

# Configuration from optimization-results-11 (CatBoost >= 0.4)
PROBABILITY_THRESHOLD = 0.40  # Best threshold from walk-forward validation
EXPECTED_ROI = 14.1  # Walk-forward average ROI
MIN_EDGE = 3.0  # Minimum edge percentage to bet (lower to match optimization)


class AwayWinTracker:
    """Track away win betting predictions."""

    def __init__(self, output_path: str = "experiments/outputs/away_win_tracking.json"):
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
        away_win_prob: float,
        odds: float,
        edge: float,
        home_elo: float = None,
        away_elo: float = None,
        elo_diff: float = None,
    ):
        key = f"{fixture_id}_AWAY_WIN"

        bet = {
            "key": key,
            "fixture_id": fixture_id,
            "match_date": match_date,
            "home_team": home_team,
            "away_team": away_team,
            "league": league,
            "away_win_prob": away_win_prob,
            "odds": odds,
            "edge": edge,
            "home_elo": home_elo,
            "away_elo": away_elo,
            "elo_diff": elo_diff,
            "actual_result": None,
            "won": None,
            "created_at": datetime.now().isoformat(),
            "status": "pending"
        }

        existing = [b for b in self.predictions["bets"] if b["key"] == key]
        if existing:
            print(f"  [EXISTS] {home_team} vs {away_team}")
            return

        self.predictions["bets"].append(bet)
        self._save_data()

        elo_info = f" [ELO: {home_elo:.0f} vs {away_elo:.0f}]" if home_elo and away_elo else ""
        print(f"  [NEW] {home_team} vs {away_team} - AWAY @ {odds:.2f} (prob={away_win_prob:.1%}, edge={edge:+.1f}%){elo_info}")

    def record_result(self, fixture_id: int, home_score: int, away_score: int):
        for bet in self.predictions["bets"]:
            if bet["fixture_id"] == fixture_id:
                bet["actual_result"] = f"{home_score}-{away_score}"
                bet["won"] = away_score > home_score
                bet["status"] = "settled"

        self._save_data()
        result_str = "AWAY WIN" if away_score > home_score else ("DRAW" if home_score == away_score else "HOME WIN")
        print(f"Recorded {home_score}-{away_score} ({result_str}) for fixture {fixture_id}")

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
            profit = sum((b["odds"] - 1) if b["won"] else -1 for b in settled)
            summary["roi"] = (profit / len(settled)) * 100
            summary["avg_edge"] = np.mean([b["edge"] for b in settled])
            summary["avg_prob"] = np.mean([b["away_win_prob"] for b in settled])

        return summary

    def print_dashboard(self):
        status = self.get_status()

        print("\n" + "=" * 70)
        print("AWAY WIN BETTING PAPER TRADE - DASHBOARD")
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
            print(f"  Average predicted prob: {status['avg_prob']:.1%}")

        print("\n" + "-" * 70)
        print("Recent Bets:")
        print("-" * 70)

        bets = self.predictions["bets"][-15:]
        for bet in bets:
            match = f"{bet['home_team'][:15]} vs {bet['away_team'][:15]}"
            date = bet['match_date'][:10] if bet['match_date'] else 'N/A'
            prob = f"{bet['away_win_prob']:.1%}"

            status_str = bet['status'].upper()
            if bet['status'] == 'settled':
                result = "WON" if bet['won'] else "LOST"
                status_str = f"{result} ({bet['actual_result']})"

            print(f"  {date} | {match:<35} | {prob:>6} | {status_str}")

        print("=" * 70)


def load_main_features():
    """Load the main features file (with SportMonks odds for accurate ROI)."""
    # Prefer SportMonks odds file (used in optimization)
    candidates = [
        Path('data/03-features/features_with_sportmonks_odds.csv'),
        Path('data/03-features/features_all_5leagues_with_odds.csv'),
    ]
    for features_path in candidates:
        if features_path.exists():
            print(f"Loading features from: {features_path.name}")
            return pd.read_csv(features_path, low_memory=False)
    raise FileNotFoundError(f"Features file not found. Tried: {[str(p) for p in candidates]}")


def train_away_win_model():
    """Train away win prediction model using OPTIMIZED parameters from optimization-results-11."""
    print("\nLoading data...")

    main_df = load_main_features()
    print(f"Main features: {len(main_df)}")

    # Filter to completed matches with results
    df = main_df[main_df['home_goals'].notna() & main_df['away_goals'].notna()].copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # Create target
    df['away_win'] = (df['away_goals'] > df['home_goals']).astype(int)

    print(f"Matches with results: {len(df)}")
    print(f"Away win rate: {df['away_win'].mean():.1%}")

    # Temporal split (60/20/20 as in optimization)
    n = len(df)
    train_end = int(0.6 * n)
    val_end = int(0.8 * n)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()

    # OPTIMIZED features from optimization-results-11 (walk-forward validated)
    optimized_features = [
        'odds_away_prob', 'odds_move_home', 'odds_home_prob', 'home_cards_ema',
        'odds_prob_diff', 'home_goals_conceded_ema', 'away_goals_scored_ema',
        'away_late_goal_rate', 'away_draws_last_n', 'home_elo',
        'poisson_away_win_prob', 'fouls_diff', 'away_early_goal_rate',
        'ref_matches', 'ppg_diff', 'odds_move_away', 'ref_home_win_pct',
        'ref_draw_pct', 'home_shot_accuracy', 'home_attack_strength',
        'away_clean_sheet_streak', 'home_defense_strength', 'ref_avg_goals',
        'home_goals_scored_ema', 'cards_diff', 'away_corners_conceded_ema',
        'home_away_ppg_diff', 'away_points_ema', 'home_home_draws',
        'odds_upset_potential_y', 'away_avg_yellows', 'poisson_draw_prob',
        'home_home_goals_conceded', 'away_goals_conceded_ema',
        'away_corners_won_ema', 'home_away_gd_diff', 'overround_change',
        'expected_total_shots', 'home_goals_scored_last_n', 'sm_cards_over_odds'
    ]

    # Use only available optimized features
    final_features = [f for f in optimized_features if f in train_df.columns]
    print(f"Using {len(final_features)}/{len(optimized_features)} optimized features")

    X_train = train_df[final_features].fillna(0).astype(float)
    X_val = val_df[final_features].fillna(0).astype(float)
    y_train = train_df['away_win'].values
    y_val = val_df['away_win'].values

    # OPTIMIZED CatBoost parameters from optimization-results-11
    model = CatBoostClassifier(
        iterations=500,
        depth=6,
        l2_leaf_reg=4.12,
        learning_rate=0.128,
        random_strength=1.38,
        bagging_temperature=0.55,
        random_state=42,
        verbose=0
    )

    model.fit(X_train, y_train)

    # Calibrate with Platt scaling (as in optimization)
    model_cal = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
    model_cal.fit(X_val, y_val)

    # Validation stats
    val_probs = model_cal.predict_proba(X_val)[:, 1]
    val_accuracy = np.mean((val_probs >= PROBABILITY_THRESHOLD) == y_val[val_probs >= PROBABILITY_THRESHOLD]) if (val_probs >= PROBABILITY_THRESHOLD).sum() > 0 else 0
    print(f"Validation accuracy at threshold {PROBABILITY_THRESHOLD}: {val_accuracy:.1%}")
    print(f"Bets at threshold: {(val_probs >= PROBABILITY_THRESHOLD).sum()}")

    return model_cal, final_features, df


def generate_predictions(tracker: AwayWinTracker, min_edge: float = MIN_EDGE):
    """Generate away win predictions for upcoming matches."""
    print("\n" + "=" * 70)
    print("GENERATING AWAY WIN PREDICTIONS")
    print("=" * 70)

    model, feature_cols, historical_df = train_away_win_model()
    print(f"\nModel trained on {len(historical_df)} matches")

    main_features = load_main_features()

    # Load upcoming fixtures
    print("\nLoading upcoming fixtures...")
    upcoming = []

    for league in ['premier_league', 'la_liga', 'serie_a', 'bundesliga', 'ligue_1']:
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

        # Find features
        match_features = main_features[
            (main_features['home_team_name'] == home_team) &
            (main_features['away_team_name'] == away_team)
        ]

        if len(match_features) == 0:
            # Use most recent away team data
            away_matches = main_features[main_features['away_team_name'] == away_team].tail(1)
            if len(away_matches) > 0:
                match_features = away_matches
            else:
                continue

        feature_row = match_features.iloc[-1:].copy()

        available_features = [f for f in feature_cols if f in feature_row.columns]
        if len(available_features) < 5:
            continue

        X = feature_row[available_features].fillna(0).astype(float)

        # Get prediction
        away_win_prob = model.predict_proba(X)[:, 1][0]

        # Get ELO features
        home_elo = feature_row['home_elo'].iloc[0] if 'home_elo' in feature_row.columns else None
        away_elo = feature_row['away_elo'].iloc[0] if 'away_elo' in feature_row.columns else None
        elo_diff = feature_row['elo_diff'].iloc[0] if 'elo_diff' in feature_row.columns else None

        # Get odds (use avg_away_open if available, otherwise estimate)
        odds = feature_row['avg_away_open'].iloc[0] if 'avg_away_open' in feature_row.columns else None
        if odds is None or np.isnan(odds):
            # Estimate odds from probability (with ~5% vig)
            odds = 0.95 / away_win_prob if away_win_prob > 0 else 10.0
            odds = min(max(odds, 1.5), 10.0)

        # Calculate edge
        edge = (odds * away_win_prob - 1) * 100

        # Check if meets criteria
        if away_win_prob >= PROBABILITY_THRESHOLD and edge > min_edge:
            tracker.add_prediction(
                fixture_id=fixture_id,
                match_date=match_date,
                home_team=home_team,
                away_team=away_team,
                league=league,
                away_win_prob=away_win_prob,
                odds=odds,
                edge=edge,
                home_elo=home_elo,
                away_elo=away_elo,
                elo_diff=elo_diff,
            )
            new_bets += 1

    print(f"\nAdded {new_bets} new predictions")


def record_results_from_data(tracker: AwayWinTracker):
    """Record results for settled matches."""
    print("\n" + "=" * 70)
    print("RECORDING RESULTS FROM MATCH DATA")
    print("=" * 70)

    pending_bets = [
        b for b in tracker.predictions["bets"]
        if b["status"] == "pending"
    ]

    if not pending_bets:
        print("No pending bets to check")
        return

    # Load match results
    all_matches = []
    for league in ['premier_league', 'la_liga', 'serie_a', 'bundesliga', 'ligue_1']:
        matches_file = Path(f'data/01-raw/{league}/2025/matches.parquet')
        if matches_file.exists():
            df = pd.read_parquet(matches_file)
            finished = df[df['fixture.status.short'] == 'FT'].copy()
            finished = finished.rename(columns={
                'fixture.id': 'fixture_id',
                'goals.home': 'home_goals',
                'goals.away': 'away_goals',
            })
            all_matches.append(finished[['fixture_id', 'home_goals', 'away_goals']])

    if not all_matches:
        print("No match results found")
        return

    matches_df = pd.concat(all_matches, ignore_index=True)

    updated = 0
    for bet in pending_bets:
        fixture_id = bet['fixture_id']
        match = matches_df[matches_df['fixture_id'] == fixture_id]
        if len(match) > 0:
            home_score = int(match.iloc[0]['home_goals'])
            away_score = int(match.iloc[0]['away_goals'])
            tracker.record_result(fixture_id, home_score, away_score)
            updated += 1

    print(f"Updated {updated} bets with results")


def main():
    tracker = AwayWinTracker()

    if len(sys.argv) < 2:
        print("Usage: python away_win_paper_trade.py [predict|settle|status]")
        tracker.print_dashboard()
        return

    command = sys.argv[1].lower()

    if command == "predict":
        min_edge = float(sys.argv[2]) if len(sys.argv) > 2 else MIN_EDGE
        generate_predictions(tracker, min_edge=min_edge)
        tracker.print_dashboard()

    elif command == "settle":
        record_results_from_data(tracker)
        tracker.print_dashboard()

    elif command == "status":
        tracker.print_dashboard()

    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
