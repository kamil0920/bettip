#!/usr/bin/env python
"""
Away Win Betting Paper Trading with Real Odds Validation

This script validates away win betting predictions using the OPTIMIZED model
from optimization-results-11 (walk-forward validated: +14.1% ROI avg).

Best strategy from optimization: CatBoost >= 0.4
- Walk-forward ROI: +14.1% (σ = 4.3%)
- P(Profit): 97.6%
- 535 bets across 3 temporal folds

Usage:
    python experiments/away_win_paper_trade.py predict    # Generate predictions with real odds
    python experiments/away_win_paper_trade.py settle     # Auto-settle from data
    python experiments/away_win_paper_trade.py status     # View dashboard
    python experiments/away_win_paper_trade.py validate   # Compare real vs estimated odds
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import os

from sklearn.calibration import CalibratedClassifierCV
from catboost import CatBoostClassifier
import requests
from dotenv import load_dotenv

load_dotenv()

import warnings
warnings.filterwarnings('ignore')

# Configuration from optimization-results-11 (CatBoost >= 0.4)
PROBABILITY_THRESHOLD = 0.40  # Best threshold from walk-forward validation
EXPECTED_ROI = 14.1  # Walk-forward average ROI
MIN_EDGE = 3.0  # Minimum edge percentage to bet (lower to match optimization)


class SportMonksOddsFetcher:
    """Fetch real-time away win odds from SportMonks API."""

    BASE_URL = "https://api.sportmonks.com/v3"

    # SportMonks league IDs
    LEAGUES = {
        "premier_league": 8,
        "bundesliga": 82,
        "ligue_1": 301,
        "serie_a": 384,
        "la_liga": 564,
    }

    def __init__(self):
        self.api_key = os.getenv("SPORTSMONK_KEY")
        if not self.api_key:
            print("Warning: SPORTSMONK_KEY not set. Using estimated odds.")
        self._cache = {}

    def _request(self, endpoint: str, params: dict = None) -> dict:
        """Make API request."""
        if not self.api_key:
            return {}

        url = f"{self.BASE_URL}/{endpoint}"
        request_params = {"api_token": self.api_key}
        if params:
            request_params.update(params)

        try:
            response = requests.get(url, params=request_params, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"API error: {e}")
            return {}

    def get_upcoming_fixtures_with_odds(self, days_ahead: int = 7) -> pd.DataFrame:
        """Get upcoming fixtures with real away win odds."""
        if not self.api_key:
            return pd.DataFrame()

        start_date = datetime.now().strftime("%Y-%m-%d")
        end_date = (datetime.now() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

        league_ids = list(self.LEAGUES.values())

        all_fixtures = []

        for league_id in league_ids:
            endpoint = f"football/fixtures/between/{start_date}/{end_date}"
            params = {
                "include": "odds;participants",
                "filters": f"fixtureLeagues:{league_id}",
                "per_page": 50
            }

            data = self._request(endpoint, params)
            fixtures = data.get("data", [])

            for fix in fixtures:
                fix_id = fix.get("id")
                name = fix.get("name", "")
                start_time = fix.get("starting_at", "")

                # Parse teams from participants
                participants = fix.get("participants", [])
                home_team = away_team = None
                for p in participants:
                    if p.get("meta", {}).get("location") == "home":
                        home_team = p.get("name")
                    elif p.get("meta", {}).get("location") == "away":
                        away_team = p.get("name")

                if not home_team or not away_team:
                    parts = name.split(" vs ")
                    if len(parts) == 2:
                        home_team, away_team = parts

                # Extract away win odds (market_id=1 is Fulltime Result)
                odds_list = fix.get("odds", [])
                away_odds = None
                home_odds = None
                draw_odds = None

                for odds in odds_list:
                    if odds.get("market_id") == 1:  # Fulltime Result
                        label = odds.get("label", "").lower()
                        value = odds.get("value")
                        if value:
                            if label == "2" or "away" in label:
                                if away_odds is None or float(value) > away_odds:
                                    away_odds = float(value)
                            elif label == "1" or "home" in label:
                                if home_odds is None or float(value) > home_odds:
                                    home_odds = float(value)
                            elif label == "x" or "draw" in label:
                                if draw_odds is None or float(value) > draw_odds:
                                    draw_odds = float(value)

                if away_odds:
                    # Find league name
                    league_name = next(
                        (k for k, v in self.LEAGUES.items() if v == league_id),
                        "unknown"
                    )

                    all_fixtures.append({
                        "fixture_id": fix_id,
                        "date": start_time,
                        "home_team": home_team,
                        "away_team": away_team,
                        "league": league_name,
                        "away_odds_real": away_odds,
                        "home_odds_real": home_odds,
                        "draw_odds_real": draw_odds,
                        "odds_source": "sportmonks"
                    })

        if all_fixtures:
            df = pd.DataFrame(all_fixtures)
            df['date'] = pd.to_datetime(df['date'])
            return df.sort_values('date')

        return pd.DataFrame()

    def get_fixture_odds(self, fixture_id: int) -> Optional[Dict]:
        """Get odds for a specific fixture."""
        if fixture_id in self._cache:
            return self._cache[fixture_id]

        data = self._request(
            f"football/fixtures/{fixture_id}",
            params={"include": "odds"}
        )

        fixture = data.get("data", {})
        odds_list = fixture.get("odds", [])

        result = {"away": None, "home": None, "draw": None}

        for odds in odds_list:
            if odds.get("market_id") == 1:
                label = odds.get("label", "").lower()
                value = odds.get("value")
                if value:
                    if label == "2" or "away" in label:
                        if result["away"] is None or float(value) > result["away"]:
                            result["away"] = float(value)
                    elif label == "1" or "home" in label:
                        if result["home"] is None or float(value) > result["home"]:
                            result["home"] = float(value)
                    elif label == "x" or "draw" in label:
                        if result["draw"] is None or float(value) > result["draw"]:
                            result["draw"] = float(value)

        self._cache[fixture_id] = result
        return result


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
        odds_real: float = None,
        odds_source: str = "estimated",
    ):
        key = f"{fixture_id}_AWAY_WIN"

        # Use real odds if available, otherwise estimated
        final_odds = odds_real if odds_real else odds
        final_edge = (final_odds * away_win_prob - 1) * 100

        bet = {
            "key": key,
            "fixture_id": fixture_id,
            "match_date": match_date,
            "home_team": home_team,
            "away_team": away_team,
            "league": league,
            "away_win_prob": away_win_prob,
            "odds": final_odds,
            "odds_estimated": odds,
            "odds_real": odds_real,
            "odds_source": odds_source,
            "edge": final_edge,
            "edge_estimated": edge,
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

        odds_tag = "REAL" if odds_real else "EST"
        elo_info = f" [ELO: {home_elo:.0f} vs {away_elo:.0f}]" if home_elo and away_elo else ""
        print(f"  [NEW] {home_team} vs {away_team} - AWAY @ {final_odds:.2f} [{odds_tag}] (prob={away_win_prob:.1%}, edge={final_edge:+.1f}%){elo_info}")

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

        # Split by odds source
        real_odds_bets = [b for b in bets if b.get("odds_source") == "sportmonks"]
        est_odds_bets = [b for b in bets if b.get("odds_source") != "sportmonks"]

        summary = {
            "total_bets": len(bets),
            "pending": len(pending),
            "settled": len(settled),
            "real_odds_bets": len(real_odds_bets),
            "estimated_odds_bets": len(est_odds_bets),
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

            # Split results by odds source
            settled_real = [b for b in settled if b.get("odds_source") == "sportmonks"]
            settled_est = [b for b in settled if b.get("odds_source") != "sportmonks"]

            if settled_real:
                wins_real = sum(1 for b in settled_real if b["won"])
                profit_real = sum((b["odds"] - 1) if b["won"] else -1 for b in settled_real)
                summary["real_odds_wins"] = wins_real
                summary["real_odds_losses"] = len(settled_real) - wins_real
                summary["real_odds_roi"] = (profit_real / len(settled_real)) * 100

            if settled_est:
                wins_est = sum(1 for b in settled_est if b["won"])
                profit_est = sum((b["odds"] - 1) if b["won"] else -1 for b in settled_est)
                summary["est_odds_wins"] = wins_est
                summary["est_odds_losses"] = len(settled_est) - wins_est
                summary["est_odds_roi"] = (profit_est / len(settled_est)) * 100

        return summary

    def print_dashboard(self):
        status = self.get_status()

        print("\n" + "=" * 70)
        print("AWAY WIN BETTING PAPER TRADE - DASHBOARD (with Real Odds Validation)")
        print("=" * 70)

        print(f"\nTotal bets tracked: {status.get('total_bets', 0)}")
        print(f"  Pending: {status.get('pending', 0)}")
        print(f"  Settled: {status.get('settled', 0)}")
        print(f"  With real odds: {status.get('real_odds_bets', 0)}")
        print(f"  With estimated odds: {status.get('estimated_odds_bets', 0)}")

        if status.get('settled', 0) > 0:
            print(f"\n📊 Overall Results:")
            print(f"  Wins: {status['wins']}, Losses: {status['losses']}")
            print(f"  Win rate: {status['win_rate']:.1%}")
            print(f"  ROI: {status['roi']:+.1f}%")
            print(f"  Average edge: {status['avg_edge']:.1f}%")
            print(f"  Average predicted prob: {status['avg_prob']:.1%}")

            # Show breakdown by odds source
            if 'real_odds_roi' in status:
                print(f"\n🎯 Real Odds Results:")
                print(f"  Wins: {status['real_odds_wins']}, Losses: {status['real_odds_losses']}")
                print(f"  ROI: {status['real_odds_roi']:+.1f}%")

            if 'est_odds_roi' in status:
                print(f"\n📝 Estimated Odds Results:")
                print(f"  Wins: {status['est_odds_wins']}, Losses: {status['est_odds_losses']}")
                print(f"  ROI: {status['est_odds_roi']:+.1f}%")

        print("\n" + "-" * 70)
        print("Recent Bets:")
        print("-" * 70)

        bets = self.predictions["bets"][-15:]
        for bet in bets:
            match = f"{bet['home_team'][:15]} vs {bet['away_team'][:15]}"
            date = bet['match_date'][:10] if bet['match_date'] else 'N/A'
            prob = f"{bet['away_win_prob']:.1%}"
            odds_tag = "R" if bet.get('odds_source') == 'sportmonks' else "E"

            status_str = bet['status'].upper()
            if bet['status'] == 'settled':
                result = "WON" if bet['won'] else "LOST"
                status_str = f"{result} ({bet['actual_result']})"

            print(f"  {date} | {match:<35} | {prob:>6} | @{bet['odds']:.2f}[{odds_tag}] | {status_str}")

        print("=" * 70)
        print("Legend: [R]=Real SportMonks odds, [E]=Estimated odds")


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
    above_threshold = val_probs >= PROBABILITY_THRESHOLD
    if above_threshold.sum() > 0:
        val_accuracy = np.mean(y_val[above_threshold])  # Win rate when we bet
    else:
        val_accuracy = 0
    print(f"Validation win rate at threshold {PROBABILITY_THRESHOLD}: {val_accuracy:.1%}")
    print(f"Bets at threshold: {above_threshold.sum()}")

    return model_cal, final_features, df


def generate_predictions(tracker: AwayWinTracker, min_edge: float = MIN_EDGE):
    """Generate away win predictions for upcoming matches with real odds validation."""
    print("\n" + "=" * 70)
    print("GENERATING AWAY WIN PREDICTIONS (with Real Odds)")
    print("=" * 70)

    model, feature_cols, historical_df = train_away_win_model()
    print(f"\nModel trained on {len(historical_df)} matches")

    main_features = load_main_features()

    # Initialize SportMonks odds fetcher
    odds_fetcher = SportMonksOddsFetcher()

    # Fetch real odds from SportMonks
    print("\nFetching real odds from SportMonks...")
    real_odds_df = odds_fetcher.get_upcoming_fixtures_with_odds(days_ahead=7)
    print(f"Found {len(real_odds_df)} fixtures with real odds")

    # Create lookup for real odds
    real_odds_lookup = {}
    if not real_odds_df.empty:
        for _, row in real_odds_df.iterrows():
            key = (row['home_team'], row['away_team'])
            real_odds_lookup[key] = {
                'fixture_id': row['fixture_id'],
                'away_odds': row['away_odds_real'],
                'home_odds': row.get('home_odds_real'),
                'draw_odds': row.get('draw_odds_real'),
            }

    # Load upcoming fixtures from local data
    print("\nLoading upcoming fixtures from local data...")
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

    print(f"Upcoming matches in local data: {len(upcoming_df)}")

    new_bets = 0
    real_odds_count = 0
    est_odds_count = 0
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

        # Get estimated odds from features
        odds_estimated = feature_row['avg_away_open'].iloc[0] if 'avg_away_open' in feature_row.columns else None
        if odds_estimated is None or np.isnan(odds_estimated):
            # Estimate odds from probability (with ~5% vig)
            odds_estimated = 0.95 / away_win_prob if away_win_prob > 0 else 10.0
            odds_estimated = min(max(odds_estimated, 1.5), 10.0)

        # Try to get real odds from SportMonks
        odds_real = None
        odds_source = "estimated"

        # Try exact match
        key = (home_team, away_team)
        if key in real_odds_lookup:
            odds_real = real_odds_lookup[key]['away_odds']
            odds_source = "sportmonks"
        else:
            # Try fuzzy matching (partial names)
            for (h, a), odds_data in real_odds_lookup.items():
                if (home_team.lower() in h.lower() or h.lower() in home_team.lower()) and \
                   (away_team.lower() in a.lower() or a.lower() in away_team.lower()):
                    odds_real = odds_data['away_odds']
                    odds_source = "sportmonks"
                    break

        # Calculate edge with best available odds
        odds_for_edge = odds_real if odds_real else odds_estimated
        edge = (odds_for_edge * away_win_prob - 1) * 100

        # Check if meets criteria
        if away_win_prob >= PROBABILITY_THRESHOLD and edge > min_edge:
            tracker.add_prediction(
                fixture_id=fixture_id,
                match_date=match_date,
                home_team=home_team,
                away_team=away_team,
                league=league,
                away_win_prob=away_win_prob,
                odds=odds_estimated,
                edge=edge,
                home_elo=home_elo,
                away_elo=away_elo,
                elo_diff=elo_diff,
                odds_real=odds_real,
                odds_source=odds_source,
            )
            new_bets += 1
            if odds_real:
                real_odds_count += 1
            else:
                est_odds_count += 1

    print(f"\nAdded {new_bets} new predictions")
    print(f"  With real odds: {real_odds_count}")
    print(f"  With estimated odds: {est_odds_count}")


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


def validate_odds_accuracy(tracker: AwayWinTracker):
    """Analyze the accuracy of estimated odds vs real odds."""
    print("\n" + "=" * 70)
    print("ODDS VALIDATION ANALYSIS")
    print("=" * 70)

    bets = tracker.predictions["bets"]
    bets_with_both = [
        b for b in bets
        if b.get("odds_real") and b.get("odds_estimated")
    ]

    if not bets_with_both:
        print("No bets with both real and estimated odds found.")
        print("Run 'predict' to generate predictions with real odds.")
        return

    print(f"\nBets with both real and estimated odds: {len(bets_with_both)}")

    # Calculate differences
    differences = []
    for b in bets_with_both:
        real = b["odds_real"]
        est = b["odds_estimated"]
        diff = real - est
        pct_diff = ((real - est) / est) * 100
        differences.append({
            "match": f"{b['home_team']} vs {b['away_team']}",
            "real": real,
            "estimated": est,
            "diff": diff,
            "pct_diff": pct_diff,
            "prob": b["away_win_prob"],
            "won": b.get("won"),
        })

    # Summary stats
    diffs = [d["diff"] for d in differences]
    pct_diffs = [d["pct_diff"] for d in differences]

    print(f"\nOdds Difference Analysis:")
    print(f"  Mean difference: {np.mean(diffs):+.3f}")
    print(f"  Std difference: {np.std(diffs):.3f}")
    print(f"  Mean % difference: {np.mean(pct_diffs):+.1f}%")
    print(f"  Range: {min(diffs):+.3f} to {max(diffs):+.3f}")

    # Check if real odds are systematically higher or lower
    higher_count = sum(1 for d in diffs if d > 0)
    lower_count = sum(1 for d in diffs if d < 0)
    print(f"\n  Real odds higher: {higher_count} ({100*higher_count/len(diffs):.0f}%)")
    print(f"  Real odds lower: {lower_count} ({100*lower_count/len(diffs):.0f}%)")

    # Show edge implications
    print("\nEdge Implications:")
    for d in differences:
        edge_real = (d["real"] * d["prob"] - 1) * 100
        edge_est = (d["estimated"] * d["prob"] - 1) * 100
        edge_diff = edge_real - edge_est
        status = ""
        if d["won"] is not None:
            status = " ✓ WON" if d["won"] else " ✗ LOST"
        print(f"  {d['match'][:40]:<40} | Est edge: {edge_est:+.1f}% | Real edge: {edge_real:+.1f}% | Δ{edge_diff:+.1f}%{status}")

    # Profit comparison
    settled_with_both = [d for d in differences if d["won"] is not None]
    if settled_with_both:
        print(f"\nProfit Comparison ({len(settled_with_both)} settled):")
        profit_real = sum((d["real"] - 1) if d["won"] else -1 for d in settled_with_both)
        profit_est = sum((d["estimated"] - 1) if d["won"] else -1 for d in settled_with_both)
        roi_real = (profit_real / len(settled_with_both)) * 100
        roi_est = (profit_est / len(settled_with_both)) * 100
        print(f"  ROI with real odds: {roi_real:+.1f}%")
        print(f"  ROI with estimated odds: {roi_est:+.1f}%")
        print(f"  Difference: {roi_real - roi_est:+.1f}%")


def analyze_prediction_improvements():
    """Analyze opportunities to improve predictions."""
    print("\n" + "=" * 70)
    print("PREDICTION IMPROVEMENT ANALYSIS")
    print("=" * 70)

    # Load features and model
    model, feature_cols, df = train_away_win_model()

    # Split data for analysis
    n = len(df)
    test_df = df.iloc[int(0.8 * n):].copy()

    X_test = test_df[feature_cols].fillna(0).astype(float)
    y_test = test_df['away_win'].values

    probs = model.predict_proba(X_test)[:, 1]

    # Analyze by probability bins
    print("\nCalibration Analysis by Probability Bin:")
    print("-" * 60)

    bins = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    for i in range(len(bins) - 1):
        mask = (probs >= bins[i]) & (probs < bins[i+1])
        if mask.sum() > 0:
            actual_rate = y_test[mask].mean()
            predicted_rate = probs[mask].mean()
            count = mask.sum()
            calibration_error = actual_rate - predicted_rate
            print(f"  {bins[i]:.0%}-{bins[i+1]:.0%}: N={count:4d} | Predicted={predicted_rate:.1%} | Actual={actual_rate:.1%} | Error={calibration_error:+.1%}")

    # Feature importance
    print("\nTop 15 Feature Importances:")
    print("-" * 60)

    try:
        importances = model.calibrated_classifiers_[0].estimator.feature_importances_
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': importances
        }).sort_values('importance', ascending=False)

        for _, row in importance_df.head(15).iterrows():
            print(f"  {row['feature']:<40} | {row['importance']:.4f}")
    except Exception as e:
        print(f"  Could not extract feature importances: {e}")

    # Analyze ELO impact
    print("\nELO Differential Analysis:")
    print("-" * 60)

    if 'elo_diff' in test_df.columns:
        test_df['elo_bucket'] = pd.cut(test_df['elo_diff'], bins=[-500, -100, -50, 0, 50, 100, 500])
        for bucket, group in test_df.groupby('elo_bucket'):
            if len(group) > 10:
                away_win_rate = group['away_win'].mean()
                avg_prob = probs[group.index - test_df.index[0]].mean() if len(group) > 0 else 0
                print(f"  ELO diff {bucket}: N={len(group):4d} | Away win rate={away_win_rate:.1%} | Predicted={avg_prob:.1%}")

    # Suggestions
    print("\n💡 Improvement Suggestions:")
    print("-" * 60)
    print("  1. Add more referee-specific features (ref_fouls_avg, ref_cards_avg)")
    print("  2. Include recent form momentum (win/loss streaks)")
    print("  3. Add weather/pitch conditions if available")
    print("  4. Consider team motivation factors (title race, relegation)")
    print("  5. Add H2H specific features for rivalry matches")


def main():
    tracker = AwayWinTracker()

    if len(sys.argv) < 2:
        print("Usage: python away_win_paper_trade.py [predict|settle|status|validate|analyze]")
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

    elif command == "validate":
        validate_odds_accuracy(tracker)

    elif command == "analyze":
        analyze_prediction_improvements()

    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
