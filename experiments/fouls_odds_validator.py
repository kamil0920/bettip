#!/usr/bin/env python
"""
Fouls Odds Validator - Compare Model Predictions with Real Bookmaker Odds

This script helps validate the Fouls model by:
1. Generating predictions for upcoming matches
2. Recording real odds from bookmakers (bet365, Pinnacle, etc.)
3. Calculating expected value and tracking profitability

Usage:
    python experiments/fouls_odds_validator.py upcoming     # Show predictions needing odds
    python experiments/fouls_odds_validator.py add          # Add real odds for a match
    python experiments/fouls_odds_validator.py analyze      # Analyze odds vs predictions
    python experiments/fouls_odds_validator.py settle       # Settle completed matches
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional

from sklearn.calibration import CalibratedClassifierCV
from lightgbm import LGBMClassifier

import warnings
warnings.filterwarnings('ignore')

# Configuration
FOULS_LINE = 24.5
OUTPUT_FILE = "experiments/outputs/fouls_odds_validation.json"

# Precision by threshold (from optimization-results-11)
PRECISION_BY_THRESHOLD = {
    0.55: 0.752,
    0.60: 0.770,
    0.65: 0.865,
    0.70: 0.909,
}


class FoulsOddsValidator:
    """Track and validate fouls predictions against real odds."""

    def __init__(self):
        self.output_path = Path(OUTPUT_FILE)
        self.data = self._load_data()

    def _load_data(self) -> Dict:
        if self.output_path.exists():
            with open(self.output_path, 'r') as f:
                return json.load(f)
        return {
            "predictions": [],
            "odds_records": [],
            "settled": [],
            "summary": {},
            "version": "v1"
        }

    def _save_data(self):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, 'w') as f:
            json.dump(self.data, f, indent=2, default=str)

    def add_prediction(self, fixture_id: int, match_date: str, home_team: str,
                       away_team: str, league: str, probability: float,
                       referee: str = None):
        """Add a model prediction for a match."""
        key = f"{fixture_id}_OVER_{FOULS_LINE}"

        # Check if exists
        existing = [p for p in self.data["predictions"] if p["key"] == key]
        if existing:
            return existing[0]

        pred = {
            "key": key,
            "fixture_id": fixture_id,
            "match_date": match_date,
            "home_team": home_team,
            "away_team": away_team,
            "league": league,
            "line": FOULS_LINE,
            "probability": probability,
            "referee": referee,
            "created_at": datetime.now().isoformat(),
            "status": "pending_odds"  # pending_odds -> has_odds -> settled
        }

        self.data["predictions"].append(pred)
        self._save_data()
        return pred

    def add_real_odds(self, fixture_id: int, bookmaker: str, over_odds: float,
                      under_odds: float = None, line: float = None):
        """Record real odds from a bookmaker."""
        line = line or FOULS_LINE
        key = f"{fixture_id}_OVER_{line}"

        # Find prediction
        pred = next((p for p in self.data["predictions"] if p["key"] == key), None)
        if not pred:
            print(f"No prediction found for fixture {fixture_id}")
            return

        odds_record = {
            "key": key,
            "fixture_id": fixture_id,
            "bookmaker": bookmaker,
            "line": line,
            "over_odds": over_odds,
            "under_odds": under_odds,
            "recorded_at": datetime.now().isoformat(),
        }

        # Calculate expected value
        prob = pred["probability"]
        ev = (over_odds * prob) - 1
        odds_record["expected_value"] = ev
        odds_record["is_value_bet"] = ev > 0

        # Calculate minimum required odds for breakeven
        min_odds = 1 / prob if prob > 0 else float('inf')
        odds_record["min_odds_breakeven"] = min_odds
        odds_record["edge_pct"] = (ev * 100) if ev > 0 else 0

        self.data["odds_records"].append(odds_record)

        # Update prediction status
        pred["status"] = "has_odds"
        pred["latest_odds"] = over_odds
        pred["latest_ev"] = ev

        self._save_data()

        return odds_record

    def settle_match(self, fixture_id: int, total_fouls: int):
        """Settle a match with actual fouls result."""
        key = f"{fixture_id}_OVER_{FOULS_LINE}"

        pred = next((p for p in self.data["predictions"] if p["key"] == key), None)
        if not pred:
            print(f"No prediction found for fixture {fixture_id}")
            return

        won = total_fouls > FOULS_LINE

        # Find best odds recorded
        odds_records = [o for o in self.data["odds_records"] if o["key"] == key]
        best_odds = max([o["over_odds"] for o in odds_records]) if odds_records else None

        settled = {
            "key": key,
            "fixture_id": fixture_id,
            "home_team": pred["home_team"],
            "away_team": pred["away_team"],
            "probability": pred["probability"],
            "actual_fouls": total_fouls,
            "won": won,
            "odds_used": best_odds,
            "profit": (best_odds - 1) if won and best_odds else (-1 if best_odds else 0),
            "settled_at": datetime.now().isoformat(),
        }

        self.data["settled"].append(settled)
        pred["status"] = "settled"
        pred["actual_fouls"] = total_fouls
        pred["won"] = won

        self._save_data()
        return settled

    def get_pending_predictions(self) -> list:
        """Get predictions that need odds."""
        return [p for p in self.data["predictions"] if p["status"] == "pending_odds"]

    def get_value_bets(self) -> list:
        """Get predictions with positive expected value."""
        value_bets = []
        for pred in self.data["predictions"]:
            if pred["status"] == "has_odds" and pred.get("latest_ev", 0) > 0:
                value_bets.append(pred)
        return value_bets

    def analyze_odds_coverage(self) -> Dict:
        """Analyze how often real odds meet our requirements."""
        if not self.data["odds_records"]:
            return {"message": "No odds recorded yet"}

        records = self.data["odds_records"]

        analysis = {
            "total_odds_recorded": len(records),
            "value_bets_found": sum(1 for r in records if r["is_value_bet"]),
            "avg_over_odds": np.mean([r["over_odds"] for r in records]),
            "avg_ev": np.mean([r["expected_value"] for r in records]),
            "by_bookmaker": {},
        }

        # By bookmaker
        for bookie in set(r["bookmaker"] for r in records):
            bookie_records = [r for r in records if r["bookmaker"] == bookie]
            analysis["by_bookmaker"][bookie] = {
                "count": len(bookie_records),
                "avg_odds": np.mean([r["over_odds"] for r in bookie_records]),
                "value_bet_rate": sum(1 for r in bookie_records if r["is_value_bet"]) / len(bookie_records),
            }

        return analysis

    def get_settled_summary(self) -> Dict:
        """Get summary of settled bets."""
        settled = self.data["settled"]
        if not settled:
            return {"message": "No settled bets yet"}

        with_odds = [s for s in settled if s["odds_used"]]

        summary = {
            "total_settled": len(settled),
            "with_odds": len(with_odds),
            "wins": sum(1 for s in settled if s["won"]),
            "losses": sum(1 for s in settled if not s["won"]),
            "win_rate": sum(1 for s in settled if s["won"]) / len(settled) if settled else 0,
        }

        if with_odds:
            summary["profit"] = sum(s["profit"] for s in with_odds)
            summary["roi"] = (summary["profit"] / len(with_odds)) * 100
            summary["avg_odds"] = np.mean([s["odds_used"] for s in with_odds])

        # By probability tier
        for threshold in [0.55, 0.60, 0.65]:
            tier_bets = [s for s in settled if s["probability"] >= threshold]
            if tier_bets:
                summary[f"tier_{threshold}"] = {
                    "count": len(tier_bets),
                    "win_rate": sum(1 for s in tier_bets if s["won"]) / len(tier_bets),
                }

        return summary


def load_main_features():
    """Load features file."""
    candidates = [
        Path('data/03-features/features_with_sportmonks_odds.csv'),
        Path('data/03-features/features_all_5leagues_with_odds.csv'),
    ]
    for p in candidates:
        if p.exists():
            return pd.read_csv(p, low_memory=False)
    raise FileNotFoundError("Features file not found")


def load_fouls_data():
    """Load fouls stats from match_stats."""
    all_data = []
    for league in ['premier_league', 'la_liga', 'serie_a', 'bundesliga', 'ligue_1']:
        league_path = Path(f'data/01-raw/{league}')
        if not league_path.exists():
            continue
        for season_dir in league_path.iterdir():
            if not season_dir.is_dir():
                continue
            stats_file = season_dir / 'match_stats.parquet'
            matches_file = season_dir / 'matches.parquet'
            if stats_file.exists() and matches_file.exists():
                stats = pd.read_parquet(stats_file)
                matches = pd.read_parquet(matches_file)
                matches_slim = matches[['fixture.id', 'fixture.referee']].rename(columns={
                    'fixture.id': 'fixture_id', 'fixture.referee': 'referee'
                })
                merged = stats.merge(matches_slim, on='fixture_id', how='left')
                merged['league'] = league
                all_data.append(merged)
    if not all_data:
        return pd.DataFrame()
    return pd.concat(all_data, ignore_index=True)


def train_model():
    """Train the optimized fouls model."""
    print("Training optimized fouls model...")

    fouls_df = load_fouls_data()
    main_df = load_main_features()

    # Merge
    fouls_cols = ['fixture_id', 'home_fouls', 'away_fouls', 'referee']
    fouls_cols = [c for c in fouls_cols if c in fouls_df.columns]
    merged = main_df.merge(fouls_df[fouls_cols], on='fixture_id', how='inner', suffixes=('', '_stats'))

    # Calculate total fouls
    home_col = 'home_fouls_stats' if 'home_fouls_stats' in merged.columns else 'home_fouls'
    away_col = 'away_fouls_stats' if 'away_fouls_stats' in merged.columns else 'away_fouls'
    merged['total_fouls'] = merged[home_col] + merged[away_col]

    merged['date'] = pd.to_datetime(merged['date'])
    df = merged.sort_values('date').reset_index(drop=True)
    df['over_24_5'] = (df['total_fouls'] > FOULS_LINE).astype(int)

    # Split
    n = len(df)
    train_end = int(0.6 * n)
    val_end = int(0.8 * n)
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]

    # Features
    optimized_features = [
        'home_goals_conceded_ema', 'away_win_prob_elo', 'away_streak',
        'home_fouls_drawn_ema', 'expected_away_fouls', 'expected_total_cards',
        'corners_attack_diff', 'expected_total_fouls', 'home_shots_conceded_ema',
        'expected_home_fouls', 'away_fouls_drawn_ema', 'away_early_goal_rate',
        'home_cards_ema', 'poisson_draw_prob', 'h2h_avg_goals',
        'goal_diff_advantage', 'away_shots_ema_y', 'away_corner_intensity',
        'away_fouls_committed_ema_y', 'home_away_ppg_diff',
    ]
    feature_cols = [f for f in optimized_features if f in train_df.columns]

    X_train = train_df[feature_cols].fillna(0).astype(float)
    X_val = val_df[feature_cols].fillna(0).astype(float)
    y_train = train_df['over_24_5'].values
    y_val = val_df['over_24_5'].values

    # Train
    model = LGBMClassifier(
        max_depth=8, num_leaves=16, min_child_samples=54,
        reg_lambda=5.90, reg_alpha=1.66, learning_rate=0.007,
        subsample=0.55, colsample_bytree=0.73, n_estimators=500,
        random_state=42, verbose=-1
    )
    model.fit(X_train, y_train)

    # Calibrate
    model_cal = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
    model_cal.fit(X_val, y_val)

    print(f"Model trained on {len(train_df)} matches")
    return model_cal, feature_cols


def generate_upcoming_predictions(validator: FoulsOddsValidator):
    """Generate predictions for upcoming matches."""
    print("\n" + "=" * 70)
    print("GENERATING PREDICTIONS FOR UPCOMING MATCHES")
    print("=" * 70)

    model, feature_cols = train_model()
    main_features = load_main_features()

    # Load upcoming
    upcoming = []
    for league in ['premier_league', 'la_liga', 'serie_a', 'bundesliga', 'ligue_1']:
        matches_file = Path(f'data/01-raw/{league}/2025/matches.parquet')
        if matches_file.exists():
            df = pd.read_parquet(matches_file)
            df['league'] = league
            not_finished = df[df['fixture.status.short'] != 'FT'].copy()
            not_finished = not_finished.rename(columns={
                'fixture.id': 'fixture_id', 'fixture.date': 'date',
                'teams.home.name': 'home_team', 'teams.away.name': 'away_team',
                'fixture.referee': 'referee',
            })
            upcoming.append(not_finished)

    if not upcoming:
        print("No upcoming matches found")
        return

    upcoming_df = pd.concat(upcoming, ignore_index=True)
    upcoming_df['date'] = pd.to_datetime(upcoming_df['date']).dt.tz_localize(None)
    upcoming_df = upcoming_df.sort_values('date')

    # Next 7 days
    today = datetime.now()
    next_week = today + timedelta(days=7)
    upcoming_df = upcoming_df[(upcoming_df['date'] >= today) & (upcoming_df['date'] <= next_week)]

    print(f"\nUpcoming matches: {len(upcoming_df)}")
    print("\n" + "-" * 70)
    print(f"{'Date':<12} {'Match':<35} {'Prob':>6} {'Min Odds':>9} {'Tier':<12}")
    print("-" * 70)

    for _, row in upcoming_df.iterrows():
        fixture_id = int(row['fixture_id'])
        match_date = str(row['date'])[:10]
        home_team = row['home_team']
        away_team = row['away_team']
        league = row['league']
        referee = row.get('referee', '')

        # Get features
        match_features = main_features[
            (main_features['home_team_name'] == home_team) &
            (main_features['away_team_name'] == away_team)
        ]
        if len(match_features) == 0:
            match_features = main_features[main_features['home_team_name'] == home_team].tail(1)
        if len(match_features) == 0:
            continue

        feature_row = match_features.iloc[-1:]
        available = [f for f in feature_cols if f in feature_row.columns]
        if len(available) < 5:
            continue

        X = feature_row[available].fillna(0).astype(float)
        prob = model.predict_proba(X)[:, 1][0]

        # Determine tier
        if prob >= 0.65:
            tier = "CONSERVATIVE"
            min_odds = 1.20
        elif prob >= 0.60:
            tier = "BALANCED"
            min_odds = 1.35
        elif prob >= 0.55:
            tier = "AGGRESSIVE"
            min_odds = 1.40
        else:
            tier = "-"
            min_odds = None

        if prob >= 0.55:  # Only show potential bets
            validator.add_prediction(
                fixture_id=fixture_id,
                match_date=match_date,
                home_team=home_team,
                away_team=away_team,
                league=league,
                probability=prob,
                referee=referee
            )

            match_str = f"{home_team[:15]} v {away_team[:15]}"
            print(f"{match_date:<12} {match_str:<35} {prob:>5.1%} {min_odds:>9.2f} {tier:<12}")

    print("-" * 70)


def show_pending_odds(validator: FoulsOddsValidator):
    """Show predictions that need real odds."""
    pending = validator.get_pending_predictions()

    print("\n" + "=" * 70)
    print("PREDICTIONS NEEDING REAL ODDS")
    print("=" * 70)

    if not pending:
        print("No pending predictions. Run 'upcoming' first.")
        return

    print(f"\n{'ID':<10} {'Date':<12} {'Match':<30} {'Prob':>6} {'Min Odds':>9}")
    print("-" * 70)

    for p in sorted(pending, key=lambda x: x['match_date']):
        match = f"{p['home_team'][:12]}v{p['away_team'][:12]}"
        min_odds = 1 / p['probability'] if p['probability'] > 0 else 999
        print(f"{p['fixture_id']:<10} {p['match_date'][:10]:<12} {match:<30} {p['probability']:>5.1%} {min_odds:>9.2f}")

    print("-" * 70)
    print("\nTo add odds: python fouls_odds_validator.py add <fixture_id> <bookmaker> <over_odds>")


def add_odds_interactive(validator: FoulsOddsValidator):
    """Interactive mode to add real odds."""
    if len(sys.argv) >= 5:
        # Command line: add <fixture_id> <bookmaker> <over_odds>
        fixture_id = int(sys.argv[2])
        bookmaker = sys.argv[3]
        over_odds = float(sys.argv[4])
        under_odds = float(sys.argv[5]) if len(sys.argv) > 5 else None

        record = validator.add_real_odds(fixture_id, bookmaker, over_odds, under_odds)
        if record:
            ev_str = f"+{record['edge_pct']:.1f}%" if record['is_value_bet'] else f"{record['expected_value']*100:.1f}%"
            print(f"Added: Over {FOULS_LINE} @ {over_odds} from {bookmaker} -> EV: {ev_str}")
            if record['is_value_bet']:
                print("  ✓ VALUE BET FOUND!")
            else:
                print(f"  ✗ Need odds >= {record['min_odds_breakeven']:.2f} for value")
    else:
        print("Usage: python fouls_odds_validator.py add <fixture_id> <bookmaker> <over_odds> [under_odds]")
        print("\nExample: python fouls_odds_validator.py add 1234567 bet365 1.85 1.95")
        show_pending_odds(validator)


def analyze_odds(validator: FoulsOddsValidator):
    """Analyze recorded odds vs predictions."""
    analysis = validator.analyze_odds_coverage()

    print("\n" + "=" * 70)
    print("ODDS ANALYSIS")
    print("=" * 70)

    if "message" in analysis:
        print(analysis["message"])
        return

    print(f"\nTotal odds recorded: {analysis['total_odds_recorded']}")
    print(f"Value bets found: {analysis['value_bets_found']} ({analysis['value_bets_found']/analysis['total_odds_recorded']*100:.1f}%)")
    print(f"Average over odds: {analysis['avg_over_odds']:.2f}")
    print(f"Average EV: {analysis['avg_ev']*100:+.1f}%")

    if analysis['by_bookmaker']:
        print("\nBy Bookmaker:")
        for bookie, stats in analysis['by_bookmaker'].items():
            print(f"  {bookie}: {stats['count']} records, avg odds {stats['avg_odds']:.2f}, value rate {stats['value_bet_rate']*100:.0f}%")

    # Show settled summary
    summary = validator.get_settled_summary()
    if "message" not in summary:
        print("\n" + "-" * 70)
        print("SETTLED RESULTS")
        print("-" * 70)
        print(f"Total settled: {summary['total_settled']}")
        print(f"Win rate: {summary['win_rate']:.1%} ({summary['wins']}W / {summary['losses']}L)")
        if 'roi' in summary:
            print(f"ROI: {summary['roi']:+.1f}%")
            print(f"Profit: {summary['profit']:+.2f} units")


def settle_matches(validator: FoulsOddsValidator):
    """Settle matches from match data."""
    print("\n" + "=" * 70)
    print("SETTLING MATCHES")
    print("=" * 70)

    # Load match stats
    fouls_df = load_fouls_data()
    if fouls_df.empty:
        print("No match stats found")
        return

    # Calculate total fouls
    fouls_df['total_fouls'] = fouls_df['home_fouls'] + fouls_df['away_fouls']

    pending = [p for p in validator.data["predictions"] if p["status"] in ["pending_odds", "has_odds"]]
    settled_count = 0

    for pred in pending:
        fixture_id = pred['fixture_id']
        match = fouls_df[fouls_df['fixture_id'] == fixture_id]
        if len(match) > 0:
            total_fouls = int(match.iloc[0]['total_fouls'])
            result = validator.settle_match(fixture_id, total_fouls)
            if result:
                won_str = "WON" if result['won'] else "LOST"
                print(f"Settled {pred['home_team']} v {pred['away_team']}: {total_fouls} fouls -> {won_str}")
                settled_count += 1

    print(f"\nSettled {settled_count} matches")

    # Show summary
    summary = validator.get_settled_summary()
    if "message" not in summary:
        print(f"\nOverall: {summary['win_rate']:.1%} win rate")


def main():
    validator = FoulsOddsValidator()

    if len(sys.argv) < 2:
        print("Fouls Odds Validator - Compare predictions with real bookmaker odds")
        print("\nUsage:")
        print("  python fouls_odds_validator.py upcoming   # Generate predictions")
        print("  python fouls_odds_validator.py pending    # Show predictions needing odds")
        print("  python fouls_odds_validator.py add <id> <book> <odds>  # Add real odds")
        print("  python fouls_odds_validator.py analyze    # Analyze odds coverage")
        print("  python fouls_odds_validator.py settle     # Settle completed matches")
        return

    command = sys.argv[1].lower()

    if command == "upcoming":
        generate_upcoming_predictions(validator)
        show_pending_odds(validator)

    elif command == "pending":
        show_pending_odds(validator)

    elif command == "add":
        add_odds_interactive(validator)

    elif command == "analyze":
        analyze_odds(validator)

    elif command == "settle":
        settle_matches(validator)
        analyze_odds(validator)

    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
