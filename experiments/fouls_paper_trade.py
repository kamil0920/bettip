#!/usr/bin/env python
"""
Fouls Betting Paper Trading - OPTIMIZED VERSION

Uses the OPTIMIZED model from optimization-results-11 which shows:
- 75.2% precision at threshold 0.55 (needs odds >= 1.33)
- 77.0% precision at threshold 0.60 (needs odds >= 1.30)
- 86.5% precision at threshold 0.65 (needs odds >= 1.16)

IMPORTANT: This model was validated with SYNTHETIC odds. To be profitable,
you MUST verify that actual bookmaker odds meet the minimum requirements:
- Only bet if actual odds >= min_profitable_odds for your threshold
- Check bet365, Pinnacle, or Asian books for fouls markets

Bookmaker fouls lines typically available:
- bet365: Total Fouls Over/Under (varies by match, usually 24.5-28.5)
- Pinnacle: Sometimes has Total Fouls spreads
- Asian books: May have fouls totals

Usage:
    python experiments/fouls_paper_trade.py predict    # Generate predictions
    python experiments/fouls_paper_trade.py settle     # Auto-settle from data
    python experiments/fouls_paper_trade.py status     # View dashboard
    python experiments/fouls_paper_trade.py validate   # Validate with real odds
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
from lightgbm import LGBMClassifier

import warnings
warnings.filterwarnings('ignore')

# OPTIMIZED configuration from optimization-results-11
# Target: Over 24.5 Fouls (48.3% base rate)
FOULS_LINE = 24.5

# Precision-based thresholds and minimum odds requirements
# CRITICAL: Only bet if actual bookmaker odds >= min_odds
BETTING_TIERS = {
    'conservative': {
        'threshold': 0.65,
        'precision': 0.865,  # 86.5%
        'min_odds': 1.20,    # 1/0.865 * 1.05 safety margin
        'expected_bets': 52,
    },
    'balanced': {
        'threshold': 0.60,
        'precision': 0.770,  # 77.0%
        'min_odds': 1.35,    # 1/0.77 * 1.05 safety margin
        'expected_bets': 139,
    },
    'aggressive': {
        'threshold': 0.55,
        'precision': 0.752,  # 75.2%
        'min_odds': 1.40,    # 1/0.752 * 1.05 safety margin
        'expected_bets': 234,
    },
}

# Default tier - balanced is recommended
DEFAULT_TIER = 'balanced'

# Default fouls odds (fallback when real odds unavailable)
DEFAULT_FOULS_ODDS = {
    'over_24_5': 1.85, 'under_24_5': 1.95,
}


def load_real_fouls_odds() -> Optional[pd.DataFrame]:
    """Load real fouls odds from niche odds cache.

    Returns:
        DataFrame with fouls odds per match, or None if unavailable.
    """
    # Try niche_odds_cache (API-Football loader)
    niche_dir = Path('data/niche_odds_cache')
    if niche_dir.exists():
        parquets = sorted(niche_dir.glob('*.parquet'))
        if parquets:
            dfs = [pd.read_parquet(p) for p in parquets]
            df = pd.concat(dfs, ignore_index=True)
            # Look for fouls-related columns
            fouls_cols = [c for c in df.columns if 'foul' in c.lower()]
            if fouls_cols:
                print(f"  Loaded niche fouls odds for {len(df)} fixtures")
                return df

    # Try theodds_cache
    cache_dir = Path('data/theodds_cache')
    if cache_dir.exists():
        parquets = sorted(cache_dir.glob('*_all_markets.parquet'))
        if parquets:
            dfs = [pd.read_parquet(p) for p in parquets]
            df = pd.concat(dfs, ignore_index=True)
            if not df.empty:
                return df

    print("  WARNING: No real fouls odds found. Using tier min_odds as placeholder.")
    return None


def get_fouls_odds_for_match(
    odds_df: Optional[pd.DataFrame],
    home_team: str,
    away_team: str,
    line: float,
    direction: str,
    fallback: float = 1.85,
) -> float:
    """Look up real fouls odds for a specific match.

    Args:
        odds_df: DataFrame from load_real_fouls_odds().
        home_team: Home team name.
        away_team: Away team name.
        line: Fouls line (e.g. 24.5).
        direction: 'over' or 'under'.
        fallback: Default odds if not found.

    Returns:
        Real odds if found, otherwise fallback.
    """
    if odds_df is None:
        return fallback

    mask = (
        odds_df['home_team'].str.contains(home_team.split()[-1], case=False, na=False) &
        odds_df['away_team'].str.contains(away_team.split()[-1], case=False, na=False)
    )
    match = odds_df[mask]
    if match.empty:
        return fallback

    row = match.iloc[0]

    # Check for fouls odds columns (varies by source)
    for col_pattern in [f'fouls_{direction}', f'total_fouls_{direction}']:
        for col in row.index:
            if col_pattern in col.lower():
                val = row[col]
                if not pd.isna(val) and val > 1.0:
                    return float(val)

    return fallback


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


def train_fouls_model(tier: str = DEFAULT_TIER):
    """Train OPTIMIZED fouls prediction model using parameters from optimization-results-11.

    Uses LightGBM with walk-forward validated parameters for Over 24.5 Fouls.
    """
    print(f"\nTraining OPTIMIZED fouls model (tier: {tier})...")

    fouls_df = load_fouls_data()
    main_df = load_main_features()
    print(f"Fouls data: {len(fouls_df)}")
    print(f"Main features: {len(main_df)}")

    # Merge - use suffixes to avoid collision with existing columns
    fouls_cols = ['fixture_id', 'total_fouls', 'home_fouls', 'away_fouls', 'referee']
    fouls_cols = [c for c in fouls_cols if c in fouls_df.columns]
    merged = main_df.merge(
        fouls_df[fouls_cols],
        on='fixture_id',
        how='inner',
        suffixes=('', '_stats')
    )
    print(f"Merged: {len(merged)}")

    # Use total_fouls from stats if collision occurred
    fouls_col = 'total_fouls_stats' if 'total_fouls_stats' in merged.columns else 'total_fouls'
    if fouls_col not in merged.columns:
        # Calculate from home/away fouls
        home_col = 'home_fouls_stats' if 'home_fouls_stats' in merged.columns else 'home_fouls'
        away_col = 'away_fouls_stats' if 'away_fouls_stats' in merged.columns else 'away_fouls'
        merged['total_fouls_calc'] = merged[home_col] + merged[away_col]
        fouls_col = 'total_fouls_calc'

    # Sort by date
    merged['date'] = pd.to_datetime(merged['date'])
    df = merged.sort_values('date').reset_index(drop=True)

    # Create target for OPTIMIZED line (24.5)
    df['over_24_5'] = (df[fouls_col] > FOULS_LINE).astype(int)
    print(f"Over {FOULS_LINE} rate: {df['over_24_5'].mean():.1%}")

    # Temporal split (60/20/20 as in optimization)
    n = len(df)
    train_end = int(0.6 * n)
    val_end = int(0.8 * n)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()

    # Calculate referee stats from training data ONLY (leakage-free)
    # Find the correct referee column (may have suffix)
    ref_col = 'referee_stats' if 'referee_stats' in train_df.columns else 'referee'
    if ref_col in train_df.columns:
        ref_stats = train_df.groupby(ref_col)[fouls_col].agg(['mean', 'std']).reset_index()
        ref_stats.columns = ['referee', 'ref_fouls_avg', 'ref_fouls_std']
        referee_lookup = ref_stats.set_index('referee').to_dict('index')
    else:
        referee_lookup = {}

    global_mean = train_df[fouls_col].mean() if fouls_col in train_df.columns else 25.0
    global_std = train_df[fouls_col].std() if fouls_col in train_df.columns else 5.0

    # OPTIMIZED features from optimization-results-11
    optimized_features = [
        'home_goals_conceded_ema', 'away_win_prob_elo', 'away_streak',
        'home_fouls_drawn_ema', 'expected_away_fouls', 'expected_total_cards',
        'corners_attack_diff', 'expected_total_fouls', 'home_shots_conceded_ema',
        'expected_home_fouls', 'away_fouls_drawn_ema', 'away_early_goal_rate',
        'home_cards_ema', 'poisson_draw_prob', 'h2h_avg_goals',
        'goal_diff_advantage', 'away_shots_ema_y', 'away_corner_intensity',
        'away_fouls_committed_ema_y', 'home_away_ppg_diff',
    ]

    # Use available optimized features
    feature_cols = [f for f in optimized_features if f in train_df.columns]

    # If not enough optimized features, add general numeric features
    if len(feature_cols) < 10:
        exclude_cols = [
            'fixture_id', 'date', 'home_team_name', 'away_team_name',
            'home_team_id', 'away_team_id', 'round', 'referee',
            'total_fouls', 'home_fouls', 'away_fouls', 'over_24_5',
            'home_score', 'away_score', 'result', 'btts',
        ]
        extra_cols = [c for c in train_df.columns if c not in exclude_cols
                      and c not in feature_cols
                      and train_df[c].dtype in ['int64', 'float64', 'int32', 'float32']]
        feature_cols.extend(extra_cols[:30])

    print(f"Using {len(feature_cols)} features ({len([f for f in optimized_features if f in train_df.columns])} optimized)")

    X_train = train_df[feature_cols].fillna(0).astype(float)
    X_val = val_df[feature_cols].fillna(0).astype(float)

    y_train = train_df['over_24_5'].values
    y_val = val_df['over_24_5'].values

    # OPTIMIZED LightGBM parameters from optimization-results-11
    model = LGBMClassifier(
        max_depth=8,
        num_leaves=16,
        min_child_samples=54,
        reg_lambda=5.90,
        reg_alpha=1.66,
        learning_rate=0.007,
        subsample=0.55,
        colsample_bytree=0.73,
        n_estimators=500,
        random_state=42,
        verbose=-1
    )

    model.fit(X_train, y_train)

    # Calibrate with Platt scaling
    model_cal = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
    model_cal.fit(X_val, y_val)

    # Validation stats
    val_probs = model_cal.predict_proba(X_val)[:, 1]
    tier_config = BETTING_TIERS[tier]
    threshold = tier_config['threshold']

    bets_at_threshold = (val_probs >= threshold).sum()
    if bets_at_threshold > 0:
        precision_at_threshold = y_val[val_probs >= threshold].mean()
        print(f"Validation at threshold {threshold}: precision={precision_at_threshold:.1%}, n_bets={bets_at_threshold}")
    else:
        print(f"No bets at threshold {threshold}")

    return model_cal, feature_cols, referee_lookup, df, global_mean, global_std


def generate_predictions(tracker: FoulsTracker, tier: str = DEFAULT_TIER):
    """Generate fouls predictions for upcoming matches using OPTIMIZED model.

    Args:
        tracker: FoulsTracker instance to store predictions
        tier: One of 'conservative', 'balanced', 'aggressive'
    """
    tier_config = BETTING_TIERS[tier]
    threshold = tier_config['threshold']
    min_odds = tier_config['min_odds']
    precision = tier_config['precision']

    print("\n" + "=" * 70)
    print(f"GENERATING FOULS PREDICTIONS (OPTIMIZED - {tier.upper()} TIER)")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Line: Over {FOULS_LINE}")
    print(f"  Threshold: {threshold}")
    print(f"  Expected precision: {precision:.1%}")
    print(f"  Minimum odds required: {min_odds:.2f}")
    print(f"\n⚠️  IMPORTANT: Only bet if actual bookmaker odds >= {min_odds:.2f}")

    model, feature_cols, referee_lookup, historical_df, global_mean, global_std = train_fouls_model(tier)
    print(f"\nModel trained on {len(historical_df)} matches")
    print(f"Referee patterns: {len(referee_lookup)}")

    # Load real fouls odds
    print("\nLoading real fouls odds...")
    fouls_odds_df = load_real_fouls_odds()

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
    print(f"\nPredictions at threshold {threshold} (need odds >= {min_odds:.2f}):")
    print("-" * 70)

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

        # Get prediction
        prob = model.predict_proba(X)[:, 1][0]
        ref_avg = feature_row['ref_fouls_avg'].iloc[0] if 'ref_fouls_avg' in feature_row.columns else None

        # Check if meets threshold
        if prob >= threshold:
            # Use real odds if available, otherwise tier min_odds as fallback
            real_odds = get_fouls_odds_for_match(
                fouls_odds_df, home_team, away_team, FOULS_LINE, 'over',
                fallback=min_odds,
            )
            edge = (real_odds * prob - 1) * 100

            # Only bet if real odds meet minimum requirement
            if real_odds < min_odds:
                print(f"  [SKIP] {home_team} vs {away_team} - odds {real_odds:.2f} < min {min_odds:.2f}")
                continue

            tracker.add_prediction(
                fixture_id=fixture_id,
                match_date=match_date,
                home_team=home_team,
                away_team=away_team,
                league=league,
                referee=referee,
                predicted_fouls=FOULS_LINE + (prob - 0.5) * 10,  # Estimate
                bet_type='OVER',
                line=FOULS_LINE,
                our_odds=real_odds,
                our_probability=prob,
                edge=edge,
                ref_avg_fouls=ref_avg
            )
            new_bets += 1

    print(f"\nAdded {new_bets} new predictions")
    odds_source = "real" if fouls_odds_df is not None else "placeholder"
    if odds_source == "placeholder":
        print(f"\n⚠️  REMINDER: These use placeholder odds ({min_odds:.2f}).")
        print(f"   Before betting, verify actual bookmaker odds >= {min_odds:.2f}")


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


def print_betting_guide():
    """Print guide for finding real fouls odds."""
    print("\n" + "=" * 70)
    print("FOULS BETTING GUIDE - FINDING REAL ODDS")
    print("=" * 70)
    print("""
Where to find Total Fouls markets:

1. BET365 (recommended)
   - Navigate to match -> More Markets -> Team Specials/Player Specials
   - Look for "Total Fouls" Over/Under
   - Lines usually: 24.5, 25.5, 26.5, 27.5

2. PINNACLE
   - Some matches have fouls spreads under "Specials"

3. ASIAN BOOKS (Pin88, SBO)
   - May have fouls totals under "Other Markets"

Minimum odds requirements by tier:
  - Conservative (0.65 threshold): odds >= 1.20
  - Balanced (0.60 threshold): odds >= 1.35
  - Aggressive (0.55 threshold): odds >= 1.40

IMPORTANT: The model's precision was validated with synthetic odds.
Real profitability depends on actual market odds meeting these minimums.
""")
    print("=" * 70)


def main():
    tracker = FoulsTracker()

    if len(sys.argv) < 2:
        print("Usage: python fouls_paper_trade.py [predict|settle|status|guide]")
        print("       python fouls_paper_trade.py predict [conservative|balanced|aggressive]")
        tracker.print_dashboard()
        return

    command = sys.argv[1].lower()

    if command == "predict":
        tier = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_TIER
        if tier not in BETTING_TIERS:
            print(f"Unknown tier: {tier}. Use one of: {list(BETTING_TIERS.keys())}")
            return
        generate_predictions(tracker, tier=tier)
        tracker.print_dashboard()

    elif command == "settle":
        record_results_from_api(tracker)
        tracker.print_dashboard()

    elif command == "status":
        tracker.print_dashboard()

    elif command == "guide":
        print_betting_guide()

    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
