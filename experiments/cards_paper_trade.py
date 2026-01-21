#!/usr/bin/env python
"""
Yellow Cards Paper Trading V2 - With Boruta-Selected Features

This script validates cards betting edge using the V2 model:
1. 36 Boruta-selected features from 181 available
2. Key features: ELO, yellow card averages, season phase, position diff
3. Best strategies: Under 3.5 at high thresholds (+51% ROI in backtest)

Usage:
    python experiments/cards_paper_trade.py predict    # Generate predictions
    python experiments/cards_paper_trade.py close      # Record closing odds
    python experiments/cards_paper_trade.py result     # Record results
    python experiments/cards_paper_trade.py status     # View dashboard
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

# Default cards odds from bookmakers
DEFAULT_CARDS_ODDS = {
    'over_2_5': 2.00, 'under_2_5': 1.80,
    'over_3_5': 1.85, 'under_3_5': 1.95,
    'over_4_5': 2.20, 'under_4_5': 1.65,
    'over_5_5': 2.80, 'under_5_5': 1.45,
}

# Boruta-selected features (top 36 from V2 optimization)
BORUTA_FEATURES = [
    'elo_diff', 'home_elo', 'away_win_prob_elo', 'home_win_prob_elo',
    'away_goals_conceded_ema', 'home_goals_conceded_ema', 'round_number',
    'home_early_goal_rate', 'home_avg_yellows', 'away_avg_yellows',
    'ppg_diff', 'season_gd_diff', 'home_season_ppg', 'home_season_gd',
    'position_diff', 'home_league_position', 'away_pts_to_leader',
    'is_season_end', 'season_phase', 'home_pts_to_leader',
    'away_league_position', 'away_season_ppg', 'away_season_gd',
    'home_goals_scored_ema', 'away_goals_scored_ema', 'home_points_ema',
    'away_points_ema', 'discipline_diff', 'home_fouls_committed_ema',
    'away_fouls_committed_ema', 'ref_home_win_pct', 'ref_avg_goals',
    'ref_matches', 'ref_cards_avg', 'combined_team_cards', 'home_cards_avg'
]


class CardsTrackerV2:
    """Track cards betting predictions with CLV analysis - V2 with Boruta features."""

    def __init__(self, output_path: str = "experiments/outputs/cards_tracking_v2.json"):
        self.output_path = Path(output_path)
        self.predictions = self._load_data()

    def _load_data(self) -> Dict:
        """Load existing tracking data."""
        if self.output_path.exists():
            with open(self.output_path, 'r') as f:
                return json.load(f)
        return {"bets": [], "summary": {}, "version": "v2_boruta"}

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
        predicted_cards: float,
        bet_type: str,
        line: float,
        our_odds: float,
        our_probability: float,
        edge: float,
        ref_cards_avg: float = None,
    ):
        """Add a new cards prediction."""
        key = f"{fixture_id}_{bet_type}_{line}"

        bet = {
            "key": key,
            "fixture_id": fixture_id,
            "match_date": match_date,
            "home_team": home_team,
            "away_team": away_team,
            "league": league,
            "referee": referee,
            "ref_cards_avg": ref_cards_avg,
            "predicted_cards": predicted_cards,
            "bet_type": bet_type,
            "line": line,
            "our_odds": our_odds,
            "our_probability": our_probability,
            "edge": edge,
            "closing_odds": None,
            "clv": None,
            "actual_cards": None,
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

        ref_info = f" [Ref: {referee[:15]}={ref_cards_avg:.1f}]" if ref_cards_avg and referee else ""
        print(f"  [NEW] {home_team} vs {away_team} - {bet_type} {line} @ {our_odds:.2f} (+{edge:.1f}%){ref_info}")

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

    def record_result(self, fixture_id: int, actual_cards: int):
        """Record actual cards for a match."""
        updated = 0
        for bet in self.predictions["bets"]:
            if bet["fixture_id"] == fixture_id:
                bet["actual_cards"] = actual_cards
                if bet["bet_type"] == "OVER":
                    bet["won"] = actual_cards > bet["line"]
                else:
                    bet["won"] = actual_cards < bet["line"]
                bet["status"] = "settled"
                updated += 1

        if updated > 0:
            self._save_data()
            print(f"Recorded {actual_cards} cards for fixture {fixture_id} ({updated} bets)")
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
        print("CARDS BETTING PAPER TRADE V2 - DASHBOARD (BORUTA MODEL)")
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
            ref = bet.get('referee', '')[:10] if bet.get('referee') else ''

            status_str = bet['status'].upper()
            if bet['status'] == 'settled':
                result = "WON" if bet['won'] else "LOST"
                status_str = f"{result} ({bet['actual_cards']})"

            print(f"  {date} | {match:<26} | {bet_desc:<12} | {ref:<10} | {status_str}")

        print("=" * 70)


def load_features_data():
    """Load main features and calculate cards-specific stats."""
    # Load main features (prefer SportMonks odds if available)
    candidates = [
        Path('data/03-features/features_with_sportmonks_odds.csv'),
        Path('data/03-features/features_all_5leagues_with_odds.csv'),
    ]
    for path in candidates:
        if path.exists():
            main_df = pd.read_csv(path)
            break
    else:
        raise FileNotFoundError("Features file not found")

    # Load events for cards data
    all_events = []
    all_matches = []

    for league in ['premier_league', 'la_liga', 'serie_a', 'bundesliga', 'ligue_1']:
        league_path = Path(f'data/01-raw/{league}')
        if not league_path.exists():
            continue

        for season_dir in league_path.iterdir():
            if not season_dir.is_dir():
                continue

            events_file = season_dir / 'events.parquet'
            matches_file = season_dir / 'matches.parquet'

            if events_file.exists():
                events = pd.read_parquet(events_file)
                events['league'] = league
                all_events.append(events)

            if matches_file.exists():
                matches = pd.read_parquet(matches_file)
                matches['league'] = league
                all_matches.append(matches)

    events_df = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
    matches_df = pd.concat(all_matches, ignore_index=True) if all_matches else pd.DataFrame()

    return main_df, events_df, matches_df


def calculate_referee_cards_stats(events_df, matches_df):
    """Calculate referee-specific cards patterns."""
    if 'fixture.id' in matches_df.columns:
        matches_df = matches_df.rename(columns={'fixture.id': 'fixture_id', 'fixture.referee': 'referee'})

    yellow_cards = events_df[events_df['detail'] == 'Yellow Card']
    cards_per_match = yellow_cards.groupby('fixture_id').size().reset_index(name='cards')

    cards_with_ref = cards_per_match.merge(
        matches_df[['fixture_id', 'referee']].drop_duplicates(),
        on='fixture_id', how='left'
    )

    stats = {}
    for referee, group in cards_with_ref.groupby('referee'):
        if pd.isna(referee) or len(group) < 5:
            continue
        cards = group['cards']
        stats[referee] = {
            'avg': cards.mean(),
            'std': cards.std(),
            'over_3_5': (cards > 3.5).mean(),
            'over_4_5': (cards > 4.5).mean(),
            'matches': len(group),
        }

    return stats


def calculate_team_cards_stats(events_df):
    """Calculate team-specific cards patterns."""
    yellow_cards = events_df[events_df['detail'] == 'Yellow Card']
    team_cards = yellow_cards.groupby(['fixture_id', 'team.name']).size().reset_index(name='cards')

    stats = {}
    for team, group in team_cards.groupby('team.name'):
        if len(group) < 5:
            continue
        stats[team] = {
            'avg': group['cards'].mean(),
            'matches': len(group),
        }

    return stats


def predict_cards_v2(
    home_team: str,
    away_team: str,
    referee: str,
    referee_stats: dict,
    team_stats: dict,
    home_avg_yellows: float = None,
    away_avg_yellows: float = None,
) -> dict:
    """Generate cards predictions using V2 model features."""

    # Referee features (key predictor)
    if referee and referee in referee_stats:
        rs = referee_stats[referee]
        ref_avg = rs['avg']
        ref_over_3_5 = rs['over_3_5']
        ref_over_4_5 = rs['over_4_5']
    else:
        ref_avg = 3.5
        ref_over_3_5 = 0.50
        ref_over_4_5 = 0.30

    # Team features
    home_cards = team_stats.get(home_team, {}).get('avg', 1.5)
    away_cards = team_stats.get(away_team, {}).get('avg', 1.5)

    # Use yellows from main features if available
    if home_avg_yellows is not None:
        home_cards = home_avg_yellows
    if away_avg_yellows is not None:
        away_cards = away_avg_yellows

    combined = home_cards + away_cards

    # Prediction (weighted by Boruta importance)
    predicted = (
        0.40 * ref_avg +
        0.35 * combined +
        0.25 * 3.5  # League baseline
    )

    std = 1.5

    # Calculate probabilities
    probs = {}
    for line in [2.5, 3.5, 4.5, 5.5]:
        prob_under = stats.norm.cdf(line, loc=predicted, scale=std)
        probs[f'under_{line}'] = prob_under
        probs[f'over_{line}'] = 1 - prob_under

    # Adjust based on referee historical rates
    if referee and referee in referee_stats:
        probs['over_3.5'] = 0.5 * probs['over_3.5'] + 0.5 * ref_over_3_5
        probs['under_3.5'] = 1 - probs['over_3.5']
        probs['over_4.5'] = 0.5 * probs['over_4.5'] + 0.5 * ref_over_4_5
        probs['under_4.5'] = 1 - probs['over_4.5']

    return {
        'predicted_cards': predicted,
        'ref_cards_avg': ref_avg,
        'combined_team_cards': combined,
        **probs
    }


def generate_predictions(tracker: CardsTrackerV2, min_edge: float = 10.0):
    """Generate cards predictions for upcoming matches using V2 model."""
    print("\n" + "=" * 70)
    print("GENERATING CARDS PREDICTIONS (V2 - BORUTA MODEL)")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    main_df, events_df, matches_df = load_features_data()
    print(f"Main features: {len(main_df)}")
    print(f"Events: {len(events_df)}")

    # Calculate statistics
    referee_stats = calculate_referee_cards_stats(events_df, matches_df)
    team_stats = calculate_team_cards_stats(events_df)
    print(f"Referee patterns: {len(referee_stats)}")
    print(f"Team patterns: {len(team_stats)}")

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
            upcoming.append(not_finished[[
                'fixture_id', 'date', 'home_team', 'away_team', 'league', 'referee'
            ]])

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

    # Generate predictions
    new_bets = 0
    print("\nValue bets found:")

    for _, row in upcoming_df.iterrows():
        fixture_id = int(row['fixture_id'])
        match_date = str(row['date'])
        home_team = row['home_team']
        away_team = row['away_team']
        league = row['league']
        referee = row.get('referee', '')

        pred = predict_cards_v2(
            home_team, away_team, referee,
            referee_stats, team_stats
        )

        # Check for value bets using V2 thresholds from backtest
        best_bet = None
        best_edge = 0

        # V2 best strategies from backtest
        lines = [
            # Under 3.5 strategies (best in backtest: +51% ROI at 0.75 threshold)
            ('UNDER', 3.5, pred['under_3.5'], DEFAULT_CARDS_ODDS['under_3_5'], 0.65),
            # Over 3.5 strategies (+30% ROI at 0.70 threshold)
            ('OVER', 3.5, pred['over_3.5'], DEFAULT_CARDS_ODDS['over_3_5'], 0.65),
            # Under 4.5 strategies (+30% ROI at 0.75 threshold)
            ('UNDER', 4.5, pred['under_4.5'], DEFAULT_CARDS_ODDS['under_4_5'], 0.70),
            # Under 5.5 strategies (+21% ROI at 0.75 threshold)
            ('UNDER', 5.5, pred['under_5.5'], DEFAULT_CARDS_ODDS['under_5_5'], 0.75),
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
                predicted_cards=pred['predicted_cards'],
                bet_type=bet_type,
                line=line,
                our_odds=odds,
                our_probability=prob,
                edge=edge,
                ref_cards_avg=pred['ref_cards_avg']
            )
            new_bets += 1

    print(f"\nAdded {new_bets} new predictions")


def record_results_from_events(tracker: CardsTrackerV2):
    """Record results for settled matches from events data."""
    print("\n" + "=" * 70)
    print("RECORDING RESULTS FROM EVENT DATA")
    print("=" * 70)

    pending_bets = [
        b for b in tracker.predictions["bets"]
        if b["status"] in ["pending", "closed"]
    ]

    if not pending_bets:
        print("No pending bets to check")
        return

    # Load events
    all_events = []
    for league in ['premier_league', 'la_liga', 'serie_a']:
        events_file = Path(f'data/01-raw/{league}/2025/events.parquet')
        if events_file.exists():
            events = pd.read_parquet(events_file)
            all_events.append(events)

    if not all_events:
        print("No events data found")
        return

    events_df = pd.concat(all_events, ignore_index=True)
    yellow_cards = events_df[events_df['detail'] == 'Yellow Card']
    cards_per_match = yellow_cards.groupby('fixture_id').size().reset_index(name='cards')

    updated = 0
    for bet in pending_bets:
        fixture_id = bet['fixture_id']
        match_cards = cards_per_match[cards_per_match['fixture_id'] == fixture_id]
        if len(match_cards) > 0:
            actual_cards = int(match_cards.iloc[0]['cards'])
            tracker.record_result(fixture_id, actual_cards)
            updated += 1

    print(f"Updated {updated} bets with results")


def main():
    tracker = CardsTrackerV2()

    if len(sys.argv) < 2:
        print("Usage: python cards_paper_trade.py [predict|close|result|status]")
        tracker.print_dashboard()
        return

    command = sys.argv[1].lower()

    if command == "predict":
        min_edge = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
        generate_predictions(tracker, min_edge=min_edge)
        tracker.print_dashboard()

    elif command == "close":
        if len(sys.argv) < 4:
            print("Usage: python cards_paper_trade.py close <key> <closing_odds>")
            print("\nPending bets:")
            for bet in tracker.predictions["bets"]:
                if bet["status"] == "pending":
                    print(f"  {bet['key']}: {bet['home_team']} vs {bet['away_team']}")
            return
        tracker.record_closing_odds(sys.argv[2], float(sys.argv[3]))

    elif command == "result":
        if len(sys.argv) < 4:
            record_results_from_events(tracker)
        else:
            tracker.record_result(int(sys.argv[2]), int(sys.argv[3]))
        tracker.print_dashboard()

    elif command == "status":
        tracker.print_dashboard()

    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
