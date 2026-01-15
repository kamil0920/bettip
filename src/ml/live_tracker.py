"""
Live Paper Trading Dashboard with CLV Tracking

This module provides real-time tracking of betting predictions with
Closing Line Value (CLV) analysis - the gold standard for validating edge.

Daily Workflow:
1. Morning: Run predictions for upcoming matches
2. Before kickoff: Fetch closing odds (automatic or manual)
3. After matches: Record results
4. Analyze: Review CLV performance

Usage:
    tracker = LiveTracker()

    # Record today's predictions
    tracker.add_predictions_from_file("experiments/outputs/next_round_predictions.json")

    # Fetch closing odds (run ~30min before first match)
    tracker.fetch_closing_odds()

    # Record results after matches
    tracker.update_results()

    # View dashboard
    tracker.show_dashboard()
"""
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from src.ml.clv_tracker import CLVTracker, calculate_clv_from_odds
from src.odds.football_data_loader import FootballDataLoader, normalize_team_name

logger = logging.getLogger(__name__)


class LiveTracker:
    """
    Live paper trading tracker with CLV analysis.

    Tracks predictions, closing odds, and results to validate
    whether your model has real edge over the market.
    """

    def __init__(
        self,
        output_dir: str = "experiments/outputs/live_tracking",
        initial_bankroll: float = 1000.0,
        kelly_fraction: float = 0.25  # Quarter Kelly for safety
    ):
        """
        Initialize live tracker.

        Args:
            output_dir: Directory for tracking data
            initial_bankroll: Starting bankroll for paper trading
            kelly_fraction: Fraction of Kelly criterion to use
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.predictions_file = self.output_dir / "live_predictions.json"
        self.results_file = self.output_dir / "results_history.json"
        self.daily_log_dir = self.output_dir / "daily_logs"
        self.daily_log_dir.mkdir(exist_ok=True)

        self.initial_bankroll = initial_bankroll
        self.kelly_fraction = kelly_fraction

        self.clv_tracker = CLVTracker(
            output_dir=str(self.output_dir),
            history_file="clv_tracking.json"
        )

        self.predictions: Dict[str, Dict] = {}
        self.results_history: List[Dict] = []

        self._load_state()

    def _load_state(self) -> None:
        """Load existing state from files."""
        if self.predictions_file.exists():
            with open(self.predictions_file, 'r') as f:
                self.predictions = json.load(f)

        if self.results_file.exists():
            with open(self.results_file, 'r') as f:
                data = json.load(f)
                self.results_history = data.get('history', [])

    def _save_state(self) -> None:
        """Save current state to files."""
        with open(self.predictions_file, 'w') as f:
            json.dump(self.predictions, f, indent=2, default=str)

        with open(self.results_file, 'w') as f:
            json.dump({
                'history': self.results_history,
                'last_updated': datetime.now().isoformat()
            }, f, indent=2, default=str)

    def add_prediction(
        self,
        match_id: str,
        home_team: str,
        away_team: str,
        match_date: str,
        league: str,
        bet_type: str,
        our_probability: float,
        market_odds: float,
        threshold: float,
        meets_threshold: bool,
        edge: float = 0.0
    ) -> Dict:
        """
        Add a single prediction to track.

        Args:
            match_id: Unique identifier (can be fixture_id or generated)
            home_team: Home team name
            away_team: Away team name
            match_date: Match datetime string
            league: League name
            bet_type: Type of bet (away_win, btts, home_win, etc.)
            our_probability: Model's predicted probability
            market_odds: Current market odds
            threshold: Probability threshold for betting
            meets_threshold: Whether this qualifies as a bet
            edge: Calculated edge over market

        Returns:
            Prediction record
        """
        key = f"{home_team}_{away_team}_{bet_type}_{match_date[:10]}"

        prediction = {
            'key': key,
            'match_id': match_id,
            'home_team': home_team,
            'away_team': away_team,
            'match_date': match_date,
            'league': league,
            'bet_type': bet_type,
            'our_probability': our_probability,
            'market_odds_at_prediction': market_odds,
            'implied_prob_at_prediction': 1 / market_odds if market_odds > 1 else 0,
            'threshold': threshold,
            'meets_threshold': meets_threshold,
            'edge': edge,
            'prediction_time': datetime.now().isoformat(),

            # To be filled later
            'closing_odds': None,
            'implied_prob_at_close': None,
            'clv': None,
            'result': None,
            'won': None,
            'profit': None,
            'status': 'pending'
        }

        # Calculate recommended stake using Kelly
        if meets_threshold and market_odds > 1:
            kelly_stake = self._calculate_kelly_stake(our_probability, market_odds)
            prediction['recommended_stake'] = kelly_stake
            prediction['recommended_amount'] = kelly_stake * self.initial_bankroll
        else:
            prediction['recommended_stake'] = 0
            prediction['recommended_amount'] = 0

        self.predictions[key] = prediction
        self._save_state()

        # Also add to CLV tracker
        self.clv_tracker.record_prediction(
            match_id=key,
            home_team=home_team,
            away_team=away_team,
            match_date=match_date,
            league=league,
            bet_type=bet_type,
            our_probability=our_probability,
            our_odds=market_odds,
            market_odds=market_odds
        )

        return prediction

    def _calculate_kelly_stake(self, prob: float, odds: float) -> float:
        """
        Calculate Kelly criterion stake.

        Kelly formula: f = (p * (odds - 1) - (1 - p)) / (odds - 1)
        We use fractional Kelly for safety.
        """
        if odds <= 1 or prob <= 0 or prob >= 1:
            return 0

        q = 1 - prob
        b = odds - 1

        kelly = (prob * b - q) / b

        # Apply fraction and cap at 5%
        stake = max(0, min(kelly * self.kelly_fraction, 0.05))

        return stake

    def add_predictions_from_file(
        self,
        filepath: str = "experiments/outputs/next_round_predictions.json",
        only_threshold_met: bool = False
    ) -> int:
        """
        Load predictions from the prediction output file.

        Args:
            filepath: Path to predictions JSON file
            only_threshold_met: If True, only add predictions that meet threshold

        Returns:
            Number of predictions added
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        predictions = data.get('predictions', [])
        added = 0

        for pred in predictions:
            meets_threshold = pred.get('meets_threshold', 'False') == 'True'

            if only_threshold_met and not meets_threshold:
                continue

            # Generate a unique match ID
            match = pred.get('match', '')
            parts = match.split(' vs ')
            home_team = parts[0] if len(parts) > 1 else match
            away_team = parts[1] if len(parts) > 1 else ''

            self.add_prediction(
                match_id=f"{home_team}_{away_team}",
                home_team=home_team,
                away_team=away_team,
                match_date=pred.get('date', ''),
                league=pred.get('league', ''),
                bet_type=pred.get('bet_type', ''),
                our_probability=pred.get('probability', pred.get('our_prob', 0)),
                market_odds=pred.get('market_odds', pred.get('odds', 2.0)),
                threshold=pred.get('threshold', 0.5),
                meets_threshold=meets_threshold,
                edge=pred.get('edge', 0)
            )
            added += 1

        print(f"Added {added} predictions from {filepath}")
        return added

    def record_closing_odds(self, key: str, closing_odds: float) -> Optional[Dict]:
        """
        Record closing odds for a prediction.

        Args:
            key: Prediction key
            closing_odds: Odds at market close (before kickoff)

        Returns:
            Updated prediction with CLV
        """
        if key not in self.predictions:
            print(f"Warning: Prediction {key} not found")
            return None

        pred = self.predictions[key]

        if closing_odds <= 1:
            print(f"Warning: Invalid closing odds {closing_odds}")
            return None

        pred['closing_odds'] = closing_odds
        pred['implied_prob_at_close'] = 1 / closing_odds

        # Calculate CLV
        clv = calculate_clv_from_odds(
            pred['market_odds_at_prediction'],
            closing_odds
        )
        pred['clv'] = clv
        pred['status'] = 'has_closing'

        self._save_state()

        # Update CLV tracker
        self.clv_tracker.record_closing_odds(
            match_id=key,
            bet_type=pred['bet_type'],
            closing_odds=closing_odds
        )

        clv_pct = clv * 100
        symbol = "+" if clv > 0 else ""
        print(f"CLV for {pred['home_team']} vs {pred['away_team']} ({pred['bet_type']}): {symbol}{clv_pct:.2f}%")

        return pred

    def record_result(
        self,
        key: str,
        won: bool,
        home_goals: Optional[int] = None,
        away_goals: Optional[int] = None
    ) -> Optional[Dict]:
        """
        Record the actual result of a match.

        Args:
            key: Prediction key
            won: Whether the bet won
            home_goals: Home team goals (optional)
            away_goals: Away team goals (optional)

        Returns:
            Updated prediction
        """
        if key not in self.predictions:
            print(f"Warning: Prediction {key} not found")
            return None

        pred = self.predictions[key]
        pred['won'] = won
        pred['result'] = 'won' if won else 'lost'
        pred['home_goals'] = home_goals
        pred['away_goals'] = away_goals
        pred['settled_at'] = datetime.now().isoformat()
        pred['status'] = 'settled'

        # Calculate profit/loss
        if pred['meets_threshold']:
            stake = pred.get('recommended_amount', 0)
            if won:
                pred['profit'] = stake * (pred['market_odds_at_prediction'] - 1)
            else:
                pred['profit'] = -stake
        else:
            pred['profit'] = 0  # No bet was placed

        self._save_state()

        # Add to history
        self.results_history.append({
            'date': pred['match_date'],
            'match': f"{pred['home_team']} vs {pred['away_team']}",
            'bet_type': pred['bet_type'],
            'probability': pred['our_probability'],
            'odds': pred['market_odds_at_prediction'],
            'closing_odds': pred.get('closing_odds'),
            'clv': pred.get('clv'),
            'bet_placed': pred['meets_threshold'],
            'won': won,
            'profit': pred['profit']
        })
        self._save_state()

        # Update CLV tracker
        self.clv_tracker.record_result(
            match_id=key,
            bet_type=pred['bet_type'],
            won=won
        )

        symbol = "+" if pred['profit'] > 0 else ""
        print(f"Result: {pred['home_team']} vs {pred['away_team']} - {'WON' if won else 'LOST'} ({symbol}${pred['profit']:.2f})")

        return pred

    def get_pending_predictions(self) -> pd.DataFrame:
        """Get all pending predictions."""
        pending = [
            p for p in self.predictions.values()
            if p['status'] == 'pending'
        ]
        if not pending:
            return pd.DataFrame()

        df = pd.DataFrame(pending)
        cols = ['home_team', 'away_team', 'match_date', 'league', 'bet_type',
                'our_probability', 'market_odds_at_prediction', 'meets_threshold',
                'recommended_amount']
        return df[[c for c in cols if c in df.columns]].sort_values('match_date')

    def get_todays_predictions(self) -> pd.DataFrame:
        """Get predictions for today's matches."""
        today = datetime.now().date()
        todays = [
            p for p in self.predictions.values()
            if pd.to_datetime(p['match_date']).date() == today
        ]
        if not todays:
            return pd.DataFrame()

        return pd.DataFrame(todays)

    def get_summary_stats(self) -> Dict:
        """Get summary statistics."""
        all_preds = list(self.predictions.values())
        settled = [p for p in all_preds if p['status'] == 'settled']
        with_clv = [p for p in all_preds if p.get('clv') is not None]
        bets_placed = [p for p in settled if p.get('meets_threshold')]

        stats = {
            'total_predictions': len(all_preds),
            'pending': sum(1 for p in all_preds if p['status'] == 'pending'),
            'with_closing_odds': len(with_clv),
            'settled': len(settled),
            'bets_placed': len(bets_placed),
        }

        # CLV stats
        if with_clv:
            clvs = [p['clv'] for p in with_clv if p['clv'] is not None]
            if clvs:
                stats['avg_clv'] = np.mean(clvs) * 100
                stats['median_clv'] = np.median(clvs) * 100
                stats['positive_clv_rate'] = sum(1 for c in clvs if c > 0) / len(clvs) * 100

        # P&L stats for actual bets
        if bets_placed:
            wins = sum(1 for p in bets_placed if p.get('won'))
            total_profit = sum(p.get('profit', 0) for p in bets_placed)
            total_staked = sum(p.get('recommended_amount', 0) for p in bets_placed)

            stats['wins'] = wins
            stats['losses'] = len(bets_placed) - wins
            stats['win_rate'] = wins / len(bets_placed) * 100 if bets_placed else 0
            stats['total_profit'] = total_profit
            stats['total_staked'] = total_staked
            stats['roi'] = (total_profit / total_staked * 100) if total_staked > 0 else 0
            stats['current_bankroll'] = self.initial_bankroll + total_profit

        return stats

    def show_dashboard(self) -> None:
        """Display the live tracking dashboard."""
        stats = self.get_summary_stats()

        print("\n" + "=" * 70)
        print("LIVE PAPER TRADING DASHBOARD")
        print("=" * 70)

        print(f"\nInitial Bankroll: ${self.initial_bankroll:,.2f}")
        if 'current_bankroll' in stats:
            change = stats['current_bankroll'] - self.initial_bankroll
            pct = (change / self.initial_bankroll) * 100
            print(f"Current Bankroll: ${stats['current_bankroll']:,.2f} ({pct:+.1f}%)")

        print("\n" + "-" * 70)
        print("PREDICTION TRACKING")
        print("-" * 70)
        print(f"Total predictions:     {stats['total_predictions']}")
        print(f"Pending:               {stats['pending']}")
        print(f"With closing odds:     {stats['with_closing_odds']}")
        print(f"Settled:               {stats['settled']}")
        print(f"Actual bets placed:    {stats['bets_placed']}")

        if 'avg_clv' in stats:
            print("\n" + "-" * 70)
            print("CLV PERFORMANCE (Key Metric)")
            print("-" * 70)
            print(f"Average CLV:           {stats['avg_clv']:+.2f}%")
            print(f"Median CLV:            {stats['median_clv']:+.2f}%")
            print(f"Positive CLV rate:     {stats['positive_clv_rate']:.1f}%")

            # Interpretation
            if stats['avg_clv'] > 2:
                print("\n>>> EXCELLENT: Consistently beating closing line!")
            elif stats['avg_clv'] > 0:
                print("\n>>> PROMISING: Slight positive CLV, continue tracking")
            else:
                print("\n>>> WARNING: Negative CLV, market is beating your model")

        if 'wins' in stats:
            print("\n" + "-" * 70)
            print("BET RESULTS")
            print("-" * 70)
            print(f"Wins:                  {stats['wins']}")
            print(f"Losses:                {stats['losses']}")
            print(f"Win rate:              {stats['win_rate']:.1f}%")
            print(f"Total staked:          ${stats['total_staked']:,.2f}")
            print(f"Total profit:          ${stats['total_profit']:+,.2f}")
            print(f"ROI:                   {stats['roi']:+.2f}%")

        # Today's action items
        today = datetime.now().date()
        todays = [p for p in self.predictions.values()
                  if pd.to_datetime(p['match_date']).date() == today]

        if todays:
            print("\n" + "-" * 70)
            print(f"TODAY'S MATCHES ({len(todays)})")
            print("-" * 70)
            for p in sorted(todays, key=lambda x: x['match_date']):
                status = p['status']
                threshold_marker = "*" if p['meets_threshold'] else " "
                print(f" {threshold_marker} {p['home_team']} vs {p['away_team']}")
                print(f"   {p['bet_type']} | Prob: {p['our_probability']:.1%} | Odds: {p['market_odds_at_prediction']:.2f} | Status: {status}")

        print("\n" + "=" * 70)
        print("* = Bet recommended (meets threshold)")
        print("=" * 70)

    def export_daily_log(self) -> str:
        """Export today's activity to a daily log file."""
        today = datetime.now().strftime("%Y-%m-%d")
        log_file = self.daily_log_dir / f"log_{today}.json"

        todays_preds = {
            k: v for k, v in self.predictions.items()
            if pd.to_datetime(v['match_date']).date() == datetime.now().date()
        }

        log_data = {
            'date': today,
            'predictions': todays_preds,
            'summary': self.get_summary_stats(),
            'exported_at': datetime.now().isoformat()
        }

        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2, default=str)

        print(f"Daily log exported to: {log_file}")
        return str(log_file)


def main():
    """CLI for live tracking."""
    import argparse

    parser = argparse.ArgumentParser(description="Live Paper Trading with CLV Tracking")
    parser.add_argument('command', choices=['dashboard', 'add', 'result', 'export'],
                        help="Command to run")
    parser.add_argument('--file', default="experiments/outputs/next_round_predictions.json",
                        help="Predictions file to load")
    parser.add_argument('--key', help="Prediction key for result recording")
    parser.add_argument('--won', type=bool, help="Whether bet won")

    args = parser.parse_args()

    tracker = LiveTracker()

    if args.command == 'dashboard':
        tracker.show_dashboard()

    elif args.command == 'add':
        tracker.add_predictions_from_file(args.file)
        tracker.show_dashboard()

    elif args.command == 'result':
        if not args.key:
            print("Error: --key required for result command")
            return
        tracker.record_result(args.key, args.won)
        tracker.show_dashboard()

    elif args.command == 'export':
        tracker.export_daily_log()


if __name__ == "__main__":
    main()
