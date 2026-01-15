"""
Closing Line Value (CLV) Tracker

CLV is the gold standard metric for validating betting edge.
If you consistently beat the closing line, you have real edge.
If not, your profits are likely due to luck.

CLV Formula (odds-based, bettor-friendly):
    CLV = (your_odds / closing_odds) - 1

Example:
    - You bet at odds 2.50
    - Line closes at 2.20 (odds dropped = market agrees with your bet)
    - CLV = (2.50 / 2.20) - 1 = +13.6% (GOOD - you got better odds than closing)

    - You bet at odds 2.50
    - Line closes at 2.80 (odds rose = market moved against you)
    - CLV = (2.50 / 2.80) - 1 = -10.7% (BAD - you got worse odds than closing)

Usage:
    tracker = CLVTracker()

    # Record a prediction
    tracker.record_prediction(
        match_id="12345",
        bet_type="away_win",
        our_odds=2.50,
        our_probability=0.45,
        prediction_time="2026-01-13T10:00:00"
    )

    # Later, record closing odds (fetched just before kickoff)
    tracker.record_closing_odds(
        match_id="12345",
        closing_odds=2.30
    )

    # After match, record result
    tracker.record_result(
        match_id="12345",
        won=True
    )

    # Analyze CLV performance
    tracker.get_clv_summary()
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class CLVTracker:
    """
    Track Closing Line Value to validate betting edge.

    The key insight: If you consistently beat the closing line,
    you have real predictive edge. The market is efficient enough
    that closing lines are the best estimate of true probability.
    """

    def __init__(
        self,
        output_dir: str = "experiments/outputs/clv_tracking",
        history_file: str = "clv_history.json"
    ):
        """
        Initialize CLV tracker.

        Args:
            output_dir: Directory to store CLV tracking data
            history_file: Filename for history storage
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history_file = self.output_dir / history_file

        self.predictions: Dict[str, Dict] = {}
        self.load_history()

    def load_history(self) -> None:
        """Load existing CLV history from file."""
        if self.history_file.exists():
            with open(self.history_file, 'r') as f:
                data = json.load(f)
                self.predictions = data.get('predictions', {})
            logger.info(f"Loaded {len(self.predictions)} predictions from history")
        else:
            self.predictions = {}

    def save_history(self) -> None:
        """Save CLV history to file."""
        data = {
            'predictions': self.predictions,
            'last_updated': datetime.now().isoformat(),
            'total_predictions': len(self.predictions)
        }
        with open(self.history_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def record_prediction(
        self,
        match_id: str,
        home_team: str,
        away_team: str,
        match_date: str,
        league: str,
        bet_type: str,
        our_probability: float,
        our_odds: float,
        market_odds: float,
        prediction_time: Optional[str] = None
    ) -> Dict:
        """
        Record a new prediction with odds at prediction time.

        Args:
            match_id: Unique match identifier
            home_team: Home team name
            away_team: Away team name
            match_date: Match date/time
            league: League name
            bet_type: Type of bet (away_win, btts, etc.)
            our_probability: Our model's probability
            our_odds: Odds we would bet at (current market)
            market_odds: Market odds at prediction time
            prediction_time: When prediction was made

        Returns:
            Prediction record
        """
        key = f"{match_id}_{bet_type}"

        prediction = {
            'match_id': match_id,
            'home_team': home_team,
            'away_team': away_team,
            'match_date': match_date,
            'league': league,
            'bet_type': bet_type,
            'our_probability': our_probability,
            'our_odds': our_odds,
            'market_odds_at_prediction': market_odds,
            'implied_prob_at_prediction': 1 / market_odds if market_odds > 0 else 0,
            'prediction_time': prediction_time or datetime.now().isoformat(),
            'closing_odds': None,
            'implied_prob_at_close': None,
            'clv_odds': None,
            'clv_probability': None,
            'result': None,
            'won': None,
            'profit': None,
            'status': 'pending_close'
        }

        self.predictions[key] = prediction
        self.save_history()

        logger.info(f"Recorded prediction: {home_team} vs {away_team} - {bet_type}")
        return prediction

    def record_closing_odds(
        self,
        match_id: str,
        bet_type: str,
        closing_odds: float
    ) -> Optional[Dict]:
        """
        Record closing odds (odds just before match starts).

        Args:
            match_id: Match identifier
            bet_type: Type of bet
            closing_odds: Odds at market close

        Returns:
            Updated prediction with CLV calculated
        """
        key = f"{match_id}_{bet_type}"

        if key not in self.predictions:
            logger.warning(f"No prediction found for {key}")
            return None

        pred = self.predictions[key]

        if closing_odds <= 1:
            logger.warning(f"Invalid closing odds: {closing_odds}")
            return None

        pred['closing_odds'] = closing_odds
        pred['implied_prob_at_close'] = 1 / closing_odds

        # Calculate CLV (bettor-friendly formula)
        # Positive CLV = you got better odds than closing (good!)
        # CLV = (your_odds / closing_odds) - 1
        if pred['our_odds'] > 0 and closing_odds > 0:
            pred['clv_odds'] = (pred['our_odds'] / closing_odds) - 1
        else:
            pred['clv_odds'] = 0

        # CLV (probability-based) - comparing our model prob to closing implied
        close_implied = 1 / closing_odds
        pred['clv_probability'] = pred['our_probability'] - close_implied

        pred['status'] = 'pending_result'

        self.save_history()

        clv_pct = pred['clv_odds'] * 100 if pred['clv_odds'] else 0
        logger.info(f"CLV for {key}: {clv_pct:+.2f}%")

        return pred

    def record_result(
        self,
        match_id: str,
        bet_type: str,
        won: bool,
        stake: float = 1.0
    ) -> Optional[Dict]:
        """
        Record the actual result of a bet.

        Args:
            match_id: Match identifier
            bet_type: Type of bet
            won: Whether the bet won
            stake: Stake amount (default 1 unit)

        Returns:
            Updated prediction with result
        """
        key = f"{match_id}_{bet_type}"

        if key not in self.predictions:
            logger.warning(f"No prediction found for {key}")
            return None

        pred = self.predictions[key]
        pred['won'] = won
        pred['result'] = 'won' if won else 'lost'

        if won:
            pred['profit'] = stake * (pred['our_odds'] - 1)
        else:
            pred['profit'] = -stake

        pred['status'] = 'settled'
        pred['settled_at'] = datetime.now().isoformat()

        self.save_history()

        return pred

    def get_clv_summary(self) -> Dict:
        """
        Get summary statistics of CLV performance.

        This is the key metric. If avg_clv > 0 over 100+ bets,
        you likely have real edge.
        """
        predictions_with_clv = [
            p for p in self.predictions.values()
            if p.get('clv_odds') is not None
        ]

        settled = [p for p in predictions_with_clv if p.get('status') == 'settled']

        if not predictions_with_clv:
            return {
                'total_predictions': len(self.predictions),
                'with_closing_odds': 0,
                'settled': 0,
                'message': 'No CLV data yet. Record closing odds to calculate CLV.'
            }

        clv_values = [p['clv_odds'] for p in predictions_with_clv]

        summary = {
            'total_predictions': len(self.predictions),
            'with_closing_odds': len(predictions_with_clv),
            'settled': len(settled),

            # CLV Statistics (THIS IS THE KEY METRIC)
            'avg_clv': np.mean(clv_values) * 100,  # in percentage
            'median_clv': np.median(clv_values) * 100,
            'std_clv': np.std(clv_values) * 100,
            'positive_clv_rate': sum(1 for c in clv_values if c > 0) / len(clv_values) * 100,

            # CLV by confidence
            'clv_when_positive': np.mean([c for c in clv_values if c > 0]) * 100 if any(c > 0 for c in clv_values) else 0,
            'clv_when_negative': np.mean([c for c in clv_values if c < 0]) * 100 if any(c < 0 for c in clv_values) else 0,
        }

        # If we have settled bets, add P&L correlation
        if settled:
            wins = sum(1 for p in settled if p.get('won'))
            total_profit = sum(p.get('profit', 0) for p in settled)
            total_staked = len(settled)

            # CLV vs actual results correlation
            clv_for_settled = [p['clv_odds'] for p in settled]
            won_for_settled = [1 if p.get('won') else 0 for p in settled]

            summary.update({
                'win_rate': wins / len(settled) * 100,
                'roi': total_profit / total_staked * 100,
                'total_profit': total_profit,
                'total_bets': len(settled),

                # This correlation tells you if CLV predicts actual results
                'clv_win_correlation': np.corrcoef(clv_for_settled, won_for_settled)[0, 1] if len(settled) > 1 else 0
            })

        return summary

    def get_clv_by_bet_type(self) -> pd.DataFrame:
        """Get CLV breakdown by bet type."""
        predictions_with_clv = [
            p for p in self.predictions.values()
            if p.get('clv_odds') is not None
        ]

        if not predictions_with_clv:
            return pd.DataFrame()

        df = pd.DataFrame(predictions_with_clv)

        summary = df.groupby('bet_type').agg({
            'clv_odds': ['mean', 'std', 'count'],
            'won': 'mean'
        }).round(4)

        summary.columns = ['avg_clv', 'std_clv', 'n_bets', 'win_rate']
        summary['avg_clv'] = summary['avg_clv'] * 100
        summary['std_clv'] = summary['std_clv'] * 100
        summary['win_rate'] = summary['win_rate'] * 100

        return summary.sort_values('avg_clv', ascending=False)

    def get_clv_by_league(self) -> pd.DataFrame:
        """Get CLV breakdown by league."""
        predictions_with_clv = [
            p for p in self.predictions.values()
            if p.get('clv_odds') is not None
        ]

        if not predictions_with_clv:
            return pd.DataFrame()

        df = pd.DataFrame(predictions_with_clv)

        summary = df.groupby('league').agg({
            'clv_odds': ['mean', 'std', 'count'],
            'won': 'mean'
        }).round(4)

        summary.columns = ['avg_clv', 'std_clv', 'n_bets', 'win_rate']
        summary['avg_clv'] = summary['avg_clv'] * 100
        summary['std_clv'] = summary['std_clv'] * 100
        summary['win_rate'] = summary['win_rate'] * 100

        return summary.sort_values('avg_clv', ascending=False)

    def print_clv_report(self) -> None:
        """Print a formatted CLV report."""
        summary = self.get_clv_summary()

        print("\n" + "=" * 60)
        print("CLOSING LINE VALUE (CLV) REPORT")
        print("=" * 60)

        if summary.get('with_closing_odds', 0) == 0:
            print("\nNo CLV data available yet.")
            print("Record closing odds for your predictions to calculate CLV.")
            return

        print(f"\nPredictions tracked: {summary['total_predictions']}")
        print(f"With closing odds:   {summary['with_closing_odds']}")
        print(f"Settled:             {summary['settled']}")

        print("\n" + "-" * 60)
        print("CLV STATISTICS (Key Metric)")
        print("-" * 60)

        avg_clv = summary['avg_clv']
        print(f"Average CLV:         {avg_clv:+.2f}%")
        print(f"Median CLV:          {summary['median_clv']:+.2f}%")
        print(f"CLV Std Dev:         {summary['std_clv']:.2f}%")
        print(f"Positive CLV Rate:   {summary['positive_clv_rate']:.1f}%")

        # Interpretation
        print("\n" + "-" * 60)
        print("INTERPRETATION")
        print("-" * 60)

        if avg_clv > 2:
            print("EXCELLENT: You're consistently beating the closing line.")
            print("           This suggests real predictive edge.")
        elif avg_clv > 0:
            print("PROMISING: Slight positive CLV, but need more data.")
            print("           Continue tracking for 100+ bets.")
        elif avg_clv > -2:
            print("NEUTRAL: CLV is around zero.")
            print("         Your model may not have edge over the market.")
        else:
            print("WARNING: Negative CLV suggests you're getting worse odds")
            print("         than closing. The market is beating your model.")

        if summary.get('settled', 0) > 0:
            print("\n" + "-" * 60)
            print("ACTUAL RESULTS")
            print("-" * 60)
            print(f"Win Rate:            {summary['win_rate']:.1f}%")
            print(f"ROI:                 {summary['roi']:+.2f}%")
            print(f"Total Profit:        {summary['total_profit']:+.2f} units")
            print(f"CLV-Win Correlation: {summary['clv_win_correlation']:.3f}")

        # By bet type
        by_type = self.get_clv_by_bet_type()
        if not by_type.empty:
            print("\n" + "-" * 60)
            print("CLV BY BET TYPE")
            print("-" * 60)
            print(by_type.to_string())

        print("\n" + "=" * 60)

    def backfill_from_historical_odds(
        self,
        predictions_df: pd.DataFrame,
        historical_odds_df: pd.DataFrame
    ) -> int:
        """
        Backfill CLV data using historical closing odds.

        This allows you to calculate CLV for past predictions
        using football-data.co.uk closing odds.

        Args:
            predictions_df: DataFrame with predictions (needs match_date, home_team, away_team, bet_type, our_odds)
            historical_odds_df: DataFrame from FootballDataLoader with closing odds

        Returns:
            Number of predictions updated
        """
        updated = 0

        # Normalize dates
        predictions_df['match_date'] = pd.to_datetime(predictions_df['match_date']).dt.date
        historical_odds_df['date'] = pd.to_datetime(historical_odds_df['date']).dt.date

        for _, pred in predictions_df.iterrows():
            # Find matching historical match
            match = historical_odds_df[
                (historical_odds_df['date'] == pred['match_date']) &
                (historical_odds_df['home_team'].str.contains(pred['home_team'].split()[0], case=False, na=False) |
                 historical_odds_df['away_team'].str.contains(pred['away_team'].split()[0], case=False, na=False))
            ]

            if match.empty:
                continue

            match = match.iloc[0]
            bet_type = pred.get('bet_type', 'away_win')

            # Get closing odds based on bet type
            closing_odds = None
            if bet_type == 'away_win' and 'avg_away_close' in match:
                closing_odds = match['avg_away_close']
            elif bet_type == 'home_win' and 'avg_home_close' in match:
                closing_odds = match['avg_home_close']
            elif bet_type == 'btts' and 'btts_yes_close' in match:
                closing_odds = match.get('btts_yes_close')

            if closing_odds and closing_odds > 1:
                key = f"{pred.get('match_id', '')}_{bet_type}"
                if key in self.predictions:
                    self.record_closing_odds(
                        match_id=pred.get('match_id', ''),
                        bet_type=bet_type,
                        closing_odds=closing_odds
                    )
                    updated += 1

        logger.info(f"Backfilled CLV for {updated} predictions")
        return updated

    def export_to_csv(self, filepath: Optional[str] = None) -> str:
        """Export CLV history to CSV for analysis."""
        if not filepath:
            filepath = self.output_dir / "clv_history.csv"

        df = pd.DataFrame(list(self.predictions.values()))
        df.to_csv(filepath, index=False)

        logger.info(f"Exported CLV history to {filepath}")
        return str(filepath)


def calculate_clv_from_odds(bet_odds: float, closing_odds: float) -> float:
    """
    Calculate CLV from odds (bettor-friendly formula).

    Positive CLV = you got BETTER odds than closing (good!)
    Negative CLV = you got WORSE odds than closing (bad)

    Args:
        bet_odds: Odds you bet at
        closing_odds: Final odds before match

    Returns:
        CLV as a decimal (0.05 = 5% positive CLV)

    Examples:
        bet=2.50, close=2.20 → CLV = +13.6% (odds dropped, you got value)
        bet=2.50, close=2.80 → CLV = -10.7% (odds rose, you missed value)
    """
    if bet_odds <= 1 or closing_odds <= 1:
        return 0.0

    # Bettor-friendly: positive when your odds > closing odds
    return (bet_odds / closing_odds) - 1


def calculate_expected_clv_roi(avg_clv: float, n_bets: int = 1000) -> Dict:
    """
    Calculate expected long-term ROI based on CLV.

    The relationship between CLV and ROI is:
    Expected ROI ≈ CLV * (1 - vig_rate)

    Args:
        avg_clv: Average CLV (as decimal, e.g., 0.03 for 3%)
        n_bets: Number of bets for projection

    Returns:
        Expected ROI projections
    """
    # Typical vig is 3-5%
    vig_rate = 0.04

    expected_roi = avg_clv * (1 - vig_rate)

    return {
        'avg_clv_pct': avg_clv * 100,
        'expected_roi_pct': expected_roi * 100,
        'projected_profit_per_100_units': expected_roi * 100,
        'confidence_interval': f"{(expected_roi - 0.02) * 100:.1f}% to {(expected_roi + 0.02) * 100:.1f}%",
        'note': 'Based on CLV-ROI correlation. Actual results may vary.'
    }
