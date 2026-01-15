"""Base tracker class for paper trading."""
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np


class BaseBetTracker:
    """
    Base class for tracking betting predictions with CLV analysis.

    Subclasses should define:
    - market_name: str (e.g., "corners", "shots", "fouls")
    - stat_column: str (e.g., "total_corners", "total_shots", "total_fouls")
    - ref_stat_name: str (e.g., "ref_avg_corners", "ref_avg_shots")
    """

    market_name: str = "generic"
    stat_column: str = "total_stat"
    ref_stat_name: str = "ref_avg_stat"

    def __init__(self, output_path: Optional[str] = None):
        """
        Initialize tracker.

        Args:
            output_path: Path to JSON file for storing predictions.
                        Defaults to experiments/outputs/{market}_tracking.json
        """
        if output_path is None:
            output_path = f"experiments/outputs/{self.market_name}_tracking.json"
        self.output_path = Path(output_path)
        self.predictions = self._load_data()

    def _load_data(self) -> Dict:
        """Load existing tracking data."""
        if self.output_path.exists():
            with open(self.output_path, 'r') as f:
                return json.load(f)
        return {"bets": [], "summary": {}, "version": "v1"}

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
        predicted_value: float,
        bet_type: str,
        line: float,
        our_odds: float,
        our_probability: float,
        edge: float,
        ref_avg: Optional[float] = None,
        extra_data: Optional[Dict[str, Any]] = None,
    ):
        """
        Add a new prediction.

        Args:
            fixture_id: Unique fixture identifier
            match_date: Match date string
            home_team: Home team name
            away_team: Away team name
            league: League name
            referee: Referee name
            predicted_value: Model's predicted value (e.g., total corners)
            bet_type: "OVER" or "UNDER"
            line: Betting line (e.g., 9.5)
            our_odds: Odds we're betting at
            our_probability: Our predicted probability
            edge: Expected edge percentage
            ref_avg: Referee's average for this stat
            extra_data: Additional market-specific data
        """
        key = f"{fixture_id}_{bet_type}_{line}"

        # Check if already exists
        existing = [b for b in self.predictions["bets"] if b["key"] == key]
        if existing:
            print(f"  [EXISTS] {home_team} vs {away_team} ({bet_type} {line})")
            return

        bet = {
            "key": key,
            "fixture_id": fixture_id,
            "match_date": match_date,
            "home_team": home_team,
            "away_team": away_team,
            "league": league,
            "referee": referee,
            self.ref_stat_name: ref_avg,
            f"predicted_{self.market_name}": predicted_value,
            "bet_type": bet_type,
            "line": line,
            "our_odds": our_odds,
            "our_probability": our_probability,
            "edge": edge,
            "closing_odds": None,
            "clv": None,
            f"actual_{self.market_name}": None,
            "won": None,
            "created_at": datetime.now().isoformat(),
            "status": "pending"
        }

        # Add extra data if provided
        if extra_data:
            bet.update(extra_data)

        self.predictions["bets"].append(bet)
        self._save_data()

        ref_info = f" [Ref: {referee[:15] if referee else 'None'}={ref_avg:.1f}]" if ref_avg else ""
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

    def record_result(self, fixture_id: int, actual_value: int):
        """
        Record actual result for a match.

        Args:
            fixture_id: Fixture identifier
            actual_value: Actual stat value (e.g., total corners)
        """
        updated = 0
        for bet in self.predictions["bets"]:
            if bet["fixture_id"] == fixture_id:
                bet[f"actual_{self.market_name}"] = actual_value
                if bet["bet_type"] == "OVER":
                    bet["won"] = actual_value > bet["line"]
                else:
                    bet["won"] = actual_value < bet["line"]
                bet["status"] = "settled"
                updated += 1

        if updated > 0:
            self._save_data()
            print(f"Recorded {actual_value} {self.market_name} for fixture {fixture_id} ({updated} bets)")
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

        # CLV stats
        clv_bets = [b for b in bets if b.get("clv") is not None]
        if clv_bets:
            clvs = [b["clv"] for b in clv_bets]
            summary["avg_clv"] = np.mean(clvs)
            summary["clv_positive_rate"] = sum(1 for c in clvs if c > 0) / len(clvs)

        # Results stats
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
        """Print tracking dashboard."""
        status = self.get_status()

        print("\n" + "=" * 70)
        print(f"{self.market_name.upper()} BETTING PAPER TRADE - DASHBOARD")
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
                actual_key = f"actual_{self.market_name}"
                actual = bet.get(actual_key, '?')
                result = "WON" if bet['won'] else "LOST"
                status_str = f"{result} ({actual})"

            print(f"  {date} | {match:<26} | {bet_desc:<12} | {ref:<10} | {status_str}")

        print("=" * 70)

    def get_pending_bets(self) -> List[Dict]:
        """Get list of pending bets."""
        return [b for b in self.predictions["bets"] if b["status"] in ("pending", "closed")]

    def export_predictions_csv(self, output_path: Optional[str] = None) -> str:
        """
        Export predictions to CSV.

        Args:
            output_path: Output file path. Defaults to {market}_predictions.csv

        Returns:
            Path to exported file
        """
        import pandas as pd

        if output_path is None:
            output_path = f"experiments/outputs/{self.market_name}_predictions.csv"

        bets = self.predictions["bets"]
        if not bets:
            print("No predictions to export")
            return ""

        df = pd.DataFrame(bets)
        df.to_csv(output_path, index=False)
        print(f"Exported {len(df)} predictions to {output_path}")
        return output_path
