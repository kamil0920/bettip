"""
Recommendation Generator

Generates betting recommendations in stable CSV format.
All recommendations follow the schema defined in data/05-recommendations/README.md
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Stable CSV columns - DO NOT CHANGE ORDER
RECOMMENDATION_COLUMNS = [
    'rec_id',           # Unique recommendation ID (R0001, R0002, ...)
    'created_at',       # When generated
    'fixture_id',       # Match ID
    'start_time',       # Match start time
    'home_team',        # Home team name
    'away_team',        # Away team name
    'market',           # CORNERS, FOULS, SHOTS, HOME_WIN, BTTS, OVER_2.5, UNDER_2.5
    'side',             # OVER, UNDER, YES, NO, HOME, AWAY
    'line',             # Betting line (null for 1X2/BTTS)
    'expected',         # Our expected value
    'our_prob',         # Our probability (0-1)
    'odds',             # Market odds (if available)
    'market_prob',      # Implied probability from odds
    'edge_pct',         # Our edge percentage
    'confidence',       # HIGH, MEDIUM, LOW
    'status',           # PENDING, WON, LOST, PUSH, VOID
    'actual_value',     # Actual result (filled after match)
    'stake',            # Amount staked
    'pnl',              # Profit/loss
    'settled_at',       # When settled
]

# Valid market types
MARKETS = ['CORNERS', 'FOULS', 'SHOTS', 'HOME_WIN', 'AWAY_WIN', 'BTTS', 'OVER_2.5', 'UNDER_2.5']

# Valid sides per market
MARKET_SIDES = {
    'CORNERS': ['OVER', 'UNDER'],
    'FOULS': ['OVER', 'UNDER'],
    'SHOTS': ['OVER', 'UNDER'],
    'HOME_WIN': ['HOME'],
    'AWAY_WIN': ['AWAY'],
    'BTTS': ['YES', 'NO'],
    'OVER_2.5': ['OVER'],
    'UNDER_2.5': ['UNDER'],
}


@dataclass
class Recommendation:
    """Single betting recommendation."""
    fixture_id: int
    start_time: str
    home_team: str
    away_team: str
    market: str
    side: str
    our_prob: float
    confidence: str
    line: Optional[float] = None
    expected: Optional[float] = None
    odds: Optional[float] = None

    def __post_init__(self):
        if self.market not in MARKETS:
            raise ValueError(f"Invalid market: {self.market}. Must be one of {MARKETS}")
        if self.side not in MARKET_SIDES.get(self.market, []):
            raise ValueError(f"Invalid side '{self.side}' for market '{self.market}'")
        if self.our_prob < 0 or self.our_prob > 1:
            raise ValueError(f"our_prob must be between 0 and 1, got {self.our_prob}")
        if self.confidence not in ['HIGH', 'MEDIUM', 'LOW']:
            raise ValueError(f"confidence must be HIGH, MEDIUM, or LOW")


class RecommendationGenerator:
    """
    Generates and saves betting recommendations in stable CSV format.

    Usage:
        gen = RecommendationGenerator()

        # Add recommendations
        gen.add(Recommendation(
            fixture_id=123,
            start_time="2026-01-17 15:00:00",
            home_team="Liverpool",
            away_team="Chelsea",
            market="CORNERS",
            side="OVER",
            line=10.5,
            expected=11.8,
            our_prob=0.65,
            odds=2.10,
            confidence="HIGH"
        ))

        # Save to file
        gen.save()  # Creates rec_YYYYMMDD_NNN.csv
    """

    def __init__(self, output_dir: str = "data/05-recommendations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.recommendations: List[Recommendation] = []
        self.created_at = datetime.now().isoformat()

    def add(self, rec: Recommendation) -> None:
        """Add a recommendation."""
        self.recommendations.append(rec)

    def add_many(self, recs: List[Recommendation]) -> None:
        """Add multiple recommendations."""
        self.recommendations.extend(recs)

    def _get_next_filename(self) -> Path:
        """Get next available filename for today."""
        today = datetime.now().strftime("%Y%m%d")

        # Find existing files for today
        existing = list(self.output_dir.glob(f"rec_{today}_*.csv"))

        if not existing:
            seq = 1
        else:
            # Extract sequence numbers and get max
            seqs = []
            for f in existing:
                try:
                    seq_str = f.stem.split('_')[-1]
                    seqs.append(int(seq_str))
                except (ValueError, IndexError):
                    pass
            seq = max(seqs) + 1 if seqs else 1

        return self.output_dir / f"rec_{today}_{seq:03d}.csv"

    def to_dataframe(self) -> pd.DataFrame:
        """Convert recommendations to DataFrame."""
        rows = []

        for i, rec in enumerate(self.recommendations, 1):
            # Calculate market probability if odds available
            market_prob = 1 / rec.odds if rec.odds else None

            # Calculate edge if we have odds
            edge_pct = None
            if rec.odds:
                edge_pct = (rec.our_prob - market_prob) * 100

            row = {
                'rec_id': f"R{i:04d}",
                'created_at': self.created_at,
                'fixture_id': rec.fixture_id,
                'start_time': rec.start_time,
                'home_team': rec.home_team,
                'away_team': rec.away_team,
                'market': rec.market,
                'side': rec.side,
                'line': rec.line,
                'expected': rec.expected,
                'our_prob': rec.our_prob,
                'odds': rec.odds,
                'market_prob': market_prob,
                'edge_pct': edge_pct,
                'confidence': rec.confidence,
                'status': 'PENDING',
                'actual_value': None,
                'stake': None,
                'pnl': None,
                'settled_at': None,
            }
            rows.append(row)

        return pd.DataFrame(rows, columns=RECOMMENDATION_COLUMNS)

    def save(self, filename: Optional[str] = None) -> Path:
        """
        Save recommendations to CSV.

        Args:
            filename: Optional custom filename. If None, auto-generates.

        Returns:
            Path to saved file.
        """
        if not self.recommendations:
            raise ValueError("No recommendations to save")

        df = self.to_dataframe()

        if filename:
            filepath = self.output_dir / filename
        else:
            filepath = self._get_next_filename()

        df.to_csv(filepath, index=False)
        logger.info(f"Saved {len(df)} recommendations to {filepath}")

        return filepath

    def clear(self) -> None:
        """Clear all recommendations."""
        self.recommendations = []
        self.created_at = datetime.now().isoformat()


def load_recommendations(filepath: str) -> pd.DataFrame:
    """Load recommendations from CSV file."""
    df = pd.read_csv(filepath)

    # Validate columns
    missing = set(RECOMMENDATION_COLUMNS) - set(df.columns)
    if missing:
        logger.warning(f"Missing columns in {filepath}: {missing}")

    return df


def update_result(
    filepath: str,
    rec_id: str,
    status: str,
    actual_value: Optional[float] = None,
    stake: Optional[float] = None,
    pnl: Optional[float] = None,
) -> None:
    """
    Update a recommendation with its result.

    Args:
        filepath: Path to CSV file
        rec_id: Recommendation ID (e.g., 'R0001')
        status: WON, LOST, PUSH, or VOID
        actual_value: Actual corners/goals/etc
        stake: Amount staked
        pnl: Profit/loss
    """
    if status not in ['WON', 'LOST', 'PUSH', 'VOID']:
        raise ValueError(f"Invalid status: {status}")

    df = pd.read_csv(filepath)

    mask = df['rec_id'] == rec_id
    if not mask.any():
        raise ValueError(f"Recommendation {rec_id} not found in {filepath}")

    df.loc[mask, 'status'] = status
    df.loc[mask, 'actual_value'] = actual_value
    df.loc[mask, 'stake'] = stake
    df.loc[mask, 'pnl'] = pnl
    df.loc[mask, 'settled_at'] = datetime.now().isoformat()

    df.to_csv(filepath, index=False)
    logger.info(f"Updated {rec_id} in {filepath}: {status}")


def calculate_performance(filepath: str) -> Dict[str, Any]:
    """
    Calculate performance metrics from recommendations file.

    Returns:
        Dict with ROI, win_rate, total_pnl, etc.
    """
    df = pd.read_csv(filepath)

    # Filter settled bets
    settled = df[df['status'].isin(['WON', 'LOST', 'PUSH'])]

    if settled.empty:
        return {
            'total_bets': len(df),
            'settled_bets': 0,
            'pending_bets': len(df[df['status'] == 'PENDING']),
            'message': 'No settled bets yet'
        }

    total_stake = settled['stake'].sum()
    total_pnl = settled['pnl'].sum()

    return {
        'total_bets': len(df),
        'settled_bets': len(settled),
        'pending_bets': len(df[df['status'] == 'PENDING']),
        'wins': len(settled[settled['status'] == 'WON']),
        'losses': len(settled[settled['status'] == 'LOST']),
        'pushes': len(settled[settled['status'] == 'PUSH']),
        'win_rate': (settled['status'] == 'WON').mean(),
        'total_stake': total_stake,
        'total_pnl': total_pnl,
        'roi': (total_pnl / total_stake * 100) if total_stake > 0 else 0,
        'avg_odds': settled['odds'].mean(),
        'avg_edge': settled['edge_pct'].mean(),
    }
