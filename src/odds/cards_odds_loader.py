"""
Cards/Bookings Totals Odds Loader

Uses The Odds API for real-time cards over/under odds.
Falls back to estimated odds for historical matches.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.odds.base_odds_loader import TotalsOddsLoader

logger = logging.getLogger(__name__)

COMMON_CARD_LINES = [3.5, 4.5, 5.5, 6.5]
DEFAULT_CARD_LINE = 4.5
DEFAULT_OVER_ODDS = 1.85
DEFAULT_UNDER_ODDS = 1.95


class CardsOddsLoader(TotalsOddsLoader):
    """Load cards totals odds from The Odds API or use estimated odds."""

    @property
    def _market_name(self) -> str:
        return "cards"

    @property
    def _api_market_key(self) -> str:
        return "alternate_totals_cards"

    @property
    def _default_over_odds(self) -> float:
        return DEFAULT_OVER_ODDS

    @property
    def _default_under_odds(self) -> float:
        return DEFAULT_UNDER_ODDS

    @property
    def _over_col(self) -> str:
        return "cards_over_odds"

    @property
    def _under_col(self) -> str:
        return "cards_under_odds"

    @property
    def _common_lines(self) -> List[float]:
        return COMMON_CARD_LINES

    @property
    def _default_line(self) -> float:
        return DEFAULT_CARD_LINE

    @property
    def _col_prefix(self) -> str:
        return "cards"

    def _apply_estimation_adjustments(
        self, df: pd.DataFrame, target_line: Optional[float] = None
    ) -> pd.DataFrame:
        """Adjust cards odds based on team yellow card averages."""
        if "home_avg_yellows" in df.columns and "away_avg_yellows" in df.columns:
            cards_factor = (df["home_avg_yellows"] + df["away_avg_yellows"]) / 4
            cards_factor = cards_factor.clip(0.8, 1.2)

            df["cards_over_odds"] = df["cards_over_odds"] / cards_factor
            df["cards_under_odds"] = df["cards_under_odds"] * cards_factor

        return df

    def estimate_historical_odds(
        self,
        matches_df: pd.DataFrame,
        total_cards_col: str = "total_cards",
        target_line: float = DEFAULT_CARD_LINE,
        use_market_efficiency: bool = True,
    ) -> pd.DataFrame:
        """Estimate cards odds for historical matches."""
        return super().estimate_historical_odds(
            matches_df, total_cards_col, target_line, use_market_efficiency
        )


def add_cards_odds_to_features(
    features_df: pd.DataFrame,
    total_cards_col: str = "total_cards",
    target_line: float = DEFAULT_CARD_LINE,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Add cards odds to existing features DataFrame."""
    loader = CardsOddsLoader()
    result = loader.estimate_historical_odds(
        features_df, total_cards_col, target_line
    )

    if output_path:
        result.to_csv(output_path, index=False)
        logger.info(f"Saved to {output_path}")

    return result
