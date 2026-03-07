"""
Corners Totals Odds Loader

Uses The Odds API for real-time corners over/under odds.
Falls back to estimated odds for historical matches.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.odds.base_odds_loader import TotalsOddsLoader

logger = logging.getLogger(__name__)

COMMON_CORNER_LINES = [8.5, 9.5, 10.5, 11.5, 12.5]
DEFAULT_CORNER_LINE = 9.5
DEFAULT_OVER_ODDS = 1.85
DEFAULT_UNDER_ODDS = 1.95


class CornersOddsLoader(TotalsOddsLoader):
    """Load corners totals odds from The Odds API or use estimated odds."""

    @property
    def _market_name(self) -> str:
        return "corners"

    @property
    def _api_market_key(self) -> str:
        return "alternate_totals_corners"

    @property
    def _default_over_odds(self) -> float:
        return DEFAULT_OVER_ODDS

    @property
    def _default_under_odds(self) -> float:
        return DEFAULT_UNDER_ODDS

    @property
    def _over_col(self) -> str:
        return "corners_over_odds"

    @property
    def _under_col(self) -> str:
        return "corners_under_odds"

    @property
    def _common_lines(self) -> List[float]:
        return COMMON_CORNER_LINES

    @property
    def _default_line(self) -> float:
        return DEFAULT_CORNER_LINE

    @property
    def _col_prefix(self) -> str:
        return "corners"

    def _apply_estimation_adjustments(
        self, df: pd.DataFrame, target_line: Optional[float] = None
    ) -> pd.DataFrame:
        """Adjust corners odds based on team corner EMAs."""
        if (
            "home_corners_won_ema" in df.columns
            and "away_corners_won_ema" in df.columns
        ):
            corner_factor = (
                df["home_corners_won_ema"] + df["away_corners_won_ema"]
            ) / 10
            corner_factor = corner_factor.clip(0.8, 1.2)

            df["corners_over_odds"] = df["corners_over_odds"] / corner_factor
            df["corners_under_odds"] = df["corners_under_odds"] * corner_factor

        return df

    def estimate_historical_odds(
        self,
        matches_df: pd.DataFrame,
        total_corners_col: str = "total_corners",
        target_line: float = DEFAULT_CORNER_LINE,
        use_market_efficiency: bool = True,
    ) -> pd.DataFrame:
        """Estimate corners odds for historical matches."""
        return super().estimate_historical_odds(
            matches_df, total_corners_col, target_line, use_market_efficiency
        )


def add_corners_odds_to_features(
    features_df: pd.DataFrame,
    total_corners_col: str = "total_corners",
    target_line: float = DEFAULT_CORNER_LINE,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Add corners odds to existing features DataFrame."""
    loader = CornersOddsLoader()
    result = loader.estimate_historical_odds(
        features_df, total_corners_col, target_line
    )

    if output_path:
        result.to_csv(output_path, index=False)
        logger.info(f"Saved to {output_path}")

    return result
