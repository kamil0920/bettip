"""
Shots Totals Odds Loader

Uses The Odds API for real-time shots over/under odds.
Falls back to estimated odds for historical matches.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.odds.base_odds_loader import TotalsOddsLoader

logger = logging.getLogger(__name__)

COMMON_SHOT_LINES = [3.5, 4.5, 5.5, 6.5, 7.5]
DEFAULT_SHOT_LINE = 4.5
DEFAULT_OVER_ODDS = 1.85
DEFAULT_UNDER_ODDS = 1.95


class ShotsOddsLoader(TotalsOddsLoader):
    """Load shots totals odds from The Odds API or use estimated odds."""

    @property
    def _market_name(self) -> str:
        return "shots"

    @property
    def _api_market_key(self) -> str:
        return "player_shots_on_target"

    @property
    def _default_over_odds(self) -> float:
        return DEFAULT_OVER_ODDS

    @property
    def _default_under_odds(self) -> float:
        return DEFAULT_UNDER_ODDS

    @property
    def _over_col(self) -> str:
        return "shots_over_odds"

    @property
    def _under_col(self) -> str:
        return "shots_under_odds"

    @property
    def _common_lines(self) -> List[float]:
        return COMMON_SHOT_LINES

    @property
    def _default_line(self) -> float:
        return DEFAULT_SHOT_LINE

    @property
    def _col_prefix(self) -> str:
        return "shots"

    def _apply_estimation_adjustments(
        self, df: pd.DataFrame, target_line: Optional[float] = None
    ) -> pd.DataFrame:
        """Adjust shots odds based on team shots EMAs (3-layer adjustment)."""
        # Primary: team shots EMA
        if "home_shots_ema" in df.columns and "away_shots_ema" in df.columns:
            shots_factor = (df["home_shots_ema"] + df["away_shots_ema"]) / 10
            shots_factor = shots_factor.clip(0.8, 1.2)

            df["shots_over_odds"] = df["shots_over_odds"] / shots_factor
            df["shots_under_odds"] = df["shots_under_odds"] * shots_factor
        # Fallback: shots on target EMA
        elif (
            "home_shots_on_target_ema" in df.columns
            and "away_shots_on_target_ema" in df.columns
        ):
            shots_factor = (
                df["home_shots_on_target_ema"] + df["away_shots_on_target_ema"]
            ) / 6
            shots_factor = shots_factor.clip(0.8, 1.2)

            df["shots_over_odds"] = df["shots_over_odds"] / shots_factor
            df["shots_under_odds"] = df["shots_under_odds"] * shots_factor

        # Tertiary: expected total shots deviation
        if target_line is not None and "expected_total_shots" in df.columns:
            deviation = (df["expected_total_shots"] - target_line) / target_line
            deviation = deviation.clip(-0.2, 0.2)

            df["shots_over_odds"] = df["shots_over_odds"] * (1 - deviation * 0.1)
            df["shots_under_odds"] = df["shots_under_odds"] * (1 + deviation * 0.1)

        return df

    def estimate_historical_odds(
        self,
        matches_df: pd.DataFrame,
        total_shots_col: str = "total_shots",
        target_line: float = DEFAULT_SHOT_LINE,
        use_market_efficiency: bool = True,
    ) -> pd.DataFrame:
        """Estimate shots odds for historical matches."""
        return super().estimate_historical_odds(
            matches_df, total_shots_col, target_line, use_market_efficiency
        )


def add_shots_odds_to_features(
    features_df: pd.DataFrame,
    total_shots_col: str = "total_shots",
    target_line: float = DEFAULT_SHOT_LINE,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Add shots odds to existing features DataFrame."""
    loader = ShotsOddsLoader()
    result = loader.estimate_historical_odds(
        features_df, total_shots_col, target_line
    )

    if output_path:
        result.to_csv(output_path, index=False)
        logger.info(f"Saved to {output_path}")

    return result
