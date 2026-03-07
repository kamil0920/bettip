"""
BTTS (Both Teams To Score) Odds Loader

Uses The Odds API for real-time BTTS odds.
Falls back to estimated odds for historical matches.
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.odds.base_odds_loader import BaseOddsLoader

logger = logging.getLogger(__name__)

DEFAULT_BTTS_YES_ODDS = 1.80
DEFAULT_BTTS_NO_ODDS = 2.00


class BTTSOddsLoader(BaseOddsLoader):
    """Load BTTS odds from The Odds API or use estimated odds."""

    @property
    def _market_name(self) -> str:
        return "btts"

    @property
    def _api_market_key(self) -> str:
        return "btts"

    @property
    def _default_over_odds(self) -> float:
        return DEFAULT_BTTS_YES_ODDS

    @property
    def _default_under_odds(self) -> float:
        return DEFAULT_BTTS_NO_ODDS

    @property
    def _over_col(self) -> str:
        return "btts_yes_odds"

    @property
    def _under_col(self) -> str:
        return "btts_no_odds"

    def _parse_event_odds(
        self, event_data: Dict, target_line: Optional[float] = None
    ) -> Optional[Dict]:
        """Parse BTTS odds (binary Yes/No outcome, no lines)."""
        btts_yes_odds = []
        btts_no_odds = []

        for bookmaker in event_data.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                if market.get("key") == "btts":
                    for outcome in market.get("outcomes", []):
                        if outcome.get("name") == "Yes":
                            btts_yes_odds.append(outcome.get("price"))
                        elif outcome.get("name") == "No":
                            btts_no_odds.append(outcome.get("price"))

        if not btts_yes_odds and not btts_no_odds:
            return None

        result = {}
        if btts_yes_odds:
            result["btts_yes_avg"] = np.mean(btts_yes_odds)
            result["btts_yes_max"] = max(btts_yes_odds)
            result["btts_yes_min"] = min(btts_yes_odds)
        if btts_no_odds:
            result["btts_no_avg"] = np.mean(btts_no_odds)
            result["btts_no_max"] = max(btts_no_odds)
            result["btts_no_min"] = min(btts_no_odds)

        return result

    def _apply_estimation_adjustments(
        self, df: pd.DataFrame, target_line: Optional[float] = None
    ) -> pd.DataFrame:
        """Adjust BTTS odds based on team attacking strength."""
        if (
            "home_goals_scored_ema" in df.columns
            and "away_goals_scored_ema" in df.columns
        ):
            attack_factor = (
                df["home_goals_scored_ema"] + df["away_goals_scored_ema"]
            ) / 3
            attack_factor = attack_factor.clip(0.5, 1.5)

            df["btts_yes_odds"] = df["btts_yes_odds"] / attack_factor.clip(0.9, 1.1)
            df["btts_no_odds"] = df["btts_no_odds"] * attack_factor.clip(0.9, 1.1)

        return df

    def estimate_historical_odds(
        self,
        matches_df: pd.DataFrame,
        btts_yes_col: str = "btts",
        use_market_efficiency: bool = True,
    ) -> pd.DataFrame:
        """Estimate BTTS odds for historical matches."""
        return super().estimate_historical_odds(
            matches_df, btts_yes_col, None, use_market_efficiency
        )


def add_btts_odds_to_features(
    features_df: pd.DataFrame,
    btts_col: str = "btts",
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Add BTTS odds to existing features DataFrame."""
    loader = BTTSOddsLoader()
    result = loader.estimate_historical_odds(features_df, btts_col)

    if output_path:
        result.to_csv(output_path, index=False)
        logger.info(f"Saved to {output_path}")

    return result
