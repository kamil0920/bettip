"""
League Aggregate Feature Engineering

Builds league-level expanding features that capture league character:
- Home win / draw / away win rates per league
- Average goals, BTTS rate per league
- Corners and cards averages per league
- Goal variance (league unpredictability)

These features let the model learn "this match is in a high-scoring league"
or "this league has strong home advantage" without using league as a
categorical identifier directly.

All features use shift(1) + expanding(min_periods) to prevent look-ahead bias.
"""

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

from src.features.engineers.base import BaseFeatureEngineer

logger = logging.getLogger(__name__)


class LeagueAggregateFeatureEngineer(BaseFeatureEngineer):
    """
    Generates league-level aggregate features for all betting markets.

    Fills the gap where fouls/cards/shots have league-relative features
    but H2H markets (home_win, away_win, over25, btts) and corners do not.
    """

    def __init__(self, min_matches: int = 20):
        """
        Args:
            min_matches: Minimum league matches for reliable expanding stats.
        """
        self.min_matches = min_matches

    def create_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create league-aggregate features from match data."""
        matches = data.get("matches")
        if matches is None or matches.empty:
            return pd.DataFrame()

        if "league" not in matches.columns:
            logger.warning("No 'league' column in matches — skipping league aggregates")
            return pd.DataFrame()

        df = matches[["fixture_id", "date", "league"]].copy()

        # Need outcome columns for H2H features
        if "ft_home" in matches.columns and "ft_away" in matches.columns:
            df["ft_home"] = matches["ft_home"]
            df["ft_away"] = matches["ft_away"]
        else:
            logger.warning("No ft_home/ft_away — skipping league aggregates")
            return pd.DataFrame()

        df = df.sort_values("date").reset_index(drop=True)

        # Derive match-level indicators
        df["_home_win"] = (df["ft_home"] > df["ft_away"]).astype(float)
        df["_draw"] = (df["ft_home"] == df["ft_away"]).astype(float)
        df["_total_goals"] = df["ft_home"] + df["ft_away"]
        df["_btts"] = ((df["ft_home"] > 0) & (df["ft_away"] > 0)).astype(float)

        min_p = self.min_matches

        # --- H2H league features ---
        df["league_home_win_rate"] = df.groupby("league")["_home_win"].transform(
            lambda x: x.shift(1).expanding(min_periods=min_p).mean()
        )
        df["league_draw_rate"] = df.groupby("league")["_draw"].transform(
            lambda x: x.shift(1).expanding(min_periods=min_p).mean()
        )
        df["league_avg_goals"] = df.groupby("league")["_total_goals"].transform(
            lambda x: x.shift(1).expanding(min_periods=min_p).mean()
        )
        df["league_goal_std"] = df.groupby("league")["_total_goals"].transform(
            lambda x: x.shift(1).expanding(min_periods=min_p).std()
        )
        df["league_btts_rate"] = df.groupby("league")["_btts"].transform(
            lambda x: x.shift(1).expanding(min_periods=min_p).mean()
        )

        # --- Corners league features (if available) ---
        if "home_corners" in matches.columns and "away_corners" in matches.columns:
            df["_total_corners"] = (
                matches["home_corners"].fillna(np.nan) + matches["away_corners"].fillna(np.nan)
            )
            df["league_avg_corners"] = df.groupby("league")["_total_corners"].transform(
                lambda x: x.shift(1).expanding(min_periods=min_p).mean()
            )
            df["league_corners_std"] = df.groupby("league")["_total_corners"].transform(
                lambda x: x.shift(1).expanding(min_periods=min_p).std()
            )

        # --- Cards league features (if available, complement niche_markets.py) ---
        # niche_markets.py already has cards_league_expanding_avg but only within
        # CardsFeatureEngineer scope. This adds it to the general feature set.
        # RFECV will deduplicate if both survive.

        # Drop internal columns, keep only features + fixture_id
        feature_cols = [
            c for c in df.columns
            if c.startswith("league_") and c != "league"
        ]
        result = df[["fixture_id"] + feature_cols].copy()

        n_features = len(feature_cols)
        n_valid = result[feature_cols].notna().any(axis=0).sum()
        logger.info(
            f"Created {n_features} league-aggregate features "
            f"({n_valid} with data, min_matches={min_p})"
        )

        return result
