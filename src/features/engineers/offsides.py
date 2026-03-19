"""
Offsides Feature Engineering

Builds predictive features for offsides betting markets.

Data distributions (typical):
- Offsides: Mean ~2.2 per team, ~4.4 per match
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.data_collection.match_stats_utils import normalize_match_stats_columns
from src.features.engineers.base import BaseFeatureEngineer
from src.leagues import EUROPEAN_LEAGUES

logger = logging.getLogger(__name__)


class OffsidesFeatureEngineer(BaseFeatureEngineer):
    """
    Generates features for predicting total match offsides.

    Default ~2.2 offsides per team per match.
    """

    DEFAULTS = {
        "offsides": 2.2,  # Average offsides per team
    }

    def __init__(
        self,
        window_sizes: List[int] = [5, 10],
        min_matches: int = 3,
        ema_span: int = 10,
        use_league_relative: bool = True,
        league_window: int = 50,
    ):
        self.window_sizes = window_sizes
        self.min_matches = min_matches
        self.ema_span = ema_span
        self.use_league_relative = use_league_relative
        self.league_window = league_window
        self.data_dir = Path("data/01-raw")

    def create_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create offsides features from match data."""
        matches = data.get("matches")
        if matches is None or matches.empty:
            return pd.DataFrame()

        match_stats = self._load_match_stats()
        if match_stats.empty:
            return pd.DataFrame()

        featured = self._build_features(match_stats)
        feature_cols = [c for c in featured.columns if "offside" in c.lower() or c == "fixture_id"]

        return featured[feature_cols]

    def _load_match_stats(self) -> pd.DataFrame:
        """Load match stats with offsides data."""
        all_stats = []
        for league in EUROPEAN_LEAGUES:
            league_dir = self.data_dir / league
            if not league_dir.exists():
                continue
            for season_dir in league_dir.iterdir():
                if not season_dir.is_dir():
                    continue
                stats_path = season_dir / "match_stats.parquet"
                if stats_path.exists():
                    try:
                        df = pd.read_parquet(stats_path)
                        df = normalize_match_stats_columns(df)
                        if "home_offsides" in df.columns:
                            df["league"] = league
                            all_stats.append(df)
                    except Exception as e:
                        logger.debug(f"Could not load {stats_path}: {e}")

        return pd.concat(all_stats, ignore_index=True) if all_stats else pd.DataFrame()

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build offsides prediction features."""
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        df = df.sort_values("date").reset_index(drop=True)

        if "total_offsides" not in df.columns:
            df["total_offsides"] = df["home_offsides"] + df["away_offsides"]

        # Team rolling offsides (EMA)
        df["home_offsides_ema"] = df.groupby("home_team")["home_offsides"].transform(
            lambda x: x.shift(1).ewm(span=self.ema_span, min_periods=self.min_matches).mean()
        )
        df["away_offsides_ema"] = df.groupby("away_team")["away_offsides"].transform(
            lambda x: x.shift(1).ewm(span=self.ema_span, min_periods=self.min_matches).mean()
        )

        # Offsides conceded
        df["home_offsides_conceded_ema"] = df.groupby("home_team")["away_offsides"].transform(
            lambda x: x.shift(1).ewm(span=self.ema_span, min_periods=self.min_matches).mean()
        )
        df["away_offsides_conceded_ema"] = df.groupby("away_team")["home_offsides"].transform(
            lambda x: x.shift(1).ewm(span=self.ema_span, min_periods=self.min_matches).mean()
        )

        # Expected offsides
        df["expected_home_offsides"] = (
            df["home_offsides_ema"].fillna(self.DEFAULTS["offsides"])
            + df["away_offsides_conceded_ema"].fillna(self.DEFAULTS["offsides"])
        ) / 2

        df["expected_away_offsides"] = (
            df["away_offsides_ema"].fillna(self.DEFAULTS["offsides"])
            + df["home_offsides_conceded_ema"].fillna(self.DEFAULTS["offsides"])
        ) / 2

        df["expected_total_offsides"] = df["expected_home_offsides"] + df["expected_away_offsides"]

        # NegBin features (overdispersed count distribution)
        from src.odds.count_distribution import DISPERSION_RATIOS, overdispersed_cdf

        d_offsides = DISPERSION_RATIOS.get("offsides", 1.0)
        expected = df["expected_total_offsides"]
        df["negbin_offsides_over_45_prob"] = 1.0 - overdispersed_cdf(
            4.5, expected.values, "offsides"
        )
        df["negbin_offsides_expected_std"] = np.sqrt(expected * d_offsides)

        # Offsides differential
        df["offsides_attack_diff"] = df["home_offsides_ema"] - df["away_offsides_ema"]

        # League-relative features (requires 'league' column)
        if "league" in df.columns:
            df["offsides_league_ema_avg"] = df.groupby("league")["total_offsides"].transform(
                lambda x: x.shift(1).ewm(span=self.league_window, min_periods=10).mean()
            )
            df["offsides_league_rolling_std"] = df.groupby("league")["total_offsides"].transform(
                lambda x: x.shift(1).rolling(self.league_window, min_periods=10).std()
            )
            league_avg = df["offsides_league_ema_avg"]
            df["expected_total_offsides_vs_league"] = df["expected_total_offsides"] - league_avg
            df["offsides_ratio_to_league"] = (
                df["expected_total_offsides"] / league_avg.replace(0, np.nan)
            ).clip(0.5, 2.0)
            df["home_offsides_vs_league"] = df["home_offsides_ema"] - (league_avg / 2)
            df["away_offsides_vs_league"] = df["away_offsides_ema"] - (league_avg / 2)

        return df
