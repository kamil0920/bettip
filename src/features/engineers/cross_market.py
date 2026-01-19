"""
Cross-Market Feature Engineering

Creates interaction features between different betting markets based on
xgbfir analysis findings:

- CORNERS: shots predict corners (away_shots × home_shots = 3364.6 gain)
- SHOTS: corners predict shots (away_corners × home_corners = 1883.5 gain)
- FOULS: yellows × odds_upset_potential = 1061.6 gain

These cross-market features capture relationships where one market's
predictors improve predictions for another market.
"""
import logging
from typing import Dict

import numpy as np
import pandas as pd

from src.features.engineers.base import BaseFeatureEngineer

logger = logging.getLogger(__name__)


class CrossMarketFeatureEngineer(BaseFeatureEngineer):
    """
    Creates cross-market interaction features.

    Based on xgbfir feature importance analysis, certain features from
    one market are strong predictors for another market. This engineer
    creates interaction features to capture these relationships.
    """

    def __init__(self):
        """Initialize cross-market feature engineer."""
        pass

    def create_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Create cross-market interaction features.

        Args:
            data: Dict containing 'matches' DataFrame

        Returns:
            DataFrame with cross-market features indexed by fixture_id
        """
        matches = data.get('matches')
        if matches is None or matches.empty:
            logger.warning("No matches data available for cross-market features")
            return pd.DataFrame()

        df = matches.copy()
        features_list = []

        for idx, match in df.iterrows():
            fixture_id = match['fixture_id']

            features = {'fixture_id': fixture_id}

            # 1. Shots-Corners interaction (for corner predictions)
            # xgbfir showed: away_shots × home_shots predicts corners
            home_shots = self._safe_get(match, ['home_shots_ema', 'home_total_shots_ema'], 12.0)
            away_shots = self._safe_get(match, ['away_shots_ema', 'away_total_shots_ema'], 10.0)

            features['cross_shots_product'] = home_shots * away_shots
            features['cross_shots_total'] = home_shots + away_shots
            features['cross_shots_diff'] = home_shots - away_shots

            # 2. Corners-Shots interaction (for shots predictions)
            # xgbfir showed: away_corners × home_corners predicts shots
            home_corners = self._safe_get(match, ['home_corners_ema', 'home_corners_won_ema'], 5.0)
            away_corners = self._safe_get(match, ['away_corners_ema', 'away_corners_won_ema'], 4.5)

            features['cross_corners_product'] = home_corners * away_corners
            features['cross_corners_total'] = home_corners + away_corners
            features['cross_corners_diff'] = home_corners - away_corners

            # 3. Yellows × Match intensity (for fouls predictions)
            # xgbfir showed: away_avg_yellows × odds_upset_potential predicts fouls
            home_yellows = self._safe_get(match, ['home_avg_yellows', 'home_yellows_ema'], 1.5)
            away_yellows = self._safe_get(match, ['away_avg_yellows', 'away_yellows_ema'], 1.5)

            features['cross_yellows_product'] = home_yellows * away_yellows
            features['cross_yellows_total'] = home_yellows + away_yellows

            # Odds-based upset potential (if available)
            home_odds = self._safe_get(match, ['avg_home_open', 'b365_home_open'], None)
            away_odds = self._safe_get(match, ['avg_away_open', 'b365_away_open'], None)

            if home_odds and away_odds and home_odds > 0 and away_odds > 0:
                # Upset potential: higher = more likely upset
                features['odds_upset_potential'] = away_odds / (home_odds + away_odds)
                features['cross_yellows_upset'] = away_yellows * features['odds_upset_potential']
            else:
                features['odds_upset_potential'] = 0.5
                features['cross_yellows_upset'] = away_yellows * 0.5

            # 4. Fouls-Cards interaction
            home_fouls = self._safe_get(match, ['home_fouls_committed_ema', 'home_fouls_ema'], 11.0)
            away_fouls = self._safe_get(match, ['away_fouls_committed_ema', 'away_fouls_ema'], 12.0)

            features['cross_fouls_product'] = home_fouls * away_fouls
            features['cross_fouls_total'] = home_fouls + away_fouls

            # Expected cards from fouls (fouls strongly predict cards)
            features['cross_fouls_cards_proxy'] = (home_fouls + away_fouls) * 0.15  # ~15% of fouls become cards

            features_list.append(features)

        result = pd.DataFrame(features_list)
        logger.info(f"Created {len(result.columns) - 1} cross-market features for {len(result)} matches")
        return result

    def _safe_get(self, match: pd.Series, columns: list, default):
        """Safely get value from match, trying multiple column names."""
        for col in columns:
            if col in match.index and pd.notna(match[col]):
                return float(match[col])
        return default
