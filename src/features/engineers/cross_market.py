"""
Cross-Market Feature Engineering

Creates interaction features between different betting markets based on
xgbfir analysis findings:

FOULS Market (Jan 2026 analysis - top interactions by Gain):
1. away_cards × expected_total_with_home_adj (2236 gain)
2. expected_total_with_home_adj × home_cards (928 gain)
3. away_cards × home_cards_ema (488 gain)
4. away_cards × home_shots (311 gain)
5. home_cards × ref_avg_goals (288 gain)
6. away_cards × home_cards (263 gain)
7. away_cards_ema × fouls_diff (111 gain)
8. corners_defense_diff × home_cards (129 gain)

CORNERS: shots predict corners (away_shots × home_shots)
SHOTS: corners predict shots (away_corners × home_corners)

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

            # =================================================================
            # FOULS MARKET INTERACTIONS (from xgbfir Jan 2026 analysis)
            # =================================================================

            # Get cards features
            home_cards = self._safe_get(match, ['home_cards', 'home_avg_cards'], 1.5)
            away_cards = self._safe_get(match, ['away_cards', 'away_avg_cards'], 1.5)
            home_cards_ema = self._safe_get(match, ['home_cards_ema', 'home_yellows_ema'], 1.5)
            away_cards_ema = self._safe_get(match, ['away_cards_ema', 'away_yellows_ema'], 1.5)

            # Get expected totals (for fouls/goals)
            expected_total = self._safe_get(match, ['expected_total_with_home_adj', 'expected_total', 'poisson_total_goals'], 2.5)

            # Get shots
            home_shots_val = self._safe_get(match, ['home_shots', 'home_shots_ema', 'home_total_shots_ema'], 12.0)

            # Get referee features
            ref_avg_goals = self._safe_get(match, ['ref_avg_goals', 'referee_avg_goals'], 2.7)

            # Get fouls diff
            fouls_diff = self._safe_get(match, ['fouls_diff', 'home_fouls_ema'], 0.0) - self._safe_get(match, ['away_fouls_ema'], 0.0)

            # Get corners defense diff
            corners_defense_diff = self._safe_get(match, ['corners_defense_diff', 'home_corners_conceded_ema'], 0.0)

            # TOP FOULS INTERACTIONS (ordered by xgbfir gain)

            # 1. away_cards × expected_total (Gain: 2236)
            features['fouls_int_cards_expected'] = away_cards * expected_total

            # 2. expected_total × home_cards (Gain: 928)
            features['fouls_int_expected_home_cards'] = expected_total * home_cards

            # 3. away_cards × home_cards_ema (Gain: 488)
            features['fouls_int_cards_cross'] = away_cards * home_cards_ema

            # 4. away_cards × home_shots (Gain: 311)
            features['fouls_int_cards_shots'] = away_cards * home_shots_val

            # 5. home_cards × ref_avg_goals (Gain: 288)
            features['fouls_int_cards_ref'] = home_cards * ref_avg_goals

            # 6. away_cards × home_cards (Gain: 263) - direct card interaction
            features['fouls_int_cards_product'] = away_cards * home_cards

            # 7. away_cards_ema × fouls_diff (Gain: 111)
            features['fouls_int_cards_fouls_diff'] = away_cards_ema * abs(fouls_diff)

            # 8. corners_defense_diff × home_cards (Gain: 129)
            features['fouls_int_corners_cards'] = corners_defense_diff * home_cards

            # Combined card intensity (sum of card-related interactions)
            features['fouls_card_intensity'] = (home_cards + away_cards) * (home_cards_ema + away_cards_ema)

            # Cards per expected goal (card density)
            if expected_total > 0:
                features['fouls_cards_per_goal'] = (home_cards + away_cards) / expected_total
            else:
                features['fouls_cards_per_goal'] = home_cards + away_cards

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
