"""
Feature Lookup for Real-Time Inference

Looks up latest historical features for teams to use in real-time predictions.
This avoids needing to regenerate all features for upcoming matches.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class FeatureLookup:
    """
    Lookup latest features for teams from historical data.

    Uses the most recent match data for each team as their "current state"
    for features like ELO, form, etc.
    """

    FEATURES_FILE = Path("data/03-features/features_all_5leagues_with_odds.parquet")

    def __init__(self, features_file: Optional[Path] = None):
        self.features_file = features_file or self.FEATURES_FILE
        self._features_df: Optional[pd.DataFrame] = None
        self._team_features: Dict[str, pd.Series] = {}
        self._available_features: List[str] = []

    def load(self) -> bool:
        """Load features dataset."""
        if not self.features_file.exists():
            logger.error(f"Features file not found: {self.features_file}")
            return False

        try:
            from src.utils.data_io import load_features
            logger.info(f"Loading features from {self.features_file}")
            self._features_df = load_features(self.features_file)

            # Parse dates if present
            if 'date' in self._features_df.columns:
                self._features_df['date'] = pd.to_datetime(self._features_df['date'])
                self._features_df = self._features_df.sort_values('date')

            # Get feature columns (exclude identifiers and targets)
            id_cols = ['fixture_id', 'date', 'home_team', 'away_team', 'home_team_id',
                       'away_team_id', 'league', 'season', 'round', 'referee', 'venue']
            target_cols = ['home_win', 'draw', 'away_win', 'btts', 'over25', 'under25',
                          'goal_margin', 'total_goals', 'home_score', 'away_score',
                          'corners_total', 'shots_total', 'fouls_total', 'cards_total']

            self._available_features = [
                col for col in self._features_df.columns
                if col not in id_cols and col not in target_cols
                and not col.endswith('_target')
            ]

            logger.info(f"Loaded {len(self._features_df)} matches, {len(self._available_features)} features")

            # Pre-compute latest features per team
            self._build_team_index()

            return True

        except Exception as e:
            logger.error(f"Failed to load features: {e}")
            return False

    def _build_team_index(self):
        """Build index of latest features per team."""
        if self._features_df is None:
            return

        df = self._features_df

        # Detect team column names (could be home_team or home_team_name)
        home_col = 'home_team_name' if 'home_team_name' in df.columns else 'home_team'
        away_col = 'away_team_name' if 'away_team_name' in df.columns else 'away_team'

        # Index by home team (most recent home game for each team)
        if home_col in df.columns:
            for team in df[home_col].unique():
                team_rows = df[df[home_col] == team]
                if not team_rows.empty:
                    latest = team_rows.iloc[-1]
                    self._team_features[f"home_{team}"] = latest

        # Index by away team
        if away_col in df.columns:
            for team in df[away_col].unique():
                team_rows = df[df[away_col] == team]
                if not team_rows.empty:
                    latest = team_rows.iloc[-1]
                    self._team_features[f"away_{team}"] = latest

        logger.info(f"Built team index with {len(self._team_features)} entries")

    def get_team_features(
        self,
        home_team: str,
        away_team: str,
        feature_list: Optional[List[str]] = None
    ) -> Optional[pd.DataFrame]:
        """
        Get features for a match between two teams.

        Combines home team's home features with away team's away features.

        Args:
            home_team: Home team name
            away_team: Away team name
            feature_list: Specific features to include (default: all available)

        Returns:
            DataFrame with one row of features, or None if lookup fails
        """
        if self._features_df is None:
            if not self.load():
                return None

        # Get latest features for each team
        home_key = f"home_{home_team}"
        away_key = f"away_{away_team}"

        # Try to find features (with fuzzy matching for team names)
        home_features = self._find_team_features(home_team, is_home=True)
        away_features = self._find_team_features(away_team, is_home=False)

        if home_features is None or away_features is None:
            logger.warning(f"Could not find features for {home_team} vs {away_team}")
            return None

        # Combine features: home_* from home team, away_* from away team
        combined = {}
        features_to_use = feature_list or self._available_features

        for feat in features_to_use:
            if feat.startswith('home_'):
                combined[feat] = home_features.get(feat)
            elif feat.startswith('away_'):
                combined[feat] = away_features.get(feat)
            elif feat in home_features:
                # Non-prefixed features (e.g., odds) - prefer home match
                combined[feat] = home_features.get(feat)

        # Recompute cross-team interaction features using the actual
        # combined home/away values (not stale values from previous opponents)
        self._recompute_cross_features(combined)

        return pd.DataFrame([combined])

    def _find_team_features(
        self,
        team_name: str,
        is_home: bool
    ) -> Optional[pd.Series]:
        """Find team features with fuzzy matching."""
        prefix = "home" if is_home else "away"
        direct_key = f"{prefix}_{team_name}"

        if direct_key in self._team_features:
            return self._team_features[direct_key]

        # Try fuzzy match
        team_lower = team_name.lower()
        for key, features in self._team_features.items():
            if key.startswith(f"{prefix}_"):
                stored_team = key[len(f"{prefix}_"):].lower()
                if team_lower in stored_team or stored_team in team_lower:
                    return features

        return None

    @staticmethod
    def _safe_get(d: dict, key: str, default: float) -> float:
        """Get a float value from dict, returning default if missing/NaN."""
        val = d.get(key)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return default
        return float(val)

    def _recompute_cross_features(self, combined: dict) -> None:
        """Recompute cross-team interaction features for the actual opponent pairing.

        When FeatureLookup combines home_* from team A's last home game and away_*
        from team B's last away game, non-prefixed interaction features (cross_*,
        fouls_int_*, etc.) still carry values from team A's PREVIOUS opponent.
        This method recomputes them using the correctly combined home/away values.
        """
        g = self._safe_get

        # --- Base values from correctly combined home/away features ---
        home_yellows = g(combined, 'home_avg_yellows', 0.0) or g(combined, 'home_yellows_ema', 1.5)
        away_yellows = g(combined, 'away_avg_yellows', 0.0) or g(combined, 'away_yellows_ema', 1.5)

        home_cards = g(combined, 'home_cards_ema', 0.0) or g(combined, 'home_avg_cards', 0.0) or g(combined, 'home_avg_yellows', 1.5)
        away_cards = g(combined, 'away_cards_ema', 0.0) or g(combined, 'away_avg_cards', 0.0) or g(combined, 'away_avg_yellows', 1.5)
        home_cards_ema = g(combined, 'home_cards_ema', 0.0) or g(combined, 'home_yellows_ema', 1.5)
        away_cards_ema = g(combined, 'away_cards_ema', 0.0) or g(combined, 'away_yellows_ema', 1.5)

        home_shots = g(combined, 'home_shots_ema', 0.0) or g(combined, 'home_total_shots_ema', 12.0)
        away_shots = g(combined, 'away_shots_ema', 0.0) or g(combined, 'away_total_shots_ema', 10.0)

        home_corners = g(combined, 'home_corners_ema', 0.0) or g(combined, 'home_corners_won_ema', 5.0)
        away_corners = g(combined, 'away_corners_ema', 0.0) or g(combined, 'away_corners_won_ema', 4.5)

        home_fouls = g(combined, 'home_fouls_committed_ema', 0.0) or g(combined, 'home_fouls_ema', 11.0)
        away_fouls = g(combined, 'away_fouls_committed_ema', 0.0) or g(combined, 'away_fouls_ema', 12.0)

        home_sot = g(combined, 'home_shots_on_target_ema', 4.0)
        away_sot = g(combined, 'away_shots_on_target_ema', 3.5)

        expected_total = g(combined, 'expected_total_with_home_adj', 0.0) or g(combined, 'expected_total', 0.0) or g(combined, 'poisson_total_goals', 2.5)
        goal_diff = g(combined, 'season_gd_diff', 0.0) or g(combined, 'home_season_gd', 0.0) or g(combined, 'gd_form_diff', 0.0)
        fouls_diff = home_fouls - away_fouls
        corners_defense_diff = g(combined, 'corners_defense_diff', 0.0) or g(combined, 'home_corners_conceded_ema', 0.0)
        ref_avg_goals = g(combined, 'ref_avg_goals', 0.0) or g(combined, 'referee_avg_goals', 2.7)

        # Odds features
        under25_odds = g(combined, 'b365_under25_close', 0.0) or g(combined, 'avg_under25_close', 0.0) or g(combined, 'odds_under25_prob', 0.5)
        over25_prob = g(combined, 'poisson_over25_prob', 0.0) or g(combined, 'xg_over25_prob', 0.5)

        # --- Yellows interactions ---
        _set = combined.__setitem__
        if 'cross_yellows_product' in combined:
            _set('cross_yellows_product', home_yellows * away_yellows)
        if 'cross_yellows_total' in combined:
            _set('cross_yellows_total', home_yellows + away_yellows)

        # --- Shots interactions ---
        if 'cross_shots_product' in combined:
            _set('cross_shots_product', home_shots * away_shots)
        if 'cross_shots_total' in combined:
            _set('cross_shots_total', home_shots + away_shots)
        if 'cross_shots_diff' in combined:
            _set('cross_shots_diff', home_shots - away_shots)

        # --- Corners interactions ---
        if 'cross_corners_product' in combined:
            _set('cross_corners_product', home_corners * away_corners)
        if 'cross_corners_total' in combined:
            _set('cross_corners_total', home_corners + away_corners)
        if 'cross_corners_diff' in combined:
            _set('cross_corners_diff', home_corners - away_corners)

        # --- Fouls interactions ---
        if 'cross_fouls_product' in combined:
            _set('cross_fouls_product', home_fouls * away_fouls)
        if 'cross_fouls_total' in combined:
            _set('cross_fouls_total', home_fouls + away_fouls)
        if 'cross_fouls_cards_proxy' in combined:
            _set('cross_fouls_cards_proxy', (home_fouls + away_fouls) * 0.15)

        # --- Fouls-cards market interactions ---
        if 'fouls_int_cards_expected' in combined:
            _set('fouls_int_cards_expected', away_cards * expected_total)
        if 'fouls_int_expected_home_cards' in combined:
            _set('fouls_int_expected_home_cards', expected_total * home_cards)
        if 'fouls_int_cards_cross' in combined:
            _set('fouls_int_cards_cross', away_cards * home_cards_ema)
        if 'fouls_int_cards_shots' in combined:
            _set('fouls_int_cards_shots', away_cards * home_shots)
        if 'fouls_int_cards_ref' in combined:
            _set('fouls_int_cards_ref', home_cards * ref_avg_goals)
        if 'fouls_int_cards_product' in combined:
            _set('fouls_int_cards_product', away_cards * home_cards)
        if 'fouls_int_cards_fouls_diff' in combined:
            _set('fouls_int_cards_fouls_diff', away_cards_ema * abs(fouls_diff))
        if 'fouls_int_corners_cards' in combined:
            _set('fouls_int_corners_cards', corners_defense_diff * home_cards)
        if 'fouls_card_intensity' in combined:
            _set('fouls_card_intensity', (home_cards + away_cards) * (home_cards_ema + away_cards_ema))
        if 'fouls_cards_per_goal' in combined:
            _set('fouls_cards_per_goal', (home_cards + away_cards) / expected_total if expected_total > 0 else home_cards + away_cards)

        # --- Corners market interactions ---
        if 'corners_int_goaldiff_shots' in combined:
            _set('corners_int_goaldiff_shots', goal_diff * home_shots)
        if 'corners_int_fouls_shots' in combined:
            _set('corners_int_fouls_shots', away_fouls * home_shots)
        if 'corners_int_homefouls_shots' in combined:
            _set('corners_int_homefouls_shots', home_fouls * home_shots)
        if 'corners_int_shots_intensity' in combined:
            _set('corners_int_shots_intensity', (home_shots + away_shots) * abs(goal_diff + 0.1))

        # --- Shots market interactions ---
        if 'shots_int_odds_corners' in combined:
            _set('shots_int_odds_corners', under25_odds * home_corners)
        if 'shots_int_corners_over25' in combined:
            _set('shots_int_corners_over25', home_corners * over25_prob)
        if 'shots_int_away_corners_over25' in combined:
            _set('shots_int_away_corners_over25', away_corners * over25_prob)

        # --- BTTS market interactions ---
        if 'btts_int_goaldiff_sot' in combined:
            _set('btts_int_goaldiff_sot', goal_diff * home_sot)
        if 'btts_int_away_sot_goaldiff' in combined:
            _set('btts_int_away_sot_goaldiff', away_sot * goal_diff)
        if 'btts_int_sot_product' in combined:
            _set('btts_int_sot_product', away_sot * home_sot)
        if 'btts_int_sot_total' in combined:
            _set('btts_int_sot_total', home_sot + away_sot)

        # --- Goals/Over25 market interactions ---
        if 'goals_int_sot_goaldiff' in combined:
            _set('goals_int_sot_goaldiff', away_sot * abs(goal_diff + 0.1))
        if 'goals_int_sot_product' in combined:
            _set('goals_int_sot_product', away_sot * home_sot)
        if 'goals_int_goaldiff_home_sot' in combined:
            _set('goals_int_goaldiff_home_sot', goal_diff * home_sot)
        if 'goals_int_attack_intensity' in combined:
            _set('goals_int_attack_intensity', (home_sot + away_sot) * (1 + abs(goal_diff)))

        # ============================================================
        # Diff features: home_X - away_X
        # These carry stale opponent data from previous matches and
        # must be recomputed from the correctly paired home/away values.
        # ============================================================
        _DIFF_FEATURES = [
            # (output_name, home_component, away_component)
            ('elo_diff', 'home_elo', 'away_elo'),
            ('cards_diff', 'home_cards_ema', 'away_cards_ema'),
            ('lineup_stability_diff', 'home_lineup_stability', 'away_lineup_stability'),
            ('xi_rating_advantage', 'home_xi_avg_rating', 'away_xi_avg_rating'),
            ('missing_rating_disadvantage', 'home_missing_rating', 'away_missing_rating'),
            ('corners_attack_diff', 'home_corners_won_ema', 'away_corners_won_ema'),
            ('corners_defense_diff', 'home_corners_conceded_ema', 'away_corners_conceded_ema'),
            ('fouls_volatility_diff', 'home_fouls_volatility', 'away_fouls_volatility'),
            ('cards_variance_ratio_diff', 'home_cards_variance_ratio', 'away_cards_variance_ratio'),
            ('shots_hurst_diff', 'home_shots_hurst', 'away_shots_hurst'),
            ('fouls_momentum_ratio_diff', 'home_fouls_momentum_ratio', 'away_fouls_momentum_ratio'),
            ('goals_conceded_momentum_advantage', 'home_goals_conceded_momentum', 'away_goals_conceded_momentum'),
            ('goals_scored_momentum_advantage', 'home_goals_scored_momentum', 'away_goals_scored_momentum'),
            ('points_momentum_advantage', 'home_points_momentum', 'away_points_momentum'),
            ('bayes_win_rate_diff', 'home_bayes_win_rate', 'away_bayes_win_rate'),
        ]
        for feat_name, home_col, away_col in _DIFF_FEATURES:
            if feat_name in combined and home_col in combined and away_col in combined:
                _set(feat_name, g(combined, home_col, 0.0) - g(combined, away_col, 0.0))

        # ============================================================
        # Expected features: computed from both teams' stats
        # These average home attack with away defense (or vice versa).
        # ============================================================

        # Expected corners: (attack + opposing defense) / 2
        if 'expected_home_corners' in combined:
            ha = g(combined, 'home_corners_won_ema', 5.0)
            ad = g(combined, 'away_corners_conceded_ema', 4.5)
            _set('expected_home_corners', (ha + ad) / 2)
        if 'expected_away_corners' in combined:
            aa = g(combined, 'away_corners_won_ema', 4.5)
            hd = g(combined, 'home_corners_conceded_ema', 5.0)
            _set('expected_away_corners', (aa + hd) / 2)
        if 'expected_total_corners' in combined:
            ehc = g(combined, 'expected_home_corners', 5.0)
            eac = g(combined, 'expected_away_corners', 4.5)
            _set('expected_total_corners', ehc + eac)

        # Expected fouls: (team commits + opposing drawn) / 2
        if 'expected_home_fouls' in combined:
            hf = g(combined, 'home_fouls_match_ema', 0.0) or g(combined, 'home_fouls_committed_ema', 11.0)
            afd = g(combined, 'away_fouls_drawn_ema', 11.0)
            _set('expected_home_fouls', (hf + afd) / 2)
        if 'expected_away_fouls' in combined:
            af = g(combined, 'away_fouls_match_ema', 0.0) or g(combined, 'away_fouls_committed_ema', 12.0)
            hfd = g(combined, 'home_fouls_drawn_ema', 12.0)
            _set('expected_away_fouls', (af + hfd) / 2)

        # Expected cards: home_cards_ema + away_cards_ema
        if 'expected_total_cards' in combined:
            _set('expected_total_cards', home_cards_ema + away_cards_ema)

        # Expected shots: (team attacks + opposing conceded) / 2
        if 'expected_home_shots' in combined:
            hsa = g(combined, 'home_shots_match_ema', 0.0) or g(combined, 'home_shots_ema', 12.0)
            asc = g(combined, 'away_shots_conceded_ema', 12.0)
            _set('expected_home_shots', (hsa + asc) / 2)
        if 'expected_away_shots' in combined:
            asa = g(combined, 'away_shots_match_ema', 0.0) or g(combined, 'away_shots_ema', 10.0)
            hsc = g(combined, 'home_shots_conceded_ema', 10.0)
            _set('expected_away_shots', (asa + hsc) / 2)
        if 'expected_total_shots' in combined:
            ehs = g(combined, 'expected_home_shots', 12.0)
            eas = g(combined, 'expected_away_shots', 10.0)
            _set('expected_total_shots', ehs + eas)

    def get_h2h_features(
        self,
        home_team: str,
        away_team: str,
        n_matches: int = 5
    ) -> Dict[str, Any]:
        """
        Get head-to-head features from historical matches.

        Args:
            home_team: Home team name
            away_team: Away team name
            n_matches: Number of past matches to consider

        Returns:
            Dict of H2H features
        """
        if self._features_df is None:
            return {}

        df = self._features_df

        # Detect team column names
        home_col = 'home_team_name' if 'home_team_name' in df.columns else 'home_team'
        away_col = 'away_team_name' if 'away_team_name' in df.columns else 'away_team'

        # Find matches between these teams
        h2h = df[
            ((df[home_col] == home_team) & (df[away_col] == away_team)) |
            ((df[home_col] == away_team) & (df[away_col] == home_team))
        ].tail(n_matches)

        if h2h.empty:
            return {}

        features = {
            'h2h_matches': len(h2h),
            'h2h_home_wins': 0,
            'h2h_away_wins': 0,
            'h2h_draws': 0,
            'h2h_avg_goals': 0.0,
        }

        for _, match in h2h.iterrows():
            home_score = match.get('home_score', 0) or 0
            away_score = match.get('away_score', 0) or 0

            # Adjust for which team was home
            if match[home_col] == home_team:
                if home_score > away_score:
                    features['h2h_home_wins'] += 1
                elif away_score > home_score:
                    features['h2h_away_wins'] += 1
                else:
                    features['h2h_draws'] += 1
            else:
                if away_score > home_score:
                    features['h2h_home_wins'] += 1
                elif home_score > away_score:
                    features['h2h_away_wins'] += 1
                else:
                    features['h2h_draws'] += 1

            features['h2h_avg_goals'] += (home_score + away_score)

        features['h2h_avg_goals'] /= len(h2h)
        features['h2h_home_win_pct'] = features['h2h_home_wins'] / len(h2h) if h2h.shape[0] > 0 else 0.33

        return features

    @property
    def available_features(self) -> List[str]:
        """Get list of available features."""
        if not self._available_features and self._features_df is None:
            self.load()
        return self._available_features


# Singleton instance
_lookup: Optional[FeatureLookup] = None


def get_feature_lookup() -> FeatureLookup:
    """Get singleton FeatureLookup instance."""
    global _lookup
    if _lookup is None:
        _lookup = FeatureLookup()
    return _lookup
