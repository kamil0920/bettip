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

    FEATURES_FILE = Path("data/03-features/features_all_5leagues_with_odds.csv")

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
            logger.info(f"Loading features from {self.features_file}")
            self._features_df = pd.read_csv(self.features_file)

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
