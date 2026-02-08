"""
Corner Feature Engineering

Builds predictive features for corners betting markets using:
- Team historical corner statistics (rolling averages)
- Attack intensity proxies (shots, shots on target)
- Home/away specific patterns
- Opponent strength adjustments

Key insight from analysis:
- total_shots correlates 0.343 with total_corners (strongest predictor)
- Home advantage: ~0.93 more corners on average
- Possession has weak correlation with corners

Validation results:
- High confidence UNDER (exp < 9.0): 68.7% accuracy vs 61.4% base rate = +7.3% edge
- High confidence OVER (exp > 11.0): 50.8% accuracy vs 38.6% base rate = +12.2% edge
"""
import logging
from typing import Dict, List, Optional
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_collection.match_stats_utils import normalize_match_stats_columns
from src.features.engineers.base import BaseFeatureEngineer
from src.leagues import EUROPEAN_LEAGUES

logger = logging.getLogger(__name__)


class CornerFeatureEngineer(BaseFeatureEngineer):
    """
    Generates features for predicting match corners.

    Features are computed using only historical data (no look-ahead bias).
    All rolling calculations use shift(1) to exclude current match.
    """

    # Default values when insufficient history
    DEFAULTS = {
        'corners_won': 5.0,      # League average home corners
        'corners_conceded': 4.5,  # League average away corners
        'shots': 12.0,
        'shots_on_target': 4.5,
        'possession': 50.0,
    }

    def __init__(
        self,
        window_sizes: List[int] = [5, 10, 20],
        min_matches: int = 3,
        use_ema: bool = True,
        ema_span: int = 10
    ):
        """
        Initialize corner feature engineer.

        Args:
            window_sizes: Rolling window sizes for averaging
            min_matches: Minimum matches required for valid rolling stats
            use_ema: Use exponential moving average in addition to simple
            ema_span: Span for EMA calculation
        """
        self.window_sizes = window_sizes
        self.min_matches = min_matches
        self.use_ema = use_ema
        self.ema_span = ema_span
        self.data_dir = Path("data/01-raw")

    def create_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Create corner features from match data.

        Implements the IFeatureEngineer interface.

        Args:
            data: Dict containing 'matches' DataFrame with fixture_id, date, home/away teams

        Returns:
            DataFrame with corner features indexed by fixture_id
        """
        matches = data.get('matches')
        if matches is None or matches.empty:
            logger.warning("No matches data provided for corner features")
            return pd.DataFrame()

        # Try to load match_stats with corner data
        match_stats = self._load_match_stats(matches)

        if match_stats.empty:
            logger.warning("No match_stats data available for corner features")
            return pd.DataFrame()

        # Generate features using fit_transform
        featured = self.fit_transform(match_stats)

        # Return only new feature columns with fixture_id
        feature_cols = [c for c in featured.columns if c not in match_stats.columns or c == 'fixture_id']

        if 'fixture_id' not in feature_cols:
            feature_cols = ['fixture_id'] + feature_cols

        return featured[feature_cols]

    def _load_match_stats(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Load match_stats data that contains corner information."""
        all_stats = []

        for league in EUROPEAN_LEAGUES:
            league_dir = self.data_dir / league
            if not league_dir.exists():
                continue

            for season_dir in league_dir.iterdir():
                if not season_dir.is_dir():
                    continue

                stats_path = season_dir / 'match_stats.parquet'
                if stats_path.exists():
                    try:
                        df = pd.read_parquet(stats_path)
                        df = normalize_match_stats_columns(df)
                        df['league'] = league
                        all_stats.append(df)
                    except Exception as e:
                        logger.debug(f"Could not load {stats_path}: {e}")

        if not all_stats:
            return pd.DataFrame()

        stats = pd.concat(all_stats, ignore_index=True)

        # Filter to matches we have in our input
        if 'fixture_id' in matches.columns and 'fixture_id' in stats.columns:
            fixture_ids = set(matches['fixture_id'].unique())
            stats = stats[stats['fixture_id'].isin(fixture_ids)]

        return stats

    def fit_transform(self, matches_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate corner features for all matches.

        Args:
            matches_df: DataFrame with columns:
                - fixture_id, date, home_team, away_team
                - home_corners, away_corners
                - home_shots, away_shots (optional but recommended)
                - home_shots_on_target, away_shots_on_target (optional)
                - home_possession, away_possession (optional)

        Returns:
            DataFrame with corner prediction features
        """
        df = matches_df.copy()

        # Ensure date is datetime and sorted (remove timezone for consistency)
        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
        df = df.sort_values('date').reset_index(drop=True)

        # Add total corners
        if 'total_corners' not in df.columns:
            df['total_corners'] = df['home_corners'] + df['away_corners']

        logger.info(f"Building corner features for {len(df)} matches")

        # Build team-level rolling stats
        features = self._build_team_features(df)

        # Build match-level features
        match_features = self._build_match_features(df, features)

        # Merge with original data
        result = df.merge(match_features, on='fixture_id', how='left')

        logger.info(f"Generated {len(match_features.columns) - 1} corner features")

        return result

    def _build_team_features(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Build rolling statistics for each team."""

        # Create unified team history (both home and away games)
        home_records = df[['date', 'home_team', 'home_corners', 'away_corners',
                          'home_shots', 'away_shots', 'home_possession']].copy()
        home_records.columns = ['date', 'team', 'corners_won', 'corners_conceded',
                               'shots', 'shots_conceded', 'possession']
        home_records['is_home'] = True

        away_records = df[['date', 'away_team', 'away_corners', 'home_corners',
                          'away_shots', 'home_shots', 'away_possession']].copy()
        away_records.columns = ['date', 'team', 'corners_won', 'corners_conceded',
                               'shots', 'shots_conceded', 'possession']
        away_records['is_home'] = False

        team_history = pd.concat([home_records, away_records], ignore_index=True)
        team_history = team_history.sort_values('date')

        # Calculate rolling stats per team
        team_stats = {}

        for team in team_history['team'].unique():
            team_df = team_history[team_history['team'] == team].copy()

            # Convert dates to timezone-naive for consistent comparison
            stats = {'team': team, 'dates': pd.to_datetime(team_df['date']).dt.tz_localize(None).values}

            # Overall rolling stats
            for col in ['corners_won', 'corners_conceded', 'shots', 'shots_conceded']:
                if col in team_df.columns:
                    # Shift to avoid look-ahead
                    shifted = team_df[col].shift(1)

                    # Rolling windows
                    for w in self.window_sizes:
                        stats[f'{col}_roll_{w}'] = shifted.rolling(w, min_periods=self.min_matches).mean().values

                    # EMA
                    if self.use_ema:
                        stats[f'{col}_ema'] = shifted.ewm(span=self.ema_span, min_periods=self.min_matches).mean().values

                    # Expanding mean (all history)
                    stats[f'{col}_expanding'] = shifted.expanding(min_periods=self.min_matches).mean().values

            # Home-specific stats
            home_df = team_df[team_df['is_home']]
            if len(home_df) >= self.min_matches:
                for col in ['corners_won', 'corners_conceded', 'shots']:
                    if col in home_df.columns:
                        shifted = home_df[col].shift(1)
                        stats[f'{col}_home_ema'] = shifted.ewm(span=self.ema_span, min_periods=self.min_matches).mean().values
                        # Pad to full length
                        full_ema = np.full(len(team_df), np.nan)
                        home_idx = team_df[team_df['is_home']].index
                        full_ema[team_df.index.isin(home_idx)] = stats[f'{col}_home_ema']
                        stats[f'{col}_home_ema'] = full_ema

            # Away-specific stats
            away_df = team_df[~team_df['is_home']]
            if len(away_df) >= self.min_matches:
                for col in ['corners_won', 'corners_conceded', 'shots']:
                    if col in away_df.columns:
                        shifted = away_df[col].shift(1)
                        stats[f'{col}_away_ema'] = shifted.ewm(span=self.ema_span, min_periods=self.min_matches).mean().values
                        # Pad to full length
                        full_ema = np.full(len(team_df), np.nan)
                        away_idx = team_df[~team_df['is_home']].index
                        full_ema[team_df.index.isin(away_idx)] = stats[f'{col}_away_ema']
                        stats[f'{col}_away_ema'] = full_ema

            team_stats[team] = pd.DataFrame(stats)

        return team_stats

    def _build_match_features(
        self,
        df: pd.DataFrame,
        team_stats: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Build match-level features from team stats."""

        features_list = []

        for idx, row in df.iterrows():
            fixture_id = row['fixture_id']
            home_team = row['home_team']
            away_team = row['away_team']
            # Ensure match_date is timezone-naive numpy datetime64 for comparison
            match_date = pd.Timestamp(row['date']).to_datetime64()

            feat = {'fixture_id': fixture_id}

            # Get home team stats at match date
            if home_team in team_stats:
                home_stats = team_stats[home_team]
                # Find latest stats before this match
                mask = home_stats['dates'] < match_date
                if mask.any():
                    latest = home_stats[mask].iloc[-1]

                    # Core features
                    feat['home_corners_won_ema'] = latest.get('corners_won_ema', self.DEFAULTS['corners_won'])
                    feat['home_corners_conceded_ema'] = latest.get('corners_conceded_ema', self.DEFAULTS['corners_conceded'])
                    feat['home_shots_ema'] = latest.get('shots_ema', self.DEFAULTS['shots'])

                    # Rolling windows
                    for w in self.window_sizes:
                        feat[f'home_corners_won_roll_{w}'] = latest.get(f'corners_won_roll_{w}', np.nan)
                        feat[f'home_corners_conceded_roll_{w}'] = latest.get(f'corners_conceded_roll_{w}', np.nan)

                    # Expanding
                    feat['home_corners_won_expanding'] = latest.get('corners_won_expanding', np.nan)
                    feat['home_corners_conceded_expanding'] = latest.get('corners_conceded_expanding', np.nan)
                else:
                    self._add_default_home_features(feat)
            else:
                self._add_default_home_features(feat)

            # Get away team stats at match date
            if away_team in team_stats:
                away_stats = team_stats[away_team]
                mask = away_stats['dates'] < match_date
                if mask.any():
                    latest = away_stats[mask].iloc[-1]

                    feat['away_corners_won_ema'] = latest.get('corners_won_ema', self.DEFAULTS['corners_won'])
                    feat['away_corners_conceded_ema'] = latest.get('corners_conceded_ema', self.DEFAULTS['corners_conceded'])
                    feat['away_shots_ema'] = latest.get('shots_ema', self.DEFAULTS['shots'])

                    for w in self.window_sizes:
                        feat[f'away_corners_won_roll_{w}'] = latest.get(f'corners_won_roll_{w}', np.nan)
                        feat[f'away_corners_conceded_roll_{w}'] = latest.get(f'corners_conceded_roll_{w}', np.nan)

                    feat['away_corners_won_expanding'] = latest.get('corners_won_expanding', np.nan)
                    feat['away_corners_conceded_expanding'] = latest.get('corners_conceded_expanding', np.nan)
                else:
                    self._add_default_away_features(feat)
            else:
                self._add_default_away_features(feat)

            # Derived features
            self._add_derived_features(feat)

            features_list.append(feat)

        return pd.DataFrame(features_list)

    def _add_default_home_features(self, feat: Dict) -> None:
        """Add default home features when no history available."""
        feat['home_corners_won_ema'] = self.DEFAULTS['corners_won']
        feat['home_corners_conceded_ema'] = self.DEFAULTS['corners_conceded']
        feat['home_shots_ema'] = self.DEFAULTS['shots']
        feat['home_corners_won_expanding'] = np.nan
        feat['home_corners_conceded_expanding'] = np.nan
        for w in self.window_sizes:
            feat[f'home_corners_won_roll_{w}'] = np.nan
            feat[f'home_corners_conceded_roll_{w}'] = np.nan

    def _add_default_away_features(self, feat: Dict) -> None:
        """Add default away features when no history available."""
        feat['away_corners_won_ema'] = self.DEFAULTS['corners_won']
        feat['away_corners_conceded_ema'] = self.DEFAULTS['corners_conceded']
        feat['away_shots_ema'] = self.DEFAULTS['shots']
        feat['away_corners_won_expanding'] = np.nan
        feat['away_corners_conceded_expanding'] = np.nan
        for w in self.window_sizes:
            feat[f'away_corners_won_roll_{w}'] = np.nan
            feat[f'away_corners_conceded_roll_{w}'] = np.nan

    def _add_derived_features(self, feat: Dict) -> None:
        """Add derived combination features."""

        # Expected corners from each perspective
        # Home team corners = home_attack vs away_defense
        home_attack = feat.get('home_corners_won_ema', self.DEFAULTS['corners_won'])
        away_defense = feat.get('away_corners_conceded_ema', self.DEFAULTS['corners_conceded'])
        feat['expected_home_corners'] = (home_attack + away_defense) / 2

        # Away team corners = away_attack vs home_defense
        away_attack = feat.get('away_corners_won_ema', self.DEFAULTS['corners_won'])
        home_defense = feat.get('home_corners_conceded_ema', self.DEFAULTS['corners_conceded'])
        feat['expected_away_corners'] = (away_attack + home_defense) / 2

        # Total expected corners
        feat['expected_total_corners'] = feat['expected_home_corners'] + feat['expected_away_corners']

        # Home advantage factor (typically ~0.93)
        feat['expected_total_with_home_adj'] = feat['expected_total_corners'] + 0.93

        # Shots-based prediction (shots correlate 0.343 with corners)
        home_shots = feat.get('home_shots_ema', self.DEFAULTS['shots'])
        away_shots = feat.get('away_shots_ema', self.DEFAULTS['shots'])
        feat['combined_shots_ema'] = home_shots + away_shots

        # Corner intensity ratio (corners per shot proxy)
        if home_shots > 0 and away_shots > 0:
            feat['home_corner_intensity'] = home_attack / home_shots
            feat['away_corner_intensity'] = away_attack / away_shots
        else:
            feat['home_corner_intensity'] = 0.4  # Default ~5 corners / 12 shots
            feat['away_corner_intensity'] = 0.4

        # Difference features
        feat['corners_attack_diff'] = home_attack - away_attack
        feat['corners_defense_diff'] = home_defense - away_defense
        feat['shots_diff'] = home_shots - away_shots


def build_corner_features(
    stats_df: pd.DataFrame,
    window_sizes: List[int] = [5, 10],
    min_matches: int = 3
) -> pd.DataFrame:
    """
    Convenience function to build corner features.

    Args:
        stats_df: DataFrame with match statistics
        window_sizes: Rolling window sizes
        min_matches: Minimum matches for valid stats

    Returns:
        DataFrame with corner features
    """
    engineer = CornerFeatureEngineer(
        window_sizes=window_sizes,
        min_matches=min_matches
    )
    return engineer.fit_transform(stats_df)
