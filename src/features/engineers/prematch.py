"""
Pre-match feature engineering from API intelligence data.

Creates features from:
1. Injuries - Key player absences
2. Lineups - Formation changes, tactical shifts
3. Predictions - API predictions, team comparisons
4. H2H - Head-to-head history

These features are designed to be collected just before match prediction
and combined with historical features.
"""
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.features.engineers.base import BaseFeatureEngineer

logger = logging.getLogger(__name__)


class PreMatchFeatureEngineer(BaseFeatureEngineer):
    """
    Creates features from pre-match intelligence data.

    Features created:
    - Injury features (count, key player flags)
    - Lineup features (formation, tactical changes)
    - Prediction features (win %, comparisons)
    - H2H features (historical matchup stats)
    """

    # Top players by market value/importance (to be expanded)
    # These players missing significantly impacts team performance
    KEY_PLAYERS = {
        # Premier League
        33: ['Bruno Fernandes', 'Marcus Rashford', 'Casemiro'],  # Man Utd
        40: ['Mohamed Salah', 'Virgil van Dijk', 'Trent Alexander-Arnold'],  # Liverpool
        50: ['Erling Haaland', 'Kevin De Bruyne', 'Rodri'],  # Man City
        47: ['Harry Kane', 'Son Heung-min'],  # Tottenham (legacy)
        42: ['Bukayo Saka', 'Martin Odegaard', 'William Saliba'],  # Arsenal
        49: ['Cole Palmer', 'Enzo Fernandez'],  # Chelsea
        # Add more teams as needed
    }

    def __init__(
        self,
        key_players: Optional[Dict[int, List[str]]] = None,
        injury_weight: float = 1.0,
    ):
        """
        Initialize feature engineer.

        Args:
            key_players: Dict mapping team_id to list of key player names
            injury_weight: Weight multiplier for injury features
        """
        self.key_players = key_players or self.KEY_PLAYERS
        self.injury_weight = injury_weight

    def create_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Create pre-match features from collected data.

        Args:
            data: Dict containing:
                - 'matches': DataFrame with fixture_id, home_team_id, away_team_id
                - 'prematch': Dict mapping fixture_id to pre-match data

        Returns:
            DataFrame with pre-match features
        """
        matches = data.get('matches', pd.DataFrame())
        prematch_data = data.get('prematch', {})

        if matches.empty:
            logger.warning("No matches provided for pre-match features")
            return pd.DataFrame()

        features_list = []

        for _, match in matches.iterrows():
            fixture_id = match.get('fixture_id')
            home_team_id = match.get('home_team_id')
            away_team_id = match.get('away_team_id')

            prematch = prematch_data.get(fixture_id, {})

            features = {
                'fixture_id': fixture_id,
            }

            # Add injury features
            features.update(self._create_injury_features(
                prematch.get('injuries', pd.DataFrame()),
                home_team_id,
                away_team_id
            ))

            # Add lineup features
            features.update(self._create_lineup_features(
                prematch.get('lineups', {})
            ))

            # Add prediction features
            features.update(self._create_prediction_features(
                prematch.get('predictions', {})
            ))

            # Add H2H features
            features.update(self._create_h2h_features(
                prematch.get('h2h_summary', {})
            ))

            features_list.append(features)

        return pd.DataFrame(features_list)

    def _create_injury_features(
        self,
        injuries: pd.DataFrame,
        home_team_id: int,
        away_team_id: int
    ) -> Dict[str, Any]:
        """Create features from injury data."""
        features = {
            # Injury counts
            'pm_home_injuries': 0,
            'pm_away_injuries': 0,
            'pm_total_injuries': 0,
            'pm_injury_diff': 0,
            # Key player flags
            'pm_home_key_player_out': 0,
            'pm_away_key_player_out': 0,
            # Injury types
            'pm_home_missing_fixture': 0,
            'pm_away_missing_fixture': 0,
            'pm_home_doubtful': 0,
            'pm_away_doubtful': 0,
        }

        if injuries.empty:
            return features

        # Count injuries per team
        home_injuries = injuries[injuries['team_id'] == home_team_id]
        away_injuries = injuries[injuries['team_id'] == away_team_id]

        features['pm_home_injuries'] = len(home_injuries)
        features['pm_away_injuries'] = len(away_injuries)
        features['pm_total_injuries'] = len(injuries)
        features['pm_injury_diff'] = len(home_injuries) - len(away_injuries)

        # Check for key players
        home_key = self.key_players.get(home_team_id, [])
        away_key = self.key_players.get(away_team_id, [])

        if not home_injuries.empty and home_key:
            home_injured_names = home_injuries['player_name'].tolist()
            for player in home_key:
                if any(player.lower() in name.lower() for name in home_injured_names):
                    features['pm_home_key_player_out'] = 1
                    break

        if not away_injuries.empty and away_key:
            away_injured_names = away_injuries['player_name'].tolist()
            for player in away_key:
                if any(player.lower() in name.lower() for name in away_injured_names):
                    features['pm_away_key_player_out'] = 1
                    break

        # Injury types
        if not home_injuries.empty:
            features['pm_home_missing_fixture'] = (
                home_injuries['injury_type'].str.contains('Missing', case=False, na=False).sum()
            )
            features['pm_home_doubtful'] = (
                home_injuries['injury_type'].str.contains('Doubtful', case=False, na=False).sum()
            )

        if not away_injuries.empty:
            features['pm_away_missing_fixture'] = (
                away_injuries['injury_type'].str.contains('Missing', case=False, na=False).sum()
            )
            features['pm_away_doubtful'] = (
                away_injuries['injury_type'].str.contains('Doubtful', case=False, na=False).sum()
            )

        return features

    def _create_lineup_features(self, lineups: Dict[str, Any]) -> Dict[str, Any]:
        """Create features from lineup data."""
        features = {
            'pm_lineups_available': 0,
            'pm_home_formation': None,
            'pm_away_formation': None,
            'pm_formation_diff': None,
            # Formation category flags
            'pm_home_attacking_formation': 0,
            'pm_away_attacking_formation': 0,
            'pm_home_defensive_formation': 0,
            'pm_away_defensive_formation': 0,
        }

        if not lineups or not lineups.get('available'):
            return features

        features['pm_lineups_available'] = 1

        home = lineups.get('home', {})
        away = lineups.get('away', {})

        if home:
            features['pm_home_formation'] = home.get('formation')
            features['pm_home_attacking_formation'] = self._is_attacking_formation(
                home.get('formation')
            )
            features['pm_home_defensive_formation'] = self._is_defensive_formation(
                home.get('formation')
            )

        if away:
            features['pm_away_formation'] = away.get('formation')
            features['pm_away_attacking_formation'] = self._is_attacking_formation(
                away.get('formation')
            )
            features['pm_away_defensive_formation'] = self._is_defensive_formation(
                away.get('formation')
            )

        # Formation comparison
        if home and away and home.get('formation') and away.get('formation'):
            home_attackers = self._count_attackers(home.get('formation'))
            away_attackers = self._count_attackers(away.get('formation'))
            features['pm_formation_diff'] = home_attackers - away_attackers

        return features

    def _is_attacking_formation(self, formation: Optional[str]) -> int:
        """Check if formation is attacking (3+ forwards)."""
        if not formation:
            return 0
        attacking = ['4-3-3', '3-4-3', '4-2-4', '3-3-4', '4-1-2-3']
        return 1 if any(f in formation for f in attacking) else 0

    def _is_defensive_formation(self, formation: Optional[str]) -> int:
        """Check if formation is defensive (5 at back or 2 DMs)."""
        if not formation:
            return 0
        defensive = ['5-4-1', '5-3-2', '4-5-1', '5-2-3', '6-3-1']
        return 1 if any(f in formation for f in defensive) else 0

    def _count_attackers(self, formation: Optional[str]) -> int:
        """Count attackers in formation string."""
        if not formation:
            return 0
        try:
            parts = formation.split('-')
            return int(parts[-1])  # Last number is typically forwards
        except:
            return 0

    def _create_prediction_features(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Create features from API predictions."""
        features = {
            'pm_predictions_available': 0,
            # Win probabilities
            'pm_pred_home_pct': 0.33,
            'pm_pred_draw_pct': 0.33,
            'pm_pred_away_pct': 0.33,
            # Comparison metrics
            'pm_comp_form_home': 0.5,
            'pm_comp_att_home': 0.5,
            'pm_comp_def_home': 0.5,
            'pm_comp_poisson_home': 0.5,
            'pm_comp_h2h_home': 0.5,
            'pm_comp_total_home': 0.5,
            # Team form (last 5)
            'pm_home_form_points': 7.5,  # Average
            'pm_away_form_points': 7.5,
            'pm_form_points_diff': 0,
            # Goals averages
            'pm_home_goals_for_avg': 1.35,
            'pm_home_goals_against_avg': 1.35,
            'pm_away_goals_for_avg': 1.35,
            'pm_away_goals_against_avg': 1.35,
            # Derived
            'pm_expected_total_goals': 2.7,
            'pm_home_xg_diff': 0,
            'pm_away_xg_diff': 0,
        }

        if not predictions:
            return features

        features['pm_predictions_available'] = 1

        def parse_pct(pct_str: str) -> float:
            if not pct_str:
                return 0.0
            try:
                return float(pct_str.replace('%', '')) / 100
            except:
                return 0.0

        # Win probabilities
        percent = predictions.get('percent', {})
        features['pm_pred_home_pct'] = parse_pct(percent.get('home', '33%'))
        features['pm_pred_draw_pct'] = parse_pct(percent.get('draw', '33%'))
        features['pm_pred_away_pct'] = parse_pct(percent.get('away', '33%'))

        # Comparison metrics
        comparison = predictions.get('comparison', {})
        for metric in ['form', 'att', 'def', 'poisson_distribution', 'h2h', 'total']:
            if metric in comparison:
                key = 'poisson' if metric == 'poisson_distribution' else metric
                features[f'pm_comp_{key}_home'] = parse_pct(
                    comparison[metric].get('home', '50%')
                )

        # Team form
        teams = predictions.get('teams', {})
        for side in ['home', 'away']:
            team = teams.get(side, {})
            form = team.get('form', '')

            last_5 = form[-5:] if form else ''
            points = last_5.count('W') * 3 + last_5.count('D')
            features[f'pm_{side}_form_points'] = points

            # Goals
            goals_for = team.get('goals_for', {})
            goals_against = team.get('goals_against', {})

            try:
                features[f'pm_{side}_goals_for_avg'] = float(
                    goals_for.get('average', {}).get('total', '1.35')
                )
            except:
                pass

            try:
                features[f'pm_{side}_goals_against_avg'] = float(
                    goals_against.get('average', {}).get('total', '1.35')
                )
            except:
                pass

        # Derived features
        features['pm_form_points_diff'] = (
            features['pm_home_form_points'] - features['pm_away_form_points']
        )
        features['pm_expected_total_goals'] = (
            features['pm_home_goals_for_avg'] +
            features['pm_away_goals_for_avg']
        )
        features['pm_home_xg_diff'] = (
            features['pm_home_goals_for_avg'] -
            features['pm_home_goals_against_avg']
        )
        features['pm_away_xg_diff'] = (
            features['pm_away_goals_for_avg'] -
            features['pm_away_goals_against_avg']
        )

        return features

    def _create_h2h_features(self, h2h_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Create features from H2H summary."""
        features = {
            'pm_h2h_available': 0,
            'pm_h2h_matches': 0,
            'pm_h2h_home_wins': 0,
            'pm_h2h_away_wins': 0,
            'pm_h2h_draws': 0,
            'pm_h2h_home_win_pct': 0.33,
            'pm_h2h_avg_goals': 2.7,
        }

        if not h2h_summary:
            return features

        features['pm_h2h_available'] = 1
        features['pm_h2h_matches'] = h2h_summary.get('matches', 0)
        features['pm_h2h_home_wins'] = h2h_summary.get('team1_wins', 0)
        features['pm_h2h_away_wins'] = h2h_summary.get('team2_wins', 0)
        features['pm_h2h_draws'] = h2h_summary.get('draws', 0)
        features['pm_h2h_home_win_pct'] = h2h_summary.get('team1_win_pct', 0.33)
        features['pm_h2h_avg_goals'] = h2h_summary.get('avg_total_goals', 2.7)

        return features


class InjuryImpactFeatureEngineer(BaseFeatureEngineer):
    """
    Creates features focused on injury impact analysis.

    Uses player market values and team dependency to estimate
    the impact of missing players.
    """

    def create_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create injury impact features."""
        matches = data.get('matches', pd.DataFrame())
        injuries = data.get('injuries', pd.DataFrame())
        player_stats = data.get('player_stats', pd.DataFrame())

        if matches.empty:
            return pd.DataFrame()

        # Implementation for detailed injury impact
        # This would use player stats to calculate impact scores
        # For now, returns basic structure
        features_list = []

        for _, match in matches.iterrows():
            features = {
                'fixture_id': match.get('fixture_id'),
                'injury_impact_home': 0.0,
                'injury_impact_away': 0.0,
                'injury_impact_diff': 0.0,
            }
            features_list.append(features)

        return pd.DataFrame(features_list)


# =============================================================================
# Utility functions
# =============================================================================

def create_prematch_features_for_fixture(
    fixture_id: int,
    home_team_id: int,
    away_team_id: int,
    prematch_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create all pre-match features for a single fixture.

    Convenience function for real-time prediction pipelines.

    Args:
        fixture_id: The fixture ID
        home_team_id: Home team ID
        away_team_id: Away team ID
        prematch_data: Pre-match data from PreMatchCollector

    Returns:
        Dict of feature values
    """
    engineer = PreMatchFeatureEngineer()

    matches = pd.DataFrame([{
        'fixture_id': fixture_id,
        'home_team_id': home_team_id,
        'away_team_id': away_team_id,
    }])

    data = {
        'matches': matches,
        'prematch': {fixture_id: prematch_data}
    }

    features_df = engineer.create_features(data)

    if features_df.empty:
        return {}

    return features_df.iloc[0].to_dict()
