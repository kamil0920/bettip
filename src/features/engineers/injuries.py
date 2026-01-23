"""
Historical injury feature engineering for training.

Creates features from historical injury data that was known BEFORE each match.
This allows training models that can use injury information.

Key insight: The API stores which players were injured FOR each fixture,
meaning we can reconstruct what was known pre-match.
"""
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.features.engineers.base import BaseFeatureEngineer

logger = logging.getLogger(__name__)


# Key players by team (high-impact absences)
# Based on market value, goals/assists contribution, and tactical importance
KEY_PLAYERS_BY_TEAM = {
    # Premier League
    33: ['Bruno Fernandes', 'Marcus Rashford', 'Casemiro', 'Kobbie Mainoo'],  # Man Utd
    40: ['Mohamed Salah', 'Virgil van Dijk', 'Trent Alexander-Arnold', 'Alisson'],  # Liverpool
    50: ['Erling Haaland', 'Kevin De Bruyne', 'Rodri', 'Bernardo Silva'],  # Man City
    42: ['Bukayo Saka', 'Martin Odegaard', 'William Saliba', 'Declan Rice'],  # Arsenal
    49: ['Cole Palmer', 'Enzo Fernandez', 'Reece James'],  # Chelsea
    47: ['Son Heung-min', 'James Maddison'],  # Tottenham
    34: ['Bruno Guimaraes', 'Alexander Isak', 'Anthony Gordon'],  # Newcastle
    66: ['Ollie Watkins', 'Emiliano Martinez', 'John McGinn'],  # Aston Villa
    48: ['Jarrod Bowen', 'Mohammed Kudus', 'Lucas Paqueta'],  # West Ham
    51: ['Kaoru Mitoma', 'Evan Ferguson'],  # Brighton

    # La Liga
    529: ['Jude Bellingham', 'Vinicius Junior', 'Kylian Mbappe'],  # Real Madrid
    530: ['Lamine Yamal', 'Pedri', 'Gavi', 'Robert Lewandowski'],  # Barcelona
    531: ['Antoine Griezmann', 'Alvaro Morata'],  # Atletico Madrid

    # Serie A
    489: ['Lautaro Martinez', 'Marcus Thuram'],  # Inter
    492: ['Rafael Leao', 'Theo Hernandez'],  # AC Milan
    496: ['Dusan Vlahovic', 'Federico Chiesa'],  # Juventus

    # Bundesliga
    157: ['Harry Kane', 'Jamal Musiala', 'Leroy Sane'],  # Bayern
    165: ['Florian Wirtz', 'Victor Boniface'],  # Leverkusen

    # Ligue 1
    85: ['Ousmane Dembele', 'Bradley Barcola', 'Vitinha'],  # PSG
}

# Position importance weights (for calculating injury impact)
POSITION_WEIGHTS = {
    'Goalkeeper': 0.8,  # Important but usually have good backup
    'Defender': 0.6,
    'Midfielder': 0.7,
    'Attacker': 0.9,  # Most impactful on scoring
}


class HistoricalInjuryFeatureEngineer(BaseFeatureEngineer):
    """
    Creates injury features from historical data for model training.

    Uses injury records that were known BEFORE each match to create
    features that can be used during training without data leakage.

    Features created:
    - Injury counts (home/away)
    - Key player absence flags
    - Injury severity scores
    - Position-based impact scores
    """

    def __init__(
        self,
        key_players: Optional[Dict[int, List[str]]] = None,
        min_injury_impact: float = 0.3,
    ):
        """
        Initialize feature engineer.

        Args:
            key_players: Dict mapping team_id to list of key player names
            min_injury_impact: Minimum impact score to flag as significant
        """
        self.key_players = key_players or KEY_PLAYERS_BY_TEAM
        self.min_injury_impact = min_injury_impact

    def create_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Create injury features for each match.

        Args:
            data: Dict containing:
                - 'matches': DataFrame with fixture_id, home_team_id, away_team_id
                - 'injuries': DataFrame with injury records per fixture

        Returns:
            DataFrame with injury features per fixture
        """
        matches = data.get('matches', pd.DataFrame())
        injuries = data.get('injuries', pd.DataFrame())

        if matches.empty:
            logger.warning("No matches provided for injury features")
            return pd.DataFrame()

        if injuries.empty:
            logger.warning("No injury data provided - using default values")
            return self._create_default_features(matches)

        features_list = []

        for _, match in matches.iterrows():
            fixture_id = match.get('fixture_id')
            home_team_id = match.get('home_team_id')
            away_team_id = match.get('away_team_id')

            # Get injuries for this fixture
            fixture_injuries = injuries[injuries['fixture_id'] == fixture_id]

            features = self._create_injury_features(
                fixture_id,
                home_team_id,
                away_team_id,
                fixture_injuries
            )

            features_list.append(features)

        result = pd.DataFrame(features_list)
        logger.info(f"Created injury features for {len(result)} matches")

        return result

    def _create_injury_features(
        self,
        fixture_id: int,
        home_team_id: int,
        away_team_id: int,
        injuries: pd.DataFrame
    ) -> Dict[str, Any]:
        """Create injury features for a single match."""

        features = {
            'fixture_id': fixture_id,
            # Basic counts
            'inj_home_count': 0,
            'inj_away_count': 0,
            'inj_total_count': 0,
            'inj_count_diff': 0,  # positive = home has more injuries (bad for home)
            # Key player flags
            'inj_home_key_player_out': 0,
            'inj_away_key_player_out': 0,
            'inj_home_key_player_count': 0,
            'inj_away_key_player_count': 0,
            # Injury types
            'inj_home_missing': 0,  # Confirmed out
            'inj_away_missing': 0,
            'inj_home_doubtful': 0,  # Questionable
            'inj_away_doubtful': 0,
            # Severity (based on reason)
            'inj_home_severe': 0,  # Long-term injuries
            'inj_away_severe': 0,
            # Derived
            'inj_advantage_home': 0.0,  # Positive = home has advantage (fewer injuries)
        }

        if injuries.empty:
            return features

        # Split by team
        home_injuries = injuries[injuries['team_id'] == home_team_id]
        away_injuries = injuries[injuries['team_id'] == away_team_id]

        # Basic counts
        features['inj_home_count'] = len(home_injuries)
        features['inj_away_count'] = len(away_injuries)
        features['inj_total_count'] = len(injuries)
        features['inj_count_diff'] = len(home_injuries) - len(away_injuries)

        # Process home team injuries
        home_stats = self._analyze_team_injuries(home_injuries, home_team_id)
        features['inj_home_key_player_out'] = home_stats['key_player_out']
        features['inj_home_key_player_count'] = home_stats['key_player_count']
        features['inj_home_missing'] = home_stats['missing_count']
        features['inj_home_doubtful'] = home_stats['doubtful_count']
        features['inj_home_severe'] = home_stats['severe_count']

        # Process away team injuries
        away_stats = self._analyze_team_injuries(away_injuries, away_team_id)
        features['inj_away_key_player_out'] = away_stats['key_player_out']
        features['inj_away_key_player_count'] = away_stats['key_player_count']
        features['inj_away_missing'] = away_stats['missing_count']
        features['inj_away_doubtful'] = away_stats['doubtful_count']
        features['inj_away_severe'] = away_stats['severe_count']

        # Calculate advantage (positive = home has fewer/less impactful injuries)
        home_impact = (
            features['inj_home_count'] * 0.3 +
            features['inj_home_key_player_count'] * 1.0 +
            features['inj_home_severe'] * 0.5
        )
        away_impact = (
            features['inj_away_count'] * 0.3 +
            features['inj_away_key_player_count'] * 1.0 +
            features['inj_away_severe'] * 0.5
        )
        features['inj_advantage_home'] = away_impact - home_impact

        return features

    def _analyze_team_injuries(
        self,
        injuries: pd.DataFrame,
        team_id: int
    ) -> Dict[str, Any]:
        """Analyze injuries for a single team."""

        stats = {
            'key_player_out': 0,
            'key_player_count': 0,
            'missing_count': 0,
            'doubtful_count': 0,
            'severe_count': 0,
        }

        if injuries.empty:
            return stats

        # Get key players for this team
        key_players = self.key_players.get(team_id, [])

        for _, inj in injuries.iterrows():
            player_name = inj.get('player_name', '')
            injury_type = str(inj.get('injury_type', '')).lower()
            injury_reason = str(inj.get('injury_reason', '')).lower()

            # Check if key player
            is_key_player = any(
                kp.lower() in player_name.lower()
                for kp in key_players
            )

            if is_key_player:
                stats['key_player_out'] = 1
                stats['key_player_count'] += 1

            # Categorize injury type
            if 'missing' in injury_type:
                stats['missing_count'] += 1
            elif 'doubt' in injury_type or 'question' in injury_type:
                stats['doubtful_count'] += 1

            # Check for severe injuries
            severe_keywords = ['acl', 'cruciate', 'achilles', 'fracture', 'surgery', 'broken']
            if any(kw in injury_reason for kw in severe_keywords):
                stats['severe_count'] += 1

        return stats

    def _create_default_features(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Create default features when no injury data available."""
        features_list = []

        for _, match in matches.iterrows():
            features = {
                'fixture_id': match.get('fixture_id'),
                'inj_home_count': 0,
                'inj_away_count': 0,
                'inj_total_count': 0,
                'inj_count_diff': 0,
                'inj_home_key_player_out': 0,
                'inj_away_key_player_out': 0,
                'inj_home_key_player_count': 0,
                'inj_away_key_player_count': 0,
                'inj_home_missing': 0,
                'inj_away_missing': 0,
                'inj_home_doubtful': 0,
                'inj_away_doubtful': 0,
                'inj_home_severe': 0,
                'inj_away_severe': 0,
                'inj_advantage_home': 0.0,
            }
            features_list.append(features)

        return pd.DataFrame(features_list)


def collect_injuries_for_training(
    league_id: int,
    season: int,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Collect historical injury data for training.

    Args:
        league_id: API-Football league ID
        season: Season year
        output_path: Optional path to save the data

    Returns:
        DataFrame with injury data
    """
    from src.data_collection.prematch_collector import PreMatchCollector

    collector = PreMatchCollector()
    injuries = collector.get_injuries_by_league(league_id, season)

    if output_path and not injuries.empty:
        injuries.to_parquet(output_path, index=False)
        logger.info(f"Saved {len(injuries)} injury records to {output_path}")

    return injuries
