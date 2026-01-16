"""
Match importance feature engineering.

Creates features based on:
- Derby/rivalry matchups
- League position context (title race, relegation battle)
- Match stakes and importance scoring

Research shows match importance affects team performance and outcome variance.
"""

import logging
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Derby/Rivalry Mappings by Team ID
# Format: team_id -> set of rival team_ids
# Using API-Football team IDs

PREMIER_LEAGUE_DERBIES: Dict[int, Set[int]] = {
    # Manchester United (33) vs Manchester City (50) - Manchester Derby
    33: {50},
    50: {33},
    # Liverpool (40) vs Everton (45) - Merseyside Derby
    40: {45, 33},  # Also vs Man United - big rivalry
    45: {40},
    # Arsenal (42) vs Tottenham (47) - North London Derby
    42: {47, 49},  # Also vs Chelsea
    47: {42, 48},  # Also vs West Ham
    # Chelsea (49) vs Arsenal (42), Tottenham (47), Fulham (36)
    49: {42, 47, 36},
    # West Ham (48) vs Tottenham (47), Millwall (not in PL usually)
    48: {47},
    # Newcastle (34) vs Sunderland (not in PL), also rivalry with Everton
    34: {45},
    # Crystal Palace (52) vs Brighton (51) - M23 Derby
    52: {51},
    51: {52},
}

LA_LIGA_DERBIES: Dict[int, Set[int]] = {
    # Real Madrid (541) vs Barcelona (529) - El Clasico
    541: {529, 530},  # Also vs Atletico
    529: {541, 530},  # Also vs Atletico (Madrid)
    # Atletico Madrid (530) vs Real Madrid (541), Barcelona
    530: {541, 529},
    # Sevilla (536) vs Real Betis (543) - Seville Derby
    536: {543},
    543: {536},
    # Athletic Bilbao (531) vs Real Sociedad (548) - Basque Derby
    531: {548},
    548: {531},
    # Valencia (532) vs Villarreal (533) - Valencian Community Derby
    532: {533},
    533: {532},
}

SERIE_A_DERBIES: Dict[int, Set[int]] = {
    # AC Milan (489) vs Inter Milan (505) - Derby della Madonnina
    489: {505, 496},  # Also vs Juventus
    505: {489, 496},  # Also vs Juventus
    # Juventus (496) vs Torino (503) - Derby della Mole
    496: {503, 489, 505},  # Also vs Milan clubs
    503: {496},
    # Roma (497) vs Lazio (487) - Derby della Capitale
    497: {487},
    487: {497},
    # Napoli (492) - no major local derby in Serie A currently
    # Genoa (495) vs Sampdoria (498) - Derby della Lanterna
    495: {498},
    498: {495},
    # Fiorentina (502) vs Siena/Empoli
    502: {511},  # Empoli
}

BUNDESLIGA_DERBIES: Dict[int, Set[int]] = {
    # Bayern Munich (157) vs Borussia Dortmund (165) - Der Klassiker
    157: {165, 168},  # Also vs 1860 Munich (not in BL usually)
    165: {157, 169},  # Also vs Schalke - Revierderby
    # Borussia Dortmund (165) vs Schalke 04 (169) - Revierderby
    169: {165},
    # Hamburg (180) vs Bremen (134) - Nordderby
    180: {134},
    134: {180},
    # Cologne (192) vs Gladbach (163) - Rhine Derby
    192: {163},
    163: {192},
}

LIGUE_1_DERBIES: Dict[int, Set[int]] = {
    # PSG (85) vs Marseille (81) - Le Classique
    85: {81},
    81: {85, 80},  # Also vs Monaco
    # Lyon (80) vs Saint-Etienne (1034) - Derby Rhone-Alpes
    80: {1034, 81},
    1034: {80},
    # Monaco (91) vs Nice (84) - Cote d'Azur Derby
    91: {84},
    84: {91},
}

# Combine all derbies
ALL_DERBIES: Dict[int, Set[int]] = {}
for derby_dict in [PREMIER_LEAGUE_DERBIES, LA_LIGA_DERBIES, SERIE_A_DERBIES,
                   BUNDESLIGA_DERBIES, LIGUE_1_DERBIES]:
    for team_id, rivals in derby_dict.items():
        if team_id in ALL_DERBIES:
            ALL_DERBIES[team_id].update(rivals)
        else:
            ALL_DERBIES[team_id] = rivals.copy()


class MatchImportanceEngineer:
    """
    Create match importance features.

    Features created:
    1. is_derby - Binary flag for local/historic rivalries
    2. relegation_battle - Both teams in bottom positions
    3. title_race - Both teams competing for title
    4. european_race - Teams competing for European spots
    5. position_gap - Absolute difference in league position
    6. match_importance_score - Combined importance metric (0-5)
    """

    def __init__(
        self,
        relegation_zone: int = 3,
        title_zone: int = 3,
        european_zone: int = 6,
        custom_derbies: Optional[Dict[int, Set[int]]] = None,
    ):
        """
        Initialize match importance engineer.

        Args:
            relegation_zone: Number of bottom positions considered relegation
            title_zone: Number of top positions considered title race
            european_zone: Number of positions for European qualification
            custom_derbies: Optional custom derby mapping to add
        """
        self.relegation_zone = relegation_zone
        self.title_zone = title_zone
        self.european_zone = european_zone

        # Build derby lookup
        self.derbies = ALL_DERBIES.copy()
        if custom_derbies:
            for team_id, rivals in custom_derbies.items():
                if team_id in self.derbies:
                    self.derbies[team_id].update(rivals)
                else:
                    self.derbies[team_id] = rivals.copy()

    def _is_derby(self, home_team_id: int, away_team_id: int) -> int:
        """Check if match is a derby."""
        if home_team_id in self.derbies:
            if away_team_id in self.derbies[home_team_id]:
                return 1
        if away_team_id in self.derbies:
            if home_team_id in self.derbies[away_team_id]:
                return 1
        return 0

    def _get_league_size(self, df: pd.DataFrame, idx: int) -> int:
        """Estimate league size from position data."""
        # Use max position seen as proxy for league size
        return max(
            df.loc[idx, 'home_league_position'] if pd.notna(df.loc[idx, 'home_league_position']) else 20,
            df.loc[idx, 'away_league_position'] if pd.notna(df.loc[idx, 'away_league_position']) else 20,
            20  # Default minimum
        )

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all match importance features.

        Args:
            df: DataFrame with team IDs and league positions

        Returns:
            DataFrame with importance features added
        """
        df = df.copy()

        # Check required columns
        required_cols = ['home_team_id', 'away_team_id']
        position_cols = ['home_league_position', 'away_league_position']

        has_ids = all(col in df.columns for col in required_cols)
        has_positions = all(col in df.columns for col in position_cols)

        if not has_ids:
            logger.warning("Missing team ID columns, skipping derby detection")
            df['is_derby'] = 0
        else:
            # 1. Derby detection
            df['is_derby'] = df.apply(
                lambda row: self._is_derby(
                    int(row['home_team_id']) if pd.notna(row['home_team_id']) else 0,
                    int(row['away_team_id']) if pd.notna(row['away_team_id']) else 0
                ),
                axis=1
            )
            logger.info(f"Found {df['is_derby'].sum()} derbies in {len(df)} matches")

        if not has_positions:
            logger.warning("Missing position columns, skipping position-based features")
            df['relegation_battle'] = 0
            df['title_race'] = 0
            df['european_race'] = 0
            df['mid_table_clash'] = 0
            df['position_gap'] = np.nan
            df['match_importance_score'] = df['is_derby'].astype(float)
        else:
            home_pos = df['home_league_position']
            away_pos = df['away_league_position']

            # Estimate league size (typically 18-20)
            league_size = df[['home_league_position', 'away_league_position']].max(axis=1).fillna(20)
            league_size = league_size.clip(lower=18, upper=24)

            # 2. Relegation battle (both teams in bottom zone)
            relegation_threshold = league_size - self.relegation_zone + 1
            df['relegation_battle'] = (
                (home_pos >= relegation_threshold) &
                (away_pos >= relegation_threshold)
            ).astype(int)

            # 3. Title race (both teams in top zone)
            df['title_race'] = (
                (home_pos <= self.title_zone) &
                (away_pos <= self.title_zone)
            ).astype(int)

            # 4. European race (both teams competing for European spots)
            df['european_race'] = (
                (home_pos <= self.european_zone) &
                (away_pos <= self.european_zone) &
                ~((home_pos <= self.title_zone) & (away_pos <= self.title_zone))  # Exclude title race
            ).astype(int)

            # 5. Mid-table clash (neither team in contention)
            mid_lower = self.european_zone + 1
            mid_upper = league_size - self.relegation_zone
            df['mid_table_clash'] = (
                (home_pos > mid_lower) & (home_pos < mid_upper) &
                (away_pos > mid_lower) & (away_pos < mid_upper)
            ).astype(int)

            # 6. Position gap (absolute difference)
            df['position_gap'] = (home_pos - away_pos).abs()

            # 7. Underdog playing up (lower positioned team is away)
            df['away_underdog'] = (away_pos > home_pos).astype(int)

            # 8. Giant killer potential (big position gap, underdog away)
            df['giant_killer_potential'] = (
                (df['position_gap'] >= 10) &
                (away_pos > home_pos)
            ).astype(int)

            # 9. Six-pointer (teams close in standings in important zone)
            df['six_pointer'] = (
                (df['position_gap'] <= 3) &
                (
                    df['relegation_battle'] |
                    df['title_race'] |
                    df['european_race']
                )
            ).astype(int)

            # 10. Combined match importance score (0-5)
            df['match_importance_score'] = (
                df['is_derby'].astype(float) * 1.5 +  # Derbies are always important
                df['relegation_battle'].astype(float) * 1.5 +  # High stakes
                df['title_race'].astype(float) * 1.5 +  # High stakes
                df['european_race'].astype(float) * 1.0 +
                df['six_pointer'].astype(float) * 0.5
            ).clip(upper=5.0)

        logger.info(f"Created {len(self.get_feature_names())} match importance features")

        return df

    def get_feature_names(self) -> List[str]:
        """Return list of feature names created by this engineer."""
        return [
            'is_derby',
            'relegation_battle',
            'title_race',
            'european_race',
            'mid_table_clash',
            'position_gap',
            'away_underdog',
            'giant_killer_potential',
            'six_pointer',
            'match_importance_score',
        ]


def get_derby_teams_for_league(league: str) -> Dict[int, Set[int]]:
    """Get derby mappings for a specific league."""
    league_map = {
        'premier_league': PREMIER_LEAGUE_DERBIES,
        'la_liga': LA_LIGA_DERBIES,
        'serie_a': SERIE_A_DERBIES,
        'bundesliga': BUNDESLIGA_DERBIES,
        'ligue_1': LIGUE_1_DERBIES,
    }
    return league_map.get(league.lower(), {})


def add_derby_by_name(
    df: pd.DataFrame,
    derby_names: Dict[str, List[str]],
    home_col: str = 'home_team_name',
    away_col: str = 'away_team_name',
) -> pd.DataFrame:
    """
    Add derby flag using team names instead of IDs.

    Useful when team IDs are not available.

    Args:
        df: DataFrame with team names
        derby_names: Dict mapping team name to list of rival names
        home_col: Column name for home team
        away_col: Column name for away team

    Returns:
        DataFrame with is_derby_by_name column added
    """
    df = df.copy()

    def check_derby(home: str, away: str) -> int:
        if pd.isna(home) or pd.isna(away):
            return 0
        home_lower = home.lower()
        away_lower = away.lower()

        for team, rivals in derby_names.items():
            team_lower = team.lower()
            rivals_lower = [r.lower() for r in rivals]

            if team_lower in home_lower:
                if any(r in away_lower for r in rivals_lower):
                    return 1
            if team_lower in away_lower:
                if any(r in home_lower for r in rivals_lower):
                    return 1
        return 0

    df['is_derby_by_name'] = df.apply(
        lambda row: check_derby(row.get(home_col), row.get(away_col)),
        axis=1
    )

    return df


# Common derby names for fallback detection
DERBY_NAMES: Dict[str, List[str]] = {
    # England
    'Manchester United': ['Manchester City'],
    'Manchester City': ['Manchester United'],
    'Liverpool': ['Everton', 'Manchester United'],
    'Everton': ['Liverpool'],
    'Arsenal': ['Tottenham', 'Chelsea'],
    'Tottenham': ['Arsenal', 'West Ham', 'Chelsea'],
    'Chelsea': ['Arsenal', 'Tottenham', 'Fulham'],
    'Crystal Palace': ['Brighton'],
    'Brighton': ['Crystal Palace'],
    # Spain
    'Real Madrid': ['Barcelona', 'Atletico Madrid'],
    'Barcelona': ['Real Madrid', 'Espanyol'],
    'Atletico Madrid': ['Real Madrid'],
    'Sevilla': ['Real Betis'],
    'Real Betis': ['Sevilla'],
    # Italy
    'AC Milan': ['Inter', 'Juventus'],
    'Inter': ['AC Milan', 'Juventus'],
    'Juventus': ['Torino', 'AC Milan', 'Inter'],
    'Torino': ['Juventus'],
    'Roma': ['Lazio'],
    'Lazio': ['Roma'],
    'Genoa': ['Sampdoria'],
    'Sampdoria': ['Genoa'],
    # Germany
    'Bayern': ['Dortmund', '1860'],
    'Dortmund': ['Bayern', 'Schalke'],
    'Schalke': ['Dortmund'],
    'Cologne': ['Gladbach', 'Monchengladbach'],
    'Gladbach': ['Cologne'],
    # France
    'PSG': ['Marseille'],
    'Paris Saint': ['Marseille'],
    'Marseille': ['PSG', 'Paris Saint', 'Lyon'],
    'Lyon': ['Saint-Etienne', 'Marseille'],
    'Saint-Etienne': ['Lyon'],
    'Monaco': ['Nice'],
    'Nice': ['Monaco'],
}
