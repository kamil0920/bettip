"""
Team Name Normalizer

Provides consistent team name mapping between different data sources:
- API-Football (used in features)
- Understat (used for xG data)
- Football-Data.co.uk (used for odds)

Usage:
    from src.utils.team_normalizer import TeamNormalizer

    normalizer = TeamNormalizer()
    canonical_name = normalizer.normalize('Wolves')  # Returns 'Wolverhampton Wanderers'
    xg_name = normalizer.to_understat('Wolves')  # Returns 'Wolverhampton Wanderers'
"""

from typing import Dict, Optional


class TeamNormalizer:
    """Normalize team names across different data sources."""

    # Map API-Football names to canonical Understat names
    # Format: 'api_football_name': 'understat_name'
    API_TO_UNDERSTAT: Dict[str, str] = {
        # Premier League
        'Wolves': 'Wolverhampton Wanderers',
        'Newcastle': 'Newcastle United',
        'Sheffield Utd': 'Sheffield United',
        'West Brom': 'West Bromwich Albion',

        # La Liga
        'Granada CF': 'Granada',
        'Huesca': 'SD Huesca',
        'Valladolid': 'Real Valladolid',
        'Oviedo': 'Real Oviedo',

        # Serie A
        'AS Roma': 'Roma',
        'Parma': 'Parma Calcio 1913',
        'Spal': 'SPAL 2013',

        # Bundesliga
        '1. FC Köln': 'FC Cologne',
        '1. FC Heidenheim': 'FC Heidenheim',
        '1899 Hoffenheim': 'Hoffenheim',
        'Borussia Monchengladbach': 'Borussia M.Gladbach',
        'Borussia Mönchengladbach': 'Borussia M.Gladbach',
        'FC Augsburg': 'Augsburg',
        'FC Schalke 04': 'Schalke 04',
        'FC St. Pauli': 'St. Pauli',
        'FSV Mainz 05': 'Mainz 05',
        'RB Leipzig': 'RasenBallsport Leipzig',
        'SC Freiburg': 'Freiburg',
        'SV Darmstadt 98': 'Darmstadt',
        'SpVgg Greuther Furth': 'Greuther Fuerth',
        'VfB Stuttgart': 'VfB Stuttgart',  # Same
        'VfL Bochum': 'Bochum',
        'Vfl Bochum': 'Bochum',
        'VfL Wolfsburg': 'Wolfsburg',
        'Fortuna Dusseldorf': 'Fortuna Duesseldorf',
        'SV Elversberg': 'Elversberg',  # May not be in Understat (Bundesliga 2)

        # Ligue 1
        'Stade Brestois 29': 'Brest',
        'Estac Troyes': 'Troyes',
        'Saint Etienne': 'Saint-Etienne',
        'Paris FC': 'Paris FC',  # Same

        # Bayern Munich variations
        'Bayern München': 'Bayern Munich',
    }

    # Reverse mapping: Understat -> API-Football
    UNDERSTAT_TO_API: Dict[str, str] = {v: k for k, v in API_TO_UNDERSTAT.items()}

    # Football-Data.co.uk mappings (for odds data)
    FOOTBALL_DATA_TO_CANONICAL: Dict[str, str] = {
        'Man United': 'Manchester United',
        'Man City': 'Manchester City',
        'Wolves': 'Wolverhampton Wanderers',
        'Newcastle': 'Newcastle United',
        'Sheffield United': 'Sheffield United',
        'West Brom': 'West Bromwich Albion',
        'Nott\'m Forest': 'Nottingham Forest',
        'Sp Gijon': 'Sporting Gijon',
        'Ath Madrid': 'Atletico Madrid',
        'Ath Bilbao': 'Athletic Club',
        'Espanol': 'Espanyol',
        'Sociedad': 'Real Sociedad',
        'La Coruna': 'Deportivo La Coruna',
        'Betis': 'Real Betis',
        'Vallecano': 'Rayo Vallecano',
        'Leverkusen': 'Bayer Leverkusen',
        'Dortmund': 'Borussia Dortmund',
        "M'gladbach": 'Borussia M.Gladbach',
        'Mainz': 'Mainz 05',
        'Hertha': 'Hertha Berlin',
        'FC Koln': 'FC Cologne',
        'Ein Frankfurt': 'Eintracht Frankfurt',
        'Fortuna Dusseldorf': 'Fortuna Duesseldorf',
        'Paderborn': 'Paderborn',
        'St Pauli': 'St. Pauli',
        'St Etienne': 'Saint-Etienne',
        'Paris SG': 'Paris Saint Germain',
    }

    def __init__(self):
        """Initialize the normalizer with all mappings."""
        # Create comprehensive canonical name set from Understat names
        # (they tend to be the most complete/consistent)
        self._all_mappings = {}

        # Add API -> Understat mappings
        for api_name, understat_name in self.API_TO_UNDERSTAT.items():
            self._all_mappings[api_name.lower()] = understat_name

        # Add Football-Data -> canonical mappings
        for fd_name, canonical in self.FOOTBALL_DATA_TO_CANONICAL.items():
            self._all_mappings[fd_name.lower()] = canonical

    def normalize(self, team_name: str) -> str:
        """
        Normalize a team name to canonical form (Understat standard).

        Args:
            team_name: Team name from any source

        Returns:
            Canonical team name (Understat format)
        """
        if not team_name or not isinstance(team_name, str):
            return team_name

        # Check if it needs mapping
        lookup = team_name.lower().strip()
        if lookup in self._all_mappings:
            return self._all_mappings[lookup]

        # Return original if no mapping needed
        return team_name

    def to_understat(self, api_football_name: str) -> str:
        """
        Convert API-Football team name to Understat format.

        Args:
            api_football_name: Team name from API-Football

        Returns:
            Understat team name
        """
        return self.API_TO_UNDERSTAT.get(api_football_name, api_football_name)

    def from_understat(self, understat_name: str) -> str:
        """
        Convert Understat team name to API-Football format.

        Args:
            understat_name: Team name from Understat

        Returns:
            API-Football team name (first match if multiple)
        """
        return self.UNDERSTAT_TO_API.get(understat_name, understat_name)

    def get_all_variants(self, team_name: str) -> list:
        """
        Get all known variants of a team name.

        Args:
            team_name: Any team name

        Returns:
            List of all known variants including the input
        """
        variants = {team_name}
        canonical = self.normalize(team_name)
        variants.add(canonical)

        # Find all keys that map to the same canonical name
        for key, value in self._all_mappings.items():
            if value == canonical:
                variants.add(key)

        return list(variants)


# Singleton instance for convenience
_normalizer = TeamNormalizer()


def normalize_team(team_name: str) -> str:
    """Convenience function to normalize a team name."""
    return _normalizer.normalize(team_name)


def to_understat(api_football_name: str) -> str:
    """Convenience function to convert API-Football name to Understat."""
    return _normalizer.to_understat(api_football_name)


def from_understat(understat_name: str) -> str:
    """Convenience function to convert Understat name to API-Football."""
    return _normalizer.from_understat(understat_name)
