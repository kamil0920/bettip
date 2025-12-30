"""
Download and process odds data from football-data.co.uk

Data source: https://www.football-data.co.uk/
Provides historical match data with odds from multiple bookmakers.
"""
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from io import StringIO

import pandas as pd
import requests

logger = logging.getLogger(__name__)


# League codes for football-data.co.uk
LEAGUE_CODES = {
    "premier_league": "E0",
    "championship": "E1",
    "league_one": "E2",
    "league_two": "E3",
    "la_liga": "SP1",
    "la_liga_2": "SP2",
    "serie_a": "I1",
    "serie_b": "I2",
    "bundesliga": "D1",
    "bundesliga_2": "D2",
    "ligue_1": "F1",
    "ligue_2": "F2",
    "eredivisie": "N1",
    "jupiler_league": "B1",
    "primeira_liga": "P1",
    "super_lig": "T1",
    "super_league": "G1",
}

# Columns we want to extract
ODDS_COLUMNS = {
    # Opening odds (Bet365)
    "B365H": "b365_home_open",
    "B365D": "b365_draw_open",
    "B365A": "b365_away_open",
    # Opening odds (Market max/avg)
    "MaxH": "max_home_open",
    "MaxD": "max_draw_open",
    "MaxA": "max_away_open",
    "AvgH": "avg_home_open",
    "AvgD": "avg_draw_open",
    "AvgA": "avg_away_open",
    # Closing odds (Bet365)
    "B365CH": "b365_home_close",
    "B365CD": "b365_draw_close",
    "B365CA": "b365_away_close",
    # Closing odds (Market max/avg)
    "MaxCH": "max_home_close",
    "MaxCD": "max_draw_close",
    "MaxCA": "max_away_close",
    "AvgCH": "avg_home_close",
    "AvgCD": "avg_draw_close",
    "AvgCA": "avg_away_close",
    # Over/Under 2.5 goals
    "B365>2.5": "b365_over25",
    "B365<2.5": "b365_under25",
    "Avg>2.5": "avg_over25",
    "Avg<2.5": "avg_under25",
    # Closing Over/Under
    "B365C>2.5": "b365_over25_close",
    "B365C<2.5": "b365_under25_close",
    "AvgC>2.5": "avg_over25_close",
    "AvgC<2.5": "avg_under25_close",
}

# Match info columns
MATCH_COLUMNS = {
    "Date": "date",
    "Time": "time",
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "FTHG": "home_goals",
    "FTAG": "away_goals",
    "FTR": "result",  # H, D, A
}


class FootballDataLoader:
    """
    Load odds data from football-data.co.uk

    Usage:
        loader = FootballDataLoader()
        df = loader.load_season("premier_league", 2024)
        df = loader.load_multiple_seasons("premier_league", [2023, 2024])
    """

    BASE_URL = "https://www.football-data.co.uk/mmz4281"

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize loader.

        Args:
            cache_dir: Directory to cache downloaded files. If None, no caching.
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_season_code(self, season: int) -> str:
        """
        Convert season year to football-data.co.uk format.

        E.g., 2024 -> "2425" (for 2024/25 season)
        """
        start = str(season)[-2:]
        end = str(season + 1)[-2:]
        return f"{start}{end}"

    def _get_url(self, league: str, season: int) -> str:
        """Build download URL for a specific league and season."""
        if league not in LEAGUE_CODES:
            raise ValueError(f"Unknown league: {league}. Available: {list(LEAGUE_CODES.keys())}")

        season_code = self._get_season_code(season)
        league_code = LEAGUE_CODES[league]
        return f"{self.BASE_URL}/{season_code}/{league_code}.csv"

    def _get_cache_path(self, league: str, season: int) -> Optional[Path]:
        """Get cache file path."""
        if not self.cache_dir:
            return None
        return self.cache_dir / f"{league}_{season}_odds.csv"

    def _download(self, url: str) -> str:
        """Download CSV content from URL."""
        logger.info(f"Downloading: {url}")

        response = requests.get(url, timeout=30)
        response.raise_for_status()

        # Handle BOM if present
        content = response.content.decode('utf-8-sig')
        return content

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date from various formats used by football-data.co.uk."""
        formats = [
            "%d/%m/%Y",  # Most common: 16/08/2024
            "%d/%m/%y",  # Older: 16/08/24
            "%Y-%m-%d",  # ISO format
        ]

        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue

        logger.warning(f"Could not parse date: {date_str}")
        return None

    def _process_dataframe(self, df: pd.DataFrame, league: str, season: int) -> pd.DataFrame:
        """Process raw CSV into standardized format."""

        # Select and rename columns
        available_match_cols = {k: v for k, v in MATCH_COLUMNS.items() if k in df.columns}
        available_odds_cols = {k: v for k, v in ODDS_COLUMNS.items() if k in df.columns}

        selected_cols = list(available_match_cols.keys()) + list(available_odds_cols.keys())
        df_selected = df[selected_cols].copy()

        # Rename columns
        df_selected = df_selected.rename(columns={**available_match_cols, **available_odds_cols})

        # Parse date
        df_selected['date'] = df_selected['date'].apply(self._parse_date)
        df_selected = df_selected.dropna(subset=['date'])

        # Add metadata
        df_selected['league'] = league
        df_selected['season'] = season

        # Normalize team names (strip whitespace, consistent casing)
        df_selected['home_team'] = df_selected['home_team'].str.strip()
        df_selected['away_team'] = df_selected['away_team'].str.strip()

        # Convert odds to numeric (exclude team name columns)
        exclude_from_numeric = {'home_team', 'away_team', 'date', 'time', 'result', 'league'}
        odds_cols = [c for c in df_selected.columns
                    if c not in exclude_from_numeric
                    and any(x in c for x in ['home', 'draw', 'away', 'over', 'under', 'max', 'avg', 'b365', 'goals'])]
        for col in odds_cols:
            df_selected[col] = pd.to_numeric(df_selected[col], errors='coerce')

        logger.info(f"Processed {len(df_selected)} matches for {league} {season}/{season+1}")

        return df_selected

    def load_season(
        self,
        league: str,
        season: int,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Load odds data for a single season.

        Args:
            league: League name (e.g., "premier_league")
            season: Season start year (e.g., 2024 for 2024/25)
            use_cache: Whether to use cached data if available

        Returns:
            DataFrame with odds data
        """
        cache_path = self._get_cache_path(league, season)

        # Try cache first
        if use_cache and cache_path and cache_path.exists():
            logger.info(f"Loading from cache: {cache_path}")
            return pd.read_csv(cache_path, parse_dates=['date'])

        # Download
        url = self._get_url(league, season)
        try:
            content = self._download(url)
        except requests.HTTPError as e:
            logger.error(f"Failed to download {url}: {e}")
            return pd.DataFrame()

        # Parse CSV
        df = pd.read_csv(StringIO(content))

        if df.empty:
            logger.warning(f"No data found for {league} {season}")
            return pd.DataFrame()

        # Process
        df_processed = self._process_dataframe(df, league, season)

        # Cache
        if cache_path:
            df_processed.to_csv(cache_path, index=False)
            logger.info(f"Cached to: {cache_path}")

        return df_processed

    def load_multiple_seasons(
        self,
        league: str,
        seasons: List[int],
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Load odds data for multiple seasons.

        Args:
            league: League name
            seasons: List of season start years
            use_cache: Whether to use cached data

        Returns:
            Combined DataFrame
        """
        dfs = []

        for season in seasons:
            df = self.load_season(league, season, use_cache)
            if not df.empty:
                dfs.append(df)

        if not dfs:
            return pd.DataFrame()

        combined = pd.concat(dfs, ignore_index=True)
        combined = combined.sort_values('date').reset_index(drop=True)

        logger.info(f"Loaded {len(combined)} total matches across {len(seasons)} seasons")

        return combined

    def get_available_columns(self, league: str, season: int) -> List[str]:
        """Check which odds columns are available for a given league/season."""
        url = self._get_url(league, season)
        content = self._download(url)
        df = pd.read_csv(StringIO(content), nrows=1)
        return list(df.columns)


# Team name mapping between football-data.co.uk and your existing data
TEAM_NAME_MAPPING = {
    # Premier League
    "Man United": "Manchester United",
    "Man City": "Manchester City",
    "Spurs": "Tottenham",
    "Tottenham": "Tottenham Hotspur",
    "Newcastle": "Newcastle United",
    "West Ham": "West Ham United",
    "Wolves": "Wolverhampton Wanderers",
    "Nott'm Forest": "Nottingham Forest",
    "Brighton": "Brighton & Hove Albion",
    "Sheffield United": "Sheffield Utd",
    "Luton": "Luton Town",
    "Leicester": "Leicester City",
    "Leeds": "Leeds United",
    "Norwich": "Norwich City",
    "Ipswich": "Ipswich Town",
    # La Liga
    "Ath Madrid": "Atletico Madrid",
    "Ath Bilbao": "Athletic Club",
    "Betis": "Real Betis",
    "Sociedad": "Real Sociedad",
    "Vallecano": "Rayo Vallecano",
    "Espanol": "Espanyol",
    "La Coruna": "Deportivo La Coruna",
    "Leganes": "CD Leganes",
    # Serie A
    "Inter": "Inter Milan",
    "AC Milan": "Milan",
    "Verona": "Hellas Verona",
    # Ligue 1
    "Paris SG": "Paris Saint Germain",
    "St Etienne": "Saint-Etienne",
}


def normalize_team_name(name: str, reverse: bool = False) -> str:
    """
    Normalize team name between football-data.co.uk and other sources.

    Args:
        name: Team name to normalize
        reverse: If True, convert from your format to football-data format

    Returns:
        Normalized team name
    """
    if reverse:
        # Create reverse mapping
        reverse_mapping = {v: k for k, v in TEAM_NAME_MAPPING.items()}
        return reverse_mapping.get(name, name)

    return TEAM_NAME_MAPPING.get(name, name)
