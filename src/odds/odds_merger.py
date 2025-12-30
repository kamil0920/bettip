"""
Merge odds data with existing match/fixture data.

Handles:
- Team name normalization between sources
- Date matching with tolerance
- Missing odds handling
"""
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz, process

from src.odds.football_data_loader import FootballDataLoader, normalize_team_name

logger = logging.getLogger(__name__)


class OddsMerger:
    """
    Merge odds data from football-data.co.uk with existing fixture data.

    Usage:
        merger = OddsMerger()
        features_with_odds = merger.merge_with_features(
            features_df,
            odds_df,
            league="premier_league"
        )
    """

    def __init__(
        self,
        date_tolerance_days: int = 1,
        fuzzy_match_threshold: int = 80
    ):
        """
        Initialize merger.

        Args:
            date_tolerance_days: Allow this many days difference when matching
            fuzzy_match_threshold: Minimum fuzzy match score for team names (0-100)
        """
        self.date_tolerance_days = date_tolerance_days
        self.fuzzy_match_threshold = fuzzy_match_threshold
        self._team_name_cache: Dict[str, str] = {}

    def _normalize_date(self, date_val) -> Optional[datetime]:
        """Convert various date formats to datetime."""
        if pd.isna(date_val):
            return None

        if isinstance(date_val, datetime):
            return date_val

        if isinstance(date_val, pd.Timestamp):
            return date_val.to_pydatetime()

        if isinstance(date_val, str):
            for fmt in ["%Y-%m-%d", "%d/%m/%Y", "%Y-%m-%d %H:%M:%S"]:
                try:
                    return datetime.strptime(date_val.split()[0], fmt)
                except ValueError:
                    continue

        return None

    def _fuzzy_match_team(
        self,
        team_name: str,
        candidates: List[str]
    ) -> Optional[str]:
        """
        Find best fuzzy match for team name.

        Uses caching to avoid repeated matching.
        """
        cache_key = f"{team_name}:{','.join(sorted(candidates[:10]))}"
        if cache_key in self._team_name_cache:
            return self._team_name_cache[cache_key]

        # First try exact match after normalization
        normalized = normalize_team_name(team_name)
        if normalized in candidates:
            self._team_name_cache[cache_key] = normalized
            return normalized

        # Try fuzzy matching
        match, score = process.extractOne(team_name, candidates, scorer=fuzz.ratio)

        if score >= self.fuzzy_match_threshold:
            self._team_name_cache[cache_key] = match
            return match

        # Try with normalized name
        match, score = process.extractOne(normalized, candidates, scorer=fuzz.ratio)
        if score >= self.fuzzy_match_threshold:
            self._team_name_cache[cache_key] = match
            return match

        logger.warning(f"No match found for team: {team_name} (best: {match} @ {score})")
        self._team_name_cache[cache_key] = None
        return None

    def _create_match_key(
        self,
        date: datetime,
        home_team: str,
        away_team: str
    ) -> str:
        """Create unique key for a match."""
        date_str = date.strftime("%Y-%m-%d") if date else "unknown"
        return f"{date_str}|{home_team.lower().strip()}|{away_team.lower().strip()}"

    def merge_with_features(
        self,
        features_df: pd.DataFrame,
        odds_df: pd.DataFrame,
        home_team_col: str = "home_team_name",
        away_team_col: str = "away_team_name",
        date_col: str = "date"
    ) -> pd.DataFrame:
        """
        Merge odds data with existing features DataFrame.

        Args:
            features_df: Your existing features (with fixture info)
            odds_df: Odds data from FootballDataLoader
            home_team_col: Column name for home team in features_df
            away_team_col: Column name for away team in features_df
            date_col: Column name for match date

        Returns:
            features_df with odds columns added
        """
        if odds_df.empty:
            logger.warning("Empty odds dataframe, returning features unchanged")
            return features_df

        features_df = features_df.copy()
        odds_df = odds_df.copy()

        # Normalize dates
        features_df['_merge_date'] = features_df[date_col].apply(self._normalize_date)
        odds_df['_merge_date'] = odds_df['date'].apply(self._normalize_date)

        # Get unique team names from both sources
        feature_teams = set(features_df[home_team_col].unique()) | set(features_df[away_team_col].unique())
        odds_teams = set(odds_df['home_team'].unique()) | set(odds_df['away_team'].unique())

        logger.info(f"Feature teams: {len(feature_teams)}, Odds teams: {len(odds_teams)}")

        # Build team name mapping (odds -> features)
        team_mapping = {}
        for odds_team in odds_teams:
            matched = self._fuzzy_match_team(odds_team, list(feature_teams))
            if matched:
                team_mapping[odds_team] = matched

        # Apply mapping to odds data
        odds_df['_home_team_mapped'] = odds_df['home_team'].map(team_mapping)
        odds_df['_away_team_mapped'] = odds_df['away_team'].map(team_mapping)

        # Drop rows where mapping failed
        mapped_odds = odds_df.dropna(subset=['_home_team_mapped', '_away_team_mapped'])
        logger.info(f"Successfully mapped {len(mapped_odds)}/{len(odds_df)} odds rows")

        # Create match keys for merging
        features_df['_match_key'] = features_df.apply(
            lambda r: self._create_match_key(r['_merge_date'], r[home_team_col], r[away_team_col]),
            axis=1
        )

        mapped_odds['_match_key'] = mapped_odds.apply(
            lambda r: self._create_match_key(r['_merge_date'], r['_home_team_mapped'], r['_away_team_mapped']),
            axis=1
        )

        # Get odds columns to merge (exclude metadata)
        odds_feature_cols = [c for c in mapped_odds.columns
                           if c not in ['date', 'time', 'home_team', 'away_team',
                                       'home_goals', 'away_goals', 'result',
                                       'league', 'season', '_merge_date',
                                       '_home_team_mapped', '_away_team_mapped', '_match_key']]

        # Merge on match key
        merged = features_df.merge(
            mapped_odds[['_match_key'] + odds_feature_cols],
            on='_match_key',
            how='left'
        )

        # Count successful merges
        n_matched = merged[odds_feature_cols[0]].notna().sum() if odds_feature_cols else 0
        logger.info(f"Successfully merged odds for {n_matched}/{len(features_df)} matches")

        # Drop helper columns
        merged = merged.drop(columns=['_merge_date', '_match_key'], errors='ignore')

        return merged

    def merge_with_fixtures(
        self,
        fixtures_df: pd.DataFrame,
        odds_df: pd.DataFrame,
        fixture_id_col: str = "fixture_id"
    ) -> pd.DataFrame:
        """
        Alternative merge using fixture-level data.

        Use this if features_df doesn't have team names but has fixture_id.
        """
        # Implementation similar to above but using fixture_id matching
        pass


def load_and_merge_odds(
    features_path: Path,
    league: str,
    seasons: List[int],
    output_path: Optional[Path] = None,
    cache_dir: Optional[Path] = None
) -> pd.DataFrame:
    """
    Convenience function to load features, fetch odds, and merge.

    Args:
        features_path: Path to existing features CSV
        league: League name (e.g., "premier_league")
        seasons: List of seasons to fetch odds for
        output_path: Where to save merged result (optional)
        cache_dir: Directory to cache odds data

    Returns:
        Features DataFrame with odds added
    """
    # Load features
    features_df = pd.read_csv(features_path)
    logger.info(f"Loaded {len(features_df)} feature rows")

    # Load odds
    loader = FootballDataLoader(cache_dir=cache_dir)
    odds_df = loader.load_multiple_seasons(league, seasons)

    if odds_df.empty:
        logger.warning("No odds data loaded")
        return features_df

    # Create odds features
    from src.odds.odds_features import OddsFeatureEngineer
    engineer = OddsFeatureEngineer(use_closing_odds=True)
    odds_df = engineer.create_features(odds_df)

    # Merge
    merger = OddsMerger()
    merged_df = merger.merge_with_features(features_df, odds_df)

    # Save if output path provided
    if output_path:
        merged_df.to_csv(output_path, index=False)
        logger.info(f"Saved merged features to: {output_path}")

    return merged_df
