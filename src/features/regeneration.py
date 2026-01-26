"""
Feature Regeneration Module

This module provides efficient feature regeneration with parameter customization
and hash-based caching. It allows regenerating features with different parameters
without modifying the base feature engineering pipeline.

Key features:
- Regenerate features with custom parameters (from BetTypeFeatureConfig)
- Hash-based caching to avoid redundant computation
- Selective regeneration (only affected engineers)
- Compatible with existing feature engineering infrastructure

Usage:
    from src.features.regeneration import FeatureRegenerator
    from src.features.config_manager import BetTypeFeatureConfig

    # Create regenerator
    regenerator = FeatureRegenerator()

    # Regenerate with custom params
    config = BetTypeFeatureConfig(bet_type='away_win', elo_k_factor=40)
    features_df = regenerator.regenerate_with_params(config)
"""
import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

import pandas as pd

from src.features.config_manager import BetTypeFeatureConfig
from src.features.registry import (
    get_registry,
    get_default_configs,
    FeatureEngineerConfig,
    DEFAULT_FEATURE_CONFIGS,
)
from src.features.loaders import ParquetDataLoader
from src.features.cleaners import MatchDataCleaner, PlayerStatsDataCleaner, LineupsDataCleaner
from src.features.merger import DataMerger

logger = logging.getLogger(__name__)


# Default paths
PREPROCESSED_DIR = Path("data/02-preprocessed")
RAW_DIR = Path("data/01-raw")
SPORTMONKS_ODDS_DIR = Path("data/sportmonks_odds/processed")
FEATURES_DIR = Path("data/03-features")
CACHE_DIR = FEATURES_DIR / "feature_cache"

# Supported leagues
LEAGUES = ["premier_league", "la_liga", "serie_a", "bundesliga", "ligue_1"]


@dataclass
class CacheManifestEntry:
    """Entry in the cache manifest tracking regenerated feature files."""
    params_hash: str
    bet_type: str
    created_at: str
    n_rows: int
    n_features: int
    config_summary: Dict[str, Any]


class FeatureRegenerator:
    """
    Regenerates features with custom parameters and caching.

    This class provides efficient feature regeneration by:
    1. Loading preprocessed data from all leagues/seasons
    2. Applying custom parameters to feature engineers
    3. Caching results by params hash to avoid redundant computation

    The cache is stored in data/03-features/feature_cache/ with:
    - {params_hash}.parquet - The cached features
    - manifest.json - Cache metadata and lookup table
    """

    def __init__(
        self,
        preprocessed_dir: Optional[Path] = None,
        raw_dir: Optional[Path] = None,
        cache_dir: Optional[Path] = None,
        leagues: Optional[List[str]] = None,
    ):
        """
        Initialize the feature regenerator.

        Args:
            preprocessed_dir: Path to preprocessed data directory
            raw_dir: Path to raw data directory (for match_stats)
            cache_dir: Path to cache directory
            leagues: List of leagues to include
        """
        self.preprocessed_dir = preprocessed_dir or PREPROCESSED_DIR
        self.raw_dir = raw_dir or RAW_DIR
        self.cache_dir = cache_dir or CACHE_DIR
        self.leagues = leagues or LEAGUES

        self.registry = get_registry()
        self._data_cache: Optional[Dict[str, pd.DataFrame]] = None

    def regenerate_with_params(
        self,
        config: BetTypeFeatureConfig,
        use_cache: bool = True,
        force_regenerate: bool = False,
    ) -> pd.DataFrame:
        """
        Regenerate features with custom parameters.

        Args:
            config: BetTypeFeatureConfig with custom parameters
            use_cache: Whether to check/use cached results
            force_regenerate: Force regeneration even if cache exists

        Returns:
            DataFrame with regenerated features
        """
        params_hash = config.params_hash()
        logger.info(f"Regenerating features for {config.bet_type} (hash: {params_hash})")

        # Check cache
        if use_cache and not force_regenerate:
            cached = self._load_from_cache(params_hash)
            if cached is not None:
                logger.info(f"Loaded {len(cached)} rows from cache")
                return cached

        # Load data (cached in memory)
        data = self._load_all_data()

        # Create modified configs with custom params
        engineer_configs = self._create_configs_with_params(config)

        # Generate features
        logger.info(f"Generating features with {len(engineer_configs)} engineers...")
        feature_dfs = self.registry.create_all_features(
            data,
            engineer_configs,
            on_error='warn'
        )

        # Merge features
        merged = self._merge_features(data, feature_dfs)
        logger.info(f"Generated {len(merged)} rows, {len(merged.columns)} columns")

        # Merge SportMonks odds (BTTS, corners, cards)
        merged = self._merge_sportmonks_odds(merged)

        # Cache results
        if use_cache:
            self._save_to_cache(merged, config, params_hash)

        return merged

    def _load_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load all preprocessed data from all leagues/seasons.

        Caches in memory for efficiency when regenerating multiple times.

        Returns:
            Dict with 'matches', 'player_stats', 'lineups', 'events', 'match_stats'
        """
        if self._data_cache is not None:
            logger.debug("Using cached data")
            return self._data_cache

        logger.info("Loading preprocessed data from all leagues...")
        loader = ParquetDataLoader()

        all_matches = []
        all_player_stats = []
        all_lineups = []
        all_events = []
        all_match_stats = []

        for league in self.leagues:
            league_dir = self.preprocessed_dir / league
            raw_league_dir = self.raw_dir / league

            if not league_dir.exists():
                logger.debug(f"League directory not found: {league_dir}")
                continue

            # Find all seasons
            for season_dir in sorted(league_dir.iterdir()):
                if not season_dir.is_dir() or not season_dir.name.isdigit():
                    continue

                raw_season_dir = raw_league_dir / season_dir.name

                # Load matches
                matches_path = season_dir / "matches.parquet"
                if matches_path.exists():
                    df = loader.load(str(matches_path))
                    df['league'] = league
                    df['season'] = int(season_dir.name)
                    all_matches.append(df)

                # Load player_stats
                player_stats_path = season_dir / "player_stats.parquet"
                if player_stats_path.exists():
                    all_player_stats.append(loader.load(str(player_stats_path)))

                # Load lineups
                lineups_path = season_dir / "lineups.parquet"
                if lineups_path.exists():
                    all_lineups.append(loader.load(str(lineups_path)))

                # Load events
                events_path = season_dir / "events.parquet"
                if events_path.exists():
                    all_events.append(loader.load(str(events_path)))

                # Load match_stats (from raw directory)
                match_stats_path = raw_season_dir / "match_stats.parquet"
                if match_stats_path.exists():
                    all_match_stats.append(loader.load(str(match_stats_path)))

        if not all_matches:
            raise FileNotFoundError(
                f"No match data found in {self.preprocessed_dir}"
            )

        # Combine all data
        combined_matches = pd.concat(all_matches, ignore_index=True)
        logger.info(f"Loaded {len(combined_matches)} matches from {len(all_matches)} season files")

        result = {'matches': combined_matches}

        if all_player_stats:
            result['player_stats'] = pd.concat(all_player_stats, ignore_index=True)
        if all_lineups:
            result['lineups'] = pd.concat(all_lineups, ignore_index=True)
        if all_events:
            result['events'] = pd.concat(all_events, ignore_index=True)
        if all_match_stats:
            result['match_stats'] = pd.concat(all_match_stats, ignore_index=True)

        # Clean data
        result = self._clean_data(result)

        self._data_cache = result
        return result

    def _clean_data(self, raw_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Clean all loaded data."""
        cleaned_data = {}

        match_cleaner = MatchDataCleaner()
        cleaned_data['matches'] = match_cleaner.clean(raw_data['matches'])

        team_name_to_id = self._build_team_mapping(cleaned_data['matches'])

        if 'player_stats' in raw_data:
            player_cleaner = PlayerStatsDataCleaner()
            cleaned_data['player_stats'] = player_cleaner.clean(raw_data['player_stats'])
            cleaned_data['player_stats'] = self._map_team_ids(
                cleaned_data['player_stats'], team_name_to_id
            )

        if 'lineups' in raw_data:
            lineups_cleaner = LineupsDataCleaner()
            cleaned_data['lineups'] = lineups_cleaner.clean(raw_data['lineups'])
            cleaned_data['lineups'] = self._map_team_ids(
                cleaned_data['lineups'], team_name_to_id
            )

        if 'events' in raw_data:
            cleaned_data['events'] = raw_data['events']

        if 'match_stats' in raw_data:
            cleaned_data['match_stats'] = raw_data['match_stats']
            # Merge match_stats into matches
            # Column names must match actual match_stats.parquet schema
            stats_cols = [
                'fixture_id', 'home_fouls', 'away_fouls',
                'home_corners', 'away_corners',  # For corners market
                'home_shots', 'away_shots',  # For shots market
                'home_shots_on_target', 'away_shots_on_target',
                'home_offsides', 'away_offsides',
                'home_possession', 'away_possession',
            ]
            available_cols = [c for c in stats_cols if c in raw_data['match_stats'].columns]
            if available_cols:
                match_stats_subset = raw_data['match_stats'][available_cols].drop_duplicates(subset=['fixture_id'])
                cleaned_data['matches'] = cleaned_data['matches'].merge(
                    match_stats_subset, on='fixture_id', how='left'
                )

        return cleaned_data

    def _build_team_mapping(self, matches: pd.DataFrame) -> dict:
        """Build mapping from team names to team IDs."""
        team_map = {}
        if 'home_team_id' in matches.columns and 'home_team_name' in matches.columns:
            for _, row in matches[['home_team_id', 'home_team_name']].drop_duplicates().iterrows():
                team_map[row['home_team_name']] = row['home_team_id']
            for _, row in matches[['away_team_id', 'away_team_name']].drop_duplicates().iterrows():
                team_map[row['away_team_name']] = row['away_team_id']
        return team_map

    def _map_team_ids(self, df: pd.DataFrame, team_map: dict) -> pd.DataFrame:
        """Map team_name strings to team_id integers if needed."""
        if df.empty or not team_map:
            return df

        if 'team_id' in df.columns and df['team_id'].dtype == 'object':
            df = df.copy()
            df['team_id'] = df['team_id'].map(team_map)
            df = df.dropna(subset=['team_id'])
            df['team_id'] = df['team_id'].astype(int)

        if 'team_name' in df.columns and 'team_id' not in df.columns:
            df = df.copy()
            df['team_id'] = df['team_name'].map(team_map)
            df = df.dropna(subset=['team_id'])
            df['team_id'] = df['team_id'].astype(int)

        return df

    def _create_configs_with_params(
        self,
        feature_config: BetTypeFeatureConfig,
    ) -> List[FeatureEngineerConfig]:
        """
        Create FeatureEngineerConfig list with custom parameters.

        Starts from default configs and overrides params based on BetTypeFeatureConfig.

        Args:
            feature_config: BetTypeFeatureConfig with custom parameters

        Returns:
            List of FeatureEngineerConfig with params applied
        """
        registry_params = feature_config.to_registry_params()
        configs = []

        for default_cfg in DEFAULT_FEATURE_CONFIGS:
            cfg = FeatureEngineerConfig(
                name=default_cfg.name,
                enabled=default_cfg.enabled,
                required=default_cfg.required,
                requires_data=default_cfg.requires_data.copy(),
                params=default_cfg.params.copy()
            )

            # Override params from feature_config
            if cfg.name in registry_params:
                cfg.params.update(registry_params[cfg.name])
                logger.debug(f"Updated {cfg.name} params: {registry_params[cfg.name]}")

            configs.append(cfg)

        return configs

    def _merge_features(
        self,
        cleaned_data: Dict[str, pd.DataFrame],
        feature_dfs: List[pd.DataFrame],
    ) -> pd.DataFrame:
        """Merge all features into single DataFrame."""
        merger = DataMerger()

        # Get base columns - handle both 'round' and 'fixture_round' naming
        round_col = 'round' if 'round' in cleaned_data['matches'].columns else 'fixture_round'
        base_cols = [
            'fixture_id', 'date', 'home_team_id', 'home_team_name',
            'away_team_id', 'away_team_name'
        ]
        if round_col in cleaned_data['matches'].columns:
            base_cols.append(round_col)

        # Add league and season if present
        for col in ['league', 'season']:
            if col in cleaned_data['matches'].columns:
                base_cols.append(col)

        # Add target/outcome columns needed for training
        # These come from the original match data and are required for ML targets
        # Note: cleaner renames goals.home -> ft_home, goals.away -> ft_away
        target_cols = [
            'home_goals', 'away_goals',  # For BTTS, over/under targets
            'ft_home', 'ft_away',  # Alternative names from cleaner
            'home_win', 'draw', 'away_win',  # Core outcome targets
            'btts',  # Both teams to score
            'total_goals', 'goal_difference',  # Goals targets
            'match_result', 'result',  # Alternative result columns
        ]
        for col in target_cols:
            if col in cleaned_data['matches'].columns and col not in base_cols:
                base_cols.append(col)

        base_df = cleaned_data['matches'][base_cols]
        final_data = merger.merge_all_features(base_df, feature_dfs)

        # Normalize score column names (ft_home/ft_away -> home_goals/away_goals)
        if 'ft_home' in final_data.columns and 'home_goals' not in final_data.columns:
            final_data = final_data.rename(columns={'ft_home': 'home_goals', 'ft_away': 'away_goals'})
            logger.debug("Renamed ft_home/ft_away to home_goals/away_goals")

        # Derive target columns if not present
        final_data = self._derive_target_columns(final_data)

        # Remove rows with missing core features (NaN in form features usually means
        # not enough historical data for that team)
        form_cols = [c for c in final_data.columns if 'wins_last_n' in c or 'form' in c.lower()]
        if form_cols:
            initial_rows = len(final_data)
            final_data = final_data.dropna(subset=form_cols[:2])
            removed = initial_rows - len(final_data)
            if removed > 0:
                logger.info(f"Removed {removed} rows with missing form features")

        return final_data

    def _derive_target_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derive target columns for ML training from raw match data.

        Creates columns like btts, home_win, away_win, total_goals from home_goals/away_goals.
        """
        # Need home_goals and away_goals to derive other targets
        if 'home_goals' not in df.columns or 'away_goals' not in df.columns:
            logger.debug("home_goals/away_goals not found, skipping target derivation")
            return df

        # Both Teams To Score
        if 'btts' not in df.columns:
            df['btts'] = ((df['home_goals'] > 0) & (df['away_goals'] > 0)).astype(int)
            # Mark matches without score data as NaN
            no_score_mask = df['home_goals'].isna() | df['away_goals'].isna()
            df.loc[no_score_mask, 'btts'] = pd.NA
            logger.debug(f"Derived btts target: {df['btts'].notna().sum()} valid values")

        # Win/Draw outcomes
        if 'home_win' not in df.columns:
            df['home_win'] = (df['home_goals'] > df['away_goals']).astype(int)
            df.loc[df['home_goals'].isna() | df['away_goals'].isna(), 'home_win'] = pd.NA

        if 'away_win' not in df.columns:
            df['away_win'] = (df['away_goals'] > df['home_goals']).astype(int)
            df.loc[df['home_goals'].isna() | df['away_goals'].isna(), 'away_win'] = pd.NA

        if 'draw' not in df.columns:
            df['draw'] = (df['home_goals'] == df['away_goals']).astype(int)
            df.loc[df['home_goals'].isna() | df['away_goals'].isna(), 'draw'] = pd.NA

        # Goals targets
        if 'total_goals' not in df.columns:
            df['total_goals'] = df['home_goals'] + df['away_goals']

        if 'goal_difference' not in df.columns:
            df['goal_difference'] = df['home_goals'] - df['away_goals']

        # Over/Under targets
        if 'over25' not in df.columns:
            df['over25'] = (df['total_goals'] > 2.5).astype(int)
            df.loc[df['total_goals'].isna(), 'over25'] = pd.NA

        if 'under25' not in df.columns:
            df['under25'] = (df['total_goals'] < 2.5).astype(int)
            df.loc[df['total_goals'].isna(), 'under25'] = pd.NA

        # Niche market targets (corners, fouls, shots)
        if 'total_corners' not in df.columns and 'home_corners' in df.columns and 'away_corners' in df.columns:
            df['total_corners'] = df['home_corners'].fillna(0) + df['away_corners'].fillna(0)
            # Mark as NA where both are missing
            both_missing = df['home_corners'].isna() & df['away_corners'].isna()
            df.loc[both_missing, 'total_corners'] = pd.NA
            logger.debug(f"Derived total_corners target: {df['total_corners'].notna().sum()} valid values")

        if 'total_fouls' not in df.columns and 'home_fouls' in df.columns and 'away_fouls' in df.columns:
            df['total_fouls'] = df['home_fouls'].fillna(0) + df['away_fouls'].fillna(0)
            both_missing = df['home_fouls'].isna() & df['away_fouls'].isna()
            df.loc[both_missing, 'total_fouls'] = pd.NA
            logger.debug(f"Derived total_fouls target: {df['total_fouls'].notna().sum()} valid values")

        if 'total_shots' not in df.columns and 'home_shots' in df.columns and 'away_shots' in df.columns:
            df['total_shots'] = df['home_shots'].fillna(0) + df['away_shots'].fillna(0)
            both_missing = df['home_shots'].isna() & df['away_shots'].isna()
            df.loc[both_missing, 'total_shots'] = pd.NA
            logger.debug(f"Derived total_shots target: {df['total_shots'].notna().sum()} valid values")

        return df

    def _merge_sportmonks_odds(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge SportMonks odds (BTTS, corners, cards) into regenerated features.

        This allows using real odds for markets not covered by football-data.co.uk.
        """
        btts_path = SPORTMONKS_ODDS_DIR / "btts_odds.csv"

        if not btts_path.exists():
            logger.debug("No SportMonks BTTS odds file found")
            return df

        try:
            btts_df = pd.read_csv(btts_path)
            logger.info(f"Loaded {len(btts_df)} SportMonks BTTS odds")

            # Prepare merge keys - normalize team names
            def normalize_name(name):
                if pd.isna(name):
                    return ""
                return str(name).lower().strip().replace(" fc", "").replace("fc ", "")

            df['_home_norm'] = df['home_team_name'].apply(normalize_name)
            df['_away_norm'] = df['away_team_name'].apply(normalize_name)

            # Create BTTS merge lookup
            btts_lookup = {}
            for _, row in btts_df.iterrows():
                home = normalize_name(row.get('home_team_normalized', row.get('home_team', '')))
                away = normalize_name(row.get('away_team_normalized', row.get('away_team', '')))
                key = (home, away)
                btts_lookup[key] = {
                    'sm_btts_yes_odds': row.get('yes_avg'),
                    'sm_btts_no_odds': row.get('no_avg'),
                }

            # Apply odds
            df['sm_btts_yes_odds'] = df.apply(
                lambda r: btts_lookup.get((r['_home_norm'], r['_away_norm']), {}).get('sm_btts_yes_odds'),
                axis=1
            )
            df['sm_btts_no_odds'] = df.apply(
                lambda r: btts_lookup.get((r['_home_norm'], r['_away_norm']), {}).get('sm_btts_no_odds'),
                axis=1
            )

            # Cleanup temp columns
            df = df.drop(columns=['_home_norm', '_away_norm'])

            btts_coverage = df['sm_btts_yes_odds'].notna().sum()
            logger.info(f"Merged BTTS odds: {btts_coverage}/{len(df)} ({btts_coverage/len(df)*100:.1f}%)")

        except Exception as e:
            logger.warning(f"Failed to merge SportMonks odds: {e}")

        return df

    def _load_from_cache(self, params_hash: str) -> Optional[pd.DataFrame]:
        """Load cached features if available."""
        cache_path = self.cache_dir / f"{params_hash}.parquet"
        if cache_path.exists():
            logger.debug(f"Loading from cache: {cache_path}")
            return pd.read_parquet(cache_path)
        return None

    def _save_to_cache(
        self,
        df: pd.DataFrame,
        config: BetTypeFeatureConfig,
        params_hash: str,
    ) -> None:
        """Save features to cache."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self.cache_dir / f"{params_hash}.parquet"
        df.to_parquet(cache_path, index=False)
        logger.info(f"Cached features to: {cache_path}")

        # Update manifest
        self._update_manifest(config, params_hash, df)

    def _update_manifest(
        self,
        config: BetTypeFeatureConfig,
        params_hash: str,
        df: pd.DataFrame,
    ) -> None:
        """Update cache manifest with new entry."""
        manifest_path = self.cache_dir / "manifest.json"

        # Load existing manifest
        manifest = {}
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)

        # Add/update entry
        from datetime import datetime
        manifest[params_hash] = {
            'bet_type': config.bet_type,
            'created_at': datetime.now().isoformat(),
            'n_rows': len(df),
            'n_features': len(df.columns),
            'config': {
                'elo_k_factor': config.elo_k_factor,
                'elo_home_advantage': config.elo_home_advantage,
                'form_window': config.form_window,
                'ema_span': config.ema_span,
                'poisson_lookback': config.poisson_lookback,
            }
        }

        # Save manifest
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

    def clear_cache(self, params_hash: Optional[str] = None) -> int:
        """
        Clear cached features.

        Args:
            params_hash: Specific hash to clear, or None to clear all

        Returns:
            Number of cache entries cleared
        """
        if not self.cache_dir.exists():
            return 0

        cleared = 0

        if params_hash:
            # Clear specific entry
            cache_path = self.cache_dir / f"{params_hash}.parquet"
            if cache_path.exists():
                cache_path.unlink()
                cleared = 1
                logger.info(f"Cleared cache entry: {params_hash}")
        else:
            # Clear all
            for cache_file in self.cache_dir.glob("*.parquet"):
                cache_file.unlink()
                cleared += 1
            # Also clear manifest
            manifest_path = self.cache_dir / "manifest.json"
            if manifest_path.exists():
                manifest_path.unlink()
            logger.info(f"Cleared {cleared} cache entries")

        return cleared

    def list_cached(self) -> Dict[str, Any]:
        """
        List all cached feature configurations.

        Returns:
            Dict mapping params_hash to cache metadata
        """
        manifest_path = self.cache_dir / "manifest.json"
        if not manifest_path.exists():
            return {}

        with open(manifest_path, 'r') as f:
            return json.load(f)

    def invalidate_data_cache(self) -> None:
        """Invalidate the in-memory data cache to force reload on next regeneration."""
        self._data_cache = None
        logger.info("Invalidated in-memory data cache")
