"""
External Feature Injector for Late-Breaking Match Intelligence.

Injects referee, weather, and lineup features at inference time for upcoming matches.
This enables the model to use late-breaking information (referee assignments,
weather forecasts, confirmed lineups) that wasn't available when historical
features were cached.

The model learns patterns like "ref_cards_avg > 5.0 -> more cards" during training.
At inference, we provide the assigned referee's stats and the model applies
these learned patterns to make better predictions.

Similarly for lineups: the model learns "higher lineup_strength -> more goals".
When lineups are announced (~1hr before kickoff), we compute lineup strength
from player stats and inject these features.

Usage:
    from src.ml.feature_injector import ExternalFeatureInjector

    injector = ExternalFeatureInjector()
    features_df = injector.inject_features(features_df, {
        'referee': 'Michael Oliver',
        'venue_city': 'Manchester',
        'kickoff': datetime(2026, 2, 8, 15, 0),
        'home_lineup': {'starting_xi': [{'id': 123, 'name': 'Player A'}, ...]},
        'away_lineup': {'starting_xi': [{'id': 456, 'name': 'Player B'}, ...]},
    })
"""
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class ExternalFeatureInjector:
    """
    Injects external features (referee, weather, lineup) for upcoming matches.

    Architecture:
    - Referee features: Looked up from pre-computed cache built during feature generation
    - Weather features: Fetched from Open-Meteo Forecast API in real-time
    - Lineup features: Computed from player stats cache when lineups announced (~1hr before)

    All sources fall back to neutral defaults if data is unavailable,
    ensuring predictions never fail due to missing external data.
    """

    # Default cache paths
    REFEREE_CACHE_PATH = Path("data/cache/referee_stats.parquet")
    PLAYER_STATS_CACHE_PATH = Path("data/cache/player_stats.parquet")

    # League average defaults (from RefereeFeatureEngineer.DEFAULTS)
    REFEREE_DEFAULTS = {
        'ref_home_win_pct': 0.46,
        'ref_draw_pct': 0.25,
        'ref_away_win_pct': 0.29,
        'ref_avg_goals': 2.7,
        'ref_matches': 0,
        'ref_home_bias': 0.0,
        'ref_cards_avg': 4.2,
        'ref_yellows_avg': 3.5,
        'ref_reds_avg': 0.2,
        'ref_fouls_avg': 22.0,
        'ref_corners_avg': 10.3,
        'ref_cards_bias': 0.0,
        'ref_fouls_bias': 0.0,
        'ref_corners_bias': 0.0,
    }

    # Neutral weather defaults (mild conditions)
    WEATHER_DEFAULTS = {
        'weather_temp': 15.0,
        'weather_temp_normalized': 0.0,
        'weather_precip': 0.0,
        'weather_is_rainy': 0,
        'weather_heavy_rain': 0,
        'weather_wind': 10.0,
        'weather_is_windy': 0,
        'weather_very_windy': 0,
        'weather_humidity': 70.0,
        'weather_humidity_normalized': 0.0,
        'weather_high_humidity': 0,
        'weather_is_clear': 1,
        'weather_is_foggy': 0,
        'weather_is_stormy': 0,
        'weather_extreme_cold': 0,
        'weather_extreme_hot': 0,
        'weather_adverse_score': 0,
    }

    # Neutral lineup defaults (league average quality)
    LINEUP_DEFAULTS = {
        'home_xi_avg_rating': 6.5,
        'away_xi_avg_rating': 6.5,
        'home_xi_goals_per_90': 0.0,
        'away_xi_goals_per_90': 0.0,
        'home_xi_assists_per_90': 0.0,
        'away_xi_assists_per_90': 0.0,
        'lineup_rating_diff': 0.0,
        'lineup_offensive_diff': 0.0,
    }

    # Minimum matches for reliable statistics
    MIN_REFEREE_MATCHES = 5
    MIN_PLAYER_MATCHES = 3

    def __init__(
        self,
        referee_cache_path: Optional[str] = None,
        player_stats_cache_path: Optional[str] = None,
        weather_cache_dir: Optional[str] = None,
        enable_referee: bool = True,
        enable_weather: bool = True,
        enable_lineups: bool = True,
    ):
        """
        Initialize the feature injector.

        Args:
            referee_cache_path: Path to referee stats cache parquet file
            player_stats_cache_path: Path to player stats cache parquet file
            weather_cache_dir: Directory for weather cache (unused, kept for API compat)
            enable_referee: Whether to inject referee features
            enable_weather: Whether to inject weather features
            enable_lineups: Whether to inject lineup features
        """
        self.enable_referee = enable_referee
        self.enable_weather = enable_weather
        self.enable_lineups = enable_lineups

        # Load referee stats cache
        self.referee_stats: Dict[str, Dict[str, float]] = {}
        cache_path = Path(referee_cache_path) if referee_cache_path else self.REFEREE_CACHE_PATH
        self._load_referee_cache(cache_path)

        # Load player stats cache for lineup features
        self.player_stats: Dict[int, Dict[str, float]] = {}
        if enable_lineups:
            ps_cache_path = Path(player_stats_cache_path) if player_stats_cache_path else self.PLAYER_STATS_CACHE_PATH
            self._load_player_stats_cache(ps_cache_path)

        # Initialize weather collector (lazy import to avoid circular deps)
        self.weather_collector = None
        if enable_weather:
            self._init_weather_collector()

    def _load_referee_cache(self, cache_path: Path) -> None:
        """Load referee statistics from cache file."""
        if not cache_path.exists():
            logger.warning(f"Referee cache not found: {cache_path}. Using defaults.")
            return

        try:
            df = pd.read_parquet(cache_path)
            logger.info(f"Loaded referee cache with {len(df)} referees from {cache_path}")

            # Convert DataFrame to dict for fast lookup
            for _, row in df.iterrows():
                name = row.get('referee_name', row.get('referee', ''))
                if not name:
                    continue

                self.referee_stats[name] = {
                    'matches': row.get('matches', 0),
                    'total_yellows': row.get('total_yellows', 0),
                    'total_reds': row.get('total_reds', 0),
                    'total_fouls': row.get('total_fouls', 0),
                    'total_corners': row.get('total_corners', 0),
                    'home_wins': row.get('home_wins', 0),
                    'draws': row.get('draws', 0),
                    'away_wins': row.get('away_wins', 0),
                    'total_goals': row.get('total_goals', 0),
                }

        except Exception as e:
            logger.error(f"Failed to load referee cache: {e}")

    def _load_player_stats_cache(self, cache_path: Path) -> None:
        """Load player statistics from cache file."""
        if not cache_path.exists():
            logger.warning(f"Player stats cache not found: {cache_path}. Using defaults.")
            return

        try:
            df = pd.read_parquet(cache_path)
            logger.info(f"Loaded player stats cache with {len(df)} players from {cache_path}")

            # Convert DataFrame to dict for fast lookup by player_id
            for _, row in df.iterrows():
                player_id = row.get('player_id')
                if pd.isna(player_id):
                    continue

                self.player_stats[int(player_id)] = {
                    'name': row.get('player_name', ''),
                    'avg_rating': row.get('avg_rating', 6.0),
                    'total_minutes': row.get('total_minutes', 0),
                    'matches_played': row.get('matches_played', 0),
                    'goals_per_90': row.get('goals_per_90', 0.0),
                    'assists_per_90': row.get('assists_per_90', 0.0),
                    'position': row.get('position', ''),
                }

        except Exception as e:
            logger.error(f"Failed to load player stats cache: {e}")

    def _init_weather_collector(self) -> None:
        """Initialize weather collector for forecast fetching."""
        try:
            from src.data_collection.weather_collector import WeatherCollector
            self.weather_collector = WeatherCollector()
        except ImportError as e:
            logger.warning(f"Could not import WeatherCollector: {e}")
            self.weather_collector = None

    def inject_features(
        self,
        features_df: pd.DataFrame,
        match_info: Dict[str, Any],
    ) -> pd.DataFrame:
        """
        Inject referee, weather, and lineup features for an upcoming match.

        Args:
            features_df: Base features DataFrame (from FeatureLookup)
            match_info: Dict with match details:
                - referee: Referee name (optional)
                - venue_city: City name for weather lookup (optional)
                - kickoff: Match datetime for weather forecast (optional)
                - home_lineup: Dict with 'starting_xi' list of player dicts (optional)
                - away_lineup: Dict with 'starting_xi' list of player dicts (optional)

        Returns:
            DataFrame with referee, weather, and lineup features added/updated
        """
        df = features_df.copy()

        if self.enable_referee:
            referee = match_info.get('referee')
            df = self._inject_referee_features(df, referee)

        if self.enable_weather:
            df = self._inject_weather_features(df, match_info)

        if self.enable_lineups:
            df = self._inject_lineup_features(df, match_info)

        return df

    def _inject_referee_features(
        self,
        df: pd.DataFrame,
        referee_name: Optional[str],
    ) -> pd.DataFrame:
        """
        Inject referee features based on referee assignment.

        Computes referee statistics from cached historical data:
        - ref_cards_avg, ref_fouls_avg, ref_corners_avg (key for niche betting)
        - ref_home_win_pct, ref_draw_pct, ref_away_win_pct (result tendencies)
        - ref_home_bias, ref_cards_bias, etc. (deviation from league average)

        Args:
            df: Features DataFrame
            referee_name: Assigned referee name (None if unknown)

        Returns:
            DataFrame with referee features injected
        """
        if not referee_name or referee_name not in self.referee_stats:
            # Use defaults for unknown referee
            if referee_name:
                logger.debug(f"Unknown referee '{referee_name}', using defaults")
            return self._apply_referee_defaults(df)

        stats = self.referee_stats[referee_name]
        n = stats.get('matches', 0)

        if n < self.MIN_REFEREE_MATCHES:
            # Insufficient data - use defaults
            logger.debug(f"Referee '{referee_name}' has only {n} matches, using defaults")
            return self._apply_referee_defaults(df)

        # Calculate features from aggregated stats
        total_cards = stats.get('total_yellows', 0) + stats.get('total_reds', 0)

        features = {
            'ref_home_win_pct': stats.get('home_wins', 0) / n,
            'ref_draw_pct': stats.get('draws', 0) / n,
            'ref_away_win_pct': stats.get('away_wins', 0) / n,
            'ref_avg_goals': stats.get('total_goals', 0) / n,
            'ref_matches': n,
            'ref_cards_avg': total_cards / n if total_cards > 0 else self.REFEREE_DEFAULTS['ref_cards_avg'],
            'ref_yellows_avg': stats.get('total_yellows', 0) / n if stats.get('total_yellows', 0) > 0 else 3.5,
            'ref_reds_avg': stats.get('total_reds', 0) / n if stats.get('total_reds', 0) > 0 else 0.2,
            'ref_fouls_avg': stats.get('total_fouls', 0) / n if stats.get('total_fouls', 0) > 0 else self.REFEREE_DEFAULTS['ref_fouls_avg'],
            'ref_corners_avg': stats.get('total_corners', 0) / n if stats.get('total_corners', 0) > 0 else self.REFEREE_DEFAULTS['ref_corners_avg'],
        }

        # Calculate bias features (deviation from league average)
        features['ref_home_bias'] = features['ref_home_win_pct'] - self.REFEREE_DEFAULTS['ref_home_win_pct']
        features['ref_cards_bias'] = features['ref_cards_avg'] - self.REFEREE_DEFAULTS['ref_cards_avg']
        features['ref_fouls_bias'] = features['ref_fouls_avg'] - self.REFEREE_DEFAULTS['ref_fouls_avg']
        features['ref_corners_bias'] = features['ref_corners_avg'] - self.REFEREE_DEFAULTS['ref_corners_avg']

        # Apply features to DataFrame
        for key, value in features.items():
            df[key] = value

        logger.debug(f"Injected referee features for '{referee_name}' (n={n})")
        return df

    def _apply_referee_defaults(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply default referee features to DataFrame."""
        for key, value in self.REFEREE_DEFAULTS.items():
            df[key] = value
        return df

    def _inject_weather_features(
        self,
        df: pd.DataFrame,
        match_info: Dict[str, Any],
    ) -> pd.DataFrame:
        """
        Inject weather features based on forecast for match location/time.

        Weather affects play style:
        - Rain reduces ball control and pass accuracy
        - Wind disrupts long passes and crosses
        - Extreme temperatures affect player stamina

        Args:
            df: Features DataFrame
            match_info: Dict with venue_city and kickoff datetime

        Returns:
            DataFrame with weather features injected
        """
        city = match_info.get('venue_city')
        kickoff = match_info.get('kickoff')

        if not city or not kickoff or not self.weather_collector:
            return self._apply_weather_defaults(df)

        # Fetch weather forecast
        try:
            weather = self.weather_collector.fetch_forecast(city, kickoff)
        except Exception as e:
            logger.warning(f"Failed to fetch weather forecast for {city}: {e}")
            weather = None

        if not weather:
            logger.debug(f"No weather data for {city}, using defaults")
            return self._apply_weather_defaults(df)

        # Extract weather values
        temp = weather.get('temperature', 15.0)
        precip = weather.get('precipitation', 0.0)
        wind = weather.get('wind_speed', 10.0)
        humidity = weather.get('humidity', 70.0)
        weather_code = weather.get('weather_code', 0)

        # Calculate derived features
        features = {
            'weather_temp': temp,
            'weather_temp_normalized': (temp - 15) / 15,
            'weather_precip': precip,
            'weather_is_rainy': 1 if precip > 0.5 else 0,
            'weather_heavy_rain': 1 if precip > 5 else 0,
            'weather_wind': wind,
            'weather_is_windy': 1 if wind > 20 else 0,
            'weather_very_windy': 1 if wind > 35 else 0,
            'weather_humidity': humidity,
            'weather_humidity_normalized': (humidity - 70) / 30,
            'weather_high_humidity': 1 if humidity > 85 else 0,
            'weather_extreme_cold': 1 if temp < 5 else 0,
            'weather_extreme_hot': 1 if temp > 28 else 0,
        }

        # Weather code flags
        weather_flags = self._get_weather_flags(weather_code)
        features.update(weather_flags)

        # Composite adverse score
        features['weather_adverse_score'] = (
            features['weather_is_rainy'] +
            features['weather_is_windy'] +
            features['weather_high_humidity'] +
            features['weather_extreme_cold'] +
            features['weather_extreme_hot']
        )

        # Apply features to DataFrame
        for key, value in features.items():
            df[key] = value

        logger.debug(f"Injected weather features for {city}: temp={temp}C, precip={precip}mm")
        return df

    def _apply_weather_defaults(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply default weather features to DataFrame."""
        for key, value in self.WEATHER_DEFAULTS.items():
            df[key] = value
        return df

    def _inject_lineup_features(
        self,
        df: pd.DataFrame,
        match_info: Dict[str, Any],
    ) -> pd.DataFrame:
        """
        Inject lineup features based on confirmed starting XI.

        Computes squad quality metrics from player stats cache:
        - home/away_xi_avg_rating: Average historical rating of starting XI
        - home/away_xi_goals_per_90: Sum of goals per 90 for starting XI
        - lineup_rating_diff: home - away XI rating difference
        - lineup_offensive_diff: home - away offensive contribution difference

        Args:
            df: Features DataFrame
            match_info: Dict with home_lineup and away_lineup

        Returns:
            DataFrame with lineup features injected
        """
        home_lineup = match_info.get('home_lineup')
        away_lineup = match_info.get('away_lineup')

        if not home_lineup or not away_lineup:
            logger.debug("No lineups provided, keeping existing features unchanged")
            return df  # Don't override - keep historical team averages from FeatureLookup

        if not self.player_stats:
            logger.debug("No player stats cache, keeping existing features unchanged")
            return df  # Don't override with arbitrary defaults

        # Extract player IDs from lineup data
        home_xi = self._extract_player_ids(home_lineup)
        away_xi = self._extract_player_ids(away_lineup)

        if not home_xi or not away_xi:
            logger.debug("Could not extract player IDs from lineups, keeping existing features")
            return df  # Don't override with defaults

        # Calculate squad quality metrics
        home_metrics = self._calculate_squad_metrics(home_xi)
        away_metrics = self._calculate_squad_metrics(away_xi)

        # If we couldn't compute metrics for either team, don't override
        if home_metrics is None or away_metrics is None:
            logger.debug("Could not compute squad metrics (unknown players), keeping existing features")
            return df

        features = {
            'home_xi_avg_rating': home_metrics['avg_rating'],
            'away_xi_avg_rating': away_metrics['avg_rating'],
            'home_xi_goals_per_90': home_metrics['goals_per_90'],
            'away_xi_goals_per_90': away_metrics['goals_per_90'],
            'home_xi_assists_per_90': home_metrics['assists_per_90'],
            'away_xi_assists_per_90': away_metrics['assists_per_90'],
            'lineup_rating_diff': home_metrics['avg_rating'] - away_metrics['avg_rating'],
            'lineup_offensive_diff': (
                (home_metrics['goals_per_90'] + home_metrics['assists_per_90']) -
                (away_metrics['goals_per_90'] + away_metrics['assists_per_90'])
            ),
        }

        # Apply features to DataFrame
        for key, value in features.items():
            df[key] = value

        logger.debug(
            f"Injected lineup features: home_rating={home_metrics['avg_rating']:.2f}, "
            f"away_rating={away_metrics['avg_rating']:.2f}"
        )
        return df

    def _extract_player_ids(self, lineup: Dict[str, Any]) -> list:
        """
        Extract player IDs from lineup data.

        Supports multiple formats:
        - {'starting_xi': [{'id': 123, 'name': 'Player'}, ...]}
        - {'starting_xi': [{'player': {'id': 123}}, ...]}
        - [{'id': 123}, ...] (direct list)
        """
        if isinstance(lineup, list):
            players = lineup
        else:
            players = lineup.get('starting_xi', lineup.get('startXI', []))

        player_ids = []
        for p in players:
            if isinstance(p, dict):
                # Try direct id
                if 'id' in p:
                    player_ids.append(int(p['id']))
                # Try nested player object
                elif 'player' in p and isinstance(p['player'], dict):
                    if 'id' in p['player']:
                        player_ids.append(int(p['player']['id']))
            elif isinstance(p, (int, float)) and not pd.isna(p):
                player_ids.append(int(p))

        return player_ids

    def _calculate_squad_metrics(self, player_ids: list) -> Optional[Dict[str, float]]:
        """
        Calculate squad quality metrics from player stats.

        Args:
            player_ids: List of player IDs in the starting XI

        Returns:
            Dict with avg_rating, goals_per_90, assists_per_90, or None if no known players
        """
        ratings = []
        goals_per_90_sum = 0.0
        assists_per_90_sum = 0.0
        players_found = 0

        for pid in player_ids:
            if pid in self.player_stats:
                stats = self.player_stats[pid]
                # Only include players with sufficient match history
                if stats.get('matches_played', 0) >= self.MIN_PLAYER_MATCHES:
                    ratings.append(stats.get('avg_rating', 6.0))
                    goals_per_90_sum += stats.get('goals_per_90', 0.0)
                    assists_per_90_sum += stats.get('assists_per_90', 0.0)
                    players_found += 1

        if not ratings:
            # No known players - return None to signal "don't override"
            return None

        avg_rating = sum(ratings) / len(ratings)

        logger.debug(f"Calculated squad metrics from {players_found}/{len(player_ids)} known players")

        return {
            'avg_rating': avg_rating,
            'goals_per_90': goals_per_90_sum,
            'assists_per_90': assists_per_90_sum,
        }

    def _apply_lineup_defaults(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply default lineup features to DataFrame."""
        for key, value in self.LINEUP_DEFAULTS.items():
            df[key] = value
        return df

    def _get_weather_flags(self, weather_code: int) -> Dict[str, int]:
        """
        Convert WMO weather code to categorical flags.

        WMO codes:
        0: Clear
        1-3: Mainly clear to overcast
        45-48: Fog
        51-67: Drizzle/Rain
        71-77: Snow
        80-82: Rain showers
        95-99: Thunderstorm
        """
        if pd.isna(weather_code):
            return {"weather_is_clear": 1, "weather_is_foggy": 0, "weather_is_stormy": 0}

        code = int(weather_code)

        return {
            "weather_is_clear": 1 if code <= 3 else 0,
            "weather_is_foggy": 1 if 45 <= code <= 48 else 0,
            "weather_is_stormy": 1 if 95 <= code <= 99 else 0,
        }


def get_feature_injector() -> ExternalFeatureInjector:
    """Get singleton ExternalFeatureInjector instance."""
    global _injector
    if '_injector' not in globals() or _injector is None:
        _injector = ExternalFeatureInjector()
    return _injector


_injector: Optional[ExternalFeatureInjector] = None
