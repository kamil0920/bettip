"""Feature engineering - External factors (referee, weather)."""
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.features.engineers.base import BaseFeatureEngineer

logger = logging.getLogger(__name__)


class RefereeFeatureEngineer(BaseFeatureEngineer):
    """
    Creates features based on referee statistics.

    Different referees have different tendencies:
    - Some are card-happy, others lenient
    - Some favor home teams more
    - Some allow more physical play

    Key features for niche betting markets:
    - ref_cards_avg: Critical for cards betting
    - ref_fouls_avg: Critical for fouls betting
    - ref_corners_avg: Useful for corners betting
    """

    # League average defaults (fallback when insufficient data)
    DEFAULTS = {
        'home_win_pct': 0.46,
        'draw_pct': 0.25,
        'away_win_pct': 0.29,
        'avg_goals': 2.7,
        'avg_cards': 4.2,
        'avg_fouls': 22.0,
        'avg_corners': 10.3,
    }

    def __init__(self, min_matches: int = 5):
        """
        Initialize with minimum matches threshold.

        Args:
            min_matches: Minimum matches for referee stats to be reliable
        """
        self.min_matches = min_matches

    def create_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculate referee-based features.

        Uses actual match statistics when available (cards, fouls, corners)
        from football-data.co.uk or similar sources.
        """
        matches = data['matches'].copy()
        matches = matches.sort_values('date').reset_index(drop=True)

        referee_stats = {}
        features_list = []

        for idx, match in matches.iterrows():
            referee = match.get('referee')

            if pd.notna(referee) and isinstance(referee, str) and referee.strip():
                stats = referee_stats.get(referee, self._init_referee_stats())

                if stats['matches'] >= self.min_matches:
                    features = self._calculate_features(match, stats)
                else:
                    features = self._default_features(match)

                # Update referee stats with this match (after calculating features)
                self._update_referee_stats(referee_stats, referee, match)
            else:
                features = self._default_features(match)

            features_list.append(features)

        print(f"Created {len(features_list)} referee features")
        return pd.DataFrame(features_list)

    def _init_referee_stats(self) -> Dict:
        """Initialize empty referee statistics dict."""
        return {
            'matches': 0,
            'home_wins': 0,
            'draws': 0,
            'away_wins': 0,
            'total_goals': 0,
            'total_yellows': 0,
            'total_reds': 0,
            'total_fouls': 0,
            'total_corners': 0,
        }

    def _calculate_features(self, match: pd.Series, stats: Dict) -> Dict:
        """Calculate referee features from accumulated statistics."""
        n = stats['matches']
        home_win_pct = stats['home_wins'] / n
        draw_pct = stats['draws'] / n
        away_win_pct = stats['away_wins'] / n
        avg_goals = stats['total_goals'] / n

        # Card statistics - key for cards betting
        total_cards = stats['total_yellows'] + stats['total_reds']
        avg_cards = total_cards / n if total_cards > 0 else self.DEFAULTS['avg_cards']
        avg_yellows = stats['total_yellows'] / n if stats['total_yellows'] > 0 else 3.5
        avg_reds = stats['total_reds'] / n if stats['total_reds'] > 0 else 0.2

        # Fouls statistics - key for fouls betting
        avg_fouls = stats['total_fouls'] / n if stats['total_fouls'] > 0 else self.DEFAULTS['avg_fouls']

        # Corners statistics - useful for corners betting
        avg_corners = stats['total_corners'] / n if stats['total_corners'] > 0 else self.DEFAULTS['avg_corners']

        return {
            'fixture_id': match.get('fixture_id', match.name),
            # Result tendencies
            'ref_home_win_pct': home_win_pct,
            'ref_draw_pct': draw_pct,
            'ref_away_win_pct': away_win_pct,
            'ref_avg_goals': avg_goals,
            'ref_matches': n,
            'ref_home_bias': home_win_pct - self.DEFAULTS['home_win_pct'],
            # Niche market features (critical for betting strategies)
            'ref_cards_avg': avg_cards,
            'ref_yellows_avg': avg_yellows,
            'ref_reds_avg': avg_reds,
            'ref_fouls_avg': avg_fouls,
            'ref_corners_avg': avg_corners,
            # Deviation from league average (indicates referee strictness)
            'ref_cards_bias': avg_cards - self.DEFAULTS['avg_cards'],
            'ref_fouls_bias': avg_fouls - self.DEFAULTS['avg_fouls'],
            'ref_corners_bias': avg_corners - self.DEFAULTS['avg_corners'],
        }

    def _default_features(self, match: pd.Series) -> Dict:
        """Return default features when referee unknown or insufficient data."""
        return {
            'fixture_id': match.get('fixture_id', match.name),
            'ref_home_win_pct': self.DEFAULTS['home_win_pct'],
            'ref_draw_pct': self.DEFAULTS['draw_pct'],
            'ref_away_win_pct': self.DEFAULTS['away_win_pct'],
            'ref_avg_goals': self.DEFAULTS['avg_goals'],
            'ref_matches': 0,
            'ref_home_bias': 0,
            'ref_cards_avg': self.DEFAULTS['avg_cards'],
            'ref_yellows_avg': 3.5,
            'ref_reds_avg': 0.2,
            'ref_fouls_avg': self.DEFAULTS['avg_fouls'],
            'ref_corners_avg': self.DEFAULTS['avg_corners'],
            'ref_cards_bias': 0,
            'ref_fouls_bias': 0,
            'ref_corners_bias': 0,
        }

    def _update_referee_stats(self, referee_stats: Dict, referee: str, match: pd.Series) -> None:
        """Update referee statistics with match data."""
        if referee not in referee_stats:
            referee_stats[referee] = self._init_referee_stats()

        stats = referee_stats[referee]
        stats['matches'] += 1

        # Get goals - try multiple column names
        home_goals = self._safe_get(match, ['ft_home', 'home_goals', 'FTHG'], 0)
        away_goals = self._safe_get(match, ['ft_away', 'away_goals', 'FTAG'], 0)
        stats['total_goals'] += home_goals + away_goals

        # Update result counts
        if home_goals > away_goals:
            stats['home_wins'] += 1
        elif home_goals == away_goals:
            stats['draws'] += 1
        else:
            stats['away_wins'] += 1

        # Card statistics (supports multiple data sources)
        # API-Football: home_yellow_cards, football-data.co.uk: home_yellows/HY
        home_yellows = self._safe_get(match, ['home_yellow_cards', 'home_yellows', 'HY'], 0)
        away_yellows = self._safe_get(match, ['away_yellow_cards', 'away_yellows', 'AY'], 0)
        home_reds = self._safe_get(match, ['home_red_cards', 'home_reds', 'HR'], 0)
        away_reds = self._safe_get(match, ['away_red_cards', 'away_reds', 'AR'], 0)
        stats['total_yellows'] += home_yellows + away_yellows
        stats['total_reds'] += home_reds + away_reds

        # Fouls statistics
        home_fouls = self._safe_get(match, ['home_fouls', 'HF'], 0)
        away_fouls = self._safe_get(match, ['away_fouls', 'AF'], 0)
        stats['total_fouls'] += home_fouls + away_fouls

        # Corners statistics
        home_corners = self._safe_get(match, ['home_corners', 'home_corner_kicks', 'HC'], 0)
        away_corners = self._safe_get(match, ['away_corners', 'away_corner_kicks', 'AC'], 0)
        stats['total_corners'] += home_corners + away_corners

    def _safe_get(self, match: pd.Series, keys: List[str], default: float = 0) -> float:
        """Safely get value from match Series, trying multiple keys."""
        for key in keys:
            if key in match.index:
                val = match[key]
                if pd.notna(val):
                    return float(val)
        return default



class MarketImpliedFeatureEngineer(BaseFeatureEngineer):
    """
    Creates features from bookmaker odds — using the crowd's wisdom.

    Opening odds encode bookmaker models + sharp money. Using them as FEATURES
    (not just for edge calculation) gives the model access to market consensus.

    Features:
    - implied_home/draw/away_prob: Vig-removed probabilities from opening odds
    - market_consensus_strength: 1/overround (how confident the market is)
    - odds_market_disagreement: Std of implied probs (market uncertainty)

    Walk-forward safe: uses opening odds available pre-match.
    Only for H2H markets where bookmaker odds exist.
    """

    def __init__(self):
        pass

    def create_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Calculate market-implied features from bookmaker odds columns."""
        matches = data['matches'].copy()
        matches = matches.sort_values('date').reset_index(drop=True)

        # Try multiple column naming conventions for odds
        home_cols = ['avg_home_open', 'avg_home_close', 'odds_home', 'B365H', 'PSH']
        draw_cols = ['avg_draw_open', 'avg_draw_close', 'odds_draw', 'B365D', 'PSD']
        away_cols = ['avg_away_open', 'avg_away_close', 'odds_away', 'B365A', 'PSA']

        home_col = self._find_col(matches, home_cols)
        draw_col = self._find_col(matches, draw_cols)
        away_col = self._find_col(matches, away_cols)

        if not home_col or not away_col:
            print("Warning: No bookmaker odds columns found, skipping MarketImpliedFeatureEngineer")
            return pd.DataFrame({'fixture_id': matches['fixture_id']})

        features_list = []
        for _, match in matches.iterrows():
            h_odds = self._safe_float(match.get(home_col))
            d_odds = self._safe_float(match.get(draw_col)) if draw_col else None
            a_odds = self._safe_float(match.get(away_col))

            features = {'fixture_id': match['fixture_id']}

            if h_odds and h_odds > 1 and a_odds and a_odds > 1:
                # Raw implied probabilities
                imp_h = 1.0 / h_odds
                imp_a = 1.0 / a_odds
                imp_d = (1.0 / d_odds) if (d_odds and d_odds > 1) else 0.0

                overround = imp_h + imp_d + imp_a

                # Vig-removed (equal-margin method)
                if overround > 0:
                    features['implied_home_prob'] = imp_h / overround
                    features['implied_draw_prob'] = imp_d / overround if imp_d > 0 else 0.0
                    features['implied_away_prob'] = imp_a / overround
                    features['market_consensus_strength'] = 1.0 / overround
                else:
                    features['implied_home_prob'] = 0.33
                    features['implied_draw_prob'] = 0.33
                    features['implied_away_prob'] = 0.33
                    features['market_consensus_strength'] = 1.0

                # Market disagreement: std of the 3 implied probs
                probs = [features['implied_home_prob'], features['implied_draw_prob'],
                         features['implied_away_prob']]
                features['odds_market_disagreement'] = float(np.std(probs))
            else:
                features['implied_home_prob'] = 0.33
                features['implied_draw_prob'] = 0.33
                features['implied_away_prob'] = 0.33
                features['market_consensus_strength'] = 1.0
                features['odds_market_disagreement'] = 0.0

            features_list.append(features)

        n_valid = sum(1 for f in features_list if f.get('market_consensus_strength', 1.0) != 1.0)
        print(f"Created {len(features_list)} market-implied features ({n_valid} with real odds)")
        return pd.DataFrame(features_list)

    def _find_col(self, df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        """Find first available column from candidates."""
        for col in candidates:
            if col in df.columns:
                return col
        return None

    def _safe_float(self, val) -> Optional[float]:
        """Safely convert to float."""
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        try:
            return float(val)
        except (TypeError, ValueError):
            return None


class WeatherFeatureEngineer(BaseFeatureEngineer):
    """
    Create weather-related features for matches.

    Research shows weather affects play style:
    - Rain reduces ball control and pass accuracy
    - Wind disrupts long passes and crosses
    - Extreme temperatures affect player stamina
    - High humidity increases fatigue

    Features created:
    - Temperature (normalized)
    - Precipitation amount
    - Wind speed
    - Humidity
    - Weather condition flags (rainy, foggy, clear, stormy)
    - Extreme weather indicators
    """

    # City coordinates for major European football cities
    CITY_COORDINATES = {
        # Premier League
        "London": (51.5074, -0.1278),
        "Manchester": (53.4808, -2.2426),
        "Liverpool": (53.4084, -2.9916),
        "Birmingham": (52.4862, -1.8904),
        "Newcastle upon Tyne": (54.9783, -1.6178),
        "Brighton": (50.8225, -0.1372),
        "Leicester": (52.6369, -1.1398),
        "Nottingham": (52.9548, -1.1581),
        "Wolverhampton": (52.5870, -2.1288),
        "Bournemouth": (50.7351, -1.8382),

        # La Liga
        "Madrid": (40.4168, -3.7038),
        "Barcelona": (41.3851, 2.1734),
        "Valencia": (39.4699, -0.3763),
        "Sevilla": (37.3891, -5.9845),
        "Seville": (37.3891, -5.9845),
        "Bilbao": (43.2630, -2.9350),

        # Serie A
        "Milano": (45.4642, 9.1900),
        "Milan": (45.4642, 9.1900),
        "Roma": (41.9028, 12.4964),
        "Rome": (41.9028, 12.4964),
        "Torino": (45.0703, 7.6869),
        "Napoli": (40.8518, 14.2681),

        # Bundesliga
        "München": (48.1351, 11.5820),
        "Munich": (48.1351, 11.5820),
        "Berlin": (52.5200, 13.4050),
        "Dortmund": (51.5136, 7.4653),
        "Frankfurt": (50.1109, 8.6821),

        # Ligue 1
        "Paris": (48.8566, 2.3522),
        "Marseille": (43.2965, 5.3698),
        "Lyon": (45.7640, 4.8357),
        "Monaco": (43.7384, 7.4246),

        # Turkish Super Lig
        "Istanbul": (41.0082, 28.9784),
        "İstanbul": (41.0082, 28.9784),
        "Ankara": (39.9334, 32.8597),
        "Izmir": (38.4237, 27.1428),
        "İzmir": (38.4237, 27.1428),
        "Trabzon": (41.0027, 39.7168),
        "Antalya": (36.8969, 30.7133),
        "Bursa": (40.1826, 29.0665),
        "Konya": (37.8746, 32.4932),
        "Gaziantep": (37.0662, 37.3833),
        "Kayseri": (38.7312, 35.4787),
        "Sivas": (39.7477, 37.0179),
        "Adana": (37.0000, 35.3213),
        "Samsun": (41.2867, 36.3300),
        "Rize": (41.0209, 40.5234),

        # Portuguese Liga
        "Lisbon": (38.7223, -9.1393),
        "Lisboa": (38.7223, -9.1393),
        "Porto": (41.1579, -8.6291),
        "Braga": (41.5518, -8.4229),
        "Guimarães": (41.4425, -8.2918),
        "Guimaraes": (41.4425, -8.2918),
        "Coimbra": (40.2033, -8.4103),
        "Funchal": (32.6669, -16.9241),
        "Famalicão": (41.4078, -8.5194),
        "Famalicao": (41.4078, -8.5194),

        # Scottish Premiership
        "Glasgow": (55.8642, -4.2518),
        "Edinburgh": (55.9533, -3.1883),
        "Aberdeen": (57.1497, -2.0943),
        "Dundee": (56.4620, -2.9707),
        "Perth": (56.3952, -3.4372),
        "Kilmarnock": (55.6111, -4.4951),
        "Motherwell": (55.7892, -3.9915),
        "Paisley": (55.8456, -4.4240),

        # Belgian Pro League
        "Brussels": (50.8503, 4.3517),
        "Bruxelles": (50.8503, 4.3517),
        "Anderlecht": (50.8342, 4.2980),
        "Bruges": (51.2093, 3.2247),
        "Brugge": (51.2093, 3.2247),
        "Ghent": (51.0543, 3.7174),
        "Gent": (51.0543, 3.7174),
        "Liège": (50.6292, 5.5797),
        "Liege": (50.6292, 5.5797),
        "Antwerp": (51.2194, 4.4025),
        "Antwerpen": (51.2194, 4.4025),
        "Genk": (50.9655, 5.5020),
        "Charleroi": (50.4108, 4.4446),
        "Leuven": (50.8798, 4.7005),
        "Sint-Truiden": (50.8167, 5.1833),

        # Eredivisie
        "Amsterdam": (52.3676, 4.9041),
        "Rotterdam": (51.9244, 4.4777),
        "Eindhoven": (51.4416, 5.4697),
        "Den Haag": (52.0705, 4.3007),
        "The Hague": (52.0705, 4.3007),
        "Utrecht": (52.0907, 5.1214),
        "Arnhem": (51.9851, 5.8987),
        "Enschede": (52.2215, 6.8937),
        "Alkmaar": (52.6324, 4.7534),
        "Groningen": (53.2194, 6.5665),
        "Heerenveen": (52.9564, 5.9119),
        "Tilburg": (51.5555, 5.0913),
        "Nijmegen": (51.8126, 5.8372),
        "Deventer": (52.2511, 6.1600),
        "Almelo": (52.3570, 6.6685),
        "Zwolle": (52.5168, 6.0830),
        "Sittard": (50.9984, 5.8687),
        "Waalwijk": (51.6881, 5.0691),
        "Emmen": (52.7792, 6.8939),
        "Volendam": (52.4945, 5.0700),
    }

    # Default weather cache paths
    WEATHER_CACHE_PATHS = [
        "data/weather_cache_recent.parquet",
        "data/weather_cache.parquet",
    ]

    def __init__(self, weather_data: pd.DataFrame = None, auto_load_cache: bool = True):
        """
        Args:
            weather_data: Pre-loaded weather data DataFrame with columns:
                - fixture_id or (city, date)
                - temperature, humidity, precipitation, wind_speed, weather_code
            auto_load_cache: If True, automatically load weather cache if available
        """
        self.weather_data = weather_data

        # Auto-load weather cache if not provided
        if self.weather_data is None and auto_load_cache:
            self._load_weather_cache()

    def create_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create weather features for each match."""
        matches = data['matches'].copy()
        matches = matches.sort_values('date')

        features_list = []

        for idx, match in matches.iterrows():
            fixture_id = match['fixture_id']

            # Try to get weather data
            weather = self._get_weather_for_match(match)

            if weather:
                # Temperature features
                temp = weather.get('temperature', 15)  # Default 15°C
                temp_normalized = (temp - 15) / 15  # Normalize around 15°C

                # Precipitation features
                precip = weather.get('precipitation', 0)
                is_rainy = 1 if precip > 0.5 else 0
                heavy_rain = 1 if precip > 5 else 0

                # Wind features
                wind = weather.get('wind_speed', 10)
                is_windy = 1 if wind > 20 else 0
                very_windy = 1 if wind > 35 else 0

                # Humidity features
                humidity = weather.get('humidity', 70)
                humidity_normalized = (humidity - 70) / 30  # Normalize around 70%
                high_humidity = 1 if humidity > 85 else 0

                # Weather code features
                weather_code = weather.get('weather_code', 0)
                weather_flags = self._get_weather_flags(weather_code)

                # Extreme conditions
                extreme_cold = 1 if temp < 5 else 0
                extreme_hot = 1 if temp > 28 else 0

                features = {
                    'fixture_id': fixture_id,
                    'weather_temp': temp,
                    'weather_temp_normalized': temp_normalized,
                    'weather_precip': precip,
                    'weather_is_rainy': is_rainy,
                    'weather_heavy_rain': heavy_rain,
                    'weather_wind': wind,
                    'weather_is_windy': is_windy,
                    'weather_very_windy': very_windy,
                    'weather_humidity': humidity,
                    'weather_humidity_normalized': humidity_normalized,
                    'weather_high_humidity': high_humidity,
                    'weather_is_clear': weather_flags['is_clear'],
                    'weather_is_foggy': weather_flags['is_foggy'],
                    'weather_is_stormy': weather_flags['is_stormy'],
                    'weather_extreme_cold': extreme_cold,
                    'weather_extreme_hot': extreme_hot,
                    # Composite adverse weather score
                    'weather_adverse_score': is_rainy + is_windy + high_humidity + extreme_cold + extreme_hot,
                }
            else:
                # No weather data - use neutral defaults
                features = {
                    'fixture_id': fixture_id,
                    'weather_temp': 15,
                    'weather_temp_normalized': 0,
                    'weather_precip': 0,
                    'weather_is_rainy': 0,
                    'weather_heavy_rain': 0,
                    'weather_wind': 10,
                    'weather_is_windy': 0,
                    'weather_very_windy': 0,
                    'weather_humidity': 70,
                    'weather_humidity_normalized': 0,
                    'weather_high_humidity': 0,
                    'weather_is_clear': 1,
                    'weather_is_foggy': 0,
                    'weather_is_stormy': 0,
                    'weather_extreme_cold': 0,
                    'weather_extreme_hot': 0,
                    'weather_adverse_score': 0,
                }

            features_list.append(features)

        print(f"Created {len(features_list)} weather features")
        return pd.DataFrame(features_list)

    def _load_weather_cache(self) -> None:
        """Try to load weather data from cache files."""
        from pathlib import Path

        for cache_path in self.WEATHER_CACHE_PATHS:
            path = Path(cache_path)
            if path.exists():
                try:
                    self.weather_data = pd.read_parquet(path)
                    # Ensure date column is string for matching
                    if 'date' in self.weather_data.columns:
                        self.weather_data['date'] = pd.to_datetime(
                            self.weather_data['date']
                        ).dt.strftime('%Y-%m-%d')
                    logger.info(f"Loaded weather cache from {cache_path}: {len(self.weather_data)} records")
                    return
                except Exception as e:
                    logger.warning(f"Failed to load weather cache {cache_path}: {e}")

        logger.info("No weather cache found - using neutral defaults")

    def _get_weather_for_match(self, match: pd.Series) -> Optional[Dict]:
        """Get weather data for a match."""
        if self.weather_data is None:
            return None

        fixture_id = match['fixture_id']

        # Try to find by fixture_id
        if 'fixture_id' in self.weather_data.columns:
            weather_row = self.weather_data[self.weather_data['fixture_id'] == fixture_id]
            if len(weather_row) > 0:
                return weather_row.iloc[0].to_dict()

        # Try to find by city and date
        city = match.get('venue_city', match.get('fixture.venue.city'))
        date = match.get('date')

        if city and date and 'city' in self.weather_data.columns:
            if isinstance(date, str):
                date_str = date[:10]
            else:
                date_str = date.strftime('%Y-%m-%d')

            weather_row = self.weather_data[
                (self.weather_data['city'] == city) &
                (self.weather_data['date'] == date_str)
            ]
            if len(weather_row) > 0:
                return weather_row.iloc[0].to_dict()

        return None

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
            return {"is_clear": 1, "is_foggy": 0, "is_stormy": 0}

        code = int(weather_code)

        return {
            "is_clear": 1 if code <= 3 else 0,
            "is_foggy": 1 if 45 <= code <= 48 else 0,
            "is_stormy": 1 if 95 <= code <= 99 else 0,
        }
