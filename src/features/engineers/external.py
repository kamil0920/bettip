"""Feature engineering - External factors (referee, weather)."""
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.features.engineers.base import BaseFeatureEngineer


class RefereeFeatureEngineer(BaseFeatureEngineer):
    """
    Creates features based on referee statistics.

    Different referees have different tendencies:
    - Some are card-happy, others lenient
    - Some favor home teams more
    - Some allow more physical play
    """

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
        """
        matches = data['matches'].copy()
        matches = matches.sort_values('date').reset_index(drop=True)

        referee_stats = {}

        features_list = []

        for idx, match in matches.iterrows():
            referee = match.get('referee')

            if referee and pd.notna(referee):
                stats = referee_stats.get(referee, {
                    'matches': 0,
                    'home_wins': 0,
                    'draws': 0,
                    'away_wins': 0,
                    'total_goals': 0,
                    'total_yellows': 0,
                    'total_reds': 0,
                })

                if stats['matches'] >= self.min_matches:
                    home_win_pct = stats['home_wins'] / stats['matches']
                    draw_pct = stats['draws'] / stats['matches']
                    away_win_pct = stats['away_wins'] / stats['matches']
                    avg_goals = stats['total_goals'] / stats['matches']
                    # Note: We don't have card data in matches, so this is placeholder
                    avg_yellows = stats['total_yellows'] / stats['matches'] if stats['total_yellows'] > 0 else 3.0
                else:
                    home_win_pct = 0.46
                    draw_pct = 0.25
                    away_win_pct = 0.29
                    avg_goals = 2.7
                    avg_yellows = 3.0

                features = {
                    'fixture_id': match['fixture_id'],
                    'ref_home_win_pct': home_win_pct,
                    'ref_draw_pct': draw_pct,
                    'ref_away_win_pct': away_win_pct,
                    'ref_avg_goals': avg_goals,
                    'ref_matches': stats['matches'],
                    'ref_home_bias': home_win_pct - 0.46,  # Deviation from average
                }

                home_goals = match['ft_home']
                away_goals = match['ft_away']

                if referee not in referee_stats:
                    referee_stats[referee] = {
                        'matches': 0,
                        'home_wins': 0,
                        'draws': 0,
                        'away_wins': 0,
                        'total_goals': 0,
                        'total_yellows': 0,
                        'total_reds': 0,
                    }

                referee_stats[referee]['matches'] += 1
                referee_stats[referee]['total_goals'] += home_goals + away_goals

                if home_goals > away_goals:
                    referee_stats[referee]['home_wins'] += 1
                elif home_goals == away_goals:
                    referee_stats[referee]['draws'] += 1
                else:
                    referee_stats[referee]['away_wins'] += 1

            else:
                features = {
                    'fixture_id': match['fixture_id'],
                    'ref_home_win_pct': 0.46,
                    'ref_draw_pct': 0.25,
                    'ref_away_win_pct': 0.29,
                    'ref_avg_goals': 2.7,
                    'ref_matches': 0,
                    'ref_home_bias': 0,
                }

            features_list.append(features)

        print(f"Created {len(features_list)} referee features")
        return pd.DataFrame(features_list)



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
    }

    def __init__(self, weather_data: pd.DataFrame = None):
        """
        Args:
            weather_data: Pre-loaded weather data DataFrame with columns:
                - fixture_id or (city, date)
                - temperature, humidity, precipitation, wind_speed, weather_code
        """
        self.weather_data = weather_data

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
