"""
Weather data collector using Open-Meteo API.

Uses Open-Meteo Historical Weather API (free, no API key required) to fetch
weather conditions for match venues on match days.

Research shows weather affects play style:
- Rain reduces ball control and pass accuracy
- Wind disrupts long passes and crosses
- Extreme temperatures affect player stamina
- High humidity increases fatigue
"""
import json
import time
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)


# Stadium coordinates for major European cities
# This is a fallback mapping when venue coordinates aren't available
CITY_COORDINATES = {
    # Premier League
    "London": (51.5074, -0.1278),
    "Manchester": (53.4808, -2.2426),
    "Liverpool": (53.4084, -2.9916),
    "Birmingham": (52.4862, -1.8904),
    "Leeds": (53.8008, -1.5491),
    "Newcastle upon Tyne": (54.9783, -1.6178),
    "Sheffield": (53.3811, -1.4701),
    "Southampton": (50.9097, -1.4044),
    "Brighton": (50.8225, -0.1372),
    "Falmer, East Sussex": (50.8617, -0.0847),  # Brighton stadium
    "Leicester": (52.6369, -1.1398),
    "Nottingham": (52.9548, -1.1581),
    "Wolverhampton": (52.5870, -2.1288),
    "Bournemouth": (50.7351, -1.8382),
    "Brentford": (51.4882, -0.2889),
    "Fulham": (51.4749, -0.2217),
    "Ipswich": (52.0567, 1.1455),
    "Burnley": (53.7897, -2.2304),
    "Watford": (51.6500, -0.4014),
    "Norwich": (52.6222, 1.3080),
    "Luton": (51.8844, -0.4316),
    "West Bromwich": (52.5090, -1.9636),
    "Huddersfield": (53.6544, -1.7682),
    "Cardiff": (51.4728, -3.2039),
    "Swansea": (51.6423, -3.9346),
    "Stoke-on-Trent": (52.9884, -2.1755),
    "Middlesbrough": (54.5728, -1.2144),
    "Sunderland": (54.9142, -1.3880),

    # La Liga
    "Madrid": (40.4168, -3.7038),
    "Barcelona": (41.3851, 2.1734),
    "Valencia": (39.4699, -0.3763),
    "Sevilla": (37.3891, -5.9845),
    "Seville": (37.3891, -5.9845),
    "Bilbao": (43.2630, -2.9350),
    "San Sebastián": (43.3183, -1.9812),
    "San Sebastian": (43.3183, -1.9812),
    "Málaga": (36.7213, -4.4214),
    "Malaga": (36.7213, -4.4214),
    "Vigo": (42.2406, -8.7207),
    "Pamplona": (42.8125, -1.6458),
    "Iruñea": (42.8125, -1.6458),  # Basque name for Pamplona
    "Valladolid": (41.6523, -4.7245),
    "Getafe": (40.3085, -3.7329),
    "Villarreal": (39.9439, -0.1006),
    "Girona": (41.9794, 2.8214),
    "Las Palmas": (28.1235, -15.4363),
    "Mallorca": (39.5696, 2.6502),
    "Palma": (39.5696, 2.6502),
    "Celta": (42.2406, -8.7207),
    "Almería": (36.8403, -2.4680),
    "Almeria": (36.8403, -2.4680),
    "Granada": (37.1773, -3.5986),
    "Cádiz": (36.5000, -6.2681),
    "Cadiz": (36.5000, -6.2681),
    "Elche": (38.2669, -0.6981),
    "Eibar": (43.1823, -2.4746),
    "Huesca": (42.1367, -0.4089),
    "Leganés": (40.3269, -3.7630),
    "Leganes": (40.3269, -3.7630),
    "Alavés": (42.8370, -2.6877),
    "Alaves": (42.8370, -2.6877),
    "Vitoria-Gasteiz": (42.8370, -2.6877),
    "Cornella": (41.3535, 2.0788),
    "Cornella de Llobregat": (41.3535, 2.0788),
    "La Nucía": (38.6168, -0.1299),
    "La Nucia": (38.6168, -0.1299),
    "Zaragoza": (41.6488, -0.8891),
    "A Coruña": (43.3623, -8.4115),

    # Serie A
    "Milano": (45.4642, 9.1900),
    "Milan": (45.4642, 9.1900),
    "Roma": (41.9028, 12.4964),
    "Rome": (41.9028, 12.4964),
    "Torino": (45.0703, 7.6869),
    "Turin": (45.0703, 7.6869),
    "Napoli": (40.8518, 14.2681),
    "Naples": (40.8518, 14.2681),
    "Firenze": (43.7696, 11.2558),
    "Florence": (43.7696, 11.2558),
    "Genova": (44.4056, 8.9463),
    "Genoa": (44.4056, 8.9463),
    "Bologna": (44.4949, 11.3426),
    "Verona": (45.4384, 10.9916),
    "Bergamo": (45.6983, 9.6773),
    "Udine": (46.0711, 13.2346),
    "Lecce": (40.3516, 18.1718),
    "Cagliari": (39.2238, 9.1217),
    "Parma": (44.8015, 10.3279),
    "Empoli": (43.7188, 10.9488),
    "Monza": (45.5845, 9.2744),
    "Como": (45.8080, 9.0852),
    "Venezia": (45.4408, 12.3155),
    "Venice": (45.4408, 12.3155),
    "Reggio Emilia": (44.6989, 10.6297),
    "Salerno": (40.6824, 14.7681),
    "Frosinone": (41.6400, 13.3452),
    "Benevento": (41.1292, 14.7826),
    "Brescia": (45.5416, 10.2118),
    "Crotone": (39.0808, 17.1170),
    "La Spezia": (44.1024, 9.8241),
    "Ferrara": (44.8381, 11.6198),
    "Cesena": (44.1391, 12.2437),
    "Cremona": (45.1333, 10.0333),
    "Sassuolo": (44.5394, 10.7847),
    "Reggio nell'Emilia": (44.6989, 10.6297),

    # Bundesliga
    "München": (48.1351, 11.5820),
    "Munich": (48.1351, 11.5820),
    "Berlin": (52.5200, 13.4050),
    "Dortmund": (51.5136, 7.4653),
    "Frankfurt": (50.1109, 8.6821),
    "Frankfurt am Main": (50.1109, 8.6821),
    "Stuttgart": (48.7758, 9.1829),
    "Gelsenkirchen": (51.5556, 7.0673),
    "Leipzig": (51.3397, 12.3731),
    "Leverkusen": (51.0459, 6.9840),
    "Bremen": (53.0793, 8.8017),
    "Wolfsburg": (52.4227, 10.7865),
    "Gladbach": (51.1805, 6.4428),
    "Mönchengladbach": (51.1805, 6.4428),
    "Freiburg": (47.9990, 7.8421),
    "Augsburg": (48.3705, 10.8978),
    "Mainz": (49.9929, 8.2473),
    "Hoffenheim": (49.2393, 8.8890),
    "Sinsheim": (49.2393, 8.8890),
    "Bochum": (51.4818, 7.2162),
    "Heidenheim": (48.6783, 10.1528),
    "St. Pauli": (53.5511, 9.9937),
    "Hamburg": (53.5511, 9.9937),
    "Kiel": (54.3233, 10.1228),
    "Köln": (50.9375, 6.9603),
    "Cologne": (50.9375, 6.9603),
    "Düsseldorf": (51.2277, 6.7735),
    "Dusseldorf": (51.2277, 6.7735),
    "Bielefeld": (52.0302, 8.5325),
    "Fürth": (49.4783, 10.9881),
    "Furth": (49.4783, 10.9881),
    "Darmstadt": (49.8728, 8.6512),
    "Paderborn": (51.7189, 8.7544),
    "Union Berlin": (52.4570, 13.5681),
    "Hertha Berlin": (52.5147, 13.2395),
    "Nürnberg": (49.4521, 11.0767),
    "Nuremberg": (49.4521, 11.0767),
    "Hannover": (52.3759, 9.7320),

    # Ligue 1
    "Paris": (48.8566, 2.3522),
    "Marseille": (43.2965, 5.3698),
    "Lyon": (45.7640, 4.8357),
    "Décines-Charpieu": (45.7659, 4.9822),  # Lyon stadium location
    "Lille": (50.6292, 3.0573),
    "Nice": (43.7102, 7.2620),
    "Bordeaux": (44.8378, -0.5792),
    "Nantes": (47.2184, -1.5536),
    "Toulouse": (43.6047, 1.4442),
    "Strasbourg": (48.5734, 7.7521),
    "Rennes": (48.1173, -1.6778),
    "Montpellier": (43.6108, 3.8767),
    "Monaco": (43.7384, 7.4246),
    "Saint-Etienne": (45.4397, 4.3872),
    "Saint-Étienne": (45.4397, 4.3872),
    "Saint-Ètienne": (45.4397, 4.3872),  # Variant accent
    "Reims": (49.2583, 4.0317),
    "Lens": (50.4262, 2.8317),
    "Brest": (48.3904, -4.4861),
    "Auxerre": (47.7980, 3.5674),
    "Angers": (47.4784, -0.5632),
    "Le Havre": (49.4944, 0.1079),
    "Dijon": (47.3220, 5.0415),
    "Clermont-Ferrand": (45.7772, 3.0870),
    "Ajaccio": (41.9192, 8.7386),
    "Lorient": (47.7485, -3.3700),
    "Metz": (49.1193, 6.1757),
    "Amiens": (49.8942, 2.2957),
    "Caen": (49.1829, -0.3707),
    "Guingamp": (48.5593, -3.1522),
    "Troyes": (48.2973, 4.0744),
    "Nîmes": (43.8367, 4.3601),
    "Nimes": (43.8367, 4.3601),
    "Villeneuve d'Ascq": (50.6263, 3.1316),  # Lille metro

    # Additional La Liga
    "Oviedo": (43.3614, -5.8593),

    # Additional Serie A
    "Pisa": (43.7228, 10.4017),

    # Additional Bundesliga
    "Spiesen-Elversberg": (49.3150, 7.1150),
    "Elversberg": (49.3150, 7.1150),

    # Additional UK/Scotland
    "Perth": (56.3950, -3.4308),
}


class WeatherCollector:
    """
    Collects historical weather data for football matches using Open-Meteo API.

    Features collected:
    - Temperature (°C)
    - Precipitation (mm)
    - Wind speed (km/h)
    - Humidity (%)
    - Weather code (WMO standard)
    """

    BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

    def __init__(self, cache_dir: str = "data/weather_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()

    def _get_cache_path(self, city: str, date: str) -> Path:
        """Get cache file path for city and date."""
        safe_city = city.replace(" ", "_").replace("/", "_")
        return self.cache_dir / f"{safe_city}_{date}.json"

    def _load_cache(self, city: str, date: str) -> Optional[Dict]:
        """Load cached weather data."""
        cache_path = self._get_cache_path(city, date)
        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        return None

    def _save_cache(self, city: str, date: str, data: Dict) -> None:
        """Save weather data to cache."""
        cache_path = self._get_cache_path(city, date)
        try:
            with open(cache_path, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def _get_coordinates(self, city: str) -> Optional[Tuple[float, float]]:
        """Get coordinates for a city."""
        # Try exact match first
        if city in CITY_COORDINATES:
            return CITY_COORDINATES[city]

        # Try partial match
        city_lower = city.lower()
        for known_city, coords in CITY_COORDINATES.items():
            if known_city.lower() in city_lower or city_lower in known_city.lower():
                return coords

        return None

    def fetch_weather(self, city: str, date: str) -> Optional[Dict]:
        """
        Fetch weather data for a city on a specific date.

        Args:
            city: City name
            date: Date in YYYY-MM-DD format

        Returns:
            Dict with weather data or None if unavailable
        """
        # Check cache first
        cached = self._load_cache(city, date)
        if cached:
            return cached

        # Get coordinates
        coords = self._get_coordinates(city)
        if not coords:
            logger.warning(f"Unknown city: {city}")
            return None

        lat, lon = coords

        # Fetch from API
        try:
            params = {
                "latitude": lat,
                "longitude": lon,
                "start_date": date,
                "end_date": date,
                "hourly": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,weather_code",
                "timezone": "Europe/London"
            }

            response = self.session.get(self.BASE_URL, params=params, timeout=30)

            if response.status_code == 200:
                data = response.json()

                # Extract afternoon values (typical match time: 15:00-20:00)
                hourly = data.get("hourly", {})
                if hourly:
                    # Get values for afternoon hours (14-20)
                    temps = hourly.get("temperature_2m", [])
                    humidity = hourly.get("relative_humidity_2m", [])
                    precip = hourly.get("precipitation", [])
                    wind = hourly.get("wind_speed_10m", [])
                    weather_code = hourly.get("weather_code", [])

                    # Average afternoon values (hours 14-20)
                    afternoon_slice = slice(14, 21)

                    result = {
                        "city": city,
                        "date": date,
                        "temperature": np.nanmean(temps[afternoon_slice]) if temps else None,
                        "humidity": np.nanmean(humidity[afternoon_slice]) if humidity else None,
                        "precipitation": np.nansum(precip[afternoon_slice]) if precip else None,
                        "wind_speed": np.nanmean(wind[afternoon_slice]) if wind else None,
                        "weather_code": max(weather_code[afternoon_slice]) if weather_code else None,
                        "latitude": lat,
                        "longitude": lon
                    }

                    # Cache the result
                    self._save_cache(city, date, result)
                    return result

            elif response.status_code == 429:
                logger.warning("Rate limited, waiting...")
                time.sleep(5)

        except Exception as e:
            logger.warning(f"Failed to fetch weather for {city} on {date}: {e}")

        return None

    def fetch_weather_batch(self, matches_df: pd.DataFrame,
                           city_col: str = "fixture.venue.city",
                           date_col: str = "fixture.date") -> pd.DataFrame:
        """
        Fetch weather data for a batch of matches.

        Args:
            matches_df: DataFrame with match data
            city_col: Column name for city
            date_col: Column name for date

        Returns:
            DataFrame with weather data for each match
        """
        weather_data = []

        for idx, row in matches_df.iterrows():
            city = row.get(city_col)
            date_raw = row.get(date_col)

            if pd.isna(city) or pd.isna(date_raw):
                weather_data.append({})
                continue

            # Parse date
            if isinstance(date_raw, str):
                date = date_raw[:10]  # YYYY-MM-DD
            elif hasattr(date_raw, 'strftime'):
                date = date_raw.strftime('%Y-%m-%d')
            else:
                weather_data.append({})
                continue

            weather = self.fetch_weather(city, date)

            if weather:
                weather_data.append(weather)
            else:
                weather_data.append({})

            # Rate limiting (Open-Meteo allows 10,000 requests/day)
            time.sleep(0.1)

        return pd.DataFrame(weather_data)


def get_weather_features_from_code(weather_code: int) -> Dict[str, int]:
    """
    Convert WMO weather code to categorical features.

    WMO Weather Codes:
    0: Clear sky
    1-3: Mainly clear, partly cloudy, overcast
    45-48: Fog
    51-57: Drizzle
    61-67: Rain
    71-77: Snow
    80-82: Rain showers
    85-86: Snow showers
    95-99: Thunderstorm
    """
    if pd.isna(weather_code):
        return {"is_clear": 0, "is_rainy": 0, "is_foggy": 0, "is_stormy": 0}

    code = int(weather_code)

    return {
        "is_clear": 1 if code <= 3 else 0,
        "is_rainy": 1 if 51 <= code <= 82 else 0,
        "is_foggy": 1 if 45 <= code <= 48 else 0,
        "is_stormy": 1 if 95 <= code <= 99 else 0
    }


if __name__ == "__main__":
    # Test the collector
    collector = WeatherCollector()

    # Test fetch for a sample match
    weather = collector.fetch_weather("London", "2024-12-01")
    if weather:
        print(f"Weather in London on 2024-12-01:")
        print(f"  Temperature: {weather['temperature']:.1f}°C")
        print(f"  Humidity: {weather['humidity']:.0f}%")
        print(f"  Precipitation: {weather['precipitation']:.1f}mm")
        print(f"  Wind: {weather['wind_speed']:.1f}km/h")
