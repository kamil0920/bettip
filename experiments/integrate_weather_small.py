#!/usr/bin/env python
"""
Weather Integration with Small Date Ranges

Fetches weather for recent matches only (last 2 seasons) to avoid rate limits.

Usage:
    python experiments/integrate_weather_small.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import requests
from datetime import datetime
import time

from src.data_collection.weather_collector import CITY_COORDINATES


def fetch_city_weather_range(city: str, start_date: str, end_date: str, max_retries: int = 3) -> pd.DataFrame:
    """Fetch weather for a city over a date range."""
    coords = None
    if city in CITY_COORDINATES:
        coords = CITY_COORDINATES[city]
    else:
        # Try partial match
        for known_city, c in CITY_COORDINATES.items():
            if known_city.lower() in city.lower() or city.lower() in known_city.lower():
                coords = c
                break

    if not coords:
        return pd.DataFrame()

    lat, lon = coords
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max,weather_code",
        "timezone": "Europe/London"
    }

    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=60)
            if response.status_code == 200:
                data = response.json()
                daily = data.get("daily", {})

                if daily and daily.get("time"):
                    df = pd.DataFrame({
                        'date': pd.to_datetime(daily['time']),
                        'city': city,
                        'temp_max': daily.get('temperature_2m_max'),
                        'temp_min': daily.get('temperature_2m_min'),
                        'precipitation': daily.get('precipitation_sum'),
                        'wind_max': daily.get('wind_speed_10m_max'),
                        'weather_code': daily.get('weather_code'),
                    })
                    df['temperature'] = (df['temp_max'] + df['temp_min']) / 2
                    return df

            elif response.status_code == 429:
                wait_time = 10 * (attempt + 1)
                print(f"  Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"  Error {response.status_code} for {city}")
                return pd.DataFrame()

        except Exception as e:
            print(f"  Error: {e}")
            time.sleep(5)

    return pd.DataFrame()


def create_weather_features(weather_df: pd.DataFrame) -> pd.DataFrame:
    """Create weather features from raw data."""
    df = weather_df.copy()

    df['weather_temp'] = df['temperature']
    df['weather_temp_normalized'] = (df['temperature'] - 15) / 15
    df['weather_precip'] = df['precipitation'].fillna(0)
    df['weather_is_rainy'] = (df['precipitation'] > 0.5).astype(int)
    df['weather_heavy_rain'] = (df['precipitation'] > 5).astype(int)
    df['weather_wind'] = df['wind_max'].fillna(10)
    df['weather_is_windy'] = (df['wind_max'] > 20).astype(int)
    df['weather_very_windy'] = (df['wind_max'] > 35).astype(int)
    df['weather_extreme_cold'] = (df['temperature'] < 5).astype(int)
    df['weather_extreme_hot'] = (df['temperature'] > 28).astype(int)
    df['weather_is_clear'] = (df['weather_code'] <= 3).astype(int)
    df['weather_is_foggy'] = ((df['weather_code'] >= 45) & (df['weather_code'] <= 48)).astype(int)
    df['weather_is_stormy'] = (df['weather_code'] >= 95).astype(int)
    df['weather_adverse_score'] = (
        df['weather_is_rainy'] + df['weather_is_windy'] +
        df['weather_extreme_cold'] + df['weather_extreme_hot']
    )

    return df


def main():
    print("=" * 70)
    print("WEATHER INTEGRATION (Recent Matches Only)")
    print("=" * 70)

    # Get unique cities for recent seasons only (2024, 2025)
    cities = set()
    venue_mapping = {}

    leagues = ['premier_league', 'la_liga', 'serie_a', 'bundesliga', 'ligue_1']
    seasons = ['2024', '2025']  # Recent only

    for league in leagues:
        for season in seasons:
            matches_path = Path(f"data/01-raw/{league}/{season}/matches.parquet")
            if matches_path.exists():
                df = pd.read_parquet(matches_path)
                if 'fixture.venue.city' in df.columns and 'fixture.id' in df.columns:
                    for _, row in df.iterrows():
                        city = row.get('fixture.venue.city')
                        if pd.notna(city):
                            cities.add(city)
                            venue_mapping[row['fixture.id']] = city

    print(f"Found {len(cities)} unique cities for 2024-2025")

    # Filter to cities with coordinates
    valid_cities = []
    for city in cities:
        if city in CITY_COORDINATES:
            valid_cities.append(city)
        else:
            for known_city in CITY_COORDINATES:
                if known_city.lower() in city.lower() or city.lower() in known_city.lower():
                    valid_cities.append(city)
                    break

    valid_cities = list(set(valid_cities))
    print(f"Cities with coordinates: {len(valid_cities)}")

    # Fetch weather for 2024-2025 only (shorter range = faster)
    start_date = "2024-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")

    all_weather = []
    for i, city in enumerate(valid_cities):
        print(f"[{i+1}/{len(valid_cities)}] {city}...", end=" ")
        weather_df = fetch_city_weather_range(city, start_date, end_date)

        if len(weather_df) > 0:
            all_weather.append(weather_df)
            print(f"{len(weather_df)} days")
        else:
            print("no data")

        time.sleep(1)  # Gentler rate limiting

    if not all_weather:
        print("No weather data collected!")
        return

    weather_combined = pd.concat(all_weather, ignore_index=True)
    print(f"\nTotal weather records: {len(weather_combined)}")

    # Create features
    weather_features = create_weather_features(weather_combined)

    # Save cache
    cache_path = Path("data/weather_cache_recent.parquet")
    weather_features.to_parquet(cache_path)
    print(f"Saved weather cache: {cache_path}")

    # Merge with features file
    print("\nMerging into features file...")
    features_path = Path("data/03-features/features_all_5leagues_with_odds.csv")
    df = pd.read_csv(features_path)
    print(f"Loaded {len(df)} rows")

    # Add venue city
    df['venue_city'] = df['fixture_id'].map(venue_mapping)

    # Parse dates
    if 'date' in df.columns:
        df['date_parsed'] = pd.to_datetime(df['date']).dt.date
    elif 'fixture_date' in df.columns:
        df['date_parsed'] = pd.to_datetime(df['fixture_date']).dt.date

    weather_features['date_parsed'] = weather_features['date'].dt.date

    # Remove existing weather columns
    existing = [c for c in df.columns if c.startswith('weather_')]
    if existing:
        df = df.drop(columns=existing)

    # Merge
    weather_cols = ['city', 'date_parsed', 'weather_temp', 'weather_temp_normalized',
                   'weather_precip', 'weather_is_rainy', 'weather_heavy_rain',
                   'weather_wind', 'weather_is_windy', 'weather_very_windy',
                   'weather_extreme_cold', 'weather_extreme_hot',
                   'weather_is_clear', 'weather_is_foggy', 'weather_is_stormy',
                   'weather_adverse_score']

    weather_merge = weather_features[weather_cols].drop_duplicates()

    df = df.merge(
        weather_merge,
        left_on=['venue_city', 'date_parsed'],
        right_on=['city', 'date_parsed'],
        how='left'
    )

    # Clean up
    df = df.drop(columns=['venue_city', 'date_parsed', 'city'], errors='ignore')

    # Fill defaults
    for col in [c for c in df.columns if c.startswith('weather_')]:
        if 'temp' in col and 'normalized' not in col:
            df[col] = df[col].fillna(15)
        else:
            df[col] = df[col].fillna(0)

    # Coverage
    coverage = (df['weather_temp'] != 15).sum()
    print(f"\nWeather coverage: {coverage}/{len(df)} ({coverage/len(df)*100:.1f}%)")

    # Save
    df.to_csv(features_path, index=False)
    print(f"Updated: {features_path}")


if __name__ == "__main__":
    main()
