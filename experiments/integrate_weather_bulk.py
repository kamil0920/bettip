#!/usr/bin/env python
"""
Bulk Weather Data Integration

Faster approach: Fetch weather by city and date range instead of per-match.
Open-Meteo allows date ranges, so we can fetch years of data in few requests.

Usage:
    python experiments/integrate_weather_bulk.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import time
import json

from src.data_collection.weather_collector import CITY_COORDINATES


def fetch_city_weather_range(city: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch weather for a city over a date range in ONE request."""
    if city not in CITY_COORDINATES:
        # Try partial match
        for known_city, coords in CITY_COORDINATES.items():
            if known_city.lower() in city.lower() or city.lower() in known_city.lower():
                lat, lon = coords
                break
        else:
            return pd.DataFrame()
    else:
        lat, lon = CITY_COORDINATES[city]

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max,weather_code",
        "timezone": "Europe/London"
    }

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
                # Calculate average temp
                df['temperature'] = (df['temp_max'] + df['temp_min']) / 2
                return df

        elif response.status_code == 429:
            print(f"Rate limited for {city}, waiting...")
            time.sleep(10)
            return fetch_city_weather_range(city, start_date, end_date)

    except Exception as e:
        print(f"Error fetching weather for {city}: {e}")

    return pd.DataFrame()


def get_unique_cities_from_matches() -> set:
    """Get all unique venue cities from match data."""
    cities = set()

    leagues = ['premier_league', 'la_liga', 'serie_a', 'bundesliga', 'ligue_1']
    seasons = ['2019', '2020', '2021', '2022', '2023', '2024', '2025']

    for league in leagues:
        for season in seasons:
            matches_path = Path(f"data/01-raw/{league}/{season}/matches.parquet")
            if matches_path.exists():
                try:
                    df = pd.read_parquet(matches_path)
                    if 'fixture.venue.city' in df.columns:
                        cities.update(df['fixture.venue.city'].dropna().unique())
                except Exception as e:
                    print(f"Error loading {matches_path}: {e}")

    return cities


def create_weather_features(weather_df: pd.DataFrame) -> pd.DataFrame:
    """Create weather features from raw data."""
    df = weather_df.copy()

    # Normalize temperature
    df['weather_temp'] = df['temperature']
    df['weather_temp_normalized'] = (df['temperature'] - 15) / 15

    # Precipitation features
    df['weather_precip'] = df['precipitation'].fillna(0)
    df['weather_is_rainy'] = (df['precipitation'] > 0.5).astype(int)
    df['weather_heavy_rain'] = (df['precipitation'] > 5).astype(int)

    # Wind features
    df['weather_wind'] = df['wind_max'].fillna(10)
    df['weather_is_windy'] = (df['wind_max'] > 20).astype(int)
    df['weather_very_windy'] = (df['wind_max'] > 35).astype(int)

    # Temperature extremes
    df['weather_extreme_cold'] = (df['temperature'] < 5).astype(int)
    df['weather_extreme_hot'] = (df['temperature'] > 28).astype(int)

    # Weather code interpretation (WMO codes)
    df['weather_is_clear'] = (df['weather_code'] <= 3).astype(int)
    df['weather_is_foggy'] = ((df['weather_code'] >= 45) & (df['weather_code'] <= 48)).astype(int)
    df['weather_is_stormy'] = (df['weather_code'] >= 95).astype(int)

    # Adverse score
    df['weather_adverse_score'] = (
        df['weather_is_rainy'] +
        df['weather_is_windy'] +
        df['weather_extreme_cold'] +
        df['weather_extreme_hot']
    )

    return df


def main():
    print("=" * 70)
    print("BULK WEATHER DATA INTEGRATION")
    print("=" * 70)

    # Get unique cities
    cities = get_unique_cities_from_matches()
    print(f"Found {len(cities)} unique venue cities")

    # Filter to cities we have coordinates for
    valid_cities = []
    for city in cities:
        if city in CITY_COORDINATES:
            valid_cities.append(city)
        else:
            # Try partial match
            for known_city in CITY_COORDINATES:
                if known_city.lower() in city.lower() or city.lower() in known_city.lower():
                    valid_cities.append(city)
                    break

    print(f"Cities with coordinates: {len(valid_cities)}")

    # Date range (2019 to now)
    start_date = "2019-08-01"
    end_date = datetime.now().strftime("%Y-%m-%d")

    # Fetch weather for each city (bulk)
    all_weather = []
    for i, city in enumerate(valid_cities):
        print(f"[{i+1}/{len(valid_cities)}] Fetching weather for {city}...")
        weather_df = fetch_city_weather_range(city, start_date, end_date)

        if len(weather_df) > 0:
            all_weather.append(weather_df)
            print(f"  Got {len(weather_df)} days of data")
        else:
            print(f"  No data")

        # Small delay to avoid rate limits
        time.sleep(0.5)

    if not all_weather:
        print("No weather data collected!")
        return

    # Combine all weather data
    weather_combined = pd.concat(all_weather, ignore_index=True)
    print(f"\nTotal weather records: {len(weather_combined)}")

    # Create features
    weather_features = create_weather_features(weather_combined)

    # Save cache
    cache_path = Path("data/weather_bulk_cache.parquet")
    weather_features.to_parquet(cache_path)
    print(f"Saved weather cache to: {cache_path}")

    # Now merge with features file
    print("\n" + "=" * 70)
    print("MERGING INTO FEATURES FILE")
    print("=" * 70)

    features_path = Path("data/03-features/features_all_5leagues_with_odds.csv")
    df = pd.read_csv(features_path)
    print(f"Loaded {len(df)} feature rows")

    # Get venue city for each match
    # We need to load matches to get venue info
    venue_mapping = {}
    leagues = ['premier_league', 'la_liga', 'serie_a', 'bundesliga', 'ligue_1']
    seasons = ['2019', '2020', '2021', '2022', '2023', '2024', '2025']

    for league in leagues:
        for season in seasons:
            matches_path = Path(f"data/01-raw/{league}/{season}/matches.parquet")
            if matches_path.exists():
                matches_df = pd.read_parquet(matches_path)
                if 'fixture.id' in matches_df.columns and 'fixture.venue.city' in matches_df.columns:
                    for _, row in matches_df.iterrows():
                        venue_mapping[row['fixture.id']] = row['fixture.venue.city']

    print(f"Venue mapping: {len(venue_mapping)} matches")

    # Add venue city to features
    df['venue_city'] = df['fixture_id'].map(venue_mapping)

    # Parse dates
    if 'date' in df.columns:
        df['date_parsed'] = pd.to_datetime(df['date']).dt.date
    elif 'fixture_date' in df.columns:
        df['date_parsed'] = pd.to_datetime(df['fixture_date']).dt.date

    weather_features['date_parsed'] = weather_features['date'].dt.date

    # Remove existing weather columns if any
    existing_weather = [c for c in df.columns if c.startswith('weather_')]
    if existing_weather:
        df = df.drop(columns=existing_weather)
        print(f"Removed {len(existing_weather)} existing weather columns")

    # Merge on city + date
    weather_merge_cols = [
        'city', 'date_parsed',
        'weather_temp', 'weather_temp_normalized',
        'weather_precip', 'weather_is_rainy', 'weather_heavy_rain',
        'weather_wind', 'weather_is_windy', 'weather_very_windy',
        'weather_extreme_cold', 'weather_extreme_hot',
        'weather_is_clear', 'weather_is_foggy', 'weather_is_stormy',
        'weather_adverse_score'
    ]

    weather_for_merge = weather_features[weather_merge_cols].drop_duplicates()

    # Merge
    df = df.merge(
        weather_for_merge,
        left_on=['venue_city', 'date_parsed'],
        right_on=['city', 'date_parsed'],
        how='left'
    )

    # Drop helper columns
    df = df.drop(columns=['venue_city', 'date_parsed', 'city'], errors='ignore')

    # Fill missing weather with defaults
    weather_cols = [c for c in df.columns if c.startswith('weather_')]
    for col in weather_cols:
        if 'temp' in col and 'normalized' not in col:
            df[col] = df[col].fillna(15)
        elif 'precip' in col or 'wind' in col:
            df[col] = df[col].fillna(0)
        elif '_normalized' in col:
            df[col] = df[col].fillna(0)
        else:
            df[col] = df[col].fillna(0)

    # Coverage check
    coverage = (df['weather_temp'] != 15).sum()
    print(f"\nWeather coverage: {coverage}/{len(df)} ({coverage/len(df)*100:.1f}%)")

    # Save
    df.to_csv(features_path, index=False)
    print(f"Updated features file: {features_path}")

    # Summary
    print("\n" + "=" * 70)
    print("WEATHER FEATURES ADDED")
    print("=" * 70)
    for col in weather_cols:
        print(f"  {col}: mean={df[col].mean():.3f}")


if __name__ == "__main__":
    main()
