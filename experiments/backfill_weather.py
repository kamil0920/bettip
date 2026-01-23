#!/usr/bin/env python
"""
Historical Weather Data Backfill

Fetches weather data from Open-Meteo API for all historical matches (2019-2025)
and merges into the features file for training.

Uses bulk date-range requests for efficiency (Open-Meteo allows full year ranges).

Usage:
    python experiments/backfill_weather.py --fetch          # Fetch weather data
    python experiments/backfill_weather.py --merge          # Merge into features
    python experiments/backfill_weather.py --fetch --merge  # Both
    python experiments/backfill_weather.py --analyze        # Analyze weather effects
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import time
from typing import Dict, List, Tuple, Optional

from src.data_collection.weather_collector import CITY_COORDINATES


# Extended city coordinates mapping (includes API-Football venue names)
VENUE_CITY_MAPPING = {
    # Premier League specific venues
    "Manchester": "Manchester",
    "Liverpool": "Liverpool",
    "London": "London",
    "Newcastle upon Tyne": "Newcastle upon Tyne",
    "Birmingham": "Birmingham",
    "Leicester": "Leicester",
    "Brighton": "Brighton",
    "Wolverhampton": "Wolverhampton",
    "Bournemouth": "Bournemouth",
    "Brentford": "London",
    "Fulham": "London",
    "Chelsea": "London",
    "Arsenal": "London",
    "Tottenham": "London",
    "West Ham": "London",
    "Crystal Palace": "London",
    "Southampton": "Southampton",
    "Leeds": "Leeds",
    "Sheffield": "Sheffield",
    "Nottingham": "Nottingham",
    "Ipswich": "Ipswich",

    # La Liga
    "Madrid": "Madrid",
    "Barcelona": "Barcelona",
    "Valencia": "Valencia",
    "Sevilla": "Sevilla",
    "Seville": "Sevilla",
    "Bilbao": "Bilbao",
    "San Sebastián": "San Sebastián",
    "Vigo": "Vigo",
    "Villarreal": "Villarreal",
    "Getafe": "Madrid",
    "Girona": "Girona",
    "Pamplona": "Pamplona",
    "Valladolid": "Valladolid",
    "Mallorca": "Mallorca",
    "Las Palmas": "Las Palmas",

    # Serie A
    "Milano": "Milan",
    "Milan": "Milan",
    "Roma": "Rome",
    "Rome": "Rome",
    "Torino": "Turin",
    "Turin": "Turin",
    "Napoli": "Naples",
    "Naples": "Naples",
    "Firenze": "Florence",
    "Florence": "Florence",
    "Genova": "Genoa",
    "Genoa": "Genoa",
    "Bologna": "Bologna",
    "Verona": "Verona",
    "Bergamo": "Bergamo",
    "Udine": "Udine",
    "Lecce": "Lecce",
    "Cagliari": "Cagliari",
    "Parma": "Parma",
    "Empoli": "Empoli",
    "Monza": "Monza",
    "Como": "Como",
    "Venezia": "Venezia",

    # Bundesliga
    "München": "Munich",
    "Munich": "Munich",
    "Berlin": "Berlin",
    "Dortmund": "Dortmund",
    "Frankfurt": "Frankfurt",
    "Frankfurt am Main": "Frankfurt",
    "Stuttgart": "Stuttgart",
    "Gelsenkirchen": "Gelsenkirchen",
    "Leipzig": "Leipzig",
    "Leverkusen": "Leverkusen",
    "Bremen": "Bremen",
    "Wolfsburg": "Wolfsburg",
    "Gladbach": "Mönchengladbach",
    "Mönchengladbach": "Mönchengladbach",
    "Freiburg": "Freiburg",
    "Augsburg": "Augsburg",
    "Mainz": "Mainz",
    "Hoffenheim": "Sinsheim",
    "Sinsheim": "Sinsheim",
    "Bochum": "Bochum",
    "Heidenheim": "Heidenheim",
    "Hamburg": "Hamburg",
    "Kiel": "Kiel",

    # Ligue 1
    "Paris": "Paris",
    "Marseille": "Marseille",
    "Lyon": "Lyon",
    "Lille": "Lille",
    "Nice": "Nice",
    "Nantes": "Nantes",
    "Toulouse": "Toulouse",
    "Strasbourg": "Strasbourg",
    "Rennes": "Rennes",
    "Montpellier": "Montpellier",
    "Monaco": "Monaco",
    "Saint-Étienne": "Saint-Étienne",
    "Reims": "Reims",
    "Lens": "Lens",
    "Brest": "Brest",
    "Auxerre": "Auxerre",
    "Angers": "Angers",
    "Le Havre": "Le Havre",
}


def fetch_city_weather_range(city: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """Fetch weather for a city over a date range using Open-Meteo archive API."""

    # Get coordinates
    coords = CITY_COORDINATES.get(city)
    if not coords:
        # Try partial match
        for known_city, known_coords in CITY_COORDINATES.items():
            if known_city.lower() in city.lower() or city.lower() in known_city.lower():
                coords = known_coords
                break

    if not coords:
        print(f"  No coordinates for city: {city}")
        return None

    lat, lon = coords

    # Cap end_date to yesterday (archive API doesn't have future or current day data)
    today = datetime.now().date()
    yesterday = today - timedelta(days=1)
    end_date_parsed = datetime.strptime(end_date, "%Y-%m-%d").date()
    if end_date_parsed >= yesterday:
        end_date = str(yesterday)

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max,relative_humidity_2m_max,weather_code",
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
                    'humidity': daily.get('relative_humidity_2m_max'),
                    'weather_code': daily.get('weather_code'),
                })
                df['temperature'] = (df['temp_max'] + df['temp_min']) / 2
                return df
            else:
                print(f"  No daily data in response for {city}")
                return None

        elif response.status_code == 429:
            print(f"  Rate limited, waiting 10s...")
            time.sleep(10)
            return fetch_city_weather_range(city, start_date, end_date)
        else:
            print(f"  API error {response.status_code} for {city}")
            return None

    except Exception as e:
        print(f"  Error fetching weather for {city}: {e}")

    return None


def get_unique_cities_from_matches() -> Dict[str, Tuple[str, str]]:
    """
    Get all unique venue cities from match data with their date ranges.

    Returns:
        Dict mapping city -> (min_date, max_date)
    """
    city_dates = {}

    leagues = ['premier_league', 'la_liga', 'serie_a', 'bundesliga', 'ligue_1']
    seasons = ['2019', '2020', '2021', '2022', '2023', '2024', '2025']

    for league in leagues:
        for season in seasons:
            matches_path = Path(f"data/01-raw/{league}/{season}/matches.parquet")
            if matches_path.exists():
                try:
                    df = pd.read_parquet(matches_path)
                    if 'fixture.venue.city' in df.columns and 'fixture.date' in df.columns:
                        df['date'] = pd.to_datetime(df['fixture.date']).dt.date

                        for city in df['fixture.venue.city'].dropna().unique():
                            city_df = df[df['fixture.venue.city'] == city]
                            min_date = city_df['date'].min()
                            max_date = city_df['date'].max()

                            if city in city_dates:
                                city_dates[city] = (
                                    min(city_dates[city][0], min_date),
                                    max(city_dates[city][1], max_date)
                                )
                            else:
                                city_dates[city] = (min_date, max_date)

                except Exception as e:
                    print(f"Error loading {matches_path}: {e}")

    return city_dates


def create_weather_features(weather_df: pd.DataFrame) -> pd.DataFrame:
    """Create weather features from raw data."""
    df = weather_df.copy()

    # Temperature features
    df['weather_temp'] = df['temperature']
    df['weather_temp_normalized'] = (df['temperature'] - 15) / 15  # Centered at 15°C

    # Precipitation features
    df['weather_precip'] = df['precipitation'].fillna(0)
    df['weather_is_rainy'] = (df['precipitation'] > 0.5).astype(int)
    df['weather_heavy_rain'] = (df['precipitation'] > 5).astype(int)

    # Wind features
    df['weather_wind'] = df['wind_max'].fillna(10)
    df['weather_is_windy'] = (df['wind_max'] > 20).astype(int)
    df['weather_very_windy'] = (df['wind_max'] > 35).astype(int)

    # Humidity features
    df['weather_humidity'] = df['humidity'].fillna(70)
    df['weather_humidity_normalized'] = (df['humidity'] - 70) / 30 if 'humidity' in df.columns else 0
    df['weather_high_humidity'] = (df['humidity'] > 85).astype(int) if 'humidity' in df.columns else 0

    # Temperature extremes
    df['weather_extreme_cold'] = (df['temperature'] < 5).astype(int)
    df['weather_extreme_hot'] = (df['temperature'] > 28).astype(int)

    # Weather code interpretation (WMO codes)
    df['weather_is_clear'] = (df['weather_code'] <= 3).astype(int)
    df['weather_is_foggy'] = ((df['weather_code'] >= 45) & (df['weather_code'] <= 48)).astype(int)
    df['weather_is_stormy'] = (df['weather_code'] >= 95).astype(int)

    # Adverse weather score (0-4)
    df['weather_adverse_score'] = (
        df['weather_is_rainy'] +
        df['weather_is_windy'] +
        df['weather_extreme_cold'] +
        df['weather_extreme_hot']
    )

    return df


def fetch_all_weather(output_path: str = "data/weather_historical_cache.parquet") -> pd.DataFrame:
    """Fetch weather data for all cities and date ranges."""
    print("=" * 70)
    print("FETCHING HISTORICAL WEATHER DATA")
    print("=" * 70)

    # Get cities and their date ranges
    city_dates = get_unique_cities_from_matches()
    print(f"Found {len(city_dates)} unique venue cities")

    # Map venue cities to known cities with coordinates
    valid_cities = {}
    for venue_city, (min_date, max_date) in city_dates.items():
        mapped = VENUE_CITY_MAPPING.get(venue_city, venue_city)
        if mapped in CITY_COORDINATES:
            if mapped in valid_cities:
                valid_cities[mapped] = (
                    min(valid_cities[mapped][0], min_date),
                    max(valid_cities[mapped][1], max_date)
                )
            else:
                valid_cities[mapped] = (min_date, max_date)
        else:
            # Try partial match
            for known_city in CITY_COORDINATES:
                if known_city.lower() in venue_city.lower() or venue_city.lower() in known_city.lower():
                    if known_city in valid_cities:
                        valid_cities[known_city] = (
                            min(valid_cities[known_city][0], min_date),
                            max(valid_cities[known_city][1], max_date)
                        )
                    else:
                        valid_cities[known_city] = (min_date, max_date)
                    break

    print(f"Cities with coordinates: {len(valid_cities)}")

    # Fetch weather for each city
    all_weather = []
    for i, (city, (min_date, max_date)) in enumerate(valid_cities.items()):
        start = str(min_date)
        end = str(max_date)
        print(f"[{i+1}/{len(valid_cities)}] {city}: {start} to {end}")

        weather_df = fetch_city_weather_range(city, start, end)
        if weather_df is not None and len(weather_df) > 0:
            all_weather.append(weather_df)
            print(f"  -> {len(weather_df)} days of data")
        else:
            print(f"  -> No data")

        time.sleep(0.3)  # Rate limiting

    if not all_weather:
        print("No weather data collected!")
        return pd.DataFrame()

    # Combine and create features
    weather_combined = pd.concat(all_weather, ignore_index=True)
    print(f"\nTotal raw weather records: {len(weather_combined)}")

    weather_features = create_weather_features(weather_combined)

    # Save cache
    output_path = Path(output_path)
    weather_features.to_parquet(output_path)
    print(f"Saved weather cache to: {output_path}")

    return weather_features


def build_venue_mapping() -> Dict[int, str]:
    """Build mapping from fixture_id to venue city."""
    venue_mapping = {}

    leagues = ['premier_league', 'la_liga', 'serie_a', 'bundesliga', 'ligue_1']
    seasons = ['2019', '2020', '2021', '2022', '2023', '2024', '2025']

    for league in leagues:
        for season in seasons:
            matches_path = Path(f"data/01-raw/{league}/{season}/matches.parquet")
            if matches_path.exists():
                try:
                    df = pd.read_parquet(matches_path)
                    if 'fixture.id' in df.columns and 'fixture.venue.city' in df.columns:
                        for _, row in df.iterrows():
                            fid = row['fixture.id']
                            city = row['fixture.venue.city']
                            if pd.notna(city):
                                # Map to known city
                                mapped_city = VENUE_CITY_MAPPING.get(city, city)
                                venue_mapping[fid] = mapped_city
                except Exception as e:
                    print(f"Error loading {matches_path}: {e}")

    return venue_mapping


def merge_weather_into_features(
    weather_cache_path: str = "data/weather_historical_cache.parquet",
    features_path: str = "data/03-features/features_all_5leagues_with_odds.csv"
) -> pd.DataFrame:
    """Merge weather data into features file."""
    print("=" * 70)
    print("MERGING WEATHER INTO FEATURES")
    print("=" * 70)

    # Load weather cache
    weather_cache_path = Path(weather_cache_path)
    if not weather_cache_path.exists():
        print(f"Weather cache not found: {weather_cache_path}")
        print("Run with --fetch first to collect weather data")
        return None

    weather_df = pd.read_parquet(weather_cache_path)
    print(f"Loaded {len(weather_df)} weather records")
    print(f"Date range: {weather_df['date'].min()} to {weather_df['date'].max()}")

    # Load features
    features_path = Path(features_path)
    df = pd.read_csv(features_path, low_memory=False)
    print(f"Loaded {len(df)} feature rows")

    # Build venue mapping
    venue_mapping = build_venue_mapping()
    print(f"Venue mapping: {len(venue_mapping)} fixtures")

    # Add venue city to features
    df['venue_city'] = df['fixture_id'].map(venue_mapping)

    # Parse dates
    if 'date' in df.columns:
        df['date_parsed'] = pd.to_datetime(df['date']).dt.date
    elif 'fixture_date' in df.columns:
        df['date_parsed'] = pd.to_datetime(df['fixture_date']).dt.date

    weather_df['date_parsed'] = weather_df['date'].dt.date

    # Remove existing weather columns
    existing_weather = [c for c in df.columns if c.startswith('weather_')]
    if existing_weather:
        df = df.drop(columns=existing_weather)
        print(f"Removed {len(existing_weather)} existing weather columns")

    # Weather columns to merge
    weather_merge_cols = [
        'city', 'date_parsed',
        'weather_temp', 'weather_temp_normalized',
        'weather_precip', 'weather_is_rainy', 'weather_heavy_rain',
        'weather_wind', 'weather_is_windy', 'weather_very_windy',
        'weather_humidity', 'weather_humidity_normalized', 'weather_high_humidity',
        'weather_extreme_cold', 'weather_extreme_hot',
        'weather_is_clear', 'weather_is_foggy', 'weather_is_stormy',
        'weather_adverse_score'
    ]

    weather_for_merge = weather_df[weather_merge_cols].drop_duplicates()

    # Merge on city + date
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
    defaults = {
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
        'weather_extreme_cold': 0,
        'weather_extreme_hot': 0,
        'weather_is_clear': 1,
        'weather_is_foggy': 0,
        'weather_is_stormy': 0,
        'weather_adverse_score': 0,
    }

    for col in weather_cols:
        df[col] = df[col].fillna(defaults.get(col, 0))

    # Coverage check
    has_real_weather = (df['weather_temp'] != 15).sum()
    print(f"\nWeather coverage: {has_real_weather}/{len(df)} ({has_real_weather/len(df)*100:.1f}%)")

    # Save
    df.to_csv(features_path, index=False)
    print(f"Updated features file: {features_path}")

    return df


def analyze_weather_effects(features_path: str = "data/03-features/features_all_5leagues_with_odds.csv"):
    """Analyze weather effects on match outcomes."""
    print("=" * 70)
    print("WEATHER EFFECT ANALYSIS")
    print("=" * 70)

    df = pd.read_csv(features_path, low_memory=False)

    # Check coverage
    has_weather = (df['weather_temp'] != 15).sum() if 'weather_temp' in df.columns else 0
    print(f"Matches with weather data: {has_weather}/{len(df)} ({has_weather/len(df)*100:.1f}%)")

    if has_weather < 100:
        print("Insufficient weather data for analysis")
        return

    # Filter to matches with weather
    df_weather = df[df['weather_temp'] != 15].copy()

    # Calculate outcomes
    targets = ['home_win', 'away_win', 'draw', 'over25', 'btts']
    weather_features = ['weather_temp', 'weather_precip', 'weather_wind',
                       'weather_is_rainy', 'weather_is_windy', 'weather_adverse_score']

    print("\n" + "-" * 50)
    print("Correlation with match outcomes:")
    print("-" * 50)

    for wf in weather_features:
        if wf not in df_weather.columns:
            continue
        print(f"\n{wf}:")
        for target in targets:
            if target in df_weather.columns:
                corr = df_weather[[wf, target]].corr().iloc[0, 1]
                print(f"  {target}: r={corr:.4f}")

    # Rain analysis
    print("\n" + "-" * 50)
    print("Rain effect on goals:")
    print("-" * 50)

    if 'weather_is_rainy' in df_weather.columns and 'total_goals' in df_weather.columns:
        rainy = df_weather[df_weather['weather_is_rainy'] == 1]['total_goals'].mean()
        dry = df_weather[df_weather['weather_is_rainy'] == 0]['total_goals'].mean()
        print(f"  Rainy matches avg goals: {rainy:.2f}")
        print(f"  Dry matches avg goals: {dry:.2f}")
        print(f"  Difference: {rainy - dry:.2f}")

    # Wind analysis
    print("\n" + "-" * 50)
    print("Wind effect on goals:")
    print("-" * 50)

    if 'weather_is_windy' in df_weather.columns and 'total_goals' in df_weather.columns:
        windy = df_weather[df_weather['weather_is_windy'] == 1]['total_goals'].mean()
        calm = df_weather[df_weather['weather_is_windy'] == 0]['total_goals'].mean()
        print(f"  Windy matches avg goals: {windy:.2f}")
        print(f"  Calm matches avg goals: {calm:.2f}")
        print(f"  Difference: {windy - calm:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Historical weather data backfill")
    parser.add_argument('--fetch', action='store_true', help='Fetch weather data from Open-Meteo')
    parser.add_argument('--merge', action='store_true', help='Merge weather into features file')
    parser.add_argument('--analyze', action='store_true', help='Analyze weather effects')
    parser.add_argument('--cache-path', default='data/weather_historical_cache.parquet',
                       help='Weather cache file path')
    parser.add_argument('--features-path', default='data/03-features/features_all_5leagues_with_odds.csv',
                       help='Features file path')

    args = parser.parse_args()

    if not args.fetch and not args.merge and not args.analyze:
        print("Specify --fetch, --merge, or --analyze (or combinations)")
        parser.print_help()
        return

    if args.fetch:
        fetch_all_weather(args.cache_path)

    if args.merge:
        merge_weather_into_features(args.cache_path, args.features_path)

    if args.analyze:
        analyze_weather_effects(args.features_path)


if __name__ == "__main__":
    main()
