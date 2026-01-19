#!/usr/bin/env python
"""
Weather Data Integration Script

Fetches weather data for all historical matches and integrates it into the feature pipeline.

Usage:
    python experiments/integrate_weather_data.py fetch     # Fetch weather for all matches
    python experiments/integrate_weather_data.py merge     # Merge weather into features file
    python experiments/integrate_weather_data.py status    # Check weather data coverage
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import pandas as pd
import numpy as np
from datetime import datetime
import json
import time

from src.data_collection.weather_collector import WeatherCollector

# Output paths
WEATHER_CACHE_PATH = Path("data/weather_cache.parquet")
FEATURES_PATH = Path("data/03-features/features_all_5leagues_with_odds.csv")


def get_match_locations():
    """Load all matches with their venue information."""
    matches_list = []

    leagues = ['premier_league', 'la_liga', 'serie_a', 'bundesliga', 'ligue_1']
    seasons = ['2019', '2020', '2021', '2022', '2023', '2024', '2025']

    for league in leagues:
        for season in seasons:
            matches_path = Path(f"data/01-raw/{league}/{season}/matches.parquet")
            if matches_path.exists():
                try:
                    df = pd.read_parquet(matches_path)
                    # Extract venue city
                    if 'fixture.venue.city' in df.columns:
                        df['venue_city'] = df['fixture.venue.city']
                    elif 'venue_city' in df.columns:
                        pass  # Already has it
                    else:
                        # Try to infer from home team
                        df['venue_city'] = None

                    # Get fixture_id and date
                    if 'fixture.id' in df.columns:
                        df['fixture_id'] = df['fixture.id']
                    if 'fixture.date' in df.columns:
                        df['date'] = pd.to_datetime(df['fixture.date']).dt.date
                    elif 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date']).dt.date

                    matches_list.append(df[['fixture_id', 'date', 'venue_city']].dropna(subset=['fixture_id', 'date']))
                except Exception as e:
                    print(f"Error loading {matches_path}: {e}")

    if matches_list:
        all_matches = pd.concat(matches_list, ignore_index=True)
        all_matches = all_matches.drop_duplicates(subset=['fixture_id'])
        print(f"Found {len(all_matches)} matches with location data")
        return all_matches

    return pd.DataFrame()


def fetch_weather_for_matches():
    """Fetch weather data for all historical matches."""
    print("="*60)
    print("Fetching Weather Data for Historical Matches")
    print("="*60)

    # Load existing cache if available
    if WEATHER_CACHE_PATH.exists():
        existing = pd.read_parquet(WEATHER_CACHE_PATH)
        existing_ids = set(existing['fixture_id'].values)
        print(f"Found {len(existing_ids)} matches already cached")
    else:
        existing = pd.DataFrame()
        existing_ids = set()

    # Get all matches
    matches = get_match_locations()
    if matches.empty:
        print("No matches found!")
        return

    # Filter to matches not yet cached
    new_matches = matches[~matches['fixture_id'].isin(existing_ids)]
    print(f"Need to fetch weather for {len(new_matches)} new matches")

    if new_matches.empty:
        print("All matches already have weather data cached!")
        return

    # Initialize collector
    collector = WeatherCollector()

    # Fetch weather in batches
    weather_records = []
    batch_size = 50

    for i, (idx, row) in enumerate(new_matches.iterrows()):
        city = row.get('venue_city')
        date = row.get('date')
        fixture_id = row.get('fixture_id')

        if pd.isna(city):
            # Try to infer city from fixture_id or skip
            city = None

        if city:
            date_str = str(date)
            weather = collector.fetch_weather(city, date_str)

            if weather:
                weather['fixture_id'] = fixture_id
                weather_records.append(weather)

        # Progress
        if (i + 1) % batch_size == 0:
            print(f"Processed {i+1}/{len(new_matches)} matches, {len(weather_records)} weather records")
            # Rate limiting
            time.sleep(0.5)

    print(f"\nFetched weather for {len(weather_records)} matches")

    # Combine with existing and save
    if weather_records:
        new_weather = pd.DataFrame(weather_records)
        if not existing.empty:
            all_weather = pd.concat([existing, new_weather], ignore_index=True)
        else:
            all_weather = new_weather

        all_weather = all_weather.drop_duplicates(subset=['fixture_id'])
        WEATHER_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        all_weather.to_parquet(WEATHER_CACHE_PATH)
        print(f"Saved {len(all_weather)} weather records to {WEATHER_CACHE_PATH}")


def merge_weather_into_features():
    """Merge weather data into the main features file."""
    print("="*60)
    print("Merging Weather Data into Features")
    print("="*60)

    # Load weather cache
    if not WEATHER_CACHE_PATH.exists():
        print(f"Weather cache not found at {WEATHER_CACHE_PATH}")
        print("Run 'python experiments/integrate_weather_data.py fetch' first")
        return

    weather_df = pd.read_parquet(WEATHER_CACHE_PATH)
    print(f"Loaded {len(weather_df)} weather records")

    # Load features file
    if not FEATURES_PATH.exists():
        print(f"Features file not found at {FEATURES_PATH}")
        return

    features_df = pd.read_csv(FEATURES_PATH)
    print(f"Loaded {len(features_df)} feature rows")

    # Check if weather features already exist
    weather_cols = [c for c in features_df.columns if c.startswith('weather_')]
    if weather_cols:
        print(f"Warning: Weather features already exist: {weather_cols[:5]}...")
        # Remove existing weather columns
        features_df = features_df.drop(columns=weather_cols)

    # Create weather features from cache
    weather_features = create_weather_features(weather_df)
    print(f"Created weather features for {len(weather_features)} matches")

    # Merge
    merged = features_df.merge(
        weather_features,
        on='fixture_id',
        how='left'
    )

    # Fill missing weather with defaults
    weather_feat_cols = [c for c in merged.columns if c.startswith('weather_')]
    for col in weather_feat_cols:
        if col == 'weather_temp':
            merged[col] = merged[col].fillna(15)
        elif col == 'weather_wind':
            merged[col] = merged[col].fillna(10)
        elif col == 'weather_humidity':
            merged[col] = merged[col].fillna(70)
        elif '_normalized' in col:
            merged[col] = merged[col].fillna(0)
        else:
            merged[col] = merged[col].fillna(0)

    # Check coverage
    coverage = merged[weather_feat_cols[0]].notna().sum() if weather_feat_cols else 0
    print(f"Weather coverage: {coverage}/{len(merged)} matches ({coverage/len(merged)*100:.1f}%)")

    # Save
    output_path = FEATURES_PATH.with_name('features_all_5leagues_with_weather.csv')
    merged.to_csv(output_path, index=False)
    print(f"Saved features with weather to: {output_path}")

    # Also update main features file
    merged.to_csv(FEATURES_PATH, index=False)
    print(f"Updated main features file: {FEATURES_PATH}")


def create_weather_features(weather_df: pd.DataFrame) -> pd.DataFrame:
    """Transform raw weather data into features."""
    features = []

    for _, row in weather_df.iterrows():
        fixture_id = row.get('fixture_id')
        temp = row.get('temperature', 15)
        humidity = row.get('humidity', 70)
        precip = row.get('precipitation', 0)
        wind = row.get('wind_speed', 10)
        weather_code = row.get('weather_code', 0)

        # Handle NaN
        temp = temp if pd.notna(temp) else 15
        humidity = humidity if pd.notna(humidity) else 70
        precip = precip if pd.notna(precip) else 0
        wind = wind if pd.notna(wind) else 10
        weather_code = weather_code if pd.notna(weather_code) else 0

        # Create features
        feat = {
            'fixture_id': fixture_id,
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

        # Weather code interpretation (WMO codes)
        # 0-3: Clear, 45-48: Fog, 51-67: Rain, 71-86: Snow, 95-99: Thunderstorm
        feat['weather_is_clear'] = 1 if weather_code <= 3 else 0
        feat['weather_is_foggy'] = 1 if 45 <= weather_code <= 48 else 0
        feat['weather_is_stormy'] = 1 if weather_code >= 95 else 0

        # Composite adverse score
        feat['weather_adverse_score'] = (
            feat['weather_is_rainy'] +
            feat['weather_is_windy'] +
            feat['weather_high_humidity'] +
            feat['weather_extreme_cold'] +
            feat['weather_extreme_hot']
        )

        features.append(feat)

    return pd.DataFrame(features)


def check_weather_status():
    """Check weather data coverage."""
    print("="*60)
    print("Weather Data Status")
    print("="*60)

    # Check cache
    if WEATHER_CACHE_PATH.exists():
        weather_df = pd.read_parquet(WEATHER_CACHE_PATH)
        print(f"Weather cache: {len(weather_df)} records")

        # Check date range
        if 'date' in weather_df.columns:
            weather_df['date'] = pd.to_datetime(weather_df['date'])
            print(f"Date range: {weather_df['date'].min()} to {weather_df['date'].max()}")

        # Check data quality
        print(f"Temperature coverage: {weather_df['temperature'].notna().sum()}/{len(weather_df)}")
        print(f"Wind coverage: {weather_df['wind_speed'].notna().sum()}/{len(weather_df)}")
        print(f"Precipitation coverage: {weather_df['precipitation'].notna().sum()}/{len(weather_df)}")
    else:
        print(f"Weather cache not found at {WEATHER_CACHE_PATH}")

    # Check features file
    print()
    if FEATURES_PATH.exists():
        features_df = pd.read_csv(FEATURES_PATH)
        weather_cols = [c for c in features_df.columns if c.startswith('weather_')]
        print(f"Features file: {len(features_df)} rows")
        print(f"Weather columns: {len(weather_cols)}")
        if weather_cols:
            print(f"Weather columns: {weather_cols}")
            sample_col = weather_cols[0]
            coverage = features_df[sample_col].notna().sum()
            print(f"Coverage: {coverage}/{len(features_df)} ({coverage/len(features_df)*100:.1f}%)")
    else:
        print(f"Features file not found at {FEATURES_PATH}")


def main():
    parser = argparse.ArgumentParser(description='Weather Data Integration')
    parser.add_argument('action', choices=['fetch', 'merge', 'status'],
                       help='Action to perform')
    args = parser.parse_args()

    if args.action == 'fetch':
        fetch_weather_for_matches()
    elif args.action == 'merge':
        merge_weather_into_features()
    elif args.action == 'status':
        check_weather_status()


if __name__ == "__main__":
    main()
