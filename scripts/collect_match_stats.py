#!/usr/bin/env python
"""Collect match statistics (corners, shots, fouls) for all completed matches."""
import sys
from pathlib import Path

# Unbuffered output
sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.data_collection.api_client import FootballAPIClient

def clean_val(val, default=0):
    """Clean API value to int."""
    if val is None:
        return default
    if isinstance(val, str):
        val = val.replace('%', '').strip()
        try:
            return int(val) if val else default
        except:
            return default
    return int(val) if isinstance(val, (int, float)) else default


def main():
    print('=== MATCH STATISTICS COLLECTOR ===')
    print()

    client = FootballAPIClient()

    # Load matches
    matches = pd.read_parquet('data/01-raw/premier_league/2025/matches.parquet')
    completed = matches[matches['fixture.status.short'] == 'FT']
    print(f'Total completed matches: {len(completed)}')

    # Load existing
    existing_path = Path('data/01-raw/premier_league/2025/match_stats.parquet')
    existing_ids = set()
    existing_data = []

    if existing_path.exists():
        try:
            existing_df = pd.read_parquet(existing_path)
            existing_ids = set(existing_df['fixture_id'].tolist())
            existing_data = existing_df.to_dict('records')
            print(f'Already collected: {len(existing_ids)}')
        except Exception as e:
            print(f'Could not read existing: {e}')

    to_collect = completed[~completed['fixture.id'].isin(existing_ids)]
    print(f'Need to collect: {len(to_collect)}')
    print()

    all_stats = list(existing_data)
    new_count = 0
    errors = 0

    for i, (_, match) in enumerate(to_collect.iterrows()):
        fixture_id = int(match['fixture.id'])

        try:
            response = client.get_fixture_statistics(fixture_id)

            if response and len(response) >= 2:
                home_stats = {s['type']: s['value'] for s in response[0].get('statistics', [])}
                away_stats = {s['type']: s['value'] for s in response[1].get('statistics', [])}

                record = {
                    'fixture_id': fixture_id,
                    'date': str(match['fixture.date']),
                    'home_team': str(match['teams.home.name']),
                    'away_team': str(match['teams.away.name']),
                    'home_goals': int(match['goals.home'] or 0),
                    'away_goals': int(match['goals.away'] or 0),
                    'home_corners': clean_val(home_stats.get('Corner Kicks')),
                    'away_corners': clean_val(away_stats.get('Corner Kicks')),
                    'home_shots': clean_val(home_stats.get('Total Shots')),
                    'away_shots': clean_val(away_stats.get('Total Shots')),
                    'home_shots_on_target': clean_val(home_stats.get('Shots on Goal')),
                    'away_shots_on_target': clean_val(away_stats.get('Shots on Goal')),
                    'home_fouls': clean_val(home_stats.get('Fouls')),
                    'away_fouls': clean_val(away_stats.get('Fouls')),
                    'home_possession': clean_val(home_stats.get('Ball Possession')),
                    'away_possession': clean_val(away_stats.get('Ball Possession')),
                    'home_offsides': clean_val(home_stats.get('Offsides')),
                    'away_offsides': clean_val(away_stats.get('Offsides')),
                }
                all_stats.append(record)
                new_count += 1

        except Exception as e:
            errors += 1
            if 'Daily limit' in str(e):
                print(f'\nDaily limit hit at {i+1}')
                break

        if (i + 1) % 25 == 0:
            print(f'Progress: {i+1}/{len(to_collect)} | New: {new_count} | Errors: {errors}')

    print()
    print(f'Collected {new_count} new matches')

    # Save
    df = pd.DataFrame(all_stats)
    df.to_parquet(existing_path, index=False)
    print(f'Saved total: {len(df)} matches to {existing_path}')

    # Summary
    df['total_corners'] = df['home_corners'] + df['away_corners']
    df['total_shots'] = df['home_shots'] + df['away_shots']
    df['total_fouls'] = df['home_fouls'] + df['away_fouls']

    print()
    print('=== CORNER STATISTICS ===')
    print(f'Avg corners/match: {df["total_corners"].mean():.1f}')
    print(f'Std dev: {df["total_corners"].std():.1f}')
    print(f'Over 9.5: {(df["total_corners"] > 9.5).mean()*100:.0f}%')
    print(f'Over 10.5: {(df["total_corners"] > 10.5).mean()*100:.0f}%')

    print()
    print('=== SHOTS STATISTICS ===')
    print(f'Avg shots/match: {df["total_shots"].mean():.1f}')
    print(f'Avg home: {df["home_shots"].mean():.1f} | Avg away: {df["away_shots"].mean():.1f}')

    print()
    print('=== FOULS STATISTICS ===')
    print(f'Avg fouls/match: {df["total_fouls"].mean():.1f}')


if __name__ == '__main__':
    main()
