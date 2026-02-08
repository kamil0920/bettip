#!/usr/bin/env python
"""Collect match statistics for all leagues and seasons."""
import argparse
import sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.data_collection.api_client import FootballAPIClient, APIError
from src.leagues import EUROPEAN_LEAGUES

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


def collect_league_season(client, league: str, season: str) -> int:
    """Collect stats for a single league/season. Returns count collected."""
    league_path = Path(f'data/01-raw/{league}/{season}')
    matches_file = league_path / 'matches.parquet'
    stats_file = league_path / 'match_stats.parquet'

    if not matches_file.exists():
        return 0

    matches = pd.read_parquet(matches_file)
    completed = matches[matches['fixture.status.short'] == 'FT']

    # Load existing stats
    existing_ids = set()
    existing_data = []
    if stats_file.exists():
        try:
            existing_df = pd.read_parquet(stats_file)
            existing_ids = set(existing_df['fixture_id'].tolist())
            existing_data = existing_df.to_dict('records')
        except:
            pass

    to_collect = completed[~completed['fixture.id'].isin(existing_ids)]

    if len(to_collect) == 0:
        return 0

    print(f'  {league}/{season}: {len(to_collect)} matches to collect...', end=' ', flush=True)

    all_stats = list(existing_data)
    new_count = 0

    for _, match in to_collect.iterrows():
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

        except APIError as e:
            if 'Daily limit' in str(e):
                print(f'LIMIT HIT after {new_count}')
                # Save what we have
                if all_stats:
                    df = pd.DataFrame(all_stats)
                    df.to_parquet(stats_file, index=False)
                raise  # Re-raise to stop all collection
        except:
            pass  # Skip individual failures

    # Save
    if all_stats:
        df = pd.DataFrame(all_stats)
        df.to_parquet(stats_file, index=False)

    print(f'{new_count} collected')
    return new_count


def main():
    parser = argparse.ArgumentParser(description='Collect match statistics')
    parser.add_argument('--leagues', type=str, default=None,
                        help='Space-separated league names (default: all European)')
    args = parser.parse_args()

    print('=== COLLECTING ALL MATCH STATISTICS ===')
    print()

    client = FootballAPIClient()

    if args.leagues:
        leagues = args.leagues.split()
    else:
        leagues = list(EUROPEAN_LEAGUES)

    total_collected = 0

    try:
        for league in leagues:
            league_path = Path(f'data/01-raw/{league}')
            if not league_path.exists():
                continue

            print(f'\n{league.upper()}:')

            # Get all seasons, sorted (newest first for relevance)
            seasons = sorted([d.name for d in league_path.iterdir() if d.is_dir()], reverse=True)

            for season in seasons:
                collected = collect_league_season(client, league, season)
                total_collected += collected

    except APIError as e:
        if 'Daily limit' in str(e):
            print(f'\n*** DAILY API LIMIT REACHED ***')
        else:
            print(f'\nError: {e}')

    print(f'\n=== COLLECTION COMPLETE ===')
    print(f'Total matches collected: {total_collected}')

    # Summary
    print('\n=== DATA SUMMARY ===')
    for league in leagues:
        league_path = Path(f'data/01-raw/{league}')
        if not league_path.exists():
            continue

        total = 0
        for season_dir in league_path.iterdir():
            if not season_dir.is_dir():
                continue
            stats_file = season_dir / 'match_stats.parquet'
            if stats_file.exists():
                df = pd.read_parquet(stats_file)
                total += len(df)

        print(f'  {league}: {total} matches with stats')


if __name__ == '__main__':
    main()
