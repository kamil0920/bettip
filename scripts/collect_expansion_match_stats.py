#!/usr/bin/env python
"""
Collect match statistics for expansion leagues + Big 5 backfill.

Day 1: 5 expansion leagues (~7,000 calls)
Day 2: Big 5 backfill (~1,327 calls)

Uses existing collect_all_stats.py infrastructure (incremental, resumable).

Usage:
    # Day 1: expansion leagues (default)
    python scripts/collect_expansion_match_stats.py

    # Day 2: Big 5 backfill
    python scripts/collect_expansion_match_stats.py --backfill

    # Specific leagues
    python scripts/collect_expansion_match_stats.py --leagues eredivisie portuguese_liga

    # Resume (safe — skips already-collected fixtures automatically)
    python scripts/collect_expansion_match_stats.py
"""
import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.collect_all_stats import collect_league_season, upload_files_to_hf
from src.data_collection.api_client import FootballAPIClient, APIError

EXPANSION_LEAGUES = [
    'eredivisie',
    'portuguese_liga',
    'turkish_super_lig',
    'scottish_premiership',
    'belgian_pro_league',
]

BIG5_LEAGUES = [
    'premier_league',
    'la_liga',
    'serie_a',
    'bundesliga',
    'ligue_1',
]

DAILY_LIMIT = 7500
BUDGET_BUFFER = 500


def get_api_calls_used() -> int:
    """Read current API call count from state.json."""
    state_path = Path('state.json')
    if state_path.exists():
        try:
            from datetime import date
            state = json.loads(state_path.read_text())
            if state.get('date') == str(date.today()):
                return state.get('count', 0)
        except Exception:
            pass
    return 0


def count_pending(league: str) -> int:
    """Count fixtures needing stats collection for a league."""
    import pandas as pd
    league_path = Path(f'data/01-raw/{league}')
    if not league_path.exists():
        return 0

    total = 0
    for season_dir in sorted(league_path.iterdir()):
        if not season_dir.is_dir() or season_dir.name.startswith('.'):
            continue
        matches_file = season_dir / 'matches.parquet'
        stats_file = season_dir / 'match_stats.parquet'
        if not matches_file.exists():
            continue

        matches = pd.read_parquet(matches_file)
        completed = len(matches[matches['fixture.status.short'] == 'FT'])

        existing = 0
        if stats_file.exists():
            try:
                existing = len(pd.read_parquet(stats_file))
            except Exception:
                pass

        total += max(0, completed - existing)

    return total


def main():
    parser = argparse.ArgumentParser(description='Collect expansion league match statistics')
    parser.add_argument('--backfill', action='store_true',
                        help='Backfill Big 5 gaps instead of expansion leagues')
    parser.add_argument('--leagues', type=str, default=None,
                        help='Space-separated league names (overrides default)')
    parser.add_argument('--backfill-cards', action='store_true',
                        help='Re-fetch fixtures missing yellow/red card columns')
    parser.add_argument('--upload', action='store_true',
                        help='Upload modified files to HF Hub after collection')
    args = parser.parse_args()

    if args.leagues:
        leagues = args.leagues.split()
    elif args.backfill:
        leagues = BIG5_LEAGUES
    else:
        leagues = EXPANSION_LEAGUES

    budget = DAILY_LIMIT - BUDGET_BUFFER
    calls_used = get_api_calls_used()
    remaining = budget - calls_used

    print('=' * 60)
    print('MATCH STATS COLLECTION')
    print('=' * 60)
    print(f'Mode:       {"Big 5 backfill" if args.backfill else "Expansion leagues"}')
    print(f'Leagues:    {", ".join(leagues)}')
    print(f'API budget: {budget} (limit {DAILY_LIMIT} - buffer {BUDGET_BUFFER})')
    print(f'Used today: {calls_used}')
    print(f'Remaining:  {remaining}')
    print()

    # Show pending counts per league
    print('Pending fixtures:')
    total_pending = 0
    for league in leagues:
        pending = count_pending(league)
        total_pending += pending
        print(f'  {league}: {pending}')
    print(f'  TOTAL: {total_pending}')

    if total_pending == 0:
        print('\nAll stats already collected!')
        return

    if remaining <= 0:
        print(f'\nNo budget remaining today (used {calls_used}/{budget}).')
        print('Run again tomorrow.')
        return

    print(f'\nEstimated time: ~{total_pending / 300:.0f} min (300 req/min)')
    print()

    client = FootballAPIClient()
    total_collected = 0
    modified_files: list[Path] = []

    try:
        for league in leagues:
            league_path = Path(f'data/01-raw/{league}')
            if not league_path.exists():
                print(f'{league}: no data directory, skipping')
                continue

            print(f'\n{"=" * 40}')
            print(f'{league.upper()}')
            print(f'{"=" * 40}')

            seasons = sorted(
                [d.name for d in league_path.iterdir()
                 if d.is_dir() and not d.name.startswith('.')],
                reverse=False  # oldest first for temporal consistency
            )

            for season in seasons:
                # Check budget before each season
                calls_now = get_api_calls_used()
                if calls_now >= budget:
                    print(f'\n*** BUDGET REACHED ({calls_now}/{budget}) ***')
                    raise APIError('Daily limit reached (budget buffer)')

                collected = collect_league_season(
                    client, league, season,
                    backfill_cards=args.backfill_cards
                )
                total_collected += collected
                if collected > 0:
                    stats_file = league_path / season / 'match_stats.parquet'
                    if stats_file.exists():
                        modified_files.append(stats_file.resolve())

    except APIError as e:
        if 'limit' in str(e).lower():
            print(f'\n*** DAILY API LIMIT REACHED ***')
            print('Collection paused. Run again tomorrow to continue.')
        else:
            print(f'\nAPI Error: {e}')
    except KeyboardInterrupt:
        print('\n\nInterrupted by user. Progress saved (resumable).')

    # Final summary
    calls_final = get_api_calls_used()
    print(f'\n{"=" * 60}')
    print('COLLECTION SUMMARY')
    print(f'{"=" * 60}')
    print(f'New matches collected: {total_collected}')
    print(f'API calls used today:  {calls_final}')
    print(f'Budget remaining:      {budget - calls_final}')
    print()

    # Per-league summary
    print('Stats coverage:')
    import pandas as pd
    for league in leagues:
        league_path = Path(f'data/01-raw/{league}')
        if not league_path.exists():
            continue

        total_completed = 0
        total_stats = 0
        for season_dir in sorted(league_path.iterdir()):
            if not season_dir.is_dir() or season_dir.name.startswith('.'):
                continue
            matches_file = season_dir / 'matches.parquet'
            stats_file = season_dir / 'match_stats.parquet'
            if matches_file.exists():
                matches = pd.read_parquet(matches_file)
                total_completed += len(matches[matches['fixture.status.short'] == 'FT'])
            if stats_file.exists():
                try:
                    total_stats += len(pd.read_parquet(stats_file))
                except Exception:
                    pass

        pct = (total_stats / total_completed * 100) if total_completed > 0 else 0
        print(f'  {league}: {total_stats}/{total_completed} ({pct:.0f}%)')

    if args.upload and modified_files:
        print(f'\nUploading {len(modified_files)} modified files to HF Hub...')
        uploaded = upload_files_to_hf(modified_files)
        print(f'Uploaded {uploaded} files')
    elif args.upload:
        print('\nNo files modified, nothing to upload')


if __name__ == '__main__':
    main()
