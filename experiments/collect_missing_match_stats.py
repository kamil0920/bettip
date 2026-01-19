#!/usr/bin/env python
"""
Collect Match Stats for Missing Leagues

Fetches match statistics for Bundesliga and Ligue 1 from API-Football.

Usage:
    python experiments/collect_missing_match_stats.py bundesliga  # Collect Bundesliga
    python experiments/collect_missing_match_stats.py ligue_1     # Collect Ligue 1
    python experiments/collect_missing_match_stats.py all         # Collect all missing
    python experiments/collect_missing_match_stats.py status      # Show status
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
from src.data_collection.match_stats_collector import MatchStatsCollector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def show_status():
    """Show which leagues/seasons have match_stats."""
    leagues = ['premier_league', 'la_liga', 'serie_a', 'bundesliga', 'ligue_1']
    seasons = [2019, 2020, 2021, 2022, 2023, 2024, 2025]

    print("\n" + "=" * 70)
    print("MATCH STATS DATA STATUS")
    print("=" * 70)

    for league in leagues:
        print(f"\n{league}:")
        for season in seasons:
            stats_path = Path(f"data/01-raw/{league}/{season}/match_stats.parquet")
            matches_path = Path(f"data/01-raw/{league}/{season}/matches.parquet")

            if stats_path.exists():
                import pandas as pd
                df = pd.read_parquet(stats_path)
                print(f"  {season}: ✓ {len(df)} matches with stats")
            elif matches_path.exists():
                import pandas as pd
                matches = pd.read_parquet(matches_path)
                completed = len(matches[matches['fixture.status.short'] == 'FT'])
                print(f"  {season}: ✗ MISSING ({completed} completed matches)")
            else:
                print(f"  {season}: - no matches data")


def collect_league_stats(league: str, seasons: list = None):
    """Collect match stats for a specific league."""
    if seasons is None:
        seasons = [2020, 2021, 2022, 2023, 2024, 2025]

    collector = MatchStatsCollector()

    print(f"\n{'=' * 70}")
    print(f"COLLECTING MATCH STATS FOR {league.upper()}")
    print(f"{'=' * 70}")

    for season in seasons:
        matches_path = Path(f"data/01-raw/{league}/{season}/matches.parquet")
        stats_path = Path(f"data/01-raw/{league}/{season}/match_stats.parquet")

        if not matches_path.exists():
            print(f"\n{season}: No matches data - skipping")
            continue

        if stats_path.exists():
            import pandas as pd
            existing = pd.read_parquet(stats_path)
            print(f"\n{season}: Already have {len(existing)} match stats - checking for updates...")
        else:
            print(f"\n{season}: Collecting new...")

        try:
            df = collector.collect_league_stats(league, season)
            if len(df) > 0:
                # Show stats summary
                if 'home_fouls' in df.columns:
                    df['total_fouls'] = df['home_fouls'] + df['away_fouls']
                    print(f"  Total matches: {len(df)}")
                    print(f"  Avg fouls: {df['total_fouls'].mean():.1f}")
                if 'home_corner_kicks' in df.columns:
                    df['total_corners'] = df['home_corner_kicks'] + df['away_corner_kicks']
                    print(f"  Avg corners: {df['total_corners'].mean():.1f}")
                if 'home_total_shots' in df.columns:
                    df['total_shots'] = df['home_total_shots'] + df['away_total_shots']
                    print(f"  Avg shots: {df['total_shots'].mean():.1f}")
        except Exception as e:
            logger.error(f"Error collecting {league} {season}: {e}")


def main():
    parser = argparse.ArgumentParser(description='Collect missing match stats')
    parser.add_argument('action', choices=['bundesliga', 'ligue_1', 'all', 'status'],
                       help='Action to perform')

    args = parser.parse_args()

    if args.action == 'status':
        show_status()
    elif args.action == 'bundesliga':
        collect_league_stats('bundesliga')
    elif args.action == 'ligue_1':
        collect_league_stats('ligue_1')
    elif args.action == 'all':
        collect_league_stats('bundesliga')
        collect_league_stats('ligue_1')


if __name__ == "__main__":
    main()
