#!/usr/bin/env python3
"""
Entry point for data collection pipeline.

This script collects raw football data from the API
and stores it in data/01-raw/ directory.

Usage:
    # Collect full season
    uv run python3 entrypoints/collect.py --mode season --league premier_league --season 2024

    # Smart update (only recent/changed fixtures - optimal for GitHub Actions)
    uv run python3 entrypoints/collect.py --mode update --strategy smart --season 2025 --days-back 26

    # Bulk collect multiple seasons
    uv run python3 entrypoints/collect.py --mode bulk --start-season 2020 --end-season 2025

    # Scheduled updates
    uv run python3 entrypoints/collect.py --mode scheduled
"""
import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data_collection.match_collector import MatchDataCollector, LEAGUES_CONFIG
from src.data_collection.scheduler import run_scheduled_updates


def setup_logging(log_file: str = None) -> None:
    """Setup logging configuration."""
    handlers = [logging.StreamHandler()]

    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def main():
    """Main entry point for data collection."""
    parser = argparse.ArgumentParser(
        description='Football Data Collection Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect single season
  uv run entrypoints/collect.py --mode season --league premier_league --season 2024

  # Bulk collect multiple seasons
  uv run entrypoints/collect.py --mode bulk --start-season 2020 --end-season 2025

  # Run scheduled updates
  uv run entrypoints/collect.py --mode scheduled

  # Smart update (update only recent/changed fixtures)
  uv run entrypoints/collect.py --mode update --strategy smart --season 2025 --days-back 26

  # Full update (re-fetch all fixtures)
  uv run entrypoints/collect.py --mode update --strategy full --season 2025

  # Live update (update only today's matches)
  uv run entrypoints/collect.py --mode update --strategy live --season 2025

  # Collect with all data types
  uv run entrypoints/collect.py --mode season --season 2024 --include-all
        """
    )

    parser.add_argument(
        '--mode',
        choices=['season', 'bulk', 'scheduled', 'update'],
        default='season',
        help='Collection mode (default: season)'
    )
    parser.add_argument(
        '--league',
        choices=list(LEAGUES_CONFIG.keys()),
        default='premier_league',
        help='League to collect (default: premier_league)'
    )
    parser.add_argument(
        '--season',
        type=int,
        help='Specific season for single season collection'
    )
    parser.add_argument(
        '--start-season',
        type=int,
        default=2020,
        help='Start season for bulk collection (default: 2020)'
    )
    parser.add_argument(
        '--end-season',
        type=int,
        help='End season for bulk collection (default: current year)'
    )
    parser.add_argument(
        '--include-lineups',
        action='store_true',
        help='Include match lineups'
    )
    parser.add_argument(
        '--include-events',
        action='store_true',
        help='Include match events'
    )
    parser.add_argument(
        '--include-player-stats',
        action='store_true',
        help='Include player statistics'
    )
    parser.add_argument(
        '--include-all',
        action='store_true',
        help='Include lineups, events, and player statistics'
    )
    parser.add_argument(
        '--max-fixtures',
        type=int,
        help='Maximum fixtures per season (for testing)'
    )
    parser.add_argument(
        '--force-refresh',
        action='store_true',
        help='Force refresh existing files'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/01-raw',
        help='Base directory for raw data (default: data/01-raw)'
    )
    parser.add_argument(
        '--log-file',
        type=str,
        help='Log file path (optional)'
    )
    parser.add_argument(
        '--strategy',
        choices=['smart', 'full', 'live'],
        default='smart',
        help='Update strategy for update mode (default: smart)'
    )
    parser.add_argument(
        '--days-back',
        type=int,
        default=30,
        help='Days back for smart update (default: 30)'
    )
    parser.add_argument(
        '--max-updates',
        type=int,
        help='Maximum fixtures to update in update mode'
    )

    args = parser.parse_args()

    setup_logging(args.log_file)

    logger = logging.getLogger(__name__)

    include_lineups = args.include_lineups or args.include_all
    include_events = args.include_events or args.include_all
    include_player_stats = args.include_player_stats or args.include_all

    logger.info("=" * 60)
    logger.info("FOOTBALL DATA COLLECTION")
    logger.info("=" * 60)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"League: {args.league}")
    logger.info(f"Data directory: {args.data_dir}")

    try:
        if args.mode == 'season':
            season = args.season or datetime.now().year
            logger.info(f"Season: {season}")
            logger.info(f"Include lineups: {include_lineups}")
            logger.info(f"Include events: {include_events}")
            logger.info(f"Include player stats: {include_player_stats}")

            collector = MatchDataCollector(args.data_dir)

            result = collector.update_fixtures_full(args.league, season)
            success = result.get('status') == 'success'

            if success and (include_lineups or include_events or include_player_stats):
                metadata, fixtures = collector.load_fixtures(args.league, season)

                if fixtures:
                    completed_fixtures = [
                        f for f in fixtures
                        if f.get('fixture.status.short') == 'FT'
                    ]

                    if args.max_fixtures:
                        completed_fixtures = completed_fixtures[:args.max_fixtures]

                    logger.info(f"Collecting detailed data for {len(completed_fixtures)} completed fixtures...")

                    collected_count = 0
                    for idx, fixture in enumerate(completed_fixtures, 1):
                        home_team = fixture['teams.home.name']
                        away_team = fixture['teams.away.name']

                        logger.info(f"[{idx}/{len(completed_fixtures)}] {home_team} vs {away_team}")

                        stats = collector.collect_fixture_details(fixture, args.league, season)

                        if any([stats['events'], stats['lineups'], stats['players']]):
                            collected_count += 1

                        if stats['errors']:
                            logger.warning(f"  Some errors occurred: {', '.join(stats['errors'])}")

                    logger.info(f"Collected detailed data for {collected_count}/{len(completed_fixtures)} fixtures")
                else:
                    logger.warning("No fixtures found to collect detailed data")

            logger.info(f"Season {season} collection: {'SUCCESS' if success else 'FAILED'}")

        elif args.mode == 'bulk':
            end_season = args.end_season or datetime.now().year
            logger.info(f"Start season: {args.start_season}")
            logger.info(f"End season: {end_season}")
            logger.info(f"Include lineups: {include_lineups}")
            logger.info(f"Include events: {include_events}")
            logger.info(f"Include player stats: {include_player_stats}")

            collector = MatchDataCollector(args.data_dir)

            total_seasons = end_season - args.start_season + 1
            successful_seasons = 0

            for season in range(args.start_season, end_season + 1):
                logger.info("=" * 60)
                logger.info(f"Processing season {season} ({season - args.start_season + 1}/{total_seasons})")
                logger.info(f"API Usage: {collector.client.state.get('count', 0)}/{collector.client.daily_limit}")

                remaining = collector.client.daily_limit - collector.client.state.get('count', 0)
                if remaining < 100:
                    logger.warning(f"Approaching daily limit. Remaining: {remaining}")
                    break

                result = collector.update_fixtures_full(args.league, season)
                success = result.get('status') == 'success'

                if success:
                    successful_seasons += 1

                    if include_lineups or include_events or include_player_stats:
                        logger.info(f"Collecting detailed data for season {season}...")
                        metadata, fixtures = collector.load_fixtures(args.league, season)

                        if fixtures:
                            completed_fixtures = [
                                f for f in fixtures
                                if f.get('fixture.status.short') == 'FT'
                            ]

                            if args.max_fixtures:
                                completed_fixtures = completed_fixtures[:args.max_fixtures]

                            for idx, fixture in enumerate(completed_fixtures, 1):
                                logger.info(f"  [{idx}/{len(completed_fixtures)}] Collecting fixture {fixture['fixture.id']}...")
                                stats = collector.collect_fixture_details(fixture, args.league, season)

                                if stats['errors']:
                                    logger.warning(f"    Errors: {', '.join(stats['errors'])}")
                else:
                    logger.error(f"Failed to collect data for season {season}")

            logger.info("=" * 60)
            logger.info("Bulk collection completed!")
            logger.info(f"Successful seasons: {successful_seasons}/{total_seasons}")
            logger.info(f"Final API usage: {collector.client.state.get('count', 0)}/{collector.client.daily_limit}")

        elif args.mode == 'scheduled':
            logger.info("Running scheduled updates...")
            results = run_scheduled_updates(
                leagues=[args.league],
                base_data_dir=args.data_dir
            )

            all_successful = all(results.values())
            if not all_successful:
                sys.exit(1)

        elif args.mode == 'update':
            season = args.season or datetime.now().year
            strategy = args.strategy

            logger.info(f"Season: {season}")
            logger.info(f"Strategy: {strategy}")
            logger.info(f"Days back: {args.days_back}")
            if args.max_updates:
                logger.info(f"Max updates: {args.max_updates}")

            collector = MatchDataCollector(args.data_dir)

            if strategy == 'smart':
                stats = collector.update_fixtures_smart(
                    args.league,
                    season,
                    max_updates=args.max_updates,
                    days_back=args.days_back
                )
            elif strategy == 'full':
                stats = collector.update_fixtures_full(args.league, season)
            elif strategy == 'live':
                stats = collector.update_live_fixtures(args.league, season)

            status = stats.get('status', 'unknown')
            if status == 'error':
                logger.error("Update failed!")
                sys.exit(1)
            elif status == 'up_to_date':
                logger.info("All fixtures are up to date")
            else:
                updated_count = stats.get('updated', 0)
                changed_count = stats.get('changed', 0)
                logger.info(f"Update completed: {updated_count} fixtures checked, {changed_count} changed")

        logger.info("=" * 60)
        logger.info("DATA COLLECTION COMPLETED")
        logger.info("=" * 60)

    except KeyboardInterrupt:
        logger.warning("Collection interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Collection failed: {e}")
        raise


if __name__ == "__main__":
    main()
