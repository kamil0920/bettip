#!/usr/bin/env python3
"""
Entry point for data collection pipeline.

This script collects raw football data from the API
and stores it in data/01-raw/ directory.

Usage:
    uv run python3 entrypoints/collect.py --league premier_league --season 2024
    uv run python3 entrypoints/collect.py --mode bulk --start-season 2020 --end-season 2025
    uv run python3 entrypoints/collect.py --mode scheduled
"""
import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data_collection.collector import (
    FootballDataCollector,
    LEAGUES_CONFIG,
    bulk_collect
)
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

  # Collect with all data types
  uv run entrypoints/collect.py --mode season --season 2024 --include-all
        """
    )

    parser.add_argument(
        '--mode',
        choices=['season', 'bulk', 'scheduled'],
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

            collector = FootballDataCollector(args.data_dir)
            success = collector.collect_season_data(
                args.league, season, args.force_refresh
            )

            if success:
                if include_lineups:
                    collector.collect_match_lineups(
                        args.league, season, args.max_fixtures
                    )
                if include_events:
                    collector.collect_match_events(
                        args.league, season, args.max_fixtures
                    )
                if include_player_stats:
                    collector.collect_player_statistics(
                        args.league, season, args.max_fixtures
                    )

            logger.info(f"Season {season} collection: {'SUCCESS' if success else 'FAILED'}")

        elif args.mode == 'bulk':
            logger.info(f"Start season: {args.start_season}")
            logger.info(f"End season: {args.end_season or datetime.now().year}")
            logger.info(f"Include lineups: {include_lineups}")
            logger.info(f"Include events: {include_events}")
            logger.info(f"Include player stats: {include_player_stats}")

            bulk_collect(
                league_key=args.league,
                start_season=args.start_season,
                end_season=args.end_season,
                include_lineups=include_lineups,
                include_events=include_events,
                include_player_stats=include_player_stats,
                max_fixtures_per_season=args.max_fixtures,
                base_data_dir=args.data_dir
            )

        elif args.mode == 'scheduled':
            logger.info("Running scheduled updates...")
            results = run_scheduled_updates(
                leagues=[args.league],
                base_data_dir=args.data_dir
            )

            all_successful = all(results.values())
            if not all_successful:
                sys.exit(1)

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
