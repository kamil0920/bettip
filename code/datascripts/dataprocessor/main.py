import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from config import ProcessingConfig
from factory import DataProcessorFactory
from exceptions import DataProcessingError, FileLoadError


def setup_logging(level: int = logging.INFO, log_file: Optional[str] = None) -> None:
    """
    Configures logging with file and console handlers.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
    """
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    handlers = [console_handler]
    if log_file:
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    logging.basicConfig(
        level=level,
        handlers=handlers
    )

    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    """
    Parses CLI arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Process football season data into ML-ready format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single season
  uv run python3 main.py --seasons 2020

  # Process multiple seasons
  uv run python3 main.py --seasons 2020 2021 2022

  # Process with custom settings
  uv run python3 main.py --seasons 2020 2021 --league premier_league --no-events --debug

  # Process all available seasons
  uv run python3 main.py --seasons 2020 2021 2022 2023 --base-dir /data/football
        """
    )

    parser.add_argument(
        "--seasons",
        nargs='+',
        type=int,
        required=True,
        help="Season years to process (e.g., 2020 2021 2022)"
    )

    parser.add_argument(
        "--base-dir",
        type=str,
        default=None,
        help="Base directory for season data (default: data/seasons)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for processed data (default: auto-detect)"
    )

    parser.add_argument(
        "--league",
        type=str,
        default="premier_league",
        help="League name (default: premier_league)"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch processing size (default: 100)"
    )

    parser.add_argument(
        "--no-players",
        action="store_true",
        help="Skip player statistics extraction"
    )

    parser.add_argument(
        "--no-events",
        action="store_true",
        help="Skip events extraction"
    )

    parser.add_argument(
        "--error-handling",
        choices=["log", "raise", "ignore"],
        default="log",
        help="Error handling strategy (default: log)"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    parser.add_argument(
        "--log-file",
        type=str,
        help="Path to log file (optional)"
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate configuration without processing"
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> bool:
    """
    Validates parsed arguments.

    Args:
        args: Parsed arguments

    Returns:
        True if valid, False otherwise
    """
    current_year = 2025
    for season in args.seasons:
        if season < 2019 or season > current_year:
            print(f"❌ Invalid season year: {season}. Must be between 2000 and {current_year}")
            return False

    if args.batch_size <= 0:
        print(f"❌ Batch size must be positive, got: {args.batch_size}")
        return False

    return True


def print_summary(results: dict) -> None:
    """
    Prints processing summary with statistics.

    Args:
        results: Dictionary with processing results
    """
    print("\n" + "=" * 70)
    print("📊 PROCESSING SUMMARY")
    print("=" * 70)

    if 'matches' in results and not results['matches'].empty:
        matches_df = results['matches']
        print(f"⚽ Matches: {len(matches_df):,} rows")
        print(
            f"   - Seasons: {sorted(matches_df['date'].str[:4].unique().tolist()) if 'date' in matches_df.columns else 'N/A'}")
        print(
            f"   - Date range: {matches_df['date'].min() if 'date' in matches_df.columns else 'N/A'} to {matches_df['date'].max() if 'date' in matches_df.columns else 'N/A'}")

    if 'events' in results and not results['events'].empty:
        events_df = results['events']
        print(f"🎯 Events: {len(events_df):,} rows")
        if 'type' in events_df.columns:
            print(f"   - Types: {', '.join(events_df['type'].value_counts().head(3).index.tolist())}")

    if 'players' in results and not results['players'].empty:
        players_df = results['players']
        print(f"👤 Player Stats: {len(players_df):,} rows")
        print(f"   - Unique players: {players_df['player_id'].nunique():,}")
        if 'rating' in players_df.columns:
            avg_rating = players_df['rating'].mean()
            print(f"   - Avg rating: {avg_rating:.2f}" if avg_rating else "   - Avg rating: N/A")

    if 'lineups' in results and not results['lineups'].empty:
        lineups_df = results['lineups']
        print(f"📋 Lineups: {len(lineups_df):,} rows")
        if 'starting' in lineups_df.columns:
            starters = lineups_df['starting'].sum()
            print(f"   - Starters: {starters:,}, Substitutes: {len(lineups_df) - starters:,}")

    if 'teams' in results and not results['teams'].empty:
        teams_df = results['teams']
        print(f"🏟️  Teams: {len(teams_df):,} unique teams")

    print("=" * 70)


def main() -> int:
    """
    Main function of the program.

    Returns:
        Exit code (0 for success, 1 for error)
    """
    args = parse_args()

    setup_logging(
        level=logging.DEBUG if args.debug else logging.INFO,
        log_file=args.log_file
    )

    logger = logging.getLogger(__name__)
    logger.info("🚀 Football Data Processor Started")

    if not validate_args(args):
        logger.error("❌ Argument validation failed")
        return 1

    try:
        logger.info(f"📋 Creating configuration for seasons: {args.seasons}")

        config = ProcessingConfig(
            base_dir=Path(args.base_dir) if args.base_dir else None,
            output_dir=Path(args.output_dir) if args.output_dir else None,
            seasons=args.seasons,
            league=args.league,
            batch_size=args.batch_size,
            error_handling=args.error_handling,
            include_player_features=not args.no_players,
            include_events=not args.no_events
        )

        logger.info(f"⚙️  Configuration:")
        logger.info(f"   - League: {config.league}")
        logger.info(f"   - Seasons: {config.seasons}")
        logger.info(f"   - Base dir: {config.base_dir}")
        logger.info(f"   - Player stats: {config.include_player_features}")
        logger.info(f"   - Events: {config.include_events}")

        if not DataProcessorFactory.validate_config(config):
            logger.error("❌ Configuration validation failed")
            return 1

        if args.validate_only:
            logger.info("✅ Configuration is valid")
            print("\n✅ Configuration validated successfully!")
            return 0

        logger.info("🔧 Creating data processor...")
        processor = DataProcessorFactory.create_season_processor(config)

        logger.info("⚡ Starting data processing...")
        results = processor.process_all_seasons()

        print_summary(results)

        print("\n✅ Processing completed successfully!")
        logger.info("✅ Processing completed successfully")

        return 0

    except FileLoadError as e:
        logger.error(f"❌ File loading error: {e}")
        print(f"\n❌ Error: Could not load required files")
        print(f"   Details: {e}")
        return 1

    except DataProcessingError as e:
        logger.error(f"❌ Data processing error: {e}")
        print(f"\n❌ Error: Data processing failed")
        print(f"   Details: {e}")
        return 1

    except KeyboardInterrupt:
        logger.warning("⚠️  Processing interrupted by user")
        print("\n⚠️  Processing interrupted")
        return 130

    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}", exc_info=True)
        print(f"\n❌ Unexpected error occurred")
        print(f"   Details: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1

    finally:
        logger.info("🏁 Program finished")


if __name__ == "__main__":
    sys.exit(main())
