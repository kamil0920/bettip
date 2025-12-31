#!/usr/bin/env python3
"""
Preprocessing entrypoint.

Transforms raw JSON data into structured Parquet files.

Usage:
    uv run python3 entrypoints/preprocess.py --config config/local.yaml
    uv run python3 entrypoints/preprocess.py --config config/prod.yaml
"""
import argparse
import logging
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.config_loader import load_config
from src.pipelines.preprocessing_pipeline import PreprocessingPipeline


def setup_logging(config) -> None:
    """Configure logging based on config."""
    logging.basicConfig(
        level=getattr(logging, config.logging.level),
        format=config.logging.format,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Preprocess raw football data into structured format"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file (e.g., config/local.yaml)"
    )
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        help="Override seasons from config (e.g., --seasons 2024 2025)"
    )
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1

    if args.seasons:
        config.seasons = args.seasons

    setup_logging(config)
    logger = logging.getLogger(__name__)

    # Resolve "auto" seasons from raw data directory
    resolved_seasons = config.resolve_seasons(config.data.raw_dir)
    config.seasons = resolved_seasons

    logger.info(f"Configuration loaded from: {args.config}")
    logger.info(f"Seasons: {config.seasons}")
    logger.info(f"League: {config.league}")

    try:
        pipeline = PreprocessingPipeline(config)
        result = pipeline.run()

        logger.info("Preprocessing completed successfully!")
        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
