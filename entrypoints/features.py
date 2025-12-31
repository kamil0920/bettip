#!/usr/bin/env python3
"""
Feature engineering entrypoint.

Creates ML-ready features from preprocessed data.

Usage:
    uv run python3 entrypoints/features.py --config config/local.yaml
    uv run python3 entrypoints/features.py --config config/prod.yaml --output ml_features.csv
"""
import argparse
import logging
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.config_loader import load_config
from src.pipelines.feature_eng_pipeline import FeatureEngineeringPipeline


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
        description="Create ML-ready features from preprocessed data"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file (e.g., config/local.yaml)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="features.csv",
        help="Output filename (default: features.csv)"
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

    # Resolve "auto" seasons from preprocessed data directory
    resolved_seasons = config.resolve_seasons(config.data.preprocessed_dir)
    config.seasons = resolved_seasons

    logger.info(f"Configuration loaded from: {args.config}")
    logger.info(f"Seasons: {config.seasons}")
    logger.info(f"League: {config.league}")
    logger.info(f"Output file: {args.output}")

    try:
        pipeline = FeatureEngineeringPipeline(config)
        result = pipeline.run(output_filename=args.output)

        logger.info("Feature engineering completed successfully!")
        logger.info(f"Features shape: {result.shape}")
        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
