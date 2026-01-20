#!/usr/bin/env python3
"""
Training entrypoint with MLflow integration.

Trains ML callibration on feature data with experiment tracking.

Usage:
    uv run python entrypoints/train.py --config config/local.yaml
    uv run python entrypoints/train.py --config config/local.yaml --target home_win
    uv run python entrypoints/train.py --config config/local.yaml --model xgboost --no-mlflow
"""
import argparse
import logging
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.config_loader import load_config
from src.pipelines.training_pipeline import TrainingPipeline

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
        description="Train ML model on feature data with MLflow tracking"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file (e.g., config/local.yaml)"
    )
    parser.add_argument(
        "--features",
        type=str,
        default="features.csv",
        help="Features filename in data/03-features/ (default: features.csv)"
    )
    parser.add_argument(
        "--target",
        type=str,
        default="home_win",
        choices=["home_win", "draw", "away_win", "match_result", "total_goals"],
        help="Target variable to predict (default: home_win)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=["random_forest", "xgboost", "lightgbm", "catboost", "logistic_regression"],
        help="Override model type from config"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="MLflow experiment name (default: bettip-{target})"
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="MLflow run name (auto-generated if not specified)"
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow tracking"
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

    if args.model:
        config.model.type = args.model

    setup_logging(config)
    logger = logging.getLogger(__name__)

    logger.info(f"Configuration loaded from: {args.config}")
    logger.info(f"Model type: {config.model.type}")
    logger.info(f"Target: {args.target}")
    logger.info(f"Features file: {args.features}")
    logger.info(f"MLflow tracking: {'disabled' if args.no_mlflow else 'enabled'}")

    try:
        pipeline = TrainingPipeline(config)
        result = pipeline.run(
            features_file=args.features,
            target_column=args.target,
            experiment_name=args.experiment,
            run_name=args.run_name,
            use_mlflow=not args.no_mlflow,
        )

        logger.info("Training completed successfully!")
        logger.info(f"Accuracy: {result['metrics'].accuracy:.4f}")
        logger.info(f"F1 Score: {result['metrics'].f1:.4f}")

        if "run_id" in result:
            logger.info(f"\nMLflow Run ID: {result['run_id']}")
            logger.info("To view results, run: mlflow ui")

        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
