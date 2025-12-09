#!/usr/bin/env python3
"""
Run baseline experiments comparing different models.

This script runs experiments with all supported models using default parameters
to establish baseline performance.

Usage:
    uv run python experiments/run_baseline.py
    uv run python experiments/run_baseline.py --target home_win
    uv run python experiments/run_baseline.py --models random_forest xgboost
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import List

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.config_loader import load_config
from src.pipelines.training_pipeline import TrainingPipeline

BASELINE_MODELS = [
    "logistic_regression",
    "random_forest",
    "xgboost",
    "lightgbm",
    "catboost",
]


def setup_logging() -> None:
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run baseline experiments with different models"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/local.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--target",
        type=str,
        default="home_win",
        choices=["home_win", "draw", "away_win", "match_result", "total_goals"],
        help="Target variable to predict"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=BASELINE_MODELS,
        help="Models to run (default: all)"
    )
    parser.add_argument(
        "--features-file",
        type=str,
        default="features.csv",
        help="Features file name"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="MLflow experiment name (default: baseline-{target})"
    )
    return parser.parse_args()


def run_baseline_experiments(
    config_path: str,
    target: str,
    models: List[str],
    features_file: str,
    experiment_name: str
) -> None:
    """
    Run baseline experiments for multiple models.

    Args:
        config_path: Path to config file
        target: Target variable
        models: List of model types to run
        features_file: Features file name
        experiment_name: MLflow experiment name
    """
    logger = logging.getLogger(__name__)

    logger.info("=" * 70)
    logger.info("BASELINE EXPERIMENTS")
    logger.info("=" * 70)
    logger.info(f"Target: {target}")
    logger.info(f"Models: {models}")
    logger.info(f"Experiment: {experiment_name}")
    logger.info("=" * 70)

    # Load base config
    config = load_config(config_path)

    results = []

    for model_type in models:
        logger.info(f"\n{'='*70}")
        logger.info(f"Running: {model_type.upper()}")
        logger.info(f"{'='*70}")

        try:
            config.model.type = model_type

            pipeline = TrainingPipeline(config)
            result = pipeline.run(
                features_file=features_file,
                target_column=target,
                experiment_name=experiment_name,
                run_name=f"baseline-{model_type}",
                use_mlflow=True
            )

            results.append({
                "model": model_type,
                "accuracy": result["metrics"].accuracy,
                "f1": result["metrics"].f1,
                "run_id": result.get("run_id", "N/A")
            })

            logger.info(f"{model_type}: accuracy={result['metrics'].accuracy:.4f}, f1={result['metrics'].f1:.4f}")

        except Exception as e:
            logger.error(f"Failed to run {model_type}: {e}")
            results.append({
                "model": model_type,
                "accuracy": None,
                "f1": None,
                "error": str(e)
            })

    logger.info("\n" + "=" * 70)
    logger.info("BASELINE RESULTS SUMMARY")
    logger.info("=" * 70)
    logger.info(f"{'Model':<25} {'Accuracy':<12} {'F1 Score':<12}")
    logger.info("-" * 50)

    for r in sorted(results, key=lambda x: x.get("accuracy") or 0, reverse=True):
        acc = f"{r['accuracy']:.4f}" if r.get('accuracy') else "FAILED"
        f1 = f"{r['f1']:.4f}" if r.get('f1') else "-"
        logger.info(f"{r['model']:<25} {acc:<12} {f1:<12}")

    logger.info("=" * 70)
    logger.info(f"\nView detailed results: mlflow ui")
    logger.info(f"Then open: http://localhost:5000")


def main() -> int:
    """Main entry point."""
    setup_logging()
    args = parse_args()

    experiment_name = args.experiment_name or f"baseline-{args.target}"

    try:
        run_baseline_experiments(
            config_path=args.config,
            target=args.target,
            models=args.models,
            features_file=args.features_file,
            experiment_name=experiment_name
        )
        return 0
    except Exception as e:
        logging.error(f"Experiments failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
