#!/usr/bin/env python3
"""
Run hyperparameter tuning experiments with Optuna.

This script tunes hyperparameters for specified callibration and logs results to MLflow.

Usage:
    uv run python experiments/run_tuning.py
    uv run python experiments/run_tuning.py --model xgboost --trials 100
    uv run python experiments/run_tuning.py --callibration lightgbm catboost --trials 50
"""
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.config_loader import load_config
from src.ml.tuning import HyperparameterTuner, SEARCH_SPACES

AVAILABLE_MODELS = list(SEARCH_SPACES.keys())


def setup_logging() -> None:
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    optuna_logger = logging.getLogger("optuna")
    optuna_logger.setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run hyperparameter tuning with Optuna"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/local.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=AVAILABLE_MODELS,
        help="Single model to tune (default: tune all)"
    )
    parser.add_argument(
        "--callibration",
        type=str,
        nargs="+",
        default=None,
        choices=AVAILABLE_MODELS,
        help="List of callibration to tune"
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=50,
        help="Number of optimization trials per model"
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds"
    )
    parser.add_argument(
        "--target",
        type=str,
        default="home_win",
        choices=["home_win", "draw", "away_win", "match_result"],
        help="Target variable to predict"
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
        help="MLflow experiment name (default: tuning-{target})"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Timeout in seconds per model (default: no timeout)"
    )
    parser.add_argument(
        "--scoring",
        type=str,
        default="accuracy",
        choices=["accuracy", "f1_weighted", "roc_auc", "neg_log_loss"],
        help="Scoring metric for optimization"
    )
    parser.add_argument(
        "--no-pruning",
        action="store_true",
        help="Disable Optuna pruning"
    )
    parser.add_argument(
        "--max-season",
        type=int,
        default=2023,
        help="Maximum season to include in tuning (default: 2023, excludes 2024+ for holdout)"
    )
    return parser.parse_args()


def load_data(config, features_file: str, target: str, max_season: int = 2023):
    """
    Load and prepare data for tuning.

    Args:
        config: Configuration object
        features_file: Name of features CSV file
        target: Target column name
        max_season: Maximum season to include (default: 2023, excludes 2024+ for holdout)

    Returns:
        X_train, y_train filtered by season
    """
    features_path = config.get_features_dir() / features_file

    from src.utils.data_io import load_features
    df = load_features(features_path)
    df['date'] = pd.to_datetime(df['date'], utc=True)

    max_date = pd.Timestamp(f"{max_season + 1}-06-01", tz='UTC')
    df_filtered = df[df['date'] < max_date].copy()

    logger = logging.getLogger(__name__)
    logger.info(f"Filtered data: {len(df)} -> {len(df_filtered)} rows (max_season={max_season})")
    logger.info(f"Date range: {df_filtered['date'].min()} to {df_filtered['date'].max()}")

    exclude_cols = [
        "fixture_id", "date", "home_team_id", "home_team_name",
        "away_team_id", "away_team_name", "round",
        "home_win", "draw", "away_win", "match_result",
        "total_goals", "goal_difference", "league"
    ]

    feature_cols = [c for c in df_filtered.columns if c not in exclude_cols]
    X = df_filtered[feature_cols]
    y = df_filtered[target]

    df_sorted = df_filtered.sort_values("date")
    X = X.loc[df_sorted.index]
    y = y.loc[df_sorted.index]

    logger.info(f"Features: {len(feature_cols)} columns")
    logger.info(f"Samples: {len(X)} rows")

    return X, y


def run_single_model_tuning(
    model_type: str,
    X: pd.DataFrame,
    y: pd.Series,
    args: argparse.Namespace,
    experiment_name: str,
) -> dict:
    """Run tuning for a single model."""
    logger = logging.getLogger(__name__)

    logger.info(f"\n{'='*70}")
    logger.info(f"TUNING: {model_type.upper()}")
    logger.info(f"{'='*70}")
    logger.info(f"Trials: {args.trials}")
    logger.info(f"CV Folds: {args.cv_folds}")
    logger.info(f"Scoring: {args.scoring}")
    logger.info(f"{'='*70}")

    tuner = HyperparameterTuner(
        model_type=model_type,
        experiment_name=f"{experiment_name}-{model_type}",
        n_trials=args.trials,
        cv_folds=args.cv_folds,
        scoring=args.scoring,
        time_series_cv=True,
        pruning=not args.no_pruning,
    )

    best_params = tuner.tune(X, y, timeout=args.timeout)

    result = {
        "model": model_type,
        "best_score": tuner.best_score,
        "best_params": best_params,
        "param_importances": tuner.get_param_importances(),
    }

    logger.info(f"\nBest {args.scoring}: {tuner.best_score:.4f}")
    logger.info(f"Best params: {best_params}")

    importances = tuner.get_param_importances()
    if importances:
        logger.info("\nParameter Importances:")
        for param, imp in sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]:
            logger.info(f"  {param}: {imp:.3f}")

    return result


def print_summary(results: list, scoring: str) -> None:
    """Print summary of all tuning results."""
    logger = logging.getLogger(__name__)

    logger.info("\n" + "=" * 70)
    logger.info("TUNING RESULTS SUMMARY")
    logger.info("=" * 70)
    logger.info(f"{'Model':<20} {'Best ' + scoring:<15} {'Top Parameters'}")
    logger.info("-" * 70)

    for r in sorted(results, key=lambda x: x.get("best_score", 0), reverse=True):
        if "error" in r:
            logger.info(f"{r['model']:<20} {'FAILED':<15} {r['error']}")
        else:
            top_params = ""
            if r.get("param_importances"):
                top = sorted(r["param_importances"].items(), key=lambda x: x[1], reverse=True)[:6]
                top_params = ", ".join([p[0] for p in top])

            logger.info(f"{r['model']:<20} {r['best_score']:<15.4f} {top_params}")

    logger.info("=" * 70)
    logger.info("\nView detailed results: mlflow ui")
    logger.info("Then open: http://localhost:5000")


def main() -> int:
    """Main entry point."""
    setup_logging()
    args = parse_args()
    logger = logging.getLogger(__name__)

    experiment_name = args.experiment_name or f"tuning-{args.target}"

    logger.info("=" * 70)
    logger.info("HYPERPARAMETER TUNING")
    logger.info("=" * 70)
    logger.info(f"Target: {args.target}")
    logger.info(f"Experiment: {experiment_name}")
    logger.info(f"Trials per model: {args.trials}")
    logger.info(f"CV Folds: {args.cv_folds}")
    logger.info(f"Scoring: {args.scoring}")
    logger.info(f"Max season (holdout after): {args.max_season}")
    logger.info("=" * 70)

    try:
        config = load_config(args.config)
        X, y = load_data(config, args.features_file, args.target, args.max_season)

        logger.info(f"Training data: {len(X)} samples, {X.shape[1]} features")
        logger.info(f"Class distribution: {y.value_counts(normalize=True).to_dict()}")

        if args.model:
            models = [args.model]
        elif args.models:
            models = args.models
        else:
            models = AVAILABLE_MODELS

        logger.info(f"Models to tune: {models}")

        results = []
        for model_type in models:
            try:
                result = run_single_model_tuning(
                    model_type, X, y, args, experiment_name
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to tune {model_type}: {e}")
                results.append({"model": model_type, "error": str(e)})

        print_summary(results, args.scoring)

        return 0

    except Exception as e:
        logger.error(f"Tuning failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
