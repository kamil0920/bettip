#!/usr/bin/env python3
"""
Run ensemble experiments comparing Voting and Stacking classifiers.

This script creates ensemble callibration from the base callibration and evaluates
their performance against individual callibration.

Usage:
    uv run python experiments/run_ensemble.py
    uv run python experiments/run_ensemble.py --target home_win
    uv run python experiments/run_ensemble.py --use-tuned-params
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import mlflow
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.config_loader import load_config
from src.ml.ensemble import (
    EnsembleFactory,
    evaluate_ensemble,
    get_ensemble_feature_importance,
)
from src.ml.metrics import SportsMetrics
from src.ml.models import ModelFactory

BASE_MODELS = ["random_forest", "xgboost", "lightgbm", "catboost"]


def setup_logging() -> None:
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run ensemble experiments"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/local.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="home_win",
        choices=["home_win", "draw", "away_win", "match_result"],
        help="Target variable to predict",
    )
    parser.add_argument(
        "--features-file",
        type=str,
        default="features.csv",
        help="Features file name",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="MLflow experiment name (default: ensemble-{target})",
    )
    parser.add_argument(
        "--use-tuned-params",
        action="store_true",
        help="Use tuned hyperparameters from MLflow (if available)",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds",
    )
    parser.add_argument(
        "--callibration",
        type=str,
        nargs="+",
        default=BASE_MODELS,
        help="Base callibration to include in ensemble",
    )
    return parser.parse_args()


def load_data(config, features_file: str, target: str):
    """Load and prepare data."""
    features_path = config.get_features_dir() / features_file

    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")

    df = pd.read_csv(features_path)

    # Sort by date for time-based split
    df = df.sort_values("date")

    # Exclude non-feature columns
    exclude_cols = [
        "fixture_id", "date", "home_team_id", "home_team_name",
        "away_team_id", "away_team_name", "round", "league",
        "home_win", "draw", "away_win", "match_result",
        "total_goals", "goal_difference",
    ]

    feature_cols = [c for c in df.columns if c not in exclude_cols]
    X = df[feature_cols].fillna(0)
    y = df[target]

    # Time-based split (80/20)
    split_idx = int(len(df) * 0.8)

    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    return X_train, X_test, y_train, y_test, feature_cols


def get_tuned_params_from_mlflow(target: str) -> Dict[str, Dict[str, Any]]:
    """Try to load tuned hyperparameters from MLflow."""
    logger = logging.getLogger(__name__)
    tuned_params = {}

    for model_type in BASE_MODELS:
        exp_name = f"phase3-tuning-{model_type}"
        try:
            exp = mlflow.get_experiment_by_name(exp_name)
            if exp:
                runs = mlflow.search_runs(
                    experiment_ids=[exp.experiment_id],
                    max_results=1,
                    order_by=["metrics.best_cv_accuracy DESC"],
                )
                if len(runs) > 0:
                    # Extract best params from run
                    param_cols = [c for c in runs.columns if c.startswith("params.best_")]
                    params = {}
                    for col in param_cols:
                        param_name = col.replace("params.best_", "")
                        value = runs.iloc[0][col]
                        if pd.notna(value):
                            # Try to convert to appropriate type
                            try:
                                if "." in str(value):
                                    params[param_name] = float(value)
                                else:
                                    params[param_name] = int(value)
                            except (ValueError, TypeError):
                                params[param_name] = value
                    if params:
                        tuned_params[model_type] = params
                        logger.info(f"Loaded tuned params for {model_type}")
        except Exception as e:
            logger.debug(f"Could not load tuned params for {model_type}: {e}")

    return tuned_params


def run_individual_baselines(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    models: List[str],
    model_params: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Run individual callibration for baseline comparison."""
    from sklearn.metrics import f1_score

    logger = logging.getLogger(__name__)
    results = {}

    for model_type in models:
        logger.info(f"Training baseline: {model_type}")
        params = model_params.get(model_type, {})
        model = ModelFactory.create(model_type, params=params)

        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        train_f1 = f1_score(y_train, train_pred, average="weighted")
        test_f1 = f1_score(y_test, test_pred, average="weighted")
        test_acc = model.score(X_test, y_test)

        results[model_type] = {
            "train_f1": train_f1,
            "test_f1": test_f1,
            "test_accuracy": test_acc,
            "model": model,
        }
        logger.info(f"  {model_type}: train_f1={train_f1:.4f}, test_f1={test_f1:.4f}")

    return results


def run_ensemble_experiments(
    config_path: str,
    target: str,
    features_file: str,
    experiment_name: str,
    use_tuned_params: bool,
    cv_folds: int,
    models: List[str],
) -> None:
    """Run ensemble experiments."""
    logger = logging.getLogger(__name__)

    logger.info("=" * 70)
    logger.info("ENSEMBLE EXPERIMENTS")
    logger.info("=" * 70)
    logger.info(f"Target: {target}")
    logger.info(f"Base callibration: {models}")
    logger.info(f"Use tuned params: {use_tuned_params}")
    logger.info("=" * 70)

    # Load config and data
    config = load_config(config_path)
    X_train, X_test, y_train, y_test, feature_cols = load_data(
        config, features_file, target
    )

    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    logger.info(f"Features: {len(feature_cols)}")

    # Setup MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(experiment_name)

    # Get model params (tuned or default)
    if use_tuned_params:
        model_params = get_tuned_params_from_mlflow(target)
        logger.info(f"Using tuned params for: {list(model_params.keys())}")
    else:
        model_params = {}

    # Run individual baselines first
    logger.info("\n" + "=" * 70)
    logger.info("INDIVIDUAL MODEL BASELINES")
    logger.info("=" * 70)

    baseline_results = run_individual_baselines(
        X_train, X_test, y_train, y_test, models, model_params
    )

    # Get F1 scores for weighted voting
    cv_scores = {m: r["test_f1"] for m, r in baseline_results.items()}

    results = []

    # Add baseline results
    for model_type, res in baseline_results.items():
        results.append({
            "model": model_type,
            "type": "baseline",
            "test_f1": res["test_f1"],
            "test_accuracy": res["test_accuracy"],
        })

    # 1. Soft Voting Ensemble
    logger.info("\n" + "=" * 70)
    logger.info("SOFT VOTING ENSEMBLE")
    logger.info("=" * 70)

    with mlflow.start_run(run_name="ensemble-voting-soft"):
        voting_soft = EnsembleFactory.create_voting_ensemble(
            base_models=models,
            model_params=model_params,
            voting="soft",
        )

        voting_soft_results = evaluate_ensemble(
            voting_soft, X_train, y_train, X_test, y_test,
            cv_folds=cv_folds, time_series_cv=True, scoring="f1_weighted"
        )

        mlflow.log_params({
            "ensemble_type": "voting_soft",
            "base_models": str(models),
            "n_base_models": len(models),
        })
        mlflow.log_metrics({
            "cv_f1": voting_soft_results["cv_score_mean"],
            "cv_f1_std": voting_soft_results["cv_score_std"],
            "test_f1": voting_soft_results["test_f1"],
            "test_accuracy": voting_soft_results["test_accuracy"],
        })

        results.append({
            "model": "Voting (Soft)",
            "type": "ensemble",
            "test_f1": voting_soft_results["test_f1"],
            "test_accuracy": voting_soft_results["test_accuracy"],
        })

        logger.info(f"Voting (Soft) - F1: {voting_soft_results['test_f1']:.4f}")

    # 2. Weighted Voting Ensemble
    logger.info("\n" + "=" * 70)
    logger.info("WEIGHTED VOTING ENSEMBLE")
    logger.info("=" * 70)

    with mlflow.start_run(run_name="ensemble-voting-weighted"):
        voting_weighted = EnsembleFactory.create_weighted_voting(
            base_models=models,
            cv_scores=cv_scores,
            model_params=model_params,
        )

        voting_weighted_results = evaluate_ensemble(
            voting_weighted, X_train, y_train, X_test, y_test,
            cv_folds=cv_folds, time_series_cv=True, scoring="f1_weighted"
        )

        mlflow.log_params({
            "ensemble_type": "voting_weighted",
            "base_models": str(models),
            "weights": str([round(w, 3) for w in voting_weighted.weights]),
        })
        mlflow.log_metrics({
            "cv_f1": voting_weighted_results["cv_score_mean"],
            "cv_f1_std": voting_weighted_results["cv_score_std"],
            "test_f1": voting_weighted_results["test_f1"],
            "test_accuracy": voting_weighted_results["test_accuracy"],
        })

        results.append({
            "model": "Voting (Weighted)",
            "type": "ensemble",
            "test_f1": voting_weighted_results["test_f1"],
            "test_accuracy": voting_weighted_results["test_accuracy"],
        })

        logger.info(f"Voting (Weighted) - F1: {voting_weighted_results['test_f1']:.4f}")

    # 3. Stacking Ensemble (Logistic Regression meta-learner)
    logger.info("\n" + "=" * 70)
    logger.info("STACKING ENSEMBLE (LogReg meta-learner)")
    logger.info("=" * 70)

    with mlflow.start_run(run_name="ensemble-stacking-lr"):
        stacking_lr = EnsembleFactory.create_stacking_ensemble(
            base_models=models,
            model_params=model_params,
            meta_learner="logistic_regression",
            cv=cv_folds,
            passthrough=False,
        )

        stacking_lr_results = evaluate_ensemble(
            stacking_lr, X_train, y_train, X_test, y_test,
            cv_folds=cv_folds, time_series_cv=True, scoring="f1_weighted"
        )

        mlflow.log_params({
            "ensemble_type": "stacking",
            "meta_learner": "logistic_regression",
            "base_models": str(models),
            "passthrough": False,
        })
        mlflow.log_metrics({
            "cv_f1": stacking_lr_results["cv_score_mean"],
            "cv_f1_std": stacking_lr_results["cv_score_std"],
            "test_f1": stacking_lr_results["test_f1"],
            "test_accuracy": stacking_lr_results["test_accuracy"],
        })

        results.append({
            "model": "Stacking (LR)",
            "type": "ensemble",
            "test_f1": stacking_lr_results["test_f1"],
            "test_accuracy": stacking_lr_results["test_accuracy"],
        })

        logger.info(f"Stacking (LR) - F1: {stacking_lr_results['test_f1']:.4f}")

    # 4. Stacking with passthrough features
    logger.info("\n" + "=" * 70)
    logger.info("STACKING ENSEMBLE (with passthrough)")
    logger.info("=" * 70)

    with mlflow.start_run(run_name="ensemble-stacking-passthrough"):
        stacking_pt = EnsembleFactory.create_stacking_ensemble(
            base_models=models,
            model_params=model_params,
            meta_learner="logistic_regression",
            cv=cv_folds,
            passthrough=True,
        )

        stacking_pt_results = evaluate_ensemble(
            stacking_pt, X_train, y_train, X_test, y_test,
            cv_folds=cv_folds, time_series_cv=True, scoring="f1_weighted"
        )

        mlflow.log_params({
            "ensemble_type": "stacking",
            "meta_learner": "logistic_regression",
            "base_models": str(models),
            "passthrough": True,
        })
        mlflow.log_metrics({
            "cv_f1": stacking_pt_results["cv_score_mean"],
            "cv_f1_std": stacking_pt_results["cv_score_std"],
            "test_f1": stacking_pt_results["test_f1"],
            "test_accuracy": stacking_pt_results["test_accuracy"],
        })

        results.append({
            "model": "Stacking (LR+features)",
            "type": "ensemble",
            "test_f1": stacking_pt_results["test_f1"],
            "test_accuracy": stacking_pt_results["test_accuracy"],
        })

        logger.info(f"Stacking (LR+features) - F1: {stacking_pt_results['test_f1']:.4f}")

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("ENSEMBLE RESULTS SUMMARY")
    logger.info("=" * 70)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("test_f1", ascending=False)

    logger.info(f"\n{'Model':<25} {'Type':<12} {'Test F1':<12} {'Test Acc':<12}")
    logger.info("-" * 65)

    for _, row in results_df.iterrows():
        logger.info(
            f"{row['model']:<25} {row['type']:<12} {row['test_f1']:.4f}       {row['test_accuracy']:.4f}"
        )

    logger.info("=" * 70)

    # Calculate improvement
    best_baseline = results_df[results_df["type"] == "baseline"]["test_f1"].max()
    best_ensemble = results_df[results_df["type"] == "ensemble"]["test_f1"].max()
    improvement = (best_ensemble - best_baseline) / best_baseline * 100

    logger.info(f"\nBest Baseline F1: {best_baseline:.4f}")
    logger.info(f"Best Ensemble F1: {best_ensemble:.4f}")
    logger.info(f"Improvement: {improvement:+.2f}%")
    logger.info("=" * 70)
    logger.info("\nView detailed results: mlflow ui")
    logger.info("Then open: http://localhost:5000")


def main() -> int:
    """Main entry point."""
    setup_logging()
    args = parse_args()

    experiment_name = args.experiment_name or f"ensemble-{args.target}"

    try:
        run_ensemble_experiments(
            config_path=args.config,
            target=args.target,
            features_file=args.features_file,
            experiment_name=experiment_name,
            use_tuned_params=args.use_tuned_params,
            cv_folds=args.cv_folds,
            models=args.models,
        )
        return 0
    except Exception as e:
        logging.error(f"Ensemble experiments failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
