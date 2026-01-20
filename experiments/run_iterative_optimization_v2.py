#!/usr/bin/env python3
"""
Iterative Feature Selection + Hyperparameter Tuning Pipeline V2.

Improvements over V1:
- Dynamic feature range: 15 to first zero-importance feature, step 2
- Optimized search spaces based on previous best params
- Better handling of feature interactions

Usage:
    uv run python experiments/run_iterative_optimization_v2.py --features-file features_all_with_odds.csv
"""
import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
import optuna
from optuna.samplers import TPESampler

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.config_loader import load_config
from src.ml.models import ModelFactory


EXCLUDE_COLUMNS = [
    "fixture_id", "date", "home_team_id", "home_team_name",
    "away_team_id", "away_team_name", "round",
    "home_win", "draw", "away_win", "match_result",
    "total_goals", "goal_difference", "league"
]

MODELS = ["lightgbm", "xgboost", "catboost", "logistic_regression"]

# Optimized search spaces based on previous best results
# Narrowed ranges around best values found
OPTIMIZED_SEARCH_SPACES = {
    "lightgbm": {
        "n_estimators": (50, 150),      # Best was 70
        "max_depth": (2, 5),             # Best was 3
        "learning_rate": (0.03, 0.1),    # Best was 0.059
        "subsample": (0.85, 1.0),        # Best was 0.94
        "colsample_bytree": (0.5, 0.8),  # Best was 0.62
        "min_child_samples": (50, 150),  # Best was 96
        "reg_alpha": (1e-4, 0.01),       # Best was 0.005
        "reg_lambda": (1e-4, 0.01),      # Best was 0.001
        "num_leaves": (30, 80),          # Best was 57
    },
    "xgboost": {
        "n_estimators": (80, 180),       # Best was 116
        "max_depth": (2, 4),             # Best was 2
        "learning_rate": (0.02, 0.08),   # Best was 0.044
        "subsample": (0.65, 0.85),       # Best was 0.75
        "colsample_bytree": (0.6, 0.9),  # Best was 0.75
        "min_child_weight": (8, 20),     # Best was 14
        "reg_alpha": (1e-4, 0.01),       # Best was 0.006
        "reg_lambda": (1e-8, 1e-4),      # Best was ~0
    },
    "catboost": {
        "iterations": (250, 500),        # Best was 356
        "depth": (4, 8),                 # Best was 6
        "learning_rate": (0.005, 0.03),  # Best was 0.011
        "subsample": (0.7, 0.9),         # Best was 0.79
        "colsample_bylevel": (0.7, 0.95),# Best was 0.84
        "l2_leaf_reg": (1e-8, 1e-4),     # Best was ~0
        "bagging_temperature": (0.3, 1.0),# Best was 0.65
        "random_strength": (0.001, 0.02),# Best was 0.005
    },
    "logistic_regression": {
        "C": (0.1, 1.0),                 # Best was 0.31
        "solver": ["lbfgs", "saga"],
    },
}


@dataclass
class ModelOptimizationResult:
    """Results for a single model's optimization."""
    model_type: str
    phase1_n_features: int = 0
    phase1_features: List[str] = field(default_factory=list)
    phase1_cv_score: float = 0.0
    phase2_best_params: Dict[str, Any] = field(default_factory=dict)
    phase2_cv_score: float = 0.0
    phase3_n_features: int = 0
    phase3_features: List[str] = field(default_factory=list)
    phase3_cv_score: float = 0.0
    final_features: List[str] = field(default_factory=list)
    final_params: Dict[str, Any] = field(default_factory=dict)
    final_cv_score: float = 0.0
    feature_stability: float = 0.0


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Iterative Optimization V2")
    parser.add_argument("--config", default="config/local.yaml")
    parser.add_argument("--features-file", default="features_all_with_odds.csv")
    parser.add_argument("--target", default="home_win")
    parser.add_argument("--callibration", nargs="+", default=MODELS)
    parser.add_argument("--min-features", type=int, default=15)
    parser.add_argument("--feature-step", type=int, default=2)
    parser.add_argument("--tuning-trials", type=int, default=100)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--scoring", default="f1_weighted")
    parser.add_argument("--output-dir", default="experiments/outputs/iterative_optimization_v2")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def load_data(config, features_file: str, target: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
    """Load data with time-based split."""
    features_path = config.get_features_dir() / features_file
    df = pd.read_csv(features_path).sort_values("date")

    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLUMNS]
    X = df[feature_cols].fillna(0)
    y = df[target]

    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    return X_train, X_test, y_train, y_test, feature_cols


def get_feature_importance(model, feature_cols: List[str]) -> pd.DataFrame:
    """Extract feature importance from trained model."""
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = model.coef_
        if len(coef.shape) > 1:
            importance = np.abs(coef).mean(axis=0)
        else:
            importance = np.abs(coef)
    else:
        importance = np.ones(len(feature_cols))

    return pd.DataFrame({
        "feature": feature_cols,
        "importance": importance
    }).sort_values("importance", ascending=False)


def get_dynamic_feature_range(
    importance_df: pd.DataFrame,
    min_features: int = 15,
    step: int = 2
) -> List[int]:
    """
    Generate feature counts from min_features to first zero-importance feature.

    Returns list like [15, 17, 19, 21, ...] up to the count where importance > 0
    """
    # Find first zero importance index
    zero_idx = (importance_df['importance'] == 0).idxmax() if (importance_df['importance'] == 0).any() else len(importance_df)
    first_zero_rank = importance_df.index.get_loc(zero_idx) if zero_idx in importance_df.index else len(importance_df)

    # If no zeros found, use all features
    if first_zero_rank == 0:
        first_zero_rank = len(importance_df)

    max_features = min(first_zero_rank, len(importance_df))

    # Generate range: min_features, min_features+step, ...
    feature_counts = list(range(min_features, max_features + 1, step))

    # Ensure we include the max if not already
    if feature_counts and feature_counts[-1] != max_features:
        feature_counts.append(max_features)

    return feature_counts


def evaluate_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    features: List[str],
    model_type: str,
    params: Optional[Dict[str, Any]] = None,
    cv_folds: int = 5,
    scoring: str = "f1_weighted"
) -> Tuple[float, float, Any]:
    """Evaluate a feature set with CV and test metrics."""
    model = ModelFactory.create(model_type, params=params)

    cv = TimeSeriesSplit(n_splits=cv_folds)
    cv_scores = cross_val_score(
        model, X_train[features], y_train,
        cv=cv, scoring=scoring, n_jobs=1
    )
    cv_score = np.mean(cv_scores)

    model.fit(X_train[features], y_train)
    y_pred = model.predict(X_test[features])

    if scoring == "accuracy":
        test_score = accuracy_score(y_test, y_pred)
    else:
        test_score = f1_score(y_test, y_pred, average="weighted")

    return cv_score, test_score, model


def select_best_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    all_features: List[str],
    model_type: str,
    params: Optional[Dict[str, Any]] = None,
    min_features: int = 15,
    feature_step: int = 2,
    cv_folds: int = 5,
    scoring: str = "f1_weighted",
    logger: logging.Logger = None
) -> Tuple[List[str], int, float, float, pd.DataFrame, List[int]]:
    """
    Select optimal feature count using dynamic range.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Train on all features to get importance ranking
    model = ModelFactory.create(model_type, params=params)
    model.fit(X_train, y_train)
    importance_df = get_feature_importance(model, all_features)

    # Get dynamic feature range
    feature_counts = get_dynamic_feature_range(importance_df, min_features, feature_step)
    logger.info(f"  Testing feature counts: {feature_counts[:5]}...{feature_counts[-3:]} ({len(feature_counts)} total)")

    best_n = len(all_features)
    best_cv_score = 0.0
    best_test_score = 0.0
    best_features = all_features

    results = []

    for n in feature_counts:
        if n > len(all_features):
            continue

        top_features = importance_df.head(n)["feature"].tolist()
        cv_score, test_score, _ = evaluate_features(
            X_train, X_test, y_train, y_test,
            top_features, model_type, params, cv_folds, scoring
        )

        results.append({
            "n_features": n,
            "cv_score": cv_score,
            "test_score": test_score
        })

        if cv_score > best_cv_score:
            best_cv_score = cv_score
            best_test_score = test_score
            best_n = n
            best_features = top_features
            logger.info(f"  Top {n:3d} features: CV={cv_score:.4f} *** NEW BEST ***")
        elif n % 10 == 0 or n == feature_counts[-1]:
            logger.info(f"  Top {n:3d} features: CV={cv_score:.4f}")

    return best_features, best_n, best_cv_score, best_test_score, importance_df, feature_counts


def create_optuna_objective(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    features: List[str],
    model_type: str,
    cv_folds: int,
    scoring: str
):
    """Create Optuna objective function with optimized search space."""

    search_space = OPTIMIZED_SEARCH_SPACES.get(model_type, {})

    def objective(trial):
        if model_type == "lightgbm":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", *search_space["n_estimators"]),
                "max_depth": trial.suggest_int("max_depth", *search_space["max_depth"]),
                "learning_rate": trial.suggest_float("learning_rate", *search_space["learning_rate"], log=True),
                "subsample": trial.suggest_float("subsample", *search_space["subsample"]),
                "colsample_bytree": trial.suggest_float("colsample_bytree", *search_space["colsample_bytree"]),
                "min_child_samples": trial.suggest_int("min_child_samples", *search_space["min_child_samples"]),
                "reg_alpha": trial.suggest_float("reg_alpha", *search_space["reg_alpha"], log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", *search_space["reg_lambda"], log=True),
                "num_leaves": trial.suggest_int("num_leaves", *search_space["num_leaves"]),
                "random_state": 42,
                "n_jobs": -1,
                "verbose": -1,
            }
        elif model_type == "xgboost":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", *search_space["n_estimators"]),
                "max_depth": trial.suggest_int("max_depth", *search_space["max_depth"]),
                "learning_rate": trial.suggest_float("learning_rate", *search_space["learning_rate"], log=True),
                "subsample": trial.suggest_float("subsample", *search_space["subsample"]),
                "colsample_bytree": trial.suggest_float("colsample_bytree", *search_space["colsample_bytree"]),
                "min_child_weight": trial.suggest_int("min_child_weight", *search_space["min_child_weight"]),
                "reg_alpha": trial.suggest_float("reg_alpha", *search_space["reg_alpha"], log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", *search_space["reg_lambda"], log=True),
                "random_state": 42,
                "n_jobs": -1,
            }
        elif model_type == "catboost":
            params = {
                "iterations": trial.suggest_int("iterations", *search_space["iterations"]),
                "depth": trial.suggest_int("depth", *search_space["depth"]),
                "learning_rate": trial.suggest_float("learning_rate", *search_space["learning_rate"], log=True),
                "subsample": trial.suggest_float("subsample", *search_space["subsample"]),
                "colsample_bylevel": trial.suggest_float("colsample_bylevel", *search_space["colsample_bylevel"]),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", *search_space["l2_leaf_reg"], log=True),
                "bagging_temperature": trial.suggest_float("bagging_temperature", *search_space["bagging_temperature"]),
                "random_strength": trial.suggest_float("random_strength", *search_space["random_strength"], log=True),
                "random_state": 42,
                "verbose": False,
            }
        elif model_type == "logistic_regression":
            params = {
                "C": trial.suggest_float("C", *search_space["C"], log=True),
                "solver": trial.suggest_categorical("solver", search_space["solver"]),
                "max_iter": 1000,
                "random_state": 42,
            }
        else:
            params = {}

        model = ModelFactory.create(model_type, params=params)
        cv = TimeSeriesSplit(n_splits=cv_folds)
        scores = cross_val_score(model, X_train[features], y_train, cv=cv, scoring=scoring, n_jobs=1)
        return np.mean(scores)

    return objective


def tune_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    features: List[str],
    model_type: str,
    n_trials: int,
    cv_folds: int,
    scoring: str,
    logger: logging.Logger
) -> Tuple[Dict[str, Any], float]:
    """Tune hyperparameters using Optuna with optimized search space."""

    objective = create_optuna_objective(X_train, y_train, features, model_type, cv_folds, scoring)

    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info(f"  Best trial: {study.best_trial.number}, CV={study.best_value:.4f}")

    return study.best_params, study.best_value


def jaccard_similarity(set1: List[str], set2: List[str]) -> float:
    """Calculate Jaccard similarity between two feature sets."""
    s1, s2 = set(set1), set(set2)
    if not s1 or not s2:
        return 0.0
    intersection = len(s1 & s2)
    union = len(s1 | s2)
    return intersection / union if union > 0 else 0.0


def optimize_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    all_features: List[str],
    model_type: str,
    min_features: int,
    feature_step: int,
    tuning_trials: int,
    cv_folds: int,
    scoring: str,
    output_dir: Path,
    logger: logging.Logger
) -> ModelOptimizationResult:
    """Run full optimization pipeline for a single model."""
    result = ModelOptimizationResult(model_type=model_type)

    logger.info(f"\n{'='*70}")
    logger.info(f"OPTIMIZING: {model_type.upper()}")
    logger.info(f"{'='*70}")

    # PHASE 1: Feature Selection with Default Params
    logger.info(f"\n[PHASE 1] Feature Selection (default params, step={feature_step})")
    logger.info("-" * 50)

    features_p1, n_p1, cv_p1, test_p1, importance_p1, counts_p1 = select_best_features(
        X_train, X_test, y_train, y_test,
        all_features, model_type, params=None,
        min_features=min_features, feature_step=feature_step,
        cv_folds=cv_folds, scoring=scoring, logger=logger
    )

    result.phase1_n_features = n_p1
    result.phase1_features = features_p1
    result.phase1_cv_score = cv_p1

    logger.info(f"\n  -> Best: {n_p1} features, CV={cv_p1:.4f}")
    importance_p1.to_csv(output_dir / f"{model_type}_phase1_importance.csv", index=False)

    # PHASE 2: Hyperparameter Tuning
    logger.info(f"\n[PHASE 2] Hyperparameter Tuning ({tuning_trials} trials, optimized search space)")
    logger.info("-" * 50)

    best_params, cv_p2 = tune_hyperparameters(
        X_train, y_train, features_p1, model_type,
        tuning_trials, cv_folds, scoring, logger
    )

    result.phase2_best_params = best_params
    result.phase2_cv_score = cv_p2

    logger.info(f"  -> Improvement: CV {cv_p1:.4f} -> {cv_p2:.4f} ({cv_p2 - cv_p1:+.4f})")

    # PHASE 3: Re-validate Feature Selection with Tuned Params
    logger.info(f"\n[PHASE 3] Re-validate Feature Selection (tuned params)")
    logger.info("-" * 50)

    features_p3, n_p3, cv_p3, test_p3, importance_p3, counts_p3 = select_best_features(
        X_train, X_test, y_train, y_test,
        all_features, model_type, params=best_params,
        min_features=min_features, feature_step=feature_step,
        cv_folds=cv_folds, scoring=scoring, logger=logger
    )

    result.phase3_n_features = n_p3
    result.phase3_features = features_p3
    result.phase3_cv_score = cv_p3
    result.feature_stability = jaccard_similarity(features_p1, features_p3)

    logger.info(f"\n  -> Best: {n_p3} features, CV={cv_p3:.4f}")
    logger.info(f"  -> Feature stability: {result.feature_stability:.2%}")
    importance_p3.to_csv(output_dir / f"{model_type}_phase3_importance.csv", index=False)

    # FINAL: Choose Best Configuration
    logger.info(f"\n[FINAL] Selecting Best Configuration")
    logger.info("-" * 50)

    if cv_p3 > cv_p2:
        result.final_features = features_p3
        result.final_params = best_params
        result.final_cv_score = cv_p3
        logger.info(f"  -> Using Phase 3 features ({n_p3} features)")
    else:
        result.final_features = features_p1
        result.final_params = best_params
        result.final_cv_score = cv_p2
        logger.info(f"  -> Using Phase 1 features ({n_p1} features)")

    logger.info(f"  -> Final CV: {result.final_cv_score:.4f}")

    return result


def save_results(results: Dict[str, ModelOptimizationResult], output_dir: Path, logger: logging.Logger):
    """Save all optimization results."""
    summary_data = []
    for model_type, result in results.items():
        summary_data.append({
            "model": model_type,
            "phase1_n_features": result.phase1_n_features,
            "phase1_cv": result.phase1_cv_score,
            "phase2_cv": result.phase2_cv_score,
            "phase3_n_features": result.phase3_n_features,
            "phase3_cv": result.phase3_cv_score,
            "final_n_features": len(result.final_features),
            "final_cv": result.final_cv_score,
            "feature_stability": result.feature_stability,
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_dir / "optimization_summary.csv", index=False)

    for model_type, result in results.items():
        config = {
            "model_type": model_type,
            "features": result.final_features,
            "n_features": len(result.final_features),
            "params": result.final_params,
            "metrics": {"cv_score": result.final_cv_score},
            "feature_stability": result.feature_stability,
            "timestamp": datetime.now().isoformat(),
        }
        with open(output_dir / f"{model_type}_config.json", "w") as f:
            json.dump(config, f, indent=2)

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("FINAL OPTIMIZATION RESULTS")
    logger.info("=" * 70)

    summary_df_sorted = summary_df.sort_values("final_cv", ascending=False)
    for _, row in summary_df_sorted.iterrows():
        logger.info(f"\n{row['model'].upper()}")
        logger.info(f"  Features: {row['final_n_features']}")
        logger.info(f"  Final CV: {row['final_cv']:.4f}")
        logger.info(f"  Stability: {row['feature_stability']:.2%}")

    best = summary_df_sorted.iloc[0]
    logger.info("\n" + "-" * 50)
    logger.info(f"BEST MODEL: {best['model'].upper()} (CV={best['final_cv']:.4f})")

    return summary_df


def main():
    args = parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("ITERATIVE OPTIMIZATION V2 (Optimized Search Space)")
    logger.info("=" * 70)
    logger.info(f"Features file: {args.features_file}")
    logger.info(f"Models: {args.models}")
    logger.info(f"Feature range: {args.min_features}+ (step {args.feature_step})")
    logger.info(f"Tuning trials: {args.tuning_trials}")

    config = load_config(args.config)
    X_train, X_test, y_train, y_test, all_features = load_data(
        config, args.features_file, args.target
    )

    logger.info(f"\nData loaded:")
    logger.info(f"  Train: {len(X_train)} samples")
    logger.info(f"  Test: {len(X_test)} samples")
    logger.info(f"  Features: {len(all_features)}")

    results = {}
    for model_type in args.models:
        try:
            result = optimize_model(
                X_train, X_test, y_train, y_test,
                all_features, model_type,
                min_features=args.min_features,
                feature_step=args.feature_step,
                tuning_trials=args.tuning_trials,
                cv_folds=args.cv_folds,
                scoring=args.scoring,
                output_dir=output_dir,
                logger=logger
            )
            results[model_type] = result
        except Exception as e:
            logger.error(f"Failed to optimize {model_type}: {e}")
            import traceback
            traceback.print_exc()

    if results:
        save_results(results, output_dir, logger)
        logger.info(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
