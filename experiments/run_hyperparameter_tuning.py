#!/usr/bin/env python3
"""
Hyperparameter Tuning for Away Win Precision (Phase 4)

This script uses Optuna to optimize model hyperparameters with precision
as the primary objective using walk-forward validation.

Models: CatBoost, LightGBM, XGBoost
Objective: Maximize precision (not accuracy)
Validation: Walk-forward (temporal splits)

Usage:
    python experiments/run_hyperparameter_tuning.py
    python experiments/run_hyperparameter_tuning.py --model catboost --n-trials 50
    python experiments/run_hyperparameter_tuning.py --model all --n-trials 30
"""

import argparse
import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from catboost import CatBoostClassifier
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Paths
FEATURES_FILE = Path("data/03-features/features_all_5leagues_with_odds.parquet")
RFE_RESULTS = Path("experiments/outputs/feature_selection/rfe_20260124_112415.json")
OUTPUT_DIR = Path("experiments/outputs/hyperparameter_tuning")

# Exclude columns (data leakage prevention)
EXCLUDE_COLUMNS = [
    "fixture_id", "date", "home_team_id", "home_team_name",
    "away_team_id", "away_team_name", "round", "season", "league",
    "home_win", "draw", "away_win", "match_result", "result",
    "total_goals", "goal_difference",
    "home_goals", "away_goals", "btts",
    "home_shots", "away_shots", "home_shots_on_target", "away_shots_on_target",
    "home_corners", "away_corners", "total_corners",
    "home_fouls", "away_fouls", "total_fouls",
    "home_yellows", "away_yellows", "home_reds", "away_reds",
    "home_possession", "away_possession",
    "under25", "over25", "under35", "over35",
]


@dataclass
class TuningResult:
    """Result of hyperparameter tuning."""
    model_type: str
    best_params: Dict[str, Any]
    best_precision: float
    best_roi: float
    n_trials: int
    study_name: str


class HyperparameterTuner:
    """
    Optuna-based hyperparameter tuning for precision optimization.
    """

    def __init__(
        self,
        n_folds: int = 5,
        threshold: float = 0.60,
        min_odds: float = 2.0,
        max_odds: float = 5.0,
        use_rfe_features: bool = True,
    ):
        self.n_folds = n_folds
        self.threshold = threshold
        self.min_odds = min_odds
        self.max_odds = max_odds
        self.use_rfe_features = use_rfe_features
        self.features_df = None
        self.feature_columns = None

    def load_data(self) -> pd.DataFrame:
        """Load and prepare feature data."""
        if not FEATURES_FILE.exists():
            raise FileNotFoundError(f"Features file not found: {FEATURES_FILE}")

        from src.utils.data_io import load_features
        df = load_features(FEATURES_FILE)
        logger.info(f"Loaded {len(df)} matches")

        # Sort by date
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)

        # Ensure target exists
        if "away_win" not in df.columns:
            if "result" in df.columns:
                df["away_win"] = (df["result"] == "A").astype(int)

        self.features_df = df

        # Load optimal features from RFE or use all
        if self.use_rfe_features and RFE_RESULTS.exists():
            try:
                with open(RFE_RESULTS, 'r') as f:
                    rfe_data = json.load(f)
                optimal_features = rfe_data.get("optimal_features", [])
                # Filter to only features that exist in df
                self.feature_columns = [f for f in optimal_features if f in df.columns]
                logger.info(f"Using {len(self.feature_columns)} RFE-optimized features")
            except Exception as e:
                logger.warning(f"Could not load RFE features: {e}")
                self.feature_columns = [c for c in df.columns if c not in EXCLUDE_COLUMNS]
        else:
            self.feature_columns = [c for c in df.columns if c not in EXCLUDE_COLUMNS]
            logger.info(f"Using all {len(self.feature_columns)} features")

        return df

    def evaluate_model(
        self,
        model,
        features: List[str],
    ) -> Dict[str, Any]:
        """Run walk-forward validation with given model."""
        df = self.features_df
        n = len(df)
        fold_size = n // (self.n_folds + 1)

        fold_results = []

        for fold in range(self.n_folds):
            train_end = (fold + 1) * fold_size
            test_start = train_end
            test_end = min(test_start + fold_size, n)

            if test_end <= test_start:
                continue

            train_df = df.iloc[:train_end].copy()
            test_df = df.iloc[test_start:test_end].copy()

            # Prepare data
            X_train = train_df[features].fillna(0)
            y_train = train_df["away_win"].values
            X_test = test_df[features].fillna(0)
            y_test = test_df["away_win"].values

            # Scale
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train
            try:
                model.fit(X_train_scaled, y_train)
            except Exception:
                return {"n_folds": 0, "total_bets": 0, "total_wins": 0,
                        "overall_precision": 0, "avg_roi": 0}

            # Calibrate
            try:
                calibrated = CalibratedClassifierCV(model, cv="prefit", method="sigmoid")
                calibrated.fit(X_train_scaled, y_train)
                probs = calibrated.predict_proba(X_test_scaled)[:, 1]
            except Exception:
                probs = model.predict_proba(X_test_scaled)[:, 1]

            # Apply threshold and odds filter
            mask = probs >= self.threshold

            if "avg_away_open" in test_df.columns:
                odds = test_df["avg_away_open"].values
                mask &= (odds >= self.min_odds) & (odds <= self.max_odds)
            else:
                odds = np.ones(len(test_df)) * 3.0

            bet_indices = np.where(mask)[0]
            if len(bet_indices) == 0:
                fold_results.append({"n_bets": 0, "wins": 0, "precision": 0, "roi": 0})
                continue

            bet_outcomes = y_test[bet_indices]
            bet_odds = odds[bet_indices]

            wins = int(bet_outcomes.sum())
            n_bets = len(bet_indices)
            precision = wins / n_bets if n_bets > 0 else 0

            returns = np.where(bet_outcomes == 1, bet_odds - 1, -1)
            roi = returns.mean() * 100

            fold_results.append({
                "n_bets": n_bets,
                "wins": wins,
                "precision": precision,
                "roi": roi,
            })

        # Aggregate
        valid_folds = [f for f in fold_results if f["n_bets"] > 0]
        if not valid_folds:
            return {
                "n_folds": 0,
                "total_bets": 0,
                "total_wins": 0,
                "overall_precision": 0,
                "avg_roi": 0,
            }

        total_bets = sum(f["n_bets"] for f in valid_folds)
        total_wins = sum(f["wins"] for f in valid_folds)

        return {
            "n_folds": len(valid_folds),
            "total_bets": total_bets,
            "total_wins": total_wins,
            "overall_precision": total_wins / total_bets if total_bets > 0 else 0,
            "avg_roi": np.mean([f["roi"] for f in valid_folds]),
        }

    def create_catboost_objective(self):
        """Create Optuna objective for CatBoost."""
        def objective(trial):
            params = {
                "iterations": trial.suggest_int("iterations", 100, 1000, step=100),
                "depth": trial.suggest_int("depth", 4, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 100, log=True),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
                "random_seed": 42,
                "verbose": False,
            }

            model = CatBoostClassifier(**params)
            result = self.evaluate_model(model, self.feature_columns)

            # Penalize if too few bets
            if result["total_bets"] < 50:
                return 0.0

            return result["overall_precision"]

        return objective

    def create_lightgbm_objective(self):
        """Create Optuna objective for LightGBM."""
        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 20, 150),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "random_state": 42,
                "verbose": -1,
            }

            model = lgb.LGBMClassifier(**params)
            result = self.evaluate_model(model, self.feature_columns)

            if result["total_bets"] < 50:
                return 0.0

            return result["overall_precision"]

        return objective

    def create_xgboost_objective(self):
        """Create Optuna objective for XGBoost."""
        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "random_state": 42,
                "verbosity": 0,
            }

            model = xgb.XGBClassifier(**params)
            result = self.evaluate_model(model, self.feature_columns)

            if result["total_bets"] < 50:
                return 0.0

            return result["overall_precision"]

        return objective

    def tune_model(
        self,
        model_type: str,
        n_trials: int = 50,
    ) -> TuningResult:
        """Run hyperparameter tuning for a specific model."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Tuning {model_type.upper()}")
        logger.info(f"{'='*60}")

        # Create objective
        if model_type == "catboost":
            objective = self.create_catboost_objective()
        elif model_type == "lightgbm":
            objective = self.create_lightgbm_objective()
        elif model_type == "xgboost":
            objective = self.create_xgboost_objective()
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Create study
        study_name = f"{model_type}_precision_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=42),
            study_name=study_name,
        )

        # Run optimization
        study.optimize(
            objective,
            n_trials=n_trials,
            show_progress_bar=True,
        )

        # Get best result
        best_params = study.best_params
        best_precision = study.best_value

        # Evaluate best model to get ROI
        if model_type == "catboost":
            best_model = CatBoostClassifier(**best_params, random_seed=42, verbose=False)
        elif model_type == "lightgbm":
            best_model = lgb.LGBMClassifier(**best_params, random_state=42, verbose=-1)
        else:
            best_model = xgb.XGBClassifier(**best_params, random_state=42, verbosity=0)

        final_result = self.evaluate_model(best_model, self.feature_columns)

        logger.info(f"Best {model_type} precision: {best_precision:.1%}")
        logger.info(f"Best {model_type} ROI: {final_result['avg_roi']:.1f}%")
        logger.info(f"Best params: {best_params}")

        return TuningResult(
            model_type=model_type,
            best_params=best_params,
            best_precision=best_precision,
            best_roi=final_result["avg_roi"],
            n_trials=n_trials,
            study_name=study_name,
        )


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter Tuning")
    parser.add_argument("--model", type=str, default="all",
                        choices=["catboost", "lightgbm", "xgboost", "all"],
                        help="Model to tune")
    parser.add_argument("--n-trials", type=int, default=50,
                        help="Number of Optuna trials per model")
    parser.add_argument("--n-folds", type=int, default=5,
                        help="Number of walk-forward folds")
    parser.add_argument("--threshold", type=float, default=0.60,
                        help="Prediction threshold")
    args = parser.parse_args()

    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║              HYPERPARAMETER TUNING (Phase 4)                                  ║
║                                                                              ║
║  Optimizing model hyperparameters with precision as objective                ║
║  Using Optuna with walk-forward validation                                   ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

    tuner = HyperparameterTuner(
        n_folds=args.n_folds,
        threshold=args.threshold,
        use_rfe_features=True,
    )

    # Load data
    logger.info("Loading data...")
    tuner.load_data()

    # Determine models to tune
    if args.model == "all":
        models = ["catboost", "lightgbm", "xgboost"]
    else:
        models = [args.model]

    # Tune each model
    results = []
    for model_type in models:
        result = tuner.tune_model(model_type, n_trials=args.n_trials)
        results.append(result)

    # Print summary
    print("\n" + "="*80)
    print("HYPERPARAMETER TUNING RESULTS")
    print("="*80)
    print(f"\n{'Model':<12} {'Best Precision':>15} {'Best ROI':>12} {'Trials':>8}")
    print("-" * 50)

    best_result = max(results, key=lambda x: x.best_precision)
    for r in results:
        is_best = r.model_type == best_result.model_type
        marker = " *" if is_best else ""
        print(f"{r.model_type:<12} {r.best_precision:>14.1%} {r.best_roi:>11.1f}% {r.n_trials:>8}{marker}")

    print("\n" + "="*80)
    print("BEST MODEL CONFIGURATION")
    print("="*80)
    print(f"Model: {best_result.model_type}")
    print(f"Precision: {best_result.best_precision:.1%}")
    print(f"ROI: {best_result.best_roi:.1f}%")
    print(f"\nBest Hyperparameters:")
    for param, value in best_result.best_params.items():
        print(f"  {param}: {value}")

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "generated_at": datetime.now().isoformat(),
        "n_folds": args.n_folds,
        "threshold": args.threshold,
        "n_features": len(tuner.feature_columns),
        "results": [asdict(r) for r in results],
        "best_model": best_result.model_type,
        "best_params": best_result.best_params,
    }

    output_path = OUTPUT_DIR / f"tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
