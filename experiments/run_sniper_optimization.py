#!/usr/bin/env python3
"""
Sniper Mode Optimization Pipeline for All Bet Types

This script applies the Phase 4-5 optimization approach (hyperparameter tuning +
threshold optimization) to achieve high-precision configurations for each bet type.

Based on away_win success: 69.1% precision, 107.3% ROI

Supported bet types:
- away_win, home_win, btts, over25, under25
- corners, shots, fouls, cards

Pipeline per bet type:
1. RFE Feature Selection (reduce to optimal subset)
2. Hyperparameter Tuning (Optuna with walk-forward validation)
3. Threshold Optimization (grid search over prob/odds thresholds)
4. Save optimal configuration

Usage:
    # Single bet type
    python experiments/run_sniper_optimization.py --bet-type away_win

    # All bet types
    python experiments/run_sniper_optimization.py --all

    # Multiple specific types
    python experiments/run_sniper_optimization.py --bet-type away_win btts fouls
"""

import argparse
import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from itertools import product

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeClassifierCV
from catboost import CatBoostClassifier
import lightgbm as lgb
import xgboost as xgb
from tqdm import tqdm

# SHAP for feature importance analysis
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not available. Install with: pip install shap")

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Paths
FEATURES_FILE = Path("data/03-features/features_all_5leagues_with_odds.csv")
OUTPUT_DIR = Path("experiments/outputs/sniper_optimization")

# Bet type configurations
BET_TYPES = {
    "away_win": {
        "target": "away_win",
        "odds_col": "odds_away",
        "approach": "classification",
        "default_threshold": 0.60,
        "threshold_search": [0.50, 0.55, 0.58, 0.60, 0.62, 0.65, 0.68, 0.70, 0.72, 0.75, 0.78, 0.80],
    },
    "home_win": {
        "target": "home_win",
        "odds_col": "odds_home",
        "approach": "classification",
        "default_threshold": 0.60,
        "threshold_search": [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80],
    },
    "btts": {
        "target": "btts",
        "odds_col": "btts_yes_odds",
        "approach": "classification",
        "default_threshold": 0.60,
        "threshold_search": [0.55, 0.60, 0.65, 0.70, 0.75, 0.80],
    },
    "over25": {
        "target": "over25",
        "odds_col": "odds_over25",
        "approach": "classification",
        "default_threshold": 0.60,
        "threshold_search": [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85],
    },
    "under25": {
        "target": "under25",
        "odds_col": "odds_under25",
        "approach": "classification",
        "default_threshold": 0.55,
        "threshold_search": [0.50, 0.55, 0.60, 0.65, 0.70, 0.75],
    },
    "fouls": {
        "target": "total_fouls",
        "target_line": 24.5,
        "odds_col": "fouls_over_odds",
        "approach": "regression_line",
        "default_threshold": 0.60,
        "threshold_search": [0.55, 0.60, 0.65, 0.70, 0.75],
    },
    "shots": {
        "target": "total_shots",
        "target_line": 24.5,
        "odds_col": "shots_over_odds",
        "approach": "regression_line",
        "default_threshold": 0.65,
        "threshold_search": [0.55, 0.60, 0.65, 0.70, 0.75, 0.80],
    },
    "corners": {
        "target": "total_corners",
        "target_line": 10.5,
        "odds_col": "corners_over_odds",
        "approach": "regression_line",
        "default_threshold": 0.65,
        "threshold_search": [0.55, 0.60, 0.65, 0.70, 0.75],
    },
    "cards": {
        "target": "total_cards",
        "target_line": 4.5,
        "odds_col": "cards_over_odds",
        "approach": "regression_line",
        "default_threshold": 0.65,
        "threshold_search": [0.55, 0.60, 0.65, 0.70, 0.75],
    },
}

# Exclude columns (data leakage prevention)
EXCLUDE_COLUMNS = [
    # Identifiers
    "fixture_id", "date", "home_team_id", "home_team_name",
    "away_team_id", "away_team_name", "round", "season", "league",
    # Target variables (match outcomes)
    "home_win", "draw", "away_win", "match_result", "result",
    "total_goals", "goal_difference",
    "home_goals", "away_goals", "btts",
    "under25", "over25", "under35", "over35",
    # Match statistics (not available pre-match - these are outcomes!)
    "home_shots", "away_shots", "home_shots_on_target", "away_shots_on_target",
    "home_corners", "away_corners", "total_corners",
    "home_fouls", "away_fouls", "total_fouls",
    "home_yellows", "away_yellows", "home_reds", "away_reds",
    "home_possession", "away_possession",
    "total_cards", "total_shots",
    "home_cards", "away_cards",  # Match outcome cards (not historical)
]

# Patterns that indicate odds/bookmaker data (leaky for predicting match outcomes)
LEAKY_PATTERNS = [
    # Direct odds
    "avg_home", "avg_away", "avg_draw", "avg_over", "avg_under", "avg_ah",
    "b365_", "pinnacle_", "max_home", "max_away", "max_draw", "max_over", "max_under", "max_ah",
    # Implied probabilities
    "odds_home_prob", "odds_away_prob", "odds_draw_prob",
    "odds_over25_prob", "odds_under25_prob",
    # Line movements
    "odds_move_", "odds_steam_", "odds_prob_move",
    "ah_line", "line_movement",
    # Derived odds features (still encode bookmaker information)
    "odds_entropy", "odds_goals_expectation", "odds_home_favorite",
    "odds_overround", "odds_prob_diff", "odds_prob_max",
    "odds_upset_potential", "odds_draw_relative",
]

MIN_ODDS_SEARCH = [1.5, 1.8, 2.0, 2.2, 2.5]
MAX_ODDS_SEARCH = [3.5, 4.0, 5.0, 6.0, 8.0]


@dataclass
class SniperResult:
    """Result of sniper optimization for a bet type."""
    bet_type: str
    target: str
    best_model: str
    best_params: Dict[str, Any]
    n_features: int
    optimal_features: List[str]
    best_threshold: float
    best_min_odds: float
    best_max_odds: float
    precision: float
    roi: float
    n_bets: int
    n_wins: int
    timestamp: str
    walkforward: Dict[str, Any] = None
    shap_analysis: Dict[str, Any] = None


class SniperOptimizer:
    """
    Unified sniper optimization pipeline.
    """

    def __init__(
        self,
        bet_type: str,
        n_folds: int = 5,
        n_rfe_features: int = 100,
        n_optuna_trials: int = 30,
        min_bets: int = 30,
        run_walkforward: bool = False,
        run_shap: bool = False,
    ):
        self.bet_type = bet_type
        self.config = BET_TYPES[bet_type]
        self.n_folds = n_folds
        self.n_rfe_features = n_rfe_features
        self.n_optuna_trials = n_optuna_trials
        self.min_bets = min_bets
        self.run_walkforward = run_walkforward
        self.run_shap = run_shap

        self.features_df = None
        self.feature_columns = None
        self.optimal_features = None
        self.best_params = None
        self.best_model_type = None
        self.all_model_params = {}

    def load_data(self) -> pd.DataFrame:
        """Load and prepare feature data."""
        df = pd.read_csv(FEATURES_FILE)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        # Check if target exists and derive if needed
        target = self.config["target"]
        if target not in df.columns:
            logger.warning(f"Target {target} not found, attempting to derive...")
            if target == "total_cards":
                df["total_cards"] = df.get("home_yellows", 0).fillna(0) + df.get("away_yellows", 0).fillna(0) + \
                                   df.get("home_reds", 0).fillna(0) + df.get("away_reds", 0).fillna(0)
            elif target == "total_shots":
                df["total_shots"] = df.get("home_shots", 0).fillna(0) + df.get("away_shots", 0).fillna(0)
            elif target == "under25":
                if "total_goals" in df.columns:
                    df["under25"] = (df["total_goals"] < 2.5).astype(int)
                else:
                    df["under25"] = ((df.get("home_goals", 0).fillna(0) + df.get("away_goals", 0).fillna(0)) < 2.5).astype(int)
            elif target == "over25":
                if "total_goals" in df.columns:
                    df["over25"] = (df["total_goals"] > 2.5).astype(int)
                else:
                    df["over25"] = ((df.get("home_goals", 0).fillna(0) + df.get("away_goals", 0).fillna(0)) > 2.5).astype(int)
            elif target == "btts":
                home_goals = df["home_goals"] if "home_goals" in df.columns else pd.Series(0, index=df.index)
                away_goals = df["away_goals"] if "away_goals" in df.columns else pd.Series(0, index=df.index)
                df["btts"] = ((home_goals.fillna(0) > 0) & (away_goals.fillna(0) > 0)).astype(int)
            elif target == "total_fouls":
                df["total_fouls"] = df.get("home_fouls", 0).fillna(0) + df.get("away_fouls", 0).fillna(0)
            elif target == "total_corners":
                df["total_corners"] = df.get("home_corners", 0).fillna(0) + df.get("away_corners", 0).fillna(0)

        logger.info(f"Loaded {len(df)} matches for {self.bet_type}")
        return df

    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get valid feature columns excluding leakage."""
        all_cols = set(df.columns)
        exclude = set(EXCLUDE_COLUMNS)

        # Exclude columns matching leaky patterns (bookmaker odds, implied probs)
        for col in all_cols:
            col_lower = col.lower()
            for pattern in LEAKY_PATTERNS:
                if pattern.lower() in col_lower:
                    exclude.add(col)
                    break

        features = [c for c in all_cols - exclude if df[c].dtype in ['float64', 'int64', 'float32', 'int32']]
        logger.info(f"Excluded {len(exclude)} columns, {len(features)} features remain")
        return sorted(features)

    def prepare_target(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare target variable."""
        target_col = self.config["target"]

        if self.config["approach"] == "classification":
            return df[target_col].values
        elif self.config["approach"] == "regression_line":
            line = self.config.get("target_line", 0)
            return (df[target_col] > line).astype(int).values
        else:
            return df[target_col].values

    def run_rfe(self, X: np.ndarray, y: np.ndarray) -> List[int]:
        """Run RFE to select optimal features."""
        logger.info(f"Running RFE to select top {self.n_rfe_features} features...")

        # Use LightGBM as base estimator
        base_model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42,
            verbose=-1,
        )

        n_features = min(self.n_rfe_features, X.shape[1])
        rfe = RFE(estimator=base_model, n_features_to_select=n_features, step=10)
        rfe.fit(X, y)

        selected_indices = np.where(rfe.support_)[0]
        logger.info(f"Selected {len(selected_indices)} features via RFE")
        return selected_indices.tolist()

    def create_objective(self, X: np.ndarray, y: np.ndarray, odds: np.ndarray, model_type: str):
        """Create Optuna objective for a specific model type."""

        def objective(trial):
            if model_type == "lightgbm":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
                    "max_depth": trial.suggest_int("max_depth", 3, 8),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                    "num_leaves": trial.suggest_int("num_leaves", 20, 100),
                    "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
                    "random_state": 42,
                    "verbose": -1,
                }
                ModelClass = lgb.LGBMClassifier
            elif model_type == "catboost":
                params = {
                    "iterations": trial.suggest_int("iterations", 100, 800, step=100),
                    "depth": trial.suggest_int("depth", 4, 8),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                    "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 100, log=True),
                    "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
                    "random_seed": 42,
                    "verbose": False,
                }
                ModelClass = CatBoostClassifier
            else:  # xgboost
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 700, step=100),
                    "max_depth": trial.suggest_int("max_depth", 3, 8),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                    "min_child_weight": trial.suggest_int("min_child_weight", 1, 50),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
                    "random_state": 42,
                    "verbosity": 0,
                }
                ModelClass = xgb.XGBClassifier

            # Walk-forward validation
            n_samples = len(y)
            fold_size = n_samples // (self.n_folds + 1)

            all_preds = []
            all_actuals = []
            all_odds = []

            for fold in range(self.n_folds):
                train_end = (fold + 1) * fold_size
                test_start = train_end
                test_end = test_start + fold_size

                if test_end > n_samples:
                    test_end = n_samples

                X_train, y_train = X[:train_end], y[:train_end]
                X_test, y_test = X[test_start:test_end], y[test_start:test_end]
                odds_test = odds[test_start:test_end]

                if len(X_train) < 100 or len(X_test) < 20:
                    continue

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                model = ModelClass(**params)
                calibrated = CalibratedClassifierCV(model, method="sigmoid", cv=3)

                try:
                    calibrated.fit(X_train_scaled, y_train)
                    probs = calibrated.predict_proba(X_test_scaled)[:, 1]
                except Exception:
                    return 0.0

                all_preds.extend(probs)
                all_actuals.extend(y_test)
                all_odds.extend(odds_test)

            if len(all_preds) == 0:
                return 0.0

            # Calculate precision at default threshold
            preds = np.array(all_preds)
            actuals = np.array(all_actuals)
            odds_arr = np.array(all_odds)

            threshold = self.config["default_threshold"]
            mask = (preds >= threshold) & (odds_arr >= 1.5) & (odds_arr <= 6.0)

            n_bets = mask.sum()
            if n_bets < self.min_bets:
                return 0.0

            precision = actuals[mask].sum() / n_bets
            return precision

        return objective

    def run_hyperparameter_tuning(
        self,
        X: np.ndarray,
        y: np.ndarray,
        odds: np.ndarray
    ) -> Tuple[str, Dict[str, Any], float]:
        """Run Optuna hyperparameter tuning for all model types."""
        logger.info("Running hyperparameter tuning...")

        best_overall = {"precision": 0.0, "model": None, "params": None}
        # Store all models' params for stacking ensemble
        self.all_model_params = {}

        for model_type in ["lightgbm", "catboost", "xgboost"]:
            logger.info(f"  Tuning {model_type}...")

            study = optuna.create_study(
                direction="maximize",
                sampler=TPESampler(seed=42),
            )

            objective = self.create_objective(X, y, odds, model_type)
            study.optimize(objective, n_trials=self.n_optuna_trials, show_progress_bar=True)

            # Store params for each model (for stacking later)
            self.all_model_params[model_type] = study.best_params

            if study.best_value > best_overall["precision"]:
                best_overall = {
                    "precision": study.best_value,
                    "model": model_type,
                    "params": study.best_params,
                }

            logger.info(f"    {model_type}: {study.best_value*100:.1f}% precision")

        logger.info(f"Best model: {best_overall['model']} ({best_overall['precision']*100:.1f}%)")
        return best_overall["model"], best_overall["params"], best_overall["precision"]

    def run_threshold_optimization(
        self,
        X: np.ndarray,
        y: np.ndarray,
        odds: np.ndarray,
    ) -> Tuple[float, float, float, float, float, int, int]:
        """Run grid search over threshold and odds filters, including stacking ensemble."""
        logger.info("Running threshold optimization (including stacking ensemble)...")

        # Generate predictions for ALL models with walk-forward
        n_samples = len(y)
        fold_size = n_samples // (self.n_folds + 1)

        # Collect predictions from all models
        model_preds = {name: [] for name in ["lightgbm", "catboost", "xgboost"]}
        all_actuals = []
        all_odds = []

        # Track validation data for stacking meta-learner training
        val_preds = {name: [] for name in ["lightgbm", "catboost", "xgboost"]}
        val_actuals = []

        for fold in range(self.n_folds):
            train_end = (fold + 1) * fold_size
            test_start = train_end
            test_end = min(test_start + fold_size, n_samples)

            X_train, y_train = X[:train_end], y[:train_end]
            X_test, y_test = X[test_start:test_end], y[test_start:test_end]
            odds_test = odds[test_start:test_end]

            if len(X_train) < 100 or len(X_test) < 20:
                continue

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Use 20% of training data for meta-learner validation
            n_val = int(len(X_train) * 0.2)
            X_val_scaled = X_train_scaled[-n_val:]
            y_val = y_train[-n_val:]

            # Train all three base models and get predictions
            for model_type in ["lightgbm", "catboost", "xgboost"]:
                if model_type not in self.all_model_params:
                    continue

                if model_type == "lightgbm":
                    ModelClass = lgb.LGBMClassifier
                    params = {**self.all_model_params[model_type], "random_state": 42, "verbose": -1}
                elif model_type == "catboost":
                    ModelClass = CatBoostClassifier
                    params = {**self.all_model_params[model_type], "random_seed": 42, "verbose": False}
                else:
                    ModelClass = xgb.XGBClassifier
                    params = {**self.all_model_params[model_type], "random_state": 42, "verbosity": 0}

                model = ModelClass(**params)
                calibrated = CalibratedClassifierCV(model, method="sigmoid", cv=3)
                calibrated.fit(X_train_scaled, y_train)

                probs = calibrated.predict_proba(X_test_scaled)[:, 1]
                model_preds[model_type].extend(probs)

                # Also get validation predictions for stacking
                val_probs = calibrated.predict_proba(X_val_scaled)[:, 1]
                val_preds[model_type].extend(val_probs)

            all_actuals.extend(y_test)
            all_odds.extend(odds_test)
            val_actuals.extend(y_val)

        # Convert to arrays
        actuals = np.array(all_actuals)
        odds_arr = np.array(all_odds)

        # Create ensemble predictions
        base_model_names = [m for m in ["lightgbm", "catboost", "xgboost"] if len(model_preds[m]) > 0]

        if len(base_model_names) >= 2:
            # Stack base model predictions
            test_stack = np.column_stack([np.array(model_preds[m]) for m in base_model_names])
            val_stack = np.column_stack([np.array(val_preds[m]) for m in base_model_names])
            y_val_arr = np.array(val_actuals)

            # Average ensemble (no training needed)
            model_preds["average"] = np.mean(test_stack, axis=1).tolist()

            # Stacking ensemble with Ridge meta-learner
            try:
                meta = RidgeClassifierCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0], cv=3)
                meta.fit(val_stack, y_val_arr)
                stacking_decision = meta.decision_function(test_stack)
                stacking_proba = 1 / (1 + np.exp(-stacking_decision))  # Sigmoid
                model_preds["stacking"] = stacking_proba.tolist()
                logger.info(f"  Stacking trained with weights: {dict(zip(base_model_names, meta.coef_[0]))}")
            except Exception as e:
                logger.warning(f"  Stacking failed: {e}")
                model_preds["stacking"] = model_preds["average"]  # Fallback to average
        else:
            logger.warning("  Not enough models for stacking, using best single model only")

        # Grid search across ALL models including ensembles
        threshold_search = self.config["threshold_search"]
        configurations = list(product(threshold_search, MIN_ODDS_SEARCH, MAX_ODDS_SEARCH))

        # Include all available models (base + ensembles)
        all_models = [m for m in ["lightgbm", "catboost", "xgboost", "stacking", "average"]
                      if m in model_preds and len(model_preds[m]) > 0]

        logger.info(f"  Testing models: {all_models}")

        best_result = {"precision": 0.0, "roi": -100.0, "model": self.best_model_type}

        for model_name in all_models:
            preds = np.array(model_preds[model_name])

            for threshold, min_odds, max_odds in configurations:
                mask = (preds >= threshold) & (odds_arr >= min_odds) & (odds_arr <= max_odds)
                n_bets = mask.sum()

                if n_bets < self.min_bets:
                    continue

                wins = actuals[mask].sum()
                precision = wins / n_bets

                # ROI
                returns = np.where(actuals[mask] == 1, odds_arr[mask] - 1, -1)
                roi = returns.mean() * 100 if len(returns) > 0 else -100.0

                if precision > best_result["precision"] or \
                   (precision == best_result["precision"] and roi > best_result["roi"]):
                    best_result = {
                        "model": model_name,
                        "threshold": threshold,
                        "min_odds": min_odds,
                        "max_odds": max_odds,
                        "precision": precision,
                        "roi": roi,
                        "n_bets": int(n_bets),
                        "n_wins": int(wins),
                    }

        if best_result["precision"] == 0:
            logger.warning("No valid configuration found!")
            return self.best_model_type, self.config["default_threshold"], 2.0, 5.0, 0.0, -100.0, 0, 0

        # Update best model type if stacking/average won
        final_model = best_result.get("model", self.best_model_type)
        if final_model != self.best_model_type:
            logger.info(f"  Ensemble '{final_model}' outperformed individual models!")
            self.best_model_type = final_model

        logger.info(f"Best model: {final_model}, threshold: {best_result['threshold']}, "
                   f"precision: {best_result['precision']*100:.1f}%, "
                   f"ROI: {best_result['roi']:.1f}%")

        return (
            final_model,
            best_result["threshold"],
            best_result["min_odds"],
            best_result["max_odds"],
            best_result["precision"],
            best_result["roi"],
            best_result["n_bets"],
            best_result["n_wins"],
        )

    def run_walkforward_validation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        odds: np.ndarray,
        threshold: float,
        min_odds: float,
        max_odds: float,
    ) -> Dict[str, Any]:
        """Run walk-forward validation to assess out-of-sample performance."""
        logger.info("Running walk-forward validation...")

        n_samples = len(y)
        fold_size = n_samples // (self.n_folds + 1)
        wf_results = []

        for fold in range(self.n_folds):
            train_end = (fold + 1) * fold_size
            test_start = train_end
            test_end = min(test_start + fold_size, n_samples)

            if test_end <= test_start or (test_end - test_start) < 20:
                continue

            X_train, y_train = X[:train_end], y[:train_end]
            X_test, y_test = X[test_start:test_end], y[test_start:test_end]
            odds_test = odds[test_start:test_end]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train all base models
            fold_preds = {}
            for model_type in ["lightgbm", "catboost", "xgboost"]:
                if model_type not in self.all_model_params:
                    continue

                if model_type == "lightgbm":
                    ModelClass = lgb.LGBMClassifier
                    params = {**self.all_model_params[model_type], "random_state": 42, "verbose": -1}
                elif model_type == "catboost":
                    ModelClass = CatBoostClassifier
                    params = {**self.all_model_params[model_type], "random_seed": 42, "verbose": False}
                else:
                    ModelClass = xgb.XGBClassifier
                    params = {**self.all_model_params[model_type], "random_state": 42, "verbosity": 0}

                model = ModelClass(**params)
                calibrated = CalibratedClassifierCV(model, method="sigmoid", cv=3)
                calibrated.fit(X_train_scaled, y_train)
                fold_preds[model_type] = calibrated.predict_proba(X_test_scaled)[:, 1]

            # Create ensemble predictions
            if len(fold_preds) >= 2:
                base_preds = np.column_stack(list(fold_preds.values()))
                fold_preds["stacking"] = np.mean(base_preds, axis=1)  # Simple average for fold
                fold_preds["average"] = np.mean(base_preds, axis=1)

            # Evaluate each model on this fold
            for model_name, proba in fold_preds.items():
                bet_mask = (proba >= threshold) & (odds_test >= min_odds) & (odds_test <= max_odds)
                n_bets = bet_mask.sum()

                if n_bets >= 5:
                    wins = y_test[bet_mask] == 1
                    profit = (wins * (odds_test[bet_mask] - 1) - (~wins) * 1).sum()
                    roi = profit / n_bets * 100
                    precision = wins.mean()

                    wf_results.append({
                        'fold': fold,
                        'model': model_name,
                        'n_bets': int(n_bets),
                        'wins': int(wins.sum()),
                        'precision': float(precision),
                        'roi': float(roi),
                    })

        # Summarize results
        if not wf_results:
            logger.warning("No walk-forward results (insufficient data per fold)")
            return {}

        wf_df = pd.DataFrame(wf_results)

        logger.info("\nWalk-Forward Validation Summary:")
        logger.info("-" * 60)

        summary = {}
        for model_name in wf_df['model'].unique():
            model_wf = wf_df[wf_df['model'] == model_name]
            avg_roi = model_wf['roi'].mean()
            std_roi = model_wf['roi'].std()
            avg_precision = model_wf['precision'].mean()
            total_bets = model_wf['n_bets'].sum()

            summary[model_name] = {
                'avg_roi': float(avg_roi),
                'std_roi': float(std_roi),
                'avg_precision': float(avg_precision),
                'total_bets': int(total_bets),
                'n_folds': len(model_wf),
            }

            logger.info(f"  {model_name:12}: ROI={avg_roi:+6.1f}% (+/-{std_roi:5.1f}%), "
                       f"Precision={avg_precision:.1%}, Bets={total_bets}")

        return {
            'summary': summary,
            'all_folds': wf_df.to_dict('records'),
            'best_model_wf': max(summary.items(), key=lambda x: x[1]['avg_roi'])[0],
        }

    def run_shap_analysis(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
    ) -> Dict[str, Any]:
        """Run SHAP analysis to understand feature importance and interactions."""
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available, skipping feature analysis")
            return {}

        logger.info("Running SHAP feature importance analysis...")

        # Use 80% for training, 20% for SHAP analysis
        n_train = int(len(X) * 0.8)
        X_train, y_train = X[:n_train], y[:n_train]
        X_shap = X[n_train:]

        # Train a LightGBM model (fast and SHAP-compatible)
        if "lightgbm" in self.all_model_params:
            params = {**self.all_model_params["lightgbm"], "random_state": 42, "verbose": -1}
        else:
            params = {"n_estimators": 100, "max_depth": 5, "random_state": 42, "verbose": -1}

        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)

        try:
            # Calculate SHAP values
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_shap)

            # Handle binary classification (get positive class)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            # Calculate mean absolute SHAP value per feature
            mean_abs_shap = np.abs(shap_values).mean(axis=0)

            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': mean_abs_shap
            }).sort_values('importance', ascending=False)

            logger.info("\nTop 15 features by SHAP importance:")
            for i, row in feature_importance.head(15).iterrows():
                logger.info(f"  {row['feature']:40} {row['importance']:.4f}")

            # Identify low-importance features
            threshold = feature_importance['importance'].max() * 0.01
            low_importance = feature_importance[feature_importance['importance'] < threshold]

            # Feature interaction analysis (top pairs)
            interactions = []
            if len(X_shap) >= 50:
                logger.info("\nTop feature interactions:")
                top_features_idx = feature_importance.head(10).index.tolist()

                for i, idx1 in enumerate(top_features_idx[:5]):
                    for idx2 in top_features_idx[i+1:6]:
                        feat1 = feature_names[idx1] if idx1 < len(feature_names) else f"feat_{idx1}"
                        feat2 = feature_names[idx2] if idx2 < len(feature_names) else f"feat_{idx2}"

                        # Calculate interaction strength via correlation of SHAP values
                        if idx1 < shap_values.shape[1] and idx2 < shap_values.shape[1]:
                            corr = np.corrcoef(shap_values[:, idx1], shap_values[:, idx2])[0, 1]
                            if not np.isnan(corr):
                                interactions.append({
                                    'feature1': feat1,
                                    'feature2': feat2,
                                    'interaction_strength': abs(corr),
                                })

                interactions = sorted(interactions, key=lambda x: x['interaction_strength'], reverse=True)[:10]
                for inter in interactions[:5]:
                    logger.info(f"  {inter['feature1']} x {inter['feature2']}: {inter['interaction_strength']:.3f}")

            return {
                'top_features': feature_importance.head(20).to_dict('records'),
                'low_importance_features': low_importance['feature'].tolist(),
                'n_low_importance': len(low_importance),
                'feature_interactions': interactions,
            }

        except Exception as e:
            logger.warning(f"SHAP analysis failed: {e}")
            return {}

    def optimize(self) -> SniperResult:
        """Run full sniper optimization pipeline."""
        logger.info(f"\n{'='*60}")
        logger.info(f"SNIPER OPTIMIZATION: {self.bet_type.upper()}")
        logger.info(f"{'='*60}\n")

        # Load data
        df = self.load_data()
        self.feature_columns = self.get_feature_columns(df)
        logger.info(f"Available features: {len(self.feature_columns)}")

        # Prepare data
        X = df[self.feature_columns].values
        X = np.nan_to_num(X, nan=0.0)
        y = self.prepare_target(df)

        # Get odds
        odds_col = self.config["odds_col"]
        if odds_col in df.columns:
            odds = df[odds_col].fillna(3.0).values
        else:
            # Fallback for missing odds columns
            logger.warning(f"Odds column {odds_col} not found, using default")
            odds = np.full(len(df), 2.5)

        # Remove samples with NaN in target
        valid_mask = ~np.isnan(y)
        if not valid_mask.all():
            logger.warning(f"Removing {(~valid_mask).sum()} samples with NaN target")
            X = X[valid_mask]
            y = y[valid_mask]
            odds = odds[valid_mask]

        # Step 1: RFE Feature Selection
        selected_indices = self.run_rfe(X, y)
        X_selected = X[:, selected_indices]
        self.optimal_features = [self.feature_columns[i] for i in selected_indices]

        # Step 2: Hyperparameter Tuning
        self.best_model_type, self.best_params, base_precision = self.run_hyperparameter_tuning(
            X_selected, y, odds
        )

        # Handle case where no model achieves any precision
        if self.best_params is None:
            logger.warning(f"No model achieved precision above minimum for {self.bet_type}")
            return SniperResult(
                bet_type=self.bet_type,
                target=self.config["target"],
                best_model="none",
                best_params={},
                n_features=len(self.optimal_features),
                optimal_features=self.optimal_features[:50],
                best_threshold=0.0,
                best_min_odds=0.0,
                best_max_odds=0.0,
                precision=0.0,
                roi=-100.0,
                n_bets=0,
                n_wins=0,
                timestamp=datetime.now().isoformat(),
            )

        # Step 3: Threshold Optimization (includes stacking/average comparison)
        final_model, threshold, min_odds, max_odds, precision, roi, n_bets, n_wins = self.run_threshold_optimization(
            X_selected, y, odds
        )

        # Get params for final model (empty dict for stacking/average)
        final_params = self.best_params if final_model not in ["stacking", "average"] else {}
        if final_model in ["stacking", "average"]:
            final_params = {"ensemble_type": final_model, "base_models": list(self.all_model_params.keys())}

        # Step 4: Walk-Forward Validation (optional)
        walkforward_results = {}
        if self.run_walkforward:
            walkforward_results = self.run_walkforward_validation(
                X_selected, y, odds, threshold, min_odds, max_odds
            )

        # Step 5: SHAP Feature Analysis (optional)
        shap_results = {}
        if self.run_shap:
            shap_results = self.run_shap_analysis(
                X_selected, y, self.optimal_features
            )

        # Create result
        result = SniperResult(
            bet_type=self.bet_type,
            target=self.config["target"],
            best_model=final_model,
            best_params=final_params,
            n_features=len(self.optimal_features),
            optimal_features=self.optimal_features[:50],  # Top 50 for storage
            best_threshold=threshold,
            best_min_odds=min_odds,
            best_max_odds=max_odds,
            precision=precision,
            roi=roi,
            n_bets=n_bets,
            n_wins=n_wins,
            timestamp=datetime.now().isoformat(),
            walkforward=walkforward_results,
            shap_analysis=shap_results,
        )

        return result


def print_summary(results: List[SniperResult]):
    """Print comprehensive summary with actionable insights."""
    print("\n" + "=" * 110)
    print("                           SNIPER OPTIMIZATION RESULTS SUMMARY")
    print("=" * 110)

    # Sort by precision descending
    sorted_results = sorted(results, key=lambda x: x.precision, reverse=True)

    # Main results table
    print(f"\n{'Bet Type':<12} {'Model':<10} {'Thresh':>7} {'Odds':>12} "
          f"{'Precision':>10} {'ROI':>10} {'Bets':>6} {'Wins':>6} {'Status':<12}")
    print("-" * 110)

    viable_count = 0
    total_bets = 0
    weighted_roi = 0

    for r in sorted_results:
        odds_range = f"{r.best_min_odds:.1f}-{r.best_max_odds:.1f}"

        # Determine status
        if r.precision >= 0.75 and r.roi > 50:
            status = "EXCELLENT"
        elif r.precision >= 0.65 and r.roi > 0:
            status = "VIABLE"
        elif r.precision >= 0.55 and r.roi > 0:
            status = "MARGINAL"
        elif r.n_bets == 0:
            status = "FAILED"
        else:
            status = "NOT VIABLE"

        if status in ["EXCELLENT", "VIABLE"]:
            viable_count += 1
            total_bets += r.n_bets
            weighted_roi += r.roi * r.n_bets

        print(f"{r.bet_type:<12} {r.best_model:<10} {r.best_threshold:>7.2f} {odds_range:>12} "
              f"{r.precision*100:>9.1f}% {r.roi:>+9.1f}% {r.n_bets:>6} {r.n_wins:>6} {status:<12}")

    print("-" * 110)

    # Portfolio analysis
    print("\n" + "=" * 110)
    print("                              PORTFOLIO ANALYSIS")
    print("=" * 110)

    if total_bets > 0:
        portfolio_roi = weighted_roi / total_bets
        print(f"\nViable strategies: {viable_count}/{len(results)}")
        print(f"Total bets (viable): {total_bets}")
        print(f"Weighted avg ROI: {portfolio_roi:+.1f}%")

        # Estimate monthly performance (assuming ~100 bets/month across all strategies)
        monthly_bets = min(100, total_bets)
        monthly_profit = monthly_bets * (portfolio_roi / 100)
        print(f"Est. monthly profit (100 unit bankroll, 1 unit/bet): {monthly_profit:+.1f} units")
    else:
        print("\nNo viable strategies found!")

    # Walk-forward validation summary (if available)
    wf_available = [r for r in results if r.walkforward and r.walkforward.get('summary')]
    if wf_available:
        print("\n" + "=" * 110)
        print("                         WALK-FORWARD VALIDATION RESULTS")
        print("=" * 110)
        print(f"\n{'Bet Type':<12} {'Best WF Model':<12} {'WF Avg ROI':>12} {'WF Std':>10} {'Overfitting?':<15}")
        print("-" * 70)

        for r in wf_available:
            wf = r.walkforward
            if wf.get('summary'):
                best_wf_model = wf.get('best_model_wf', 'unknown')
                best_summary = wf['summary'].get(best_wf_model, {})
                avg_roi = best_summary.get('avg_roi', 0)
                std_roi = best_summary.get('std_roi', 0)

                # Check for overfitting (if backtest ROI >> walk-forward ROI)
                overfit_ratio = r.roi / avg_roi if avg_roi > 0 else float('inf')
                if overfit_ratio > 2:
                    overfit_status = "HIGH RISK"
                elif overfit_ratio > 1.5:
                    overfit_status = "MODERATE"
                else:
                    overfit_status = "LOW"

                print(f"{r.bet_type:<12} {best_wf_model:<12} {avg_roi:>+11.1f}% {std_roi:>9.1f}% {overfit_status:<15}")

    # SHAP analysis highlights (if available)
    shap_available = [r for r in results if r.shap_analysis and r.shap_analysis.get('top_features')]
    if shap_available:
        print("\n" + "=" * 110)
        print("                            TOP FEATURES BY SHAP IMPORTANCE")
        print("=" * 110)

        # Collect all top features across bet types
        feature_counts = {}
        for r in shap_available:
            for feat in r.shap_analysis.get('top_features', [])[:10]:
                fname = feat.get('feature', '')
                if fname:
                    feature_counts[fname] = feature_counts.get(fname, 0) + 1

        if feature_counts:
            print("\nMost important features across all bet types:")
            for feat, count in sorted(feature_counts.items(), key=lambda x: -x[1])[:15]:
                print(f"  {feat:<45} (appears in {count}/{len(shap_available)} bet types)")

    # Recommendations
    print("\n" + "=" * 110)
    print("                              RECOMMENDATIONS")
    print("=" * 110)

    excellent = [r for r in results if r.precision >= 0.75 and r.roi > 50]
    marginal = [r for r in results if 0.55 <= r.precision < 0.65 and r.roi > 0]
    failed = [r for r in results if r.n_bets == 0 or r.precision < 0.55]

    if excellent:
        print(f"\n DEPLOY NOW: {', '.join(r.bet_type for r in excellent)}")
        print("   These strategies show strong precision and ROI. Ready for paper trading.")

    if marginal:
        print(f"\n NEEDS WORK: {', '.join(r.bet_type for r in marginal)}")
        print("   Consider: more features, different models, or tighter odds filters.")

    if failed:
        print(f"\n NOT VIABLE: {', '.join(r.bet_type for r in failed)}")
        print("   These markets may lack predictability with current features.")

    # Model distribution
    model_counts = {}
    for r in results:
        if r.best_model:
            model_counts[r.best_model] = model_counts.get(r.best_model, 0) + 1

    print(f"\n Model selection: {model_counts}")
    if 'stacking' in model_counts or 'average' in model_counts:
        ensemble_count = model_counts.get('stacking', 0) + model_counts.get('average', 0)
        print(f"   Ensemble methods won {ensemble_count}/{len(results)} bet types")

    print("\n" + "=" * 110)


def save_markdown_summary(results: List[SniperResult], output_path: Path):
    """Save comprehensive markdown summary for documentation and sharing."""
    lines = []
    lines.append("# Sniper Optimization Results\n")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

    # Results table
    lines.append("## Results Summary\n")
    lines.append("| Bet Type | Model | Threshold | Odds Range | Precision | ROI | Bets | Status |")
    lines.append("|----------|-------|-----------|------------|-----------|-----|------|--------|")

    sorted_results = sorted(results, key=lambda x: x.precision, reverse=True)

    for r in sorted_results:
        odds_range = f"{r.best_min_odds:.1f}-{r.best_max_odds:.1f}"
        if r.precision >= 0.75 and r.roi > 50:
            status = "EXCELLENT"
        elif r.precision >= 0.65 and r.roi > 0:
            status = "VIABLE"
        elif r.precision >= 0.55 and r.roi > 0:
            status = "MARGINAL"
        elif r.n_bets == 0:
            status = "FAILED"
        else:
            status = "NOT VIABLE"

        lines.append(f"| {r.bet_type} | {r.best_model} | {r.best_threshold:.2f} | {odds_range} | "
                    f"{r.precision*100:.1f}% | {r.roi:+.1f}% | {r.n_bets} | {status} |")

    # Walk-forward section
    wf_available = [r for r in results if r.walkforward and r.walkforward.get('summary')]
    if wf_available:
        lines.append("\n## Walk-Forward Validation\n")
        lines.append("| Bet Type | Best WF Model | Avg ROI | Std ROI | Overfitting Risk |")
        lines.append("|----------|---------------|---------|---------|------------------|")

        for r in wf_available:
            wf = r.walkforward
            best_wf_model = wf.get('best_model_wf', 'unknown')
            best_summary = wf['summary'].get(best_wf_model, {})
            avg_roi = best_summary.get('avg_roi', 0)
            std_roi = best_summary.get('std_roi', 0)
            overfit_ratio = r.roi / avg_roi if avg_roi > 0 else float('inf')
            overfit_status = "HIGH" if overfit_ratio > 2 else ("MODERATE" if overfit_ratio > 1.5 else "LOW")
            lines.append(f"| {r.bet_type} | {best_wf_model} | {avg_roi:+.1f}% | {std_roi:.1f}% | {overfit_status} |")

    # SHAP section
    shap_available = [r for r in results if r.shap_analysis and r.shap_analysis.get('top_features')]
    if shap_available:
        lines.append("\n## Top Features (SHAP Analysis)\n")

        feature_counts = {}
        for r in shap_available:
            for feat in r.shap_analysis.get('top_features', [])[:10]:
                fname = feat.get('feature', '')
                if fname:
                    feature_counts[fname] = feature_counts.get(fname, 0) + 1

        lines.append("| Feature | Appears In |")
        lines.append("|---------|------------|")
        for feat, count in sorted(feature_counts.items(), key=lambda x: -x[1])[:15]:
            lines.append(f"| {feat} | {count}/{len(shap_available)} bet types |")

    # Recommendations
    lines.append("\n## Recommendations\n")

    excellent = [r for r in results if r.precision >= 0.75 and r.roi > 50]
    viable = [r for r in results if 0.65 <= r.precision < 0.75 and r.roi > 0]
    marginal = [r for r in results if 0.55 <= r.precision < 0.65 and r.roi > 0]
    failed = [r for r in results if r.n_bets == 0 or r.precision < 0.55]

    if excellent:
        lines.append(f"### Deploy Now\n")
        lines.append(f"**{', '.join(r.bet_type for r in excellent)}**\n")
        lines.append("Strong precision and ROI. Ready for paper trading.\n")

    if viable:
        lines.append(f"### Ready for Testing\n")
        lines.append(f"**{', '.join(r.bet_type for r in viable)}**\n")
        lines.append("Good performance. Monitor in paper trading before deployment.\n")

    if marginal:
        lines.append(f"### Needs Improvement\n")
        lines.append(f"**{', '.join(r.bet_type for r in marginal)}**\n")
        lines.append("Consider: more features, different models, or tighter odds filters.\n")

    if failed:
        lines.append(f"### Not Viable\n")
        lines.append(f"**{', '.join(r.bet_type for r in failed)}**\n")
        lines.append("These markets may lack predictability with current features.\n")

    # Best configurations for each viable bet type
    viable_results = [r for r in results if r.precision >= 0.55 and r.roi > 0]
    if viable_results:
        lines.append("\n## Best Configurations\n")
        for r in sorted(viable_results, key=lambda x: x.roi, reverse=True):
            lines.append(f"### {r.bet_type}\n")
            lines.append(f"```yaml")
            lines.append(f"model: {r.best_model}")
            lines.append(f"threshold: {r.best_threshold}")
            lines.append(f"min_odds: {r.best_min_odds}")
            lines.append(f"max_odds: {r.best_max_odds}")
            lines.append(f"expected_precision: {r.precision*100:.1f}%")
            lines.append(f"expected_roi: {r.roi:+.1f}%")
            lines.append(f"```\n")

    # Write to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    logger.info(f"Saved markdown summary to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Sniper Mode Optimization Pipeline")
    parser.add_argument("--bet-type", nargs="+", default=None,
                       help="Bet type(s) to optimize")
    parser.add_argument("--all", action="store_true",
                       help="Run for all bet types")
    parser.add_argument("--n-folds", type=int, default=5,
                       help="Walk-forward folds")
    parser.add_argument("--n-rfe-features", type=int, default=100,
                       help="Target features after RFE")
    parser.add_argument("--n-optuna-trials", type=int, default=30,
                       help="Optuna trials per model")
    parser.add_argument("--min-bets", type=int, default=30,
                       help="Minimum bets for valid configuration")
    parser.add_argument("--walkforward", action="store_true",
                       help="Run walk-forward validation after optimization")
    parser.add_argument("--shap", action="store_true",
                       help="Run SHAP feature importance and interaction analysis")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Determine bet types to run
    if args.all:
        bet_types = list(BET_TYPES.keys())
    elif args.bet_type:
        bet_types = args.bet_type
    else:
        bet_types = ["away_win"]  # Default

    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              SNIPER MODE OPTIMIZATION PIPELINE                                ║
║                                                                              ║
║  High-precision betting configurations via:                                   ║
║  1. RFE Feature Selection                                                    ║
║  2. Optuna Hyperparameter Tuning (incl. Stacking Ensemble)                   ║
║  3. Threshold + Odds Filter Optimization                                     ║
║  4. Walk-Forward Validation: {'ENABLED' if args.walkforward else 'disabled'}                                     ║
║  5. SHAP Feature Analysis: {'ENABLED' if args.shap else 'disabled'}                                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

    results = []

    for bet_type in bet_types:
        if bet_type not in BET_TYPES:
            logger.warning(f"Unknown bet type: {bet_type}, skipping")
            continue

        optimizer = SniperOptimizer(
            bet_type=bet_type,
            n_folds=args.n_folds,
            n_rfe_features=args.n_rfe_features,
            n_optuna_trials=args.n_optuna_trials,
            min_bets=args.min_bets,
            run_walkforward=args.walkforward,
            run_shap=args.shap,
        )

        result = optimizer.optimize()
        results.append(result)

        # Save individual result
        output_path = OUTPUT_DIR / f"sniper_{bet_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_path, "w") as f:
            json.dump(asdict(result), f, indent=2)
        logger.info(f"Saved result to {output_path}")

    # Print summary
    print_summary(results)

    # Save combined results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    combined_path = OUTPUT_DIR / f"sniper_all_{timestamp}.json"
    with open(combined_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    logger.info(f"\nSaved combined results to {combined_path}")

    # Save markdown summary
    markdown_path = OUTPUT_DIR / f"SUMMARY_{timestamp}.md"
    save_markdown_summary(results, markdown_path)


if __name__ == "__main__":
    main()
