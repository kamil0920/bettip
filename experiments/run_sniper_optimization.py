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
from catboost import CatBoostClassifier
import lightgbm as lgb
import xgboost as xgb
from tqdm import tqdm

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
    # Match statistics (not available pre-match)
    "home_shots", "away_shots", "home_shots_on_target", "away_shots_on_target",
    "home_corners", "away_corners", "total_corners",
    "home_fouls", "away_fouls", "total_fouls",
    "home_yellows", "away_yellows", "home_reds", "away_reds",
    "home_possession", "away_possession",
    "total_cards", "total_shots",
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
    ):
        self.bet_type = bet_type
        self.config = BET_TYPES[bet_type]
        self.n_folds = n_folds
        self.n_rfe_features = n_rfe_features
        self.n_optuna_trials = n_optuna_trials
        self.min_bets = min_bets

        self.features_df = None
        self.feature_columns = None
        self.optimal_features = None
        self.best_params = None
        self.best_model_type = None

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
                df["btts"] = ((df.get("home_goals", 0).fillna(0) > 0) & (df.get("away_goals", 0).fillna(0) > 0)).astype(int)
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

        for model_type in ["lightgbm", "catboost", "xgboost"]:
            logger.info(f"  Tuning {model_type}...")

            study = optuna.create_study(
                direction="maximize",
                sampler=TPESampler(seed=42),
            )

            objective = self.create_objective(X, y, odds, model_type)
            study.optimize(objective, n_trials=self.n_optuna_trials, show_progress_bar=True)

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
        """Run grid search over threshold and odds filters."""
        logger.info("Running threshold optimization...")

        # Get model class
        if self.best_model_type == "lightgbm":
            ModelClass = lgb.LGBMClassifier
            params = {**self.best_params, "random_state": 42, "verbose": -1}
        elif self.best_model_type == "catboost":
            ModelClass = CatBoostClassifier
            params = {**self.best_params, "random_seed": 42, "verbose": False}
        else:
            ModelClass = xgb.XGBClassifier
            params = {**self.best_params, "random_state": 42, "verbosity": 0}

        # Generate predictions with walk-forward
        n_samples = len(y)
        fold_size = n_samples // (self.n_folds + 1)

        all_preds = []
        all_actuals = []
        all_odds = []

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

            model = ModelClass(**params)
            calibrated = CalibratedClassifierCV(model, method="sigmoid", cv=3)
            calibrated.fit(X_train_scaled, y_train)
            probs = calibrated.predict_proba(X_test_scaled)[:, 1]

            all_preds.extend(probs)
            all_actuals.extend(y_test)
            all_odds.extend(odds_test)

        preds = np.array(all_preds)
        actuals = np.array(all_actuals)
        odds_arr = np.array(all_odds)

        # Grid search
        threshold_search = self.config["threshold_search"]
        configurations = list(product(threshold_search, MIN_ODDS_SEARCH, MAX_ODDS_SEARCH))

        best_result = {"precision": 0.0, "roi": -100.0}

        for threshold, min_odds, max_odds in tqdm(configurations, desc="Threshold search"):
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
            return self.config["default_threshold"], 2.0, 5.0, 0.0, -100.0, 0, 0

        logger.info(f"Best threshold: {best_result['threshold']}, "
                   f"precision: {best_result['precision']*100:.1f}%, "
                   f"ROI: {best_result['roi']:.1f}%")

        return (
            best_result["threshold"],
            best_result["min_odds"],
            best_result["max_odds"],
            best_result["precision"],
            best_result["roi"],
            best_result["n_bets"],
            best_result["n_wins"],
        )

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

        # Step 1: RFE Feature Selection
        selected_indices = self.run_rfe(X, y)
        X_selected = X[:, selected_indices]
        self.optimal_features = [self.feature_columns[i] for i in selected_indices]

        # Step 2: Hyperparameter Tuning
        self.best_model_type, self.best_params, base_precision = self.run_hyperparameter_tuning(
            X_selected, y, odds
        )

        # Step 3: Threshold Optimization
        threshold, min_odds, max_odds, precision, roi, n_bets, n_wins = self.run_threshold_optimization(
            X_selected, y, odds
        )

        # Create result
        result = SniperResult(
            bet_type=self.bet_type,
            target=self.config["target"],
            best_model=self.best_model_type,
            best_params=self.best_params,
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
        )

        return result


def print_summary(results: List[SniperResult]):
    """Print summary table of all results."""
    print("\n" + "=" * 100)
    print("SNIPER OPTIMIZATION RESULTS SUMMARY")
    print("=" * 100)
    print(f"{'Bet Type':<12} {'Model':<10} {'Features':>8} {'Threshold':>10} "
          f"{'Precision':>10} {'ROI':>10} {'Bets':>8} {'Wins':>8}")
    print("-" * 100)

    for r in sorted(results, key=lambda x: x.precision, reverse=True):
        print(f"{r.bet_type:<12} {r.best_model:<10} {r.n_features:>8} {r.best_threshold:>10.2f} "
              f"{r.precision*100:>9.1f}% {r.roi:>9.1f}% {r.n_bets:>8} {r.n_wins:>8}")

    print("=" * 100)


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
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Determine bet types to run
    if args.all:
        bet_types = list(BET_TYPES.keys())
    elif args.bet_type:
        bet_types = args.bet_type
    else:
        bet_types = ["away_win"]  # Default

    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║              SNIPER MODE OPTIMIZATION PIPELINE                                ║
║                                                                              ║
║  High-precision betting configurations via:                                   ║
║  1. RFE Feature Selection                                                    ║
║  2. Optuna Hyperparameter Tuning                                             ║
║  3. Threshold + Odds Filter Optimization                                     ║
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
    combined_path = OUTPUT_DIR / f"sniper_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(combined_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    logger.info(f"\nSaved combined results to {combined_path}")


if __name__ == "__main__":
    main()
