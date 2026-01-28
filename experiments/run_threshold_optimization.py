#!/usr/bin/env python3
"""
Threshold Optimization for Away Win Precision (Phase 5)

This script searches through different probability thresholds and odds filters
to find the optimal configuration for precision using the best model from Phase 4.

Objective: Maximize precision while maintaining sufficient bet volume
Uses: Best LightGBM hyperparameters from Phase 4

Usage:
    python experiments/run_threshold_optimization.py
"""

import argparse
import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
from itertools import product

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb
from tqdm import tqdm

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Paths
FEATURES_FILE = Path("data/03-features/features_all_5leagues_with_odds.parquet")
RFE_RESULTS = Path("experiments/outputs/feature_selection/rfe_20260124_112415.json")
LGBM_RESULTS = Path("experiments/outputs/hyperparameter_tuning/tuning_20260124_144721.json")
OUTPUT_DIR = Path("experiments/outputs/threshold_optimization")

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

# Search space for threshold optimization
THRESHOLD_SEARCH = [0.50, 0.55, 0.58, 0.60, 0.62, 0.65, 0.68, 0.70, 0.72, 0.75]
MIN_ODDS_SEARCH = [1.8, 2.0, 2.2, 2.5, 2.8, 3.0]
MAX_ODDS_SEARCH = [4.0, 5.0, 6.0, 8.0, 10.0]


@dataclass
class ThresholdResult:
    """Result of a threshold configuration."""
    threshold: float
    min_odds: float
    max_odds: float
    precision: float
    roi: float
    n_bets: int
    n_wins: int


class ThresholdOptimizer:
    """
    Grid search over probability thresholds and odds filters.
    """

    def __init__(
        self,
        n_folds: int = 5,
        min_bets: int = 50,
    ):
        self.n_folds = n_folds
        self.min_bets = min_bets
        self.features_df = None
        self.feature_columns = None
        self.best_params = None

    def load_data(self) -> pd.DataFrame:
        """Load and prepare feature data."""
        from src.utils.data_io import load_features
        df = load_features(FEATURES_FILE)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        logger.info(f"Loaded {len(df)} matches")
        return df

    def load_rfe_features(self) -> List[str]:
        """Load RFE-optimized feature list."""
        with open(RFE_RESULTS, "r") as f:
            rfe_data = json.load(f)
        features = rfe_data["optimal_features"]
        logger.info(f"Using {len(features)} RFE-optimized features")
        return features

    def load_best_lgbm_params(self) -> Dict[str, Any]:
        """Load best LightGBM parameters from Phase 4."""
        with open(LGBM_RESULTS, "r") as f:
            data = json.load(f)
        # Handle both single-model and multi-model result formats
        if "results" in data:
            for r in data["results"]:
                if r["model_type"] == "lightgbm":
                    return r["best_params"]
        return data["best_params"]

    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare feature matrix and target."""
        # Get valid feature columns
        available = set(df.columns) - set(EXCLUDE_COLUMNS)

        # Use RFE features
        rfe_features = self.load_rfe_features()
        self.feature_columns = [f for f in rfe_features if f in available]

        X = df[self.feature_columns].values
        y = df["away_win"].values

        # Handle NaN
        X = np.nan_to_num(X, nan=0.0)

        return X, y

    def walk_forward_evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        odds: np.ndarray,
        threshold: float,
        min_odds: float,
        max_odds: float,
    ) -> ThresholdResult:
        """
        Evaluate a threshold configuration using walk-forward validation.
        """
        n_samples = len(y)
        fold_size = n_samples // (self.n_folds + 1)

        all_predictions = []
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

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train with best LightGBM params
            model = lgb.LGBMClassifier(**self.best_params, random_state=42, verbose=-1)

            # Calibrate
            calibrated = CalibratedClassifierCV(model, method="sigmoid", cv=3)
            calibrated.fit(X_train_scaled, y_train)

            # Predict probabilities
            probs = calibrated.predict_proba(X_test_scaled)[:, 1]

            all_predictions.extend(probs)
            all_actuals.extend(y_test)
            all_odds.extend(odds_test)

        # Apply threshold and odds filters
        preds = np.array(all_predictions)
        actuals = np.array(all_actuals)
        odds_arr = np.array(all_odds)

        # Filter by threshold and odds range
        mask = (preds >= threshold) & (odds_arr >= min_odds) & (odds_arr <= max_odds)

        n_bets = mask.sum()
        if n_bets < self.min_bets:
            return ThresholdResult(
                threshold=threshold,
                min_odds=min_odds,
                max_odds=max_odds,
                precision=0.0,
                roi=-100.0,
                n_bets=n_bets,
                n_wins=0,
            )

        wins = actuals[mask].sum()
        precision = wins / n_bets if n_bets > 0 else 0.0

        # Calculate ROI
        returns = np.where(actuals[mask] == 1, odds_arr[mask] - 1, -1)
        roi = returns.mean() * 100 if len(returns) > 0 else -100.0

        return ThresholdResult(
            threshold=threshold,
            min_odds=min_odds,
            max_odds=max_odds,
            precision=precision,
            roi=roi,
            n_bets=int(n_bets),
            n_wins=int(wins),
        )

    def optimize(self) -> List[ThresholdResult]:
        """
        Run grid search over threshold configurations.
        """
        # Load data
        df = self.load_data()
        self.best_params = self.load_best_lgbm_params()
        logger.info(f"Using LightGBM params: {self.best_params}")

        X, y = self.prepare_features(df)

        # Get away odds
        odds_col = "odds_away" if "odds_away" in df.columns else "away_odds"
        if odds_col not in df.columns:
            # Try finding odds column
            for col in df.columns:
                if "away" in col.lower() and "odd" in col.lower():
                    odds_col = col
                    break
        odds = df[odds_col].fillna(3.0).values

        # Grid search
        configurations = list(product(THRESHOLD_SEARCH, MIN_ODDS_SEARCH, MAX_ODDS_SEARCH))
        logger.info(f"Testing {len(configurations)} threshold configurations...")

        results = []
        for threshold, min_odds, max_odds in tqdm(configurations, desc="Searching thresholds"):
            result = self.walk_forward_evaluate(X, y, odds, threshold, min_odds, max_odds)
            results.append(result)

        # Sort by precision (descending), then by ROI
        results.sort(key=lambda r: (r.precision, r.roi), reverse=True)

        return results

    def save_results(self, results: List[ThresholdResult], output_path: Path):
        """Save results to JSON."""
        # Convert numpy types to Python types
        def convert_types(d):
            result = {}
            for k, v in d.items():
                if hasattr(v, 'item'):  # numpy scalar
                    result[k] = v.item()
                else:
                    result[k] = v
            return result

        data = {
            "timestamp": datetime.now().isoformat(),
            "n_configurations": len(results),
            "best_params_lgbm": self.best_params,
            "top_results": [convert_types(asdict(r)) for r in results[:20]],  # Top 20
            "all_results": [convert_types(asdict(r)) for r in results],
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved results to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Threshold Optimization (Phase 5)")
    parser.add_argument("--n-folds", type=int, default=5, help="Number of walk-forward folds")
    parser.add_argument("--min-bets", type=int, default=50, help="Minimum bets threshold")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║              THRESHOLD OPTIMIZATION (Phase 5)                                 ║
║                                                                              ║
║  Searching optimal probability threshold and odds filters                    ║
║  Using best LightGBM model from Phase 4                                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

    optimizer = ThresholdOptimizer(
        n_folds=args.n_folds,
        min_bets=args.min_bets,
    )

    results = optimizer.optimize()

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"threshold_{timestamp}.json"
    optimizer.save_results(results, output_path)

    # Print top results
    print("\n" + "=" * 80)
    print("TOP 10 THRESHOLD CONFIGURATIONS")
    print("=" * 80)
    print(f"{'Threshold':>10} {'Min Odds':>10} {'Max Odds':>10} {'Precision':>12} {'ROI':>10} {'Bets':>8}")
    print("-" * 80)

    for r in results[:10]:
        if r.precision > 0:
            print(f"{r.threshold:>10.2f} {r.min_odds:>10.1f} {r.max_odds:>10.1f} "
                  f"{r.precision*100:>11.1f}% {r.roi:>9.1f}% {r.n_bets:>8}")

    # Best result summary
    best = results[0] if results and results[0].precision > 0 else None
    if best:
        print("\n" + "=" * 80)
        print("BEST CONFIGURATION")
        print("=" * 80)
        print(f"  Probability Threshold: {best.threshold:.2f}")
        print(f"  Min Odds: {best.min_odds}")
        print(f"  Max Odds: {best.max_odds}")
        print(f"  Precision: {best.precision*100:.1f}%")
        print(f"  ROI: {best.roi:.1f}%")
        print(f"  Total Bets: {best.n_bets}")
        print(f"  Wins: {best.n_wins}")


if __name__ == "__main__":
    main()
