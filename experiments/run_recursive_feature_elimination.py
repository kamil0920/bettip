#!/usr/bin/env python3
"""
Recursive Feature Elimination for Away Win Precision

This script iteratively removes features and tracks walk-forward precision
to find the optimal feature subset.

Approach:
1. Start with all features
2. Train model, get feature importance
3. Remove least important feature
4. Repeat, tracking precision at each step
5. Find optimal subset size

Usage:
    python experiments/run_recursive_feature_elimination.py
    python experiments/run_recursive_feature_elimination.py --min-features 10 --n-folds 5
"""

import argparse
import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from catboost import CatBoostClassifier
import lightgbm as lgb

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Paths
FEATURES_FILE = Path("data/03-features/features_all_5leagues_with_odds.csv")
OUTPUT_DIR = Path("experiments/outputs/feature_selection")

# Exclude columns
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
class RFEStep:
    """Result of one RFE step."""
    n_features: int
    features_removed: List[str]
    remaining_features: List[str]
    n_folds: int
    total_bets: int
    total_wins: int
    overall_precision: float
    avg_precision: float
    std_precision: float
    avg_roi: float


class RecursiveFeatureElimination:
    """
    Recursive Feature Elimination for precision optimization.
    """

    def __init__(
        self,
        n_folds: int = 5,
        threshold: float = 0.60,
        min_features: int = 5,
        step_size: int = 1,
        min_odds: float = 2.0,
        max_odds: float = 5.0,
    ):
        self.n_folds = n_folds
        self.threshold = threshold
        self.min_features = min_features
        self.step_size = step_size
        self.min_odds = min_odds
        self.max_odds = max_odds
        self.features_df = None

    def load_data(self) -> pd.DataFrame:
        """Load and prepare feature data."""
        if not FEATURES_FILE.exists():
            raise FileNotFoundError(f"Features file not found: {FEATURES_FILE}")

        df = pd.read_csv(FEATURES_FILE)
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
        return df

    def get_initial_features(self) -> List[str]:
        """Get initial feature set (all non-excluded columns)."""
        return [c for c in self.features_df.columns if c not in EXCLUDE_COLUMNS]

    def train_and_get_importance(
        self,
        train_df: pd.DataFrame,
        features: List[str],
    ) -> Dict[str, float]:
        """Train model and get feature importance."""
        X_train = train_df[features].fillna(0)
        y_train = train_df["away_win"].values

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # Train CatBoost
        model = CatBoostClassifier(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            l2_leaf_reg=10,
            random_seed=42,
            verbose=False,
        )
        model.fit(X_train_scaled, y_train)

        # Get importance
        importance = model.feature_importances_
        importance = importance / importance.sum()  # Normalize

        return {feat: float(imp) for feat, imp in zip(features, importance)}

    def evaluate_features(
        self,
        features: List[str],
    ) -> Dict[str, Any]:
        """Run walk-forward validation with given features."""
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

            # Train
            X_train = train_df[features].fillna(0)
            y_train = train_df["away_win"].values
            X_test = test_df[features].fillna(0)
            y_test = test_df["away_win"].values

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model = CatBoostClassifier(
                iterations=300,
                depth=6,
                learning_rate=0.05,
                l2_leaf_reg=10,
                random_seed=42,
                verbose=False,
            )
            model.fit(X_train_scaled, y_train)

            # Calibrate
            calibrated = CalibratedClassifierCV(model, cv="prefit", method="sigmoid")
            calibrated.fit(X_train_scaled, y_train)

            # Predict
            probs = calibrated.predict_proba(X_test_scaled)[:, 1]

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
                "avg_precision": 0,
                "std_precision": 0,
                "avg_roi": 0,
            }

        total_bets = sum(f["n_bets"] for f in valid_folds)
        total_wins = sum(f["wins"] for f in valid_folds)
        precisions = [f["precision"] for f in valid_folds]
        rois = [f["roi"] for f in valid_folds]

        return {
            "n_folds": len(valid_folds),
            "total_bets": total_bets,
            "total_wins": total_wins,
            "overall_precision": total_wins / total_bets if total_bets > 0 else 0,
            "avg_precision": np.mean(precisions),
            "std_precision": np.std(precisions) if len(precisions) > 1 else 0,
            "avg_roi": np.mean(rois),
        }

    def run_rfe(self) -> List[RFEStep]:
        """Run recursive feature elimination."""
        current_features = self.get_initial_features()
        logger.info(f"Starting with {len(current_features)} features")

        results = []
        removed_features = []

        while len(current_features) >= self.min_features:
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing with {len(current_features)} features")

            # Evaluate current feature set
            eval_result = self.evaluate_features(current_features)

            step = RFEStep(
                n_features=len(current_features),
                features_removed=removed_features.copy(),
                remaining_features=current_features.copy(),
                n_folds=eval_result["n_folds"],
                total_bets=eval_result["total_bets"],
                total_wins=eval_result["total_wins"],
                overall_precision=eval_result["overall_precision"],
                avg_precision=eval_result["avg_precision"],
                std_precision=eval_result["std_precision"],
                avg_roi=eval_result["avg_roi"],
            )
            results.append(step)

            logger.info(
                f"  Precision: {step.overall_precision:.1%} ± {step.std_precision:.1%}, "
                f"ROI: {step.avg_roi:.1f}%, Bets: {step.total_bets}"
            )

            if len(current_features) <= self.min_features:
                break

            # Get feature importance
            # Use first 80% of data for importance calculation
            n = len(self.features_df)
            train_df = self.features_df.iloc[:int(n * 0.8)]
            importance = self.train_and_get_importance(train_df, current_features)

            # Remove least important features
            sorted_features = sorted(importance.items(), key=lambda x: x[1])
            to_remove = [f for f, _ in sorted_features[:self.step_size]]

            logger.info(f"  Removing: {to_remove}")
            removed_features.extend(to_remove)
            current_features = [f for f in current_features if f not in to_remove]

        return results


def main():
    parser = argparse.ArgumentParser(description="Recursive Feature Elimination")
    parser.add_argument("--n-folds", type=int, default=5, help="Number of folds")
    parser.add_argument("--threshold", type=float, default=0.60, help="Prediction threshold")
    parser.add_argument("--min-features", type=int, default=10, help="Minimum features to keep")
    parser.add_argument("--step-size", type=int, default=5, help="Features to remove per step")
    args = parser.parse_args()

    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║              RECURSIVE FEATURE ELIMINATION                                    ║
║                                                                              ║
║  Iteratively removing least important features to find optimal subset       ║
║  Tracking walk-forward precision at each step                               ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

    rfe = RecursiveFeatureElimination(
        n_folds=args.n_folds,
        threshold=args.threshold,
        min_features=args.min_features,
        step_size=args.step_size,
    )

    # Load data
    logger.info("Loading data...")
    rfe.load_data()

    # Run RFE
    results = rfe.run_rfe()

    # Print summary
    print("\n" + "="*80)
    print("RECURSIVE FEATURE ELIMINATION RESULTS")
    print("="*80)
    print(f"\n{'#Features':<12} {'Bets':>8} {'Wins':>8} {'Precision':>12} {'±Std':>8} {'ROI':>10}")
    print("-" * 60)

    best_result = max(results, key=lambda x: x.overall_precision if x.total_bets >= 20 else 0)

    for r in results:
        is_best = r.n_features == best_result.n_features
        marker = " *" if is_best else ""
        print(
            f"{r.n_features:<12} {r.total_bets:>8} {r.total_wins:>8} "
            f"{r.overall_precision:>11.1%} ±{r.std_precision:>6.1%} {r.avg_roi:>9.1f}%{marker}"
        )

    print("\n" + "="*80)
    print("OPTIMAL FEATURE SET")
    print("="*80)
    print(f"Number of features: {best_result.n_features}")
    print(f"Precision: {best_result.overall_precision:.1%}")
    print(f"ROI: {best_result.avg_roi:.1f}%")
    print(f"\nFeatures ({len(best_result.remaining_features)}):")
    for i, f in enumerate(sorted(best_result.remaining_features), 1):
        print(f"  {i:2d}. {f}")

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "generated_at": datetime.now().isoformat(),
        "n_folds": args.n_folds,
        "threshold": args.threshold,
        "min_features": args.min_features,
        "step_size": args.step_size,
        "results": [asdict(r) for r in results],
        "optimal_features": best_result.remaining_features,
    }

    output_path = OUTPUT_DIR / f"rfe_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
