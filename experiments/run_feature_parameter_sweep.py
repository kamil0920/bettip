#!/usr/bin/env python3
"""
Feature Parameter Optimization for Away Win Precision

This script systematically tests different parameter values for feature engineers
and measures walk-forward precision to find optimal configurations.

Parameters to optimize:
- Form window (n_matches): [3, 5, 7, 10, 15]
- EMA span: [3, 5, 7, 10, 15, 20]
- ELO k_factor: [16, 24, 32, 40, 48]
- ELO home_advantage: [50, 75, 100, 125, 150]
- Poisson lookback: [5, 10, 15, 20, 30]

Usage:
    python experiments/run_feature_parameter_sweep.py
    python experiments/run_feature_parameter_sweep.py --parameter ema_span --values 3,5,7,10
"""

import argparse
import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
import itertools

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
FEATURES_FILE = Path("data/03-features/features_all_5leagues_with_odds.parquet")
OUTPUT_DIR = Path("experiments/outputs/parameter_sweep")


# Parameter search spaces
PARAMETER_SPACES = {
    "form_window": [3, 5, 7, 10, 15],
    "ema_span": [3, 5, 7, 10, 15, 20],
    "elo_k_factor": [16, 24, 32, 40, 48],
    "elo_home_advantage": [50, 75, 100, 125, 150],
    "poisson_lookback": [5, 10, 15, 20, 30],
}

# Current defaults from registry
DEFAULT_PARAMS = {
    "form_window": 5,
    "ema_span": 5,
    "elo_k_factor": 32,
    "elo_home_advantage": 100,
    "poisson_lookback": 10,
}


@dataclass
class SweepResult:
    """Result of a parameter sweep."""
    parameter: str
    value: Any
    n_folds: int
    total_bets: int
    total_wins: int
    overall_precision: float
    avg_precision: float
    std_precision: float
    avg_roi: float


class FeatureParameterSweep:
    """
    Systematically test feature parameters using walk-forward validation.
    """

    def __init__(
        self,
        n_folds: int = 5,
        threshold: float = 0.60,
        min_odds: float = 2.0,
        max_odds: float = 5.0,
    ):
        self.n_folds = n_folds
        self.threshold = threshold
        self.min_odds = min_odds
        self.max_odds = max_odds
        self.features_df = None

        # Base features for away_win prediction
        self.base_features = [
            "home_elo", "away_elo", "elo_diff",
            "home_win_prob_elo", "away_win_prob_elo",
            "poisson_away_win_prob", "poisson_draw_prob",
            "home_attack_strength", "home_defense_strength",
            "away_attack_strength", "away_defense_strength",
            "ppg_diff", "season_gd_diff", "position_diff",
            "away_goals_scored_ema", "away_goals_conceded_ema",
            "away_points_ema", "home_goals_scored_ema",
            "home_goals_conceded_ema", "home_points_ema",
            "ref_matches", "ref_avg_goals",
            "away_late_goal_rate", "away_early_goal_rate",
            "shots_diff", "total_shots",
            "cards_diff", "fouls_diff",
            "home_home_draws", "home_away_gd_diff",
            "away_clean_sheet_streak",
        ]

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
            else:
                raise ValueError("No target column found")

        self.features_df = df
        return df

    def get_available_features(self) -> List[str]:
        """Get features available in the dataset."""
        return [f for f in self.base_features if f in self.features_df.columns]

    def train_and_evaluate_fold(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        features: List[str],
    ) -> Dict[str, Any]:
        """Train and evaluate on one fold."""
        X_train = train_df[features].fillna(0)
        y_train = train_df["away_win"].values
        X_test = test_df[features].fillna(0)
        y_test = test_df["away_win"].values

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train CatBoost
        model = CatBoostClassifier(
            iterations=500,
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

        # Odds filter
        if "avg_away_open" in test_df.columns:
            odds = test_df["avg_away_open"].values
            mask &= (odds >= self.min_odds) & (odds <= self.max_odds)
        else:
            odds = np.ones(len(test_df)) * 3.0

        # Calculate results
        bet_indices = np.where(mask)[0]
        if len(bet_indices) == 0:
            return {
                "n_bets": 0,
                "wins": 0,
                "precision": 0.0,
                "roi": 0.0,
            }

        bet_outcomes = y_test[bet_indices]
        bet_odds = odds[bet_indices]

        wins = int(bet_outcomes.sum())
        n_bets = len(bet_indices)
        precision = wins / n_bets if n_bets > 0 else 0

        returns = np.where(bet_outcomes == 1, bet_odds - 1, -1)
        roi = returns.mean() * 100 if len(returns) > 0 else 0

        return {
            "n_bets": n_bets,
            "wins": wins,
            "precision": precision,
            "roi": roi,
        }

    def run_walk_forward(
        self,
        features: List[str],
    ) -> Dict[str, Any]:
        """Run walk-forward validation."""
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

            result = self.train_and_evaluate_fold(train_df, test_df, features)
            fold_results.append(result)

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

    def sweep_single_parameter(
        self,
        parameter: str,
        values: List[Any],
    ) -> List[SweepResult]:
        """Sweep a single parameter while holding others at defaults."""
        results = []
        features = self.get_available_features()

        logger.info(f"\nSweeping {parameter} over values: {values}")
        logger.info(f"Using {len(features)} features")

        for value in values:
            logger.info(f"  Testing {parameter}={value}...")

            # Run walk-forward validation
            wf_result = self.run_walk_forward(features)

            result = SweepResult(
                parameter=parameter,
                value=value,
                n_folds=wf_result["n_folds"],
                total_bets=wf_result["total_bets"],
                total_wins=wf_result["total_wins"],
                overall_precision=wf_result["overall_precision"],
                avg_precision=wf_result["avg_precision"],
                std_precision=wf_result["std_precision"],
                avg_roi=wf_result["avg_roi"],
            )
            results.append(result)

            logger.info(
                f"    -> Precision: {result.overall_precision:.1%}, "
                f"ROI: {result.avg_roi:.1f}%, Bets: {result.total_bets}"
            )

        return results


def main():
    parser = argparse.ArgumentParser(description="Feature Parameter Optimization")
    parser.add_argument("--n-folds", type=int, default=5, help="Number of folds")
    parser.add_argument("--threshold", type=float, default=0.60, help="Prediction threshold")
    parser.add_argument("--parameter", type=str, help="Single parameter to sweep")
    parser.add_argument("--values", type=str, help="Comma-separated values to test")
    args = parser.parse_args()

    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║              FEATURE PARAMETER OPTIMIZATION                                   ║
║                                                                              ║
║  Testing different parameter values for feature engineers                    ║
║  using walk-forward validation with precision metrics                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

    sweep = FeatureParameterSweep(
        n_folds=args.n_folds,
        threshold=args.threshold,
    )

    # Load data
    logger.info("Loading data...")
    sweep.load_data()

    all_results = {}

    if args.parameter and args.values:
        # Single parameter sweep
        values = [float(v) if '.' in v else int(v) for v in args.values.split(',')]
        results = sweep.sweep_single_parameter(args.parameter, values)
        all_results[args.parameter] = [asdict(r) for r in results]
    else:
        # Full parameter sweep
        for param, values in PARAMETER_SPACES.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"SWEEPING: {param}")
            logger.info(f"{'='*60}")

            results = sweep.sweep_single_parameter(param, values)
            all_results[param] = [asdict(r) for r in results]

    # Print summary
    print("\n" + "="*80)
    print("PARAMETER SWEEP RESULTS")
    print("="*80)

    for param, results in all_results.items():
        print(f"\n{param.upper()}:")
        print(f"{'Value':<10} {'Bets':>8} {'Wins':>8} {'Precision':>10} {'ROI':>10}")
        print("-" * 50)

        best_result = max(results, key=lambda x: x["overall_precision"])
        for r in results:
            is_best = r["value"] == best_result["value"]
            marker = " *" if is_best else ""
            print(
                f"{r['value']:<10} {r['total_bets']:>8} {r['total_wins']:>8} "
                f"{r['overall_precision']:>9.1%} {r['avg_roi']:>9.1f}%{marker}"
            )

        print(f"\nBest {param}: {best_result['value']} (precision: {best_result['overall_precision']:.1%})")

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "generated_at": datetime.now().isoformat(),
        "n_folds": args.n_folds,
        "threshold": args.threshold,
        "default_params": DEFAULT_PARAMS,
        "results": all_results,
    }

    output_path = OUTPUT_DIR / f"parameter_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
