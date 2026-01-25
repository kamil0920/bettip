#!/usr/bin/env python3
"""
Feature Parameter Optimization Pipeline

This script optimizes feature engineering parameters (ELO k-factor, form windows,
decay rates, etc.) independently per bet type using Optuna and walk-forward validation.

Different markets benefit from different parameters:
- away_win may benefit from higher ELO k-factor (more reactive ratings)
- fouls may need longer form windows (more stable estimates)
- BTTS may benefit from different poisson lookback periods

Pipeline:
1. For each bet type, Optuna suggests feature params
2. Features are regenerated with those params
3. Model is trained and evaluated via walk-forward validation
4. Best params are saved to config/feature_params/{bet_type}.yaml

Usage:
    # Single bet type (quick test)
    python experiments/run_feature_param_optimization.py --bet-type away_win --n-trials 10

    # Full optimization
    python experiments/run_feature_param_optimization.py --bet-type away_win --n-trials 30

    # All bet types
    python experiments/run_feature_param_optimization.py --all --n-trials 30
"""

import argparse
import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb
from tqdm import tqdm

from src.features.config_manager import (
    BetTypeFeatureConfig,
    PARAMETER_SEARCH_SPACES,
    BET_TYPE_PARAM_PRIORITIES,
    get_search_space_for_bet_type,
)
from src.features.regeneration import FeatureRegenerator

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Paths
FEATURES_FILE = Path("data/03-features/features_all_5leagues_with_odds.csv")
OUTPUT_DIR = Path("experiments/outputs/feature_param_optimization")

# Bet type configurations (same as sniper optimization)
BET_TYPES = {
    "away_win": {
        "target": "away_win",
        "odds_col": "odds_away",
        "approach": "classification",
        "default_threshold": 0.60,
    },
    "home_win": {
        "target": "home_win",
        "odds_col": "odds_home",
        "approach": "classification",
        "default_threshold": 0.60,
    },
    "btts": {
        "target": "btts",
        "odds_col": "btts_yes_odds",
        "approach": "classification",
        "default_threshold": 0.60,
    },
    "over25": {
        "target": "over25",
        "odds_col": "odds_over25",
        "approach": "classification",
        "default_threshold": 0.60,
    },
    "under25": {
        "target": "under25",
        "odds_col": "odds_under25",
        "approach": "classification",
        "default_threshold": 0.55,
    },
    "fouls": {
        "target": "total_fouls",
        "target_line": 24.5,
        "odds_col": "fouls_over_odds",
        "approach": "regression_line",
        "default_threshold": 0.60,
    },
    "shots": {
        "target": "total_shots",
        "target_line": 24.5,
        "odds_col": "shots_over_odds",
        "approach": "regression_line",
        "default_threshold": 0.65,
    },
    "corners": {
        "target": "total_corners",
        "target_line": 10.5,
        "odds_col": "corners_over_odds",
        "approach": "regression_line",
        "default_threshold": 0.65,
    },
    "cards": {
        "target": "total_cards",
        "target_line": 4.5,
        "odds_col": "cards_over_odds",
        "approach": "regression_line",
        "default_threshold": 0.65,
    },
}

# Exclude columns (data leakage prevention)
EXCLUDE_COLUMNS = [
    # Identifiers
    "fixture_id", "date", "home_team_id", "home_team_name",
    "away_team_id", "away_team_name", "round", "season", "league",
    # Target variables
    "home_win", "draw", "away_win", "match_result", "result",
    "total_goals", "goal_difference",
    "home_goals", "away_goals", "btts",
    "under25", "over25", "under35", "over35",
    # Match statistics (outcomes)
    "home_shots", "away_shots", "home_shots_on_target", "away_shots_on_target",
    "home_corners", "away_corners", "total_corners",
    "home_fouls", "away_fouls", "total_fouls",
    "home_yellows", "away_yellows", "home_reds", "away_reds",
    "home_possession", "away_possession",
    "total_cards", "total_shots",
    "home_cards", "away_cards",
]

# Leaky patterns
LEAKY_PATTERNS = [
    "avg_home", "avg_away", "avg_draw", "avg_over", "avg_under", "avg_ah",
    "b365_", "pinnacle_", "max_home", "max_away", "max_draw", "max_over", "max_under", "max_ah",
    "odds_home_prob", "odds_away_prob", "odds_draw_prob",
    "odds_over25_prob", "odds_under25_prob",
    "odds_move_", "odds_steam_", "odds_prob_move",
    "ah_line", "line_movement",
    "odds_entropy", "odds_goals_expectation", "odds_home_favorite",
    "odds_overround", "odds_prob_diff", "odds_prob_max",
    "odds_upset_potential", "odds_draw_relative",
]


@dataclass
class FeatureOptimizationResult:
    """Result of feature parameter optimization."""
    bet_type: str
    best_params: Dict[str, Any]
    precision: float
    roi: float
    n_bets: int
    n_trials: int
    n_folds: int
    search_space: Dict[str, List]
    all_trials: List[Dict[str, Any]]
    timestamp: str


class FeatureParamOptimizer:
    """
    Optimizes feature engineering parameters for a specific bet type.

    Uses Optuna to search over parameter space and walk-forward validation
    to evaluate each configuration.
    """

    def __init__(
        self,
        bet_type: str,
        n_trials: int = 30,
        n_folds: int = 5,
        min_bets: int = 30,
        use_regeneration: bool = False,
    ):
        """
        Initialize the optimizer.

        Args:
            bet_type: Bet type to optimize
            n_trials: Number of Optuna trials
            n_folds: Walk-forward folds
            min_bets: Minimum bets required for valid result
            use_regeneration: If True, actually regenerate features (slower but more accurate).
                             If False, use existing features (faster for testing).
        """
        self.bet_type = bet_type
        self.config = BET_TYPES[bet_type]
        self.n_trials = n_trials
        self.n_folds = n_folds
        self.min_bets = min_bets
        self.use_regeneration = use_regeneration

        self.search_space = get_search_space_for_bet_type(bet_type)
        self.regenerator = FeatureRegenerator() if use_regeneration else None

        self.features_df = None
        self.feature_columns = None

    def load_base_features(self) -> pd.DataFrame:
        """Load existing features file (for non-regeneration mode)."""
        df = pd.read_csv(FEATURES_FILE)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        # Derive target if needed
        target = self.config["target"]
        if target not in df.columns:
            self._derive_target(df, target)

        return df

    def _derive_target(self, df: pd.DataFrame, target: str) -> None:
        """Derive target column if not present."""
        if target == "total_cards":
            df["total_cards"] = (
                df.get("home_yellows", 0).fillna(0) +
                df.get("away_yellows", 0).fillna(0) +
                df.get("home_reds", 0).fillna(0) +
                df.get("away_reds", 0).fillna(0)
            )
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

    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get valid feature columns excluding leakage."""
        all_cols = set(df.columns)
        exclude = set(EXCLUDE_COLUMNS)

        for col in all_cols:
            col_lower = col.lower()
            for pattern in LEAKY_PATTERNS:
                if pattern.lower() in col_lower:
                    exclude.add(col)
                    break

        features = [c for c in all_cols - exclude if df[c].dtype in ['float64', 'int64', 'float32', 'int32']]
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

    def create_feature_config_from_trial(self, trial: optuna.Trial) -> BetTypeFeatureConfig:
        """Create BetTypeFeatureConfig from Optuna trial suggestions."""
        config_kwargs = {'bet_type': self.bet_type}

        for param_name, values in self.search_space.items():
            suggested = trial.suggest_categorical(param_name, values)
            config_kwargs[param_name] = suggested

        return BetTypeFeatureConfig(**config_kwargs)

    def evaluate_config(
        self,
        feature_config: BetTypeFeatureConfig,
        features_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[float, float, int]:
        """
        Evaluate a feature configuration using walk-forward validation.

        Args:
            feature_config: Configuration to evaluate
            features_df: Pre-loaded features (for non-regeneration mode)

        Returns:
            Tuple of (precision, roi, n_bets)
        """
        if self.use_regeneration:
            # Regenerate features with custom params
            df = self.regenerator.regenerate_with_params(feature_config)
            # Derive target if needed
            target = self.config["target"]
            if target not in df.columns:
                self._derive_target(df, target)
        else:
            df = features_df

        feature_columns = self.get_feature_columns(df)
        X = df[feature_columns].values
        X = np.nan_to_num(X, nan=0.0)
        y = self.prepare_target(df)

        # Get odds
        odds_col = self.config["odds_col"]
        if odds_col in df.columns:
            odds = df[odds_col].fillna(3.0).values
        else:
            odds = np.full(len(df), 2.5)

        # Remove NaN targets
        valid_mask = ~np.isnan(y)
        if not valid_mask.all():
            X = X[valid_mask]
            y = y[valid_mask]
            odds = odds[valid_mask]

        # Walk-forward validation
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

            # Use LightGBM with fixed params (we're optimizing feature params, not model params)
            model = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                random_state=42,
                verbose=-1,
            )
            calibrated = CalibratedClassifierCV(model, method="sigmoid", cv=3)

            try:
                calibrated.fit(X_train_scaled, y_train)
                probs = calibrated.predict_proba(X_test_scaled)[:, 1]
            except Exception:
                continue

            all_preds.extend(probs)
            all_actuals.extend(y_test)
            all_odds.extend(odds_test)

        if len(all_preds) == 0:
            return 0.0, -100.0, 0

        preds = np.array(all_preds)
        actuals = np.array(all_actuals)
        odds_arr = np.array(all_odds)

        # Evaluate at default threshold
        threshold = self.config["default_threshold"]
        mask = (preds >= threshold) & (odds_arr >= 1.5) & (odds_arr <= 6.0)

        n_bets = mask.sum()
        if n_bets < self.min_bets:
            return 0.0, -100.0, n_bets

        wins = actuals[mask].sum()
        precision = wins / n_bets

        # ROI calculation
        returns = np.where(actuals[mask] == 1, odds_arr[mask] - 1, -1)
        roi = returns.mean() * 100 if len(returns) > 0 else -100.0

        return precision, roi, n_bets

    def create_objective(self, features_df: pd.DataFrame):
        """Create Optuna objective function."""

        def objective(trial):
            feature_config = self.create_feature_config_from_trial(trial)
            precision, roi, n_bets = self.evaluate_config(feature_config, features_df)

            # Store additional metrics in trial
            trial.set_user_attr("roi", roi)
            trial.set_user_attr("n_bets", n_bets)
            trial.set_user_attr("params_hash", feature_config.params_hash())

            return precision

        return objective

    def optimize(self) -> FeatureOptimizationResult:
        """Run feature parameter optimization."""
        logger.info(f"\n{'='*60}")
        logger.info(f"FEATURE PARAMETER OPTIMIZATION: {self.bet_type.upper()}")
        logger.info(f"{'='*60}\n")

        logger.info(f"Search space: {self.search_space}")
        logger.info(f"Trials: {self.n_trials}, Folds: {self.n_folds}")

        # Load features
        if not self.use_regeneration:
            logger.info("Loading base features (fast mode)...")
            features_df = self.load_base_features()
            logger.info(f"Loaded {len(features_df)} matches")
        else:
            logger.info("Using feature regeneration (accurate mode)...")
            features_df = None

        # Run Optuna
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=42),
        )

        objective = self.create_objective(features_df)
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)

        # Get best result
        best_trial = study.best_trial
        best_precision = best_trial.value
        best_roi = best_trial.user_attrs.get("roi", 0.0)
        best_n_bets = best_trial.user_attrs.get("n_bets", 0)

        logger.info(f"\nBest trial:")
        logger.info(f"  Precision: {best_precision*100:.1f}%")
        logger.info(f"  ROI: {best_roi:+.1f}%")
        logger.info(f"  Bets: {best_n_bets}")
        logger.info(f"  Params: {best_trial.params}")

        # Collect all trials
        all_trials = []
        for trial in study.trials:
            all_trials.append({
                "number": trial.number,
                "params": trial.params,
                "precision": trial.value if trial.value is not None else 0.0,
                "roi": trial.user_attrs.get("roi", 0.0),
                "n_bets": trial.user_attrs.get("n_bets", 0),
            })

        # Create result
        result = FeatureOptimizationResult(
            bet_type=self.bet_type,
            best_params=best_trial.params,
            precision=best_precision,
            roi=best_roi,
            n_bets=best_n_bets,
            n_trials=self.n_trials,
            n_folds=self.n_folds,
            search_space=self.search_space,
            all_trials=all_trials,
            timestamp=datetime.now().isoformat(),
        )

        return result

    def save_optimal_config(self, result: FeatureOptimizationResult) -> Path:
        """Save optimal configuration to YAML file."""
        # Create BetTypeFeatureConfig with optimal params
        config = BetTypeFeatureConfig(bet_type=self.bet_type, **result.best_params)
        config.update_metadata(
            precision=result.precision,
            roi=result.roi,
            n_trials=result.n_trials,
        )

        # Save to config directory
        output_path = config.save()
        logger.info(f"Saved optimal config to: {output_path}")
        return output_path


def print_summary(results: List[FeatureOptimizationResult]):
    """Print optimization summary."""
    print("\n" + "=" * 90)
    print("                   FEATURE PARAMETER OPTIMIZATION RESULTS")
    print("=" * 90)

    print(f"\n{'Bet Type':<12} {'Precision':>10} {'ROI':>10} {'Bets':>8} {'Best Parameters':<40}")
    print("-" * 90)

    for r in sorted(results, key=lambda x: x.precision, reverse=True):
        params_str = ", ".join(f"{k}={v}" for k, v in r.best_params.items())
        if len(params_str) > 38:
            params_str = params_str[:35] + "..."
        print(f"{r.bet_type:<12} {r.precision*100:>9.1f}% {r.roi:>+9.1f}% {r.n_bets:>8} {params_str:<40}")

    print("-" * 90)

    # Comparison with defaults
    print("\n" + "=" * 90)
    print("                   PARAMETER VALUE DISTRIBUTION")
    print("=" * 90)

    param_values = {}
    for r in results:
        for param, value in r.best_params.items():
            if param not in param_values:
                param_values[param] = []
            param_values[param].append((r.bet_type, value))

    for param, values in sorted(param_values.items()):
        print(f"\n{param}:")
        for bet_type, value in values:
            print(f"  {bet_type:<12}: {value}")


def main():
    parser = argparse.ArgumentParser(description="Feature Parameter Optimization")
    parser.add_argument("--bet-type", nargs="+", default=None,
                       help="Bet type(s) to optimize")
    parser.add_argument("--all", action="store_true",
                       help="Optimize all bet types")
    parser.add_argument("--n-trials", type=int, default=30,
                       help="Optuna trials per bet type")
    parser.add_argument("--n-folds", type=int, default=5,
                       help="Walk-forward folds")
    parser.add_argument("--min-bets", type=int, default=30,
                       help="Minimum bets for valid configuration")
    parser.add_argument("--regenerate", action="store_true",
                       help="Actually regenerate features (slower but more accurate)")
    parser.add_argument("--save-config", action="store_true",
                       help="Save optimal configs to config/feature_params/")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Determine bet types
    if args.all:
        bet_types = list(BET_TYPES.keys())
    elif args.bet_type:
        bet_types = args.bet_type
    else:
        bet_types = ["away_win"]

    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              FEATURE PARAMETER OPTIMIZATION PIPELINE                          ║
║                                                                              ║
║  Optimizing feature engineering parameters per bet type:                      ║
║  - ELO k-factor, home advantage                                              ║
║  - Form window, EMA span                                                     ║
║  - Poisson lookback, market-specific spans                                   ║
║                                                                              ║
║  Mode: {'REGENERATION (accurate)' if args.regenerate else 'FAST (using existing features)':^30}                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

    results = []

    for bet_type in bet_types:
        if bet_type not in BET_TYPES:
            logger.warning(f"Unknown bet type: {bet_type}, skipping")
            continue

        optimizer = FeatureParamOptimizer(
            bet_type=bet_type,
            n_trials=args.n_trials,
            n_folds=args.n_folds,
            min_bets=args.min_bets,
            use_regeneration=args.regenerate,
        )

        result = optimizer.optimize()
        results.append(result)

        # Save individual result
        output_path = OUTPUT_DIR / f"feature_params_{bet_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_path, "w") as f:
            json.dump(asdict(result), f, indent=2, cls=NumpyEncoder)
        logger.info(f"Saved result to {output_path}")

        # Save optimal config
        if args.save_config:
            optimizer.save_optimal_config(result)

    # Print summary
    print_summary(results)

    # Save combined results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    combined_path = OUTPUT_DIR / f"feature_params_all_{timestamp}.json"
    with open(combined_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2, cls=NumpyEncoder)
    logger.info(f"\nSaved combined results to {combined_path}")

    if args.save_config:
        print("\n" + "=" * 90)
        print("OPTIMAL CONFIGS SAVED")
        print("=" * 90)
        print("\nConfigs saved to config/feature_params/")
        print("Use with sniper optimization:")
        print("  python experiments/run_sniper_optimization.py --bet-type away_win --feature-params config/feature_params/away_win.yaml")


if __name__ == "__main__":
    main()
