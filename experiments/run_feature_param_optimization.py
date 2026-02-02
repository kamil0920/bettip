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

# Paths - Try SportMonks backup first, fall back to standard location
# Both files have sm_* columns (corners, cards, btts odds)
_SPORTMONKS_BACKUP = Path("data/sportmonks_backup/features_with_sportmonks_odds_FULL.parquet")
_SPORTMONKS_STANDARD = Path("data/03-features/features_with_sportmonks_odds.parquet")
FEATURES_FILE = _SPORTMONKS_BACKUP if _SPORTMONKS_BACKUP.exists() else _SPORTMONKS_STANDARD
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
        "odds_col": "sm_btts_yes_odds",  # SportMonks BTTS odds
        "approach": "classification",
        "default_threshold": 0.55,  # Lower threshold for BTTS (high base rate ~50%)
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
        "target_line": 24.5,  # Our data median=25, gives ~50% base rate
        # Note: SportMonks shots odds are for "shots on target" - different market
        "odds_col": "shots_over_odds",  # Will use fallback odds
        "approach": "regression_line",
        "default_threshold": 0.55,
    },
    "corners": {
        "target": "total_corners",
        "target_line": 9.5,  # SportMonks line - gives ~50% base rate
        "odds_col": "sm_corners_over_odds",  # SportMonks odds
        "approach": "regression_line",
        "default_threshold": 0.50,  # Lower for ~32% base rate
    },
    "cards": {
        "target": "total_cards",
        "target_line": 4.5,  # Matches SportMonks
        "odds_col": "sm_cards_over_odds",  # SportMonks odds
        "approach": "regression_line",
        "default_threshold": 0.50,  # Lower for ~37% base rate
    },
}

# Exclude columns (data leakage prevention)
EXCLUDE_COLUMNS = [
    # Identifiers
    "fixture_id", "date", "home_team_id", "home_team_name",
    "away_team_id", "away_team_name", "round", "season", "league",
    "sm_fixture_id",  # SportMonks fixture ID
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
    # SportMonks odds (used for ROI calc, not features)
    "sm_btts_", "sm_corners_", "sm_cards_", "sm_shots_",
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
    """Result of feature parameter optimization.

    Optimizes for Sharpe-like consistency score, not just mean precision.
    """
    bet_type: str
    best_params: Dict[str, Any]
    sharpe: float  # Sharpe-like consistency score (precision / std_precision)
    precision: float  # Mean precision across folds
    roi: float  # Mean ROI across folds
    n_bets: int
    n_trials: int
    n_folds: int
    fold_precisions: List[float]  # Per-fold precision for transparency
    search_space: Dict[str, tuple]  # (min, max, type) tuples
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
        from src.utils.data_io import load_features
        df = load_features(FEATURES_FILE)
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
        """Create BetTypeFeatureConfig from Optuna trial suggestions.

        Uses Bayesian optimization with continuous ranges:
        - suggest_int() for integer parameters (elo_k_factor, form_window, etc.)
        - suggest_float() for float parameters (half_life_days, etc.)
        """
        config_kwargs = {'bet_type': self.bet_type}

        for param_name, space_def in self.search_space.items():
            min_val, max_val, param_type = space_def
            if param_type == 'float':
                suggested = trial.suggest_float(param_name, min_val, max_val)
            else:  # 'int'
                suggested = trial.suggest_int(param_name, min_val, max_val)
            config_kwargs[param_name] = suggested

        return BetTypeFeatureConfig(**config_kwargs)

    def evaluate_config(
        self,
        feature_config: BetTypeFeatureConfig,
        features_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a feature configuration using walk-forward validation.

        Returns per-fold metrics to enable Sharpe-like optimization that
        rewards consistency across time periods, not just mean performance.

        Args:
            feature_config: Configuration to evaluate
            features_df: Pre-loaded features (for non-regeneration mode)

        Returns:
            Dict with per-fold and aggregate metrics:
            - fold_precisions: List of precision per fold
            - fold_rois: List of ROI per fold
            - fold_n_bets: List of bet count per fold
            - precision: Mean precision across folds
            - roi: Mean ROI across folds
            - n_bets: Total bets across folds
            - sharpe: Sharpe-like consistency score
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

        # Walk-forward validation - track per-fold metrics
        n_samples = len(y)
        fold_size = n_samples // (self.n_folds + 1)

        fold_precisions = []
        fold_rois = []
        fold_n_bets = []
        threshold = self.config["default_threshold"]

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

            # Per-fold evaluation
            mask = (probs >= threshold) & (odds_test >= 1.5) & (odds_test <= 6.0)
            n_bets_fold = mask.sum()

            if n_bets_fold >= 5:  # Minimum bets per fold for valid measurement
                wins = y_test[mask].sum()
                precision_fold = wins / n_bets_fold
                returns = np.where(y_test[mask] == 1, odds_test[mask] - 1, -1)
                roi_fold = returns.mean() * 100

                fold_precisions.append(precision_fold)
                fold_rois.append(roi_fold)
                fold_n_bets.append(n_bets_fold)

        # Need at least 2 folds for Sharpe calculation
        if len(fold_precisions) < 2:
            return {
                'fold_precisions': [],
                'fold_rois': [],
                'fold_n_bets': [],
                'precision': 0.0,
                'roi': -100.0,
                'n_bets': 0,
                'sharpe': -10.0,
            }

        # Aggregate metrics
        mean_precision = np.mean(fold_precisions)
        std_precision = np.std(fold_precisions)
        mean_roi = np.mean(fold_rois)
        total_bets = sum(fold_n_bets)

        # ROI lower-confidence-bound: optimizes for profit, not just win rate
        # 55% precision @ 2.50 odds beats 65% precision @ 1.50 odds
        std_roi = np.std(fold_rois) if len(fold_rois) > 1 else 0.0
        sharpe = mean_roi - std_roi

        return {
            'fold_precisions': fold_precisions,
            'fold_rois': fold_rois,
            'fold_n_bets': fold_n_bets,
            'precision': mean_precision,
            'roi': mean_roi,
            'n_bets': total_bets,
            'sharpe': sharpe,
        }

    def create_objective(self, features_df: pd.DataFrame):
        """Create Optuna objective function.

        Optimizes for Sharpe-like score (precision / std_precision) which
        rewards consistent performance across time periods, not just mean.
        This helps avoid overfitting to specific market conditions.
        """

        def objective(trial):
            feature_config = self.create_feature_config_from_trial(trial)
            metrics = self.evaluate_config(feature_config, features_df)

            # Store all metrics in trial for analysis
            trial.set_user_attr("precision", metrics['precision'])
            trial.set_user_attr("roi", metrics['roi'])
            trial.set_user_attr("n_bets", metrics['n_bets'])
            trial.set_user_attr("sharpe", metrics['sharpe'])
            trial.set_user_attr("fold_precisions", metrics['fold_precisions'])
            trial.set_user_attr("params_hash", feature_config.params_hash())

            # Optimize for Sharpe-like consistency score
            return metrics['sharpe']

        return objective

    def optimize(self) -> FeatureOptimizationResult:
        """Run feature parameter optimization."""
        logger.info(f"\n{'='*60}")
        logger.info(f"FEATURE PARAMETER OPTIMIZATION: {self.bet_type.upper()}")
        logger.info(f"{'='*60}\n")

        # Log Bayesian search space with ranges
        logger.info("Search space (Bayesian optimization with TPE):")
        for param, (min_val, max_val, ptype) in self.search_space.items():
            logger.info(f"  {param}: [{min_val}, {max_val}] ({ptype})")
        logger.info(f"\nTrials: {self.n_trials}, Folds: {self.n_folds}")

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

        # Get best result (by Sharpe-like consistency score)
        best_trial = study.best_trial
        best_sharpe = best_trial.value
        best_precision = best_trial.user_attrs.get("precision", 0.0)
        best_roi = best_trial.user_attrs.get("roi", 0.0)
        best_n_bets = best_trial.user_attrs.get("n_bets", 0)
        fold_precisions = best_trial.user_attrs.get("fold_precisions", [])

        logger.info(f"\nBest trial (optimized for consistency):")
        logger.info(f"  Sharpe score: {best_sharpe:.3f} (higher = more consistent)")
        logger.info(f"  Mean Precision: {best_precision*100:.1f}%")
        if fold_precisions:
            logger.info(f"  Fold Precisions: {[f'{p*100:.1f}%' for p in fold_precisions]}")
            logger.info(f"  Std Precision: {np.std(fold_precisions)*100:.1f}%")
        logger.info(f"  Mean ROI: {best_roi:+.1f}%")
        logger.info(f"  Total Bets: {best_n_bets}")
        logger.info(f"  Params: {best_trial.params}")

        # Collect all trials
        all_trials = []
        for trial in study.trials:
            all_trials.append({
                "number": trial.number,
                "params": trial.params,
                "sharpe": trial.value if trial.value is not None else -10.0,
                "precision": trial.user_attrs.get("precision", 0.0),
                "roi": trial.user_attrs.get("roi", 0.0),
                "n_bets": trial.user_attrs.get("n_bets", 0),
            })

        # Create result
        result = FeatureOptimizationResult(
            bet_type=self.bet_type,
            best_params=best_trial.params,
            sharpe=best_sharpe,
            precision=best_precision,
            roi=best_roi,
            n_bets=best_n_bets,
            n_trials=self.n_trials,
            n_folds=self.n_folds,
            fold_precisions=fold_precisions,
            search_space=self.search_space,
            all_trials=all_trials,
            timestamp=datetime.now().isoformat(),
        )

        return result

    def save_optimal_config(self, result: FeatureOptimizationResult, params_dir: Optional[Path] = None) -> Path:
        """Save optimal configuration to YAML file."""
        # Create BetTypeFeatureConfig with optimal params
        config = BetTypeFeatureConfig(bet_type=self.bet_type, **result.best_params)
        config.update_metadata(
            precision=result.precision,
            roi=result.roi,
            n_trials=result.n_trials,
        )

        # Save to config directory
        output_path = config.save(params_dir=params_dir)
        logger.info(f"Saved optimal config to: {output_path}")
        return output_path


def print_summary(results: List[FeatureOptimizationResult]):
    """Print optimization summary."""
    print("\n" + "=" * 100)
    print("                     FEATURE PARAMETER OPTIMIZATION RESULTS")
    print("                     (Optimized for Sharpe-like consistency)")
    print("=" * 100)

    print(f"\n{'Bet Type':<12} {'Sharpe':>8} {'Precision':>10} {'Std':>8} {'ROI':>10} {'Bets':>8}")
    print("-" * 100)

    for r in sorted(results, key=lambda x: x.sharpe, reverse=True):
        std_prec = np.std(r.fold_precisions) * 100 if r.fold_precisions else 0.0
        print(f"{r.bet_type:<12} {r.sharpe:>8.2f} {r.precision*100:>9.1f}% {std_prec:>7.1f}% {r.roi:>+9.1f}% {r.n_bets:>8}")

    print("-" * 100)
    print("\nSharpe = mean(precision) / std(precision). Higher = more consistent across time periods.")

    # Show per-fold details for best result
    if results:
        best = max(results, key=lambda x: x.sharpe)
        if best.fold_precisions:
            print(f"\nBest ({best.bet_type}) fold precisions: {[f'{p*100:.1f}%' for p in best.fold_precisions]}")

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
    parser.add_argument("--feature-params-dir", type=str, default=None,
                       help="Custom feature params output directory (e.g., config/feature_params/americas)")
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
            params_dir = Path(args.feature_params_dir) if args.feature_params_dir else None
            optimizer.save_optimal_config(result, params_dir=params_dir)

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
