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
1. (Optional) Feature Parameter Optimization / Loading
2. RFE Feature Selection (reduce to optimal subset)
3. Hyperparameter Tuning (Optuna with walk-forward validation)
4. Threshold Optimization (grid search over prob/odds thresholds)
5. Save optimal configuration

Usage:
    # Single bet type (default features)
    python experiments/run_sniper_optimization.py --bet-type away_win

    # With custom feature params from file
    python experiments/run_sniper_optimization.py --bet-type away_win \
        --feature-params config/feature_params/away_win.yaml

    # With feature parameter optimization first
    python experiments/run_sniper_optimization.py --bet-type away_win \
        --optimize-features --n-feature-trials 20

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
from sklearn.feature_selection import RFE, RFECV
from sklearn.linear_model import RidgeClassifierCV
from catboost import CatBoostClassifier
import lightgbm as lgb
import xgboost as xgb
import joblib
from tqdm import tqdm

# Feature parameter optimization
from src.features.config_manager import BetTypeFeatureConfig
from src.features.regeneration import FeatureRegenerator

# Sample weighting (retail forecasting integration)
from src.ml.sample_weighting import (
    calculate_time_decay_weights,
    decay_rate_from_half_life,
    get_recommended_decay_rate,
)

# Disagreement ensemble for high-confidence betting
from src.ml.ensemble_disagreement import DisagreementEnsemble, create_disagreement_ensemble

# SHAP for feature importance analysis
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not available. Install with: pip install shap")

# Deep learning models (optional)
try:
    from src.ml.models import FastAITabularModel
    # Check if actual fastai library is installed (not just our wrapper class)
    import fastai.tabular.all  # noqa: F401
    FASTAI_AVAILABLE = True
except ImportError:
    FASTAI_AVAILABLE = False
    try:
        from src.ml.models import FastAITabularModel  # noqa: F811
    except ImportError:
        pass

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Paths - Use unified features file with football-data.co.uk odds
FEATURES_FILE = Path("data/03-features/features_all_5leagues_with_odds.parquet")
OUTPUT_DIR = Path("experiments/outputs/sniper_optimization")
MODELS_DIR = Path("models")

# Bet type configurations
BET_TYPES = {
    "away_win": {
        "target": "away_win",
        "odds_col": "avg_away_close",
        "approach": "classification",
        "default_threshold": 0.60,
        # R36 selected 0.80; floor at 0.65 to prevent threshold collapse
        "threshold_search": [0.65, 0.68, 0.70, 0.72, 0.75, 0.78, 0.80, 0.85],
        # Away wins have wide odds range (underdogs can be 3.0-8.0)
        "min_odds_search": [1.4, 1.6, 1.8, 2.0, 2.5],
        "max_odds_search": [4.0, 5.0, 6.0, 8.0],
    },
    "home_win": {
        "target": "home_win",
        "odds_col": "avg_home_close",
        "approach": "classification",
        "default_threshold": 0.60,
        # R36 selected 0.80; floor at 0.65 to prevent threshold collapse
        "threshold_search": [0.65, 0.70, 0.75, 0.80, 0.85],
        # Home wins typically 1.2-3.0 for favorites, but underdogs can go higher
        "min_odds_search": [1.2, 1.4, 1.6, 1.8, 2.0],
        "max_odds_search": [3.0, 4.0, 5.0, 6.0],
    },
    "btts": {
        "target": "btts",
        "odds_col": "btts_yes_odds",  # No bulk historical BTTS odds; uses fallback
        "approach": "classification",
        "default_threshold": 0.55,  # Lower threshold for BTTS (high base rate ~50%)
        # R36 selected 0.75; floor at 0.60
        "threshold_search": [0.60, 0.65, 0.70, 0.75, 0.80],
        # BTTS odds typically 1.6-2.5 range
        "min_odds_search": [1.4, 1.5, 1.6, 1.8, 2.0],
        "max_odds_search": [2.5, 3.0, 3.5],
    },
    "over25": {
        "target": "over25",
        "odds_col": "avg_over25_close",
        "approach": "classification",
        "default_threshold": 0.60,
        # R36 selected 0.85; floor raised to 0.75 (R47-49: 0.65 floor → poor calibration; R59-62: always selected 0.70 floor)
        "threshold_search": [0.75, 0.80, 0.85],
        # Over 2.5 odds typically 1.5-2.2 range; R53-56 failed with min_odds=2.0
        "min_odds_search": [1.4, 1.5, 1.6, 1.8],
        "max_odds_search": [2.5, 3.0, 3.5],
    },
    "under25": {
        "target": "under25",
        "odds_col": "avg_under25_close",
        "approach": "classification",
        "default_threshold": 0.55,
        # R36 selected 0.75; floor raised to 0.65 (R47-49: 0.55-0.60 produced garbage holdout ROI +18-37%)
        "threshold_search": [0.65, 0.70, 0.75, 0.80],
        # Under 2.5 odds typically 1.6-2.5 range; R53-56 failed with min_odds=2.0
        "min_odds_search": [1.4, 1.5, 1.6, 1.8],
        "max_odds_search": [2.5, 3.0, 3.5],
    },
    "fouls": {
        "target": "total_fouls",
        "target_line": 24.5,
        "odds_col": "fouls_over_odds",
        "approach": "regression_line",
        "default_threshold": 0.60,
        # R36 selected 0.75; floor at 0.60
        "threshold_search": [0.60, 0.65, 0.70, 0.75, 0.80],
        # Fouls market typically uses fallback odds around 1.9
        "min_odds_search": [1.2, 1.4, 1.6, 1.8],
        "max_odds_search": [2.5, 3.0, 3.5],
    },
    "shots": {
        "target": "total_shots",
        "target_line": 24.5,  # Our data median=25, gives ~50% base rate
        # Note: SportMonks shots odds (10.5 line) are for "shots on target" - different market
        # Using fallback odds since markets don't match
        "odds_col": "shots_over_odds",  # Will use fallback (no real odds for total shots)
        "approach": "regression_line",
        "default_threshold": 0.55,
        # R36 selected 0.70; floor at 0.55
        "threshold_search": [0.55, 0.60, 0.65, 0.70, 0.75],
        # Shots market uses fallback odds around 1.9
        "min_odds_search": [1.2, 1.4, 1.6, 1.8],
        "max_odds_search": [2.5, 3.0, 3.5],
    },
    "corners": {
        "target": "total_corners",
        "target_line": 9.5,  # SportMonks line (was 10.5) - gives ~50% base rate
        "odds_col": "corners_over_odds",  # No bulk historical odds; uses fallback
        "approach": "regression_line",
        "default_threshold": 0.50,  # Lower threshold for ~32% base rate at this line
        "threshold_search": [0.40, 0.45, 0.50, 0.55, 0.60],
        # Corners odds typically 1.7-2.3 range
        "min_odds_search": [1.2, 1.4, 1.6, 1.8],
        "max_odds_search": [2.5, 3.0, 3.5],
    },
    "cards": {
        "target": "total_cards",
        "target_line": 4.5,  # Matches SportMonks line
        "odds_col": "cards_over_odds",  # No bulk historical odds; uses fallback
        "approach": "regression_line",
        "default_threshold": 0.50,  # Lower threshold for ~37% base rate
        "threshold_search": [0.40, 0.45, 0.50, 0.55, 0.60],
        # Cards odds typically 1.7-2.3 range
        "min_odds_search": [1.2, 1.4, 1.6, 1.8],
        "max_odds_search": [2.5, 3.0, 3.5],
    },
}

# Exclude columns (data leakage prevention)
EXCLUDE_COLUMNS = [
    # Identifiers
    "fixture_id", "date", "home_team_id", "home_team_name",
    "away_team_id", "away_team_name", "round", "season", "league",
    "sm_fixture_id",  # SportMonks fixture ID
    # Target variables (match outcomes)
    "home_win", "draw", "away_win", "match_result", "result",
    "total_goals", "goal_difference", "xg_diff",
    "home_goals", "away_goals", "btts",
    "under25", "over25", "under35", "over35",
    # Match statistics (not available pre-match - these are outcomes!)
    "home_shots", "away_shots", "home_shots_on_target", "away_shots_on_target",
    "home_corners", "away_corners", "total_corners",
    "home_fouls", "away_fouls", "total_fouls",
    "home_yellows", "away_yellows", "home_reds", "away_reds",
    "home_yellow_cards", "away_yellow_cards",  # Alternate naming from events pipeline
    "home_red_cards", "away_red_cards",  # Alternate naming from events pipeline
    "home_possession", "away_possession",
    "total_cards", "total_shots",
    "home_cards", "away_cards",  # Match outcome cards (not historical)
]

# Per-bet-type low-importance feature exclusions (R33 SHAP analysis).
# Only excludes features that are low-importance AND NOT in that bet type's top-20.
# Features important for specific markets are preserved where they matter.
LOW_IMPORTANCE_EXCLUSIONS: Dict[str, List[str]] = {
    "away_win": [
        "away_corners_won_ema", "away_first_half_rate", "discipline_diff",
        "expected_home_corners", "h2h_away_wins", "home_cards_ema",
        "home_corners_won_ema", "home_corners_won_roll_10", "home_corners_won_roll_5",
        "home_importance", "home_points_last_n", "home_pts_to_cl",
        "home_shot_accuracy", "home_shots_conceded_ema", "home_shots_ema_x",
        "home_unbeaten_streak", "importance_diff", "match_importance",
        "away_corners_won_roll_10", "away_shots_ema_y",
    ],
    "btts": [
        "away_corners_conceded_ema", "away_corners_won_ema", "away_corners_won_roll_10",
        "away_first_half_rate", "away_shots_ema_x", "away_shots_ema_y",
        "discipline_diff", "expected_home_corners", "h2h_away_wins",
        "home_cards_ema", "home_corners_won_ema", "home_corners_won_roll_10",
        "home_corners_won_roll_5", "home_importance", "home_points_last_n",
        "home_pts_to_cl", "home_shot_accuracy", "home_shots_conceded_ema",
        "home_shots_ema_x", "home_unbeaten_streak", "importance_diff",
        "match_importance",
    ],
    "cards": [
        "away_corners_conceded_ema", "away_corners_won_ema", "away_corners_won_roll_10",
        "away_first_half_rate", "away_shots_ema_x", "expected_home_corners",
        "h2h_away_wins", "home_cards_ema", "home_corners_won_ema",
        "home_corners_won_roll_10", "home_corners_won_roll_5", "home_importance",
        "home_points_last_n", "home_pts_to_cl", "home_shot_accuracy",
        "home_shots_conceded_ema", "home_shots_ema_x", "home_unbeaten_streak",
        "importance_diff", "match_importance",
    ],
    "corners": [
        "away_corners_conceded_ema", "away_corners_won_ema", "away_corners_won_roll_10",
        "away_first_half_rate", "away_shots_ema_x", "away_shots_ema_y",
        "discipline_diff", "expected_home_corners", "h2h_away_wins",
        "home_corners_won_ema", "home_corners_won_roll_10", "home_corners_won_roll_5",
        "home_importance", "home_points_last_n", "home_shot_accuracy",
        "home_shots_conceded_ema", "home_unbeaten_streak", "importance_diff",
        "match_importance",
    ],
    "fouls": [
        "away_corners_conceded_ema", "away_corners_won_ema", "away_corners_won_roll_10",
        "away_first_half_rate", "away_shots_ema_x", "away_shots_ema_y",
        "discipline_diff", "expected_home_corners", "h2h_away_wins",
        "home_cards_ema", "home_corners_won_ema", "home_corners_won_roll_10",
        "home_corners_won_roll_5", "home_importance", "home_points_last_n",
        "home_pts_to_cl", "home_shot_accuracy", "home_shots_conceded_ema",
        "home_shots_ema_x", "home_unbeaten_streak", "importance_diff",
    ],
    "home_win": [
        "away_corners_won_ema", "away_first_half_rate", "discipline_diff",
        "expected_home_corners", "h2h_away_wins", "home_cards_ema",
        "home_corners_won_ema", "home_corners_won_roll_10", "home_corners_won_roll_5",
        "home_importance", "home_points_last_n", "home_pts_to_cl",
        "home_shot_accuracy", "home_shots_conceded_ema", "home_shots_ema_x",
        "home_unbeaten_streak", "importance_diff", "match_importance",
        "away_corners_conceded_ema", "away_corners_won_roll_10", "away_shots_ema_y",
    ],
    "over25": [
        "away_corners_conceded_ema", "away_corners_won_ema", "away_corners_won_roll_10",
        "away_first_half_rate", "away_shots_ema_x", "away_shots_ema_y",
        "discipline_diff", "expected_home_corners", "h2h_away_wins",
        "home_cards_ema", "home_corners_won_ema", "home_corners_won_roll_10",
        "home_corners_won_roll_5", "home_importance", "home_points_last_n",
        "home_pts_to_cl", "home_shot_accuracy", "home_shots_conceded_ema",
        "home_shots_ema_x", "home_unbeaten_streak", "importance_diff",
        "match_importance",
    ],
    "shots": [
        "away_corners_conceded_ema", "away_corners_won_ema", "away_first_half_rate",
        "away_shots_ema_x", "away_shots_ema_y", "discipline_diff",
        "expected_home_corners", "h2h_away_wins", "home_cards_ema",
        "home_importance", "home_points_last_n", "home_shot_accuracy",
        "home_shots_conceded_ema", "home_unbeaten_streak", "importance_diff",
        "match_importance",
    ],
    "under25": [
        "away_corners_conceded_ema", "away_corners_won_ema", "away_corners_won_roll_10",
        "away_first_half_rate", "away_shots_ema_x", "away_shots_ema_y",
        "discipline_diff", "expected_home_corners", "h2h_away_wins",
        "home_cards_ema", "home_corners_won_ema", "home_corners_won_roll_10",
        "home_corners_won_roll_5", "home_importance", "home_points_last_n",
        "home_pts_to_cl", "home_shot_accuracy", "home_shots_conceded_ema",
        "home_shots_ema_x", "home_unbeaten_streak", "importance_diff",
        "match_importance",
    ],
}

# Patterns that indicate odds/bookmaker data (leaky for predicting match outcomes)
LEAKY_PATTERNS = [
    # Direct odds
    "avg_home", "avg_away", "avg_draw", "avg_over", "avg_under", "avg_ah",
    "b365_", "pinnacle_", "max_home", "max_away", "max_draw", "max_over", "max_under", "max_ah",
    # SportMonks odds (used for ROI calc, not features)
    "sm_btts_", "sm_corners_", "sm_cards_", "sm_shots_",
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

MIN_ODDS_SEARCH = [1.2, 1.4, 1.5, 1.8, 2.0, 2.5]
MAX_ODDS_SEARCH = [3.0, 3.5, 4.0, 5.0, 6.0, 8.0]


def _adversarial_validation(
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: List[str],
) -> Tuple[float, List[Tuple[str, float]]]:
    """Train LGB classifier to distinguish train vs test. AUC > 0.6 = distribution shift.

    Args:
        X_train: Training features.
        X_test: Test features.
        feature_names: Feature names for interpretability.

    Returns:
        Tuple of (auc, top_shifting_features) where top_shifting_features
        is a list of (feature_name, importance) tuples sorted by importance.
    """
    from sklearn.metrics import roc_auc_score

    y_adv = np.concatenate([np.zeros(len(X_train)), np.ones(len(X_test))])
    X_adv = np.vstack([X_train, X_test])
    clf = lgb.LGBMClassifier(n_estimators=50, max_depth=3, verbose=-1, random_state=42)
    clf.fit(X_adv, y_adv)
    auc = roc_auc_score(y_adv, clf.predict_proba(X_adv)[:, 1])

    importances = dict(zip(feature_names, clf.feature_importances_))
    top_shift = sorted(importances.items(), key=lambda x: -x[1])[:10]
    return auc, top_shift


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
    saved_models: List[str] = None
    # Retail forecasting integration
    sample_decay_rate: float = None
    sample_min_weight: float = None
    threshold_alpha: float = None  # Odds-dependent threshold parameter
    # Held-out (unbiased) metrics from final walk-forward fold
    holdout_metrics: Dict[str, Any] = None
    # Meta-learner stacking weights from RidgeClassifierCV
    stacking_weights: Dict[str, float] = None
    stacking_alpha: float = None
    # Adversarial validation diagnostics
    adversarial_validation: Dict[str, Any] = None
    # Calibration method selected by Optuna
    calibration_method: str = None
    # Per-league calibration metrics
    per_league_ece: Dict[str, float] = None


class SniperOptimizer:
    """
    Unified sniper optimization pipeline.

    Supports optional feature parameter optimization/loading:
    - Load custom feature params from YAML file
    - Run feature parameter optimization before model optimization
    - Regenerate features with optimal params
    """

    def __init__(
        self,
        bet_type: str,
        n_folds: int = 5,
        n_rfe_features: int = 100,
        auto_rfe: bool = False,
        min_rfe_features: int = 20,
        max_rfe_features: int = 80,
        n_optuna_trials: int = 150,
        min_bets: int = 30,
        run_walkforward: bool = False,
        run_shap: bool = False,
        feature_params_path: Optional[str] = None,
        optimize_features: bool = False,
        n_feature_trials: int = 20,
        feature_params_dir: Optional[Path] = None,
        # Retail forecasting integration parameters
        use_sample_weights: bool = False,
        sample_decay_rate: Optional[float] = None,
        use_odds_threshold: bool = False,
        threshold_alpha: float = 0.0,
        filter_missing_odds: bool = True,
        calibration_method: str = "beta",
        temporal_buffer: int = 50,
        seed: int = 42,
        fast_mode: bool = False,
    ):
        self.bet_type = bet_type
        self.config = BET_TYPES[bet_type]
        self.n_folds = n_folds
        self.n_rfe_features = n_rfe_features
        self.auto_rfe = auto_rfe
        self.min_rfe_features = min_rfe_features
        self.max_rfe_features = max_rfe_features
        self.n_optuna_trials = n_optuna_trials
        self.min_bets = min_bets
        self.run_walkforward = run_walkforward
        self.run_shap = run_shap

        # Feature parameter options
        self.feature_params_path = feature_params_path
        self.optimize_features = optimize_features
        self.n_feature_trials = n_feature_trials
        self.feature_params_dir = feature_params_dir
        self.feature_config: Optional[BetTypeFeatureConfig] = None
        self.regenerator: Optional[FeatureRegenerator] = None

        # Retail forecasting integration
        self.use_sample_weights = use_sample_weights
        self.sample_decay_rate = sample_decay_rate or get_recommended_decay_rate("football")
        self.use_odds_threshold = use_odds_threshold
        self.threshold_alpha = threshold_alpha
        self.filter_missing_odds = filter_missing_odds
        self.temporal_buffer = temporal_buffer
        self.seed = seed
        self.fast_mode = fast_mode
        self.use_two_stage = not fast_mode  # Two-stage models only in full mode

        # Calibration method: "sigmoid", "isotonic", "beta", "temperature"
        self.calibration_method = calibration_method
        # CalibratedClassifierCV only supports sigmoid/isotonic; others are post-hoc
        self._sklearn_cal_method = (
            calibration_method if calibration_method in ("sigmoid", "isotonic") else "sigmoid"
        )
        self._use_custom_calibration = calibration_method in ("beta", "temperature")

        self.features_df = None
        self.feature_columns = None
        self.optimal_features = None
        self.best_params = None
        self.best_model_type = None
        self.all_model_params = {}
        self.dates = None  # Store dates for sample weight calculation
        self.league_col = None  # Store league column for per-league calibration

        # Deep learning integration
        self.use_fastai = FASTAI_AVAILABLE

    @staticmethod
    def _get_base_model_types(
        include_fastai: bool = True,
        fast_mode: bool = False,
        include_two_stage: bool = False,
    ) -> List[str]:
        """Return list of base model types to use."""
        if fast_mode:
            models = ["lightgbm", "xgboost"]
        else:
            models = ["lightgbm", "catboost", "xgboost"]
            if include_fastai and FASTAI_AVAILABLE:
                models.append("fastai")
        if include_two_stage and not fast_mode:
            models.extend(["two_stage_lgb", "two_stage_xgb"])
        return models

    @staticmethod
    def _create_model_instance(model_type: str, params: Dict[str, Any], seed: int = 42):
        """Create a model instance for the given type and params."""
        if model_type == "lightgbm":
            return lgb.LGBMClassifier(**params, random_state=seed, verbose=-1)
        elif model_type == "catboost":
            return CatBoostClassifier(**params, random_seed=seed, verbose=False)
        elif model_type == "xgboost":
            return xgb.XGBClassifier(**params, random_state=seed, verbosity=0)
        elif model_type == "fastai":
            return FastAITabularModel(**params, random_state=seed)
        elif model_type.startswith("two_stage_"):
            from src.ml.two_stage_model import create_two_stage_model
            base_type = model_type.replace("two_stage_", "")
            type_map = {"lgb": "lightgbm", "xgb": "lightgbm"}  # Both use lightgbm-backed stages
            if base_type == "lgb":
                return create_two_stage_model("lightgbm")
            elif base_type == "xgb":
                # Use CatBoost as alternative Stage 1 for diversity
                return create_two_stage_model("catboost")
            return create_two_stage_model("logistic")
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def load_data(self) -> pd.DataFrame:
        """Load and prepare feature data."""
        from src.utils.data_io import load_features
        df = load_features(FEATURES_FILE)
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

    def load_or_optimize_feature_config(self) -> Optional[BetTypeFeatureConfig]:
        """
        Load or optimize feature parameters.

        Returns:
            BetTypeFeatureConfig if using custom params, None for default behavior
        """
        if self.feature_params_path:
            # Load from file
            logger.info(f"Loading feature params from {self.feature_params_path}")
            config = BetTypeFeatureConfig.load(Path(self.feature_params_path))
            logger.info(f"Loaded feature config: {config.summary()}")
            return config

        elif self.optimize_features:
            # Run feature parameter optimization
            logger.info(f"Running feature parameter optimization ({self.n_feature_trials} trials)...")

            # Import here to avoid circular dependency
            from experiments.run_feature_param_optimization import FeatureParamOptimizer

            optimizer = FeatureParamOptimizer(
                bet_type=self.bet_type,
                n_trials=self.n_feature_trials,
                n_folds=self.n_folds,
                min_bets=self.min_bets,
                use_regeneration=True,  # Regenerate features with different params for true optimization
            )

            result = optimizer.optimize()

            # Create config from result
            config = BetTypeFeatureConfig(bet_type=self.bet_type, **result.best_params)
            config.update_metadata(
                precision=result.precision,
                roi=result.roi,
                n_trials=result.n_trials,
            )

            # Save for future use
            output_path = config.save(params_dir=self.feature_params_dir)
            logger.info(f"Saved optimized feature params to {output_path}")

            return config

        return None

    def load_data_with_feature_config(self) -> pd.DataFrame:
        """
        Load feature data, optionally regenerating with custom params.

        If feature_config is set and has custom params, regenerates features
        using the FeatureRegenerator. Otherwise, loads from default file.

        Returns:
            DataFrame with features
        """
        if self.feature_config is not None and self.feature_config.optimized:
            # Regenerate features with custom params
            logger.info("Regenerating features with optimized params...")

            if self.regenerator is None:
                self.regenerator = FeatureRegenerator()

            df = self.regenerator.regenerate_with_params(self.feature_config)

            # Derive target if needed (regenerated features may not have targets)
            target = self.config["target"]
            if target not in df.columns:
                self._derive_target(df, target)

            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)

            logger.info(f"Regenerated {len(df)} matches with custom feature params")
            return df
        else:
            # Use default feature loading
            return self.load_data()

    def _derive_target(self, df: pd.DataFrame, target: str) -> None:
        """Derive target column if not present in dataframe."""
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

    def calculate_sample_weights(self, dates: pd.Series) -> np.ndarray:
        """
        Calculate time-decayed sample weights for training.

        Implements the retail forecasting insight: recent observations
        should have higher weight during training.

        Args:
            dates: Series of match dates

        Returns:
            Array of sample weights
        """
        if not self.use_sample_weights:
            return None

        min_weight = getattr(self, 'sample_min_weight', 0.1)
        weights = calculate_time_decay_weights(
            dates,
            decay_rate=self.sample_decay_rate,
            min_weight=min_weight,
        )

        logger.info(f"Sample weights: min={weights.min():.3f}, max={weights.max():.3f}, "
                   f"mean={weights.mean():.3f}, decay_rate={self.sample_decay_rate:.4f}")

        return weights

    def calculate_odds_adjusted_threshold(
        self,
        base_threshold: float,
        odds: np.ndarray,
        alpha: float = None,
    ) -> np.ndarray:
        """
        Calculate odds-dependent betting thresholds.

        Implements the newsvendor critical fractile concept from retail forecasting:
        - Lower threshold for longshots (high odds) - more tolerant of uncertainty
        - Higher threshold for favorites (low odds) - need higher confidence

        Formula: threshold = base_threshold * (1 / odds)^alpha
        - alpha = 0: Fixed threshold (current behavior)
        - alpha = 1: Full newsvendor adjustment
        - Recommended: alpha in [0.1, 0.3] for sports betting

        Args:
            base_threshold: Base probability threshold
            odds: Array of decimal odds
            alpha: Adjustment strength (default: self.threshold_alpha)

        Returns:
            Array of adjusted thresholds per bet
        """
        if alpha is None:
            alpha = self.threshold_alpha

        if alpha == 0 or not self.use_odds_threshold:
            return np.full(len(odds), base_threshold)

        # Normalize odds to avoid extreme adjustments
        # Use 2.0 as reference point (even odds)
        odds_ratio = 2.0 / np.clip(odds, 1.2, 10.0)

        # Apply adjustment: higher odds -> lower threshold
        adjusted = base_threshold * (odds_ratio ** alpha)

        # Clamp to reasonable range
        adjusted = np.clip(adjusted, 0.3, 0.9)

        return adjusted

    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get valid feature columns excluding leakage and low-importance features."""
        all_cols = set(df.columns)
        exclude = set(EXCLUDE_COLUMNS)

        # Per-bet-type low-importance exclusions (R33 SHAP analysis)
        bt_exclusions = LOW_IMPORTANCE_EXCLUSIONS.get(self.bet_type, [])
        exclude.update(bt_exclusions)

        # Exclude columns matching leaky patterns (bookmaker odds, implied probs)
        for col in all_cols:
            col_lower = col.lower()
            for pattern in LEAKY_PATTERNS:
                if pattern.lower() in col_lower:
                    exclude.add(col)
                    break

        features = [c for c in all_cols - exclude if df[c].dtype in ['float64', 'int64', 'float32', 'int32']]
        n_low_imp = len(set(bt_exclusions) & all_cols)
        logger.info(f"Excluded {len(exclude)} columns ({n_low_imp} low-importance), {len(features)} features remain")
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

    def run_rfe(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
    ) -> List[int]:
        """Run RFE or RFECV to select optimal features.

        If auto_rfe=True, uses RFECV with cross-validation to find optimal count.
        Otherwise, uses fixed n_rfe_features.

        When sample_weights are provided and auto_rfe=False, uses weighted
        importance ranking instead of sklearn RFE (which doesn't support
        sample_weight), ensuring feature selection is consistent with the
        weighted training objective.
        """
        # Use LightGBM as base estimator
        base_model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=5,
            importance_type='gain',
            reg_alpha=0.5,
            reg_lambda=0.5,
            colsample_bytree=0.8,
            class_weight='balanced',
            random_state=self.seed,
            n_jobs=1,
            verbose=-1,
        )

        if self.auto_rfe:
            # RFECV: automatically find optimal number of features via CV
            # Cap at max_rfe_features to prevent bloated feature sets (R52: under25 got 155 features)
            logger.info(f"Running RFECV to find optimal feature count "
                       f"(min={self.min_rfe_features}, max={self.max_rfe_features})...")

            from sklearn.model_selection import TimeSeriesSplit
            cv = TimeSeriesSplit(n_splits=3)

            rfecv = RFECV(
                estimator=base_model,
                step=10,
                cv=cv,
                scoring='roc_auc',
                min_features_to_select=self.min_rfe_features,
                n_jobs=-1,
            )
            rfecv.fit(X, y)

            selected_indices = np.where(rfecv.support_)[0]
            optimal_n = rfecv.n_features_
            logger.info(f"RFECV found optimal feature count: {optimal_n}")
            logger.info(f"CV scores by n_features: min={min(rfecv.cv_results_['mean_test_score']):.3f}, "
                       f"max={max(rfecv.cv_results_['mean_test_score']):.3f}")

            # Enforce max cap: if RFECV selected too many, trim to top N by importance
            if len(selected_indices) > self.max_rfe_features:
                logger.warning(f"RFECV selected {len(selected_indices)} features, "
                             f"capping to {self.max_rfe_features} by importance ranking")
                base_model.fit(X[:, selected_indices], y)
                importances = base_model.feature_importances_
                top_k = np.argsort(importances)[::-1][:self.max_rfe_features]
                selected_indices = np.sort(selected_indices[top_k])
                logger.info(f"Capped to {len(selected_indices)} features")
        elif sample_weights is not None:
            # Weighted importance ranking: train with sample weights, select by gain importance
            logger.info(f"Running weighted feature selection (top {self.n_rfe_features} by weighted gain)...")

            base_model.fit(X, y, sample_weight=sample_weights)
            importances = base_model.feature_importances_

            n_features = min(self.n_rfe_features, X.shape[1])
            selected_indices = np.argsort(importances)[::-1][:n_features]
            selected_indices = np.sort(selected_indices)  # Restore original order

            logger.info(f"Selected {len(selected_indices)} features via weighted importance ranking")
            return selected_indices.tolist()
        else:
            # Fixed RFE: use specified n_rfe_features
            logger.info(f"Running RFE to select top {self.n_rfe_features} features...")

            n_features = min(self.n_rfe_features, X.shape[1])
            rfe = RFE(estimator=base_model, n_features_to_select=n_features, step=10)
            rfe.fit(X, y)

            selected_indices = np.where(rfe.support_)[0]

        logger.info(f"Selected {len(selected_indices)} features via {'RFECV' if self.auto_rfe else 'RFE'}")
        return selected_indices.tolist()

    def create_objective(
        self,
        X: np.ndarray,
        y: np.ndarray,
        odds: np.ndarray,
        model_type: str,
        dates: Optional[np.ndarray] = None,
    ):
        """Create Optuna objective for a specific model type."""

        def objective(trial):
            # Calibration method (tuned per trial)
            trial_cal_method = trial.suggest_categorical("calibration_method", ["sigmoid", "isotonic"])

            # Sample weight hyperparameters (tuned per trial)
            if self.use_sample_weights and dates is not None:
                # R47/R48 decay collapsed to 0.0006; R59-62 shots hit 0.002 floor. Allow slower decay.
                trial_decay_rate = trial.suggest_float("decay_rate", 0.001, 0.01, log=True)
                trial_min_weight = trial.suggest_float("min_weight", 0.05, 0.5)
            else:
                trial_decay_rate = None
                trial_min_weight = None

            if model_type == "lightgbm":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
                    "max_depth": trial.suggest_int("max_depth", 3, 8),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                    "num_leaves": trial.suggest_int("num_leaves", 20, 100),
                    "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
                    "random_state": self.seed,
                    "verbose": -1,
                }
                ModelClass = lgb.LGBMClassifier
            elif model_type == "catboost":
                params = {
                    "iterations": trial.suggest_int("iterations", 100, 800, step=100),
                    "depth": trial.suggest_int("depth", 4, 8),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.35, log=True),
                    "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 100, log=True),
                    "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
                    "random_seed": self.seed,
                    "verbose": False,
                }
                ModelClass = CatBoostClassifier
            elif model_type == "xgboost":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 700, step=100),
                    "max_depth": trial.suggest_int("max_depth", 3, 8),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.35, log=True),
                    "min_child_weight": trial.suggest_int("min_child_weight", 1, 50),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
                    "random_state": self.seed,
                    "verbosity": 0,
                }
                ModelClass = xgb.XGBClassifier
            elif model_type == "fastai":
                layer1 = trial.suggest_categorical("layer1", [200, 400, 600])
                layer2 = trial.suggest_categorical("layer2", [100, 200, 300])
                params = {
                    "layers": [layer1, layer2],
                    "epochs": trial.suggest_int("epochs", 30, 100, step=10),
                    "ps": [
                        trial.suggest_float("ps1", 0.001, 0.2, log=True),
                        trial.suggest_float("ps2", 0.001, 0.2, log=True),
                    ],
                    "embed_p": trial.suggest_float("embed_p", 0.01, 0.1),
                    "random_state": self.seed,
                }
                ModelClass = FastAITabularModel
            elif model_type.startswith("two_stage_"):
                # Two-stage models use their own internal hyperparameters.
                # Just pass min_edge_threshold as a tunable parameter.
                params = {
                    "min_edge_threshold": trial.suggest_float("min_edge", 0.0, 0.05),
                }
                ModelClass = None  # Handled separately in the fold loop

            # Walk-forward validation
            n_samples = len(y)
            fold_size = n_samples // (self.n_folds + 1)

            all_preds = []
            all_actuals = []
            all_odds = []

            for fold in range(self.n_folds):
                train_end = (fold + 1) * fold_size
                test_start = train_end + self.temporal_buffer
                test_end = test_start + fold_size

                if test_start >= n_samples:
                    continue
                if test_end > n_samples:
                    test_end = n_samples

                X_train, y_train = X[:train_end], y[:train_end]
                X_test, y_test = X[test_start:test_end], y[test_start:test_end]
                odds_test = odds[test_start:test_end]

                # Calculate sample weights for training data (using trial params)
                sample_weights = None
                if self.use_sample_weights and dates is not None and trial_decay_rate is not None:
                    train_dates = pd.to_datetime(dates[:train_end])
                    sample_weights = calculate_time_decay_weights(
                        train_dates,
                        decay_rate=trial_decay_rate,
                        min_weight=trial_min_weight,
                    )

                if len(X_train) < 100 or len(X_test) < 20:
                    continue

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                try:
                    if model_type.startswith("two_stage_"):
                        # Two-stage models have non-standard API
                        from src.ml.two_stage_model import create_two_stage_model
                        base = "lightgbm" if model_type == "two_stage_lgb" else "catboost"
                        ts_model = create_two_stage_model(base)
                        # Get odds for training/test sets
                        odds_train = odds[:train_end]
                        ts_model.fit(X_train_scaled, y_train, odds_train)
                        result_dict = ts_model.predict_proba(X_test_scaled, odds_test)
                        probs = result_dict['combined_score']
                    else:
                        model = ModelClass(**params)
                        calibrated = CalibratedClassifierCV(model, method=trial_cal_method, cv=3)

                        # Use sample weights if available
                        if sample_weights is not None:
                            calibrated.fit(X_train_scaled, y_train, sample_weight=sample_weights)
                        else:
                            calibrated.fit(X_train_scaled, y_train)
                        probs = calibrated.predict_proba(X_test_scaled)[:, 1]
                except Exception as e:
                    logger.debug(f"Trial failed during model fitting: {e}")
                    return float("-inf")

                all_preds.extend(probs)
                all_actuals.extend(y_test)
                all_odds.extend(odds_test)

            if len(all_preds) == 0:
                return float("-inf")

            preds = np.array(all_preds)
            actuals = np.array(all_actuals)
            odds_arr = np.array(all_odds)

            # Use negative log_loss as objective for better-calibrated probabilities.
            # Lower log_loss = better calibrated = better downstream betting decisions.
            from sklearn.metrics import log_loss as sklearn_log_loss
            eps = 1e-15
            clipped_preds = np.clip(preds, eps, 1 - eps)
            try:
                ll = sklearn_log_loss(actuals, clipped_preds)
            except Exception as e:
                logger.debug(f"Trial failed during log_loss calculation: {e}")
                return float("-inf")

            # Also require minimum bet volume at threshold floor to avoid
            # degenerate solutions that are well-calibrated but never bet.
            # Uses threshold_search[0] (not default_threshold) so the min_bets
            # check is consistent with the grid search floor — a model that only
            # produces bets below the floor will correctly be rejected here.
            base_threshold = self.config["threshold_search"][0]
            if self.use_odds_threshold and self.threshold_alpha > 0:
                thresholds = self.calculate_odds_adjusted_threshold(base_threshold, odds_arr)
                mask = (preds >= thresholds) & (odds_arr >= 1.5) & (odds_arr <= 6.0)
            else:
                mask = (preds >= base_threshold) & (odds_arr >= 1.5) & (odds_arr <= 6.0)

            n_bets = mask.sum()
            if n_bets < self.min_bets:
                return float("-inf")

            # Return negative log_loss (higher = better for Optuna maximize)
            return -ll

        return objective

    def run_hyperparameter_tuning(
        self,
        X: np.ndarray,
        y: np.ndarray,
        odds: np.ndarray,
        dates: Optional[np.ndarray] = None,
    ) -> Tuple[str, Dict[str, Any], float]:
        """Run Optuna hyperparameter tuning for all model types."""
        logger.info("Running hyperparameter tuning...")
        if self.use_sample_weights:
            logger.info(f"  Using time-decayed sample weights (decay_rate={self.sample_decay_rate:.4f})")
        if self.use_odds_threshold:
            logger.info(f"  Using odds-dependent thresholds (alpha={self.threshold_alpha:.2f})")

        best_overall = {"precision": float("-inf"), "model": None, "params": None}
        # Store all models' params for stacking ensemble
        self.all_model_params = {}

        for model_type in self._get_base_model_types(include_fastai=self.use_fastai, fast_mode=self.fast_mode, include_two_stage=self.use_two_stage):
            logger.info(f"  Tuning {model_type}...")

            study = optuna.create_study(
                direction="maximize",
                sampler=TPESampler(seed=self.seed),
            )

            if model_type == "catboost":
                n_trials_for_run = 50
            elif model_type == "fastai":
                n_trials_for_run = 20  # DL tuning is slower
            elif model_type.startswith("two_stage_"):
                n_trials_for_run = 30  # Two-stage fits 2 models per trial
            else:
                n_trials_for_run = self.n_optuna_trials

            if self.fast_mode:
                n_trials_for_run = min(n_trials_for_run, 5)

            objective = self.create_objective(X, y, odds, model_type, dates)

            study.optimize(
                objective,
                n_trials=n_trials_for_run,
                show_progress_bar=True
            )

            # Store params for each model (for stacking later)
            # For fastai, reconstruct list params from flat Optuna params
            best_params = dict(study.best_params)
            if model_type == "fastai":
                best_params["layers"] = [best_params.pop("layer1"), best_params.pop("layer2")]
                best_params["ps"] = [best_params.pop("ps1"), best_params.pop("ps2")]

            # Extract non-model params (tuned per trial but not passed to model constructor)
            best_cal_method = best_params.pop("calibration_method", "sigmoid")
            best_params.pop("decay_rate", None)
            best_params.pop("min_weight", None)
            best_params.pop("min_edge", None)  # two-stage param

            self.all_model_params[model_type] = best_params
            # Store per-model calibration method
            if not hasattr(self, '_model_cal_methods'):
                self._model_cal_methods = {}
            self._model_cal_methods[model_type] = best_cal_method

            if study.best_value > best_overall["precision"]:
                best_overall = {
                    "precision": study.best_value,
                    "model": model_type,
                    "params": best_params,
                    "calibration_method": best_cal_method,
                }

            logger.info(f"    {model_type}: log_loss={-study.best_value:.4f}, calibration={best_cal_method}")

        # Set calibration method from winning model for downstream use
        winning_cal = best_overall.get("calibration_method", "sigmoid")
        self._sklearn_cal_method = winning_cal
        self._use_custom_calibration = False  # Only sklearn methods in search space
        self.calibration_method = winning_cal

        logger.info(f"Best model: {best_overall['model']} (log_loss={-best_overall['precision']:.4f}, calibration={winning_cal})")
        return best_overall["model"], best_overall["params"], best_overall["precision"]

    def run_threshold_optimization(
        self,
        X: np.ndarray,
        y: np.ndarray,
        odds: np.ndarray,
        dates: Optional[np.ndarray] = None,
    ) -> Tuple[float, float, float, float, float, int, int]:
        """Run grid search over threshold and odds filters, including stacking ensemble.

        Note on model selection: The model selected here may differ from the
        walk-forward best. This is by design — threshold optimization selects
        on pooled OOS predictions (highest ROI with precision >= 0.60), while
        walk-forward evaluates per-fold average metrics. Different data
        aggregation methods can legitimately produce different winners.

        Uses held-out final fold for unbiased metric reporting:
        - Folds 0..N-2: pool OOS predictions for threshold grid search (optimization set)
        - Fold N-1: apply selected thresholds for unbiased reporting (held-out set)
        """
        from src.ml.metrics import sharpe_ratio, sortino_ratio, expected_calibration_error

        logger.info("Running threshold optimization (including stacking ensemble)...")
        logger.info(f"  Reserving final fold (fold {self.n_folds - 1}) as held-out reporting set")

        # Use stored dates if not provided
        if dates is None:
            dates = self.dates

        # Generate predictions for ALL models with walk-forward
        n_samples = len(y)
        fold_size = n_samples // (self.n_folds + 1)

        # Collect predictions separately for optimization and held-out folds
        _base_types = self._get_base_model_types(include_fastai=self.use_fastai, fast_mode=self.fast_mode, include_two_stage=self.use_two_stage)
        opt_preds = {name: [] for name in _base_types}
        opt_actuals = []
        opt_odds = []

        holdout_preds = {name: [] for name in _base_types}
        holdout_actuals = []
        holdout_odds = []

        # Track validation data for stacking meta-learner training
        val_preds = {name: [] for name in _base_types}
        val_actuals = []

        # Adversarial validation diagnostics
        adv_results = []

        # Per-league calibration tracking
        opt_leagues = []
        holdout_leagues = []

        # Uncertainty tracking (MAPIE conformal prediction)
        opt_uncertainties = []
        holdout_uncertainties = []

        n_opt_folds = self.n_folds - 1  # Folds 0..N-2 for optimization

        for fold in range(self.n_folds):
            train_end = (fold + 1) * fold_size
            test_start = train_end + self.temporal_buffer
            test_end = min(test_start + fold_size, n_samples)

            if test_start >= n_samples:
                continue

            X_train, y_train = X[:train_end], y[:train_end]
            X_test, y_test = X[test_start:test_end], y[test_start:test_end]
            odds_test = odds[test_start:test_end]

            # Calculate sample weights for training data
            sample_weights = None
            if self.use_sample_weights and dates is not None:
                train_dates = pd.to_datetime(dates[:train_end])
                sample_weights = self.calculate_sample_weights(train_dates)

            if len(X_train) < 100 or len(X_test) < 20:
                continue

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Adversarial validation: detect distribution shift between train and test
            try:
                adv_auc, shift_features = _adversarial_validation(
                    X_train_scaled, X_test_scaled, self.optimal_features
                )
                adv_results.append({"fold": fold, "auc": float(adv_auc)})
                logger.info(f"  Fold {fold} adversarial AUC: {adv_auc:.3f} (>0.6 = shift)")
                if adv_auc > 0.7:
                    logger.warning(f"  Significant distribution shift! Top features: {shift_features[:5]}")
            except Exception as e:
                logger.debug(f"  Adversarial validation failed: {e}")

            # Track league membership for per-league calibration
            is_holdout = (fold == self.n_folds - 1)
            if self.league_col is not None:
                test_leagues = self.league_col[test_start:test_end]
                if is_holdout:
                    holdout_leagues.extend(test_leagues)
                else:
                    opt_leagues.extend(test_leagues)

            # Use 20% of training data for meta-learner validation
            n_val = int(len(X_train) * 0.2)
            X_val_scaled = X_train_scaled[-n_val:]
            y_val = y_train[-n_val:]

            target_preds = holdout_preds if is_holdout else opt_preds

            # Train all base models and get predictions
            for model_type in _base_types:
                if model_type not in self.all_model_params:
                    continue

                try:
                    if model_type.startswith("two_stage_"):
                        # Two-stage models have non-standard API
                        from src.ml.two_stage_model import create_two_stage_model
                        base = "lightgbm" if model_type == "two_stage_lgb" else "catboost"
                        ts_model = create_two_stage_model(base)
                        odds_train = odds[:train_end]
                        ts_model.fit(X_train_scaled, y_train, odds_train)
                        result_dict = ts_model.predict_proba(X_test_scaled, odds_test)
                        probs = result_dict['combined_score']
                        target_preds[model_type].extend(probs)
                        if not is_holdout:
                            val_odds = odds[train_end - n_val:train_end]
                            val_result = ts_model.predict_proba(X_val_scaled, val_odds)
                            val_preds[model_type].extend(val_result['combined_score'])
                        continue
                    else:
                        model = self._create_model_instance(model_type, self.all_model_params[model_type], seed=self.seed)
                        cal_method = getattr(self, '_model_cal_methods', {}).get(model_type, self._sklearn_cal_method)
                        calibrated = CalibratedClassifierCV(model, method=cal_method, cv=3)
                        if sample_weights is not None:
                            calibrated.fit(X_train_scaled, y_train, sample_weight=sample_weights)
                        else:
                            calibrated.fit(X_train_scaled, y_train)
                        probs = calibrated.predict_proba(X_test_scaled)[:, 1]
                except Exception as e:
                    logger.warning(f"  {model_type} fold failed: {e}")
                    continue

                target_preds[model_type].extend(probs)

                # Also get validation predictions for stacking (only from opt folds)
                if not is_holdout:
                    val_probs = calibrated.predict_proba(X_val_scaled)[:, 1]
                    val_preds[model_type].extend(val_probs)

            # Collect uncertainty estimates via MAPIE (using best single model's calibrated output)
            best_single = self.best_model_type if self.best_model_type in self.all_model_params else None
            if best_single and best_single in self.all_model_params and not best_single.startswith("two_stage_"):
                try:
                    from src.ml.uncertainty import ConformalClassifier
                    # Use the calibrated model already trained above (re-train for MAPIE)
                    mapie_model = self._create_model_instance(
                        best_single, self.all_model_params[best_single], seed=self.seed
                    )
                    mapie_cal = CalibratedClassifierCV(mapie_model, method=self._sklearn_cal_method, cv=3)
                    if sample_weights is not None:
                        mapie_cal.fit(X_train_scaled, y_train, sample_weight=sample_weights)
                    else:
                        mapie_cal.fit(X_train_scaled, y_train)

                    conformal = ConformalClassifier(mapie_cal, alpha=0.1)
                    conformal.calibrate(X_val_scaled, y_val)
                    _, _, uncertainty = conformal.predict_with_uncertainty(X_test_scaled)

                    if is_holdout:
                        holdout_uncertainties.extend(uncertainty)
                    else:
                        opt_uncertainties.extend(uncertainty)
                except Exception as e:
                    logger.warning(f"  MAPIE uncertainty failed for fold {fold}: {e}")
                    # Fill with default 0.5 uncertainty
                    n_test = len(y_test)
                    if is_holdout:
                        holdout_uncertainties.extend([0.5] * n_test)
                    else:
                        opt_uncertainties.extend([0.5] * n_test)
            else:
                n_test = len(y_test)
                if is_holdout:
                    holdout_uncertainties.extend([0.5] * n_test)
                else:
                    opt_uncertainties.extend([0.5] * n_test)

            if is_holdout:
                holdout_actuals.extend(y_test)
                holdout_odds.extend(odds_test)
            else:
                opt_actuals.extend(y_test)
                opt_odds.extend(odds_test)
                val_actuals.extend(y_val)

        # Convert optimization set to arrays
        opt_actuals_arr = np.array(opt_actuals)
        opt_odds_arr = np.array(opt_odds)
        opt_uncertainties_arr = np.array(opt_uncertainties) if opt_uncertainties else None

        # Build ensemble predictions for optimization set
        base_model_names = [m for m in _base_types if len(opt_preds[m]) > 0]

        if len(base_model_names) >= 2:
            opt_stack = np.column_stack([np.array(opt_preds[m]) for m in base_model_names])
            val_stack = np.column_stack([np.array(val_preds[m]) for m in base_model_names])
            y_val_arr = np.array(val_actuals)

            opt_preds["average"] = np.mean(opt_stack, axis=1).tolist()

            # Stacking ensemble with Ridge meta-learner
            meta = None
            try:
                meta = RidgeClassifierCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0], cv=3)
                meta.fit(val_stack, y_val_arr)
                stacking_decision = np.atleast_1d(meta.decision_function(opt_stack))
                stacking_proba = 1 / (1 + np.exp(-stacking_decision))
                opt_preds["stacking"] = stacking_proba.tolist()
                coefs = np.atleast_2d(meta.coef_)[0]
                self._stacking_weights = dict(zip(base_model_names, coefs.tolist()))
                self._stacking_alpha = float(meta.alpha_)
                logger.info(f"  Stacking trained with weights: {self._stacking_weights}, alpha={self._stacking_alpha}")
            except Exception as e:
                logger.warning(f"  Stacking failed: {e}")
                opt_preds["stacking"] = opt_preds["average"]

            # DisagreementEnsemble: only bet when models agree with each other
            # AND disagree with the market. Test conservative/balanced/aggressive presets.
            market_probs = np.clip(1.0 / opt_odds_arr, 0.05, 0.95)
            for strategy in ['conservative', 'balanced', 'aggressive']:
                try:
                    # Build models list from individual predictions (no refit needed)
                    # Use a lightweight wrapper that returns pre-computed probabilities
                    class _PrecomputedModel:
                        def __init__(self, probs):
                            self._probs = probs
                        def predict_proba(self, X):
                            return np.column_stack([1 - self._probs, self._probs])

                    models = [_PrecomputedModel(np.array(opt_preds[m])) for m in base_model_names]
                    ensemble = create_disagreement_ensemble(models, base_model_names, strategy=strategy)
                    result = ensemble.predict_with_disagreement(
                        np.zeros((len(opt_odds_arr), 1)),  # X unused by precomputed models
                        market_probs,
                    )
                    opt_preds[f"disagree_{strategy}"] = result['avg_prob'].tolist()
                    # For non-signal samples, zero out probability to prevent betting
                    signal_probs = np.where(result['bet_signal'], result['avg_prob'], 0.0)
                    opt_preds[f"disagree_{strategy}_filtered"] = signal_probs.tolist()
                    n_signals = result['bet_signal'].sum()
                    logger.info(f"  Disagreement ({strategy}): {n_signals} bet signals "
                               f"({n_signals/len(opt_odds_arr)*100:.1f}%)")
                except Exception as e:
                    logger.warning(f"  Disagreement ({strategy}) failed: {e}")

            # Also keep simple agreement (backward compatible)
            opt_preds["agreement"] = np.min(opt_stack, axis=1).tolist()
            logger.info(f"  Agreement ensemble: uses minimum probability across {base_model_names}")
        else:
            meta = None
            logger.warning("  Not enough models for stacking, using best single model only")

        # Temporal blending: blend full-history and recent-only models
        # Only activate with sufficient training data (2000+ samples)
        if n_samples >= 2000 and self.best_model_type in self.all_model_params and not self.best_model_type.startswith("two_stage_"):
            try:
                blend_opt_preds = []
                blend_holdout_preds = []
                for fold in range(self.n_folds):
                    train_end = (fold + 1) * fold_size
                    test_start = train_end + self.temporal_buffer
                    test_end = min(test_start + fold_size, n_samples)
                    if test_start >= n_samples:
                        continue
                    X_train_f, y_train_f = X[:train_end], y[:train_end]
                    X_test_f = X[test_start:test_end]
                    if len(X_train_f) < 600 or len(X_test_f) < 20:
                        continue

                    scaler_b = StandardScaler()
                    X_train_scaled_b = scaler_b.fit_transform(X_train_f)
                    X_test_scaled_b = scaler_b.transform(X_test_f)

                    # Full-history model
                    model_full = self._create_model_instance(
                        self.best_model_type, self.all_model_params[self.best_model_type], seed=self.seed
                    )
                    cal_full = CalibratedClassifierCV(model_full, method=self._sklearn_cal_method, cv=3)
                    cal_full.fit(X_train_scaled_b, y_train_f)
                    probs_full = cal_full.predict_proba(X_test_scaled_b)[:, 1]

                    # Recent-only model (last 30% of training)
                    cutoff = int(len(X_train_f) * 0.7)
                    model_recent = self._create_model_instance(
                        self.best_model_type, self.all_model_params[self.best_model_type], seed=self.seed
                    )
                    cal_recent = CalibratedClassifierCV(model_recent, method=self._sklearn_cal_method, cv=3)
                    cal_recent.fit(X_train_scaled_b[cutoff:], y_train_f[cutoff:])
                    probs_recent = cal_recent.predict_proba(X_test_scaled_b)[:, 1]

                    # Blend with alpha=0.4 (slightly favoring recent data)
                    blend_alpha = 0.4
                    blended = blend_alpha * probs_recent + (1 - blend_alpha) * probs_full

                    is_holdout = (fold == self.n_folds - 1)
                    if is_holdout:
                        blend_holdout_preds.extend(blended)
                    else:
                        blend_opt_preds.extend(blended)

                if blend_opt_preds:
                    opt_preds["temporal_blend"] = blend_opt_preds
                    logger.info(f"  Temporal blend: {len(blend_opt_preds)} predictions (alpha=0.4)")
                if blend_holdout_preds:
                    holdout_preds["temporal_blend"] = blend_holdout_preds
            except Exception as e:
                logger.warning(f"  Temporal blending failed: {e}")

        # Log uncertainty collection status
        non_default_unc = sum(1 for u in opt_uncertainties if u != 0.5)
        if non_default_unc > 0:
            logger.info(f"  MAPIE uncertainty: {non_default_unc}/{len(opt_uncertainties)} predictions with real estimates")
        elif opt_uncertainties:
            logger.info(f"  MAPIE uncertainty: all {len(opt_uncertainties)} predictions used default (0.5)")

        # Grid search on OPTIMIZATION SET (folds 0..N-2)
        threshold_search = self.config["threshold_search"]
        # Per-market odds bounds (fall back to globals if not defined)
        min_odds_search = self.config.get("min_odds_search", MIN_ODDS_SEARCH)
        max_odds_search = self.config.get("max_odds_search", MAX_ODDS_SEARCH)
        configurations = list(product(threshold_search, min_odds_search, max_odds_search))

        _ensemble_methods = [
            "stacking", "average", "agreement",
            "disagree_conservative_filtered", "disagree_balanced_filtered",
            "disagree_aggressive_filtered",
            "temporal_blend",
        ]
        all_models = [m for m in _base_types + _ensemble_methods
                      if m in opt_preds and len(opt_preds[m]) > 0]

        logger.info(f"  Testing models: {all_models}")

        best_result = {"precision": 0.0, "roi": -100.0, "sharpe_roi": -100.0, "model": self.best_model_type}

        for model_name in all_models:
            preds = np.array(opt_preds[model_name])

            for threshold, min_odds, max_odds in configurations:
                mask = (preds >= threshold) & (opt_odds_arr >= min_odds) & (opt_odds_arr <= max_odds)
                n_bets = mask.sum()

                if n_bets < self.min_bets:
                    continue

                wins = opt_actuals_arr[mask].sum()
                precision = wins / n_bets

                returns = np.where(opt_actuals_arr[mask] == 1, opt_odds_arr[mask] - 1, -1)
                roi = returns.mean() * 100 if len(returns) > 0 else -100.0

                # Uncertainty-adjusted ROI (MAPIE): weight returns by confidence
                uncertainty_roi = roi  # Default: same as flat ROI
                if opt_uncertainties_arr is not None and len(opt_uncertainties_arr) == len(opt_actuals_arr):
                    from src.ml.uncertainty import batch_adjust_stakes
                    bet_uncertainties = opt_uncertainties_arr[mask]
                    stakes = batch_adjust_stakes(np.ones(n_bets), bet_uncertainties, uncertainty_penalty=1.0)
                    if stakes.sum() > 0:
                        uncertainty_roi = (returns * stakes).sum() / stakes.sum() * 100

                # Multi-objective: combine ROI with Sharpe ratio for risk-adjusted selection
                # sharpe_roi = ROI * min(1, sharpe / 1.5)
                # This penalizes high-ROI configs with high variance (ruin risk)
                sr = sharpe_ratio(returns)
                sharpe_mult = min(1.0, sr / 1.5) if sr > 0 else 0.0
                sharpe_roi = roi * sharpe_mult if roi > 0 else roi

                min_precision = 0.60  # Minimum viable precision floor
                if precision >= min_precision and (
                    sharpe_roi > best_result["sharpe_roi"] or
                    (sharpe_roi == best_result["sharpe_roi"] and precision > best_result["precision"])
                ):
                    best_result = {
                        "model": model_name,
                        "threshold": threshold,
                        "min_odds": min_odds,
                        "max_odds": max_odds,
                        "precision": precision,
                        "roi": roi,
                        "uncertainty_roi": uncertainty_roi,
                        "sharpe_roi": sharpe_roi,
                        "n_bets": int(n_bets),
                        "n_wins": int(wins),
                    }

        if best_result["precision"] == 0:
            logger.warning("No valid configuration found!")
            fallback_threshold = self.config["threshold_search"][0]
            return self.best_model_type, fallback_threshold, 2.0, 5.0, 0.0, -100.0, 0, 0

        # Update best model type if ensemble method won
        final_model = best_result.get("model", self.best_model_type)
        if final_model != self.best_model_type:
            logger.info(f"  Ensemble '{final_model}' outperformed individual models!")
            self.best_model_type = final_model

        uncertainty_roi_val = best_result.get('uncertainty_roi', best_result['roi'])
        unc_suffix = ""
        if abs(uncertainty_roi_val - best_result['roi']) > 0.1:
            unc_suffix = f", Uncertainty-adj ROI: {uncertainty_roi_val:.1f}%"
        logger.info(f"Optimization set - Best model: {final_model}, threshold: {best_result['threshold']}, "
                   f"precision: {best_result['precision']*100:.1f}%, "
                   f"ROI: {best_result['roi']:.1f}%, Sharpe-ROI: {best_result.get('sharpe_roi', 0):.1f}%{unc_suffix}")

        # --- HELD-OUT EVALUATION (fold N-1) for unbiased metrics ---
        holdout_actuals_arr = np.array(holdout_actuals)
        holdout_odds_arr = np.array(holdout_odds)

        # Build ensemble predictions for holdout set
        holdout_base_names = [m for m in _base_types if len(holdout_preds[m]) > 0]

        if len(holdout_base_names) >= 2:
            ho_stack = np.column_stack([np.array(holdout_preds[m]) for m in holdout_base_names])
            holdout_preds["average"] = np.mean(ho_stack, axis=1).tolist()

            if meta is not None:
                try:
                    ho_decision = np.atleast_1d(meta.decision_function(ho_stack))
                    holdout_preds["stacking"] = (1 / (1 + np.exp(-ho_decision))).tolist()
                except Exception:
                    holdout_preds["stacking"] = holdout_preds["average"]

            holdout_preds["agreement"] = np.min(ho_stack, axis=1).tolist()

            # DisagreementEnsemble for holdout set
            ho_market_probs = np.clip(1.0 / holdout_odds_arr, 0.05, 0.95)
            for strategy in ['conservative', 'balanced', 'aggressive']:
                try:
                    class _PrecomputedModel:
                        def __init__(self, probs):
                            self._probs = probs
                        def predict_proba(self, X):
                            return np.column_stack([1 - self._probs, self._probs])

                    models = [_PrecomputedModel(np.array(holdout_preds[m])) for m in holdout_base_names]
                    ensemble = create_disagreement_ensemble(models, holdout_base_names, strategy=strategy)
                    result = ensemble.predict_with_disagreement(
                        np.zeros((len(holdout_odds_arr), 1)),
                        ho_market_probs,
                    )
                    holdout_preds[f"disagree_{strategy}"] = result['avg_prob'].tolist()
                    signal_probs = np.where(result['bet_signal'], result['avg_prob'], 0.0)
                    holdout_preds[f"disagree_{strategy}_filtered"] = signal_probs.tolist()
                except Exception:
                    pass

        # Apply best thresholds to held-out fold
        if final_model in holdout_preds and len(holdout_preds[final_model]) > 0 and len(holdout_actuals_arr) > 0:
            ho_preds_arr = np.array(holdout_preds[final_model])
            ho_mask = (
                (ho_preds_arr >= best_result["threshold"]) &
                (holdout_odds_arr >= best_result["min_odds"]) &
                (holdout_odds_arr <= best_result["max_odds"])
            )
            ho_n_bets = ho_mask.sum()

            if ho_n_bets > 0:
                ho_wins = holdout_actuals_arr[ho_mask].sum()
                ho_precision = ho_wins / ho_n_bets
                ho_returns = np.where(holdout_actuals_arr[ho_mask] == 1, holdout_odds_arr[ho_mask] - 1, -1)
                ho_roi = ho_returns.mean() * 100

                ho_sharpe = sharpe_ratio(ho_returns)
                ho_sortino = sortino_ratio(ho_returns)
                ho_ece = expected_calibration_error(
                    holdout_actuals_arr, ho_preds_arr
                )

                logger.info(f"Held-out fold (UNBIASED) - {final_model}:")
                logger.info(f"  Precision: {ho_precision*100:.1f}% ({int(ho_wins)}/{ho_n_bets})")
                logger.info(f"  ROI: {ho_roi:.1f}%")
                logger.info(f"  Sharpe: {ho_sharpe:.3f}, Sortino: {ho_sortino:.3f}, ECE: {ho_ece:.4f}")

                # Store held-out metrics for downstream use
                self._holdout_metrics = {
                    "precision": float(ho_precision),
                    "roi": float(ho_roi),
                    "n_bets": int(ho_n_bets),
                    "n_wins": int(ho_wins),
                    "sharpe": float(ho_sharpe),
                    "sortino": float(ho_sortino),
                    "ece": float(ho_ece),
                }
            else:
                logger.info("Held-out fold: No qualifying bets with selected thresholds")
                self._holdout_metrics = {}
        else:
            logger.info("Held-out fold: No predictions available for selected model")
            self._holdout_metrics = {}

        # Per-league ECE calculation (on optimization set)
        self._per_league_ece = {}
        if opt_leagues and final_model in opt_preds and len(opt_preds[final_model]) > 0:
            from src.calibration.calibration import calibration_metrics
            opt_league_arr = np.array(opt_leagues)
            opt_preds_arr = np.array(opt_preds[final_model])
            for league in np.unique(opt_league_arr):
                mask = opt_league_arr == league
                if mask.sum() >= 30:
                    metrics = calibration_metrics(opt_actuals_arr[mask], opt_preds_arr[mask])
                    self._per_league_ece[str(league)] = float(metrics['ece'])
            if self._per_league_ece:
                logger.info(f"Per-league ECE: {self._per_league_ece}")

        # Store adversarial validation results
        self._adv_results = adv_results

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
            test_start = train_end + self.temporal_buffer
            test_end = min(test_start + fold_size, n_samples)

            if test_start >= n_samples or test_end <= test_start or (test_end - test_start) < 20:
                continue

            X_train, y_train = X[:train_end], y[:train_end]
            X_test, y_test = X[test_start:test_end], y[test_start:test_end]
            odds_test = odds[test_start:test_end]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train all base models
            fold_preds = {}
            for model_type in self._get_base_model_types(include_fastai=self.use_fastai, fast_mode=self.fast_mode, include_two_stage=self.use_two_stage):
                if model_type not in self.all_model_params:
                    continue

                try:
                    if model_type.startswith("two_stage_"):
                        from src.ml.two_stage_model import create_two_stage_model
                        base = "lightgbm" if model_type == "two_stage_lgb" else "catboost"
                        ts_model = create_two_stage_model(base)
                        odds_train = odds[:train_end]
                        ts_model.fit(X_train_scaled, y_train, odds_train)
                        result_dict = ts_model.predict_proba(X_test_scaled, odds_test)
                        fold_preds[model_type] = result_dict['combined_score']
                    else:
                        model = self._create_model_instance(model_type, self.all_model_params[model_type], seed=self.seed)
                        calibrated = CalibratedClassifierCV(model, method=self._sklearn_cal_method, cv=3)
                        calibrated.fit(X_train_scaled, y_train)
                        fold_preds[model_type] = calibrated.predict_proba(X_test_scaled)[:, 1]
                except Exception as e:
                    logger.warning(f"  {model_type} walkforward fold failed: {e}")
                    continue

            # Create ensemble predictions
            if len(fold_preds) >= 2:
                base_preds = np.column_stack(list(fold_preds.values()))
                fold_preds["stacking"] = np.mean(base_preds, axis=1)  # Simple average for fold
                fold_preds["average"] = np.mean(base_preds, axis=1)
                fold_preds["agreement"] = np.min(base_preds, axis=1)  # Min across models (conservative)

                # DisagreementEnsemble for walkforward
                wf_market_probs = np.clip(1.0 / odds_test, 0.05, 0.95)
                base_names = list(fold_preds.keys())[:len(fold_preds) - 3]  # Exclude stacking/average/agreement
                for strategy in ['conservative', 'balanced', 'aggressive']:
                    try:
                        class _PrecomputedModel:
                            def __init__(self, probs):
                                self._probs = probs
                            def predict_proba(self, X):
                                return np.column_stack([1 - self._probs, self._probs])

                        models = [_PrecomputedModel(fold_preds[m]) for m in base_names]
                        ensemble = create_disagreement_ensemble(models, base_names, strategy=strategy)
                        result = ensemble.predict_with_disagreement(
                            np.zeros((len(odds_test), 1)),
                            wf_market_probs,
                        )
                        signal_probs = np.where(result['bet_signal'], result['avg_prob'], 0.0)
                        fold_preds[f"disagree_{strategy}_filtered"] = signal_probs
                    except Exception:
                        pass

                # Temporal blend for walk-forward
                if len(X_train) >= 2000 and self.best_model_type in self.all_model_params and not self.best_model_type.startswith("two_stage_"):
                    try:
                        # Full-history model
                        model_full = self._create_model_instance(
                            self.best_model_type, self.all_model_params[self.best_model_type], seed=self.seed
                        )
                        cal_full = CalibratedClassifierCV(model_full, method=self._sklearn_cal_method, cv=3)
                        cal_full.fit(X_train_scaled, y_train)
                        probs_full = cal_full.predict_proba(X_test_scaled)[:, 1]

                        # Recent-only model (last 30% of training)
                        cutoff = int(len(X_train) * 0.7)
                        model_recent = self._create_model_instance(
                            self.best_model_type, self.all_model_params[self.best_model_type], seed=self.seed
                        )
                        cal_recent = CalibratedClassifierCV(model_recent, method=self._sklearn_cal_method, cv=3)
                        cal_recent.fit(X_train_scaled[cutoff:], y_train[cutoff:])
                        probs_recent = cal_recent.predict_proba(X_test_scaled)[:, 1]

                        fold_preds["temporal_blend"] = 0.4 * probs_recent + 0.6 * probs_full
                    except Exception:
                        pass

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

    @staticmethod
    def _safe_to_float(x):
        """Convert various formats to float, handling string-wrapped numbers."""
        if pd.isna(x):
            return np.nan
        if isinstance(x, (int, float, np.integer, np.floating)):
            return float(x)
        try:
            # Strip brackets from strings like '[3.9479554E-1]'
            s = str(x).strip('[]() ')
            return float(s)
        except (ValueError, TypeError):
            return np.nan

    def _convert_array_to_float(
        self,
        X: np.ndarray,
        feature_names: List[str],
    ) -> np.ndarray:
        """Convert array to float, handling string-wrapped numbers."""
        X_df = pd.DataFrame(X, columns=feature_names)
        for col in X_df.columns:
            X_df[col] = X_df[col].apply(self._safe_to_float)
        X_df = X_df.fillna(X_df.median())
        return X_df.astype(float).values

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
        X_train_raw, y_train = X[:n_train], y[:n_train]
        X_shap_raw = X[n_train:]

        # Convert to float for SHAP compatibility (handles string-wrapped numbers)
        X_train = self._convert_array_to_float(X_train_raw, feature_names)
        X_shap = self._convert_array_to_float(X_shap_raw, feature_names)

        # Train a LightGBM model (fast and SHAP-compatible)
        if "lightgbm" in self.all_model_params:
            params = {**self.all_model_params["lightgbm"], "random_state": self.seed, "verbose": -1}
        else:
            params = {"n_estimators": 100, "max_depth": 5, "random_state": self.seed, "verbose": -1}

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

    def validate_features_with_shap(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        threshold_pct: float = 0.01,
    ) -> Tuple[List[str], List[int], Dict[str, Any]]:
        """
        Validate features using SHAP and remove near-zero importance features.

        Args:
            model: Trained model (tree-based)
            X: Feature matrix
            y: Target values
            feature_names: List of feature names
            threshold_pct: Remove features below this percentage of max importance

        Returns:
            Tuple of (refined_feature_names, refined_indices, shap_results_dict)
        """
        if not SHAP_AVAILABLE:
            logger.info("SHAP not available, skipping feature validation")
            return feature_names, list(range(len(feature_names))), {}

        logger.info("Validating features with SHAP...")

        # Convert data for SHAP compatibility
        X_clean = self._convert_array_to_float(X, feature_names)

        # Get base model from calibrated wrapper if needed
        base_model = model
        if hasattr(model, 'estimator'):
            base_model = model.estimator
        elif hasattr(model, 'calibrated_classifiers_'):
            base_model = model.calibrated_classifiers_[0].estimator

        try:
            # Sample for speed (max 500 samples)
            n_samples = min(500, len(X_clean))
            X_sample = X_clean[:n_samples]

            explainer = shap.TreeExplainer(base_model)
            shap_values = explainer.shap_values(X_sample)

            # Handle binary classification (get positive class)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            # Calculate mean absolute SHAP value per feature
            mean_abs_shap = np.abs(shap_values).mean(axis=0)

            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': mean_abs_shap
            }).sort_values('importance', ascending=False)

            # Identify and remove low-importance features
            max_importance = feature_importance['importance'].max()
            threshold = max_importance * threshold_pct

            high_importance = feature_importance[feature_importance['importance'] >= threshold]
            low_importance = feature_importance[feature_importance['importance'] < threshold]

            refined_features = high_importance['feature'].tolist()
            removed_features = low_importance['feature'].tolist()

            # Get indices of refined features
            refined_indices = [feature_names.index(f) for f in refined_features
                              if f in feature_names]

            if removed_features:
                logger.info(f"Removing {len(removed_features)} low-importance features (<{threshold_pct*100:.1f}% of max):")
                for feat in removed_features[:10]:
                    logger.info(f"  - {feat}")
                if len(removed_features) > 10:
                    logger.info(f"  ... and {len(removed_features) - 10} more")

            shap_results = {
                'top_features': feature_importance.head(20).to_dict('records'),
                'removed_features': removed_features,
                'n_removed': len(removed_features),
                'n_kept': len(refined_features),
            }

            return refined_features, refined_indices, shap_results

        except Exception as e:
            logger.warning(f"SHAP validation failed: {e}")
            return feature_names, list(range(len(feature_names))), {'error': str(e)}

    def optimize(self) -> SniperResult:
        """Run full sniper optimization pipeline."""
        logger.info(f"\n{'='*60}")
        logger.info(f"SNIPER OPTIMIZATION: {self.bet_type.upper()}")
        logger.info(f"{'='*60}\n")

        # Log retail forecasting integration settings
        if self.use_sample_weights:
            logger.info(f"Sample weights: ENABLED (decay_rate={self.sample_decay_rate:.4f})")
        if self.use_odds_threshold:
            logger.info(f"Odds-dependent thresholds: ENABLED (alpha={self.threshold_alpha:.2f})")
        if self.filter_missing_odds:
            logger.info("Missing odds filtering: ENABLED")

        # Step 0: Load or optimize feature parameters
        self.feature_config = self.load_or_optimize_feature_config()
        if self.feature_config:
            logger.info(f"Using feature config: elo_k={self.feature_config.elo_k_factor}, "
                       f"form_window={self.feature_config.form_window}, "
                       f"ema_span={self.feature_config.ema_span}")

        # Load data (with potential feature regeneration)
        df = self.load_data_with_feature_config()
        self.features_df = df  # Store for later use in model training
        self.feature_columns = self.get_feature_columns(df)
        logger.info(f"Available features: {len(self.feature_columns)}")

        # Prepare data
        X = df[self.feature_columns].values
        X = np.nan_to_num(X, nan=0.0)
        # Ensure all values are numeric (handles string-wrapped floats like '[3.167E-1]')
        X = self._convert_array_to_float(X, self.feature_columns)
        y = self.prepare_target(df)

        # Store dates for sample weighting
        self.dates = df["date"].values

        # Preserve league column for per-league calibration (before feature dropping)
        if "league" in df.columns:
            self.league_col = df["league"].values
        else:
            self.league_col = None

        # Preserve odds columns for two-stage model fitting
        odds_preserve_cols = ["odds_home", "odds_away", "odds_draw"]
        self._preserved_odds = {}
        for oc in odds_preserve_cols:
            if oc in df.columns:
                self._preserved_odds[oc] = df[oc].values

        # Get odds
        odds_col = self.config["odds_col"]
        if odds_col in df.columns:
            odds = df[odds_col].values  # Don't fill NaN yet - we'll use them for filtering
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
            self.dates = self.dates[valid_mask]
            if self.league_col is not None:
                self.league_col = self.league_col[valid_mask]
            for oc in self._preserved_odds:
                self._preserved_odds[oc] = self._preserved_odds[oc][valid_mask]

        # Filter out rows with missing odds (retail forecasting: stockout-aware masking)
        # Only train on rows where we can actually calculate ROI
        if self.filter_missing_odds:
            odds_valid_mask = ~np.isnan(odds) & (odds > 1.0)
            n_missing_odds = (~odds_valid_mask).sum()
            if n_missing_odds > 0:
                logger.info(f"Filtering {n_missing_odds} rows with missing/invalid odds for training")
                X = X[odds_valid_mask]
                y = y[odds_valid_mask]
                odds = odds[odds_valid_mask]
                self.dates = self.dates[odds_valid_mask]
                if self.league_col is not None:
                    self.league_col = self.league_col[odds_valid_mask]
                for oc in self._preserved_odds:
                    self._preserved_odds[oc] = self._preserved_odds[oc][odds_valid_mask]

        # Fill any remaining NaN odds with default for evaluation
        odds = np.nan_to_num(odds, nan=3.0)

        logger.info(f"Training data: {len(X)} samples after filtering")

        # Step 1: RFE Feature Selection (with sample weights if enabled)
        rfe_weights = None
        if self.use_sample_weights and self.dates is not None:
            rfe_dates = pd.to_datetime(self.dates)
            rfe_weights = self.calculate_sample_weights(rfe_dates)
        selected_indices = self.run_rfe(X, y, sample_weights=rfe_weights)

        # Step 1b: Force-include cross-market interaction features (high CLV edge in R36)
        interaction_prefixes = ('btts_int_', 'goals_int_', 'fouls_int_')
        selected_set = set(selected_indices)
        forced_count = 0
        for i, col in enumerate(self.feature_columns):
            if col.startswith(interaction_prefixes) and i not in selected_set:
                selected_set.add(i)
                forced_count += 1
        if forced_count > 0:
            selected_indices = sorted(selected_set)
            logger.info(f"Force-included {forced_count} cross-market interaction features")

        # Step 1c: Remove highly correlated features (>0.95) to reduce redundancy
        X_temp = pd.DataFrame(X[:, selected_indices], columns=[self.feature_columns[i] for i in selected_indices])
        corr_matrix = X_temp.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        # Protect interaction features from correlation removal
        to_drop = []
        for col in upper.columns:
            if col.startswith(interaction_prefixes):
                continue
            if any(upper[col] > 0.95):
                to_drop.append(col)
        if to_drop:
            keep_cols = [c for c in X_temp.columns if c not in to_drop]
            selected_indices = [i for i in selected_indices if self.feature_columns[i] in keep_cols]
            logger.info(f"Removed {len(to_drop)} correlated features (r>0.95), {len(selected_indices)} remain")

        X_selected = X[:, selected_indices]
        self.optimal_features = [self.feature_columns[i] for i in selected_indices]

        # Step 2: Hyperparameter Tuning (with sample weights and dates)
        self.best_model_type, self.best_params, base_precision = self.run_hyperparameter_tuning(
            X_selected, y, odds, dates=self.dates
        )

        # Extract tuned sample weight params and update instance state
        if self.best_params is not None and self.use_sample_weights:
            if 'decay_rate' in self.best_params:
                self.sample_decay_rate = self.best_params.pop('decay_rate')
                logger.info(f"  Tuned decay_rate: {self.sample_decay_rate:.4f}")
            if 'min_weight' in self.best_params:
                self.sample_min_weight = self.best_params.pop('min_weight')
                logger.info(f"  Tuned min_weight: {self.sample_min_weight:.3f}")
            # Also clean from all_model_params so model constructors don't get them
            for mtype in self.all_model_params:
                self.all_model_params[mtype].pop('decay_rate', None)
                self.all_model_params[mtype].pop('min_weight', None)

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

        # Step 2.5: SHAP Feature Validation (remove low-importance features)
        shap_validation_results = {}
        if self.run_shap and SHAP_AVAILABLE:
            # Train best model for SHAP validation
            if self.best_model_type == "lightgbm":
                ModelClass = lgb.LGBMClassifier
                params = {**self.best_params, "random_state": self.seed, "verbose": -1}
            elif self.best_model_type == "catboost":
                ModelClass = CatBoostClassifier
                params = {**self.best_params, "random_seed": self.seed, "verbose": False}
            else:  # xgboost
                ModelClass = xgb.XGBClassifier
                params = {**self.best_params, "random_state": self.seed, "verbosity": 0}

            # Split data for training
            n_train = int(len(X_selected) * 0.8)
            X_train_shap = X_selected[:n_train]
            y_train_shap = y[:n_train]

            # Convert to float for SHAP compatibility
            X_train_clean = self._convert_array_to_float(X_train_shap, self.optimal_features)

            validation_model = ModelClass(**params)
            validation_model.fit(X_train_clean, y_train_shap)

            # Validate features with SHAP
            refined_features, refined_indices, shap_validation_results = self.validate_features_with_shap(
                validation_model, X_selected, y, self.optimal_features
            )

            # If features were removed, update feature set
            if shap_validation_results.get('n_removed', 0) > 0:
                logger.info(f"Refining features: {len(self.optimal_features)} -> {len(refined_features)}")
                X_selected = X_selected[:, refined_indices]
                self.optimal_features = refined_features

        # Step 3: Threshold Optimization (includes stacking/average/agreement ensembles)
        final_model, threshold, min_odds, max_odds, precision, roi, n_bets, n_wins = self.run_threshold_optimization(
            X_selected, y, odds, dates=self.dates
        )

        # Get params for final model (empty dict for ensemble methods)
        ensemble_methods = [
            "stacking", "average", "agreement",
            "disagree_conservative_filtered", "disagree_balanced_filtered",
            "disagree_aggressive_filtered",
            "temporal_blend",
        ]
        final_params = self.best_params if final_model not in ensemble_methods else {}
        if final_model in ensemble_methods:
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
            # Merge validation results if available
            if shap_validation_results:
                shap_results['validation'] = shap_validation_results

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
            # Retail forecasting integration params
            sample_decay_rate=self.sample_decay_rate if self.use_sample_weights else None,
            sample_min_weight=getattr(self, 'sample_min_weight', None) if self.use_sample_weights else None,
            threshold_alpha=self.threshold_alpha if self.use_odds_threshold else None,
            holdout_metrics=getattr(self, '_holdout_metrics', None),
            stacking_weights=getattr(self, '_stacking_weights', None),
            stacking_alpha=getattr(self, '_stacking_alpha', None),
            adversarial_validation={"folds": getattr(self, '_adv_results', [])},
            calibration_method=self.calibration_method,
            per_league_ece=getattr(self, '_per_league_ece', None),
        )

        return result

    def train_and_save_models(self, X: np.ndarray, y: np.ndarray) -> List[str]:
        """
        Train final calibrated models on full data and save them.

        Only saves the models needed for the winning strategy:
        - Individual model winner: saves only that model
        - Ensemble winner (agreement/stacking/average): saves all base models

        Uses joblib compression (level 3) to reduce file sizes.

        Returns list of saved model filenames.
        """
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        saved_models = []

        # Determine which models to save based on the winning strategy
        ensemble_methods = {
            "stacking", "average", "agreement",
            "disagree_conservative_filtered", "disagree_balanced_filtered",
            "disagree_aggressive_filtered",
            "temporal_blend",
        }
        if self.best_model_type in ensemble_methods:
            # Ensemble: need all base models
            models_to_save = [m for m in self._get_base_model_types(include_fastai=self.use_fastai, fast_mode=self.fast_mode, include_two_stage=self.use_two_stage)
                              if m in self.all_model_params]
            logger.info(f"  Ensemble winner ({self.best_model_type}): saving {len(models_to_save)} base models")
        else:
            # Individual model: save only the winner
            models_to_save = [self.best_model_type] if self.best_model_type in self.all_model_params else []
            logger.info(f"  Individual winner: saving only {self.best_model_type}")

        # Prepare scaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        for model_name in models_to_save:
            params = self.all_model_params.get(model_name, {})
            if not params:
                logger.info(f"  Skipping {model_name} - no params available")
                continue

            try:
                base_model = self._create_model_instance(model_name, params, seed=self.seed)

                cal_method = getattr(self, '_model_cal_methods', {}).get(model_name, self._sklearn_cal_method)
                calibrated = CalibratedClassifierCV(
                    base_model, method=cal_method, cv=3
                )
                calibrated.fit(X_scaled, y)
                model_data = {
                    "model": calibrated,
                    "features": self.optimal_features,
                    "bet_type": self.bet_type,
                    "scaler": scaler,
                    "calibration": cal_method,
                    "best_params": params,
                }

                model_path = MODELS_DIR / f"{self.bet_type}_{model_name}.joblib"
                joblib.dump(model_data, model_path, compress=3)
                size_mb = model_path.stat().st_size / (1024 * 1024)
                saved_models.append(model_path.name)
                logger.info(f"  Saved: {model_path} ({size_mb:.1f} MB)")

            except Exception as e:
                logger.warning(f"  Failed to save {model_name}: {e}")

        return saved_models


def save_models_to_hf(bet_types: List[str]) -> bool:
    """Upload saved models to Hugging Face Hub."""
    import os

    TOKEN = os.environ.get("HF_TOKEN")
    if not TOKEN:
        logger.warning("HF_TOKEN not set, skipping model upload")
        return False

    try:
        from huggingface_hub import HfApi
    except ImportError:
        logger.warning("huggingface_hub not installed, skipping upload")
        return False

    api = HfApi(token=TOKEN)
    repo_id = "czlowiekZplanety/bettip-data"

    uploaded = 0
    for bet_type in bet_types:
        for model_file in MODELS_DIR.glob(f"{bet_type}_*.joblib"):
            try:
                api.upload_file(
                    path_or_fileobj=str(model_file),
                    path_in_repo=f"models/{model_file.name}",
                    repo_id=repo_id,
                    repo_type="dataset",
                )
                logger.info(f"Uploaded: {model_file.name}")
                uploaded += 1
            except Exception as e:
                logger.warning(f"Failed to upload {model_file.name}: {e}")

    logger.info(f"Uploaded {uploaded} model files to HF Hub")
    return uploaded > 0


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
    ensemble_types = ['stacking', 'average', 'agreement']
    ensemble_count = sum(model_counts.get(e, 0) for e in ensemble_types)
    if ensemble_count > 0:
        print(f"   Ensemble methods won {ensemble_count}/{len(results)} bet types")

    # Tuned hyperparameters section
    print("\n" + "=" * 110)
    print("                           TUNED HYPERPARAMETERS")
    print("=" * 110)

    viable_with_params = [r for r in results if r.best_params and r.precision >= 0.55 and r.roi > 0]
    for r in sorted(viable_with_params, key=lambda x: x.roi, reverse=True):
        print(f"\n{r.bet_type} ({r.best_model}):")
        if r.best_model in ["stacking", "average", "agreement"]:
            print(f"  ensemble_type: {r.best_params.get('ensemble_type', 'N/A')}")
            print(f"  base_models: {r.best_params.get('base_models', [])}")
        else:
            for param, value in r.best_params.items():
                if isinstance(value, float):
                    print(f"  {param}: {value:.6g}")
                else:
                    print(f"  {param}: {value}")

    # Selected features section
    print("\n" + "=" * 110)
    print("                           SELECTED FEATURES (RFE)")
    print("=" * 110)

    viable_with_features = [r for r in results if r.optimal_features and r.precision >= 0.55 and r.roi > 0]

    # Show features per bet type
    for r in sorted(viable_with_features, key=lambda x: x.roi, reverse=True):
        print(f"\n{r.bet_type} ({r.n_features} features):")
        # Show top 10 features
        for i, feat in enumerate(r.optimal_features[:10], 1):
            print(f"  {i:2}. {feat}")
        if r.n_features > 10:
            print(f"  ... and {r.n_features - 10} more")

    # Feature overlap analysis - which features appear across multiple bet types
    if len(viable_with_features) >= 2:
        print("\n" + "-" * 110)
        print("FEATURE OVERLAP ANALYSIS (features important across multiple bet types):")
        print("-" * 110)

        feature_counts = {}
        for r in viable_with_features:
            for feat in r.optimal_features:
                feature_counts[feat] = feature_counts.get(feat, 0) + 1

        # Features appearing in 2+ bet types
        shared_features = [(f, c) for f, c in feature_counts.items() if c >= 2]
        shared_features.sort(key=lambda x: -x[1])

        if shared_features:
            print(f"\nFeatures appearing in multiple bet types ({len(shared_features)} total):")
            for feat, count in shared_features[:20]:
                pct = count / len(viable_with_features) * 100
                print(f"  {feat:<45} {count}/{len(viable_with_features)} bet types ({pct:.0f}%)")
        else:
            print("\nNo features shared across multiple bet types.")

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

    # Best configurations for each viable bet type (with hyperparameters)
    viable_results = [r for r in results if r.precision >= 0.55 and r.roi > 0]
    if viable_results:
        lines.append("\n## Best Configurations (with Tuned Hyperparameters)\n")
        for r in sorted(viable_results, key=lambda x: x.roi, reverse=True):
            lines.append(f"### {r.bet_type}\n")
            lines.append(f"```yaml")
            lines.append(f"# Strategy Configuration")
            lines.append(f"model: {r.best_model}")
            lines.append(f"threshold: {r.best_threshold}")
            lines.append(f"min_odds: {r.best_min_odds}")
            lines.append(f"max_odds: {r.best_max_odds}")
            lines.append(f"expected_precision: {r.precision*100:.1f}%")
            lines.append(f"expected_roi: {r.roi:+.1f}%")
            lines.append(f"n_features: {r.n_features}")
            lines.append(f"")
            lines.append(f"# Tuned Hyperparameters")
            if r.best_params:
                for param, value in r.best_params.items():
                    if isinstance(value, float):
                        lines.append(f"{param}: {value:.6g}")
                    elif isinstance(value, list):
                        lines.append(f"{param}: {value}")
                    else:
                        lines.append(f"{param}: {value}")
            lines.append(f"```\n")

            # Add top features for this bet type
            if r.optimal_features:
                lines.append(f"**Top 10 Features:**")
                for i, feat in enumerate(r.optimal_features[:10], 1):
                    lines.append(f"{i}. `{feat}`")
                lines.append("")

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
                       help="Target features after RFE (ignored if --auto-rfe)")
    parser.add_argument("--auto-rfe", action="store_true",
                       help="Use RFECV to automatically find optimal feature count")
    parser.add_argument("--min-rfe-features", type=int, default=20,
                       help="Minimum features for RFECV (only with --auto-rfe)")
    parser.add_argument("--max-rfe-features", type=int, default=80,
                       help="Maximum features for RFECV cap (prevents bloat, R36 used 38-48)")
    parser.add_argument("--n-optuna-trials", type=int, default=150,
                       help="Optuna trials per model")
    parser.add_argument("--min-bets", type=int, default=30,
                       help="Minimum bets for valid configuration")
    parser.add_argument("--walkforward", action="store_true",
                       help="Run walk-forward validation after optimization")
    parser.add_argument("--shap", action="store_true",
                       help="Run SHAP feature importance and interaction analysis")
    # Feature parameter options
    parser.add_argument("--feature-params", type=str, default=None,
                       help="Path to feature params YAML file (e.g., config/feature_params/away_win.yaml)")
    parser.add_argument("--optimize-features", action="store_true",
                       help="Run feature parameter optimization before model optimization")
    parser.add_argument("--n-feature-trials", type=int, default=50,
                       help="Optuna trials for feature parameter optimization")
    # Model saving options
    parser.add_argument("--save-models", action="store_true",
                       help="Train and save final calibrated models to models/ directory")
    parser.add_argument("--upload-models", action="store_true",
                       help="Upload saved models to HF Hub (requires HF_TOKEN)")
    # Retail forecasting integration options
    parser.add_argument("--sample-weights", action="store_true",
                       help="Use time-decayed sample weights during training (recent matches weighted higher)")
    parser.add_argument("--decay-rate", type=float, default=None,
                       help="Sample weight decay rate per day (default: ~0.002 for 1-year half-life)")
    parser.add_argument("--odds-threshold", action="store_true",
                       help="Use odds-dependent betting thresholds (newsvendor-inspired)")
    parser.add_argument("--threshold-alpha", type=float, default=0.2,
                       help="Odds-threshold adjustment strength (0=fixed, 1=full adjustment, default: 0.2)")
    parser.add_argument("--no-filter-missing-odds", action="store_true",
                       help="Disable filtering of rows with missing odds during training")
    parser.add_argument("--calibration-method", type=str, default="sigmoid",
                       choices=["sigmoid", "isotonic", "beta", "temperature"],
                       help="Initial calibration method (Optuna searches sigmoid/isotonic per model)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--fast", action="store_true",
                       help="Fast mode: LightGBM + XGBoost only, max 5 Optuna trials")
    parser.add_argument("--data", type=str, default=None,
                       help="Path to features parquet file (overrides default FEATURES_FILE)")
    parser.add_argument("--output-config", type=str, default=None,
                       help="Output deployment config path (default: config/sniper_deployment.json)")
    parser.add_argument("--league-group", type=str, default="",
                       help="League group namespace (e.g., 'americas'). Isolates feature params, models, and deployment config.")
    parser.add_argument("--markets", nargs="+", default=None,
                       help="Alias for --bet-type (e.g., --markets home_win shots fouls)")
    args = parser.parse_args()

    # --markets is an alias for --bet-type
    if args.markets and not args.bet_type:
        args.bet_type = args.markets

    # Override global FEATURES_FILE if --data is provided
    global FEATURES_FILE, MODELS_DIR
    if args.data:
        FEATURES_FILE = Path(args.data)

    # League group namespacing: isolate models and feature params
    league_group = args.league_group.strip()
    if league_group:
        MODELS_DIR = Path("models") / league_group
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Determine bet types to run
    if args.all:
        bet_types = list(BET_TYPES.keys())
    elif args.bet_type:
        bet_types = args.bet_type
    else:
        bet_types = ["away_win"]  # Default

    # Determine feature params mode
    feature_mode = "default"
    if args.feature_params:
        feature_mode = f"from file: {args.feature_params}"
    elif args.optimize_features:
        feature_mode = f"optimize ({args.n_feature_trials} trials)"

    # Determine retail forecasting mode
    retail_mode = []
    if args.sample_weights:
        decay_info = f"decay={args.decay_rate:.4f}" if args.decay_rate else "auto"
        retail_mode.append(f"sample_weights({decay_info})")
    if args.odds_threshold:
        retail_mode.append(f"odds_thresh(alpha={args.threshold_alpha})")
    if not args.no_filter_missing_odds:
        retail_mode.append("filter_missing_odds")
    retail_str = ", ".join(retail_mode) if retail_mode else "disabled"

    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              SNIPER MODE OPTIMIZATION PIPELINE                                ║
║                                                                              ║
║  High-precision betting configurations via:                                   ║
║  League Group: {league_group or 'default (European)':<46}        ║
║  0. Feature Params: {feature_mode:<42}            ║
║  1. RFE Feature Selection: {'RFECV (auto-optimal)' if args.auto_rfe else f'Fixed {args.n_rfe_features} features':<30}             ║
║  2. Optuna Hyperparameter Tuning (incl. Stacking Ensemble)                   ║
║  3. Threshold + Odds Filter Optimization                                     ║
║  4. Walk-Forward Validation: {'ENABLED' if args.walkforward else 'disabled'}                                     ║
║  5. SHAP Feature Analysis: {'ENABLED' if args.shap else 'disabled'}                                        ║
║  6. Save Models: {'ENABLED' if args.save_models else 'disabled'}                                               ║
║  7. Upload to HF Hub: {'ENABLED' if args.upload_models else 'disabled'}                                          ║
║  8. Retail Forecasting: {retail_str:<38}                     ║
║  9. Seed: {args.seed:<52}            ║
║  10. Fast Mode: {"ENABLED (LGB+XGB, 5 trials)" if args.fast else "disabled":<40}              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

    results = []

    for bet_type in bet_types:
        if bet_type not in BET_TYPES:
            logger.warning(f"Unknown bet type: {bet_type}, skipping")
            continue

        # Resolve feature params dir for league group namespacing
        feature_params_dir = Path("config/feature_params") / league_group if league_group else None

        optimizer = SniperOptimizer(
            bet_type=bet_type,
            n_folds=args.n_folds,
            n_rfe_features=args.n_rfe_features,
            auto_rfe=args.auto_rfe,
            min_rfe_features=args.min_rfe_features,
            max_rfe_features=args.max_rfe_features,
            n_optuna_trials=args.n_optuna_trials,
            min_bets=args.min_bets,
            run_walkforward=args.walkforward,
            run_shap=args.shap,
            feature_params_path=args.feature_params,
            optimize_features=args.optimize_features,
            n_feature_trials=args.n_feature_trials,
            feature_params_dir=feature_params_dir,
            # Retail forecasting integration
            use_sample_weights=args.sample_weights,
            sample_decay_rate=args.decay_rate,
            use_odds_threshold=args.odds_threshold,
            threshold_alpha=args.threshold_alpha,
            filter_missing_odds=not args.no_filter_missing_odds,
            calibration_method=args.calibration_method,
            seed=args.seed,
            fast_mode=args.fast,
        )

        result = optimizer.optimize()

        # Train and save models if requested
        if args.save_models and result.precision > 0.5 and result.n_bets > 0:
            if optimizer.features_df is None or optimizer.feature_columns is None:
                logger.warning(f"Cannot save models for {bet_type}: features_df or feature_columns not set")
            else:
                logger.info(f"\nTraining final models for {bet_type}...")
                # Get data for training
                df = optimizer.features_df
                X = df[optimizer.feature_columns].values
                X = np.nan_to_num(X, nan=0.0)
                y = optimizer.prepare_target(df)
                valid_mask = ~np.isnan(y)
                X = X[valid_mask]
                y = y[valid_mask]
                # Select optimal features
                selected_indices = [optimizer.feature_columns.index(f) for f in optimizer.optimal_features
                                   if f in optimizer.feature_columns]
                X_selected = X[:, selected_indices]
                saved_models = optimizer.train_and_save_models(X_selected, y)
                result = SniperResult(
                    **{**asdict(result), 'saved_models': saved_models}
                )

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

    # Upload models to HF Hub if requested
    if args.upload_models:
        logger.info("\nUploading models to HF Hub...")
        save_models_to_hf(bet_types)


if __name__ == "__main__":
    main()
