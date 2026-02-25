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
import os
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from optuna.samplers import TPESampler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import RFE, RFECV
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Beta calibration for post-hoc probability recalibration
from src.calibration.calibration import BetaCalibrator

# Feature parameter optimization
from src.features.config_manager import BetTypeFeatureConfig
from src.features.regeneration import FeatureRegenerator

# Enhanced CatBoost wrapper for transfer learning and baseline injection
from src.ml.catboost_wrapper import EnhancedCatBoost

# Disagreement ensemble for high-confidence betting
from src.ml.ensemble_disagreement import DisagreementEnsemble, create_disagreement_ensemble

# Sample weighting (retail forecasting integration)
from src.ml.sample_weighting import (
    calculate_time_decay_weights,
    decay_rate_from_half_life,
    get_recommended_decay_rate,
)

# SHAP for feature importance analysis
try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not available. Install with: pip install shap")

# Deep learning models (optional)
try:
    # Check if actual fastai library is installed (not just our wrapper class)
    import fastai.tabular.all  # noqa: F401

    from src.ml.models import FastAITabularModel

    FASTAI_AVAILABLE = True
except ImportError:
    FASTAI_AVAILABLE = False
    try:
        from src.ml.models import FastAITabularModel  # noqa: F811
    except ImportError:
        pass

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _get_calibration_cv(model_type: str, n_splits: int = 3):
    """Get appropriate CV strategy for CalibratedClassifierCV.

    CatBoost with has_time=True requires temporal ordering — use TimeSeriesSplit.
    Other models use the default StratifiedKFold (integer n_splits).
    """
    if model_type == "catboost":
        return TimeSeriesSplit(n_splits=n_splits)
    return n_splits


def _numpy_serializer(obj):
    """JSON serializer for numpy types (used when saving model params)."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
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
        # R36 selected 0.85; floor lowered to 0.65 for real odds (R86: fake odds inflated ROI, real odds need more volume)
        "threshold_search": [0.65, 0.70, 0.75, 0.80, 0.85],
        # Over 2.5 odds typically 1.5-2.2 range; R53-56 failed with min_odds=2.0
        "min_odds_search": [1.4, 1.5, 1.6, 1.8],
        "max_odds_search": [2.5, 3.0, 3.5],
    },
    "under25": {
        "target": "under25",
        "odds_col": "avg_under25_close",
        "approach": "classification",
        "default_threshold": 0.55,
        # R90 selected 0.65 (HIT FLOOR); extended to 0.60. R47-49 garbage was pre-odds-threshold.
        "threshold_search": [0.60, 0.65, 0.70, 0.75, 0.80],
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
        "odds_col": "theodds_shots_over_odds",  # Will use fallback (no real odds for total shots)
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
        "odds_col": "theodds_corners_over_odds",  # No bulk historical odds; uses fallback
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
        "odds_col": "theodds_cards_over_odds",  # No bulk historical odds; uses fallback
        "approach": "regression_line",
        "default_threshold": 0.50,  # Lower threshold for ~37% base rate
        "threshold_search": [0.40, 0.45, 0.50, 0.55, 0.60],
        # Cards odds typically 1.7-2.3 range
        "min_odds_search": [1.2, 1.4, 1.6, 1.8],
        "max_odds_search": [2.5, 3.0, 3.5],
    },
    # --- Niche market line variants ---
    # Cards variants (1.5-6.5, step 1.0)
    "cards_over_15": {
        "target": "total_cards",
        "target_line": 1.5,
        "odds_col": "theodds_cards_over_odds",
        "approach": "regression_line",
        "default_threshold": 0.50,
        "threshold_search": [0.40, 0.45, 0.50, 0.55, 0.60],
        "min_odds_search": [1.2, 1.4, 1.6, 1.8],
        "max_odds_search": [2.5, 3.0, 3.5],
    },
    "cards_over_25": {
        "target": "total_cards",
        "target_line": 2.5,
        "odds_col": "theodds_cards_over_odds",
        "approach": "regression_line",
        "default_threshold": 0.50,
        "threshold_search": [0.40, 0.45, 0.50, 0.55, 0.60],
        "min_odds_search": [1.2, 1.4, 1.6, 1.8],
        "max_odds_search": [2.5, 3.0, 3.5],
    },
    "cards_over_35": {
        "target": "total_cards",
        "target_line": 3.5,
        "odds_col": "theodds_cards_over_odds",
        "approach": "regression_line",
        "default_threshold": 0.50,
        "threshold_search": [0.40, 0.45, 0.50, 0.55, 0.60],
        "min_odds_search": [1.2, 1.4, 1.6, 1.8],
        "max_odds_search": [2.5, 3.0, 3.5],
    },
    "cards_over_45": {
        "target": "total_cards",
        "target_line": 4.5,
        "odds_col": "theodds_cards_over_odds",
        "approach": "regression_line",
        "default_threshold": 0.50,
        "threshold_search": [0.40, 0.45, 0.50, 0.55, 0.60],
        "min_odds_search": [1.2, 1.4, 1.6, 1.8],
        "max_odds_search": [2.5, 3.0, 3.5],
    },
    "cards_over_55": {
        "target": "total_cards",
        "target_line": 5.5,
        "odds_col": "theodds_cards_over_odds",
        "approach": "regression_line",
        "default_threshold": 0.50,
        "threshold_search": [0.40, 0.45, 0.50, 0.55, 0.60],
        "min_odds_search": [1.2, 1.4, 1.6, 1.8],
        "max_odds_search": [2.5, 3.0, 3.5],
    },
    "cards_over_65": {
        "target": "total_cards",
        "target_line": 6.5,
        "odds_col": "theodds_cards_over_odds",
        "approach": "regression_line",
        "default_threshold": 0.50,
        "threshold_search": [0.40, 0.45, 0.50, 0.55, 0.60],
        "min_odds_search": [1.2, 1.4, 1.6, 1.8],
        "max_odds_search": [2.5, 3.0, 3.5],
    },
    # Corners variants (8.5-11.5, step 1.0)
    "corners_over_85": {
        "target": "total_corners",
        "target_line": 8.5,
        "odds_col": "theodds_corners_over_odds",
        "approach": "regression_line",
        "default_threshold": 0.50,
        "threshold_search": [0.40, 0.45, 0.50, 0.55, 0.60],
        "min_odds_search": [1.2, 1.4, 1.6, 1.8],
        "max_odds_search": [2.5, 3.0, 3.5],
    },
    "corners_over_95": {
        "target": "total_corners",
        "target_line": 9.5,
        "odds_col": "theodds_corners_over_odds",
        "approach": "regression_line",
        "default_threshold": 0.50,
        "threshold_search": [0.40, 0.45, 0.50, 0.55, 0.60],
        "min_odds_search": [1.2, 1.4, 1.6, 1.8],
        "max_odds_search": [2.5, 3.0, 3.5],
    },
    "corners_over_105": {
        "target": "total_corners",
        "target_line": 10.5,
        "odds_col": "theodds_corners_over_odds",
        "approach": "regression_line",
        "default_threshold": 0.50,
        "threshold_search": [0.40, 0.45, 0.50, 0.55, 0.60],
        "min_odds_search": [1.2, 1.4, 1.6, 1.8],
        "max_odds_search": [2.5, 3.0, 3.5],
    },
    "corners_over_115": {
        "target": "total_corners",
        "target_line": 11.5,
        "odds_col": "theodds_corners_over_odds",
        "approach": "regression_line",
        "default_threshold": 0.50,
        "threshold_search": [0.40, 0.45, 0.50, 0.55, 0.60],
        "min_odds_search": [1.2, 1.4, 1.6, 1.8],
        "max_odds_search": [2.5, 3.0, 3.5],
    },
    # Shots variants (24.5-29.5, step 1.0)
    "shots_over_245": {
        "target": "total_shots",
        "target_line": 24.5,
        "odds_col": "theodds_shots_over_odds",
        "approach": "regression_line",
        "default_threshold": 0.55,
        "threshold_search": [0.55, 0.60, 0.65, 0.70, 0.75],
        "min_odds_search": [1.2, 1.4, 1.6, 1.8],
        "max_odds_search": [2.5, 3.0, 3.5],
    },
    "shots_over_255": {
        "target": "total_shots",
        "target_line": 25.5,
        "odds_col": "theodds_shots_over_odds",
        "approach": "regression_line",
        "default_threshold": 0.55,
        "threshold_search": [0.55, 0.60, 0.65, 0.70, 0.75],
        "min_odds_search": [1.2, 1.4, 1.6, 1.8],
        "max_odds_search": [2.5, 3.0, 3.5],
    },
    "shots_over_265": {
        "target": "total_shots",
        "target_line": 26.5,
        "odds_col": "theodds_shots_over_odds",
        "approach": "regression_line",
        "default_threshold": 0.55,
        "threshold_search": [0.55, 0.60, 0.65, 0.70, 0.75],
        "min_odds_search": [1.2, 1.4, 1.6, 1.8],
        "max_odds_search": [2.5, 3.0, 3.5],
    },
    "shots_over_275": {
        "target": "total_shots",
        "target_line": 27.5,
        "odds_col": "theodds_shots_over_odds",
        "approach": "regression_line",
        "default_threshold": 0.55,
        "threshold_search": [0.55, 0.60, 0.65, 0.70, 0.75],
        "min_odds_search": [1.2, 1.4, 1.6, 1.8],
        "max_odds_search": [2.5, 3.0, 3.5],
    },
    "shots_over_285": {
        "target": "total_shots",
        "target_line": 28.5,
        "odds_col": "theodds_shots_over_odds",
        "approach": "regression_line",
        "default_threshold": 0.55,
        "threshold_search": [0.55, 0.60, 0.65, 0.70, 0.75],
        "min_odds_search": [1.2, 1.4, 1.6, 1.8],
        "max_odds_search": [2.5, 3.0, 3.5],
    },
    "shots_over_295": {
        "target": "total_shots",
        "target_line": 29.5,
        "odds_col": "theodds_shots_over_odds",
        "approach": "regression_line",
        "default_threshold": 0.55,
        "threshold_search": [0.55, 0.60, 0.65, 0.70, 0.75],
        "min_odds_search": [1.2, 1.4, 1.6, 1.8],
        "max_odds_search": [2.5, 3.0, 3.5],
    },
    # Fouls variants (22.5-27.5, step 1.0)
    "fouls_over_225": {
        "target": "total_fouls",
        "target_line": 22.5,
        "odds_col": "fouls_over_odds",
        "approach": "regression_line",
        "default_threshold": 0.60,
        "threshold_search": [0.55, 0.60, 0.65, 0.70, 0.75],
        "min_odds_search": [1.2, 1.4, 1.6, 1.8],
        "max_odds_search": [2.5, 3.0, 3.5],
    },
    "fouls_over_235": {
        "target": "total_fouls",
        "target_line": 23.5,
        "odds_col": "fouls_over_odds",
        "approach": "regression_line",
        "default_threshold": 0.60,
        "threshold_search": [0.55, 0.60, 0.65, 0.70, 0.75],
        "min_odds_search": [1.2, 1.4, 1.6, 1.8],
        "max_odds_search": [2.5, 3.0, 3.5],
    },
    "fouls_over_245": {
        "target": "total_fouls",
        "target_line": 24.5,
        "odds_col": "fouls_over_odds",
        "approach": "regression_line",
        "default_threshold": 0.60,
        "threshold_search": [0.55, 0.60, 0.65, 0.70, 0.75],
        "min_odds_search": [1.2, 1.4, 1.6, 1.8],
        "max_odds_search": [2.5, 3.0, 3.5],
    },
    "fouls_over_255": {
        "target": "total_fouls",
        "target_line": 25.5,
        "odds_col": "fouls_over_odds",
        "approach": "regression_line",
        "default_threshold": 0.60,
        "threshold_search": [0.55, 0.60, 0.65, 0.70, 0.75],
        "min_odds_search": [1.2, 1.4, 1.6, 1.8],
        "max_odds_search": [2.5, 3.0, 3.5],
    },
    "fouls_over_265": {
        "target": "total_fouls",
        "target_line": 26.5,
        "odds_col": "fouls_over_odds",
        "approach": "regression_line",
        "default_threshold": 0.60,
        "threshold_search": [0.55, 0.60, 0.65, 0.70, 0.75],
        "min_odds_search": [1.2, 1.4, 1.6, 1.8],
        "max_odds_search": [2.5, 3.0, 3.5],
    },
    "fouls_over_275": {
        "target": "total_fouls",
        "target_line": 27.5,
        "odds_col": "fouls_over_odds",
        "approach": "regression_line",
        "default_threshold": 0.60,
        "threshold_search": [0.55, 0.60, 0.65, 0.70, 0.75],
        "min_odds_search": [1.2, 1.4, 1.6, 1.8],
        "max_odds_search": [2.5, 3.0, 3.5],
    },
    # --- UNDER line variants (direction="under" flips target to total < line) ---
    # Shots UNDER (24.5-29.5)
    "shots_under_245": {
        "target": "total_shots",
        "target_line": 24.5,
        "direction": "under",
        "odds_col": "shots_under_odds",
        "approach": "regression_line",
        "default_threshold": 0.55,
        "threshold_search": [0.55, 0.60, 0.65, 0.70, 0.75],
        "min_odds_search": [1.2, 1.4, 1.6, 1.8],
        "max_odds_search": [2.5, 3.0, 3.5],
    },
    "shots_under_255": {
        "target": "total_shots",
        "target_line": 25.5,
        "direction": "under",
        "odds_col": "shots_under_odds",
        "approach": "regression_line",
        "default_threshold": 0.55,
        "threshold_search": [0.55, 0.60, 0.65, 0.70, 0.75],
        "min_odds_search": [1.2, 1.4, 1.6, 1.8],
        "max_odds_search": [2.5, 3.0, 3.5],
    },
    "shots_under_265": {
        "target": "total_shots",
        "target_line": 26.5,
        "direction": "under",
        "odds_col": "shots_under_odds",
        "approach": "regression_line",
        "default_threshold": 0.55,
        "threshold_search": [0.55, 0.60, 0.65, 0.70, 0.75],
        "min_odds_search": [1.2, 1.4, 1.6, 1.8],
        "max_odds_search": [2.5, 3.0, 3.5],
    },
    "shots_under_275": {
        "target": "total_shots",
        "target_line": 27.5,
        "direction": "under",
        "odds_col": "shots_under_odds",
        "approach": "regression_line",
        "default_threshold": 0.55,
        "threshold_search": [0.55, 0.60, 0.65, 0.70, 0.75],
        "min_odds_search": [1.2, 1.4, 1.6, 1.8],
        "max_odds_search": [2.5, 3.0, 3.5],
    },
    "shots_under_285": {
        "target": "total_shots",
        "target_line": 28.5,
        "direction": "under",
        "odds_col": "shots_under_odds",
        "approach": "regression_line",
        "default_threshold": 0.55,
        "threshold_search": [0.55, 0.60, 0.65, 0.70, 0.75],
        "min_odds_search": [1.2, 1.4, 1.6, 1.8],
        "max_odds_search": [2.5, 3.0, 3.5],
    },
    "shots_under_295": {
        "target": "total_shots",
        "target_line": 29.5,
        "direction": "under",
        "odds_col": "shots_under_odds",
        "approach": "regression_line",
        "default_threshold": 0.55,
        "threshold_search": [0.55, 0.60, 0.65, 0.70, 0.75],
        "min_odds_search": [1.2, 1.4, 1.6, 1.8],
        "max_odds_search": [2.5, 3.0, 3.5],
    },
    # Fouls UNDER (22.5-27.5)
    "fouls_under_225": {
        "target": "total_fouls",
        "target_line": 22.5,
        "direction": "under",
        "odds_col": "fouls_under_odds",
        "approach": "regression_line",
        "default_threshold": 0.60,
        "threshold_search": [0.55, 0.60, 0.65, 0.70, 0.75],
        "min_odds_search": [1.2, 1.4, 1.6, 1.8],
        "max_odds_search": [2.5, 3.0, 3.5],
    },
    "fouls_under_235": {
        "target": "total_fouls",
        "target_line": 23.5,
        "direction": "under",
        "odds_col": "fouls_under_odds",
        "approach": "regression_line",
        "default_threshold": 0.60,
        "threshold_search": [0.55, 0.60, 0.65, 0.70, 0.75],
        "min_odds_search": [1.2, 1.4, 1.6, 1.8],
        "max_odds_search": [2.5, 3.0, 3.5],
    },
    "fouls_under_245": {
        "target": "total_fouls",
        "target_line": 24.5,
        "direction": "under",
        "odds_col": "fouls_under_odds",
        "approach": "regression_line",
        "default_threshold": 0.60,
        "threshold_search": [0.55, 0.60, 0.65, 0.70, 0.75],
        "min_odds_search": [1.2, 1.4, 1.6, 1.8],
        "max_odds_search": [2.5, 3.0, 3.5],
    },
    "fouls_under_255": {
        "target": "total_fouls",
        "target_line": 25.5,
        "direction": "under",
        "odds_col": "fouls_under_odds",
        "approach": "regression_line",
        "default_threshold": 0.60,
        "threshold_search": [0.55, 0.60, 0.65, 0.70, 0.75],
        "min_odds_search": [1.2, 1.4, 1.6, 1.8],
        "max_odds_search": [2.5, 3.0, 3.5],
    },
    "fouls_under_265": {
        "target": "total_fouls",
        "target_line": 26.5,
        "direction": "under",
        "odds_col": "fouls_under_odds",
        "approach": "regression_line",
        "default_threshold": 0.60,
        "threshold_search": [0.55, 0.60, 0.65, 0.70, 0.75],
        "min_odds_search": [1.2, 1.4, 1.6, 1.8],
        "max_odds_search": [2.5, 3.0, 3.5],
    },
    "fouls_under_275": {
        "target": "total_fouls",
        "target_line": 27.5,
        "direction": "under",
        "odds_col": "fouls_under_odds",
        "approach": "regression_line",
        "default_threshold": 0.60,
        "threshold_search": [0.55, 0.60, 0.65, 0.70, 0.75],
        "min_odds_search": [1.2, 1.4, 1.6, 1.8],
        "max_odds_search": [2.5, 3.0, 3.5],
    },
    # Cards UNDER (1.5-6.5)
    "cards_under_15": {
        "target": "total_cards",
        "target_line": 1.5,
        "direction": "under",
        "odds_col": "cards_under_odds",
        "approach": "regression_line",
        "default_threshold": 0.50,
        "threshold_search": [0.50, 0.55, 0.60, 0.65, 0.70],
        "min_odds_search": [1.2, 1.4, 1.6, 1.8],
        "max_odds_search": [2.5, 3.0, 3.5],
    },
    "cards_under_25": {
        "target": "total_cards",
        "target_line": 2.5,
        "direction": "under",
        "odds_col": "cards_under_odds",
        "approach": "regression_line",
        "default_threshold": 0.50,
        "threshold_search": [0.50, 0.55, 0.60, 0.65, 0.70],
        "min_odds_search": [1.2, 1.4, 1.6, 1.8],
        "max_odds_search": [2.5, 3.0, 3.5],
    },
    "cards_under_35": {
        "target": "total_cards",
        "target_line": 3.5,
        "direction": "under",
        "odds_col": "cards_under_odds",
        "approach": "regression_line",
        "default_threshold": 0.50,
        "threshold_search": [0.50, 0.55, 0.60, 0.65, 0.70],
        "min_odds_search": [1.2, 1.4, 1.6, 1.8],
        "max_odds_search": [2.5, 3.0, 3.5],
    },
    "cards_under_45": {
        "target": "total_cards",
        "target_line": 4.5,
        "direction": "under",
        "odds_col": "cards_under_odds",
        "approach": "regression_line",
        "default_threshold": 0.50,
        "threshold_search": [0.50, 0.55, 0.60, 0.65, 0.70],
        "min_odds_search": [1.2, 1.4, 1.6, 1.8],
        "max_odds_search": [2.5, 3.0, 3.5],
    },
    "cards_under_55": {
        "target": "total_cards",
        "target_line": 5.5,
        "direction": "under",
        "odds_col": "cards_under_odds",
        "approach": "regression_line",
        "default_threshold": 0.50,
        "threshold_search": [0.50, 0.55, 0.60, 0.65, 0.70],
        "min_odds_search": [1.2, 1.4, 1.6, 1.8],
        "max_odds_search": [2.5, 3.0, 3.5],
    },
    "cards_under_65": {
        "target": "total_cards",
        "target_line": 6.5,
        "direction": "under",
        "odds_col": "cards_under_odds",
        "approach": "regression_line",
        "default_threshold": 0.50,
        "threshold_search": [0.50, 0.55, 0.60, 0.65, 0.70],
        "min_odds_search": [1.2, 1.4, 1.6, 1.8],
        "max_odds_search": [2.5, 3.0, 3.5],
    },
    # Corners UNDER (8.5-11.5)
    "corners_under_85": {
        "target": "total_corners",
        "target_line": 8.5,
        "direction": "under",
        "odds_col": "corners_under_odds",
        "approach": "regression_line",
        "default_threshold": 0.50,
        "threshold_search": [0.50, 0.55, 0.60, 0.65, 0.70],
        "min_odds_search": [1.2, 1.4, 1.6, 1.8],
        "max_odds_search": [2.5, 3.0, 3.5],
    },
    "corners_under_95": {
        "target": "total_corners",
        "target_line": 9.5,
        "direction": "under",
        "odds_col": "corners_under_odds",
        "approach": "regression_line",
        "default_threshold": 0.50,
        "threshold_search": [0.50, 0.55, 0.60, 0.65, 0.70],
        "min_odds_search": [1.2, 1.4, 1.6, 1.8],
        "max_odds_search": [2.5, 3.0, 3.5],
    },
    "corners_under_105": {
        "target": "total_corners",
        "target_line": 10.5,
        "direction": "under",
        "odds_col": "corners_under_odds",
        "approach": "regression_line",
        "default_threshold": 0.50,
        "threshold_search": [0.50, 0.55, 0.60, 0.65, 0.70],
        "min_odds_search": [1.2, 1.4, 1.6, 1.8],
        "max_odds_search": [2.5, 3.0, 3.5],
    },
    "corners_under_115": {
        "target": "total_corners",
        "target_line": 11.5,
        "direction": "under",
        "odds_col": "corners_under_odds",
        "approach": "regression_line",
        "default_threshold": 0.50,
        "threshold_search": [0.50, 0.55, 0.60, 0.65, 0.70],
        "min_odds_search": [1.2, 1.4, 1.6, 1.8],
        "max_odds_search": [2.5, 3.0, 3.5],
    },
    # Goals variants (1.5-3.5, step 1.0)
    "goals_over_15": {
        "target": "total_goals",
        "target_line": 1.5,
        "odds_col": "goals_over_odds",
        "approach": "regression_line",
        "default_threshold": 0.50,
        "threshold_search": [0.50, 0.55, 0.60, 0.65, 0.70, 0.75],
        "min_odds_search": [1.05, 1.10, 1.15, 1.20],
        "max_odds_search": [1.5, 1.8, 2.0],
    },
    "goals_over_25": {
        "target": "total_goals",
        "target_line": 2.5,
        "odds_col": "goals_over_odds",
        "approach": "regression_line",
        "default_threshold": 0.55,
        "threshold_search": [0.55, 0.60, 0.65, 0.70, 0.75],
        "min_odds_search": [1.4, 1.6, 1.8, 2.0],
        "max_odds_search": [2.5, 3.0, 3.5],
    },
    "goals_over_35": {
        "target": "total_goals",
        "target_line": 3.5,
        "odds_col": "goals_over_odds",
        "approach": "regression_line",
        "default_threshold": 0.50,
        "threshold_search": [0.50, 0.55, 0.60, 0.65, 0.70],
        "min_odds_search": [1.8, 2.0, 2.2, 2.5],
        "max_odds_search": [3.0, 3.5, 4.0],
    },
    # Goals UNDER (1.5-3.5)
    "goals_under_15": {
        "target": "total_goals",
        "target_line": 1.5,
        "direction": "under",
        "odds_col": "goals_under_odds",
        "approach": "regression_line",
        "default_threshold": 0.50,
        "threshold_search": [0.50, 0.55, 0.60, 0.65, 0.70],
        "min_odds_search": [1.8, 2.0, 2.5, 3.0],
        "max_odds_search": [4.0, 5.0, 6.0],
    },
    "goals_under_25": {
        "target": "total_goals",
        "target_line": 2.5,
        "direction": "under",
        "odds_col": "goals_under_odds",
        "approach": "regression_line",
        "default_threshold": 0.55,
        "threshold_search": [0.55, 0.60, 0.65, 0.70, 0.75],
        "min_odds_search": [1.4, 1.6, 1.8, 2.0],
        "max_odds_search": [2.5, 3.0, 3.5],
    },
    "goals_under_35": {
        "target": "total_goals",
        "target_line": 3.5,
        "direction": "under",
        "odds_col": "goals_under_odds",
        "approach": "regression_line",
        "default_threshold": 0.50,
        "threshold_search": [0.50, 0.55, 0.60, 0.65, 0.70, 0.75],
        "min_odds_search": [1.05, 1.10, 1.15, 1.20],
        "max_odds_search": [1.5, 1.8, 2.0],
    },
    # Home goals variants (0.5-2.5)
    "hgoals_over_05": {
        "target": "home_goals",
        "target_line": 0.5,
        "odds_col": "hgoals_over_odds",
        "approach": "regression_line",
        "default_threshold": 0.50,
        "threshold_search": [0.50, 0.55, 0.60, 0.65, 0.70, 0.75],
        "min_odds_search": [1.05, 1.10, 1.15, 1.20],
        "max_odds_search": [1.5, 1.8, 2.0],
    },
    "hgoals_over_15": {
        "target": "home_goals",
        "target_line": 1.5,
        "odds_col": "hgoals_over_odds",
        "approach": "regression_line",
        "default_threshold": 0.50,
        "threshold_search": [0.50, 0.55, 0.60, 0.65, 0.70],
        "min_odds_search": [1.4, 1.6, 1.8, 2.0],
        "max_odds_search": [2.5, 3.0, 3.5],
    },
    "hgoals_over_25": {
        "target": "home_goals",
        "target_line": 2.5,
        "odds_col": "hgoals_over_odds",
        "approach": "regression_line",
        "default_threshold": 0.50,
        "threshold_search": [0.50, 0.55, 0.60, 0.65, 0.70],
        "min_odds_search": [1.8, 2.0, 2.5, 3.0],
        "max_odds_search": [3.5, 4.0, 5.0],
    },
    "hgoals_under_05": {
        "target": "home_goals",
        "target_line": 0.5,
        "direction": "under",
        "odds_col": "hgoals_under_odds",
        "approach": "regression_line",
        "default_threshold": 0.50,
        "threshold_search": [0.50, 0.55, 0.60, 0.65, 0.70],
        "min_odds_search": [1.8, 2.0, 2.5, 3.0],
        "max_odds_search": [4.0, 5.0, 6.0],
    },
    "hgoals_under_15": {
        "target": "home_goals",
        "target_line": 1.5,
        "direction": "under",
        "odds_col": "hgoals_under_odds",
        "approach": "regression_line",
        "default_threshold": 0.50,
        "threshold_search": [0.50, 0.55, 0.60, 0.65, 0.70],
        "min_odds_search": [1.2, 1.4, 1.6, 1.8],
        "max_odds_search": [2.5, 3.0, 3.5],
    },
    "hgoals_under_25": {
        "target": "home_goals",
        "target_line": 2.5,
        "direction": "under",
        "odds_col": "hgoals_under_odds",
        "approach": "regression_line",
        "default_threshold": 0.50,
        "threshold_search": [0.50, 0.55, 0.60, 0.65, 0.70, 0.75],
        "min_odds_search": [1.05, 1.10, 1.15, 1.20],
        "max_odds_search": [1.5, 1.8, 2.0],
    },
    # Away goals variants (0.5-2.5)
    "agoals_over_05": {
        "target": "away_goals",
        "target_line": 0.5,
        "odds_col": "agoals_over_odds",
        "approach": "regression_line",
        "default_threshold": 0.50,
        "threshold_search": [0.50, 0.55, 0.60, 0.65, 0.70, 0.75],
        "min_odds_search": [1.05, 1.10, 1.15, 1.20],
        "max_odds_search": [1.5, 1.8, 2.0],
    },
    "agoals_over_15": {
        "target": "away_goals",
        "target_line": 1.5,
        "odds_col": "agoals_over_odds",
        "approach": "regression_line",
        "default_threshold": 0.50,
        "threshold_search": [0.50, 0.55, 0.60, 0.65, 0.70],
        "min_odds_search": [1.4, 1.6, 1.8, 2.0],
        "max_odds_search": [2.5, 3.0, 3.5],
    },
    "agoals_over_25": {
        "target": "away_goals",
        "target_line": 2.5,
        "odds_col": "agoals_over_odds",
        "approach": "regression_line",
        "default_threshold": 0.50,
        "threshold_search": [0.50, 0.55, 0.60, 0.65, 0.70],
        "min_odds_search": [1.8, 2.0, 2.5, 3.0],
        "max_odds_search": [3.5, 4.0, 5.0],
    },
    "agoals_under_05": {
        "target": "away_goals",
        "target_line": 0.5,
        "direction": "under",
        "odds_col": "agoals_under_odds",
        "approach": "regression_line",
        "default_threshold": 0.50,
        "threshold_search": [0.50, 0.55, 0.60, 0.65, 0.70],
        "min_odds_search": [1.4, 1.6, 1.8, 2.0],
        "max_odds_search": [2.5, 3.0, 3.5],
    },
    "agoals_under_15": {
        "target": "away_goals",
        "target_line": 1.5,
        "direction": "under",
        "odds_col": "agoals_under_odds",
        "approach": "regression_line",
        "default_threshold": 0.50,
        "threshold_search": [0.50, 0.55, 0.60, 0.65, 0.70],
        "min_odds_search": [1.2, 1.4, 1.6, 1.8],
        "max_odds_search": [2.5, 3.0, 3.5],
    },
    "agoals_under_25": {
        "target": "away_goals",
        "target_line": 2.5,
        "direction": "under",
        "odds_col": "agoals_under_odds",
        "approach": "regression_line",
        "default_threshold": 0.50,
        "threshold_search": [0.50, 0.55, 0.60, 0.65, 0.70, 0.75],
        "min_odds_search": [1.05, 1.10, 1.15, 1.20],
        "max_odds_search": [1.5, 1.8, 2.0],
    },
    # Corners handicap variants (0.5-2.5)
    "cornershc_over_05": {
        "target": "corner_diff",
        "target_line": 0.5,
        "odds_col": "cornershc_over_odds",
        "approach": "regression_line",
        "default_threshold": 0.50,
        "threshold_search": [0.50, 0.55, 0.60, 0.65, 0.70],
        "min_odds_search": [1.4, 1.6, 1.8, 2.0],
        "max_odds_search": [2.5, 3.0, 3.5],
    },
    "cornershc_over_15": {
        "target": "corner_diff",
        "target_line": 1.5,
        "odds_col": "cornershc_over_odds",
        "approach": "regression_line",
        "default_threshold": 0.50,
        "threshold_search": [0.50, 0.55, 0.60, 0.65, 0.70],
        "min_odds_search": [1.6, 1.8, 2.0, 2.2],
        "max_odds_search": [3.0, 3.5, 4.0],
    },
    "cornershc_over_25": {
        "target": "corner_diff",
        "target_line": 2.5,
        "odds_col": "cornershc_over_odds",
        "approach": "regression_line",
        "default_threshold": 0.50,
        "threshold_search": [0.50, 0.55, 0.60, 0.65, 0.70],
        "min_odds_search": [1.8, 2.0, 2.5, 3.0],
        "max_odds_search": [3.5, 4.0, 5.0],
    },
    "cornershc_under_05": {
        "target": "corner_diff",
        "target_line": 0.5,
        "direction": "under",
        "odds_col": "cornershc_under_odds",
        "approach": "regression_line",
        "default_threshold": 0.50,
        "threshold_search": [0.50, 0.55, 0.60, 0.65, 0.70],
        "min_odds_search": [1.4, 1.6, 1.8, 2.0],
        "max_odds_search": [2.5, 3.0, 3.5],
    },
    "cornershc_under_15": {
        "target": "corner_diff",
        "target_line": 1.5,
        "direction": "under",
        "odds_col": "cornershc_under_odds",
        "approach": "regression_line",
        "default_threshold": 0.50,
        "threshold_search": [0.50, 0.55, 0.60, 0.65, 0.70],
        "min_odds_search": [1.2, 1.4, 1.6, 1.8],
        "max_odds_search": [2.5, 3.0, 3.5],
    },
    "cornershc_under_25": {
        "target": "corner_diff",
        "target_line": 2.5,
        "direction": "under",
        "odds_col": "cornershc_under_odds",
        "approach": "regression_line",
        "default_threshold": 0.50,
        "threshold_search": [0.50, 0.55, 0.60, 0.65, 0.70, 0.75],
        "min_odds_search": [1.05, 1.10, 1.15, 1.20],
        "max_odds_search": [1.5, 1.8, 2.0],
    },
    # Cards handicap variants (0.5 only — single bookmaker)
    "cardshc_over_05": {
        "target": "card_diff",
        "target_line": 0.5,
        "odds_col": "cardshc_over_odds",
        "approach": "regression_line",
        "default_threshold": 0.50,
        "threshold_search": [0.50, 0.55, 0.60, 0.65, 0.70],
        "min_odds_search": [1.4, 1.6, 1.8, 2.0],
        "max_odds_search": [2.5, 3.0, 3.5],
    },
    "cardshc_under_05": {
        "target": "card_diff",
        "target_line": 0.5,
        "direction": "under",
        "odds_col": "cardshc_under_odds",
        "approach": "regression_line",
        "default_threshold": 0.50,
        "threshold_search": [0.50, 0.55, 0.60, 0.65, 0.70],
        "min_odds_search": [1.4, 1.6, 1.8, 2.0],
        "max_odds_search": [2.5, 3.0, 3.5],
    },
    # --- Half-time markets ---
    # HT 1X2 (classification)
    "home_win_h1": {
        "target": "home_win_h1",
        "odds_col": "h2h_h1_home_avg",
        "approach": "classification",
        "default_threshold": 0.55,
        "threshold_search": [0.55, 0.60, 0.65, 0.70, 0.75, 0.80],
        "min_odds_search": [1.4, 1.6, 1.8, 2.0, 2.5],
        "max_odds_search": [4.0, 5.0, 6.0, 8.0],
    },
    "away_win_h1": {
        "target": "away_win_h1",
        "odds_col": "h2h_h1_away_avg",
        "approach": "classification",
        "default_threshold": 0.50,
        "threshold_search": [0.50, 0.55, 0.60, 0.65, 0.70, 0.75],
        "min_odds_search": [1.6, 1.8, 2.0, 2.5, 3.0],
        "max_odds_search": [5.0, 6.0, 8.0, 10.0],
    },
    # HT Totals (regression_line on ht_total_goals)
    "ht_over_05": {
        "target": "ht_total_goals",
        "target_line": 0.5,
        "odds_col": "totals_h1_over_odds",
        "approach": "regression_line",
        "default_threshold": 0.50,
        "threshold_search": [0.50, 0.55, 0.60, 0.65, 0.70, 0.75],
        "min_odds_search": [1.05, 1.10, 1.15, 1.20],
        "max_odds_search": [1.5, 1.8, 2.0],
    },
    "ht_over_15": {
        "target": "ht_total_goals",
        "target_line": 1.5,
        "odds_col": "totals_h1_over_odds",
        "approach": "regression_line",
        "default_threshold": 0.50,
        "threshold_search": [0.50, 0.55, 0.60, 0.65, 0.70],
        "min_odds_search": [1.8, 2.0, 2.2, 2.5],
        "max_odds_search": [3.0, 3.5, 4.0],
    },
    "ht_under_05": {
        "target": "ht_total_goals",
        "target_line": 0.5,
        "direction": "under",
        "odds_col": "totals_h1_under_odds",
        "approach": "regression_line",
        "default_threshold": 0.50,
        "threshold_search": [0.50, 0.55, 0.60, 0.65, 0.70],
        "min_odds_search": [1.8, 2.0, 2.5, 3.0],
        "max_odds_search": [4.0, 5.0, 6.0],
    },
    "ht_under_15": {
        "target": "ht_total_goals",
        "target_line": 1.5,
        "direction": "under",
        "odds_col": "totals_h1_under_odds",
        "approach": "regression_line",
        "default_threshold": 0.50,
        "threshold_search": [0.50, 0.55, 0.60, 0.65, 0.70, 0.75],
        "min_odds_search": [1.05, 1.10, 1.15, 1.20],
        "max_odds_search": [1.5, 1.8, 2.0],
    },
}

# Maps line variants to their base market for feature params sharing
BASE_MARKET_MAP = {
    # Cards (1.5-6.5)
    "cards_over_15": "cards",
    "cards_over_25": "cards",
    "cards_over_35": "cards",
    "cards_over_45": "cards",
    "cards_over_55": "cards",
    "cards_over_65": "cards",
    "cards_under_15": "cards",
    "cards_under_25": "cards",
    "cards_under_35": "cards",
    "cards_under_45": "cards",
    "cards_under_55": "cards",
    "cards_under_65": "cards",
    # Corners (8.5-11.5)
    "corners_over_85": "corners",
    "corners_over_95": "corners",
    "corners_over_105": "corners",
    "corners_over_115": "corners",
    "corners_under_85": "corners",
    "corners_under_95": "corners",
    "corners_under_105": "corners",
    "corners_under_115": "corners",
    # Shots (24.5-29.5)
    "shots_over_245": "shots",
    "shots_over_255": "shots",
    "shots_over_265": "shots",
    "shots_over_275": "shots",
    "shots_over_285": "shots",
    "shots_over_295": "shots",
    "shots_under_245": "shots",
    "shots_under_255": "shots",
    "shots_under_265": "shots",
    "shots_under_275": "shots",
    "shots_under_285": "shots",
    "shots_under_295": "shots",
    # Fouls (22.5-27.5)
    "fouls_over_225": "fouls",
    "fouls_over_235": "fouls",
    "fouls_over_245": "fouls",
    "fouls_over_255": "fouls",
    "fouls_over_265": "fouls",
    "fouls_over_275": "fouls",
    "fouls_under_225": "fouls",
    "fouls_under_235": "fouls",
    "fouls_under_245": "fouls",
    "fouls_under_255": "fouls",
    "fouls_under_265": "fouls",
    "fouls_under_275": "fouls",
    # Goals (1.5-3.5)
    "goals_over_15": "goals",
    "goals_over_25": "goals",
    "goals_over_35": "goals",
    "goals_under_15": "goals",
    "goals_under_25": "goals",
    "goals_under_35": "goals",
    # Home goals (0.5-2.5)
    "hgoals_over_05": "hgoals", "hgoals_over_15": "hgoals", "hgoals_over_25": "hgoals",
    "hgoals_under_05": "hgoals", "hgoals_under_15": "hgoals", "hgoals_under_25": "hgoals",
    # Away goals (0.5-2.5)
    "agoals_over_05": "agoals", "agoals_over_15": "agoals", "agoals_over_25": "agoals",
    "agoals_under_05": "agoals", "agoals_under_15": "agoals", "agoals_under_25": "agoals",
    # Corners handicap (0.5-2.5)
    "cornershc_over_05": "cornershc", "cornershc_over_15": "cornershc", "cornershc_over_25": "cornershc",
    "cornershc_under_05": "cornershc", "cornershc_under_15": "cornershc", "cornershc_under_25": "cornershc",
    # Cards handicap (0.5)
    "cardshc_over_05": "cardshc", "cardshc_under_05": "cardshc",
    # Half-time totals (0.5-1.5)
    "ht_over_05": "ht", "ht_over_15": "ht",
    "ht_under_05": "ht", "ht_under_15": "ht",
}

# Exclude columns (data leakage prevention)
EXCLUDE_COLUMNS = [
    # Identifiers
    "fixture_id",
    "date",
    "home_team_id",
    "home_team_name",
    "away_team_id",
    "away_team_name",
    "round",
    "season",
    "league",
    "sm_fixture_id",  # SportMonks fixture ID
    # Target variables (match outcomes)
    "home_win",
    "draw",
    "away_win",
    "match_result",
    "result",
    "total_goals",
    "goal_difference",
    "xg_diff",
    "home_goals",
    "away_goals",
    "ft_home",
    "ft_away",  # Cleaner-renamed full-time scores (== home_goals/away_goals)
    "btts",
    "under25",
    "over25",
    "under35",
    "over35",
    # Match statistics (not available pre-match - these are outcomes!)
    "home_shots",
    "away_shots",
    "home_shots_on_target",
    "away_shots_on_target",
    "home_corners",
    "away_corners",
    "total_corners",
    "home_fouls",
    "away_fouls",
    "total_fouls",
    "home_yellows",
    "away_yellows",
    "home_reds",
    "away_reds",
    "total_yellows",
    "total_reds",  # Aggregate card counts (match outcome)
    "home_yellow_cards",
    "away_yellow_cards",  # Alternate naming from events pipeline
    "home_red_cards",
    "away_red_cards",  # Alternate naming from events pipeline
    "home_possession",
    "away_possession",
    "total_cards",
    "total_shots",
    "total_shots_on_target",  # Match outcome - was causing cards leakage
    "home_cards",
    "away_cards",  # Match outcome cards (not historical)
    "corner_diff",
    "card_diff",  # Handicap targets (match outcomes)
    "ht_home",
    "ht_away",
    "ht_total_goals",  # Half-time scores (match outcomes)
    "home_win_h1",
    "away_win_h1",  # Derived HT result targets
    # Match-level synonyms of target columns (S39 leakage discovery)
    # These exist in HF Hub parquet but encode match outcomes exactly
    "away_goals_conceded",  # == home_goals (corr=1.0)
    "home_goals_conceded",  # == away_goals (corr=1.0)
    "home_goals_per_shot",  # == home_goals / home_shots (corr=1.0)
    "away_goals_per_shot",  # == away_goals / away_shots (corr=1.0)
    "home_points",  # Standings points (post-match update)
    "away_points",  # Standings points (post-match update)
    # API-Football detailed match-level shot breakdown (post-match only)
    "home_shots_insidebox",
    "away_shots_insidebox",
    "home_shots_outsidebox",
    "away_shots_outsidebox",
    "home_blocked_shots",
    "away_blocked_shots",
    "home_shots_off_goal",
    "away_shots_off_goal",
    # API-Football other match-level stats (post-match only)
    "home_goalkeeper_saves",
    "away_goalkeeper_saves",
    "home_offsides",
    "away_offsides",
    "home_passes_total",
    "away_passes_total",
    "home_passes_accurate",
    "away_passes_accurate",
    "home_passes_%",
    "away_passes_%",
    "home_expected_goals",
    "away_expected_goals",
    # Pandas merge-suffix variants of target columns (leak through _x/_y renaming)
    "total_corners_x",
    "total_corners_y",
    "total_fouls_x",
    "total_fouls_y",
    "total_cards_x",
    "total_cards_y",
    "total_shots_x",
    "total_shots_y",
    "total_goals_x",
    "total_goals_y",
    "home_goals_x",
    "home_goals_y",
    "away_goals_x",
    "away_goals_y",
    # S39: merge-suffix variants of goal synonym columns
    "away_goals_conceded_x",
    "away_goals_conceded_y",
    "home_goals_conceded_x",
    "home_goals_conceded_y",
    "home_goals_per_shot_x",
    "home_goals_per_shot_y",
    "away_goals_per_shot_x",
    "away_goals_per_shot_y",
    "home_points_x",
    "home_points_y",
    "away_points_x",
    "away_points_y",
]

# Per-bet-type low-importance feature exclusions (R33 SHAP analysis).
# Only excludes features that are low-importance AND NOT in that bet type's top-20.
# Features important for specific markets are preserved where they matter.
LOW_IMPORTANCE_EXCLUSIONS: Dict[str, List[str]] = {
    "away_win": [
        "away_corners_won_ema",
        "away_first_half_rate",
        "discipline_diff",
        "expected_home_corners",
        "h2h_away_wins",
        "home_cards_ema",
        "home_corners_won_ema",
        "home_corners_won_roll_10",
        "home_corners_won_roll_5",
        "home_importance",
        "home_points_last_n",
        "home_pts_to_cl",
        "home_shot_accuracy",
        "home_shots_conceded_ema",
        "home_shots_ema_x",
        "home_unbeaten_streak",
        "importance_diff",
        "match_importance",
        "away_corners_won_roll_10",
        "away_shots_ema_y",
    ],
    "btts": [
        "away_corners_conceded_ema",
        "away_corners_won_ema",
        "away_corners_won_roll_10",
        "away_first_half_rate",
        "away_shots_ema_x",
        "away_shots_ema_y",
        "discipline_diff",
        "expected_home_corners",
        "h2h_away_wins",
        "home_cards_ema",
        "home_corners_won_ema",
        "home_corners_won_roll_10",
        "home_corners_won_roll_5",
        "home_importance",
        "home_points_last_n",
        "home_pts_to_cl",
        "home_shot_accuracy",
        "home_shots_conceded_ema",
        "home_shots_ema_x",
        "home_unbeaten_streak",
        "importance_diff",
        "match_importance",
    ],
    "cards": [
        "away_corners_conceded_ema",
        "away_corners_won_ema",
        "away_corners_won_roll_10",
        "away_first_half_rate",
        "away_shots_ema_x",
        "expected_home_corners",
        "h2h_away_wins",
        "home_cards_ema",
        "home_corners_won_ema",
        "home_corners_won_roll_10",
        "home_corners_won_roll_5",
        "home_importance",
        "home_points_last_n",
        "home_pts_to_cl",
        "home_shot_accuracy",
        "home_shots_conceded_ema",
        "home_shots_ema_x",
        "home_unbeaten_streak",
        "importance_diff",
        "match_importance",
    ],
    "corners": [
        "away_corners_conceded_ema",
        "away_corners_won_ema",
        "away_corners_won_roll_10",
        "away_first_half_rate",
        "away_shots_ema_x",
        "away_shots_ema_y",
        "discipline_diff",
        "expected_home_corners",
        "h2h_away_wins",
        "home_corners_won_ema",
        "home_corners_won_roll_10",
        "home_corners_won_roll_5",
        "home_importance",
        "home_points_last_n",
        "home_shot_accuracy",
        "home_shots_conceded_ema",
        "home_unbeaten_streak",
        "importance_diff",
        "match_importance",
    ],
    "fouls": [
        "away_corners_conceded_ema",
        "away_corners_won_ema",
        "away_corners_won_roll_10",
        "away_first_half_rate",
        "away_shots_ema_x",
        "away_shots_ema_y",
        "discipline_diff",
        "expected_home_corners",
        "h2h_away_wins",
        "home_cards_ema",
        "home_corners_won_ema",
        "home_corners_won_roll_10",
        "home_corners_won_roll_5",
        "home_importance",
        "home_points_last_n",
        "home_pts_to_cl",
        "home_shot_accuracy",
        "home_shots_conceded_ema",
        "home_shots_ema_x",
        "home_unbeaten_streak",
        "importance_diff",
    ],
    "home_win": [
        "away_corners_won_ema",
        "away_first_half_rate",
        "discipline_diff",
        "expected_home_corners",
        "h2h_away_wins",
        "home_cards_ema",
        "home_corners_won_ema",
        "home_corners_won_roll_10",
        "home_corners_won_roll_5",
        "home_importance",
        "home_points_last_n",
        "home_pts_to_cl",
        "home_shot_accuracy",
        "home_shots_conceded_ema",
        "home_shots_ema_x",
        "home_unbeaten_streak",
        "importance_diff",
        "match_importance",
        "away_corners_conceded_ema",
        "away_corners_won_roll_10",
        "away_shots_ema_y",
    ],
    "over25": [
        "away_corners_conceded_ema",
        "away_corners_won_ema",
        "away_corners_won_roll_10",
        "away_first_half_rate",
        "away_shots_ema_x",
        "away_shots_ema_y",
        "discipline_diff",
        "expected_home_corners",
        "h2h_away_wins",
        "home_cards_ema",
        "home_corners_won_ema",
        "home_corners_won_roll_10",
        "home_corners_won_roll_5",
        "home_importance",
        "home_points_last_n",
        "home_pts_to_cl",
        "home_shot_accuracy",
        "home_shots_conceded_ema",
        "home_shots_ema_x",
        "home_unbeaten_streak",
        "importance_diff",
        "match_importance",
    ],
    "shots": [
        "away_corners_conceded_ema",
        "away_corners_won_ema",
        "away_first_half_rate",
        "away_shots_ema_x",
        "away_shots_ema_y",
        "discipline_diff",
        "expected_home_corners",
        "h2h_away_wins",
        "home_cards_ema",
        "home_importance",
        "home_points_last_n",
        "home_shot_accuracy",
        "home_shots_conceded_ema",
        "home_unbeaten_streak",
        "importance_diff",
        "match_importance",
    ],
    "under25": [
        "away_corners_conceded_ema",
        "away_corners_won_ema",
        "away_corners_won_roll_10",
        "away_first_half_rate",
        "away_shots_ema_x",
        "away_shots_ema_y",
        "discipline_diff",
        "expected_home_corners",
        "h2h_away_wins",
        "home_cards_ema",
        "home_corners_won_ema",
        "home_corners_won_roll_10",
        "home_corners_won_roll_5",
        "home_importance",
        "home_points_last_n",
        "home_pts_to_cl",
        "home_shot_accuracy",
        "home_shots_conceded_ema",
        "home_shots_ema_x",
        "home_unbeaten_streak",
        "importance_diff",
        "match_importance",
    ],
}

# Patterns that indicate odds/bookmaker data (leaky for predicting match outcomes)
LEAKY_PATTERNS = [
    # Direct odds
    "avg_home",
    "avg_away",
    "avg_draw",
    "avg_over",
    "avg_under",
    "avg_ah",
    "b365_",
    "pinnacle_",
    "max_home",
    "max_away",
    "max_draw",
    "max_over",
    "max_under",
    "max_ah",
    # SportMonks odds (used for ROI calc, not features)
    "sm_btts_",
    "sm_corners_",
    "sm_cards_",
    "sm_shots_",
    # Implied probabilities
    "odds_home_prob",
    "odds_away_prob",
    "odds_draw_prob",
    "odds_over25_prob",
    "odds_under25_prob",
    # Line movements
    "odds_move_",
    "odds_steam_",
    "odds_prob_move",
    "ah_line",
    "line_movement",
    # Derived odds features (still encode bookmaker information)
    "odds_entropy",
    "odds_goals_expectation",
    "odds_home_favorite",
    "odds_overround",
    "odds_prob_diff",
    "odds_prob_max",
    "odds_upset_potential",
    "odds_draw_relative",
    # API-Football match-level stat patterns (safety net for dynamic column names)
    "_insidebox",
    "_outsidebox",
    "_off_goal",
    "goalkeeper_saves",
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


def _adversarial_filter(
    X: np.ndarray,
    feature_names: List[str],
    max_passes: int = 2,
    auc_threshold: float = 0.75,
    importance_threshold: float = 0.05,
    max_features_per_pass: int = 10,
) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
    """Pre-screen and remove temporally leaky features before model training.

    Splits data into first 70% (train) and last 30% (test) to mimic temporal split,
    then uses adversarial validation to find features that distinguish time periods.
    Features with high importance are removed iteratively.

    Args:
        X: Feature matrix (n_samples, n_features), assumed sorted by time.
        feature_names: List of feature names corresponding to columns of X.
        max_passes: Maximum number of iterative removal passes.
        auc_threshold: Only filter if adversarial AUC exceeds this threshold.
        importance_threshold: Remove features with importance > this fraction of total.
        max_features_per_pass: Cap on features removed per pass.

    Returns:
        Tuple of (filtered_X, filtered_feature_names, diagnostics_dict).
    """
    from sklearn.metrics import roc_auc_score

    diagnostics = {
        "passes": [],
        "initial_n_features": len(feature_names),
        "removed_features": [],
    }

    current_X = X.copy()
    current_features = list(feature_names)

    split_idx = int(len(current_X) * 0.7)

    for pass_num in range(max_passes):
        X_train = current_X[:split_idx]
        X_test = current_X[split_idx:]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        auc, top_shift = _adversarial_validation(X_train_scaled, X_test_scaled, current_features)

        pass_info = {"pass": pass_num, "auc": float(auc), "top_features": top_shift[:5]}

        if auc <= auc_threshold:
            pass_info["action"] = "stop_below_threshold"
            diagnostics["passes"].append(pass_info)
            logger.info(
                f"  Adversarial filter pass {pass_num}: AUC={auc:.3f} <= {auc_threshold} — stopping"
            )
            break

        # Calculate total importance and find leaky features
        total_importance = sum(imp for _, imp in top_shift)
        if total_importance == 0:
            pass_info["action"] = "stop_zero_importance"
            diagnostics["passes"].append(pass_info)
            break

        to_remove = []
        for feat_name, imp in top_shift:
            if imp / total_importance > importance_threshold:
                to_remove.append(feat_name)
            if len(to_remove) >= max_features_per_pass:
                break

        if not to_remove:
            pass_info["action"] = "stop_no_features_above_threshold"
            diagnostics["passes"].append(pass_info)
            logger.info(
                f"  Adversarial filter pass {pass_num}: AUC={auc:.3f} but no features above {importance_threshold:.0%} importance"
            )
            break

        # Safety: never remove so many features that fewer than 5 remain
        max_removable = len(current_features) - 5
        if max_removable <= 0:
            pass_info["action"] = "stop_too_few_features"
            diagnostics["passes"].append(pass_info)
            logger.info(
                f"  Adversarial filter pass {pass_num}: only {len(current_features)} features left, stopping"
            )
            break
        to_remove = to_remove[:max_removable]

        # Remove features
        keep_mask = [f not in to_remove for f in current_features]
        current_X = current_X[:, keep_mask]
        current_features = [f for f, keep in zip(current_features, keep_mask) if keep]
        diagnostics["removed_features"].extend(to_remove)

        pass_info["action"] = "removed"
        pass_info["removed"] = to_remove
        diagnostics["passes"].append(pass_info)

        logger.info(
            f"  Adversarial filter pass {pass_num}: AUC={auc:.3f}, removed {len(to_remove)} features: {to_remove}"
        )

        # If AUC still high after removal, do another pass (up to max)
        if pass_num < max_passes - 1 and auc > auc_threshold:
            continue
        else:
            break

    diagnostics["final_n_features"] = len(current_features)
    diagnostics["total_removed"] = len(diagnostics["removed_features"])

    return current_X, current_features, diagnostics


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
    # Meta-learner stacking weights from non-negative Ridge
    stacking_weights: Dict[str, float] = None
    stacking_alpha: float = None
    # Adversarial validation diagnostics
    adversarial_validation: Dict[str, Any] = None
    # Calibration method selected by Optuna
    calibration_method: str = None
    # Per-league calibration metrics
    per_league_ece: Dict[str, float] = None
    # Calibration validation result (ECE check)
    calibration_validation: Dict[str, Any] = None
    # Brier Score (calibration + sharpness combined)
    brier_score: float = None
    # Forecast Value Added vs market-implied baseline
    fva: float = None
    # Forecastability diagnostics
    mean_pe_residual: float = None
    forecastability_gate: str = None  # "passed", "rejected", or None
    # Measurement hardening diagnostics (S26+)
    embargo_days_computed: int = None
    embargo_days_effective: int = None
    aggressive_regularization_applied: bool = None
    adversarial_auc_mean: float = None
    regularization_overrides: Dict[str, Any] = None
    mrmr_result: Dict[str, Any] = None
    per_fold_ks: List[Dict[str, Any]] = None
    # Uncertainty (MAPIE conformal)
    uncertainty_penalty: float = None
    holdout_uncertainty_roi: float = None


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
        use_two_stage: Optional[bool] = None,
        only_catboost: bool = False,
        no_catboost: bool = False,
        no_fastai: bool = False,
        merge_params_path: Optional[str] = None,
        adversarial_filter: bool = False,
        adversarial_max_passes: int = 2,
        adversarial_max_features: int = 10,
        adversarial_auc_threshold: float = 0.75,
        use_monotonic: bool = False,
        use_transfer_learning: bool = False,
        use_baseline: bool = False,
        deterministic: bool = False,
        n_holdout_folds: int = 1,
        max_ece: float = 0.15,
        cv_method: str = "walk_forward",
        embargo_days: int = 14,
        pe_gate: float = 1.0,
        no_aggressive_reg: bool = False,
        mrmr_k: int = 0,
    ):
        self.bet_type = bet_type
        self.config = BET_TYPES[bet_type]
        self.n_folds = n_folds
        self.n_holdout_folds = n_holdout_folds
        self.max_ece = max_ece
        self.cv_method = cv_method
        self.embargo_days = embargo_days
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
        self.use_two_stage = use_two_stage if use_two_stage is not None else (not fast_mode)
        self.only_catboost = only_catboost
        self.no_catboost = no_catboost
        self.no_fastai = no_fastai
        self.merge_params_path = merge_params_path
        self.adversarial_filter = adversarial_filter
        self.adversarial_max_passes = adversarial_max_passes
        self.adversarial_max_features = adversarial_max_features
        self.adversarial_auc_threshold = adversarial_auc_threshold
        self.use_monotonic = use_monotonic
        self.use_transfer_learning = use_transfer_learning
        self.use_baseline = use_baseline
        self.deterministic = deterministic
        self.pe_gate = pe_gate
        self.no_aggressive_reg = no_aggressive_reg
        self.mrmr_k = mrmr_k
        self._base_model_path: Optional[str] = None  # Path to transfer learning base model
        self._adversarial_auc_mean: Optional[float] = None

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
        self.use_fastai = FASTAI_AVAILABLE and not no_fastai

    def _compute_pe_gate(self, df: pd.DataFrame) -> Optional[float]:
        """Compute mean permutation entropy for PE gate check.

        Uses a fast rough estimate: MIN_SERIES_LENGTH=50, up to 200 teams.

        Returns:
            Mean PE across teams, or None if computation fails.
        """
        try:
            from experiments.forecastability_analysis import (
                _set_min_series_length,
                build_team_series,
            )
            from experiments.forecastability_analysis import permutation_entropy as pe_func
        except ImportError:
            logger.warning("Cannot import forecastability_analysis for PE gate")
            return None

        target_col = self.config["target"]
        odds_col = self.config.get("odds_col")
        target_line = self.config.get("line")

        # Use lower min series for fast gate check
        original_min = 100  # default MIN_SERIES_LENGTH
        _set_min_series_length(50)
        try:
            team_series = build_team_series(df, target_col, odds_col, target_line)
        finally:
            _set_min_series_length(original_min)

        if not team_series:
            logger.warning(f"PE gate: no qualifying teams for {self.bet_type}")
            return None

        # Sample up to 200 teams for speed
        teams = list(team_series.keys())[:200]
        pe_values = []
        for team in teams:
            residuals = team_series[team]["residuals"]
            pe = pe_func(residuals, order=3, delay=1)
            if not np.isnan(pe):
                pe_values.append(pe)

        if not pe_values:
            return None

        mean_pe = float(np.mean(pe_values))
        logger.info(f"PE gate: {self.bet_type} mean_pe={mean_pe:.4f} ({len(pe_values)} teams)")
        return mean_pe

    def _compute_embargo_days(self) -> int:
        """Compute embargo from feature config's max lookback window.

        Derives the minimum safe embargo period by examining the maximum
        lookback used by any feature engineer, then converting matches to
        calendar days (~3.5 days/match at 10 leagues).

        Returns:
            Embargo period in calendar days.
        """
        fc = self.feature_config
        max_lookback_matches = max(
            getattr(fc, "form_window", 5) if fc else 5,
            getattr(fc, "ema_span", 10) if fc else 10,
            getattr(fc, "poisson_lookback", 10) if fc else 10,
            (getattr(fc, "h2h_matches", 5) if fc else 5) * 5,  # H2H spans multiple seasons
            20,  # corner/niche window_sizes max
        )
        # ~3.5 days per match (10 leagues, ~38 matches/season/league)
        embargo_days = int(max_lookback_matches * 3.5) + 7  # safety buffer
        return max(embargo_days, 14)  # floor at 14 days

    def _get_cv_splits(self, n_samples: int, dates: Optional[np.ndarray] = None):
        """
        Generate cross-validation splits based on cv_method.

        Returns list of (train_start, train_end, test_start, test_end) tuples.
        """
        fold_size = n_samples // (self.n_folds + 1)

        if self.cv_method == "purged_kfold" and dates is not None:
            # Purged Walk-Forward CV with date-based embargo
            # Instead of fixed sample-count buffer, use calendar days
            dates_dt = pd.to_datetime(dates)
            splits = []

            for fold in range(self.n_folds):
                train_end = (fold + 1) * fold_size
                if train_end >= n_samples:
                    continue

                # Purge: remove training samples within embargo_days of test boundary
                test_boundary_date = (
                    dates_dt.iloc[train_end] if train_end < len(dates_dt) else dates_dt.iloc[-1]
                )
                embargo_start = test_boundary_date - pd.Timedelta(days=self.embargo_days)

                # Find purged train end: last sample before embargo zone
                purge_mask = dates_dt[:train_end] <= embargo_start
                if purge_mask.sum() < 100:
                    # Not enough data after purging, fall back to temporal buffer
                    purged_train_end = max(0, train_end - self.temporal_buffer)
                else:
                    purged_train_end = purge_mask.sum()

                # Embargo: test starts embargo_days after train boundary
                embargo_end_date = test_boundary_date + pd.Timedelta(days=self.embargo_days)
                test_mask = dates_dt[train_end:] >= embargo_end_date
                if test_mask.sum() < 20:
                    # Fall back to temporal buffer
                    test_start = train_end + self.temporal_buffer
                else:
                    test_start = train_end + (~test_mask[: self.temporal_buffer * 3]).sum()
                    test_start = min(test_start, train_end + self.temporal_buffer * 3)

                test_end = min(test_start + fold_size, n_samples)

                if test_start >= n_samples or purged_train_end < 100:
                    continue

                splits.append((0, purged_train_end, test_start, test_end))

            logger.info(f"  Purged CV: {len(splits)} folds, embargo={self.embargo_days} days")
            return splits
        else:
            # Standard walk-forward with feature-aware date-based embargo
            computed_embargo = self._compute_embargo_days()
            # CLI --embargo-days overrides the auto-computed value
            effective_embargo = max(self.embargo_days, computed_embargo)

            splits = []
            for fold in range(self.n_folds):
                train_end = (fold + 1) * fold_size

                if dates is not None:
                    # Date-based embargo: skip samples within embargo window
                    dates_dt = pd.to_datetime(dates)
                    boundary_date = dates_dt.iloc[min(train_end, len(dates_dt) - 1)]
                    embargo_end_date = boundary_date + pd.Timedelta(days=effective_embargo)
                    # Find first test sample after embargo
                    remaining = dates_dt.iloc[train_end:]
                    post_embargo = remaining >= embargo_end_date
                    if post_embargo.sum() < 20:
                        # Not enough data after embargo, fall back to sample buffer
                        test_start = train_end + self.temporal_buffer
                    else:
                        test_start = train_end + int((~post_embargo).sum())
                else:
                    test_start = train_end + self.temporal_buffer

                test_end = min(test_start + fold_size, n_samples)

                if test_start >= n_samples:
                    continue

                splits.append((0, train_end, test_start, test_end))

            if dates is not None:
                logger.info(
                    f"  Walk-forward CV: {len(splits)} folds, embargo={effective_embargo} days "
                    f"(auto={computed_embargo}, cli={self.embargo_days})"
                )
            return splits

    def _build_monotonic_constraints(self, feature_names: List[str]) -> Optional[List[int]]:
        """Build monotonic constraints vector from strategies.yaml for CatBoost/LightGBM/XGBoost.

        Returns list of 0/1/-1 per feature, or None if no constraints defined.
        """
        import yaml

        strategies_path = Path("config/strategies.yaml")
        if not strategies_path.exists():
            return None

        with open(strategies_path) as f:
            strategies = yaml.safe_load(f)

        # Look up constraints for this bet type, also check base market for variants
        # e.g. fouls_over_265 → fouls
        bet_type = self.bet_type
        base_market = bet_type.split("_over_")[0].split("_under_")[0]

        constraints_dict = None
        strats = strategies.get("strategies", {})
        if bet_type in strats and "monotonic_constraints" in strats[bet_type]:
            constraints_dict = strats[bet_type]["monotonic_constraints"]
        elif base_market in strats and "monotonic_constraints" in strats[base_market]:
            constraints_dict = strats[base_market]["monotonic_constraints"]

        if not constraints_dict:
            return None

        # Build constraint vector: 0 = unconstrained, 1 = monotone increasing, -1 = decreasing
        constraints = []
        n_constrained = 0
        for feat in feature_names:
            if feat in constraints_dict:
                constraints.append(constraints_dict[feat])
                n_constrained += 1
            else:
                constraints.append(0)

        if n_constrained == 0:
            return None

        logger.info(f"  Monotonic constraints: {n_constrained} features constrained for {bet_type}")
        return constraints

    @staticmethod
    def _get_base_model_types(
        include_fastai: bool = True,
        fast_mode: bool = False,
        include_two_stage: bool = False,
        only_catboost: bool = False,
        no_catboost: bool = False,
    ) -> List[str]:
        """Return list of base model types to use."""
        if only_catboost:
            return ["catboost"]
        if fast_mode:
            models = ["lightgbm", "xgboost"]
        else:
            models = ["lightgbm", "catboost", "xgboost"]
            if no_catboost:
                models.remove("catboost")
            if include_fastai and FASTAI_AVAILABLE:
                models.append("fastai")
        if include_two_stage and not fast_mode:
            models.extend(["two_stage_lgb", "two_stage_xgb"])
        return models

    # Keys stored in Optuna trials but not valid CatBoost constructor args
    _CATBOOST_STRIP_KEYS = {"use_monotonic", "ft_iterations"}

    @staticmethod
    def _safe_predict_proba(model, X) -> Optional[np.ndarray]:
        """Call predict_proba and handle the 1-column edge case.

        CalibratedClassifierCV can return a single column when an inner CV fold
        sees only one class. This method returns the positive-class probabilities
        or None if only one column is present (caller should skip the fold).
        """
        proba = model.predict_proba(X)
        if proba.ndim == 1 or proba.shape[1] == 1:
            return None
        return proba[:, 1]

    def _create_model_instance(self, model_type: str, params: Dict[str, Any], seed: int = 42):
        """Create a model instance for the given type and params."""
        if model_type == "lightgbm":
            return lgb.LGBMClassifier(**params, random_state=seed, verbose=-1)
        elif model_type == "catboost":
            # Strip non-CatBoost keys (e.g. use_monotonic from Optuna trial)
            params = {k: v for k, v in params.items() if k not in self._CATBOOST_STRIP_KEYS}
            extra_params = {}
            # Use EnhancedCatBoost when transfer learning or baseline is enabled
            if self.use_transfer_learning or self.use_baseline:
                return EnhancedCatBoost(
                    init_model_path=self._base_model_path if self.use_transfer_learning else None,
                    use_baseline=self.use_baseline,
                    **params,
                    **extra_params,
                    random_seed=seed,
                    verbose=False,
                    has_time=True,
                )
            return CatBoostClassifier(
                **params, **extra_params, random_seed=seed, verbose=False, has_time=True
            )
        elif model_type == "xgboost":
            return xgb.XGBClassifier(**params, random_state=seed, verbosity=0)
        elif model_type == "fastai":
            return FastAITabularModel(**params, random_state=seed)
        elif model_type.startswith("two_stage_"):
            from src.ml.two_stage_model import create_two_stage_model

            base = "lightgbm" if model_type == "two_stage_lgb" else "catboost"
            return create_two_stage_model(
                base,
                stage1_params=params.get("stage1_params"),
                stage2_params=params.get("stage2_params"),
                calibration_method=params.get("calibration_method", "sigmoid"),
                min_edge_threshold=params.get("min_edge_threshold", 0.02),
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def load_data(self) -> pd.DataFrame:
        """Load and prepare feature data."""
        from src.utils.data_io import load_features

        df = load_features(FEATURES_FILE)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        # Note: bracketed string cleaning (e.g. '[5.07E-1]') is now handled
        # centrally in load_features() in src/utils/data_io.py

        # Recover home_goals/away_goals from total_goals + goal_difference
        if "total_goals" in df.columns and "goal_difference" in df.columns:
            derived_hg = (df["total_goals"] + df["goal_difference"]) / 2
            derived_ag = (df["total_goals"] - df["goal_difference"]) / 2
            if "home_goals" in df.columns:
                df["home_goals"] = df["home_goals"].fillna(derived_hg)
            else:
                df["home_goals"] = derived_hg
            if "away_goals" in df.columns:
                df["away_goals"] = df["away_goals"].fillna(derived_ag)
            else:
                df["away_goals"] = derived_ag

        # Derive target if needed (or fill gaps from components)
        target = self.config["target"]
        self._derive_target(df, target)

        logger.info(f"Loaded {len(df)} matches for {self.bet_type}")

        # Filter implausible training rows for niche line markets
        from src.utils.line_plausibility import filter_implausible_training_rows

        df = filter_implausible_training_rows(df, self.bet_type)

        return df

    def load_or_optimize_feature_config(self) -> Optional[BetTypeFeatureConfig]:
        """
        Load or optimize feature parameters.

        Returns:
            BetTypeFeatureConfig if using custom params, None for default behavior
        """
        if self.feature_params_path:
            # Load from file
            logger.debug(f"Loading feature params from {self.feature_params_path}")
            config = BetTypeFeatureConfig.load(Path(self.feature_params_path))
            logger.debug(f"Loaded feature config: {config.summary()}")
            return config

        elif self.optimize_features:
            # Run feature parameter optimization
            logger.info(f"Running feature parameter optimization ({self.n_feature_trials} trials)")

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
            logger.debug(f"Saved optimized feature params to {output_path}")

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
            logger.debug("Regenerating features with optimized params...")

            if self.regenerator is None:
                self.regenerator = FeatureRegenerator()

            df = self.regenerator.regenerate_with_params(self.feature_config)

            # Derive target if needed (regenerated features may not have targets)
            target = self.config["target"]
            if target not in df.columns:
                self._derive_target(df, target)

            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)

            # Filter implausible training rows for niche line markets
            from src.utils.line_plausibility import filter_implausible_training_rows

            df = filter_implausible_training_rows(df, self.bet_type)

            logger.debug(f"Regenerated {len(df)} matches with custom feature params")
            return df
        else:
            # Use default feature loading
            return self.load_data()

    @staticmethod
    def _safe_col(df: pd.DataFrame, col: str) -> pd.Series:
        """Get column as Series, returning NaN series if column missing."""
        if col in df.columns:
            return df[col]
        return pd.Series(np.nan, index=df.index)

    def _derive_target(self, df: pd.DataFrame, target: str) -> None:
        """Derive target column if not present (or fill gaps) in dataframe."""
        derived = None

        if target == "total_cards":
            # Prefer home_yellow_cards (54.5% coverage) over home_yellows (19.4%)
            for yellow_h, yellow_a, red_h, red_a in [
                ("home_yellow_cards", "away_yellow_cards", "home_red_cards", "away_red_cards"),
                ("home_yellows", "away_yellows", "home_reds", "away_reds"),
            ]:
                if yellow_h in df.columns and yellow_a in df.columns:
                    part = df[yellow_h].fillna(0) + df[yellow_a].fillna(0)
                    if red_h in df.columns:
                        part = part + df[red_h].fillna(0)
                    if red_a in df.columns:
                        part = part + df[red_a].fillna(0)
                    both_missing = df[yellow_h].isna() & df[yellow_a].isna()
                    part[both_missing] = np.nan
                    derived = part if derived is None else derived.fillna(part)
        elif target == "total_shots":
            if "home_shots" in df.columns and "away_shots" in df.columns:
                derived = df["home_shots"].fillna(0) + df["away_shots"].fillna(0)
                both_missing = df["home_shots"].isna() & df["away_shots"].isna()
                derived[both_missing] = np.nan
        elif target == "total_fouls":
            if "home_fouls" in df.columns and "away_fouls" in df.columns:
                derived = df["home_fouls"].fillna(0) + df["away_fouls"].fillna(0)
                both_missing = df["home_fouls"].isna() & df["away_fouls"].isna()
                derived[both_missing] = np.nan
        elif target == "total_corners":
            if "home_corners" in df.columns and "away_corners" in df.columns:
                derived = df["home_corners"].fillna(0) + df["away_corners"].fillna(0)
                both_missing = df["home_corners"].isna() & df["away_corners"].isna()
                derived[both_missing] = np.nan
        elif target == "total_goals":
            if "home_goals" in df.columns and "away_goals" in df.columns:
                derived = df["home_goals"].fillna(0) + df["away_goals"].fillna(0)
                both_missing = df["home_goals"].isna() & df["away_goals"].isna()
                derived[both_missing] = np.nan
        elif target == "corner_diff":
            if "home_corners" in df.columns and "away_corners" in df.columns:
                derived = df["home_corners"].fillna(0) - df["away_corners"].fillna(0)
                both_missing = df["home_corners"].isna() & df["away_corners"].isna()
                derived[both_missing] = np.nan
        elif target == "card_diff":
            if "home_cards" in df.columns and "away_cards" in df.columns:
                derived = df["home_cards"].fillna(0) - df["away_cards"].fillna(0)
                both_missing = df["home_cards"].isna() & df["away_cards"].isna()
                derived[both_missing] = np.nan
        elif target == "ht_total_goals":
            if "ht_home" in df.columns and "ht_away" in df.columns:
                derived = df["ht_home"].fillna(0) + df["ht_away"].fillna(0)
                both_missing = df["ht_home"].isna() & df["ht_away"].isna()
                derived[both_missing] = np.nan
        elif target == "home_win_h1":
            if "ht_home" in df.columns and "ht_away" in df.columns:
                df["home_win_h1"] = (df["ht_home"] > df["ht_away"]).astype(float)
                both_missing = df["ht_home"].isna() | df["ht_away"].isna()
                df.loc[both_missing, "home_win_h1"] = np.nan
            return
        elif target == "away_win_h1":
            if "ht_home" in df.columns and "ht_away" in df.columns:
                df["away_win_h1"] = (df["ht_away"] > df["ht_home"]).astype(float)
                both_missing = df["ht_home"].isna() | df["ht_away"].isna()
                df.loc[both_missing, "away_win_h1"] = np.nan
            return
        elif target == "under25":
            if "total_goals" in df.columns:
                df["under25"] = (df["total_goals"] < 2.5).astype(int)
            elif "home_goals" in df.columns and "away_goals" in df.columns:
                df["under25"] = (
                    (df["home_goals"].fillna(0) + df["away_goals"].fillna(0)) < 2.5
                ).astype(int)
            return
        elif target == "over25":
            if "total_goals" in df.columns:
                df["over25"] = (df["total_goals"] > 2.5).astype(int)
            elif "home_goals" in df.columns and "away_goals" in df.columns:
                df["over25"] = (
                    (df["home_goals"].fillna(0) + df["away_goals"].fillna(0)) > 2.5
                ).astype(int)
            return
        elif target == "btts":
            home_goals = (
                df["home_goals"] if "home_goals" in df.columns else pd.Series(0, index=df.index)
            )
            away_goals = (
                df["away_goals"] if "away_goals" in df.columns else pd.Series(0, index=df.index)
            )
            df["btts"] = ((home_goals.fillna(0) > 0) & (away_goals.fillna(0) > 0)).astype(int)
            return

        # Fill gaps: if column already exists with some values, fill NaN from derived
        if derived is not None:
            if target in df.columns:
                df[target] = df[target].fillna(derived)
            else:
                df[target] = derived

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

        min_weight = getattr(self, "sample_min_weight", 0.1)
        weights = calculate_time_decay_weights(
            dates,
            decay_rate=self.sample_decay_rate,
            min_weight=min_weight,
        )

        logger.debug(
            f"Sample weights: min={weights.min():.3f}, max={weights.max():.3f}, "
            f"mean={weights.mean():.3f}, decay_rate={self.sample_decay_rate:.4f}"
        )

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
        adjusted = base_threshold * (odds_ratio**alpha)

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

        features = [
            c for c in all_cols - exclude if df[c].dtype in ["float64", "int64", "float32", "int32"]
        ]
        n_low_imp = len(set(bt_exclusions) & all_cols)
        logger.debug(
            f"Excluded {len(exclude)} columns ({n_low_imp} low-importance), {len(features)} features remain"
        )
        return sorted(features)

    def prepare_target(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare target variable.

        For regression_line targets, NaN values in the source column are
        preserved as NaN (not silently converted to 0) so that the downstream
        valid_mask can filter them out.
        """
        target_col = self.config["target"]

        if self.config["approach"] == "classification":
            return df[target_col].values.astype(float)
        elif self.config["approach"] == "regression_line":
            line = self.config.get("target_line", 0)
            direction = self.config.get("direction", "over")
            raw = df[target_col].values.astype(float)
            if direction == "under":
                result = np.where(np.isnan(raw), np.nan, (raw < line).astype(float))
            else:
                result = np.where(np.isnan(raw), np.nan, (raw > line).astype(float))
            return result
        else:
            return df[target_col].values.astype(float)

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
            importance_type="gain",
            reg_alpha=0.5,
            reg_lambda=0.5,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=self.seed,
            n_jobs=1,
            verbose=-1,
        )

        if self.auto_rfe:
            # RFECV: automatically find optimal number of features via CV
            # Cap at max_rfe_features to prevent bloated feature sets (R52: under25 got 155 features)
            logger.info(
                f"Running RFECV to find optimal feature count "
                f"(min={self.min_rfe_features}, max={self.max_rfe_features})..."
            )

            from sklearn.model_selection import TimeSeriesSplit

            cv = TimeSeriesSplit(n_splits=3)

            rfecv = RFECV(
                estimator=base_model,
                step=10,
                cv=cv,
                scoring="roc_auc",
                min_features_to_select=self.min_rfe_features,
                n_jobs=-1,
            )
            rfecv.fit(X, y)

            selected_indices = np.where(rfecv.support_)[0]
            optimal_n = rfecv.n_features_
            logger.info(f"RFECV found optimal feature count: {optimal_n}")
            logger.debug(
                f"CV scores by n_features: min={min(rfecv.cv_results_['mean_test_score']):.3f}, "
                f"max={max(rfecv.cv_results_['mean_test_score']):.3f}"
            )

            # Enforce max cap: if RFECV selected too many, trim to top N by importance
            if len(selected_indices) > self.max_rfe_features:
                logger.warning(
                    f"RFECV selected {len(selected_indices)} features, "
                    f"capping to {self.max_rfe_features} by importance ranking"
                )
                base_model.fit(X[:, selected_indices], y)
                importances = base_model.feature_importances_
                top_k = np.argsort(importances)[::-1][: self.max_rfe_features]
                selected_indices = np.sort(selected_indices[top_k])
                logger.debug(f"Capped to {len(selected_indices)} features")
        elif sample_weights is not None:
            # Weighted importance ranking: train with sample weights, select by gain importance
            logger.info(
                f"Running weighted feature selection (top {self.n_rfe_features} by weighted gain)..."
            )

            base_model.fit(X, y, sample_weight=sample_weights)
            importances = base_model.feature_importances_

            n_features = min(self.n_rfe_features, X.shape[1])
            selected_indices = np.argsort(importances)[::-1][:n_features]
            selected_indices = np.sort(selected_indices)  # Restore original order

            logger.info(
                f"Selected {len(selected_indices)} features via weighted importance ranking"
            )
            return selected_indices.tolist()
        else:
            # Fixed RFE: use specified n_rfe_features
            logger.info(f"Running RFE to select top {self.n_rfe_features} features...")

            n_features = min(self.n_rfe_features, X.shape[1])
            rfe = RFE(estimator=base_model, n_features_to_select=n_features, step=10)
            rfe.fit(X, y)

            selected_indices = np.where(rfe.support_)[0]

        logger.info(
            f"Selected {len(selected_indices)} features via {'RFECV' if self.auto_rfe else 'RFE'}"
        )
        return selected_indices.tolist()

    def _per_fold_ks_test(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        feature_names: List[str],
        top_k: int = 20,
    ) -> Dict[str, Any]:
        """Run KS test on top features to detect distribution shift per fold.

        Tests whether train and test distributions differ significantly for
        each of the top-k features (by variance). Reports shifted features
        (p < 0.05) for diagnostic purposes.

        Args:
            X_train: Training feature matrix.
            X_test: Test feature matrix.
            feature_names: Feature names.
            top_k: Number of features to test (by variance).

        Returns:
            Dict with n_shifted, n_tested, shifted_features, high_shift flag.
        """
        from scipy.stats import ks_2samp

        n_features = min(top_k, len(feature_names))

        # Select top features by variance (most informative for shift detection)
        variances = np.var(X_train, axis=0)
        top_indices = np.argsort(variances)[::-1][:n_features]

        shifted = []
        for idx in top_indices:
            stat, pval = ks_2samp(X_train[:, idx], X_test[:, idx])
            if pval < 0.05:
                shifted.append(
                    {
                        "feature": feature_names[idx],
                        "ks_stat": round(float(stat), 4),
                        "p_value": round(float(pval), 6),
                    }
                )

        return {
            "n_shifted": len(shifted),
            "n_tested": n_features,
            "high_shift": len(shifted) > n_features * 0.5,
            "shifted_features": shifted[:10],  # Top 10 for storage
        }

    def _mrmr_select(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        k: int,
    ) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
        """Minimum Redundancy Maximum Relevance (mRMR) feature selection.

        Greedy forward selection that maximizes mutual information with target
        while minimizing average correlation with already-selected features.

        Args:
            X: Feature matrix.
            y: Target vector.
            feature_names: Feature names corresponding to X columns.
            k: Number of features to select.

        Returns:
            Tuple of (selected_X, selected_feature_names, diagnostics).
        """
        from sklearn.feature_selection import mutual_info_classif

        n_features = X.shape[1]
        k = min(k, n_features)

        # Compute MI between each feature and target
        mi_scores = mutual_info_classif(X, y, random_state=self.seed, n_neighbors=5)

        # Greedy forward selection
        selected = []
        remaining = list(range(n_features))

        # First feature: highest MI
        best_idx = int(np.argmax(mi_scores))
        selected.append(best_idx)
        remaining.remove(best_idx)

        # Pre-compute correlation matrix for redundancy
        corr = np.abs(np.corrcoef(X.T))

        while len(selected) < k and remaining:
            best_score = -np.inf
            best_feat = None

            for feat in remaining:
                relevance = mi_scores[feat]
                # Average absolute correlation with already-selected features
                redundancy = np.mean([corr[feat, s] for s in selected])
                # mRMR score: relevance - redundancy
                score = relevance - redundancy
                if score > best_score:
                    best_score = score
                    best_feat = feat

            if best_feat is None:
                break
            selected.append(best_feat)
            remaining.remove(best_feat)

        selected = sorted(selected)
        removed = [feature_names[i] for i in range(n_features) if i not in selected]
        selected_names = [feature_names[i] for i in selected]
        X_selected = X[:, selected]

        diagnostics = {
            "pre_count": n_features,
            "post_count": len(selected),
            "removed_features": removed[:20],  # Top 20 for storage
            "n_removed": len(removed),
        }

        logger.info(f"mRMR: {n_features} → {len(selected)} features (removed {len(removed)})")
        return X_selected, selected_names, diagnostics

    def create_objective(
        self,
        X: np.ndarray,
        y: np.ndarray,
        odds: np.ndarray,
        model_type: str,
        dates: Optional[np.ndarray] = None,
    ):
        """Create Optuna objective for a specific model type."""
        # Pre-compute CV splits once (same for all trials)
        n_samples = len(y)
        cv_splits = self._get_cv_splits(
            n_samples,
            dates=pd.Series(dates) if dates is not None else None,
        )

        def objective(trial):
            # Calibration method (tuned per trial)
            # "beta" uses sigmoid for CalibratedClassifierCV + BetaCalibrator post-hoc
            trial_cal_method = trial.suggest_categorical(
                "calibration_method", ["sigmoid", "beta", "temperature"]
            )

            # Uncertainty penalty for MAPIE conformal stake adjustment
            trial.suggest_float("uncertainty_penalty", 0.5, 3.0, step=0.25)

            # Sample weight hyperparameters (tuned per trial)
            if self.use_sample_weights and dates is not None:
                # R47/R48 decay collapsed to 0.0006; R59-62 shots hit 0.002 floor. Allow slower decay.
                trial_decay_rate = trial.suggest_float("decay_rate", 0.001, 0.01, log=True)
                trial_min_weight = trial.suggest_float("min_weight", 0.05, 0.5)
            else:
                trial_decay_rate = None
                trial_min_weight = None

            # Aggressive regularization: tighten bounds when adversarial AUC > 0.8
            _agg = getattr(self, "_aggressive_reg_applied", False)

            if model_type == "lightgbm":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
                    "max_depth": trial.suggest_int("max_depth", 3, 4 if _agg else 8),
                    "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
                    "num_leaves": trial.suggest_int("num_leaves", 20, 50 if _agg else 100),
                    "min_child_samples": trial.suggest_int(
                        "min_child_samples", 50 if _agg else 20, 200 if _agg else 100
                    ),
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
                    "random_state": self.seed,
                    "verbose": -1,
                }
                ModelClass = lgb.LGBMClassifier
            elif model_type == "catboost":
                # Transfer learning base model uses SymmetricTree — fine-tuning must match
                if self.use_transfer_learning and self._base_model_path:
                    grow_policy = "SymmetricTree"
                else:
                    grow_policy = trial.suggest_categorical(
                        "grow_policy", ["SymmetricTree", "Depthwise"]
                    )
                params = {
                    "iterations": trial.suggest_int("iterations", 100, 600, step=100),
                    "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.35, log=True),
                    "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 200, log=True),
                    "min_data_in_leaf": trial.suggest_int(
                        "min_data_in_leaf", 50 if _agg else 1, 200 if _agg else 100
                    ),
                    "random_strength": trial.suggest_float(
                        "random_strength", 0.001, 10.0, log=True
                    ),
                    "rsm": trial.suggest_float("rsm", 0.5, 0.6 if _agg else 1.0),
                    "grow_policy": grow_policy,
                    "random_seed": self.seed,
                    "verbose": False,
                }
                params["depth"] = trial.suggest_int("depth", 3 if _agg else 4, 4 if _agg else 8)
                # Model shrink rate for regularization (incompatible with transfer learning)
                if not (self.use_transfer_learning and self._base_model_path):
                    shrink_rate = trial.suggest_float("model_shrink_rate", 0.0, 0.1)
                    if shrink_rate > 0:
                        params["model_shrink_rate"] = shrink_rate
                        params["model_shrink_mode"] = "Constant"
                # Monotonic constraints (Optuna toggle)
                # CatBoost only supports monotonic constraints with SymmetricTree grow_policy
                if self.use_monotonic and self.optimal_features and grow_policy == "SymmetricTree":
                    use_mono = trial.suggest_categorical("use_monotonic", [True, False])
                    if use_mono:
                        constraints = self._build_monotonic_constraints(self.optimal_features)
                        if constraints is not None:
                            params["monotone_constraints"] = constraints
                # Deterministic mode (debug only — slows training)
                if self.deterministic:
                    params["task_type"] = "CPU"
                    params["bootstrap_type"] = "No"
                # Use EnhancedCatBoost wrapper when transfer learning or baseline enabled
                if self.use_transfer_learning or self.use_baseline:
                    # Transfer learning: reduce fine-tune iterations
                    if self.use_transfer_learning and self._base_model_path:
                        params["iterations"] = trial.suggest_int("ft_iterations", 50, 200, step=50)
                    ModelClass = lambda **p: EnhancedCatBoost(
                        init_model_path=(
                            self._base_model_path if self.use_transfer_learning else None
                        ),
                        use_baseline=self.use_baseline,
                        **p,
                    )
                else:
                    ModelClass = CatBoostClassifier
            elif model_type == "xgboost":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
                    "max_depth": trial.suggest_int("max_depth", 3, 4 if _agg else 8),
                    "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.35, log=True),
                    "min_child_weight": trial.suggest_int(
                        "min_child_weight", 50 if _agg else 20, 200 if _agg else 50
                    ),
                    "subsample": trial.suggest_float("subsample", 0.5, 0.7 if _agg else 1.0),
                    "colsample_bytree": trial.suggest_float(
                        "colsample_bytree", 0.5, 0.7 if _agg else 1.0
                    ),
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
                # Tune actual stage1/stage2 hyperparameters + calibration
                ts_cal = trial.suggest_categorical("ts_calibration", ["sigmoid"])
                min_edge = trial.suggest_float("min_edge", 0.0, 0.05)

                if model_type == "two_stage_lgb":
                    lr = trial.suggest_float("ts_learning_rate", 0.005, 0.2, log=True)
                    s1_params = {
                        "n_estimators": trial.suggest_int("ts_s1_n_estimators", 100, 800, step=100),
                        "max_depth": trial.suggest_int("ts_s1_max_depth", 3, 8),
                        "learning_rate": lr,
                        "reg_alpha": trial.suggest_float("ts_reg_alpha", 0.1, 10.0, log=True),
                        "reg_lambda": trial.suggest_float("ts_reg_lambda", 0.1, 10.0, log=True),
                        "random_state": self.seed,
                        "verbose": -1,
                    }
                    s2_params = {
                        "n_estimators": trial.suggest_int("ts_s2_n_estimators", 100, 800, step=100),
                        "max_depth": trial.suggest_int("ts_s2_max_depth", 3, 8),
                        "learning_rate": lr,
                        "reg_alpha": s1_params["reg_alpha"],
                        "reg_lambda": s1_params["reg_lambda"],
                        "random_state": self.seed,
                        "verbose": -1,
                    }
                else:  # two_stage_xgb -> CatBoost
                    lr = trial.suggest_float("ts_learning_rate", 0.005, 0.2, log=True)
                    s1_params = {
                        "iterations": trial.suggest_int("ts_s1_iterations", 100, 800, step=100),
                        "depth": trial.suggest_int("ts_s1_depth", 3, 8),
                        "learning_rate": lr,
                        "l2_leaf_reg": trial.suggest_float("ts_l2_leaf_reg", 0.1, 10.0, log=True),
                        "random_seed": self.seed,
                        "verbose": False,
                    }
                    s2_params = {
                        "iterations": trial.suggest_int("ts_s2_iterations", 100, 800, step=100),
                        "depth": trial.suggest_int("ts_s2_depth", 3, 8),
                        "learning_rate": lr,
                        "l2_leaf_reg": s1_params["l2_leaf_reg"],
                        "random_seed": self.seed,
                        "verbose": False,
                    }

                params = {
                    "stage1_params": s1_params,
                    "stage2_params": s2_params,
                    "calibration_method": ts_cal,
                    "min_edge_threshold": min_edge,
                }
                ModelClass = None  # Handled separately in the fold loop

            # Walk-forward validation using pre-computed splits
            all_preds = []
            all_actuals = []
            all_odds = []

            for fold, (_, train_end, test_start, test_end) in enumerate(cv_splits):
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
                        from src.ml.two_stage_model import create_two_stage_model

                        base = "lightgbm" if model_type == "two_stage_lgb" else "catboost"
                        ts_model = create_two_stage_model(
                            base,
                            stage1_params=params.get("stage1_params"),
                            stage2_params=params.get("stage2_params"),
                            calibration_method=params.get("calibration_method", "sigmoid"),
                            min_edge_threshold=params.get("min_edge_threshold", 0.02),
                        )
                        odds_train = odds[:train_end]
                        ts_model.fit(X_train_scaled, y_train, odds_train)
                        result_dict = ts_model.predict_proba(X_test_scaled, odds_test)
                        probs = result_dict["combined_score"]
                    else:
                        model = ModelClass(**params)
                        # Beta calibration: use sigmoid for sklearn, then apply BetaCalibrator post-hoc
                        sklearn_cal = (
                            "sigmoid"
                            if trial_cal_method in ("beta", "temperature")
                            else trial_cal_method
                        )

                        # Baseline injection: use prefit calibration to avoid cloning losing baseline odds
                        if (
                            self.use_baseline
                            and model_type == "catboost"
                            and isinstance(model, EnhancedCatBoost)
                        ):
                            odds_train = odds[:train_end]
                            split_idx = int(len(X_train_scaled) * 0.8)
                            X_fit, X_cal = X_train_scaled[:split_idx], X_train_scaled[split_idx:]
                            y_fit, y_cal = y_train[:split_idx], y_train[split_idx:]
                            model.set_baseline_odds(odds_train[:split_idx])
                            sw_fit = (
                                sample_weights[:split_idx] if sample_weights is not None else None
                            )
                            model.fit(X_fit, y_fit, sample_weight=sw_fit)
                            calibrated = CalibratedClassifierCV(
                                model, method=sklearn_cal, cv="prefit"
                            )
                            calibrated.fit(X_cal, y_cal)
                        else:
                            calibrated = CalibratedClassifierCV(
                                model, method=sklearn_cal, cv=_get_calibration_cv(model_type)
                            )
                            # Use sample weights if available (skip for FastAI - it doesn't support them properly)
                            if sample_weights is not None and model_type != "fastai":
                                calibrated.fit(
                                    X_train_scaled, y_train, sample_weight=sample_weights
                                )
                            else:
                                calibrated.fit(X_train_scaled, y_train)

                        proba = calibrated.predict_proba(X_test_scaled)
                        if proba.shape[1] == 1:
                            # CalibratedClassifierCV saw only one class in an inner fold
                            logger.debug(f"predict_proba returned 1 column, skipping fold")
                            continue
                        probs = proba[:, 1]

                        # Apply BetaCalibrator post-hoc recalibration
                        if trial_cal_method == "beta":
                            train_proba = calibrated.predict_proba(X_train_scaled)
                            if train_proba.shape[1] > 1:
                                beta_cal = BetaCalibrator(method="abm")
                                beta_cal.fit(train_proba[:, 1], y_train)
                                probs = beta_cal.transform(probs)
                        elif trial_cal_method == "temperature":
                            train_proba = calibrated.predict_proba(X_train_scaled)
                            if train_proba.shape[1] > 1:
                                from src.calibration.calibration import TemperatureScaling

                                temp_cal = TemperatureScaling()
                                temp_cal.fit(train_proba[:, 1], y_train)
                                probs = temp_cal.transform(probs)
                except Exception as e:
                    logger.warning(f"Trial failed during model fitting ({model_type}): {e}")
                    import traceback

                    logger.warning(f"Traceback: {traceback.format_exc()}")
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

        best_overall = {"precision": float("-inf"), "model": None, "params": None}
        # Store all models' params for stacking ensemble
        self.all_model_params = {}

        # Phase 2 merge: pre-populate with Phase 1 params
        if self.merge_params_path:
            with open(self.merge_params_path) as f:
                loaded = json.load(f)
            self.all_model_params = loaded["all_model_params"]
            self._model_cal_methods = loaded.get("model_cal_methods", {})
            logger.debug(f"  Loaded Phase 1 params for: {list(self.all_model_params.keys())}")

        # Train transfer learning base model if enabled (before Optuna)
        if self.use_transfer_learning:
            import tempfile

            logger.info("  Training CatBoost base model for transfer learning...")
            scaler_tl = StandardScaler()
            X_tl = scaler_tl.fit_transform(X)
            base_cb = CatBoostClassifier(
                iterations=200,
                depth=6,
                learning_rate=0.05,
                random_seed=self.seed,
                verbose=False,
                has_time=True,
            )
            base_cb.fit(X_tl, y)
            self._base_model_path = tempfile.mktemp(suffix=".cbm")
            base_cb.save_model(self._base_model_path)
            logger.info(f"  Base model saved to {self._base_model_path}")

        for model_type in self._get_base_model_types(
            include_fastai=self.use_fastai,
            fast_mode=self.fast_mode,
            include_two_stage=self.use_two_stage,
            only_catboost=self.only_catboost,
            no_catboost=self.no_catboost,
        ):
            if model_type in self.all_model_params:
                logger.debug(f"  Skipping {model_type} (params loaded from Phase 1)")
                continue

            logger.info(f"  Tuning {model_type}...")

            study = optuna.create_study(
                direction="maximize",
                sampler=TPESampler(seed=self.seed),
            )

            # Scale trial counts proportionally (ratios based on per-trial cost)
            if model_type == "catboost":
                n_trials_for_run = max(30, int(self.n_optuna_trials * 0.67))
            elif model_type == "fastai":
                n_trials_for_run = max(10, int(self.n_optuna_trials * 0.13))
            elif model_type.startswith("two_stage_"):
                n_trials_for_run = max(15, int(self.n_optuna_trials * 0.20))
            else:
                n_trials_for_run = self.n_optuna_trials

            if self.fast_mode:
                n_trials_for_run = min(n_trials_for_run, 5)

            objective = self.create_objective(X, y, odds, model_type, dates)

            study.optimize(objective, n_trials=n_trials_for_run, show_progress_bar=True)

            # Store params for each model (for stacking later)
            # Reconstruct structured params from flat Optuna params
            best_params = dict(study.best_params)

            if model_type.startswith("two_stage_"):
                # Reconstruct nested dict from flat Optuna params
                ts_cal = best_params.pop("ts_calibration", "sigmoid")
                min_edge = best_params.pop("min_edge", 0.02)

                if model_type == "two_stage_lgb":
                    s1_params = {
                        "n_estimators": best_params.pop("ts_s1_n_estimators"),
                        "max_depth": best_params.pop("ts_s1_max_depth"),
                        "learning_rate": best_params.pop("ts_learning_rate"),
                        "reg_alpha": best_params.pop("ts_reg_alpha"),
                        "reg_lambda": best_params.pop("ts_reg_lambda"),
                        "random_state": self.seed,
                        "verbose": -1,
                    }
                    s2_params = {
                        "n_estimators": best_params.pop("ts_s2_n_estimators"),
                        "max_depth": best_params.pop("ts_s2_max_depth"),
                        "learning_rate": s1_params["learning_rate"],
                        "reg_alpha": s1_params["reg_alpha"],
                        "reg_lambda": s1_params["reg_lambda"],
                        "random_state": self.seed,
                        "verbose": -1,
                    }
                else:  # two_stage_xgb -> CatBoost
                    s1_params = {
                        "iterations": best_params.pop("ts_s1_iterations"),
                        "depth": best_params.pop("ts_s1_depth"),
                        "learning_rate": best_params.pop("ts_learning_rate"),
                        "l2_leaf_reg": best_params.pop("ts_l2_leaf_reg"),
                        "random_seed": self.seed,
                        "verbose": False,
                    }
                    s2_params = {
                        "iterations": best_params.pop("ts_s2_iterations"),
                        "depth": best_params.pop("ts_s2_depth"),
                        "learning_rate": s1_params["learning_rate"],
                        "l2_leaf_reg": s1_params["l2_leaf_reg"],
                        "random_seed": self.seed,
                        "verbose": False,
                    }

                best_params = {
                    "stage1_params": s1_params,
                    "stage2_params": s2_params,
                    "calibration_method": ts_cal,
                    "min_edge_threshold": min_edge,
                }
                # Pop shared params that leaked into flat space
                best_params.pop("calibration_method_shared", None)
                best_params.pop("decay_rate", None)
                best_params.pop("min_weight", None)
                best_params.pop("uncertainty_penalty", None)
                best_cal_method = ts_cal
            else:
                if model_type == "fastai":
                    best_params["layers"] = [best_params.pop("layer1"), best_params.pop("layer2")]
                    best_params["ps"] = [best_params.pop("ps1"), best_params.pop("ps2")]

                # Extract non-model params (tuned per trial but not passed to model constructor)
                best_cal_method = best_params.pop("calibration_method", "sigmoid")
                best_params.pop("decay_rate", None)
                best_params.pop("min_weight", None)
                best_params.pop("uncertainty_penalty", None)
                best_params.pop("use_monotonic", None)  # Optuna toggle, not a CatBoost arg
                best_params.pop(
                    "ft_iterations", None
                )  # Transfer learning param, not a CatBoost arg

            self.all_model_params[model_type] = best_params
            # Store per-model calibration method
            if not hasattr(self, "_model_cal_methods"):
                self._model_cal_methods = {}
            self._model_cal_methods[model_type] = best_cal_method

            if study.best_value > best_overall["precision"]:
                best_overall = {
                    "precision": study.best_value,
                    "model": model_type,
                    "params": best_params,
                    "calibration_method": best_cal_method,
                    "uncertainty_penalty": study.best_params.get("uncertainty_penalty", 1.0),
                }

            logger.info(
                f"    {model_type}: log_loss={-study.best_value:.4f}, calibration={best_cal_method}"
            )

        # Set calibration method from winning model for downstream use
        winning_cal = best_overall.get("calibration_method", "sigmoid")
        # CalibratedClassifierCV only supports sigmoid/isotonic; beta uses sigmoid + post-hoc
        self._sklearn_cal_method = "sigmoid" if winning_cal == "beta" else winning_cal
        self._use_custom_calibration = winning_cal == "beta"
        self.calibration_method = winning_cal
        # Uncertainty penalty from winning model's best trial
        self._uncertainty_penalty = best_overall.get("uncertainty_penalty", 1.0)

        logger.info(
            f"Best model: {best_overall['model']} (log_loss={-best_overall['precision']:.4f}, calibration={winning_cal})"
        )
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
        from src.ml.metrics import expected_calibration_error, sharpe_ratio, sortino_ratio

        logger.info("Running threshold optimization (including stacking ensemble)...")

        # Use stored dates if not provided
        if dates is None:
            dates = self.dates

        # Generate predictions for ALL models with walk-forward
        n_samples = len(y)
        fold_size = n_samples // (self.n_folds + 1)

        # Collect predictions separately for optimization and held-out folds
        _base_types = self._get_base_model_types(
            include_fastai=self.use_fastai,
            fast_mode=self.fast_mode,
            include_two_stage=self.use_two_stage,
            only_catboost=self.only_catboost,
            no_catboost=self.no_catboost,
        )
        opt_preds = {name: [] for name in _base_types}
        opt_actuals = []
        opt_odds = []

        holdout_preds = {name: [] for name in _base_types}
        holdout_actuals = []
        holdout_odds = []
        holdout_fixture_ids = []
        holdout_dates = []

        # Track validation data for stacking meta-learner training
        val_preds = {name: [] for name in _base_types}
        val_actuals = []

        # Adversarial validation diagnostics
        adv_results = []
        # Per-fold KS test diagnostics
        ks_results = []

        # Per-league calibration tracking
        opt_leagues = []
        holdout_leagues = []

        # Uncertainty tracking (MAPIE conformal prediction)
        opt_uncertainties = []
        holdout_uncertainties = []

        n_opt_folds = self.n_folds - self.n_holdout_folds  # Folds for optimization

        # Generate CV splits based on method (walk_forward or purged_kfold)
        cv_splits = self._get_cv_splits(
            n_samples,
            dates=pd.Series(dates) if dates is not None else None,
        )
        if self.cv_method == "purged_kfold":
            logger.info(
                f"  Using purged CV with {self.embargo_days}-day embargo ({len(cv_splits)} folds)"
            )

        for fold, (_, train_end, test_start, test_end) in enumerate(cv_splits):

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
                logger.debug(f"  Fold {fold} adversarial AUC: {adv_auc:.3f} (>0.6 = shift)")
                if adv_auc > 0.7:
                    logger.warning(
                        f"  Significant distribution shift! Top features: {shift_features[:5]}"
                    )
            except Exception as e:
                logger.debug(f"  Adversarial validation failed: {e}")

            # Per-fold KS test: detect which features shift between train/test
            try:
                ks_result = self._per_fold_ks_test(
                    X_train_scaled, X_test_scaled, self.optimal_features
                )
                ks_result["fold"] = fold
                ks_results.append(ks_result)
                if ks_result["high_shift"]:
                    logger.warning(
                        f"  Fold {fold} HIGH SHIFT: {ks_result['n_shifted']}/{ks_result['n_tested']} features shifted"
                    )
                else:
                    logger.debug(
                        f"  Fold {fold} KS: {ks_result['n_shifted']}/{ks_result['n_tested']} features shifted"
                    )
            except Exception as e:
                logger.debug(f"  KS test failed: {e}")

            # Track league membership for per-league calibration
            is_holdout = fold >= self.n_folds - self.n_holdout_folds
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
                        from src.ml.two_stage_model import create_two_stage_model

                        base = "lightgbm" if model_type == "two_stage_lgb" else "catboost"
                        ts_params = self.all_model_params[model_type]
                        ts_model = create_two_stage_model(
                            base,
                            stage1_params=ts_params.get("stage1_params"),
                            stage2_params=ts_params.get("stage2_params"),
                            calibration_method=ts_params.get("calibration_method", "sigmoid"),
                            min_edge_threshold=ts_params.get("min_edge_threshold", 0.02),
                        )
                        odds_train = odds[:train_end]
                        ts_model.fit(X_train_scaled, y_train, odds_train)
                        result_dict = ts_model.predict_proba(X_test_scaled, odds_test)
                        probs = result_dict["combined_score"]
                        target_preds[model_type].extend(probs)
                        if not is_holdout:
                            val_odds = odds[train_end - n_val : train_end]
                            val_result = ts_model.predict_proba(X_val_scaled, val_odds)
                            val_preds[model_type].extend(val_result["combined_score"])
                        continue
                    else:
                        model = self._create_model_instance(
                            model_type, self.all_model_params[model_type], seed=self.seed
                        )
                        cal_method = getattr(self, "_model_cal_methods", {}).get(
                            model_type, self._sklearn_cal_method
                        )
                        # Beta calibration: use sigmoid for sklearn, apply BetaCalibrator post-hoc
                        sklearn_cal = "sigmoid" if cal_method == "beta" else cal_method
                        calibrated = CalibratedClassifierCV(
                            model, method=sklearn_cal, cv=_get_calibration_cv(model_type)
                        )
                        # Skip sample_weights for FastAI - it doesn't support them properly
                        if sample_weights is not None and model_type != "fastai":
                            calibrated.fit(X_train_scaled, y_train, sample_weight=sample_weights)
                        else:
                            calibrated.fit(X_train_scaled, y_train)
                        proba = calibrated.predict_proba(X_test_scaled)
                        if proba.shape[1] == 1:
                            logger.warning(
                                f"  {model_type} fold: predict_proba returned 1 column, skipping"
                            )
                            continue
                        probs = proba[:, 1]

                        # Apply BetaCalibrator post-hoc recalibration
                        beta_cal = None
                        if cal_method == "beta":
                            train_proba = calibrated.predict_proba(X_train_scaled)
                            if train_proba.shape[1] > 1:
                                beta_cal = BetaCalibrator(method="abm")
                                beta_cal.fit(train_proba[:, 1], y_train)
                                probs = beta_cal.transform(probs)
                except Exception as e:
                    logger.warning(f"  {model_type} fold failed: {e}")
                    continue

                target_preds[model_type].extend(probs)

                # Also get validation predictions for stacking (only from opt folds)
                if not is_holdout:
                    val_probs = calibrated.predict_proba(X_val_scaled)[:, 1]
                    if beta_cal is not None:
                        val_probs = beta_cal.transform(val_probs)
                    val_preds[model_type].extend(val_probs)

            # Collect uncertainty estimates via MAPIE (using best single model's calibrated output)
            best_single = (
                self.best_model_type if self.best_model_type in self.all_model_params else None
            )
            if (
                best_single
                and best_single in self.all_model_params
                and not best_single.startswith("two_stage_")
            ):
                try:
                    from src.ml.uncertainty import ConformalClassifier

                    # Use the calibrated model already trained above (re-train for MAPIE)
                    mapie_model = self._create_model_instance(
                        best_single, self.all_model_params[best_single], seed=self.seed
                    )
                    mapie_cal = CalibratedClassifierCV(
                        mapie_model,
                        method=self._sklearn_cal_method,
                        cv=_get_calibration_cv(best_single),
                    )
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
                if self.fixture_ids is not None:
                    holdout_fixture_ids.extend(self.fixture_ids[test_start:test_end])
                holdout_dates.extend(self.dates[test_start:test_end])
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

            # Stacking ensemble with non-negative Ridge meta-learner
            # Non-negative constraint prevents extreme/inverted weights (e.g., LGB=-6, CB=+10)
            # that caused overfitting in previous runs.
            meta = None
            try:
                from sklearn.model_selection import cross_val_score

                best_alpha = 1.0
                best_score = -np.inf
                for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
                    ridge = Ridge(alpha=alpha, positive=True, fit_intercept=True)
                    scores = cross_val_score(
                        ridge, val_stack, y_val_arr, cv=min(3, len(y_val_arr)), scoring="r2"
                    )
                    mean_score = scores.mean()
                    if mean_score > best_score:
                        best_score = mean_score
                        best_alpha = alpha

                meta = Ridge(alpha=best_alpha, positive=True, fit_intercept=True)
                meta.fit(val_stack, y_val_arr)
                stacking_raw = np.atleast_1d(meta.predict(opt_stack))
                stacking_proba = np.clip(stacking_raw, 0.0, 1.0)
                opt_preds["stacking"] = stacking_proba.tolist()
                self._stacking_weights = dict(zip(base_model_names, meta.coef_.tolist()))
                self._stacking_alpha = float(best_alpha)
                logger.info(
                    f"  Stacking trained (non-negative Ridge) weights: {self._stacking_weights}, alpha={self._stacking_alpha}"
                )
            except Exception as e:
                logger.warning(f"  Stacking failed: {e}")
                opt_preds["stacking"] = opt_preds["average"]

            # DisagreementEnsemble: only bet when models agree with each other
            # AND disagree with the market. Test conservative/balanced/aggressive presets.
            market_probs = np.clip(1.0 / opt_odds_arr, 0.05, 0.95)
            for strategy in ["conservative", "balanced", "aggressive"]:
                try:
                    # Build models list from individual predictions (no refit needed)
                    # Use a lightweight wrapper that returns pre-computed probabilities
                    class _PrecomputedModel:
                        def __init__(self, probs):
                            self._probs = probs

                        def predict_proba(self, X):
                            return np.column_stack([1 - self._probs, self._probs])

                    models = [_PrecomputedModel(np.array(opt_preds[m])) for m in base_model_names]
                    ensemble = create_disagreement_ensemble(
                        models, base_model_names, strategy=strategy
                    )
                    result = ensemble.predict_with_disagreement(
                        np.zeros((len(opt_odds_arr), 1)),  # X unused by precomputed models
                        market_probs,
                    )
                    opt_preds[f"disagree_{strategy}"] = result["avg_prob"].tolist()
                    # For non-signal samples, zero out probability to prevent betting
                    signal_probs = np.where(result["bet_signal"], result["avg_prob"], 0.0)
                    opt_preds[f"disagree_{strategy}_filtered"] = signal_probs.tolist()
                    n_signals = result["bet_signal"].sum()
                    logger.debug(
                        f"  Disagreement ({strategy}): {n_signals} bet signals "
                        f"({n_signals/len(opt_odds_arr)*100:.1f}%)"
                    )
                except Exception as e:
                    logger.warning(f"  Disagreement ({strategy}) failed: {e}")

            # Also keep simple agreement (backward compatible)
            opt_preds["agreement"] = np.min(opt_stack, axis=1).tolist()
            logger.debug(
                f"  Agreement ensemble: uses minimum probability across {base_model_names}"
            )
        else:
            meta = None
            logger.warning("  Not enough models for stacking, using best single model only")

        # Temporal blending: blend full-history and recent-only models
        # Only activate with sufficient training data (2000+ samples)
        if (
            n_samples >= 2000
            and self.best_model_type in self.all_model_params
            and not self.best_model_type.startswith("two_stage_")
        ):
            try:
                blend_opt_preds = []
                blend_holdout_preds = []
                for fold, (_, train_end, test_start, test_end) in enumerate(cv_splits):
                    X_train_f, y_train_f = X[:train_end], y[:train_end]
                    X_test_f = X[test_start:test_end]
                    if len(X_train_f) < 600 or len(X_test_f) < 20:
                        continue

                    scaler_b = StandardScaler()
                    X_train_scaled_b = scaler_b.fit_transform(X_train_f)
                    X_test_scaled_b = scaler_b.transform(X_test_f)

                    # Full-history model
                    model_full = self._create_model_instance(
                        self.best_model_type,
                        self.all_model_params[self.best_model_type],
                        seed=self.seed,
                    )
                    cal_full = CalibratedClassifierCV(
                        model_full,
                        method=self._sklearn_cal_method,
                        cv=_get_calibration_cv(self.best_model_type),
                    )
                    cal_full.fit(X_train_scaled_b, y_train_f)
                    probs_full = self._safe_predict_proba(cal_full, X_test_scaled_b)
                    if probs_full is None:
                        continue

                    # Recent-only model (last 30% of training)
                    cutoff = int(len(X_train_f) * 0.7)
                    model_recent = self._create_model_instance(
                        self.best_model_type,
                        self.all_model_params[self.best_model_type],
                        seed=self.seed,
                    )
                    cal_recent = CalibratedClassifierCV(
                        model_recent,
                        method=self._sklearn_cal_method,
                        cv=_get_calibration_cv(self.best_model_type),
                    )
                    cal_recent.fit(X_train_scaled_b[cutoff:], y_train_f[cutoff:])
                    probs_recent = self._safe_predict_proba(cal_recent, X_test_scaled_b)
                    if probs_recent is None:
                        continue

                    # Blend with alpha=0.4 (slightly favoring recent data)
                    blend_alpha = 0.4
                    blended = blend_alpha * probs_recent + (1 - blend_alpha) * probs_full

                    is_holdout = fold >= self.n_folds - self.n_holdout_folds
                    if is_holdout:
                        blend_holdout_preds.extend(blended)
                    else:
                        blend_opt_preds.extend(blended)

                if blend_opt_preds:
                    opt_preds["temporal_blend"] = blend_opt_preds
                    logger.debug(
                        f"  Temporal blend: {len(blend_opt_preds)} predictions (alpha=0.4)"
                    )
                if blend_holdout_preds:
                    holdout_preds["temporal_blend"] = blend_holdout_preds
            except Exception as e:
                logger.warning(f"  Temporal blending failed: {e}")

        # Log uncertainty collection status
        non_default_unc = sum(1 for u in opt_uncertainties if u != 0.5)
        if non_default_unc > 0:
            logger.debug(
                f"  MAPIE uncertainty: {non_default_unc}/{len(opt_uncertainties)} predictions with real estimates"
            )
        elif opt_uncertainties:
            logger.debug(
                f"  MAPIE uncertainty: all {len(opt_uncertainties)} predictions used default (0.5)"
            )

        # Deflated Sharpe Ratio helper — penalizes for multiple testing
        # Based on Harvey & Liu (2015) "Haircut Sharpe Ratios"
        n_total_configs = 0  # Will be counted during grid search

        def _deflate_sharpe(sr: float, n_configs: int, n_obs: int) -> float:
            """Deflate Sharpe ratio for multiple testing (Haircut SR formula).

            sr: raw Sharpe ratio
            n_configs: number of configurations tested
            n_obs: number of observations (bets)
            Returns: deflated Sharpe ratio
            """
            if sr <= 0 or n_configs <= 1 or n_obs < 10:
                return sr
            # Expected max Sharpe under null for N independent tests
            # E[max(Z_1..Z_N)] ≈ sqrt(2 * ln(N)) for standard normals
            import math

            expected_max_sr = math.sqrt(2 * math.log(n_configs))
            # Haircut: subtract the expected inflation from random search
            # Scale by sqrt(n_obs) since SR estimation error ~ 1/sqrt(n)
            haircut = expected_max_sr / max(math.sqrt(n_obs), 1)
            deflated = max(0.0, sr - haircut)
            return deflated

        # Grid search on OPTIMIZATION SET (folds 0..N-2)
        threshold_search = self.config["threshold_search"]
        # Per-market odds bounds (fall back to globals if not defined)
        min_odds_search = self.config.get("min_odds_search", MIN_ODDS_SEARCH)
        max_odds_search = self.config.get("max_odds_search", MAX_ODDS_SEARCH)
        if self.use_odds_threshold:
            alpha_search = [0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4]
            configurations = list(
                product(threshold_search, min_odds_search, max_odds_search, alpha_search)
            )
            logger.info(
                f"  Grid search: {len(configurations)} configs (incl. alpha search {alpha_search})"
            )
        else:
            configurations = list(product(threshold_search, min_odds_search, max_odds_search))

        _ensemble_methods = [
            "stacking",
            "average",
            "agreement",
            "disagree_conservative_filtered",
            "disagree_balanced_filtered",
            "disagree_aggressive_filtered",
            "temporal_blend",
        ]
        all_models = [
            m for m in _base_types + _ensemble_methods if m in opt_preds and len(opt_preds[m]) > 0
        ]

        logger.info(f"  Testing models: {all_models}")
        n_total_configs = len(all_models) * len(configurations)
        logger.info(
            f"  Total configs to test: {n_total_configs} ({len(all_models)} models × {len(configurations)} threshold combos)"
        )

        best_result = {
            "precision": 0.0,
            "roi": -100.0,
            "sharpe_roi": -100.0,
            "model": self.best_model_type,
        }

        for model_name in all_models:
            preds = np.array(opt_preds[model_name])

            # Skip models whose predictions don't align with optimization set
            # (e.g., temporal_blend predicts on a subset of folds)
            if len(preds) != len(opt_odds_arr):
                logger.warning(
                    f"  Skipping {model_name}: {len(preds)} preds vs {len(opt_odds_arr)} opt samples"
                )
                continue

            for config in configurations:
                if self.use_odds_threshold:
                    threshold, min_odds, max_odds, alpha = config
                    if alpha > 0:
                        adj_thresholds = self.calculate_odds_adjusted_threshold(
                            threshold, opt_odds_arr, alpha=alpha
                        )
                        mask = (
                            (preds >= adj_thresholds)
                            & (opt_odds_arr >= min_odds)
                            & (opt_odds_arr <= max_odds)
                        )
                    else:
                        mask = (
                            (preds >= threshold)
                            & (opt_odds_arr >= min_odds)
                            & (opt_odds_arr <= max_odds)
                        )
                else:
                    threshold, min_odds, max_odds = config
                    mask = (
                        (preds >= threshold)
                        & (opt_odds_arr >= min_odds)
                        & (opt_odds_arr <= max_odds)
                    )
                n_bets = mask.sum()

                if n_bets < self.min_bets:
                    continue

                wins = opt_actuals_arr[mask].sum()
                precision = wins / n_bets

                returns = np.where(opt_actuals_arr[mask] == 1, opt_odds_arr[mask] - 1, -1)
                roi = returns.mean() * 100 if len(returns) > 0 else -100.0

                # Uncertainty-adjusted ROI (MAPIE): weight returns by confidence
                uncertainty_roi = roi  # Default: same as flat ROI
                if opt_uncertainties_arr is not None and len(opt_uncertainties_arr) == len(
                    opt_actuals_arr
                ):
                    from src.ml.uncertainty import batch_adjust_stakes

                    bet_uncertainties = opt_uncertainties_arr[mask]
                    stakes = batch_adjust_stakes(
                        np.ones(n_bets),
                        bet_uncertainties,
                        uncertainty_penalty=getattr(self, "_uncertainty_penalty", 1.0),
                    )
                    if stakes.sum() > 0:
                        uncertainty_roi = (returns * stakes).sum() / stakes.sum() * 100

                # Multi-objective: combine ROI with Sharpe ratio for risk-adjusted selection
                # sharpe_roi = ROI * min(1, sharpe / 1.5)
                # This penalizes high-ROI configs with high variance (ruin risk)
                sr = sharpe_ratio(returns)
                # Deflate Sharpe for multiple testing (Harvey & Liu, 2015)
                deflated_sr = _deflate_sharpe(sr, n_total_configs, n_bets)
                sharpe_mult = min(1.0, deflated_sr / 1.5) if deflated_sr > 0 else 0.0
                sharpe_roi = roi * sharpe_mult if roi > 0 else roi

                # ECE penalty: penalize overconfident configs (calibration-first)
                # 0 penalty if ECE < 5%, linearly increasing, full penalty at ECE > 15%
                ece = expected_calibration_error(opt_actuals_arr[mask], preds[mask])
                if ece > self.max_ece:
                    continue  # Hard-reject configs with ECE above threshold
                ece_penalty = max(0.0, (ece - 0.05) / 0.10)
                calibrated_sharpe_roi = sharpe_roi * (1 - ece_penalty)

                # Brier Score (calibration + sharpness in one metric)
                from sklearn.metrics import brier_score_loss

                brier = brier_score_loss(opt_actuals_arr[mask], preds[mask])

                # FVA vs market-implied baseline
                # implied_prob = 1/odds (already clipped in market_probs)
                market_probs_bet = np.clip(1.0 / opt_odds_arr[mask], 0.05, 0.95)
                brier_market = brier_score_loss(opt_actuals_arr[mask], market_probs_bet)
                fva = 1.0 - (brier / brier_market) if brier_market > 0 else 0.0

                min_precision = 0.60  # Minimum viable precision floor
                if precision >= min_precision and (
                    calibrated_sharpe_roi > best_result["sharpe_roi"]
                    or (
                        calibrated_sharpe_roi == best_result["sharpe_roi"]
                        and precision > best_result["precision"]
                    )
                ):
                    best_result = {
                        "model": model_name,
                        "threshold": threshold,
                        "min_odds": min_odds,
                        "max_odds": max_odds,
                        "precision": precision,
                        "roi": roi,
                        "uncertainty_roi": uncertainty_roi,
                        "sharpe_roi": calibrated_sharpe_roi,
                        "ece": ece,
                        "brier": brier,
                        "fva": fva,
                        "n_bets": int(n_bets),
                        "n_wins": int(wins),
                        "alpha": alpha if self.use_odds_threshold else None,
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

        uncertainty_roi_val = best_result.get("uncertainty_roi", best_result["roi"])
        unc_suffix = ""
        if abs(uncertainty_roi_val - best_result["roi"]) > 0.1:
            unc_suffix = f", Uncertainty-adj ROI: {uncertainty_roi_val:.1f}%"
        alpha_suffix = (
            f", alpha: {best_result['alpha']:.2f}" if best_result.get("alpha") is not None else ""
        )
        ece_suffix = (
            f", ECE: {best_result['ece']:.3f}" if best_result.get("ece") is not None else ""
        )
        brier_suffix = (
            f", Brier: {best_result['brier']:.4f}" if best_result.get("brier") is not None else ""
        )
        fva_suffix = (
            f", FVA: {best_result['fva']:+.3f}" if best_result.get("fva") is not None else ""
        )
        logger.info(
            f"Optimization set - Best model: {final_model}, threshold: {best_result['threshold']}{alpha_suffix}, "
            f"precision: {best_result['precision']*100:.1f}%, "
            f"ROI: {best_result['roi']:.1f}%, Sharpe-ROI: {best_result.get('sharpe_roi', 0):.1f}%{ece_suffix}{brier_suffix}{fva_suffix}{unc_suffix}"
        )

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
                    ho_raw = np.atleast_1d(meta.predict(ho_stack))
                    holdout_preds["stacking"] = np.clip(ho_raw, 0.0, 1.0).tolist()
                except Exception:
                    holdout_preds["stacking"] = holdout_preds["average"]

            holdout_preds["agreement"] = np.min(ho_stack, axis=1).tolist()

            # DisagreementEnsemble for holdout set
            ho_market_probs = np.clip(1.0 / holdout_odds_arr, 0.05, 0.95)
            for strategy in ["conservative", "balanced", "aggressive"]:
                try:

                    class _PrecomputedModel:
                        def __init__(self, probs):
                            self._probs = probs

                        def predict_proba(self, X):
                            return np.column_stack([1 - self._probs, self._probs])

                    models = [
                        _PrecomputedModel(np.array(holdout_preds[m])) for m in holdout_base_names
                    ]
                    ensemble = create_disagreement_ensemble(
                        models, holdout_base_names, strategy=strategy
                    )
                    result = ensemble.predict_with_disagreement(
                        np.zeros((len(holdout_odds_arr), 1)),
                        ho_market_probs,
                    )
                    holdout_preds[f"disagree_{strategy}"] = result["avg_prob"].tolist()
                    signal_probs = np.where(result["bet_signal"], result["avg_prob"], 0.0)
                    holdout_preds[f"disagree_{strategy}_filtered"] = signal_probs.tolist()
                except Exception:
                    pass

        # Apply best thresholds to held-out fold
        if (
            final_model in holdout_preds
            and len(holdout_preds[final_model]) > 0
            and len(holdout_actuals_arr) > 0
        ):
            ho_preds_arr = np.array(holdout_preds[final_model])
            # Apply odds-adjusted thresholds when enabled (consistent with grid search)
            best_alpha = best_result.get("alpha", 0) or 0
            if self.use_odds_threshold and best_alpha > 0:
                ho_adj_thresholds = self.calculate_odds_adjusted_threshold(
                    best_result["threshold"], holdout_odds_arr, alpha=best_alpha
                )
                ho_mask = (
                    (ho_preds_arr >= ho_adj_thresholds)
                    & (holdout_odds_arr >= best_result["min_odds"])
                    & (holdout_odds_arr <= best_result["max_odds"])
                )
            else:
                ho_mask = (
                    (ho_preds_arr >= best_result["threshold"])
                    & (holdout_odds_arr >= best_result["min_odds"])
                    & (holdout_odds_arr <= best_result["max_odds"])
                )
            ho_n_bets = ho_mask.sum()

            if ho_n_bets > 0:
                ho_wins = holdout_actuals_arr[ho_mask].sum()
                ho_precision = ho_wins / ho_n_bets
                ho_returns = np.where(
                    holdout_actuals_arr[ho_mask] == 1, holdout_odds_arr[ho_mask] - 1, -1
                )
                ho_roi = ho_returns.mean() * 100

                ho_sharpe = sharpe_ratio(ho_returns)
                ho_sortino = sortino_ratio(ho_returns)
                ho_ece = expected_calibration_error(holdout_actuals_arr, ho_preds_arr)

                # Holdout Brier Score + FVA
                from sklearn.metrics import brier_score_loss

                ho_brier = brier_score_loss(holdout_actuals_arr[ho_mask], ho_preds_arr[ho_mask])
                ho_market_probs_bet = np.clip(1.0 / holdout_odds_arr[ho_mask], 0.05, 0.95)
                ho_brier_market = brier_score_loss(
                    holdout_actuals_arr[ho_mask], ho_market_probs_bet
                )
                ho_fva = 1.0 - (ho_brier / ho_brier_market) if ho_brier_market > 0 else 0.0

                logger.info(f"Held-out fold (UNBIASED) - {final_model}:")
                logger.info(f"  Precision: {ho_precision*100:.1f}% ({int(ho_wins)}/{ho_n_bets})")
                logger.info(f"  ROI: {ho_roi:.1f}%")
                logger.info(
                    f"  Sharpe: {ho_sharpe:.3f}, Sortino: {ho_sortino:.3f}, ECE: {ho_ece:.4f}"
                )
                logger.info(f"  Brier: {ho_brier:.4f}, FVA: {ho_fva:+.3f}")

                # Bootstrap confidence interval for holdout ROI
                rng = np.random.RandomState(self.seed)
                n_bootstrap = 1000
                bootstrap_rois = []
                for _ in range(n_bootstrap):
                    idx = rng.choice(len(ho_returns), len(ho_returns), replace=True)
                    bootstrap_rois.append(np.mean(ho_returns[idx]) * 100)
                roi_ci_lower = float(np.percentile(bootstrap_rois, 2.5))
                roi_ci_upper = float(np.percentile(bootstrap_rois, 97.5))
                logger.info(f"  ROI 95% CI: [{roi_ci_lower:.1f}%, {roi_ci_upper:.1f}%]")
                if roi_ci_lower < 0:
                    logger.warning("  CI lower bound < 0% — not significantly profitable")

                # Holdout uncertainty-adjusted ROI
                ho_uncertainty_roi = ho_roi
                if holdout_uncertainties and len(holdout_uncertainties) == len(
                    holdout_actuals_arr
                ):
                    from src.ml.uncertainty import batch_adjust_stakes

                    ho_unc_arr = np.array(holdout_uncertainties)
                    ho_bet_unc = ho_unc_arr[ho_mask]
                    if len(ho_bet_unc) == ho_n_bets:
                        unc_penalty = getattr(self, "_uncertainty_penalty", 1.0)
                        ho_stakes = batch_adjust_stakes(
                            np.ones(ho_n_bets), ho_bet_unc, uncertainty_penalty=unc_penalty
                        )
                        if ho_stakes.sum() > 0:
                            ho_uncertainty_roi = (
                                (ho_returns * ho_stakes).sum() / ho_stakes.sum() * 100
                            )
                    logger.info(f"  Uncertainty-adj ROI: {ho_uncertainty_roi:.1f}%")

                # Store held-out metrics for downstream use
                self._holdout_metrics = {
                    "precision": float(ho_precision),
                    "roi": float(ho_roi),
                    "roi_ci_lower": roi_ci_lower,
                    "roi_ci_upper": roi_ci_upper,
                    "n_bets": int(ho_n_bets),
                    "n_wins": int(ho_wins),
                    "sharpe": float(ho_sharpe),
                    "sortino": float(ho_sortino),
                    "ece": float(ho_ece),
                    "brier": float(ho_brier),
                    "fva": float(ho_fva),
                    "uncertainty_roi": float(ho_uncertainty_roi),
                    "uncertainty_penalty": float(
                        getattr(self, "_uncertainty_penalty", 1.0)
                    ),
                }
            else:
                logger.info("Held-out fold: No qualifying bets with selected thresholds")
                self._holdout_metrics = {}
        else:
            logger.info("Held-out fold: No predictions available for selected model")
            self._holdout_metrics = {}

        # --- Save holdout predictions CSV for weekend backtest ---
        if (
            final_model in holdout_preds
            and len(holdout_preds[final_model]) > 0
            and len(holdout_actuals_arr) > 0
        ):
            ho_preds_export = np.array(holdout_preds[final_model])
            holdout_csv = pd.DataFrame(
                {
                    "date": holdout_dates,
                    "fixture_id": holdout_fixture_ids if holdout_fixture_ids else range(len(holdout_dates)),
                    "league": list(holdout_leagues) if holdout_leagues else ["unknown"] * len(holdout_dates),
                    "prob": ho_preds_export,
                    "odds": holdout_odds_arr,
                    "actual": holdout_actuals_arr,
                    "market": self.bet_type,
                    "threshold": best_result["threshold"],
                    "model": final_model,
                    "qualifies": ho_mask if len(ho_mask) == len(holdout_dates) else False,
                }
            )
            holdout_csv_path = OUTPUT_DIR / f"holdout_preds_{self.bet_type}.csv"
            holdout_csv_path.parent.mkdir(parents=True, exist_ok=True)
            holdout_csv.to_csv(holdout_csv_path, index=False)
            logger.info(
                f"Saved holdout predictions: {holdout_csv_path} "
                f"({len(holdout_csv)} total, {holdout_csv['qualifies'].sum()} qualifying)"
            )

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
                    self._per_league_ece[str(league)] = float(metrics["ece"])
            if self._per_league_ece:
                logger.debug(f"Per-league ECE: {self._per_league_ece}")

        # Calibration validation: check if calibrated predictions have acceptable ECE
        self._calibration_validation = None
        if final_model in opt_preds and len(opt_preds[final_model]) > 0:
            from src.ml.calibration_validator import validate_calibration

            cal_result = validate_calibration(
                opt_actuals_arr,
                np.array(opt_preds[final_model]),
                method_used=self.calibration_method,
                ece_threshold=0.10,
            )
            self._calibration_validation = cal_result

        # Store adversarial validation results
        self._adv_results = adv_results

        # Store per-fold KS test results
        self._per_fold_ks = ks_results if ks_results else None

        # Store Brier Score + FVA from best config for SniperResult
        self._brier_score = best_result.get("brier")
        self._fva = best_result.get("fva")

        # Update threshold_alpha with best alpha from grid search (used by walk-forward)
        if self.use_odds_threshold and best_result.get("alpha") is not None:
            self.threshold_alpha = best_result["alpha"]
            logger.debug(f"  Best alpha from grid search: {self.threshold_alpha:.2f}")

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
        wf_results = []

        # Use unified CV splits (respects date-based embargo)
        cv_splits = self._get_cv_splits(
            n_samples,
            dates=pd.Series(self.dates) if self.dates is not None else None,
        )

        for fold, (_, train_end, test_start, test_end) in enumerate(cv_splits):
            if test_end <= test_start or (test_end - test_start) < 20:
                continue

            X_train, y_train = X[:train_end], y[:train_end]
            X_test, y_test = X[test_start:test_end], y[test_start:test_end]
            odds_test = odds[test_start:test_end]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Calculate sample weights for walk-forward training (consistent with Optuna/threshold optimization)
            wf_sample_weights = None
            if self.use_sample_weights and self.dates is not None:
                train_dates = pd.to_datetime(self.dates[:train_end])
                wf_sample_weights = self.calculate_sample_weights(train_dates)

            # Train all base models
            fold_preds = {}
            for model_type in self._get_base_model_types(
                include_fastai=self.use_fastai,
                fast_mode=self.fast_mode,
                include_two_stage=self.use_two_stage,
                only_catboost=self.only_catboost,
                no_catboost=self.no_catboost,
            ):
                if model_type not in self.all_model_params:
                    continue

                try:
                    if model_type.startswith("two_stage_"):
                        from src.ml.two_stage_model import create_two_stage_model

                        base = "lightgbm" if model_type == "two_stage_lgb" else "catboost"
                        ts_params = self.all_model_params[model_type]
                        ts_model = create_two_stage_model(
                            base,
                            stage1_params=ts_params.get("stage1_params"),
                            stage2_params=ts_params.get("stage2_params"),
                            calibration_method=ts_params.get("calibration_method", "sigmoid"),
                            min_edge_threshold=ts_params.get("min_edge_threshold", 0.02),
                        )
                        odds_train = odds[:train_end]
                        ts_model.fit(X_train_scaled, y_train, odds_train)
                        result_dict = ts_model.predict_proba(X_test_scaled, odds_test)
                        fold_preds[model_type] = result_dict["combined_score"]
                    else:
                        model = self._create_model_instance(
                            model_type, self.all_model_params[model_type], seed=self.seed
                        )
                        cal_method = getattr(self, "_model_cal_methods", {}).get(
                            model_type, self._sklearn_cal_method
                        )
                        # Beta calibration: use sigmoid for sklearn, apply BetaCalibrator post-hoc
                        sklearn_cal = "sigmoid" if cal_method == "beta" else cal_method
                        calibrated = CalibratedClassifierCV(
                            model, method=sklearn_cal, cv=_get_calibration_cv(model_type)
                        )
                        # Use sample weights if available (skip for FastAI)
                        if wf_sample_weights is not None and model_type != "fastai":
                            calibrated.fit(X_train_scaled, y_train, sample_weight=wf_sample_weights)
                        else:
                            calibrated.fit(X_train_scaled, y_train)
                        probs = self._safe_predict_proba(calibrated, X_test_scaled)
                        if probs is None:
                            logger.warning(
                                f"  {model_type} walkforward fold: predict_proba returned 1 column, skipping"
                            )
                            continue

                        # Apply BetaCalibrator post-hoc recalibration
                        if cal_method == "beta":
                            train_proba = calibrated.predict_proba(X_train_scaled)
                            if train_proba.shape[1] > 1:
                                beta_cal = BetaCalibrator(method="abm")
                                beta_cal.fit(train_proba[:, 1], y_train)
                                probs = beta_cal.transform(probs)

                        fold_preds[model_type] = probs
                except Exception as e:
                    logger.warning(f"  {model_type} walkforward fold failed: {e}")
                    continue

            # Create ensemble predictions
            if len(fold_preds) >= 2:
                base_preds = np.column_stack(list(fold_preds.values()))
                fold_preds["stacking"] = np.mean(base_preds, axis=1)  # Simple average for fold
                fold_preds["average"] = np.mean(base_preds, axis=1)
                fold_preds["agreement"] = np.min(
                    base_preds, axis=1
                )  # Min across models (conservative)

                # DisagreementEnsemble for walkforward
                wf_market_probs = np.clip(1.0 / odds_test, 0.05, 0.95)
                base_names = list(fold_preds.keys())[
                    : len(fold_preds) - 3
                ]  # Exclude stacking/average/agreement
                for strategy in ["conservative", "balanced", "aggressive"]:
                    try:

                        class _PrecomputedModel:
                            def __init__(self, probs):
                                self._probs = probs

                            def predict_proba(self, X):
                                return np.column_stack([1 - self._probs, self._probs])

                        models = [_PrecomputedModel(fold_preds[m]) for m in base_names]
                        ensemble = create_disagreement_ensemble(
                            models, base_names, strategy=strategy
                        )
                        result = ensemble.predict_with_disagreement(
                            np.zeros((len(odds_test), 1)),
                            wf_market_probs,
                        )
                        signal_probs = np.where(result["bet_signal"], result["avg_prob"], 0.0)
                        fold_preds[f"disagree_{strategy}_filtered"] = signal_probs
                    except Exception:
                        pass

                # Temporal blend for walk-forward
                if (
                    len(X_train) >= 2000
                    and self.best_model_type in self.all_model_params
                    and not self.best_model_type.startswith("two_stage_")
                ):
                    try:
                        # Full-history model
                        model_full = self._create_model_instance(
                            self.best_model_type,
                            self.all_model_params[self.best_model_type],
                            seed=self.seed,
                        )
                        cal_full = CalibratedClassifierCV(
                            model_full,
                            method=self._sklearn_cal_method,
                            cv=_get_calibration_cv(self.best_model_type),
                        )
                        cal_full.fit(X_train_scaled, y_train)
                        probs_full = self._safe_predict_proba(cal_full, X_test_scaled)
                        if probs_full is None:
                            raise ValueError("predict_proba returned 1 column")

                        # Recent-only model (last 30% of training)
                        cutoff = int(len(X_train) * 0.7)
                        model_recent = self._create_model_instance(
                            self.best_model_type,
                            self.all_model_params[self.best_model_type],
                            seed=self.seed,
                        )
                        cal_recent = CalibratedClassifierCV(
                            model_recent,
                            method=self._sklearn_cal_method,
                            cv=_get_calibration_cv(self.best_model_type),
                        )
                        cal_recent.fit(X_train_scaled[cutoff:], y_train[cutoff:])
                        probs_recent = self._safe_predict_proba(cal_recent, X_test_scaled)
                        if probs_recent is None:
                            raise ValueError("predict_proba returned 1 column")

                        fold_preds["temporal_blend"] = 0.4 * probs_recent + 0.6 * probs_full
                    except Exception:
                        pass

            # Evaluate each model on this fold
            for model_name, proba in fold_preds.items():
                # Apply odds-adjusted thresholds when enabled (consistent with grid search)
                if self.use_odds_threshold and self.threshold_alpha > 0:
                    wf_adj_thresholds = self.calculate_odds_adjusted_threshold(threshold, odds_test)
                    bet_mask = (
                        (proba >= wf_adj_thresholds)
                        & (odds_test >= min_odds)
                        & (odds_test <= max_odds)
                    )
                else:
                    bet_mask = (
                        (proba >= threshold) & (odds_test >= min_odds) & (odds_test <= max_odds)
                    )
                n_bets = bet_mask.sum()

                if n_bets >= 5:
                    wins = y_test[bet_mask] == 1
                    profit = (wins * (odds_test[bet_mask] - 1) - (~wins) * 1).sum()
                    roi = profit / n_bets * 100
                    precision = wins.mean()

                    wf_results.append(
                        {
                            "fold": fold,
                            "model": model_name,
                            "n_bets": int(n_bets),
                            "wins": int(wins.sum()),
                            "precision": float(precision),
                            "roi": float(roi),
                        }
                    )

        # Summarize results
        if not wf_results:
            logger.warning("No walk-forward results (insufficient data per fold)")
            return {}

        wf_df = pd.DataFrame(wf_results)

        logger.info("\nWalk-Forward Validation Summary:")
        logger.info("-" * 60)

        summary = {}
        for model_name in wf_df["model"].unique():
            model_wf = wf_df[wf_df["model"] == model_name]
            avg_roi = model_wf["roi"].mean()
            std_roi = model_wf["roi"].std()
            avg_precision = model_wf["precision"].mean()
            total_bets = model_wf["n_bets"].sum()

            summary[model_name] = {
                "avg_roi": float(avg_roi),
                "std_roi": float(std_roi),
                "avg_precision": float(avg_precision),
                "total_bets": int(total_bets),
                "n_folds": len(model_wf),
            }

            logger.info(
                f"  {model_name:12}: ROI={avg_roi:+6.1f}% (+/-{std_roi:5.1f}%), "
                f"Precision={avg_precision:.1%}, Bets={total_bets}"
            )

        return {
            "summary": summary,
            "all_folds": wf_df.to_dict("records"),
            "best_model_wf": max(summary.items(), key=lambda x: x[1]["avg_roi"])[0],
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
            s = str(x).strip("[]() ")
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
            if X_df[col].dtype == object:
                # Strip brackets from strings like '[4.3119267E-1]'
                X_df[col] = X_df[col].apply(self._safe_to_float)
            # Force numeric — catches any residual non-numeric values
            X_df[col] = pd.to_numeric(X_df[col], errors="coerce")
        X_df = X_df.replace([np.inf, -np.inf], np.nan)
        X_df = X_df.fillna(X_df.median())
        # Final fallback: fill any remaining NaN (e.g., all-NaN columns) with 0
        X_df = X_df.fillna(0)
        return X_df.values.astype(np.float64)

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

        # Use CatBoost native SHAP when it's the best model (faster, exact)
        use_native_catboost_shap = (
            self.best_model_type == "catboost" and "catboost" in self.all_model_params
        )

        if use_native_catboost_shap:
            from catboost import Pool

            cb_params = {
                k: v
                for k, v in self.all_model_params["catboost"].items()
                if k not in self._CATBOOST_STRIP_KEYS
            }
            cb_params.update({"random_seed": self.seed, "verbose": False, "has_time": True})
            model = CatBoostClassifier(**cb_params)
            model.fit(X_train, y_train)
            logger.info("  Using CatBoost native SHAP (exact, GPU-accelerated)")
        else:
            # Train a LightGBM model (fast and SHAP-compatible)
            if "lightgbm" in self.all_model_params:
                params = {
                    **self.all_model_params["lightgbm"],
                    "random_state": self.seed,
                    "verbose": -1,
                }
            else:
                params = {
                    "n_estimators": 100,
                    "max_depth": 5,
                    "random_state": self.seed,
                    "verbose": -1,
                }
            model = lgb.LGBMClassifier(**params)
            model.fit(X_train, y_train)

        try:
            # Calculate SHAP values
            if use_native_catboost_shap:
                pool = Pool(X_shap)
                shap_vals_raw = model.get_feature_importance(type="ShapValues", data=pool)
                shap_values = shap_vals_raw[:, :-1]  # Remove bias column
            else:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_shap)

            # Handle binary classification (get positive class)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            # Calculate mean absolute SHAP value per feature
            mean_abs_shap = np.abs(shap_values).mean(axis=0)

            feature_importance = pd.DataFrame(
                {"feature": feature_names, "importance": mean_abs_shap}
            ).sort_values("importance", ascending=False)

            logger.info("\nTop 15 features by SHAP importance:")
            for i, row in feature_importance.head(15).iterrows():
                logger.info(f"  {row['feature']:40} {row['importance']:.4f}")

            # Identify low-importance features
            threshold = feature_importance["importance"].max() * 0.01
            low_importance = feature_importance[feature_importance["importance"] < threshold]

            # Feature interaction analysis (top pairs)
            interactions = []
            if len(X_shap) >= 50:
                logger.debug("\nTop feature interactions:")
                top_features_idx = feature_importance.head(10).index.tolist()

                for i, idx1 in enumerate(top_features_idx[:5]):
                    for idx2 in top_features_idx[i + 1 : 6]:
                        feat1 = feature_names[idx1] if idx1 < len(feature_names) else f"feat_{idx1}"
                        feat2 = feature_names[idx2] if idx2 < len(feature_names) else f"feat_{idx2}"

                        # Calculate interaction strength via correlation of SHAP values
                        if idx1 < shap_values.shape[1] and idx2 < shap_values.shape[1]:
                            corr = np.corrcoef(shap_values[:, idx1], shap_values[:, idx2])[0, 1]
                            if not np.isnan(corr):
                                interactions.append(
                                    {
                                        "feature1": feat1,
                                        "feature2": feat2,
                                        "interaction_strength": abs(corr),
                                    }
                                )

                interactions = sorted(
                    interactions, key=lambda x: x["interaction_strength"], reverse=True
                )[:10]
                for inter in interactions[:5]:
                    logger.debug(
                        f"  {inter['feature1']} x {inter['feature2']}: {inter['interaction_strength']:.3f}"
                    )

            # Store top features for per-feature border optimization
            self._shap_top_features = set(feature_importance.head(20)["feature"].tolist())

            return {
                "top_features": feature_importance.head(20).to_dict("records"),
                "low_importance_features": low_importance["feature"].tolist(),
                "n_low_importance": len(low_importance),
                "feature_interactions": interactions,
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
        if hasattr(model, "estimator"):
            base_model = model.estimator
        elif hasattr(model, "calibrated_classifiers_"):
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

            feature_importance = pd.DataFrame(
                {"feature": feature_names, "importance": mean_abs_shap}
            ).sort_values("importance", ascending=False)

            # Identify and remove low-importance features
            max_importance = feature_importance["importance"].max()
            threshold = max_importance * threshold_pct

            high_importance = feature_importance[feature_importance["importance"] >= threshold]
            low_importance = feature_importance[feature_importance["importance"] < threshold]

            refined_features = high_importance["feature"].tolist()
            removed_features = low_importance["feature"].tolist()

            # Get indices of refined features
            refined_indices = [
                feature_names.index(f) for f in refined_features if f in feature_names
            ]

            if removed_features:
                logger.info(
                    f"Removing {len(removed_features)} low-importance features (<{threshold_pct*100:.1f}% of max)"
                )
                for feat in removed_features[:10]:
                    logger.debug(f"  - {feat}")
                if len(removed_features) > 10:
                    logger.debug(f"  ... and {len(removed_features) - 10} more")

            shap_results = {
                "top_features": feature_importance.head(20).to_dict("records"),
                "removed_features": removed_features,
                "n_removed": len(removed_features),
                "n_kept": len(refined_features),
            }

            return refined_features, refined_indices, shap_results

        except Exception as e:
            logger.warning(f"SHAP validation failed: {e}")
            return feature_names, list(range(len(feature_names))), {"error": str(e)}

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
            logger.info(
                f"Using feature config: elo_k={self.feature_config.elo_k_factor}, "
                f"form_window={self.feature_config.form_window}, "
                f"ema_span={self.feature_config.ema_span}"
            )

        # Load data (with potential feature regeneration)
        df = self.load_data_with_feature_config()
        self.features_df = df  # Store for later use in model training
        self.feature_columns = self.get_feature_columns(df)
        logger.info(f"Available features: {len(self.feature_columns)}")

        # Pre-optimization forecastability gate (PE check)
        if self.pe_gate < 1.0:
            mean_pe = self._compute_pe_gate(df)
            if mean_pe is not None and mean_pe > self.pe_gate:
                logger.warning(
                    f"FORECASTABILITY GATE REJECTED: {self.bet_type} "
                    f"(mean PE={mean_pe:.4f} > threshold={self.pe_gate:.2f}). "
                    f"Skipping optimization."
                )
                return SniperResult(
                    bet_type=self.bet_type,
                    target=self.config["target"],
                    best_model="none",
                    best_params={},
                    n_features=0,
                    optimal_features=[],
                    best_threshold=0.0,
                    best_min_odds=0.0,
                    best_max_odds=0.0,
                    precision=0.0,
                    roi=-100.0,
                    n_bets=0,
                    n_wins=0,
                    timestamp=datetime.now().isoformat(),
                    mean_pe_residual=mean_pe,
                    forecastability_gate="rejected",
                )
            if mean_pe is not None:
                self._mean_pe = mean_pe
            if mean_pe is not None and mean_pe <= self.pe_gate:
                logger.info(
                    f"Forecastability gate PASSED: {self.bet_type} "
                    f"(mean PE={mean_pe:.4f} <= threshold={self.pe_gate:.2f})"
                )

        # Prepare data
        X = df[self.feature_columns].values
        X = np.nan_to_num(X, nan=0.0)
        # Ensure all values are numeric (handles string-wrapped floats like '[3.167E-1]')
        X = self._convert_array_to_float(X, self.feature_columns)
        y = self.prepare_target(df)

        # Store dates for sample weighting
        self.dates = df["date"].values

        # Preserve fixture_id for holdout prediction export
        if "fixture_id" in df.columns:
            self.fixture_ids = df["fixture_id"].values
        else:
            self.fixture_ids = None

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
            if self.fixture_ids is not None:
                self.fixture_ids = self.fixture_ids[valid_mask]
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
                logger.info(
                    f"Filtering {n_missing_odds} rows with missing/invalid odds for training"
                )
                X = X[odds_valid_mask]
                y = y[odds_valid_mask]
                odds = odds[odds_valid_mask]
                self.dates = self.dates[odds_valid_mask]
                if self.fixture_ids is not None:
                    self.fixture_ids = self.fixture_ids[odds_valid_mask]
                if self.league_col is not None:
                    self.league_col = self.league_col[odds_valid_mask]
                for oc in self._preserved_odds:
                    self._preserved_odds[oc] = self._preserved_odds[oc][odds_valid_mask]

        # Fill any remaining NaN odds with default for evaluation
        odds = np.nan_to_num(odds, nan=3.0)
        self._full_odds = odds  # Store for model saving (two-stage models need odds)

        logger.info(f"Training data: {len(X)} samples after filtering")

        # Step 1: RFE Feature Selection (with sample weights if enabled)
        rfe_weights = None
        if self.use_sample_weights and self.dates is not None:
            rfe_dates = pd.to_datetime(self.dates)
            rfe_weights = self.calculate_sample_weights(rfe_dates)
        selected_indices = self.run_rfe(X, y, sample_weights=rfe_weights)

        # Step 1b: Force-include cross-market interaction features (high CLV edge in R36)
        interaction_prefixes = ("btts_int_", "goals_int_", "fouls_int_")
        selected_set = set(selected_indices)
        forced_count = 0
        for i, col in enumerate(self.feature_columns):
            if col.startswith(interaction_prefixes) and i not in selected_set:
                selected_set.add(i)
                forced_count += 1
        if forced_count > 0:
            selected_indices = sorted(selected_set)
            logger.debug(f"Force-included {forced_count} cross-market interaction features")

        # Step 1c: Remove highly correlated features (>0.95) to reduce redundancy
        X_temp = pd.DataFrame(
            X[:, selected_indices], columns=[self.feature_columns[i] for i in selected_indices]
        )
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
            logger.info(
                f"Removed {len(to_drop)} correlated features (r>0.95), {len(selected_indices)} remain"
            )

        X_selected = X[:, selected_indices]
        self.optimal_features = [self.feature_columns[i] for i in selected_indices]

        # Step 1c.5: mRMR feature refinement (reduces post-RFECV bloat)
        self._mrmr_result = None
        if self.mrmr_k > 0 and len(self.optimal_features) > self.mrmr_k:
            logger.info(
                f"Running mRMR to refine {len(self.optimal_features)} → {self.mrmr_k} features..."
            )
            X_selected, self.optimal_features, mrmr_diag = self._mrmr_select(
                X_selected, y, self.optimal_features, self.mrmr_k
            )
            self._mrmr_result = mrmr_diag
        elif self.mrmr_k > 0:
            logger.info(
                f"mRMR skipped: {len(self.optimal_features)} features <= target {self.mrmr_k}"
            )
            self._mrmr_result = {
                "pre_count": len(self.optimal_features),
                "post_count": len(self.optimal_features),
                "removed_features": [],
                "n_removed": 0,
                "skipped": True,
            }

        # Step 1d: Adversarial feature filtering (remove temporally leaky features)
        self._adversarial_filter_diagnostics = None
        if self.adversarial_filter:
            logger.info(
                f"Running adversarial feature filtering (max_passes={self.adversarial_max_passes}, "
                f"max_features={self.adversarial_max_features}, auc_threshold={self.adversarial_auc_threshold})..."
            )
            X_selected, self.optimal_features, adv_filter_diag = _adversarial_filter(
                X_selected,
                self.optimal_features,
                max_passes=self.adversarial_max_passes,
                auc_threshold=self.adversarial_auc_threshold,
                max_features_per_pass=self.adversarial_max_features,
            )
            self._adversarial_filter_diagnostics = adv_filter_diag
            # Extract mean adversarial AUC for aggressive regularization
            pass_aucs = [p["auc"] for p in adv_filter_diag.get("passes", []) if "auc" in p]
            if pass_aucs:
                self._adversarial_auc_mean = float(np.mean(pass_aucs))
                logger.info(f"  Adversarial AUC mean: {self._adversarial_auc_mean:.3f}")
            if adv_filter_diag["total_removed"] > 0:
                logger.info(
                    f"Adversarial filter removed {adv_filter_diag['total_removed']} features, "
                    f"{len(self.optimal_features)} remain"
                )
            else:
                logger.info("Adversarial filter: no features removed")

        # Determine if aggressive regularization should apply
        self._aggressive_reg_applied = False
        self._regularization_overrides = None
        adv_auc = getattr(self, "_adversarial_auc_mean", None)
        if adv_auc is not None and adv_auc > 0.8 and not self.no_aggressive_reg:
            self._aggressive_reg_applied = True
            self._regularization_overrides = {
                "max_depth": "3-4",
                "rsm": "0.5-0.6",
                "min_data_in_leaf": "50-200",
                "min_child_samples": "50-200",
                "subsample": "0.5-0.7",
                "colsample_bytree": "0.5-0.7",
            }
            logger.info(
                f"  Aggressive regularization ENABLED (adversarial AUC {adv_auc:.3f} > 0.8)"
            )

        # Step 2: Hyperparameter Tuning (with sample weights and dates)
        self.best_model_type, self.best_params, base_precision = self.run_hyperparameter_tuning(
            X_selected, y, odds, dates=self.dates
        )

        # Extract tuned sample weight params and update instance state
        if self.best_params is not None and self.use_sample_weights:
            if "decay_rate" in self.best_params:
                self.sample_decay_rate = self.best_params.pop("decay_rate")
                logger.info(f"  Tuned decay_rate: {self.sample_decay_rate:.4f}")
            if "min_weight" in self.best_params:
                self.sample_min_weight = self.best_params.pop("min_weight")
                logger.info(f"  Tuned min_weight: {self.sample_min_weight:.3f}")
            # Also clean from all_model_params so model constructors don't get them
            # (skip two-stage models — their params are nested dicts, not flat)
            for mtype in self.all_model_params:
                if not mtype.startswith("two_stage_"):
                    self.all_model_params[mtype].pop("decay_rate", None)
                    self.all_model_params[mtype].pop("min_weight", None)

        # Save model params for two-phase merge pipeline (Phase 1 → Phase 2)
        if self.all_model_params:
            model_params_data = {
                "all_model_params": self.all_model_params,
                "model_cal_methods": getattr(self, "_model_cal_methods", {}),
                "bet_type": self.bet_type,
                "seed": self.seed,
            }
            params_path = OUTPUT_DIR / f"model_params_{self.bet_type}.json"
            params_path.parent.mkdir(parents=True, exist_ok=True)
            with open(params_path, "w") as f:
                json.dump(model_params_data, f, indent=2, default=_numpy_serializer)
            logger.info(f"Saved model params to {params_path}")

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
                params = {
                    k: v for k, v in self.best_params.items() if k not in self._CATBOOST_STRIP_KEYS
                }
                params.update({"random_seed": self.seed, "verbose": False})
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
            refined_features, refined_indices, shap_validation_results = (
                self.validate_features_with_shap(
                    validation_model, X_selected, y, self.optimal_features
                )
            )

            # If features were removed, update feature set
            if shap_validation_results.get("n_removed", 0) > 0:
                logger.info(
                    f"Refining features: {len(self.optimal_features)} -> {len(refined_features)}"
                )
                X_selected = X_selected[:, refined_indices]
                self.optimal_features = refined_features

        # Step 3: Threshold Optimization (includes stacking/average/agreement ensembles)
        final_model, threshold, min_odds, max_odds, precision, roi, n_bets, n_wins = (
            self.run_threshold_optimization(X_selected, y, odds, dates=self.dates)
        )

        # Get params for final model (empty dict for ensemble methods)
        ensemble_methods = [
            "stacking",
            "average",
            "agreement",
            "disagree_conservative_filtered",
            "disagree_balanced_filtered",
            "disagree_aggressive_filtered",
            "temporal_blend",
        ]
        final_params = self.best_params if final_model not in ensemble_methods else {}
        if final_model in ensemble_methods:
            final_params = {
                "ensemble_type": final_model,
                "base_models": list(self.all_model_params.keys()),
            }

        # Step 4: Walk-Forward Validation (optional)
        walkforward_results = {}
        if self.run_walkforward:
            walkforward_results = self.run_walkforward_validation(
                X_selected, y, odds, threshold, min_odds, max_odds
            )

        # Step 5: SHAP Feature Analysis (optional)
        shap_results = {}
        if self.run_shap:
            shap_results = self.run_shap_analysis(X_selected, y, self.optimal_features)
            # Merge validation results if available
            if shap_validation_results:
                shap_results["validation"] = shap_validation_results

        # Store final training data for model saving (already filtered)
        self._final_X = X_selected
        self._final_y = y
        self._final_odds = odds

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
            sample_min_weight=(
                getattr(self, "sample_min_weight", None) if self.use_sample_weights else None
            ),
            threshold_alpha=self.threshold_alpha if self.use_odds_threshold else None,
            holdout_metrics=getattr(self, "_holdout_metrics", None),
            stacking_weights=getattr(self, "_stacking_weights", None),
            stacking_alpha=getattr(self, "_stacking_alpha", None),
            adversarial_validation={
                "folds": getattr(self, "_adv_results", []),
                "filter": getattr(self, "_adversarial_filter_diagnostics", None),
            },
            calibration_method=self.calibration_method,
            per_league_ece=getattr(self, "_per_league_ece", None),
            calibration_validation=getattr(self, "_calibration_validation", None),
            brier_score=getattr(self, "_brier_score", None),
            fva=getattr(self, "_fva", None),
            mean_pe_residual=getattr(self, "_mean_pe", None),
            forecastability_gate="passed" if self.pe_gate < 1.0 else None,
            # Measurement hardening diagnostics
            embargo_days_computed=self._compute_embargo_days(),
            embargo_days_effective=max(self.embargo_days, self._compute_embargo_days()),
            aggressive_regularization_applied=getattr(self, "_aggressive_reg_applied", None),
            adversarial_auc_mean=getattr(self, "_adversarial_auc_mean", None),
            regularization_overrides=getattr(self, "_regularization_overrides", None),
            mrmr_result=getattr(self, "_mrmr_result", None),
            per_fold_ks=getattr(self, "_per_fold_ks", None),
            # Uncertainty (MAPIE conformal)
            uncertainty_penalty=getattr(self, "_uncertainty_penalty", None),
            holdout_uncertainty_roi=(
                getattr(self, "_holdout_metrics", {}).get("uncertainty_roi")
            ),
        )

        return result

    def train_and_save_models(
        self, X: np.ndarray, y: np.ndarray, odds: Optional[np.ndarray] = None
    ) -> List[str]:
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
            "stacking",
            "average",
            "agreement",
            "disagree_conservative_filtered",
            "disagree_balanced_filtered",
            "disagree_aggressive_filtered",
            "temporal_blend",
        }
        if self.best_model_type in ensemble_methods:
            # Ensemble: need all base models
            models_to_save = [
                m
                for m in self._get_base_model_types(
                    include_fastai=self.use_fastai,
                    fast_mode=self.fast_mode,
                    include_two_stage=self.use_two_stage,
                    only_catboost=self.only_catboost,
                    no_catboost=self.no_catboost,
                )
                if m in self.all_model_params
            ]
            logger.info(
                f"  Ensemble winner ({self.best_model_type}): saving {len(models_to_save)} base models"
            )
        else:
            # Individual model: save only the winner
            models_to_save = (
                [self.best_model_type] if self.best_model_type in self.all_model_params else []
            )
            logger.info(f"  Individual winner: saving only {self.best_model_type}")

        # Prepare scaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        for model_name in models_to_save:
            params = self.all_model_params.get(model_name, {})
            if not params:
                logger.debug(f"  Skipping {model_name} - no params available")
                continue

            try:
                if model_name.startswith("two_stage_"):
                    # Two-stage models need odds and have their own training path
                    from src.ml.two_stage_model import create_two_stage_model

                    base = "lightgbm" if model_name == "two_stage_lgb" else "catboost"
                    ts_model = create_two_stage_model(
                        base,
                        stage1_params=params.get("stage1_params"),
                        stage2_params=params.get("stage2_params"),
                        calibration_method=params.get("calibration_method", "sigmoid"),
                        min_edge_threshold=params.get("min_edge_threshold", 0.02),
                    )
                    train_odds = odds if odds is not None else np.full(len(y), 2.5)
                    # Split BEFORE fitting: first 80% for training, last 20% for conformal
                    n = len(X_scaled)
                    cal_start = int(n * 0.8)
                    ts_model.fit(X_scaled[:cal_start], y[:cal_start], train_odds[:cal_start])
                    model_data = {
                        "model": ts_model,
                        "features": self.optimal_features,
                        "bet_type": self.bet_type,
                        "scaler": scaler,
                        "model_type": "two_stage",
                        "best_params": params,
                    }
                    # Train conformal calibrator for production uncertainty
                    try:
                        from src.ml.uncertainty import ConformalClassifier

                        X_cal, y_cal = X_scaled[cal_start:], y[cal_start:]
                        if len(X_cal) >= 50:
                            conformal = ConformalClassifier(ts_model, alpha=0.1)
                            conformal.calibrate(X_cal, y_cal)
                            model_data["conformal"] = conformal.to_dict()
                            logger.info(
                                f"  Conformal calibrator saved ({len(X_cal)} cal samples)"
                            )
                    except Exception as e:
                        logger.warning(f"  Conformal calibration failed: {e}")
                else:
                    # Apply per-feature border optimization for CatBoost final models
                    if (
                        model_name == "catboost"
                        and hasattr(self, "_shap_top_features")
                        and self._shap_top_features
                        and self.optimal_features
                    ):
                        # CatBoost expects format: ["idx:border_count=N", ...]
                        per_feature_quantization = []
                        n_high = 0
                        for idx, feat in enumerate(self.optimal_features):
                            if feat in self._shap_top_features:
                                per_feature_quantization.append(f"{idx}:border_count=1024")
                                n_high += 1
                            else:
                                per_feature_quantization.append(f"{idx}:border_count=128")
                        params = {
                            **params,
                            "per_float_feature_quantization": per_feature_quantization,
                        }
                        logger.info(
                            f"  Applied per-feature borders: {n_high} features @ 1024 borders"
                        )

                    base_model = self._create_model_instance(model_name, params, seed=self.seed)

                    cal_method = getattr(self, "_model_cal_methods", {}).get(
                        model_name, self._sklearn_cal_method
                    )
                    # Split BEFORE fitting: first 80% for training, last 20% for conformal
                    n = len(X_scaled)
                    cal_start = int(n * 0.8)

                    # Beta calibration: use sigmoid for sklearn, save BetaCalibrator alongside
                    sklearn_cal = "sigmoid" if cal_method == "beta" else cal_method
                    calibrated = CalibratedClassifierCV(
                        base_model, method=sklearn_cal, cv=_get_calibration_cv(model_name)
                    )
                    calibrated.fit(X_scaled[:cal_start], y[:cal_start])

                    # Train BetaCalibrator for post-hoc recalibration if needed
                    saved_beta_cal = None
                    if cal_method == "beta":
                        train_proba = calibrated.predict_proba(X_scaled[:cal_start])
                        if train_proba.shape[1] > 1:
                            saved_beta_cal = BetaCalibrator(method="abm")
                            saved_beta_cal.fit(train_proba[:, 1], y[:cal_start])

                    model_data = {
                        "model": calibrated,
                        "features": self.optimal_features,
                        "bet_type": self.bet_type,
                        "scaler": scaler,
                        "calibration": cal_method,
                        "best_params": params,
                    }
                    if saved_beta_cal is not None:
                        model_data["beta_calibrator"] = saved_beta_cal

                    # Train conformal calibrator for production uncertainty
                    try:
                        from src.ml.uncertainty import ConformalClassifier

                        X_cal, y_cal = X_scaled[cal_start:], y[cal_start:]
                        if len(X_cal) >= 50:
                            conformal = ConformalClassifier(calibrated, alpha=0.1)
                            conformal.calibrate(X_cal, y_cal)
                            model_data["conformal"] = conformal.to_dict()
                            logger.info(
                                f"  Conformal calibrator saved ({len(X_cal)} cal samples)"
                            )
                    except Exception as e:
                        logger.warning(f"  Conformal calibration failed: {e}")

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
    print(
        f"\n{'Bet Type':<12} {'Model':<10} {'Thresh':>7} {'Odds':>12} "
        f"{'Precision':>10} {'ROI':>10} {'Bets':>6} {'Wins':>6} {'Status':<12}"
    )
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

        print(
            f"{r.bet_type:<12} {r.best_model:<10} {r.best_threshold:>7.2f} {odds_range:>12} "
            f"{r.precision*100:>9.1f}% {r.roi:>+9.1f}% {r.n_bets:>6} {r.n_wins:>6} {status:<12}"
        )

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
    wf_available = [r for r in results if r.walkforward and r.walkforward.get("summary")]
    if wf_available:
        print("\n" + "=" * 110)
        print("                         WALK-FORWARD VALIDATION RESULTS")
        print("=" * 110)
        print(
            f"\n{'Bet Type':<12} {'Best WF Model':<12} {'WF Avg ROI':>12} {'WF Std':>10} {'Overfitting?':<15}"
        )
        print("-" * 70)

        for r in wf_available:
            wf = r.walkforward
            if wf.get("summary"):
                best_wf_model = wf.get("best_model_wf", "unknown")
                best_summary = wf["summary"].get(best_wf_model, {})
                avg_roi = best_summary.get("avg_roi", 0)
                std_roi = best_summary.get("std_roi", 0)

                # Check for overfitting (if backtest ROI >> walk-forward ROI)
                overfit_ratio = r.roi / avg_roi if avg_roi > 0 else float("inf")
                if overfit_ratio > 2:
                    overfit_status = "HIGH RISK"
                elif overfit_ratio > 1.5:
                    overfit_status = "MODERATE"
                else:
                    overfit_status = "LOW"

                print(
                    f"{r.bet_type:<12} {best_wf_model:<12} {avg_roi:>+11.1f}% {std_roi:>9.1f}% {overfit_status:<15}"
                )

    # SHAP analysis highlights (if available)
    shap_available = [r for r in results if r.shap_analysis and r.shap_analysis.get("top_features")]
    if shap_available:
        print("\n" + "=" * 110)
        print("                            TOP FEATURES BY SHAP IMPORTANCE")
        print("=" * 110)

        # Collect all top features across bet types
        feature_counts = {}
        for r in shap_available:
            for feat in r.shap_analysis.get("top_features", [])[:10]:
                fname = feat.get("feature", "")
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
    ensemble_types = ["stacking", "average", "agreement"]
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

    viable_with_features = [
        r for r in results if r.optimal_features and r.precision >= 0.55 and r.roi > 0
    ]

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

        lines.append(
            f"| {r.bet_type} | {r.best_model} | {r.best_threshold:.2f} | {odds_range} | "
            f"{r.precision*100:.1f}% | {r.roi:+.1f}% | {r.n_bets} | {status} |"
        )

    # Walk-forward section
    wf_available = [r for r in results if r.walkforward and r.walkforward.get("summary")]
    if wf_available:
        lines.append("\n## Walk-Forward Validation\n")
        lines.append("| Bet Type | Best WF Model | Avg ROI | Std ROI | Overfitting Risk |")
        lines.append("|----------|---------------|---------|---------|------------------|")

        for r in wf_available:
            wf = r.walkforward
            best_wf_model = wf.get("best_model_wf", "unknown")
            best_summary = wf["summary"].get(best_wf_model, {})
            avg_roi = best_summary.get("avg_roi", 0)
            std_roi = best_summary.get("std_roi", 0)
            overfit_ratio = r.roi / avg_roi if avg_roi > 0 else float("inf")
            overfit_status = (
                "HIGH" if overfit_ratio > 2 else ("MODERATE" if overfit_ratio > 1.5 else "LOW")
            )
            lines.append(
                f"| {r.bet_type} | {best_wf_model} | {avg_roi:+.1f}% | {std_roi:.1f}% | {overfit_status} |"
            )

    # SHAP section
    shap_available = [r for r in results if r.shap_analysis and r.shap_analysis.get("top_features")]
    if shap_available:
        lines.append("\n## Top Features (SHAP Analysis)\n")

        feature_counts = {}
        for r in shap_available:
            for feat in r.shap_analysis.get("top_features", [])[:10]:
                fname = feat.get("feature", "")
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
    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    logger.info(f"Saved markdown summary to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Sniper Mode Optimization Pipeline")
    parser.add_argument("--bet-type", nargs="+", default=None, help="Bet type(s) to optimize")
    parser.add_argument("--all", action="store_true", help="Run for all bet types")
    parser.add_argument("--n-folds", type=int, default=5, help="Walk-forward folds")
    parser.add_argument(
        "--n-holdout-folds",
        type=int,
        default=1,
        help="Number of folds reserved for holdout (default: 1, max: n_folds-2)",
    )
    parser.add_argument(
        "--max-ece",
        type=float,
        default=0.15,
        help="Hard-reject configs with ECE above this threshold (default: 0.15)",
    )
    parser.add_argument(
        "--n-rfe-features",
        type=int,
        default=100,
        help="Target features after RFE (ignored if --auto-rfe)",
    )
    parser.add_argument(
        "--auto-rfe",
        action="store_true",
        help="Use RFECV to automatically find optimal feature count",
    )
    parser.add_argument(
        "--min-rfe-features",
        type=int,
        default=20,
        help="Minimum features for RFECV (only with --auto-rfe)",
    )
    parser.add_argument(
        "--max-rfe-features",
        type=int,
        default=80,
        help="Maximum features for RFECV cap (prevents bloat, R36 used 38-48)",
    )
    parser.add_argument("--n-optuna-trials", type=int, default=150, help="Optuna trials per model")
    parser.add_argument(
        "--min-bets", type=int, default=30, help="Minimum bets for valid configuration"
    )
    parser.add_argument(
        "--walkforward", action="store_true", help="Run walk-forward validation after optimization"
    )
    parser.add_argument(
        "--shap", action="store_true", help="Run SHAP feature importance and interaction analysis"
    )
    # Feature parameter options
    parser.add_argument(
        "--feature-params",
        type=str,
        default=None,
        help="Path to feature params YAML file (e.g., config/feature_params/away_win.yaml)",
    )
    parser.add_argument(
        "--optimize-features",
        action="store_true",
        help="Run feature parameter optimization before model optimization",
    )
    parser.add_argument(
        "--n-feature-trials",
        type=int,
        default=50,
        help="Optuna trials for feature parameter optimization",
    )
    # Model saving options
    parser.add_argument(
        "--save-models",
        action="store_true",
        help="Train and save final calibrated models to models/ directory",
    )
    parser.add_argument(
        "--upload-models",
        action="store_true",
        help="Upload saved models to HF Hub (requires HF_TOKEN)",
    )
    # Retail forecasting integration options
    parser.add_argument(
        "--sample-weights",
        action="store_true",
        help="Use time-decayed sample weights during training (recent matches weighted higher)",
    )
    parser.add_argument(
        "--decay-rate",
        type=float,
        default=None,
        help="Sample weight decay rate per day (default: ~0.002 for 1-year half-life)",
    )
    parser.add_argument(
        "--odds-threshold",
        action="store_true",
        help="Use odds-dependent betting thresholds (newsvendor-inspired)",
    )
    parser.add_argument(
        "--threshold-alpha",
        type=float,
        default=0.2,
        help="Odds-threshold adjustment strength (0=fixed, 1=full adjustment, default: 0.2)",
    )
    parser.add_argument(
        "--no-filter-missing-odds",
        action="store_true",
        help="Disable filtering of rows with missing odds during training",
    )
    parser.add_argument(
        "--calibration-method",
        type=str,
        default="sigmoid",
        choices=["sigmoid", "isotonic", "beta", "temperature", "venn_abers"],
        help="Initial calibration method (Optuna searches sigmoid/isotonic per model)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast mode: LightGBM + XGBoost only, max 5 Optuna trials",
    )
    parser.add_argument(
        "--no-two-stage",
        action="store_true",
        help="Disable two-stage models (saves ~60 trials of execution time)",
    )
    parser.add_argument(
        "--only-catboost",
        action="store_true",
        help="Run ONLY CatBoost models (for dedicated CatBoost optimization runs)",
    )
    parser.add_argument(
        "--no-catboost",
        action="store_true",
        help="Exclude CatBoost from model list (Phase 1 of two-phase merge pipeline)",
    )
    parser.add_argument(
        "--no-fastai",
        action="store_true",
        help="Exclude FastAI from model list (test boosting-only ensembles)",
    )
    parser.add_argument(
        "--merge-catboost",
        type=str,
        default=None,
        help="Path to Phase 1 model_params JSON for two-phase CatBoost merge (Phase 2)",
    )
    parser.add_argument(
        "--adversarial-filter",
        action="store_true",
        help="Pre-screen and remove temporally leaky features before training",
    )
    parser.add_argument(
        "--adversarial-max-passes",
        type=int,
        default=2,
        help="Max passes for adversarial filter (default: 2, try 5+ for H2H)",
    )
    parser.add_argument(
        "--adversarial-max-features",
        type=int,
        default=10,
        help="Max features removed per pass (default: 10, try 15+ for H2H)",
    )
    parser.add_argument(
        "--adversarial-auc-threshold",
        type=float,
        default=0.75,
        help="AUC threshold to stop filtering (default: 0.75, try 0.65 for aggressive)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to features parquet file (overrides default FEATURES_FILE)",
    )
    parser.add_argument(
        "--output-config",
        type=str,
        default=None,
        help="Output deployment config path (default: config/sniper_deployment.json)",
    )
    parser.add_argument(
        "--league-group",
        type=str,
        default="",
        help="League group namespace (e.g., 'americas'). Isolates feature params, models, and deployment config.",
    )
    parser.add_argument(
        "--no-monotonic",
        action="store_true",
        help="Disable monotonic constraints for CatBoost (enabled by default)",
    )
    parser.add_argument(
        "--no-transfer-learning",
        action="store_true",
        help="Disable CatBoost transfer learning (enabled by default)",
    )
    parser.add_argument(
        "--use-baseline",
        action="store_true",
        help="Enable CatBoost baseline injection from market odds (loses 20%% training data)",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable CatBoost deterministic mode (CPU-only, no bootstrap, debug only)",
    )
    parser.add_argument(
        "--cv-method",
        type=str,
        default="walk_forward",
        choices=["walk_forward", "purged_kfold"],
        help="Cross-validation method (default: walk_forward)",
    )
    parser.add_argument(
        "--embargo-days",
        type=int,
        default=14,
        help="Embargo period in days for purged CV (default: 14)",
    )
    parser.add_argument(
        "--markets",
        nargs="+",
        default=None,
        help="Alias for --bet-type (e.g., --markets home_win shots fouls)",
    )
    parser.add_argument(
        "--pe-gate",
        type=float,
        default=1.0,
        help="PE forecastability gate threshold (default: 1.0=disabled, recommended: 0.95). "
        "Markets with mean PE > threshold are skipped.",
    )
    parser.add_argument(
        "--no-aggressive-reg",
        action="store_true",
        help="Disable aggressive regularization when adversarial AUC > 0.8",
    )
    parser.add_argument(
        "--mrmr",
        type=int,
        default=0,
        help="mRMR feature selection target count (0=disabled, e.g. 40 to refine to 40 features)",
    )
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
            use_two_stage=False if args.no_two_stage else None,
            only_catboost=args.only_catboost,
            no_catboost=args.no_catboost,
            no_fastai=args.no_fastai,
            merge_params_path=args.merge_catboost,
            adversarial_filter=args.adversarial_filter,
            adversarial_max_passes=args.adversarial_max_passes,
            adversarial_max_features=args.adversarial_max_features,
            adversarial_auc_threshold=args.adversarial_auc_threshold,
            use_monotonic=not args.no_monotonic,
            use_transfer_learning=not args.no_transfer_learning,
            use_baseline=args.use_baseline,
            deterministic=args.deterministic,
            n_holdout_folds=args.n_holdout_folds,
            max_ece=args.max_ece,
            cv_method=args.cv_method,
            embargo_days=args.embargo_days,
            pe_gate=args.pe_gate,
            no_aggressive_reg=args.no_aggressive_reg,
            mrmr_k=args.mrmr,
        )

        result = optimizer.optimize()

        # Train and save models if requested
        if args.save_models and result.precision > 0.5 and result.n_bets > 0:
            # Use final training data stored during optimize() — already filtered
            X_final = getattr(optimizer, "_final_X", None)
            y_final = getattr(optimizer, "_final_y", None)
            odds_final = getattr(optimizer, "_final_odds", None)

            if X_final is None or y_final is None:
                logger.warning(
                    f"Cannot save models for {bet_type}: final training data not available"
                )
            else:
                logger.info(f"\nTraining final models for {bet_type}...")
                saved_models = optimizer.train_and_save_models(X_final, y_final, odds=odds_final)
                result = SniperResult(**{**asdict(result), "saved_models": saved_models})

        results.append(result)

        # Save individual result (atomic write to prevent truncated JSON)
        output_path = (
            OUTPUT_DIR / f"sniper_{bet_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        tmp_path = output_path.with_suffix(".json.tmp")
        with open(tmp_path, "w") as f:
            json.dump(asdict(result), f, indent=2, default=_numpy_serializer)
        tmp_path.rename(output_path)
        logger.info(f"Saved result to {output_path}")

    # Print summary
    print_summary(results)

    # Save combined results (atomic write to prevent truncated JSON)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    combined_path = OUTPUT_DIR / f"sniper_all_{timestamp}.json"
    tmp_combined = combined_path.with_suffix(".json.tmp")
    with open(tmp_combined, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2, default=_numpy_serializer)
    tmp_combined.rename(combined_path)
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
