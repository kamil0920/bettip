"""
Comprehensive Optimization Pipeline for All Bet Types.

This pipeline runs the full optimization flow:
1. Feature Selection (Boruta)
2. Model Comparison (XGBoost, LightGBM, CatBoost, RF, Stacking)
3. Hyperparameter Tuning (Optuna)
4. SHAP Analysis (feature validation)
5. Calibration (Isotonic/Platt/Beta)
6. Business Optimization (ROI, Kelly, CLV)

Usage:
    python experiments/run_comprehensive_optimization.py [--bet-types all]
"""

import argparse
import json
import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class BetTypeConfig:
    """Configuration for a single bet type."""
    name: str
    target_col: str
    target_type: str  # 'classification' or 'regression'
    enabled: bool = True
    min_samples: int = 100
    odds_col: Optional[str] = None
    line: Optional[float] = None


# Define all bet types
BET_TYPES = {
    "home_win": BetTypeConfig(
        name="Home Win",
        target_col="home_win",
        target_type="classification",
        odds_col="avg_odds_home",
    ),
    "away_win": BetTypeConfig(
        name="Away Win",
        target_col="away_win",
        target_type="classification",
        odds_col="avg_odds_away",
    ),
    "draw": BetTypeConfig(
        name="Draw",
        target_col="draw",
        target_type="classification",
        odds_col="avg_odds_draw",
    ),
    "btts": BetTypeConfig(
        name="BTTS",
        target_col="btts",
        target_type="classification",
        odds_col="btts_yes_odds",
    ),
    "over25": BetTypeConfig(
        name="Over 2.5 Goals",
        target_col="over25",
        target_type="classification",
        odds_col="over25_odds",
    ),
    "under25": BetTypeConfig(
        name="Under 2.5 Goals",
        target_col="under25",
        target_type="classification",
        odds_col="under25_odds",
    ),
    # Corners markets
    "corners_over_9": BetTypeConfig(
        name="Corners Over 9.5",
        target_col="corners_over_9",
        target_type="classification",
        line=9.5,
    ),
    "corners_over_10": BetTypeConfig(
        name="Corners Over 10.5",
        target_col="corners_over_10",
        target_type="classification",
        line=10.5,
    ),
    "corners_over_11": BetTypeConfig(
        name="Corners Over 11.5",
        target_col="corners_over_11",
        target_type="classification",
        line=11.5,
    ),
    "corners_under_10": BetTypeConfig(
        name="Corners Under 10.5",
        target_col="corners_under_10",
        target_type="classification",
        line=10.5,
    ),
    # Shots markets
    "shots_over_22": BetTypeConfig(
        name="Shots Over 22.5",
        target_col="shots_over_22",
        target_type="classification",
        line=22.5,
    ),
    "shots_over_24": BetTypeConfig(
        name="Shots Over 24.5",
        target_col="shots_over_24",
        target_type="classification",
        line=24.5,
    ),
    "shots_over_26": BetTypeConfig(
        name="Shots Over 26.5",
        target_col="shots_over_26",
        target_type="classification",
        line=26.5,
    ),
    "shots_under_24": BetTypeConfig(
        name="Shots Under 24.5",
        target_col="shots_under_24",
        target_type="classification",
        line=24.5,
    ),
    # Fouls markets
    "fouls_over_22": BetTypeConfig(
        name="Fouls Over 22.5",
        target_col="fouls_over_22",
        target_type="classification",
        line=22.5,
    ),
    "fouls_over_24": BetTypeConfig(
        name="Fouls Over 24.5",
        target_col="fouls_over_24",
        target_type="classification",
        line=24.5,
    ),
    "fouls_over_26": BetTypeConfig(
        name="Fouls Over 26.5",
        target_col="fouls_over_26",
        target_type="classification",
        line=26.5,
    ),
    "fouls_under_24": BetTypeConfig(
        name="Fouls Under 24.5",
        target_col="fouls_under_24",
        target_type="classification",
        line=24.5,
    ),
}


@dataclass
class OptimizationResult:
    """Results from optimizing a single bet type."""
    bet_type: str
    selected_features: List[str]
    best_model: str
    best_params: Dict[str, Any]
    cv_auc: float
    cv_accuracy: float
    cv_log_loss: float
    shap_top_features: List[str]
    calibration_method: str
    calibration_improvement: float
    optimal_threshold: float
    expected_roi: float
    roi_ci_lower: float
    roi_ci_upper: float
    win_rate: float
    avg_odds: float
    n_bets: int
    sharpe_ratio: float
    kelly_fraction: float
    training_time: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


# =============================================================================
# DATA LOADING & PREPARATION
# =============================================================================

def load_features(data_path: str = "data/03-features/features_with_niche_targets.csv") -> pd.DataFrame:
    """Load and prepare feature dataset."""
    logger.info(f"Loading features from {data_path}")
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    logger.info(f"Loaded {len(df)} matches with {len(df.columns)} columns")
    return df


def create_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Create all target columns for bet types."""
    df = df.copy()

    # Match outcome targets
    if "ft_home" in df.columns and "ft_away" in df.columns:
        df["home_win"] = (df["ft_home"] > df["ft_away"]).astype(int)
        df["draw"] = (df["ft_home"] == df["ft_away"]).astype(int)
        df["away_win"] = (df["ft_home"] < df["ft_away"]).astype(int)
        df["btts"] = ((df["ft_home"] > 0) & (df["ft_away"] > 0)).astype(int)
        df["over25"] = ((df["ft_home"] + df["ft_away"]) > 2.5).astype(int)
        df["under25"] = ((df["ft_home"] + df["ft_away"]) < 2.5).astype(int)

    # Niche market targets
    if "home_corners" in df.columns and "away_corners" in df.columns:
        df["total_corners"] = df["home_corners"] + df["away_corners"]
        df["corners_over_9"] = (df["total_corners"] > 9).astype(int)
        df["corners_over_10"] = (df["total_corners"] > 10).astype(int)

    if "home_yellow" in df.columns and "away_yellow" in df.columns:
        df["total_cards"] = (
            df["home_yellow"] + df["away_yellow"] +
            df.get("home_red", 0) + df.get("away_red", 0)
        )
        df["cards_over_4"] = (df["total_cards"] > 4).astype(int)

    if "home_shots_total" in df.columns and "away_shots_total" in df.columns:
        df["total_shots"] = df["home_shots_total"] + df["away_shots_total"]
        df["shots_over_24"] = (df["total_shots"] > 24).astype(int)
    elif "home_shots" in df.columns and "away_shots" in df.columns:
        df["total_shots"] = df["home_shots"] + df["away_shots"]
        df["shots_over_24"] = (df["total_shots"] > 24).astype(int)

    if "home_fouls" in df.columns and "away_fouls" in df.columns:
        df["total_fouls"] = df["home_fouls"] + df["away_fouls"]
        df["fouls_over_24"] = (df["total_fouls"] > 24).astype(int)

    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Get list of feature columns (exclude IDs, targets, odds, dates, leaky features)."""
    # EXACT column names that are outcomes (MUST be excluded)
    exact_exclude = {
        # Goals outcomes
        "total_goals", "goal_difference", "home_goals", "away_goals",
        "match_result", "result", "ft_home", "ft_away", "ht_home", "ht_away",
        # Standard betting targets
        "home_win", "draw", "away_win", "btts", "over25", "under25",
        # Niche market totals (actual match outcomes)
        "total_corners", "total_cards", "total_shots", "total_fouls",
        "home_corners", "away_corners", "home_shots", "away_shots",
        "home_fouls", "away_fouls", "home_yellow", "away_yellow",
        "home_red", "away_red", "home_shots_total", "away_shots_total",
        "home_shots_on_target", "away_shots_on_target",
        "home_shots_on", "away_shots_on",
        "home_possession", "away_possession",
        "home_offsides", "away_offsides",
        # Niche market binary targets (these ARE the target columns)
        "corners_over_9", "corners_over_10", "corners_over_11",
        "corners_under_9", "corners_under_10",
        "shots_over_22", "shots_over_24", "shots_over_26",
        "shots_under_22", "shots_under_24",
        "fouls_over_22", "fouls_over_24", "fouls_over_26",
        "fouls_under_22", "fouls_under_24",
    }

    # Patterns to exclude (substring match)
    exclude_patterns = [
        # Identifiers
        "fixture_id", "date", "timestamp", "team_id", "team_name",
        "league", "season", "round", "venue", "referee", "status",
        # Target-related patterns (catch any we missed)
        "corners_over", "corners_under", "cards_over", "cards_under",
        "shots_over", "shots_under", "fouls_over", "fouls_under",
        # Odds columns (encode market expectations, not pure features)
        "odds", "prob_", "b365_", "pinnacle", "betfair", "avg_odds",
        "implied", "_open", "_close", "overround", "max_home", "max_draw",
        "max_away", "min_home", "min_draw", "min_away",
        # Post-match statistics that leak outcome
        "xg_home", "xg_away", "home_xg", "away_xg",
    ]

    feature_cols = []
    for col in df.columns:
        if df[col].dtype in ["float64", "int64", "int32", "float32"]:
            # Check exact match first
            if col in exact_exclude:
                continue
            # Check pattern match
            col_lower = col.lower()
            is_excluded = any(pat in col_lower for pat in exclude_patterns)
            if not is_excluded:
                feature_cols.append(col)

    return feature_cols


# =============================================================================
# FEATURE SELECTION - BORUTA
# =============================================================================

def boruta_feature_selection(
    X: pd.DataFrame,
    y: pd.Series,
    max_features: int = 50,
    n_estimators: int = 100,
    max_iter: int = 50,
) -> List[str]:
    """
    Perform Boruta feature selection using shadow features.

    Boruta creates shadow features (shuffled copies), trains RF,
    and keeps features that outperform the best shadow feature.
    """
    logger.info(f"Running Boruta selection on {X.shape[1]} features...")

    from sklearn.ensemble import RandomForestClassifier

    # Clean data
    X_clean = X.fillna(X.median())

    # Initialize
    n_features = X_clean.shape[1]
    feature_names = list(X_clean.columns)
    hit_counts = np.zeros(n_features)

    for iteration in range(max_iter):
        # Create shadow features
        X_shadow = X_clean.apply(np.random.permutation).values
        X_extended = np.hstack([X_clean.values, X_shadow])

        # Train RF
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=5,
            n_jobs=-1,
            random_state=iteration,
        )
        rf.fit(X_extended, y)

        # Get importance
        importances = rf.feature_importances_
        real_imp = importances[:n_features]
        shadow_imp = importances[n_features:]
        shadow_max = shadow_imp.max()

        # Count hits (real features beating shadow max)
        hits = real_imp > shadow_max
        hit_counts += hits.astype(int)

    # Select features with > 50% hit rate
    hit_rate = hit_counts / max_iter
    selected_mask = hit_rate > 0.5
    selected_features = [f for f, s in zip(feature_names, selected_mask) if s]

    # Limit to max_features by importance
    if len(selected_features) > max_features:
        # Rank by hit rate
        feature_scores = [(f, hr) for f, hr in zip(feature_names, hit_rate) if hr > 0.5]
        feature_scores.sort(key=lambda x: x[1], reverse=True)
        selected_features = [f for f, _ in feature_scores[:max_features]]

    logger.info(f"Boruta selected {len(selected_features)} features")
    return selected_features


# =============================================================================
# MODEL COMPARISON
# =============================================================================

def get_models() -> Dict[str, Any]:
    """Get dictionary of models to compare."""
    try:
        from xgboost import XGBClassifier
        from lightgbm import LGBMClassifier
        from catboost import CatBoostClassifier
        has_boosting = True
    except ImportError:
        has_boosting = False
        logger.warning("XGBoost/LightGBM/CatBoost not available")

    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=100, max_depth=6, min_samples_leaf=10,
            n_jobs=-1, random_state=42
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            min_samples_leaf=10, random_state=42
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=1000, C=1.0, random_state=42
        ),
    }

    if has_boosting:
        models["XGBoost"] = XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="logloss",
            random_state=42, verbosity=0
        )
        models["LightGBM"] = LGBMClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbose=-1
        )
        models["CatBoost"] = CatBoostClassifier(
            iterations=100, depth=4, learning_rate=0.1,
            min_data_in_leaf=20, random_state=42, verbose=0
        )

    return models


def compare_models(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
) -> Tuple[str, Dict[str, float]]:
    """Compare models using time-series cross-validation."""
    logger.info("Comparing model architectures...")

    models = get_models()
    tscv = TimeSeriesSplit(n_splits=n_splits)

    results = {}

    for name, model in models.items():
        aucs = []
        log_losses = []

        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Fill NaN
            X_train = X_train.fillna(X_train.median())
            X_test = X_test.fillna(X_train.median())

            try:
                model_copy = model.__class__(**model.get_params())
                model_copy.fit(X_train, y_train)

                pred_proba = model_copy.predict_proba(X_test)[:, 1]
                aucs.append(roc_auc_score(y_test, pred_proba))
                log_losses.append(log_loss(y_test, pred_proba))
            except Exception as e:
                logger.warning(f"{name} failed: {e}")
                aucs.append(0.5)
                log_losses.append(1.0)

        results[name] = {
            "auc": np.mean(aucs),
            "auc_std": np.std(aucs),
            "log_loss": np.mean(log_losses),
        }
        logger.info(f"  {name}: AUC={results[name]['auc']:.4f} (±{results[name]['auc_std']:.4f})")

    # Add stacking ensemble
    try:
        base_models = [
            ("rf", RandomForestClassifier(n_estimators=50, max_depth=4, n_jobs=-1, random_state=42)),
            ("gb", GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)),
        ]

        if "XGBoost" in models:
            from xgboost import XGBClassifier
            base_models.append(("xgb", XGBClassifier(
                n_estimators=50, max_depth=3, random_state=42, verbosity=0, use_label_encoder=False
            )))

        stacking_lr = StackingClassifier(
            estimators=base_models,
            final_estimator=LogisticRegression(max_iter=1000),
            cv=3, n_jobs=-1
        )

        aucs = []
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            X_train = X_train.fillna(X_train.median())
            X_test = X_test.fillna(X_train.median())

            stacking_lr.fit(X_train, y_train)
            pred_proba = stacking_lr.predict_proba(X_test)[:, 1]
            aucs.append(roc_auc_score(y_test, pred_proba))

        results["Stacking_LR"] = {"auc": np.mean(aucs), "auc_std": np.std(aucs), "log_loss": 0.5}
        logger.info(f"  Stacking_LR: AUC={results['Stacking_LR']['auc']:.4f}")
    except Exception as e:
        logger.warning(f"Stacking failed: {e}")

    # Find best model
    best_model = max(results, key=lambda x: results[x]["auc"])
    logger.info(f"Best model: {best_model} (AUC={results[best_model]['auc']:.4f})")

    return best_model, results


# =============================================================================
# HYPERPARAMETER TUNING
# =============================================================================

def tune_hyperparameters(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int = 50,
) -> Tuple[Dict[str, Any], float]:
    """Tune hyperparameters using Optuna."""
    logger.info(f"Tuning {model_name} with {n_trials} trials...")

    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        logger.warning("Optuna not available, using default params")
        return {}, 0.5

    X_clean = X.fillna(X.median())
    tscv = TimeSeriesSplit(n_splits=3)

    def objective(trial):
        if model_name in ["XGBoost", "xgboost"]:
            from xgboost import XGBClassifier
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "use_label_encoder": False,
                "eval_metric": "logloss",
                "random_state": 42,
                "verbosity": 0,
            }
            model = XGBClassifier(**params)

        elif model_name in ["LightGBM", "lightgbm"]:
            from lightgbm import LGBMClassifier
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "random_state": 42,
                "verbose": -1,
            }
            model = LGBMClassifier(**params)

        elif model_name in ["CatBoost", "catboost"]:
            from catboost import CatBoostClassifier
            params = {
                "iterations": trial.suggest_int("iterations", 50, 300),
                "depth": trial.suggest_int("depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 50),
                "random_state": 42,
                "verbose": 0,
            }
            model = CatBoostClassifier(**params)

        elif model_name in ["RandomForest", "rf"]:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 50),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
                "random_state": 42,
                "n_jobs": -1,
            }
            model = RandomForestClassifier(**params)

        else:
            # Default to gradient boosting
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 200),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 30),
                "random_state": 42,
            }
            model = GradientBoostingClassifier(**params)

        # Cross-validate
        aucs = []
        for train_idx, test_idx in tscv.split(X_clean):
            X_train, X_test = X_clean.iloc[train_idx], X_clean.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model.fit(X_train, y_train)
            pred_proba = model.predict_proba(X_test)[:, 1]
            aucs.append(roc_auc_score(y_test, pred_proba))

        return np.mean(aucs)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    logger.info(f"Best AUC: {study.best_value:.4f}")
    return study.best_params, study.best_value


# =============================================================================
# SHAP ANALYSIS
# =============================================================================

def shap_analysis(
    model: Any,
    X: pd.DataFrame,
    top_n: int = 20,
) -> List[str]:
    """Get top features by SHAP importance or fallback to feature_importances_."""
    logger.info("Running SHAP/importance analysis...")

    X_clean = X.fillna(X.median())

    # Try SHAP first, but fallback to feature_importances_ if issues
    try:
        import shap

        model_type = type(model).__name__.lower()

        # Use permutation importance for problematic models (XGBoost version issues)
        if "xgb" in model_type:
            # XGBoost has version compatibility issues with SHAP
            # Use feature_importances_ instead
            raise ValueError("Using feature_importances_ for XGBoost")

        if "tree" in model_type or "forest" in model_type or "lgb" in model_type or "catboost" in model_type:
            explainer = shap.TreeExplainer(model)
            # Sample for speed
            X_sample = X_clean.sample(min(1000, len(X_clean)), random_state=42)
            shap_values = explainer.shap_values(X_sample)
        else:
            # Sample for speed
            X_sample = X_clean.sample(min(300, len(X_clean)), random_state=42)
            explainer = shap.KernelExplainer(model.predict_proba, X_sample)
            shap_values = explainer.shap_values(X_sample)

        # Handle different output formats
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Take positive class

        # Get mean absolute SHAP
        mean_shap = np.abs(shap_values).mean(axis=0)
        indices = np.argsort(mean_shap)[::-1][:top_n]
        top_features = [X.columns[i] for i in indices]

        logger.info(f"Top SHAP features: {top_features[:5]}")
        return top_features

    except Exception as e:
        logger.info(f"SHAP failed ({e}), using feature_importances_")

        # Fallback to feature_importances_
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:top_n]
            top_features = [X.columns[i] for i in indices]
            logger.info(f"Top importance features: {top_features[:5]}")
            return top_features

        # Last resort: use correlation
        logger.warning("No feature importance available, returning all features")
        return list(X.columns[:top_n])


# =============================================================================
# CALIBRATION
# =============================================================================

def calibrate_model(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> Tuple[Any, str, float]:
    """Calibrate model and return best method."""
    logger.info("Calibrating probabilities...")

    X_train_clean = X_train.fillna(X_train.median())
    X_val_clean = X_val.fillna(X_train.median())

    # Get uncalibrated predictions
    model.fit(X_train_clean, y_train)
    uncal_proba = model.predict_proba(X_val_clean)[:, 1]
    uncal_brier = brier_score_loss(y_val, uncal_proba)

    methods = ["isotonic", "sigmoid"]
    best_method = None
    best_brier = uncal_brier
    best_model = model

    for method in methods:
        try:
            cal_model = CalibratedClassifierCV(
                model.__class__(**model.get_params()),
                method=method,
                cv=3,
            )
            cal_model.fit(X_train_clean, y_train)
            cal_proba = cal_model.predict_proba(X_val_clean)[:, 1]
            cal_brier = brier_score_loss(y_val, cal_proba)

            if cal_brier < best_brier:
                best_brier = cal_brier
                best_method = method
                best_model = cal_model

        except Exception as e:
            logger.warning(f"Calibration {method} failed: {e}")

    improvement = (uncal_brier - best_brier) / uncal_brier * 100 if best_method else 0
    logger.info(f"Best calibration: {best_method or 'none'} (Brier improvement: {improvement:.1f}%)")

    return best_model, best_method or "none", improvement


# =============================================================================
# BUSINESS OPTIMIZATION
# =============================================================================

def optimize_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    odds: np.ndarray,
    thresholds: List[float] = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7],
) -> Tuple[float, Dict[str, float]]:
    """Find optimal betting threshold by ROI."""
    best_threshold = 0.5
    best_roi = -1.0

    results = {}

    for threshold in thresholds:
        # Filter to bets above threshold
        mask = y_proba >= threshold
        if mask.sum() < 10:
            continue

        # Calculate ROI
        bet_odds = odds[mask]
        bet_outcomes = y_true[mask]

        returns = bet_outcomes * (bet_odds - 1) - (1 - bet_outcomes)
        roi = returns.mean()
        win_rate = bet_outcomes.mean()

        results[threshold] = {
            "roi": roi,
            "win_rate": win_rate,
            "n_bets": mask.sum(),
            "avg_odds": bet_odds.mean(),
        }

        if roi > best_roi:
            best_roi = roi
            best_threshold = threshold

    return best_threshold, results


def bootstrap_roi(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    odds: np.ndarray,
    threshold: float,
    n_bootstrap: int = 1000,
) -> Tuple[float, float, float]:
    """Calculate ROI with bootstrap confidence intervals."""
    mask = y_proba >= threshold
    if mask.sum() < 10:
        return 0.0, 0.0, 0.0

    bet_odds = odds[mask]
    bet_outcomes = y_true[mask]

    rois = []
    n_bets = len(bet_outcomes)

    for _ in range(n_bootstrap):
        idx = np.random.choice(n_bets, n_bets, replace=True)
        sample_odds = bet_odds[idx]
        sample_outcomes = bet_outcomes[idx]

        returns = sample_outcomes * (sample_odds - 1) - (1 - sample_outcomes)
        rois.append(returns.mean())

    roi_mean = np.mean(rois)
    roi_lower = np.percentile(rois, 2.5)
    roi_upper = np.percentile(rois, 97.5)

    return roi_mean, roi_lower, roi_upper


def calculate_kelly(
    win_prob: float,
    odds: float,
) -> float:
    """Calculate Kelly criterion fraction."""
    if odds <= 1:
        return 0.0

    # Kelly = (bp - q) / b
    # where b = decimal odds - 1, p = win prob, q = 1 - p
    b = odds - 1
    p = win_prob
    q = 1 - p

    kelly = (b * p - q) / b
    return max(0, kelly)


def calculate_sharpe(returns: np.ndarray) -> float:
    """Calculate Sharpe ratio of returns."""
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    return returns.mean() / returns.std() * np.sqrt(len(returns))


# =============================================================================
# MAIN OPTIMIZATION PIPELINE
# =============================================================================

def optimize_bet_type(
    df: pd.DataFrame,
    bet_config: BetTypeConfig,
    feature_cols: List[str],
    n_optuna_trials: int = 50,
) -> Optional[OptimizationResult]:
    """Run full optimization pipeline for a single bet type."""
    import time
    start_time = time.time()

    logger.info(f"\n{'='*60}")
    logger.info(f"OPTIMIZING: {bet_config.name}")
    logger.info(f"{'='*60}")

    # Check if target exists
    if bet_config.target_col not in df.columns:
        logger.warning(f"Target column {bet_config.target_col} not found, skipping")
        return None

    # Prepare data
    target = df[bet_config.target_col]
    valid_mask = target.notna()

    # Get odds if available
    odds = None
    if bet_config.odds_col and bet_config.odds_col in df.columns:
        odds = df[bet_config.odds_col].values
        valid_mask &= df[bet_config.odds_col].notna()
    else:
        # Use implied odds from target rate
        base_rate = target[valid_mask].mean()
        odds = np.full(len(df), 1 / base_rate if base_rate > 0 else 2.0)

    df_valid = df[valid_mask].copy()
    target_valid = target[valid_mask]
    odds_valid = odds[valid_mask]

    if len(df_valid) < bet_config.min_samples:
        logger.warning(f"Insufficient samples ({len(df_valid)}), skipping")
        return None

    logger.info(f"Samples: {len(df_valid)}, Target rate: {target_valid.mean():.1%}")

    # Get features
    available_features = [f for f in feature_cols if f in df_valid.columns]
    X = df_valid[available_features]
    y = target_valid

    # Time-series split
    split_idx = int(len(df_valid) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    odds_test = odds_valid[split_idx:]

    # 1. BORUTA FEATURE SELECTION
    selected_features = boruta_feature_selection(X_train, y_train, max_features=50)
    if len(selected_features) < 5:
        logger.warning("Boruta selected too few features, using top correlations")
        correlations = X_train.corrwith(y_train).abs().sort_values(ascending=False)
        selected_features = list(correlations.head(30).index)

    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]

    # 2. MODEL COMPARISON
    best_model_name, model_results = compare_models(X_train_selected, y_train)

    # 3. HYPERPARAMETER TUNING
    best_params, tuned_auc = tune_hyperparameters(
        best_model_name, X_train_selected, y_train, n_trials=n_optuna_trials
    )

    # 4. Train final model with best params
    if best_model_name == "XGBoost":
        from xgboost import XGBClassifier
        final_model = XGBClassifier(**best_params, use_label_encoder=False, eval_metric="logloss", verbosity=0)
    elif best_model_name == "LightGBM":
        from lightgbm import LGBMClassifier
        final_model = LGBMClassifier(**best_params, verbose=-1)
    elif best_model_name == "CatBoost":
        from catboost import CatBoostClassifier
        final_model = CatBoostClassifier(**best_params, verbose=0)
    elif best_model_name == "RandomForest":
        final_model = RandomForestClassifier(**best_params, n_jobs=-1)
    else:
        final_model = GradientBoostingClassifier(**best_params)

    X_train_clean = X_train_selected.fillna(X_train_selected.median())
    X_test_clean = X_test_selected.fillna(X_train_selected.median())
    final_model.fit(X_train_clean, y_train)

    # 5. SHAP ANALYSIS
    shap_features = shap_analysis(final_model, X_train_clean)

    # 6. CALIBRATION
    # Use a validation split for calibration
    cal_split = int(len(X_train_clean) * 0.8)
    X_cal_train = X_train_clean.iloc[:cal_split]
    y_cal_train = y_train.iloc[:cal_split]
    X_cal_val = X_train_clean.iloc[cal_split:]
    y_cal_val = y_train.iloc[cal_split:]

    calibrated_model, cal_method, cal_improvement = calibrate_model(
        final_model, X_cal_train, y_cal_train, X_cal_val, y_cal_val
    )

    # Retrain calibrated model on full training data
    calibrated_model.fit(X_train_clean, y_train)

    # 7. BUSINESS OPTIMIZATION
    y_proba = calibrated_model.predict_proba(X_test_clean)[:, 1]
    y_true = y_test.values

    # Optimize threshold
    opt_threshold, threshold_results = optimize_threshold(y_true, y_proba, odds_test)

    # Bootstrap ROI
    roi_mean, roi_lower, roi_upper = bootstrap_roi(y_true, y_proba, odds_test, opt_threshold)

    # Calculate metrics at optimal threshold
    bet_mask = y_proba >= opt_threshold
    if bet_mask.sum() > 0:
        bet_outcomes = y_true[bet_mask]
        bet_odds = odds_test[bet_mask]
        win_rate = bet_outcomes.mean()
        avg_odds = bet_odds.mean()
        n_bets = int(bet_mask.sum())

        # Calculate returns for Sharpe
        returns = bet_outcomes * (bet_odds - 1) - (1 - bet_outcomes)
        sharpe = calculate_sharpe(returns)

        # Kelly fraction
        kelly = calculate_kelly(win_rate, avg_odds)
    else:
        win_rate = avg_odds = n_bets = sharpe = kelly = 0.0

    # CV metrics
    cv_auc = model_results[best_model_name]["auc"]

    # Final predictions for log loss and accuracy
    y_pred = (y_proba > 0.5).astype(int)
    cv_accuracy = accuracy_score(y_true, y_pred)
    cv_log_loss = log_loss(y_true, y_proba)

    elapsed = time.time() - start_time

    result = OptimizationResult(
        bet_type=bet_config.name,
        selected_features=selected_features,
        best_model=best_model_name,
        best_params=best_params,
        cv_auc=cv_auc,
        cv_accuracy=cv_accuracy,
        cv_log_loss=cv_log_loss,
        shap_top_features=shap_features[:10],
        calibration_method=cal_method,
        calibration_improvement=cal_improvement,
        optimal_threshold=opt_threshold,
        expected_roi=roi_mean,
        roi_ci_lower=roi_lower,
        roi_ci_upper=roi_upper,
        win_rate=win_rate,
        avg_odds=avg_odds,
        n_bets=n_bets,
        sharpe_ratio=sharpe,
        kelly_fraction=kelly,
        training_time=elapsed,
    )

    logger.info(f"\nResults for {bet_config.name}:")
    logger.info(f"  Best Model: {best_model_name}")
    logger.info(f"  AUC: {cv_auc:.4f}")
    logger.info(f"  Optimal Threshold: {opt_threshold:.2f}")
    logger.info(f"  Expected ROI: {roi_mean:.1%} [{roi_lower:.1%}, {roi_upper:.1%}]")
    logger.info(f"  Win Rate: {win_rate:.1%}")
    logger.info(f"  # Bets: {n_bets}")
    logger.info(f"  Sharpe: {sharpe:.2f}")
    logger.info(f"  Kelly: {kelly:.2%}")
    logger.info(f"  Time: {elapsed:.1f}s")

    return result


def run_comprehensive_optimization(
    bet_types: List[str] = None,
    n_optuna_trials: int = 50,
    output_dir: str = "experiments/outputs/comprehensive_optimization",
) -> Dict[str, OptimizationResult]:
    """Run comprehensive optimization for all bet types."""

    # Setup output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_features()
    df = create_targets(df)

    # Get feature columns
    feature_cols = get_feature_columns(df)
    logger.info(f"Available features: {len(feature_cols)}")

    # Determine bet types to optimize
    if bet_types is None or "all" in bet_types:
        bet_types_to_run = list(BET_TYPES.keys())
    else:
        bet_types_to_run = bet_types

    # Run optimization for each bet type
    results = {}

    for bet_type in bet_types_to_run:
        if bet_type not in BET_TYPES:
            logger.warning(f"Unknown bet type: {bet_type}")
            continue

        config = BET_TYPES[bet_type]
        if not config.enabled:
            logger.info(f"Skipping disabled bet type: {bet_type}")
            continue

        result = optimize_bet_type(df, config, feature_cols, n_optuna_trials)
        if result:
            results[bet_type] = result

    # Generate summary
    generate_summary(results, output_path)

    # Save results
    save_results(results, output_path)

    return results


def generate_summary(results: Dict[str, OptimizationResult], output_path: Path):
    """Generate summary report."""
    logger.info("\n" + "="*70)
    logger.info("COMPREHENSIVE OPTIMIZATION SUMMARY")
    logger.info("="*70)

    # Sort by expected ROI
    sorted_results = sorted(
        results.values(),
        key=lambda x: x.expected_roi,
        reverse=True
    )

    # Create summary table
    print("\n" + "="*90)
    print(f"{'Bet Type':<20} {'Model':<12} {'AUC':>6} {'Thresh':>6} {'ROI':>8} {'Win%':>6} {'Bets':>5} {'Sharpe':>6}")
    print("="*90)

    for r in sorted_results:
        roi_str = f"{r.expected_roi:+.1%}"
        print(f"{r.bet_type:<20} {r.best_model:<12} {r.cv_auc:>6.3f} {r.optimal_threshold:>6.2f} {roi_str:>8} {r.win_rate:>5.1%} {r.n_bets:>5} {r.sharpe_ratio:>6.2f}")

    print("="*90)

    # Recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)

    # Top picks by ROI with positive confidence interval
    profitable = [r for r in sorted_results if r.roi_ci_lower > 0]

    if profitable:
        print("\n✓ RECOMMENDED BETS (95% CI lower bound > 0):")
        for r in profitable[:5]:
            print(f"  • {r.bet_type}: ROI {r.expected_roi:+.1%} [{r.roi_ci_lower:+.1%}, {r.roi_ci_upper:+.1%}]")
            print(f"    Model: {r.best_model}, Threshold: {r.optimal_threshold:.2f}, Kelly: {r.kelly_fraction:.1%}")

    # Borderline
    borderline = [r for r in sorted_results if r.expected_roi > 0 and r.roi_ci_lower <= 0]
    if borderline:
        print("\n⚠ BORDERLINE (positive ROI but CI includes 0):")
        for r in borderline[:3]:
            print(f"  • {r.bet_type}: ROI {r.expected_roi:+.1%} [{r.roi_ci_lower:+.1%}, {r.roi_ci_upper:+.1%}]")

    # Not recommended
    not_recommended = [r for r in sorted_results if r.expected_roi <= 0]
    if not_recommended:
        print("\n✗ NOT RECOMMENDED (negative expected ROI):")
        for r in not_recommended:
            print(f"  • {r.bet_type}: ROI {r.expected_roi:+.1%}")

    # Top features across all models
    print("\n" + "="*70)
    print("TOP FEATURES ACROSS ALL MODELS")
    print("="*70)

    feature_counts = {}
    for r in sorted_results:
        for f in r.shap_top_features[:5]:
            # Handle if f is a string or other type
            if isinstance(f, str):
                feature_counts[f] = feature_counts.get(f, 0) + 1
            else:
                # Skip non-string features
                continue

    top_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    for feat, count in top_features:
        print(f"  {feat}: appears in {count} models")


def save_results(results: Dict[str, OptimizationResult], output_path: Path):
    """Save results to files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save detailed JSON
    json_data = {
        bet_type: {
            "bet_type": r.bet_type,
            "best_model": r.best_model,
            "best_params": r.best_params,
            "cv_auc": r.cv_auc,
            "cv_accuracy": r.cv_accuracy,
            "cv_log_loss": r.cv_log_loss,
            "selected_features": r.selected_features[:20],
            "shap_top_features": [str(f) for f in r.shap_top_features] if r.shap_top_features else [],
            "calibration_method": r.calibration_method,
            "calibration_improvement": r.calibration_improvement,
            "optimal_threshold": r.optimal_threshold,
            "expected_roi": r.expected_roi,
            "roi_ci_lower": r.roi_ci_lower,
            "roi_ci_upper": r.roi_ci_upper,
            "win_rate": r.win_rate,
            "avg_odds": r.avg_odds,
            "n_bets": int(r.n_bets),
            "sharpe_ratio": r.sharpe_ratio,
            "kelly_fraction": r.kelly_fraction,
            "training_time": r.training_time,
            "timestamp": r.timestamp,
        }
        for bet_type, r in results.items()
    }

    json_path = output_path / f"optimization_results_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)

    logger.info(f"\nResults saved to: {json_path}")

    # Save summary CSV
    summary_data = []
    for r in results.values():
        summary_data.append({
            "bet_type": r.bet_type,
            "model": r.best_model,
            "auc": r.cv_auc,
            "threshold": r.optimal_threshold,
            "roi": r.expected_roi,
            "roi_lower": r.roi_ci_lower,
            "roi_upper": r.roi_ci_upper,
            "win_rate": r.win_rate,
            "n_bets": int(r.n_bets),
            "sharpe": r.sharpe_ratio,
            "kelly": r.kelly_fraction,
            "recommended": r.roi_ci_lower > 0,
        })

    csv_path = output_path / f"optimization_summary_{timestamp}.csv"
    pd.DataFrame(summary_data).to_csv(csv_path, index=False)
    logger.info(f"Summary saved to: {csv_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Comprehensive Betting Optimization Pipeline")
    parser.add_argument(
        "--bet-types",
        nargs="+",
        default=["all"],
        help="Bet types to optimize (default: all)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of Optuna trials per model (default: 50)",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/outputs/comprehensive_optimization",
        help="Output directory",
    )

    args = parser.parse_args()

    logger.info("Starting Comprehensive Optimization Pipeline")
    logger.info(f"Bet types: {args.bet_types}")
    logger.info(f"Optuna trials: {args.n_trials}")

    results = run_comprehensive_optimization(
        bet_types=args.bet_types,
        n_optuna_trials=args.n_trials,
        output_dir=args.output_dir,
    )

    logger.info("\nOptimization complete!")
    return results


if __name__ == "__main__":
    main()
