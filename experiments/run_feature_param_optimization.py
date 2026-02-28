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
import re
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict

import catboost as cb
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from sklearn.metrics import log_loss as sklearn_log_loss
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
from tqdm import tqdm

from src.ml.adversarial import _adversarial_filter

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

# Paths - Use unified features file with football-data.co.uk odds
FEATURES_FILE = Path("data/03-features/features_all_5leagues_with_odds.parquet")
OUTPUT_DIR = Path("experiments/outputs/feature_param_optimization")

# Bet type configurations (same as sniper optimization)
BET_TYPES = {
    "away_win": {
        "target": "away_win",
        "odds_col": "avg_away_close",
        "approach": "classification",
        "default_threshold": 0.60,
    },
    "home_win": {
        "target": "home_win",
        "odds_col": "avg_home_close",
        "approach": "classification",
        "default_threshold": 0.60,
    },
    "btts": {
        "target": "btts",
        "odds_col": "btts_yes_odds",  # No bulk historical BTTS odds; uses fallback
        "approach": "classification",
        "default_threshold": 0.55,  # Lower threshold for BTTS (high base rate ~50%)
    },
    "over25": {
        "target": "over25",
        "odds_col": "avg_over25_close",
        "approach": "classification",
        "default_threshold": 0.60,
    },
    "under25": {
        "target": "under25",
        "odds_col": "avg_under25_close",
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
        "odds_col": "theodds_shots_over_odds",  # No real odds for total shots; uses fallback
        "approach": "regression_line",
        "default_threshold": 0.55,
    },
    "corners": {
        "target": "total_corners",
        "target_line": 9.5,  # Gives ~50% base rate
        "odds_col": "theodds_corners_over_odds",  # No bulk historical odds; uses fallback
        "approach": "regression_line",
        "default_threshold": 0.50,  # Lower for ~32% base rate
    },
    "cards": {
        "target": "total_cards",
        "target_line": 4.5,
        "odds_col": "theodds_cards_over_odds",  # No bulk historical odds; uses fallback
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
    "total_yellows", "total_reds",  # Aggregate card counts (match outcome)
    "home_yellow_cards", "away_yellow_cards",  # Alternate naming
    "home_red_cards", "away_red_cards",  # Alternate naming
    "home_possession", "away_possession",
    "total_cards", "total_shots",
    "total_shots_on_target",  # Match outcome - was causing cards leakage
    "home_cards", "away_cards",
    # S43: Niche odds columns (target-encoding leakage)
    "theodds_cards_over_odds", "cards_under_odds",
    "theodds_corners_over_odds", "corners_under_odds",
    "theodds_shots_over_odds", "shots_under_odds",
    "fouls_over_odds", "fouls_under_odds",
    "cornershc_over_odds", "cornershc_under_odds",
    "cardshc_over_odds", "cardshc_under_odds",
    "h2h_h1_home_avg", "h2h_h1_away_avg",
    "totals_h1_over_odds", "totals_h1_under_odds",
    # S44: Dead constant features (verified against HF Hub parquet 2026-02-26)
    # Weather (17 constants: temp=15, wind=10, humidity=70, is_clear=1, rest 0)
    "weather_temp", "weather_wind", "weather_humidity",
    "weather_is_clear", "weather_is_rainy", "weather_is_stormy",
    "weather_is_foggy", "weather_is_windy", "weather_very_windy",
    "weather_extreme_hot", "weather_extreme_cold", "weather_high_humidity",
    "weather_adverse_score", "weather_heavy_rain", "weather_precip",
    "weather_temp_normalized", "weather_humidity_normalized",
    # CLV diagnostics (10 constants, all 0.0)
    "home_avg_historical_clv", "away_avg_historical_clv",
    "home_clv_ema", "away_clv_ema",
    "home_clv_std", "away_clv_std",
    "home_clv_trend", "away_clv_trend",
    "clv_edge_diff", "both_positive_clv",
    # Cross-market / odds interactions (7 constants)
    "odds_upset_potential", "steam_x_elo_diff", "movement_x_form",
    "sharp_x_upset", "velocity_x_rest", "league_cluster",
    "one_team_nothing_to_play",
    # Sample entropy features (93-99.9% null — effectively dead)
    "fouls_sampen_diff", "fouls_sampen_sum",
    "shots_sampen_diff", "shots_sampen_sum",
    "corners_sampen_diff", "corners_sampen_sum",
    "home_fouls_sampen", "away_fouls_sampen",
    "home_shots_sampen", "away_shots_sampen",
    "home_corners_sampen", "away_corners_sampen",
    # Also high-null sampen (49-75%)
    "goals_sampen_diff", "goals_sampen_sum",
    "cards_sampen_diff", "cards_sampen_sum",
    "home_goals_sampen", "away_goals_sampen",
    "home_cards_sampen", "away_cards_sampen",
    # Redundant features (r=1.0 with another feature)
    "ref_corners_bias", "ref_fouls_bias", "ref_cards_bias",
    "ref_home_bias", "expected_total_with_home_adj",
    "home_stars_ratio", "away_stars_ratio", "away_win_prob_elo",
]

# Leaky patterns
LEAKY_PATTERNS = [
    "avg_home", "avg_away", "avg_draw", "avg_over", "avg_under", "avg_ah",
    "b365_", "pinnacle_", "max_home", "max_away", "max_draw", "max_over", "max_under", "max_ah",
    # Odds columns (used for ROI calc, not features)
    "sm_btts_", "sm_corners_", "sm_cards_", "sm_shots_",
    "theodds_", "btts_yes_odds", "fouls_over_odds", "fouls_under_odds",
    "shots_over_odds", "shots_under_odds", "cards_over_odds", "cards_under_odds",
    "corners_over_odds", "corners_under_odds",
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

    Optimizes for neg_log_loss (proper scoring rule, threshold-independent).
    """
    bet_type: str
    best_params: Dict[str, Any]
    neg_log_loss: float  # Primary objective: negative log loss (higher = better)
    sharpe: float  # Legacy: ROI consistency score (kept for logging)
    precision: float  # Mean precision across folds
    roi: float  # Mean ROI across folds
    n_bets: int
    n_trials: int
    n_folds: int
    fold_precisions: List[float]  # Per-fold precision for transparency
    search_space: Dict[str, tuple]  # (min, max, type) tuples
    all_trials: List[Dict[str, Any]]
    timestamp: str
    embargo_days: int = 0  # Embargo applied during evaluation
    adversarial_features_removed: int = 0  # Features removed by adversarial filter


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
        n_folds: int = 3,
        min_bets: int = 30,
        use_regeneration: bool = False,
        time_budget_minutes: int = 0,
    ):
        """
        Initialize the optimizer.

        Args:
            bet_type: Bet type to optimize
            n_trials: Number of Optuna trials
            n_folds: Walk-forward folds (3 is sufficient for comparing feature configs)
            min_bets: Minimum bets required for valid result
            use_regeneration: If True, actually regenerate features (slower but more accurate).
                             If False, use existing features (faster for testing).
            time_budget_minutes: Time budget in minutes (0=unlimited). When exceeded,
                                Optuna stops and saves best-so-far result.
        """
        self.bet_type = bet_type
        self.config = BET_TYPES[bet_type]
        self.n_trials = n_trials
        self.n_folds = n_folds
        self.min_bets = min_bets
        self.use_regeneration = use_regeneration
        self.time_budget_minutes = time_budget_minutes

        self.search_space = get_search_space_for_bet_type(bet_type)
        self.regenerator = FeatureRegenerator() if use_regeneration else None

        self.features_df = None
        self.feature_columns = None

    def load_base_features(self) -> pd.DataFrame:
        """Load existing features file (for non-regeneration mode)."""
        from src.utils.data_io import load_features
        from src.utils.line_plausibility import filter_implausible_training_rows
        df = load_features(FEATURES_FILE)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        # Derive target if needed
        target = self.config["target"]
        if target not in df.columns:
            self._derive_target(df, target)

        # Filter implausible training rows for niche line markets
        df = filter_implausible_training_rows(df, self.bet_type)

        return df

    def _derive_target(self, df: pd.DataFrame, target: str) -> None:
        """Derive target column if not present (or fill gaps from components)."""
        derived = None

        if target == "total_cards":
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
                derived[df["home_shots"].isna() & df["away_shots"].isna()] = np.nan
        elif target == "total_fouls":
            if "home_fouls" in df.columns and "away_fouls" in df.columns:
                derived = df["home_fouls"].fillna(0) + df["away_fouls"].fillna(0)
                derived[df["home_fouls"].isna() & df["away_fouls"].isna()] = np.nan
        elif target == "total_corners":
            if "home_corners" in df.columns and "away_corners" in df.columns:
                derived = df["home_corners"].fillna(0) + df["away_corners"].fillna(0)
                derived[df["home_corners"].isna() & df["away_corners"].isna()] = np.nan
        elif target == "under25":
            if "total_goals" in df.columns:
                df["under25"] = (df["total_goals"] < 2.5).astype(int)
            elif "home_goals" in df.columns and "away_goals" in df.columns:
                df["under25"] = ((df["home_goals"].fillna(0) + df["away_goals"].fillna(0)) < 2.5).astype(int)
            return
        elif target == "over25":
            if "total_goals" in df.columns:
                df["over25"] = (df["total_goals"] > 2.5).astype(int)
            elif "home_goals" in df.columns and "away_goals" in df.columns:
                df["over25"] = ((df["home_goals"].fillna(0) + df["away_goals"].fillna(0)) > 2.5).astype(int)
            return
        elif target == "btts":
            home_goals = df["home_goals"] if "home_goals" in df.columns else pd.Series(0, index=df.index)
            away_goals = df["away_goals"] if "away_goals" in df.columns else pd.Series(0, index=df.index)
            df["btts"] = ((home_goals.fillna(0) > 0) & (away_goals.fillna(0) > 0)).astype(int)
            return

        if derived is not None:
            if target in df.columns:
                df[target] = df[target].fillna(derived)
            else:
                df[target] = derived

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

        # Safety net: drop any remaining zero-variance features
        variances = df[features].var()
        zero_var = variances[variances == 0].index.tolist()
        if zero_var:
            logger.info(f"Dropping {len(zero_var)} zero-variance features: {zero_var[:5]}...")
            features = [c for c in features if c not in zero_var]

        return sorted(features)

    def prepare_target(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare target variable.

        Preserves NaN so downstream valid_mask can filter them out.
        """
        target_col = self.config["target"]

        if self.config["approach"] == "classification":
            return df[target_col].values.astype(float)
        elif self.config["approach"] == "regression_line":
            line = self.config.get("target_line", 0)
            raw = df[target_col].values.astype(float)
            if self.config.get("direction") == "under":
                return np.where(np.isnan(raw), np.nan, (raw < line).astype(float))
            return np.where(np.isnan(raw), np.nan, (raw > line).astype(float))
        else:
            return df[target_col].values.astype(float)

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
        trial: Optional["optuna.Trial"] = None,
        filtered_feature_info: Optional[Tuple[List[str], int]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a feature configuration using walk-forward validation.

        Uses log_loss as primary objective (proper scoring rule, threshold-independent).
        Multi-model evaluation (CatBoost/LGB/XGB) weighted toward production architecture.
        Date-based temporal embargo prevents feature lookback leakage between folds.

        Args:
            feature_config: Configuration to evaluate
            features_df: Pre-loaded features (for non-regeneration mode)
            trial: Optuna trial for pruning support (report intermediate values)
            filtered_feature_info: Tuple of (filtered_feature_names, n_removed) from
                adversarial pre-filtering. If None, uses all features.

        Returns:
            Dict with per-fold and aggregate metrics including neg_log_loss (primary).
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

        # P3: Use adversarial-filtered feature set if available
        if filtered_feature_info is not None:
            filtered_names, _ = filtered_feature_info
            feature_columns = [c for c in feature_columns if c in filtered_names]

        X = df[feature_columns].values
        X = np.nan_to_num(X, nan=0.0)
        y = self.prepare_target(df)

        # Get dates for embargo computation
        dates = pd.to_datetime(df["date"]) if "date" in df.columns else None

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
            if dates is not None:
                dates = dates[valid_mask].reset_index(drop=True)

        # P0: Compute embargo from trial's feature params (dynamic per-trial)
        max_lookback = max(
            getattr(feature_config, "form_window", 5),
            getattr(feature_config, "ema_span", 10),
            getattr(feature_config, "poisson_lookback", 10),
            20,  # niche window max
        )
        embargo_days = int(max_lookback * 3.5) + 7

        # Walk-forward validation - track per-fold metrics
        n_samples = len(y)
        fold_size = n_samples // (self.n_folds + 1)

        fold_precisions = []
        fold_rois = []
        fold_n_bets = []
        fold_log_losses = []
        threshold = self.config["default_threshold"]

        for fold in range(self.n_folds):
            train_end = (fold + 1) * fold_size

            # P0: Date-based embargo to prevent feature lookback leakage
            if dates is not None:
                boundary_date = dates.iloc[min(train_end, len(dates) - 1)]
                embargo_end = boundary_date + pd.Timedelta(days=embargo_days)
                remaining = dates.iloc[train_end:]
                post_embargo = remaining >= embargo_end
                if post_embargo.sum() >= 20:
                    test_start = train_end + int((~post_embargo).sum())
                else:
                    # Fallback: fixed sample buffer
                    test_start = train_end + max(50, int(n_samples * 0.02))
            else:
                test_start = train_end + max(50, int(n_samples * 0.02))

            test_end = min(test_start + fold_size, n_samples)

            X_train, y_train = X[:train_end], y[:train_end]
            X_test, y_test = X[test_start:test_end], y[test_start:test_end]
            odds_test = odds[test_start:test_end]

            if len(X_train) < 100 or len(X_test) < 20:
                continue

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # P2: Multi-model evaluation weighted toward production architecture
            models = {
                "catboost": (
                    cb.CatBoostClassifier(
                        iterations=200,
                        depth=4,
                        learning_rate=0.1,
                        random_seed=42,
                        verbose=0,
                    ),
                    0.5,
                    False,  # use_scaled: CatBoost handles raw features natively
                ),
                "lightgbm": (
                    lgb.LGBMClassifier(
                        n_estimators=200,
                        max_depth=5,
                        learning_rate=0.1,
                        random_state=42,
                        verbose=-1,
                    ),
                    0.25,
                    True,
                ),
                "xgboost": (
                    xgb.XGBClassifier(
                        n_estimators=200,
                        max_depth=5,
                        learning_rate=0.1,
                        random_state=42,
                        verbosity=0,
                    ),
                    0.25,
                    True,
                ),
            }

            weighted_ll_parts = []
            total_weight = 0.0

            for name, (model, weight, use_scaled) in models.items():
                try:
                    X_tr = X_train_scaled if use_scaled else X_train
                    X_te = X_test_scaled if use_scaled else X_test
                    model.fit(X_tr, y_train)
                    probs = model.predict_proba(X_te)[:, 1]
                    ll = sklearn_log_loss(y_test, probs, labels=[0, 1])
                    weighted_ll_parts.append(ll * weight)
                    total_weight += weight
                except Exception:
                    # Penalty: random baseline log_loss
                    weighted_ll_parts.append(0.693 * weight)
                    total_weight += weight

            fold_ll = sum(weighted_ll_parts) / total_weight if total_weight > 0 else 0.693
            fold_log_losses.append(fold_ll)

            # P1: Prune on running mean log_loss (negate: lower loss = higher value)
            if trial is not None and fold_log_losses:
                trial.report(-np.mean(fold_log_losses), fold)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            # Also compute ROI/precision for logging (use CatBoost probs for betting sim)
            try:
                cb_model = models["catboost"][0]
                probs = cb_model.predict_proba(X_test)[:, 1]
            except Exception:
                continue

            mask = (probs >= threshold) & (odds_test >= 1.5) & (odds_test <= 6.0)
            n_bets_fold = mask.sum()

            if n_bets_fold >= 5:
                wins = y_test[mask].sum()
                precision_fold = wins / n_bets_fold
                returns = np.where(y_test[mask] == 1, odds_test[mask] - 1, -1)
                roi_fold = returns.mean() * 100

                fold_precisions.append(precision_fold)
                fold_rois.append(roi_fold)
                fold_n_bets.append(n_bets_fold)

        # Need at least 2 folds for meaningful evaluation
        if len(fold_log_losses) < 2:
            return {
                "fold_precisions": [],
                "fold_rois": [],
                "fold_n_bets": [],
                "fold_log_losses": [],
                "precision": 0.0,
                "roi": -100.0,
                "n_bets": 0,
                "neg_log_loss": -0.693,  # random baseline penalty
                "sharpe": -10.0,
                "embargo_days": embargo_days,
            }

        # P1: Primary objective — neg_log_loss (proper scoring rule)
        neg_log_loss = -np.mean(fold_log_losses)

        # Legacy metrics for logging
        mean_precision = np.mean(fold_precisions) if fold_precisions else 0.0
        mean_roi = np.mean(fold_rois) if fold_rois else -100.0
        total_bets = sum(fold_n_bets)
        std_roi = np.std(fold_rois) if len(fold_rois) > 1 else 0.0
        sharpe = mean_roi - std_roi

        return {
            "fold_precisions": fold_precisions,
            "fold_rois": fold_rois,
            "fold_n_bets": fold_n_bets,
            "fold_log_losses": fold_log_losses,
            "precision": mean_precision,
            "roi": mean_roi,
            "n_bets": total_bets,
            "neg_log_loss": neg_log_loss,
            "sharpe": sharpe,
            "embargo_days": embargo_days,
        }

    def create_objective(
        self,
        features_df: pd.DataFrame,
        filtered_feature_info: Optional[Tuple[List[str], int]] = None,
    ):
        """Create Optuna objective function.

        Optimizes for neg_log_loss (proper scoring rule). Log loss is threshold-
        independent and measures prediction quality directly, unlike ROI/Sharpe
        which conflate prediction with bet sizing/threshold effects.
        """

        def objective(trial):
            feature_config = self.create_feature_config_from_trial(trial)
            metrics = self.evaluate_config(
                feature_config,
                features_df,
                trial=trial,
                filtered_feature_info=filtered_feature_info,
            )

            # Store all metrics in trial for analysis
            trial.set_user_attr("neg_log_loss", metrics["neg_log_loss"])
            trial.set_user_attr("precision", metrics["precision"])
            trial.set_user_attr("roi", metrics["roi"])
            trial.set_user_attr("n_bets", metrics["n_bets"])
            trial.set_user_attr("sharpe", metrics["sharpe"])
            trial.set_user_attr("embargo_days", metrics["embargo_days"])
            trial.set_user_attr("fold_precisions", metrics["fold_precisions"])
            trial.set_user_attr("fold_log_losses", metrics.get("fold_log_losses", []))
            trial.set_user_attr("params_hash", feature_config.params_hash())

            # Primary objective: neg_log_loss (higher = better predictions)
            return metrics["neg_log_loss"]

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

        # P3: Adversarial filter — remove temporally leaky features before Optuna loop
        filtered_feature_info = None
        adversarial_removed = 0
        if features_df is not None:
            feature_columns = self.get_feature_columns(features_df)
            X_all = features_df[feature_columns].values
            X_all = np.nan_to_num(X_all, nan=0.0)

            logger.info(f"Running adversarial filter on {len(feature_columns)} features...")
            _, filtered_names, adv_diag = _adversarial_filter(
                X_all,
                feature_columns,
                max_passes=2,
                auc_threshold=0.75,
                importance_threshold=0.05,
                max_features_per_pass=10,
            )
            adversarial_removed = adv_diag["total_removed"]
            if adversarial_removed > 0:
                filtered_feature_info = (filtered_names, adversarial_removed)
                logger.info(
                    f"Adversarial filter removed {adversarial_removed} features "
                    f"({len(feature_columns)} → {len(filtered_names)})"
                )
            else:
                logger.info("Adversarial filter: no temporally leaky features found")

        # Run Optuna — maximize neg_log_loss (proper scoring rule)
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
        )

        objective = self.create_objective(features_df, filtered_feature_info)

        # Time budget callback: gracefully stop when time runs out
        callbacks = []
        start_time = time.time()
        if self.time_budget_minutes > 0:
            time_budget_seconds = self.time_budget_minutes * 60
            logger.info(f"Time budget: {self.time_budget_minutes} minutes")

            def time_budget_callback(study, trial):
                elapsed = time.time() - start_time
                if elapsed > time_budget_seconds:
                    remaining_trials = self.n_trials - trial.number - 1
                    logger.warning(
                        f"Time budget exceeded ({elapsed/60:.1f} min > {self.time_budget_minutes} min). "
                        f"Stopping with {trial.number + 1} trials completed "
                        f"({remaining_trials} remaining trials skipped)."
                    )
                    study.stop()

            callbacks.append(time_budget_callback)

        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True, callbacks=callbacks)
        elapsed_min = (time.time() - start_time) / 60
        logger.info(f"Optimization completed in {elapsed_min:.1f} minutes ({len(study.trials)} trials)")

        # Get best result (by neg_log_loss)
        best_trial = study.best_trial
        best_neg_ll = best_trial.value
        best_sharpe = best_trial.user_attrs.get("sharpe", 0.0)
        best_precision = best_trial.user_attrs.get("precision", 0.0)
        best_roi = best_trial.user_attrs.get("roi", 0.0)
        best_n_bets = best_trial.user_attrs.get("n_bets", 0)
        best_embargo = best_trial.user_attrs.get("embargo_days", 0)
        fold_precisions = best_trial.user_attrs.get("fold_precisions", [])
        fold_log_losses = best_trial.user_attrs.get("fold_log_losses", [])

        logger.info(f"\nBest trial (optimized for neg_log_loss):")
        logger.info(f"  neg_log_loss: {best_neg_ll:.4f} (log_loss: {-best_neg_ll:.4f})")
        if fold_log_losses:
            logger.info(f"  Fold log_losses: {[f'{ll:.4f}' for ll in fold_log_losses]}")
        logger.info(f"  Embargo: {best_embargo} days")
        logger.info(f"  Mean Precision: {best_precision*100:.1f}%")
        if fold_precisions:
            logger.info(f"  Fold Precisions: {[f'{p*100:.1f}%' for p in fold_precisions]}")
        logger.info(f"  Mean ROI: {best_roi:+.1f}% (Sharpe: {best_sharpe:.2f})")
        logger.info(f"  Total Bets: {best_n_bets}")
        logger.info(f"  Params: {best_trial.params}")

        # Collect all trials
        all_trials = []
        for trial in study.trials:
            all_trials.append({
                "number": trial.number,
                "params": trial.params,
                "neg_log_loss": trial.value if trial.value is not None else -0.693,
                "sharpe": trial.user_attrs.get("sharpe", -10.0),
                "precision": trial.user_attrs.get("precision", 0.0),
                "roi": trial.user_attrs.get("roi", 0.0),
                "n_bets": trial.user_attrs.get("n_bets", 0),
            })

        # Create result
        result = FeatureOptimizationResult(
            bet_type=self.bet_type,
            best_params=best_trial.params,
            neg_log_loss=best_neg_ll,
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
            embargo_days=best_embargo,
            adversarial_features_removed=adversarial_removed,
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
    print("\n" + "=" * 110)
    print("                     FEATURE PARAMETER OPTIMIZATION RESULTS")
    print("                     (Optimized for neg_log_loss — proper scoring rule)")
    print("=" * 110)

    print(f"\n{'Bet Type':<12} {'LogLoss':>8} {'Precision':>10} {'ROI':>10} {'Bets':>8} {'Embargo':>8} {'AdvRem':>7}")
    print("-" * 110)

    for r in sorted(results, key=lambda x: x.neg_log_loss, reverse=True):
        print(
            f"{r.bet_type:<12} {-r.neg_log_loss:>8.4f} {r.precision*100:>9.1f}% "
            f"{r.roi:>+9.1f}% {r.n_bets:>8} {r.embargo_days:>7}d {r.adversarial_features_removed:>7}"
        )

    print("-" * 110)
    print("\nLogLoss = mean log_loss across folds (lower = better). Embargo = days between train/test.")
    print("AdvRem = features removed by adversarial filter.")

    # Show per-fold details for best result
    if results:
        best = max(results, key=lambda x: x.neg_log_loss)
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
    parser.add_argument("--n-folds", type=int, default=3,
                       help="Walk-forward folds (3 is sufficient for comparing feature configs)")
    parser.add_argument("--min-bets", type=int, default=30,
                       help="Minimum bets for valid configuration")
    parser.add_argument("--regenerate", action="store_true",
                       help="Actually regenerate features (slower but more accurate)")
    parser.add_argument("--save-config", action="store_true",
                       help="Save optimal configs to config/feature_params/")
    parser.add_argument("--feature-params-dir", type=str, default=None,
                       help="Custom feature params output directory (e.g., config/feature_params/americas)")
    parser.add_argument("--time-budget-minutes", type=int, default=0,
                       help="Time budget in minutes for optimization (0=unlimited). "
                            "Stops gracefully when exceeded, saving best-so-far result.")
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
        # Map line variants to base market (e.g. corners_over_85 -> corners, fouls_under_265 -> fouls)
        base_bet_type = re.sub(r'_(over|under)_\d+$', '', bet_type)
        if base_bet_type not in BET_TYPES:
            logger.warning(f"Unknown bet type: {bet_type}, skipping")
            continue
        if base_bet_type != bet_type:
            logger.info(f"Mapped line variant {bet_type} -> base market {base_bet_type}")

        optimizer = FeatureParamOptimizer(
            bet_type=base_bet_type,
            n_trials=args.n_trials,
            n_folds=args.n_folds,
            min_bets=args.min_bets,
            use_regeneration=args.regenerate,
            time_budget_minutes=args.time_budget_minutes,
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
