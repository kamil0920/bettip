#!/usr/bin/env python3
"""
Rigorous Walk-Forward Validation for Sniper Mode

This script performs TRUE out-of-sample validation by:
1. For each fold, training models ONLY on past data
2. Testing on future data that the model has never seen
3. No data leakage - models are retrained fresh for each fold

This is the gold standard for validating trading/betting strategies.

Usage:
    python experiments/sniper_walkforward_validation.py
    python experiments/sniper_walkforward_validation.py --n-folds 5 --target-precision 0.85
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
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Paths
FEATURES_FILE = Path("data/03-features/features_all_5leagues_with_odds.parquet")
OUTPUT_DIR = Path("experiments/outputs/sniper_mode")


@dataclass
class FoldResult:
    """Result for a single fold."""
    fold: int
    train_size: int
    test_size: int
    n_bets: int
    wins: int
    losses: int
    precision: float
    roi: float
    avg_odds: float
    bets_detail: List[Dict]
    feature_importance: Dict[str, Dict[str, float]] = None  # model -> {feature: importance}
    optimized_thresholds: Dict[str, float] = None  # Thresholds selected on validation set


@dataclass
class FeatureStability:
    """Feature importance stability across folds."""
    feature: str
    mean_importance: float
    std_importance: float
    stability_score: float  # mean / (std + 1e-6) - higher = more stable
    rank: int


@dataclass
class ValidationResult:
    """Overall validation result."""
    config_name: str
    n_folds: int
    total_bets: int
    total_wins: int
    overall_precision: float
    avg_precision: float
    std_precision: float
    min_precision: float
    max_precision: float
    avg_roi: float
    std_roi: float
    fold_results: List[Dict]
    meets_target: bool


class WalkForwardValidator:
    """
    Rigorous walk-forward validation with model retraining.

    For each fold:
    1. Train models on all data BEFORE the test period
    2. Optimize thresholds on inner validation split (NOT test data)
    3. Apply sniper filters with optimized thresholds
    4. Test on the next time period
    5. Record results
    6. Move forward in time
    """

    def __init__(
        self,
        target_precision: float = 0.90,
        min_bets_per_fold: int = 2,
        n_folds: int = 5,
        validation_ratio: float = 0.2,
    ):
        self.target_precision = target_precision
        self.min_bets_per_fold = min_bets_per_fold
        self.n_folds = n_folds
        self.validation_ratio = validation_ratio  # Inner validation split ratio
        self.features_df = None

        # Threshold search space for nested optimization
        self.threshold_search_space = {
            "primary_threshold": [0.55, 0.60, 0.65, 0.70, 0.75],
            "consensus_threshold": [0.50, 0.55, 0.60, 0.65],
        }

        # Features to use (from optimization results)
        self.feature_cols = [
            "home_elo", "odds_move_home", "odds_away_prob", "odds_home_prob",
            "away_draws_last_n", "home_goals_conceded_ema", "away_late_goal_rate",
            "home_cards_ema", "odds_prob_diff", "away_goals_scored_ema",
            "poisson_away_win_prob", "fouls_diff", "away_early_goal_rate",
            "ref_matches", "odds_move_away", "ppg_diff", "ref_home_win_pct",
            "ref_draw_pct", "home_shot_accuracy", "home_attack_strength",
            "away_clean_sheet_streak", "home_defense_strength",
            "home_goals_scored_ema", "ref_avg_goals", "cards_diff",
            "away_corners_conceded_ema", "away_points_ema", "home_away_ppg_diff",
            "home_home_draws", "poisson_draw_prob", "away_avg_yellows",
            "home_home_goals_conceded", "away_goals_conceded_ema",
            "away_corners_won_ema", "home_away_gd_diff", "overround_change",
            "home_goals_scored_last_n", "expected_total_shots",
        ]

    def load_data(self) -> pd.DataFrame:
        """Load and prepare feature data."""
        if not FEATURES_FILE.exists():
            raise FileNotFoundError(f"Features file not found: {FEATURES_FILE}")

        from src.utils.data_io import load_features
        df = load_features(FEATURES_FILE)
        logger.info(f"Loaded {len(df)} matches")

        # Sort by date
        date_col = None
        for col in ["date", "match_date", "kickoff", "fixture_date"]:
            if col in df.columns:
                date_col = col
                df[col] = pd.to_datetime(df[col])
                break

        if date_col:
            df = df.sort_values(date_col).reset_index(drop=True)
            logger.info(f"Sorted by {date_col}")
        else:
            logger.warning("No date column found!")

        # Ensure target exists
        if "away_win" not in df.columns:
            if "result" in df.columns:
                df["away_win"] = (df["result"] == "A").astype(int)
            else:
                raise ValueError("No target column found")

        # Get available features
        self.available_features = [f for f in self.feature_cols if f in df.columns]
        logger.info(f"Using {len(self.available_features)}/{len(self.feature_cols)} features")

        self.features_df = df
        return df

    def train_models(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray
    ) -> Dict[str, Any]:
        """Train all models on training data."""
        models = {}

        # Prepare features
        X = X_train[self.available_features].copy()
        X = X.fillna(0)

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 1. CatBoost
        try:
            cb_model = CatBoostClassifier(
                iterations=500,
                depth=6,
                learning_rate=0.05,
                l2_leaf_reg=10,
                random_seed=42,
                verbose=False,
            )
            cb_model.fit(X_scaled, y_train)

            # Calibrate
            cb_calibrated = CalibratedClassifierCV(cb_model, cv="prefit", method="sigmoid")
            cb_calibrated.fit(X_scaled, y_train)

            models["catboost"] = {
                "model": cb_calibrated,
                "scaler": scaler,
            }
        except Exception as e:
            logger.warning(f"CatBoost training failed: {e}")

        # 2. LightGBM
        try:
            lgb_model = lgb.LGBMClassifier(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                reg_alpha=2,
                reg_lambda=3,
                random_state=42,
                verbose=-1,
            )
            lgb_model.fit(X_scaled, y_train)

            # Calibrate
            lgb_calibrated = CalibratedClassifierCV(lgb_model, cv="prefit", method="sigmoid")
            lgb_calibrated.fit(X_scaled, y_train)

            models["lightgbm"] = {
                "model": lgb_calibrated,
                "scaler": scaler,
            }
        except Exception as e:
            logger.warning(f"LightGBM training failed: {e}")

        # 3. XGBoost
        try:
            xgb_model = xgb.XGBClassifier(
                n_estimators=500,
                max_depth=4,
                learning_rate=0.1,
                reg_alpha=4,
                reg_lambda=2,
                random_state=42,
                verbosity=0,
            )
            xgb_model.fit(X_scaled, y_train)

            # Calibrate
            xgb_calibrated = CalibratedClassifierCV(xgb_model, cv="prefit", method="sigmoid")
            xgb_calibrated.fit(X_scaled, y_train)

            models["xgboost"] = {
                "model": xgb_calibrated,
                "scaler": scaler,
            }
        except Exception as e:
            logger.warning(f"XGBoost training failed: {e}")

        return models

    def extract_feature_importance(
        self,
        models: Dict[str, Any],
    ) -> Dict[str, Dict[str, float]]:
        """Extract feature importance from all trained models."""
        all_importance = {}

        for name, model_data in models.items():
            model = model_data["model"]

            try:
                # CalibratedClassifierCV wraps the actual model
                if hasattr(model, 'calibrated_classifiers_'):
                    # Get the base estimator from the first calibrated classifier
                    base_model = model.calibrated_classifiers_[0].estimator
                elif hasattr(model, 'estimator'):
                    base_model = model.estimator
                else:
                    base_model = model

                # Extract importance based on model type
                importance = None

                if hasattr(base_model, 'feature_importances_'):
                    # CatBoost, LightGBM, XGBoost all have this
                    importance = base_model.feature_importances_
                elif hasattr(base_model, 'coef_'):
                    # Logistic regression
                    importance = np.abs(base_model.coef_[0])

                if importance is not None:
                    # Normalize to sum to 1
                    importance = importance / importance.sum() if importance.sum() > 0 else importance

                    # Map to feature names
                    feature_dict = {}
                    for i, feat in enumerate(self.available_features):
                        if i < len(importance):
                            feature_dict[feat] = float(importance[i])

                    all_importance[name] = feature_dict

            except Exception as e:
                logger.warning(f"Could not extract importance for {name}: {e}")

        return all_importance

    def predict_proba(
        self,
        models: Dict[str, Any],
        X: pd.DataFrame
    ) -> Dict[str, np.ndarray]:
        """Get probability predictions from all models."""
        predictions = {}

        X_feat = X[self.available_features].copy()
        X_feat = X_feat.fillna(0)

        for name, model_data in models.items():
            model = model_data["model"]
            scaler = model_data["scaler"]

            try:
                X_scaled = scaler.transform(X_feat)
                probs = model.predict_proba(X_scaled)[:, 1]
                predictions[name] = probs
            except Exception as e:
                logger.warning(f"Prediction failed for {name}: {e}")

        return predictions

    def apply_sniper_filter(
        self,
        predictions: Dict[str, np.ndarray],
        df: pd.DataFrame,
        config: Dict[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply sniper mode filters.

        Config options:
        - primary_model: Main model to use
        - primary_threshold: Min probability for primary model
        - consensus_model: Secondary model for consensus
        - consensus_threshold: Min probability for consensus
        - min_odds, max_odds: Odds range filter
        """
        n = len(df)
        mask = np.ones(n, dtype=bool)

        # Primary model filter
        primary = config.get("primary_model", "catboost")
        primary_thresh = config.get("primary_threshold", 0.60)

        if primary in predictions:
            mask &= predictions[primary] >= primary_thresh
        else:
            return np.array([]), np.array([])

        # Consensus filter
        consensus = config.get("consensus_model")
        consensus_thresh = config.get("consensus_threshold", 0.55)

        if consensus and consensus in predictions:
            mask &= predictions[consensus] >= consensus_thresh

        # Odds filter
        min_odds = config.get("min_odds", 2.0)
        max_odds = config.get("max_odds", 5.0)

        if "avg_away_open" in df.columns:
            odds = df["avg_away_open"].values
            mask &= (odds >= min_odds) & (odds <= max_odds)
        elif "odds_away_prob" in df.columns:
            odds = 1 / df["odds_away_prob"].values
            mask &= (odds >= min_odds) & (odds <= max_odds)
        else:
            odds = np.ones(n) * 3.0

        return mask, odds

    def _optimize_threshold_inner(
        self,
        models: Dict[str, Any],
        val_df: pd.DataFrame,
        base_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Optimize thresholds on inner validation split.

        CRITICAL: This ensures thresholds are never optimized using test data.
        We search over threshold combinations and select the one that maximizes
        expected value (precision × avg_odds - 1) on the validation set.

        Args:
            models: Trained models
            val_df: Validation dataframe (NOT test data)
            base_config: Base configuration with model choices

        Returns:
            Optimized config with best thresholds
        """
        predictions = self.predict_proba(models, val_df)
        y_val = val_df["away_win"].values

        # Get odds for expected value calculation
        if "avg_away_open" in val_df.columns:
            odds = val_df["avg_away_open"].values
        elif "odds_away_prob" in val_df.columns:
            odds = 1 / val_df["odds_away_prob"].values
        else:
            odds = np.ones(len(val_df)) * 3.0

        best_config = base_config.copy()
        best_ev = -np.inf  # Expected value

        primary_model = base_config.get("primary_model", "catboost")
        consensus_model = base_config.get("consensus_model")

        if primary_model not in predictions:
            return best_config

        # Search over threshold combinations
        for primary_thresh in self.threshold_search_space["primary_threshold"]:
            consensus_thresholds = [None] if consensus_model is None else self.threshold_search_space["consensus_threshold"]

            for consensus_thresh in consensus_thresholds:
                # Build test config
                test_config = base_config.copy()
                test_config["primary_threshold"] = primary_thresh
                if consensus_thresh is not None:
                    test_config["consensus_threshold"] = consensus_thresh

                # Apply filter
                mask, _ = self.apply_sniper_filter(predictions, val_df, test_config)
                bet_indices = np.where(mask)[0]

                if len(bet_indices) < self.min_bets_per_fold:
                    continue

                # Calculate expected value on validation set
                bet_outcomes = y_val[bet_indices]
                bet_odds = odds[bet_indices]

                precision = bet_outcomes.mean()
                avg_odds = bet_odds.mean()

                # Expected value = precision × avg_odds - 1
                # This is the expected return per unit bet
                ev = precision * avg_odds - 1

                # Prefer configurations with positive EV and reasonable volume
                score = ev * np.sqrt(len(bet_indices))  # Volume-adjusted EV

                if score > best_ev and ev > 0:
                    best_ev = score
                    best_config = test_config.copy()
                    logger.debug(
                        f"  New best threshold: primary={primary_thresh}, "
                        f"consensus={consensus_thresh}, EV={ev:.3f}, n={len(bet_indices)}"
                    )

        return best_config

    def validate_fold(
        self,
        fold: int,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        config: Dict[str, Any],
    ) -> FoldResult:
        """
        Validate one fold with nested threshold optimization.

        CRITICAL: Thresholds are optimized on an inner validation split,
        NOT on the test data. This prevents threshold overfitting.

        Steps:
        1. Split train into train_inner (80%) and val_inner (20%)
        2. Train models on train_inner
        3. Optimize thresholds on val_inner
        4. Evaluate on test_df with optimized thresholds
        """
        # Split training data into inner train and validation
        n_train = len(train_df)
        val_size = int(n_train * self.validation_ratio)
        train_inner_size = n_train - val_size

        # Ensure we keep temporal ordering - validation is the LAST part of training period
        train_inner_df = train_df.iloc[:train_inner_size].copy()
        val_inner_df = train_df.iloc[train_inner_size:].copy()

        logger.debug(
            f"  Fold {fold}: train_inner={len(train_inner_df)}, "
            f"val_inner={len(val_inner_df)}, test={len(test_df)}"
        )

        # Train models on inner training data only
        X_train = train_inner_df
        y_train = train_inner_df["away_win"].values

        models = self.train_models(X_train, y_train)

        if not models:
            return FoldResult(
                fold=fold,
                train_size=len(train_df),
                test_size=len(test_df),
                n_bets=0,
                wins=0,
                losses=0,
                precision=0.0,
                roi=0.0,
                avg_odds=0.0,
                bets_detail=[],
                feature_importance={},
                optimized_thresholds=None,
            )

        # Extract feature importance
        feature_importance = self.extract_feature_importance(models)

        # CRITICAL: Optimize thresholds on inner validation set, NOT test data
        # This prevents threshold overfitting
        optimized_config = self._optimize_threshold_inner(models, val_inner_df, config)

        logger.debug(
            f"  Fold {fold} optimized thresholds: "
            f"primary={optimized_config.get('primary_threshold')}, "
            f"consensus={optimized_config.get('consensus_threshold')}"
        )

        # Get predictions on test data
        predictions = self.predict_proba(models, test_df)

        # Apply sniper filter with OPTIMIZED thresholds (not original config)
        mask, odds = self.apply_sniper_filter(predictions, test_df, optimized_config)

        # Get qualifying bets
        bet_indices = np.where(mask)[0]
        n_bets = len(bet_indices)

        if n_bets == 0:
            return FoldResult(
                fold=fold,
                train_size=len(train_df),
                test_size=len(test_df),
                n_bets=0,
                wins=0,
                losses=0,
                precision=0.0,
                roi=0.0,
                avg_odds=0.0,
                bets_detail=[],
                feature_importance=feature_importance,
                optimized_thresholds={
                    "primary_threshold": optimized_config.get("primary_threshold"),
                    "consensus_threshold": optimized_config.get("consensus_threshold"),
                },
            )

        # Calculate results
        y_test = test_df["away_win"].values
        bet_outcomes = y_test[bet_indices]
        bet_odds = odds[bet_indices]

        wins = int(bet_outcomes.sum())
        losses = n_bets - wins
        precision = wins / n_bets if n_bets > 0 else 0

        # ROI calculation
        returns = np.where(bet_outcomes == 1, bet_odds - 1, -1)
        roi = returns.mean() * 100 if len(returns) > 0 else 0

        # Bet details
        bets_detail = []
        for i, idx in enumerate(bet_indices[:10]):  # First 10
            row = test_df.iloc[idx]
            bets_detail.append({
                "home_team": row.get("home_team", "Unknown"),
                "away_team": row.get("away_team", "Unknown"),
                "league": row.get("league", "Unknown"),
                "outcome": "WIN" if bet_outcomes[i] == 1 else "LOSS",
                "odds": float(bet_odds[i]),
            })

        return FoldResult(
            fold=fold,
            train_size=len(train_df),
            test_size=len(test_df),
            n_bets=n_bets,
            wins=wins,
            losses=losses,
            precision=precision,
            roi=roi,
            avg_odds=float(bet_odds.mean()) if len(bet_odds) > 0 else 0,
            bets_detail=bets_detail,
            feature_importance=feature_importance,
            optimized_thresholds={
                "primary_threshold": optimized_config.get("primary_threshold"),
                "consensus_threshold": optimized_config.get("consensus_threshold"),
            },
        )

    def validate_config(
        self,
        config: Dict[str, Any],
    ) -> ValidationResult:
        """Run full walk-forward validation for a config."""
        df = self.features_df
        n = len(df)

        # Calculate fold boundaries
        # Use expanding window: train on all data up to fold, test on fold
        fold_size = n // (self.n_folds + 1)

        fold_results = []

        for fold in range(self.n_folds):
            # Training: all data up to test start
            train_end = (fold + 1) * fold_size
            test_start = train_end
            test_end = min(test_start + fold_size, n)

            if test_end <= test_start:
                continue

            train_df = df.iloc[:train_end].copy()
            test_df = df.iloc[test_start:test_end].copy()

            logger.info(f"Fold {fold + 1}: Train {len(train_df)}, Test {len(test_df)}")

            result = self.validate_fold(fold + 1, train_df, test_df, config)
            fold_results.append(result)

            if result.n_bets > 0:
                logger.info(
                    f"  -> {result.wins}/{result.n_bets} = {result.precision:.1%} precision, "
                    f"ROI: {result.roi:.1f}%"
                )
            else:
                logger.info("  -> No qualifying bets")

        # Aggregate results
        valid_folds = [f for f in fold_results if f.n_bets > 0]

        if not valid_folds:
            return ValidationResult(
                config_name=config.get("name", "unknown"),
                n_folds=0,
                total_bets=0,
                total_wins=0,
                overall_precision=0,
                avg_precision=0,
                std_precision=0,
                min_precision=0,
                max_precision=0,
                avg_roi=0,
                std_roi=0,
                fold_results=[asdict(f) for f in fold_results],
                meets_target=False,
            )

        total_bets = sum(f.n_bets for f in valid_folds)
        total_wins = sum(f.wins for f in valid_folds)
        precisions = [f.precision for f in valid_folds]
        rois = [f.roi for f in valid_folds]

        return ValidationResult(
            config_name=config.get("name", "unknown"),
            n_folds=len(valid_folds),
            total_bets=total_bets,
            total_wins=total_wins,
            overall_precision=total_wins / total_bets if total_bets > 0 else 0,
            avg_precision=np.mean(precisions),
            std_precision=np.std(precisions) if len(precisions) > 1 else 0,
            min_precision=min(precisions),
            max_precision=max(precisions),
            avg_roi=np.mean(rois),
            std_roi=np.std(rois) if len(rois) > 1 else 0,
            fold_results=[asdict(f) for f in fold_results],
            meets_target=total_wins / total_bets >= self.target_precision if total_bets > 0 else False,
        )


def aggregate_feature_importance(
    fold_results: List[FoldResult],
    model_name: str = "catboost",
) -> List[FeatureStability]:
    """
    Aggregate feature importance across folds and compute stability metrics.

    Returns features ranked by stability score (high mean, low variance).
    """
    # Collect importance values per feature
    feature_values = {}

    for fold in fold_results:
        if fold.feature_importance and model_name in fold.feature_importance:
            imp = fold.feature_importance[model_name]
            for feat, value in imp.items():
                if feat not in feature_values:
                    feature_values[feat] = []
                feature_values[feat].append(value)

    if not feature_values:
        return []

    # Compute statistics
    stability_results = []
    for feat, values in feature_values.items():
        if len(values) > 0:
            mean_imp = np.mean(values)
            std_imp = np.std(values) if len(values) > 1 else 0
            stability = mean_imp / (std_imp + 1e-6)  # Higher = more stable

            stability_results.append(FeatureStability(
                feature=feat,
                mean_importance=mean_imp,
                std_importance=std_imp,
                stability_score=stability,
                rank=0,  # Will be set after sorting
            ))

    # Sort by stability score (high mean, low variance)
    stability_results.sort(key=lambda x: x.stability_score, reverse=True)

    # Assign ranks
    for i, r in enumerate(stability_results):
        r.rank = i + 1

    return stability_results


def main():
    parser = argparse.ArgumentParser(description="Rigorous Walk-Forward Validation")
    parser.add_argument("--n-folds", type=int, default=5, help="Number of folds")
    parser.add_argument("--target-precision", type=float, default=0.90, help="Target precision")
    args = parser.parse_args()

    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║           RIGOROUS WALK-FORWARD VALIDATION - SNIPER MODE                     ║
║                                                                              ║
║  This is the TRUE test: models retrained on PAST data only for each fold    ║
║  No data leakage - each test period is completely out-of-sample             ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

    validator = WalkForwardValidator(
        target_precision=args.target_precision,
        n_folds=args.n_folds,
    )

    # Load data
    logger.info("Loading data...")
    validator.load_data()

    # Define configs to test (based on previous results)
    configs = [
        {
            "name": "catboost_0.60+lightgbm_0.55",
            "primary_model": "catboost",
            "primary_threshold": 0.60,
            "consensus_model": "lightgbm",
            "consensus_threshold": 0.55,
            "min_odds": 2.0,
            "max_odds": 5.0,
        },
        {
            "name": "catboost_0.65+lightgbm_0.55",
            "primary_model": "catboost",
            "primary_threshold": 0.65,
            "consensus_model": "lightgbm",
            "consensus_threshold": 0.55,
            "min_odds": 2.0,
            "max_odds": 5.0,
        },
        {
            "name": "catboost_0.70+lightgbm_0.60",
            "primary_model": "catboost",
            "primary_threshold": 0.70,
            "consensus_model": "lightgbm",
            "consensus_threshold": 0.60,
            "min_odds": 2.0,
            "max_odds": 5.0,
        },
        {
            "name": "lightgbm_0.60+catboost_0.55",
            "primary_model": "lightgbm",
            "primary_threshold": 0.60,
            "consensus_model": "catboost",
            "consensus_threshold": 0.55,
            "min_odds": 2.0,
            "max_odds": 5.0,
        },
        {
            "name": "xgboost_0.65+catboost_0.60",
            "primary_model": "xgboost",
            "primary_threshold": 0.65,
            "consensus_model": "catboost",
            "consensus_threshold": 0.60,
            "min_odds": 2.0,
            "max_odds": 5.0,
        },
        {
            "name": "catboost_0.60_solo",
            "primary_model": "catboost",
            "primary_threshold": 0.60,
            "consensus_model": None,
            "min_odds": 2.0,
            "max_odds": 5.0,
        },
        {
            "name": "catboost_0.70_solo",
            "primary_model": "catboost",
            "primary_threshold": 0.70,
            "consensus_model": None,
            "min_odds": 2.0,
            "max_odds": 5.0,
        },
        {
            "name": "lightgbm_0.65_solo",
            "primary_model": "lightgbm",
            "primary_threshold": 0.65,
            "consensus_model": None,
            "min_odds": 2.0,
            "max_odds": 5.0,
        },
    ]

    results = []

    print(f"\nTesting {len(configs)} configurations with {args.n_folds} folds each...\n")
    print("="*80)

    for config in configs:
        print(f"\n>>> Testing: {config['name']}")
        result = validator.validate_config(config)
        results.append(result)

    # Print summary
    print("\n" + "="*80)
    print("WALK-FORWARD VALIDATION RESULTS (Models Retrained Per Fold)")
    print("="*80)
    print(f"\n{'Config':<40} {'Folds':>6} {'Bets':>6} {'Wins':>6} {'Prec':>8} {'±Std':>8} {'ROI':>8}")
    print("-"*80)

    for r in sorted(results, key=lambda x: x.overall_precision, reverse=True):
        if r.total_bets > 0:
            print(
                f"{r.config_name:<40} "
                f"{r.n_folds:>6} "
                f"{r.total_bets:>6} "
                f"{r.total_wins:>6} "
                f"{r.overall_precision:>7.1%} "
                f"±{r.std_precision:>5.1%} "
                f"{r.avg_roi:>7.1f}%"
            )
        else:
            print(f"{r.config_name:<40} {'No qualifying bets':>50}")

    # Best result
    valid_results = [r for r in results if r.total_bets >= 5]
    if valid_results:
        best = max(valid_results, key=lambda x: x.overall_precision)

        print("\n" + "="*80)
        print("BEST VALIDATED CONFIGURATION")
        print("="*80)
        print(f"Config: {best.config_name}")
        print(f"\nOverall Results ({best.n_folds} folds with bets):")
        print(f"  Total bets: {best.total_bets}")
        print(f"  Total wins: {best.total_wins}")
        print(f"  Overall precision: {best.overall_precision:.1%}")
        print(f"  Precision range: {best.min_precision:.1%} - {best.max_precision:.1%}")
        print(f"  Avg ROI: {best.avg_roi:.1f}% (±{best.std_roi:.1f}%)")

        print(f"\nFold-by-Fold Breakdown:")
        for fold in best.fold_results:
            if fold["n_bets"] > 0:
                status = "✓" if fold["precision"] >= args.target_precision else "✗"
                print(
                    f"  Fold {fold['fold']}: {fold['wins']}/{fold['n_bets']} = "
                    f"{fold['precision']:.0%} precision, ROI: {fold['roi']:.1f}% {status}"
                )

        if best.overall_precision >= args.target_precision:
            print(f"\n✓ VALIDATED: {best.overall_precision:.1%} precision achieved!")
            print("  This strategy is ready for paper trading.")
        else:
            print(f"\n✗ Target {args.target_precision:.0%} not met.")
            print(f"  Best achieved: {best.overall_precision:.1%}")
            print("  Consider higher thresholds or additional filters.")

    # Feature importance analysis across folds
    if valid_results:
        best_config_results = [
            fold for r in results
            if r.config_name == best.config_name
            for fold in r.fold_results
        ]

        # Convert back to FoldResult objects
        fold_result_objects = []
        for fold_dict in best.fold_results:
            fr = FoldResult(
                fold=fold_dict.get("fold", 0),
                train_size=fold_dict.get("train_size", 0),
                test_size=fold_dict.get("test_size", 0),
                n_bets=fold_dict.get("n_bets", 0),
                wins=fold_dict.get("wins", 0),
                losses=fold_dict.get("losses", 0),
                precision=fold_dict.get("precision", 0),
                roi=fold_dict.get("roi", 0),
                avg_odds=fold_dict.get("avg_odds", 0),
                bets_detail=fold_dict.get("bets_detail", []),
                feature_importance=fold_dict.get("feature_importance", {}),
            )
            fold_result_objects.append(fr)

        print("\n" + "="*80)
        print("STABLE PREDICTORS (High Mean, Low Variance Importance)")
        print("="*80)

        for model_name in ["catboost", "lightgbm", "xgboost"]:
            stability = aggregate_feature_importance(fold_result_objects, model_name)
            if stability:
                print(f"\n{model_name.upper()} Top 15 Stable Features:")
                print(f"{'Rank':<6} {'Feature':<35} {'Mean':>10} {'Std':>10} {'Stability':>12}")
                print("-" * 75)
                for s in stability[:15]:
                    print(
                        f"{s.rank:<6} {s.feature:<35} "
                        f"{s.mean_importance:>10.4f} {s.std_importance:>10.4f} "
                        f"{s.stability_score:>12.2f}"
                    )

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Include feature stability in output
    feature_stability_output = {}
    if valid_results:
        fold_result_objects = []
        for fold_dict in best.fold_results:
            fr = FoldResult(
                fold=fold_dict.get("fold", 0),
                train_size=fold_dict.get("train_size", 0),
                test_size=fold_dict.get("test_size", 0),
                n_bets=fold_dict.get("n_bets", 0),
                wins=fold_dict.get("wins", 0),
                losses=fold_dict.get("losses", 0),
                precision=fold_dict.get("precision", 0),
                roi=fold_dict.get("roi", 0),
                avg_odds=fold_dict.get("avg_odds", 0),
                bets_detail=fold_dict.get("bets_detail", []),
                feature_importance=fold_dict.get("feature_importance", {}),
            )
            fold_result_objects.append(fr)

        for model_name in ["catboost", "lightgbm", "xgboost"]:
            stability = aggregate_feature_importance(fold_result_objects, model_name)
            if stability:
                feature_stability_output[model_name] = [
                    {
                        "rank": s.rank,
                        "feature": s.feature,
                        "mean_importance": s.mean_importance,
                        "std_importance": s.std_importance,
                        "stability_score": s.stability_score,
                    }
                    for s in stability[:20]
                ]

    output = {
        "generated_at": datetime.now().isoformat(),
        "target_precision": args.target_precision,
        "n_folds": args.n_folds,
        "results": [asdict(r) for r in results],
        "feature_stability": feature_stability_output,
    }

    output_path = OUTPUT_DIR / f"rigorous_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
