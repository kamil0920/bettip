"""
Model Loader Utility

Loads trained models for inference in pre-match intelligence system.
Supports both:
- Full optimization models (dict with model + metadata)
- Niche optimization models (just CalibratedClassifierCV + JSON features)
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import pandas as pd

from src.ml.prediction_health import (
    CalibrationStatus,
    FeatureMismatchSeverity,
    MarketHealthReport,
    classify_feature_mismatch,
)

logger = logging.getLogger(__name__)


class _UncalibratedEnsemble:
    """Fallback for models with degenerate isotonic calibration.

    Averages raw predict_proba from the base estimators across CV folds,
    bypassing the broken isotonic step function.
    """

    def __init__(self, calibrated_classifiers):
        self._estimators = [cc.estimator for cc in calibrated_classifiers]

    def predict_proba(self, X):
        import numpy as np

        probas = [est.predict_proba(X) for est in self._estimators]
        return np.mean(probas, axis=0)


class ModelLoader:
    """Load and manage trained betting models."""

    MODELS_DIR = Path("models")
    OUTPUTS_DIR = Path("experiments/outputs")

    # Map bet types to their model files and feature configs
    # Niche models: specific lines (e.g., fouls over 24.5)
    NICHE_MODELS = {
        # Fouls markets
        "fouls_over_24_5": {
            "model_file": "fouls_over_24_5_model.joblib",
            "config_file": "fouls_over_24_5_pipeline.json",
        },
        "fouls_over_22_5": {
            "model_file": "fouls_over_22_5_model.joblib",
            "config_file": "fouls_over_22_5_pipeline.json",
        },
        "fouls_over_26_5": {
            "model_file": "fouls_over_26_5_model.joblib",
            "config_file": "fouls_over_26_5_pipeline.json",
        },
        # Shots markets
        "shots_over_24_5": {
            "model_file": "shots_over_24_5_model.joblib",
            "config_file": "shots_over_24_5_pipeline.json",
        },
        "shots_over_22_5": {
            "model_file": "shots_over_22_5_model.joblib",
            "config_file": "shots_over_22_5_pipeline.json",
        },
        "shots_over_26_5": {
            "model_file": "shots_over_26_5_model.joblib",
            "config_file": "shots_over_26_5_pipeline.json",
        },
        # Corners markets
        "corners_over_10_5": {
            "model_file": "corners_over_10_5_model.joblib",
            "config_file": "corners_over_10_5_pipeline.json",
        },
        "corners_over_9_5": {
            "model_file": "corners_over_9_5_model.joblib",
            "config_file": "corners_over_9_5_pipeline.json",
        },
        "corners_over_11_5": {
            "model_file": "corners_over_11_5_model.joblib",
            "config_file": "corners_over_11_5_pipeline.json",
        },
        # Cards markets
        "cards_over_4_5": {
            "model_file": "cards_over_4_5_model.joblib",
            "config_file": "cards_over_4_5_pipeline.json",
        },
        "cards_over_3_5": {
            "model_file": "cards_over_3_5_model.joblib",
            "config_file": "cards_over_3_5_pipeline.json",
        },
        "cards_over_5_5": {
            "model_file": "cards_over_5_5_model.joblib",
            "config_file": "cards_over_5_5_pipeline.json",
        },
    }

    # Full optimization models: general markets (away_win, btts, etc.)
    FULL_OPTIMIZATION_MARKETS = [
        "away_win",
        "home_win",
        "btts",
        "over25",
        "under25",
        "asian_handicap",
        "shots",
        "fouls",
        "corners",
        "cards",
    ]

    # Model variants to look for per market
    MODEL_VARIANTS = [
        "xgboost",
        "lightgbm",
        "catboost",
        "logisticreg",
        "fastai",
        "two_stage_lgb",
        "two_stage_xgb",
    ]

    def __init__(self, models_dir: Optional[Path] = None, outputs_dir: Optional[Path] = None):
        self.models_dir = models_dir or self.MODELS_DIR
        self.outputs_dir = outputs_dir or self.OUTPUTS_DIR
        self._loaded_models: Dict[str, Dict[str, Any]] = {}

    def list_available_models(self) -> List[str]:
        """List all available trained models.

        Discovers models in two ways:
        1. Known markets x known variants (e.g. home_win_xgboost.joblib)
        2. Dynamic scan for line variant models (e.g. cards_over_35_lightgbm.joblib)
        """
        available = []
        found_files = set()

        # 1. Check for full optimization models (base markets)
        for market in self.FULL_OPTIMIZATION_MARKETS:
            for variant in self.MODEL_VARIANTS:
                model_path = self.models_dir / f"{market}_{variant}.joblib"
                if model_path.exists():
                    available.append(f"{market}_{variant}")
                    found_files.add(model_path.name)

        # 2. Dynamic scan for line variant models (cards_over_35_lightgbm.joblib, etc.)
        import re

        for model_path in sorted(self.models_dir.glob("*_over_*_*.joblib")):
            if model_path.name in found_files:
                continue
            name = model_path.stem  # e.g. "cards_over_35_lightgbm"
            # Validate it matches expected pattern: {base}_over_{line}_{variant}
            match = re.match(r"^(cards|corners|fouls|shots)_over_(\d+)_([a-z_]+)$", name)
            if match:
                available.append(name)
                found_files.add(model_path.name)

        # 2b. Dynamic scan for UNDER line variant models (fouls_under_265_lightgbm.joblib, etc.)
        for model_path in sorted(self.models_dir.glob("*_under_*_*.joblib")):
            if model_path.name in found_files:
                continue
            name = model_path.stem
            match = re.match(r"^(cards|corners|fouls|shots)_under_(\d+)_([a-z_]+)$", name)
            if match:
                available.append(name)
                found_files.add(model_path.name)

        # 3. Check for legacy niche models (specific lines like fouls_over_24_5)
        for model_name, config in self.NICHE_MODELS.items():
            model_path = self.models_dir / config["model_file"]
            config_path = self.outputs_dir / config["config_file"]
            if model_path.exists() and config_path.exists():
                if model_name not in available:
                    available.append(model_name)

        return available

    def load_model(self, model_name: str) -> Dict[str, Any]:
        """
        Load a model with its features and metadata.

        Returns dict with:
            - model: The trained model (CalibratedClassifierCV or similar)
            - features: List of feature names expected by model
            - bet_type: Type of bet (fouls, shots, away_win, etc.)
            - scaler: Optional scaler for LogisticReg models
            - metadata: Additional info (calibration method, params, etc.)
        """
        if model_name in self._loaded_models:
            return self._loaded_models[model_name]

        # Try niche model first
        if model_name in self.NICHE_MODELS:
            result = self._load_niche_model(model_name)
        else:
            # Try full optimization model (new format)
            result = self._load_full_model(model_name)

        if result:
            self._loaded_models[model_name] = result

        return result

    def _load_niche_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Load niche optimization model (fouls, shots, corners)."""
        config = self.NICHE_MODELS.get(model_name)
        if not config:
            return None

        model_path = self.models_dir / config["model_file"]
        config_path = self.outputs_dir / config["config_file"]

        if not model_path.exists():
            logger.warning(f"Model file not found: {model_path}")
            return None

        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}")
            return None

        try:
            model = joblib.load(model_path)

            with open(config_path) as f:
                pipeline_config = json.load(f)

            features = pipeline_config.get("features", {}).get("list", [])
            best_strategy = pipeline_config.get("best_strategy", {})

            return {
                "model": model,
                "features": features,
                "bet_type": model_name,
                "scaler": None,
                "metadata": {
                    "source": "niche_optimization",
                    "best_threshold": best_strategy.get("threshold"),
                    "expected_roi": best_strategy.get("roi"),
                    "n_bets": best_strategy.get("n_bets"),
                },
            }
        except Exception as e:
            logger.error(f"Failed to load niche model {model_name}: {e}")
            return None

    def _load_full_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Load full optimization model (new format with metadata)."""
        model_path = self.models_dir / f"{model_name}.joblib"

        if not model_path.exists():
            logger.warning(f"Model file not found: {model_path}")
            return None

        try:
            data = joblib.load(model_path)

            # New format: dict with model + metadata
            if isinstance(data, dict) and "model" in data:
                return {
                    "model": data["model"],
                    "features": data.get("features", []),
                    "bet_type": data.get("bet_type", model_name),
                    "scaler": data.get("scaler"),
                    "metadata": {
                        "source": "full_optimization",
                        "calibration": data.get("calibration"),
                        "best_params": data.get("best_params", {}),
                        "is_regression": data.get("is_regression", False),
                    },
                }
            else:
                # Old format: just the model (shouldn't happen for full optimization)
                logger.warning(f"Model {model_name} in old format, no features available")
                return {
                    "model": data,
                    "features": [],
                    "bet_type": model_name,
                    "scaler": None,
                    "metadata": {},
                }
        except Exception as e:
            logger.error(f"Failed to load full model {model_name}: {e}")
            return None

    def _check_calibration(self, model, model_name: str) -> Tuple[Any, CalibrationStatus]:
        """Detect degenerate calibration and fall back to base estimator.

        Checks for two types of calibration collapse:
        1. Isotonic: step function where raw > ~0.45 maps to 0.9+
        2. Sigmoid: extreme Platt parameters that compress output to near-constant
        When detected, extract the uncalibrated base estimator and average across folds.

        Returns:
            Tuple of (model_or_fallback, CalibrationStatus).
        """
        if not hasattr(model, "calibrated_classifiers_"):
            return model, CalibrationStatus.UNKNOWN

        import numpy as np

        n_folds = len(model.calibrated_classifiers_)
        degenerate_folds = 0

        for cc in model.calibrated_classifiers_:
            for cal in getattr(cc, "calibrators", []):
                # Check isotonic calibration
                if hasattr(cal, "X_thresholds_"):
                    x_max = cal.X_thresholds_[-1]
                    y_at_top = cal.y_thresholds_[-1]
                    if x_max < 0.55 and y_at_top >= 0.75:
                        degenerate_folds += 1
                # Check sigmoid calibration (Platt scaling)
                elif hasattr(cal, "a_") and hasattr(cal, "b_"):
                    # Sigmoid: P = 1 / (1 + exp(a*f + b))
                    # Degenerate when output range is tiny (near-constant)
                    test_points = np.array([0.2, 0.5, 0.8])
                    outputs = 1.0 / (1.0 + np.exp(cal.a_ * test_points + cal.b_))
                    output_range = outputs.max() - outputs.min()
                    if output_range < 0.15:
                        degenerate_folds += 1

        if degenerate_folds >= 2:
            cal_type = (
                "isotonic"
                if hasattr(
                    getattr(model.calibrated_classifiers_[0], "calibrators", [None])[0],
                    "X_thresholds_",
                )
                else "sigmoid"
            )
            logger.warning(
                f"[CALIBRATION COLLAPSE] {model_name}: {cal_type} calibration is degenerate "
                f"({degenerate_folds}/{n_folds} folds). "
                f"Bypassing calibration — using raw base estimator average."
            )
            return (
                _UncalibratedEnsemble(model.calibrated_classifiers_),
                CalibrationStatus.UNCALIBRATED,
            )

        return model, CalibrationStatus.CALIBRATED

    def _check_zero_fill_ratio(self, X_df, model_name: str) -> bool:
        """Check if too many features are zero-filled (indicates stale model)."""
        row = X_df.iloc[0]
        zero_ratio = (row == 0.0).sum() / len(row)
        if zero_ratio > 0.5:
            logger.warning(f"[DEGRADED] {model_name}: {zero_ratio:.0%} features are 0.0. Skipping.")
            return False
        elif zero_ratio > 0.3:
            logger.warning(f"[CAUTION] {model_name}: {zero_ratio:.0%} features are 0.0.")
        return True

    def predict_with_health(
        self,
        model_name: str,
        features_df: pd.DataFrame,
        health_report: MarketHealthReport,
        odds: Optional[float] = None,
    ) -> Optional[Tuple[float, float]]:
        """Run prediction with structured health reporting.

        Like predict(), but populates the given MarketHealthReport with
        feature match, calibration status, and two-stage fallback info.

        When a TwoStageModel has no odds, falls back to Stage 1
        probability-only prediction with reduced confidence instead
        of returning None.

        Args:
            model_name: Name of the model to use.
            features_df: DataFrame with features.
            health_report: MarketHealthReport to populate.
            odds: Decimal odds (required for TwoStageModel Stage 2).

        Returns:
            Tuple of (probability, confidence) or None if prediction fails.
        """
        import numpy as np

        model_data = self.load_model(model_name)
        if not model_data:
            health_report.record_skip(f"Model {model_name} failed to load")
            return None

        model = model_data["model"]
        expected_features = model_data["features"]
        scaler = model_data.get("scaler")

        # Calibration check
        model, cal_status = self._check_calibration(model, model_name)
        health_report.record_calibration(cal_status)

        try:
            # Feature alignment
            if expected_features:
                missing = set(expected_features) - set(features_df.columns)
                n_missing = len(missing)
                n_expected = len(expected_features)

                health_report.record_feature_match(
                    expected=n_expected,
                    missing=n_missing,
                    missing_names=sorted(missing)[:10] if missing else None,
                )

                # Check severity — HIGH means skip
                if health_report.feature_mismatch_severity == FeatureMismatchSeverity.HIGH:
                    return None

                if missing:
                    missing_pct = n_missing / n_expected
                    # Also enforce original 10% hard cap
                    if missing_pct > 0.10:
                        health_report.record_skip(
                            f"Feature mismatch too high: "
                            f"{n_missing}/{n_expected} "
                            f"({missing_pct:.1%} > 10%)"
                        )
                        return None

                    logger.warning(
                        f"[FEATURE MISMATCH] {model_name}: filling "
                        f"{n_missing}/{n_expected} features with 0.0 "
                        f"({missing_pct:.1%} missing)"
                    )

                # Build feature DataFrame in expected order
                data = {}
                for feat in expected_features:
                    if feat in features_df.columns:
                        data[feat] = features_df[feat].values
                    else:
                        data[feat] = 0.0
                X_df = pd.DataFrame(data, index=features_df.index)

                # Fill NaN with median/0
                for col in X_df.columns:
                    if X_df[col].isna().any():
                        median = X_df[col].median()
                        fill_val = median if pd.notna(median) else 0
                        X_df[col] = X_df[col].fillna(fill_val)

                if not self._check_zero_fill_ratio(X_df, model_name):
                    health_report.record_skip(f"{model_name}: too many zero-filled features")
                    return None

                X = X_df
            else:
                X = features_df

            # Scaler
            if scaler:
                X_scaled = scaler.transform(X.values if hasattr(X, "values") else X)
                if hasattr(X, "columns"):
                    X = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
                else:
                    X = X_scaled

            # Prediction
            if hasattr(model, "predict_proba"):
                model_cls = type(model).__name__
                if "TwoStage" in model_cls:
                    if odds is None:
                        # Fallback: Stage 1 only, reduced confidence
                        health_report.record_two_stage_fallback()
                        try:
                            X_vals = X.values if hasattr(X, "values") else X
                            X_scaled = model._stage1_scaler.transform(X_vals)
                            stage1_proba = model._get_stage1_proba(X_scaled)
                            prob = float(stage1_proba[0])
                        except Exception as e:
                            logger.warning(
                                f"[TWO STAGE] {model_name}: " f"Stage 1 fallback failed: {e}"
                            )
                            health_report.record_skip(f"Two-stage Stage 1 fallback failed: {e}")
                            return None
                    else:
                        odds_arr = np.array([odds])
                        result_dict = model.predict_proba(X, odds_arr)
                        prob = float(result_dict["combined_score"][0])
                else:
                    proba = model.predict_proba(X)
                    prob = float(proba[0][1]) if proba.shape[1] == 2 else float(proba[0].max())

                if prob >= 0.99 or prob <= 0.01:
                    health_report.record_skip(f"Degenerate probability: {prob:.4f}")
                    return None

                confidence = abs(prob - 0.5) * 2
                # Apply health penalties
                confidence *= health_report.confidence_penalty
                return (prob, confidence)
            else:
                pred = model.predict(X)
                return (float(pred[0]), 0.5)

        except Exception as e:
            logger.error(f"Prediction failed for {model_name}: {e}")
            health_report.record_skip(f"Prediction error: {e}")
            return None

    def predict(
        self, model_name: str, features_df, odds: float = None
    ) -> Optional[Tuple[float, float]]:
        """
        Run prediction using loaded model.

        Args:
            model_name: Name of the model to use
            features_df: DataFrame with features (will be filtered to model's expected features)
            odds: Decimal odds for the match (required for TwoStageModel)

        Returns:
            Tuple of (probability, confidence) or None if prediction fails
        """
        model_data = self.load_model(model_name)
        if not model_data:
            return None

        model = model_data["model"]
        expected_features = model_data["features"]
        scaler = model_data.get("scaler")

        # Detect and bypass degenerate calibration (isotonic or sigmoid)
        model, _cal_status = self._check_calibration(model, model_name)

        try:
            # Filter to expected features
            if expected_features:
                missing = set(expected_features) - set(features_df.columns)
                available = [f for f in expected_features if f in features_df.columns]

                if missing:
                    # Allow up to 10% missing features, fill with median/0
                    missing_pct = len(missing) / len(expected_features)
                    if missing_pct > 0.10:
                        logger.warning(
                            f"Too many missing features for {model_name}: "
                            f"{len(missing)}/{len(expected_features)} ({missing_pct:.1%})"
                        )
                        return None

                    logger.warning(
                        f"[FEATURE MISMATCH] {model_name}: filling {len(missing)}/{len(expected_features)} "
                        f"features with 0.0 ({missing_pct:.1%} missing). "
                        f"Missing: {sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}"
                    )

                # Create feature DataFrame with expected order
                data = {}
                for feat in expected_features:
                    if feat in features_df.columns:
                        data[feat] = features_df[feat].values
                    else:
                        data[feat] = 0.0
                X_df = pd.DataFrame(data, index=features_df.index)

                # Fill NaN values with column median or 0
                nan_filled = []
                for col in X_df.columns:
                    if X_df[col].isna().any():
                        median = X_df[col].median()
                        fill_val = median if pd.notna(median) else 0
                        X_df[col] = X_df[col].fillna(fill_val)
                        nan_filled.append(col)
                if nan_filled:
                    logger.info(
                        f"[NaN FILL] {model_name}: filled {len(nan_filled)} NaN features with median/0: "
                        f"{nan_filled[:5]}{'...' if len(nan_filled) > 5 else ''}"
                    )

                # Check for degraded predictions (too many zeros from missing features)
                if not self._check_zero_fill_ratio(X_df, model_name):
                    return None

                X = X_df
            else:
                X = features_df

            # Apply scaler if present (convert to array if scaler was fitted without feature names)
            if scaler:
                X_scaled = scaler.transform(X.values if hasattr(X, "values") else X)
                if hasattr(X, "columns"):
                    X = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
                else:
                    X = X_scaled

            # Get probability prediction
            if hasattr(model, "predict_proba"):
                # TwoStageModel returns a dict and requires odds
                import numpy as np

                model_cls = type(model).__name__
                if "TwoStage" in model_cls:
                    if odds is None:
                        logger.warning(
                            f"[TWO STAGE] {model_name}: requires odds but none provided. Skipping."
                        )
                        return None
                    odds_arr = np.array([odds])
                    result_dict = model.predict_proba(X, odds_arr)
                    prob = float(result_dict["combined_score"][0])
                else:
                    proba = model.predict_proba(X)
                    # For binary classification, return probability of positive class
                    prob = float(proba[0][1]) if proba.shape[1] == 2 else float(proba[0].max())

                # Reject degenerate predictions from any model
                if prob >= 0.99 or prob <= 0.01:
                    logger.warning(
                        f"Degenerate probability from {model_name}: {prob:.4f}. "
                        f"Skipping prediction."
                    )
                    return None

                confidence = abs(prob - 0.5) * 2  # 0 at 50%, 1 at 0% or 100%
                return (prob, confidence)
            else:
                # Regression model
                pred = model.predict(X)
                return (float(pred[0]), 0.5)  # Default confidence for regression

        except Exception as e:
            logger.error(f"Prediction failed for {model_name}: {e}")
            return None


# Singleton instance for easy access
_loader: Optional[ModelLoader] = None


def get_model_loader() -> ModelLoader:
    """Get singleton ModelLoader instance."""
    global _loader
    if _loader is None:
        _loader = ModelLoader()
    return _loader
