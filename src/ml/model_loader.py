"""
Model Loader Utility

Loads trained models for inference in pre-match intelligence system.
Supports both:
- Full optimization models (dict with model + metadata)
- Niche optimization models (just CalibratedClassifierCV + JSON features)
"""

import json
import joblib
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import logging

import pandas as pd

logger = logging.getLogger(__name__)


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
        "away_win", "home_win", "btts", "over25", "under25", "asian_handicap",
        "shots", "fouls", "corners", "cards",
    ]

    def __init__(self, models_dir: Optional[Path] = None, outputs_dir: Optional[Path] = None):
        self.models_dir = models_dir or self.MODELS_DIR
        self.outputs_dir = outputs_dir or self.OUTPUTS_DIR
        self._loaded_models: Dict[str, Dict[str, Any]] = {}

    def list_available_models(self) -> List[str]:
        """List all available trained models."""
        available = []

        # Check for full optimization models (new format with metadata)
        # These are saved as {bet_type}_{model_name}.joblib (e.g., away_win_xgboost.joblib)
        for market in self.FULL_OPTIMIZATION_MARKETS:
            # Look for any model variant (xgboost, lightgbm, catboost, logisticreg)
            for variant in ["xgboost", "lightgbm", "catboost", "logisticreg", "fastai"]:
                model_path = self.models_dir / f"{market}_{variant}.joblib"
                if model_path.exists():
                    available.append(f"{market}_{variant}")

        # Check for niche models (specific lines like fouls_over_24_5)
        for model_name, config in self.NICHE_MODELS.items():
            model_path = self.models_dir / config["model_file"]
            config_path = self.outputs_dir / config["config_file"]
            if model_path.exists() and config_path.exists():
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
                return {"model": data, "features": [], "bet_type": model_name, "scaler": None, "metadata": {}}
        except Exception as e:
            logger.error(f"Failed to load full model {model_name}: {e}")
            return None

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

    def predict(self, model_name: str, features_df) -> Optional[Tuple[float, float]]:
        """
        Run prediction using loaded model.

        Args:
            model_name: Name of the model to use
            features_df: DataFrame with features (will be filtered to model's expected features)

        Returns:
            Tuple of (probability, confidence) or None if prediction fails
        """
        model_data = self.load_model(model_name)
        if not model_data:
            return None

        model = model_data["model"]
        expected_features = model_data["features"]
        scaler = model_data.get("scaler")

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
                X_scaled = scaler.transform(X.values if hasattr(X, 'values') else X)
                if hasattr(X, 'columns'):
                    X = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
                else:
                    X = X_scaled

            # Get probability prediction
            if hasattr(model, "predict_proba"):
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
