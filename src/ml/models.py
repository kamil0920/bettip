"""
Model factory for creating and configuring ML models.

Supports:
- Random Forest
- XGBoost
- LightGBM
- CatBoost
- Logistic Regression (baseline)
"""
import logging
from enum import Enum
from typing import Any, Dict, Optional

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Supported model types."""
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"
    LOGISTIC_REGRESSION = "logistic_regression"


# Default hyperparameters for each model type
DEFAULT_PARAMS: Dict[ModelType, Dict[str, Any]] = {
    ModelType.RANDOM_FOREST: {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": 42,
        "n_jobs": -1,
    },
    ModelType.XGBOOST: {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0,
    },
    ModelType.LIGHTGBM: {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    },
    ModelType.CATBOOST: {
        "iterations": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "random_state": 42,
        "verbose": False,
        "thread_count": -1,
    },
    ModelType.LOGISTIC_REGRESSION: {
        "max_iter": 1000,
        "random_state": 42,
        "n_jobs": -1,
    },
}


class ModelFactory:
    """Factory for creating ML models with consistent interface."""

    @staticmethod
    def create(
            model_type: ModelType | str,
            params: Optional[Dict[str, Any]] = None,
            **kwargs
    ) -> Any:
        """
        Create a model instance using DEFAULT_PARAMS as a strict schema (allowlist).
        """
        if isinstance(model_type, str):
            model_type = ModelType(model_type)

        model_class = ModelFactory._get_model_class(model_type)

        default_config = DEFAULT_PARAMS.get(model_type, {})
        allowed_keys = set(default_config.keys())

        input_params = {}
        if params:
            input_params.update(params)
        input_params.update(kwargs)

        final_params = default_config.copy()

        for key, value in input_params.items():
            if key in allowed_keys:
                final_params[key] = value
            else:
                logger.debug(f"Ignored param '{key}' for {model_type.value} (not in defaults)")
                pass

        logger.info(f"Creating {model_type.value} with params: {final_params}")

        return model_class(**final_params)

    @staticmethod
    def _get_model_class(model_type: ModelType) -> type:
        """Get the model class for a given type."""
        mapping = {
            ModelType.RANDOM_FOREST: RandomForestClassifier,
            ModelType.XGBOOST: XGBClassifier,
            ModelType.LIGHTGBM: LGBMClassifier,
            ModelType.CATBOOST: CatBoostClassifier,
            ModelType.LOGISTIC_REGRESSION: LogisticRegression,
        }
        return mapping[model_type]

    @staticmethod
    def get_default_params(model_type: ModelType | str) -> Dict[str, Any]:
        """Get default parameters for a model type."""
        if isinstance(model_type, str):
            model_type = ModelType(model_type)
        return DEFAULT_PARAMS.get(model_type, {}).copy()

    @staticmethod
    def list_models() -> list[str]:
        """List all available model types."""
        return [m.value for m in ModelType]


def get_feature_importance(model: Any, feature_names: list[str]) -> Dict[str, float]:
    """
    Extract feature importance from a trained model.

    Args:
        model: Trained model
        feature_names: List of feature names

    Returns:
        Dictionary mapping feature names to importance scores
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        # For logistic regression, use absolute coefficients
        importances = abs(model.coef_[0]) if len(model.coef_.shape) > 1 else abs(model.coef_)
    else:
        logger.warning(f"Model {type(model).__name__} doesn't support feature importance")
        return {}

    # Normalize to sum to 1
    total = sum(importances)
    if total > 0:
        importances = importances / total

    return dict(sorted(
        zip(feature_names, importances),
        key=lambda x: x[1],
        reverse=True
    ))
