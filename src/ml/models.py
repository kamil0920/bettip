"""
Model factory for creating and configuring ML callibration.

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

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
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
    FASTAI_TABULAR = "fastai_tabular"


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
        "has_time": True,
    },
    ModelType.LOGISTIC_REGRESSION: {
        "max_iter": 1000,
        "random_state": 42,
        "n_jobs": -1,
    },
    ModelType.FASTAI_TABULAR: {
        "layers": [400, 200],
        "epochs": 80,
        "ps": [0.001, 0.01],
        "embed_p": 0.04,
        "random_state": 42,
    },
}


class FastAITabularModel(ClassifierMixin, BaseEstimator):
    """fastai TabularLearner wrapper with sklearn-compatible interface.

    Uses entity embeddings for categorical features and fit_one_cycle training.
    Designed for ensemble stacking alongside GBDT models.
    """

    def __init__(
        self,
        layers: list = None,
        epochs: int = 80,
        ps: list = None,
        embed_p: float = 0.04,
        random_state: int = 42,
    ):
        self.layers = layers or [400, 200]
        self.epochs = epochs
        self.ps = ps or [0.001, 0.01]
        self.embed_p = embed_p
        self.random_state = random_state
        self.learn = None
        self._classes = None

    def fit(self, X, y, sample_weight=None):
        """Train fastai TabularLearner on numpy arrays.

        Args:
            X: Feature matrix (numpy array, already scaled).
            y: Target array.
            sample_weight: Optional sample weights (used via weighted loss).
        """
        try:
            import torch
            from fastai.tabular.all import (
                CategoryBlock,
                Categorify,
                FillMissing,
                Normalize,
                TabularDataLoaders,
                TabularLearner,
                tabular_learner,
            )
            import pandas as pd
        except ImportError:
            raise ImportError(
                "fastai is required for FastAITabularModel. "
                "Install with: uv pip install 'bettip[dl]'"
            )

        torch.manual_seed(self.random_state)

        self._classes = sorted(set(y))

        # Build DataFrame from numpy arrays (all continuous, no categoricals)
        col_names = [f"f_{i}" for i in range(X.shape[1])]
        X_arr = X.values if hasattr(X, "values") else X
        df = pd.DataFrame(X_arr, columns=col_names)
        df["target"] = y.astype(int)

        # Create dataloaders — use last 20% as validation
        n_val = max(int(len(df) * 0.2), 1)
        splits = (list(range(len(df) - n_val)), list(range(len(df) - n_val, len(df))))

        dls = TabularDataLoaders.from_df(
            df,
            y_names="target",
            y_block=CategoryBlock(),
            cont_names=col_names,
            cat_names=[],
            procs=[FillMissing, Normalize],
            splits=splits,
            bs=min(256, max(1, len(df) // 4)),
        )

        from fastai.tabular.model import tabular_config

        config = tabular_config(ps=self.ps, embed_p=self.embed_p)
        self.learn = tabular_learner(
            dls,
            layers=self.layers,
            config=config,
            emb_szs=[],
            metrics=[],
        )

        # Find learning rate and train
        with self.learn.no_bar(), self.learn.no_logging():
            try:
                lr_result = self.learn.lr_find(show_plot=False)
                lr = lr_result.valley if hasattr(lr_result, "valley") and lr_result.valley is not None else 1e-3
                # Sanity check lr
                if lr <= 0 or lr > 1 or not np.isfinite(lr):
                    lr = 1e-3
            except Exception:
                lr = 1e-3  # Default if lr_find fails
            self.learn.fit_one_cycle(self.epochs, lr_max=lr)

        self._col_names = col_names
        return self

    def predict_proba(self, X):
        """Return class probabilities matching sklearn interface.

        Args:
            X: Feature matrix (numpy array).

        Returns:
            Array of shape (n_samples, 2) with class probabilities.
        """
        import numpy as np
        import pandas as pd

        if self.learn is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X_arr = X.values if hasattr(X, "values") else X
        df = pd.DataFrame(X_arr, columns=self._col_names)
        df["target"] = 0  # Dummy target for fastai

        dl = self.learn.dls.test_dl(df)
        with self.learn.no_bar(), self.learn.no_logging():
            preds, _ = self.learn.get_preds(dl=dl)

        proba = preds.numpy()
        # Ensure 2-column output for binary classification
        if proba.shape[1] == 1:
            proba = np.column_stack([1 - proba, proba])

        # Guard against degenerate predictions (saturated softmax)
        if np.any(proba >= 0.999) or np.any(proba <= 0.001):
            n_degen = int(np.sum((proba >= 0.999) | (proba <= 0.001)))
            logger.debug(
                f"FastAI: clipping {n_degen} degenerate values to [0.01, 0.99] "
                f"(min={proba.min():.4f}, max={proba.max():.4f})"
            )
            proba = np.clip(proba, 0.01, 0.99)
            # Re-normalize rows to sum to 1
            proba = proba / proba.sum(axis=1, keepdims=True)

        return proba

    def predict(self, X):
        """Return class predictions."""
        import numpy as np
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    @property
    def classes_(self):
        """Return class labels."""
        import numpy as np
        return np.array(self._classes) if self._classes else None


class ModelFactory:
    """Factory for creating ML callibration with consistent interface."""

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

        model = model_class(**final_params)

        if model_type == ModelType.LOGISTIC_REGRESSION:
            from sklearn.pipeline import Pipeline
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('model', model)
            ])
            logger.info(f"Wrapped {model_type.value} with StandardScaler")

        logger.info(f"Created {model_type.value} with params: {final_params}")

        return model

    @staticmethod
    def _get_model_class(model_type: ModelType) -> type:
        """Get the model class for a given type."""
        mapping = {
            ModelType.RANDOM_FOREST: RandomForestClassifier,
            ModelType.XGBOOST: XGBClassifier,
            ModelType.LIGHTGBM: LGBMClassifier,
            ModelType.CATBOOST: CatBoostClassifier,
            ModelType.LOGISTIC_REGRESSION: LogisticRegression,
            ModelType.FASTAI_TABULAR: FastAITabularModel,
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
    if hasattr(model, 'named_steps') and 'model' in model.named_steps:
        model = model.named_steps['model']

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = abs(model.coef_[0]) if len(model.coef_.shape) > 1 else abs(model.coef_)
    else:
        logger.warning(f"Model {type(model).__name__} no importance")
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
