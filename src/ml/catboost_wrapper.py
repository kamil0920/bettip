"""Enhanced CatBoost wrapper enabling transfer learning and baseline injection.

This wrapper provides sklearn-compatible interface while supporting CatBoost-specific
features that don't work with CalibratedClassifierCV directly:
- `init_model`: Transfer learning from a pre-trained base model
- `baseline`: Learn residuals from market odds (log-odds injection)

Compatible with CalibratedClassifierCV through BaseEstimator/ClassifierMixin interface.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

logger = logging.getLogger(__name__)


class EnhancedCatBoost(ClassifierMixin, BaseEstimator):
    """CatBoost wrapper enabling transfer learning + baseline injection.

    Compatible with CalibratedClassifierCV through sklearn interface.

    Args:
        init_model_path: Path to pre-trained CatBoost model for transfer learning.
        use_baseline: Whether to inject market odds as log-odds baseline.
        **catboost_params: Parameters passed to CatBoostClassifier.
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        init_model_path: Optional[str] = None,
        use_baseline: bool = False,
        **catboost_params: Any,
    ):
        self.init_model_path = init_model_path
        self.use_baseline = use_baseline
        self.catboost_params = catboost_params
        self._model: Optional[CatBoostClassifier] = None
        self._baseline_odds: Optional[np.ndarray] = None

    def set_baseline_odds(self, odds: np.ndarray) -> None:
        """Store odds array. Must be called before fit() when use_baseline=True."""
        self._baseline_odds = odds

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None, **kwargs) -> "EnhancedCatBoost":
        """Fit CatBoost with optional transfer learning and baseline injection.

        Args:
            X: Feature matrix.
            y: Target vector.
            sample_weight: Optional sample weights.

        Returns:
            self
        """
        self._model = CatBoostClassifier(**self.catboost_params)

        fit_kwargs: Dict[str, Any] = {}

        # Transfer learning: initialize from pre-trained model
        if self.init_model_path and Path(self.init_model_path).exists():
            fit_kwargs["init_model"] = self.init_model_path
            logger.info(f"Using transfer learning from: {self.init_model_path}")

        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight

        # Baseline injection: convert decimal odds to log-odds
        if self.use_baseline and self._baseline_odds is not None:
            implied = np.clip(1.0 / self._baseline_odds, 0.01, 0.99)
            fit_kwargs["baseline"] = np.log(implied / (1 - implied))
            logger.debug(f"Injected baseline from {len(self._baseline_odds)} odds values")

        self._model.fit(X, y, **fit_kwargs)
        self.classes_ = self._model.classes_
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        return self._model.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        return self._model.predict(X)

    @property
    def feature_importances_(self) -> np.ndarray:
        """Return feature importances from the underlying CatBoost model."""
        return self._model.feature_importances_

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for sklearn compatibility."""
        return {
            "init_model_path": self.init_model_path,
            "use_baseline": self.use_baseline,
            **self.catboost_params,
        }

    def set_params(self, **params: Any) -> "EnhancedCatBoost":
        """Set parameters for sklearn compatibility."""
        if "init_model_path" in params:
            self.init_model_path = params.pop("init_model_path")
        if "use_baseline" in params:
            self.use_baseline = params.pop("use_baseline")
        self.catboost_params.update(params)
        return self

    def save_model(self, path: str) -> None:
        """Save the underlying CatBoost model."""
        if self._model is not None:
            self._model.save_model(path)

    def get_feature_importance(self, **kwargs) -> np.ndarray:
        """Proxy for CatBoost native feature importance (including SHAP)."""
        return self._model.get_feature_importance(**kwargs)
