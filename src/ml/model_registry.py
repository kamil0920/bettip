"""
Model Registry for ML Models

Provides:
- Registry pattern for model types (easy to add new models)
- Per-model feature selection
- Configurable ensemble (choose which models to use)
- Early stopping support
- Unified interface for training and prediction
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Type
import logging

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score, mean_absolute_error
from sklearn.calibration import CalibratedClassifierCV

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for a single model."""
    enabled: bool = True
    use_early_stopping: bool = True
    early_stopping_rounds: int = 50
    calibrate: bool = True  # Use probability calibration for classifiers
    # Feature selection
    select_own_features: bool = False  # If True, each model selects its own features
    n_top_features: int = 50
    # Tuning
    n_optuna_trials: int = 80
    # Custom params (override tuned params)
    fixed_params: Dict = field(default_factory=dict)


@dataclass
class EnsembleConfig:
    """Configuration for model ensemble."""
    models: List[str] = field(default_factory=lambda: ['xgboost', 'lightgbm', 'catboost'])
    weights: Optional[Dict[str, float]] = None  # None = equal weights
    use_stacking: bool = False  # Future: stacking ensemble
    per_model_features: bool = False  # Each model uses its own selected features


class BaseModelWrapper(ABC):
    """Abstract base class for model wrappers."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.best_params = {}
        self.selected_features: List[str] = []

    @property
    @abstractmethod
    def name(self) -> str:
        """Model name identifier."""
        pass

    @abstractmethod
    def get_param_space(self, trial) -> Dict:
        """Get Optuna parameter space for tuning."""
        pass

    @abstractmethod
    def create_model(self, params: Dict, is_regression: bool) -> Any:
        """Create model instance with given params."""
        pass

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        is_regression: bool = False
    ) -> 'BaseModelWrapper':
        """Fit the model with optional early stopping."""
        params = {**self.best_params, **self.config.fixed_params}
        self.model = self.create_model(params, is_regression)

        # Early stopping if supported and validation data provided
        if self.config.use_early_stopping and X_val is not None:
            fit_params = self._get_early_stopping_params(X_val, y_val)
            self.model.fit(X_train, y_train, **fit_params)
        else:
            self.model.fit(X_train, y_train)

        # Calibrate classifier
        if not is_regression and self.config.calibrate and X_val is not None:
            self.model = CalibratedClassifierCV(self.model, method='sigmoid', cv='prefit')
            self.model.fit(X_val, y_val)

        return self

    def _get_early_stopping_params(self, X_val, y_val) -> Dict:
        """Get early stopping parameters. Override in subclasses."""
        return {}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability predictions for classifiers."""
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)[:, 1]
        return self.predict(X)

    def select_features(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: List[str],
        is_regression: bool = False
    ) -> List[str]:
        """Select top features using permutation importance."""
        # Create a quick model for feature selection
        quick_params = self._get_quick_params()
        model = self.create_model(quick_params, is_regression)
        model.fit(X_train, y_train)

        perm = permutation_importance(
            model, X_val, y_val,
            n_repeats=10, random_state=42, n_jobs=-1
        )

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': perm.importances_mean
        }).sort_values('importance', ascending=False)

        self.selected_features = importance_df.head(self.config.n_top_features)['feature'].tolist()
        logger.info(f"{self.name} selected {len(self.selected_features)} features")
        logger.info(f"  Top 5: {self.selected_features[:5]}")

        return self.selected_features

    def _get_quick_params(self) -> Dict:
        """Get quick training params for feature selection."""
        return {'n_estimators': 100, 'max_depth': 5, 'random_state': 42}


class XGBoostWrapper(BaseModelWrapper):
    """XGBoost model wrapper."""

    @property
    def name(self) -> str:
        return "xgboost"

    def get_param_space(self, trial) -> Dict:
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 400),
            'max_depth': trial.suggest_int('max_depth', 2, 8),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 30),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        }

    def create_model(self, params: Dict, is_regression: bool) -> Any:
        from xgboost import XGBClassifier, XGBRegressor

        full_params = {**params, 'random_state': 42, 'verbosity': 0}
        if is_regression:
            return XGBRegressor(**full_params)
        return XGBClassifier(**full_params)

    def _get_early_stopping_params(self, X_val, y_val) -> Dict:
        return {
            'eval_set': [(X_val, y_val)],
            'verbose': False
        }

    def _get_quick_params(self) -> Dict:
        return {'n_estimators': 100, 'max_depth': 5, 'random_state': 42, 'verbosity': 0}


class LightGBMWrapper(BaseModelWrapper):
    """LightGBM model wrapper."""

    @property
    def name(self) -> str:
        return "lightgbm"

    def get_param_space(self, trial) -> Dict:
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 400),
            'max_depth': trial.suggest_int('max_depth', 2, 10),
            'num_leaves': trial.suggest_int('num_leaves', 10, 100),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 60),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        }

    def create_model(self, params: Dict, is_regression: bool) -> Any:
        from lightgbm import LGBMClassifier, LGBMRegressor

        full_params = {**params, 'random_state': 42, 'verbose': -1}
        if is_regression:
            return LGBMRegressor(**full_params)
        return LGBMClassifier(**full_params)

    def _get_early_stopping_params(self, X_val, y_val) -> Dict:
        return {
            'eval_set': [(X_val, y_val)],
            'callbacks': [self._early_stopping_callback()]
        }

    def _early_stopping_callback(self):
        from lightgbm import early_stopping
        return early_stopping(self.config.early_stopping_rounds, verbose=False)

    def _get_quick_params(self) -> Dict:
        return {'n_estimators': 100, 'max_depth': 5, 'random_state': 42, 'verbose': -1}


class CatBoostWrapper(BaseModelWrapper):
    """CatBoost model wrapper."""

    @property
    def name(self) -> str:
        return "catboost"

    def get_param_space(self, trial) -> Dict:
        return {
            'iterations': trial.suggest_int('iterations', 50, 400),
            'depth': trial.suggest_int('depth', 3, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.1, 30.0),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        }

    def create_model(self, params: Dict, is_regression: bool) -> Any:
        from catboost import CatBoostClassifier, CatBoostRegressor

        full_params = {
            **params,
            'random_state': 42,
            'verbose': 0,
            'early_stopping_rounds': self.config.early_stopping_rounds if self.config.use_early_stopping else None
        }
        if is_regression:
            return CatBoostRegressor(**full_params)
        return CatBoostClassifier(**full_params)

    def _get_early_stopping_params(self, X_val, y_val) -> Dict:
        return {'eval_set': (X_val, y_val)}

    def _get_quick_params(self) -> Dict:
        return {'iterations': 100, 'depth': 5, 'random_state': 42, 'verbose': 0}


# Model Registry
MODEL_REGISTRY: Dict[str, Type[BaseModelWrapper]] = {
    'xgboost': XGBoostWrapper,
    'lightgbm': LightGBMWrapper,
    'catboost': CatBoostWrapper,
}


def get_model(name: str, config: Optional[ModelConfig] = None) -> BaseModelWrapper:
    """Get a model wrapper by name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}")

    wrapper_class = MODEL_REGISTRY[name]
    return wrapper_class(config or ModelConfig())


def register_model(name: str, wrapper_class: Type[BaseModelWrapper]) -> None:
    """Register a new model type."""
    if not issubclass(wrapper_class, BaseModelWrapper):
        raise TypeError(f"{wrapper_class} must inherit from BaseModelWrapper")
    MODEL_REGISTRY[name] = wrapper_class


class ModelEnsemble:
    """
    Ensemble of models with configurable composition.

    Supports:
    - Configurable model selection (use 2 or 3 models)
    - Per-model feature selection
    - Weighted averaging
    - Early stopping
    """

    def __init__(self, config: Optional[EnsembleConfig] = None):
        self.config = config or EnsembleConfig()
        self.models: Dict[str, BaseModelWrapper] = {}
        self.feature_indices: Dict[str, List[int]] = {}
        self.shared_features: List[str] = []

    def tune_and_train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: List[str],
        is_regression: bool = False,
        n_trials: int = 80
    ) -> 'ModelEnsemble':
        """Tune hyperparameters and train all models in ensemble."""
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        for model_name in self.config.models:
            logger.info(f"\n--- Training {model_name} ---")

            model_config = ModelConfig(n_optuna_trials=n_trials)
            wrapper = get_model(model_name, model_config)

            # Per-model feature selection
            if self.config.per_model_features:
                selected = wrapper.select_features(
                    X_train, y_train, X_val, y_val, feature_names, is_regression
                )
                indices = [feature_names.index(f) for f in selected]
                self.feature_indices[model_name] = indices
                X_train_sel = X_train[:, indices]
                X_val_sel = X_val[:, indices]
            else:
                X_train_sel, X_val_sel = X_train, X_val

            # Tune hyperparameters
            wrapper.best_params = self._tune_model(
                wrapper, X_train_sel, y_train, X_val_sel, y_val, is_regression, n_trials
            )

            # Train with best params
            wrapper.fit(X_train_sel, y_train, X_val_sel, y_val, is_regression)

            self.models[model_name] = wrapper
            logger.info(f"{model_name} trained successfully")

        return self

    def _tune_model(
        self,
        wrapper: BaseModelWrapper,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        is_regression: bool,
        n_trials: int
    ) -> Dict:
        """Tune model hyperparameters with Optuna."""
        import optuna

        def objective(trial):
            params = wrapper.get_param_space(trial)
            model = wrapper.create_model(params, is_regression)
            model.fit(X_train, y_train)

            if is_regression:
                pred = model.predict(X_val)
                return mean_absolute_error(y_val, pred)
            else:
                pred = model.predict(X_val)
                return f1_score(y_val, pred, average='binary')

        direction = 'minimize' if is_regression else 'maximize'
        study = optuna.create_study(
            direction=direction,
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        logger.info(f"Best {wrapper.name} score: {study.best_value:.4f}")
        return study.best_params

    def predict(self, X: np.ndarray, is_regression: bool = False) -> np.ndarray:
        """Make ensemble predictions."""
        predictions = []
        weights = []

        for name, wrapper in self.models.items():
            # Use model-specific features if configured
            if name in self.feature_indices:
                X_model = X[:, self.feature_indices[name]]
            else:
                X_model = X

            if is_regression:
                pred = wrapper.predict(X_model)
            else:
                pred = wrapper.predict_proba(X_model)

            predictions.append(pred)

            # Get weight
            if self.config.weights and name in self.config.weights:
                weights.append(self.config.weights[name])
            else:
                weights.append(1.0)

        # Weighted average
        weights = np.array(weights) / sum(weights)
        ensemble_pred = np.average(predictions, axis=0, weights=weights)

        return ensemble_pred

    def get_individual_predictions(
        self,
        X: np.ndarray,
        is_regression: bool = False
    ) -> Dict[str, np.ndarray]:
        """Get predictions from each model individually."""
        predictions = {}

        for name, wrapper in self.models.items():
            if name in self.feature_indices:
                X_model = X[:, self.feature_indices[name]]
            else:
                X_model = X

            if is_regression:
                predictions[name] = wrapper.predict(X_model)
            else:
                predictions[name] = wrapper.predict_proba(X_model)

        predictions['ensemble'] = self.predict(X, is_regression)
        return predictions