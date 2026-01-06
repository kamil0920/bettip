"""
Hyperparameter optimization with Optuna.

Provides:
- Search space definitions for each model type
- Optuna study integration with MLflow tracking
- Cross-validation and time-series aware splitting
"""
import logging
from typing import Any, Callable, Dict, Optional

import mlflow
import numpy as np
import optuna
from optuna.integration.mlflow import MLflowCallback
import pandas as pd
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from src.ml.models import ModelFactory, ModelType

logger = logging.getLogger(__name__)


SCALERS = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
    "none": None,
}

SEARCH_SPACES: Dict[str, Callable[[optuna.Trial], Dict[str, Any]]] = {}


def search_space(model_type: str):
    """Decorator to register a search space for a model type."""
    def decorator(func: Callable[[optuna.Trial], Dict[str, Any]]):
        SEARCH_SPACES[model_type] = func
        return func
    return decorator


@search_space("logistic_regression")
def logistic_space(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        "C": trial.suggest_float("C", 0.01, 100, log=True),
        "solver": trial.suggest_categorical("solver", ["lbfgs", "liblinear"]),
        "max_iter": 1000,
        "penalty": "l2",
        "random_state": 42,
    }

@search_space("random_forest")
def random_forest_space(trial: optuna.Trial) -> Dict[str, Any]:
    """Search space for Random Forest.

    Tuned based on previous experiments:
    - Best: n_est=490, depth=13, min_leaf=9, min_split=20
    - Model prefers more trees and higher regularization
    """
    return {
        "n_estimators": trial.suggest_int("n_estimators", 300, 700),
        "max_depth": trial.suggest_int("max_depth", 8, 18),
        "min_samples_split": trial.suggest_int("min_samples_split", 10, 30),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 15),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5, 0.7]),
        "random_state": 42,
        "n_jobs": -1,
    }


@search_space("xgboost")
def xgboost_space(trial: optuna.Trial) -> Dict[str, Any]:
    """Search space for XGBoost.

    Tuned based on previous experiments:
    - Best: depth=3, lr=0.017, n_est=176, min_child=8, subsample=0.61
    - Model prefers shallow trees, low learning rate, high min_child_weight
    - Very low regularization worked best (reg_lambda~0)
    """
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 400),
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 0.8),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
        "min_child_weight": trial.suggest_int("min_child_weight", 5, 15),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 0.1, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 0.01, log=True),
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0,
    }

@search_space("lightgbm")
def lightgbm_space(trial: optuna.Trial) -> Dict[str, Any]:
    """Search space for LightGBM.

    Tuned based on previous experiments:
    - Best: depth=4, lr=0.035, n_est=89, num_leaves=74, min_child=80
    - Model prefers shallow depth but many leaves
    - High min_child_samples for regularization
    - Low regularization (reg_alpha/lambda near 0)
    """
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 200),  # was 50-500, best=89
        "max_depth": trial.suggest_int("max_depth", 3, 7),  # was 3-10, best=4
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),  # was 0.01-0.3, best=0.035
        "subsample": trial.suggest_float("subsample", 0.8, 1.0),  # was 0.6-1.0, best=0.97
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.8),  # was 0.6-1.0, best=0.60
        "min_child_samples": trial.suggest_int("min_child_samples", 50, 150),  # was implicit, best=80
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 0.01, log=True),  # was 0.01-10, best=0.0003
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 0.1, log=True),  # was 0.01-10, best=0.009
        "num_leaves": trial.suggest_int("num_leaves", 50, 120),  # was 10-100, best=74
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

@search_space("catboost")
def catboost_space(trial: optuna.Trial) -> Dict[str, Any]:
    """Search space for CatBoost.

    Tuned based on previous experiments:
    - Best: iter=375, depth=6, lr=0.010, bagging_temp=0.84, l2_reg~0
    - Model prefers more iterations with low learning rate
    - Very low l2 regularization
    - High bagging temperature
    """
    return {
        "iterations": trial.suggest_int("iterations", 250, 600),  # was 50-500, best=375
        "depth": trial.suggest_int("depth", 4, 8),  # was 3-10, best=6
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.03, log=True),  # was 0.01-0.3, best=0.010
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),  # was 0.6-1.0
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.6, 1.0),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 1.0, log=True),  # was 1-10, best~0
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.5, 1.0),  # was 0-1, best=0.84
        "random_strength": trial.suggest_float("random_strength", 0.001, 0.1, log=True),  # best=0.019
        "random_seed": 42,
        "thread_count": -1,
        "verbose": False,
    }

class HyperparameterTuner:
    """
    Optuna-based hyperparameter tuner with MLflow integration.

    Example:
        tuner = HyperparameterTuner(
            model_type="xgboost",
            experiment_name="tuning-xgboost",
            n_trials=50
        )
        best_params = tuner.tune(X_train, y_train)
    """

    def __init__(
        self,
        model_type: str,
        experiment_name: str,
        n_trials: int = 50,
        cv_folds: int = 5,
        scoring: str = "accuracy",
        direction: str = "maximize",
        time_series_cv: bool = True,
        tracking_uri: str = "sqlite:///mlflow.db",
        pruning: bool = True,
    ):
        """
        Initialize the tuner.

        Args:
            model_type: Type of model to tune
            experiment_name: MLflow experiment name
            n_trials: Number of optimization trials
            cv_folds: Number of cross-validation folds
            scoring: Scoring metric for CV
            direction: "maximize" or "minimize"
            time_series_cv: Use TimeSeriesSplit for CV
            tracking_uri: MLflow tracking URI
            pruning: Enable Optuna pruning for early stopping
        """
        self.model_type = model_type
        self.experiment_name = experiment_name
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.direction = direction
        self.time_series_cv = time_series_cv
        self.tracking_uri = tracking_uri
        self.pruning = pruning

        self.study: Optional[optuna.Study] = None
        self.best_params: Dict[str, Any] = {}
        self.best_score: float = 0.0

        if model_type not in SEARCH_SPACES:
            raise ValueError(
                f"Unknown model type: {model_type}. "
                f"Available: {list(SEARCH_SPACES.keys())}"
            )

    def tune(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run hyperparameter optimization.

        Args:
            X: Training features
            y: Training target
            timeout: Optional timeout in seconds

        Returns:
            Best hyperparameters found
        """
        logger.info(f"Starting hyperparameter tuning for {self.model_type}")
        logger.info(f"Trials: {self.n_trials}, CV folds: {self.cv_folds}")
        logger.info(f"Scoring: {self.scoring}, Direction: {self.direction}")

        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

        mlflow_callback = MLflowCallback(
            tracking_uri=self.tracking_uri,
            metric_name=f"cv_{self.scoring}",
            create_experiment=False,
            mlflow_kwargs={"nested": True},
        )

        sampler = optuna.samplers.TPESampler(seed=42)

        if self.pruning:
            pruner = optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10,
            )
        else:
            pruner = optuna.pruners.NopPruner()

        self.study = optuna.create_study(
            direction=self.direction,
            sampler=sampler,
            pruner=pruner,
            study_name=f"{self.experiment_name}_{self.model_type}",
        )

        def objective(trial: optuna.Trial) -> float:
            return self._objective(trial, X, y)

        with mlflow.start_run(run_name=f"tuning_{self.model_type}") as parent_run:
            mlflow.log_params({
                "model_type": self.model_type,
                "n_trials": self.n_trials,
                "cv_folds": self.cv_folds,
                "scoring": self.scoring,
                "time_series_cv": self.time_series_cv,
            })

            self.study.optimize(
                objective,
                n_trials=self.n_trials,
                timeout=timeout,
                callbacks=[mlflow_callback],
                show_progress_bar=True,
            )

            self.best_params = self.study.best_params
            self.best_score = self.study.best_value

            mlflow.log_params({f"best_{k}": v for k, v in self.best_params.items()})
            mlflow.log_metric(f"best_cv_{self.scoring}", self.best_score)

            logger.info(f"Best {self.scoring}: {self.best_score:.4f}")
            logger.info(f"Best params: {self.best_params}")

        return self.best_params

    def _objective(
        self,
        trial: optuna.Trial,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> float:
        """Objective function for a single trial."""
        search_space_fn = SEARCH_SPACES[self.model_type]
        params = search_space_fn(trial)

        try:
            model = self._create_model_with_params(params)
        except Exception as e:
            logger.warning(f"Failed to create model: {e}")
            raise optuna.TrialPruned()

        if self.time_series_cv:
            cv = TimeSeriesSplit(n_splits=self.cv_folds)
        else:
            cv = self.cv_folds

        try:
            scores = cross_val_score(
                model, X, y,
                cv=cv,
                scoring=self.scoring,
                n_jobs=1,
            )
            mean_score = np.mean(scores)
            std_score = np.std(scores)

            trial.report(mean_score, step=0)

            if trial.should_prune():
                raise optuna.TrialPruned()

            trial.set_user_attr("cv_std", std_score)
            trial.set_user_attr("cv_scores", scores.tolist())

            return mean_score

        except Exception as e:
            logger.warning(f"CV failed: {e}")
            raise optuna.TrialPruned()

    def _create_model_with_params(self, params: Dict[str, Any]):
        """Create model via factory - handles scaling for LR."""
        model_type_enum = ModelType(self.model_type)
        return ModelFactory.create(model_type_enum, params=params)

    def get_best_model(self, X: pd.DataFrame, y: pd.Series):
        """
        Get the best model trained on full data.

        Args:
            X: Full training features
            y: Full training target

        Returns:
            Trained model with best hyperparameters
        """
        if not self.best_params:
            raise ValueError("No tuning results. Run tune() first.")

        search_space_fn = SEARCH_SPACES[self.model_type]

        class DummyTrial:
            def __init__(self, params):
                self._params = params
            def suggest_int(self, name, *args, **kwargs):
                return self._params.get(name)
            def suggest_float(self, name, *args, **kwargs):
                return self._params.get(name)
            def suggest_categorical(self, name, *args, **kwargs):
                return self._params.get(name)

        dummy = DummyTrial(self.best_params)
        full_params = search_space_fn(dummy)

        full_params.update(self.best_params)

        model = self._create_model_with_params(full_params)
        model.fit(X, y)

        return model

    def get_optimization_history(self) -> pd.DataFrame:
        """Get optimization history as DataFrame."""
        if self.study is None:
            raise ValueError("No study available. Run tune() first.")

        trials_data = []
        for trial in self.study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                data = {
                    "trial_number": trial.number,
                    "value": trial.value,
                    **trial.params,
                }
                trials_data.append(data)

        return pd.DataFrame(trials_data)

    def get_param_importances(self) -> Dict[str, float]:
        """Get hyperparameter importances using fANOVA."""
        if self.study is None:
            raise ValueError("No study available. Run tune() first.")

        try:
            importances = optuna.importance.get_param_importances(self.study)
            return dict(importances)
        except Exception as e:
            logger.warning(f"Could not calculate param importances: {e}")
            return {}


def tune_all_models(
    X: pd.DataFrame,
    y: pd.Series,
    experiment_name: str,
    n_trials: int = 30,
    models: Optional[list] = None,
    **kwargs
) -> Dict[str, Dict[str, Any]]:
    """
    Tune all models and return best params for each.

    Args:
        X: Training features
        y: Training target
        experiment_name: Base experiment name
        n_trials: Trials per model
        models: List of models to tune (default: all)
        **kwargs: Additional args for HyperparameterTuner

    Returns:
        Dict mapping model_type to best_params
    """
    if models is None:
        models = list(SEARCH_SPACES.keys())

    results = {}

    for model_type in models:
        logger.info(f"\n{'='*60}")
        logger.info(f"Tuning {model_type.upper()}")
        logger.info(f"{'='*60}")

        tuner = HyperparameterTuner(
            model_type=model_type,
            experiment_name=f"{experiment_name}-{model_type}",
            n_trials=n_trials,
            **kwargs
        )

        try:
            best_params = tuner.tune(X, y)
            results[model_type] = {
                "best_params": best_params,
                "best_score": tuner.best_score,
                "param_importances": tuner.get_param_importances(),
            }
        except Exception as e:
            logger.error(f"Failed to tune {model_type}: {e}")
            results[model_type] = {"error": str(e)}

    return results
