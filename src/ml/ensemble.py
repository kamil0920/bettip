"""
Ensemble methods for combining multiple models.

Supports:
- VotingClassifier (soft voting with probability averaging)
- StackingClassifier (meta-learner approach)
"""
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, TimeSeriesSplit

from src.ml.models import ModelFactory

logger = logging.getLogger(__name__)


DEFAULT_BASE_MODELS = ["random_forest", "xgboost", "lightgbm", "catboost"]


class EnsembleFactory:
    """Factory for creating ensemble models."""

    @staticmethod
    def create_voting_ensemble(
        base_models: Optional[List[str]] = None,
        model_params: Optional[Dict[str, Dict[str, Any]]] = None,
        voting: str = "soft",
        weights: Optional[List[float]] = None,
    ) -> VotingClassifier:
        """
        Create a VotingClassifier ensemble.

        Args:
            base_models: List of model types to include (default: all 4 boosting models)
            model_params: Dict mapping model_type to custom params
            voting: "soft" (probability averaging) or "hard" (majority vote)
            weights: Optional weights for each model

        Returns:
            VotingClassifier instance
        """
        if base_models is None:
            base_models = DEFAULT_BASE_MODELS

        model_params = model_params or {}

        estimators = []
        for model_type in base_models:
            params = model_params.get(model_type, {})
            model = ModelFactory.create(model_type, params=params)
            estimators.append((model_type, model))

        logger.info(f"Creating VotingClassifier with {len(estimators)} models: {base_models}")
        logger.info(f"Voting method: {voting}, Weights: {weights}")

        return VotingClassifier(
            estimators=estimators,
            voting=voting,
            weights=weights,
            n_jobs=-1,
        )

    @staticmethod
    def create_stacking_ensemble(
        base_models: Optional[List[str]] = None,
        model_params: Optional[Dict[str, Dict[str, Any]]] = None,
        meta_learner: str = "logistic_regression",
        meta_params: Optional[Dict[str, Any]] = None,
        cv: int = 5,
        passthrough: bool = False,
    ) -> StackingClassifier:
        """
        Create a StackingClassifier ensemble.

        Args:
            base_models: List of model types for base layer
            model_params: Dict mapping model_type to custom params
            meta_learner: Model type for final estimator
            meta_params: Params for meta learner
            cv: Cross-validation folds for generating meta-features
            passthrough: Whether to pass original features to meta-learner

        Returns:
            StackingClassifier instance
        """
        if base_models is None:
            base_models = DEFAULT_BASE_MODELS

        model_params = model_params or {}
        meta_params = meta_params or {}

        estimators = []
        for model_type in base_models:
            params = model_params.get(model_type, {})
            model = ModelFactory.create(model_type, params=params)
            estimators.append((model_type, model))

        if meta_learner == "logistic_regression":
            final_estimator = LogisticRegression(
                max_iter=1000,
                random_state=42,
                **meta_params
            )
        else:
            final_estimator = ModelFactory.create(meta_learner, params=meta_params)

        logger.info(f"Creating StackingClassifier with {len(estimators)} base models")
        logger.info(f"Base models: {base_models}")
        logger.info(f"Meta-learner: {meta_learner}")
        logger.info(f"CV folds: {cv}, Passthrough: {passthrough}")

        return StackingClassifier(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=cv,
            stack_method="predict_proba",
            passthrough=passthrough,
            n_jobs=-1,
        )

    @staticmethod
    def create_weighted_voting(
        base_models: List[str],
        cv_scores: Dict[str, float],
        model_params: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> VotingClassifier:
        """
        Create a VotingClassifier with weights based on CV scores.

        Args:
            base_models: List of model types
            cv_scores: Dict mapping model_type to CV score (for weight calculation)
            model_params: Dict mapping model_type to custom params

        Returns:
            VotingClassifier with performance-based weights
        """
        model_params = model_params or {}

        scores = [cv_scores.get(m, 0.5) for m in base_models]
        min_score = min(scores)
        max_score = max(scores)

        if max_score > min_score:
            weights = [0.5 + (s - min_score) / (max_score - min_score) for s in scores]
        else:
            weights = [1.0] * len(base_models)

        logger.info(f"Calculated weights based on CV scores: {dict(zip(base_models, weights))}")

        return EnsembleFactory.create_voting_ensemble(
            base_models=base_models,
            model_params=model_params,
            voting="soft",
            weights=weights,
        )


def evaluate_ensemble(
    ensemble,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv_folds: int = 5,
    time_series_cv: bool = True,
    scoring: str = "f1_weighted",
) -> Dict[str, Any]:
    """
    Evaluate an ensemble model with cross-validation and holdout test.

    Args:
        ensemble: Ensemble model instance
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        cv_folds: Number of CV folds
        time_series_cv: Use TimeSeriesSplit
        scoring: Scoring metric for CV (default: f1_weighted)

    Returns:
        Dict with CV and test metrics
    """
    from sklearn.metrics import accuracy_score, f1_score

    logger.info("Evaluating ensemble model...")

    if time_series_cv:
        cv = TimeSeriesSplit(n_splits=cv_folds)
    else:
        cv = cv_folds

    cv_scores = cross_val_score(ensemble, X_train, y_train, cv=cv, scoring=scoring)

    logger.info(f"CV {scoring}: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

    ensemble.fit(X_train, y_train)

    test_predictions = ensemble.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_predictions)
    test_f1 = f1_score(y_test, test_predictions, average="weighted")

    if hasattr(ensemble, "predict_proba"):
        test_proba = ensemble.predict_proba(X_test)
    else:
        test_proba = None

    logger.info(f"Test Accuracy: {test_accuracy:.4f}, F1: {test_f1:.4f}")

    return {
        "cv_score_mean": cv_scores.mean(),
        "cv_score_std": cv_scores.std(),
        "cv_scores": cv_scores.tolist(),
        "test_accuracy": test_accuracy,
        "test_f1": test_f1,
        "test_predictions": test_predictions,
        "test_proba": test_proba,
    }


def get_ensemble_feature_importance(
    ensemble,
    feature_names: List[str],
) -> Dict[str, float]:
    """
    Extract aggregated feature importance from ensemble.

    For VotingClassifier: average importances across base models
    For StackingClassifier: average importances from base models

    Args:
        ensemble: Trained ensemble model
        feature_names: List of feature names

    Returns:
        Dict mapping feature names to importance scores
    """
    importances = np.zeros(len(feature_names))
    n_models = 0

    if hasattr(ensemble, "estimators_"):
        estimators = ensemble.estimators_
    elif hasattr(ensemble, "named_estimators_"):
        estimators = list(ensemble.named_estimators_.values())
    else:
        logger.warning("Cannot extract estimators from ensemble")
        return {}

    for estimator in estimators:
        if hasattr(estimator, "named_steps") and "model" in estimator.named_steps:
            estimator = estimator.named_steps["model"]

        if hasattr(estimator, "feature_importances_"):
            importances += estimator.feature_importances_
            n_models += 1
        elif hasattr(estimator, "coef_"):
            coef = estimator.coef_
            if len(coef.shape) > 1:
                coef = np.abs(coef).mean(axis=0)
            else:
                coef = np.abs(coef)
            importances += coef
            n_models += 1

    if n_models > 0:
        importances /= n_models
        total = importances.sum()
        if total > 0:
            importances /= total

    return dict(sorted(
        zip(feature_names, importances),
        key=lambda x: x[1],
        reverse=True
    ))