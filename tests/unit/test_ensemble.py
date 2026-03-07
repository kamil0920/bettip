"""Unit tests for ML ensemble module."""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import VotingClassifier, StackingClassifier

from src.ml.ensemble import EnsembleFactory, get_ensemble_feature_importance


@pytest.fixture
def binary_data():
    """Create a small binary classification dataset."""
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=5,
        random_state=42,
    )
    return X, y


class TestEnsembleFactory:
    def test_create_voting_ensemble_default(self):
        """Test voting ensemble creates with default models."""
        ensemble = EnsembleFactory.create_voting_ensemble()
        assert isinstance(ensemble, VotingClassifier)
        assert len(ensemble.estimators) > 0
        assert ensemble.voting == "soft"

    def test_create_voting_ensemble_custom_models(self):
        """Test voting ensemble with specific model list."""
        ensemble = EnsembleFactory.create_voting_ensemble(
            base_models=["xgboost", "lightgbm"]
        )
        assert isinstance(ensemble, VotingClassifier)
        assert len(ensemble.estimators) == 2
        model_names = [name for name, _ in ensemble.estimators]
        assert "xgboost" in model_names
        assert "lightgbm" in model_names

    def test_create_voting_ensemble_with_weights(self):
        """Test voting ensemble with custom weights."""
        ensemble = EnsembleFactory.create_voting_ensemble(
            base_models=["xgboost", "lightgbm"],
            weights=[0.7, 0.3],
        )
        assert ensemble.weights == [0.7, 0.3]

    def test_create_stacking_ensemble_default(self):
        """Test stacking ensemble creates with defaults."""
        ensemble = EnsembleFactory.create_stacking_ensemble()
        assert isinstance(ensemble, StackingClassifier)
        assert len(ensemble.estimators) > 0

    def test_create_stacking_ensemble_custom_meta(self):
        """Test stacking with non-default meta learner."""
        ensemble = EnsembleFactory.create_stacking_ensemble(
            base_models=["xgboost", "lightgbm"],
            meta_learner="logistic_regression",
        )
        assert isinstance(ensemble, StackingClassifier)
        assert len(ensemble.estimators) == 2

    def test_create_weighted_voting(self):
        """Test performance-weighted voting ensemble."""
        models = ["xgboost", "lightgbm"]
        scores = {"xgboost": 0.8, "lightgbm": 0.6}
        ensemble = EnsembleFactory.create_weighted_voting(models, scores)
        assert isinstance(ensemble, VotingClassifier)
        assert ensemble.weights is not None
        # xgboost scored higher, should have higher weight
        assert ensemble.weights[0] > ensemble.weights[1]

    def test_weighted_voting_equal_scores(self):
        """Test weighted voting with equal scores produces equal weights."""
        models = ["xgboost", "lightgbm"]
        scores = {"xgboost": 0.7, "lightgbm": 0.7}
        ensemble = EnsembleFactory.create_weighted_voting(models, scores)
        assert ensemble.weights[0] == ensemble.weights[1]


class TestFeatureImportance:
    def test_importance_from_fitted_voting(self, binary_data):
        """Test feature importance extraction from fitted voting ensemble."""
        X, y = binary_data
        feature_names = [f"f{i}" for i in range(X.shape[1])]

        ensemble = EnsembleFactory.create_voting_ensemble(
            base_models=["xgboost", "lightgbm"]
        )
        ensemble.fit(X, y)

        importances = get_ensemble_feature_importance(ensemble, feature_names)
        assert isinstance(importances, dict)
        assert len(importances) == len(feature_names)
        # Importances should sum to ~1 (normalized)
        total = sum(importances.values())
        assert total > 0

    def test_importance_empty_for_unfitted(self):
        """Test that importance extraction handles missing estimators."""
        importances = get_ensemble_feature_importance(object(), ["f1", "f2"])
        assert importances == {}
