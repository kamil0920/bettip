"""Unit tests for ML callibration module."""
import pytest
import numpy as np
from sklearn.pipeline import Pipeline

from src.ml.models import (
    ModelFactory,
    ModelType,
    DEFAULT_PARAMS,
    get_feature_importance,
)


class TestModelType:
    """Tests for ModelType enum."""

    def test_all_model_types_defined(self):
        """Test all expected model types are defined."""
        expected = [
            "random_forest",
            "xgboost",
            "lightgbm",
            "catboost",
            "logistic_regression",
        ]
        actual = [m.value for m in ModelType]
        assert sorted(actual) == sorted(expected)

    def test_model_type_from_string(self):
        """Test creating ModelType from string."""
        assert ModelType("random_forest") == ModelType.RANDOM_FOREST
        assert ModelType("xgboost") == ModelType.XGBOOST


class TestDefaultParams:
    """Tests for default parameters."""

    def test_all_model_types_have_defaults(self):
        """Test all model types have default parameters."""
        for model_type in ModelType:
            assert model_type in DEFAULT_PARAMS
            assert isinstance(DEFAULT_PARAMS[model_type], dict)

    def test_random_state_consistency(self):
        """Test all callibration have random_state=42 for reproducibility."""
        for model_type, params in DEFAULT_PARAMS.items():
            assert "random_state" in params, f"{model_type} missing random_state"
            assert params["random_state"] == 42


class TestModelFactory:
    """Tests for ModelFactory."""

    def test_create_random_forest(self):
        """Test creating Random Forest model."""
        model = ModelFactory.create(ModelType.RANDOM_FOREST)
        assert model is not None
        assert model.__class__.__name__ == "RandomForestClassifier"

    def test_create_xgboost(self):
        """Test creating XGBoost model."""
        model = ModelFactory.create(ModelType.XGBOOST)
        assert model is not None
        assert model.__class__.__name__ == "XGBClassifier"

    def test_create_lightgbm(self):
        """Test creating LightGBM model."""
        model = ModelFactory.create(ModelType.LIGHTGBM)
        assert model is not None
        assert model.__class__.__name__ == "LGBMClassifier"

    def test_create_catboost(self):
        """Test creating CatBoost model."""
        model = ModelFactory.create(ModelType.CATBOOST)
        assert model is not None
        assert model.__class__.__name__ == "CatBoostClassifier"

    def test_create_logistic_regression_wrapped_in_pipeline(self):
        """Test LogisticRegression is wrapped with StandardScaler."""
        model = ModelFactory.create(ModelType.LOGISTIC_REGRESSION)
        assert isinstance(model, Pipeline)
        assert "scaler" in model.named_steps
        assert "model" in model.named_steps

    def test_create_from_string(self):
        """Test creating model from string type."""
        model = ModelFactory.create("random_forest")
        assert model.__class__.__name__ == "RandomForestClassifier"

    def test_create_with_custom_params(self):
        """Test creating model with custom parameters."""
        model = ModelFactory.create(
            ModelType.RANDOM_FOREST,
            params={"n_estimators": 50, "max_depth": 5}
        )
        assert model.n_estimators == 50
        assert model.max_depth == 5

    def test_create_ignores_unknown_params(self):
        """Test unknown parameters are ignored (allowlist behavior)."""
        model = ModelFactory.create(
            ModelType.RANDOM_FOREST,
            params={"unknown_param": 999, "n_estimators": 50}
        )
        assert model.n_estimators == 50
        assert not hasattr(model, "unknown_param")

    def test_create_with_kwargs(self):
        """Test creating model with kwargs."""
        model = ModelFactory.create(
            ModelType.RANDOM_FOREST,
            n_estimators=75
        )
        assert model.n_estimators == 75

    def test_get_default_params(self):
        """Test getting default parameters."""
        params = ModelFactory.get_default_params(ModelType.RANDOM_FOREST)
        assert "n_estimators" in params
        assert params["n_estimators"] == 100

    def test_get_default_params_returns_copy(self):
        """Test default params returns a copy, not original."""
        params1 = ModelFactory.get_default_params(ModelType.RANDOM_FOREST)
        params2 = ModelFactory.get_default_params(ModelType.RANDOM_FOREST)
        params1["n_estimators"] = 999
        assert params2["n_estimators"] == 100

    def test_list_models(self):
        """Test listing available callibration."""
        models = ModelFactory.list_models()
        assert "random_forest" in models
        assert "xgboost" in models
        assert len(models) == 5


class TestGetFeatureImportance:
    """Tests for feature importance extraction."""

    def test_get_importance_from_tree_model(self):
        """Test extracting importance from tree-based model."""
        from sklearn.ensemble import RandomForestClassifier

        X = np.random.rand(100, 3)
        y = (X[:, 0] > 0.5).astype(int)

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        importance = get_feature_importance(model, ["f1", "f2", "f3"])

        assert isinstance(importance, dict)
        assert len(importance) == 3
        assert all(f in importance for f in ["f1", "f2", "f3"])
        # Importance should sum to ~1 (normalized)
        assert abs(sum(importance.values()) - 1.0) < 0.01

    def test_get_importance_from_pipeline(self):
        """Test extracting importance from model wrapped in pipeline."""
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression

        X = np.random.rand(100, 3)
        y = (X[:, 0] > 0.5).astype(int)

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression())
        ])
        model.fit(X, y)

        importance = get_feature_importance(model, ["f1", "f2", "f3"])

        assert isinstance(importance, dict)
        assert len(importance) == 3

    def test_get_importance_sorted_descending(self):
        """Test feature importance is sorted in descending order."""
        from sklearn.ensemble import RandomForestClassifier

        X = np.random.rand(100, 3)
        y = (X[:, 0] > 0.5).astype(int)

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        importance = get_feature_importance(model, ["f1", "f2", "f3"])
        values = list(importance.values())

        assert values == sorted(values, reverse=True)


class TestModelTraining:
    """Integration tests for model training."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample training data."""
        np.random.seed(42)
        X = np.random.rand(200, 5)
        y = (X[:, 0] + X[:, 1] > 1).astype(int)
        return X, y

    def test_random_forest_fit_predict(self, sample_data):
        """Test Random Forest can fit and predict."""
        X, y = sample_data
        model = ModelFactory.create(ModelType.RANDOM_FOREST)
        model.fit(X, y)

        predictions = model.predict(X)
        assert len(predictions) == len(y)
        assert set(predictions).issubset({0, 1})

    def test_xgboost_fit_predict(self, sample_data):
        """Test XGBoost can fit and predict."""
        X, y = sample_data
        model = ModelFactory.create(ModelType.XGBOOST)
        model.fit(X, y)

        proba = model.predict_proba(X)
        assert proba.shape == (len(y), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_lightgbm_fit_predict(self, sample_data):
        """Test LightGBM can fit and predict."""
        X, y = sample_data
        model = ModelFactory.create(ModelType.LIGHTGBM)
        model.fit(X, y)

        proba = model.predict_proba(X)
        assert proba.shape == (len(y), 2)

    def test_catboost_fit_predict(self, sample_data):
        """Test CatBoost can fit and predict."""
        X, y = sample_data
        model = ModelFactory.create(ModelType.CATBOOST)
        model.fit(X, y)

        proba = model.predict_proba(X)
        assert proba.shape == (len(y), 2)

    def test_logistic_regression_fit_predict(self, sample_data):
        """Test Logistic Regression (in pipeline) can fit and predict."""
        X, y = sample_data
        model = ModelFactory.create(ModelType.LOGISTIC_REGRESSION)
        model.fit(X, y)

        proba = model.predict_proba(X)
        assert proba.shape == (len(y), 2)
