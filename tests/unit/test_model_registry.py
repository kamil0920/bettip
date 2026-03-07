"""Unit tests for model registry and wrappers."""

import numpy as np
import pytest
from sklearn.datasets import make_classification

from src.ml.model_registry import (
    MODEL_REGISTRY,
    BaseModelWrapper,
    CatBoostWrapper,
    EnsembleConfig,
    LightGBMWrapper,
    ModelConfig,
    ModelEnsemble,
    XGBoostWrapper,
    get_model,
    register_model,
)


@pytest.fixture
def binary_data():
    X, y = make_classification(
        n_samples=200, n_features=10, n_informative=5, random_state=42
    )
    return X, y


class TestModelConfig:
    def test_defaults(self):
        cfg = ModelConfig()
        assert cfg.enabled is True
        assert cfg.use_early_stopping is True
        assert cfg.calibrate is True
        assert cfg.n_top_features == 50

    def test_override(self):
        cfg = ModelConfig(enabled=False, n_top_features=20)
        assert cfg.enabled is False
        assert cfg.n_top_features == 20


class TestEnsembleConfig:
    def test_defaults(self):
        cfg = EnsembleConfig()
        assert 'xgboost' in cfg.models
        assert cfg.weights is None
        assert cfg.use_stacking is False


class TestGetModel:
    def test_xgboost(self):
        wrapper = get_model('xgboost')
        assert isinstance(wrapper, XGBoostWrapper)
        assert wrapper.name == 'xgboost'

    def test_lightgbm(self):
        wrapper = get_model('lightgbm')
        assert isinstance(wrapper, LightGBMWrapper)
        assert wrapper.name == 'lightgbm'

    def test_catboost(self):
        wrapper = get_model('catboost')
        assert isinstance(wrapper, CatBoostWrapper)
        assert wrapper.name == 'catboost'

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            get_model('nonexistent')

    def test_custom_config(self):
        cfg = ModelConfig(use_early_stopping=False)
        wrapper = get_model('xgboost', cfg)
        assert wrapper.config.use_early_stopping is False


class TestRegisterModel:
    def test_register_custom_model(self):
        class CustomWrapper(BaseModelWrapper):
            @property
            def name(self):
                return "custom"

            def get_param_space(self, trial):
                return {}

            def create_model(self, params, is_regression):
                from sklearn.linear_model import LogisticRegression
                return LogisticRegression()

        register_model('custom_test', CustomWrapper)
        assert 'custom_test' in MODEL_REGISTRY
        wrapper = get_model('custom_test')
        assert wrapper.name == 'custom'
        # Cleanup
        del MODEL_REGISTRY['custom_test']

    def test_register_non_subclass_raises(self):
        with pytest.raises(TypeError):
            register_model('bad', object)


class TestModelWrappers:
    def test_xgboost_create_classifier(self):
        wrapper = get_model('xgboost')
        model = wrapper.create_model({'n_estimators': 10, 'max_depth': 3}, is_regression=False)
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict_proba')

    def test_xgboost_create_regressor(self):
        wrapper = get_model('xgboost')
        model = wrapper.create_model({'n_estimators': 10, 'max_depth': 3}, is_regression=True)
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')

    def test_lightgbm_create_classifier(self):
        wrapper = get_model('lightgbm')
        model = wrapper.create_model({'n_estimators': 10, 'max_depth': 3}, is_regression=False)
        assert hasattr(model, 'predict_proba')

    def test_catboost_create_classifier(self):
        wrapper = get_model('catboost')
        model = wrapper.create_model({'iterations': 10, 'depth': 3}, is_regression=False)
        assert hasattr(model, 'predict_proba')


class TestModelFit:
    def test_xgboost_fit_and_predict(self, binary_data):
        X, y = binary_data
        wrapper = get_model('xgboost', ModelConfig(calibrate=False, use_early_stopping=False))
        wrapper.best_params = {'n_estimators': 10, 'max_depth': 3}
        wrapper.fit(X, y)
        probs = wrapper.predict_proba(X)
        assert len(probs) == len(y)
        assert np.all((probs >= 0) & (probs <= 1))

    def test_lightgbm_fit_and_predict(self, binary_data):
        X, y = binary_data
        wrapper = get_model('lightgbm', ModelConfig(calibrate=False, use_early_stopping=False))
        wrapper.best_params = {'n_estimators': 10, 'max_depth': 3}
        wrapper.fit(X, y)
        probs = wrapper.predict_proba(X)
        assert len(probs) == len(y)

    def test_fit_with_calibration(self, binary_data):
        X, y = binary_data
        X_train, X_val = X[:150], X[150:]
        y_train, y_val = y[:150], y[150:]
        wrapper = get_model('xgboost', ModelConfig(calibrate=True, use_early_stopping=False))
        wrapper.best_params = {'n_estimators': 10, 'max_depth': 3}
        wrapper.fit(X_train, y_train, X_val, y_val)
        probs = wrapper.predict_proba(X_val)
        assert len(probs) == len(y_val)


class TestModelEnsemble:
    def test_predict_averages_models(self, binary_data):
        X, y = binary_data
        ensemble = ModelEnsemble(EnsembleConfig(models=['xgboost', 'lightgbm']))
        # Manually train models
        for name in ['xgboost', 'lightgbm']:
            wrapper = get_model(name, ModelConfig(calibrate=False, use_early_stopping=False))
            wrapper.best_params = {'n_estimators': 10, 'max_depth': 3}
            wrapper.fit(X, y)
            ensemble.models[name] = wrapper

        preds = ensemble.predict(X)
        assert len(preds) == len(y)
        assert np.all((preds >= 0) & (preds <= 1))

    def test_weighted_predict(self, binary_data):
        X, y = binary_data
        config = EnsembleConfig(
            models=['xgboost', 'lightgbm'],
            weights={'xgboost': 0.8, 'lightgbm': 0.2},
        )
        ensemble = ModelEnsemble(config)
        for name in ['xgboost', 'lightgbm']:
            wrapper = get_model(name, ModelConfig(calibrate=False, use_early_stopping=False))
            wrapper.best_params = {'n_estimators': 10, 'max_depth': 3}
            wrapper.fit(X, y)
            ensemble.models[name] = wrapper

        preds = ensemble.predict(X)
        assert len(preds) == len(y)

    def test_get_individual_predictions(self, binary_data):
        X, y = binary_data
        ensemble = ModelEnsemble(EnsembleConfig(models=['xgboost']))
        wrapper = get_model('xgboost', ModelConfig(calibrate=False, use_early_stopping=False))
        wrapper.best_params = {'n_estimators': 10, 'max_depth': 3}
        wrapper.fit(X, y)
        ensemble.models['xgboost'] = wrapper

        individual = ensemble.get_individual_predictions(X)
        assert 'xgboost' in individual
        assert 'ensemble' in individual
        assert len(individual['xgboost']) == len(y)
