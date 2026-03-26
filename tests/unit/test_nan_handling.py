"""Tests for native NaN handling in tree models and feature NaN gate."""

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler


@pytest.fixture
def data_with_nans():
    """Synthetic data with NaN values mimicking real pipeline data."""
    rng = np.random.RandomState(42)
    n = 500
    n_features = 10
    X = rng.randn(n, n_features)
    y = (rng.rand(n) > 0.5).astype(int)

    # Inject NaN at varying rates per feature
    for i in range(n_features):
        nan_rate = i * 0.05  # 0%, 5%, 10%, ..., 45%
        mask = rng.rand(n) < nan_rate
        X[mask, i] = np.nan

    return X, y


class TestCatBoostNativeNaN:
    def test_catboost_trains_with_nan(self, data_with_nans):
        """CatBoost should train successfully with NaN features."""
        try:
            from catboost import CatBoostClassifier
        except ImportError:
            pytest.skip("CatBoost not installed")

        X, y = data_with_nans
        model = CatBoostClassifier(
            iterations=10, depth=3, verbose=0, random_seed=42
        )
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 2)
        assert not np.any(np.isnan(proba))

    def test_catboost_nan_vs_zero_differ(self, data_with_nans):
        """CatBoost predictions should differ between NaN and zero-filled data."""
        try:
            from catboost import CatBoostClassifier
        except ImportError:
            pytest.skip("CatBoost not installed")

        X, y = data_with_nans
        X_zero = np.nan_to_num(X, nan=0.0)

        model_nan = CatBoostClassifier(
            iterations=50, depth=3, verbose=0, random_seed=42
        )
        model_nan.fit(X, y)
        proba_nan = model_nan.predict_proba(X)[:, 1]

        model_zero = CatBoostClassifier(
            iterations=50, depth=3, verbose=0, random_seed=42
        )
        model_zero.fit(X_zero, y)
        proba_zero = model_zero.predict_proba(X_zero)[:, 1]

        # Predictions should differ (NaN routing vs zero routing)
        assert not np.allclose(proba_nan, proba_zero, atol=1e-3)


class TestLightGBMNativeNaN:
    def test_lgb_trains_with_nan(self, data_with_nans):
        """LightGBM should train successfully with NaN features."""
        try:
            import lightgbm as lgb
        except ImportError:
            pytest.skip("LightGBM not installed")

        X, y = data_with_nans
        model = lgb.LGBMClassifier(
            n_estimators=10, max_depth=3, verbose=-1, random_state=42
        )
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 2)
        assert not np.any(np.isnan(proba))


class TestXGBoostNativeNaN:
    def test_xgb_trains_with_nan(self, data_with_nans):
        """XGBoost should train successfully with NaN features."""
        try:
            from xgboost import XGBClassifier
        except ImportError:
            pytest.skip("XGBoost not installed")

        X, y = data_with_nans
        model = XGBClassifier(
            n_estimators=10, max_depth=3, verbosity=0, random_state=42
        )
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 2)
        assert not np.any(np.isnan(proba))


class TestStandardScalerNaN:
    def test_scaler_passes_nan_through(self, data_with_nans):
        """StandardScaler should preserve NaN positions."""
        X, _ = data_with_nans
        nan_mask = np.isnan(X)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # NaN positions should be preserved
        np.testing.assert_array_equal(np.isnan(X_scaled), nan_mask)

        # Non-NaN values should be scaled (mean~0, std~1)
        for col in range(X.shape[1]):
            valid = X_scaled[~nan_mask[:, col], col]
            if len(valid) > 10:
                assert abs(valid.mean()) < 0.1
                assert abs(valid.std() - 1.0) < 0.2


class TestCalibratedClassifierCVWithNaN:
    def test_prefit_calibration_with_nan_features(self, data_with_nans):
        """CalibratedClassifierCV with prefit should work when tree model handles NaN."""
        try:
            from catboost import CatBoostClassifier
            from sklearn.calibration import CalibratedClassifierCV
        except ImportError:
            pytest.skip("CatBoost or sklearn not installed")

        X, y = data_with_nans
        n_train = 400

        model = CatBoostClassifier(
            iterations=10, depth=3, verbose=0, random_seed=42
        )
        model.fit(X[:n_train], y[:n_train])

        calibrated = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
        # CalibratedClassifierCV calls model.predict_proba() internally
        # which should handle NaN if the model was trained on NaN data
        calibrated.fit(X[n_train:], y[n_train:])

        proba = calibrated.predict_proba(X[:10])
        assert proba.shape == (10, 2)
        assert not np.any(np.isnan(proba))
