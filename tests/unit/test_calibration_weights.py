"""Tests for importance-weighted calibration (sample_weight parameter)."""

import numpy as np
import pytest

from src.calibration.calibration import (
    BetaCalibrator,
    IsotonicCalibrator,
    PlattScaling,
    TemperatureScaling,
)


@pytest.fixture
def calibration_data():
    """Synthetic calibration data with known structure."""
    rng = np.random.RandomState(42)
    n = 200
    y_prob = rng.beta(2, 5, size=n)  # Skewed predictions
    y_true = (rng.rand(n) < y_prob).astype(int)
    return y_prob, y_true


@pytest.fixture
def non_uniform_weights():
    """Weights that emphasize recent samples (like density ratios)."""
    rng = np.random.RandomState(42)
    w = rng.exponential(1.0, size=200)
    w = w / w.mean()  # Normalize to mean=1
    return w


class TestBetaCalibrator:
    def test_fit_without_weights(self, calibration_data):
        y_prob, y_true = calibration_data
        cal = BetaCalibrator(method="abm")
        cal.fit(y_prob, y_true)
        result = cal.transform(y_prob)
        assert result.shape == y_prob.shape
        assert np.all(result >= 0) and np.all(result <= 1)

    def test_fit_with_weights(self, calibration_data, non_uniform_weights):
        y_prob, y_true = calibration_data
        cal = BetaCalibrator(method="abm")
        cal.fit(y_prob, y_true, sample_weight=non_uniform_weights)
        result = cal.transform(y_prob)
        assert result.shape == y_prob.shape
        assert np.all(result >= 0) and np.all(result <= 1)

    def test_weights_change_params(self, calibration_data, non_uniform_weights):
        """Non-uniform weights should produce different parameters."""
        y_prob, y_true = calibration_data

        cal_no_w = BetaCalibrator(method="abm")
        cal_no_w.fit(y_prob, y_true)

        cal_w = BetaCalibrator(method="abm")
        cal_w.fit(y_prob, y_true, sample_weight=non_uniform_weights)

        # Parameters should differ with non-uniform weights
        assert not (
            np.isclose(cal_no_w.a_, cal_w.a_, atol=1e-6)
            and np.isclose(cal_no_w.b_, cal_w.b_, atol=1e-6)
            and np.isclose(cal_no_w.c_, cal_w.c_, atol=1e-6)
        )

    def test_none_weights_same_as_no_weights(self, calibration_data):
        """sample_weight=None should produce identical results."""
        y_prob, y_true = calibration_data

        cal1 = BetaCalibrator(method="abm")
        cal1.fit(y_prob, y_true)

        cal2 = BetaCalibrator(method="abm")
        cal2.fit(y_prob, y_true, sample_weight=None)

        np.testing.assert_allclose(cal1.a_, cal2.a_, atol=1e-8)
        np.testing.assert_allclose(cal1.b_, cal2.b_, atol=1e-8)

    def test_uniform_weights_same_as_no_weights(self, calibration_data):
        """All-ones weights should produce same results as no weights."""
        y_prob, y_true = calibration_data
        uniform_w = np.ones(len(y_prob))

        cal1 = BetaCalibrator(method="am")
        cal1.fit(y_prob, y_true)

        cal2 = BetaCalibrator(method="am")
        cal2.fit(y_prob, y_true, sample_weight=uniform_w)

        np.testing.assert_allclose(cal1.a_, cal2.a_, atol=1e-6)

    def test_all_methods_accept_weights(self, calibration_data, non_uniform_weights):
        """All beta calibration methods (abm, am, ab) should accept weights."""
        y_prob, y_true = calibration_data
        for method in ["abm", "am", "ab"]:
            cal = BetaCalibrator(method=method)
            cal.fit(y_prob, y_true, sample_weight=non_uniform_weights)
            result = cal.transform(y_prob)
            assert result.shape == y_prob.shape


class TestTemperatureScaling:
    def test_fit_with_weights(self, calibration_data, non_uniform_weights):
        y_prob, y_true = calibration_data
        cal = TemperatureScaling()
        cal.fit(y_prob, y_true, sample_weight=non_uniform_weights)
        result = cal.transform(y_prob)
        assert result.shape == y_prob.shape

    def test_none_weights_same_as_no_weights(self, calibration_data):
        y_prob, y_true = calibration_data
        cal1 = TemperatureScaling()
        cal1.fit(y_prob, y_true)
        cal2 = TemperatureScaling()
        cal2.fit(y_prob, y_true, sample_weight=None)
        np.testing.assert_allclose(cal1.temperature_, cal2.temperature_, atol=1e-8)


class TestPlattScaling:
    def test_fit_with_weights(self, calibration_data, non_uniform_weights):
        y_prob, y_true = calibration_data
        cal = PlattScaling()
        cal.fit(y_prob, y_true, sample_weight=non_uniform_weights)
        result = cal.transform(y_prob)
        assert result.shape == y_prob.shape


class TestIsotonicCalibrator:
    def test_fit_with_weights(self, calibration_data, non_uniform_weights):
        y_prob, y_true = calibration_data
        cal = IsotonicCalibrator()
        cal.fit(y_prob, y_true, sample_weight=non_uniform_weights)
        result = cal.transform(y_prob)
        assert result.shape == y_prob.shape
