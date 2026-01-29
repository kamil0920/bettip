"""Unit tests for risk-adjusted metrics: Sharpe, Sortino, ECE."""

import numpy as np
import pytest

from src.ml.metrics import expected_calibration_error, sharpe_ratio, sortino_ratio


class TestSharpeRatio:
    def test_positive_returns(self):
        returns = np.array([0.5, 1.0, 0.8, 0.3, 0.6])
        result = sharpe_ratio(returns)
        assert result > 0

    def test_negative_returns(self):
        returns = np.array([-1.0, -0.5, -1.0, -0.2])
        result = sharpe_ratio(returns)
        assert result < 0

    def test_zero_std(self):
        returns = np.array([1.0, 1.0, 1.0])
        assert sharpe_ratio(returns) == 0.0

    def test_single_return(self):
        assert sharpe_ratio(np.array([1.0])) == 0.0

    def test_empty(self):
        assert sharpe_ratio(np.array([])) == 0.0

    def test_mixed_returns(self):
        returns = np.array([2.0, -1.0, 1.5, -1.0, 3.0])
        result = sharpe_ratio(returns)
        expected = returns.mean() / returns.std(ddof=1)
        assert abs(result - expected) < 1e-10


class TestSortinoRatio:
    def test_all_positive(self):
        returns = np.array([1.0, 2.0, 3.0])
        result = sortino_ratio(returns)
        assert result == float('inf')

    def test_all_negative(self):
        returns = np.array([-1.0, -2.0, -0.5])
        result = sortino_ratio(returns)
        assert result < 0

    def test_mixed(self):
        returns = np.array([2.0, -1.0, 1.5, -0.5, -1.0, 3.0])
        result = sortino_ratio(returns)
        downside = returns[returns < 0]
        expected = returns.mean() / downside.std(ddof=1)
        assert abs(result - expected) < 1e-10

    def test_single(self):
        assert sortino_ratio(np.array([1.0])) == 0.0

    def test_empty(self):
        assert sortino_ratio(np.array([])) == 0.0


class TestExpectedCalibrationError:
    def test_perfect_calibration(self):
        """Perfectly calibrated predictions should have ECE near 0."""
        np.random.seed(42)
        n = 10000
        y_prob = np.random.uniform(0, 1, n)
        y_true = (np.random.random(n) < y_prob).astype(float)
        ece = expected_calibration_error(y_true, y_prob)
        assert ece < 0.05  # Should be very close to 0

    def test_overconfident(self):
        """Overconfident predictions (always predict 0.9) should have high ECE."""
        y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        y_prob = np.full(10, 0.9)
        ece = expected_calibration_error(y_true, y_prob)
        assert ece > 0.3  # Big gap between 0.9 confidence and 0.5 accuracy

    def test_ece_bounded(self):
        """ECE should be between 0 and 1."""
        y_true = np.array([1, 0, 1, 1, 0])
        y_prob = np.array([0.8, 0.2, 0.7, 0.9, 0.1])
        ece = expected_calibration_error(y_true, y_prob)
        assert 0 <= ece <= 1

    def test_all_correct_high_conf(self):
        """All correct at high confidence should have low ECE."""
        y_true = np.ones(100)
        y_prob = np.full(100, 0.95)
        ece = expected_calibration_error(y_true, y_prob)
        assert ece < 0.1
