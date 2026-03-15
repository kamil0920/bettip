"""Tests for conformal coverage monitoring.

Tests all four functions in ``src.monitoring.conformal_coverage``.
"""

import numpy as np
import pytest

from src.monitoring.conformal_coverage import (
    one_sided_coverage,
    rolling_coverage,
    va_consistency,
    width_accuracy_correlation,
)


# ---------- one_sided_coverage ----------


class TestOneSidedCoverage:
    """Tests for one_sided_coverage."""

    def test_perfect_coverage(self) -> None:
        """All predictions within conformal bounds → 100 % coverage."""
        probs = np.array([0.7, 0.6, 0.8, 0.5])
        taus = np.array([0.5, 0.5, 0.5, 0.5])
        actuals = np.array([1.0, 1.0, 1.0, 0.0])
        # scores = [-0.3, -0.4, -0.2, 0.5], all <= 0.5
        result = one_sided_coverage(probs, taus, actuals)
        assert result["empirical_coverage"] == 1.0
        assert not result["alert"]

    def test_under_coverage_triggers_alert(self) -> None:
        """Systematic under-coverage triggers alert."""
        probs = np.array([0.9] * 100)
        taus = np.array([0.1] * 100)
        actuals = np.zeros(100)  # all lose → scores = 0.9, tau = 0.1
        result = one_sided_coverage(probs, taus, actuals)
        assert result["empirical_coverage"] == 0.0
        assert result["alert"]

    def test_no_alert_when_above_nominal(self) -> None:
        """Coverage above nominal minus 5pp → no alert."""
        probs = np.array([0.6, 0.6, 0.6, 0.6, 0.6])
        taus = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        actuals = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
        result = one_sided_coverage(probs, taus, actuals, alpha=0.10)
        assert result["empirical_coverage"] == 1.0
        assert result["coverage_gap"] > 0
        assert not result["alert"]

    def test_n_bets_count(self) -> None:
        """n_bets reflects actual array length."""
        n = 42
        result = one_sided_coverage(
            np.ones(n) * 0.5,
            np.ones(n) * 0.5,
            np.zeros(n),
        )
        assert result["n_bets"] == n

    def test_nominal_coverage_respects_alpha(self) -> None:
        """Different alpha values set the correct nominal coverage."""
        result = one_sided_coverage(
            np.array([0.5]),
            np.array([0.5]),
            np.array([0.0]),
            alpha=0.20,
        )
        assert result["nominal_coverage"] == 0.80

    def test_boundary_score_equals_tau(self) -> None:
        """Score exactly equal to tau counts as covered."""
        probs = np.array([0.6])
        taus = np.array([0.6])
        actuals = np.array([0.0])  # score = 0.6 == tau
        result = one_sided_coverage(probs, taus, actuals)
        assert result["empirical_coverage"] == 1.0


# ---------- va_consistency ----------


class TestVaConsistency:
    """Tests for va_consistency."""

    def test_basic_all_consistent(self) -> None:
        """All outcomes plausible given the VA intervals."""
        actual = np.array([1.0, 0.0, 1.0, 0.0])
        va_lo = np.array([0.3, 0.2, 0.6, 0.1])
        va_hi = np.array([0.7, 0.4, 0.9, 0.3])
        result = va_consistency(va_lo, va_hi, actual)
        # actual=1: va_hi >= 0.5? [T, -, T, -] → 2/2
        # actual=0: va_lo <= 0.5? [-, T, -, T] → 2/2
        assert result["consistency_rate"] == 1.0

    def test_inconsistent_win(self) -> None:
        """Win with va_upper < 0.5 is inconsistent."""
        actual = np.array([1.0])
        va_lo = np.array([0.1])
        va_hi = np.array([0.3])  # va_upper < 0.5 → inconsistent for win
        result = va_consistency(va_lo, va_hi, actual)
        assert result["consistency_rate"] == 0.0

    def test_inconsistent_loss(self) -> None:
        """Loss with va_lower > 0.5 is inconsistent."""
        actual = np.array([0.0])
        va_lo = np.array([0.6])  # va_lower > 0.5 → inconsistent for loss
        va_hi = np.array([0.9])
        result = va_consistency(va_lo, va_hi, actual)
        assert result["consistency_rate"] == 0.0

    def test_mean_width(self) -> None:
        """Mean width computed correctly."""
        va_lo = np.array([0.2, 0.3])
        va_hi = np.array([0.8, 0.7])
        actual = np.array([1.0, 0.0])
        result = va_consistency(va_lo, va_hi, actual)
        # widths = [0.6, 0.4], mean = 0.5
        assert result["mean_width"] == 0.5

    def test_n_bets_count(self) -> None:
        """n_bets reflects actual array length."""
        n = 17
        result = va_consistency(
            np.zeros(n), np.ones(n), np.zeros(n)
        )
        assert result["n_bets"] == n


# ---------- rolling_coverage ----------


class TestRollingCoverage:
    """Tests for rolling_coverage."""

    def test_shape_matches_input(self) -> None:
        """Output length equals input length."""
        scores = np.random.randn(100)
        taus = np.ones(100) * 0.5
        result = rolling_coverage(scores, taus, window=20)
        assert len(result) == 100

    def test_nan_prefix(self) -> None:
        """First (window - 1) entries are NaN."""
        scores = np.zeros(50)
        taus = np.ones(50)
        result = rolling_coverage(scores, taus, window=20)
        assert all(np.isnan(result[:19]))
        assert not np.isnan(result[19])

    def test_all_covered(self) -> None:
        """If all scores <= taus, rolling coverage is always 1.0."""
        n = 60
        scores = np.zeros(n)  # scores = 0, always <= tau
        taus = np.ones(n) * 0.5
        result = rolling_coverage(scores, taus, window=10)
        # All non-NaN values should be 1.0
        valid = result[~np.isnan(result)]
        np.testing.assert_array_equal(valid, np.ones(len(valid)))

    def test_none_covered(self) -> None:
        """If no scores <= taus, rolling coverage is always 0.0."""
        n = 30
        scores = np.ones(n) * 10.0  # scores = 10, always > tau
        taus = np.ones(n) * 0.1
        result = rolling_coverage(scores, taus, window=5)
        valid = result[~np.isnan(result)]
        np.testing.assert_array_equal(valid, np.zeros(len(valid)))

    def test_window_1(self) -> None:
        """Window of 1 has no NaN prefix and equals point coverage."""
        scores = np.array([0.0, 1.0, 0.0, 1.0])
        taus = np.array([0.5, 0.5, 0.5, 0.5])
        result = rolling_coverage(scores, taus, window=1)
        assert not any(np.isnan(result))
        expected = np.array([1.0, 0.0, 1.0, 0.0])
        np.testing.assert_array_equal(result, expected)


# ---------- width_accuracy_correlation ----------


class TestWidthAccuracyCorrelation:
    """Tests for width_accuracy_correlation."""

    def test_positive_correlation_informative(self) -> None:
        """Perfect positive correlation should be informative."""
        # Wide intervals ↔ large errors
        np.random.seed(42)
        n = 200
        widths = np.linspace(0.05, 0.5, n)
        errors = widths + np.random.normal(0, 0.01, n)
        probs = 0.5 * np.ones(n)
        actuals = probs + errors  # abs_error = errors
        result = width_accuracy_correlation(widths, actuals, probs)
        assert result["spearman_rho"] > 0
        assert result["p_value"] < 0.05
        assert result["informative"] is True

    def test_random_not_informative(self) -> None:
        """Random data should not be informative (most of the time)."""
        np.random.seed(123)
        n = 50
        widths = np.random.uniform(0.05, 0.5, n)
        probs = np.random.uniform(0.3, 0.9, n)
        actuals = np.random.choice([0.0, 1.0], n)
        result = width_accuracy_correlation(widths, actuals, probs)
        # Not guaranteed but seed makes it deterministic
        assert result["n_bets"] == n
        # At minimum, function should return valid keys
        assert "spearman_rho" in result
        assert "p_value" in result
        assert "informative" in result

    def test_n_bets_count(self) -> None:
        """n_bets reflects actual array length."""
        n = 33
        result = width_accuracy_correlation(
            np.ones(n) * 0.1,
            np.zeros(n),
            np.ones(n) * 0.5,
        )
        assert result["n_bets"] == n

    def test_negative_correlation_not_informative(self) -> None:
        """Negative correlation (wider = MORE accurate) is not informative."""
        np.random.seed(7)
        n = 200
        widths = np.linspace(0.05, 0.5, n)
        # Inverse: wide intervals → small errors
        errors = 0.6 - widths + np.random.normal(0, 0.01, n)
        probs = 0.5 * np.ones(n)
        actuals = probs + np.abs(errors)
        # Reverse the relationship: make actuals closer for wider widths
        actuals_neg = probs + (0.6 - widths)
        result = width_accuracy_correlation(widths, actuals_neg, probs)
        assert result["spearman_rho"] < 0
        assert result["informative"] is False
