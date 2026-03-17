"""Tests for cross-market covariates (implied goal intensities) and match-varying dispersion."""
import numpy as np
import pandas as pd
import pytest

from src.features.engineers.cross_market import _extract_implied_goal_intensities
from src.odds.count_distribution import match_varying_dispersion, overdispersed_cdf


class TestImpliedGoalIntensities:
    """Tests for L-BFGS-B inversion of HAD odds to goal intensities."""

    def test_balanced_match(self):
        """Balanced match should have TG ≈ 2.5-2.8 and |SUP| ≈ 0."""
        # Typical balanced match: H 2.60, D 3.30, A 2.80
        result = _extract_implied_goal_intensities(2.60, 3.30, 2.80)
        assert not np.isnan(result['implied_total_goals'])
        assert 1.5 < result['implied_total_goals'] < 4.0
        assert abs(result['implied_goal_supremacy']) < 1.0

    def test_strong_favourite(self):
        """Strong favourite should have positive goal supremacy (home favoured)."""
        # Home strong favourite: H 1.40, D 4.50, A 8.00
        result = _extract_implied_goal_intensities(1.40, 4.50, 8.00)
        assert not np.isnan(result['implied_goal_supremacy'])
        assert result['implied_goal_supremacy'] > 0, "Home favourite → positive SUP"

    def test_away_favourite(self):
        """Away favourite should have negative goal supremacy."""
        # Away favourite: H 5.00, D 4.00, A 1.60
        result = _extract_implied_goal_intensities(5.00, 4.00, 1.60)
        assert not np.isnan(result['implied_goal_supremacy'])
        assert result['implied_goal_supremacy'] < 0, "Away favourite → negative SUP"

    def test_abs_goal_supremacy(self):
        """abs_goal_supremacy should be |SUP|."""
        result = _extract_implied_goal_intensities(2.00, 3.50, 4.00)
        if not np.isnan(result['implied_goal_supremacy']):
            assert abs(
                result['abs_goal_supremacy'] - abs(result['implied_goal_supremacy'])
            ) < 1e-10

    def test_invalid_odds_returns_nan(self):
        """Invalid odds should return NaN values."""
        result = _extract_implied_goal_intensities(0.5, 3.0, 4.0)
        assert np.isnan(result['implied_total_goals'])

    def test_nan_odds_returns_nan(self):
        """NaN odds should return NaN values."""
        result = _extract_implied_goal_intensities(np.nan, 3.0, 4.0)
        assert np.isnan(result['implied_total_goals'])


class TestMatchVaryingDispersion:
    """Tests for match-varying NegBin dispersion."""

    def test_zero_supremacy_returns_base(self):
        """When goal supremacy is 0, dispersion should be near base value."""
        d = match_varying_dispersion("corners", np.array([0.0]))
        assert abs(d[0] - 1.35) < 0.1  # Should be close to base

    def test_higher_supremacy_higher_dispersion(self):
        """More one-sided matches should have higher dispersion."""
        d_balanced = match_varying_dispersion("corners", np.array([0.1]))
        d_onesided = match_varying_dispersion("corners", np.array([2.0]))
        assert d_onesided[0] > d_balanced[0]

    def test_array_input(self):
        """Should work with array inputs."""
        sup = np.array([0.0, 0.5, 1.0, 2.0])
        d = match_varying_dispersion("corners", sup)
        assert len(d) == 4
        # Dispersion should be monotonically increasing with |SUP|
        assert all(d[i] <= d[i + 1] for i in range(len(d) - 1))

    def test_unknown_stat_returns_base(self):
        """Unknown stat should return base dispersion (1.0) array."""
        d = match_varying_dispersion("unknown_stat", np.array([1.0]))
        # Unknown stat has d=1.0 in DISPERSION_RATIOS and alpha_1=0
        assert d[0] == 1.0

    def test_dispersion_clamped(self):
        """Dispersion should be clamped to [1.01, 5.0]."""
        # Very extreme supremacy
        d = match_varying_dispersion("corners", np.array([100.0]))
        assert d[0] <= 5.0
        assert d[0] >= 1.01


class TestOverdispersedCDFWithVaryingDispersion:
    """Tests for overdispersed_cdf with per-match dispersion."""

    def test_fixed_dispersion_unchanged(self):
        """Without dispersion arg, should behave as before."""
        cdf1 = overdispersed_cdf(10, 9.5, "corners")
        cdf2 = overdispersed_cdf(10, 9.5, "corners", dispersion=None)
        assert abs(cdf1 - cdf2) < 1e-10

    def test_per_match_dispersion_array(self):
        """Per-match dispersion should produce different CDFs."""
        k = np.array([10, 10, 10])
        lam = np.array([9.5, 9.5, 9.5])
        d = np.array([1.2, 1.5, 2.0])
        result = overdispersed_cdf(k, lam, "corners", dispersion=d)
        assert len(result) == 3
        # Higher dispersion → heavier tails → lower CDF at same point
        assert result[0] > result[1] > result[2]

    def test_poisson_fallback_in_array(self):
        """Elements with d <= 1.0 should use Poisson."""
        from scipy.stats import poisson
        k = np.array([5, 5])
        lam = np.array([4.0, 4.0])
        d = np.array([0.9, 1.5])  # First element should use Poisson
        result = overdispersed_cdf(k, lam, "test", dispersion=d)
        expected_poisson = poisson.cdf(5, 4.0)
        assert abs(result[0] - expected_poisson) < 1e-10
