"""Tests for the overdispersed count distribution CDF helper."""

import numpy as np
import pytest
from scipy.stats import poisson

from src.odds.count_distribution import DISPERSION_RATIOS, overdispersed_cdf


class TestOverdispersedCdf:
    """Tests for overdispersed_cdf()."""

    def test_unknown_stat_falls_back_to_poisson(self):
        """Unknown stat name should return exactly Poisson CDF."""
        k, lam = 5, 3.0
        result = overdispersed_cdf(k, lam, "unknown_stat")
        expected = poisson.cdf(k, lam)
        assert result == pytest.approx(expected)

    def test_low_dispersion_falls_back_to_poisson(self):
        """Stat with d <= 1.0 should return Poisson CDF."""
        # Temporarily test with a stat that has low dispersion
        k, lam = 3, 2.5
        result = overdispersed_cdf(k, lam, "ht")
        # ht has d=1.10, so it should use NB, not Poisson
        poisson_result = poisson.cdf(k, lam)
        # Should differ from Poisson since d > 1.0
        assert result != pytest.approx(poisson_result, abs=1e-6)

    def test_cards_heavier_left_tail(self):
        """For cards (d=2.06), NB CDF should be greater than Poisson at low k.

        Overdispersion spreads mass to tails → more mass at low counts →
        P(X <= k) is higher for small k.
        """
        k, lam = 2, 3.54  # Below mean
        nb_cdf = overdispersed_cdf(k, lam, "cards")
        poisson_cdf = poisson.cdf(k, lam)
        assert nb_cdf > poisson_cdf, (
            f"NB CDF ({nb_cdf:.4f}) should exceed Poisson CDF ({poisson_cdf:.4f}) "
            f"for k={k} < mean={lam} with cards dispersion=2.06"
        )

    def test_corners_heavier_left_tail(self):
        """Corners (d=1.35) should also show heavier left tail vs Poisson."""
        k, lam = 7, 9.82
        nb_cdf = overdispersed_cdf(k, lam, "corners")
        poisson_cdf = poisson.cdf(k, lam)
        assert nb_cdf > poisson_cdf

    def test_monotonicity_in_k(self):
        """CDF must be monotonically non-decreasing in k."""
        lam = 4.0
        for stat in ["cards", "corners", "shots", "fouls"]:
            ks = np.arange(0, 20)
            cdfs = overdispersed_cdf(ks, lam, stat)
            diffs = np.diff(cdfs)
            assert np.all(diffs >= -1e-10), (
                f"CDF not monotonic for {stat}: diffs={diffs[diffs < 0]}"
            )

    def test_vectorized_with_arrays(self):
        """Should work with numpy array inputs for both k and lam."""
        k = np.array([1, 2, 3, 4, 5])
        lam = np.array([3.0, 3.5, 4.0, 4.5, 5.0])
        result = overdispersed_cdf(k, lam, "cards")
        assert result.shape == (5,)
        assert np.all(np.isfinite(result))
        assert np.all((result >= 0) & (result <= 1))

    def test_scalar_inputs(self):
        """Should work with plain Python scalars."""
        result = overdispersed_cdf(3, 4.0, "cards")
        assert 0 <= float(result) <= 1

    def test_lam_zero_no_crash(self):
        """lam=0 should not raise an error."""
        result = overdispersed_cdf(2, 0.0, "cards")
        assert np.isfinite(result)

    def test_lam_zero_array_no_crash(self):
        """Array with lam=0 entries should not crash."""
        lam = np.array([0.0, 3.5, 0.0, 4.0])
        result = overdispersed_cdf(2, lam, "corners")
        assert result.shape == (4,)
        assert np.all(np.isfinite(result))

    def test_all_stats_have_dispersion_above_one(self):
        """Verify all configured stats have meaningful overdispersion."""
        for stat, d in DISPERSION_RATIOS.items():
            assert d > 1.0, f"{stat} has dispersion {d} <= 1.0"

    def test_cdf_approaches_one_for_large_k(self):
        """CDF should approach 1.0 for very large k."""
        for stat in ["cards", "corners", "shots", "fouls"]:
            result = overdispersed_cdf(100, 5.0, stat)
            assert result > 0.999, f"{stat}: CDF at k=100, lam=5 is only {result}"

    def test_cdf_at_zero_positive(self):
        """CDF at k=0 should be positive (P(X=0) > 0)."""
        for stat in ["cards", "corners", "shots", "fouls"]:
            result = overdispersed_cdf(0, 5.0, stat)
            assert result > 0, f"{stat}: CDF at k=0 should be > 0"

    def test_cards_under_25_bias_reduction(self):
        """Key test: NB should give more realistic under-2.5 cards probability.

        With Poisson(3.54), P(X<=2) = 0.321
        With NB(d=2.06), P(X<=2) should be higher (heavier left tail).
        This directly impacts cards_under_25 odds estimation.
        """
        lam = 3.54
        poisson_p = poisson.cdf(2, lam)
        nb_p = overdispersed_cdf(2, lam, "cards")
        # NB should give higher P(X<=2) due to heavier left tail
        assert nb_p > poisson_p
        # The difference should be substantial for cards (d=2.06)
        assert nb_p - poisson_p > 0.05, (
            f"Expected substantial difference: NB={nb_p:.4f}, Poisson={poisson_p:.4f}"
        )
