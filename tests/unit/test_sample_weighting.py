"""
Unit tests for sample weighting utilities.

Tests the retail forecasting integration for time-decayed sample weights
and odds-dependent thresholds.
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

from src.ml.sample_weighting import (
    calculate_time_decay_weights,
    calculate_tiered_weights,
    decay_rate_from_half_life,
    half_life_from_decay_rate,
    get_recommended_decay_rate,
    normalize_weights,
)


class TestTimeDecayWeights:
    """Test time-decayed sample weight calculations."""

    def test_recent_dates_get_higher_weights(self):
        """Recent observations should have weights close to 1.0."""
        dates = pd.to_datetime([
            "2024-01-15",  # Most recent
            "2024-01-01",  # 14 days ago
            "2023-12-01",  # 45 days ago
            "2023-06-01",  # ~7 months ago
        ])
        reference = pd.to_datetime("2024-01-15")

        weights = calculate_time_decay_weights(dates, decay_rate=0.002, reference_date=reference)

        assert weights[0] == 1.0, "Most recent date should have weight 1.0"
        assert weights[1] > weights[2] > weights[3], "Weights should decrease with age"

    def test_exponential_decay_formula(self):
        """Verify exponential decay formula: weight = exp(-decay_rate * days_ago)."""
        reference = pd.to_datetime("2024-01-01")
        dates = pd.to_datetime(["2024-01-01", "2023-06-01"])
        decay_rate = 0.01

        # Calculate actual days between dates
        days_diff = (reference - pd.to_datetime("2023-06-01")).days

        weights = calculate_time_decay_weights(
            dates, decay_rate=decay_rate, reference_date=reference, min_weight=0.0
        )

        expected_recent = 1.0
        expected_old = np.exp(-decay_rate * days_diff)

        assert np.isclose(weights[0], expected_recent), f"Recent weight should be {expected_recent}"
        assert np.isclose(weights[1], expected_old, rtol=0.01), f"Old weight should be ~{expected_old:.4f}"

    def test_min_weight_floor(self):
        """Minimum weight should be enforced for very old observations."""
        dates = pd.to_datetime(["2020-01-01"])  # Very old
        reference = pd.to_datetime("2024-01-15")
        min_weight = 0.1

        weights = calculate_time_decay_weights(
            dates, decay_rate=0.01, reference_date=reference, min_weight=min_weight
        )

        assert weights[0] >= min_weight, f"Weight should be at least {min_weight}"

    def test_uses_max_date_as_reference_by_default(self):
        """If no reference date, should use most recent date in data."""
        dates = pd.to_datetime(["2024-01-15", "2024-01-01", "2023-12-01"])

        weights = calculate_time_decay_weights(dates, decay_rate=0.002)

        assert weights[0] == 1.0, "First date (most recent) should have weight 1.0"

    def test_handles_string_dates(self):
        """Should handle string date inputs."""
        dates = ["2024-01-15", "2024-01-01"]

        weights = calculate_time_decay_weights(dates, decay_rate=0.002)

        assert len(weights) == 2
        assert weights[0] > weights[1]

    def test_handles_numpy_array(self):
        """Should handle numpy array date inputs."""
        dates = np.array(["2024-01-15", "2024-01-01"], dtype="datetime64")

        weights = calculate_time_decay_weights(dates, decay_rate=0.002)

        assert len(weights) == 2

    def test_decay_rate_sensitivity(self):
        """Higher decay rate should cause faster weight decay."""
        dates = pd.to_datetime(["2024-01-15", "2023-01-15"])  # 1 year apart
        reference = pd.to_datetime("2024-01-15")

        weights_slow = calculate_time_decay_weights(dates, decay_rate=0.001, reference_date=reference)
        weights_fast = calculate_time_decay_weights(dates, decay_rate=0.005, reference_date=reference)

        assert weights_slow[1] > weights_fast[1], "Slower decay should give higher weight to old data"


class TestTieredWeights:
    """Test tiered weight calculations (discrete tiers instead of continuous decay)."""

    def test_default_tiers(self):
        """Test default year-based tiers: 1.0/0.5/0.25."""
        reference = pd.to_datetime("2024-01-15")
        # Default tier_days is (365, 730) meaning:
        # < 365 days -> weight 1.0
        # 365-730 days -> weight 0.5
        # > 730 days -> weight 0.25
        dates = pd.to_datetime([
            "2024-01-01",   # 14 days ago: < 365 days -> tier 1 (1.0)
            "2023-03-01",   # ~320 days ago: < 365 days -> tier 1 (1.0)
            "2022-12-01",   # ~410 days ago: 365-730 days -> tier 2 (0.5)
            "2021-06-01",   # ~960 days ago: > 730 days -> tier 3 (0.25)
        ])

        weights = calculate_tiered_weights(dates, reference_date=reference)

        assert weights[0] == 1.0
        assert weights[1] == 1.0
        assert weights[2] == 0.5
        assert weights[3] == 0.25

    def test_custom_tiers(self):
        """Test custom tier configuration."""
        reference = pd.to_datetime("2024-01-15")
        dates = pd.to_datetime([
            "2024-01-01",   # < 180 days
            "2023-06-01",   # > 180 days
        ])

        weights = calculate_tiered_weights(
            dates,
            tier_days=(180,),
            tier_weights=(1.0, 0.3),
            reference_date=reference
        )

        assert weights[0] == 1.0
        assert weights[1] == 0.3


class TestDecayRateConversion:
    """Test conversion between decay rate and half-life."""

    def test_decay_rate_from_half_life(self):
        """Verify decay rate calculation from half-life."""
        half_life_days = 365  # 1 year

        rate = decay_rate_from_half_life(half_life_days)

        # At half-life, weight should be 0.5
        weight_at_half_life = np.exp(-rate * half_life_days)
        assert np.isclose(weight_at_half_life, 0.5, rtol=0.01)

    def test_half_life_from_decay_rate(self):
        """Verify half-life calculation from decay rate."""
        decay_rate = 0.002

        half_life = half_life_from_decay_rate(decay_rate)
        weight_at_half_life = np.exp(-decay_rate * half_life)

        assert np.isclose(weight_at_half_life, 0.5, rtol=0.01)

    def test_round_trip_conversion(self):
        """Converting rate->half_life->rate should give same result."""
        original_rate = 0.003

        half_life = half_life_from_decay_rate(original_rate)
        recovered_rate = decay_rate_from_half_life(half_life)

        assert np.isclose(original_rate, recovered_rate)


class TestRecommendedDecayRate:
    """Test recommended decay rate for different sports."""

    def test_football_decay_rate(self):
        """Football should have ~1 season half-life."""
        rate = get_recommended_decay_rate("football")

        # Half-life should be roughly 1 season (~300-365 days)
        half_life = half_life_from_decay_rate(rate)
        assert 250 < half_life < 400, f"Football half-life should be ~1 season, got {half_life:.0f} days"

    def test_default_decay_rate(self):
        """Unknown sports should get reasonable default."""
        rate = get_recommended_decay_rate("unknown_sport")

        assert rate > 0
        assert rate < 0.01  # Reasonable range


class TestNormalizeWeights:
    """Test weight normalization."""

    def test_normalize_to_mean_one(self):
        """Normalized weights should have mean of target."""
        weights = np.array([1.0, 0.8, 0.6, 0.4, 0.2])
        target_mean = 1.0

        normalized = normalize_weights(weights, target_mean)

        assert np.isclose(normalized.mean(), target_mean)

    def test_normalize_preserves_relative_ordering(self):
        """Normalization should preserve relative weight ordering."""
        weights = np.array([1.0, 0.5, 0.25])

        normalized = normalize_weights(weights, target_mean=1.0)

        assert normalized[0] > normalized[1] > normalized[2]

    def test_normalize_handles_zero_mean(self):
        """Should handle edge case of zero mean weights."""
        weights = np.array([0.0, 0.0, 0.0])

        normalized = normalize_weights(weights, target_mean=1.0)

        # Should not crash, returns original
        assert len(normalized) == 3


class TestOddsAdjustedThreshold:
    """Test odds-dependent threshold calculations.

    Note: The actual implementation is in SniperOptimizer.calculate_odds_adjusted_threshold(),
    but we test the concept here.
    """

    def test_higher_odds_lower_threshold_concept(self):
        """Verify the newsvendor concept: longshots get lower thresholds."""
        # Formula: threshold = base * (2.0 / odds)^alpha
        base_threshold = 0.6
        alpha = 0.3

        # Low odds (favorite)
        low_odds = 1.5
        low_thresh = base_threshold * (2.0 / low_odds) ** alpha

        # High odds (longshot)
        high_odds = 4.0
        high_thresh = base_threshold * (2.0 / high_odds) ** alpha

        assert high_thresh < low_thresh, "Longshots should have lower threshold"
        assert low_thresh > base_threshold, "Favorites should have higher threshold"
        assert high_thresh < base_threshold, "Longshots should have lower threshold"

    def test_alpha_zero_gives_fixed_threshold(self):
        """When alpha=0, threshold should be fixed regardless of odds."""
        base_threshold = 0.6
        alpha = 0.0

        odds = np.array([1.5, 2.0, 3.0, 5.0])
        thresholds = base_threshold * (2.0 / odds) ** alpha

        assert np.allclose(thresholds, base_threshold)

    def test_alpha_one_gives_full_adjustment(self):
        """When alpha=1, threshold scales inversely with odds."""
        base_threshold = 0.6
        alpha = 1.0

        # At odds=2.0, threshold should be base (odds_ratio = 1)
        odds_even = 2.0
        thresh_even = base_threshold * (2.0 / odds_even) ** alpha
        assert np.isclose(thresh_even, base_threshold)

        # At odds=4.0, threshold should be base/2
        odds_longshot = 4.0
        thresh_longshot = base_threshold * (2.0 / odds_longshot) ** alpha
        assert np.isclose(thresh_longshot, base_threshold / 2)
