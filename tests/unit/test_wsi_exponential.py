"""Tests for WSI exponential weighting option."""

import numpy as np
import pytest

from src.features.engineers.form import StreakFeatureEngineer


class TestWSIExponentialWeighting:
    """Test exponential vs linear weighting for Weighted Streak Index."""

    def test_exponential_differs_from_linear(self):
        """Exponential weighting should produce different result than linear."""
        results = [1, -1, 1, 1, -1, 1]
        wsi_linear = StreakFeatureEngineer._compute_wsi(results, weighting="linear")
        wsi_exp = StreakFeatureEngineer._compute_wsi(results, weighting="exponential")
        assert wsi_linear != wsi_exp

    def test_default_is_linear(self):
        """Default weighting should be linear."""
        results = [1, -1, 1]
        wsi_default = StreakFeatureEngineer._compute_wsi(results)
        wsi_linear = StreakFeatureEngineer._compute_wsi(results, weighting="linear")
        assert wsi_default == wsi_linear

    def test_exponential_weights_recent_more(self):
        """Exponential should weight recent results more heavily than linear."""
        # Recent win after losses
        results = [-1, -1, -1, -1, -1, 1]
        wsi_linear = StreakFeatureEngineer._compute_wsi(results, weighting="linear")
        wsi_exp = StreakFeatureEngineer._compute_wsi(results, weighting="exponential")
        # With exponential, the recent win gets ~50% of total weight
        # So exponential WSI should be higher (less negative) than linear
        assert wsi_exp > wsi_linear

    def test_all_wins_equal(self):
        """With all identical results, weighting shouldn't matter."""
        results = [1, 1, 1, 1, 1]
        wsi_linear = StreakFeatureEngineer._compute_wsi(results, weighting="linear")
        wsi_exp = StreakFeatureEngineer._compute_wsi(results, weighting="exponential")
        # All +1 * normalized_weights = 1.0 regardless of weighting
        assert wsi_linear == pytest.approx(1.0)
        assert wsi_exp == pytest.approx(1.0)

    def test_empty_results(self):
        """Empty results should return NaN for both weightings."""
        assert np.isnan(StreakFeatureEngineer._compute_wsi([], weighting="linear"))
        assert np.isnan(StreakFeatureEngineer._compute_wsi([], weighting="exponential"))

    def test_constructor_accepts_weighting(self):
        """StreakFeatureEngineer should accept wsi_weighting param."""
        eng = StreakFeatureEngineer(wsi_window=6, wsi_weighting="exponential")
        assert eng.wsi_weighting == "exponential"

    def test_constructor_default_linear(self):
        """Default constructor should use linear weighting."""
        eng = StreakFeatureEngineer()
        assert eng.wsi_weighting == "linear"
