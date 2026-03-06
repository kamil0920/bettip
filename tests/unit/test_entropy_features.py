"""Tests for NaN-aware entropy feature computation."""

import numpy as np
import pytest

from src.features.engineers.entropy import _permutation_entropy, _sample_entropy


class TestPermutationEntropy:
    def test_normal_sequence(self):
        x = np.array([1.0, 3.0, 2.0, 4.0, 1.0, 3.0, 2.0, 5.0])
        pe = _permutation_entropy(x, order=3, delay=1)
        assert 0.0 < pe <= 1.0

    def test_constant_sequence(self):
        x = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        pe = _permutation_entropy(x, order=3, delay=1)
        assert pe == 0.0

    def test_with_nan_values(self):
        x = np.array([1.0, np.nan, 3.0, np.nan, 2.0, 4.0, 1.0, 3.0])
        pe = _permutation_entropy(x, order=3, delay=1)
        # After NaN drop: [1, 3, 2, 4, 1, 3] — should produce valid PE
        assert not np.isnan(pe)
        assert 0.0 <= pe <= 1.0

    def test_all_nan(self):
        x = np.array([np.nan, np.nan, np.nan])
        pe = _permutation_entropy(x, order=3, delay=1)
        assert np.isnan(pe)

    def test_insufficient_after_nan_drop(self):
        x = np.array([1.0, np.nan, np.nan])
        pe = _permutation_entropy(x, order=3, delay=1)
        assert np.isnan(pe)


class TestSampleEntropy:
    def test_normal_sequence(self):
        # Need enough data and enough matching patterns
        x = np.array([2.0, 4.0, 6.0, 2.0, 4.0, 6.0, 2.0, 4.0, 6.0, 2.0])
        se = _sample_entropy(x, m=2, r_factor=0.3)
        # Repeating pattern should have finite sample entropy
        assert not np.isnan(se)

    def test_constant_sequence(self):
        x = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        se = _sample_entropy(x, m=2, r_factor=0.2)
        assert se == 0.0

    def test_with_nan_values(self):
        # Repeating pattern with NaN gaps
        x = np.array([2.0, np.nan, 4.0, np.nan, 6.0, 2.0, 4.0, 6.0, 2.0, 4.0])
        se = _sample_entropy(x, m=2, r_factor=0.3)
        # After NaN drop: [2, 4, 6, 2, 4, 6, 2, 4] — repeating pattern
        assert not np.isnan(se)

    def test_all_nan(self):
        x = np.array([np.nan, np.nan, np.nan])
        se = _sample_entropy(x, m=2, r_factor=0.2)
        assert np.isnan(se)
