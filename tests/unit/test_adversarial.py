"""Unit tests for adversarial validation and filtering."""

import numpy as np
import pytest

from src.ml.adversarial import _adversarial_filter, _adversarial_validation


def _make_temporal_data(n_train=200, n_test=100, n_features=10, leaky_features=2):
    """Create data where first `leaky_features` columns have temporal shift."""
    rng = np.random.RandomState(42)
    X_train = rng.randn(n_train, n_features)
    X_test = rng.randn(n_test, n_features)
    # Inject temporal shift in first `leaky_features` columns
    for i in range(leaky_features):
        X_train[:, i] += 0  # train centered at 0
        X_test[:, i] += 3   # test shifted to 3 — easy to distinguish
    feature_names = [f"f{i}" for i in range(n_features)]
    return X_train, X_test, feature_names


class TestAdversarialValidation:
    def test_detects_shift(self):
        """When train/test distributions differ, AUC should be high."""
        X_train, X_test, names = _make_temporal_data(leaky_features=3)
        auc, top_shift, _ = _adversarial_validation(X_train, X_test, names)
        assert auc > 0.7

    def test_no_shift_lower_auc_than_shifted(self):
        """Without shift, AUC should be lower than with strong shift."""
        rng = np.random.RandomState(42)
        X_same = rng.randn(300, 10)
        names = [f"f{i}" for i in range(10)]
        auc_no_shift, _, _ = _adversarial_validation(X_same[:200], X_same[200:], names)

        X_train, X_test, _ = _make_temporal_data(leaky_features=5)
        auc_shifted, _, _ = _adversarial_validation(X_train, X_test, names)
        assert auc_shifted > auc_no_shift

    def test_returns_top_features(self):
        X_train, X_test, names = _make_temporal_data(leaky_features=2)
        auc, top_shift, _ = _adversarial_validation(X_train, X_test, names)
        assert len(top_shift) <= 10
        # Leaky features should be in top shift
        top_names = [name for name, _ in top_shift]
        assert 'f0' in top_names or 'f1' in top_names


class TestAdversarialFilter:
    def _make_temporal_X(self, n=300, n_features=20, leaky=3):
        """Create single matrix with temporal structure (first 70% train, last 30% test)."""
        rng = np.random.RandomState(42)
        X = rng.randn(n, n_features)
        split = int(n * 0.7)
        for i in range(leaky):
            X[split:, i] += 4  # shift test portion
        names = [f"f{i}" for i in range(n_features)]
        return X, names

    def test_removes_leaky_features(self):
        X, names = self._make_temporal_X(leaky=3)
        filtered_X, filtered_names, diag = _adversarial_filter(
            X, names, max_passes=2, auc_threshold=0.6
        )
        assert len(filtered_names) < len(names)
        assert diag['total_removed'] > 0
        # At least some leaky features removed
        removed = diag['removed_features']
        assert any(f in removed for f in ['f0', 'f1', 'f2'])

    def test_high_threshold_prevents_removal(self):
        """With very high AUC threshold, no features should be removed."""
        rng = np.random.RandomState(42)
        X = rng.randn(300, 15)
        names = [f"f{i}" for i in range(15)]
        filtered_X, filtered_names, diag = _adversarial_filter(
            X, names, max_passes=2, auc_threshold=0.999
        )
        assert len(filtered_names) == len(names)
        assert diag['total_removed'] == 0

    def test_preserves_minimum_features(self):
        """Should never reduce below 5 features."""
        rng = np.random.RandomState(42)
        X = rng.randn(300, 8)
        split = int(300 * 0.7)
        # Make ALL features leaky
        for i in range(8):
            X[split:, i] += 5
        names = [f"f{i}" for i in range(8)]
        filtered_X, filtered_names, diag = _adversarial_filter(
            X, names, max_passes=5, auc_threshold=0.5
        )
        assert len(filtered_names) >= 5

    def test_diagnostics_structure(self):
        X, names = self._make_temporal_X()
        _, _, diag = _adversarial_filter(X, names)
        assert 'initial_n_features' in diag
        assert 'final_n_features' in diag
        assert 'removed_features' in diag
        assert 'passes' in diag
        assert isinstance(diag['passes'], list)

    def test_respects_max_passes(self):
        X, names = self._make_temporal_X(leaky=5)
        _, _, diag = _adversarial_filter(
            X, names, max_passes=1, auc_threshold=0.5
        )
        assert len(diag['passes']) <= 1

    def test_does_not_modify_input(self):
        X, names = self._make_temporal_X()
        X_original = X.copy()
        names_original = list(names)
        _adversarial_filter(X, names)
        np.testing.assert_array_equal(X, X_original)
        assert names == names_original
