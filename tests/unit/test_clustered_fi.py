"""Tests for clustered feature importance."""
import numpy as np
import pytest

from src.ml.explainability import cluster_features, clustered_feature_importance


class TestClusterFeatures:
    """Tests for feature clustering."""

    def test_identical_features_same_cluster(self):
        """Perfectly correlated features should be in one cluster."""
        rng = np.random.RandomState(42)
        x1 = rng.normal(0, 1, 100)
        X = np.column_stack([x1, x1 * 1.1 + 0.01, rng.normal(0, 1, 100)])
        result = cluster_features(X, ["a", "b", "c"], threshold=0.5)
        # a and b should be in the same cluster, c separate
        clusters = result["clusters"]
        assert result["n_clusters"] == 2
        # Find which cluster has a and b
        for cluster_feats in clusters.values():
            if "a" in cluster_feats:
                assert "b" in cluster_feats

    def test_independent_features_separate(self):
        """Independent features should be in separate clusters."""
        rng = np.random.RandomState(42)
        X = rng.normal(0, 1, (100, 5))
        result = cluster_features(X, ["a", "b", "c", "d", "e"], threshold=0.3)
        # With independent features, expect ~5 clusters at low threshold
        assert result["n_clusters"] >= 4

    def test_single_feature(self):
        """Single feature should return one cluster."""
        X = np.random.RandomState(42).normal(0, 1, (50, 1))
        result = cluster_features(X, ["a"], threshold=0.5)
        assert result["n_clusters"] == 1
        assert result["clusters"] == {0: ["a"]}

    def test_threshold_effect(self):
        """Higher threshold should produce fewer clusters."""
        rng = np.random.RandomState(42)
        x1 = rng.normal(0, 1, 100)
        X = np.column_stack([
            x1, x1 + rng.normal(0, 0.3, 100),
            rng.normal(0, 1, 100), rng.normal(0, 1, 100),
        ])
        names = ["a", "b", "c", "d"]
        r_low = cluster_features(X, names, threshold=0.3)
        r_high = cluster_features(X, names, threshold=1.5)
        assert r_high["n_clusters"] <= r_low["n_clusters"]


class TestClusteredFeatureImportance:
    """Tests for CFI."""

    def test_cfi_returns_all_features(self):
        """CFI should return importance for all features."""
        from sklearn.ensemble import GradientBoostingClassifier

        rng = np.random.RandomState(42)
        X = rng.normal(0, 1, (200, 5))
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        model = GradientBoostingClassifier(n_estimators=20, max_depth=3, random_state=42)
        model.fit(X, y)

        result = clustered_feature_importance(
            model, X, y, ["f1", "f2", "f3", "f4", "f5"],
            cluster_threshold=0.5, n_repeats=3, random_state=42,
        )
        assert len(result["feature_importance"]) == 5
        assert result["n_clusters"] >= 1

    def test_important_cluster_has_higher_importance(self):
        """Cluster containing signal features should rank higher."""
        from sklearn.ensemble import GradientBoostingClassifier

        rng = np.random.RandomState(42)
        x_signal = rng.normal(0, 1, 200)
        X = np.column_stack([
            x_signal, x_signal + rng.normal(0, 0.1, 200),  # Correlated signal pair
            rng.normal(0, 1, 200), rng.normal(0, 1, 200),  # Noise
        ])
        y = (x_signal > 0).astype(int)
        model = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
        model.fit(X, y)

        result = clustered_feature_importance(
            model, X, y, ["sig1", "sig2", "noise1", "noise2"],
            cluster_threshold=0.5, n_repeats=5, random_state=42,
        )
        # Signal features should have higher importance than noise
        sig_imp = result["feature_importance"]["sig1"] + result["feature_importance"]["sig2"]
        noise_imp = result["feature_importance"]["noise1"] + result["feature_importance"]["noise2"]
        assert sig_imp > noise_imp
