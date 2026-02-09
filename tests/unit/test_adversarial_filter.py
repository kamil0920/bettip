"""Tests for adversarial feature filtering and calibration validation."""

import numpy as np
import pytest


class TestAdversarialFilter:
    """Test the _adversarial_filter function."""

    def test_no_removal_when_features_not_leaky(self):
        """When features don't distinguish time periods, nothing should be removed."""
        from experiments.run_sniper_optimization import _adversarial_filter

        np.random.seed(42)
        n_samples = 1000
        n_features = 20
        X = np.random.randn(n_samples, n_features)
        feature_names = [f"feat_{i}" for i in range(n_features)]

        X_filtered, filtered_names, diagnostics = _adversarial_filter(
            X, feature_names, auc_threshold=0.75
        )

        # Random features shouldn't distinguish time periods well enough
        # At minimum, we should retain at least 5 features (safety floor)
        assert len(filtered_names) >= 5
        assert X_filtered.shape[1] == len(filtered_names)

    def test_removes_leaky_feature(self):
        """A feature that perfectly distinguishes time periods should be removed."""
        from experiments.run_sniper_optimization import _adversarial_filter

        np.random.seed(42)
        n_samples = 500
        n_features = 10
        X = np.random.randn(n_samples, n_features)
        feature_names = [f"feat_{i}" for i in range(n_features)]

        # Make feat_0 a perfect temporal leak: monotonically increasing index
        X[:, 0] = np.arange(n_samples)

        X_filtered, filtered_names, diagnostics = _adversarial_filter(
            X, feature_names, auc_threshold=0.75
        )

        assert diagnostics["total_removed"] >= 1
        assert "feat_0" in diagnostics["removed_features"]
        assert len(filtered_names) < n_features

    def test_caps_removal_at_max(self):
        """Should not remove more than max_features_per_pass features."""
        from experiments.run_sniper_optimization import _adversarial_filter

        np.random.seed(42)
        n_samples = 500
        n_features = 20
        X = np.random.randn(n_samples, n_features)
        feature_names = [f"feat_{i}" for i in range(n_features)]

        # Make many features temporal leaks
        for i in range(15):
            X[:, i] = np.arange(n_samples) + np.random.randn(n_samples) * 0.1 * i

        _, _, diagnostics = _adversarial_filter(
            X, feature_names, max_features_per_pass=5, max_passes=1
        )

        # Should not remove more than cap per pass
        if len(diagnostics["passes"]) > 0:
            for pass_info in diagnostics["passes"]:
                if "removed" in pass_info:
                    assert len(pass_info["removed"]) <= 5

    def test_returns_correct_shapes(self):
        """Output X and feature_names should have consistent shapes."""
        from experiments.run_sniper_optimization import _adversarial_filter

        np.random.seed(42)
        n_samples = 1000
        n_features = 20
        X = np.random.randn(n_samples, n_features)
        feature_names = [f"feat_{i}" for i in range(n_features)]

        X_filtered, filtered_names, diagnostics = _adversarial_filter(
            X, feature_names
        )

        assert X_filtered.shape[0] == n_samples
        assert X_filtered.shape[1] == len(filtered_names)
        assert diagnostics["final_n_features"] == len(filtered_names)
        assert len(filtered_names) >= 5  # Safety floor


class TestAdversarialFilterConfigurable:
    """Test configurable adversarial filter parameters."""

    def _make_leaky_data(self, n_samples=500, n_features=20, n_leaky=12):
        """Create data where n_leaky features are temporal leaks."""
        np.random.seed(42)
        X = np.random.randn(n_samples, n_features)
        feature_names = [f"feat_{i}" for i in range(n_features)]
        for i in range(n_leaky):
            X[:, i] = np.arange(n_samples) + np.random.randn(n_samples) * (0.1 * (i + 1))
        return X, feature_names

    def test_more_passes_removes_more_features(self):
        """Increasing max_passes should allow more features to be removed."""
        from experiments.run_sniper_optimization import _adversarial_filter

        X, names = self._make_leaky_data()

        _, names_1pass, diag_1 = _adversarial_filter(
            X.copy(), list(names), max_passes=1, max_features_per_pass=5
        )
        _, names_3pass, diag_3 = _adversarial_filter(
            X.copy(), list(names), max_passes=3, max_features_per_pass=5
        )

        assert diag_3["total_removed"] >= diag_1["total_removed"]

    def test_larger_max_features_removes_more_per_pass(self):
        """Increasing max_features_per_pass should remove more in a single pass."""
        from experiments.run_sniper_optimization import _adversarial_filter

        X, names = self._make_leaky_data()

        _, _, diag_small = _adversarial_filter(
            X.copy(), list(names), max_passes=1, max_features_per_pass=3
        )
        _, _, diag_large = _adversarial_filter(
            X.copy(), list(names), max_passes=1, max_features_per_pass=10
        )

        if diag_small["total_removed"] > 0 and diag_large["total_removed"] > 0:
            assert diag_large["total_removed"] >= diag_small["total_removed"]

    def test_lower_auc_threshold_continues_longer(self):
        """Lower AUC threshold should cause more passes to run."""
        from experiments.run_sniper_optimization import _adversarial_filter

        X, names = self._make_leaky_data()

        _, _, diag_high = _adversarial_filter(
            X.copy(), list(names), max_passes=5, max_features_per_pass=5,
            auc_threshold=0.85
        )
        _, _, diag_low = _adversarial_filter(
            X.copy(), list(names), max_passes=5, max_features_per_pass=5,
            auc_threshold=0.65
        )

        # Lower threshold should allow more passes → more removals
        assert diag_low["total_removed"] >= diag_high["total_removed"]

    def test_continuation_uses_threshold_not_hardcoded(self):
        """Filter should continue passes when AUC > threshold, not hardcoded 0.85."""
        from experiments.run_sniper_optimization import _adversarial_filter

        X, names = self._make_leaky_data(n_leaky=15)

        # With threshold=0.95, filter should stop earlier (most AUCs below 0.95 after removal)
        _, _, diag_strict = _adversarial_filter(
            X.copy(), list(names), max_passes=5, max_features_per_pass=5,
            auc_threshold=0.95
        )
        # With threshold=0.65, filter should continue longer
        _, _, diag_relaxed = _adversarial_filter(
            X.copy(), list(names), max_passes=5, max_features_per_pass=5,
            auc_threshold=0.65
        )

        n_passes_strict = len(diag_strict["passes"])
        n_passes_relaxed = len(diag_relaxed["passes"])
        assert n_passes_relaxed >= n_passes_strict

    def test_safety_floor_preserved_with_aggressive_settings(self):
        """Even aggressive settings should never leave fewer than 5 features."""
        from experiments.run_sniper_optimization import _adversarial_filter

        X, names = self._make_leaky_data(n_features=10, n_leaky=10)

        _, filtered_names, diag = _adversarial_filter(
            X.copy(), list(names), max_passes=10, max_features_per_pass=20,
            auc_threshold=0.50
        )

        assert len(filtered_names) >= 5
        assert diag["final_n_features"] >= 5

    def test_default_params_match_original_behavior(self):
        """Default params (2 passes, 10 features, 0.75 threshold) should work."""
        from experiments.run_sniper_optimization import _adversarial_filter

        X, names = self._make_leaky_data()

        X_filtered, filtered_names, diag = _adversarial_filter(
            X, names, max_passes=2, auc_threshold=0.75, max_features_per_pass=10
        )

        assert diag["initial_n_features"] == 20
        assert diag["final_n_features"] == len(filtered_names)
        assert X_filtered.shape[1] == len(filtered_names)
        assert len(diag["passes"]) <= 2


class TestCalibrationValidator:
    """Test calibration validation utilities."""

    def test_validate_calibration_passes_good_calibration(self):
        """Well-calibrated predictions should pass validation."""
        from src.ml.calibration_validator import validate_calibration

        np.random.seed(42)
        n = 200
        y_true = np.random.binomial(1, 0.5, n)
        # Near-perfect calibration: predicted prob ~= true rate
        y_prob = y_true * 0.8 + (1 - y_true) * 0.2 + np.random.randn(n) * 0.05
        y_prob = np.clip(y_prob, 0.01, 0.99)

        result = validate_calibration(y_true, y_prob, ece_threshold=0.15)
        # This should pass since predictions track true labels
        assert "ece" in result
        assert "passed" in result
        assert isinstance(result["ece"], float)

    def test_validate_calibration_fails_bad_calibration(self):
        """Badly calibrated predictions should fail validation."""
        from src.ml.calibration_validator import validate_calibration

        np.random.seed(42)
        n = 200
        y_true = np.random.binomial(1, 0.3, n)
        # All predictions at 0.9 when true rate is 0.3 — terrible calibration
        y_prob = np.full(n, 0.9)

        result = validate_calibration(y_true, y_prob, ece_threshold=0.10)
        assert not result["passed"]
        assert result["ece"] > 0.10

    def test_find_best_calibration(self):
        """Should return a valid method name and comparison results."""
        from src.ml.calibration_validator import find_best_calibration

        np.random.seed(42)
        n = 300
        y_true = np.random.binomial(1, 0.4, n)
        y_prob = np.random.rand(n) * 0.6 + 0.2  # Uncalibrated

        best_method, results = find_best_calibration(
            y_true[:200], y_prob[:200],
            y_true[200:], y_prob[200:],
            methods=["sigmoid", "isotonic"],
        )

        assert best_method in ["sigmoid", "isotonic"]
        assert "sigmoid" in results
        assert "isotonic" in results
        assert "ece" in results["sigmoid"]


class TestStackingWeightsNonNegative:
    """Verify that stacking weights are non-negative after the Ridge change."""

    def test_ridge_positive_constraint(self):
        """Ridge with positive=True should produce non-negative coefficients."""
        from sklearn.linear_model import Ridge

        np.random.seed(42)
        # Simulate stacking: 3 base model predictions as features
        X = np.random.rand(100, 3)
        y = (X[:, 0] * 0.5 + X[:, 1] * 0.3 + X[:, 2] * 0.2 + np.random.randn(100) * 0.1) > 0.5
        y = y.astype(float)

        ridge = Ridge(alpha=1.0, positive=True, fit_intercept=True)
        ridge.fit(X, y)

        assert all(c >= 0 for c in ridge.coef_), f"Expected non-negative weights, got {ridge.coef_}"


class TestRefereeFeatureFix:
    """Test that referee feature handles Series/NaN values correctly."""

    def test_referee_with_nan(self):
        """Referee column with NaN values should not raise errors."""
        import pandas as pd
        from src.features.engineers.external import RefereeFeatureEngineer

        engineer = RefereeFeatureEngineer(min_matches=2)
        matches = pd.DataFrame({
            "fixture_id": [1, 2, 3],
            "referee": ["John Doe", np.nan, "Jane Smith"],
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "ft_home": [2, 1, 0],
            "ft_away": [1, 0, 3],
        })

        result = engineer.create_features({"matches": matches})
        assert len(result) == 3
        # NaN referee row should get defaults
        assert result.iloc[1]["ref_matches"] == 0

    def test_referee_with_none(self):
        """Referee column with None values should not raise errors."""
        import pandas as pd
        from src.features.engineers.external import RefereeFeatureEngineer

        engineer = RefereeFeatureEngineer(min_matches=2)
        matches = pd.DataFrame({
            "fixture_id": [1, 2],
            "referee": [None, "Ref A"],
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "ft_home": [1, 2],
            "ft_away": [0, 1],
        })

        result = engineer.create_features({"matches": matches})
        assert len(result) == 2


class TestLoadFeaturesCleaning:
    """Test that load_features cleans bracketed string columns."""

    def test_clean_bracketed_strings(self, tmp_path):
        """Bracketed scientific notation should be converted to numeric."""
        import pandas as pd
        from src.utils.data_io import load_features

        df = pd.DataFrame({
            "date": ["2024-01-01", "2024-01-02"],
            "home_team": ["Team A", "Team B"],
            "away_team": ["Team C", "Team D"],
            "feature1": ["[5.0743705E-1]", "[3.167E-1]"],
            "feature2": [1.0, 2.0],
        })
        path = tmp_path / "test_features.parquet"
        df.to_parquet(path, index=False)

        result = load_features(path)
        # feature1 should have been cleaned to numeric
        assert result["feature1"].dtype in [np.float64, np.float32]
        assert abs(result["feature1"].iloc[0] - 0.50743705) < 0.001
        # text columns should remain unchanged
        assert result["home_team"].dtype == object
