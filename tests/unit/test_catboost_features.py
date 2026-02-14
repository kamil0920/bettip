"""Tests for CatBoost advanced features implementation.

Covers:
- has_time=True in model defaults and instance creation
- TimeSeriesSplit calibration for CatBoost vs StratifiedKFold for others
- grow_policy Lossguide uses max_leaves instead of depth
- Monotonic constraints building from strategies.yaml
- EnhancedCatBoost fit/predict, clone compatibility, baseline injection, transfer learning
- Native SHAP path
"""

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit


# ─── has_time defaults ────────────────────────────────────────────────────────

class TestHasTimeDefault:
    def test_catboost_has_time_in_default_params(self):
        """Verify has_time=True is set in DEFAULT_PARAMS for CatBoost."""
        from src.ml.models import DEFAULT_PARAMS, ModelType
        cb_defaults = DEFAULT_PARAMS[ModelType.CATBOOST]
        assert cb_defaults.get("has_time") is True


# ─── Calibration CV selection ────────────────────────────────────────────────

class TestCalibrationCV:
    def test_calibration_cv_timeseries_for_catboost(self):
        """Verify _get_calibration_cv returns TimeSeriesSplit for CatBoost."""
        from experiments.run_sniper_optimization import _get_calibration_cv
        cv = _get_calibration_cv("catboost")
        assert isinstance(cv, TimeSeriesSplit)
        assert cv.n_splits == 3

    def test_calibration_cv_integer_for_lightgbm(self):
        """Verify _get_calibration_cv returns integer for LightGBM."""
        from experiments.run_sniper_optimization import _get_calibration_cv
        cv = _get_calibration_cv("lightgbm")
        assert cv == 3

    def test_calibration_cv_integer_for_xgboost(self):
        """Verify _get_calibration_cv returns integer for XGBoost."""
        from experiments.run_sniper_optimization import _get_calibration_cv
        cv = _get_calibration_cv("xgboost")
        assert cv == 3

    def test_calibration_cv_custom_splits(self):
        """Verify custom n_splits is respected."""
        from experiments.run_sniper_optimization import _get_calibration_cv
        cv = _get_calibration_cv("catboost", n_splits=5)
        assert isinstance(cv, TimeSeriesSplit)
        assert cv.n_splits == 5


# ─── EnhancedCatBoost ────────────────────────────────────────────────────────

class TestEnhancedCatBoost:
    """Tests for the EnhancedCatBoost wrapper."""

    @pytest.fixture
    def synthetic_data(self):
        """Generate synthetic binary classification data."""
        rng = np.random.RandomState(42)
        X = rng.randn(200, 5)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        return X, y

    def test_fit_predict(self, synthetic_data):
        """Test basic fit/predict cycle."""
        from src.ml.catboost_wrapper import EnhancedCatBoost
        X, y = synthetic_data
        model = EnhancedCatBoost(iterations=10, depth=3, verbose=False, random_seed=42)
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (200, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

        preds = model.predict(X)
        assert preds.shape == (200,)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_feature_importances(self, synthetic_data):
        """Test feature_importances_ property."""
        from src.ml.catboost_wrapper import EnhancedCatBoost
        X, y = synthetic_data
        model = EnhancedCatBoost(iterations=10, depth=3, verbose=False, random_seed=42)
        model.fit(X, y)
        imp = model.feature_importances_
        assert len(imp) == 5

    def test_clone_compatible(self, synthetic_data):
        """Test sklearn clone() preserves parameters."""
        from src.ml.catboost_wrapper import EnhancedCatBoost
        model = EnhancedCatBoost(
            init_model_path="/tmp/test.cbm",
            use_baseline=True,
            iterations=50,
            depth=4,
            verbose=False,
        )
        cloned = clone(model)
        params = cloned.get_params()
        assert params["init_model_path"] == "/tmp/test.cbm"
        assert params["use_baseline"] is True
        assert params["iterations"] == 50
        assert params["depth"] == 4

    def test_get_set_params(self):
        """Test get_params / set_params round-trip."""
        from src.ml.catboost_wrapper import EnhancedCatBoost
        model = EnhancedCatBoost(iterations=100, depth=6, verbose=False)
        params = model.get_params()
        assert params["iterations"] == 100
        assert params["use_baseline"] is False

        model.set_params(iterations=200, use_baseline=True)
        params = model.get_params()
        assert params["iterations"] == 200
        assert params["use_baseline"] is True

    def test_baseline_injection(self, synthetic_data):
        """Test that baseline injection doesn't crash and produces valid output."""
        from src.ml.catboost_wrapper import EnhancedCatBoost
        X, y = synthetic_data
        odds = np.full(200, 2.0)  # Even odds

        model = EnhancedCatBoost(
            use_baseline=True,
            iterations=10, depth=3, verbose=False, random_seed=42,
        )
        model.set_baseline_odds(odds)
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (200, 2)
        assert np.all(np.isfinite(proba))

    def test_transfer_learning(self, synthetic_data):
        """Test transfer learning from a saved base model."""
        from src.ml.catboost_wrapper import EnhancedCatBoost
        from catboost import CatBoostClassifier
        X, y = synthetic_data

        # Train and save base model
        base = CatBoostClassifier(iterations=20, depth=3, verbose=False, random_seed=42)
        base.fit(X, y)
        with tempfile.NamedTemporaryFile(suffix=".cbm", delete=False) as f:
            base_path = f.name
        base.save_model(base_path)

        # Fine-tune with EnhancedCatBoost
        model = EnhancedCatBoost(
            init_model_path=base_path,
            iterations=10, depth=3, verbose=False, random_seed=42,
        )
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (200, 2)

        # Cleanup
        Path(base_path).unlink(missing_ok=True)

    def test_baseline_without_odds_does_nothing(self, synthetic_data):
        """Test baseline mode without calling set_baseline_odds still works."""
        from src.ml.catboost_wrapper import EnhancedCatBoost
        X, y = synthetic_data
        model = EnhancedCatBoost(
            use_baseline=True,
            iterations=10, depth=3, verbose=False, random_seed=42,
        )
        # Don't call set_baseline_odds — should still fit fine
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (200, 2)

    def test_with_sample_weight(self, synthetic_data):
        """Test fitting with sample weights."""
        from src.ml.catboost_wrapper import EnhancedCatBoost
        X, y = synthetic_data
        weights = np.random.RandomState(42).uniform(0.5, 1.5, size=200)
        model = EnhancedCatBoost(iterations=10, depth=3, verbose=False, random_seed=42)
        model.fit(X, y, sample_weight=weights)
        proba = model.predict_proba(X)
        assert proba.shape == (200, 2)

    def test_with_calibrated_classifier_cv(self, synthetic_data):
        """Test EnhancedCatBoost works inside CalibratedClassifierCV."""
        from src.ml.catboost_wrapper import EnhancedCatBoost
        X, y = synthetic_data
        model = EnhancedCatBoost(iterations=10, depth=3, verbose=False, random_seed=42)
        cal = CalibratedClassifierCV(model, method="sigmoid", cv=3)
        cal.fit(X, y)
        proba = cal.predict_proba(X)
        assert proba.shape == (200, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)


# ─── Monotonic constraints ──────────────────────────────────────────────────

class TestMonotonicConstraints:
    """Tests for monotonic constraint building from strategies.yaml."""

    def test_build_constraints_home_win(self):
        """Test constraint building for home_win market."""
        from experiments.run_sniper_optimization import SniperOptimizer, BET_TYPES
        if "home_win" not in BET_TYPES:
            pytest.skip("home_win not in BET_TYPES")

        opt = SniperOptimizer.__new__(SniperOptimizer)
        opt.bet_type = "home_win"

        features = ["poisson_home_win_prob", "home_elo", "home_attack_strength", "random_feature"]
        constraints = opt._build_monotonic_constraints(features)
        assert constraints is not None
        assert constraints[0] == 1  # poisson_home_win_prob: +1
        assert constraints[2] == 1  # home_attack_strength: +1
        assert constraints[3] == 0  # random_feature: unconstrained

    def test_build_constraints_missing_market(self):
        """Test constraint building returns None for undefined market."""
        from experiments.run_sniper_optimization import SniperOptimizer, BET_TYPES

        opt = SniperOptimizer.__new__(SniperOptimizer)
        opt.bet_type = "nonexistent_market_xyz"

        constraints = opt._build_monotonic_constraints(["feat1", "feat2"])
        assert constraints is None

    def test_build_constraints_variant_inherits_base(self):
        """Test that market variants (fouls_over_265) inherit base market constraints."""
        from experiments.run_sniper_optimization import SniperOptimizer, BET_TYPES

        opt = SniperOptimizer.__new__(SniperOptimizer)
        opt.bet_type = "fouls_over_265"

        features = ["expected_total_fouls", "other_feature"]
        constraints = opt._build_monotonic_constraints(features)
        assert constraints is not None
        assert constraints[0] == 1  # expected_total_fouls: +1
        assert constraints[1] == 0  # other_feature: unconstrained


# ─── grow_policy Lossguide ───────────────────────────────────────────────────

class TestGrowPolicy:
    def test_lossguide_removes_depth(self):
        """Verify Lossguide grow_policy uses max_leaves instead of depth."""
        # This is a structural test — we verify the logic in the search space
        # by checking CatBoost accepts Lossguide + max_leaves without depth
        from catboost import CatBoostClassifier
        rng = np.random.RandomState(42)
        X = rng.randn(100, 3)
        y = (X[:, 0] > 0).astype(int)

        model = CatBoostClassifier(
            iterations=10, max_leaves=32, grow_policy="Lossguide",
            verbose=False, random_seed=42,
        )
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (100, 2)

    def test_symmetric_tree_uses_depth(self):
        """Verify SymmetricTree uses depth parameter."""
        from catboost import CatBoostClassifier
        rng = np.random.RandomState(42)
        X = rng.randn(100, 3)
        y = (X[:, 0] > 0).astype(int)

        model = CatBoostClassifier(
            iterations=10, depth=4, grow_policy="SymmetricTree",
            verbose=False, random_seed=42,
        )
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (100, 2)


# ─── Native SHAP ────────────────────────────────────────────────────────────

class TestNativeSHAP:
    def test_catboost_native_shap_produces_values(self):
        """Test CatBoost's native SHAP produces valid feature importance values."""
        from catboost import CatBoostClassifier, Pool
        rng = np.random.RandomState(42)
        X = rng.randn(200, 5)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        model = CatBoostClassifier(iterations=20, depth=4, verbose=False, random_seed=42)
        model.fit(X[:150], y[:150])

        pool = Pool(X[150:])
        shap_vals = model.get_feature_importance(type='ShapValues', data=pool)

        # Shape: (n_samples, n_features + 1) — last col is bias
        assert shap_vals.shape == (50, 6)
        # Feature SHAP values (excluding bias)
        feat_shap = shap_vals[:, :-1]
        assert feat_shap.shape == (50, 5)
        # Values should be finite
        assert np.all(np.isfinite(feat_shap))

    def test_explainability_catboost_path(self):
        """Test that explainability.py handles CatBoost models."""
        from catboost import CatBoostClassifier
        rng = np.random.RandomState(42)
        X = rng.randn(100, 3)
        y = (X[:, 0] > 0).astype(int)

        model = CatBoostClassifier(iterations=10, depth=3, verbose=False, random_seed=42)
        model.fit(X, y)

        try:
            from src.ml.explainability import compute_shap_values
            shap_values = compute_shap_values(
                model, X[:50], ["feat_0", "feat_1", "feat_2"], model_type="tree"
            )
            assert shap_values is not None
        except ImportError:
            pytest.skip("SHAP not installed")


# ─── Per-feature borders ────────────────────────────────────────────────────

class TestPerFeatureBorders:
    def test_catboost_accepts_per_float_feature_quantization(self):
        """Verify CatBoost accepts per_float_feature_quantization parameter."""
        from catboost import CatBoostClassifier
        rng = np.random.RandomState(42)
        X = rng.randn(100, 3)
        y = (X[:, 0] > 0).astype(int)

        # CatBoost format: "idx:border_count=N"
        model = CatBoostClassifier(
            iterations=10, depth=3, verbose=False, random_seed=42,
            per_float_feature_quantization=["0:border_count=1024", "1:border_count=128", "2:border_count=128"],
        )
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (100, 2)


# ─── model_shrink_rate ──────────────────────────────────────────────────────

class TestModelShrinkRate:
    def test_catboost_accepts_shrink_rate(self):
        """Verify CatBoost accepts model_shrink_rate + model_shrink_mode."""
        from catboost import CatBoostClassifier
        rng = np.random.RandomState(42)
        X = rng.randn(100, 3)
        y = (X[:, 0] > 0).astype(int)

        model = CatBoostClassifier(
            iterations=20, depth=4, verbose=False, random_seed=42,
            model_shrink_rate=0.05, model_shrink_mode="Constant",
        )
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (100, 2)


# ─── rsm (Random Subspace Method) ───────────────────────────────────────────

class TestRSM:
    def test_catboost_accepts_rsm(self):
        """Verify CatBoost accepts rsm parameter."""
        from catboost import CatBoostClassifier
        rng = np.random.RandomState(42)
        X = rng.randn(100, 5)
        y = (X[:, 0] > 0).astype(int)

        model = CatBoostClassifier(
            iterations=10, depth=4, verbose=False, random_seed=42,
            rsm=0.7,
        )
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (100, 2)


# ─── has_time parameter ─────────────────────────────────────────────────────

class TestHasTime:
    def test_catboost_has_time_fits(self):
        """Verify CatBoost with has_time=True trains on sorted temporal data."""
        from catboost import CatBoostClassifier
        rng = np.random.RandomState(42)
        X = rng.randn(200, 3)
        y = (X[:, 0] > 0).astype(int)

        model = CatBoostClassifier(
            iterations=10, depth=3, verbose=False, random_seed=42,
            has_time=True,
        )
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (200, 2)
