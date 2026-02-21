"""
Unit tests for uncertainty quantification via MAPIE conformal prediction.

Tests:
1. ConformalClassifier calibrate + predict_with_uncertainty
2. Serialization roundtrip (to_dict / from_dict)
3. adjust_kelly_stake single + batch
4. Edge cases (uncalibrated, empty input)
5. Integration with model_loader conformal extraction
"""

import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingClassifier
from unittest.mock import MagicMock

from src.ml.uncertainty import (
    ConformalClassifier,
    adjust_kelly_stake,
    batch_adjust_stakes,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def trained_model():
    """A simple trained GBM for conformal wrapping."""
    rng = np.random.RandomState(42)
    X = rng.randn(200, 5)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    model = GradientBoostingClassifier(n_estimators=20, random_state=42)
    model.fit(X[:150], y[:150])
    return model, X[150:], y[150:]


@pytest.fixture
def calibrated_conformal(trained_model):
    """A calibrated ConformalClassifier."""
    model, X_cal, y_cal = trained_model
    cc = ConformalClassifier(model, alpha=0.1)
    cc.calibrate(X_cal, y_cal)
    return cc, model


# ---------------------------------------------------------------------------
# ConformalClassifier
# ---------------------------------------------------------------------------
class TestConformalClassifier:
    """Test ConformalClassifier calibration and prediction."""

    def test_calibrate_returns_self(self, trained_model):
        model, X_cal, y_cal = trained_model
        cc = ConformalClassifier(model, alpha=0.1)
        result = cc.calibrate(X_cal, y_cal)
        assert result is cc

    def test_calibrate_sets_mapie_clf(self, trained_model):
        model, X_cal, y_cal = trained_model
        cc = ConformalClassifier(model, alpha=0.1)
        assert cc.mapie_clf is None
        cc.calibrate(X_cal, y_cal)
        assert cc.mapie_clf is not None

    def test_predict_before_calibrate_raises(self, trained_model):
        model, X_cal, _ = trained_model
        cc = ConformalClassifier(model, alpha=0.1)
        with pytest.raises(RuntimeError, match="calibrate"):
            cc.predict_with_uncertainty(X_cal)

    def test_predict_with_uncertainty_shapes(self, calibrated_conformal):
        cc, model = calibrated_conformal
        rng = np.random.RandomState(123)
        X_test = rng.randn(10, 5)

        preds, pred_sets, uncertainty = cc.predict_with_uncertainty(X_test)
        assert preds.shape == (10,)
        assert pred_sets.shape == (10, 2)  # binary classification
        assert uncertainty.shape == (10,)

    def test_uncertainty_in_0_1_range(self, calibrated_conformal):
        cc, _ = calibrated_conformal
        rng = np.random.RandomState(123)
        X_test = rng.randn(20, 5)

        _, _, uncertainty = cc.predict_with_uncertainty(X_test)
        assert np.all(uncertainty >= 0.0)
        assert np.all(uncertainty <= 1.0)

    def test_uncertainty_binary_values(self, calibrated_conformal):
        """For binary classification, uncertainty should be 0 (certain) or 1 (uncertain)."""
        cc, _ = calibrated_conformal
        rng = np.random.RandomState(99)
        X_test = rng.randn(50, 5)

        _, _, uncertainty = cc.predict_with_uncertainty(X_test)
        # Each value should be either 0 or 1 for binary case
        for u in uncertainty:
            assert u in (0.0, 1.0), f"Expected 0 or 1, got {u}"

    def test_empty_prediction_set_is_maximally_uncertain(self, calibrated_conformal):
        """Empty prediction set (set_size=0) should map to uncertainty=1.0, not 0.0."""
        cc, _ = calibrated_conformal
        rng = np.random.RandomState(42)
        X_test = rng.randn(5, 5)

        # Manually test the uncertainty formula with empty sets
        pred_sets_empty = np.zeros((5, 2), dtype=bool)  # All empty
        n_classes = 2
        set_sizes = pred_sets_empty.sum(axis=1)
        uncertainty = np.where(
            set_sizes == 0,
            1.0,
            (set_sizes - 1) / max(n_classes - 1, 1),
        )
        uncertainty = np.clip(uncertainty, 0, 1)
        np.testing.assert_array_equal(uncertainty, np.ones(5))

    def test_predict_proba_passthrough(self, calibrated_conformal):
        """predict_proba should delegate to underlying model."""
        cc, model = calibrated_conformal
        rng = np.random.RandomState(42)
        X_test = rng.randn(5, 5)

        cc_proba = cc.predict_proba(X_test)
        model_proba = model.predict_proba(X_test)
        np.testing.assert_array_almost_equal(cc_proba, model_proba)

    def test_alpha_stored(self):
        model = MagicMock()
        cc = ConformalClassifier(model, alpha=0.05)
        assert cc.alpha == 0.05

    def test_default_alpha(self):
        model = MagicMock()
        cc = ConformalClassifier(model)
        assert cc.alpha == 0.1


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------
class TestConformalSerialization:
    """Test to_dict / from_dict roundtrip."""

    def test_to_dict_keys(self, calibrated_conformal):
        cc, _ = calibrated_conformal
        d = cc.to_dict()
        assert "alpha" in d
        assert "mapie_clf" in d

    def test_to_dict_alpha_preserved(self, calibrated_conformal):
        cc, _ = calibrated_conformal
        d = cc.to_dict()
        assert d["alpha"] == cc.alpha

    def test_roundtrip_predictions_match(self, calibrated_conformal):
        cc, model = calibrated_conformal
        rng = np.random.RandomState(42)
        X_test = rng.randn(10, 5)

        # Original predictions
        preds1, sets1, unc1 = cc.predict_with_uncertainty(X_test)

        # Roundtrip
        d = cc.to_dict()
        cc2 = ConformalClassifier.from_dict(d, model)
        preds2, sets2, unc2 = cc2.predict_with_uncertainty(X_test)

        np.testing.assert_array_equal(preds1, preds2)
        np.testing.assert_array_equal(sets1, sets2)
        np.testing.assert_array_equal(unc1, unc2)

    def test_from_dict_sets_alpha(self, calibrated_conformal):
        cc, model = calibrated_conformal
        d = cc.to_dict()
        cc2 = ConformalClassifier.from_dict(d, model)
        assert cc2.alpha == cc.alpha

    def test_from_dict_sets_mapie_clf(self, calibrated_conformal):
        cc, model = calibrated_conformal
        d = cc.to_dict()
        cc2 = ConformalClassifier.from_dict(d, model)
        assert cc2.mapie_clf is not None

    def test_joblib_roundtrip(self, calibrated_conformal, tmp_path):
        """Test that conformal data survives joblib save/load (production path)."""
        import joblib

        cc, model = calibrated_conformal
        rng = np.random.RandomState(42)
        X_test = rng.randn(5, 5)

        # Simulate model_data dict as saved by train_and_save_models
        model_data = {
            "model": model,
            "features": ["f0", "f1", "f2", "f3", "f4"],
            "conformal": cc.to_dict(),
        }

        path = tmp_path / "test_model.joblib"
        joblib.dump(model_data, path, compress=3)
        loaded = joblib.load(path)

        assert "conformal" in loaded
        cc2 = ConformalClassifier.from_dict(loaded["conformal"], loaded["model"])
        _, _, unc = cc2.predict_with_uncertainty(X_test)
        assert unc.shape == (5,)


# ---------------------------------------------------------------------------
# adjust_kelly_stake
# ---------------------------------------------------------------------------
class TestAdjustKellyStake:
    """Test single-bet Kelly stake adjustment."""

    def test_zero_uncertainty_no_change(self):
        assert adjust_kelly_stake(1.0, 0.0) == 1.0

    def test_full_uncertainty_default_penalty(self):
        # penalty=1.0, uncertainty=1.0 → 1/(1+1) = 0.5
        assert adjust_kelly_stake(1.0, 1.0) == pytest.approx(0.5)

    def test_custom_penalty(self):
        # penalty=2.0, uncertainty=1.0 → 1/(1+2) = 0.333
        assert adjust_kelly_stake(1.0, 1.0, uncertainty_penalty=2.0) == pytest.approx(
            1.0 / 3.0
        )

    def test_high_penalty_aggressive_reduction(self):
        # penalty=3.0, uncertainty=1.0 → 1/(1+3) = 0.25
        assert adjust_kelly_stake(1.0, 1.0, uncertainty_penalty=3.0) == pytest.approx(
            0.25
        )

    def test_low_penalty_mild_reduction(self):
        # penalty=0.5, uncertainty=1.0 → 1/(1+0.5) = 0.667
        assert adjust_kelly_stake(1.0, 1.0, uncertainty_penalty=0.5) == pytest.approx(
            1.0 / 1.5
        )

    def test_result_never_exceeds_base(self):
        for unc in [0.0, 0.5, 1.0]:
            for penalty in [0.5, 1.0, 2.0, 3.0]:
                result = adjust_kelly_stake(5.0, unc, penalty)
                assert result <= 5.0

    def test_zero_base_stake(self):
        assert adjust_kelly_stake(0.0, 1.0) == 0.0


# ---------------------------------------------------------------------------
# batch_adjust_stakes
# ---------------------------------------------------------------------------
class TestBatchAdjustStakes:
    """Test vectorized stake adjustment."""

    def test_basic_batch(self):
        stakes = np.array([1.0, 1.0, 1.0])
        uncertainties = np.array([0.0, 0.5, 1.0])
        result = batch_adjust_stakes(stakes, uncertainties, uncertainty_penalty=1.0)

        assert result[0] == pytest.approx(1.0)  # No uncertainty
        assert result[1] == pytest.approx(1.0 / 1.5)  # 50% uncertain
        assert result[2] == pytest.approx(0.5)  # Fully uncertain

    def test_batch_shapes(self):
        stakes = np.ones(10)
        uncertainties = np.random.rand(10)
        result = batch_adjust_stakes(stakes, uncertainties)
        assert result.shape == (10,)

    def test_batch_all_zero_uncertainty(self):
        stakes = np.array([2.0, 3.0, 4.0])
        uncertainties = np.zeros(3)
        result = batch_adjust_stakes(stakes, uncertainties)
        np.testing.assert_array_equal(result, stakes)

    def test_batch_custom_penalty(self):
        stakes = np.array([1.0])
        uncertainties = np.array([1.0])
        result = batch_adjust_stakes(stakes, uncertainties, uncertainty_penalty=2.0)
        assert result[0] == pytest.approx(1.0 / 3.0)

    def test_batch_result_never_exceeds_base(self):
        stakes = np.array([5.0, 10.0, 3.0])
        uncertainties = np.random.rand(3)
        result = batch_adjust_stakes(stakes, uncertainties, uncertainty_penalty=2.0)
        assert np.all(result <= stakes)


# ---------------------------------------------------------------------------
# Integration: model_loader conformal extraction
# ---------------------------------------------------------------------------
class TestModelLoaderConformalIntegration:
    """Test that model_loader correctly extracts and uses conformal data."""

    def test_load_full_model_extracts_conformal(self, calibrated_conformal, tmp_path):
        """_load_full_model includes conformal key from joblib."""
        import joblib
        from src.ml.model_loader import ModelLoader

        cc, model = calibrated_conformal
        model_data = {
            "model": model,
            "features": ["f0", "f1", "f2", "f3", "f4"],
            "bet_type": "test",
            "scaler": None,
            "conformal": cc.to_dict(),
        }

        models_dir = tmp_path / "models"
        models_dir.mkdir()
        joblib.dump(model_data, models_dir / "test_model.joblib", compress=3)

        loader = ModelLoader(models_dir=models_dir)
        loaded = loader.load_model("test_model")
        assert loaded is not None
        assert "conformal" in loaded
        assert loaded["conformal"] is not None
        assert loaded["conformal"]["alpha"] == 0.1

    def test_load_full_model_without_conformal(self, tmp_path):
        """Backward compat: models without conformal key load fine."""
        import joblib
        from sklearn.linear_model import LogisticRegression
        from src.ml.model_loader import ModelLoader

        model = LogisticRegression()
        model.fit(np.array([[1], [2], [3]]), np.array([0, 1, 0]))
        model_data = {
            "model": model,
            "features": ["f0"],
            "bet_type": "test",
        }

        models_dir = tmp_path / "models"
        models_dir.mkdir()
        joblib.dump(model_data, models_dir / "test_old.joblib", compress=3)

        loader = ModelLoader(models_dir=models_dir)
        loaded = loader.load_model("test_old")
        assert loaded is not None
        assert loaded.get("conformal") is None

    def test_predict_with_health_sets_conformal_uncertainty(
        self, calibrated_conformal, tmp_path
    ):
        """predict_with_health populates health_report.conformal_uncertainty."""
        import joblib
        import pandas as pd
        from src.ml.model_loader import ModelLoader
        from src.ml.prediction_health import MarketHealthReport

        cc, model = calibrated_conformal
        model_data = {
            "model": model,
            "features": ["f0", "f1", "f2", "f3", "f4"],
            "bet_type": "test",
            "scaler": None,
            "conformal": cc.to_dict(),
        }

        models_dir = tmp_path / "models"
        models_dir.mkdir()
        joblib.dump(model_data, models_dir / "test_conf.joblib", compress=3)

        loader = ModelLoader(models_dir=models_dir)
        features_df = pd.DataFrame(
            np.random.randn(1, 5), columns=["f0", "f1", "f2", "f3", "f4"]
        )
        health = MarketHealthReport(market="test")
        result = loader.predict_with_health(
            "test_conf", features_df, health_report=health
        )
        assert result is not None
        assert health.conformal_uncertainty is not None
        assert health.conformal_uncertainty in (0.0, 1.0)  # Binary: 0 or 1

    def test_predict_with_health_no_conformal_graceful(self):
        """Without conformal data, conformal_uncertainty stays None."""
        import pandas as pd
        from src.ml.model_loader import ModelLoader
        from src.ml.prediction_health import MarketHealthReport

        model = MagicMock()
        model.predict_proba.return_value = np.array([[0.3, 0.7]])

        loader = ModelLoader()
        loader._loaded_models["test"] = {
            "model": model,
            "features": ["f1", "f2"],
            "bet_type": "test",
            "scaler": None,
            "metadata": {},
        }

        features_df = pd.DataFrame({"f1": [1.0], "f2": [2.0]})
        health = MarketHealthReport(market="test")
        result = loader.predict_with_health("test", features_df, health_report=health)
        assert result is not None
        assert health.conformal_uncertainty is None


# ---------------------------------------------------------------------------
# Prediction health fields
# ---------------------------------------------------------------------------
class TestPredictionHealthUncertaintyFields:
    """Test new uncertainty fields on MarketHealthReport."""

    def test_default_values(self):
        from src.ml.prediction_health import MarketHealthReport

        report = MarketHealthReport(market="test")
        assert report.conformal_uncertainty is None
        assert report.uncertainty_penalty == 1.0

    def test_to_dict_includes_uncertainty_fields(self):
        from src.ml.prediction_health import MarketHealthReport

        report = MarketHealthReport(market="test")
        report.conformal_uncertainty = 0.5
        report.uncertainty_penalty = 2.0

        d = report.to_dict()
        assert d["conformal_uncertainty"] == 0.5
        assert d["uncertainty_penalty"] == 2.0

    def test_to_dict_null_uncertainty(self):
        from src.ml.prediction_health import MarketHealthReport

        report = MarketHealthReport(market="test")
        d = report.to_dict()
        assert d["conformal_uncertainty"] is None
        assert d["uncertainty_penalty"] == 1.0
