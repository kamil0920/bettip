"""
Unit tests for prediction health reporting module.

Tests all degradation paths:
1. Feature mismatch severity classification (<2%, 2-5%, >5%)
2. Calibration collapse detection and threshold multiplier
3. Two-stage model fallback when odds unavailable
4. Model mismatch auto-resolution (skip, don't crash)
5. Summary JSON output with per-market status
6. HealthTracker accumulation and finalization
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.ml.prediction_health import (
    MEDIUM_MISMATCH_CONFIDENCE_PENALTY,
    UNCALIBRATED_THRESHOLD_MULTIPLIER,
    CalibrationStatus,
    FeatureMismatchSeverity,
    HealthTracker,
    MarketHealthReport,
    MarketStatus,
    classify_feature_mismatch,
)


# ---------------------------------------------------------------------------
# classify_feature_mismatch
# ---------------------------------------------------------------------------
class TestClassifyFeatureMismatch:
    """Test feature mismatch severity classification."""

    def test_no_missing_features(self):
        assert classify_feature_mismatch(30, 0) == FeatureMismatchSeverity.NONE

    def test_low_severity_below_2pct(self):
        # 1 of 100 = 1%
        assert classify_feature_mismatch(100, 1) == FeatureMismatchSeverity.LOW

    def test_low_severity_just_below_2pct(self):
        # 1 of 51 ~ 1.96% < 2%
        assert classify_feature_mismatch(51, 1) == FeatureMismatchSeverity.LOW

    def test_medium_severity_at_2pct(self):
        # 2 of 100 = 2%
        assert classify_feature_mismatch(100, 2) == FeatureMismatchSeverity.MEDIUM

    def test_medium_severity_at_5pct(self):
        # 5 of 100 = 5%
        assert classify_feature_mismatch(100, 5) == FeatureMismatchSeverity.MEDIUM

    def test_high_severity_above_5pct(self):
        # 6 of 100 = 6%
        assert classify_feature_mismatch(100, 6) == FeatureMismatchSeverity.HIGH

    def test_high_severity_10pct(self):
        # 10 of 100 = 10%
        assert classify_feature_mismatch(100, 10) == FeatureMismatchSeverity.HIGH

    def test_zero_expected_features(self):
        assert classify_feature_mismatch(0, 5) == FeatureMismatchSeverity.NONE

    def test_negative_missing(self):
        assert classify_feature_mismatch(30, -1) == FeatureMismatchSeverity.NONE


# ---------------------------------------------------------------------------
# MarketHealthReport
# ---------------------------------------------------------------------------
class TestMarketHealthReport:
    """Test per-market health report."""

    def test_default_status_is_ok(self):
        report = MarketHealthReport(market="cards_under_35")
        assert report.status == MarketStatus.OK
        assert report.threshold_multiplier == 1.0
        assert report.confidence_penalty == 1.0

    def test_record_models_loaded(self):
        report = MarketHealthReport(market="over25")
        report.record_models_loaded(
            loaded=["over25_xgboost", "over25_lightgbm"],
            missing=["over25_catboost"],
        )
        assert report.models_loaded == [
            "over25_xgboost",
            "over25_lightgbm",
        ]
        assert report.models_missing == ["over25_catboost"]
        assert len(report.warnings) == 1
        assert "over25_catboost" in report.warnings[0]

    def test_record_models_loaded_no_missing(self):
        report = MarketHealthReport(market="btts")
        report.record_models_loaded(loaded=["btts_xgboost"])
        assert report.models_loaded == ["btts_xgboost"]
        assert report.models_missing == []
        assert len(report.warnings) == 0

    # --- Feature mismatch ---
    def test_feature_match_low_severity(self):
        report = MarketHealthReport(market="cards_under_35")
        report.record_feature_match(expected=100, missing=1, missing_names=["feat_x"])
        assert report.feature_mismatch_severity == FeatureMismatchSeverity.LOW
        assert report.status == MarketStatus.OK
        assert report.confidence_penalty == 1.0
        assert len(report.warnings) == 1
        assert "proceed with warning" in report.warnings[0]

    def test_feature_match_medium_degrades_confidence(self):
        report = MarketHealthReport(market="cards_under_35")
        report.record_feature_match(expected=100, missing=3)
        assert report.feature_mismatch_severity == FeatureMismatchSeverity.MEDIUM
        assert report.status == MarketStatus.DEGRADED
        assert report.confidence_penalty == MEDIUM_MISMATCH_CONFIDENCE_PENALTY
        assert "confidence degraded" in report.warnings[0]

    def test_feature_match_high_skips_market(self):
        report = MarketHealthReport(market="cards_under_35")
        report.record_feature_match(
            expected=30,
            missing=5,
            missing_names=["a", "b", "c", "d", "e"],
        )
        assert report.feature_mismatch_severity == FeatureMismatchSeverity.HIGH
        assert report.status == MarketStatus.SKIPPED
        assert report.skip_reason is not None
        assert "> 5%" in report.skip_reason

    def test_feature_match_truncates_names(self):
        names = [f"feat_{i}" for i in range(20)]
        report = MarketHealthReport(market="x")
        report.record_feature_match(expected=100, missing=20, missing_names=names)
        assert len(report.missing_feature_names) == 10

    # --- Calibration ---
    def test_record_calibrated(self):
        report = MarketHealthReport(market="btts")
        report.record_calibration(CalibrationStatus.CALIBRATED)
        assert report.calibration_status == CalibrationStatus.CALIBRATED
        assert report.threshold_multiplier == 1.0
        assert report.status == MarketStatus.OK

    def test_record_uncalibrated_raises_threshold(self):
        report = MarketHealthReport(market="btts")
        report.record_calibration(CalibrationStatus.UNCALIBRATED)
        assert report.calibration_status == CalibrationStatus.UNCALIBRATED
        assert report.threshold_multiplier == UNCALIBRATED_THRESHOLD_MULTIPLIER
        assert report.status == MarketStatus.DEGRADED
        assert any("threshold multiplier" in w for w in report.warnings)

    def test_uncalibrated_does_not_unskip(self):
        """If market already SKIPPED, uncalibrated shouldn't change to DEGRADED."""
        report = MarketHealthReport(market="btts")
        report.record_skip("test skip")
        report.record_calibration(CalibrationStatus.UNCALIBRATED)
        assert report.status == MarketStatus.SKIPPED

    # --- Odds ---
    def test_record_real_odds(self):
        report = MarketHealthReport(market="over25")
        report.record_odds("real", odds_value=1.85)
        assert report.odds_source == "real"
        assert report.odds_value == 1.85

    def test_record_baseline_odds(self):
        report = MarketHealthReport(market="over25")
        report.record_odds("baseline")
        assert report.odds_source == "baseline"
        assert report.odds_value is None

    # --- Two-stage fallback ---
    def test_two_stage_fallback(self):
        report = MarketHealthReport(market="home_win")
        report.record_two_stage_fallback()
        assert report.two_stage_fallback is True
        assert report.status == MarketStatus.DEGRADED
        assert any("Stage 1" in w for w in report.warnings)

    def test_two_stage_fallback_does_not_unskip(self):
        report = MarketHealthReport(market="home_win")
        report.record_skip("no features")
        report.record_two_stage_fallback()
        assert report.status == MarketStatus.SKIPPED

    # --- Skip ---
    def test_record_skip(self):
        report = MarketHealthReport(market="fouls")
        report.record_skip("Model file missing")
        assert report.status == MarketStatus.SKIPPED
        assert report.skip_reason == "Model file missing"

    # --- Serialization ---
    def test_to_dict_structure(self):
        report = MarketHealthReport(market="cards_under_35")
        report.record_models_loaded(["m1"], missing=["m2"])
        # 1/100 = 1% -> LOW severity -> status stays OK
        report.record_feature_match(expected=100, missing=1)
        report.record_calibration(CalibrationStatus.CALIBRATED)
        report.record_odds("real", 1.95)

        d = report.to_dict()
        assert d["market"] == "cards_under_35"
        assert d["status"] == "ok"
        assert d["models_loaded"] == ["m1"]
        assert d["models_missing"] == ["m2"]
        assert d["features"]["expected"] == 100
        assert d["features"]["missing"] == 1
        assert d["features"]["severity"] == "low"
        assert d["calibration_status"] == "calibrated"
        assert d["odds_source"] == "real"
        assert d["odds_value"] == 1.95
        assert d["two_stage_fallback"] is False


# ---------------------------------------------------------------------------
# HealthTracker
# ---------------------------------------------------------------------------
class TestHealthTracker:
    """Test health tracker accumulation and summary."""

    def test_create_and_finalize(self):
        tracker = HealthTracker()
        report = tracker.create_market_report("btts")
        report.record_models_loaded(["btts_xgb"])
        tracker.finalize(report)
        assert tracker.get_report("btts") is not None
        assert tracker.get_report("btts").models_loaded == ["btts_xgb"]

    def test_summary_counts(self):
        tracker = HealthTracker()

        r1 = tracker.create_market_report("cards_under_35")
        tracker.finalize(r1)

        r2 = tracker.create_market_report("btts")
        r2.record_calibration(CalibrationStatus.UNCALIBRATED)
        tracker.finalize(r2)

        r3 = tracker.create_market_report("fouls")
        r3.record_skip("No models")
        tracker.finalize(r3)

        summary = tracker.summary()
        assert summary["total_markets"] == 3
        assert summary["markets_ok"] == 1
        assert summary["markets_degraded"] == 1
        assert summary["markets_skipped"] == 1

    def test_summary_has_run_timestamp(self):
        tracker = HealthTracker()
        summary = tracker.summary()
        assert "run_timestamp" in summary

    def test_summary_has_schema_version(self):
        tracker = HealthTracker()
        summary = tracker.summary()
        assert summary["schema_version"] == "1.0"

    def test_global_warnings(self):
        tracker = HealthTracker()
        tracker.add_global_warning("No odds file")
        summary = tracker.summary()
        assert "No odds file" in summary["global_warnings"]

    def test_get_report_missing(self):
        tracker = HealthTracker()
        assert tracker.get_report("nonexistent") is None

    def test_write_summary_creates_file(self):
        tracker = HealthTracker()
        r = tracker.create_market_report("test_market")
        r.record_models_loaded(["model_a"])
        tracker.finalize(r)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "subdir" / "health.json"
            tracker.write_summary(out_path)
            assert out_path.exists()

            data = json.loads(out_path.read_text())
            assert data["total_markets"] == 1
            assert "test_market" in data["markets"]

    def test_finalize_duplicate_market_merges_warnings(self):
        tracker = HealthTracker()

        r1 = tracker.create_market_report("btts")
        r1.warnings.append("warn1")
        tracker.finalize(r1)

        r2 = tracker.create_market_report("btts")
        r2.warnings.append("warn2")
        tracker.finalize(r2)

        report = tracker.get_report("btts")
        assert "warn1" in report.warnings
        assert "warn2" in report.warnings

    def test_finalize_duplicate_market_escalates_status(self):
        """Bug regression: finalize must keep the WORSE status."""
        tracker = HealthTracker()

        r1 = tracker.create_market_report("btts")
        # Status stays OK
        tracker.finalize(r1)
        assert tracker.get_report("btts").status == MarketStatus.OK

        r2 = tracker.create_market_report("btts")
        r2.record_calibration(CalibrationStatus.UNCALIBRATED)
        # Status is now DEGRADED on the new report
        tracker.finalize(r2)
        assert tracker.get_report("btts").status == MarketStatus.DEGRADED

    def test_finalize_duplicate_market_escalates_to_skipped(self):
        """Finalizing a SKIPPED report after an OK one escalates to SKIPPED."""
        tracker = HealthTracker()

        r1 = tracker.create_market_report("btts")
        tracker.finalize(r1)

        r2 = tracker.create_market_report("btts")
        r2.record_skip("Model crashed")
        tracker.finalize(r2)
        assert tracker.get_report("btts").status == MarketStatus.SKIPPED
        assert tracker.get_report("btts").skip_reason == "Model crashed"

    def test_finalize_duplicate_does_not_downgrade(self):
        """Finalizing OK report after DEGRADED should not downgrade."""
        tracker = HealthTracker()

        r1 = tracker.create_market_report("btts")
        r1.record_calibration(CalibrationStatus.UNCALIBRATED)
        tracker.finalize(r1)
        assert tracker.get_report("btts").status == MarketStatus.DEGRADED

        r2 = tracker.create_market_report("btts")
        # r2 status is OK
        tracker.finalize(r2)
        # Should stay DEGRADED, not downgrade to OK
        assert tracker.get_report("btts").status == MarketStatus.DEGRADED

    def test_summary_markets_sorted(self):
        tracker = HealthTracker()
        for m in ["fouls", "btts", "cards"]:
            r = tracker.create_market_report(m)
            tracker.finalize(r)
        summary = tracker.summary()
        assert list(summary["markets"].keys()) == ["btts", "cards", "fouls"]


# ---------------------------------------------------------------------------
# ModelLoader.predict_with_health
# ---------------------------------------------------------------------------
class TestPredictWithHealth:
    """Test ModelLoader.predict_with_health for all degradation paths."""

    def _make_model_data(
        self,
        features=None,
        model=None,
        scaler=None,
    ):
        """Helper: create model data dict."""
        if features is None:
            features = ["f1", "f2", "f3"]
        if model is None:
            model = MagicMock()
            model.predict_proba.return_value = np.array([[0.3, 0.7]])
        return {
            "model": model,
            "features": features,
            "bet_type": "test",
            "scaler": scaler,
            "metadata": {},
        }

    def _make_features_df(self, cols=None):
        """Helper: create features DataFrame."""
        if cols is None:
            cols = ["f1", "f2", "f3"]
        return pd.DataFrame({c: [1.0] for c in cols})

    def test_basic_prediction(self):
        from src.ml.model_loader import ModelLoader

        loader = ModelLoader()
        model_data = self._make_model_data()
        loader._loaded_models["test_model"] = model_data

        health = MarketHealthReport(market="test")
        result = loader.predict_with_health(
            "test_model",
            self._make_features_df(),
            health_report=health,
        )
        assert result is not None
        prob, conf = result
        assert 0.0 < prob < 1.0
        assert health.status == MarketStatus.OK

    def test_model_not_found_skips(self):
        from src.ml.model_loader import ModelLoader

        loader = ModelLoader(models_dir=Path("/nonexistent"))
        health = MarketHealthReport(market="test")
        result = loader.predict_with_health(
            "missing_model",
            self._make_features_df(),
            health_report=health,
        )
        assert result is None
        assert health.status == MarketStatus.SKIPPED
        assert "failed to load" in health.skip_reason

    def test_feature_mismatch_low(self):
        """<2% missing: proceed with warning, no confidence penalty."""
        from src.ml.model_loader import ModelLoader

        features = [f"f{i}" for i in range(100)]
        model = MagicMock()
        model.predict_proba.return_value = np.array([[0.3, 0.7]])
        model_data = self._make_model_data(features=features, model=model)

        loader = ModelLoader()
        loader._loaded_models["test_model"] = model_data

        # Provide 99 of 100 features
        df = self._make_features_df(cols=features[:99])
        health = MarketHealthReport(market="test")
        result = loader.predict_with_health("test_model", df, health_report=health)
        assert result is not None
        assert health.feature_mismatch_severity == FeatureMismatchSeverity.LOW
        assert health.status != MarketStatus.SKIPPED
        assert health.confidence_penalty == 1.0

    def test_feature_mismatch_medium_degrades(self):
        """2-5% missing: degrade confidence."""
        from src.ml.model_loader import ModelLoader

        features = [f"f{i}" for i in range(100)]
        model = MagicMock()
        model.predict_proba.return_value = np.array([[0.3, 0.7]])
        model_data = self._make_model_data(features=features, model=model)

        loader = ModelLoader()
        loader._loaded_models["test_model"] = model_data

        # Provide 97 of 100 features (3% missing)
        df = self._make_features_df(cols=features[:97])
        health = MarketHealthReport(market="test")
        result = loader.predict_with_health("test_model", df, health_report=health)
        assert result is not None
        assert health.feature_mismatch_severity == FeatureMismatchSeverity.MEDIUM
        assert health.status == MarketStatus.DEGRADED
        assert health.confidence_penalty == MEDIUM_MISMATCH_CONFIDENCE_PENALTY
        # Confidence should be penalized
        _, conf = result
        expected_raw_conf = abs(0.7 - 0.5) * 2
        assert conf == pytest.approx(expected_raw_conf * MEDIUM_MISMATCH_CONFIDENCE_PENALTY)

    def test_feature_mismatch_high_skips(self):
        """>5% missing: skip market."""
        from src.ml.model_loader import ModelLoader

        features = [f"f{i}" for i in range(30)]
        model = MagicMock()
        model.predict_proba.return_value = np.array([[0.3, 0.7]])
        model_data = self._make_model_data(features=features, model=model)

        loader = ModelLoader()
        loader._loaded_models["test_model"] = model_data

        # Provide only 25 of 30 features (16.7% missing)
        df = self._make_features_df(cols=features[:25])
        health = MarketHealthReport(market="test")
        result = loader.predict_with_health("test_model", df, health_report=health)
        assert result is None
        assert health.feature_mismatch_severity == FeatureMismatchSeverity.HIGH

    def test_calibration_collapse_sets_uncalibrated(self):
        """Degenerate calibration -> UNCALIBRATED status + threshold multiplier."""
        from src.ml.model_loader import ModelLoader

        # Create a model with degenerate isotonic calibration
        mock_cal = MagicMock()
        mock_cal.X_thresholds_ = np.array([0.1, 0.3, 0.45])
        mock_cal.y_thresholds_ = np.array([0.2, 0.5, 0.95])

        mock_cc = MagicMock()
        mock_cc.calibrators = [mock_cal]
        # Need a working base estimator
        base_est = MagicMock()
        base_est.predict_proba.return_value = np.array([[0.3, 0.7]])
        mock_cc.estimator = base_est

        model = MagicMock()
        model.calibrated_classifiers_ = [mock_cc, mock_cc, mock_cc]

        model_data = self._make_model_data(model=model)
        loader = ModelLoader()
        loader._loaded_models["test_model"] = model_data

        health = MarketHealthReport(market="test")
        result = loader.predict_with_health(
            "test_model",
            self._make_features_df(),
            health_report=health,
        )
        assert health.calibration_status == CalibrationStatus.UNCALIBRATED
        assert health.threshold_multiplier == UNCALIBRATED_THRESHOLD_MULTIPLIER
        assert health.status == MarketStatus.DEGRADED

    def test_two_stage_fallback_when_no_odds(self):
        """Two-stage model falls back to Stage 1 when odds=None."""
        from src.ml.model_loader import ModelLoader

        # Create mock TwoStageModel
        model = MagicMock()
        type(model).__name__ = "TwoStageModel"
        model._stage1_scaler = MagicMock()
        model._stage1_scaler.transform.return_value = np.array([[1.0, 2.0, 3.0]])
        model._get_stage1_proba.return_value = np.array([0.65])

        model_data = self._make_model_data(model=model)
        loader = ModelLoader()
        loader._loaded_models["test_model"] = model_data

        health = MarketHealthReport(market="test")
        result = loader.predict_with_health(
            "test_model",
            self._make_features_df(),
            health_report=health,
            odds=None,
        )
        assert result is not None
        prob, conf = result
        assert prob == pytest.approx(0.65)
        assert health.two_stage_fallback is True
        assert health.status == MarketStatus.DEGRADED

    def test_two_stage_with_odds_no_fallback(self):
        """Two-stage model uses both stages when odds available."""
        from src.ml.model_loader import ModelLoader

        model = MagicMock()
        type(model).__name__ = "TwoStageModel"
        model.predict_proba.return_value = {
            "combined_score": np.array([0.72]),
            "outcome_prob": np.array([0.75]),
            "profit_prob": np.array([0.68]),
            "edge": np.array([0.05]),
        }

        model_data = self._make_model_data(model=model)
        loader = ModelLoader()
        loader._loaded_models["test_model"] = model_data

        health = MarketHealthReport(market="test")
        result = loader.predict_with_health(
            "test_model",
            self._make_features_df(),
            health_report=health,
            odds=2.5,
        )
        assert result is not None
        prob, conf = result
        assert prob == pytest.approx(0.72)
        assert health.two_stage_fallback is False

    def test_degenerate_probability_skips(self):
        """Probabilities >= 0.99 or <= 0.01 skip."""
        from src.ml.model_loader import ModelLoader

        model = MagicMock()
        model.predict_proba.return_value = np.array([[0.005, 0.995]])

        model_data = self._make_model_data(model=model)
        loader = ModelLoader()
        loader._loaded_models["test_model"] = model_data

        health = MarketHealthReport(market="test")
        result = loader.predict_with_health(
            "test_model",
            self._make_features_df(),
            health_report=health,
        )
        assert result is None
        assert health.status == MarketStatus.SKIPPED
        assert "Degenerate" in health.skip_reason

    def test_prediction_exception_records_skip(self):
        """Model.predict_proba raising should record skip, not crash."""
        from src.ml.model_loader import ModelLoader

        model = MagicMock()
        model.predict_proba.side_effect = RuntimeError("GPU OOM")

        model_data = self._make_model_data(model=model)
        loader = ModelLoader()
        loader._loaded_models["test_model"] = model_data

        health = MarketHealthReport(market="test")
        result = loader.predict_with_health(
            "test_model",
            self._make_features_df(),
            health_report=health,
        )
        assert result is None
        assert health.status == MarketStatus.SKIPPED
        assert "Prediction error" in health.skip_reason

    def test_zero_fill_ratio_skips(self):
        """Too many zero-filled features should skip."""
        from src.ml.model_loader import ModelLoader

        features = [f"f{i}" for i in range(10)]
        model = MagicMock()
        model.predict_proba.return_value = np.array([[0.3, 0.7]])

        model_data = self._make_model_data(features=features, model=model)
        loader = ModelLoader()
        loader._loaded_models["test_model"] = model_data

        # Provide only 4 of 10, so 6 are zero-filled (60%)
        df = self._make_features_df(cols=features[:4])
        health = MarketHealthReport(market="test")
        result = loader.predict_with_health("test_model", df, health_report=health)
        assert result is None
        assert health.status == MarketStatus.SKIPPED


# ---------------------------------------------------------------------------
# Integration: generate_daily_recommendations with health tracker
# ---------------------------------------------------------------------------
class TestHealthTrackerIntegration:
    """Test that generate_daily_recommendations integrates health tracking."""

    def test_health_tracker_param_accepted(self):
        """Verify generate_sniper_predictions accepts health_tracker param."""
        # Import should not fail
        import inspect

        from experiments.generate_daily_recommendations import (
            generate_sniper_predictions,
        )

        sig = inspect.signature(generate_sniper_predictions)
        assert "health_tracker" in sig.parameters

    def test_health_summary_json_valid(self):
        """Round-trip: build tracker, write summary, read back."""
        tracker = HealthTracker()

        ok = tracker.create_market_report("cards_under_35")
        ok.record_models_loaded(["cu35_lgb", "cu35_xgb"])
        ok.record_feature_match(expected=30, missing=0)
        ok.record_calibration(CalibrationStatus.CALIBRATED)
        ok.record_odds("real", 1.85)
        tracker.finalize(ok)

        degraded = tracker.create_market_report("shots_under_295")
        degraded.record_models_loaded(["su295_lgb"])
        degraded.record_feature_match(expected=30, missing=1)
        degraded.record_calibration(CalibrationStatus.UNCALIBRATED)
        degraded.record_odds("baseline")
        tracker.finalize(degraded)

        skipped = tracker.create_market_report("fouls")
        skipped.record_skip("No model files")
        tracker.finalize(skipped)

        tracker.add_global_warning("No odds file found")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "health.json"
            tracker.write_summary(path)

            data = json.loads(path.read_text())
            assert data["total_markets"] == 3
            assert data["markets_ok"] == 1
            assert data["markets_degraded"] == 1
            assert data["markets_skipped"] == 1
            assert "No odds file found" in data["global_warnings"]

            # Check per-market detail
            cu35 = data["markets"]["cards_under_35"]
            assert cu35["status"] == "ok"
            assert cu35["odds_source"] == "real"
            assert cu35["calibration_status"] == "calibrated"

            su295 = data["markets"]["shots_under_295"]
            assert su295["status"] == "degraded"
            assert su295["calibration_status"] == "uncalibrated"
            assert su295["threshold_multiplier"] == UNCALIBRATED_THRESHOLD_MULTIPLIER

            fouls = data["markets"]["fouls"]
            assert fouls["status"] == "skipped"
            assert fouls["skip_reason"] == "No model files"


# ---------------------------------------------------------------------------
# Backward compatibility: original predict() still works
# ---------------------------------------------------------------------------
class TestOriginalPredictUnchanged:
    """Ensure the original predict() method is unaffected."""

    def test_predict_returns_tuple(self):
        from src.ml.model_loader import ModelLoader

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

        df = pd.DataFrame({"f1": [1.0], "f2": [2.0]})
        result = loader.predict("test", df)
        assert result is not None
        prob, conf = result
        assert prob == pytest.approx(0.7)

    def test_predict_returns_none_for_missing(self):
        from src.ml.model_loader import ModelLoader

        loader = ModelLoader(models_dir=Path("/nonexistent"))
        result = loader.predict("missing", pd.DataFrame({"x": [1]}))
        assert result is None
