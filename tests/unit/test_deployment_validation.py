"""Tests for deployment config validation in generate_deployment_config.py."""
import math
import sys
from pathlib import Path

import pytest

# Ensure project root is on sys.path so we can import the script
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root / 'scripts'))

from generate_deployment_config import (  # noqa: E402
    _get_holdout_metric,
    is_better,
    validate_config,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(markets: dict) -> dict:
    return {"markets": markets}


def _market(
    enabled=True,
    model="stacking",
    saved_models=None,
    **extra,
) -> dict:
    cfg = {
        "enabled": enabled,
        "model": model,
        "threshold": 0.5,
        "roi": 10.0,
        "p_profit": 0.8,
        "saved_models": saved_models or [],
        "holdout_metrics": {
            "n_bets": 100,
            "ece": 0.05,
        },
    }
    cfg.update(extra)
    return cfg


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEnabledButEmptyModels:
    def test_blocks_when_enabled_and_no_saved_models(self):
        config = _make_config({
            "home_win": _market(enabled=True, saved_models=[]),
        })
        warnings = validate_config(config)
        assert any("BLOCKED" in w and "no saved_models" in w for w in warnings)
        assert config["markets"]["home_win"]["enabled"] is False

    def test_no_warning_when_disabled_and_no_saved_models(self):
        config = _make_config({
            "home_win": _market(enabled=False, saved_models=[]),
        })
        warnings = validate_config(config)
        assert len(warnings) == 0

    def test_no_warning_when_enabled_with_saved_models(self):
        config = _make_config({
            "home_win": _market(
                enabled=True,
                model="lightgbm",
                saved_models=["home_win_lightgbm.joblib"],
            ),
        })
        warnings = validate_config(config)
        assert len(warnings) == 0


class TestStrategyModelCountMismatch:
    def test_ensemble_with_one_model_warns(self):
        config = _make_config({
            "over25": _market(
                model="stacking",
                saved_models=["over25_lightgbm.joblib"],
            ),
        })
        warnings = validate_config(config)
        assert any("Ensemble strategy" in w and "needs at least 2" in w for w in warnings)

    def test_ensemble_with_four_models_ok(self):
        config = _make_config({
            "over25": _market(
                model="stacking",
                saved_models=[
                    "over25_lightgbm.joblib",
                    "over25_xgboost.joblib",
                    "over25_catboost.joblib",
                    "over25_fastai.joblib",
                ],
            ),
        })
        warnings = validate_config(config)
        assert len(warnings) == 0

    def test_single_model_with_four_models_warns(self):
        config = _make_config({
            "shots": _market(
                model="lightgbm",
                saved_models=[
                    "shots_lightgbm.joblib",
                    "shots_xgboost.joblib",
                    "shots_catboost.joblib",
                    "shots_fastai.joblib",
                ],
            ),
        })
        warnings = validate_config(config)
        assert any("Single-model strategy" in w and "extra models" in w for w in warnings)

    def test_single_model_with_one_model_ok(self):
        config = _make_config({
            "shots": _market(
                model="lightgbm",
                saved_models=["shots_lightgbm.joblib"],
            ),
        })
        warnings = validate_config(config)
        assert len(warnings) == 0

    @pytest.mark.parametrize("strategy", [
        "stacking", "average", "agreement", "temporal_blend",
        "disagree_lgb_filtered",
    ])
    def test_all_ensemble_strategies_checked(self, strategy):
        config = _make_config({
            "btts": _market(
                model=strategy,
                saved_models=["btts_lightgbm.joblib"],
            ),
        })
        warnings = validate_config(config)
        assert any("Ensemble strategy" in w for w in warnings)

    @pytest.mark.parametrize("strategy", [
        "lightgbm", "catboost", "xgboost", "fastai",
        "two_stage_lgb", "two_stage_xgb",
    ])
    def test_all_single_model_strategies_checked(self, strategy):
        config = _make_config({
            "cards": _market(
                model=strategy,
                saved_models=[
                    "cards_lightgbm.joblib",
                    "cards_xgboost.joblib",
                ],
            ),
        })
        warnings = validate_config(config)
        assert any("Single-model strategy" in w for w in warnings)


class TestArtifactExistence:
    def test_warns_when_model_file_missing(self, tmp_path):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        # Create only one of the two expected models
        (models_dir / "fouls_lightgbm.joblib").touch()

        config = _make_config({
            "fouls": _market(
                model="stacking",
                saved_models=[
                    "fouls_lightgbm.joblib",
                    "fouls_xgboost.joblib",
                ],
            ),
        })
        warnings = validate_config(config, models_dir=models_dir)
        missing_warnings = [w for w in warnings if "missing locally" in w]
        assert len(missing_warnings) == 1
        assert "fouls_xgboost.joblib" in missing_warnings[0]

    def test_no_warning_when_all_models_exist(self, tmp_path):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "fouls_lightgbm.joblib").touch()
        (models_dir / "fouls_xgboost.joblib").touch()

        config = _make_config({
            "fouls": _market(
                model="stacking",
                saved_models=[
                    "fouls_lightgbm.joblib",
                    "fouls_xgboost.joblib",
                ],
            ),
        })
        warnings = validate_config(config, models_dir=models_dir)
        assert len(warnings) == 0

    def test_skips_artifact_check_when_models_dir_none(self):
        config = _make_config({
            "fouls": _market(
                model="stacking",
                saved_models=[
                    "fouls_lightgbm.joblib",
                    "fouls_xgboost.joblib",
                ],
            ),
        })
        warnings = validate_config(config, models_dir=None)
        # No artifact warnings when models_dir is None
        assert not any("missing locally" in w for w in warnings)


class TestMultipleMarkets:
    def test_multiple_warnings_from_different_markets(self):
        config = _make_config({
            "home_win": _market(enabled=True, saved_models=[]),
            "shots": _market(
                model="lightgbm",
                saved_models=[
                    "shots_lightgbm.joblib",
                    "shots_xgboost.joblib",
                    "shots_catboost.joblib",
                    "shots_fastai.joblib",
                ],
            ),
            "over25": _market(
                model="stacking",
                saved_models=[
                    "over25_lightgbm.joblib",
                    "over25_xgboost.joblib",
                    "over25_catboost.joblib",
                ],
            ),
        })
        warnings = validate_config(config)
        # home_win: BLOCKED (no saved_models) + shots: extra models warning
        assert any("home_win" in w and "BLOCKED" in w for w in warnings)
        assert any("shots" in w and "Single-model strategy" in w for w in warnings)

    def test_clean_config_produces_no_warnings(self):
        config = _make_config({
            "home_win": _market(
                model="stacking",
                saved_models=[
                    "home_win_lightgbm.joblib",
                    "home_win_xgboost.joblib",
                    "home_win_catboost.joblib",
                ],
            ),
            "shots": _market(
                model="lightgbm",
                saved_models=["shots_lightgbm.joblib"],
            ),
        })
        warnings = validate_config(config)
        assert len(warnings) == 0


class TestMinBetsGate:
    def test_disables_market_below_min_bets(self):
        config = _make_config({
            "shots_under_255": _market(
                enabled=True,
                model="lightgbm",
                saved_models=["shots_under_255_lightgbm.joblib"],
                holdout_metrics={"n_bets": 6, "ece": 0.05},
            ),
        })
        warnings = validate_config(config, min_n_bets=20)
        assert any("BLOCKED" in w and "n_bets 6" in w for w in warnings)
        assert config["markets"]["shots_under_255"]["enabled"] is False

    def test_ok_above_min_bets(self):
        config = _make_config({
            "cards_under_25": _market(
                enabled=True,
                model="lightgbm",
                saved_models=["cards_under_25_lightgbm.joblib"],
                holdout_metrics={"n_bets": 100, "ece": 0.05},
            ),
        })
        warnings = validate_config(config, min_n_bets=20)
        blocked = [w for w in warnings if "BLOCKED" in w]
        assert len(blocked) == 0
        assert config["markets"]["cards_under_25"]["enabled"] is True

    def test_min_bets_skipped_for_disabled(self):
        config = _make_config({
            "fouls": _market(
                enabled=False,
                model="lightgbm",
                saved_models=["fouls_lightgbm.joblib"],
                n_bets=3,
            ),
        })
        warnings = validate_config(config, min_n_bets=20)
        blocked = [w for w in warnings if "BLOCKED" in w]
        assert len(blocked) == 0


class TestMaxEceGate:
    def test_disables_market_above_max_ece(self):
        config = _make_config({
            "home_win": _market(
                enabled=True,
                model="lightgbm",
                saved_models=["home_win_lightgbm.joblib"],
                holdout_metrics={"ece": 0.20, "n_bets": 100},
            ),
        })
        warnings = validate_config(config, max_ece=0.15)
        assert any("BLOCKED" in w and "ECE" in w for w in warnings)
        assert config["markets"]["home_win"]["enabled"] is False

    def test_ok_below_max_ece(self):
        config = _make_config({
            "btts": _market(
                enabled=True,
                model="lightgbm",
                saved_models=["btts_lightgbm.joblib"],
                holdout_metrics={"ece": 0.08, "n_bets": 100},
            ),
        })
        warnings = validate_config(config, max_ece=0.15)
        blocked = [w for w in warnings if "BLOCKED" in w]
        assert len(blocked) == 0
        assert config["markets"]["btts"]["enabled"] is True

    def test_ece_from_holdout_metrics(self):
        """ECE only in holdout_metrics sub-dict should still be validated."""
        config = _make_config({
            "over25": _market(
                enabled=True,
                model="lightgbm",
                saved_models=["over25_lightgbm.joblib"],
                holdout_metrics={"ece": 0.18, "n_bets": 100},
            ),
        })
        warnings = validate_config(config, max_ece=0.15)
        assert any("BLOCKED" in w and "ECE" in w for w in warnings)
        assert config["markets"]["over25"]["enabled"] is False

    def test_missing_ece_blocks(self):
        """Missing ECE should block — ECE is required for deployment."""
        config = _make_config({
            "cards": _market(
                enabled=True,
                model="lightgbm",
                saved_models=["cards_lightgbm.joblib"],
                holdout_metrics={"n_bets": 100},
            ),
        })
        warnings = validate_config(config, max_ece=0.15)
        blocked = [w for w in warnings if "BLOCKED" in w]
        assert len(blocked) == 1
        assert any("ECE missing" in w for w in blocked)
        assert config["markets"]["cards"]["enabled"] is False


class TestSavedModelsPathHandling:
    def test_handles_full_path_in_saved_models(self):
        """saved_models may contain paths like 'models/home_win_lightgbm.joblib'."""
        config = _make_config({
            "home_win": _market(
                model="lightgbm",
                saved_models=["models/home_win_lightgbm.joblib"],
            ),
        })
        warnings = validate_config(config)
        assert len(warnings) == 0


# ---------------------------------------------------------------------------
# is_better() hard-gate tests
# ---------------------------------------------------------------------------

def _new_market(
    precision=0.80,
    roi=20.0,
    n_bets=50,
    ece=0.05,
    ts_rejected=False,
    tracking_signal=None,
    **extra,
) -> dict:
    """Build a new-market dict for is_better() tests."""
    holdout: dict = {
        "n_bets": n_bets,
        "precision": precision,
        "roi": roi,
        "ece": ece,
        "ts_rejected": ts_rejected,
    }
    if tracking_signal is not None:
        holdout["tracking_signal"] = tracking_signal
    return {"precision": precision, "roi": roi, "holdout_metrics": holdout, **extra}


def _old_market(
    precision=0.75,
    roi=15.0,
    n_bets=60,
    ece=0.04,
    tracking_signal=None,
    **extra,
) -> dict:
    """Build an old-market dict for is_better() tests."""
    holdout: dict = {
        "n_bets": n_bets,
        "precision": precision,
        "roi": roi,
        "ece": ece,
    }
    if tracking_signal is not None:
        holdout["tracking_signal"] = tracking_signal
    return {"precision": precision, "roi": roi, "holdout_metrics": holdout, **extra}


class TestIsBetterHardGates:
    """Hard gates in is_better() that fire before metric comparison."""

    # -- Gate 1: holdout n_bets ------------------------------------------

    def test_rejects_zero_holdout_bets(self):
        new = _new_market(n_bets=0, precision=0.99)
        old = _old_market(precision=0.50)
        ok, reason = is_better(new, old, metric="precision")
        assert ok is False
        assert "REJECTED" in reason
        assert "n_bets=0" in reason

    def test_rejects_below_minimum_holdout_bets(self):
        new = _new_market(n_bets=15, precision=0.99)
        old = _old_market(precision=0.50)
        ok, reason = is_better(new, old, metric="precision", min_holdout_bets=20)
        assert ok is False
        assert "REJECTED" in reason
        assert "n_bets=15" in reason

    def test_rejects_missing_holdout_metrics(self):
        """holdout_metrics=None should be treated like 0 bets."""
        new = {"precision": 0.99, "holdout_metrics": None}
        old = _old_market(precision=0.50)
        ok, reason = is_better(new, old, metric="precision")
        assert ok is False
        assert "REJECTED" in reason
        assert "n_bets=0" in reason

    def test_rejects_missing_n_bets_key(self):
        """holdout_metrics dict without n_bets key should reject."""
        new = {"precision": 0.99, "holdout_metrics": {"ece": 0.03}}
        old = _old_market(precision=0.50)
        ok, reason = is_better(new, old, metric="precision")
        assert ok is False
        assert "REJECTED" in reason

    # -- Gate 2: ts_rejected ---------------------------------------------

    def test_rejects_ts_rejected_config(self):
        new = _new_market(n_bets=50, ts_rejected=True, precision=0.99)
        old = _old_market(precision=0.50)
        ok, reason = is_better(new, old, metric="precision")
        assert ok is False
        assert "REJECTED" in reason
        assert "tracking signal" in reason

    # -- Gate 3: ECE > 0.10 ----------------------------------------------

    def test_rejects_high_ece_config(self):
        new = _new_market(n_bets=50, ece=0.15, precision=0.99)
        old = _old_market(precision=0.50)
        ok, reason = is_better(new, old, metric="precision")
        assert ok is False
        assert "REJECTED" in reason
        assert "ECE" in reason

    def test_accepts_ece_at_boundary(self):
        """ECE exactly 0.10 should NOT be rejected (> 0.10 rejects)."""
        new = _new_market(n_bets=50, ece=0.10, precision=0.80)
        old = _old_market(precision=0.75)
        ok, reason = is_better(new, old, metric="precision")
        assert ok is True

    def test_accepts_none_ece(self):
        """Missing ECE should not trigger the ECE gate (let downstream gates handle it)."""
        new = _new_market(n_bets=50, precision=0.80)
        new["holdout_metrics"]["ece"] = None
        old = _old_market(precision=0.75)
        ok, reason = is_better(new, old, metric="precision")
        assert ok is True

    # -- Gate 4: CLEAN→CAUTION regression --------------------------------

    def test_rejects_clean_to_caution_regression(self):
        new = _new_market(n_bets=50, tracking_signal=5.0, precision=0.99)
        old = _old_market(tracking_signal=2.0, precision=0.50)
        ok, reason = is_better(new, old, metric="precision")
        assert ok is False
        assert "REJECTED" in reason
        assert "CLEAN" in reason and "CAUTION" in reason

    def test_accepts_caution_to_caution(self):
        """Both old and new CAUTION — gate should not fire."""
        new = _new_market(n_bets=50, tracking_signal=5.0, precision=0.80)
        old = _old_market(tracking_signal=4.5, precision=0.75)
        ok, reason = is_better(new, old, metric="precision")
        assert ok is True

    def test_accepts_caution_to_clean(self):
        """Improvement from CAUTION to CLEAN should be accepted."""
        new = _new_market(n_bets=50, tracking_signal=2.0, precision=0.80)
        old = _old_market(tracking_signal=5.0, precision=0.75)
        ok, reason = is_better(new, old, metric="precision")
        assert ok is True

    def test_handles_nan_tracking_signal_old(self):
        """NaN TS in old market should not trigger CLEAN→CAUTION gate."""
        new = _new_market(n_bets=50, tracking_signal=5.0, precision=0.80)
        old = _old_market(tracking_signal=float('nan'), precision=0.75)
        ok, reason = is_better(new, old, metric="precision")
        # NaN old TS is NOT "clean", so gate should not fire
        assert ok is True

    def test_handles_nan_tracking_signal_new(self):
        """NaN TS in new market should not trigger CLEAN→CAUTION gate."""
        new = _new_market(n_bets=50, tracking_signal=float('nan'), precision=0.80)
        old = _old_market(tracking_signal=2.0, precision=0.75)
        ok, reason = is_better(new, old, metric="precision")
        # NaN new TS is NOT "caution", so gate should not fire
        assert ok is True

    def test_handles_none_tracking_signal(self):
        """None TS values should not trigger CLEAN→CAUTION gate."""
        new = _new_market(n_bets=50, precision=0.80)  # no TS
        old = _old_market(precision=0.75)  # no TS
        ok, reason = is_better(new, old, metric="precision")
        assert ok is True

    # -- Normal comparison after gates pass ------------------------------

    def test_accepts_improvement_with_sufficient_bets(self):
        new = _new_market(n_bets=50, precision=0.85, ece=0.04)
        old = _old_market(precision=0.75)
        ok, reason = is_better(new, old, metric="precision")
        assert ok is True
        assert "0.75" in reason and "0.85" in reason

    def test_accepts_new_market_with_sufficient_bets(self):
        """A brand new market (old_market={}) should pass if gates pass."""
        new = _new_market(n_bets=50, precision=0.80, ece=0.04)
        ok, reason = is_better(new, {}, metric="precision")
        assert ok is True
        assert "new market" in reason

    def test_rejects_new_market_with_zero_bets(self):
        """New market with 0 holdout bets should still be rejected."""
        new = _new_market(n_bets=0, precision=0.90)
        ok, reason = is_better(new, {}, metric="precision")
        assert ok is False
        assert "REJECTED" in reason

    def test_rejects_regression_with_sufficient_bets(self):
        """Normal metric regression (without any gate violations) still rejected."""
        new = _new_market(n_bets=50, precision=0.70, ece=0.04)
        old = _old_market(precision=0.80)
        ok, reason = is_better(new, old, metric="precision")
        assert ok is False
        assert "REJECTED" not in reason  # normal rejection, not hard gate

    def test_staleness_tolerance_still_works(self):
        """Staleness tolerance logic must still work after gates pass."""
        new = _new_market(n_bets=50, precision=0.73, ece=0.04)
        old = _old_market(precision=0.75, trained_date="2024-01-01")
        ok, reason = is_better(
            new, old, metric="precision",
            max_model_age_days=30, staleness_tolerance=0.10,
        )
        assert ok is True
        assert "STALE" in reason


# ---------------------------------------------------------------------------
# _get_holdout_metric() tests
# ---------------------------------------------------------------------------


class TestGetHoldoutMetric:
    """Ensure holdout metrics are read from the correct source in all config formats."""

    def test_reads_from_holdout_metrics_dict(self):
        """Preferred source: nested holdout_metrics dict."""
        market = {"holdout_metrics": {"precision": 0.85, "roi": 20.0}}
        assert _get_holdout_metric(market, "precision") == 0.85
        assert _get_holdout_metric(market, "roi") == 20.0

    def test_reads_from_flat_holdout_fields(self):
        """Legacy format: flat holdout_precision, holdout_roi fields."""
        market = {"holdout_precision": 0.90, "holdout_roi": 48.5}
        assert _get_holdout_metric(market, "precision") == 0.90
        assert _get_holdout_metric(market, "roi") == 48.5

    def test_reads_from_top_level_as_last_resort(self):
        """generate_config S31+ format: top-level precision/roi (already holdout-sourced)."""
        market = {"precision": 0.78, "roi": 15.0}
        assert _get_holdout_metric(market, "precision") == 0.78
        assert _get_holdout_metric(market, "roi") == 15.0

    def test_prefers_holdout_metrics_over_flat(self):
        """holdout_metrics dict wins over flat holdout_* fields."""
        market = {
            "holdout_metrics": {"precision": 0.85},
            "holdout_precision": 0.70,
            "precision": 0.60,
        }
        assert _get_holdout_metric(market, "precision") == 0.85

    def test_prefers_flat_over_top_level(self):
        """flat holdout_* wins over top-level when holdout_metrics is missing."""
        market = {
            "holdout_precision": 0.80,
            "precision": 0.60,
        }
        assert _get_holdout_metric(market, "precision") == 0.80

    def test_returns_zero_when_missing(self):
        market = {}
        assert _get_holdout_metric(market, "precision") == 0.0

    def test_returns_zero_for_none_holdout_metrics(self):
        market = {"holdout_metrics": None}
        assert _get_holdout_metric(market, "precision") == 0.0

    def test_handles_none_value_in_holdout_metrics(self):
        """precision=None in holdout_metrics should fall through to flat/top-level."""
        market = {
            "holdout_metrics": {"precision": None},
            "holdout_precision": 0.75,
        }
        assert _get_holdout_metric(market, "precision") == 0.75


# ---------------------------------------------------------------------------
# is_better() holdout comparison regression tests (the auto-deploy bug)
# ---------------------------------------------------------------------------


class TestIsBetterHoldoutComparison:
    """
    Verify is_better() uses holdout metrics for comparison, not WF metrics.

    The bug: old deployed configs have precision only inside holdout_metrics
    (no top-level 'precision'), so old_market.get('precision') returned None/0.
    This made ANY new result appear "better" and auto-deploy regressions.
    """

    def test_old_config_no_top_level_precision_still_compared(self):
        """Old config with precision ONLY in holdout_metrics must be compared correctly."""
        new = _new_market(n_bets=50, precision=0.70, ece=0.04)
        # Old config WITHOUT top-level precision (real production format)
        old = {
            "model": "temporal_blend",
            "holdout_metrics": {
                "n_bets": 100,
                "precision": 0.85,
                "roi": 30.0,
                "ece": 0.03,
            },
        }
        ok, reason = is_better(new, old, metric="precision")
        assert ok is False, (
            "New precision 0.70 < old holdout precision 0.85 should be rejected"
        )
        assert "0.8500" in reason
        assert "0.7000" in reason

    def test_old_config_flat_holdout_precision_compared(self):
        """Old config with flat holdout_precision field must be compared correctly."""
        new = _new_market(n_bets=50, precision=0.72, ece=0.04)
        # Old config with FLAT holdout_precision (legacy format)
        old = {
            "model": "stacking",
            "holdout_precision": 0.88,
            "holdout_metrics": {"n_bets": 80, "ece": 0.03},
        }
        ok, reason = is_better(new, old, metric="precision")
        assert ok is False

    def test_new_genuinely_better_holdout_accepted(self):
        """New config with genuinely higher holdout precision should be accepted."""
        new = _new_market(n_bets=50, precision=0.90, ece=0.04)
        old = {
            "model": "temporal_blend",
            "holdout_metrics": {
                "n_bets": 100,
                "precision": 0.80,
                "roi": 20.0,
                "ece": 0.04,
            },
        }
        ok, reason = is_better(new, old, metric="precision")
        assert ok is True
        assert "0.8000" in reason
        assert "0.9000" in reason

    def test_roi_comparison_also_uses_holdout(self):
        """ROI metric comparison should also use holdout values."""
        new = _new_market(n_bets=50, precision=0.80, roi=10.0, ece=0.04)
        old = {
            "holdout_metrics": {
                "n_bets": 100,
                "precision": 0.75,
                "roi": 25.0,
                "ece": 0.04,
            },
        }
        ok, reason = is_better(new, old, metric="roi")
        assert ok is False, "New ROI 10.0 < old holdout ROI 25.0 should be rejected"
