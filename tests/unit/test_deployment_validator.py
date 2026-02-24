"""Unit tests for deployment_validator.py gate enforcement."""

import json
import tempfile
from pathlib import Path

import pytest

from src.ml.deployment_validator import (
    MAX_ECE,
    MIN_HOLDOUT_BETS,
    _get_holdout_ece,
    _get_holdout_n_bets,
    _validate_market,
    auto_fix,
    validate_deployment_config,
)


def _make_config(markets: dict) -> Path:
    """Write a temporary sniper_deployment.json and return its path."""
    config = {
        "generated_at": "2026-02-24T00:00:00",
        "source": "test",
        "markets": markets,
    }
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, dir=tempfile.gettempdir()
    )
    json.dump(config, tmp)
    tmp.flush()
    return Path(tmp.name)


def _valid_market(**overrides) -> dict:
    """Return a valid enabled market config with optional overrides."""
    base = {
        "enabled": True,
        "model": "catboost",
        "threshold": 0.60,
        "roi": 110.0,
        "n_bets": 500,
        "source_run": "S36 R400: catboost, beta, 500 HO bets",
        "selected_features": ["feat_a", "feat_b"],
        "saved_models": ["market_catboost.joblib"],
        "holdout_metrics": {
            "precision": 0.80,
            "roi": 110.0,
            "n_bets": 500,
            "n_wins": 400,
            "ece": 0.05,
        },
    }
    base.update(overrides)
    return base


class TestPassesValidMarket:
    """Market with valid ECE and n_bets should pass."""

    def test_passes_valid_market(self):
        result = _validate_market("cards_under_25", _valid_market())
        assert result.passed is True
        assert result.violations == []

    def test_passes_at_boundary_ece(self):
        market = _valid_market()
        market["holdout_metrics"]["ece"] = 0.10  # exactly at limit
        result = _validate_market("corners_over_85", market)
        assert result.passed is True

    def test_passes_at_boundary_n_bets(self):
        market = _valid_market()
        market["holdout_metrics"]["n_bets"] = 20  # exactly at minimum
        result = _validate_market("shots_over_245", market)
        assert result.passed is True


class TestFailsHighECE:
    """Markets with ECE above threshold should fail."""

    def test_fails_high_ece(self):
        market = _valid_market()
        market["holdout_metrics"]["ece"] = 0.15
        result = _validate_market("cards_over_15", market)
        assert result.passed is False
        assert any("ECE" in v for v in result.violations)

    def test_fails_very_high_ece(self):
        market = _valid_market()
        market["holdout_metrics"]["ece"] = 0.228
        result = _validate_market("cards_over_15", market)
        assert result.passed is False
        assert "ECE 0.228 > 0.1" in result.violations[0]


class TestFailsLowNBets:
    """Markets with too few holdout bets should fail."""

    def test_fails_low_n_bets(self):
        market = _valid_market()
        market["holdout_metrics"]["n_bets"] = 5
        result = _validate_market("under25", market)
        assert result.passed is False
        assert any("n_bets" in v for v in result.violations)


class TestFailsNullHoldout:
    """Markets without holdout_metrics should fail."""

    def test_fails_null_holdout(self):
        market = _valid_market()
        market["holdout_metrics"] = None
        result = _validate_market("fouls_over_235", market)
        assert result.passed is False
        assert "no holdout_metrics" in result.violations

    def test_fails_missing_holdout(self):
        market = _valid_market()
        del market["holdout_metrics"]
        result = _validate_market("fouls_over_235", market)
        assert result.passed is False
        assert "no holdout_metrics" in result.violations


class TestECEFallbackToToplevel:
    """When holdout_metrics.ece is missing, use top-level ece."""

    def test_ece_fallback_to_toplevel(self):
        market = _valid_market()
        del market["holdout_metrics"]["ece"]
        market["ece"] = 0.15  # top-level ECE above threshold
        result = _validate_market("cards_over_15", market)
        assert result.passed is False
        assert any("ECE" in v for v in result.violations)

    def test_ece_fallback_passes(self):
        market = _valid_market()
        del market["holdout_metrics"]["ece"]
        market["ece"] = 0.05  # top-level ECE below threshold
        result = _validate_market("cards_under_35", market)
        assert result.passed is True


class TestAutoFixDisablesMarket:
    """Failing markets should be disabled when --auto-fix is used."""

    def test_auto_fix_disables_market(self):
        markets = {
            "good_market": _valid_market(),
            "bad_market": _valid_market(),
        }
        markets["bad_market"]["holdout_metrics"]["ece"] = 0.20
        config_path = _make_config(markets)

        report, n_disabled = auto_fix(config_path)
        assert n_disabled == 1

        # Verify file was updated
        with open(config_path) as f:
            updated = json.load(f)
        assert updated["markets"]["bad_market"]["enabled"] is False
        assert "disabled_reason" in updated["markets"]["bad_market"]
        assert updated["markets"]["good_market"]["enabled"] is True

    def test_auto_fix_no_changes_when_all_pass(self):
        markets = {"good_market": _valid_market()}
        config_path = _make_config(markets)

        report, n_disabled = auto_fix(config_path)
        assert n_disabled == 0
        assert report.all_passed is True


class TestDisabledMarketsSkipped:
    """Disabled markets should not be validated."""

    def test_disabled_markets_skipped(self):
        markets = {
            "enabled_market": _valid_market(),
            "disabled_market": _valid_market(),
        }
        markets["disabled_market"]["enabled"] = False
        markets["disabled_market"]["holdout_metrics"]["ece"] = 0.50  # would fail
        config_path = _make_config(markets)

        report = validate_deployment_config(config_path)
        assert len(report.results) == 1
        assert report.results[0].market == "enabled_market"
        assert report.all_passed is True


class TestSourceRunWarning:
    """Missing or unknown source_run should produce a warning."""

    def test_source_run_missing(self):
        market = _valid_market()
        del market["source_run"]
        result = _validate_market("corners_over_85", market)
        assert result.passed is True  # warning, not failure
        assert any("source_run" in w for w in result.warnings)

    def test_source_run_unknown(self):
        market = _valid_market()
        market["source_run"] = "?"
        result = _validate_market("corners_over_85", market)
        assert result.passed is True
        assert any("source_run" in w for w in result.warnings)


class TestModelFileWarning:
    """Missing model files should produce a warning."""

    def test_missing_model_file(self, tmp_path):
        market = _valid_market()
        market["saved_models"] = ["nonexistent_model.joblib"]
        result = _validate_market(
            "cards_under_25", market, models_dir=tmp_path
        )
        assert result.passed is True
        assert any("model file missing" in w for w in result.warnings)


class TestMultipleViolations:
    """Markets can have multiple violations."""

    def test_multiple_violations(self):
        market = _valid_market()
        market["holdout_metrics"]["ece"] = 0.20
        market["holdout_metrics"]["n_bets"] = 5
        result = _validate_market("bad_market", market)
        assert result.passed is False
        assert len(result.violations) == 2


class TestGetHelpers:
    """Test ECE and n_bets extraction helpers."""

    def test_get_holdout_ece_from_holdout_metrics(self):
        config = {"holdout_metrics": {"ece": 0.05}}
        assert _get_holdout_ece(config) == 0.05

    def test_get_holdout_ece_fallback(self):
        config = {"holdout_metrics": {}, "ece": 0.08}
        assert _get_holdout_ece(config) == 0.08

    def test_get_holdout_ece_none(self):
        config = {"holdout_metrics": {}}
        assert _get_holdout_ece(config) is None

    def test_get_holdout_n_bets(self):
        config = {"holdout_metrics": {"n_bets": 100}}
        assert _get_holdout_n_bets(config) == 100

    def test_get_holdout_n_bets_none(self):
        config = {"holdout_metrics": {}}
        assert _get_holdout_n_bets(config) is None


class TestValidateDeploymentConfig:
    """Integration tests for validate_deployment_config."""

    def test_full_report(self):
        markets = {
            "pass_market": _valid_market(),
            "fail_ece": _valid_market(),
            "fail_nbets": _valid_market(),
            "disabled": _valid_market(),
        }
        markets["fail_ece"]["holdout_metrics"]["ece"] = 0.15
        markets["fail_nbets"]["holdout_metrics"]["n_bets"] = 5
        markets["disabled"]["enabled"] = False
        config_path = _make_config(markets)

        report = validate_deployment_config(config_path)
        assert report.n_passed == 1
        assert report.n_failed == 2
        assert not report.all_passed

    def test_handles_nan_in_json(self):
        """Config with NaN values should be handled gracefully."""
        config = {
            "generated_at": "2026-02-24",
            "source": "test",
            "markets": {
                "test_market": _valid_market(),
            },
        }
        # Write with NaN manually
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, dir=tempfile.gettempdir()
        )
        content = json.dumps(config).replace('"sortino": null', '"sortino": NaN')
        # Ensure holdout_metrics still has valid ece/n_bets
        tmp.write(content)
        tmp.flush()

        report = validate_deployment_config(Path(tmp.name))
        assert len(report.results) == 1
        assert report.results[0].passed is True
