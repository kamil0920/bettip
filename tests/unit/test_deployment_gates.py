"""Unit tests for the shared deployment gate module."""

import pytest

from src.ml.deployment_gates import (
    MAX_ECE,
    MIN_HOLDOUT_BETS_FALLBACK,
    check_deployment_gates,
)


def _valid_config(**overrides) -> dict:
    """Return a market config that passes all gates."""
    base = {
        "enabled": True,
        "model": "catboost",
        "threshold": 0.60,
        "saved_models": ["market_catboost.joblib"],
        "holdout_metrics": {
            "n_bets": 100,
            "ece": 0.05,
        },
    }
    base.update(overrides)
    return base


class TestAllGatesPass:
    """Configs that satisfy every gate should return no violations."""

    def test_valid_config_passes(self):
        v = check_deployment_gates("over25", _valid_config())
        assert v == []

    def test_boundary_ece_passes(self):
        cfg = _valid_config()
        cfg["holdout_metrics"]["ece"] = MAX_ECE  # exactly at limit
        v = check_deployment_gates("over25", cfg)
        assert v == []

    def test_boundary_n_bets_at_fallback(self):
        cfg = _valid_config()
        cfg["holdout_metrics"]["n_bets"] = MIN_HOLDOUT_BETS_FALLBACK
        v = check_deployment_gates("over25", cfg)
        assert v == []

    def test_n_bets_at_mintrl(self):
        cfg = _valid_config()
        cfg["holdout_metrics"]["min_track_record_length"] = 30
        cfg["holdout_metrics"]["n_bets"] = 30
        v = check_deployment_gates("sot", cfg)
        assert v == []


class TestECEGate:
    """ECE must be present and below threshold."""

    def test_ece_none_fails(self):
        cfg = _valid_config()
        cfg["holdout_metrics"]["ece"] = None
        v = check_deployment_gates("btts", cfg)
        assert any("ECE missing" in s for s in v)

    def test_ece_missing_key_fails(self):
        cfg = _valid_config()
        del cfg["holdout_metrics"]["ece"]
        v = check_deployment_gates("btts", cfg)
        assert any("ECE missing" in s for s in v)

    def test_ece_too_high_fails(self):
        cfg = _valid_config()
        cfg["holdout_metrics"]["ece"] = 0.15
        v = check_deployment_gates("cards", cfg)
        assert any("ECE 0.150 > 0.1" in s for s in v)

    def test_ece_fallback_to_toplevel(self):
        cfg = _valid_config()
        del cfg["holdout_metrics"]["ece"]
        cfg["ece"] = 0.04
        v = check_deployment_gates("corners", cfg)
        assert not any("ECE" in s for s in v)

    def test_ece_toplevel_too_high(self):
        cfg = _valid_config()
        del cfg["holdout_metrics"]["ece"]
        cfg["ece"] = 0.15
        v = check_deployment_gates("corners", cfg)
        assert any("ECE" in s for s in v)


class TestSavedModelsGate:
    """saved_models must be a non-empty list."""

    def test_empty_saved_models_fails(self):
        cfg = _valid_config(saved_models=[])
        v = check_deployment_gates("fouls", cfg)
        assert any("no saved_models" in s for s in v)

    def test_missing_saved_models_fails(self):
        cfg = _valid_config()
        del cfg["saved_models"]
        v = check_deployment_gates("fouls", cfg)
        assert any("no saved_models" in s for s in v)

    def test_none_saved_models_fails(self):
        cfg = _valid_config(saved_models=None)
        v = check_deployment_gates("fouls", cfg)
        assert any("no saved_models" in s for s in v)


class TestNBetsGate:
    """n_bets must meet MinTRL or fallback threshold."""

    def test_n_bets_below_mintrl_fails(self):
        cfg = _valid_config()
        cfg["holdout_metrics"]["min_track_record_length"] = 40
        cfg["holdout_metrics"]["n_bets"] = 30
        v = check_deployment_gates("sot", cfg)
        assert any("n_bets 30 < MinTRL 40" in s for s in v)

    def test_n_bets_above_mintrl_passes(self):
        cfg = _valid_config()
        cfg["holdout_metrics"]["min_track_record_length"] = 12
        cfg["holdout_metrics"]["n_bets"] = 28
        v = check_deployment_gates("sot", cfg)
        assert not any("n_bets" in s for s in v)

    def test_no_mintrl_below_fallback_fails(self):
        cfg = _valid_config()
        cfg["holdout_metrics"]["n_bets"] = 30
        v = check_deployment_gates("cards", cfg)
        assert any("n_bets 30 < fallback min 50" in s for s in v)

    def test_no_mintrl_above_fallback_passes(self):
        cfg = _valid_config()
        cfg["holdout_metrics"]["n_bets"] = 50
        v = check_deployment_gates("cards", cfg)
        assert not any("n_bets" in s for s in v)

    def test_n_bets_missing_fails(self):
        cfg = _valid_config()
        del cfg["holdout_metrics"]["n_bets"]
        v = check_deployment_gates("dc_1x", cfg)
        assert any("n_bets missing" in s for s in v)


class TestHoldoutMetricsGate:
    """holdout_metrics must be a dict."""

    def test_no_holdout_metrics_fails(self):
        cfg = _valid_config()
        del cfg["holdout_metrics"]
        v = check_deployment_gates("fouls", cfg)
        assert v == ["no holdout_metrics dict"]

    def test_null_holdout_metrics_fails(self):
        cfg = _valid_config(holdout_metrics=None)
        v = check_deployment_gates("fouls", cfg)
        assert v == ["no holdout_metrics dict"]

    def test_holdout_metrics_not_dict_fails(self):
        cfg = _valid_config(holdout_metrics="bad")
        v = check_deployment_gates("fouls", cfg)
        assert v == ["no holdout_metrics dict"]


class TestMultipleViolations:
    """A single market can have multiple violations."""

    def test_multiple_violations(self):
        cfg = _valid_config(saved_models=[])
        cfg["holdout_metrics"]["ece"] = 0.20
        cfg["holdout_metrics"]["n_bets"] = 5
        v = check_deployment_gates("bad_market", cfg)
        assert len(v) == 3  # saved_models + ECE + n_bets

    def test_ece_missing_and_no_models(self):
        cfg = _valid_config(saved_models=[])
        del cfg["holdout_metrics"]["ece"]
        v = check_deployment_gates("phantom", cfg)
        assert any("no saved_models" in s for s in v)
        assert any("ECE missing" in s for s in v)


class TestCustomThresholds:
    """Custom max_ece and min_bets_fallback parameters."""

    def test_custom_max_ece(self):
        cfg = _valid_config()
        cfg["holdout_metrics"]["ece"] = 0.06
        v = check_deployment_gates("x", cfg, max_ece=0.05)
        assert any("ECE" in s for s in v)

    def test_custom_min_bets_fallback(self):
        cfg = _valid_config()
        cfg["holdout_metrics"]["n_bets"] = 25
        v = check_deployment_gates("x", cfg, min_bets_fallback=30)
        assert any("n_bets 25 < fallback min 30" in s for s in v)
