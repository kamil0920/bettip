"""Tests for strategy resolution in match_scheduler.

Covers:
- get_enabled_markets() reading walkforward.best_model / best_model_wf
- Strategy propagation: wf_best and stacking_weights passed through
- Strategies.yaml fallback (no wf_best → empty string)
"""
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data_collection.match_scheduler import get_enabled_markets


def _make_sniper_config(markets_dict):
    """Wrap market configs in sniper deployment format."""
    return {"markets": markets_dict}


# ---------------------------------------------------------------------------
# get_enabled_markets — walkforward field resolution
# ---------------------------------------------------------------------------

class TestGetEnabledMarketsWalkforward:
    """Test that get_enabled_markets reads walkforward.best_model correctly."""

    def test_best_model_read(self):
        """wf_best should come from walkforward.best_model."""
        sniper = _make_sniper_config({
            "btts": {
                "enabled": True,
                "threshold": 0.6,
                "model": "average",
                "walkforward": {"best_model": "lightgbm"},
                "saved_models": ["btts_lightgbm.joblib"],
            }
        })
        with patch("src.data_collection.match_scheduler.load_sniper_deployment", return_value=sniper):
            enabled = get_enabled_markets({})

        assert "btts" in enabled
        assert enabled["btts"]["wf_best"] == "lightgbm"

    def test_best_model_wf_fallback(self):
        """When only best_model_wf exists, it should be used."""
        sniper = _make_sniper_config({
            "corners": {
                "enabled": True,
                "threshold": 0.6,
                "model": "disagree_aggressive_filtered",
                "walkforward": {
                    "best_model_wf": "disagree_conservative_filtered",
                    "summary": {},
                },
                "saved_models": ["corners_lightgbm.joblib"],
            }
        })
        with patch("src.data_collection.match_scheduler.load_sniper_deployment", return_value=sniper):
            enabled = get_enabled_markets({})

        assert enabled["corners"]["wf_best"] == "disagree_conservative_filtered"

    def test_best_model_takes_precedence_over_best_model_wf(self):
        """best_model should win when both keys exist."""
        sniper = _make_sniper_config({
            "shots": {
                "enabled": True,
                "threshold": 0.65,
                "model": "lightgbm",
                "walkforward": {
                    "best_model": "temporal_blend",
                    "best_model_wf": "temporal_blend",
                },
                "saved_models": ["shots_lightgbm.joblib"],
            }
        })
        with patch("src.data_collection.match_scheduler.load_sniper_deployment", return_value=sniper):
            enabled = get_enabled_markets({})

        assert enabled["shots"]["wf_best"] == "temporal_blend"

    def test_empty_best_model_falls_to_best_model_wf(self):
        """Empty string best_model should fall back to best_model_wf."""
        sniper = _make_sniper_config({
            "fouls": {
                "enabled": True,
                "threshold": 0.8,
                "model": "catboost",
                "walkforward": {
                    "best_model": "",
                    "best_model_wf": "catboost",
                },
                "saved_models": ["fouls_catboost.joblib"],
            }
        })
        with patch("src.data_collection.match_scheduler.load_sniper_deployment", return_value=sniper):
            enabled = get_enabled_markets({})

        assert enabled["fouls"]["wf_best"] == "catboost"

    def test_no_walkforward_skips_market(self):
        """Missing walkforward section → market skipped (can't select model)."""
        sniper = _make_sniper_config({
            "btts": {
                "enabled": True,
                "threshold": 0.6,
                "model": "xgboost",
                "saved_models": ["btts_xgboost.joblib"],
            }
        })
        with patch("src.data_collection.match_scheduler.load_sniper_deployment", return_value=sniper):
            enabled = get_enabled_markets({})

        assert "btts" not in enabled


# ---------------------------------------------------------------------------
# get_enabled_markets — stacking weights propagation
# ---------------------------------------------------------------------------

class TestGetEnabledMarketsStackingWeights:
    """Test that stacking_weights are passed through."""

    def test_stacking_weights_propagated(self):
        weights = {"lightgbm": 1.5, "catboost": 2.0, "xgboost": -0.5}
        sniper = _make_sniper_config({
            "cards": {
                "enabled": True,
                "threshold": 0.6,
                "model": "disagree_aggressive_filtered",
                "walkforward": {"best_model": "stacking"},
                "stacking_weights": weights,
                "saved_models": ["cards_lightgbm.joblib"],
            }
        })
        with patch("src.data_collection.match_scheduler.load_sniper_deployment", return_value=sniper):
            enabled = get_enabled_markets({})

        assert enabled["cards"]["stacking_weights"] == weights

    def test_missing_stacking_weights_gives_empty_dict(self):
        sniper = _make_sniper_config({
            "btts": {
                "enabled": True,
                "threshold": 0.6,
                "model": "average",
                "walkforward": {"best_model": "lightgbm"},
                "saved_models": ["btts_lightgbm.joblib"],
            }
        })
        with patch("src.data_collection.match_scheduler.load_sniper_deployment", return_value=sniper):
            enabled = get_enabled_markets({})

        assert enabled["btts"]["stacking_weights"] == {}


# ---------------------------------------------------------------------------
# get_enabled_markets — disabled markets excluded
# ---------------------------------------------------------------------------

class TestGetEnabledMarketsFiltering:
    """Test that disabled markets are excluded."""

    def test_disabled_market_excluded(self):
        sniper = _make_sniper_config({
            "btts": {
                "enabled": True,
                "threshold": 0.6,
                "model": "average",
                "walkforward": {"best_model": "lightgbm"},
                "saved_models": [],
            },
            "fouls": {
                "enabled": False,
                "threshold": 0.8,
                "model": "catboost",
                "saved_models": [],
            },
        })
        with patch("src.data_collection.match_scheduler.load_sniper_deployment", return_value=sniper):
            enabled = get_enabled_markets({})

        assert "btts" in enabled
        assert "fouls" not in enabled


# ---------------------------------------------------------------------------
# get_enabled_markets — strategies.yaml fallback
# ---------------------------------------------------------------------------

class TestGetEnabledMarketsYamlFallback:
    """Test fallback to strategies.yaml when no sniper config."""

    def test_yaml_fallback_has_no_wf_best(self):
        """strategies.yaml path doesn't set wf_best — should default to empty."""
        yaml_config = {
            "strategies": {
                "btts": {
                    "enabled": True,
                    "probability_threshold": 0.75,
                    "model_type": "xgboost",
                    "expected_roi": 10,
                    "p_profit": 0.8,
                }
            }
        }
        with patch("src.data_collection.match_scheduler.load_sniper_deployment", return_value={}):
            enabled = get_enabled_markets(yaml_config)

        assert "btts" in enabled
        # strategies.yaml path doesn't set wf_best
        assert "wf_best" not in enabled["btts"]
        assert enabled["btts"]["model_type"] == "xgboost"


# ---------------------------------------------------------------------------
# Strategy alignment: deployment config produces consistent wf_best
# for all 9 markets (integration-style test with real config)
# ---------------------------------------------------------------------------

class TestDeploymentConfigStrategyAlignment:
    """Verify the actual deployment config produces expected wf_best values."""

    @pytest.fixture
    def real_config(self):
        config_path = project_root / "config" / "sniper_deployment.json"
        if not config_path.exists():
            pytest.skip("sniper_deployment.json not on disk")
        with open(config_path) as f:
            return json.load(f)

    def test_all_markets_have_wf_best(self, real_config):
        """Every enabled market should have a non-empty wf_best."""
        with patch("src.data_collection.match_scheduler.load_sniper_deployment", return_value=real_config):
            enabled = get_enabled_markets({})

        for market, cfg in enabled.items():
            assert cfg["wf_best"], f"{market} has empty wf_best"

    def test_niche_markets_have_best_model(self, real_config):
        """Niche markets (corners, cards, fouls, shots) should resolve wf_best."""
        with patch("src.data_collection.match_scheduler.load_sniper_deployment", return_value=real_config):
            enabled = get_enabled_markets({})

        niche = ["corners", "cards", "fouls", "shots"]
        for market in niche:
            if market in enabled:
                assert enabled[market]["wf_best"], f"{market} wf_best is empty"
