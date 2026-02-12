"""Tests for ensemble strategy selection in daily recommendations.

Covers:
- walkforward.best_model / best_model_wf field resolution
- average / temporal_blend strategy
- disagree_*_filtered strategies (conservative, balanced, aggressive)
- stacking strategy (existing, regression test)
- agreement strategy (existing, regression test)
- fallback to single model when strategy unknown
"""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from experiments.generate_daily_recommendations import (
    generate_sniper_predictions,
    calculate_edge,
    MARKET_ODDS_COLUMNS,
    MARKET_BASELINES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_match(home="TeamA", away="TeamB", fixture_id=1001):
    return {
        "home_team": home,
        "away_team": away,
        "league": "Test League",
        "fixture_id": fixture_id,
        "kickoff": "2026-02-07T20:00:00",
    }


def _make_market_config(
    *,
    threshold=0.6,
    model="lightgbm",
    wf_best_model=None,
    wf_best_model_wf=None,
    saved_models=None,
    stacking_weights=None,
):
    """Build a minimal market config dict for testing."""
    cfg = {
        "enabled": True,
        "model": model,
        "threshold": threshold,
        "saved_models": saved_models or ["mkt_lightgbm.joblib", "mkt_catboost.joblib", "mkt_xgboost.joblib"],
    }
    wf = {}
    if wf_best_model is not None:
        wf["best_model"] = wf_best_model
    if wf_best_model_wf is not None:
        wf["best_model_wf"] = wf_best_model_wf
    if wf:
        cfg["walkforward"] = wf
    if stacking_weights:
        cfg["stacking_weights"] = stacking_weights
    return cfg


def _run_strategy(
    market_name,
    market_config,
    model_probs,
    match_odds=None,
    min_edge_pct=0.0,
):
    """
    Run generate_sniper_predictions with mocked dependencies, returning
    the list of prediction dicts (or empty if filtered out).

    model_probs: list of (model_name, probability, confidence) tuples
    """
    match = _make_match()
    if match_odds is None:
        match_odds = {}

    # Mock all heavy dependencies
    mock_feature_lookup = MagicMock()
    mock_feature_lookup.load.return_value = True
    # Return a minimal DataFrame (1 row) — model_loader.predict is mocked anyway
    import pandas as pd
    mock_feature_lookup.get_team_features.return_value = pd.DataFrame({"x": [1.0]})

    mock_model_loader = MagicMock()
    available = {name for name, _, _ in model_probs}
    mock_model_loader.list_available_models.return_value = list(available)

    # predict() returns (prob, confidence) for each model
    prob_map = {name: (prob, conf) for name, prob, conf in model_probs}
    mock_model_loader.predict.side_effect = lambda name, df, **kwargs: prob_map.get(name)

    mock_injector = MagicMock()
    mock_injector.inject_features.side_effect = lambda df, ctx: df

    with patch("experiments.generate_daily_recommendations.load_sniper_config", return_value={market_name: market_config}), \
         patch("experiments.generate_daily_recommendations.ModelLoader", return_value=mock_model_loader), \
         patch("experiments.generate_daily_recommendations.FeatureLookup", return_value=mock_feature_lookup), \
         patch("experiments.generate_daily_recommendations.ExternalFeatureInjector", return_value=mock_injector), \
         patch("experiments.generate_daily_recommendations.load_odds", return_value=None), \
         patch("experiments.generate_daily_recommendations.get_match_odds", return_value=match_odds), \
         patch("experiments.generate_daily_recommendations._load_prematch_lineups", return_value=(None, None)), \
         patch("experiments.generate_daily_recommendations._score_all_strategies"):

        return generate_sniper_predictions([match], min_edge_pct=min_edge_pct)


# ---------------------------------------------------------------------------
# Walkforward field resolution
# ---------------------------------------------------------------------------

class TestWalkforwardFieldResolution:
    """Test that best_model and best_model_wf are both handled."""

    def test_best_model_takes_precedence(self):
        """When both best_model and best_model_wf exist, best_model wins."""
        cfg = _make_market_config(
            wf_best_model="xgboost",
            wf_best_model_wf="catboost",
            threshold=0.5,
        )
        preds = _run_strategy(
            "corners", cfg,
            [
                ("mkt_lightgbm", 0.7, 0.8),
                ("mkt_catboost", 0.6, 0.7),
                ("mkt_xgboost", 0.75, 0.85),
            ],
        )
        assert len(preds) == 1
        # xgboost should be selected (best_model takes precedence)
        assert preds[0]["probability"] == 0.75

    def test_best_model_wf_fallback(self):
        """When only best_model_wf exists (no best_model), it's used."""
        cfg = _make_market_config(
            wf_best_model_wf="catboost",
            threshold=0.5,
        )
        # No best_model key — should fall back to best_model_wf
        preds = _run_strategy(
            "corners", cfg,
            [
                ("mkt_lightgbm", 0.7, 0.8),
                ("mkt_catboost", 0.8, 0.9),
                ("mkt_xgboost", 0.65, 0.7),
            ],
        )
        assert len(preds) == 1
        assert preds[0]["probability"] == 0.8

    def test_empty_best_model_falls_to_best_model_wf(self):
        """When best_model is empty string, best_model_wf is used."""
        cfg = _make_market_config(
            wf_best_model="",
            wf_best_model_wf="lightgbm",
            threshold=0.5,
        )
        preds = _run_strategy(
            "corners", cfg,
            [
                ("mkt_lightgbm", 0.72, 0.8),
                ("mkt_catboost", 0.6, 0.7),
                ("mkt_xgboost", 0.65, 0.7),
            ],
        )
        assert len(preds) == 1
        assert preds[0]["probability"] == 0.72


# ---------------------------------------------------------------------------
# Average / temporal_blend strategies
# ---------------------------------------------------------------------------

class TestAverageStrategy:
    """Test average and temporal_blend ensemble strategies."""

    def test_average_uses_mean_of_all_models(self):
        cfg = _make_market_config(wf_best_model="average", threshold=0.5)
        probs = [0.7, 0.8, 0.6]
        preds = _run_strategy(
            "corners", cfg,
            [
                ("mkt_lightgbm", probs[0], 0.8),
                ("mkt_catboost", probs[1], 0.9),
                ("mkt_xgboost", probs[2], 0.7),
            ],
        )
        assert len(preds) == 1
        expected = round(sum(probs) / len(probs), 4)
        assert preds[0]["probability"] == expected

    def test_temporal_blend_uses_mean(self):
        """temporal_blend approximated as average at prediction time."""
        cfg = _make_market_config(wf_best_model="temporal_blend", threshold=0.5)
        probs = [0.65, 0.75, 0.70]
        preds = _run_strategy(
            "shots", cfg,
            [
                ("mkt_lightgbm", probs[0], 0.8),
                ("mkt_catboost", probs[1], 0.9),
                ("mkt_xgboost", probs[2], 0.85),
            ],
        )
        assert len(preds) == 1
        expected = round(sum(probs) / len(probs), 4)
        assert preds[0]["probability"] == expected

    def test_average_below_threshold_filtered(self):
        """Average prob below threshold produces no prediction."""
        cfg = _make_market_config(wf_best_model="average", threshold=0.8)
        preds = _run_strategy(
            "corners", cfg,
            [
                ("mkt_lightgbm", 0.6, 0.8),
                ("mkt_catboost", 0.7, 0.9),
                ("mkt_xgboost", 0.5, 0.7),
            ],
        )
        assert len(preds) == 0

    def test_average_single_model_falls_to_else(self):
        """With only 1 model, average can't run — falls to single model logic."""
        cfg = _make_market_config(
            wf_best_model="average",
            threshold=0.5,
            saved_models=["mkt_lightgbm.joblib"],
        )
        preds = _run_strategy(
            "corners", cfg,
            [("mkt_lightgbm", 0.7, 0.8)],
        )
        assert len(preds) == 1
        assert preds[0]["probability"] == 0.7


# ---------------------------------------------------------------------------
# Disagree filtered strategies
# ---------------------------------------------------------------------------

class TestDisagreeFilteredStrategy:
    """Test disagree_*_filtered ensemble strategies."""

    def test_conservative_passes_when_models_agree(self):
        """Conservative: tight agreement (std < 0.08), high edge, prob in [0.55, 0.80]."""
        cfg = _make_market_config(wf_best_model="disagree_conservative_filtered", threshold=0.5)
        # Models agree tightly: std ~ 0.028
        preds = _run_strategy(
            "corners", cfg,
            [
                ("mkt_lightgbm", 0.68, 0.8),
                ("mkt_catboost", 0.72, 0.9),
                ("mkt_xgboost", 0.70, 0.85),
            ],
            # corners odds → implied prob ~ 0.5 → edge ~ 0.2
            match_odds={"corners_over_avg": 2.0},
        )
        assert len(preds) == 1
        expected = round((0.68 + 0.72 + 0.70) / 3, 4)
        assert preds[0]["probability"] == expected

    def test_conservative_rejected_models_disagree(self):
        """Conservative: high std → rejected."""
        cfg = _make_market_config(wf_best_model="disagree_conservative_filtered", threshold=0.5)
        # Models disagree: std > 0.08
        preds = _run_strategy(
            "corners", cfg,
            [
                ("mkt_lightgbm", 0.55, 0.8),
                ("mkt_catboost", 0.80, 0.9),
                ("mkt_xgboost", 0.60, 0.7),
            ],
            match_odds={"corners_over_avg": 2.0},
        )
        assert len(preds) == 0

    def test_aggressive_allows_wider_disagreement(self):
        """Aggressive: std < 0.15 passes (would fail conservative at 0.08)."""
        cfg = _make_market_config(wf_best_model="disagree_aggressive_filtered", threshold=0.4)
        # std ~ 0.10 — fails conservative (0.08) but passes aggressive (0.15)
        preds = _run_strategy(
            "corners", cfg,
            [
                ("mkt_lightgbm", 0.55, 0.8),
                ("mkt_catboost", 0.70, 0.9),
                ("mkt_xgboost", 0.58, 0.7),
            ],
            match_odds={"corners_over_avg": 2.0},
        )
        assert len(preds) == 1

    def test_balanced_rejected_low_edge(self):
        """Balanced: edge below 0.03 → rejected."""
        cfg = _make_market_config(wf_best_model="disagree_balanced_filtered", threshold=0.5)
        # Models agree, but avg_prob ~ 0.52 with implied ~ 0.50 → edge = 0.02 < 0.03
        preds = _run_strategy(
            "corners", cfg,
            [
                ("mkt_lightgbm", 0.52, 0.8),
                ("mkt_catboost", 0.53, 0.9),
                ("mkt_xgboost", 0.51, 0.7),
            ],
            match_odds={"corners_over_avg": 2.0},
        )
        assert len(preds) == 0

    def test_conservative_rejected_prob_out_of_range(self):
        """Conservative: prob > 0.80 → out of range → rejected."""
        cfg = _make_market_config(wf_best_model="disagree_conservative_filtered", threshold=0.5)
        # avg_prob ~ 0.85 > max_prob 0.80
        preds = _run_strategy(
            "corners", cfg,
            [
                ("mkt_lightgbm", 0.84, 0.8),
                ("mkt_catboost", 0.86, 0.9),
                ("mkt_xgboost", 0.85, 0.85),
            ],
            match_odds={"corners_over_avg": 2.0},
        )
        assert len(preds) == 0

    def test_aggressive_accepts_high_prob(self):
        """Aggressive: max_prob = 0.90, so 0.85 passes."""
        cfg = _make_market_config(wf_best_model="disagree_aggressive_filtered", threshold=0.5)
        preds = _run_strategy(
            "corners", cfg,
            [
                ("mkt_lightgbm", 0.84, 0.8),
                ("mkt_catboost", 0.86, 0.9),
                ("mkt_xgboost", 0.85, 0.85),
            ],
            match_odds={"corners_over_avg": 2.0},
        )
        assert len(preds) == 1


# ---------------------------------------------------------------------------
# Stacking strategy (regression test)
# ---------------------------------------------------------------------------

class TestStackingStrategy:
    """Regression tests for stacking ensemble."""

    def test_stacking_with_weights(self):
        """Stacking uses Ridge meta-learner weights (probability-space weighted average)."""
        # Use a non-niche market (cards/fouls have a niche override that replaces stacking result)
        cfg = _make_market_config(
            wf_best_model="stacking",
            threshold=0.3,
            stacking_weights={"lightgbm": 1.0, "catboost": 1.0, "xgboost": 1.0},
        )
        preds = _run_strategy(
            "corners", cfg,
            [
                ("mkt_lightgbm", 0.7, 0.8),
                ("mkt_catboost", 0.8, 0.9),
                ("mkt_xgboost", 0.6, 0.7),
            ],
        )
        assert len(preds) == 1
        # With equal weights [1,1,1], weighted avg = (0.7 + 0.8 + 0.6) / 3 = 0.7
        # Ridge is trained in probability space, so we use normalized weighted average.
        expected = round((1.0 * 0.7 + 1.0 * 0.8 + 1.0 * 0.6) / (1.0 + 1.0 + 1.0), 4)
        assert preds[0]["probability"] == expected

    def test_stacking_without_weights_falls_to_average(self):
        """Stacking without weights falls back to simple average."""
        cfg = _make_market_config(wf_best_model="stacking", threshold=0.3)
        preds = _run_strategy(
            "corners", cfg,
            [
                ("mkt_lightgbm", 0.7, 0.8),
                ("mkt_catboost", 0.8, 0.9),
                ("mkt_xgboost", 0.6, 0.7),
            ],
        )
        assert len(preds) == 1
        expected = round((0.7 + 0.8 + 0.6) / 3, 4)
        assert preds[0]["probability"] == expected

    def test_stacking_large_weights_no_sigmoid(self):
        """Stacking with large weights stays in probability range (no sigmoid)."""
        # Regression test: large Ridge coefficients previously caused sigmoid
        # to inflate probabilities to 95%+. The fix uses weighted average instead.
        cfg = _make_market_config(
            wf_best_model="stacking",
            threshold=0.3,
            stacking_weights={"lightgbm": 0.0, "catboost": 5.79, "xgboost": 0.0},
        )
        preds = _run_strategy(
            "corners", cfg,
            [
                ("mkt_lightgbm", 0.70, 0.4),
                ("mkt_catboost", 0.72, 0.44),
                ("mkt_xgboost", 0.68, 0.36),
            ],
        )
        assert len(preds) == 1
        # Weighted avg = (0*0.70 + 5.79*0.72 + 0*0.68) / 5.79 = 0.72
        # Old sigmoid: 1/(1+exp(-(5.79*0.72))) = 0.985 — WRONG
        prob = preds[0]["probability"]
        assert prob < 0.80, f"Probability {prob} too high — sigmoid regression?"
        assert abs(prob - 0.72) < 0.01

    def test_stacking_includes_fastai(self):
        """Stacking model_map includes fastai when it has non-zero weight."""
        cfg = _make_market_config(
            wf_best_model="stacking",
            threshold=0.3,
            stacking_weights={"lightgbm": 0.0, "catboost": 2.0, "xgboost": 0.0, "fastai": 3.0},
            saved_models=["mkt_lightgbm.joblib", "mkt_catboost.joblib",
                          "mkt_xgboost.joblib", "mkt_fastai.joblib"],
        )
        preds = _run_strategy(
            "corners", cfg,
            [
                ("mkt_lightgbm", 0.40, 0.1),
                ("mkt_catboost", 0.50, 0.1),
                ("mkt_xgboost", 0.45, 0.1),
                ("mkt_fastai", 0.60, 0.2),
            ],
        )
        assert len(preds) == 1
        # Weighted avg = (0*0.40 + 2.0*0.50 + 0*0.45 + 3.0*0.60) / (0+2+0+3) = 2.80/5 = 0.56
        prob = preds[0]["probability"]
        assert abs(prob - 0.56) < 0.01, f"Expected ~0.56 with fastai included, got {prob}"


# ---------------------------------------------------------------------------
# Agreement strategy (regression test)
# ---------------------------------------------------------------------------

class TestAgreementStrategy:
    """Regression tests for agreement ensemble."""

    def test_agreement_uses_min_prob(self):
        cfg = _make_market_config(wf_best_model="agreement", threshold=0.5)
        preds = _run_strategy(
            "corners", cfg,
            [
                ("mkt_lightgbm", 0.7, 0.8),
                ("mkt_catboost", 0.8, 0.9),
                ("mkt_xgboost", 0.65, 0.7),
            ],
        )
        assert len(preds) == 1
        assert preds[0]["probability"] == 0.65

    def test_agreement_below_threshold_skipped(self):
        cfg = _make_market_config(wf_best_model="agreement", threshold=0.75)
        preds = _run_strategy(
            "corners", cfg,
            [
                ("mkt_lightgbm", 0.7, 0.8),
                ("mkt_catboost", 0.8, 0.9),
                ("mkt_xgboost", 0.65, 0.7),
            ],
        )
        # min_prob = 0.65 < threshold 0.75 → skipped
        assert len(preds) == 0


# ---------------------------------------------------------------------------
# Single model fallback
# ---------------------------------------------------------------------------

class TestSingleModelFallback:
    """Test fallback to specific model name in else branch."""

    def test_specific_model_name_match(self):
        cfg = _make_market_config(wf_best_model="catboost", threshold=0.5)
        preds = _run_strategy(
            "fouls", cfg,
            [
                ("mkt_lightgbm", 0.6, 0.8),
                ("mkt_catboost", 0.75, 0.9),
                ("mkt_xgboost", 0.65, 0.7),
            ],
        )
        assert len(preds) == 1
        assert preds[0]["probability"] == 0.75

    def test_no_wf_best_uses_model_type(self):
        """When walkforward is empty, falls back to config 'model' field."""
        cfg = _make_market_config(threshold=0.5)
        cfg["model"] = "xgboost"
        # No walkforward at all
        preds = _run_strategy(
            "fouls", cfg,
            [
                ("mkt_lightgbm", 0.6, 0.8),
                ("mkt_catboost", 0.65, 0.7),
                ("mkt_xgboost", 0.72, 0.9),
            ],
        )
        assert len(preds) == 1
        assert preds[0]["probability"] == 0.72
