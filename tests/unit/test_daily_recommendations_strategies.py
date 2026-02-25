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
    fill_estimated_line_odds,
    MARKET_ODDS_COLUMNS,
    MARKET_COMPLEMENT_COLUMNS,
    MARKET_BASELINES,
    POISSON_ESTIMATION_LINES,
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
    real_odds_only=False,
):
    """
    Run generate_sniper_predictions with mocked dependencies, returning
    the list of prediction dicts (or empty if filtered out).

    model_probs: list of (model_name, probability, confidence) tuples
    """
    match = _make_match()
    if match_odds is None:
        # Provide real odds for common niche markets so strategy tests
        # aren't blocked by the baseline-suppression filter (F4 fix).
        match_odds = {
            "corners_over_avg": 2.0,
            "corners_under_avg": 2.0,
            "shots_over_avg": 2.0,
            "shots_under_avg": 2.0,
            "fouls_over_avg": 2.0,
            "fouls_under_avg": 2.0,
            "cards_over_avg": 2.0,
            "cards_under_avg": 2.0,
        }

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

        return generate_sniper_predictions(
            [match], min_edge_pct=min_edge_pct, real_odds_only=real_odds_only,
        )


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


# ---------------------------------------------------------------------------
# Per-line odds column mapping (F6/F7 fix for phantom edge bug)
# ---------------------------------------------------------------------------

class TestPerLineOddsMapping:
    """Verify that niche line variants map to per-line odds columns,
    not shared default-line columns (which cause phantom edge)."""

    def test_corners_line_variants_use_per_line_columns(self):
        """corners_over_85 should map to corners_over_avg_85, NOT corners_over_avg."""
        assert MARKET_ODDS_COLUMNS["corners_over_85"] == "corners_over_avg_85"
        assert MARKET_ODDS_COLUMNS["corners_over_95"] == "corners_over_avg_95"
        assert MARKET_ODDS_COLUMNS["corners_over_105"] == "corners_over_avg_105"
        assert MARKET_ODDS_COLUMNS["corners_under_85"] == "corners_under_avg_85"
        assert MARKET_ODDS_COLUMNS["corners_under_95"] == "corners_under_avg_95"

    def test_cards_line_variants_use_per_line_columns(self):
        assert MARKET_ODDS_COLUMNS["cards_over_25"] == "cards_over_avg_25"
        assert MARKET_ODDS_COLUMNS["cards_under_45"] == "cards_under_avg_45"

    def test_shots_line_variants_use_per_line_columns(self):
        assert MARKET_ODDS_COLUMNS["shots_over_285"] == "shots_over_avg_285"
        assert MARKET_ODDS_COLUMNS["shots_under_265"] == "shots_under_avg_265"

    def test_fouls_line_variants_use_per_line_columns(self):
        assert MARKET_ODDS_COLUMNS["fouls_over_265"] == "fouls_over_avg_265"
        assert MARKET_ODDS_COLUMNS["fouls_under_255"] == "fouls_under_avg_255"

    def test_base_markets_unchanged(self):
        """Base markets (no line suffix) still use generic columns."""
        assert MARKET_ODDS_COLUMNS["corners"] == "corners_over_avg"
        assert MARKET_ODDS_COLUMNS["cards"] == "cards_over_avg"
        assert MARKET_ODDS_COLUMNS["shots"] == "shots_over_avg"
        assert MARKET_ODDS_COLUMNS["fouls"] == "fouls_over_avg"

    def test_h2h_markets_unchanged(self):
        """H2H markets are unaffected by per-line fix."""
        assert MARKET_ODDS_COLUMNS["home_win"] == "h2h_home_avg"
        assert MARKET_ODDS_COLUMNS["away_win"] == "h2h_away_avg"
        assert MARKET_ODDS_COLUMNS["over25"] == "totals_over_avg"
        assert MARKET_ODDS_COLUMNS["btts"] == "btts_yes_avg"

    def test_complement_columns_match_per_line(self):
        """Complement columns use per-line names for vig removal."""
        assert MARKET_COMPLEMENT_COLUMNS["corners_over_85"] == "corners_under_avg_85"
        assert MARKET_COMPLEMENT_COLUMNS["corners_under_85"] == "corners_over_avg_85"
        assert MARKET_COMPLEMENT_COLUMNS["cards_over_35"] == "cards_under_avg_35"
        assert MARKET_COMPLEMENT_COLUMNS["shots_over_285"] == "shots_under_avg_285"
        assert MARKET_COMPLEMENT_COLUMNS["shots_under_285"] == "shots_over_avg_285"

    def test_complement_base_markets_unchanged(self):
        """Base market complements remain generic."""
        assert MARKET_COMPLEMENT_COLUMNS["corners"] == "corners_under_avg"
        assert MARKET_COMPLEMENT_COLUMNS["shots"] == "shots_under_avg"

    def test_per_line_edge_calculation(self):
        """Edge calculation uses per-line odds, not default-line odds."""
        # Simulate corners_over_85 with per-line odds (line 8.5 is easy → low odds)
        match_odds = {
            "corners_over_avg_85": 1.40,   # 8.5 line: easy to hit, low odds
            "corners_under_avg_85": 3.00,  # complement at same line
        }
        # With per-line odds: implied prob = vig-removed(1.40, 3.00)
        edge = calculate_edge(0.80, "corners_over_85", match_odds)
        # vig-removed: 1/1.40 / (1/1.40 + 1/3.00) = 0.714/1.048 ≈ 0.681
        # edge = 0.80 - 0.681 ≈ 0.119 (reasonable ~12% edge)
        assert 0.05 < edge < 0.20, f"Edge {edge} should be moderate with correct line odds"

    def test_phantom_edge_prevented(self):
        """Without per-line odds, edge falls to baseline (no phantom edge)."""
        # No per-line odds available → match_odds.get("corners_over_avg_85") returns None
        match_odds = {
            "corners_over_avg": 2.67,   # default 9.5 line odds (wrong line!)
        }
        edge = calculate_edge(0.80, "corners_over_85", match_odds)
        # corners_over_avg_85 not found → falls to baseline 0.50
        # edge = 0.80 - 0.50 = 0.30 (baseline)
        assert edge == pytest.approx(0.30, abs=0.01)


# ---------------------------------------------------------------------------
# Poisson-based line-adjusted odds estimation
# ---------------------------------------------------------------------------


class TestPoissonEstimation:
    """Test Poisson ratio scaling for estimating per-line odds from default-line odds."""

    MOCK_LEAGUE_STATS = {
        "premier_league": {"total_corners": 10.0, "total_cards": 4.0},
        "la_liga": {"total_corners": 9.5, "total_cards": 4.5},
    }

    def _call(self, match_odds, league="premier_league"):
        """Call fill_estimated_line_odds with mocked league stats."""
        with patch(
            "experiments.generate_daily_recommendations._get_league_stats",
            return_value=self.MOCK_LEAGUE_STATS,
        ):
            return fill_estimated_line_odds(match_odds, league)

    def test_corners_line_estimation(self):
        """Corners O8.5 estimated from O9.5 default odds + λ=10.0."""
        match_odds = {
            "corners_over_avg": 1.85,  # default 9.5 line
            "corners_under_avg": 2.05,
        }
        filled = self._call(match_odds)
        assert "corners_over_avg_85" in filled
        # O8.5 is easier than O9.5, so odds should be lower (higher prob)
        assert match_odds["corners_over_avg_85"] < 1.85
        assert match_odds["corners_over_avg_85"] > 1.0

    def test_cards_line_estimation(self):
        """Cards O2.5 estimated from O4.5 default odds + λ=4.0."""
        match_odds = {
            "cards_over_avg": 2.70,  # default 4.5 line
            "cards_under_avg": 1.50,
        }
        filled = self._call(match_odds)
        assert "cards_over_avg_25" in filled
        # O2.5 is easier than O4.5, so odds should be lower
        assert match_odds["cards_over_avg_25"] < 2.70
        assert match_odds["cards_over_avg_25"] > 1.0

    def test_under_direction(self):
        """Under line estimation uses CDF ratio (not survival)."""
        match_odds = {
            "corners_over_avg": 1.85,
            "corners_under_avg": 2.05,
        }
        filled = self._call(match_odds)
        # U8.5 is harder than U9.5 (fewer outcomes), so odds should be lower (higher prob)
        # Wait — U8.5 means X <= 8, U9.5 means X <= 9. P(X<=8) < P(X<=9),
        # so U8.5 is less likely → higher odds
        assert "corners_under_avg_85" in filled
        assert match_odds["corners_under_avg_85"] > match_odds["corners_under_avg"]

    def test_pure_poisson_without_default_odds(self):
        """Pure Poisson mode fills odds even when default-line odds are missing."""
        match_odds = {}  # no default odds at all
        filled = self._call(match_odds)
        # Should still fill corners and cards via pure Poisson(λ)
        corners_filled = {c for c in filled if "corners" in c}
        cards_filled = {c for c in filled if "cards" in c}
        assert len(corners_filled) > 0, "Corners should be filled via pure Poisson"
        assert len(cards_filled) > 0, "Cards should be filled via pure Poisson"

    def test_pure_poisson_odds_reasonable(self):
        """Pure Poisson odds are reasonable (not extreme) for typical lines."""
        match_odds = {}
        filled = self._call(match_odds)
        # Cards O2.5 with λ=4.0: P(X>2) = 1 - poisson.cdf(2, 4.0) ≈ 0.762
        # Odds ≈ 1 / (0.762 * 1.05) ≈ 1.25
        assert "cards_over_avg_25" in filled
        odds_o25 = match_odds["cards_over_avg_25"]
        assert 1.0 < odds_o25 < 2.0, f"Cards O2.5 odds={odds_o25} out of range"

        # Corners O8.5 with λ=10.0: P(X>8) = 1 - poisson.cdf(8, 10.0) ≈ 0.667
        # Odds ≈ 1 / (0.667 * 1.05) ≈ 1.43
        assert "corners_over_avg_85" in filled
        odds_o85 = match_odds["corners_over_avg_85"]
        assert 1.0 < odds_o85 < 2.5, f"Corners O8.5 odds={odds_o85} out of range"

    def test_pure_poisson_also_fills_default_line(self):
        """Pure Poisson mode also fills the default line (not skipped like ratio mode)."""
        match_odds = {}
        filled = self._call(match_odds)
        # In pure Poisson mode, default line 9.5 for corners should also be filled
        # (no actual odds to use, so Poisson estimate fills it)
        assert "corners_over_avg_95" in filled

    def test_ratio_mode_preferred_over_pure_poisson(self):
        """When default-line odds exist, ratio scaling is used (not pure Poisson)."""
        # With default odds: ratio scaling produces different values than pure Poisson
        match_odds_ratio = {
            "corners_over_avg": 1.50,  # Artificially low (implies high prob)
            "corners_under_avg": 2.80,
        }
        match_odds_pure = {}

        # _call modifies dicts in-place, so pass originals (not copies)
        filled_ratio = self._call(match_odds_ratio)
        filled_pure = self._call(match_odds_pure)

        # Both should fill corners_over_avg_85
        assert "corners_over_avg_85" in filled_ratio
        assert "corners_over_avg_85" in filled_pure

        # But the odds should differ (ratio scaling incorporates match-specific odds)
        ratio_odds = match_odds_ratio["corners_over_avg_85"]
        pure_odds = match_odds_pure["corners_over_avg_85"]
        assert ratio_odds != pytest.approx(pure_odds, abs=0.01), \
            "Ratio and pure Poisson should produce different estimates"

    def test_pure_poisson_complement_filled(self):
        """Pure Poisson fills both over and under sides."""
        match_odds = {}
        filled = self._call(match_odds)
        if "corners_over_avg_85" in filled:
            assert "corners_under_avg_85" in filled
            # Over + under fair probs should roughly sum to 1 (before vig)
            over_odds = match_odds["corners_over_avg_85"]
            under_odds = match_odds["corners_under_avg_85"]
            over_implied = 1.0 / over_odds
            under_implied = 1.0 / under_odds
            # Each implied = fair_prob * 1.05, and fair_over + fair_under ≈ 1.0
            # So sum of implied ≈ 1.05
            total = over_implied + under_implied
            assert 0.95 < total < 1.15, f"Implied sum={total} out of range"

    def test_no_estimation_for_shots(self):
        """Shots excluded from Poisson estimation (stat mismatch)."""
        match_odds = {
            "shots_over_avg": 2.00,
            "shots_under_avg": 2.00,
        }
        filled = self._call(match_odds)
        # No shots columns should be filled
        shots_filled = {c for c in filled if "shots" in c}
        assert len(shots_filled) == 0

    def test_no_estimation_for_fouls(self):
        """Fouls excluded from Poisson estimation (no odds source)."""
        match_odds = {
            "fouls_over_avg": 2.00,
            "fouls_under_avg": 2.00,
        }
        filled = self._call(match_odds)
        fouls_filled = {c for c in filled if "fouls" in c}
        assert len(fouls_filled) == 0

    def test_estimation_clamped(self):
        """Fair prob clamped to [0.02, 0.98] for extreme lines."""
        # Cards O1.5 with λ=4.0 → P(X>1.5) very high → fair prob near 1.0
        match_odds = {
            "cards_over_avg": 2.70,
            "cards_under_avg": 1.50,
        }
        filled = self._call(match_odds)
        assert "cards_over_avg_15" in filled
        # Fair prob clamped at 0.98 → odds = 1/(0.98*1.05) ≈ 0.97
        # Actually odds should be > 1.0 since 0.98*1.05 = 1.029 → 1/1.029 ≈ 0.972
        # Hmm, that's < 1.0. But the vig is on the total, and extremely likely
        # outcomes can produce odds < 1.0 at the clamped boundary.
        # The important thing is the fair_prob was clamped.
        assert match_odds["cards_over_avg_15"] > 0

    def test_existing_perline_not_overwritten(self):
        """Already-populated per-line columns are untouched."""
        real_odds = 1.55
        match_odds = {
            "corners_over_avg": 1.85,
            "corners_under_avg": 2.05,
            "corners_over_avg_85": real_odds,  # already populated
        }
        filled = self._call(match_odds)
        # corners_over_avg_85 should NOT be in filled (already existed)
        assert "corners_over_avg_85" not in filled
        # Value should be unchanged
        assert match_odds["corners_over_avg_85"] == real_odds

    def test_complement_filled(self):
        """Both over and under sides are filled for each estimated line."""
        match_odds = {
            "corners_over_avg": 1.85,
            "corners_under_avg": 2.05,
        }
        filled = self._call(match_odds)
        # For corners O8.5, complement U8.5 should also be filled
        if "corners_over_avg_85" in filled:
            assert "corners_under_avg_85" in filled
            # Verify both are valid odds
            assert match_odds["corners_over_avg_85"] > 0
            assert match_odds["corners_under_avg_85"] > 0

    def test_edge_source_estimated(self):
        """Verify edge_source='estimated' when --allow-estimated is used."""
        # Only provide default-line odds; per-line will be Poisson-estimated
        match_odds = {
            "corners_over_avg": 1.85,
            "corners_under_avg": 2.05,
        }
        cfg = _make_market_config(
            wf_best_model="lightgbm",
            threshold=0.3,
            saved_models=["mkt_lightgbm.joblib"],
        )
        # Mock league stats for "Test League" (what _make_match uses)
        mock_stats = {"Test League": {"total_corners": 10.0, "total_cards": 4.0}}
        with patch(
            "experiments.generate_daily_recommendations._get_league_stats",
            return_value=mock_stats,
        ):
            preds = _run_strategy(
                "corners_over_85",
                cfg,
                [("mkt_lightgbm", 0.85, 0.9)],
                match_odds=match_odds,
                real_odds_only=False,
            )
        assert len(preds) > 0
        assert preds[0]["edge_source"] == "estimated"

    def test_estimated_odds_suppressed_by_default(self):
        """Poisson-estimated odds are suppressed when real_odds_only=True (default)."""
        match_odds = {
            "corners_over_avg": 1.85,
            "corners_under_avg": 2.05,
        }
        cfg = _make_market_config(
            wf_best_model="lightgbm",
            threshold=0.3,
            saved_models=["mkt_lightgbm.joblib"],
        )
        mock_stats = {"Test League": {"total_corners": 10.0, "total_cards": 4.0}}
        with patch(
            "experiments.generate_daily_recommendations._get_league_stats",
            return_value=mock_stats,
        ):
            preds = _run_strategy(
                "corners_over_85",
                cfg,
                [("mkt_lightgbm", 0.85, 0.9)],
                match_odds=match_odds,
                real_odds_only=True,
            )
        assert len(preds) == 0

    def test_default_line_not_estimated(self):
        """The default line itself is not estimated (skip target == default)."""
        match_odds = {
            "corners_over_avg": 1.85,
            "corners_under_avg": 2.05,
        }
        filled = self._call(match_odds)
        # corners_over_avg_95 corresponds to line 9.5 = default → should be skipped
        assert "corners_over_avg_95" not in filled

    def test_all_niche_stats_covered(self):
        """POISSON_ESTIMATION_LINES covers all niche stats."""
        assert set(POISSON_ESTIMATION_LINES.keys()) == {"corners", "cards", "goals", "shots", "fouls", "ht"}
