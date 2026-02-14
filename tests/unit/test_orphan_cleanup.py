"""Tests for orphan model cleanup scoping.

Regression tests for the bug where orphan cleanup deleted models for markets
NOT in the current run, causing cascading model loss across multiple runs.
"""

import sys
from pathlib import Path

import pytest

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root / 'scripts'))

from cleanup_orphan_models import (  # noqa: E402
    KNOWN_MARKETS,
    SNIPER_MODEL_PATTERN,
    find_orphans,
    get_model_market,
    get_referenced_models,
)


class TestGetModelMarket:
    """Test market name extraction from model filenames."""

    def test_simple_market(self):
        assert get_model_market("home_win_lightgbm.joblib") == "home_win"

    def test_line_variant(self):
        assert get_model_market("corners_over_85_catboost.joblib") == "corners_over_85"

    def test_greedy_match_prefers_longest(self):
        """'shots_over_255_xgboost' should match 'shots_over_255', not 'shots'."""
        assert get_model_market("shots_over_255_xgboost.joblib") == "shots_over_255"

    def test_fouls_variants(self):
        assert get_model_market("fouls_over_235_lightgbm.joblib") == "fouls_over_235"
        assert get_model_market("fouls_over_265_lightgbm.joblib") == "fouls_over_265"
        assert get_model_market("fouls_lightgbm.joblib") == "fouls"

    def test_two_stage_model(self):
        assert get_model_market("home_win_two_stage_lgb.joblib") == "home_win"
        assert get_model_market("home_win_two_stage_xgb.joblib") == "home_win"

    def test_unknown_model_returns_none(self):
        assert get_model_market("unknown_market_lightgbm.joblib") is None

    def test_non_sniper_model_matches_prefix(self):
        """get_model_market does prefix matching; SNIPER_MODEL_PATTERN guards against non-sniper."""
        # 'fouls_over_24_5_model.joblib' has 'fouls' as prefix, so get_model_market returns 'fouls'.
        # The regex pattern (not get_model_market) is what rejects this file as a non-sniper model.
        assert get_model_market("fouls_over_24_5_model.joblib") == "fouls"


class TestSniperModelPattern:
    """Test the regex pattern that identifies sniper-produced models."""

    def test_matches_standard_models(self):
        assert SNIPER_MODEL_PATTERN.match("home_win_lightgbm.joblib")
        assert SNIPER_MODEL_PATTERN.match("btts_catboost.joblib")
        assert SNIPER_MODEL_PATTERN.match("corners_over_85_fastai.joblib")

    def test_matches_two_stage(self):
        assert SNIPER_MODEL_PATTERN.match("home_win_two_stage_lgb.joblib")
        assert SNIPER_MODEL_PATTERN.match("home_win_two_stage_xgb.joblib")

    def test_rejects_non_sniper(self):
        assert not SNIPER_MODEL_PATTERN.match("fouls_over_24_5_model.joblib")
        assert not SNIPER_MODEL_PATTERN.match("random_file.joblib")


class TestGetReferencedModels:
    """Test extraction of model filenames from deployment config."""

    def test_extracts_filenames(self):
        config = {
            "markets": {
                "home_win": {"saved_models": ["home_win_lightgbm.joblib", "home_win_catboost.joblib"]},
                "btts": {"saved_models": ["btts_xgboost.joblib"]},
            }
        }
        refs = get_referenced_models(config)
        assert refs == {"home_win_lightgbm.joblib", "home_win_catboost.joblib", "btts_xgboost.joblib"}

    def test_handles_path_prefix(self):
        config = {
            "markets": {
                "shots": {"saved_models": ["models/shots_lightgbm.joblib"]},
            }
        }
        refs = get_referenced_models(config)
        assert "shots_lightgbm.joblib" in refs

    def test_empty_config(self):
        assert get_referenced_models({"markets": {}}) == set()


class TestFindOrphans:
    """Test orphan detection with and without market scoping."""

    def _hub_models(self, *names):
        return [f"models/{n}" for n in names]

    def test_finds_unreferenced_models(self):
        hub = self._hub_models(
            "home_win_lightgbm.joblib",
            "home_win_catboost.joblib",
            "btts_xgboost.joblib",
        )
        referenced = {"home_win_lightgbm.joblib"}
        orphans = find_orphans(hub, referenced)
        assert len(orphans) == 2
        assert "models/home_win_catboost.joblib" in orphans
        assert "models/btts_xgboost.joblib" in orphans

    def test_no_orphans_when_all_referenced(self):
        hub = self._hub_models("home_win_lightgbm.joblib", "btts_xgboost.joblib")
        referenced = {"home_win_lightgbm.joblib", "btts_xgboost.joblib"}
        assert find_orphans(hub, referenced) == []

    def test_scoped_to_current_run_markets(self):
        """The key regression test: orphan cleanup scoped to specific markets."""
        hub = self._hub_models(
            "home_win_lightgbm.joblib",  # unreferenced, IN scope
            "home_win_catboost.joblib",  # unreferenced, IN scope
            "cards_lightgbm.joblib",     # unreferenced, NOT in scope
            "shots_xgboost.joblib",      # unreferenced, NOT in scope
            "btts_fastai.joblib",        # referenced
        )
        referenced = {"btts_fastai.joblib"}

        # Without scoping — all 4 unreferenced are orphans
        orphans_all = find_orphans(hub, referenced, scope_markets=None)
        assert len(orphans_all) == 4

        # With scoping to home_win only — only 2 home_win models are orphans
        orphans_scoped = find_orphans(hub, referenced, scope_markets={"home_win"})
        assert len(orphans_scoped) == 2
        assert all("home_win" in o for o in orphans_scoped)

    def test_scope_preserves_other_markets(self):
        """Models for markets NOT in the current run must never be deleted."""
        hub = self._hub_models(
            "cards_lightgbm.joblib",
            "cards_catboost.joblib",
            "cards_xgboost.joblib",
            "cards_fastai.joblib",
            "shots_lightgbm.joblib",
            "shots_catboost.joblib",
            "home_win_lightgbm.joblib",
        )
        # Imagine a run that only optimized "home_win" and produced a new lightgbm
        # The new config only references home_win_lightgbm (the rest are from other markets)
        referenced = {"home_win_lightgbm.joblib"}

        # Scoped to home_win — should NOT touch cards or shots
        orphans = find_orphans(hub, referenced, scope_markets={"home_win"})
        assert orphans == []  # home_win_lightgbm is referenced, no orphans

    def test_scope_with_line_variants(self):
        """Line variant markets are correctly scoped."""
        hub = self._hub_models(
            "fouls_over_235_lightgbm.joblib",  # unreferenced, IN scope
            "fouls_over_235_catboost.joblib",   # referenced
            "fouls_over_265_lightgbm.joblib",   # unreferenced, NOT in scope
            "fouls_lightgbm.joblib",            # unreferenced, NOT in scope
        )
        referenced = {"fouls_over_235_catboost.joblib"}

        orphans = find_orphans(hub, referenced, scope_markets={"fouls_over_235"})
        assert len(orphans) == 1
        assert "fouls_over_235_lightgbm.joblib" in orphans[0]

    def test_scope_multiple_markets(self):
        """Scoping to multiple markets works correctly."""
        hub = self._hub_models(
            "home_win_lightgbm.joblib",  # unreferenced, IN scope
            "over25_lightgbm.joblib",    # unreferenced, IN scope
            "btts_lightgbm.joblib",      # unreferenced, NOT in scope
        )
        referenced = set()

        orphans = find_orphans(hub, referenced, scope_markets={"home_win", "over25"})
        assert len(orphans) == 2
        names = {o.split("/")[-1] for o in orphans}
        assert "home_win_lightgbm.joblib" in names
        assert "over25_lightgbm.joblib" in names
        assert "btts_lightgbm.joblib" not in names

    def test_ignores_non_sniper_models(self):
        """Non-sniper pattern models are never orphan candidates."""
        hub = self._hub_models(
            "fouls_over_24_5_model.joblib",  # non-sniper pattern
            "home_win_lightgbm.joblib",      # sniper pattern, unreferenced
        )
        referenced = set()
        orphans = find_orphans(hub, referenced)
        assert len(orphans) == 1
        assert "home_win_lightgbm.joblib" in orphans[0]
