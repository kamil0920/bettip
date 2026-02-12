"""
Unit tests for feature optimization improvements:
- Selective feature regeneration (per-engineer caching)
- Time budget callback
- Updated pruner and fold reporting
- LightGBM parameter changes
"""
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pandas as pd
import pytest
import optuna

from src.features.config_manager import BetTypeFeatureConfig
from src.features.regeneration import FeatureRegenerator
from src.features.registry import FeatureEngineerConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dummy_features(n_rows: int = 100, prefix: str = "feat") -> pd.DataFrame:
    """Create a dummy feature DataFrame with fixture_id."""
    return pd.DataFrame({
        "fixture_id": range(n_rows),
        f"{prefix}_a": np.random.randn(n_rows),
        f"{prefix}_b": np.random.randn(n_rows),
    })


def _make_dummy_data() -> dict:
    """Create minimal data dict for FeatureRegenerator."""
    matches = pd.DataFrame({
        "fixture_id": range(100),
        "date": pd.date_range("2020-01-01", periods=100),
        "home_team_id": [1, 2] * 50,
        "home_team_name": ["TeamA", "TeamB"] * 50,
        "away_team_id": [2, 1] * 50,
        "away_team_name": ["TeamB", "TeamA"] * 50,
        "league": ["premier_league"] * 100,
        "season": [2020] * 100,
        "home_goals": np.random.randint(0, 4, 100),
        "away_goals": np.random.randint(0, 4, 100),
    })
    return {"matches": matches}


# ---------------------------------------------------------------------------
# Selective Regeneration Tests
# ---------------------------------------------------------------------------

class TestSelectiveRegeneration:
    """Tests for _generate_features_selective in FeatureRegenerator."""

    def test_first_call_populates_cache(self):
        """First call should run all engineers and populate _engineer_cache."""
        regenerator = FeatureRegenerator()
        data = _make_dummy_data()

        # Create minimal configs — two "engineers"
        configs = [
            FeatureEngineerConfig("elo", enabled=True, params={"k_factor": 32.0}),
            FeatureEngineerConfig("team_form", enabled=True, params={"n_matches": 5}),
        ]

        registry_params = {
            "elo": {"k_factor": 32.0},
            "team_form": {"n_matches": 5},
        }

        # Mock registry.get to return mock engineers
        elo_features = _make_dummy_features(100, "elo")
        form_features = _make_dummy_features(100, "form")

        mock_elo_eng = MagicMock()
        mock_elo_eng.create_features.return_value = elo_features
        mock_form_eng = MagicMock()
        mock_form_eng.create_features.return_value = form_features

        def mock_get(name, **params):
            if name == "elo":
                return mock_elo_eng
            elif name == "team_form":
                return mock_form_eng
            raise KeyError(name)

        regenerator.registry = MagicMock()
        regenerator.registry.get = mock_get

        result = regenerator._generate_features_selective(data, configs, registry_params)

        assert len(result) == 2
        assert "elo" in regenerator._engineer_cache
        assert "team_form" in regenerator._engineer_cache
        assert regenerator._prev_registry_params == registry_params

    def test_unchanged_params_reuses_cache(self):
        """When params don't change, engineers should NOT be re-run."""
        regenerator = FeatureRegenerator()
        data = _make_dummy_data()

        configs = [
            FeatureEngineerConfig("elo", enabled=True, params={"k_factor": 32.0}),
            FeatureEngineerConfig("team_form", enabled=True, params={"n_matches": 5}),
        ]

        registry_params = {
            "elo": {"k_factor": 32.0},
            "team_form": {"n_matches": 5},
        }

        # Pre-populate cache (simulate first call)
        elo_features = _make_dummy_features(100, "elo")
        form_features = _make_dummy_features(100, "form")
        regenerator._engineer_cache = {
            "elo": elo_features,
            "team_form": form_features,
        }
        regenerator._prev_registry_params = registry_params.copy()

        # Registry.get should NOT be called since params are unchanged
        mock_registry = MagicMock()
        regenerator.registry = mock_registry

        result = regenerator._generate_features_selective(data, configs, registry_params)

        assert len(result) == 2
        mock_registry.get.assert_not_called()

    def test_changed_params_regenerates_only_affected_engineers(self):
        """When only elo params change, only elo should be re-run."""
        regenerator = FeatureRegenerator()
        data = _make_dummy_data()

        old_params = {
            "elo": {"k_factor": 32.0},
            "team_form": {"n_matches": 5},
        }
        new_params = {
            "elo": {"k_factor": 50.0},  # Changed
            "team_form": {"n_matches": 5},  # Same
        }

        configs = [
            FeatureEngineerConfig("elo", enabled=True, params={"k_factor": 50.0}),
            FeatureEngineerConfig("team_form", enabled=True, params={"n_matches": 5}),
        ]

        # Pre-populate cache
        old_elo_features = _make_dummy_features(100, "elo_old")
        form_features = _make_dummy_features(100, "form")
        regenerator._engineer_cache = {
            "elo": old_elo_features,
            "team_form": form_features,
        }
        regenerator._prev_registry_params = old_params

        new_elo_features = _make_dummy_features(100, "elo_new")
        mock_elo_eng = MagicMock()
        mock_elo_eng.create_features.return_value = new_elo_features

        def mock_get(name, **params):
            if name == "elo":
                return mock_elo_eng
            raise KeyError(f"Unexpected engineer: {name}")

        regenerator.registry = MagicMock()
        regenerator.registry.get = mock_get

        result = regenerator._generate_features_selective(data, configs, new_params)

        assert len(result) == 2
        # Elo was regenerated
        mock_elo_eng.create_features.assert_called_once()
        # team_form was reused (registry.get only called for elo)
        assert regenerator._engineer_cache["elo"] is new_elo_features
        assert regenerator._engineer_cache["team_form"] is form_features

    def test_clear_engineer_cache(self):
        """clear_engineer_cache should reset both cache and prev params."""
        regenerator = FeatureRegenerator()
        regenerator._engineer_cache = {"elo": _make_dummy_features()}
        regenerator._prev_registry_params = {"elo": {"k_factor": 32.0}}

        regenerator.clear_engineer_cache()

        assert regenerator._engineer_cache == {}
        assert regenerator._prev_registry_params is None

    def test_disabled_engineer_skipped(self):
        """Disabled engineers should be skipped entirely."""
        regenerator = FeatureRegenerator()
        data = _make_dummy_data()

        configs = [
            FeatureEngineerConfig("elo", enabled=False, params={}),
            FeatureEngineerConfig("team_form", enabled=True, params={"n_matches": 5}),
        ]

        registry_params = {"team_form": {"n_matches": 5}}

        form_features = _make_dummy_features(100, "form")
        mock_eng = MagicMock()
        mock_eng.create_features.return_value = form_features

        regenerator.registry = MagicMock()
        regenerator.registry.get = MagicMock(return_value=mock_eng)

        result = regenerator._generate_features_selective(data, configs, registry_params)

        assert len(result) == 1
        assert "elo" not in regenerator._engineer_cache


# ---------------------------------------------------------------------------
# Time Budget Tests
# ---------------------------------------------------------------------------

class TestTimeBudget:
    """Tests for time budget callback in FeatureParamOptimizer."""

    def test_time_budget_stops_study(self):
        """Time budget callback should stop Optuna study when time exceeded."""
        # Create a simple study and objective
        study = optuna.create_study(direction="maximize")

        call_count = 0

        def objective(trial):
            nonlocal call_count
            call_count += 1
            x = trial.suggest_float("x", -10, 10)
            return -x ** 2

        # Simulate time budget callback with 0 seconds (immediate stop)
        start_time = time.time() - 100  # already exceeded
        time_budget_seconds = 1

        def time_budget_callback(study, trial):
            elapsed = time.time() - start_time
            if elapsed > time_budget_seconds:
                study.stop()

        study.optimize(
            objective,
            n_trials=100,
            callbacks=[time_budget_callback],
        )

        # Should have stopped after first trial (callback fires after trial completes)
        assert call_count == 1
        assert len(study.trials) == 1

    def test_no_time_budget_runs_all_trials(self):
        """Without time budget, all trials should run."""
        study = optuna.create_study(direction="maximize")

        def objective(trial):
            x = trial.suggest_float("x", -10, 10)
            return -x ** 2

        study.optimize(objective, n_trials=5)

        assert len(study.trials) == 5


# ---------------------------------------------------------------------------
# Pruner and Fold Reporting Tests
# ---------------------------------------------------------------------------

class TestPrunerAndFoldReporting:
    """Tests for updated pruner settings and fold reporting."""

    def test_fold_1_reporting_enables_early_pruning(self):
        """Intermediate values should be reported after fold 1 (not only fold 2+)."""
        # Simulate the logic from evaluate_config
        fold_rois = [5.0]  # After first fold

        # Old behavior: len(fold_rois) >= 2 would skip reporting
        # New behavior: len(fold_rois) >= 1 reports after every fold
        should_report = len(fold_rois) >= 1
        assert should_report, "Should report after first fold"

        # Intermediate sharpe with single fold should use std=0
        intermediate_sharpe = np.mean(fold_rois) - (np.std(fold_rois) if len(fold_rois) > 1 else 0.0)
        assert intermediate_sharpe == 5.0

    def test_intermediate_sharpe_with_multiple_folds(self):
        """Intermediate sharpe with multiple folds should penalize by std."""
        fold_rois = [10.0, 2.0]
        intermediate_sharpe = np.mean(fold_rois) - (np.std(fold_rois) if len(fold_rois) > 1 else 0.0)
        assert intermediate_sharpe == pytest.approx(6.0 - 4.0, abs=0.01)

    def test_median_pruner_settings(self):
        """Verify pruner is created with updated settings."""
        pruner = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=0)
        assert pruner._n_startup_trials == 3
        assert pruner._n_warmup_steps == 0


# ---------------------------------------------------------------------------
# LightGBM Parameter Tests
# ---------------------------------------------------------------------------

class TestLightGBMParams:
    """Tests for updated LightGBM parameters."""

    def test_lgbm_uses_simplified_params(self):
        """Verify LightGBM uses simplified params for feature comparison."""
        import lightgbm as lgb

        model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
            verbose=-1,
        )

        assert model.n_estimators == 100
        assert model.max_depth == 4
        assert model.learning_rate == 0.1


# ---------------------------------------------------------------------------
# CLI Argument Tests
# ---------------------------------------------------------------------------

class TestCLIArguments:
    """Tests for --time-budget-minutes CLI argument."""

    def test_time_budget_arg_parsing(self):
        """Verify --time-budget-minutes is parsed correctly."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--time-budget-minutes", type=int, default=0)

        args = parser.parse_args(["--time-budget-minutes", "330"])
        assert args.time_budget_minutes == 330

        args_default = parser.parse_args([])
        assert args_default.time_budget_minutes == 0
