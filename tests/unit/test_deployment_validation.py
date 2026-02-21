"""Tests for deployment config validation in generate_deployment_config.py."""
import sys
from pathlib import Path

import pytest

# Ensure project root is on sys.path so we can import the script
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root / 'scripts'))

from generate_deployment_config import validate_config  # noqa: E402


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
    }
    cfg.update(extra)
    return cfg


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEnabledButEmptyModels:
    def test_warns_when_enabled_and_no_saved_models(self):
        config = _make_config({
            "home_win": _market(enabled=True, saved_models=[]),
        })
        warnings = validate_config(config)
        assert len(warnings) == 1
        assert "Enabled but saved_models is empty" in warnings[0]
        assert "home_win" in warnings[0]

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
        assert len(warnings) == 2  # home_win empty + shots extra models
        assert any("home_win" in w for w in warnings)
        assert any("shots" in w for w in warnings)

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
            "shots_under_285": _market(
                enabled=True,
                model="lightgbm",
                saved_models=["shots_under_285_lightgbm.joblib"],
                n_bets=6,
            ),
        })
        warnings = validate_config(config, min_n_bets=20)
        assert any("BLOCKED" in w and "6 holdout bets" in w for w in warnings)
        assert config["markets"]["shots_under_285"]["enabled"] is False

    def test_ok_above_min_bets(self):
        config = _make_config({
            "cards_under_25": _market(
                enabled=True,
                model="lightgbm",
                saved_models=["cards_under_25_lightgbm.joblib"],
                n_bets=25,
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
                ece=0.20,
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
                ece=0.08,
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
                holdout_metrics={"ece": 0.18, "sharpe": 1.2},
            ),
        })
        warnings = validate_config(config, max_ece=0.15)
        assert any("BLOCKED" in w and "ECE" in w for w in warnings)
        assert config["markets"]["over25"]["enabled"] is False

    def test_missing_ece_no_warning(self):
        """Missing ECE should not block — can't gate on missing data."""
        config = _make_config({
            "cards": _market(
                enabled=True,
                model="lightgbm",
                saved_models=["cards_lightgbm.joblib"],
            ),
        })
        warnings = validate_config(config, max_ece=0.15)
        blocked = [w for w in warnings if "BLOCKED" in w]
        assert len(blocked) == 0
        assert config["markets"]["cards"]["enabled"] is True


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
