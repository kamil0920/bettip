"""Tests for _get_base_model_types() flags, model params serialization, and result JSON safety."""

import json
import tempfile
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from experiments.run_sniper_optimization import SniperOptimizer, SniperResult, _numpy_serializer


class TestGetBaseModelTypes:
    """Test _get_base_model_types with various flag combinations."""

    def test_default_includes_catboost(self):
        models = SniperOptimizer._get_base_model_types()
        assert "catboost" in models
        assert "lightgbm" in models
        assert "xgboost" in models

    def test_no_catboost_excludes_catboost(self):
        models = SniperOptimizer._get_base_model_types(no_catboost=True)
        assert "catboost" not in models
        assert "lightgbm" in models
        assert "xgboost" in models

    def test_only_catboost_overrides_no_catboost(self):
        """only_catboost takes precedence — returns ["catboost"] regardless."""
        models = SniperOptimizer._get_base_model_types(
            only_catboost=True, no_catboost=True
        )
        assert models == ["catboost"]

    def test_no_catboost_with_fast_mode(self):
        """fast_mode already excludes catboost, no_catboost has no extra effect."""
        models = SniperOptimizer._get_base_model_types(
            fast_mode=True, no_catboost=True
        )
        assert "catboost" not in models
        assert models == ["lightgbm", "xgboost"]

    def test_no_catboost_with_two_stage(self):
        models = SniperOptimizer._get_base_model_types(
            no_catboost=True, include_two_stage=True
        )
        assert "catboost" not in models
        assert "lightgbm" in models
        assert "two_stage_lgb" in models

    def test_no_catboost_with_fastai(self):
        """When fastai is available, no_catboost still removes catboost."""
        with patch("experiments.run_sniper_optimization.FASTAI_AVAILABLE", True):
            models = SniperOptimizer._get_base_model_types(
                include_fastai=True, no_catboost=True
            )
        assert "catboost" not in models
        assert "fastai" in models


class TestNumpySerializer:
    """Test _numpy_serializer handles numpy types correctly."""

    def test_numpy_int(self):
        assert _numpy_serializer(np.int64(42)) == 42
        assert isinstance(_numpy_serializer(np.int64(42)), int)

    def test_numpy_float(self):
        assert _numpy_serializer(np.float64(3.14)) == 3.14
        assert isinstance(_numpy_serializer(np.float64(3.14)), float)

    def test_numpy_bool(self):
        assert _numpy_serializer(np.bool_(True)) is True
        assert isinstance(_numpy_serializer(np.bool_(False)), bool)

    def test_numpy_array(self):
        result = _numpy_serializer(np.array([1, 2, 3]))
        assert result == [1, 2, 3]

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError, match="not JSON serializable"):
            _numpy_serializer(set())

    def test_roundtrip_with_numpy_values(self):
        """Full JSON roundtrip with numpy-typed model params."""
        params = {
            "all_model_params": {
                "lightgbm": {
                    "n_estimators": np.int64(500),
                    "learning_rate": np.float64(0.05),
                    "max_depth": np.int32(6),
                    "subsample": np.float32(0.8),
                },
                "catboost": {
                    "iterations": np.int64(300),
                    "depth": np.int64(8),
                    "l2_leaf_reg": np.float64(3.0),
                },
            },
            "model_cal_methods": {"lightgbm": "sigmoid", "catboost": "isotonic"},
            "bet_type": "home_win",
            "seed": 42,
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(params, f, indent=2, default=_numpy_serializer)
            tmp_path = f.name

        with open(tmp_path) as f:
            loaded = json.load(f)

        assert loaded["all_model_params"]["lightgbm"]["n_estimators"] == 500
        assert loaded["all_model_params"]["lightgbm"]["learning_rate"] == pytest.approx(0.05)
        assert loaded["all_model_params"]["catboost"]["iterations"] == 300
        assert loaded["model_cal_methods"]["lightgbm"] == "sigmoid"
        assert loaded["bet_type"] == "home_win"
        assert loaded["seed"] == 42

        Path(tmp_path).unlink()


class TestSniperResultSerialization:
    """Test SniperResult JSON serialization with numpy types.

    Regression tests for the truncated JSON bug: json.dump crashed with
    'Object of type int32 is not JSON serializable' because _numpy_serializer
    was not passed as `default`. The file was left truncated on disk.
    """

    def _make_result(self, **overrides):
        """Create a minimal SniperResult with optional numpy-typed overrides."""
        defaults = dict(
            bet_type="home_win",
            target="result_home_win",
            best_model="stacking",
            best_params={"ensemble_type": "average", "base_models": ["lightgbm"]},
            n_features=10,
            optimal_features=["feat_a", "feat_b"],
            best_threshold=0.65,
            best_min_odds=1.5,
            best_max_odds=5.0,
            precision=0.72,
            roi=45.3,
            n_bets=120,
            n_wins=80,
            timestamp="2026-02-11 08:00:00",
        )
        defaults.update(overrides)
        return SniperResult(**defaults)

    def test_basic_result_serializes(self):
        """A plain SniperResult should serialize without _numpy_serializer."""
        result = self._make_result()
        data = json.dumps(asdict(result))
        loaded = json.loads(data)
        assert loaded["bet_type"] == "home_win"
        assert loaded["roi"] == 45.3

    def test_numpy_int32_in_walkforward_serializes(self):
        """numpy int32 inside walkforward dict must not crash json.dump."""
        result = self._make_result(
            walkforward={
                "folds": [
                    {
                        "n_bets": np.int32(42),
                        "n_wins": np.int32(30),
                        "roi": np.float64(55.3),
                        "predictions": [
                            {"prob": np.float32(0.73), "actual": np.int32(1)},
                            {"prob": np.float32(0.65), "actual": np.int32(0)},
                        ],
                    }
                ],
                "total_bets": np.int64(42),
            }
        )
        # Without _numpy_serializer this would raise TypeError
        data = json.dumps(asdict(result), default=_numpy_serializer)
        loaded = json.loads(data)
        assert loaded["walkforward"]["folds"][0]["n_bets"] == 42
        assert loaded["walkforward"]["total_bets"] == 42
        assert isinstance(loaded["walkforward"]["folds"][0]["n_bets"], int)

    def test_numpy_int32_without_serializer_crashes(self):
        """Confirm the bug: numpy int32 crashes standard json.dumps."""
        result = self._make_result(
            walkforward={"n_bets": np.int32(10)}
        )
        with pytest.raises(TypeError, match="not JSON serializable"):
            json.dumps(asdict(result))

    def test_numpy_int64_in_holdout_metrics_serializes(self):
        """numpy int64 in holdout_metrics must serialize correctly."""
        result = self._make_result(
            holdout_metrics={
                "n_bets": np.int64(48),
                "n_wins": np.int64(35),
                "roi": np.float64(113.5),
                "precision": np.float64(0.729),
            }
        )
        data = json.dumps(asdict(result), default=_numpy_serializer)
        loaded = json.loads(data)
        assert loaded["holdout_metrics"]["n_bets"] == 48
        assert loaded["holdout_metrics"]["roi"] == pytest.approx(113.5)

    def test_deeply_nested_numpy_values(self):
        """Reproduce the exact nesting depth from the traceback:
        result -> walkforward -> fold -> predictions -> [list] -> {dict} -> int32
        """
        result = self._make_result(
            walkforward={
                "summary": {
                    "model_results": {
                        "lightgbm": {
                            "folds": [
                                {
                                    "predictions": [
                                        [np.float32(0.7), np.int32(1), np.float64(2.1)],
                                        [np.float32(0.3), np.int32(0), np.float64(1.8)],
                                    ]
                                }
                            ]
                        }
                    }
                }
            }
        )
        data = json.dumps(asdict(result), default=_numpy_serializer)
        loaded = json.loads(data)
        preds = loaded["walkforward"]["summary"]["model_results"]["lightgbm"]["folds"][0]["predictions"]
        assert preds[0][1] == 1
        assert isinstance(preds[0][1], int)

    def test_atomic_write_produces_valid_json(self):
        """Atomic write pattern: write to .tmp then rename."""
        result = self._make_result(
            walkforward={"n_bets": np.int32(10), "roi": np.float64(50.0)}
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "sniper_home_win_test.json"
            tmp_path = output_path.with_suffix(".json.tmp")

            with open(tmp_path, "w") as f:
                json.dump(asdict(result), f, indent=2, default=_numpy_serializer)
            tmp_path.rename(output_path)

            # Verify the file is valid JSON
            with open(output_path) as f:
                loaded = json.load(f)
            assert loaded["bet_type"] == "home_win"
            assert loaded["walkforward"]["n_bets"] == 10
            # .tmp file should not exist after rename
            assert not tmp_path.exists()

    def test_all_numpy_scalar_types_handled(self):
        """Ensure all common numpy scalar types serialize through _numpy_serializer."""
        result = self._make_result(
            walkforward={
                "int8": np.int8(1),
                "int16": np.int16(2),
                "int32": np.int32(3),
                "int64": np.int64(4),
                "uint8": np.uint8(5),
                "float16": np.float16(0.5),
                "float32": np.float32(0.6),
                "float64": np.float64(0.7),
                "bool_": np.bool_(True),
                "array": np.array([1, 2, 3]),
            }
        )
        data = json.dumps(asdict(result), default=_numpy_serializer)
        loaded = json.loads(data)
        wf = loaded["walkforward"]
        assert wf["int32"] == 3
        assert wf["uint8"] == 5
        assert wf["bool_"] is True
        assert wf["array"] == [1, 2, 3]
