"""Tests for _get_base_model_types() flags and model params serialization."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from experiments.run_sniper_optimization import SniperOptimizer, _numpy_serializer


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
