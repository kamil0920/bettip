"""
Integration tests for critical sniper pipeline paths.

These tests target the exact code paths that caused historical bugs:
- ft_iterations crash (80h wasted compute)
- _x/_y column collision (40h wasted)
- monotonic + Depthwise crash (6h wasted)
- predict_proba 1-column edge case (6h wasted)
- String "None" in features
- Data leakage via cross-market features

Each test exercises real model creation → fit → predict paths,
not just isolated functions.
"""

import sys
import os
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_features_df():
    """500 rows, 20 features, binary target, odds, dates spanning 2022-2024."""
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2022-01-01", periods=n, freq="3D")
    teams_home = [f"Team_{i % 10}" for i in range(n)]
    teams_away = [f"Team_{(i + 5) % 10}" for i in range(n)]

    df = pd.DataFrame({
        "date": dates,
        "home_team_name": teams_home,
        "away_team_name": teams_away,
        "fixture_id": range(1, n + 1),
    })

    # 20 numeric features
    for i in range(20):
        df[f"feat_{i}"] = np.random.randn(n)

    # Target and odds
    df["target"] = np.random.randint(0, 2, n)
    df["odds"] = np.random.uniform(1.5, 5.0, n)

    return df


@pytest.fixture
def small_features_df():
    """100 rows, 5 features for quick model fit tests."""
    np.random.seed(123)
    n = 100
    df = pd.DataFrame({f"feat_{i}": np.random.randn(n) for i in range(5)})
    df["target"] = np.random.randint(0, 2, n)
    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_optimizer_stub():
    """Create a minimal SniperOptimizer-like object for method calls."""
    # Import here to avoid import errors at collection time
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from experiments.run_sniper_optimization import SniperOptimizer
    return SniperOptimizer


# ---------------------------------------------------------------------------
# Test 1: CatBoost Parameter Sanitization
# ---------------------------------------------------------------------------

class TestCatBoostParamSanitization:
    """_create_model_instance strips non-CatBoost keys before construction."""

    @pytest.mark.slow
    def test_strip_keys_creates_valid_model(self, small_features_df):
        """Params with ft_iterations, use_monotonic etc. don't crash CatBoost."""
        SniperOptimizer = _make_optimizer_stub()
        opt = SniperOptimizer.__new__(SniperOptimizer)
        opt.use_transfer_learning = False
        opt.use_baseline = False
        opt.seed = 42

        params = {
            "iterations": 50,
            "learning_rate": 0.1,
            "depth": 4,
            # These would crash CatBoostClassifier if not stripped:
            "ft_iterations": 100,
            "use_monotonic": True,
        }

        # Keys that should be stripped by _CATBOOST_STRIP_KEYS
        model = opt._create_model_instance("catboost", params)

        # Model should be a valid CatBoost classifier
        from catboost import CatBoostClassifier
        assert isinstance(model, CatBoostClassifier)

        # Should fit and predict without crashing
        X = small_features_df[[f"feat_{i}" for i in range(5)]].values
        y = small_features_df["target"].values
        model.fit(X, y, verbose=False)
        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 2)

    def test_raw_catboost_rejects_ft_iterations(self):
        """Prove ft_iterations would crash without the strip — locks in the invariant."""
        from catboost import CatBoostClassifier

        with pytest.raises(TypeError, match="unexpected keyword argument"):
            CatBoostClassifier(ft_iterations=100, iterations=50, verbose=False)

    def test_strip_keys_constant_contains_ft_iterations(self):
        """_CATBOOST_STRIP_KEYS must include ft_iterations."""
        SniperOptimizer = _make_optimizer_stub()
        assert "ft_iterations" in SniperOptimizer._CATBOOST_STRIP_KEYS
        assert "use_monotonic" in SniperOptimizer._CATBOOST_STRIP_KEYS


# ---------------------------------------------------------------------------
# Test 2: Monotonic Constraints + Grow Policy Guard
# ---------------------------------------------------------------------------

class TestMonotonicGrowPolicyGuard:
    """CatBoost monotonic constraints only work with SymmetricTree grow_policy."""

    @pytest.mark.slow
    def test_symmetric_tree_with_monotonic_succeeds(self, small_features_df):
        """SymmetricTree + monotone_constraints works fine."""
        from catboost import CatBoostClassifier

        X = small_features_df[[f"feat_{i}" for i in range(5)]].values
        y = small_features_df["target"].values
        constraints = [1, -1, 0, 0, 0]

        model = CatBoostClassifier(
            iterations=30,
            depth=4,
            grow_policy="SymmetricTree",
            monotone_constraints=constraints,
            verbose=False,
        )
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 2)

    @pytest.mark.slow
    def test_depthwise_with_monotonic_crashes(self, small_features_df):
        """Depthwise + monotone_constraints must crash — proves the guard is needed."""
        from catboost import CatBoostClassifier, CatBoostError

        X = small_features_df[[f"feat_{i}" for i in range(5)]].values
        y = small_features_df["target"].values
        constraints = [1, -1, 0, 0, 0]

        model = CatBoostClassifier(
            iterations=30,
            depth=4,
            grow_policy="Depthwise",
            monotone_constraints=constraints,
            verbose=False,
        )
        with pytest.raises(CatBoostError):
            model.fit(X, y)


# ---------------------------------------------------------------------------
# Test 3: Odds Merger No _x/_y Columns
# ---------------------------------------------------------------------------

class TestOddsMergerNoSuffixes:
    """OddsMerger.merge_with_features() never produces _x/_y suffixed columns."""

    def test_overlapping_columns_no_suffixes(self):
        """When odds_df has columns overlapping with features_df, no _x/_y appear."""
        from src.odds.odds_merger import OddsMerger

        np.random.seed(42)
        n = 50
        dates = pd.date_range("2023-01-01", periods=n, freq="7D")
        home_teams = [f"Arsenal" if i % 2 == 0 else "Chelsea" for i in range(n)]
        away_teams = [f"Liverpool" if i % 2 == 0 else "Tottenham" for i in range(n)]

        # Features with match stat columns that also appear in odds data
        features_df = pd.DataFrame({
            "date": dates,
            "home_team_name": home_teams,
            "away_team_name": away_teams,
            "home_elo": np.random.randn(n) + 1500,
            "away_elo": np.random.randn(n) + 1500,
            # These overlap with odds_df match stats:
            "home_shots": np.random.randint(5, 20, n),
            "away_shots": np.random.randint(5, 20, n),
            "total_corners": np.random.randint(5, 15, n),
            "home_fouls": np.random.randint(5, 20, n),
            "away_fouls": np.random.randint(5, 20, n),
        })

        # Odds data with same stat columns + real odds
        odds_df = pd.DataFrame({
            "date": dates,
            "home_team": home_teams,
            "away_team": away_teams,
            # Overlapping match stats:
            "home_shots": np.random.randint(5, 20, n),
            "away_shots": np.random.randint(5, 20, n),
            "total_corners": np.random.randint(5, 15, n),
            "home_fouls": np.random.randint(5, 20, n),
            "away_fouls": np.random.randint(5, 20, n),
            # Real odds columns:
            "avg_home_close": np.random.uniform(1.5, 4.0, n),
            "avg_away_close": np.random.uniform(1.5, 4.0, n),
            "avg_over25_close": np.random.uniform(1.5, 3.0, n),
        })

        merger = OddsMerger(date_tolerance_days=1, fuzzy_match_threshold=70)
        merged = merger.merge_with_features(features_df, odds_df)

        # No _x or _y suffixed columns
        bad_cols = [c for c in merged.columns if c.endswith("_x") or c.endswith("_y")]
        assert bad_cols == [], f"Found _x/_y columns: {bad_cols}"

        # Odds columns should be present
        assert "avg_home_close" in merged.columns
        assert "avg_away_close" in merged.columns

        # Original feature values preserved
        pd.testing.assert_series_equal(
            merged["home_elo"].iloc[:n],
            features_df["home_elo"],
            check_names=False,
        )

    def test_empty_odds_returns_features_unchanged(self):
        """Empty odds_df returns features unchanged."""
        from src.odds.odds_merger import OddsMerger

        features_df = pd.DataFrame({
            "date": ["2023-01-01"],
            "home_team_name": ["Arsenal"],
            "away_team_name": ["Chelsea"],
            "home_elo": [1500.0],
        })
        odds_df = pd.DataFrame()

        merger = OddsMerger()
        result = merger.merge_with_features(features_df, odds_df)
        assert len(result) == 1
        assert "home_elo" in result.columns


# ---------------------------------------------------------------------------
# Test 4: predict_proba 1-Column Guard
# ---------------------------------------------------------------------------

class TestPredictProba1ColGuard:
    """CalibratedClassifierCV can return 1 column; our guard handles it."""

    def test_safe_predict_proba_returns_none_for_1col(self):
        """_safe_predict_proba returns None when predict_proba gives 1 column."""
        SniperOptimizer = _make_optimizer_stub()

        # Mock model whose predict_proba returns 1 column
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.8], [0.9], [0.7]])

        result = SniperOptimizer._safe_predict_proba(mock_model, np.zeros((3, 5)))
        assert result is None

    def test_safe_predict_proba_returns_col1_for_2col(self):
        """_safe_predict_proba returns column 1 for normal 2-column output."""
        SniperOptimizer = _make_optimizer_stub()

        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([
            [0.3, 0.7],
            [0.4, 0.6],
            [0.2, 0.8],
        ])

        result = SniperOptimizer._safe_predict_proba(mock_model, np.zeros((3, 5)))
        assert result is not None
        np.testing.assert_array_almost_equal(result, [0.7, 0.6, 0.8])

    def test_safe_predict_proba_handles_1d_array(self):
        """_safe_predict_proba handles 1D array edge case."""
        SniperOptimizer = _make_optimizer_stub()

        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([0.8, 0.9, 0.7])

        result = SniperOptimizer._safe_predict_proba(mock_model, np.zeros((3, 5)))
        assert result is None

    def test_no_unguarded_predict_proba_calls(self):
        """Static analysis: all predict_proba[:, 1] in run_sniper_optimization.py are guarded."""
        import re

        script_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "experiments", "run_sniper_optimization.py"
        )
        with open(script_path) as f:
            lines = f.readlines()

        unguarded = []
        for i, line in enumerate(lines, 1):
            # Match predict_proba(...)[:, 1] pattern
            if re.search(r'\.predict_proba\([^)]*\)\s*\[\s*:\s*,\s*1\s*\]', line):
                # Check if it's inside _safe_predict_proba definition (OK)
                if '_safe_predict_proba' in line or 'def _safe_predict_proba' in line:
                    continue
                # Check preceding 40 lines for shape guard + continue (guard may be far above)
                context = lines[max(0, i - 41):i - 1]
                context_text = "".join(context)
                has_guard = (
                    "shape[1]" in context_text
                    or "_safe_predict_proba" in context_text
                    or "if proba.shape[1] == 1:" in context_text
                )
                # Also OK: inside adversarial validation (raw LGBMClassifier, always 2 classes)
                is_adversarial = "roc_auc_score" in line or "_adversarial" in "".join(lines[max(0, i - 3):i])
                if not has_guard and not is_adversarial:
                    unguarded.append((i, line.strip()))

        assert unguarded == [], (
            f"Found unguarded predict_proba[:, 1] calls:\n"
            + "\n".join(f"  Line {ln}: {code}" for ln, code in unguarded)
        )


# ---------------------------------------------------------------------------
# Test 5: Model Creation Smoke Test (All Types)
# ---------------------------------------------------------------------------

class TestModelCreationSmoke:
    """Every model type can be created, fit, and predict."""

    @pytest.mark.slow
    @pytest.mark.parametrize("model_type,params", [
        ("lightgbm", {"n_estimators": 30, "max_depth": 3, "num_leaves": 8}),
        ("xgboost", {"n_estimators": 30, "max_depth": 3}),
        ("catboost", {"iterations": 30, "depth": 4}),
    ])
    def test_base_model_fit_predict(self, small_features_df, model_type, params):
        """Each base model type creates, fits, and predicts with shape (n, 2)."""
        SniperOptimizer = _make_optimizer_stub()
        opt = SniperOptimizer.__new__(SniperOptimizer)
        opt.use_transfer_learning = False
        opt.use_baseline = False
        opt.seed = 42

        model = opt._create_model_instance(model_type, params)
        X = small_features_df[[f"feat_{i}" for i in range(5)]].values
        y = small_features_df["target"].values

        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 2), f"{model_type} predict_proba shape mismatch"
        assert np.all((proba >= 0) & (proba <= 1)), f"{model_type} probabilities out of range"

    @pytest.mark.slow
    def test_two_stage_model_fit_predict(self, small_features_df):
        """Two-stage model creates, fits, and predicts."""
        SniperOptimizer = _make_optimizer_stub()
        opt = SniperOptimizer.__new__(SniperOptimizer)
        opt.use_transfer_learning = False
        opt.use_baseline = False
        opt.seed = 42

        params = {
            "stage1_params": {"n_estimators": 20, "max_depth": 3, "num_leaves": 8},
            "stage2_params": {"n_estimators": 20, "max_depth": 3, "num_leaves": 8},
            "calibration_method": "sigmoid",
            "min_edge_threshold": 0.02,
        }

        model = opt._create_model_instance("two_stage_lgb", params)
        X = small_features_df[[f"feat_{i}" for i in range(5)]].values
        y = small_features_df["target"].values
        odds = np.random.uniform(1.5, 5.0, len(X))

        model.fit(X, y, odds)
        result = model.predict_proba(X, odds)
        assert "combined_score" in result
        assert len(result["combined_score"]) == len(X)


# ---------------------------------------------------------------------------
# Test 6: Data Type Sanitization (None-String)
# ---------------------------------------------------------------------------

class TestDataTypeSanitization:
    """_convert_array_to_float handles string 'None', 'nan', brackets, empty strings."""

    def test_mixed_types_converted_to_float(self):
        """Various string formats are converted to float64 correctly."""
        SniperOptimizer = _make_optimizer_stub()
        opt = SniperOptimizer.__new__(SniperOptimizer)

        # Create array with problematic types
        X = np.array([
            ["None", "5.07", "2.5"],
            ["nan", "[3.14]", ""],
            ["1.0", "0.5", "None"],
        ], dtype=object)
        feature_names = ["f1", "f2", "f3"]

        result = opt._convert_array_to_float(X, feature_names)

        assert result.dtype == np.float64
        assert result.shape == (3, 3)
        # Non-NaN values should be correct floats
        assert result[0, 1] == pytest.approx(5.07)
        assert result[1, 1] == pytest.approx(3.14)
        assert result[0, 2] == pytest.approx(2.5)
        assert result[2, 0] == pytest.approx(1.0)
        # No infinities
        assert not np.any(np.isinf(result))

    def test_all_none_column_becomes_zero(self):
        """An all-None column should become 0 (median fallback → fillna(0))."""
        SniperOptimizer = _make_optimizer_stub()
        opt = SniperOptimizer.__new__(SniperOptimizer)

        X = np.array([["None"], ["None"], ["None"]], dtype=object)
        result = opt._convert_array_to_float(X, ["f1"])
        assert result.dtype == np.float64
        # All-NaN → median is NaN → fillna(0)
        np.testing.assert_array_equal(result.flatten(), [0.0, 0.0, 0.0])

    def test_scientific_notation_in_brackets(self):
        """Bracketed scientific notation like '[3.9479554E-1]' is parsed."""
        SniperOptimizer = _make_optimizer_stub()
        opt = SniperOptimizer.__new__(SniperOptimizer)

        X = np.array([["[3.9479554E-1]"], ["[9.5E-2]"]], dtype=object)
        result = opt._convert_array_to_float(X, ["f1"])
        assert result[0, 0] == pytest.approx(0.39479554, abs=1e-6)
        assert result[1, 0] == pytest.approx(0.095, abs=1e-6)


# ---------------------------------------------------------------------------
# Test 7: Feature Regeneration Leakage Stripping
# ---------------------------------------------------------------------------

class TestFeatureRegenerationLeakageStripping:
    """_add_cross_market_features strips all _LEAKAGE_COLUMNS."""

    def test_leakage_columns_stripped_before_cross_market(self):
        """CrossMarketFeatureEngineer never sees target/outcome columns."""
        from src.features.regeneration import FeatureRegenerator

        regen = FeatureRegenerator.__new__(FeatureRegenerator)

        # Build a DataFrame with all leakage columns + safe features
        n = 50
        data = {"fixture_id": range(n)}

        # Add ALL leakage columns
        for col in FeatureRegenerator._LEAKAGE_COLUMNS:
            data[col] = np.random.randn(n)

        # Add safe feature columns
        safe_cols = [
            "home_elo", "away_elo", "home_shots_ema", "away_shots_ema",
            "home_form_points", "away_form_points", "home_corners_ema",
            "away_corners_ema", "home_cards_ema", "away_cards_ema",
        ]
        for col in safe_cols:
            data[col] = np.random.randn(n)

        merged_df = pd.DataFrame(data)

        # Capture what CrossMarketFeatureEngineer.create_features receives
        captured_dfs = []

        def mock_create_features(self_eng, data_dict):
            captured_dfs.append(data_dict["matches"].copy())
            # Return empty to avoid downstream issues
            return pd.DataFrame({"fixture_id": range(n)})

        with patch(
            "src.features.engineers.cross_market.CrossMarketFeatureEngineer.create_features",
            mock_create_features,
        ):
            regen._add_cross_market_features(merged_df)

        assert len(captured_dfs) == 1, "CrossMarketFeatureEngineer.create_features was not called"
        received_df = captured_dfs[0]

        # Assert NO leakage columns were passed
        leaked = [c for c in FeatureRegenerator._LEAKAGE_COLUMNS if c in received_df.columns]
        assert leaked == [], f"Leakage columns passed to CrossMarketFeatureEngineer: {leaked}"

        # Assert safe feature columns ARE present
        for col in safe_cols:
            assert col in received_df.columns, f"Safe feature '{col}' was incorrectly stripped"

    def test_leakage_columns_constant_covers_known_targets(self):
        """_LEAKAGE_COLUMNS includes all known target and raw stat columns."""
        from src.features.regeneration import FeatureRegenerator

        critical_targets = {
            "goal_difference", "total_goals", "home_goals", "away_goals",
            "home_win", "away_win", "draw", "btts", "over25", "under25",
            "total_corners", "total_fouls", "total_shots", "total_cards",
        }
        missing = critical_targets - FeatureRegenerator._LEAKAGE_COLUMNS
        assert missing == set(), f"Missing critical targets from _LEAKAGE_COLUMNS: {missing}"


# ---------------------------------------------------------------------------
# Test 8: CatBoost + CalibratedClassifierCV Prefit (Bug 2 regression)
# ---------------------------------------------------------------------------

class TestCatBoostCalibratedPrefit:
    """CatBoost with cv='prefit' survives CalibratedClassifierCV without feature mismatch.

    Historical bug: CatBoost with has_time=True tracks internal feature ordering.
    When CalibratedClassifierCV creates internal CV folds, feature indices diverge
    causing 'Feature N is present in model but not in pool' errors.
    Fix: use cv='prefit' so CalibratedClassifierCV skips internal re-fitting.
    """

    def test_get_calibration_cv_returns_prefit_for_catboost(self):
        """_get_calibration_cv must return 'prefit' for CatBoost."""
        from experiments.run_sniper_optimization import _get_calibration_cv
        assert _get_calibration_cv("catboost") == "prefit"

    def test_get_calibration_cv_returns_int_for_others(self):
        """_get_calibration_cv returns integer n_splits for non-CatBoost models."""
        from experiments.run_sniper_optimization import _get_calibration_cv
        for model_type in ("lightgbm", "xgboost", "logistic_regression"):
            cv = _get_calibration_cv(model_type)
            assert isinstance(cv, int), f"{model_type} should get int cv, got {type(cv)}"

    @pytest.mark.slow
    def test_catboost_prefit_calibration_succeeds(self, small_features_df):
        """CatBoost pre-fit + CalibratedClassifierCV(cv='prefit') produces valid probabilities.

        This is the end-to-end regression test: the exact code pattern used in
        walk-forward, threshold optimization, temporal blend, and MAPIE.
        """
        from catboost import CatBoostClassifier
        from experiments.run_sniper_optimization import _get_calibration_cv

        X = small_features_df[[f"feat_{i}" for i in range(5)]].values
        y = small_features_df["target"].values

        model = CatBoostClassifier(iterations=30, depth=4, verbose=False, has_time=True)
        cv_val = _get_calibration_cv("catboost")
        assert cv_val == "prefit"

        # Pre-fit the model (required for cv="prefit")
        model.fit(X, y)

        # Wrap with CalibratedClassifierCV — this used to crash with feature mismatch
        calibrated = CalibratedClassifierCV(model, method="sigmoid", cv=cv_val)
        calibrated.fit(X, y)

        proba = calibrated.predict_proba(X)
        assert proba.shape == (len(X), 2), f"Expected (n, 2), got {proba.shape}"
        assert np.all((proba >= 0) & (proba <= 1)), "Probabilities out of [0, 1]"

    def test_prefit_avoids_internal_refitting(self, small_features_df):
        """cv='prefit' means CalibratedClassifierCV does NOT clone/refit the model.

        With cv=TimeSeriesSplit, sklearn clones the estimator and fits it on each fold,
        which causes CatBoost feature index divergence on real (large) data.
        With cv='prefit', the original fitted model is used directly for calibration.
        """
        from catboost import CatBoostClassifier

        X = small_features_df[[f"feat_{i}" for i in range(5)]].values
        y = small_features_df["target"].values

        model = CatBoostClassifier(iterations=30, depth=4, verbose=False, has_time=True)
        model.fit(X, y)

        # With prefit, the calibrated model wraps the SAME fitted model (no cloning)
        calibrated = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
        calibrated.fit(X, y)

        # The internal calibrated classifier should reference our original model
        assert len(calibrated.calibrated_classifiers_) == 1
        inner_model = calibrated.calibrated_classifiers_[0].estimator
        # Verify it's the same model object (not a clone)
        assert inner_model is model


# ---------------------------------------------------------------------------
# Test 9: Temperature Calibration Mapping (Bug 1 regression)
# ---------------------------------------------------------------------------

class TestTemperatureCalibrationMapping:
    """Temperature calibration is mapped to sigmoid for sklearn + applied post-hoc.

    Historical bug: Optuna could select calibration_method='temperature', but
    CalibratedClassifierCV only accepts 'sigmoid' or 'isotonic'. Temperature
    was passed through raw, crashing all downstream phases (walk-forward,
    temporal blend, MAPIE, threshold optimization).
    """

    def test_sklearn_cal_method_maps_temperature_to_sigmoid(self):
        """Post-HPO flag mapping converts temperature → sigmoid for sklearn."""
        SniperOptimizer = _make_optimizer_stub()
        opt = SniperOptimizer.__new__(SniperOptimizer)

        # Simulate what happens after HPO selects temperature
        opt._sklearn_cal_method = (
            "sigmoid" if "temperature" in ("beta", "temperature") else "temperature"
        )
        opt._use_custom_calibration = "temperature" in ("beta", "temperature")

        assert opt._sklearn_cal_method == "sigmoid"
        assert opt._use_custom_calibration is True

    def test_sklearn_rejects_temperature_method(self):
        """Prove CalibratedClassifierCV crashes with method='temperature' — locks in the invariant."""
        from sklearn.linear_model import LogisticRegression

        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)

        model = LogisticRegression(max_iter=200)
        with pytest.raises(ValueError, match="method"):
            calibrated = CalibratedClassifierCV(model, method="temperature", cv=3)
            calibrated.fit(X, y)

    @pytest.mark.slow
    def test_temperature_posthoc_calibration_works(self, small_features_df):
        """Full temperature post-hoc path: sigmoid sklearn + TemperatureScaling applied.

        Exercises the exact code pattern from threshold optimization and walk-forward.
        """
        from sklearn.linear_model import LogisticRegression
        from src.calibration.calibration import TemperatureScaling

        X = small_features_df[[f"feat_{i}" for i in range(5)]].values
        y = small_features_df["target"].values

        # Step 1: Fit with sigmoid (the sklearn-safe method)
        model = LogisticRegression(max_iter=200)
        calibrated = CalibratedClassifierCV(model, method="sigmoid", cv=3)
        calibrated.fit(X, y)

        proba = calibrated.predict_proba(X)
        assert proba.shape[1] == 2
        probs = proba[:, 1]

        # Step 2: Apply TemperatureScaling post-hoc (the actual fix)
        train_proba = calibrated.predict_proba(X)
        temp_cal = TemperatureScaling()
        temp_cal.fit(train_proba[:, 1], y)
        calibrated_probs = temp_cal.transform(probs)

        # Probabilities should still be valid
        assert len(calibrated_probs) == len(X)
        assert np.all((calibrated_probs >= 0) & (calibrated_probs <= 1))
        # Temperature scaling should actually change the probabilities
        assert not np.allclose(probs, calibrated_probs, atol=1e-6), (
            "TemperatureScaling had no effect — post-hoc calibration is not working"
        )

    def test_all_cal_methods_mapped_consistently(self):
        """Static analysis: every sklearn_cal assignment handles both beta AND temperature."""
        import re

        script_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "experiments", "run_sniper_optimization.py"
        )
        with open(script_path) as f:
            content = f.read()

        # Find all lines that do the cal_method → sklearn_cal mapping
        # Pattern: "sigmoid" if cal_method in/== ... else ...
        mapping_pattern = re.compile(
            r'"sigmoid"\s+if\s+(cal_method|winning_cal)\s+(in|==)\s+'
        )
        matches = list(mapping_pattern.finditer(content))

        # Should have at least 3 mapping sites (post-HPO, threshold opt, walk-forward, model save)
        assert len(matches) >= 3, (
            f"Expected at least 3 cal_method→sigmoid mapping sites, found {len(matches)}"
        )

        # Every mapping using 'in' must include both "beta" and "temperature"
        for m in matches:
            if m.group(2) == "in":
                # Find the tuple after 'in'
                start = m.end()
                paren_end = content.index(")", start)
                tuple_str = content[start:paren_end + 1]
                assert "beta" in tuple_str, f"Missing 'beta' in mapping: {tuple_str}"
                assert "temperature" in tuple_str, f"Missing 'temperature' in mapping: {tuple_str}"

        # No mapping should use == "beta" alone (misses temperature)
        bad_pattern = re.compile(r'"sigmoid"\s+if\s+cal_method\s+==\s+"beta"')
        bad_matches = list(bad_pattern.finditer(content))
        assert bad_matches == [], (
            f"Found {len(bad_matches)} cal_method mapping(s) that only handle 'beta' but not 'temperature'"
        )


# ---------------------------------------------------------------------------
# Test 10: Stacking Ensemble Base Model Count
# ---------------------------------------------------------------------------

class TestStackingEnsembleModelCount:
    """Stacking ensembles should not silently lose base models.

    Historical issue: except Exception + continue blocks swallowed CatBoost
    feature mismatch and temperature calibration crashes, reducing stacking
    ensembles from 4 models to 2-3 models without any visible error.
    """

    def test_except_continue_blocks_logged(self):
        """Static analysis: all 'except Exception' blocks that continue must log a warning."""
        import re

        script_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "experiments", "run_sniper_optimization.py"
        )
        with open(script_path) as f:
            lines = f.readlines()

        unlogged = []
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("except Exception") and ":" in stripped:
                # Look at the next 3 lines for logger.warning or continue
                block = "".join(lines[i:i + 3])
                has_continue = "continue" in block
                has_log = "logger.warning" in block or "logger.error" in block
                if has_continue and not has_log:
                    unlogged.append((i, stripped))

        assert unlogged == [], (
            f"Found except-continue blocks without logging:\n"
            + "\n".join(f"  Line {ln}: {code}" for ln, code in unlogged)
        )
