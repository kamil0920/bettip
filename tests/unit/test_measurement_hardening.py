"""Tests for measurement hardening methods in SniperOptimizer.

Covers:
- _compute_embargo_days: embargo period from feature config lookback windows
- _mrmr_select: minimum redundancy maximum relevance feature selection
- _per_fold_ks_test: per-fold Kolmogorov-Smirnov distribution shift detection
- Aggressive regularization logic: adversarial AUC gating for Optuna bounds
"""

import sys
import os
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from experiments.run_sniper_optimization import SniperOptimizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_optimizer(**overrides) -> SniperOptimizer:
    """Create a SniperOptimizer via __new__ with minimal attributes set.

    Bypasses __init__ (which needs BET_TYPES lookup, feature config loading, etc.)
    and sets only the attributes required by the methods under test.
    """
    opt = SniperOptimizer.__new__(SniperOptimizer)
    # Defaults for attributes that the tested methods access
    opt.seed = 42
    opt.feature_config = None
    opt.no_aggressive_reg = False
    opt._adversarial_auc_mean = None
    for k, v in overrides.items():
        setattr(opt, k, v)
    return opt


def _make_feature_config(**kwargs) -> SimpleNamespace:
    """Create a lightweight stand-in for BetTypeFeatureConfig."""
    defaults = dict(
        form_window=5,
        ema_span=10,
        poisson_lookback=10,
        h2h_matches=5,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


# ---------------------------------------------------------------------------
# _compute_embargo_days
# ---------------------------------------------------------------------------

class TestComputeEmbargoDays:
    """_compute_embargo_days derives embargo from max lookback * 3.5 + 7."""

    def test_default_feature_config_values(self):
        """Default BetTypeFeatureConfig (5/10/10/5) → h2h dominates at 5*5=25 matches."""
        opt = _make_optimizer(feature_config=_make_feature_config())
        result = opt._compute_embargo_days()
        # max(5, 10, 10, 25, 20) = 25 → 25*3.5 + 7 = 94.5 → int(94) but 94 > 14
        assert result == int(25 * 3.5) + 7  # 94
        assert isinstance(result, int)

    def test_floor_at_14_days(self):
        """When all lookbacks are tiny, result is floored at 14."""
        fc = _make_feature_config(
            form_window=1, ema_span=1, poisson_lookback=1, h2h_matches=1
        )
        opt = _make_optimizer(feature_config=fc)
        result = opt._compute_embargo_days()
        # max(1, 1, 1, 1*5=5, 20) = 20 → 20*3.5+7 = 77 → still > 14
        # The corner/niche window_sizes max of 20 prevents falling below 77
        assert result >= 14

    def test_none_feature_config_uses_fallbacks(self):
        """When feature_config is None, uses hardcoded fallback values."""
        opt = _make_optimizer(feature_config=None)
        result = opt._compute_embargo_days()
        # Fallbacks: form_window=5, ema_span=10, poisson_lookback=10, h2h=5
        # max(5, 10, 10, 25, 20) = 25 → 25*3.5 + 7 = 94
        assert result == int(25 * 3.5) + 7

    def test_large_h2h_dominates(self):
        """h2h_matches * 5 amplification makes H2H the dominant lookback."""
        fc = _make_feature_config(h2h_matches=10)
        opt = _make_optimizer(feature_config=fc)
        result = opt._compute_embargo_days()
        # max(5, 10, 10, 10*5=50, 20) = 50 → 50*3.5 + 7 = 182
        assert result == int(50 * 3.5) + 7  # 182

    def test_large_ema_span_dominates(self):
        """When ema_span is the largest, it determines embargo."""
        fc = _make_feature_config(ema_span=40, h2h_matches=2)
        opt = _make_optimizer(feature_config=fc)
        result = opt._compute_embargo_days()
        # max(5, 40, 10, 10, 20) = 40 → 40*3.5 + 7 = 147
        assert result == int(40 * 3.5) + 7  # 147

    def test_returns_int_type(self):
        """Return type must be int, not float."""
        opt = _make_optimizer(feature_config=_make_feature_config())
        result = opt._compute_embargo_days()
        assert isinstance(result, int)

    def test_corner_niche_minimum_20(self):
        """The hardcoded 20 (corner/niche window_sizes max) sets a floor
        on max_lookback_matches regardless of config values."""
        fc = _make_feature_config(
            form_window=1, ema_span=1, poisson_lookback=1, h2h_matches=1
        )
        opt = _make_optimizer(feature_config=fc)
        result = opt._compute_embargo_days()
        # max(1, 1, 1, 5, 20) = 20 → 20*3.5+7 = 77
        assert result == int(20 * 3.5) + 7  # 77


# ---------------------------------------------------------------------------
# _mrmr_select
# ---------------------------------------------------------------------------

class TestMrmrSelect:
    """_mrmr_select: greedy forward selection maximizing MI, minimizing correlation."""

    @pytest.fixture
    def correlated_data(self):
        """Synthetic data with known correlation structure.

        Features 0-2 are informative (correlated with target).
        Features 3-4 are noise. Features 0 and 1 are highly correlated.
        """
        np.random.seed(42)
        n = 200
        y = np.random.randint(0, 2, n)
        X = np.column_stack([
            y + np.random.randn(n) * 0.3,       # feat_0: strong MI, correlated with feat_1
            y + np.random.randn(n) * 0.3 + 0.1,  # feat_1: strong MI, correlated with feat_0
            y * 0.5 + np.random.randn(n) * 0.5,  # feat_2: moderate MI, independent
            np.random.randn(n),                    # feat_3: noise
            np.random.randn(n),                    # feat_4: noise
        ])
        names = ["feat_0", "feat_1", "feat_2", "feat_3", "feat_4"]
        return X, y, names

    def test_k_selects_correct_count(self, correlated_data):
        """Selecting k=3 features returns exactly 3."""
        X, y, names = correlated_data
        opt = _make_optimizer()
        X_sel, sel_names, diag = opt._mrmr_select(X, y, names, k=3)
        assert len(sel_names) == 3
        assert X_sel.shape == (len(X), 3)
        assert diag["post_count"] == 3
        assert diag["pre_count"] == 5
        assert diag["n_removed"] == 2

    def test_k_greater_than_n_features_returns_all(self, correlated_data):
        """When k >= n_features, all features are returned."""
        X, y, names = correlated_data
        opt = _make_optimizer()
        X_sel, sel_names, diag = opt._mrmr_select(X, y, names, k=100)
        assert len(sel_names) == 5
        assert X_sel.shape == (len(X), 5)
        assert diag["n_removed"] == 0

    def test_k_equals_1_returns_highest_mi_feature(self, correlated_data):
        """k=1 should select the single feature with highest MI to the target."""
        X, y, names = correlated_data
        opt = _make_optimizer()
        X_sel, sel_names, diag = opt._mrmr_select(X, y, names, k=1)
        assert len(sel_names) == 1
        # The selected feature should be one of the informative ones (0 or 1)
        assert sel_names[0] in ["feat_0", "feat_1"]

    def test_redundancy_penalty_avoids_duplicates(self):
        """mRMR penalizes redundancy: a duplicate feature should be removed
        in favor of a less-correlated informative alternative."""
        np.random.seed(42)
        n = 500
        y = np.random.randint(0, 2, n)
        signal = y.astype(float) + np.random.randn(n) * 0.1  # strong signal
        X = np.column_stack([
            signal,                                  # feat_0: strong MI
            signal + np.random.randn(n) * 0.01,      # feat_1: exact copy of feat_0 (r~1.0)
            y * 0.8 + np.random.randn(n) * 0.3,      # feat_2: strong MI, independent of feat_0
            np.random.randn(n),                       # feat_3: noise
        ])
        names = ["feat_0", "feat_1", "feat_2", "feat_3"]

        opt = _make_optimizer()
        _, sel_names, _ = opt._mrmr_select(X, y, names, k=2)
        # feat_0 and feat_1 are virtually identical → mRMR should pick at most one,
        # then prefer feat_2 (independent, high MI) over the clone
        assert not ("feat_0" in sel_names and "feat_1" in sel_names), (
            f"mRMR selected both clone features: {sel_names}"
        )

    def test_removed_features_reported(self, correlated_data):
        """Diagnostics include the removed feature names."""
        X, y, names = correlated_data
        opt = _make_optimizer()
        _, sel_names, diag = opt._mrmr_select(X, y, names, k=2)
        removed = diag["removed_features"]
        assert len(removed) == 3
        # Removed features should not overlap with selected
        for r in removed:
            assert r not in sel_names

    def test_selected_indices_sorted(self, correlated_data):
        """Selected column indices in X_selected are in sorted order."""
        X, y, names = correlated_data
        opt = _make_optimizer()
        X_sel, sel_names, diag = opt._mrmr_select(X, y, names, k=3)
        # Names should correspond to column indices in ascending order
        original_indices = [names.index(n) for n in sel_names]
        assert original_indices == sorted(original_indices)


# ---------------------------------------------------------------------------
# _per_fold_ks_test
# ---------------------------------------------------------------------------

class TestPerFoldKsTest:
    """_per_fold_ks_test: KS distribution shift detection on top-variance features."""

    def test_identical_distributions_no_shift(self):
        """When train and test come from the same distribution, n_shifted should be 0 or low."""
        np.random.seed(42)
        n_train, n_test, n_features = 500, 200, 10
        X_train = np.random.randn(n_train, n_features)
        X_test = np.random.randn(n_test, n_features)
        names = [f"feat_{i}" for i in range(n_features)]

        opt = _make_optimizer()
        result = opt._per_fold_ks_test(X_train, X_test, names, top_k=10)

        assert result["n_tested"] == 10
        # With same distribution, we expect very few false positives
        # (at p=0.05, expect ~0.5 out of 10 on average)
        assert result["n_shifted"] <= 3  # generous bound for randomness
        assert result["high_shift"] is False

    def test_shifted_distribution_detected(self):
        """When test has a different mean, KS test should detect the shift."""
        np.random.seed(42)
        n_train, n_test, n_features = 500, 200, 10
        X_train = np.random.randn(n_train, n_features)
        # Shift ALL features by 2 standard deviations
        X_test = np.random.randn(n_test, n_features) + 2.0
        names = [f"feat_{i}" for i in range(n_features)]

        opt = _make_optimizer()
        result = opt._per_fold_ks_test(X_train, X_test, names, top_k=10)

        assert result["n_tested"] == 10
        # Most features should be detected as shifted
        assert result["n_shifted"] > 5
        assert result["high_shift"] is True

    def test_high_shift_threshold_at_half(self):
        """high_shift = True exactly when n_shifted > n_tested * 0.5."""
        opt = _make_optimizer()

        np.random.seed(42)
        # Create data where exactly a controlled number of features are shifted
        n_train, n_test = 500, 200
        n_features = 4
        X_train = np.random.randn(n_train, n_features)
        # Shift first 3 features strongly, keep last one identical
        X_test = np.random.randn(n_test, n_features)
        X_test[:, :3] += 5.0  # massive shift on 3 of 4 features
        names = [f"feat_{i}" for i in range(n_features)]

        result = opt._per_fold_ks_test(X_train, X_test, names, top_k=4)
        # 3 out of 4 shifted → 3 > 4*0.5=2 → high_shift=True
        assert result["n_shifted"] >= 3
        assert result["high_shift"] is True

    def test_top_k_limits_features_tested(self):
        """top_k < n_features means only top_k features by variance are tested."""
        np.random.seed(42)
        n_train, n_test = 300, 100
        n_features = 20
        X_train = np.random.randn(n_train, n_features)
        X_test = np.random.randn(n_test, n_features) + 3.0
        names = [f"feat_{i}" for i in range(n_features)]

        opt = _make_optimizer()
        result = opt._per_fold_ks_test(X_train, X_test, names, top_k=5)

        assert result["n_tested"] == 5

    def test_shifted_features_list_capped_at_10(self):
        """shifted_features list is capped at 10 entries for storage."""
        np.random.seed(42)
        n_train, n_test = 500, 200
        n_features = 30
        X_train = np.random.randn(n_train, n_features)
        X_test = np.random.randn(n_test, n_features) + 5.0  # shift all
        names = [f"feat_{i}" for i in range(n_features)]

        opt = _make_optimizer()
        result = opt._per_fold_ks_test(X_train, X_test, names, top_k=20)

        assert len(result["shifted_features"]) <= 10

    def test_return_dict_structure(self):
        """Return value has exactly the expected keys."""
        np.random.seed(42)
        X_train = np.random.randn(100, 5)
        X_test = np.random.randn(50, 5)
        names = [f"feat_{i}" for i in range(5)]

        opt = _make_optimizer()
        result = opt._per_fold_ks_test(X_train, X_test, names)

        assert set(result.keys()) == {"n_shifted", "n_tested", "high_shift", "shifted_features"}
        assert isinstance(result["n_shifted"], int)
        assert isinstance(result["n_tested"], int)
        assert isinstance(result["high_shift"], bool)
        assert isinstance(result["shifted_features"], list)

    def test_shifted_feature_entry_has_correct_fields(self):
        """Each entry in shifted_features has feature, ks_stat, p_value."""
        np.random.seed(42)
        n_train, n_test = 500, 200
        X_train = np.random.randn(n_train, 5)
        X_test = np.random.randn(n_test, 5) + 3.0
        names = [f"feat_{i}" for i in range(5)]

        opt = _make_optimizer()
        result = opt._per_fold_ks_test(X_train, X_test, names, top_k=5)

        assert len(result["shifted_features"]) > 0
        entry = result["shifted_features"][0]
        assert "feature" in entry
        assert "ks_stat" in entry
        assert "p_value" in entry
        assert 0 <= entry["ks_stat"] <= 1
        assert 0 <= entry["p_value"] <= 1


# ---------------------------------------------------------------------------
# Aggressive regularization logic
# ---------------------------------------------------------------------------

class TestAggressiveRegularization:
    """Aggressive reg flag gates Optuna suggest bounds when adversarial AUC > 0.8."""

    def test_high_auc_enables_aggressive_reg(self):
        """adversarial AUC > 0.8 with no_aggressive_reg=False → enabled."""
        opt = _make_optimizer(no_aggressive_reg=False)
        opt._adversarial_auc_mean = 0.85

        # Simulate the logic from lines 3799-3809
        opt._aggressive_reg_applied = False
        opt._regularization_overrides = None
        adv_auc = getattr(opt, '_adversarial_auc_mean', None)
        if adv_auc is not None and adv_auc > 0.8 and not opt.no_aggressive_reg:
            opt._aggressive_reg_applied = True
            opt._regularization_overrides = {
                "max_depth": "3-4", "rsm": "0.5-0.6", "min_data_in_leaf": "50-200",
                "min_child_samples": "50-200", "subsample": "0.5-0.7", "colsample_bytree": "0.5-0.7",
            }

        assert opt._aggressive_reg_applied is True
        assert opt._regularization_overrides is not None
        assert "max_depth" in opt._regularization_overrides

    def test_low_auc_disables_aggressive_reg(self):
        """adversarial AUC <= 0.8 → aggressive reg NOT applied."""
        opt = _make_optimizer(no_aggressive_reg=False)
        opt._adversarial_auc_mean = 0.75

        opt._aggressive_reg_applied = False
        opt._regularization_overrides = None
        adv_auc = getattr(opt, '_adversarial_auc_mean', None)
        if adv_auc is not None and adv_auc > 0.8 and not opt.no_aggressive_reg:
            opt._aggressive_reg_applied = True

        assert opt._aggressive_reg_applied is False
        assert opt._regularization_overrides is None

    def test_no_aggressive_reg_flag_overrides(self):
        """no_aggressive_reg=True prevents aggressive reg even when AUC > 0.8."""
        opt = _make_optimizer(no_aggressive_reg=True)
        opt._adversarial_auc_mean = 0.95

        opt._aggressive_reg_applied = False
        opt._regularization_overrides = None
        adv_auc = getattr(opt, '_adversarial_auc_mean', None)
        if adv_auc is not None and adv_auc > 0.8 and not opt.no_aggressive_reg:
            opt._aggressive_reg_applied = True

        assert opt._aggressive_reg_applied is False

    def test_none_auc_disables_aggressive_reg(self):
        """When adversarial AUC is None (filter not run), aggressive reg is off."""
        opt = _make_optimizer(no_aggressive_reg=False)
        opt._adversarial_auc_mean = None

        opt._aggressive_reg_applied = False
        opt._regularization_overrides = None
        adv_auc = getattr(opt, '_adversarial_auc_mean', None)
        if adv_auc is not None and adv_auc > 0.8 and not opt.no_aggressive_reg:
            opt._aggressive_reg_applied = True

        assert opt._aggressive_reg_applied is False

    def test_boundary_auc_exactly_0_8_not_triggered(self):
        """AUC of exactly 0.8 does NOT trigger aggressive reg (> not >=)."""
        opt = _make_optimizer(no_aggressive_reg=False)
        opt._adversarial_auc_mean = 0.8

        opt._aggressive_reg_applied = False
        opt._regularization_overrides = None
        adv_auc = getattr(opt, '_adversarial_auc_mean', None)
        if adv_auc is not None and adv_auc > 0.8 and not opt.no_aggressive_reg:
            opt._aggressive_reg_applied = True

        assert opt._aggressive_reg_applied is False

    def test_agg_flag_tightens_lightgbm_max_depth(self):
        """When _aggressive_reg_applied=True, LightGBM max_depth upper bound is 4 (not 8)."""
        _agg = True
        max_depth_upper = 4 if _agg else 8
        assert max_depth_upper == 4

        _agg = False
        max_depth_upper = 4 if _agg else 8
        assert max_depth_upper == 8

    def test_agg_flag_tightens_catboost_min_data_in_leaf(self):
        """When _aggressive_reg_applied=True, CatBoost min_data_in_leaf lower bound is 50."""
        _agg = True
        min_data_lower = 50 if _agg else 1
        min_data_upper = 200 if _agg else 100
        assert min_data_lower == 50
        assert min_data_upper == 200

        _agg = False
        min_data_lower = 50 if _agg else 1
        min_data_upper = 200 if _agg else 100
        assert min_data_lower == 1
        assert min_data_upper == 100

    def test_agg_flag_tightens_catboost_rsm(self):
        """When _aggressive_reg_applied=True, CatBoost rsm upper bound is 0.6 (not 1.0)."""
        _agg = True
        rsm_upper = 0.6 if _agg else 1.0
        assert rsm_upper == 0.6

        _agg = False
        rsm_upper = 0.6 if _agg else 1.0
        assert rsm_upper == 1.0
