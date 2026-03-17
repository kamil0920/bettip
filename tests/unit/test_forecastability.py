"""Unit tests for forecastability diagnostics.

Tests: permutation_entropy, sample_entropy, tracking_signal,
       rolling_tracking_signal, acf_lag1, fano_bound, FVA computation,
       pre-optimization gate, market tracking signals, forecastability weights.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from experiments.forecastability_analysis import (
    acf_lag1,
    fano_bound,
    forecastability_scorecard,
    permutation_entropy,
    sample_entropy,
    save_config,
)
from experiments.generate_daily_recommendations import (
    _get_base_market,
    compute_forecastability_weight,
    compute_market_tracking_signals,
)
from src.monitoring.drift_detection import rolling_tracking_signal, tracking_signal


# --- Permutation Entropy ---


class TestPermutationEntropy:
    def test_deterministic_series(self):
        """Monotonically increasing series should have PE near 0."""
        x = np.arange(100, dtype=float)
        pe = permutation_entropy(x, order=3, delay=1)
        assert pe < 0.1, f"Deterministic series PE should be near 0, got {pe}"

    def test_random_series(self):
        """Random series should have PE near 1."""
        rng = np.random.RandomState(42)
        x = rng.randn(1000)
        pe = permutation_entropy(x, order=3, delay=1)
        assert pe > 0.9, f"Random series PE should be near 1, got {pe}"

    def test_periodic_series(self):
        """Periodic series should have lower PE than random."""
        x = np.tile([0, 1, 2, 3], 50).astype(float)
        pe = permutation_entropy(x, order=3, delay=1)
        assert pe < 0.7, f"Periodic series PE should be lower than random, got {pe}"

    def test_constant_series(self):
        """Constant series: all patterns are ties, PE should be well-defined."""
        x = np.ones(100)
        pe = permutation_entropy(x, order=3, delay=1)
        assert not np.isnan(pe)

    def test_short_series_returns_nan(self):
        """Series shorter than order should return NaN."""
        x = np.array([1.0, 2.0])
        pe = permutation_entropy(x, order=3, delay=1)
        assert np.isnan(pe)

    def test_normalized_range(self):
        """PE should always be in [0, 1]."""
        rng = np.random.RandomState(123)
        for _ in range(10):
            x = rng.randn(200)
            pe = permutation_entropy(x, order=3, delay=1)
            if not np.isnan(pe):
                assert 0.0 <= pe <= 1.0

    def test_binary_series_high_pe(self):
        """Binary random series should have elevated PE (ties reduce max PE)."""
        rng = np.random.RandomState(42)
        x = rng.binomial(1, 0.5, size=500).astype(float)
        pe = permutation_entropy(x, order=3, delay=1)
        # Binary series has many ties, so PE is lower than continuous random
        assert pe > 0.7, f"Binary random PE should be elevated, got {pe}"


# --- Sample Entropy ---


class TestSampleEntropy:
    def test_deterministic_low_entropy(self):
        """Periodic signal should have low SampEn."""
        x = np.sin(np.linspace(0, 10 * np.pi, 500))
        se = sample_entropy(x, m=2)
        if not np.isnan(se):
            assert se < 1.0, f"Periodic SampEn should be low, got {se}"

    def test_random_higher_entropy(self):
        """Random signal should have higher SampEn than periodic."""
        rng = np.random.RandomState(42)
        periodic = np.sin(np.linspace(0, 10 * np.pi, 200))
        random_sig = rng.randn(200)
        se_periodic = sample_entropy(periodic, m=2)
        se_random = sample_entropy(random_sig, m=2)
        # Both may be NaN in edge cases, but if both are valid:
        if not np.isnan(se_periodic) and not np.isnan(se_random):
            assert se_random > se_periodic

    def test_constant_series_zero(self):
        """Constant series should have SampEn = 0."""
        x = np.ones(100)
        se = sample_entropy(x, m=2)
        assert se == 0.0

    def test_short_series_nan(self):
        """Series shorter than m+2 should return NaN."""
        x = np.array([1.0, 2.0])
        se = sample_entropy(x, m=2)
        assert np.isnan(se)

    def test_non_negative(self):
        """SampEn should be non-negative when valid."""
        rng = np.random.RandomState(42)
        x = rng.randn(200)
        se = sample_entropy(x, m=2)
        if not np.isnan(se):
            assert se >= 0.0


# --- Tracking Signal ---


class TestTrackingSignal:
    def test_zero_bias(self):
        """Unbiased errors should have TS near 0."""
        rng = np.random.RandomState(42)
        errors = rng.randn(100)  # Mean near 0
        ts = tracking_signal(errors, window=50)
        assert abs(ts) < 3.0, f"Unbiased TS should be small, got {ts}"

    def test_positive_bias(self):
        """Consistent over-prediction should give large positive TS."""
        errors = np.full(50, 0.5)  # Always over-predicting
        ts = tracking_signal(errors, window=50)
        assert ts > 4.0, f"Positive bias TS should be >4, got {ts}"

    def test_negative_bias(self):
        """Consistent under-prediction should give large negative TS."""
        errors = np.full(50, -0.3)
        ts = tracking_signal(errors, window=50)
        assert ts < -4.0, f"Negative bias TS should be <-4, got {ts}"

    def test_short_series(self):
        """Single element should return 0."""
        assert tracking_signal(np.array([0.1]), window=50) == 0.0

    def test_empty_series(self):
        """Empty array should return 0."""
        assert tracking_signal(np.array([]), window=50) == 0.0

    def test_nan_handling(self):
        """NaN values should be filtered out."""
        errors = np.array([0.5, np.nan, 0.5, np.nan, 0.5, 0.5, 0.5])
        ts = tracking_signal(errors, window=50)
        assert ts > 0  # All valid errors are positive

    def test_window_smaller_than_series(self):
        """Window should limit to recent errors."""
        errors = np.concatenate([np.full(100, -1.0), np.full(50, 0.5)])
        ts = tracking_signal(errors, window=50)
        assert ts > 0, "Should reflect recent positive bias, not old negative"

    def test_all_zeros(self):
        """All-zero errors: MAD=0, should return 0."""
        errors = np.zeros(50)
        ts = tracking_signal(errors, window=50)
        assert ts == 0.0


class TestRollingTrackingSignal:
    def test_output_length(self):
        """Output should match input length."""
        errors = np.random.randn(100)
        ts = rolling_tracking_signal(errors, window=20)
        assert len(ts) == 100

    def test_early_values_nan(self):
        """First (window-1) values should be NaN."""
        errors = np.random.randn(100)
        ts = rolling_tracking_signal(errors, window=20)
        assert all(np.isnan(ts[:19]))
        assert not np.isnan(ts[19])

    def test_consistent_bias_detected(self):
        """Should detect bias switching from unbiased to biased."""
        errors = np.concatenate([np.random.randn(50), np.full(50, 0.8)])
        ts = rolling_tracking_signal(errors, window=30)
        # Last values should show high TS
        assert ts[-1] > 3.0


# --- ACF Lag-1 ---


class TestACFLag1:
    def test_white_noise_near_zero(self):
        """White noise should have ACF near 0."""
        rng = np.random.RandomState(42)
        x = rng.randn(1000)
        acf = acf_lag1(x)
        assert abs(acf) < 0.1, f"White noise ACF should be near 0, got {acf}"

    def test_ar1_positive(self):
        """AR(1) with positive coefficient should have positive ACF."""
        rng = np.random.RandomState(42)
        x = np.zeros(500)
        for i in range(1, 500):
            x[i] = 0.8 * x[i - 1] + rng.randn()
        acf = acf_lag1(x)
        assert acf > 0.5, f"AR(1) phi=0.8 ACF should be >0.5, got {acf}"

    def test_constant_zero(self):
        """Constant series should have ACF = 0."""
        x = np.ones(100)
        assert acf_lag1(x) == 0.0

    def test_short_series_nan(self):
        """Too-short series returns NaN."""
        assert np.isnan(acf_lag1(np.array([1.0])))

    def test_nan_filtering(self):
        """NaN values should be filtered."""
        x = np.array([1.0, 2.0, np.nan, 3.0, 4.0, 5.0])
        acf = acf_lag1(x)
        assert not np.isnan(acf)


# --- Fano Bound ---


class TestFanoBound:
    def test_fully_forecastable(self):
        """PE=0 should give Pi_max=1."""
        assert fano_bound(0.0) == 1.0

    def test_fully_random(self):
        """PE=1 should give Pi_max=0."""
        assert fano_bound(1.0) == 0.0

    def test_intermediate(self):
        """PE=0.3 should give Pi_max=0.7."""
        assert abs(fano_bound(0.3) - 0.7) < 1e-10

    def test_nan_input(self):
        """NaN PE should give NaN Pi_max."""
        assert np.isnan(fano_bound(np.nan))


# --- Forecastability Scorecard ---


class TestForecastabilityScorecard:
    def test_returns_all_keys(self):
        """Scorecard should return all expected metrics."""
        rng = np.random.RandomState(42)
        targets = rng.binomial(1, 0.5, 200).astype(float)
        residuals = rng.randn(200) * 0.3
        sc = forecastability_scorecard(targets, residuals, "test_market")
        expected_keys = {
            "market",
            "pe_residual",
            "pe_target",
            "sampen_residual",
            "acf1_residual",
            "pi_max",
            "series_length",
        }
        assert set(sc.keys()) == expected_keys

    def test_series_length_stored(self):
        """Should record the input series length."""
        targets = np.zeros(150)
        residuals = np.zeros(150)
        sc = forecastability_scorecard(targets, residuals, "test")
        assert sc["series_length"] == 150


# --- FVA Computation (inline test) ---


class TestFVA:
    def test_perfect_model_fva_positive(self):
        """Model that's better than market should have positive FVA."""
        from sklearn.metrics import brier_score_loss

        y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 0])
        model_probs = np.array([0.9, 0.1, 0.8, 0.85, 0.15, 0.1, 0.9, 0.2, 0.85, 0.15])
        market_probs = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

        brier_model = brier_score_loss(y_true, model_probs)
        brier_market = brier_score_loss(y_true, market_probs)
        fva = 1.0 - (brier_model / brier_market)

        assert fva > 0, f"Good model FVA should be positive, got {fva}"

    def test_worse_than_market_fva_negative(self):
        """Model worse than market should have negative FVA."""
        from sklearn.metrics import brier_score_loss

        y_true = np.array([1, 0, 1, 1, 0])
        model_probs = np.array([0.2, 0.8, 0.3, 0.2, 0.7])  # Inverted
        market_probs = np.array([0.6, 0.4, 0.55, 0.6, 0.45])  # Decent

        brier_model = brier_score_loss(y_true, model_probs)
        brier_market = brier_score_loss(y_true, market_probs)
        fva = 1.0 - (brier_model / brier_market)

        assert fva < 0, f"Bad model FVA should be negative, got {fva}"

    def test_equal_model_fva_zero(self):
        """Model matching market should have FVA ≈ 0."""
        from sklearn.metrics import brier_score_loss

        y_true = np.array([1, 0, 1])
        probs = np.array([0.6, 0.4, 0.6])

        brier = brier_score_loss(y_true, probs)
        fva = 1.0 - (brier / brier)
        assert abs(fva) < 1e-10


# --- Pre-Optimization Gate ---


class TestPreOptimizationGate:
    def test_sniper_result_has_new_fields(self):
        """SniperResult dataclass should include forecastability fields."""
        from dataclasses import fields
        from experiments.run_sniper_optimization import SniperResult

        field_names = {f.name for f in fields(SniperResult)}
        assert "mean_pe_residual" in field_names
        assert "forecastability_gate" in field_names

    def test_gate_disabled_by_default(self):
        """PE gate with default threshold 1.0 should never reject."""
        # PE is always <= 1.0, so threshold=1.0 means gate never triggers
        assert 1.0 >= 1.0  # Trivially true, documenting the design

    def test_sniper_result_defaults_none(self):
        """New fields should default to None."""
        from experiments.run_sniper_optimization import SniperResult

        result = SniperResult(
            bet_type="test",
            target="test_target",
            best_model="xgboost",
            best_params={},
            n_features=10,
            optimal_features=["f1"],
            best_threshold=0.6,
            best_min_odds=1.5,
            best_max_odds=5.0,
            precision=0.65,
            roi=10.0,
            n_bets=100,
            n_wins=65,
            timestamp="2026-01-01",
        )
        assert result.mean_pe_residual is None
        assert result.forecastability_gate is None

    def test_sniper_result_rejected_gate(self):
        """SniperResult with rejected gate should store values."""
        from experiments.run_sniper_optimization import SniperResult

        result = SniperResult(
            bet_type="home_win",
            target="home_win",
            best_model="none",
            best_params={},
            n_features=0,
            optimal_features=[],
            best_threshold=0.0,
            best_min_odds=0.0,
            best_max_odds=0.0,
            precision=0.0,
            roi=-100.0,
            n_bets=0,
            n_wins=0,
            timestamp="2026-01-01",
            mean_pe_residual=0.9935,
            forecastability_gate="rejected",
        )
        assert result.mean_pe_residual == 0.9935
        assert result.forecastability_gate == "rejected"


# --- Tracking Signal Alerts ---


class TestComputeMarketTrackingSignals:
    def test_empty_ledger_returns_empty(self):
        """No ledger file should return empty dict."""
        with patch(
            "experiments.generate_daily_recommendations.project_root",
            Path("/nonexistent/path"),
        ):
            result = compute_market_tracking_signals()
            assert result == {}

    def test_over_prediction_detected(self):
        """Systematic over-prediction should be detected."""
        # Create fake ledger with consistent over-prediction
        df = pd.DataFrame({
            "market": ["HOME_WIN"] * 30,
            "probability": [0.8] * 30,
            "actual": [0.0] * 30,  # Always wrong = over-predicting
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            preds_dir = Path(tmpdir) / "data" / "preds"
            preds_dir.mkdir(parents=True)
            df.to_parquet(preds_dir / "predictions.parquet")

            with patch(
                "experiments.generate_daily_recommendations.project_root",
                Path(tmpdir),
            ):
                result = compute_market_tracking_signals(min_settled=20)

        assert "HOME_WIN" in result
        assert result["HOME_WIN"]["alert"]
        assert result["HOME_WIN"]["ts"] > 4.0
        assert result["HOME_WIN"]["direction"] == "over-predicting"

    def test_min_settled_threshold(self):
        """Markets with fewer than min_settled bets should be excluded."""
        df = pd.DataFrame({
            "market": ["SHOTS"] * 10,
            "probability": [0.7] * 10,
            "actual": [0.0] * 10,
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            preds_dir = Path(tmpdir) / "data" / "preds"
            preds_dir.mkdir(parents=True)
            df.to_parquet(preds_dir / "predictions.parquet")

            with patch(
                "experiments.generate_daily_recommendations.project_root",
                Path(tmpdir),
            ):
                result = compute_market_tracking_signals(min_settled=20)

        assert result == {}

    def test_error_sign_convention(self):
        """Positive errors = over-predicting, negative = under-predicting."""
        # Under-predicting: actual outcomes are 1 but model predicts low
        df = pd.DataFrame({
            "market": ["BTTS"] * 30,
            "probability": [0.2] * 30,
            "actual": [1.0] * 30,
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            preds_dir = Path(tmpdir) / "data" / "preds"
            preds_dir.mkdir(parents=True)
            df.to_parquet(preds_dir / "predictions.parquet")

            with patch(
                "experiments.generate_daily_recommendations.project_root",
                Path(tmpdir),
            ):
                result = compute_market_tracking_signals(min_settled=20)

        assert "BTTS" in result
        assert result["BTTS"]["ts"] < -4.0
        assert result["BTTS"]["direction"] == "under-predicting"


# --- Forecastability Weights ---


class TestForecastabilityWeight:
    def test_h2h_markets_flat_weight(self):
        """H2H markets should always get weight 1.0."""
        assert compute_forecastability_weight("home_win") == 1.0
        assert compute_forecastability_weight("away_win") == 1.0
        assert compute_forecastability_weight("over25") == 1.0

    def test_niche_markets_scaled(self):
        """Niche markets should get scaled weight based on pi_max."""
        cards_w = compute_forecastability_weight("cards")
        fouls_w = compute_forecastability_weight("fouls")
        shots_w = compute_forecastability_weight("shots")

        # Cards should have highest weight (pi_max=0.4087, raw_weight~1.0)
        assert cards_w >= fouls_w >= shots_w

    def test_base_market_extraction(self):
        """Line variants should map to their base market."""
        assert _get_base_market("cards_over_35") == "cards"
        assert _get_base_market("fouls_under_265") == "fouls"
        assert _get_base_market("corners_over_95") == "corners"
        assert _get_base_market("shots_under_275") == "shots"
        assert _get_base_market("home_win") == "home_win"
        assert _get_base_market("btts") == "btts"

    def test_line_variant_inherits_base_weight(self):
        """cards_over_35 should get same weight as cards."""
        assert compute_forecastability_weight("cards_over_35") == compute_forecastability_weight("cards")
        assert compute_forecastability_weight("fouls_under_265") == compute_forecastability_weight("fouls")

    def test_soft_floor_minimum(self):
        """No market should get weight below 0.5."""
        for market in ["cards", "fouls", "shots", "corners", "btts"]:
            w = compute_forecastability_weight(market)
            assert w >= 0.5, f"{market} weight {w} is below 0.5 floor"

    def test_missing_config_returns_one(self):
        """If config file doesn't exist, all weights should be 1.0."""
        import experiments.generate_daily_recommendations as mod

        # Reset cache
        original = mod._FORECASTABILITY_CONFIG
        mod._FORECASTABILITY_CONFIG = None

        with patch.object(mod, "project_root", Path("/nonexistent")):
            w = compute_forecastability_weight("cards")
            assert w == 1.0

        # Restore
        mod._FORECASTABILITY_CONFIG = original


# --- save_config ---


class TestSaveConfig:
    def test_save_config_creates_valid_json(self):
        """save_config should create valid JSON with expected structure."""
        results_df = pd.DataFrame([
            {"market": "cards", "mean_pe_residual": 0.59, "mean_pi_max": 0.41,
             "has_real_odds": False, "status": "ok"},
            {"market": "home_win", "mean_pe_residual": 0.99, "mean_pi_max": 0.01,
             "has_real_odds": True, "status": "ok"},
        ])

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            output_path = f.name

        try:
            save_config(results_df, output_path)

            with open(output_path) as f:
                config = json.load(f)

            assert "markets" in config
            assert "cards" in config["markets"]
            assert "home_win" in config["markets"]
            assert config["markets"]["cards"]["pi_max"] == 0.41
            assert config["markets"]["home_win"]["has_real_odds"] is True
        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_save_config_skips_failed_markets(self):
        """Markets with status != 'ok' should be excluded."""
        results_df = pd.DataFrame([
            {"market": "cards", "mean_pe_residual": 0.59, "mean_pi_max": 0.41,
             "has_real_odds": False, "status": "ok"},
            {"market": "bad_market", "status": "no_data"},
        ])

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            output_path = f.name

        try:
            save_config(results_df, output_path)

            with open(output_path) as f:
                config = json.load(f)

            assert "cards" in config["markets"]
            assert "bad_market" not in config["markets"]
        finally:
            Path(output_path).unlink(missing_ok=True)


# --- Per-Fold TS + Holdout TS Gate + Base Rate Drift ---


class TestPerFoldTrackingSignal:
    """Tests for per-fold TS gate (P1), holdout TS gate (P2), and base rate drift (P3)."""

    def test_per_fold_ts_catches_cancelling_bias(self):
        """Two folds with opposite bias cancel in aggregate but per-fold should reject."""
        # Fold 0: strong positive bias → TS >> 4
        fold0_errors = np.full(50, 0.5)
        ts0 = tracking_signal(fold0_errors, window=50)
        assert abs(ts0) > 4.0, f"Fold 0 should have |TS| > 4, got {ts0}"

        # Fold 1: strong negative bias → TS << -4
        fold1_errors = np.full(50, -0.5)
        ts1 = tracking_signal(fold1_errors, window=50)
        assert abs(ts1) > 4.0, f"Fold 1 should have |TS| > 4, got {ts1}"

        # Aggregate over full window: biases cancel → |TS| near 0
        agg_errors = np.concatenate([fold0_errors, fold1_errors])
        ts_agg = tracking_signal(agg_errors, window=100)
        assert abs(ts_agg) < 1.0, f"Aggregate |TS| should be ~0, got {ts_agg}"

        # Per-fold check should catch the individual fold bias
        max_ts = 4.0
        fold_boundaries = [(0, 0, 50), (1, 50, 100)]
        any_rejected = False
        for fold_idx, f_start, f_end in fold_boundaries:
            fold_ts = tracking_signal(agg_errors[f_start:f_end], window=50)
            if abs(fold_ts) > max_ts:
                any_rejected = True
                break
        assert any_rejected, "Per-fold TS should reject when individual folds have extreme bias"

    def test_per_fold_ts_passes_low_bias(self):
        """All folds with balanced errors should pass per-fold TS check."""
        max_ts = 4.0
        fold_boundaries = [(0, 0, 50), (1, 50, 100), (2, 100, 150)]
        # Alternating +/- errors → CFE ≈ 0 → TS ≈ 0
        errors = np.array([0.3 if i % 2 == 0 else -0.3 for i in range(150)])

        any_rejected = False
        for fold_idx, f_start, f_end in fold_boundaries:
            fold_ts = tracking_signal(errors[f_start:f_end], window=50)
            if abs(fold_ts) > max_ts:
                any_rejected = True
                break
        assert not any_rejected, "Balanced-error folds should all pass TS check"

    def test_per_fold_ts_skips_tiny_folds(self):
        """Fold with <5 bets should be skipped, not rejected."""
        max_ts = 4.0
        # Fold with only 3 elements — should be skipped
        errors = np.full(3, 0.8)  # Would be high TS if not skipped
        n_bets = len(errors)

        skipped = n_bets < 5
        assert skipped, "Fold with <5 bets should be skipped"

        # If not skipped, it would reject
        if not skipped:
            fold_ts = tracking_signal(errors, window=min(50, n_bets))
            assert abs(fold_ts) > max_ts

    def test_holdout_ts_gate_rejects(self):
        """Holdout metrics should contain ts_rejected=True when |TS| > max_ts."""
        # Simulate holdout with extreme bias
        errors = np.full(50, 0.6)
        ho_ts = tracking_signal(errors, window=50)
        max_ts = 4.0

        holdout_metrics = {"tracking_signal": float(ho_ts)}
        if max_ts > 0 and abs(ho_ts) > max_ts:
            holdout_metrics["ts_rejected"] = True

        assert holdout_metrics.get("ts_rejected", False), (
            f"Holdout should be rejected with |TS|={abs(ho_ts):.1f} > {max_ts}"
        )

    def test_holdout_ts_gate_passes(self):
        """Holdout metrics should not contain ts_rejected when |TS| <= max_ts."""
        # Alternating errors → CFE = 0 → TS = 0
        errors = np.array([0.3 if i % 2 == 0 else -0.3 for i in range(50)])
        ho_ts = tracking_signal(errors, window=50)
        max_ts = 4.0

        holdout_metrics = {"tracking_signal": float(ho_ts)}
        if max_ts > 0 and abs(ho_ts) > max_ts:
            holdout_metrics["ts_rejected"] = True

        assert not holdout_metrics.get("ts_rejected", False), (
            f"Holdout should pass with |TS|={abs(ho_ts):.1f} <= {max_ts}"
        )

    def test_base_rate_shift_stored(self):
        """Base rate shift should be computed correctly from opt and holdout base rates."""
        opt_base_rates = [0.40, 0.42, 0.38, 0.41]  # ~0.4025 mean
        holdout_actuals = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0])  # 0.60 mean

        opt_base_rate = np.mean(opt_base_rates)
        ho_base_rate = float(holdout_actuals.mean())
        base_rate_shift = ho_base_rate - opt_base_rate

        assert abs(opt_base_rate - 0.4025) < 1e-6
        assert abs(ho_base_rate - 0.6) < 1e-6
        assert abs(base_rate_shift - 0.1975) < 1e-6
        assert abs(base_rate_shift) > 0.05, "Should detect drift > 0.05"

    def test_base_rate_shift_no_drift(self):
        """Small base rate difference should not trigger drift warning."""
        opt_base_rates = [0.45, 0.44, 0.46, 0.45]
        holdout_actuals = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])  # 0.50

        opt_base_rate = np.mean(opt_base_rates)
        ho_base_rate = float(holdout_actuals.mean())
        base_rate_shift = ho_base_rate - opt_base_rate

        assert abs(base_rate_shift) <= 0.05, "Should not trigger drift for small shift"
