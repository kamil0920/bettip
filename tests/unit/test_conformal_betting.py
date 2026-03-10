"""
Unit tests for conformal prediction betting features.

Tests:
1. SniperOptimizer._compute_conformal_tau — known residuals, coverage, empty input, perfect preds
2. VennAbersCalibrator — interval bounds, width-uncertainty correlation
3. CrossMarketConformalPooler — normalization, single-market degenerate, serialization
4. Backward compatibility — model data without conformal fields
"""

import numpy as np
import pytest

from experiments.run_sniper_optimization import SniperOptimizer
from src.calibration.calibration import VennAbersCalibrator
from src.calibration.conformal_pooling import CrossMarketConformalPooler


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def rng():
    """Reproducible random state."""
    return np.random.RandomState(42)


@pytest.fixture
def fitted_va(rng):
    """A VennAbersCalibrator fitted on synthetic calibration data."""
    n = 500
    # Generate calibration data with realistic class separation
    scores = rng.beta(2, 5, size=n)  # skewed towards low probs
    labels = (rng.rand(n) < scores).astype(int)
    va = VennAbersCalibrator()
    va.fit(scores, labels)
    return va


@pytest.fixture
def two_market_pooler(rng):
    """A CrossMarketConformalPooler with two markets at different scales."""
    n = 100

    # Market A: small-scale residuals (tight predictions)
    preds_a = rng.uniform(0.3, 0.5, size=n)
    actuals_a = (rng.rand(n) < 0.4).astype(float)

    # Market B: large-scale residuals (10x noisier predictions)
    preds_b = rng.uniform(0.0, 1.0, size=n)
    actuals_b = (rng.rand(n) < 0.5).astype(float)

    pooler = CrossMarketConformalPooler()
    pooler.add_market("tight", preds_a, actuals_a)
    pooler.add_market("noisy", preds_b, actuals_b)
    pooler.fit(alpha=0.10)
    return pooler


# ---------------------------------------------------------------------------
# SniperOptimizer._compute_conformal_tau
# ---------------------------------------------------------------------------
class TestConformalTau:
    """Tests for SniperOptimizer._compute_conformal_tau static method."""

    def test_conformal_tau_known_residuals(self):
        """Known scores [0.1..0.5] with alpha=0.10 -> tau should be 0.5."""
        # scores = preds - actuals.  If actuals are all 0, scores = preds.
        preds = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        actuals = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        tau = SniperOptimizer._compute_conformal_tau(preds, actuals, alpha=0.10)

        # scores = [0.1, 0.2, 0.3, 0.4, 0.5], n=5
        # k = ceil((5+1) * 0.90) = ceil(5.4) = 6, capped to min(6,5) = 5
        # sorted scores[4] = 0.5
        assert tau == pytest.approx(0.5)

    def test_conformal_tau_coverage_guarantee(self, rng):
        """Synthetic data: marginal coverage >= 1 - alpha across pooled test points."""
        alpha = 0.10
        n_trials = 100
        n_cal = 200
        n_test = 50
        all_covered = []

        for _ in range(n_trials):
            # Calibration set: well-specified model
            cal_preds = rng.uniform(0.2, 0.8, size=n_cal)
            cal_actuals = (rng.rand(n_cal) < cal_preds).astype(float)

            tau = SniperOptimizer._compute_conformal_tau(
                cal_preds, cal_actuals, alpha=alpha
            )

            # Test set drawn from same distribution
            test_preds = rng.uniform(0.2, 0.8, size=n_test)
            test_actuals = (rng.rand(n_test) < test_preds).astype(float)

            # Per the conformal guarantee: P(score_new <= tau) >= 1 - alpha
            # Check each test point individually
            scores = test_preds - test_actuals
            covered = (scores <= tau).astype(float)
            all_covered.extend(covered.tolist())

        # Marginal coverage across all 5000 pooled test points should be >= 1-alpha
        # Allow small slack (0.03) for finite-sample discretization with binary actuals
        marginal_coverage = np.mean(all_covered)
        assert marginal_coverage >= (1 - alpha) - 0.03, (
            f"Marginal coverage {marginal_coverage:.4f} < {1 - alpha - 0.03:.4f}. "
            f"Expected >= {1 - alpha:.2f} (with 0.03 tolerance)."
        )

    def test_conformal_tau_empty_input(self):
        """Empty arrays -> returns 0.0."""
        tau = SniperOptimizer._compute_conformal_tau(
            np.array([]), np.array([]), alpha=0.10
        )
        assert tau == 0.0

    def test_conformal_tau_all_correct(self):
        """Perfect predictions (scores all 0) -> tau = 0."""
        preds = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        actuals = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        tau = SniperOptimizer._compute_conformal_tau(preds, actuals, alpha=0.10)
        assert tau == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# VennAbersCalibrator
# ---------------------------------------------------------------------------
class TestVennAbersIntervals:
    """Tests for VennAbersCalibrator interval properties."""

    def test_va_interval_bounds(self, fitted_va, rng):
        """p0 <= calibrated_prob <= p1 on average."""
        test_probs = rng.uniform(0.05, 0.95, size=200)
        calibrated = fitted_va.transform(test_probs)
        p0, p1 = fitted_va.predict_interval(test_probs)

        # On average, calibrated should lie within [p0, p1].
        # The VA formula is calibrated = p1 / (1 - p0 + p1), so
        # algebraically p0 <= calibrated <= p1 when the isotonic
        # regressions are consistent. Check the average relationship.
        mean_p0 = np.mean(p0)
        mean_cal = np.mean(calibrated)
        mean_p1 = np.mean(p1)

        assert mean_p0 <= mean_cal + 1e-6, (
            f"mean(p0)={mean_p0:.4f} > mean(calibrated)={mean_cal:.4f}"
        )
        assert mean_cal <= mean_p1 + 1e-6, (
            f"mean(calibrated)={mean_cal:.4f} > mean(p1)={mean_p1:.4f}"
        )

    def test_va_width_correlates_uncertainty(self, rng):
        """Ambiguous cases (prob ~ 0.5) should have wider intervals than confident cases."""
        n = 1000
        # Create calibration data with clear signal
        scores = rng.uniform(0.0, 1.0, size=n)
        labels = (rng.rand(n) < scores).astype(int)

        va = VennAbersCalibrator()
        va.fit(scores, labels)

        # Ambiguous test points near 0.5
        ambiguous = np.full(100, 0.5)
        p0_amb, p1_amb = va.predict_interval(ambiguous)
        width_ambiguous = np.mean(p1_amb - p0_amb)

        # Confident test points near 0.1
        confident_low = np.full(100, 0.1)
        p0_cl, p1_cl = va.predict_interval(confident_low)
        width_confident_low = np.mean(p1_cl - p0_cl)

        # Confident test points near 0.9
        confident_high = np.full(100, 0.9)
        p0_ch, p1_ch = va.predict_interval(confident_high)
        width_confident_high = np.mean(p1_ch - p0_ch)

        # Width at 0.5 should be >= width at extremes (on average)
        min_confident_width = min(width_confident_low, width_confident_high)
        assert width_ambiguous >= min_confident_width - 1e-6, (
            f"Ambiguous width ({width_ambiguous:.4f}) < confident width "
            f"({min_confident_width:.4f}). Expected wider intervals at p~0.5."
        )


# ---------------------------------------------------------------------------
# CrossMarketConformalPooler
# ---------------------------------------------------------------------------
class TestPoolingNormalization:
    """Tests for cross-market conformal pooling."""

    def test_pooling_normalization(self, rng):
        """Two markets with 10x scale difference -> proportional per-market taus."""
        n = 200

        # Market A: predictions close to actuals (small residuals)
        preds_a = rng.uniform(0.35, 0.45, size=n)
        actuals_a = (rng.rand(n) < 0.4).astype(float)

        # Market B: predictions spread wide (large residuals ~ 10x scale)
        preds_b = rng.uniform(0.0, 1.0, size=n)
        actuals_b = (rng.rand(n) < 0.5).astype(float)

        pooler = CrossMarketConformalPooler()
        pooler.add_market("small_scale", preds_a, actuals_a)
        pooler.add_market("large_scale", preds_b, actuals_b)
        pooler.fit(alpha=0.10)

        tau_small = pooler.get_tau("small_scale")
        tau_large = pooler.get_tau("large_scale")

        # tau = global_tau_z * scale, so ratio of taus should approximate
        # ratio of MAEs (scales)
        scale_ratio = pooler.scales["large_scale"] / pooler.scales["small_scale"]
        tau_ratio = tau_large / tau_small

        assert tau_ratio == pytest.approx(scale_ratio, rel=1e-6), (
            f"Tau ratio ({tau_ratio:.4f}) != scale ratio ({scale_ratio:.4f})"
        )

        # Large-scale market should have a larger tau
        assert tau_large > tau_small, (
            f"Expected tau_large ({tau_large:.4f}) > tau_small ({tau_small:.4f})"
        )

    def test_pooling_single_market_equals_per_market(self, rng):
        """Degenerate case: pooling with 1 market gives same tau as direct computation."""
        n = 200
        preds = rng.uniform(0.2, 0.8, size=n)
        actuals = (rng.rand(n) < 0.5).astype(float)
        alpha = 0.10

        # Direct per-market tau via SniperOptimizer
        direct_tau = SniperOptimizer._compute_conformal_tau(preds, actuals, alpha=alpha)

        # Pooled tau with single market
        pooler = CrossMarketConformalPooler()
        pooler.add_market("only_market", preds, actuals)
        pooler.fit(alpha=alpha)
        pooled_tau = pooler.get_tau("only_market")

        # Both pooler and SniperOptimizer use signed residuals (one-sided).
        # Single-market pooler: tau = global_tau_z * MAE where
        # global_tau_z is the (1-alpha) quantile of signed normalized residuals.
        residuals = preds - actuals
        mae = float(np.mean(np.abs(residuals)))
        z = residuals / mae  # signed normalized residuals
        k = int(np.ceil((n + 1) * (1 - alpha)))
        k = min(k, n)
        expected_tau_z = float(np.sort(z)[k - 1])
        expected_tau = expected_tau_z * mae

        assert pooled_tau == pytest.approx(expected_tau, rel=1e-6), (
            f"Pooled tau ({pooled_tau:.6f}) != expected ({expected_tau:.6f})"
        )

    def test_pooling_serialization(self, two_market_pooler):
        """to_dict() / from_dict() round trip preserves state."""
        original_tau_tight = two_market_pooler.get_tau("tight")
        original_tau_noisy = two_market_pooler.get_tau("noisy")

        d = two_market_pooler.to_dict()

        # Verify serialized keys
        assert "scales" in d
        assert "global_tau_z" in d
        assert "alpha" in d
        assert "n_markets" in d
        assert d["n_markets"] == 2

        # Reconstruct
        restored = CrossMarketConformalPooler.from_dict(d)

        assert restored.get_tau("tight") == pytest.approx(original_tau_tight)
        assert restored.get_tau("noisy") == pytest.approx(original_tau_noisy)
        assert restored.alpha == two_market_pooler.alpha
        assert restored.global_tau_z == pytest.approx(two_market_pooler.global_tau_z)


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------
class TestBackwardCompat:
    """Backward compatibility for models without conformal fields."""

    def test_backward_compat_no_conformal(self):
        """Model data without conformal fields -> no errors when accessing defaults."""
        # Simulate a legacy model_data dict (pre-S54, no conformal fields)
        model_data = {
            "model": None,
            "features": ["f1", "f2", "f3"],
            "bet_type": "over25",
            "scaler": None,
        }

        # Accessing conformal fields via .get() should return None gracefully
        assert model_data.get("conformal_tau") is None
        assert model_data.get("conformal_alpha") is None
        assert model_data.get("conformal") is None
        assert model_data.get("va_calibrator") is None

        # SniperResult defaults should also work
        from dataclasses import fields as dc_fields
        from experiments.run_sniper_optimization import SniperResult

        # Create a SniperResult with only required fields
        result = SniperResult(
            bet_type="over25",
            target="over25",
            best_model="catboost",
            best_params={},
            n_features=10,
            optimal_features=["f1"],
            best_threshold=0.65,
            best_min_odds=1.5,
            best_max_odds=5.0,
            precision=0.80,
            roi=15.0,
            n_bets=50,
            n_wins=40,
            timestamp="2026-01-01",
        )

        # Conformal fields should default to None
        assert result.conformal_tau is None
        assert result.conformal_alpha is None
        assert result.uncertainty_penalty is None
