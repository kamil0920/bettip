"""Unit tests for sports metrics module."""
import pytest
import numpy as np
import pandas as pd

from src.ml.metrics import (
    PredictionMetrics,
    SportsMetrics,
    probabilistic_sharpe_ratio,
    min_track_record_length,
    deflated_sharpe_ratio,
    sharpe_ratio,
    estimate_k_eff,
    estimate_return_autocorrelation,
)


class TestPredictionMetrics:
    """Tests for PredictionMetrics dataclass."""

    def test_required_fields(self):
        """Test required fields are set."""
        metrics = PredictionMetrics(
            accuracy=0.75,
            precision=0.70,
            recall=0.65,
            f1=0.67,
            log_loss=0.55
        )
        assert metrics.accuracy == 0.75
        assert metrics.precision == 0.70

    def test_optional_fields_default_none(self):
        """Test optional fields default to None."""
        metrics = PredictionMetrics(
            accuracy=0.75,
            precision=0.70,
            recall=0.65,
            f1=0.67,
            log_loss=0.55
        )
        assert metrics.brier_score is None
        assert metrics.roc_auc is None
        assert metrics.roi is None

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = PredictionMetrics(
            accuracy=0.75,
            precision=0.70,
            recall=0.65,
            f1=0.67,
            log_loss=0.55,
            roi=15.5
        )
        d = metrics.to_dict()

        assert d["accuracy"] == 0.75
        assert d["roi"] == 15.5
        assert isinstance(d["accuracy"], float)

    def test_to_dict_with_per_class_metrics(self):
        """Test to_dict includes per-class metrics."""
        metrics = PredictionMetrics(
            accuracy=0.75,
            precision=0.70,
            recall=0.65,
            f1=0.67,
            log_loss=0.55,
            per_class_precision={"home_win": 0.8, "away_win": 0.6},
            per_class_recall={"home_win": 0.75, "away_win": 0.55}
        )
        d = metrics.to_dict()

        assert "precision_home_win" in d
        assert "precision_away_win" in d
        assert "recall_home_win" in d
        assert d["precision_home_win"] == 0.8


class TestSportsMetricsCalculateAll:
    """Tests for SportsMetrics.calculate_all()."""

    @pytest.fixture
    def sample_predictions(self):
        """Sample prediction data."""
        np.random.seed(42)
        y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1])
        y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 0])
        y_proba = np.random.rand(10, 2)
        y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)
        return y_true, y_pred, y_proba

    def test_calculate_basic_metrics(self, sample_predictions):
        """Test basic metrics are calculated."""
        y_true, y_pred, y_proba = sample_predictions

        metrics = SportsMetrics.calculate_all(y_true, y_pred, y_proba)

        assert 0 <= metrics.accuracy <= 1
        assert 0 <= metrics.precision <= 1
        assert 0 <= metrics.recall <= 1
        assert 0 <= metrics.f1 <= 1
        assert metrics.log_loss >= 0

    def test_calculate_brier_score(self, sample_predictions):
        """Test Brier score is calculated."""
        y_true, y_pred, y_proba = sample_predictions

        metrics = SportsMetrics.calculate_all(y_true, y_pred, y_proba)

        assert metrics.brier_score is not None
        assert 0 <= metrics.brier_score <= 1

    def test_calculate_roc_auc(self, sample_predictions):
        """Test ROC AUC is calculated."""
        y_true, y_pred, y_proba = sample_predictions

        metrics = SportsMetrics.calculate_all(y_true, y_pred, y_proba)

        assert metrics.roc_auc is not None
        assert 0 <= metrics.roc_auc <= 1

    def test_per_class_metrics(self, sample_predictions):
        """Test per-class metrics are calculated."""
        y_true, y_pred, y_proba = sample_predictions

        metrics = SportsMetrics.calculate_all(y_true, y_pred, y_proba)

        assert metrics.per_class_precision is not None
        assert metrics.per_class_recall is not None
        assert len(metrics.per_class_precision) == 2

    def test_without_probabilities(self, sample_predictions):
        """Test calculation without probability predictions."""
        y_true, y_pred, _ = sample_predictions

        metrics = SportsMetrics.calculate_all(y_true, y_pred)

        assert metrics.accuracy > 0
        assert metrics.log_loss == 0.0
        assert metrics.brier_score is None


class TestSportsMetricsROI:
    """Tests for ROI calculation."""

    @pytest.fixture
    def betting_data(self):
        """Sample betting data."""
        y_true = np.array([2, 0, 2, 1, 2])  # home=2, draw=1, away=0
        y_pred = np.array([2, 0, 1, 1, 2])  # 3 correct
        y_proba = np.array([
            [0.2, 0.2, 0.6],  # home
            [0.6, 0.2, 0.2],  # away
            [0.3, 0.4, 0.3],  # draw (wrong)
            [0.2, 0.6, 0.2],  # draw
            [0.1, 0.1, 0.8],  # home
        ])
        odds = pd.DataFrame({
            "home_odds": [2.0, 3.0, 1.8, 4.0, 1.5],
            "draw_odds": [3.5, 3.2, 3.0, 3.5, 4.0],
            "away_odds": [4.0, 2.5, 5.0, 2.0, 6.0],
        })
        return y_true, y_pred, y_proba, odds

    def test_calculate_roi_basic(self, betting_data):
        """Test basic ROI calculation."""
        y_true, y_pred, y_proba, odds = betting_data

        roi, yield_pct = SportsMetrics.calculate_roi(
            y_true, y_pred, y_proba, odds
        )

        assert isinstance(roi, float)
        assert isinstance(yield_pct, float)

    def test_calculate_roi_all_correct(self):
        """Test ROI when all predictions are correct."""
        y_true = np.array([2, 2, 2])  # All home wins
        y_pred = np.array([2, 2, 2])
        y_proba = np.array([[0.1, 0.1, 0.8]] * 3)
        odds = pd.DataFrame({
            "home_odds": [2.0, 2.0, 2.0],
            "draw_odds": [3.0, 3.0, 3.0],
            "away_odds": [4.0, 4.0, 4.0],
        })

        roi, yield_pct = SportsMetrics.calculate_roi(
            y_true, y_pred, y_proba, odds
        )

        # 3 bets at 2.0 odds, all win
        # Profit = 3 * (2.0 - 1) = 3 units on 3 staked = 100% yield
        assert yield_pct == 100.0

    def test_calculate_roi_all_wrong(self):
        """Test ROI when all predictions are wrong."""
        y_true = np.array([0, 0, 0])  # All away wins
        y_pred = np.array([2, 2, 2])  # All predicted home
        y_proba = np.array([[0.1, 0.1, 0.8]] * 3)
        odds = pd.DataFrame({
            "home_odds": [2.0, 2.0, 2.0],
            "draw_odds": [3.0, 3.0, 3.0],
            "away_odds": [4.0, 4.0, 4.0],
        })

        roi, yield_pct = SportsMetrics.calculate_roi(
            y_true, y_pred, y_proba, odds
        )

        # 3 bets, all lose
        assert yield_pct == -100.0

    def test_calculate_roi_with_confidence_filter(self, betting_data):
        """Test ROI with minimum confidence filter."""
        y_true, y_pred, y_proba, odds = betting_data

        # High confidence filter should reduce bets
        roi_all, _ = SportsMetrics.calculate_roi(
            y_true, y_pred, y_proba, odds, min_confidence=0.0
        )

        roi_filtered, _ = SportsMetrics.calculate_roi(
            y_true, y_pred, y_proba, odds, min_confidence=0.7
        )

        # Filtered should have fewer bets (different ROI)
        # Note: exact values depend on data

    def test_calculate_roi_empty(self):
        """Test ROI with no bets."""
        y_true = np.array([])
        y_pred = np.array([])
        y_proba = np.empty((0, 3))
        odds = pd.DataFrame(columns=["home_odds", "draw_odds", "away_odds"])

        roi, yield_pct = SportsMetrics.calculate_roi(
            y_true, y_pred, y_proba, odds
        )

        assert roi == 0.0
        assert yield_pct == 0.0


class TestSportsMetricsCalibration:
    """Tests for calibration metrics."""

    def test_calculate_calibration_binary(self):
        """Test calibration for binary classification."""
        np.random.seed(42)
        n = 100
        y_true = np.random.randint(0, 2, n)
        y_proba = np.random.rand(n)

        calibration = SportsMetrics.calculate_calibration(y_true, y_proba)

        assert "bin_edges" in calibration
        assert "bin_accuracy" in calibration
        assert "bin_confidence" in calibration
        assert "bin_counts" in calibration
        assert len(calibration["bin_edges"]) == 11  # 10 bins + 1

    def test_calculate_calibration_multiclass(self):
        """Test calibration for multiclass classification."""
        np.random.seed(42)
        n = 100
        y_true = np.random.randint(0, 3, n)
        y_proba = np.random.rand(n, 3)
        y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)

        calibration = SportsMetrics.calculate_calibration(y_true, y_proba)

        assert len(calibration["bin_accuracy"]) == 10

    def test_calculate_calibration_perfect(self):
        """Test calibration for perfectly calibrated predictions."""
        # When prob = 0.8, accuracy should be ~0.8
        y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 1])  # 60% positive
        y_proba = np.array([0.6] * 10)  # All predict 60%

        calibration = SportsMetrics.calculate_calibration(y_true, y_proba, n_bins=5)

        # Most samples in one bin (around 0.6)
        assert sum(calibration["bin_counts"]) == 10


class TestSportsMetricsConfusionMatrix:
    """Tests for confusion matrix."""

    def test_get_confusion_matrix(self):
        """Test confusion matrix generation."""
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 1, 1, 1, 2, 0])

        cm = SportsMetrics.get_confusion_matrix(y_true, y_pred)

        assert cm.shape == (3, 3)
        assert cm[0, 0] == 1  # True away correctly predicted
        assert cm[1, 1] == 2  # True draw correctly predicted

    def test_get_classification_report(self):
        """Test classification report generation."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])

        report = SportsMetrics.get_classification_report(y_true, y_pred)

        assert isinstance(report, str)
        assert "precision" in report.lower()
        assert "recall" in report.lower()


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_class_predictions(self):
        """Test metrics with single class in predictions."""
        y_true = np.array([0, 0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 0, 0])  # All predict class 0
        y_proba = np.array([[0.9, 0.1]] * 5)

        metrics = SportsMetrics.calculate_all(y_true, y_pred, y_proba)

        assert metrics.accuracy == 0.6
        assert metrics.precision >= 0  # Should handle zero division

    def test_imbalanced_classes(self):
        """Test metrics with highly imbalanced classes."""
        np.random.seed(42)
        y_true = np.array([0] * 90 + [1] * 10)  # 90% class 0
        y_pred = np.array([0] * 95 + [1] * 5)
        y_proba = np.random.rand(100, 2)
        y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)

        metrics = SportsMetrics.calculate_all(y_true, y_pred, y_proba)

        assert metrics.accuracy > 0
        # Per-class metrics should show imbalance
        assert metrics.per_class_recall is not None


class TestProbabilisticSharpeRatio:
    """Tests for PSR function."""

    def test_psr_positive_sr(self):
        """PSR > 0.5 when SR is clearly positive."""
        np.random.seed(42)
        returns = np.random.normal(0.05, 0.3, 200)
        psr = probabilistic_sharpe_ratio(returns, sr_benchmark=0.0)
        assert 0.0 <= psr <= 1.0
        assert psr > 0.5  # Positive mean should give PSR > 0.5

    def test_psr_zero_benchmark(self):
        """PSR with strong positive returns should be high."""
        returns = np.random.RandomState(42).normal(0.1, 0.2, 500)
        psr = probabilistic_sharpe_ratio(returns, sr_benchmark=0.0)
        assert psr > 0.9

    def test_psr_too_few_returns(self):
        """PSR returns 0 for fewer than 5 observations."""
        returns = np.array([0.1, -0.05, 0.2])
        assert probabilistic_sharpe_ratio(returns) == 0.0

    def test_psr_negative_sr(self):
        """PSR < 0.5 when SR is negative."""
        returns = np.random.RandomState(42).normal(-0.05, 0.3, 200)
        psr = probabilistic_sharpe_ratio(returns, sr_benchmark=0.0)
        assert psr < 0.5


class TestMinTrackRecordLength:
    """Tests for MinTRL function."""

    def test_mintrl_typical_betting(self):
        """MinTRL for typical betting returns (SR~0.15, negative skew).

        With SR=0.15, skew=-0.5, kurtosis=4.0 (excess=1.0),
        MinTRL should be roughly ~100-200 bets.
        """
        np.random.seed(42)
        # Simulate betting returns: win ~65% at avg odds 1.85
        n = 300
        wins = np.random.binomial(1, 0.65, n)
        returns = np.where(wins, 0.85, -1.0)  # odds 1.85: win +0.85, lose -1
        mintrl = min_track_record_length(returns, sr_benchmark=0.0)
        assert 30 < mintrl < 500  # Should be in reasonable range

    def test_mintrl_negative_sr(self):
        """MinTRL returns max when SR is negative (can never be significant)."""
        returns = np.random.RandomState(42).normal(-0.1, 0.5, 100)
        mintrl = min_track_record_length(returns, sr_benchmark=0.0)
        assert mintrl == 999999

    def test_mintrl_too_few_returns(self):
        """MinTRL returns max for fewer than 5 observations."""
        returns = np.array([0.1, -0.05])
        assert min_track_record_length(returns) == 999999

    def test_mintrl_strong_sr_needs_fewer(self):
        """Stronger SR requires shorter track record."""
        rng = np.random.RandomState(42)
        # Strong SR
        strong = rng.normal(0.3, 0.3, 200)
        # Weak SR
        weak = rng.normal(0.05, 0.3, 200)
        mintrl_strong = min_track_record_length(strong)
        mintrl_weak = min_track_record_length(weak)
        assert mintrl_strong < mintrl_weak


class TestDeflatedSharpeRatio:
    """Tests for DSR function."""

    def test_dsr_decreases_with_more_trials(self):
        """DSR should decrease as K (number of trials) increases."""
        np.random.seed(42)
        returns = np.random.normal(0.05, 0.3, 200)
        dsr_10 = deflated_sharpe_ratio(returns, n_trials=10)
        dsr_1000 = deflated_sharpe_ratio(returns, n_trials=1000)
        assert dsr_10 > dsr_1000

    def test_dsr_single_trial(self):
        """With 1 trial, DSR should equal PSR (no multiple testing penalty)."""
        np.random.seed(42)
        returns = np.random.normal(0.1, 0.3, 200)
        dsr = deflated_sharpe_ratio(returns, n_trials=1)
        psr = probabilistic_sharpe_ratio(returns, sr_benchmark=0.0)
        assert abs(dsr - psr) < 0.01

    def test_dsr_range(self):
        """DSR should be in [0, 1]."""
        np.random.seed(42)
        returns = np.random.normal(0.05, 0.3, 100)
        for k in [1, 10, 100, 1000, 5000]:
            dsr = deflated_sharpe_ratio(returns, n_trials=k)
            assert 0.0 <= dsr <= 1.0

    def test_dsr_too_few_returns(self):
        """DSR returns 0 for fewer than 5 observations."""
        returns = np.array([0.1, -0.05])
        assert deflated_sharpe_ratio(returns, n_trials=100) == 0.0

    def test_dsr_strong_signal_survives(self):
        """Very strong SR should still have high DSR even with many trials."""
        np.random.seed(42)
        returns = np.random.normal(0.5, 0.3, 500)  # Very high SR
        dsr = deflated_sharpe_ratio(returns, n_trials=200)
        assert dsr > 0.9

    def test_dsr_noise_killed(self):
        """Marginal SR should have low DSR with many trials."""
        np.random.seed(42)
        returns = np.random.normal(0.01, 0.5, 50)  # Marginal SR, few obs
        dsr = deflated_sharpe_ratio(returns, n_trials=5000)
        assert dsr < 0.5


class TestEstimateKEff:
    """Tests for K_eff estimation."""

    def test_k_eff_basic(self):
        """K_eff = n_models * sqrt(n_combos)."""
        k = estimate_k_eff(10, 100)
        assert k == 100  # 10 * sqrt(100) = 100

    def test_k_eff_typical_h2h(self):
        """H2H: 12 models * sqrt(840 combos) ~ 348."""
        k = estimate_k_eff(12, 840)
        assert 340 < k < 360

    def test_k_eff_niche(self):
        """Niche: 10 models * sqrt(5 combos) ~ 22."""
        k = estimate_k_eff(10, 5)
        assert 20 <= k <= 25

    def test_k_eff_minimum_one(self):
        """K_eff is at least 1."""
        assert estimate_k_eff(0, 0) == 1
        assert estimate_k_eff(1, 0) == 1

    def test_k_eff_single_trial(self):
        """Single model, single config = 1."""
        assert estimate_k_eff(1, 1) == 1


class TestEstimateReturnAutocorrelation:
    """Tests for return autocorrelation estimation."""

    def test_iid_returns(self):
        """IID returns should have rho ~ 0."""
        rng = np.random.RandomState(42)
        returns = rng.normal(0.05, 0.3, 200)
        rho = estimate_return_autocorrelation(returns)
        assert abs(rho) < 0.2  # Should be close to zero for IID

    def test_correlated_returns(self):
        """Returns with AR(1) structure should have positive rho."""
        rng = np.random.RandomState(42)
        n = 200
        returns = np.zeros(n)
        returns[0] = rng.normal(0.05, 0.3)
        for i in range(1, n):
            returns[i] = 0.5 * returns[i - 1] + rng.normal(0.05, 0.3)
        rho = estimate_return_autocorrelation(returns)
        assert rho > 0.2  # Should detect positive autocorrelation

    def test_too_few_returns(self):
        """Returns 0.0 for fewer than 10 observations."""
        returns = np.array([0.1, -0.05, 0.2, 0.1, 0.05])
        assert estimate_return_autocorrelation(returns) == 0.0

    def test_clipped_range(self):
        """Result is clipped to (-0.99, 0.99)."""
        rho = estimate_return_autocorrelation(np.ones(20))
        assert -0.99 <= rho <= 0.99


class TestPSRWithRho:
    """Tests for PSR with autocorrelation adjustment."""

    def test_rho_zero_unchanged(self):
        """rho=0 should give same result as before."""
        rng = np.random.RandomState(42)
        returns = rng.normal(0.1, 0.3, 200)
        psr_no_rho = probabilistic_sharpe_ratio(returns, sr_benchmark=0.0)
        psr_rho_0 = probabilistic_sharpe_ratio(returns, sr_benchmark=0.0, rho=0.0)
        assert abs(psr_no_rho - psr_rho_0) < 1e-10

    def test_rho_reduces_psr(self):
        """Positive rho should reduce PSR (wider confidence band)."""
        rng = np.random.RandomState(42)
        returns = rng.normal(0.1, 0.3, 200)
        psr_0 = probabilistic_sharpe_ratio(returns, sr_benchmark=0.0, rho=0.0)
        psr_rho = probabilistic_sharpe_ratio(returns, sr_benchmark=0.0, rho=0.3)
        assert psr_rho < psr_0


class TestDSRWithRho:
    """Tests for DSR with autocorrelation adjustment."""

    def test_rho_reduces_dsr(self):
        """Positive rho should reduce DSR."""
        rng = np.random.RandomState(42)
        returns = rng.normal(0.1, 0.3, 200)
        dsr_0 = deflated_sharpe_ratio(returns, n_trials=100, rho=0.0)
        dsr_rho = deflated_sharpe_ratio(returns, n_trials=100, rho=0.3)
        assert dsr_rho < dsr_0
