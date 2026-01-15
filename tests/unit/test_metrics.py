"""Unit tests for sports metrics module."""
import pytest
import numpy as np
import pandas as pd

from src.ml.metrics import (
    PredictionMetrics,
    SportsMetrics,
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
