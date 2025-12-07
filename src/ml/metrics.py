"""
Custom metrics for sports prediction evaluation.

Standard ML metrics + sports-specific metrics:
- ROI (Return on Investment) simulation
- Calibration metrics
- Per-class performance
"""
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    brier_score_loss,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


@dataclass
class PredictionMetrics:
    """Container for all prediction metrics."""
    accuracy: float
    precision: float
    recall: float
    f1: float
    log_loss: float

    brier_score: Optional[float] = None
    roc_auc: Optional[float] = None

    per_class_precision: Optional[Dict[str, float]] = None
    per_class_recall: Optional[Dict[str, float]] = None

    roi: Optional[float] = None
    yield_pct: Optional[float] = None

    def to_dict(self) -> Dict[str, float]:
        """Convert to flat dictionary for MLflow logging."""
        result = {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "log_loss": self.log_loss,
        }
        if self.brier_score is not None:
            result["brier_score"] = self.brier_score
        if self.roc_auc is not None:
            result["roc_auc"] = self.roc_auc
        if self.roi is not None:
            result["roi"] = self.roi
        if self.yield_pct is not None:
            result["yield_pct"] = self.yield_pct

        # Add per-class metrics
        if self.per_class_precision:
            for cls, val in self.per_class_precision.items():
                result[f"precision_{cls}"] = val
        if self.per_class_recall:
            for cls, val in self.per_class_recall.items():
                result[f"recall_{cls}"] = val

        return result


class SportsMetrics:
    """Calculator for sports prediction metrics."""

    CLASS_LABELS = {0: "away_win", 1: "draw", 2: "home_win"}

    @classmethod
    def calculate_all(
        cls,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        odds: Optional[pd.DataFrame] = None,
    ) -> PredictionMetrics:
        """
        Calculate all metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)
            odds: DataFrame with betting odds (optional, for ROI calculation)

        Returns:
            PredictionMetrics with all calculated metrics
        """
        metrics = PredictionMetrics(
            accuracy=accuracy_score(y_true, y_pred),
            precision=precision_score(y_true, y_pred, average="weighted", zero_division=0),
            recall=recall_score(y_true, y_pred, average="weighted", zero_division=0),
            f1=f1_score(y_true, y_pred, average="weighted", zero_division=0),
            log_loss=log_loss(y_true, y_proba) if y_proba is not None else 0.0,
        )

        if y_proba is not None:
            try:
                if y_proba.shape[1] == 2:
                    metrics.brier_score = brier_score_loss(y_true, y_proba[:, 1])
                else:
                    brier_scores = []
                    for i in range(y_proba.shape[1]):
                        binary_true = (y_true == i).astype(int)
                        brier_scores.append(brier_score_loss(binary_true, y_proba[:, i]))
                    metrics.brier_score = np.mean(brier_scores)

                if len(np.unique(y_true)) == 2:
                    metrics.roc_auc = roc_auc_score(y_true, y_proba[:, 1])
                else:
                    metrics.roc_auc = roc_auc_score(
                        y_true, y_proba, multi_class="ovr", average="weighted"
                    )
            except Exception as e:
                logger.warning(f"Could not calculate probability metrics: {e}")

        try:
            unique_classes = np.unique(y_true)
            precision_per_class = precision_score(
                y_true, y_pred, average=None, zero_division=0
            )
            recall_per_class = recall_score(
                y_true, y_pred, average=None, zero_division=0
            )

            metrics.per_class_precision = {
                cls.CLASS_LABELS.get(c, f"class_{c}"): float(precision_per_class[i])
                for i, c in enumerate(unique_classes)
            }
            metrics.per_class_recall = {
                cls.CLASS_LABELS.get(c, f"class_{c}"): float(recall_per_class[i])
                for i, c in enumerate(unique_classes)
            }
        except Exception as e:
            logger.warning(f"Could not calculate per-class metrics: {e}")

        if odds is not None:
            try:
                roi, yield_pct = cls.calculate_roi(y_true, y_pred, y_proba, odds)
                metrics.roi = roi
                metrics.yield_pct = yield_pct
            except Exception as e:
                logger.warning(f"Could not calculate ROI: {e}")

        return metrics

    @classmethod
    def calculate_roi(
        cls,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray],
        odds: pd.DataFrame,
        stake: float = 1.0,
        min_confidence: float = 0.0,
    ) -> Tuple[float, float]:
        """
        Calculate Return on Investment for betting simulation.

        Args:
            y_true: True outcomes
            y_pred: Predicted outcomes
            y_proba: Predicted probabilities
            odds: DataFrame with columns ['home_odds', 'draw_odds', 'away_odds']
            stake: Stake per bet
            min_confidence: Minimum probability to place bet

        Returns:
            Tuple of (total_roi, yield_percentage)
        """
        total_stake = 0.0
        total_return = 0.0

        odds_columns = {
            2: "home_odds",
            1: "draw_odds",
            0: "away_odds",
        }

        for i, (true, pred) in enumerate(zip(y_true, y_pred)):
            if y_proba is not None and min_confidence > 0:
                if y_proba[i, pred] < min_confidence:
                    continue

            total_stake += stake

            if true == pred:
                odds_col = odds_columns.get(pred)
                if odds_col and odds_col in odds.columns:
                    bet_odds = odds.iloc[i][odds_col]
                    total_return += stake * bet_odds

        if total_stake == 0:
            return 0.0, 0.0

        roi = total_return - total_stake
        yield_pct = (roi / total_stake) * 100

        return roi, yield_pct

    @staticmethod
    def get_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: Optional[list] = None
    ) -> np.ndarray:
        """Get confusion matrix."""
        return confusion_matrix(y_true, y_pred, labels=labels)

    @staticmethod
    def get_classification_report(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        target_names: Optional[list] = None
    ) -> str:
        """Get detailed classification report as string."""
        return classification_report(
            y_true, y_pred,
            target_names=target_names or ["away_win", "draw", "home_win"],
            zero_division=0
        )

    @staticmethod
    def calculate_calibration(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        n_bins: int = 10
    ) -> Dict[str, np.ndarray]:
        """
        Calculate calibration metrics.

        Returns dict with:
        - bin_edges: Probability bin edges
        - bin_accuracy: Actual accuracy in each bin
        - bin_confidence: Mean predicted probability in each bin
        - bin_counts: Number of samples in each bin
        """
        bins = np.linspace(0, 1, n_bins + 1)
        bin_accuracy = []
        bin_confidence = []
        bin_counts = []

        if len(y_proba.shape) > 1:
            proba_pred = y_proba.max(axis=1)
            y_pred = y_proba.argmax(axis=1)
            correct = (y_true == y_pred).astype(float)
        else:
            proba_pred = y_proba
            correct = y_true

        for i in range(n_bins):
            mask = (proba_pred >= bins[i]) & (proba_pred < bins[i + 1])
            if mask.sum() > 0:
                bin_accuracy.append(correct[mask].mean())
                bin_confidence.append(proba_pred[mask].mean())
                bin_counts.append(mask.sum())
            else:
                bin_accuracy.append(np.nan)
                bin_confidence.append(np.nan)
                bin_counts.append(0)

        return {
            "bin_edges": bins,
            "bin_accuracy": np.array(bin_accuracy),
            "bin_confidence": np.array(bin_confidence),
            "bin_counts": np.array(bin_counts),
        }
