"""
Uncertainty Quantification via Conformal Prediction (MAPIE)

Wraps trained classifiers with MAPIE to produce prediction sets
and calibrated uncertainty estimates. Uses prediction set size
to adjust Kelly staking — larger prediction sets = less certainty = smaller stakes.
"""

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ConformalClassifier:
    """Wraps a trained classifier with MAPIE conformal prediction.

    Provides:
    - Prediction sets at configurable confidence level
    - Uncertainty score (prediction set size as fraction of classes)
    - Uncertainty-adjusted Kelly stakes
    """

    def __init__(self, model: Any, alpha: float = 0.1):
        """
        Args:
            model: Fitted sklearn-compatible classifier with predict_proba.
            alpha: Significance level (default 0.1 = 90% confidence).
        """
        self.model = model
        self.alpha = alpha
        self.mapie_clf = None

    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray) -> "ConformalClassifier":
        """Calibrate conformal predictor on a held-out calibration set.

        Args:
            X_cal: Calibration features.
            y_cal: Calibration targets.

        Returns:
            self
        """
        from mapie.classification import SplitConformalClassifier

        self.mapie_clf = SplitConformalClassifier(
            estimator=self.model,
            prefit=True,
            conformity_score='lac',  # Least Ambiguous set-valued Classifier
            confidence_level=1 - self.alpha,
        )
        self.mapie_clf.conformalize(X_cal, y_cal)
        logger.info(f"Conformal predictor calibrated on {len(X_cal)} samples (alpha={self.alpha})")
        return self

    def predict_with_uncertainty(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate predictions with uncertainty estimates.

        Args:
            X: Feature matrix.

        Returns:
            Tuple of (predictions, prediction_sets, uncertainty_scores)
            - predictions: shape (n,) class predictions
            - prediction_sets: shape (n, n_classes) boolean mask of included classes
            - uncertainty_scores: shape (n,) in [0, 1], higher = more uncertain
        """
        if self.mapie_clf is None:
            raise RuntimeError("Call calibrate() first")

        y_pred, prediction_sets = self.mapie_clf.predict_set(X)

        # prediction_sets shape: (n_samples, n_classes, 1) — squeeze last dim
        if prediction_sets.ndim == 3:
            prediction_sets = prediction_sets[:, :, 0]

        # Uncertainty = fraction of classes in prediction set
        # For binary: 0 classes = impossible, 1 class = certain, 2 classes = uncertain
        n_classes = prediction_sets.shape[1]
        set_sizes = prediction_sets.sum(axis=1)
        uncertainty = (set_sizes - 1) / max(n_classes - 1, 1)
        uncertainty = np.clip(uncertainty, 0, 1)

        logger.info(f"Uncertainty: mean={uncertainty.mean():.3f}, "
                    f"certain={np.sum(uncertainty == 0)}/{len(uncertainty)}")

        return y_pred, prediction_sets, uncertainty

    def to_dict(self) -> Dict[str, Any]:
        """Serialize conformal state for persistence (model stored separately)."""
        return {
            "alpha": self.alpha,
            "mapie_clf": self.mapie_clf,  # MAPIE internal state is picklable
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], model: Any) -> "ConformalClassifier":
        """Reconstruct from serialized state."""
        cc = cls(model=model, alpha=data["alpha"])
        cc.mapie_clf = data["mapie_clf"]
        return cc

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Passthrough to underlying model's predict_proba."""
        return self.model.predict_proba(X)


def adjust_kelly_stake(
    base_stake: float,
    uncertainty: float,
    uncertainty_penalty: float = 1.0,
) -> float:
    """Reduce Kelly stake based on prediction uncertainty.

    stake = base_stake / (1 + uncertainty_penalty * uncertainty)

    Args:
        base_stake: Original Kelly-calculated stake.
        uncertainty: Uncertainty score in [0, 1].
        uncertainty_penalty: How aggressively to penalize uncertainty.

    Returns:
        Adjusted stake (always <= base_stake).
    """
    return base_stake / (1 + uncertainty_penalty * uncertainty)


def batch_adjust_stakes(
    stakes: np.ndarray,
    uncertainties: np.ndarray,
    uncertainty_penalty: float = 1.0,
) -> np.ndarray:
    """Vectorized stake adjustment for a batch of bets.

    Args:
        stakes: Array of base stakes.
        uncertainties: Array of uncertainty scores.
        uncertainty_penalty: Penalty factor.

    Returns:
        Array of adjusted stakes.
    """
    return stakes / (1 + uncertainty_penalty * uncertainties)
