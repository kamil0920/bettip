"""Saerens-Latinne-Decaestecker (2002) prior shift correction.

Adjusts a classifier's posterior probabilities to a new class prior
using Bayes' theorem, without retraining. When the target prior is
unknown, an EM algorithm estimates it from unlabeled predictions.

Reference:
    Saerens, M., Latinne, P., & Decaestecker, C. (2002).
    Adjusting the Outputs of a Classifier to New a Priori Probabilities:
    A Simple Procedure. Neural Computation, 14(1), 21-41.
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class SLDPriorAdjuster:
    """EM-based prior shift correction for binary classifiers."""

    def adjust(
        self,
        probs: np.ndarray,
        pi_train: float,
        pi_new: Optional[float] = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        max_shift: float = 0.15,
    ) -> np.ndarray:
        """Adjust posteriors from training prior to new prior.

        Args:
            probs: Calibrated posterior probabilities P(y=1|x) under training prior.
            pi_train: Training set class prior P(y=1) used during calibration.
            pi_new: Target prior. If None, estimated via EM from unlabeled predictions.
            max_iter: Maximum EM iterations (only used when pi_new is None).
            tol: Convergence tolerance for EM.
            max_shift: Maximum allowed shift from pi_train (prevents EM divergence).

        Returns:
            Adjusted posteriors under the new (or estimated) prior.
        """
        probs = np.asarray(probs, dtype=np.float64)
        probs = np.clip(probs, 1e-8, 1 - 1e-8)

        if pi_train <= 0 or pi_train >= 1:
            logger.warning(f"SLD: invalid pi_train={pi_train:.4f}, returning original probs")
            return probs

        if pi_new is not None:
            # Direct adjustment with known target prior
            pi_new = np.clip(pi_new, 1e-8, 1 - 1e-8)
            adjusted = self._bayes_adjust(probs, pi_train, pi_new)
            logger.info(
                f"SLD direct: pi_train={pi_train:.4f} → pi_new={pi_new:.4f}, "
                f"mean_prob {probs.mean():.4f} → {adjusted.mean():.4f}"
            )
            return adjusted

        # EM: estimate pi_new from unlabeled predictions
        pi_lo = max(1e-4, pi_train - max_shift)
        pi_hi = min(1 - 1e-4, pi_train + max_shift)
        pi_est = float(np.clip(probs.mean(), pi_lo, pi_hi))

        for i in range(max_iter):
            adjusted = self._bayes_adjust(probs, pi_train, pi_est)
            pi_new_est = float(np.clip(adjusted.mean(), pi_lo, pi_hi))

            if abs(pi_new_est - pi_est) < tol:
                logger.info(
                    f"SLD EM converged in {i + 1} iterations: "
                    f"pi_train={pi_train:.4f} → pi_est={pi_new_est:.4f}, "
                    f"mean_prob {probs.mean():.4f} → {adjusted.mean():.4f}"
                )
                return adjusted

            pi_est = pi_new_est

        logger.warning(f"SLD EM did not converge after {max_iter} iterations (pi_est={pi_est:.4f})")
        return self._bayes_adjust(probs, pi_train, pi_est)

    @staticmethod
    def _bayes_adjust(probs: np.ndarray, pi_old: float, pi_new: float) -> np.ndarray:
        """Apply Bayes' theorem to shift prior from pi_old to pi_new."""
        ratio_pos = pi_new / pi_old
        ratio_neg = (1 - pi_new) / (1 - pi_old)
        numerator = probs * ratio_pos
        denominator = numerator + (1 - probs) * ratio_neg
        return np.clip(numerator / denominator, 1e-8, 1 - 1e-8)
