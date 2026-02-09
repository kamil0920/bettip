"""Post-hoc calibration validation with ECE-based quality checks.

After calibration, validates on held-out data by checking Expected Calibration Error (ECE)
per bucket. If ECE > threshold, logs a warning and tries alternative calibration methods.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.calibration.calibration import (
    calibration_metrics,
    get_calibrator,
)

logger = logging.getLogger(__name__)


def validate_calibration(
    y_true: np.ndarray,
    y_prob_calibrated: np.ndarray,
    method_used: str = "sigmoid",
    ece_threshold: float = 0.10,
    n_bins: int = 10,
) -> Dict:
    """Validate calibration quality on held-out predictions.

    Args:
        y_true: True binary labels.
        y_prob_calibrated: Calibrated predicted probabilities.
        method_used: Name of calibration method that was used.
        ece_threshold: ECE threshold above which calibration is considered poor.
        n_bins: Number of bins for ECE calculation.

    Returns:
        Dict with validation results including ECE, MCE, Brier, and pass/fail status.
    """
    metrics = calibration_metrics(y_true, y_prob_calibrated, n_bins=n_bins)

    result = {
        "method": method_used,
        "ece": float(metrics["ece"]),
        "mce": float(metrics["mce"]),
        "brier": float(metrics["brier"]),
        "passed": metrics["ece"] <= ece_threshold,
        "threshold": ece_threshold,
    }

    if not result["passed"]:
        logger.warning(
            f"Calibration validation FAILED: ECE={metrics['ece']:.4f} > {ece_threshold:.2f} "
            f"(method={method_used}, MCE={metrics['mce']:.4f}, Brier={metrics['brier']:.4f})"
        )
    else:
        logger.info(
            f"Calibration validation passed: ECE={metrics['ece']:.4f} <= {ece_threshold:.2f} "
            f"(method={method_used})"
        )

    return result


def find_best_calibration(
    y_true_train: np.ndarray,
    y_prob_train: np.ndarray,
    y_true_val: np.ndarray,
    y_prob_val: np.ndarray,
    methods: Optional[List[str]] = None,
    n_bins: int = 10,
) -> Tuple[str, Dict]:
    """Compare calibration methods and return the best one by ECE.

    Args:
        y_true_train: Training labels for fitting calibrators.
        y_prob_train: Training probabilities for fitting calibrators.
        y_true_val: Validation labels for evaluation.
        y_prob_val: Validation probabilities for evaluation.
        methods: List of methods to compare. Defaults to all available.
        n_bins: Number of bins for ECE calculation.

    Returns:
        Tuple of (best_method_name, comparison_results_dict).
    """
    if methods is None:
        methods = ["sigmoid", "isotonic", "beta", "temperature"]

    results = {}
    best_method = methods[0]
    best_ece = float("inf")

    for method in methods:
        try:
            calibrator = get_calibrator(method)
            calibrator.fit(y_prob_train, y_true_train)
            calibrated_val = calibrator.transform(y_prob_val)

            metrics = calibration_metrics(y_true_val, calibrated_val, n_bins=n_bins)
            results[method] = {
                "ece": float(metrics["ece"]),
                "mce": float(metrics["mce"]),
                "brier": float(metrics["brier"]),
            }

            if metrics["ece"] < best_ece:
                best_ece = metrics["ece"]
                best_method = method

        except Exception as e:
            logger.warning(f"Calibration method {method} failed: {e}")
            results[method] = {"ece": float("inf"), "error": str(e)}

    logger.info(f"Best calibration method: {best_method} (ECE={best_ece:.4f})")
    for method, res in results.items():
        marker = " <-- best" if method == best_method else ""
        logger.info(f"  {method}: ECE={res.get('ece', 'N/A'):.4f}{marker}")

    return best_method, results
