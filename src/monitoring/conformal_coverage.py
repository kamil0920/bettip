"""Production coverage monitoring for conformal predictions.

Provides functions to check whether conformal prediction intervals
achieve their stated coverage on settled bets. Pure numpy/scipy —
no side effects, no I/O.

Key function: ``width_accuracy_correlation`` determines whether
Venn-Abers width carries a real uncertainty signal. If it does not,
Phases 2-5 of the conformal improvement plan are wasted effort.
"""

from typing import Dict

import numpy as np


def one_sided_coverage(
    probabilities: np.ndarray,
    conformal_taus: np.ndarray,
    actuals: np.ndarray,
    alpha: float = 0.10,
) -> Dict[str, float]:
    """Check one-sided conformal coverage on settled bets.

    For each bet the nonconformity score is ``pred - actual``
    (positive means overconfident). A bet is *covered* when its
    score does not exceed tau.

    Parameters
    ----------
    probabilities : np.ndarray
        Model predicted probabilities, shape ``(n,)``.
    conformal_taus : np.ndarray
        Conformal thresholds (quantile of calibration scores),
        shape ``(n,)``.
    actuals : np.ndarray
        Binary outcomes (1.0 = win, 0.0 = loss), shape ``(n,)``.
    alpha : float
        Nominal miscoverage rate. Default 0.10 → 90 % coverage.

    Returns
    -------
    Dict[str, float]
        ``empirical_coverage``, ``nominal_coverage``, ``coverage_gap``,
        ``n_bets``, ``alert`` (True when gap < -0.05).
    """
    scores = probabilities - actuals
    covered = (scores <= conformal_taus).astype(float)
    empirical = float(np.mean(covered))
    gap = empirical - (1 - alpha)
    return {
        "empirical_coverage": round(empirical, 4),
        "nominal_coverage": 1 - alpha,
        "coverage_gap": round(gap, 4),
        "n_bets": len(actuals),
        "alert": gap < -0.05,
    }


def va_consistency(
    va_lowers: np.ndarray,
    va_uppers: np.ndarray,
    actuals: np.ndarray,
) -> Dict[str, float]:
    """Check Venn-Abers interval consistency on settled bets.

    For binary outcomes a VA interval is *consistent* when the
    outcome is plausible given the interval:

    - ``actual == 1`` requires ``va_upper >= 0.5``
    - ``actual == 0`` requires ``va_lower <= 0.5``

    Parameters
    ----------
    va_lowers : np.ndarray
        Lower bounds of VA intervals, shape ``(n,)``.
    va_uppers : np.ndarray
        Upper bounds of VA intervals, shape ``(n,)``.
    actuals : np.ndarray
        Binary outcomes (1.0 = win, 0.0 = loss), shape ``(n,)``.

    Returns
    -------
    Dict[str, float]
        ``consistency_rate``, ``mean_width``, ``n_bets``.
    """
    consistent = np.where(
        actuals == 1,
        va_uppers >= 0.5,
        va_lowers <= 0.5,
    ).astype(float)
    return {
        "consistency_rate": round(float(np.mean(consistent)), 4),
        "mean_width": round(float(np.mean(va_uppers - va_lowers)), 4),
        "n_bets": len(actuals),
    }


def rolling_coverage(
    scores: np.ndarray,
    taus: np.ndarray,
    window: int = 50,
) -> np.ndarray:
    """Compute rolling conformal coverage rate.

    Returns an array of the same length as the input.  The first
    ``window - 1`` entries are ``NaN`` (insufficient history).

    Parameters
    ----------
    scores : np.ndarray
        Nonconformity scores (``pred - actual``), shape ``(n,)``.
    taus : np.ndarray
        Conformal thresholds, shape ``(n,)``.
    window : int
        Rolling window size. Default 50.

    Returns
    -------
    np.ndarray
        Rolling coverage values, shape ``(n,)``.
    """
    covered = (scores <= taus).astype(float)
    n = len(covered)
    result = np.full(n, np.nan)
    for i in range(window - 1, n):
        result[i] = np.mean(covered[i - window + 1 : i + 1])
    return result


def width_accuracy_correlation(
    va_widths: np.ndarray,
    actuals: np.ndarray,
    probabilities: np.ndarray,
) -> Dict[str, float]:
    """Key diagnostic: does VA width predict prediction accuracy?

    Computes the Spearman rank correlation between VA interval width
    and absolute prediction error ``|actual - predicted|``.

    A **positive**, statistically significant correlation means wider
    intervals correctly flag harder-to-predict bets → the uncertainty
    signal is informative and Phases 2-5 are worth pursuing.

    Parameters
    ----------
    va_widths : np.ndarray
        VA interval widths (``va_upper - va_lower``), shape ``(n,)``.
    actuals : np.ndarray
        Binary outcomes (1.0 = win, 0.0 = loss), shape ``(n,)``.
    probabilities : np.ndarray
        Model predicted probabilities, shape ``(n,)``.

    Returns
    -------
    Dict[str, float]
        ``spearman_rho``, ``p_value``, ``informative`` (bool),
        ``n_bets``.
    """
    from scipy.stats import spearmanr

    abs_errors = np.abs(actuals - probabilities)
    rho, pval = spearmanr(va_widths, abs_errors)
    return {
        "spearman_rho": round(float(rho), 4),
        "p_value": round(float(pval), 4),
        "informative": bool(pval < 0.05 and rho > 0),
        "n_bets": len(actuals),
    }
