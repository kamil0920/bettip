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


def sharpe_ratio(returns_per_bet: np.ndarray) -> float:
    """
    Calculate Sharpe ratio of per-bet returns.

    Args:
        returns_per_bet: Array of returns per bet (e.g., odds-1 for win, -1 for loss).

    Returns:
        Sharpe ratio (mean / std). Returns 0.0 if std is 0 or fewer than 2 bets.
    """
    returns = np.asarray(returns_per_bet, dtype=float)
    if len(returns) < 2:
        return 0.0
    std = returns.std(ddof=1)
    if std == 0:
        return 0.0
    return float(returns.mean() / std)


def sortino_ratio(returns_per_bet: np.ndarray) -> float:
    """
    Calculate Sortino ratio of per-bet returns.

    Uses downside deviation (std of negative returns only) instead of full std.

    Args:
        returns_per_bet: Array of returns per bet.

    Returns:
        Sortino ratio (mean / downside_std). Returns 0.0 if downside_std is 0.
    """
    returns = np.asarray(returns_per_bet, dtype=float)
    if len(returns) < 2:
        return 0.0
    downside = returns[returns < 0]
    if len(downside) == 0:
        return float('inf') if returns.mean() > 0 else 0.0
    downside_std = downside.std(ddof=1)
    if downside_std == 0:
        return 0.0
    return float(returns.mean() / downside_std)


def probabilistic_sharpe_ratio(
    returns_per_bet: np.ndarray,
    sr_benchmark: float = 0.0,
    rho: float = 0.0,
) -> float:
    """Probabilistic Sharpe Ratio (PSR) — Bailey & Lopez de Prado [2012].

    Estimates the probability that the true Sharpe ratio exceeds sr_benchmark,
    accounting for skewness and kurtosis of returns.

    PSR[SR*] = Z[ ((SR_hat - SR*) * sqrt(T-1)) /
                   sqrt(1 - gamma3*SR_hat + (gamma4-1)/4 * SR_hat^2) ]

    When rho > 0, applies Newey-West variance adjustment to account for
    serial correlation in returns (Lo 2002).

    Args:
        returns_per_bet: Array of per-bet returns.
        sr_benchmark: Benchmark Sharpe ratio to test against (default 0).
        rho: Lag-1 autocorrelation of returns (default 0 = IID assumption).

    Returns:
        Probability (0-1) that true SR > sr_benchmark.
    """
    from scipy.stats import norm

    returns = np.asarray(returns_per_bet, dtype=float)
    T = len(returns)
    if T < 5:
        return 0.0

    sr_hat = sharpe_ratio(returns)
    gamma3 = float(pd.Series(returns).skew())
    gamma4 = float(pd.Series(returns).kurtosis() + 3)  # excess -> raw

    denom_sq = 1.0 - gamma3 * sr_hat + (gamma4 - 1) / 4.0 * sr_hat**2
    if denom_sq <= 0:
        return 0.0

    if rho > 0:
        denom_sq *= (1 + 2 * rho / (1 - rho))

    z = (sr_hat - sr_benchmark) * np.sqrt(T - 1) / np.sqrt(denom_sq)
    return float(norm.cdf(z))


def min_track_record_length(
    returns_per_bet: np.ndarray,
    sr_benchmark: float = 0.0,
    alpha: float = 0.05,
    rho: float = 0.0,
) -> int:
    """Minimum Track Record Length (MinTRL) — Lopez de Prado.

    Minimum number of observations needed for observed SR to be
    statistically significant at confidence level (1 - alpha).

    MinTRL = 1 + [1 - gamma3*SR_hat + (gamma4-1)/4 * SR_hat^2]
                 * (Z_alpha / (SR_hat - SR*))^2

    When rho > 0, applies Newey-West variance adjustment to account for
    serial correlation in returns (Lo 2002).

    Args:
        returns_per_bet: Array of per-bet returns.
        sr_benchmark: Benchmark SR (default 0 = break-even).
        alpha: Significance level (default 0.05 = 95% confidence).
        rho: Lag-1 autocorrelation of returns (default 0 = IID assumption).

    Returns:
        MinTRL as integer. Returns 999999 if SR <= benchmark (infinite record needed).
    """
    from scipy.stats import norm

    returns = np.asarray(returns_per_bet, dtype=float)
    if len(returns) < 5:
        return 999999

    sr_hat = sharpe_ratio(returns)
    if sr_hat <= sr_benchmark:
        return 999999

    gamma3 = float(pd.Series(returns).skew())
    gamma4 = float(pd.Series(returns).kurtosis() + 3)  # excess -> raw

    z_alpha = norm.ppf(1 - alpha)
    variance_factor = 1.0 - gamma3 * sr_hat + (gamma4 - 1) / 4.0 * sr_hat**2
    if variance_factor <= 0:
        return 999999

    if rho > 0:
        variance_factor *= (1 + 2 * rho / (1 - rho))

    mintrl = 1 + variance_factor * (z_alpha / (sr_hat - sr_benchmark)) ** 2
    return int(np.ceil(mintrl))


def deflated_sharpe_ratio(
    returns_per_bet: np.ndarray,
    n_trials: int,
    sr_benchmark: float = 0.0,
    rho: float = 0.0,
) -> float:
    """Deflated Sharpe Ratio (DSR) — Lopez de Prado [2014].

    Corrects the Sharpe ratio for selection bias under multiple testing,
    non-Normal returns, and finite sample length. Uses the False Strategy
    theorem to compute the expected maximum SR under the null.

    DSR = PSR(SR_0), where SR_0 = expected max SR from K independent trials
    under the null hypothesis that all strategies have SR = 0.

    When rho > 0, applies Newey-West variance adjustment to account for
    serial correlation in returns (Lo 2002).

    Args:
        returns_per_bet: Array of per-bet returns from the selected strategy.
        n_trials: Number of independent configurations/trials tested (K).
        sr_benchmark: Base benchmark before inflation (default 0).
        rho: Lag-1 autocorrelation of returns (default 0 = IID assumption).

    Returns:
        DSR as probability (0-1) that the observed SR is genuine after
        controlling for multiple testing.
    """
    from scipy.stats import norm

    returns = np.asarray(returns_per_bet, dtype=float)
    T = len(returns)
    if T < 5 or n_trials < 1:
        return 0.0

    sr_hat = sharpe_ratio(returns)

    # False Strategy theorem: expected max SR under null for K trials
    # E[max{SR_k}] ≈ (1 - gamma_em) * Z^{-1}[1 - 1/K]
    #                + gamma_em * Z^{-1}[1 - 1/(K*e)]
    # where gamma_em = Euler-Mascheroni constant ≈ 0.5772
    EULER_MASCHERONI = 0.5772156649
    e = np.e

    if n_trials <= 1:
        sr_0 = sr_benchmark
    else:
        # Variance of SR estimates under null: Var[SR] ≈ 1/T (IID Normal)
        # For non-Normal, we use the full Mertens formula but SR=0 under null
        # simplifies to Var[SR_hat] = 1/T, so sqrt(V) = 1/sqrt(T)
        sr_std = 1.0 / np.sqrt(T)
        q1 = norm.ppf(1 - 1.0 / n_trials) if n_trials > 1 else 0.0
        q2 = norm.ppf(1 - 1.0 / (n_trials * e)) if n_trials * e > 1 else 0.0
        expected_max = (1 - EULER_MASCHERONI) * q1 + EULER_MASCHERONI * q2
        sr_0 = sr_std * expected_max + sr_benchmark

    # DSR = PSR with SR_0 as the benchmark
    gamma3 = float(pd.Series(returns).skew())
    gamma4 = float(pd.Series(returns).kurtosis() + 3)

    denom_sq = 1.0 - gamma3 * sr_hat + (gamma4 - 1) / 4.0 * sr_hat**2
    if denom_sq <= 0:
        return 0.0

    if rho > 0:
        denom_sq *= (1 + 2 * rho / (1 - rho))

    z = (sr_hat - sr_0) * np.sqrt(T - 1) / np.sqrt(denom_sq)
    return float(norm.cdf(z))


def estimate_k_eff(n_models: int, n_threshold_combos: int, method: str = "sqrt") -> int:
    """Estimate effective number of independent trials for DSR correction.

    Adjacent thresholds and same-model variants are highly correlated,
    so raw K (n_models * n_combos) massively overstates the true number
    of independent tests. This heuristic uses sqrt(n_combos) to approximate
    the effective degrees of freedom.

    Args:
        n_models: Number of distinct model types tested.
        n_threshold_combos: Number of threshold/odds/alpha configurations.
        method: Estimation method. Only "sqrt" supported.

    Returns:
        Effective number of independent trials (minimum 1).
    """
    if n_models < 1 or n_threshold_combos < 1:
        return 1
    k_eff = int(n_models * np.sqrt(n_threshold_combos))
    return max(1, k_eff)


def wilson_lower_bound(wins: int, n: int, z: float = 1.645) -> float:
    """Wilson score interval lower bound for a binomial proportion.

    Provides a conservative estimate of the true win rate that naturally
    penalizes small samples. Used as the volume-aware replacement for
    raw precision in grid search objectives.

    With z=1.645 (90% one-sided confidence):
      12/12 wins → 0.816  (vs raw 1.000)
      65/80 wins → 0.731  (vs raw 0.813)
      10/10 wins → 0.787  (vs raw 1.000)

    Args:
        wins: Number of successes (winning bets).
        n: Total number of trials (total bets).
        z: Z-score for confidence level (default 1.645 = 90% one-sided).

    Returns:
        Wilson lower bound [0, 1]. Returns 0.0 if n == 0.
    """
    if n == 0:
        return 0.0
    p_hat = wins / n
    z2 = z * z
    denominator = 1.0 + z2 / n
    center = p_hat + z2 / (2.0 * n)
    spread = z * np.sqrt(p_hat * (1.0 - p_hat) / n + z2 / (4.0 * n * n))
    return float(max(0.0, (center - spread) / denominator))


def wilson_expected_roi(
    wins: int,
    n_bets: int,
    mean_win_return: float,
    z: float = 1.645,
) -> float:
    """Wilson-adjusted expected ROI for betting strategies.

    Uses Wilson lower bound on the binary win RATE (not continuous returns)
    to compute a conservative expected return per bet. This correctly handles
    the degenerate case where all bets win (std=0) which breaks standard
    LCB on continuous returns.

    Formula: E[R] = wilson_p * mean_win_return - (1 - wilson_p) * 1.0

    Args:
        wins: Number of winning bets.
        n_bets: Total number of bets.
        mean_win_return: Average profit per winning bet (e.g. odds-1 or odds*(1-tax)-1).
        z: Z-score for Wilson confidence (default 1.645).

    Returns:
        Expected return per bet (can be negative). Multiply by 100 for ROI%.
    """
    if n_bets == 0 or mean_win_return <= 0:
        return -1.0
    wilson_p = wilson_lower_bound(wins, n_bets, z)
    return wilson_p * mean_win_return - (1.0 - wilson_p)


def estimate_return_autocorrelation(returns_per_bet: np.ndarray) -> float:
    """Estimate lag-1 autocorrelation of per-bet returns.

    Same-day bets create positive autocorrelation, inflating the
    Sharpe ratio by up to 4x (Lo 2002, Bailey & Lopez de Prado 2012).

    Args:
        returns_per_bet: Array of per-bet returns.

    Returns:
        Lag-1 autocorrelation coefficient. Returns 0.0 if fewer than 10 observations.
    """
    returns = np.asarray(returns_per_bet, dtype=float)
    if len(returns) < 10:
        return 0.0
    corr = np.corrcoef(returns[:-1], returns[1:])[0, 1]
    if not np.isfinite(corr):
        return 0.0
    return float(np.clip(corr, -0.99, 0.99))


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Calculate Expected Calibration Error (ECE).

    Weighted average of absolute difference between predicted probability
    and observed frequency across bins.

    Args:
        y_true: True binary labels (0 or 1).
        y_prob: Predicted probabilities for the positive class.
        n_bins: Number of equal-width bins.

    Returns:
        ECE value (lower is better, 0 = perfectly calibrated).
    """
    y_true = np.asarray(y_true, dtype=float).flatten()
    y_prob = np.asarray(y_prob, dtype=float).flatten()

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n_total = len(y_true)

    for i in range(n_bins):
        in_bin = (y_prob > bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])
        prop_in_bin = in_bin.sum()
        if prop_in_bin > 0:
            avg_confidence = y_prob[in_bin].mean()
            avg_accuracy = y_true[in_bin].mean()
            ece += (prop_in_bin / n_total) * abs(avg_accuracy - avg_confidence)

    return float(ece)


def ranked_probability_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Ranked Probability Score for ordered multi-class outcomes.

    RPS penalizes predictions that are further from the true outcome in the
    ordered class space (e.g., home_win > draw > away_win).

    Formula: RPS = (1/(K-1)) * sum_k (cum_pred_k - cum_obs_k)^2

    Args:
        y_true: True class labels (0, 1, 2 for 3 classes).
        y_prob: Predicted probabilities, shape (n_samples, K).

    Returns:
        Mean RPS across all samples (lower is better, 0 = perfect).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    if y_prob.ndim != 2:
        raise ValueError(f"y_prob must be 2D, got shape {y_prob.shape}")

    n_samples, K = y_prob.shape
    if K < 2:
        return 0.0

    # Build one-hot observed matrix
    y_obs = np.zeros_like(y_prob)
    y_obs[np.arange(n_samples), y_true] = 1.0

    # Cumulative sums along class axis
    cum_pred = np.cumsum(y_prob, axis=1)
    cum_obs = np.cumsum(y_obs, axis=1)

    # RPS per sample: (1/(K-1)) * sum((cum_pred - cum_obs)^2)
    rps_per_sample = np.sum((cum_pred - cum_obs) ** 2, axis=1) / (K - 1)

    return float(np.mean(rps_per_sample))


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
            "accuracy": float(self.accuracy),
            "precision": float(self.precision),
            "recall": float(self.recall),
            "f1": float(self.f1),
            "log_loss": float(self.log_loss),
        }
        if self.brier_score is not None:
            result["brier_score"] = float(self.brier_score)
        if self.roc_auc is not None:
            result["roc_auc"] = float(self.roc_auc)
        if self.roi is not None:
            result["roi"] = float(self.roi)
        if self.yield_pct is not None:
            result["yield_pct"] = float(self.yield_pct)

        if self.per_class_precision:
            for cls, val in self.per_class_precision.items():
                result[f"precision_{cls}"] = float(val)
        if self.per_class_recall:
            for cls, val in self.per_class_recall.items():
                result[f"recall_{cls}"] = float(val)

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
        if target_names is None:
            unique_labels = np.unique(y_true)
            if len(unique_labels) == 2:
                target_names = ["negative", "positive"]
            elif len(unique_labels) == 3:
                target_names = ["away_win", "draw", "home_win"]
            else:
                logger.info(f"Determined target names: {target_names}")
                target_names = [str(label) for label in unique_labels]

        return classification_report(
            y_true, y_pred,
            target_names=target_names,
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


# ---------------------------------------------------------------------------
# Time-series diagnostic metrics (Phase 2 — audit against ML best practices)
# ---------------------------------------------------------------------------


def tracking_signal(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute bias tracking signal: CFE / MAD.

    Detects systematic over- or under-prediction. |TS| > 4 signals
    persistent directional bias and should trigger retraining.

    Note: a complementary (errors, window) API lives in
    ``src.monitoring.drift_detection.tracking_signal`` for production monitoring.

    Args:
        y_true: True outcomes (binary 0/1 or continuous).
        y_pred: Predicted probabilities or values.

    Returns:
        Tracking signal value. Positive = over-predicting, negative = under-predicting.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    errors = y_true - y_pred
    cfe = np.sum(errors)
    mad = np.mean(np.abs(errors))
    if mad < 1e-10:
        return 0.0
    return float(cfe / mad)


def rolling_tracking_signal(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    window: int = 20,
) -> np.ndarray:
    """Rolling tracking signal for walk-forward monitoring.

    Returns array of same length as inputs. First (window - 1) values are NaN.

    Args:
        y_true: True outcomes.
        y_pred: Predicted probabilities or values.
        window: Rolling window size.

    Returns:
        Array of per-window tracking signal values.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    errors = y_true - y_pred
    ts_values = np.full(len(errors), np.nan)
    for i in range(window, len(errors) + 1):
        w = errors[i - window : i]
        cfe = np.sum(w)
        mad = np.mean(np.abs(w))
        ts_values[i - 1] = cfe / mad if mad > 1e-10 else 0.0
    return ts_values


def mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
) -> float:
    """Mean Absolute Scaled Error — scale-independent accuracy vs naive lag-1 forecast.

    MASE < 1 means the model beats the naive forecast (predict previous value).
    MASE = 1 means equivalent to naive. MASE > 1 means worse than naive.

    Args:
        y_true: True test outcomes.
        y_pred: Predicted values/probabilities for the test set.
        y_train: Training outcomes (used to compute naive forecast error).

    Returns:
        MASE value. Returns inf if naive error is zero (constant training series).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_train = np.asarray(y_train, dtype=float)

    naive_errors = np.abs(np.diff(y_train))
    mae_naive = np.mean(naive_errors)
    if mae_naive < 1e-10:
        return float("inf")
    mae_model = np.mean(np.abs(y_true - y_pred))
    return float(mae_model / mae_naive)


def forecast_value_added(
    model_errors: np.ndarray,
    baseline_errors: np.ndarray,
) -> float:
    """Forecast Value Added: percentage improvement over a baseline model.

    FVA > 0 means the model adds value. FVA <= 0 means do not deploy —
    the baseline (e.g., market-implied probabilities) is at least as good.

    Uses MAE as loss function. For Brier-score-based FVA (used in sniper
    optimization), compute ``1 - brier_model / brier_baseline`` directly.

    Args:
        model_errors: Residuals from the candidate model (y_true - y_pred).
        baseline_errors: Residuals from the baseline model (y_true - y_baseline).

    Returns:
        FVA as a percentage (e.g., 15.0 means 15% improvement).
    """
    model_errors = np.asarray(model_errors, dtype=float)
    baseline_errors = np.asarray(baseline_errors, dtype=float)
    mae_model = np.mean(np.abs(model_errors))
    mae_baseline = np.mean(np.abs(baseline_errors))
    if mae_baseline < 1e-10:
        return 0.0
    return float((1 - mae_model / mae_baseline) * 100)


def diebold_mariano_test(
    errors_1: np.ndarray,
    errors_2: np.ndarray,
    loss: str = "squared",
) -> dict:
    """Diebold-Mariano test for equal predictive accuracy.

    H0: both models have equal predictive accuracy.
    p < 0.05 ⇒ the models are significantly different.
    Negative DM stat ⇒ model 1 is better; positive ⇒ model 2 is better.

    Includes Harvey-Leybourne-Newbold small-sample correction and a
    stationarity pre-check on the loss differential (ADF test).

    Args:
        errors_1: Residuals from model 1.
        errors_2: Residuals from model 2.
        loss: Loss function — "squared" or "absolute".

    Returns:
        Dict with dm_stat, p_value, mean_loss_diff, adf_p, and optional warning.
    """
    from scipy.stats import t as t_dist

    errors_1 = np.asarray(errors_1, dtype=float)
    errors_2 = np.asarray(errors_2, dtype=float)

    if loss == "squared":
        d = errors_1**2 - errors_2**2
    elif loss == "absolute":
        d = np.abs(errors_1) - np.abs(errors_2)
    else:
        raise ValueError(f"Unknown loss: {loss}")

    n = len(d)
    if n < 10:
        return {
            "dm_stat": float("nan"),
            "p_value": float("nan"),
            "warning": f"Too few observations ({n}) for DM test",
        }

    # Stationarity pre-check on the loss differential
    try:
        from statsmodels.tsa.stattools import adfuller

        adf_p = adfuller(d, autolag="AIC")[1]
    except Exception:
        adf_p = 0.0  # Assume stationary if statsmodels unavailable

    if adf_p > 0.10:
        return {
            "dm_stat": float("nan"),
            "p_value": float("nan"),
            "adf_p": float(adf_p),
            "warning": "Loss differential non-stationary; DM invalid",
        }

    d_mean = np.mean(d)
    d_var = np.var(d, ddof=1)

    if d_var < 1e-10:
        return {
            "dm_stat": 0.0,
            "p_value": 1.0,
            "mean_loss_diff": float(d_mean),
            "adf_p": float(adf_p),
        }

    # Harvey-Leybourne-Newbold small-sample correction
    dm_stat = d_mean / np.sqrt(d_var / n)
    hln_correction = np.sqrt((n + 1 - 2 + n ** (-1)) / n)
    dm_corrected = dm_stat * hln_correction

    p_value = 2 * t_dist.cdf(-abs(dm_corrected), df=n - 1)
    return {
        "dm_stat": float(dm_corrected),
        "p_value": float(p_value),
        "mean_loss_diff": float(d_mean),
        "adf_p": float(adf_p),
    }
