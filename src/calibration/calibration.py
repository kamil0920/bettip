"""
Advanced probability calibration methods for betting callibration.

Includes:
- Beta Calibration: Uses beta distribution for more flexible calibration
- Platt Scaling: Standard logistic calibration (from sklearn)
- Isotonic Regression: Non-parametric calibration (from sklearn)
- Temperature Scaling: Simple single-parameter scaling

Research shows calibration optimization leads to 69.86% higher returns
than accuracy optimization (Walsh and Joshi, 2024).
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from typing import Optional
import warnings


class BetaCalibrator(BaseEstimator, TransformerMixin):
    """
    Beta Calibration for probability calibration.

    Maps predicted probabilities through a beta distribution CDF,
    providing more flexible calibration than Platt scaling.

    The calibration function is:
        calibrated_p = 1 / (1 + 1/(exp(a) * p^b / (1-p)^c))

    Where a, b, c are learned parameters.

    This is particularly effective for:
    - Skewed probability distributions
    - When Platt scaling (sigmoid) is too restrictive
    - Sports betting where extreme probabilities matter

    Reference:
    Kull, M., Silva Filho, T., & Flach, P. (2017).
    "Beta calibration: a well-founded and easily implemented improvement on logistic calibration for binary classifiers"
    """

    def __init__(self, method: str = 'am'):
        """
        Args:
            method: Calibration method
                - 'abm': Full 3-parameter model (a, b, m where c = 1-m+b*m)
                - 'am': 2-parameter model with b=1 (a, m)
                - 'ab': 2-parameter model with m=0.5 (a, b, c=b)
        """
        self.method = method
        self.a_ = None
        self.b_ = None
        self.c_ = None

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray) -> 'BetaCalibrator':
        """
        Fit the beta calibration parameters.

        Args:
            y_prob: Predicted probabilities (uncalibrated)
            y_true: True binary labels (0 or 1)

        Returns:
            self
        """
        y_prob = np.asarray(y_prob).flatten()
        y_true = np.asarray(y_true).flatten()

        # Clip probabilities to avoid log(0)
        eps = 1e-10
        y_prob = np.clip(y_prob, eps, 1 - eps)

        # Log-odds transformation
        log_odds = np.log(y_prob / (1 - y_prob))

        if self.method == 'abm':
            # Full 3-parameter model
            result = self._fit_abm(y_prob, y_true)
        elif self.method == 'am':
            # 2-parameter with b=1
            result = self._fit_am(y_prob, y_true)
        else:  # 'ab'
            # 2-parameter with m=0.5 (symmetric)
            result = self._fit_ab(y_prob, y_true)

        return self

    def _fit_abm(self, y_prob: np.ndarray, y_true: np.ndarray):
        """Fit full 3-parameter model."""
        eps = 1e-10

        def neg_log_likelihood(params):
            a, b, m = params
            c = 1 - m + b * m

            # Beta calibration transformation
            log_odds_new = a + b * np.log(y_prob + eps) - c * np.log(1 - y_prob + eps)
            calibrated = expit(log_odds_new)
            calibrated = np.clip(calibrated, eps, 1 - eps)

            # Binary cross-entropy loss
            loss = -np.mean(y_true * np.log(calibrated) + (1 - y_true) * np.log(1 - calibrated))
            return loss

        # Initialize with reasonable values
        x0 = [0.0, 1.0, 0.5]
        bounds = [(-10, 10), (0.01, 10), (0, 1)]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(neg_log_likelihood, x0, method='L-BFGS-B', bounds=bounds)

        self.a_ = result.x[0]
        self.b_ = result.x[1]
        m = result.x[2]
        self.c_ = 1 - m + self.b_ * m

        return result

    def _fit_am(self, y_prob: np.ndarray, y_true: np.ndarray):
        """Fit 2-parameter model with b=1."""
        eps = 1e-10

        def neg_log_likelihood(params):
            a, m = params
            b = 1.0
            c = 1 - m + m  # = 1

            log_odds_new = a + b * np.log(y_prob + eps) - c * np.log(1 - y_prob + eps)
            calibrated = expit(log_odds_new)
            calibrated = np.clip(calibrated, eps, 1 - eps)

            loss = -np.mean(y_true * np.log(calibrated) + (1 - y_true) * np.log(1 - calibrated))
            return loss

        x0 = [0.0, 0.5]
        bounds = [(-10, 10), (0, 1)]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(neg_log_likelihood, x0, method='L-BFGS-B', bounds=bounds)

        self.a_ = result.x[0]
        self.b_ = 1.0
        m = result.x[1]
        self.c_ = 1.0

        return result

    def _fit_ab(self, y_prob: np.ndarray, y_true: np.ndarray):
        """Fit 2-parameter symmetric model (m=0.5, so c=b)."""
        eps = 1e-10

        def neg_log_likelihood(params):
            a, b = params
            c = b  # symmetric

            log_odds_new = a + b * np.log(y_prob + eps) - c * np.log(1 - y_prob + eps)
            calibrated = expit(log_odds_new)
            calibrated = np.clip(calibrated, eps, 1 - eps)

            loss = -np.mean(y_true * np.log(calibrated) + (1 - y_true) * np.log(1 - calibrated))
            return loss

        x0 = [0.0, 1.0]
        bounds = [(-10, 10), (0.01, 10)]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(neg_log_likelihood, x0, method='L-BFGS-B', bounds=bounds)

        self.a_ = result.x[0]
        self.b_ = result.x[1]
        self.c_ = result.x[1]

        return result

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        """
        Apply beta calibration to probabilities.

        Args:
            y_prob: Uncalibrated probabilities

        Returns:
            Calibrated probabilities
        """
        if self.a_ is None:
            raise ValueError("Calibrator not fitted. Call fit() first.")

        y_prob = np.asarray(y_prob).flatten()
        eps = 1e-10
        y_prob = np.clip(y_prob, eps, 1 - eps)

        log_odds_new = self.a_ + self.b_ * np.log(y_prob) - self.c_ * np.log(1 - y_prob)
        calibrated = expit(log_odds_new)

        return np.clip(calibrated, eps, 1 - eps)

    def fit_transform(self, y_prob: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(y_prob, y_true)
        return self.transform(y_prob)

    def get_params_str(self) -> str:
        """Get string representation of calibration parameters."""
        return f"a={self.a_:.4f}, b={self.b_:.4f}, c={self.c_:.4f}"


class TemperatureScaling(BaseEstimator, TransformerMixin):
    """
    Temperature Scaling for probability calibration.

    Simple single-parameter calibration that divides logits by a temperature T:
        calibrated_p = sigmoid(logit(p) / T)

    When T > 1: Makes probabilities less confident (closer to 0.5)
    When T < 1: Makes probabilities more confident (closer to 0 or 1)
    When T = 1: No change

    This is commonly used in neural networks but works for any classifier.
    """

    def __init__(self):
        self.temperature_ = 1.0

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray) -> 'TemperatureScaling':
        """Fit the temperature parameter."""
        y_prob = np.asarray(y_prob).flatten()
        y_true = np.asarray(y_true).flatten()

        eps = 1e-10
        y_prob = np.clip(y_prob, eps, 1 - eps)

        def neg_log_likelihood(T):
            T = max(T[0], 0.01)  # Ensure T > 0
            log_odds = logit(y_prob) / T
            calibrated = expit(log_odds)
            calibrated = np.clip(calibrated, eps, 1 - eps)

            loss = -np.mean(y_true * np.log(calibrated) + (1 - y_true) * np.log(1 - calibrated))
            return loss

        result = minimize(neg_log_likelihood, [1.0], method='L-BFGS-B', bounds=[(0.01, 10)])
        self.temperature_ = result.x[0]

        return self

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        """Apply temperature scaling."""
        y_prob = np.asarray(y_prob).flatten()
        eps = 1e-10
        y_prob = np.clip(y_prob, eps, 1 - eps)

        log_odds = logit(y_prob) / self.temperature_
        calibrated = expit(log_odds)

        return np.clip(calibrated, eps, 1 - eps)

    def fit_transform(self, y_prob: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        self.fit(y_prob, y_true)
        return self.transform(y_prob)


class EnsembleCalibrator(BaseEstimator, TransformerMixin):
    """
    Ensemble of multiple calibration methods.

    Combines predictions from multiple calibrators using averaging or
    learned weights.
    """

    def __init__(self, methods: list = None, weights: str = 'equal'):
        """
        Args:
            methods: List of calibrator names ['beta', 'platt', 'isotonic', 'temperature']
            weights: 'equal' for equal weights, 'learned' to optimize weights
        """
        self.methods = methods or ['beta', 'platt', 'isotonic']
        self.weights = weights
        self.calibrators_ = {}
        self.weights_ = None

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray) -> 'EnsembleCalibrator':
        """Fit all calibrators."""
        y_prob = np.asarray(y_prob).flatten()
        y_true = np.asarray(y_true).flatten()

        for method in self.methods:
            if method == 'beta':
                cal = BetaCalibrator(method='abm')
            elif method == 'platt':
                cal = PlattScaling()
            elif method == 'isotonic':
                cal = IsotonicCalibrator()
            elif method == 'temperature':
                cal = TemperatureScaling()
            else:
                raise ValueError(f"Unknown calibration method: {method}")

            cal.fit(y_prob, y_true)
            self.calibrators_[method] = cal

        if self.weights == 'equal':
            self.weights_ = np.ones(len(self.methods)) / len(self.methods)
        else:
            # Learn optimal weights by minimizing calibration error
            self._learn_weights(y_prob, y_true)

        return self

    def _learn_weights(self, y_prob: np.ndarray, y_true: np.ndarray):
        """Learn optimal ensemble weights."""
        predictions = np.column_stack([
            self.calibrators_[m].transform(y_prob) for m in self.methods
        ])

        def neg_log_likelihood(w):
            w = np.abs(w) / np.sum(np.abs(w))  # Normalize to sum to 1
            combined = predictions @ w
            combined = np.clip(combined, 1e-10, 1 - 1e-10)
            loss = -np.mean(y_true * np.log(combined) + (1 - y_true) * np.log(1 - combined))
            return loss

        x0 = np.ones(len(self.methods)) / len(self.methods)
        result = minimize(neg_log_likelihood, x0, method='L-BFGS-B')
        self.weights_ = np.abs(result.x) / np.sum(np.abs(result.x))

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        """Apply ensemble calibration."""
        predictions = np.column_stack([
            self.calibrators_[m].transform(y_prob) for m in self.methods
        ])
        return predictions @ self.weights_

    def fit_transform(self, y_prob: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        self.fit(y_prob, y_true)
        return self.transform(y_prob)


class PlattScaling(BaseEstimator, TransformerMixin):
    """Platt Scaling (logistic calibration) wrapper."""

    def __init__(self):
        self.lr_ = LogisticRegression(solver='lbfgs', max_iter=1000)

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray) -> 'PlattScaling':
        y_prob = np.asarray(y_prob).reshape(-1, 1)
        y_true = np.asarray(y_true).flatten()
        self.lr_.fit(y_prob, y_true)
        return self

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        y_prob = np.asarray(y_prob).reshape(-1, 1)
        return self.lr_.predict_proba(y_prob)[:, 1]

    def fit_transform(self, y_prob: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        self.fit(y_prob, y_true)
        return self.transform(y_prob)


class IsotonicCalibrator(BaseEstimator, TransformerMixin):
    """Isotonic regression calibration wrapper."""

    def __init__(self):
        self.ir_ = IsotonicRegression(out_of_bounds='clip')

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray) -> 'IsotonicCalibrator':
        y_prob = np.asarray(y_prob).flatten()
        y_true = np.asarray(y_true).flatten()
        self.ir_.fit(y_prob, y_true)
        return self

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        y_prob = np.asarray(y_prob).flatten()
        return self.ir_.transform(y_prob)

    def fit_transform(self, y_prob: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        self.fit(y_prob, y_true)
        return self.transform(y_prob)


def calibration_metrics(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> dict:
    """
    Calculate calibration metrics.

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        n_bins: Number of bins for ECE calculation

    Returns:
        Dict with calibration metrics:
        - ece: Expected Calibration Error
        - mce: Maximum Calibration Error
        - brier: Brier Score
        - reliability: Reliability diagram data
    """
    y_true = np.asarray(y_true).flatten()
    y_prob = np.asarray(y_prob).flatten()

    # Brier score
    brier = np.mean((y_prob - y_true) ** 2)

    # ECE and MCE
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    mce = 0.0
    reliability_data = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            avg_confidence = y_prob[in_bin].mean()
            avg_accuracy = y_true[in_bin].mean()
            calibration_error = abs(avg_accuracy - avg_confidence)

            ece += prop_in_bin * calibration_error
            mce = max(mce, calibration_error)

            reliability_data.append({
                'bin_lower': bin_lower,
                'bin_upper': bin_upper,
                'avg_confidence': avg_confidence,
                'avg_accuracy': avg_accuracy,
                'count': in_bin.sum(),
                'error': calibration_error
            })

    return {
        'ece': ece,
        'mce': mce,
        'brier': brier,
        'reliability': reliability_data
    }


def compare_calibrators(y_prob: np.ndarray, y_true: np.ndarray,
                       val_prob: np.ndarray = None, val_true: np.ndarray = None) -> dict:
    """
    Compare different calibration methods.

    Args:
        y_prob: Training probabilities
        y_true: Training labels
        val_prob: Validation probabilities (optional, uses training if None)
        val_true: Validation labels (optional)

    Returns:
        Dict with comparison results for each method
    """
    if val_prob is None:
        val_prob = y_prob
        val_true = y_true

    calibrators = {
        'uncalibrated': None,
        'platt': PlattScaling(),
        'isotonic': IsotonicCalibrator(),
        'beta_abm': BetaCalibrator(method='abm'),
        'beta_am': BetaCalibrator(method='am'),
        'beta_ab': BetaCalibrator(method='ab'),
        'temperature': TemperatureScaling(),
        'ensemble': EnsembleCalibrator(methods=['beta', 'platt', 'isotonic'], weights='learned'),
    }

    results = {}

    for name, calibrator in calibrators.items():
        if calibrator is None:
            calibrated = val_prob
        else:
            calibrator.fit(y_prob, y_true)
            calibrated = calibrator.transform(val_prob)

        metrics = calibration_metrics(val_true, calibrated)
        results[name] = {
            'ece': metrics['ece'],
            'mce': metrics['mce'],
            'brier': metrics['brier'],
        }

    return results


def get_calibrator(method: str = "sigmoid"):
    """
    Factory function to create a calibrator by name.

    Args:
        method: One of "sigmoid" (Platt), "isotonic", "beta", "temperature".

    Returns:
        Calibrator instance with fit/transform interface.

    Raises:
        ValueError: If method is unknown.
    """
    method = method.lower()
    if method in ("sigmoid", "platt"):
        return PlattScaling()
    elif method == "isotonic":
        return IsotonicCalibrator()
    elif method == "beta":
        return BetaCalibrator(method="abm")
    elif method == "temperature":
        return TemperatureScaling()
    else:
        raise ValueError(
            f"Unknown calibration method: {method}. "
            f"Available: sigmoid, isotonic, beta, temperature"
        )


if __name__ == "__main__":
    # Test calibration methods
    np.random.seed(42)

    # Generate synthetic uncalibrated probabilities
    n = 1000
    y_true = np.random.binomial(1, 0.3, n)  # 30% positive rate

    # Uncalibrated predictions (overconfident)
    noise = np.random.normal(0, 0.3, n)
    y_prob = 1 / (1 + np.exp(-2 * (y_true - 0.5 + noise)))
    y_prob = np.clip(y_prob, 0.01, 0.99)

    print("Comparing calibration methods:")
    print("-" * 50)

    results = compare_calibrators(y_prob, y_true)

    for method, metrics in sorted(results.items(), key=lambda x: x[1]['ece']):
        print(f"{method:15s}: ECE={metrics['ece']:.4f}, MCE={metrics['mce']:.4f}, Brier={metrics['brier']:.4f}")
