"""
Data and Model Drift Detection with Evidently

Generates drift reports comparing reference (training) data against
current (production/new) data. Alerts when significant drift is detected,
which signals the model should be retrained.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DriftDetector:
    """Detects data drift and model performance degradation.

    Uses Evidently to compare feature distributions between reference
    (training) and current (production) data.
    """

    def __init__(
        self,
        drift_threshold: float = 0.3,
        feature_drift_threshold: float = 0.1,
    ):
        """
        Args:
            drift_threshold: Fraction of features that must drift to trigger alert.
            feature_drift_threshold: p-value threshold for individual feature drift.
        """
        self.drift_threshold = drift_threshold
        self.feature_drift_threshold = feature_drift_threshold

    def generate_drift_report(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate a data drift report.

        Args:
            reference_data: Training/reference DataFrame.
            current_data: Current/production DataFrame.
            feature_columns: Columns to check for drift (default: all numeric).
            output_path: Optional path to save HTML report.

        Returns:
            Dict with drift summary: n_drifted, fraction_drifted, alert, per_feature results.
        """
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset

        if feature_columns is None:
            feature_columns = reference_data.select_dtypes(include=[np.number]).columns.tolist()

        ref = reference_data[feature_columns].copy()
        cur = current_data[feature_columns].copy()

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=ref, current_data=cur)

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            report.save_html(output_path)
            logger.info(f"Drift report saved to {output_path}")

        # Extract results
        report_dict = report.as_dict()
        metrics = report_dict.get('metrics', [])

        # Find the dataset drift metric
        dataset_drift = {}
        feature_drift = {}

        for metric in metrics:
            result = metric.get('result', {})
            if 'drift_by_columns' in result:
                for col_name, col_data in result['drift_by_columns'].items():
                    feature_drift[col_name] = {
                        'drifted': col_data.get('drift_detected', False),
                        'stattest': col_data.get('stattest_name', ''),
                        'p_value': col_data.get('drift_score', 1.0),
                    }
            if 'share_of_drifted_columns' in result:
                dataset_drift = {
                    'fraction_drifted': result['share_of_drifted_columns'],
                    'n_drifted': result.get('number_of_drifted_columns', 0),
                    'n_columns': result.get('number_of_columns', len(feature_columns)),
                    'dataset_drift': result.get('dataset_drift', False),
                }

        fraction_drifted = dataset_drift.get('fraction_drifted', 0)
        alert = fraction_drifted > self.drift_threshold

        summary = {
            'alert': alert,
            'fraction_drifted': fraction_drifted,
            'n_drifted': dataset_drift.get('n_drifted', 0),
            'n_features': dataset_drift.get('n_columns', len(feature_columns)),
            'threshold': self.drift_threshold,
            'per_feature': feature_drift,
        }

        level = 'WARNING' if alert else 'INFO'
        logger.log(
            logging.WARNING if alert else logging.INFO,
            f"Drift report: {summary['n_drifted']}/{summary['n_features']} features drifted "
            f"({fraction_drifted:.1%}) — {'ALERT: retrain recommended' if alert else 'within tolerance'}"
        )

        return summary

    def generate_performance_report(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        target_column: str,
        prediction_column: str,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate a classification performance drift report.

        Args:
            reference_data: Reference DataFrame with target + predictions.
            current_data: Current DataFrame with target + predictions.
            target_column: Name of the target column.
            prediction_column: Name of the prediction column.
            output_path: Optional path to save HTML report.

        Returns:
            Dict with performance comparison metrics.
        """
        from evidently.report import Report
        from evidently.metric_preset import ClassificationPreset

        report = Report(metrics=[ClassificationPreset()])

        ref = reference_data[[target_column, prediction_column]].copy()
        cur = current_data[[target_column, prediction_column]].copy()

        # Evidently expects 'target' and 'prediction' columns
        ref = ref.rename(columns={target_column: 'target', prediction_column: 'prediction'})
        cur = cur.rename(columns={target_column: 'target', prediction_column: 'prediction'})

        report.run(reference_data=ref, current_data=cur)

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            report.save_html(output_path)
            logger.info(f"Performance report saved to {output_path}")

        return report.as_dict()

    def should_retrain(self, drift_summary: Dict[str, Any]) -> bool:
        """Check if retraining is recommended based on drift report.

        Args:
            drift_summary: Output from generate_drift_report().

        Returns:
            True if drift exceeds threshold and retraining is recommended.
        """
        return drift_summary.get('alert', False)

    def should_retrain_extended(
        self,
        drift_summary: Dict[str, Any],
        model_age_days: int | None = None,
        max_model_age_days: int = 60,
        tracking_signal_value: float | None = None,
        tracking_signal_threshold: float = 4.0,
        live_roi: float | None = None,
        backtest_roi: float | None = None,
        roi_gap_threshold: float = -30.0,
        live_ece: float | None = None,
        training_ece: float | None = None,
        ece_alert_multiplier: float = 2.0,
    ) -> tuple[bool, list[str]]:
        """Extended retraining check combining multiple signals.

        Combines 5 independent signals:
        1. Feature drift (existing Evidently-based detection)
        2. Model staleness (age exceeds max_model_age_days)
        3. Tracking signal (persistent directional bias, |TS| > threshold)
        4. Live ROI gap (live performance significantly worse than backtest)
        5. ECE drift (live calibration error exceeds multiplier × training ECE)

        Args:
            drift_summary: Output from generate_drift_report().
            model_age_days: Age of the deployed model in days (None = unknown).
            max_model_age_days: Maximum acceptable model age before flagging.
            tracking_signal_value: Current tracking signal value (None = unavailable).
            tracking_signal_threshold: |TS| threshold for bias alert.
            live_roi: Live ROI percentage (None = unavailable).
            backtest_roi: Backtest ROI percentage (None = unavailable).
            roi_gap_threshold: Maximum acceptable (live - backtest) gap in percentage points.
            live_ece: Live ECE computed from settled bets (None = unavailable).
            training_ece: ECE from training/holdout evaluation (None = unavailable).
            ece_alert_multiplier: Alert when live ECE > multiplier × training ECE.

        Returns:
            Tuple of (should_retrain, list_of_reasons).
        """
        reasons: list[str] = []

        # Signal 1: Feature drift
        if drift_summary.get('alert', False):
            frac = drift_summary.get('fraction_drifted', 0)
            reasons.append(f"feature_drift: {frac:.1%} features drifted")

        # Signal 2: Model staleness
        if model_age_days is not None and model_age_days > max_model_age_days:
            reasons.append(f"stale_model: {model_age_days}d old (max {max_model_age_days}d)")

        # Signal 3: Tracking signal (asymmetric — overprediction is dangerous)
        if tracking_signal_value is not None and tracking_signal_value > tracking_signal_threshold:
            reasons.append(
                f"tracking_signal: TS={tracking_signal_value:+.2f} "
                f"(OVER-predicting, dangerous, threshold +{tracking_signal_threshold})"
            )
        elif tracking_signal_value is not None and tracking_signal_value < -15.0:
            reasons.append(
                f"tracking_signal: TS={tracking_signal_value:+.2f} "
                f"(under-predicting, safe but too conservative)"
            )

        # Signal 4: Live ROI gap
        if live_roi is not None and backtest_roi is not None:
            gap = live_roi - backtest_roi
            if gap < roi_gap_threshold:
                reasons.append(
                    f"roi_gap: live={live_roi:.1f}% vs backtest={backtest_roi:.1f}% "
                    f"(gap={gap:.1f}pp, threshold {roi_gap_threshold}pp)"
                )

        # Signal 5: ECE drift — the #1 predictor of live market failure.
        # Calibration degrades when the probability distribution shifts between
        # training and production data. Alert when live ECE > 2× training ECE.
        if live_ece is not None and training_ece is not None and training_ece > 0:
            ece_ratio = live_ece / training_ece
            if ece_ratio > ece_alert_multiplier:
                reasons.append(
                    f"ece_drift: live ECE={live_ece:.4f} vs training={training_ece:.4f} "
                    f"({ece_ratio:.1f}x, threshold {ece_alert_multiplier}x)"
                )

        should = len(reasons) > 0
        if should:
            logger.warning(
                f"Retraining recommended — {len(reasons)} signal(s): {'; '.join(reasons)}"
            )
        else:
            logger.info("No retraining signals triggered")

        return should, reasons


def tracking_signal(errors: np.ndarray, window: int = 50) -> float:
    """Cumulative Forecast Error / Mean Absolute Deviation.

    Detects persistent directional bias in predictions.
    |TS| > 4 indicates systematic over/under-prediction.

    Args:
        errors: Signed errors (prediction - actual). Positive = over-predicting.
        window: Use last N errors for computation.

    Returns:
        Tracking signal value. |TS| > 4 is the standard alert threshold.
    """
    errors = np.asarray(errors, dtype=float)
    errors = errors[~np.isnan(errors)]
    if len(errors) < 2:
        return 0.0
    recent = errors[-window:]
    cfe = np.sum(recent)
    mad = np.mean(np.abs(recent))
    return cfe / mad if mad > 0 else 0.0


def rolling_tracking_signal(
    errors: np.ndarray, window: int = 50
) -> np.ndarray:
    """Compute tracking signal over a rolling window.

    Args:
        errors: Full series of signed errors (prediction - actual).
        window: Rolling window size.

    Returns:
        Array of TS values, same length as errors. First (window-1) values are NaN.
    """
    errors = np.asarray(errors, dtype=float)
    n = len(errors)
    ts = np.full(n, np.nan)
    for i in range(window - 1, n):
        chunk = errors[i - window + 1 : i + 1]
        valid = chunk[~np.isnan(chunk)]
        if len(valid) < 2:
            continue
        cfe = np.sum(valid)
        mad = np.mean(np.abs(valid))
        ts[i] = cfe / mad if mad > 0 else 0.0
    return ts


def ece_drift_monitor(
    recent_probs: np.ndarray,
    recent_outcomes: np.ndarray,
    training_ece: float,
    n_bins: int = 10,
    alert_multiplier: float = 2.0,
) -> dict:
    """Monitor ECE drift in production.

    Compares live calibration error against the training baseline.
    Alert when live ECE exceeds ``alert_multiplier × training_ece``.
    ECE drift is the #1 predictor of live market failure in this project.

    Args:
        recent_probs: Predicted probabilities from settled bets.
        recent_outcomes: Actual outcomes (0/1) for those bets.
        training_ece: ECE from training/holdout evaluation.
        n_bins: Number of equal-width bins for ECE computation.
        alert_multiplier: Alert threshold as a multiple of training ECE.

    Returns:
        Dict with live_ece, training_ece, ece_ratio, alert flag, and recommendation.
    """
    recent_probs = np.asarray(recent_probs, dtype=float)
    recent_outcomes = np.asarray(recent_outcomes, dtype=float)

    if len(recent_probs) < 5:
        return {
            "live_ece": float("nan"),
            "training_ece": float(training_ece),
            "ece_ratio": float("nan"),
            "alert": False,
            "n_bets": len(recent_probs),
            "recommendation": "INSUFFICIENT_DATA",
        }

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (recent_probs >= bin_edges[i]) & (recent_probs < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_acc = recent_outcomes[mask].mean()
            bin_conf = recent_probs[mask].mean()
            ece += mask.sum() * abs(bin_acc - bin_conf)
    ece /= len(recent_probs)

    ece_ratio = ece / training_ece if training_ece > 0 else float("inf")
    alert = ece_ratio > alert_multiplier

    return {
        "live_ece": float(ece),
        "training_ece": float(training_ece),
        "ece_ratio": float(ece_ratio),
        "alert": alert,
        "n_bets": len(recent_probs),
        "recommendation": "RETRAIN" if alert else "OK",
    }
