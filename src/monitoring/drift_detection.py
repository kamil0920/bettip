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
