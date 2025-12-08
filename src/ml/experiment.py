"""
MLflow experiment tracking integration.

Provides:
- Experiment configuration and management
- Automatic logging of parameters, metrics, and artifacts
- Model versioning and registry integration
"""
import json
import logging
import tempfile
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.ml.models import ModelFactory, get_feature_importance
from src.ml.metrics import SportsMetrics, PredictionMetrics

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for an experiment run."""
    experiment_name: str
    run_name: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)

    model_type: str = "random_forest"
    model_params: Dict[str, Any] = field(default_factory=dict)

    features_file: str = "features.csv"
    target_column: str = "home_win"
    feature_columns: Optional[List[str]] = None
    exclude_columns: List[str] = field(default_factory=lambda: [
        "fixture_id", "date", "home_team_id", "home_team_name",
        "away_team_id", "away_team_name", "round",
        "home_win", "draw", "away_win", "match_result",
        "total_goals", "goal_difference"
    ])

    test_size: float = 0.2
    random_state: int = 42
    time_based_split: bool = True

    tracking_uri: str = "sqlite:///mlflow.db"
    artifact_location: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {k: v for k, v in asdict(self).items() if v is not None}


class Experiment:
    """
    MLflow experiment wrapper for sports prediction.

    Example usage:
        config = ExperimentConfig(
            experiment_name="feature_selection",
            model_type="xgboost",
            target_column="home_win"
        )
        exp = Experiment(config)
        results = exp.run(X_train, X_test, y_train, y_test)
    """

    def __init__(self, config: ExperimentConfig):
        """
        Initialize experiment.

        Args:
            config: Experiment configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        mlflow.set_tracking_uri(config.tracking_uri)
        mlflow.set_experiment(config.experiment_name)

        self.model = None
        self.metrics: Optional[PredictionMetrics] = None
        self.feature_importance: Dict[str, float] = {}

    def run(
            self,
            X_train: pd.DataFrame,
            X_test: pd.DataFrame,
            y_train: pd.Series,
            y_test: pd.Series,
            odds_test: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Execute a single experiment run.

        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training target
            y_test: Test target
            odds_test: Optional betting odds for ROI calculation

        Returns:
            Dictionary with run results
        """
        run_name = self.config.run_name or f"{self.config.model_type}_{datetime.now():%Y%m%d_%H%M%S}"

        with mlflow.start_run(run_name=run_name) as run:
            self.logger.info(f"Starting run: {run_name}")
            self.logger.info(f"Run ID: {run.info.run_id}")

            self._log_config()

            self._log_dataset_info(X_train, X_test, y_train, y_test)

            self.logger.info(f"Training {self.config.model_type}...")
            self.model = ModelFactory.create(
                self.config.model_type,
                self.config.model_params
            )
            self.model.fit(X_train, y_train)

            y_pred = self.model.predict(X_test)
            y_proba = None
            if hasattr(self.model, "predict_proba"):
                y_proba = self.model.predict_proba(X_test)

            self.logger.info("Calculating metrics...")
            self.metrics = SportsMetrics.calculate_all(
                y_test.values, y_pred, y_proba, odds_test
            )

            sanitized_metrics = _sanitize_for_mlflow(self.metrics.to_dict())
            mlflow.log_metrics(sanitized_metrics)

            self.feature_importance = get_feature_importance(
                self.model, list(X_train.columns)
            )

            if self.feature_importance:
                self.feature_importance = _sanitize_for_mlflow(self.feature_importance)
                self._log_feature_importance()

            mlflow.sklearn.log_model(
                self.model,
                name="model",
                registered_model_name=None
            )

            self._log_artifacts(y_test.values, y_pred, y_proba)

            self.logger.info(f"Run completed. Accuracy: {self.metrics.accuracy:.4f}")

            return {
                "run_id": run.info.run_id,
                "model": self.model,
                "metrics": self.metrics,
                "feature_importance": self.feature_importance,
            }

    def _log_config(self) -> None:
        """Log experiment configuration."""
        mlflow.log_params({
            "model_type": self.config.model_type,
            "target_column": self.config.target_column,
            "test_size": self.config.test_size,
            "random_state": self.config.random_state,
            "time_based_split": self.config.time_based_split,
        })

        clean_params = _sanitize_for_mlflow(self.config.model_params)

        for key, value in clean_params.items():
            mlflow.log_param(f"model_{key}", value)

        if self.config.tags:
            mlflow.set_tags(self.config.tags)

    def _log_dataset_info(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series
    ) -> None:
        """Log dataset information."""
        mlflow.log_params({
            "n_train_samples": len(X_train),
            "n_test_samples": len(X_test),
            "n_features": X_train.shape[1],
        })

        train_dist = y_train.value_counts(normalize=True).to_dict()
        test_dist = y_test.value_counts(normalize=True).to_dict()

        for cls, pct in train_dist.items():
            mlflow.log_param(f"train_class_{cls}_pct", f"{pct:.3f}")
        for cls, pct in test_dist.items():
            mlflow.log_param(f"test_class_{cls}_pct", f"{pct:.3f}")

    def _log_feature_importance(self) -> None:
        """Log feature importance as artifact."""
        top_features = dict(list(self.feature_importance.items())[:10])
        for name, importance in top_features.items():
            safe_name = name.replace(" ", "_").replace("-", "_")[:50]
            mlflow.log_metric(f"importance_{safe_name}", importance)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.feature_importance, f, indent=2)
            mlflow.log_artifact(f.name, "feature_importance")

    def _log_artifacts(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray]
    ) -> None:
        """Log plots and reports as artifacts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            report = SportsMetrics.get_classification_report(y_true, y_pred)
            report_path = tmpdir / "classification_report.txt"
            report_path.write_text(report)
            mlflow.log_artifact(str(report_path))

            self._plot_confusion_matrix(y_true, y_pred, tmpdir)

            if self.feature_importance:
                self._plot_feature_importance(tmpdir)

            if y_proba is not None:
                self._plot_calibration(y_true, y_proba, tmpdir)

    def _plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        output_dir: Path
    ) -> None:
        """Create and save confusion matrix plot."""
        cm = SportsMetrics.get_confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)

        classes = ["Away Win", "Draw", "Home Win"]
        ax.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=classes[:cm.shape[1]],
            yticklabels=classes[:cm.shape[0]],
            title="Confusion Matrix",
            ylabel="True label",
            xlabel="Predicted label"
        )

        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")

        fig.tight_layout()
        fig_path = output_dir / "confusion_matrix.png"
        fig.savefig(fig_path, dpi=100)
        plt.close(fig)
        mlflow.log_artifact(str(fig_path))

    def _plot_feature_importance(self, output_dir: Path) -> None:
        """Create and save feature importance plot."""
        top_n = 20
        features = list(self.feature_importance.keys())[:top_n]
        importances = list(self.feature_importance.values())[:top_n]

        fig, ax = plt.subplots(figsize=(10, 8))
        y_pos = np.arange(len(features))
        ax.barh(y_pos, importances, align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()
        ax.set_xlabel('Importance')
        ax.set_title(f'Top {top_n} Feature Importance')

        fig.tight_layout()
        fig_path = output_dir / "feature_importance.png"
        fig.savefig(fig_path, dpi=100)
        plt.close(fig)
        mlflow.log_artifact(str(fig_path))

    def _plot_calibration(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        output_dir: Path
    ) -> None:
        """Create and save calibration plot."""
        calibration = SportsMetrics.calculate_calibration(y_true, y_proba)

        fig, ax = plt.subplots(figsize=(8, 6))

        ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')

        mask = ~np.isnan(calibration['bin_accuracy'])
        ax.plot(
            calibration['bin_confidence'][mask],
            calibration['bin_accuracy'][mask],
            'o-', label='Model calibration'
        )

        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title('Calibration Plot')
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig_path = output_dir / "calibration.png"
        fig.savefig(fig_path, dpi=100)
        plt.close(fig)
        mlflow.log_artifact(str(fig_path))


def _sanitize_for_mlflow(data):
    """
    Recursively convert NumPy types to native Python types for JSON serialization.
    """
    if isinstance(data, dict):
        return {k: _sanitize_for_mlflow(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_sanitize_for_mlflow(v) for v in data]
    elif isinstance(data, (np.floating, float)):
        return float(data)
    elif isinstance(data, (np.integer, int)):
        return int(data)
    elif isinstance(data, np.ndarray):
        return data.tolist()
    else:
        return data

def compare_runs(
    experiment_name: str,
    metric: str = "accuracy",
    top_n: int = 10
) -> pd.DataFrame:
    """
    Compare runs in an experiment.

    Args:
        experiment_name: Name of the experiment
        metric: Metric to sort by
        top_n: Number of top runs to return

    Returns:
        DataFrame with run comparison
    """
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} DESC"],
        max_results=top_n
    )

    results = []
    for run in runs:
        result = {
            "run_id": run.info.run_id[:8],
            "run_name": run.info.run_name,
            "model_type": run.data.params.get("model_type", "unknown"),
        }
        result.update(run.data.metrics)
        results.append(result)

    return pd.DataFrame(results)
