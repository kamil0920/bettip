"""
MLflow Configuration and Model Registry Setup

This module provides:
1. MLflow tracking configuration
2. Model registry utilities
3. Experiment management
4. Model versioning and promotion
"""

import os
import json
import mlflow
from mlflow.tracking import MlflowClient
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)


@dataclass
class MLflowConfig:
    """MLflow configuration."""
    tracking_uri: str = "sqlite:///mlflow.db"
    artifact_location: str = "./mlruns"
    experiment_name: str = "bettip"
    registry_uri: Optional[str] = None

    @classmethod
    def from_env(cls) -> "MLflowConfig":
        """Load config from environment variables."""
        return cls(
            tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"),
            artifact_location=os.getenv("MLFLOW_ARTIFACT_LOCATION", "./mlruns"),
            experiment_name=os.getenv("MLFLOW_EXPERIMENT_NAME", "bettip"),
            registry_uri=os.getenv("MLFLOW_REGISTRY_URI"),
        )


class MLflowManager:
    """Manages MLflow experiments, runs, and model registry."""

    def __init__(self, config: Optional[MLflowConfig] = None):
        self.config = config or MLflowConfig.from_env()
        self._setup_mlflow()
        self.client = MlflowClient()

    def _setup_mlflow(self):
        """Initialize MLflow tracking."""
        mlflow.set_tracking_uri(self.config.tracking_uri)

        artifact_path = Path(self.config.artifact_location)
        artifact_path.mkdir(parents=True, exist_ok=True)

        experiment = mlflow.get_experiment_by_name(self.config.experiment_name)
        if experiment is None:
            mlflow.create_experiment(
                self.config.experiment_name,
                artifact_location=self.config.artifact_location
            )
        mlflow.set_experiment(self.config.experiment_name)

        logger.info(f"MLflow configured: {self.config.tracking_uri}")

    def start_run(self, run_name: str, tags: Optional[Dict[str, str]] = None) -> mlflow.ActiveRun:
        """Start a new MLflow run."""
        return mlflow.start_run(run_name=run_name, tags=tags)

    def log_params(self, params: Dict[str, Any]):
        """Log parameters to current run."""
        for key, value in params.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    mlflow.log_param(f"{key}.{k}", v)
            else:
                mlflow.log_param(key, value)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to current run."""
        mlflow.log_metrics(metrics, step=step)

    def log_model(self, model, artifact_path: str, registered_model_name: Optional[str] = None,
                  signature=None, input_example=None):
        """Log model to MLflow."""
        model_type = type(model).__name__

        clean_path = artifact_path.replace('/', '_').replace(':', '_')

        if "XGB" in model_type:
            mlflow.xgboost.log_model(
                model, clean_path,
                registered_model_name=registered_model_name,
                signature=signature,
                input_example=input_example
            )
        elif "LGBM" in model_type or "LightGBM" in model_type:
            mlflow.lightgbm.log_model(
                model, clean_path,
                registered_model_name=registered_model_name,
                signature=signature,
                input_example=input_example
            )
        elif "CatBoost" in model_type:
            mlflow.catboost.log_model(
                model, clean_path,
                registered_model_name=registered_model_name,
                signature=signature,
                input_example=input_example
            )
        else:
            mlflow.sklearn.log_model(
                model, clean_path,
                registered_model_name=registered_model_name,
                signature=signature,
                input_example=input_example
            )

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log artifact file."""
        mlflow.log_artifact(local_path, artifact_path)

    def log_dict(self, data: Dict, artifact_file: str):
        """Log dictionary as JSON artifact."""
        mlflow.log_dict(data, artifact_file)

    def register_model(self, model_uri: str, name: str) -> str:
        """Register a model in the model registry."""
        result = mlflow.register_model(model_uri, name)
        return result.version

    def promote_model(self, name: str, version: str, stage: str = "Production"):
        """Promote a model version to a stage (Staging, Production, Archived)."""
        self.client.transition_model_version_stage(
            name=name,
            version=version,
            stage=stage
        )
        logger.info(f"Model {name} v{version} promoted to {stage}")

    def get_latest_model(self, name: str, stage: str = "Production") -> Optional[str]:
        """Get the latest model URI for a given stage."""
        try:
            latest = self.client.get_latest_versions(name, stages=[stage])
            if latest:
                return f"models:/{name}/{stage}"
            return None
        except Exception as e:
            logger.warning(f"Could not find model {name} in stage {stage}: {e}")
            return None

    def load_model(self, model_uri: str):
        """Load a model from MLflow."""
        return mlflow.pyfunc.load_model(model_uri)

    def get_run_metrics(self, run_id: str) -> Dict[str, float]:
        """Get metrics from a specific run."""
        run = self.client.get_run(run_id)
        return run.data.metrics

    def search_runs(self, filter_string: str = "", max_results: int = 100) -> List[mlflow.entities.Run]:
        """Search for runs matching criteria."""
        experiment = mlflow.get_experiment_by_name(self.config.experiment_name)
        if experiment is None:
            return []

        return self.client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=filter_string,
            max_results=max_results
        )

    def get_best_run(self, metric: str = "roi", maximize: bool = True) -> Optional[mlflow.entities.Run]:
        """Get the best run based on a metric."""
        order = "DESC" if maximize else "ASC"
        runs = self.search_runs(
            filter_string=f"metrics.{metric} IS NOT NULL",
            max_results=1
        )

        if runs:
            runs.sort(key=lambda r: r.data.metrics.get(metric, 0), reverse=maximize)
            return runs[0]
        return None


_mlflow_manager: Optional[MLflowManager] = None


def get_mlflow_manager(config: Optional[MLflowConfig] = None) -> MLflowManager:
    """Get or create the global MLflow manager."""
    global _mlflow_manager
    if _mlflow_manager is None:
        _mlflow_manager = MLflowManager(config)
    return _mlflow_manager


def init_mlflow(tracking_uri: Optional[str] = None, experiment_name: str = "bettip"):
    """Initialize MLflow with optional custom settings."""
    config = MLflowConfig(
        tracking_uri=tracking_uri or "sqlite:///mlflow.db",
        experiment_name=experiment_name
    )
    return get_mlflow_manager(config)
