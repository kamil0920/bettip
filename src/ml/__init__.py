"""
Machine Learning module for BetTip.

This module provides:
- Model factory for creating ML models (RF, XGBoost, LightGBM, CatBoost)
- Custom metrics for sports prediction evaluation
- MLflow experiment tracking integration
- Optuna hyperparameter tuning
"""
from src.ml.models import ModelFactory, ModelType
from src.ml.metrics import SportsMetrics
from src.ml.experiment import Experiment, ExperimentConfig

__all__ = [
    "ModelFactory",
    "ModelType",
    "SportsMetrics",
    "Experiment",
    "ExperimentConfig",
]
