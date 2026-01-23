"""
Machine Learning module for BetTip.

This module provides:
- Model factory for creating ML callibration (RF, XGBoost, LightGBM, CatBoost)
- Custom metrics for sports prediction evaluation
- MLflow experiment tracking integration
- Optuna hyperparameter tuning
- Model loading for inference
- Feature lookup for real-time predictions
"""
from src.ml.models import ModelFactory, ModelType
from src.ml.metrics import SportsMetrics
from src.ml.experiment import Experiment, ExperimentConfig
from src.ml.tuning import HyperparameterTuner, tune_all_models
from src.ml.model_loader import ModelLoader, get_model_loader
from src.ml.feature_lookup import FeatureLookup, get_feature_lookup

__all__ = [
    "ModelFactory",
    "ModelType",
    "SportsMetrics",
    "Experiment",
    "ExperimentConfig",
    "HyperparameterTuner",
    "tune_all_models",
    "ModelLoader",
    "get_model_loader",
    "FeatureLookup",
    "get_feature_lookup",
]
