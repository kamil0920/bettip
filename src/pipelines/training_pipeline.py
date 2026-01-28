"""
Training pipeline for ML model training with MLflow integration.

This pipeline orchestrates the training process:
1. Load features from data/03-features/
2. Split data into train/test sets (time-based for sports data)
3. Train model with MLflow tracking
4. Evaluate model with sports-specific metrics
5. Save model and artifacts
"""
import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config_loader import Config
from src.ml.models import ModelFactory, get_feature_importance
from src.ml.metrics import SportsMetrics, PredictionMetrics
from src.ml.experiment import Experiment, ExperimentConfig

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """
    Pipeline for training ML callibration with MLflow experiment tracking.

    Supports multiple model types and provides comprehensive evaluation
    metrics tailored for sports prediction.
    """

    NON_FEATURE_COLUMNS = [
        "fixture_id", "date", "home_team_id", "home_team_name",
        "away_team_id", "away_team_name", "round",
    ]

    TARGET_COLUMNS = [
        "home_win", "draw", "away_win", "match_result",
        "total_goals", "goal_difference"
    ]

    def __init__(self, config: Config):
        """
        Initialize the training pipeline.

        Args:
            config: Configuration object loaded from YAML
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = None
        self.metrics: Optional[PredictionMetrics] = None
        self.feature_columns: List[str] = []

    def run(
        self,
        features_file: str = "features.csv",
        target_column: str = "home_win",
        experiment_name: Optional[str] = None,
        run_name: Optional[str] = None,
        use_mlflow: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute the training pipeline.

        Args:
            features_file: Name of features CSV file in data/03-features/
            target_column: Column to predict (home_win, draw, match_result, etc.)
            experiment_name: MLflow experiment name (default: bettip-{target})
            run_name: MLflow run name (auto-generated if None)
            use_mlflow: Whether to use MLflow tracking

        Returns:
            Dictionary with training results:
            - 'model': Trained model object
            - 'metrics': PredictionMetrics object
            - 'feature_importance': Feature importance scores
            - 'run_id': MLflow run ID (if use_mlflow=True)
        """
        self.logger.info("=" * 60)
        self.logger.info("TRAINING PIPELINE")
        self.logger.info("=" * 60)

        self.logger.info("[1/5] Loading features...")
        features_df = self._load_features(features_file)

        self.logger.info("[2/5] Preparing data...")
        X_train, X_test, y_train, y_test = self._prepare_data(
            features_df, target_column
        )

        self.logger.info("[3/5] Training model...")
        if use_mlflow:
            result = self._train_with_mlflow(
                X_train, X_test, y_train, y_test,
                target_column, experiment_name, run_name
            )
        else:
            result = self._train_simple(X_train, X_test, y_train, y_test)

        self._log_summary(result)

        return result

    def _load_features(self, features_file: str) -> pd.DataFrame:
        """Load features file (Parquet preferred, CSV fallback)."""
        features_path = self.config.get_features_dir() / features_file

        from src.utils.data_io import load_features
        df = load_features(features_path)

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])

        self.logger.info(f"Loaded features: {df.shape[0]} rows, {df.shape[1]} columns")

        return df

    def _prepare_data(
        self,
        features_df: pd.DataFrame,
        target_column: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare data for training with time-based split.

        For sports prediction, we use chronological split to prevent
        data leakage (future data predicting past).

        Args:
            features_df: DataFrame with all features
            target_column: Target column name

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        if target_column not in features_df.columns:
            available = [c for c in features_df.columns if c in self.TARGET_COLUMNS]
            raise ValueError(
                f"Target column '{target_column}' not found. "
                f"Available targets: {available}"
            )

        exclude_cols = set(self.NON_FEATURE_COLUMNS + self.TARGET_COLUMNS)
        self.feature_columns = [
            col for col in features_df.columns
            if col not in exclude_cols
        ]

        self.logger.info(f"Using {len(self.feature_columns)} features")
        self.logger.info(f"Target: {target_column}")

        X = features_df[self.feature_columns].copy()
        y = features_df[target_column].copy()

        if X.isnull().any().any():
            null_counts = X.isnull().sum()
            null_cols = null_counts[null_counts > 0]
            self.logger.warning(f"Found null values in columns: {null_cols.to_dict()}")
            X = X.fillna(0)  # TODO: improve

        if "date" in features_df.columns:
            self.logger.info("Using time-based split (chronological)")
            features_df_sorted = features_df.sort_values("date")
            split_idx = int(len(features_df_sorted) * (1 - self.config.model.test_size))

            train_indices = features_df_sorted.index[:split_idx]
            test_indices = features_df_sorted.index[split_idx:]

            X_train = X.loc[train_indices]
            X_test = X.loc[test_indices]
            y_train = y.loc[train_indices]
            y_test = y.loc[test_indices]

            self.logger.info(
                f"Train: {features_df_sorted.iloc[0]['date'].date()} to "
                f"{features_df_sorted.iloc[split_idx-1]['date'].date()}"
            )
            self.logger.info(
                f"Test: {features_df_sorted.iloc[split_idx]['date'].date()} to "
                f"{features_df_sorted.iloc[-1]['date'].date()}"
            )
        else:
            self.logger.info("Using random split (no date column)")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.config.model.test_size,
                random_state=self.config.model.random_state
            )

        self.logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

        self._log_class_distribution(y_train, y_test)

        return X_train, X_test, y_train, y_test

    def _log_class_distribution(self, y_train: pd.Series, y_test: pd.Series) -> None:
        """Log class distribution for train and test sets."""
        self.logger.info("Class distribution:")
        train_dist = y_train.value_counts(normalize=True)
        test_dist = y_test.value_counts(normalize=True)

        for cls in sorted(train_dist.index):
            self.logger.info(
                f"  Class {cls}: train={train_dist.get(cls, 0):.1%}, "
                f"test={test_dist.get(cls, 0):.1%}"
            )

    def _train_with_mlflow(
            self,
            X_train: pd.DataFrame,
            X_test: pd.DataFrame,
            y_train: pd.Series,
            y_test: pd.Series,
            target_column: str,
            experiment_name: Optional[str],
            run_name: Optional[str],
    ) -> Dict[str, Any]:
        """Train model with MLflow tracking."""
        params_to_log = getattr(self.config.model, "params", {})

        if not isinstance(params_to_log, dict):
            params_to_log = {}

        exp_config = ExperimentConfig(
            experiment_name=experiment_name or f"bettip-{target_column}",
            run_name=run_name,
            model_type=self.config.model.type,
            model_params=params_to_log,
            target_column=target_column,
            test_size=self.config.model.test_size,
            random_state=self.config.model.random_state,
            tags={
                "league": self.config.league,
                "seasons": str(self.config.seasons),
            }
        )

        experiment = Experiment(exp_config)
        result = experiment.run(X_train, X_test, y_train, y_test)

        self.model = result["model"]
        self.metrics = result["metrics"]

        return result

    def _train_simple(
            self,
            X_train: pd.DataFrame,
            X_test: pd.DataFrame,
            y_train: pd.Series,
            y_test: pd.Series,
    ) -> Dict[str, Any]:
        """Train model without MLflow (for quick testing)."""

        self.model = ModelFactory.create(
            model_type=self.config.model.type,
            params=self.config.model.params,
            random_state=self.config.model.random_state
        )

        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        y_proba = None
        if hasattr(self.model, "predict_proba"):
            y_proba = self.model.predict_proba(X_test)

        self.metrics = SportsMetrics.calculate_all(
            y_test.values, y_pred, y_proba
        )

        feature_importance = get_feature_importance(
            self.model, self.feature_columns
        )

        return {
            "model": self.model,
            "metrics": self.metrics,
            "feature_importance": feature_importance,
        }

    def _log_summary(self, result: Dict[str, Any]) -> None:
        """Log training summary."""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("TRAINING PIPELINE COMPLETED")
        self.logger.info("=" * 60)

        if self.metrics:
            self.logger.info(f"Accuracy:  {self.metrics.accuracy:.4f}")
            self.logger.info(f"Precision: {self.metrics.precision:.4f}")
            self.logger.info(f"Recall:    {self.metrics.recall:.4f}")
            self.logger.info(f"F1 Score:  {self.metrics.f1:.4f}")

            if self.metrics.roc_auc:
                self.logger.info(f"ROC AUC:   {self.metrics.roc_auc:.4f}")

        if "run_id" in result:
            self.logger.info(f"\nMLflow Run ID: {result['run_id']}")
            self.logger.info("View results: mlflow ui")

        self.logger.info("=" * 60)


def run_quick_experiment(
    config_path: str = "config/local.yaml",
    target: str = "home_win",
    model_type: str = "random_forest"
) -> Dict[str, Any]:
    """
    Quick helper to run a single experiment.

    Args:
        config_path: Path to config file
        target: Target column
        model_type: Model type

    Returns:
        Experiment results
    """
    from src.config_loader import load_config

    config = load_config(config_path)
    config.model.type = model_type

    pipeline = TrainingPipeline(config)
    return pipeline.run(target_column=target)
