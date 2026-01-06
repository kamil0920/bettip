"""
Inference pipeline for making predictions.

This pipeline orchestrates the prediction process:
1. Load trained model
2. Load input data
3. Preprocess input
4. Make predictions
5. Save predictions to data/04-predictions/
"""
import logging
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.config_loader import Config

logger = logging.getLogger(__name__)


class InferencePipeline:
    """
    Pipeline for making predictions with trained models.

    """

    def __init__(self, config: Config):
        """
        Initialize the inference pipeline.

        Args:
            config: Configuration object loaded from YAML
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = None

    def run(
            self,
            input_file: str,
            model_path: Optional[str] = None,
            output_file: str = "predictions.csv"
    ) -> pd.DataFrame:
        """
        Execute the inference pipeline.

        Args:
            input_file: Path to input data (CSV with features)
            model_path: Path to trained model file (optional, uses default if None)
            output_file: Name of output predictions file

        Returns:
            DataFrame with predictions

        Raises:
            FileNotFoundError: If model or input file doesn't exist
            Exception: If inference fails
        """
        self.logger.info("=" * 60)
        self.logger.info("INFERENCE PIPELINE")
        self.logger.info("=" * 60)

        self.logger.info("[1/4] Loading model...")
        self.model = self._load_model(model_path)

        self.logger.info("[2/4] Loading input data...")
        input_df = self._load_input(input_file)

        self.logger.info("[3/4] Making predictions...")
        predictions = self._predict(input_df)

        self.logger.info("[4/4] Saving predictions...")
        output_path = self._save_predictions(predictions, output_file)

        self._log_summary(predictions, output_path)

        return predictions

    def _load_model(self, model_path: Optional[str] = None) -> Any:
        """
        Load trained model.

        Args:
            model_path: Path to model file (uses default if None)

        Returns:
            Loaded model object
        """
        # TODO: Implement model loading
        # Example:
        # import joblib
        # if model_path is None:
        #     model_path = self.config.get_predictions_dir() / "model.joblib"
        # model = joblib.load(model_path)
        # return model

        raise NotImplementedError("TODO: Implement _load_model method")

    def _load_input(self, input_file: str) -> pd.DataFrame:
        """
        Load input data for prediction.

        Args:
            input_file: Path to input CSV file

        Returns:
            DataFrame with input features
        """
        input_path = Path(input_file)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        df = pd.read_csv(input_path)
        self.logger.info(f"Loaded input: {df.shape[0]} rows, {df.shape[1]} columns")

        return df

    def _predict(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions on input data.

        TODO: Implement prediction logic:
        - Select features
        - Handle missing values
        - Make predictions
        - Add probabilities if available

        Args:
            input_df: DataFrame with input features

        Returns:
            DataFrame with predictions added
        """
        # TODO: Implement prediction
        # Example:
        # feature_cols = [...]  # Same features used in training
        # X = input_df[feature_cols]
        #
        # predictions = self.model.predict(X)
        # input_df['prediction'] = predictions
        #
        # # Add probabilities if available
        # if hasattr(self.model, 'predict_proba'):
        #     proba = self.model.predict_proba(X)
        #     input_df['home_win_prob'] = proba[:, 1]  # Adjust based on your classes
        #
        # return input_df

        raise NotImplementedError("TODO: Implement _predict method")

    def _save_predictions(self, predictions: pd.DataFrame, output_file: str) -> Path:
        """
        Save predictions to file.

        Args:
            predictions: DataFrame with predictions
            output_file: Output filename

        Returns:
            Path to saved file
        """
        output_dir = self.config.get_predictions_dir()
        output_path = output_dir / output_file

        predictions.to_csv(output_path, index=False)
        self.logger.info(f"Saved predictions to: {output_path}")

        return output_path

    def _log_summary(self, predictions: pd.DataFrame, output_path: Path) -> None:
        """Log inference summary."""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("INFERENCE PIPELINE COMPLETED")
        self.logger.info("=" * 60)
        self.logger.info(f"Total predictions: {len(predictions)}")
        self.logger.info(f"Output saved to: {output_path}")

        # TODO: Add prediction statistics
        # if 'prediction' in predictions.columns:
        #     self.logger.info(f"Prediction distribution:")
        #     self.logger.info(predictions['prediction'].value_counts())

        self.logger.info("=" * 60)
