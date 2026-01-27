"""
Two-Stage Model Architecture for Betting Predictions.

Stage 1: Predict outcome probability (classification)
Stage 2: Predict bet profitability given Stage 1 prediction + odds (regression)

The two-stage approach separates "what will happen" from "is this a good bet",
allowing the model to learn betting-specific patterns beyond pure prediction.

Benefits:
1. Stage 2 can learn market inefficiencies that Stage 1 misses
2. Stage 2 can incorporate odds-based features that would cause leakage in Stage 1
3. Better calibration for betting decisions vs pure classification
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class TwoStageModel(BaseEstimator):
    """
    Two-stage model for betting predictions.

    Stage 1: Predicts outcome probability using features
    Stage 2: Predicts bet profitability using Stage 1 output + odds features

    The model outputs both:
    - outcome_prob: Probability of the outcome occurring
    - profit_prob: Probability that betting on this outcome is profitable

    A bet is recommended when both probabilities are high.
    """

    def __init__(
        self,
        stage1_model: Optional[Any] = None,
        stage2_model: Optional[Any] = None,
        calibrate_stage1: bool = True,
        calibration_method: str = 'sigmoid',
        min_edge_threshold: float = 0.02,
    ):
        """
        Initialize two-stage model.

        Args:
            stage1_model: Model for outcome prediction (default: LogisticRegression)
            stage2_model: Model for profitability prediction (default: LogisticRegression)
            calibrate_stage1: Whether to calibrate Stage 1 probabilities
            calibration_method: 'sigmoid' (Platt) or 'isotonic'
            min_edge_threshold: Minimum edge to consider bet potentially profitable
        """
        self.stage1_model = stage1_model
        self.stage2_model = stage2_model
        self.calibrate_stage1 = calibrate_stage1
        self.calibration_method = calibration_method
        self.min_edge_threshold = min_edge_threshold

        # Will be set during fit
        self._stage1_fitted = None
        self._stage2_fitted = None
        self._stage1_scaler = None
        self._stage2_scaler = None
        self._feature_names = None
        self._is_fitted = False

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y_outcome: np.ndarray,
        odds: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> 'TwoStageModel':
        """
        Fit the two-stage model.

        Args:
            X: Feature matrix (N x D)
            y_outcome: Binary outcome labels (N,) - did the outcome occur?
            odds: Decimal odds for the outcome (N,)
            feature_names: Optional feature names for interpretability

        Returns:
            self (fitted model)
        """
        # Store feature names
        if isinstance(X, pd.DataFrame):
            self._feature_names = list(X.columns)
            X = X.values
        elif feature_names is not None:
            self._feature_names = feature_names

        n_samples = X.shape[0]
        logger.info(f"Fitting two-stage model on {n_samples} samples")

        # Initialize models if not provided
        if self.stage1_model is None:
            self.stage1_model = LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=42,
            )

        if self.stage2_model is None:
            self.stage2_model = LogisticRegression(
                C=0.5,
                max_iter=1000,
                random_state=42,
            )

        # === Stage 1: Fit outcome predictor ===
        self._stage1_scaler = StandardScaler()
        X_scaled = self._stage1_scaler.fit_transform(X)

        self.stage1_model.fit(X_scaled, y_outcome)
        logger.info(f"Stage 1 fitted, train accuracy: {self.stage1_model.score(X_scaled, y_outcome):.3f}")

        # Calibrate Stage 1 if requested
        if self.calibrate_stage1:
            try:
                calibrated = CalibratedClassifierCV(
                    self.stage1_model,
                    cv='prefit',
                    method=self.calibration_method,
                )
                calibrated.fit(X_scaled, y_outcome)
                self._stage1_fitted = calibrated
                logger.info("Stage 1 calibrated")
            except Exception as e:
                logger.warning(f"Calibration failed: {e}, using uncalibrated model")
                self._stage1_fitted = self.stage1_model
        else:
            self._stage1_fitted = self.stage1_model

        # Get Stage 1 predictions
        stage1_proba = self._get_stage1_proba(X_scaled)

        # === Stage 2: Fit profitability predictor ===
        # Create profit target: was this bet profitable?
        y_profit = self._calculate_profit_target(y_outcome, odds)

        # Create Stage 2 features: original features + Stage 1 proba + odds features
        X_stage2 = self._create_stage2_features(X, stage1_proba, odds)

        self._stage2_scaler = StandardScaler()
        X_stage2_scaled = self._stage2_scaler.fit_transform(X_stage2)

        self.stage2_model.fit(X_stage2_scaled, y_profit)
        logger.info(f"Stage 2 fitted, train accuracy: {self.stage2_model.score(X_stage2_scaled, y_profit):.3f}")

        self._stage2_fitted = self.stage2_model
        self._is_fitted = True

        return self

    def predict_proba(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        odds: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Predict outcome and profitability probabilities.

        Args:
            X: Feature matrix
            odds: Decimal odds for the outcome

        Returns:
            Dict with keys:
            - 'outcome_prob': Probability of outcome occurring
            - 'profit_prob': Probability of bet being profitable
            - 'combined_score': Combined betting score
            - 'edge': Estimated edge vs implied probability
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if isinstance(X, pd.DataFrame):
            X = X.values

        # Stage 1: Outcome prediction
        X_scaled = self._stage1_scaler.transform(X)
        stage1_proba = self._get_stage1_proba(X_scaled)

        # Stage 2: Profitability prediction
        X_stage2 = self._create_stage2_features(X, stage1_proba, odds)
        X_stage2_scaled = self._stage2_scaler.transform(X_stage2)

        if hasattr(self._stage2_fitted, 'predict_proba'):
            stage2_proba = self._stage2_fitted.predict_proba(X_stage2_scaled)[:, 1]
        else:
            stage2_proba = self._stage2_fitted.predict(X_stage2_scaled)

        # Calculate edge
        implied_prob = 1 / odds
        edge = stage1_proba - implied_prob

        # Combined score: geometric mean of outcome prob and profit prob
        # Only count as valuable if both are high
        combined_score = np.sqrt(stage1_proba * stage2_proba)

        return {
            'outcome_prob': stage1_proba,
            'profit_prob': stage2_proba,
            'combined_score': combined_score,
            'edge': edge,
        }

    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        odds: np.ndarray,
        outcome_threshold: float = 0.5,
        profit_threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Generate bet recommendations.

        Args:
            X: Feature matrix
            odds: Decimal odds
            outcome_threshold: Minimum outcome probability
            profit_threshold: Minimum profit probability

        Returns:
            Binary array indicating recommended bets
        """
        predictions = self.predict_proba(X, odds)

        # Bet when both conditions are met
        bet_mask = (
            (predictions['outcome_prob'] >= outcome_threshold) &
            (predictions['profit_prob'] >= profit_threshold) &
            (predictions['edge'] >= self.min_edge_threshold)
        )

        return bet_mask.astype(int)

    def _get_stage1_proba(self, X_scaled: np.ndarray) -> np.ndarray:
        """Get Stage 1 probability predictions."""
        if hasattr(self._stage1_fitted, 'predict_proba'):
            return self._stage1_fitted.predict_proba(X_scaled)[:, 1]
        else:
            return self._stage1_fitted.predict(X_scaled)

    def _calculate_profit_target(
        self,
        y_outcome: np.ndarray,
        odds: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate binary profit target.

        A bet is profitable if:
        - Outcome occurred AND bet would have returned profit (always true if outcome occurs)
        - For more nuanced: we could consider if the implied probability was beaten

        For simplicity: profit = outcome occurred (since we're betting on that outcome)
        """
        # Binary: was the bet profitable?
        # If outcome occurred, profit = odds - 1 (positive)
        # If outcome didn't occur, profit = -1 (loss)
        profit = np.where(y_outcome == 1, odds - 1, -1)

        # Convert to binary: was it profitable?
        return (profit > 0).astype(int)

    def _create_stage2_features(
        self,
        X: np.ndarray,
        stage1_proba: np.ndarray,
        odds: np.ndarray,
    ) -> np.ndarray:
        """
        Create features for Stage 2 model.

        Stage 2 sees:
        - Original features (subset or all)
        - Stage 1 probability prediction
        - Odds-derived features
        """
        n_samples = X.shape[0]

        # Odds-derived features (safe to use in Stage 2)
        implied_prob = 1 / odds
        edge = stage1_proba - implied_prob
        value_ratio = stage1_proba / (implied_prob + 1e-10)

        # Kelly criterion value
        kelly = (stage1_proba * odds - 1) / (odds - 1 + 1e-10)
        kelly = np.clip(kelly, 0, 1)

        # Expected value
        expected_value = stage1_proba * (odds - 1) - (1 - stage1_proba)

        # Stack features
        stage2_features = np.column_stack([
            X,  # Original features
            stage1_proba.reshape(-1, 1),  # Stage 1 prediction
            odds.reshape(-1, 1),  # Raw odds
            implied_prob.reshape(-1, 1),  # Implied probability
            edge.reshape(-1, 1),  # Edge vs market
            value_ratio.reshape(-1, 1),  # Value ratio
            kelly.reshape(-1, 1),  # Kelly value
            expected_value.reshape(-1, 1),  # Expected value
        ])

        return stage2_features

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from both stages."""
        importance = {}

        # Stage 1 importance
        if hasattr(self._stage1_fitted, 'coef_'):
            base_model = self._stage1_fitted
            if hasattr(self._stage1_fitted, 'calibrated_classifiers_'):
                base_model = self._stage1_fitted.calibrated_classifiers_[0].estimator

            if hasattr(base_model, 'coef_') and self._feature_names:
                coef = np.abs(base_model.coef_[0])
                for i, name in enumerate(self._feature_names):
                    if i < len(coef):
                        importance[f'stage1_{name}'] = float(coef[i])

        # Stage 2 importance for added features
        if hasattr(self._stage2_fitted, 'coef_'):
            coef = np.abs(self._stage2_fitted.coef_[0])
            stage2_added = [
                'stage1_prob', 'odds', 'implied_prob', 'edge',
                'value_ratio', 'kelly', 'expected_value'
            ]
            start_idx = len(self._feature_names) if self._feature_names else 0

            for i, name in enumerate(stage2_added):
                idx = start_idx + i
                if idx < len(coef):
                    importance[f'stage2_{name}'] = float(coef[idx])

        return importance


class TwoStageCatBoost(TwoStageModel):
    """Two-stage model using CatBoost for both stages."""

    def __init__(
        self,
        stage1_params: Optional[Dict] = None,
        stage2_params: Optional[Dict] = None,
        **kwargs,
    ):
        """
        Initialize with CatBoost models.

        Args:
            stage1_params: CatBoost parameters for Stage 1
            stage2_params: CatBoost parameters for Stage 2
        """
        try:
            from catboost import CatBoostClassifier

            default_params = {
                'iterations': 500,
                'depth': 6,
                'learning_rate': 0.05,
                'l2_leaf_reg': 5,
                'random_seed': 42,
                'verbose': False,
            }

            stage1_model = CatBoostClassifier(**(stage1_params or default_params))
            stage2_model = CatBoostClassifier(**(stage2_params or default_params))

        except ImportError:
            logger.warning("CatBoost not available, using LogisticRegression")
            stage1_model = None
            stage2_model = None

        super().__init__(
            stage1_model=stage1_model,
            stage2_model=stage2_model,
            **kwargs,
        )


class TwoStageLightGBM(TwoStageModel):
    """Two-stage model using LightGBM for both stages."""

    def __init__(
        self,
        stage1_params: Optional[Dict] = None,
        stage2_params: Optional[Dict] = None,
        **kwargs,
    ):
        """
        Initialize with LightGBM models.

        Args:
            stage1_params: LightGBM parameters for Stage 1
            stage2_params: LightGBM parameters for Stage 2
        """
        try:
            import lightgbm as lgb

            default_params = {
                'n_estimators': 500,
                'max_depth': 6,
                'learning_rate': 0.05,
                'reg_alpha': 2,
                'reg_lambda': 3,
                'random_state': 42,
                'verbose': -1,
            }

            stage1_model = lgb.LGBMClassifier(**(stage1_params or default_params))
            stage2_model = lgb.LGBMClassifier(**(stage2_params or default_params))

        except ImportError:
            logger.warning("LightGBM not available, using LogisticRegression")
            stage1_model = None
            stage2_model = None

        super().__init__(
            stage1_model=stage1_model,
            stage2_model=stage2_model,
            **kwargs,
        )


def create_two_stage_model(
    model_type: str = 'logistic',
    **kwargs,
) -> TwoStageModel:
    """
    Factory function to create two-stage models.

    Args:
        model_type: 'logistic', 'catboost', or 'lightgbm'
        **kwargs: Additional arguments passed to model

    Returns:
        TwoStageModel instance
    """
    if model_type == 'catboost':
        return TwoStageCatBoost(**kwargs)
    elif model_type == 'lightgbm':
        return TwoStageLightGBM(**kwargs)
    else:
        return TwoStageModel(**kwargs)
