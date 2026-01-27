"""
Ensemble Disagreement Model for High-Confidence Betting.

Implements a conservative betting strategy that only bets when:
1. Multiple models AGREE on the prediction (low disagreement)
2. The models collectively disagree with the MARKET (positive edge)

This approach filters out uncertain predictions where models
disagree with each other, focusing on high-conviction opportunities.

Research shows ensemble agreement is a strong signal of prediction quality.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

logger = logging.getLogger(__name__)


class DisagreementEnsemble(BaseEstimator):
    """
    Ensemble model that only recommends bets when models agree against market.

    Betting criteria:
    1. Model agreement: Standard deviation of predictions < threshold
       (models agree with each other)
    2. Market edge: Average model prediction - market probability > edge threshold
       (models disagree with market)
    3. Minimum probability: All models predict above minimum threshold

    This is a conservative strategy prioritizing precision over recall.
    """

    def __init__(
        self,
        models: Optional[List[Any]] = None,
        model_names: Optional[List[str]] = None,
        agreement_threshold: float = 0.10,
        min_edge_vs_market: float = 0.03,
        min_probability: float = 0.50,
        max_probability: float = 0.85,
        agreement_method: str = 'std',  # 'std', 'range', or 'iqr'
    ):
        """
        Initialize disagreement ensemble.

        Args:
            models: List of fitted models (must have predict_proba method)
            model_names: Optional names for models (for logging)
            agreement_threshold: Max std/range for models to be considered "agreeing"
            min_edge_vs_market: Minimum edge required vs market probability
            min_probability: Minimum average model probability
            max_probability: Maximum probability (avoid extreme predictions)
            agreement_method: How to measure disagreement ('std', 'range', 'iqr')
        """
        self.models = models or []
        self.model_names = model_names or [f"model_{i}" for i in range(len(self.models))]
        self.agreement_threshold = agreement_threshold
        self.min_edge_vs_market = min_edge_vs_market
        self.min_probability = min_probability
        self.max_probability = max_probability
        self.agreement_method = agreement_method

        self._is_fitted = len(self.models) > 0

    def add_model(self, model: Any, name: Optional[str] = None) -> None:
        """Add a fitted model to the ensemble."""
        self.models.append(model)
        self.model_names.append(name or f"model_{len(self.models)-1}")
        self._is_fitted = True

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Get ensemble probability predictions.

        Returns the AVERAGE probability across all models.

        Args:
            X: Feature matrix

        Returns:
            Array of average probabilities (N,)
        """
        if not self._is_fitted:
            raise ValueError("No models in ensemble. Call add_model() first.")

        if isinstance(X, pd.DataFrame):
            X = X.values

        all_probs = self._get_all_model_probas(X)
        return np.mean(all_probs, axis=0)

    def predict_with_disagreement(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        market_probs: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Predict with disagreement analysis.

        Args:
            X: Feature matrix
            market_probs: Market implied probabilities (1 / odds)

        Returns:
            Dict with:
            - 'avg_prob': Average model probability
            - 'disagreement': Model disagreement measure
            - 'edge_vs_market': Edge over market probability
            - 'models_agree': Boolean - models agree with each other
            - 'beat_market': Boolean - models beat market
            - 'bet_signal': Boolean - recommended bet (agree AND beat market)
            - 'individual_probs': Dict of individual model probabilities
        """
        if not self._is_fitted:
            raise ValueError("No models in ensemble. Call add_model() first.")

        if isinstance(X, pd.DataFrame):
            X = X.values

        n_samples = X.shape[0]

        # Get all model predictions
        all_probs = self._get_all_model_probas(X)  # (n_models, n_samples)

        # Calculate agreement metrics
        avg_prob = np.mean(all_probs, axis=0)
        disagreement = self._calculate_disagreement(all_probs)

        # Calculate edge vs market
        edge_vs_market = avg_prob - market_probs

        # Determine agreement status
        models_agree = disagreement < self.agreement_threshold

        # Determine market beat status
        beat_market = edge_vs_market >= self.min_edge_vs_market

        # Probability within acceptable range
        prob_in_range = (avg_prob >= self.min_probability) & (avg_prob <= self.max_probability)

        # Final bet signal: all conditions must be met
        bet_signal = models_agree & beat_market & prob_in_range

        # Individual model probabilities
        individual_probs = {
            name: all_probs[i]
            for i, name in enumerate(self.model_names)
        }

        return {
            'avg_prob': avg_prob,
            'disagreement': disagreement,
            'edge_vs_market': edge_vs_market,
            'models_agree': models_agree,
            'beat_market': beat_market,
            'bet_signal': bet_signal,
            'individual_probs': individual_probs,
            'n_models_above_market': np.sum(all_probs > market_probs, axis=0),
        }

    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        market_probs: np.ndarray,
    ) -> np.ndarray:
        """
        Generate binary bet recommendations.

        Args:
            X: Feature matrix
            market_probs: Market implied probabilities

        Returns:
            Binary array of bet recommendations
        """
        result = self.predict_with_disagreement(X, market_probs)
        return result['bet_signal'].astype(int)

    def _get_all_model_probas(self, X: np.ndarray) -> np.ndarray:
        """Get probability predictions from all models."""
        all_probs = []

        for model in self.models:
            try:
                if hasattr(model, 'predict_proba'):
                    probs = model.predict_proba(X)
                    # Handle both binary and multiclass
                    if probs.ndim > 1:
                        probs = probs[:, 1]  # Positive class probability
                else:
                    probs = model.predict(X)

                all_probs.append(probs)

            except Exception as e:
                logger.warning(f"Model prediction failed: {e}")
                continue

        if not all_probs:
            raise ValueError("No successful predictions from any model")

        return np.array(all_probs)  # (n_models, n_samples)

    def _calculate_disagreement(self, all_probs: np.ndarray) -> np.ndarray:
        """
        Calculate disagreement between models.

        Args:
            all_probs: (n_models, n_samples) array of probabilities

        Returns:
            (n_samples,) array of disagreement scores
        """
        if self.agreement_method == 'std':
            # Standard deviation: lower = more agreement
            return np.std(all_probs, axis=0)

        elif self.agreement_method == 'range':
            # Range between max and min: lower = more agreement
            return np.max(all_probs, axis=0) - np.min(all_probs, axis=0)

        elif self.agreement_method == 'iqr':
            # Interquartile range: robust to outliers
            q75 = np.percentile(all_probs, 75, axis=0)
            q25 = np.percentile(all_probs, 25, axis=0)
            return q75 - q25

        else:
            return np.std(all_probs, axis=0)

    def get_agreement_analysis(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        market_probs: np.ndarray,
    ) -> pd.DataFrame:
        """
        Get detailed analysis of model agreement and edge.

        Useful for understanding where models agree/disagree
        and how they compare to market.

        Returns:
            DataFrame with columns for each model's prediction,
            disagreement metrics, and market comparison.
        """
        result = self.predict_with_disagreement(X, market_probs)

        df = pd.DataFrame({
            'avg_prob': result['avg_prob'],
            'market_prob': market_probs,
            'edge': result['edge_vs_market'],
            'disagreement': result['disagreement'],
            'models_agree': result['models_agree'],
            'beat_market': result['beat_market'],
            'bet_signal': result['bet_signal'],
            'n_models_above_market': result['n_models_above_market'],
        })

        # Add individual model predictions
        for name, probs in result['individual_probs'].items():
            df[f'prob_{name}'] = probs

        return df

    def get_confidence_score(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        market_probs: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate confidence score for each prediction.

        Confidence is based on:
        1. How much models agree (low disagreement = high confidence)
        2. How much edge vs market (higher edge = higher confidence)
        3. How many models beat the market (consensus)

        Returns:
            Array of confidence scores (0-1)
        """
        result = self.predict_with_disagreement(X, market_probs)

        # Normalize components
        n_models = len(self.models)

        # Agreement component (0-1, higher = more agreement)
        agreement_score = 1 - np.clip(result['disagreement'] / 0.20, 0, 1)

        # Edge component (0-1, higher = more edge)
        edge_score = np.clip(result['edge_vs_market'] / 0.10, 0, 1)

        # Consensus component (0-1, higher = more models beat market)
        consensus_score = result['n_models_above_market'] / n_models

        # Combined confidence (geometric mean)
        confidence = (agreement_score * edge_score * consensus_score) ** (1/3)

        # Only non-zero for actual bet signals
        confidence = confidence * result['bet_signal']

        return confidence


class AdaptiveDisagreementEnsemble(DisagreementEnsemble):
    """
    Disagreement ensemble with adaptive thresholds based on market conditions.

    Adjusts agreement and edge thresholds based on:
    - Market volatility (wider thresholds in volatile markets)
    - Historical performance (tighten thresholds if losing)
    - Sample size (wider thresholds with more models)
    """

    def __init__(
        self,
        base_agreement_threshold: float = 0.10,
        base_edge_threshold: float = 0.03,
        volatility_adjustment: float = 0.5,
        **kwargs,
    ):
        """
        Initialize adaptive ensemble.

        Args:
            base_agreement_threshold: Base threshold before adjustment
            base_edge_threshold: Base edge threshold before adjustment
            volatility_adjustment: How much to adjust for volatility
        """
        super().__init__(
            agreement_threshold=base_agreement_threshold,
            min_edge_vs_market=base_edge_threshold,
            **kwargs,
        )
        self.base_agreement_threshold = base_agreement_threshold
        self.base_edge_threshold = base_edge_threshold
        self.volatility_adjustment = volatility_adjustment

        # Track performance for adaptive adjustment
        self._bet_history = []
        self._win_history = []

    def adapt_thresholds(
        self,
        market_volatility: float = 0.0,
        recent_win_rate: Optional[float] = None,
    ) -> None:
        """
        Adapt thresholds based on conditions.

        Args:
            market_volatility: Current market volatility (0-1)
            recent_win_rate: Recent win rate of bets (0-1)
        """
        # Adjust for volatility
        volatility_mult = 1 + (market_volatility * self.volatility_adjustment)

        # Adjust for recent performance (tighten if losing)
        if recent_win_rate is not None and len(self._bet_history) >= 20:
            if recent_win_rate < 0.45:
                # Losing: tighten thresholds
                performance_mult = 0.8
            elif recent_win_rate > 0.55:
                # Winning: can slightly loosen
                performance_mult = 1.1
            else:
                performance_mult = 1.0
        else:
            performance_mult = 1.0

        # Apply adjustments
        self.agreement_threshold = self.base_agreement_threshold * volatility_mult * performance_mult
        self.min_edge_vs_market = self.base_edge_threshold / performance_mult

        logger.info(
            f"Adapted thresholds: agreement={self.agreement_threshold:.3f}, "
            f"edge={self.min_edge_vs_market:.3f}"
        )

    def record_result(self, bet: bool, won: bool) -> None:
        """Record bet result for performance tracking."""
        if bet:
            self._bet_history.append(1)
            self._win_history.append(1 if won else 0)

    def get_recent_win_rate(self, window: int = 50) -> float:
        """Get win rate over recent bets."""
        if len(self._win_history) == 0:
            return 0.5  # Default

        recent = self._win_history[-window:]
        return np.mean(recent)


def create_disagreement_ensemble(
    models: List[Any],
    model_names: Optional[List[str]] = None,
    strategy: str = 'conservative',
) -> DisagreementEnsemble:
    """
    Factory function to create disagreement ensemble with preset configurations.

    Args:
        models: List of fitted models
        model_names: Optional model names
        strategy: Preset configuration
            - 'conservative': Tight thresholds, high precision
            - 'balanced': Moderate thresholds
            - 'aggressive': Looser thresholds, higher recall

    Returns:
        Configured DisagreementEnsemble
    """
    configs = {
        'conservative': {
            'agreement_threshold': 0.08,
            'min_edge_vs_market': 0.05,
            'min_probability': 0.55,
            'max_probability': 0.80,
        },
        'balanced': {
            'agreement_threshold': 0.12,
            'min_edge_vs_market': 0.03,
            'min_probability': 0.50,
            'max_probability': 0.85,
        },
        'aggressive': {
            'agreement_threshold': 0.15,
            'min_edge_vs_market': 0.02,
            'min_probability': 0.45,
            'max_probability': 0.90,
        },
    }

    config = configs.get(strategy, configs['balanced'])

    return DisagreementEnsemble(
        models=models,
        model_names=model_names,
        **config,
    )
