"""
Betting Inference Pipeline

Uses the Strategy Pattern for different bet types.
Each betting strategy encapsulates its own:
- Feature engineering
- Bet recommendation generation

This pipeline:
1. Loads trained callibration from MLflow registry
2. Fetches upcoming fixtures
3. Generates features for new matches
4. Makes predictions using ensemble
5. Applies betting strategies to generate recommendations
6. Outputs recommended bets with confidence scores
"""

import pandas as pd
import numpy as np
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import mlflow

from src.ml.mlflow_config import get_mlflow_manager
from src.ml.betting_strategies import (
    BettingStrategy,
    StrategyConfig,
    get_strategy,
    STRATEGY_REGISTRY
)
from src.ml.bankroll_manager import BankrollManager, create_bankroll_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BetRecommendation:
    """A single bet recommendation."""
    fixture_id: str
    date: str
    home_team: str
    away_team: str
    league: str
    bet_type: str
    bet_side: str
    odds: float
    probability: float
    edge: float
    confidence: float
    expected_value: float
    kelly_fraction: float
    recommended_stake: float

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'BetRecommendation':
        """Create from dictionary."""
        return cls(**d)


@dataclass
class InferenceConfig:
    """Inference configuration."""
    strategies_path: str = "config/strategies.yaml"
    models_dir: str = "outputs/callibration"
    min_edge: float = 0.02  # Minimum 2% edge (enforced by BankrollManager)
    min_confidence: float = 0.6
    bankroll: float = 1000.0
    max_stake_fraction: float = 0.05
    enforce_daily_limits: bool = True  # Enable BankrollManager enforcement


class BettingInferencePipeline:
    """
    Generate betting recommendations from trained callibration.

    Uses Strategy pattern - each bet type is handled by a BettingStrategy class.
    """

    def __init__(self, config: InferenceConfig):
        self.config = config
        self.mlflow_manager = get_mlflow_manager()
        self.strategies_config = self._load_strategies_config()
        self.models = {}
        self.features_config = {}

        # Initialize BankrollManager for risk control enforcement
        self.bankroll_manager = create_bankroll_manager(
            bankroll=config.bankroll,
            strategies_path=config.strategies_path,
        )

    def _load_strategies_config(self) -> Dict:
        """Load betting strategies configuration."""
        with open(self.config.strategies_path) as f:
            return yaml.safe_load(f)

    def _get_strategy(self, bet_type: str) -> BettingStrategy:
        """Get strategy instance for a bet type."""
        strategy_cfg = self.strategies_config['strategies'].get(bet_type, {})

        config = StrategyConfig(
            enabled=strategy_cfg.get('enabled', False),
            approach=strategy_cfg.get('approach', 'classification'),
            odds_column=strategy_cfg.get('odds_column'),
            probability_threshold=strategy_cfg.get('probability_threshold', 0.5),
            edge_threshold=strategy_cfg.get('edge_threshold', 0.3),
            min_edge=self.config.min_edge,
            max_stake_fraction=self.config.max_stake_fraction,
            line_filter=strategy_cfg.get('line_filter', {'min': -4, 'max': -1.5}),
        )

        return get_strategy(bet_type, config)

    def load_models(self, bet_types: Optional[List[str]] = None):
        """Load trained callibration from MLflow registry."""
        if bet_types is None:
            bet_types = [
                bt for bt, cfg in self.strategies_config['strategies'].items()
                if cfg.get('enabled', False)
            ]

        for bet_type in bet_types:
            strategy_cfg = self.strategies_config['strategies'].get(bet_type, {})
            model_type = strategy_cfg.get('model_type', 'ensemble')

            logger.info(f"Loading callibration for {bet_type}")

            if model_type == 'ensemble':
                self.models[bet_type] = {}
                for mt in ['xgboost', 'lightgbm', 'catboost']:
                    model_name = f"{bet_type}_{mt}"
                    try:
                        model_uri = self.mlflow_manager.get_latest_model(model_name, stage="Production")
                        if model_uri:
                            self.models[bet_type][mt] = mlflow.pyfunc.load_model(model_uri)
                            logger.info(f"  Loaded {model_name}")
                        else:
                            local_path = Path(self.config.models_dir) / bet_type / mt
                            if local_path.exists():
                                self.models[bet_type][mt] = mlflow.pyfunc.load_model(str(local_path))
                                logger.info(f"  Loaded {model_name} from local")
                    except Exception as e:
                        logger.warning(f"  Could not load {model_name}: {e}")
            else:
                model_name = f"{bet_type}_{model_type}"
                try:
                    model_uri = self.mlflow_manager.get_latest_model(model_name, stage="Production")
                    if model_uri:
                        self.models[bet_type] = mlflow.pyfunc.load_model(model_uri)
                        logger.info(f"  Loaded {model_name}")
                except Exception as e:
                    logger.warning(f"  Could not load {model_name}: {e}")

            self._load_features_config(bet_type)

    def _load_features_config(self, bet_type: str):
        """Load feature configuration for a bet type."""
        features_path = Path(self.config.models_dir) / bet_type / 'features.json'
        if features_path.exists():
            with open(features_path) as f:
                self.features_config[bet_type] = json.load(f)

    def prepare_features(self, df: pd.DataFrame, bet_type: str) -> pd.DataFrame:
        """
        Prepare features for prediction using strategy.

        Args:
            df: Input DataFrame with match data
            bet_type: Type of bet

        Returns:
            DataFrame with strategy-specific features added
        """
        strategy = self._get_strategy(bet_type)
        return strategy.create_features(df.copy())

    def predict(self, df: pd.DataFrame, bet_type: str) -> pd.DataFrame:
        """Generate predictions for a bet type."""
        if bet_type not in self.models:
            logger.warning(f"No callibration loaded for {bet_type}")
            return df

        strategy = self._get_strategy(bet_type)

        features = self.features_config.get(bet_type, {}).get('features', [])
        if not features:
            strategy_cfg = self.strategies_config['strategies'].get(bet_type, {})
            features = strategy_cfg.get('top_features', [])

        available_features = [f for f in features if f in df.columns]
        if not available_features:
            logger.warning(f"No features available for {bet_type}")
            return df

        X = df[available_features].copy()
        for col in X.columns:
            X[col] = X[col].fillna(X[col].median())

        predictions = {}
        models = self.models[bet_type]

        if isinstance(models, dict):
            for name, model in models.items():
                try:
                    pred = model.predict(X)
                    predictions[name] = pred
                except Exception as e:
                    logger.warning(f"Error predicting with {name}: {e}")
        else:
            predictions['model'] = models.predict(X)

        if predictions:
            df[f'{bet_type}_pred'] = np.mean(list(predictions.values()), axis=0)

            if not strategy.is_regression:
                df[f'{bet_type}_prob'] = df[f'{bet_type}_pred']
            else:
                df[f'{bet_type}_margin'] = df[f'{bet_type}_pred']

        return df

    def apply_strategy(self, df: pd.DataFrame, bet_type: str) -> List[BetRecommendation]:
        """
        Apply betting strategy and generate recommendations.

        Uses the Strategy pattern - delegates to strategy.create_recommendation()
        Now integrates BankrollManager for risk control enforcement.

        Args:
            df: DataFrame with predictions
            bet_type: Type of bet

        Returns:
            List of bet recommendations
        """
        strategy = self._get_strategy(bet_type)
        recommendations = []

        # Check if we can still bet today (enforce daily limits)
        if self.config.enforce_daily_limits:
            can_bet, reason = self.bankroll_manager.can_bet_today(bet_type)
            if not can_bet:
                logger.info(f"Skipping {bet_type}: {reason}")
                return recommendations

        if strategy.is_regression:
            pred_col = f'{bet_type}_margin'
        else:
            pred_col = f'{bet_type}_prob'

        if pred_col not in df.columns:
            logger.warning(f"No predictions found for {bet_type}")
            return recommendations

        threshold = strategy.config.probability_threshold

        for idx, row in df.iterrows():
            # Re-check daily limits after each recommendation
            if self.config.enforce_daily_limits:
                can_bet, reason = self.bankroll_manager.can_bet_today(bet_type)
                if not can_bet:
                    logger.info(f"Daily limit reached for {bet_type}: {reason}")
                    break

            prediction = row.get(pred_col, 0)

            if not strategy.is_regression and prediction < threshold:
                continue

            rec_dict = strategy.create_recommendation(
                row,
                prediction,
                bankroll=self.config.bankroll
            )

            if rec_dict:
                # Apply Sharpe-dampened Kelly stake calculation
                odds = rec_dict['odds']
                probability = rec_dict['probability']
                edge = rec_dict['edge']

                # Use BankrollManager's stake calculation
                stake = self.bankroll_manager.calculate_stake(
                    market=bet_type,
                    probability=probability,
                    odds=odds,
                    edge=edge,
                )

                if stake <= 0:
                    # Edge below minimum or market not profitable
                    continue

                # Update recommendation with BankrollManager stake
                rec_dict['recommended_stake'] = stake
                rec_dict['kelly_fraction'] = stake / self.config.bankroll

                recommendations.append(BetRecommendation.from_dict(rec_dict))

                # Record the bet for daily tracking
                if self.config.enforce_daily_limits:
                    self.bankroll_manager.record_bet(bet_type, stake)

        return recommendations

    def run(self, fixtures_df: pd.DataFrame,
            bet_types: Optional[List[str]] = None) -> List[BetRecommendation]:
        """
        Run inference pipeline on fixtures.

        Args:
            fixtures_df: DataFrame with fixture data
            bet_types: List of bet types to process (default: all enabled)

        Returns:
            List of bet recommendations sorted by expected value
        """
        logger.info(f"Running inference on {len(fixtures_df)} fixtures")
        logger.info(f"Available strategies: {list(STRATEGY_REGISTRY.keys())}")

        if not self.models:
            self.load_models(bet_types)

        if bet_types is None:
            bet_types = list(self.models.keys())

        logger.info(f"Processing bet types: {bet_types}")

        all_recommendations = []

        for bet_type in bet_types:
            if bet_type not in self.models:
                continue

            logger.info(f"Processing {bet_type}")

            df = self.prepare_features(fixtures_df, bet_type)
            df = self.predict(df, bet_type)

            recommendations = self.apply_strategy(df, bet_type)
            all_recommendations.extend(recommendations)

            logger.info(f"  Generated {len(recommendations)} recommendations")

        all_recommendations.sort(key=lambda x: x.expected_value, reverse=True)

        return all_recommendations

    def format_recommendations(self, recommendations: List[BetRecommendation]) -> str:
        """Format recommendations as readable string."""
        if not recommendations:
            return "No betting recommendations for today."

        lines = [
            "=" * 80,
            f"BETTING RECOMMENDATIONS - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "=" * 80,
            ""
        ]

        by_type = {}
        for rec in recommendations:
            if rec.bet_type not in by_type:
                by_type[rec.bet_type] = []
            by_type[rec.bet_type].append(rec)

        for bet_type, recs in by_type.items():
            lines.append(f"\n{bet_type.upper().replace('_', ' ')}")
            lines.append("-" * 40)

            for rec in recs[:5]:
                lines.append(
                    f"  {rec.home_team} vs {rec.away_team}"
                )
                lines.append(
                    f"    Bet: {rec.bet_side} @ {rec.odds:.2f}"
                )
                lines.append(
                    f"    Edge: {rec.edge:.1%} | EV: {rec.expected_value:.1%} | Stake: ${rec.recommended_stake:.2f}"
                )
                lines.append("")

        total_stake = sum(r.recommended_stake for r in recommendations)
        avg_ev = np.mean([r.expected_value for r in recommendations]) if recommendations else 0

        # Add bankroll manager summary
        daily_summary = self.bankroll_manager.get_daily_summary()

        lines.extend([
            "=" * 80,
            "SUMMARY",
            f"Total recommendations: {len(recommendations)}",
            f"Total suggested stake: ${total_stake:.2f}",
            f"Average expected value: {avg_ev:.1%}",
            "",
            "DAILY LIMITS STATUS",
            f"Bets placed today: {daily_summary['bets_placed']}/{daily_summary['max_daily_bets']}",
            f"Remaining bets: {daily_summary['remaining_bets']}",
            f"Daily P&L: ${daily_summary['total_pnl']:.2f} ({daily_summary['pnl_percentage']:.1f}%)",
            f"Stop-loss triggered: {'YES' if daily_summary['stop_loss_triggered'] else 'No'}",
            f"Take-profit reached: {'YES' if daily_summary['take_profit_reached'] else 'No'}",
            "=" * 80
        ])

        return "\n".join(lines)

    def save_recommendations(self, recommendations: List[BetRecommendation],
                             output_path: str):
        """Save recommendations to JSON."""
        data = {
            'timestamp': datetime.now().isoformat(),
            'count': len(recommendations),
            'recommendations': [
                {
                    'fixture_id': r.fixture_id,
                    'date': r.date,
                    'home_team': r.home_team,
                    'away_team': r.away_team,
                    'league': r.league,
                    'bet_type': r.bet_type,
                    'bet_side': r.bet_side,
                    'odds': r.odds,
                    'probability': r.probability,
                    'edge': r.edge,
                    'confidence': r.confidence,
                    'expected_value': r.expected_value,
                    'kelly_fraction': r.kelly_fraction,
                    'recommended_stake': r.recommended_stake
                }
                for r in recommendations
            ]
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved {len(recommendations)} recommendations to {output_path}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Betting Inference Pipeline')
    parser.add_argument('--fixtures', type=str, required=True, help='Path to fixtures CSV')
    parser.add_argument('--strategies', type=str, default='config/strategies.yaml')
    parser.add_argument('--callibration-dir', type=str, default='outputs/callibration')
    parser.add_argument('--output', type=str, default='outputs/recommendations.json')
    parser.add_argument('--bankroll', type=float, default=1000.0)

    args = parser.parse_args()

    config = InferenceConfig(
        strategies_path=args.strategies,
        models_dir=args.models_dir,
        bankroll=args.bankroll
    )

    pipeline = BettingInferencePipeline(config)
    fixtures = pd.read_csv(args.fixtures)
    recommendations = pipeline.run(fixtures)
    print(pipeline.format_recommendations(recommendations))
    pipeline.save_recommendations(recommendations, args.output)


if __name__ == '__main__':
    main()
