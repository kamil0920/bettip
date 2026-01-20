#!/usr/bin/env python3
"""
Main Orchestration Script for Bettip Pipeline

This script orchestrates the entire betting prediction pipeline:
1. Data collection (optional)
2. Preprocessing
3. Feature engineering
4. Model training
5. Inference and recommendations

Usage:
    # Full pipeline
    python entrypoints/run_pipeline.py --mode full

    # Training only
    python entrypoints/run_pipeline.py --mode train --data data/03-features/features.csv

    # Inference only
    python entrypoints/run_pipeline.py --mode inference --fixtures upcoming_fixtures.csv

    # Daily update (collect + preprocess + features + inference)
    python entrypoints/run_pipeline.py --mode daily
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
import yaml
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.pipelines.preprocessing_pipeline import PreprocessingPipeline
from src.pipelines.feature_eng_pipeline import FeatureEngineeringPipeline
from src.pipelines.betting_training_pipeline import BettingTrainingPipeline, TrainingConfig
from src.pipelines.betting_inference_pipeline import BettingInferencePipeline, InferenceConfig
from src.ml.mlflow_config import init_mlflow

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Orchestrates the entire betting prediction pipeline."""

    def __init__(self, config_path: str = "config/prod.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.output_dir = Path("outputs")
        self.output_dir.mkdir(exist_ok=True)

        # Initialize MLflow
        init_mlflow(experiment_name="bettip")

    def _load_config(self) -> dict:
        """Load configuration."""
        with open(self.config_path) as f:
            return yaml.safe_load(f)

    def run_preprocessing(self, leagues: list = None) -> Path:
        """Run preprocessing pipeline."""
        logger.info("=" * 60)
        logger.info("STEP 1: PREPROCESSING")
        logger.info("=" * 60)

        if leagues is None:
            leagues = self.config.get('leagues', ['premier_league'])

        for league in leagues:
            logger.info(f"Preprocessing {league}")
            try:
                pipeline = PreprocessingPipeline(
                    config_path=f"config/{league}.yaml"
                )
                pipeline.run()
            except Exception as e:
                logger.error(f"Error preprocessing {league}: {e}")

        preprocessed_dir = Path(self.config.get('data', {}).get('preprocessed_dir', 'data/02-preprocessed'))
        return preprocessed_dir

    def run_feature_engineering(self, leagues: list = None) -> Path:
        """Run feature engineering pipeline."""
        logger.info("=" * 60)
        logger.info("STEP 2: FEATURE ENGINEERING")
        logger.info("=" * 60)

        if leagues is None:
            leagues = self.config.get('leagues', ['premier_league'])

        for league in leagues:
            logger.info(f"Engineering features for {league}")
            try:
                pipeline = FeatureEngineeringPipeline(
                    config_path=f"config/{league}.yaml"
                )
                pipeline.run()
            except Exception as e:
                logger.error(f"Error in feature engineering for {league}: {e}")

        features_dir = Path(self.config.get('data', {}).get('features_dir', 'data/03-features'))
        return features_dir

    def run_training(self, data_path: str, bet_types: list = None) -> dict:
        """Run model training pipeline."""
        logger.info("=" * 60)
        logger.info("STEP 3: MODEL TRAINING")
        logger.info("=" * 60)

        config = TrainingConfig(
            data_path=data_path,
            strategies_path="config/strategies.yaml",
            output_dir=str(self.output_dir / "training"),
            n_optuna_trials=80
        )

        pipeline = BettingTrainingPipeline(config)
        results = pipeline.run(bet_types)

        # Save training results
        results_path = self.output_dir / "training" / "results.json"
        self._save_training_results(results, results_path)

        return results

    def run_inference(self, fixtures_path: str, bet_types: list = None) -> list:
        """Run inference pipeline."""
        logger.info("=" * 60)
        logger.info("STEP 4: INFERENCE")
        logger.info("=" * 60)

        import pandas as pd
        fixtures = pd.read_csv(fixtures_path)

        config = InferenceConfig(
            strategies_path="config/strategies.yaml",
            models_dir=str(self.output_dir / "callibration"),
            bankroll=1000.0
        )

        pipeline = BettingInferencePipeline(config)
        recommendations = pipeline.run(fixtures, bet_types)

        # Save recommendations
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / "recommendations" / f"recs_{timestamp}.json"
        pipeline.save_recommendations(recommendations, str(output_path))

        # Print recommendations
        print(pipeline.format_recommendations(recommendations))

        return recommendations

    def _save_training_results(self, results: dict, path: Path):
        """Save training results to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)

        summary = {
            'timestamp': datetime.now().isoformat(),
            'results': {}
        }

        for bet_type, result in results.items():
            if result and 'results' in result and result['results']:
                best = result['results'][0]
                summary['results'][bet_type] = {
                    'roi': best.get('roi'),
                    'p_profit': best.get('p_profit'),
                    'bets': best.get('bets'),
                    'model': best.get('model'),
                    'threshold': best.get('threshold')
                }

        with open(path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Training results saved to {path}")

    def run_full_pipeline(self, leagues: list = None):
        """Run the complete pipeline."""
        logger.info("=" * 80)
        logger.info("RUNNING FULL PIPELINE")
        logger.info("=" * 80)

        start_time = datetime.now()

        # Step 1: Preprocessing
        self.run_preprocessing(leagues)

        # Step 2: Feature Engineering
        features_dir = self.run_feature_engineering(leagues)

        # Step 3: Training
        features_path = features_dir / "features_all_5leagues_with_odds.csv"
        if not features_path.exists():
            features_path = features_dir / "features.csv"

        if features_path.exists():
            self.run_training(str(features_path))
        else:
            logger.warning(f"Features file not found: {features_path}")

        elapsed = datetime.now() - start_time
        logger.info(f"\nPipeline completed in {elapsed}")

    def run_daily_update(self, leagues: list = None):
        """Run daily update pipeline (no training)."""
        logger.info("=" * 80)
        logger.info("RUNNING DAILY UPDATE")
        logger.info("=" * 80)

        start_time = datetime.now()

        # Preprocessing & Features
        self.run_preprocessing(leagues)
        features_dir = self.run_feature_engineering(leagues)

        # Find upcoming fixtures for inference
        predictions_dir = Path(self.config.get('data', {}).get('predictions_dir', 'data/04-predictions'))
        fixtures_path = predictions_dir / "upcoming_fixtures.csv"

        if fixtures_path.exists():
            self.run_inference(str(fixtures_path))
        else:
            logger.warning(f"No upcoming fixtures found at {fixtures_path}")

        elapsed = datetime.now() - start_time
        logger.info(f"\nDaily update completed in {elapsed}")

    def run_retrain(self, data_path: str, bet_types: list = None):
        """Run retraining pipeline."""
        logger.info("=" * 80)
        logger.info("RUNNING RETRAINING")
        logger.info("=" * 80)

        self.run_training(data_path, bet_types)


def main():
    parser = argparse.ArgumentParser(
        description='Bettip Pipeline Orchestrator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run full pipeline
    python entrypoints/run_pipeline.py --mode full

    # Train callibration only
    python entrypoints/run_pipeline.py --mode train --data data/03-features/features.csv

    # Generate predictions
    python entrypoints/run_pipeline.py --mode inference --fixtures upcoming.csv

    # Daily update (preprocess + features + inference)
    python entrypoints/run_pipeline.py --mode daily

    # Retrain specific bet types
    python entrypoints/run_pipeline.py --mode train --data features.csv --bet-types asian_handicap away_win
        """
    )

    parser.add_argument(
        '--mode',
        choices=['full', 'preprocess', 'features', 'train', 'inference', 'daily', 'retrain'],
        default='full',
        help='Pipeline mode to run'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/prod.yaml',
        help='Configuration file path'
    )
    parser.add_argument(
        '--data',
        type=str,
        help='Path to features data (for train mode)'
    )
    parser.add_argument(
        '--fixtures',
        type=str,
        help='Path to fixtures CSV (for inference mode)'
    )
    parser.add_argument(
        '--leagues',
        nargs='+',
        help='Leagues to process'
    )
    parser.add_argument(
        '--bet-types',
        nargs='+',
        help='Bet types to train/predict'
    )

    args = parser.parse_args()

    # Initialize orchestrator
    orchestrator = PipelineOrchestrator(config_path=args.config)

    # Run appropriate mode
    if args.mode == 'full':
        orchestrator.run_full_pipeline(args.leagues)

    elif args.mode == 'preprocess':
        orchestrator.run_preprocessing(args.leagues)

    elif args.mode == 'features':
        orchestrator.run_feature_engineering(args.leagues)

    elif args.mode == 'train':
        if not args.data:
            parser.error("--data is required for train mode")
        orchestrator.run_training(args.data, args.bet_types)

    elif args.mode == 'inference':
        if not args.fixtures:
            parser.error("--fixtures is required for inference mode")
        orchestrator.run_inference(args.fixtures, args.bet_types)

    elif args.mode == 'daily':
        orchestrator.run_daily_update(args.leagues)

    elif args.mode == 'retrain':
        if not args.data:
            parser.error("--data is required for retrain mode")
        orchestrator.run_retrain(args.data, args.bet_types)


if __name__ == '__main__':
    main()
