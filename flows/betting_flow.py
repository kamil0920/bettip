"""
Betting Prediction Pipeline using Metaflow

Metaflow provides:
- DAG-based workflow definition
- Automatic data versioning
- Easy parallelization with @foreach
- Cloud scaling (AWS Batch, Kubernetes)
- Resume failed runs
- Artifact tracking

Combined with MLflow for experiment tracking.

Usage:
    # Run locally
    uv run python flows/betting_flow.py run

    # Run with specific bet types
    uv run python flows/betting_flow.py run --bet_types '["asian_handicap", "away_win"]'

    # Resume failed run
    uv run python flows/betting_flow.py resume

    # Show all runs
    uv run python flows/betting_flow.py list

    # Visualize DAG
    uv run python flows/betting_flow.py show
"""
import sys

from metaflow import FlowSpec, step, Parameter, JSONType, retry, timeout, catch
import pandas as pd
import json
import yaml
from datetime import datetime
from pathlib import Path


class BettingPredictionFlow(FlowSpec):
    """
    End-to-end betting prediction pipeline.

    DAG:
        start
           ↓
        load_data
           ↓
        prepare_strategies (fan-out)
           ↓ (foreach bet_type)
        train_model (parallel)
           ↓
        evaluate_model (parallel)
           ↓ (join)
        combine_results
           ↓
        generate_recommendations
           ↓
        end
    """

    data_path = Parameter(
        'data_path',
        default='data/03-features/features_all_5leagues_with_odds.csv',
        help='Path to features CSV'
    )

    strategies_path = Parameter(
        'strategies_path',
        default='config/strategies.yaml',
        help='Path to strategies config'
    )

    bet_types = Parameter(
        'bet_types',
        type=JSONType,
        default='null',
        help='List of bet types to train (null = all enabled)'
    )

    n_trials = Parameter(
        'n_trials',
        default=40,
        help='Optuna trials per model'
    )

    @step
    def start(self):
        """Initialize the pipeline."""
        print(f"Starting Betting Prediction Pipeline")
        print(f"Data: {self.data_path}")
        print(f"Strategies: {self.strategies_path}")

        with open(self.strategies_path) as f:
            self.strategies = yaml.safe_load(f)

        self.next(self.load_data)

    @step
    def load_data(self):
        """Load and validate the features dataset."""
        print(f"Loading data from {self.data_path}")

        self.df = pd.read_csv(self.data_path)
        self.n_matches = len(self.df)
        self.n_features = len(self.df.columns)

        print(f"Loaded {self.n_matches} matches with {self.n_features} features")

        assert self.n_matches > 1000, "Not enough matches for training"
        assert 'date' in self.df.columns, "Missing date column"

        self.next(self.prepare_strategies)

    @step
    def prepare_strategies(self):
        """Determine which strategies to train."""
        if self.bet_types is None:
            self.active_bet_types = [
                bt for bt, cfg in self.strategies['strategies'].items()
                if cfg.get('enabled', False)
            ]
        else:
            self.active_bet_types = self.bet_types

        print(f"Active bet types: {self.active_bet_types}")

        self.next(self.train_model, foreach='active_bet_types')

    @retry(times=2)
    @timeout(minutes=30)
    @step
    def train_model(self):
        """Train callibration for a single bet type (runs in parallel)."""
        self.bet_type = self.input

        print(f"\n{'='*60}")
        print(f"TRAINING: {self.bet_type.upper()}")
        print(f"{'='*60}")

        project_root = Path(__file__).resolve().parent.parent
        sys.path.insert(0, str(project_root))

        from src.pipelines.betting_training_pipeline import BettingTrainingPipeline, TrainingConfig
        import tempfile

        config = TrainingConfig(
            data_path=self.data_path,
            strategies_path=self.strategies_path,
            output_dir=tempfile.mkdtemp(),
            n_optuna_trials=self.n_trials
        )

        pipeline = BettingTrainingPipeline(config)

        result = pipeline.train_bet_type(self.df, self.bet_type)

        self.training_result = result
        self.model_features = result.get('features', []) if result else []

        if result and 'results' in result and result['results']:
            best = result['results'][0]
            self.best_roi = best.get('roi', 0)
            self.best_p_profit = best.get('p_profit', 0)
            self.best_model = best.get('model', 'unknown')
            self.best_threshold = best.get('threshold', 0.5)
            self.n_bets = best.get('bets', 0)

            print(f"\nBest: {self.best_model} >= {self.best_threshold}")
            print(f"ROI: {self.best_roi:+.1f}%, P(profit): {self.best_p_profit:.0%}")
        else:
            self.best_roi = 0
            self.best_p_profit = 0
            self.best_model = None
            self.best_threshold = None
            self.n_bets = 0

        self.next(self.evaluate_model)

    @step
    def evaluate_model(self):
        """Evaluate trained model and compute additional metrics."""
        print(f"Evaluating {self.bet_type}")

        self.evaluation = {
            'bet_type': self.bet_type,
            'roi': self.best_roi,
            'p_profit': self.best_p_profit,
            'model': self.best_model,
            'threshold': self.best_threshold,
            'n_bets': self.n_bets,
            'n_features': len(self.model_features),
            'top_features': self.model_features[:5]
        }

        self.is_production_ready = (
            self.best_roi > 5.0 and
            self.best_p_profit > 0.7 and
            self.n_bets >= 30
        )

        print(f"Production ready: {self.is_production_ready}")

        self.next(self.combine_results)

    @step
    def combine_results(self, inputs):
        """Combine results from all parallel training runs."""
        print("\n" + "=" * 60)
        print("COMBINING RESULTS")
        print("=" * 60)

        self.all_results = []
        self.production_ready = []

        for inp in inputs:
            self.all_results.append(inp.evaluation)
            if inp.is_production_ready:
                self.production_ready.append(inp.bet_type)

        self.all_results.sort(key=lambda x: x['roi'], reverse=True)

        print(f"\n{'Bet Type':<20} {'ROI':>10} {'P(profit)':>12} {'Bets':>8} {'Ready':>8}")
        print("-" * 60)
        for r in self.all_results:
            ready = "✓" if r['bet_type'] in self.production_ready else "✗"
            print(f"{r['bet_type']:<20} {r['roi']:>+9.1f}% {r['p_profit']:>11.0%} {r['n_bets']:>8} {ready:>8}")

        print(f"\nProduction-ready strategies: {self.production_ready}")

        self.df = inputs[0].df
        self.strategies = inputs[0].strategies

        self.next(self.generate_recommendations)

    @step
    def generate_recommendations(self):
        """Generate betting recommendations for production-ready strategies."""
        print("\n" + "=" * 60)
        print("GENERATING RECOMMENDATIONS")
        print("=" * 60)

        self.recommendations = []
        self.summary = {
            'timestamp': datetime.now().isoformat(),
            'n_strategies_trained': len(self.all_results),
            'n_production_ready': len(self.production_ready),
            'results': self.all_results,
            'production_ready': self.production_ready
        }

        output_dir = Path('outputs/metaflow')
        output_dir.mkdir(parents=True, exist_ok=True)

        summary_path = output_dir / f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(summary_path, 'w') as f:
            json.dump(self.summary, f, indent=2)

        print(f"Summary saved to {summary_path}")

        self.next(self.end)

    @step
    def end(self):
        """Pipeline complete."""
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)

        if self.production_ready:
            print(f"\n✓ {len(self.production_ready)} strategies ready for production:")
            for bt in self.production_ready:
                r = next(x for x in self.all_results if x['bet_type'] == bt)
                print(f"  - {bt}: ROI={r['roi']:+.1f}%, P(profit)={r['p_profit']:.0%}")
        else:
            print("\n✗ No strategies met production criteria")

        print(f"\nTotal strategies trained: {len(self.all_results)}")


if __name__ == '__main__':
    BettingPredictionFlow()
