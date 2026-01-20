"""
Daily Inference Flow using Metaflow

This flow runs daily to:
1. Fetch latest fixtures
2. Generate features for upcoming matches
3. Load trained callibration
4. Generate predictions
5. Apply betting strategies
6. Output recommendations

Usage:
    # Run locally
    uv run python flows/daily_inference_flow.py run

    # Schedule with Argo/AWS Step Functions
    uv run python flows/daily_inference_flow.py argo-workflows create
"""

from metaflow import FlowSpec, step, Parameter, schedule, retry, timeout, catch
import pandas as pd
import numpy as np
import json
import yaml
from datetime import datetime, timedelta
from pathlib import Path


class DailyInferenceFlow(FlowSpec):
    """
    Daily inference pipeline for betting recommendations.

    DAG:
        start
           ↓
        fetch_fixtures
           ↓
        generate_features
           ↓
        load_models
           ↓
        make_predictions
           ↓
        apply_strategies
           ↓
        output_recommendations
           ↓
        end
    """

    strategies_path = Parameter(
        'strategies_path',
        default='config/strategies.yaml',
        help='Path to strategies config'
    )

    bankroll = Parameter(
        'bankroll',
        default=1000.0,
        help='Bankroll for stake calculation'
    )

    lookforward_days = Parameter(
        'lookforward_days',
        default=7,
        help='Days ahead to fetch fixtures'
    )

    @step
    def start(self):
        """Initialize daily inference."""
        print(f"Daily Inference - {datetime.now().strftime('%Y-%m-%d %H:%M')}")

        with open(self.strategies_path) as f:
            self.strategies = yaml.safe_load(f)

        self.enabled_strategies = [
            bt for bt, cfg in self.strategies['strategies'].items()
            if cfg.get('enabled', False)
        ]

        print(f"Enabled strategies: {self.enabled_strategies}")

        self.next(self.fetch_fixtures)

    @retry(times=3)
    @timeout(minutes=10)
    @step
    def fetch_fixtures(self):
        """Fetch upcoming fixtures from API or cache."""
        print("Fetching upcoming fixtures...")

        fixtures_path = Path('data/04-predictions/upcoming_fixtures.csv')

        if fixtures_path.exists():
            self.fixtures = pd.read_csv(fixtures_path)
            print(f"Loaded {len(self.fixtures)} fixtures from cache")
        else:
            print("No cached fixtures found. Using empty DataFrame.")
            self.fixtures = pd.DataFrame()

        if len(self.fixtures) > 0:
            self.fixtures['date'] = pd.to_datetime(self.fixtures['date'])
            today = datetime.now()
            future = today + timedelta(days=self.lookforward_days)

            self.fixtures = self.fixtures[
                (self.fixtures['date'] >= today) &
                (self.fixtures['date'] <= future)
            ]

            print(f"Upcoming fixtures (next {self.lookforward_days} days): {len(self.fixtures)}")

        self.next(self.generate_features)

    @step
    def generate_features(self):
        """Generate features for upcoming fixtures."""
        if len(self.fixtures) == 0:
            print("No fixtures to process")
            self.features_df = pd.DataFrame()
            self.next(self.load_models)
            return

        print(f"Generating features for {len(self.fixtures)} fixtures...")

        self._add_ah_features()
        self._add_btts_features()
        self.features_df = self.fixtures
        print(f"Features shape: {self.features_df.shape}")

        self.next(self.load_models)

    def _add_ah_features(self):
        """Add Asian Handicap specific features."""
        df = self.fixtures

        if 'home_xg_poisson' in df.columns and 'away_xg_poisson' in df.columns:
            df['xg_margin'] = df['home_xg_poisson'] - df['away_xg_poisson']

        if 'elo_diff' in df.columns:
            df['elo_expected_margin'] = df['elo_diff'] / 400

        if 'season_gd_diff' in df.columns and 'round_number' in df.columns:
            df['season_margin_per_game'] = df['season_gd_diff'] / (df['round_number'] + 1)

        margin_cols = ['elo_expected_margin', 'xg_margin', 'season_margin_per_game']
        available = [c for c in margin_cols if c in df.columns]
        if available:
            df['composite_expected_margin'] = df[available].mean(axis=1)

        if 'ah_line' in df.columns and 'composite_expected_margin' in df.columns:
            df['margin_edge'] = df['composite_expected_margin'] - (-df['ah_line'])

    def _add_btts_features(self):
        """Add BTTS specific features."""
        df = self.fixtures

        if 'home_goals_scored_ema' in df.columns:
            df['home_scores_prob'] = 1 - np.exp(-df['home_goals_scored_ema'].fillna(1.5))
        if 'away_goals_scored_ema' in df.columns:
            df['away_scores_prob'] = 1 - np.exp(-df['away_goals_scored_ema'].fillna(1.2))

        if 'home_scores_prob' in df.columns and 'away_scores_prob' in df.columns:
            df['btts_composite'] = df['home_scores_prob'] * df['away_scores_prob']

    @step
    def load_models(self):
        """Load trained callibration from MLflow registry."""
        print("Loading trained callibration...")

        self.models = {}

        try:
            import mlflow
            mlflow.set_tracking_uri("sqlite:///mlflow.db")

            for bet_type in self.enabled_strategies:
                self.models[bet_type] = {}

                for model_type in ['xgboost', 'lightgbm', 'catboost']:
                    model_name = f"{bet_type}_{model_type}"
                    try:
                        model_uri = f"callibration:/{model_name}/Production"
                        self.models[bet_type][model_type] = mlflow.pyfunc.load_model(model_uri)
                        print(f"  Loaded {model_name}")
                    except Exception:
                        try:
                            model_uri = f"callibration:/{model_name}/latest"
                            self.models[bet_type][model_type] = mlflow.pyfunc.load_model(model_uri)
                            print(f"  Loaded {model_name} (latest)")
                        except Exception as e:
                            print(f"  Could not load {model_name}: {e}")

        except Exception as e:
            print(f"Error loading callibration: {e}")

        self.next(self.make_predictions)

    @step
    def make_predictions(self):
        """Generate predictions for each bet type."""
        if len(self.features_df) == 0:
            print("No fixtures to predict")
            self.predictions = {}
            self.next(self.apply_strategies)
            return

        print("Generating predictions...")

        self.predictions = {}

        for bet_type, models in self.models.items():
            if not models:
                continue

            strategy = self.strategies['strategies'].get(bet_type, {})
            features = strategy.get('top_features', [])

            available = [f for f in features if f in self.features_df.columns]
            if not available:
                print(f"  No features available for {bet_type}")
                continue

            X = self.features_df[available].fillna(0)

            preds = []
            for model_type, model in models.items():
                try:
                    pred = model.predict(X)
                    preds.append(pred)
                except Exception as e:
                    print(f"  Error with {model_type}: {e}")

            if preds:
                self.predictions[bet_type] = np.mean(preds, axis=0)
                print(f"  {bet_type}: {len(preds)} callibration averaged")

        self.next(self.apply_strategies)

    @step
    def apply_strategies(self):
        """Apply betting strategies to predictions."""
        print("Applying betting strategies...")

        self.recommendations = []

        for bet_type, preds in self.predictions.items():
            strategy = self.strategies['strategies'].get(bet_type, {})

            if bet_type == 'asian_handicap':
                recs = self._apply_ah_strategy(preds, strategy)
            elif bet_type == 'btts':
                recs = self._apply_btts_strategy(preds, strategy)
            else:
                recs = self._apply_classification_strategy(bet_type, preds, strategy)

            self.recommendations.extend(recs)
            print(f"  {bet_type}: {len(recs)} recommendations")

        self.recommendations.sort(key=lambda x: x.get('expected_value', 0), reverse=True)

        self.next(self.output_recommendations)

    def _apply_ah_strategy(self, preds, strategy):
        """Apply Asian Handicap strategy."""
        recs = []
        df = self.features_df

        edge_threshold = strategy.get('edge_threshold', 0.3)
        line_filter = strategy.get('line_filter', {})
        min_line = line_filter.get('min', -4)
        max_line = line_filter.get('max', -1.5)

        for i, row in df.iterrows():
            ah_line = row.get('ah_line', 0)

            if not (min_line <= ah_line <= max_line):
                continue

            pred_margin = preds[i] if i < len(preds) else 0
            bookie_margin = -ah_line
            edge = pred_margin - bookie_margin

            if edge < -edge_threshold:
                odds = row.get('avg_ah_away', 1.9)
                ev = 0.5 * odds - 1

                recs.append({
                    'fixture_id': row.get('fixture_id', i),
                    'home_team': row.get('home_team_name', 'Home'),
                    'away_team': row.get('away_team_name', 'Away'),
                    'date': str(row.get('date', '')),
                    'bet_type': 'asian_handicap',
                    'bet': f'Away {ah_line:+.2f}',
                    'odds': odds,
                    'edge': abs(edge),
                    'expected_value': ev,
                    'stake': min(ev * self.bankroll * 0.1, self.bankroll * 0.05)
                })

        return recs

    def _apply_btts_strategy(self, preds, strategy):
        """Apply BTTS strategy."""
        recs = []
        df = self.features_df

        threshold = strategy.get('probability_threshold', 0.6)

        for i, row in df.iterrows():
            prob = preds[i] if i < len(preds) else 0

            if prob >= threshold:
                odds = row.get('btts_yes_odds', 1.8)
                ev = prob * odds - 1

                if ev > 0.05:
                    recs.append({
                        'fixture_id': row.get('fixture_id', i),
                        'home_team': row.get('home_team_name', 'Home'),
                        'away_team': row.get('away_team_name', 'Away'),
                        'date': str(row.get('date', '')),
                        'bet_type': 'btts',
                        'bet': 'Yes',
                        'odds': odds,
                        'probability': prob,
                        'expected_value': ev,
                        'stake': min(ev * self.bankroll * 0.1, self.bankroll * 0.05)
                    })

        return recs

    def _apply_classification_strategy(self, bet_type, preds, strategy):
        """Apply classification strategy."""
        recs = []
        df = self.features_df

        threshold = strategy.get('probability_threshold', 0.5)
        odds_col = strategy.get('odds_column', 'odds')

        bet_labels = {
            'away_win': 'Away Win',
            'home_win': 'Home Win',
            'over25': 'Over 2.5',
            'under25': 'Under 2.5'
        }

        for i, row in df.iterrows():
            prob = preds[i] if i < len(preds) else 0

            if prob >= threshold:
                odds = row.get(odds_col, 2.0)
                ev = prob * odds - 1

                if ev > 0.05:
                    recs.append({
                        'fixture_id': row.get('fixture_id', i),
                        'home_team': row.get('home_team_name', 'Home'),
                        'away_team': row.get('away_team_name', 'Away'),
                        'date': str(row.get('date', '')),
                        'bet_type': bet_type,
                        'bet': bet_labels.get(bet_type, bet_type),
                        'odds': odds,
                        'probability': prob,
                        'expected_value': ev,
                        'stake': min(ev * self.bankroll * 0.1, self.bankroll * 0.05)
                    })

        return recs

    @step
    def output_recommendations(self):
        """Output and save recommendations."""
        print("\n" + "=" * 60)
        print("RECOMMENDATIONS")
        print("=" * 60)

        if not self.recommendations:
            print("No recommendations for today")
            self.next(self.end)
            return

        for rec in self.recommendations[:10]:
            print(f"\n{rec['home_team']} vs {rec['away_team']}")
            print(f"  Bet: {rec['bet']} @ {rec['odds']:.2f}")
            print(f"  EV: {rec['expected_value']:.1%} | Stake: ${rec['stake']:.2f}")

        output_dir = Path('outputs/recommendations')
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"daily_{datetime.now().strftime('%Y%m%d')}.json"
        with open(output_path, 'w') as f:
            json.dump({
                'date': datetime.now().isoformat(),
                'count': len(self.recommendations),
                'total_stake': sum(r['stake'] for r in self.recommendations),
                'recommendations': self.recommendations
            }, f, indent=2)

        print(f"\nSaved {len(self.recommendations)} recommendations to {output_path}")

        self.next(self.end)

    @step
    def end(self):
        """Pipeline complete."""
        print("\n" + "=" * 60)
        print("DAILY INFERENCE COMPLETE")
        print("=" * 60)

        if self.recommendations:
            total_stake = sum(r['stake'] for r in self.recommendations)
            avg_ev = np.mean([r['expected_value'] for r in self.recommendations])
            print(f"Total recommendations: {len(self.recommendations)}")
            print(f"Total stake: ${total_stake:.2f}")
            print(f"Average EV: {avg_ev:.1%}")


if __name__ == '__main__':
    DailyInferenceFlow()
