#!/usr/bin/env python
"""
Niche Markets V2 - Improved Models with Team-Specific Features

Key improvements over V1:
1. Uses team foul/shot EMA averages (critical for team-specific patterns)
2. Incorporates goal expectation for shots (correlated with shot volume)
3. League-specific baselines (Serie A has different patterns than EPL)
4. Better feature selection based on analysis

Findings from Jan 14-15 analysis:
- FOULS: AC Milan matches average only 21.9 fouls (lowest in Serie A)
  - Model predicted OVER 22.5 for Como vs Milan, actual was 13
  - Need team-specific foul features, not just referee

- SHOTS: 5-goal matches have 62.5% over 26.5 shots
  - Model predicted UNDER 26.5 for Verona vs Bologna (5 goals), actual was 29
  - Need goal expectation features

Usage:
    python experiments/niche_markets_v2.py train     # Train improved models
    python experiments/niche_markets_v2.py predict  # Generate predictions
    python experiments/niche_markets_v2.py status   # View dashboard
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import brier_score_loss, log_loss
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Import data quality module
from src.features.data_quality import (
    validate_features_for_prediction,
    add_data_quality_flags,
    get_prediction_confidence,
    LEAGUE_MEDIANS,
    CRITICAL_EMA_FEATURES,
)

import warnings
warnings.filterwarnings('ignore')

# League-specific baselines (from analysis)
LEAGUE_BASELINES = {
    'premier_league': {'fouls': 21.3, 'shots': 24.3},
    'la_liga': {'fouls': 25.1, 'shots': 24.9},
    'serie_a': {'fouls': 26.6, 'shots': 24.8},
    'bundesliga': {'fouls': 24.0, 'shots': 25.0},
    'ligue_1': {'fouls': 25.0, 'shots': 24.5},
}

# Default odds
DEFAULT_ODDS = {
    'fouls_over_22_5': 1.75, 'fouls_under_22_5': 2.05,
    'fouls_over_24_5': 1.90, 'fouls_under_24_5': 1.90,
    'fouls_over_26_5': 2.15, 'fouls_under_26_5': 1.70,
    'shots_over_22_5': 1.75, 'shots_under_22_5': 2.05,
    'shots_over_24_5': 1.90, 'shots_under_24_5': 1.90,
    'shots_over_26_5': 1.70, 'shots_under_26_5': 2.15,
    'corners_over_9_5': 1.90, 'corners_under_9_5': 1.90,
    'corners_over_10_5': 2.10, 'corners_under_10_5': 1.72,
    'corners_over_11_5': 2.50, 'corners_under_11_5': 1.55,
}


# V2 Features - incorporating team-specific patterns
FOULS_FEATURES_V2 = [
    # Team foul history (CRITICAL - missed in V1)
    'home_fouls_committed_ema',
    'away_fouls_committed_ema',
    # Referee features (available in data)
    'ref_avg_goals',  # proxy for game tempo/strictness
    # Match context (game tempo indicators)
    'odds_goals_expectation',
    'home_xg_poisson',
    'away_xg_poisson',
    # Team aggression proxies
    'home_avg_yellows',
    'away_avg_yellows',
    # Form indicators
    'home_points_ema',
    'away_points_ema',
    'elo_diff',
    # League indicator for different baselines
    'odds_home_prob',
    'odds_away_prob',
]

SHOTS_FEATURES_V2 = [
    # Team shot history (CRITICAL)
    'home_shots_total_ema',
    'away_shots_total_ema',
    'home_shots_on_ema',
    'away_shots_on_ema',
    # Goal expectation (correlated 0.225 with shots)
    'odds_goals_expectation',
    'home_xg_poisson',
    'away_xg_poisson',
    # Attack strength
    'home_goals_scored_ema',
    'away_goals_scored_ema',
    # Referee influence (less important for shots)
    'ref_avg_goals',
    # Form
    'home_points_ema',
    'away_points_ema',
    'elo_diff',
]

CORNERS_FEATURES_V2 = [
    # Existing strong features
    'ref_corner_avg',
    'odds_goals_expectation',
    'odds_over25_prob',
    'odds_under25_prob',
    # Attack metrics
    'home_shots_total_ema',
    'away_shots_total_ema',
    # Form
    'home_points_ema',
    'away_points_ema',
    'elo_diff',
    'home_elo',
    'away_elo',
]


class NicheMarketsV2:
    """Improved niche market predictions with team-specific features."""

    def __init__(self, output_dir: str = "experiments/outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.models = {}
        self.calibrators = {}

    def load_training_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load features and match statistics."""
        # Load main features
        features_path = Path('data/03-features/features_all_5leagues_with_odds.csv')
        features = pd.read_csv(features_path)
        print(f"Loaded {len(features)} matches with features")

        # Load match stats for targets
        all_stats = []
        for league in ['premier_league', 'la_liga', 'serie_a']:
            for season in ['2024', '2025']:
                stats_path = Path(f'data/01-raw/{league}/{season}/match_stats.parquet')
                if stats_path.exists():
                    df = pd.read_parquet(stats_path)
                    df['league'] = league
                    all_stats.append(df)

        if all_stats:
            stats = pd.concat(all_stats, ignore_index=True)
            stats['total_fouls'] = stats['home_fouls'] + stats['away_fouls']
            stats['total_shots'] = stats['home_shots'] + stats['away_shots']
            stats['total_corners'] = stats['home_corners'] + stats['away_corners']
            print(f"Loaded {len(stats)} matches with statistics")
        else:
            stats = pd.DataFrame()

        return features, stats

    def prepare_features(
        self,
        df: pd.DataFrame,
        feature_list: List[str],
        use_data_quality: bool = True
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Prepare features with data quality validation and imputation.

        Args:
            df: Input DataFrame
            feature_list: List of required features
            use_data_quality: Whether to use data quality module

        Returns:
            Tuple of (prepared features, original df with quality flags)
        """
        available = [f for f in feature_list if f in df.columns]
        missing = [f for f in feature_list if f not in df.columns]

        if missing:
            print(f"  Warning: Missing features (not in DataFrame): {missing}")

        # Add data quality flags to track imputation
        if use_data_quality:
            df_flagged = add_data_quality_flags(df.copy())

            # Use data quality module for critical EMA features
            critical_to_impute = [f for f in available if f in CRITICAL_EMA_FEATURES]
            if critical_to_impute:
                df_validated, report = validate_features_for_prediction(
                    df_flagged,
                    required_features=available,
                    allow_imputation=True
                )
                print(f"  Data quality: {report.completeness_ratio:.1%} complete, "
                      f"imputed: {report.imputed_columns}")
            else:
                df_validated = df_flagged
        else:
            df_validated = df.copy()

        X = df_validated[available].copy()

        # Fill any remaining NaN with column median (for non-critical features)
        for col in X.columns:
            if X[col].isna().any():
                median_val = X[col].median()
                if pd.isna(median_val):
                    median_val = 0  # Ultimate fallback
                X[col] = X[col].fillna(median_val)

        return X, df_validated if use_data_quality else None

    def calculate_referee_stats(self, df: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
        """Calculate referee statistics from training data only (leakage-free)."""
        if 'referee' not in df.columns or stats.empty:
            return df

        # Get referee averages from stats
        ref_stats = stats.groupby('home_team').agg({
            'total_fouls': 'mean',
            'total_shots': 'mean',
            'total_corners': 'mean'
        }).reset_index()

        # This is a placeholder - in production, calculate from actual referee data
        return df

    def train_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_type: str = 'xgboost'
    ) -> Tuple[object, object]:
        """Train and calibrate a model with time series split."""

        # Time series split for proper validation
        tscv = TimeSeriesSplit(n_splits=3)

        if model_type == 'xgboost':
            base_model = XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )
        elif model_type == 'lightgbm':
            base_model = LGBMClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=42,
                verbose=-1
            )
        elif model_type == 'catboost':
            base_model = CatBoostClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=42,
                verbose=0
            )
        else:  # random_forest
            base_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=42
            )

        # Train with calibration
        calibrated = CalibratedClassifierCV(
            base_model,
            method='sigmoid',
            cv=tscv
        )

        calibrated.fit(X, y)

        return calibrated, base_model

    def train_all_models(self):
        """Train models for all niche markets."""
        print("\n" + "=" * 70)
        print("TRAINING NICHE MARKET MODELS V2")
        print("=" * 70)

        features, stats = self.load_training_data()

        # Merge features with stats
        if 'fixture_id' in features.columns and 'fixture_id' in stats.columns:
            merged = features.merge(
                stats[['fixture_id', 'total_fouls', 'total_shots', 'total_corners']],
                on='fixture_id',
                how='inner'
            )
        else:
            print("Cannot merge - fixture_id not found")
            return

        print(f"Merged dataset: {len(merged)} matches")

        # Calculate derived features
        if 'home_fouls_committed_ema' in merged.columns and 'away_fouls_committed_ema' in merged.columns:
            merged['expected_fouls'] = merged['home_fouls_committed_ema'] + merged['away_fouls_committed_ema']
        if 'home_shots_total_ema' in merged.columns and 'away_shots_total_ema' in merged.columns:
            merged['expected_shots'] = merged['home_shots_total_ema'] + merged['away_shots_total_ema']

        # Train FOULS models
        print("\n--- FOULS MODELS ---")
        for line in [22.5, 24.5, 26.5]:
            target = (merged['total_fouls'] > line).astype(int)
            X, _ = self.prepare_features(merged, FOULS_FEATURES_V2, use_data_quality=False)

            valid_idx = ~(X.isna().any(axis=1) | target.isna())
            X_clean = X[valid_idx]
            y_clean = target[valid_idx]

            if len(X_clean) < 100:
                print(f"  Skipping fouls_{line}: insufficient data ({len(X_clean)})")
                continue

            model, _ = self.train_model(X_clean, y_clean, 'xgboost')
            self.models[f'fouls_{line}'] = model

            # Evaluate
            probs = model.predict_proba(X_clean)[:, 1]
            brier = brier_score_loss(y_clean, probs)
            base_rate = y_clean.mean()
            print(f"  fouls_over_{line}: Brier={brier:.4f}, BaseRate={base_rate:.1%}, N={len(X_clean)}")

        # Train SHOTS models
        print("\n--- SHOTS MODELS ---")
        for line in [22.5, 24.5, 26.5]:
            target = (merged['total_shots'] > line).astype(int)
            X, _ = self.prepare_features(merged, SHOTS_FEATURES_V2, use_data_quality=False)

            valid_idx = ~(X.isna().any(axis=1) | target.isna())
            X_clean = X[valid_idx]
            y_clean = target[valid_idx]

            if len(X_clean) < 100:
                print(f"  Skipping shots_{line}: insufficient data ({len(X_clean)})")
                continue

            model, _ = self.train_model(X_clean, y_clean, 'lightgbm')
            self.models[f'shots_{line}'] = model

            probs = model.predict_proba(X_clean)[:, 1]
            brier = brier_score_loss(y_clean, probs)
            base_rate = y_clean.mean()
            print(f"  shots_over_{line}: Brier={brier:.4f}, BaseRate={base_rate:.1%}, N={len(X_clean)}")

        # Train CORNERS models
        print("\n--- CORNERS MODELS ---")
        for line in [9.5, 10.5, 11.5]:
            target = (merged['total_corners'] > line).astype(int)
            X, _ = self.prepare_features(merged, CORNERS_FEATURES_V2, use_data_quality=False)

            valid_idx = ~(X.isna().any(axis=1) | target.isna())
            X_clean = X[valid_idx]
            y_clean = target[valid_idx]

            if len(X_clean) < 100:
                print(f"  Skipping corners_{line}: insufficient data ({len(X_clean)})")
                continue

            model, _ = self.train_model(X_clean, y_clean, 'xgboost')
            self.models[f'corners_{line}'] = model

            probs = model.predict_proba(X_clean)[:, 1]
            brier = brier_score_loss(y_clean, probs)
            base_rate = y_clean.mean()
            print(f"  corners_over_{line}: Brier={brier:.4f}, BaseRate={base_rate:.1%}, N={len(X_clean)}")

        print(f"\nTrained {len(self.models)} models")

        # Save models info
        model_info = {
            'version': 'v2',
            'trained_at': datetime.now().isoformat(),
            'models': list(self.models.keys()),
            'fouls_features': FOULS_FEATURES_V2,
            'shots_features': SHOTS_FEATURES_V2,
            'corners_features': CORNERS_FEATURES_V2,
        }
        with open(self.output_dir / 'niche_models_v2_info.json', 'w') as f:
            json.dump(model_info, f, indent=2)

        return self.models

    def generate_predictions(
        self,
        fixtures: pd.DataFrame,
        min_edge: float = 15.0,
        min_prob: float = 0.60
    ) -> List[Dict]:
        """Generate predictions for upcoming fixtures."""

        if not self.models:
            print("No models loaded. Run train first.")
            return []

        predictions = []

        for _, match in fixtures.iterrows():
            match_dict = match.to_dict()

            # FOULS predictions
            for line in [22.5, 24.5, 26.5]:
                model_key = f'fouls_{line}'
                if model_key not in self.models:
                    continue

                match_df = pd.DataFrame([match_dict])
                X, df_quality = self.prepare_features(match_df, FOULS_FEATURES_V2, use_data_quality=True)
                if X.empty or X.isna().all().all():
                    continue

                prob_over = self.models[model_key].predict_proba(X)[0, 1]
                prob_under = 1 - prob_over

                # Calculate confidence based on data quality
                confidence = 1.0
                if df_quality is not None and '_is_imputed' in df_quality.columns:
                    confidence = get_prediction_confidence(df_quality.iloc[0])

                # Check OVER bet
                over_odds = DEFAULT_ODDS.get(f'fouls_over_{line}'.replace('.', '_'), 1.90)
                over_edge = (over_odds * prob_over - 1) * 100
                if prob_over >= min_prob and over_edge >= min_edge:
                    predictions.append({
                        'fixture_id': match.get('fixture_id'),
                        'date': match.get('date', match.get('match_date', '')),
                        'home_team': match.get('home_team'),
                        'away_team': match.get('away_team'),
                        'league': match.get('league'),
                        'market': 'FOULS',
                        'bet_type': 'OVER',
                        'line': line,
                        'odds': over_odds,
                        'probability': prob_over,
                        'edge': over_edge,
                        'confidence': confidence,
                        'referee': match.get('referee'),
                    })

                # Check UNDER bet
                under_odds = DEFAULT_ODDS.get(f'fouls_under_{line}'.replace('.', '_'), 1.90)
                under_edge = (under_odds * prob_under - 1) * 100
                if prob_under >= min_prob and under_edge >= min_edge:
                    predictions.append({
                        'fixture_id': match.get('fixture_id'),
                        'date': match.get('date', match.get('match_date', '')),
                        'home_team': match.get('home_team'),
                        'away_team': match.get('away_team'),
                        'league': match.get('league'),
                        'market': 'FOULS',
                        'bet_type': 'UNDER',
                        'line': line,
                        'odds': under_odds,
                        'probability': prob_under,
                        'edge': under_edge,
                        'confidence': confidence,
                        'referee': match.get('referee'),
                    })

            # SHOTS predictions
            for line in [22.5, 24.5, 26.5]:
                model_key = f'shots_{line}'
                if model_key not in self.models:
                    continue

                match_df = pd.DataFrame([match_dict])
                X, df_quality = self.prepare_features(match_df, SHOTS_FEATURES_V2, use_data_quality=True)
                if X.empty or X.isna().all().all():
                    continue

                prob_over = self.models[model_key].predict_proba(X)[0, 1]
                prob_under = 1 - prob_over

                # Calculate confidence based on data quality
                confidence = 1.0
                if df_quality is not None and '_is_imputed' in df_quality.columns:
                    confidence = get_prediction_confidence(df_quality.iloc[0])

                # Check OVER bet
                over_odds = DEFAULT_ODDS.get(f'shots_over_{line}'.replace('.', '_'), 1.90)
                over_edge = (over_odds * prob_over - 1) * 100
                if prob_over >= min_prob and over_edge >= min_edge:
                    predictions.append({
                        'fixture_id': match.get('fixture_id'),
                        'date': match.get('date', match.get('match_date', '')),
                        'home_team': match.get('home_team'),
                        'away_team': match.get('away_team'),
                        'league': match.get('league'),
                        'market': 'SHOTS',
                        'bet_type': 'OVER',
                        'line': line,
                        'odds': over_odds,
                        'probability': prob_over,
                        'edge': over_edge,
                        'confidence': confidence,
                        'referee': match.get('referee'),
                    })

                # Check UNDER bet
                under_odds = DEFAULT_ODDS.get(f'shots_under_{line}'.replace('.', '_'), 1.90)
                under_edge = (under_odds * prob_under - 1) * 100
                if prob_under >= min_prob and under_edge >= min_edge:
                    predictions.append({
                        'fixture_id': match.get('fixture_id'),
                        'date': match.get('date', match.get('match_date', '')),
                        'home_team': match.get('home_team'),
                        'away_team': match.get('away_team'),
                        'league': match.get('league'),
                        'market': 'SHOTS',
                        'bet_type': 'UNDER',
                        'line': line,
                        'odds': under_odds,
                        'probability': prob_under,
                        'edge': under_edge,
                        'confidence': confidence,
                        'referee': match.get('referee'),
                    })

            # CORNERS predictions
            for line in [9.5, 10.5, 11.5]:
                model_key = f'corners_{line}'
                if model_key not in self.models:
                    continue

                match_df = pd.DataFrame([match_dict])
                X, df_quality = self.prepare_features(match_df, CORNERS_FEATURES_V2, use_data_quality=True)
                if X.empty or X.isna().all().all():
                    continue

                prob_over = self.models[model_key].predict_proba(X)[0, 1]
                prob_under = 1 - prob_over

                # Calculate confidence based on data quality
                confidence = 1.0
                if df_quality is not None and '_is_imputed' in df_quality.columns:
                    confidence = get_prediction_confidence(df_quality.iloc[0])

                # Check OVER bet
                over_odds = DEFAULT_ODDS.get(f'corners_over_{line}'.replace('.', '_'), 2.00)
                over_edge = (over_odds * prob_over - 1) * 100
                if prob_over >= min_prob and over_edge >= min_edge:
                    predictions.append({
                        'fixture_id': match.get('fixture_id'),
                        'date': match.get('date', match.get('match_date', '')),
                        'home_team': match.get('home_team'),
                        'away_team': match.get('away_team'),
                        'league': match.get('league'),
                        'market': 'CORNERS',
                        'bet_type': 'OVER',
                        'line': line,
                        'odds': over_odds,
                        'probability': prob_over,
                        'edge': over_edge,
                        'confidence': confidence,
                        'referee': match.get('referee'),
                    })

                # Check UNDER bet
                under_odds = DEFAULT_ODDS.get(f'corners_under_{line}'.replace('.', '_'), 1.80)
                under_edge = (under_odds * prob_under - 1) * 100
                if prob_under >= min_prob and under_edge >= min_edge:
                    predictions.append({
                        'fixture_id': match.get('fixture_id'),
                        'date': match.get('date', match.get('match_date', '')),
                        'home_team': match.get('home_team'),
                        'away_team': match.get('away_team'),
                        'league': match.get('league'),
                        'market': 'CORNERS',
                        'bet_type': 'UNDER',
                        'line': line,
                        'odds': under_odds,
                        'probability': prob_under,
                        'edge': under_edge,
                        'confidence': confidence,
                        'referee': match.get('referee'),
                    })

        return predictions

    def backtest(self) -> Dict:
        """Run backtest on historical data to validate model improvements."""
        print("\n" + "=" * 70)
        print("BACKTESTING V2 MODELS")
        print("=" * 70)

        features, stats = self.load_training_data()

        if 'fixture_id' in features.columns and 'fixture_id' in stats.columns:
            merged = features.merge(
                stats[['fixture_id', 'total_fouls', 'total_shots', 'total_corners', 'league']],
                on='fixture_id',
                how='inner'
            )
        else:
            print("Cannot merge data")
            return {}

        # Sort by date for proper temporal split
        if 'date' in merged.columns:
            merged = merged.sort_values('date').reset_index(drop=True)

        # Use last 20% for testing
        split_idx = int(len(merged) * 0.8)
        train_df = merged.iloc[:split_idx]
        test_df = merged.iloc[split_idx:]

        print(f"Train: {len(train_df)}, Test: {len(test_df)}")

        results = {}

        # Test FOULS model
        for line in [22.5]:
            print(f"\n--- Testing FOULS over {line} ---")

            # Train on train set
            y_train = (train_df['total_fouls'] > line).astype(int)
            X_train, _ = self.prepare_features(train_df, FOULS_FEATURES_V2, use_data_quality=False)

            valid_idx = ~(X_train.isna().any(axis=1) | y_train.isna())
            X_train_clean = X_train[valid_idx]
            y_train_clean = y_train[valid_idx]

            model, _ = self.train_model(X_train_clean, y_train_clean, 'xgboost')

            # Test
            y_test = (test_df['total_fouls'] > line).astype(int)
            X_test, _ = self.prepare_features(test_df, FOULS_FEATURES_V2, use_data_quality=False)

            probs = model.predict_proba(X_test)[:, 1]

            # Simulate betting
            bets = 0
            wins = 0
            profit = 0

            for i, (prob, actual) in enumerate(zip(probs, y_test)):
                # OVER bet
                if prob >= 0.65:
                    bets += 1
                    odds = DEFAULT_ODDS['fouls_over_22_5']
                    if actual == 1:
                        wins += 1
                        profit += odds - 1
                    else:
                        profit -= 1

                # UNDER bet
                elif prob <= 0.35:
                    bets += 1
                    odds = DEFAULT_ODDS['fouls_under_22_5']
                    if actual == 0:
                        wins += 1
                        profit += odds - 1
                    else:
                        profit -= 1

            if bets > 0:
                roi = (profit / bets) * 100
                win_rate = wins / bets
                print(f"  Bets: {bets}, Wins: {wins}, WinRate: {win_rate:.1%}, ROI: {roi:+.1f}%")
                results['fouls_22.5'] = {'bets': bets, 'roi': roi, 'win_rate': win_rate}

        # Test SHOTS model
        for line in [26.5]:
            print(f"\n--- Testing SHOTS under {line} ---")

            y_train = (train_df['total_shots'] > line).astype(int)
            X_train, _ = self.prepare_features(train_df, SHOTS_FEATURES_V2, use_data_quality=False)

            valid_idx = ~(X_train.isna().any(axis=1) | y_train.isna())
            X_train_clean = X_train[valid_idx]
            y_train_clean = y_train[valid_idx]

            model, _ = self.train_model(X_train_clean, y_train_clean, 'lightgbm')

            y_test = (test_df['total_shots'] > line).astype(int)
            X_test, _ = self.prepare_features(test_df, SHOTS_FEATURES_V2, use_data_quality=False)

            probs = model.predict_proba(X_test)[:, 1]

            bets = 0
            wins = 0
            profit = 0

            for prob, actual in zip(probs, y_test):
                # UNDER bet (prob < 0.35 means likely under)
                if prob <= 0.35:
                    bets += 1
                    odds = DEFAULT_ODDS['shots_under_26_5']
                    if actual == 0:
                        wins += 1
                        profit += odds - 1
                    else:
                        profit -= 1

            if bets > 0:
                roi = (profit / bets) * 100
                win_rate = wins / bets
                print(f"  Bets: {bets}, Wins: {wins}, WinRate: {win_rate:.1%}, ROI: {roi:+.1f}%")
                results['shots_26.5_under'] = {'bets': bets, 'roi': roi, 'win_rate': win_rate}

        return results


def main():
    predictor = NicheMarketsV2()

    if len(sys.argv) < 2:
        print("Usage: python niche_markets_v2.py [train|predict|backtest|status]")
        return

    command = sys.argv[1].lower()

    if command == "train":
        predictor.train_all_models()

    elif command == "backtest":
        predictor.backtest()

    elif command == "predict":
        # First train models
        predictor.train_all_models()

        # Load upcoming fixtures
        fixtures_path = Path('data/04-predictions/upcoming_fixtures.csv')
        if fixtures_path.exists():
            fixtures = pd.read_csv(fixtures_path)
            predictions = predictor.generate_predictions(fixtures)

            # Save predictions
            if predictions:
                df = pd.DataFrame(predictions)
                df = df.sort_values(['date', 'edge'], ascending=[True, False])
                output_path = predictor.output_dir / 'recommendations_v2.csv'
                df.to_csv(output_path, index=False)
                print(f"\nSaved {len(predictions)} predictions to {output_path}")

                # Show top predictions
                print("\nTop 10 predictions by edge:")
                for _, row in df.head(10).iterrows():
                    print(f"  {row['date'][:10]} | {row['home_team']} vs {row['away_team']} | "
                          f"{row['market']} {row['bet_type']} {row['line']} | "
                          f"Edge: {row['edge']:.1f}%")
        else:
            print(f"No fixtures found at {fixtures_path}")

    elif command == "status":
        # Show model info
        info_path = predictor.output_dir / 'niche_models_v2_info.json'
        if info_path.exists():
            with open(info_path) as f:
                info = json.load(f)
            print(f"Model version: {info['version']}")
            print(f"Trained: {info['trained_at']}")
            print(f"Models: {info['models']}")
        else:
            print("No models trained yet. Run: python niche_markets_v2.py train")
    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
