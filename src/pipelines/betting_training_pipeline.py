"""
Unified Betting Training Pipeline

Uses the Strategy Pattern for different bet types.
Each betting strategy encapsulates its own:
- Target variable creation
- Feature engineering
- Evaluation logic

Adding a new bet type = adding a new Strategy class (no pipeline changes needed)
"""

import pandas as pd
import numpy as np
import yaml
import json
import warnings
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime

import mlflow
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
import optuna

from src.ml.mlflow_config import get_mlflow_manager
from src.ml.betting_strategies import (
    BettingStrategy,
    StrategyConfig,
    get_strategy,
    STRATEGY_REGISTRY
)
from src.ml.model_registry import (
    ModelEnsemble,
    EnsembleConfig,
    ModelConfig,
    get_model,
    MODEL_REGISTRY
)

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration."""
    data_path: str
    strategies_path: str
    output_dir: str
    n_optuna_trials: int = 80
    n_top_features: int = 50
    test_size: float = 0.2
    val_size: float = 0.2
    random_state: int = 42
    # Model ensemble configuration
    ensemble_models: List[str] = None  # None = ['xgboost', 'lightgbm', 'catboost']
    per_model_features: bool = False  # Each model selects its own features
    use_early_stopping: bool = True
    model_weights: Dict[str, float] = None  # Custom weights for ensemble

    def __post_init__(self):
        if self.ensemble_models is None:
            self.ensemble_models = ['xgboost', 'lightgbm', 'catboost']


class BettingTrainingPipeline:
    """
    Unified training pipeline for all betting strategies.

    Uses Strategy pattern - each bet type is handled by a BettingStrategy class.
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.mlflow_manager = get_mlflow_manager()
        self.strategies_config = self._load_strategies_config()
        self.results = {}

    def _load_strategies_config(self) -> Dict:
        """Load betting strategies configuration from YAML."""
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
            expected_roi=strategy_cfg.get('expected_roi', 0.0),
            p_profit=strategy_cfg.get('p_profit', 0.0),
        )

        return get_strategy(bet_type, config)

    def load_data(self) -> pd.DataFrame:
        """Load the features dataset."""
        logger.info(f"Loading data from {self.config.data_path}")
        df = pd.read_csv(self.config.data_path)
        logger.info(f"Loaded {len(df)} matches")
        return df

    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get feature columns, excluding targets and odds."""
        exclude_patterns = [
            'fixture_id', 'date', 'team_id', 'team_name', 'round', 'match_result',
            'home_win', 'draw', 'away_win', 'total_goals', 'goal_difference',
            'league', 'target', 'btts', 'goal_margin', 'ah_result',
            'home_goals', 'away_goals', 'home_score', 'away_score', 'result',
            'b365', 'avg_', 'max_', 'pinnacle', 'ah_line',
        ]

        feature_cols = []
        for col in df.columns:
            if df[col].dtype not in ['float64', 'int64', 'float32', 'int32']:
                continue
            if any(pattern in col.lower() for pattern in exclude_patterns):
                continue
            if df[col].notna().sum() > len(df) * 0.5:
                feature_cols.append(col)

        return feature_cols

    def time_split(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create time-based train/val/test split indices."""
        dates = pd.to_datetime(df['date'])
        sorted_idx = dates.argsort()
        n = len(df)

        train_end = int(n * (1 - self.config.test_size - self.config.val_size))
        val_end = int(n * (1 - self.config.test_size))

        train_idx = sorted_idx[:train_end]
        val_idx = sorted_idx[train_end:val_end]
        test_idx = sorted_idx[val_end:]

        return train_idx, val_idx, test_idx

    def select_features(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_cols: List[str],
        is_regression: bool = False
    ) -> List[str]:
        """Select top features using permutation importance."""
        logger.info(f"Selecting top {self.config.n_top_features} features from {len(feature_cols)}")

        # Defensive NaN handling - XGBoost fails with NaN in target
        train_nan_mask = np.isnan(y_train)
        val_nan_mask = np.isnan(y_val)
        if train_nan_mask.any() or val_nan_mask.any():
            n_train_nan = train_nan_mask.sum()
            n_val_nan = val_nan_mask.sum()
            logger.warning(f"Found NaN values in target: {n_train_nan} train, {n_val_nan} val - removing")
            X_train = X_train[~train_nan_mask]
            y_train = y_train[~train_nan_mask]
            X_val = X_val[~val_nan_mask]
            y_val = y_val[~val_nan_mask]

        # Also handle NaN in features
        X_train = np.nan_to_num(X_train, nan=0.0)
        X_val = np.nan_to_num(X_val, nan=0.0)

        if is_regression:
            model = XGBRegressor(n_estimators=100, max_depth=5, random_state=42, verbosity=0, objective='reg:squarederror')
        else:
            model = XGBClassifier(n_estimators=100, max_depth=5, random_state=42, verbosity=0, objective='binary:logistic', eval_metric='logloss')

        model.fit(X_train, y_train)
        perm = permutation_importance(model, X_val, y_val, n_repeats=15, random_state=42, n_jobs=-1)

        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': perm.importances_mean
        }).sort_values('importance', ascending=False)

        selected = importance_df.head(self.config.n_top_features)['feature'].tolist()
        logger.info(f"Top 5 features: {selected[:5]}")

        return selected

    def tune_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        model_type: str,
        is_regression: bool = False
    ) -> Dict:
        """Tune model hyperparameters with Optuna."""
        logger.info(f"Tuning {model_type} with {self.config.n_optuna_trials} trials")

        # Defensive NaN handling
        train_nan_mask = np.isnan(y_train)
        val_nan_mask = np.isnan(y_val)
        if train_nan_mask.any() or val_nan_mask.any():
            X_train = X_train[~train_nan_mask]
            y_train = y_train[~train_nan_mask]
            X_val = X_val[~val_nan_mask]
            y_val = y_val[~val_nan_mask]
        X_train = np.nan_to_num(X_train, nan=0.0)
        X_val = np.nan_to_num(X_val, nan=0.0)

        def objective(trial):
            if model_type == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 400),
                    'max_depth': trial.suggest_int('max_depth', 2, 8),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 30),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'random_state': 42, 'verbosity': 0
                }
                model = XGBRegressor(**params) if is_regression else XGBClassifier(**params)

            elif model_type == 'lightgbm':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 400),
                    'max_depth': trial.suggest_int('max_depth', 2, 10),
                    'num_leaves': trial.suggest_int('num_leaves', 10, 100),
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 60),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                    'random_state': 42, 'verbose': -1
                }
                model = LGBMRegressor(**params) if is_regression else LGBMClassifier(**params)

            elif model_type == 'catboost':
                params = {
                    'iterations': trial.suggest_int('iterations', 50, 400),
                    'depth': trial.suggest_int('depth', 3, 10),
                    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.1, 30.0),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                    'random_state': 42, 'verbose': 0
                }
                model = CatBoostRegressor(**params) if is_regression else CatBoostClassifier(**params)

            else:
                raise ValueError(f"Unknown model type: {model_type}")

            model.fit(X_train, y_train)

            if is_regression:
                pred = model.predict(X_val)
                return np.mean(np.abs(pred - y_val))  # MAE for regression
            else:
                # Use F1 score instead of accuracy for imbalanced betting data
                from sklearn.metrics import f1_score
                pred = model.predict(X_val)
                return f1_score(y_val, pred, average='binary')

        direction = 'minimize' if is_regression else 'maximize'
        study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=self.config.n_optuna_trials, show_progress_bar=False)

        logger.info(f"Best {model_type} score: {study.best_value:.4f}")
        return study.best_params

    def train_ensemble(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        is_regression: bool = False
    ) -> Dict[str, Any]:
        """Train ensemble of models."""
        # Defensive NaN handling
        train_nan_mask = np.isnan(y_train)
        val_nan_mask = np.isnan(y_val)
        if train_nan_mask.any() or val_nan_mask.any():
            X_train = X_train[~train_nan_mask]
            y_train = y_train[~train_nan_mask]
            X_val = X_val[~val_nan_mask]
            y_val = y_val[~val_nan_mask]
        X_train = np.nan_to_num(X_train, nan=0.0)
        X_val = np.nan_to_num(X_val, nan=0.0)

        models = {}

        for model_type in ['xgboost', 'lightgbm', 'catboost']:
            best_params = self.tune_model(X_train, y_train, X_val, y_val, model_type, is_regression)

            if model_type == 'xgboost':
                params = {**best_params, 'random_state': 42, 'verbosity': 0}
                model = XGBRegressor(**params) if is_regression else XGBClassifier(**params)
            elif model_type == 'lightgbm':
                params = {**best_params, 'random_state': 42, 'verbose': -1}
                model = LGBMRegressor(**params) if is_regression else LGBMClassifier(**params)
            else:
                params = {**best_params, 'random_state': 42, 'verbose': 0}
                model = CatBoostRegressor(**params) if is_regression else CatBoostClassifier(**params)

            model.fit(X_train, y_train)

            if not is_regression:
                model = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
                model.fit(X_val, y_val)

            models[model_type] = model

        return models

    def train_bet_type(self, df: pd.DataFrame, bet_type: str) -> Dict:
        """
        Train models for a specific bet type using its strategy.

        Args:
            df: Features DataFrame
            bet_type: Type of bet (e.g., 'asian_handicap', 'btts')

        Returns:
            Training results dict
        """
        strategy = self._get_strategy(bet_type)

        if not strategy.config.enabled:
            logger.info(f"Skipping {bet_type} (disabled)")
            return {}

        logger.info(f"\n{'='*70}")
        logger.info(f"TRAINING: {bet_type.upper()}")
        logger.info(f"Strategy: {strategy.__class__.__name__}")
        logger.info(f"Approach: {'Regression' if strategy.is_regression else 'Classification'}")
        logger.info(f"{'='*70}")

        df_filtered, target_col = strategy.create_target(df)

        # Remove rows with NaN target values (critical for XGBoost classification)
        valid_target_mask = df_filtered[target_col].notna()
        if not valid_target_mask.all():
            n_invalid = (~valid_target_mask).sum()
            logger.warning(f"Removing {n_invalid} rows with NaN target values")
            df_filtered = df_filtered[valid_target_mask].copy()

        df_filtered = strategy.create_features(df_filtered)

        feature_cols = self.get_feature_columns(df_filtered)

        X = df_filtered[feature_cols].copy()
        y = df_filtered[target_col].values

        odds_col = strategy.get_odds_column()
        odds = df_filtered[odds_col].values if odds_col in df_filtered.columns else None

        for col in X.columns:
            X[col] = X[col].fillna(X[col].median())

        train_idx, val_idx, test_idx = self.time_split(df_filtered)
        X_train, y_train = X.iloc[train_idx].values, y[train_idx]
        X_val, y_val = X.iloc[val_idx].values, y[val_idx]
        X_test, y_test = X.iloc[test_idx].values, y[test_idx]
        odds_test = odds[test_idx] if odds is not None else None
        df_test = df_filtered.iloc[test_idx]

        logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        selected_features = self.select_features(
            X_train, y_train, X_val, y_val, feature_cols, strategy.is_regression
        )

        feat_indices = [feature_cols.index(f) for f in selected_features]
        X_train_sel = X_train[:, feat_indices]
        X_val_sel = X_val[:, feat_indices]
        X_test_sel = X_test[:, feat_indices]

        models = self.train_ensemble(
            X_train_sel, y_train, X_val_sel, y_val, strategy.is_regression
        )

        predictions = {}
        for name, model in models.items():
            if strategy.is_regression:
                predictions[name] = model.predict(X_test_sel)
            else:
                predictions[name] = model.predict_proba(X_test_sel)[:, 1]

        predictions['ensemble'] = np.mean(list(predictions.values()), axis=0)

        results = strategy.evaluate(predictions, y_test, odds_test, df_test)

        self._log_to_mlflow(bet_type, models, selected_features, results)

        if results:
            best = results[0]
            logger.info(
                f"Best: {best['model']} thresh={best['threshold']} | "
                f"ROI: {best['roi']:+.1f}% | P(profit): {best['p_profit']:.0%}"
            )

        return {
            'bet_type': bet_type,
            'strategy': strategy.__class__.__name__,
            'models': models,
            'features': selected_features,
            'results': results
        }

    def _log_to_mlflow(
        self,
        bet_type: str,
        models: Dict,
        features: List[str],
        results: List[Dict]
    ):
        """Log training results to MLflow."""
        run_name = f"{bet_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        with self.mlflow_manager.start_run(run_name=run_name, tags={'bet_type': bet_type}):
            self.mlflow_manager.log_params({
                'bet_type': bet_type,
                'n_features': len(features),
                'n_optuna_trials': self.config.n_optuna_trials,
            })

            if results:
                best = results[0]
                self.mlflow_manager.log_metrics({
                    'best_roi': best['roi'],
                    'best_p_profit': best['p_profit'],
                    'best_bets': best['bets'],
                    'best_win_rate': best['win_rate'],
                })

            for name, model in models.items():
                model_name = f"{bet_type}_{name}"
                self.mlflow_manager.log_model(
                    model,
                    artifact_path=f"models/{name}",
                    registered_model_name=model_name
                )

            self.mlflow_manager.log_dict({'features': features}, 'features.json')
            self.mlflow_manager.log_dict({'results': results}, 'results.json')

    def run(self, bet_types: Optional[List[str]] = None) -> Dict:
        """
        Run training pipeline for specified bet types.

        Args:
            bet_types: List of bet types to train. If None, trains all enabled strategies.

        Returns:
            Dict of results for each bet type
        """
        logger.info("Starting Betting Training Pipeline")
        logger.info(f"Available strategies: {list(STRATEGY_REGISTRY.keys())}")

        df = self.load_data()

        if bet_types is None:
            bet_types = [
                bt for bt, cfg in self.strategies_config['strategies'].items()
                if cfg.get('enabled', False)
            ]

        logger.info(f"Training bet types: {bet_types}")

        for bet_type in bet_types:
            try:
                result = self.train_bet_type(df, bet_type)
                self.results[bet_type] = result
            except Exception as e:
                logger.error(f"Error training {bet_type}: {e}")
                raise

        self._save_summary()

        return self.results

    def _save_summary(self):
        """Save training summary."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        summary = {
            'timestamp': datetime.now().isoformat(),
            'bet_types': list(self.results.keys()),
            'results': {}
        }

        for bet_type, result in self.results.items():
            if result and 'results' in result:
                best = result['results'][0] if result['results'] else {}
                summary['results'][bet_type] = {
                    'strategy': result.get('strategy'),
                    'best_model': best.get('model'),
                    'best_threshold': best.get('threshold'),
                    'roi': best.get('roi'),
                    'p_profit': best.get('p_profit'),
                    'features': result.get('features', [])[:10]  # Top 10
                }

        summary_path = output_dir / 'training_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Summary saved to {summary_path}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Betting Training Pipeline')
    parser.add_argument('--data', type=str, required=True, help='Path to features CSV')
    parser.add_argument('--strategies', type=str, default='config/strategies.yaml', help='Strategies config')
    parser.add_argument('--output', type=str, default='outputs/training', help='Output directory')
    parser.add_argument('--bet-types', nargs='+', help='Bet types to train (default: all enabled)')
    parser.add_argument('--n-trials', type=int, default=80, help='Optuna trials per model')

    args = parser.parse_args()

    config = TrainingConfig(
        data_path=args.data,
        strategies_path=args.strategies,
        output_dir=args.output,
        n_optuna_trials=args.n_trials
    )

    pipeline = BettingTrainingPipeline(config)
    results = pipeline.run(args.bet_types)

    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    for bet_type, result in results.items():
        if result and 'results' in result and result['results']:
            best = result['results'][0]
            print(f"{bet_type}: ROI={best['roi']:+.1f}%, P(profit)={best['p_profit']:.0%}")


if __name__ == '__main__':
    main()
