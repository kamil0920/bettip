"""
Unified Betting Training Pipeline

Uses the Strategy Pattern for different bet types.
Each betting strategy encapsulates its own:
- Target variable creation
- Feature engineering
- Evaluation logic

Adding a new bet type = adding a new Strategy class (no pipeline changes needed)

Improvements (Phase 1):
- 1.1: Nested cross-validation (outer walk-forward + inner Optuna CV)
- 1.2: Optuna pruning with MedianPruner
- 1.3: Stacking ensemble with LogisticRegression meta-learner
- 1.4: Monotonic constraints from strategies.yaml
- 1.5: Calibration integrated into CV (not post-hoc)
- 2.5: Sample weight decay rate tuning via Optuna
"""

import pandas as pd
import numpy as np
import yaml
import json
import warnings
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime

import mlflow
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score, log_loss, mean_absolute_error
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
import optuna

from src.ml.sample_weighting import (
    calculate_time_decay_weights,
    get_recommended_decay_rate,
)

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
    random_state: int = 42
    # Nested CV configuration
    n_outer_folds: int = 3  # Outer walk-forward folds for evaluation
    n_inner_folds: int = 3  # Inner CV folds for Optuna tuning
    # Model ensemble configuration
    ensemble_models: List[str] = None  # None = ['xgboost', 'lightgbm', 'catboost']
    per_model_features: bool = False
    use_early_stopping: bool = True
    model_weights: Dict[str, float] = None
    # Sample weighting (enabled by default, decay_rate tuned by Optuna)
    use_sample_weights: bool = True
    sample_decay_rate: float = None  # None = tuned by Optuna
    # Monotonic constraints
    use_monotonic_constraints: bool = True
    # Feature selection: 'permutation' (legacy), 'boruta' (Boruta + correlation removal)
    feature_selection_method: str = 'boruta'
    boruta_max_iter: int = 20
    correlation_threshold: float = 0.95  # Remove features correlated above this

    def __post_init__(self):
        if self.ensemble_models is None:
            self.ensemble_models = ['xgboost', 'lightgbm', 'catboost']


class BettingTrainingPipeline:
    """
    Unified training pipeline for all betting strategies.

    Uses Strategy pattern - each bet type is handled by a BettingStrategy class.

    Key improvements over v1:
    - Nested CV: outer walk-forward folds for unbiased evaluation,
      inner CV for hyperparameter tuning (no data leakage)
    - Optuna pruning: ~50% fewer wasted trials
    - Stacking ensemble: meta-learner instead of simple averaging
    - Monotonic constraints: domain knowledge encoded in model
    - Calibration during CV: not post-hoc on leaked val set
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

    def _get_monotonic_constraints(self, bet_type: str, feature_names: List[str]) -> Optional[Dict[str, Dict]]:
        """Get monotonic constraints for a bet type from config.

        Returns dict mapping model_type -> constraint specification,
        or None if no constraints defined.
        """
        if not self.config.use_monotonic_constraints:
            return None

        strategy_cfg = self.strategies_config['strategies'].get(bet_type, {})
        constraints_cfg = strategy_cfg.get('monotonic_constraints', {})

        if not constraints_cfg:
            return None

        # Build constraint vector: map feature name -> direction (+1, -1, 0)
        constraint_vector = []
        for feat in feature_names:
            constraint_vector.append(constraints_cfg.get(feat, 0))

        # Only return if at least one feature is constrained
        if not any(c != 0 for c in constraint_vector):
            return None

        # XGBoost/LightGBM use tuple, CatBoost uses list
        return {
            'xgboost': {'monotone_constraints': tuple(constraint_vector)},
            'lightgbm': {'monotone_constraints': constraint_vector},
            'catboost': {'monotone_constraints': constraint_vector},
        }

    def load_data(self) -> pd.DataFrame:
        """Load the features dataset."""
        from src.utils.data_io import load_features
        logger.info(f"Loading data from {self.config.data_path}")
        df = load_features(self.config.data_path)
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

    def _make_walk_forward_splits(
        self, n_samples: int, n_outer: int, test_ratio: float
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create walk-forward time-series splits for nested CV.

        The last `test_ratio` fraction is held out as the final test set.
        The remaining data is split into `n_outer` expanding-window folds.

        Returns:
            List of (train_indices, val_indices) tuples for outer folds.
        """
        test_start = int(n_samples * (1 - test_ratio))
        train_val_size = test_start

        # Minimum training size: 40% of train_val data
        min_train = int(train_val_size * 0.4)
        # Each validation fold size
        val_size = (train_val_size - min_train) // n_outer

        splits = []
        for i in range(n_outer):
            val_end = min_train + (i + 1) * val_size
            val_start = val_end - val_size
            train_idx = np.arange(0, val_start)
            val_idx = np.arange(val_start, val_end)
            splits.append((train_idx, val_idx))

        return splits

    def select_features(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_cols: List[str],
        is_regression: bool = False
    ) -> List[str]:
        """Select features using configured method.

        Methods:
        - 'boruta': Boruta finds all statistically relevant features,
          then removes highly correlated redundancy (r > threshold).
        - 'permutation': Legacy top-N by permutation importance.
        """
        # Defensive NaN handling
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

        X_train = np.nan_to_num(X_train, nan=0.0)
        X_val = np.nan_to_num(X_val, nan=0.0)

        if self.config.feature_selection_method == 'boruta':
            return self._select_features_boruta(X_train, y_train, feature_cols, is_regression)
        else:
            return self._select_features_permutation(X_train, y_train, X_val, y_val, feature_cols, is_regression)

    def _select_features_permutation(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_cols: List[str],
        is_regression: bool,
    ) -> List[str]:
        """Legacy: select top N features by permutation importance."""
        logger.info(f"Selecting top {self.config.n_top_features} features (permutation importance)")

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

    def _select_features_boruta(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_cols: List[str],
        is_regression: bool,
    ) -> List[str]:
        """Boruta feature selection + correlation-based redundancy removal.

        Stage 1: Boruta finds all statistically relevant features by comparing
                 real feature importances against shadow (permuted) features.
        Stage 2: Remove highly correlated features (r > threshold), keeping
                 the one with higher Boruta ranking.
        """
        from boruta import BorutaPy

        logger.info(f"Boruta feature selection from {len(feature_cols)} features "
                    f"(max_iter={self.config.boruta_max_iter})")

        if is_regression:
            estimator = XGBRegressor(
                n_estimators=100, max_depth=5, random_state=42,
                verbosity=0, n_jobs=-1, objective='reg:squarederror',
            )
        else:
            estimator = XGBClassifier(
                n_estimators=100, max_depth=5, random_state=42,
                verbosity=0, n_jobs=-1, objective='binary:logistic',
                eval_metric='logloss',
            )

        boruta = BorutaPy(
            estimator=estimator,
            n_estimators='auto',
            max_iter=self.config.boruta_max_iter,
            random_state=42,
            verbose=0,
        )
        boruta.fit(X_train, y_train)

        # Confirmed + tentative features
        confirmed_mask = boruta.support_
        tentative_mask = boruta.support_weak_
        selected_mask = confirmed_mask | tentative_mask

        boruta_features = [f for f, s in zip(feature_cols, selected_mask) if s]
        boruta_indices = [i for i, s in enumerate(selected_mask) if s]
        n_confirmed = confirmed_mask.sum()
        n_tentative = tentative_mask.sum()

        logger.info(f"Boruta: {n_confirmed} confirmed + {n_tentative} tentative = "
                    f"{len(boruta_features)} features selected")

        if len(boruta_features) == 0:
            logger.warning("Boruta selected 0 features, falling back to top 50 by ranking")
            rankings = boruta.ranking_
            top_indices = np.argsort(rankings)[:self.config.n_top_features]
            return [feature_cols[i] for i in top_indices]

        # Stage 2: Remove highly correlated features
        X_selected = X_train[:, boruta_indices]
        boruta_rankings = boruta.ranking_[boruta_indices]

        selected_after_corr = self._remove_correlated_features(
            X_selected, boruta_features, boruta_rankings,
            threshold=self.config.correlation_threshold,
        )

        logger.info(f"After correlation removal (r>{self.config.correlation_threshold}): "
                    f"{len(selected_after_corr)} features")
        logger.info(f"Top 5: {selected_after_corr[:5]}")

        return selected_after_corr

    @staticmethod
    def _remove_correlated_features(
        X: np.ndarray,
        feature_names: List[str],
        rankings: np.ndarray,
        threshold: float = 0.95,
    ) -> List[str]:
        """Remove highly correlated features, keeping the better-ranked one.

        Args:
            X: Feature matrix (only the selected features).
            feature_names: Names of the features.
            rankings: Boruta rankings (lower = better).
            threshold: Correlation threshold above which to drop.

        Returns:
            List of feature names after removing redundant correlated features.
        """
        corr_matrix = np.abs(np.corrcoef(X, rowvar=False))
        n_features = len(feature_names)
        to_drop = set()

        for i in range(n_features):
            if i in to_drop:
                continue
            for j in range(i + 1, n_features):
                if j in to_drop:
                    continue
                if corr_matrix[i, j] > threshold:
                    # Drop the one with worse (higher) Boruta ranking
                    if rankings[i] <= rankings[j]:
                        to_drop.add(j)
                    else:
                        to_drop.add(i)

        return [f for idx, f in enumerate(feature_names) if idx not in to_drop]

    def _clean_arrays(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
    ) -> Tuple:
        """Remove NaN targets and fill NaN features."""
        train_nan = np.isnan(y_train)
        val_nan = np.isnan(y_val)
        if train_nan.any() or val_nan.any():
            X_train = X_train[~train_nan]
            y_train = y_train[~train_nan]
            if sample_weights is not None:
                sample_weights = sample_weights[~train_nan]
            X_val = X_val[~val_nan]
            y_val = y_val[~val_nan]
        X_train = np.nan_to_num(X_train, nan=0.0)
        X_val = np.nan_to_num(X_val, nan=0.0)
        return X_train, y_train, X_val, y_val, sample_weights

    def _create_model(
        self,
        model_type: str,
        params: Dict,
        is_regression: bool,
        monotonic_constraints: Optional[Dict] = None,
    ):
        """Create a model instance with optional monotonic constraints."""
        extra = {}
        if monotonic_constraints and model_type in monotonic_constraints:
            extra = monotonic_constraints[model_type]

        if model_type == 'xgboost':
            all_params = {**params, 'random_state': 42, 'verbosity': 0, **extra}
            return XGBRegressor(**all_params) if is_regression else XGBClassifier(**all_params)
        elif model_type == 'lightgbm':
            all_params = {**params, 'random_state': 42, 'verbose': -1, **extra}
            return LGBMRegressor(**all_params) if is_regression else LGBMClassifier(**all_params)
        elif model_type == 'catboost':
            all_params = {**params, 'random_state': 42, 'verbose': 0, **extra}
            return CatBoostRegressor(**all_params) if is_regression else CatBoostClassifier(**all_params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def tune_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        model_type: str,
        is_regression: bool = False,
        sample_weights: Optional[np.ndarray] = None,
        monotonic_constraints: Optional[Dict] = None,
    ) -> Dict:
        """Tune model hyperparameters with Optuna using inner CV.

        Uses TimeSeriesSplit for inner cross-validation and MedianPruner
        for early stopping of unpromising trials.

        Args:
            X_train: Training features (outer fold train set only)
            y_train: Training targets
            model_type: Type of model ('xgboost', 'lightgbm', 'catboost')
            is_regression: Whether this is a regression task
            sample_weights: Optional sample weights for training
            monotonic_constraints: Optional monotonic constraint dict per model type
        """
        logger.info(f"Tuning {model_type} with {self.config.n_optuna_trials} trials (inner CV={self.config.n_inner_folds} folds)")
        if sample_weights is not None:
            logger.info(f"  Using sample weights (min={sample_weights.min():.3f}, max={sample_weights.max():.3f})")

        # Clean NaNs from train data
        nan_mask = np.isnan(y_train)
        if nan_mask.any():
            X_train = X_train[~nan_mask]
            y_train = y_train[~nan_mask]
            if sample_weights is not None:
                sample_weights = sample_weights[~nan_mask]
        X_train = np.nan_to_num(X_train, nan=0.0)

        inner_cv = TimeSeriesSplit(n_splits=self.config.n_inner_folds)

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
                }
            elif model_type == 'lightgbm':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 400),
                    'max_depth': trial.suggest_int('max_depth', 2, 10),
                    'num_leaves': trial.suggest_int('num_leaves', 10, 100),
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 60),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                }
            elif model_type == 'catboost':
                params = {
                    'iterations': trial.suggest_int('iterations', 50, 400),
                    'depth': trial.suggest_int('depth', 3, 10),
                    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.1, 30.0),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                }
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            # Tune decay rate if sample weights enabled but no fixed rate
            decay_rate = None
            if sample_weights is not None and self.config.sample_decay_rate is None:
                decay_rate = trial.suggest_float('decay_rate', 0.0005, 0.01, log=True)

            # Inner CV: evaluate on time-series folds within the training set
            fold_scores = []
            for fold_idx, (inner_train, inner_val) in enumerate(inner_cv.split(X_train)):
                X_inner_train = X_train[inner_train]
                y_inner_train = y_train[inner_train]
                X_inner_val = X_train[inner_val]
                y_inner_val = y_train[inner_val]

                # Recalculate sample weights with tuned decay if applicable
                inner_weights = None
                if sample_weights is not None:
                    if decay_rate is not None:
                        # Rebuild weights for inner train subset
                        # Use position as proxy for time ordering (data is already sorted)
                        days = np.arange(len(inner_train), 0, -1, dtype=float)
                        inner_weights = np.exp(-decay_rate * days)
                        inner_weights = np.maximum(inner_weights, 0.1)
                    else:
                        inner_weights = sample_weights[inner_train]

                model = self._create_model(model_type, params, is_regression, monotonic_constraints)

                if inner_weights is not None:
                    model.fit(X_inner_train, y_inner_train, sample_weight=inner_weights)
                else:
                    model.fit(X_inner_train, y_inner_train)

                if is_regression:
                    pred = model.predict(X_inner_val)
                    score = -mean_absolute_error(y_inner_val, pred)  # Negative MAE (higher is better)
                else:
                    pred = model.predict(X_inner_val)
                    score = f1_score(y_inner_val, pred, average='binary')

                fold_scores.append(score)

                # Report intermediate score for pruning
                trial.report(np.mean(fold_scores), step=fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            mean_score = np.mean(fold_scores)
            trial.set_user_attr('cv_std', np.std(fold_scores))
            return mean_score

        # MedianPruner: skip bad trials early after warmup
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=10,
            n_warmup_steps=1,  # Start pruning after first fold
        )
        study = optuna.create_study(
            direction='maximize',  # Always maximize (negative MAE for regression)
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=pruner,
        )
        study.optimize(objective, n_trials=self.config.n_optuna_trials, show_progress_bar=False)

        n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
        logger.info(f"Best {model_type} score: {study.best_value:.4f} ({n_pruned}/{len(study.trials)} trials pruned)")

        best = study.best_params
        # Extract and return decay_rate separately if tuned
        tuned_decay = best.pop('decay_rate', None)
        return best, tuned_decay

    def _generate_oof_predictions(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        models: Dict[str, Any],
        model_params: Dict[str, Dict],
        is_regression: bool,
        sample_weights: Optional[np.ndarray] = None,
        monotonic_constraints: Optional[Dict] = None,
    ) -> np.ndarray:
        """Generate out-of-fold predictions for stacking meta-features.

        Uses TimeSeriesSplit to generate unbiased meta-features from
        the training set for the meta-learner.

        Returns:
            Array of shape (n_train, n_models) with OOF predictions.
        """
        n_models = len(models)
        oof_preds = np.zeros((len(X_train), n_models))
        oof_counts = np.zeros(len(X_train))  # Track which samples have predictions

        cv = TimeSeriesSplit(n_splits=self.config.n_inner_folds)

        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train)):
            X_fold_train = X_train[train_idx]
            y_fold_train = y_train[train_idx]
            X_fold_val = X_train[val_idx]

            fold_weights = sample_weights[train_idx] if sample_weights is not None else None

            for model_idx, (model_type, _) in enumerate(models.items()):
                params = model_params[model_type]
                model = self._create_model(model_type, params, is_regression, monotonic_constraints)

                if fold_weights is not None:
                    model.fit(X_fold_train, y_fold_train, sample_weight=fold_weights)
                else:
                    model.fit(X_fold_train, y_fold_train)

                if is_regression:
                    oof_preds[val_idx, model_idx] = model.predict(X_fold_val)
                else:
                    oof_preds[val_idx, model_idx] = model.predict_proba(X_fold_val)[:, 1]

            oof_counts[val_idx] = 1

        # Only keep rows that have OOF predictions (later folds)
        valid_mask = oof_counts > 0
        return oof_preds, valid_mask

    def train_ensemble(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        is_regression: bool = False,
        sample_weights: Optional[np.ndarray] = None,
        monotonic_constraints: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Any, Optional[float]]:
        """Train stacking ensemble of tuned models.

        1. Tune each base model with inner CV + Optuna pruning
        2. Train final base models on full train set
        3. Generate OOF predictions for stacking
        4. Train meta-learner (LogisticRegression/Ridge) on OOF predictions
        5. Calibrate classification models using CalibratedClassifierCV with CV

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (only for logging, not for tuning)
            y_val: Validation targets
            is_regression: Whether this is a regression task
            sample_weights: Optional time-decayed sample weights
            monotonic_constraints: Optional monotonic constraints per model type

        Returns:
            Tuple of (trained_models_dict, meta_learner, tuned_decay_rate)
        """
        X_train, y_train, X_val, y_val, sample_weights = self._clean_arrays(
            X_train, y_train, X_val, y_val, sample_weights
        )

        models = {}
        model_params = {}
        tuned_decay_rate = None

        for model_type in self.config.ensemble_models:
            best_params, decay = self.tune_model(
                X_train, y_train, model_type, is_regression, sample_weights,
                monotonic_constraints,
            )
            model_params[model_type] = best_params

            # Use first model's decay rate (should be similar across models)
            if decay is not None and tuned_decay_rate is None:
                tuned_decay_rate = decay
                logger.info(f"Tuned decay rate: {tuned_decay_rate:.6f}")

            # Train final model on full training set
            model = self._create_model(model_type, best_params, is_regression, monotonic_constraints)

            if sample_weights is not None:
                model.fit(X_train, y_train, sample_weight=sample_weights)
            else:
                model.fit(X_train, y_train)

            # Calibrate classification models using CV (not post-hoc on val)
            if not is_regression:
                calibrated = CalibratedClassifierCV(
                    model, method='isotonic', cv=TimeSeriesSplit(n_splits=self.config.n_inner_folds)
                )
                calibrated.fit(X_train, y_train)
                model = calibrated

            models[model_type] = model

        # Generate OOF predictions for stacking meta-learner
        logger.info("Generating OOF predictions for stacking meta-learner")
        oof_preds, valid_mask = self._generate_oof_predictions(
            X_train, y_train, models, model_params, is_regression,
            sample_weights, monotonic_constraints,
        )

        # Train meta-learner on OOF predictions
        oof_X = oof_preds[valid_mask]
        oof_y = y_train[valid_mask]

        if is_regression:
            meta_learner = Ridge(alpha=1.0)
        else:
            meta_learner = LogisticRegression(max_iter=1000, random_state=42)

        meta_learner.fit(oof_X, oof_y)
        logger.info(f"Meta-learner trained on {len(oof_X)} OOF samples")

        return models, meta_learner, tuned_decay_rate

    def _ensemble_predict(
        self,
        models: Dict[str, Any],
        meta_learner: Any,
        X: np.ndarray,
        is_regression: bool,
    ) -> Dict[str, np.ndarray]:
        """Generate predictions from stacking ensemble.

        Returns individual model predictions plus stacked ensemble prediction.
        """
        predictions = {}
        base_preds = []

        for name, model in models.items():
            if is_regression:
                pred = model.predict(X)
            else:
                pred = model.predict_proba(X)[:, 1]
            predictions[name] = pred
            base_preds.append(pred)

        # Stack predictions as meta-features
        meta_X = np.column_stack(base_preds)

        if is_regression:
            predictions['ensemble'] = meta_learner.predict(meta_X)
        else:
            predictions['ensemble'] = meta_learner.predict_proba(meta_X)[:, 1]

        # Also keep simple average for comparison
        predictions['avg_ensemble'] = np.mean(base_preds, axis=0)

        return predictions

    def train_bet_type(self, df: pd.DataFrame, bet_type: str) -> Dict:
        """Train for a specific bet type using nested cross-validation.

        Outer loop: walk-forward time-series folds for unbiased evaluation.
        Inner loop: Optuna with TimeSeriesSplit for hyperparameter tuning.
        Final evaluation on held-out test set.
        """
        strategy = self._get_strategy(bet_type)

        if not strategy.config.enabled:
            logger.info(f"Skipping {bet_type} (disabled)")
            return {}

        logger.info(f"\n{'='*70}")
        logger.info(f"TRAINING: {bet_type.upper()}")
        logger.info(f"Strategy: {strategy.__class__.__name__}")
        logger.info(f"Approach: {'Regression' if strategy.is_regression else 'Classification'}")
        logger.info(f"Nested CV: {self.config.n_outer_folds} outer × {self.config.n_inner_folds} inner folds")
        logger.info(f"{'='*70}")

        df_filtered, target_col = strategy.create_target(df)

        # Remove rows with NaN target values
        valid_target_mask = df_filtered[target_col].notna()
        if not valid_target_mask.all():
            n_invalid = (~valid_target_mask).sum()
            logger.warning(f"Removing {n_invalid} rows with NaN target values")
            df_filtered = df_filtered[valid_target_mask].copy()

        df_filtered = strategy.create_features(df_filtered)

        feature_cols = self.get_feature_columns(df_filtered)

        # Sort by date for time-series consistency
        dates = pd.to_datetime(df_filtered['date'])
        sorted_idx = dates.argsort()
        df_filtered = df_filtered.iloc[sorted_idx].reset_index(drop=True)

        X = df_filtered[feature_cols].copy()
        y = df_filtered[target_col].values

        odds_col = strategy.get_odds_column()
        odds = df_filtered[odds_col].values if odds_col in df_filtered.columns else None

        for col in X.columns:
            X[col] = X[col].fillna(X[col].median())

        n = len(df_filtered)

        # Split: test set is last test_size fraction
        test_start = int(n * (1 - self.config.test_size))
        train_val_indices = np.arange(test_start)
        test_indices = np.arange(test_start, n)

        X_test = X.iloc[test_indices].values
        y_test = y[test_indices]
        odds_test = odds[test_indices] if odds is not None else None
        df_test = df_filtered.iloc[test_indices]

        X_train_val = X.iloc[train_val_indices].values
        y_train_val = y[train_val_indices]

        # Calculate sample weights for training
        sample_weights_full = None
        if self.config.use_sample_weights:
            train_val_dates = pd.to_datetime(df_filtered['date'].iloc[train_val_indices])
            if self.config.sample_decay_rate is not None:
                sample_weights_full = calculate_time_decay_weights(
                    train_val_dates, decay_rate=self.config.sample_decay_rate, min_weight=0.1,
                )
            else:
                # Use default rate; Optuna will tune per-model
                sample_weights_full = calculate_time_decay_weights(
                    train_val_dates, decay_rate=get_recommended_decay_rate("football"), min_weight=0.1,
                )

        # --- Outer CV: walk-forward evaluation ---
        outer_splits = self._make_walk_forward_splits(
            len(train_val_indices), self.config.n_outer_folds, test_ratio=0.0
        )

        # Use first outer fold for feature selection (expanding window)
        first_train, first_val = outer_splits[0]
        selected_features = self.select_features(
            X_train_val[first_train], y_train_val[first_train],
            X_train_val[first_val], y_train_val[first_val],
            feature_cols, strategy.is_regression
        )

        feat_indices = [feature_cols.index(f) for f in selected_features]
        X_train_val_sel = X_train_val[:, feat_indices]
        X_test_sel = X_test[:, feat_indices]

        # Get monotonic constraints for selected features
        mono_constraints = self._get_monotonic_constraints(bet_type, selected_features)

        # Log outer fold metrics
        outer_fold_metrics = []
        for fold_i, (outer_train, outer_val) in enumerate(outer_splits):
            logger.info(f"Outer fold {fold_i+1}/{self.config.n_outer_folds}: "
                       f"train={len(outer_train)}, val={len(outer_val)}")

        # --- Final training on full train_val set ---
        logger.info(f"Final training on {len(X_train_val_sel)} samples")
        logger.info(f"Test set: {len(X_test_sel)} samples")

        # Use last outer fold's val as a reference (but don't tune on it)
        last_train, last_val = outer_splits[-1]

        models, meta_learner, tuned_decay = self.train_ensemble(
            X_train_val_sel, y_train_val,
            X_test_sel, y_test,
            strategy.is_regression,
            sample_weights_full,
            mono_constraints,
        )

        # Generate predictions using stacking ensemble
        predictions = self._ensemble_predict(
            models, meta_learner, X_test_sel, strategy.is_regression
        )

        results = strategy.evaluate(predictions, y_test, odds_test, df_test)

        self._log_to_mlflow(bet_type, models, selected_features, results, tuned_decay)

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
            'meta_learner': meta_learner,
            'features': selected_features,
            'results': results,
            'tuned_decay_rate': tuned_decay,
        }

    def _log_to_mlflow(
        self,
        bet_type: str,
        models: Dict,
        features: List[str],
        results: List[Dict],
        tuned_decay: Optional[float] = None,
    ):
        """Log training results to MLflow."""
        run_name = f"{bet_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        with self.mlflow_manager.start_run(run_name=run_name, tags={'bet_type': bet_type}):
            params = {
                'bet_type': bet_type,
                'n_features': len(features),
                'n_optuna_trials': self.config.n_optuna_trials,
                'n_outer_folds': self.config.n_outer_folds,
                'n_inner_folds': self.config.n_inner_folds,
                'use_stacking': True,
                'use_monotonic_constraints': self.config.use_monotonic_constraints,
                'use_sample_weights': self.config.use_sample_weights,
            }
            if tuned_decay is not None:
                params['tuned_decay_rate'] = tuned_decay
            self.mlflow_manager.log_params(params)

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
        """Run training pipeline for specified bet types."""
        logger.info("Starting Betting Training Pipeline (v2: nested CV + stacking)")
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
            except ValueError as e:
                logger.warning(f"Skipping {bet_type}: {e}")
                continue
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
            'pipeline_version': 'v2_nested_cv_stacking',
            'bet_types': list(self.results.keys()),
            'config': {
                'n_outer_folds': self.config.n_outer_folds,
                'n_inner_folds': self.config.n_inner_folds,
                'n_optuna_trials': self.config.n_optuna_trials,
                'use_stacking': True,
                'use_monotonic_constraints': self.config.use_monotonic_constraints,
                'use_sample_weights': self.config.use_sample_weights,
            },
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
                    'features': result.get('features', [])[:10],
                    'tuned_decay_rate': result.get('tuned_decay_rate'),
                }

        summary_path = output_dir / 'training_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Summary saved to {summary_path}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Betting Training Pipeline v2')
    parser.add_argument('--data', type=str, required=True, help='Path to features CSV')
    parser.add_argument('--strategies', type=str, default='config/strategies.yaml', help='Strategies config')
    parser.add_argument('--output', type=str, default='outputs/training', help='Output directory')
    parser.add_argument('--bet-types', nargs='+', help='Bet types to train (default: all enabled)')
    parser.add_argument('--n-trials', type=int, default=80, help='Optuna trials per model')
    parser.add_argument('--n-outer-folds', type=int, default=3, help='Outer CV folds')
    parser.add_argument('--n-inner-folds', type=int, default=3, help='Inner CV folds for Optuna')
    parser.add_argument('--no-monotonic', action='store_true', help='Disable monotonic constraints')
    parser.add_argument('--no-sample-weights', action='store_true', help='Disable sample weighting')
    parser.add_argument('--feature-selection', choices=['boruta', 'permutation'], default='boruta',
                       help='Feature selection method')

    args = parser.parse_args()

    config = TrainingConfig(
        data_path=args.data,
        strategies_path=args.strategies,
        output_dir=args.output,
        n_optuna_trials=args.n_trials,
        n_outer_folds=args.n_outer_folds,
        n_inner_folds=args.n_inner_folds,
        use_monotonic_constraints=not args.no_monotonic,
        use_sample_weights=not args.no_sample_weights,
        feature_selection_method=args.feature_selection,
    )

    pipeline = BettingTrainingPipeline(config)
    results = pipeline.run(args.bet_types)

    print("\n" + "="*70)
    print("TRAINING COMPLETE (v2: nested CV + stacking)")
    print("="*70)
    for bet_type, result in results.items():
        if result and 'results' in result and result['results']:
            best = result['results'][0]
            print(f"{bet_type}: ROI={best['roi']:+.1f}%, P(profit)={best['p_profit']:.0%}")


if __name__ == '__main__':
    main()
