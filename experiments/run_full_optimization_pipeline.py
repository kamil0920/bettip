"""
Full Optimization Pipeline for Any Bet Type

Based on "Effective XGBoost" book by Matt Harrison.

Usage:
    # Traditional markets
    python run_full_optimization_pipeline.py --bet_type asian_handicap
    python run_full_optimization_pipeline.py --bet_type home_win
    python run_full_optimization_pipeline.py --bet_type over25
    python run_full_optimization_pipeline.py --bet_type under25
    python run_full_optimization_pipeline.py --bet_type btts --n_trials 150

    # Niche markets (corners, shots, fouls, cards)
    python run_full_optimization_pipeline.py --bet_type corners
    python run_full_optimization_pipeline.py --bet_type shots
    python run_full_optimization_pipeline.py --bet_type fouls
    python run_full_optimization_pipeline.py --bet_type cards

Pipeline:
1. Permutation importance feature selection PER MODEL
2. Optuna tuning per model with its own best features (default 150 trials)
   - Uses early stopping during tuning (prevents overfitting)
   - Optimized parameter ranges from "Effective XGBoost" book
   - Optional step-wise tuning for faster optimization
3. Probability calibration
4. Stacking ensemble
5. Betting optimization with bootstrap CI
"""
import argparse
import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV, Ridge
from src.calibration.calibration import BetaCalibrator, calibration_metrics
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from boruta import BorutaPy
from sklearn.metrics import mean_absolute_error
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping as lgb_early_stopping
from catboost import CatBoostClassifier, CatBoostRegressor
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_match_stats():
    """Load and combine match_stats from all leagues/seasons."""
    all_stats = []
    base_dir = Path('data/01-raw')
    if not base_dir.exists():
        base_dir = Path(__file__).parent.parent / 'data/01-raw'

    for league in ['premier_league', 'la_liga', 'serie_a', 'bundesliga', 'ligue_1']:
        league_dir = base_dir / league
        if not league_dir.exists():
            continue
        for season_dir in league_dir.iterdir():
            if not season_dir.is_dir():
                continue
            stats_path = season_dir / 'match_stats.parquet'
            if stats_path.exists():
                try:
                    df = pd.read_parquet(stats_path)
                    all_stats.append(df)
                except Exception:
                    pass

    if not all_stats:
        return pd.DataFrame()
    return pd.concat(all_stats, ignore_index=True)


def load_data(bet_type, data_path=None):
    """Load data and create target for specific bet type."""
    if data_path is None:
        # Try relative path first, then absolute
        candidates = [
            Path('data/03-features/features_all_5leagues_with_odds.csv'),
            Path(__file__).parent.parent / 'data/03-features/features_all_5leagues_with_odds.csv',
        ]
        for p in candidates:
            if p.exists():
                data_path = p
                break
        else:
            raise FileNotFoundError("Features file not found")
    df = pd.read_csv(data_path, low_memory=False)

    is_regression = False  # Default to classification

    if bet_type == 'asian_handicap':
        # Filter matches with AH odds
        df_filtered = df[df['ah_line'].notna() & df['avg_ah_home'].notna()].copy()

        # Regression target: predict goal margin (goal_difference)
        # Betting decision: if predicted_margin > -ah_line, bet home covers
        df_filtered['target'] = df_filtered['goal_difference'].astype(float)
        is_regression = True

        # Odds columns
        odds_col_bet = 'avg_ah_home'  # Bet on home covering
        odds_col_opposite = 'avg_ah_away'  # Bet on away covering

        # Exclude AH odds from features
        exclude_odds = ['ah_line', 'b365_ah_home', 'b365_ah_away', 'avg_ah_home', 'avg_ah_away',
                       'pinnacle_ah_home', 'pinnacle_ah_away', 'max_ah_home', 'max_ah_away',
                       'ah_line_close', 'b365_ah_home_close', 'b365_ah_away_close',
                       'avg_ah_home_close', 'avg_ah_away_close']

        target_name = 'Asian Handicap (Goal Margin Regression)'
        base_rate = df_filtered['target'].mean()  # Average goal margin

    elif bet_type == 'home_win':
        df_filtered = df[df['avg_home_open'].notna()].copy()
        df_filtered['target'] = df_filtered['home_win'].astype(int)

        odds_col_bet = 'avg_home_open'
        odds_col_opposite = 'avg_away_open'

        exclude_odds = ['b365_home_open', 'b365_draw_open', 'b365_away_open',
                       'avg_home_open', 'avg_draw_open', 'avg_away_open',
                       'b365_home_close', 'b365_draw_close', 'b365_away_close',
                       'avg_home_close', 'avg_draw_close', 'avg_away_close']

        target_name = 'Home Win'
        base_rate = df_filtered['target'].mean()

    elif bet_type == 'over25':
        df_filtered = df[df['avg_over25'].notna()].copy()
        df_filtered['target'] = (df_filtered['total_goals'] > 2.5).astype(int)

        odds_col_bet = 'avg_over25'
        odds_col_opposite = 'avg_under25'

        exclude_odds = ['b365_over25', 'b365_under25', 'avg_over25', 'avg_under25',
                       'b365_over25_close', 'b365_under25_close', 'avg_over25_close', 'avg_under25_close']

        target_name = 'Over 2.5 Goals'
        base_rate = df_filtered['target'].mean()

    elif bet_type == 'under25':
        df_filtered = df[df['avg_under25'].notna()].copy()
        df_filtered['target'] = (df_filtered['total_goals'] <= 2.5).astype(int)

        odds_col_bet = 'avg_under25'
        odds_col_opposite = 'avg_over25'

        exclude_odds = ['b365_over25', 'b365_under25', 'avg_over25', 'avg_under25',
                       'b365_over25_close', 'b365_under25_close', 'avg_over25_close', 'avg_under25_close']

        target_name = 'Under 2.5 Goals'
        base_rate = df_filtered['target'].mean()

    elif bet_type == 'away_win':
        df_filtered = df[df['avg_away_open'].notna()].copy()
        df_filtered['target'] = df_filtered['away_win'].astype(int)

        odds_col_bet = 'avg_away_open'
        odds_col_opposite = 'avg_home_open'

        exclude_odds = ['b365_home_open', 'b365_draw_open', 'b365_away_open',
                       'avg_home_open', 'avg_draw_open', 'avg_away_open',
                       'b365_home_close', 'b365_draw_close', 'b365_away_close',
                       'avg_home_close', 'avg_draw_close', 'avg_away_close']

        target_name = 'Away Win'
        base_rate = df_filtered['target'].mean()

    elif bet_type == 'btts':
        # Both Teams To Score
        df_filtered = df[df['btts_yes_avg'].notna()].copy()
        df_filtered['target'] = ((df_filtered['home_goals'] > 0) &
                                 (df_filtered['away_goals'] > 0)).astype(int)

        odds_col_bet = 'btts_yes_avg'
        odds_col_opposite = 'btts_no_avg'

        exclude_odds = ['btts_yes_avg', 'btts_no_avg', 'btts_yes_max', 'btts_no_max',
                       'btts_yes_b365', 'btts_no_b365']

        target_name = 'Both Teams To Score'
        base_rate = df_filtered['target'].mean()

    # ==================== NICHE MARKETS ====================
    # These require merging match_stats data for outcomes
    elif bet_type in ['corners', 'shots', 'fouls', 'cards']:
        # Load and merge match_stats
        match_stats = load_match_stats()
        if match_stats.empty:
            raise ValueError(f"No match_stats data found for {bet_type}")

        # Merge on fixture_id
        df = df.merge(
            match_stats[['fixture_id', 'home_corners', 'away_corners',
                        'home_shots', 'away_shots', 'home_fouls', 'away_fouls']],
            on='fixture_id', how='inner'
        )
        print(f"Merged match_stats: {len(df)} matches with niche market outcomes")

        # Calculate totals
        df['total_corners'] = df['home_corners'] + df['away_corners']
        df['total_shots'] = df['home_shots'] + df['away_shots']
        df['total_fouls'] = df['home_fouls'] + df['away_fouls']

        # Niche market configurations
        # Lines based on historical distributions
        NICHE_CONFIGS = {
            'corners': {
                'line': 9.5,  # Over/Under 9.5 corners (mean ~9.7)
                'total_col': 'total_corners',
                'name': 'Over 9.5 Corners'
            },
            'shots': {
                'line': 24.5,  # Over/Under 24.5 shots (mean ~24.8)
                'total_col': 'total_shots',
                'name': 'Over 24.5 Shots'
            },
            'fouls': {
                'line': 24.5,  # Over/Under 24.5 fouls (mean ~24.5)
                'total_col': 'total_fouls',
                'name': 'Over 24.5 Fouls'
            },
            'cards': {
                'line': 4.5,  # Over/Under 4.5 cards - use fouls as proxy
                'total_col': 'total_fouls',  # Cards correlate with fouls
                'name': 'Over 4.5 Cards (Fouls Proxy)'
            }
        }

        config = NICHE_CONFIGS[bet_type]
        df_filtered = df.copy()

        # Create target: 1 if over the line
        df_filtered['target'] = (df_filtered[config['total_col']] > config['line']).astype(int)

        # No real odds for niche markets - use synthetic odds based on base rate
        # This allows ROI calculation assuming fair-ish market odds
        base_rate = df_filtered['target'].mean()
        # Synthetic odds: ~1.90 for both sides (typical vig)
        df_filtered['synthetic_odds_over'] = 1.90
        df_filtered['synthetic_odds_under'] = 1.90

        odds_col_bet = 'synthetic_odds_over'
        odds_col_opposite = 'synthetic_odds_under'

        # Exclude niche market columns from features (prevent leakage)
        exclude_odds = ['home_corners', 'away_corners', 'total_corners',
                       'home_shots', 'away_shots', 'total_shots',
                       'home_fouls', 'away_fouls', 'total_fouls',
                       'synthetic_odds_over', 'synthetic_odds_under']

        target_name = config['name']

    else:
        raise ValueError(f"Unknown bet type: {bet_type}")

    return df_filtered, odds_col_bet, odds_col_opposite, exclude_odds, target_name, base_rate, is_regression


def get_feature_columns(df, exclude_odds):
    """Get feature columns excluding identifiers and odds."""
    exclude_cols = [
        'fixture_id', 'date', 'home_team_id', 'home_team_name', 'away_team_id',
        'away_team_name', 'round', 'match_result', 'home_win', 'draw', 'away_win',
        'total_goals', 'goal_difference', 'league', 'target', 'ah_result',
        'home_goals', 'away_goals', 'season', 'round_num', 'result',
        # Prevent data leakage - these are outcome flags, not features!
        'over25', 'under25', 'btts', 'home_score', 'away_score'
    ] + exclude_odds

    # Also exclude all odds columns
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    feature_cols = [c for c in feature_cols if 'b365' not in c.lower()
                    and not (c.startswith('avg_') and any(x in c for x in ['home', 'away', 'draw', 'over', 'under', 'ah']))
                    and not (c.startswith('max_') and any(x in c for x in ['home', 'away', 'draw', 'over', 'under', 'ah']))
                    and 'pinnacle' not in c.lower()
                    and 'btts' not in c.lower()]  # Exclude BTTS odds
    feature_cols = [c for c in feature_cols if df[c].notna().sum() > len(df) * 0.5]

    # Only keep numeric columns
    numeric_cols = df[feature_cols].select_dtypes(include=['number']).columns.tolist()

    return numeric_cols


def stepwise_tune_xgboost(X_train, y_train, X_val, y_val, features, is_regression=False, trials_per_step=30):
    """
    Step-wise hyperparameter tuning for XGBoost based on "Effective XGBoost" book.

    This method tunes hyperparameters in groups:
    1. Tree parameters (max_depth, min_child_weight)
    2. Sampling parameters (subsample, colsample_bytree)
    3. Regularization parameters (reg_alpha, reg_lambda, gamma)
    4. Learning rate

    This is much faster than tuning all parameters at once (~4x speedup),
    but may find a local optimum instead of global.
    """
    direction = 'minimize' if is_regression else 'maximize'
    best_params = {'random_state': 42, 'verbosity': 0, 'n_estimators': 500, 'early_stopping_rounds': 50}

    # Define parameter groups (from the book, Chapter 13)
    param_groups = [
        # Step 1: Tree parameters
        {
            'max_depth': lambda t: t.suggest_int('max_depth', 2, 8),
            'min_child_weight': lambda t: t.suggest_float('min_child_weight', 0.1, 20.0, log=True),
        },
        # Step 2: Sampling parameters
        {
            'subsample': lambda t: t.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': lambda t: t.suggest_float('colsample_bytree', 0.5, 1.0),
        },
        # Step 3: Regularization parameters
        {
            'reg_alpha': lambda t: t.suggest_float('reg_alpha', 0.0, 10.0),
            'reg_lambda': lambda t: t.suggest_float('reg_lambda', 1.0, 10.0),
            'gamma': lambda t: t.suggest_float('gamma', 1e-8, 10.0, log=True),
        },
        # Step 4: Learning rate (tune last, as recommended by the book)
        {
            'learning_rate': lambda t: t.suggest_float('learning_rate', 0.001, 0.3, log=True),
        },
    ]

    step_names = ['Tree', 'Sampling', 'Regularization', 'Learning Rate']

    for step_idx, (param_group, step_name) in enumerate(zip(param_groups, step_names)):
        print(f"  Step {step_idx+1}/4: {step_name} parameters...")

        def objective(trial):
            # Start with current best params
            params = best_params.copy()
            # Add trial suggestions for this group
            for param_name, suggest_fn in param_group.items():
                params[param_name] = suggest_fn(trial)

            if is_regression:
                model = XGBRegressor(**params)
                model.fit(X_train[features], y_train,
                         eval_set=[(X_val[features], y_val)], verbose=False)
                pred = model.predict(X_val[features])
                return mean_absolute_error(y_val, pred)
            else:
                model = XGBClassifier(**params)
                model.fit(X_train[features], y_train,
                         eval_set=[(X_val[features], y_val)], verbose=False)
                return model.score(X_val[features], y_val)

        study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=42+step_idx))
        study.optimize(objective, n_trials=trials_per_step, show_progress_bar=False)

        # Update best_params with the best from this step
        best_params.update(study.best_params)
        print(f"    Best score: {study.best_value:.4f}")

    # Remove early_stopping_rounds from output (it's a training param, not model param)
    output_params = {k: v for k, v in best_params.items() if k != 'early_stopping_rounds'}
    return output_params


def run_pipeline(bet_type, n_trials=150, revalidate_features=False, walkforward=False, stepwise=False, optimize_sharpe=False, calibration='platt'):
    """Run full optimization pipeline for a bet type."""

    print("=" * 70)
    print(f"FULL OPTIMIZATION: {bet_type.upper()}")
    print("=" * 70)

    # Load data
    df, odds_col_bet, odds_col_opposite, exclude_odds, target_name, base_rate, is_regression = load_data(bet_type)
    feature_cols = get_feature_columns(df, exclude_odds)

    print(f"\nTarget: {target_name}")
    print(f"Task: {'Regression' if is_regression else 'Classification'}")
    print(f"Matches: {len(df)}")
    if is_regression:
        print(f"Mean target: {base_rate:.2f}")
    else:
        print(f"Base rate: {base_rate:.1%}")
    print(f"Features: {len(feature_cols)}")

    # Prepare data
    X = df[feature_cols].copy()
    if is_regression:
        y = df['target'].values.astype(float)
    else:
        y = df['target'].values.astype(int)
    odds = df[odds_col_bet].values
    ah_line = df['ah_line'].values if bet_type == 'asian_handicap' else None
    dates = pd.to_datetime(df['date'])

    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())

    # Time split
    sorted_indices = dates.argsort()
    n = len(X)
    train_idx = sorted_indices[:int(0.6*n)]
    val_idx = sorted_indices[int(0.6*n):int(0.8*n)]
    test_idx = sorted_indices[int(0.8*n):]

    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_val, y_val = X.iloc[val_idx], y[val_idx]
    X_test, y_test = X.iloc[test_idx], y[test_idx]
    odds_test = odds[test_idx]

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # ============================================================
    # STEP 1: Boruta Feature Selection
    # ============================================================
    print("\n" + "=" * 70)
    print("STEP 1: Boruta Feature Selection (finds ALL relevant features)")
    print("=" * 70)

    # Boruta uses Random Forest as base estimator
    # It compares feature importance against shadow (shuffled) features
    # to find statistically significant features

    if is_regression:
        rf_estimator = RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            n_jobs=-1,
            random_state=42
        )
    else:
        rf_estimator = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            n_jobs=-1,
            class_weight='balanced',
            random_state=42
        )

    # Run Boruta feature selection
    print("\nRunning Boruta algorithm...")
    print("  (Comparing features against shadow features to find relevant ones)")

    boruta_selector = BorutaPy(
        rf_estimator,
        n_estimators='auto',
        verbose=0,
        random_state=42,
        max_iter=100,  # Maximum iterations
        perc=100,  # Percentile of shadow features to compare against
    )

    # Fit Boruta on training data
    boruta_selector.fit(X_train.values, y_train)

    # Get selected features
    selected_mask = boruta_selector.support_
    tentative_mask = boruta_selector.support_weak_

    # Combine confirmed and tentative features
    all_selected_mask = selected_mask | tentative_mask

    selected_features = [f for f, sel in zip(feature_cols, all_selected_mask) if sel]
    confirmed_features = [f for f, sel in zip(feature_cols, selected_mask) if sel]
    tentative_features = [f for f, sel in zip(feature_cols, tentative_mask) if sel]

    print(f"\nBoruta Results:")
    print(f"  Confirmed features: {len(confirmed_features)}")
    print(f"  Tentative features: {len(tentative_features)}")
    print(f"  Total selected: {len(selected_features)}")

    # Get feature rankings for additional info
    feature_ranks = pd.DataFrame({
        'feature': feature_cols,
        'rank': boruta_selector.ranking_,
        'confirmed': selected_mask,
        'tentative': tentative_mask
    }).sort_values('rank')

    print(f"\nTop 10 features by Boruta ranking:")
    for _, row in feature_ranks.head(10).iterrows():
        status = "CONFIRMED" if row['confirmed'] else ("tentative" if row['tentative'] else "rejected")
        print(f"    {row['feature']}: rank={row['rank']} ({status})")

    # If Boruta selected too few features, fall back to ranking
    if len(selected_features) < 20:
        print(f"\n  Warning: Only {len(selected_features)} features selected, using top 40 by rank")
        selected_features = feature_ranks.head(40)['feature'].tolist()

    # All callibration use the same Boruta-selected features
    # This is more robust than per-model selection
    features_per_model = {}

    # For tree-based callibration, use all selected features
    for model_name in ['XGBoost', 'LightGBM', 'CatBoost']:
        features_per_model[model_name] = selected_features
        print(f"\n{model_name}: Using {len(selected_features)} Boruta-selected features")

    # For LogisticReg, also use Boruta features (only for classification)
    if not is_regression:
        features_per_model['LogisticReg'] = selected_features
        print(f"LogisticReg: Using {len(selected_features)} Boruta-selected features")

    # ============================================================
    # STEP 2: Optuna Tuning Per Model
    # ============================================================
    print("\n" + "=" * 70)
    if stepwise:
        print("STEP 2: Step-wise Optuna Tuning (from 'Effective XGBoost' book)")
    else:
        print(f"STEP 2: Optuna Tuning ({n_trials} trials per model)")
    print("=" * 70)

    best_params = {}
    direction = 'minimize' if is_regression else 'maximize'
    metric_name = 'MAE' if is_regression else 'accuracy'

    # XGBoost
    # Parameter ranges optimized based on "Effective XGBoost" book by Matt Harrison
    print("\nTuning XGBoost...")
    xgb_features = features_per_model['XGBoost']

    if stepwise:
        # Use step-wise tuning (faster, ~4x speedup)
        print("  Using step-wise tuning (4 steps x 30 trials = 120 total)")
        best_params['XGBoost'] = stepwise_tune_xgboost(
            X_train, y_train, X_val, y_val, xgb_features, is_regression, trials_per_step=30
        )
        print(f"  XGBoost step-wise tuning complete")
    else:
        # Standard full tuning
        def xgb_objective(trial):
            # Tree parameters (book recommends: max_depth 1-8, min_child_weight loguniform)
            params = {
                'n_estimators': 500,  # High value, rely on early stopping
                'max_depth': trial.suggest_int('max_depth', 2, 8),
                'min_child_weight': trial.suggest_float('min_child_weight', 0.1, 20.0, log=True),
                # Sampling parameters (book: 0.5-1.0 for both)
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                # Regularization parameters (book: reg_alpha 0-10, reg_lambda 1-10)
                'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 10.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
                # Gamma (book: loguniform, can be very small to large)
                'gamma': trial.suggest_float('gamma', 1e-8, 10.0, log=True),
                # Learning rate (book recommends tuning last, range loguniform -7 to 0)
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                'random_state': 42, 'verbosity': 0,
                'early_stopping_rounds': 50  # From book: use early stopping
            }
            if is_regression:
                model = XGBRegressor(**params)
                model.fit(
                    X_train[xgb_features], y_train,
                    eval_set=[(X_val[xgb_features], y_val)],
                    verbose=False
                )
                pred = model.predict(X_val[xgb_features])
                return mean_absolute_error(y_val, pred)
            else:
                model = XGBClassifier(**params)
                model.fit(
                    X_train[xgb_features], y_train,
                    eval_set=[(X_val[xgb_features], y_val)],
                    verbose=False
                )
                return model.score(X_val[xgb_features], y_val)

        study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(xgb_objective, n_trials=n_trials, show_progress_bar=False)
        best_params['XGBoost'] = study.best_params
        print(f"  Best val {metric_name}: {study.best_value:.4f}")

    # LightGBM - with early stopping and optimized ranges
    print("\nTuning LightGBM...")
    lgbm_features = features_per_model['LightGBM']

    def lgbm_objective(trial):
        params = {
            'n_estimators': 500,  # High value, rely on early stopping
            'max_depth': trial.suggest_int('max_depth', 2, 10),
            'num_leaves': trial.suggest_int('num_leaves', 10, 100),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 60),
            'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 10.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'random_state': 42, 'verbose': -1
        }
        if is_regression:
            model = LGBMRegressor(**params)
            model.fit(
                X_train[lgbm_features], y_train,
                eval_set=[(X_val[lgbm_features], y_val)],
                callbacks=[lgb_early_stopping(50, verbose=False)]
            )
            pred = model.predict(X_val[lgbm_features])
            return mean_absolute_error(y_val, pred)
        else:
            model = LGBMClassifier(**params)
            model.fit(
                X_train[lgbm_features], y_train,
                eval_set=[(X_val[lgbm_features], y_val)],
                callbacks=[lgb_early_stopping(50, verbose=False)]
            )
            return model.score(X_val[lgbm_features], y_val)

    study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lgbm_objective, n_trials=n_trials, show_progress_bar=False)
    best_params['LightGBM'] = study.best_params
    print(f"  Best val {metric_name}: {study.best_value:.4f}")

    # CatBoost - with early stopping and optimized ranges
    print("\nTuning CatBoost...")
    cat_features = features_per_model['CatBoost']

    def cat_objective(trial):
        params = {
            'iterations': 500,  # High value, rely on early stopping
            'depth': trial.suggest_int('depth', 3, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 30.0),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
            'random_strength': trial.suggest_float('random_strength', 0.0, 3.0),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
            'random_state': 42, 'verbose': 0,
            'early_stopping_rounds': 50
        }
        if is_regression:
            model = CatBoostRegressor(**params)
            model.fit(
                X_train[cat_features], y_train,
                eval_set=(X_val[cat_features], y_val)
            )
            pred = model.predict(X_val[cat_features])
            return mean_absolute_error(y_val, pred)
        else:
            model = CatBoostClassifier(**params)
            model.fit(
                X_train[cat_features], y_train,
                eval_set=(X_val[cat_features], y_val)
            )
            return model.score(X_val[cat_features], y_val)

    study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(cat_objective, n_trials=n_trials, show_progress_bar=False)
    best_params['CatBoost'] = study.best_params
    print(f"  Best val {metric_name}: {study.best_value:.4f}")

    # LogisticRegression / Ridge (only for classification)
    scaler_lr = None
    if not is_regression:
        print("\nTuning LogisticRegression...")
        lr_features = features_per_model['LogisticReg']
        scaler_lr = StandardScaler()
        X_train_lr = scaler_lr.fit_transform(X_train[lr_features])
        X_val_lr = scaler_lr.transform(X_val[lr_features])

        def lr_objective(trial):
            params = {
                'C': trial.suggest_float('C', 0.0001, 10.0, log=True),
                'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
                'solver': 'saga',
                'max_iter': 1000,
                'random_state': 42
            }
            model = LogisticRegression(**params)
            model.fit(X_train_lr, y_train)
            return model.score(X_val_lr, y_val)

        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(lr_objective, n_trials=n_trials, show_progress_bar=False)
        best_params['LogisticReg'] = study.best_params
        print(f"  Best val accuracy: {study.best_value:.4f}")

    # ============================================================
    # STEP 2.5: Re-run Boruta with Tuned Params (Optional)
    # ============================================================
    if revalidate_features:
        print("\n" + "=" * 70)
        print("STEP 2.5: Re-run Boruta Feature Selection with Tuned Models")
        print("=" * 70)

        # Use best XGBoost params for Boruta re-run (most robust)
        xgb_params = best_params.get('XGBoost', {})
        print(f"\nRe-running Boruta with tuned XGBoost parameters...")

        if is_regression:
            # For regression, use XGBoost with tuned params
            tuned_estimator = XGBRegressor(
                n_estimators=100,
                max_depth=xgb_params.get('max_depth', 5),
                learning_rate=xgb_params.get('learning_rate', 0.1),
                subsample=xgb_params.get('subsample', 0.8),
                colsample_bytree=xgb_params.get('colsample_bytree', 0.8),
                random_state=42,
                verbosity=0,
                n_jobs=-1
            )
        else:
            tuned_estimator = XGBClassifier(
                n_estimators=100,
                max_depth=xgb_params.get('max_depth', 5),
                learning_rate=xgb_params.get('learning_rate', 0.1),
                subsample=xgb_params.get('subsample', 0.8),
                colsample_bytree=xgb_params.get('colsample_bytree', 0.8),
                random_state=42,
                verbosity=0,
                n_jobs=-1
            )

        boruta_revalidate = BorutaPy(
            tuned_estimator,
            n_estimators='auto',
            verbose=0,
            random_state=42,
            max_iter=50,  # Fewer iterations for revalidation
        )

        boruta_revalidate.fit(X_train.values, y_train)

        # Get new selected features
        new_selected_mask = boruta_revalidate.support_ | boruta_revalidate.support_weak_
        new_selected_features = [f for f, sel in zip(feature_cols, new_selected_mask) if sel]

        # Compare with original selection
        original_count = len(selected_features)
        new_count = len(new_selected_features)

        # Find differences
        added = set(new_selected_features) - set(selected_features)
        removed = set(selected_features) - set(new_selected_features)

        print(f"\nRevalidation Results:")
        print(f"  Original: {original_count} features")
        print(f"  Revalidated: {new_count} features")
        print(f"  Added: {len(added)} features")
        print(f"  Removed: {len(removed)} features")

        if len(added) > 0:
            print(f"  New features: {list(added)[:5]}...")
        if len(removed) > 0:
            print(f"  Dropped features: {list(removed)[:5]}...")

        # Use new features if they're reasonable
        if new_count >= 15:
            selected_features = new_selected_features
            for model_name in features_per_model:
                features_per_model[model_name] = selected_features
            print(f"\n  Updated all callibration to use {len(selected_features)} revalidated features")
        else:
            print(f"\n  Keeping original {original_count} features (revalidation too restrictive)")

        # Update feature references
        xgb_features = features_per_model['XGBoost']
        lgbm_features = features_per_model['LightGBM']
        cat_features = features_per_model['CatBoost']

    # ============================================================
    # STEP 3: Train Final Models with Calibration
    # ============================================================
    print("\n" + "=" * 70)
    cal_name = {'platt': 'Platt', 'isotonic': 'Isotonic', 'beta': 'Beta'}
    print(f"STEP 3: Train Final Models {'(Regression)' if is_regression else f'with {cal_name.get(calibration, calibration)} Calibration'}")
    print("=" * 70)

    final_models = {}
    beta_calibrators = {}  # Store beta calibrators for later use

    if is_regression:
        # Regression callibration (no calibration needed)
        xgb_params = {**best_params['XGBoost'], 'random_state': 42, 'verbosity': 0}
        xgb = XGBRegressor(**xgb_params)
        xgb.fit(X_train[xgb_features], y_train)
        final_models['XGBoost'] = (xgb, xgb_features)

        lgbm_params = {**best_params['LightGBM'], 'random_state': 42, 'verbose': -1}
        lgbm = LGBMRegressor(**lgbm_params)
        lgbm.fit(X_train[lgbm_features], y_train)
        final_models['LightGBM'] = (lgbm, lgbm_features)

        cat_params = {**best_params['CatBoost'], 'random_state': 42, 'verbose': 0}
        cat = CatBoostRegressor(**cat_params)
        cat.fit(X_train[cat_features], y_train)
        final_models['CatBoost'] = (cat, cat_features)

        print("Regression callibration trained")
    else:
        # Classification callibration with calibration
        # Choose calibration method
        if calibration == 'beta':
            # Beta calibration - train raw model first, then apply beta calibration
            print(f"  Using Beta Calibration (more flexible than Platt scaling)")

            # XGBoost with Beta calibration
            xgb_params = {**best_params['XGBoost'], 'random_state': 42, 'verbosity': 0}
            xgb = XGBClassifier(**xgb_params)
            xgb.fit(X_train[xgb_features], y_train)
            xgb_proba_val = xgb.predict_proba(X_val[xgb_features])[:, 1]
            beta_cal_xgb = BetaCalibrator(method='abm')
            beta_cal_xgb.fit(xgb_proba_val, y_val)
            beta_calibrators['XGBoost'] = beta_cal_xgb
            final_models['XGBoost'] = (xgb, xgb_features)
            print(f"    XGBoost Beta params: {beta_cal_xgb.get_params_str()}")

            # LightGBM with Beta calibration
            lgbm_params = {**best_params['LightGBM'], 'random_state': 42, 'verbose': -1}
            lgbm = LGBMClassifier(**lgbm_params)
            lgbm.fit(X_train[lgbm_features], y_train)
            lgbm_proba_val = lgbm.predict_proba(X_val[lgbm_features])[:, 1]
            beta_cal_lgbm = BetaCalibrator(method='abm')
            beta_cal_lgbm.fit(lgbm_proba_val, y_val)
            beta_calibrators['LightGBM'] = beta_cal_lgbm
            final_models['LightGBM'] = (lgbm, lgbm_features)
            print(f"    LightGBM Beta params: {beta_cal_lgbm.get_params_str()}")

            # CatBoost with Beta calibration
            cat_params = {**best_params['CatBoost'], 'random_state': 42, 'verbose': 0}
            cat = CatBoostClassifier(**cat_params)
            cat.fit(X_train[cat_features], y_train)
            cat_proba_val = cat.predict_proba(X_val[cat_features])[:, 1]
            beta_cal_cat = BetaCalibrator(method='abm')
            beta_cal_cat.fit(cat_proba_val, y_val)
            beta_calibrators['CatBoost'] = beta_cal_cat
            final_models['CatBoost'] = (cat, cat_features)
            print(f"    CatBoost Beta params: {beta_cal_cat.get_params_str()}")

            # LogisticReg with Beta calibration
            lr_features = features_per_model['LogisticReg']
            lr_params = {**best_params['LogisticReg'], 'solver': 'saga', 'max_iter': 1000, 'random_state': 42}
            lr = LogisticRegression(**lr_params)
            lr.fit(X_train_lr, y_train)
            lr_proba_val = lr.predict_proba(X_val_lr)[:, 1]
            beta_cal_lr = BetaCalibrator(method='abm')
            beta_cal_lr.fit(lr_proba_val, y_val)
            beta_calibrators['LogisticReg'] = beta_cal_lr
            final_models['LogisticReg'] = (lr, lr_features, scaler_lr)
            print(f"    LogisticReg Beta params: {beta_cal_lr.get_params_str()}")

        else:
            # Standard sklearn calibration (Platt or Isotonic)
            sklearn_method = 'sigmoid' if calibration == 'platt' else 'isotonic'

            xgb_params = {**best_params['XGBoost'], 'random_state': 42, 'verbosity': 0}
            xgb = XGBClassifier(**xgb_params)
            xgb.fit(X_train[xgb_features], y_train)
            xgb_cal = CalibratedClassifierCV(xgb, method=sklearn_method, cv='prefit')
            xgb_cal.fit(X_val[xgb_features], y_val)
            final_models['XGBoost'] = (xgb_cal, xgb_features)

            lgbm_params = {**best_params['LightGBM'], 'random_state': 42, 'verbose': -1}
            lgbm = LGBMClassifier(**lgbm_params)
            lgbm.fit(X_train[lgbm_features], y_train)
            lgbm_cal = CalibratedClassifierCV(lgbm, method=sklearn_method, cv='prefit')
            lgbm_cal.fit(X_val[lgbm_features], y_val)
            final_models['LightGBM'] = (lgbm_cal, lgbm_features)

            cat_params = {**best_params['CatBoost'], 'random_state': 42, 'verbose': 0}
            cat = CatBoostClassifier(**cat_params)
            cat.fit(X_train[cat_features], y_train)
            cat_cal = CalibratedClassifierCV(cat, method=sklearn_method, cv='prefit')
            cat_cal.fit(X_val[cat_features], y_val)
            final_models['CatBoost'] = (cat_cal, cat_features)

            # LogisticReg
            lr_features = features_per_model['LogisticReg']
            lr_params = {**best_params['LogisticReg'], 'solver': 'saga', 'max_iter': 1000, 'random_state': 42}
            lr = LogisticRegression(**lr_params)
            lr.fit(X_train_lr, y_train)
            lr_cal = CalibratedClassifierCV(lr, method=sklearn_method, cv='prefit')
            lr_cal.fit(X_val_lr, y_val)
            final_models['LogisticReg'] = (lr_cal, lr_features, scaler_lr)

        print(f"Models trained and calibrated ({calibration})")

    # ============================================================
    # STEP 4: Evaluation
    # ============================================================
    print("\n" + "=" * 70)
    print("STEP 4: Evaluation")
    print("=" * 70)

    test_preds = {}
    val_preds = {}
    ah_line_test = ah_line[test_idx] if ah_line is not None else None
    ah_line_val = ah_line[val_idx] if ah_line is not None else None

    if is_regression:
        # Regression: get predictions (goal margin predictions)
        for name, model_data in final_models.items():
            model, features = model_data
            X_t = X_test[features]
            X_v = X_val[features]

            pred = model.predict(X_t)
            test_preds[name] = pred
            mae = mean_absolute_error(y_test, pred)
            print(f"{name} MAE: {mae:.4f}")

            val_preds[name] = model.predict(X_v)

        # Ensemble average for regression
        avg_pred = np.mean([test_preds[n] for n in test_preds], axis=0)
        avg_val_pred = np.mean([val_preds[n] for n in val_preds], axis=0)
        print(f"\nSimple Average MAE: {mean_absolute_error(y_test, avg_pred):.4f}")

        # Stacking ensemble for regression
        X_stack_val = np.column_stack([val_preds[n] for n in ['XGBoost', 'LightGBM', 'CatBoost']])
        X_stack_test = np.column_stack([test_preds[n] for n in ['XGBoost', 'LightGBM', 'CatBoost']])

        meta = Ridge(alpha=1.0)
        meta.fit(X_stack_val, y_val)
        stack_pred = meta.predict(X_stack_test)
        print(f"Stacking MAE: {mean_absolute_error(y_test, stack_pred):.4f}")

        test_preds['Average'] = avg_pred
        test_preds['Stacking'] = stack_pred

    else:
        # Classification: get probabilities
        for name, model_data in final_models.items():
            if name == 'LogisticReg':
                model, features, sc = model_data
                X_t = sc.transform(X_test[features])
                X_v = sc.transform(X_val[features])
            else:
                model, features = model_data
                X_t = X_test[features]
                X_v = X_val[features]

            proba = model.predict_proba(X_t)[:, 1]
            val_proba = model.predict_proba(X_v)[:, 1]

            # Apply beta calibration if used
            if calibration == 'beta' and name in beta_calibrators:
                proba = beta_calibrators[name].transform(proba)
                val_proba = beta_calibrators[name].transform(val_proba)

            test_preds[name] = proba
            acc = ((proba >= 0.5) == y_test).mean()

            # Calculate calibration metrics
            cal_metrics = calibration_metrics(y_test, proba)
            print(f"{name}: Acc={acc:.4f}, ECE={cal_metrics['ece']:.4f}, Brier={cal_metrics['brier']:.4f}")

            val_preds[name] = val_proba

        # Ensembles for classification
        avg_proba = np.mean([test_preds[n] for n in test_preds], axis=0)
        avg_cal = calibration_metrics(y_test, avg_proba)
        print(f"\nSimple Average: Acc={((avg_proba >= 0.5) == y_test).mean():.4f}, ECE={avg_cal['ece']:.4f}")

        X_stack_val = np.column_stack([val_preds[n] for n in ['XGBoost', 'LightGBM', 'CatBoost', 'LogisticReg']])
        X_stack_test = np.column_stack([test_preds[n] for n in ['XGBoost', 'LightGBM', 'CatBoost', 'LogisticReg']])

        meta = RidgeClassifierCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0])
        meta.fit(X_stack_val, y_val)
        stack_proba = 1 / (1 + np.exp(-meta.decision_function(X_stack_test)))
        print(f"Stacking: {((stack_proba >= 0.5) == y_test).mean():.4f}")

        test_preds['Average'] = avg_proba
        test_preds['Stacking'] = stack_proba

    # ============================================================
    # STEP 5: Betting Optimization
    # ============================================================
    print("\n" + "=" * 70)
    print("STEP 5: Betting Optimization")
    print("=" * 70)

    def calc_roi_bootstrap(bet_mask, actual_win, odds_arr, n_boot=1000):
        """Calculate ROI with bootstrap confidence intervals and Sharpe ratio.

        Returns:
            tuple: (mean_roi, ci_low, ci_high, p_profit, sharpe_ratio, sortino_ratio)

        Sharpe Ratio: mean_return / std_return (risk-adjusted return)
        Sortino Ratio: mean_return / downside_std (penalizes only negative volatility)
        """
        rois = []
        # Also calculate per-bet returns for Sharpe ratio
        per_bet_returns = []

        for _ in range(n_boot):
            idx = np.random.choice(len(bet_mask), len(bet_mask), replace=True)
            mask = bet_mask[idx]
            a = actual_win[idx]
            o = odds_arr[idx]
            if mask.sum() == 0:
                continue
            wins = a[mask] == 1
            profit = (wins * (o[mask] - 1) - (~wins) * 1).sum()
            rois.append(profit / mask.sum() * 100)

        if not rois:
            return 0, 0, 0, 0, 0, 0

        # Calculate per-bet returns for Sharpe/Sortino
        if bet_mask.sum() > 0:
            wins_actual = actual_win[bet_mask] == 1
            bet_odds = odds_arr[bet_mask]
            per_bet_returns = np.where(wins_actual, bet_odds - 1, -1)  # Returns per unit staked

            # Sharpe ratio (mean / std) - higher is better risk-adjusted return
            mean_return = np.mean(per_bet_returns)
            std_return = np.std(per_bet_returns)
            sharpe = mean_return / std_return if std_return > 0 else 0

            # Sortino ratio (mean / downside std) - only penalizes negative returns
            downside_returns = per_bet_returns[per_bet_returns < 0]
            downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0
            sortino = mean_return / downside_std if downside_std > 0 else (mean_return if mean_return > 0 else 0)
        else:
            sharpe = 0
            sortino = 0

        return (np.mean(rois), np.percentile(rois, 2.5), np.percentile(rois, 97.5),
                (np.array(rois) > 0).mean(), sharpe, sortino)

    results = []

    if is_regression:
        # Asian Handicap: bet if predicted_margin > -ah_line (home covers)
        # Actual win: goal_difference + ah_line > 0
        actual_home_covers = (y_test + ah_line_test > 0).astype(int)

        # Test different margins for betting decision
        for model_name in ['XGBoost', 'LightGBM', 'CatBoost', 'Average', 'Stacking']:
            pred_margin = test_preds[model_name]

            for margin_buffer in [0, 0.25, 0.5, 0.75, 1.0]:
                # Bet home covers when predicted margin > -ah_line + buffer
                bet_mask = (pred_margin > -ah_line_test + margin_buffer)
                n_bets = bet_mask.sum()
                if n_bets < 30:
                    continue

                prec = actual_home_covers[bet_mask].mean()
                roi, ci_low, ci_high, p_profit, sharpe, sortino = calc_roi_bootstrap(
                    bet_mask, actual_home_covers, odds_test
                )

                results.append({
                    'strategy': f'{model_name} margin>{margin_buffer}',
                    'model': model_name,
                    'threshold': margin_buffer,
                    'bets': n_bets,
                    'precision': prec,
                    'roi': roi,
                    'ci_low': ci_low,
                    'ci_high': ci_high,
                    'p_profit': p_profit,
                    'sharpe': sharpe,
                    'sortino': sortino
                })
    else:
        # Classification: standard threshold-based betting
        for model_name in ['XGBoost', 'LightGBM', 'CatBoost', 'LogisticReg', 'Average', 'Stacking']:
            proba = test_preds[model_name]

            for thresh in [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]:
                bet_mask = (proba >= thresh)
                n_bets = bet_mask.sum()
                if n_bets < 30:
                    continue

                prec = y_test[bet_mask].mean()
                roi, ci_low, ci_high, p_profit, sharpe, sortino = calc_roi_bootstrap(
                    bet_mask, y_test, odds_test
                )

                results.append({
                    'strategy': f'{model_name} >= {thresh}',
                    'model': model_name,
                    'threshold': thresh,
                    'bets': n_bets,
                    'precision': prec,
                    'roi': roi,
                    'ci_low': ci_low,
                    'ci_high': ci_high,
                    'p_profit': p_profit,
                    'sharpe': sharpe,
                    'sortino': sortino
                })

    # Print results with Sharpe ratio
    print(f"\n{'Strategy':<25} {'Bets':>5} {'Prec':>7} {'ROI':>9} {'Sharpe':>7} {'Sortino':>8} {'P(profit)':>10}")
    print("-" * 90)

    # Sort by chosen metric (Sharpe for risk-adjusted, ROI for raw returns)
    sort_metric = 'sharpe' if optimize_sharpe else 'roi'
    results_df = pd.DataFrame(results).sort_values(sort_metric, ascending=False)
    for _, row in results_df.head(15).iterrows():
        print(f"{row['strategy']:<25} {row['bets']:>5} {row['precision']:>6.1%} {row['roi']:>+8.1f}% "
              f"{row['sharpe']:>+6.3f} {row['sortino']:>+7.3f} {row['p_profit']:>9.0%}")

    # Also show best by Sharpe ratio
    best_by_sharpe = results_df.sort_values('sharpe', ascending=False).head(5)
    print(f"\n--- Best by Sharpe Ratio (Risk-Adjusted) ---")
    for _, row in best_by_sharpe.iterrows():
        print(f"{row['strategy']:<25} Sharpe: {row['sharpe']:>+.3f} | ROI: {row['roi']:>+.1f}% | Sortino: {row['sortino']:>+.3f}")

    # Best result
    best = None
    best_sharpe_row = None
    if len(results_df) > 0:
        best = results_df.iloc[0]
        best_sharpe_row = best_by_sharpe.iloc[0]
        print("\n" + "=" * 70)
        print(f"BEST ROI: {best['strategy']} | ROI: {best['roi']:.1f}% | Sharpe: {best['sharpe']:.3f}")
        print(f"BEST SHARPE: {best_sharpe_row['strategy']} | Sharpe: {best_sharpe_row['sharpe']:.3f} | ROI: {best_sharpe_row['roi']:.1f}%")
        print("=" * 70)

    # ============================================================
    # STEP 6: Walk-Forward Validation (Optional)
    # ============================================================
    walkforward_results = None
    if walkforward:
        print("\n" + "=" * 70)
        print("STEP 6: Walk-Forward Validation")
        print("=" * 70)

        # Walk-forward with 5 folds (each ~20% of data)
        n_folds = 5
        fold_size = len(X) // n_folds
        wf_results = []

        print(f"\nRunning {n_folds}-fold walk-forward validation...")
        print(f"  Each fold: ~{fold_size} matches")

        for fold in range(2, n_folds + 1):  # Start from fold 2 (need training data)
            train_end = fold * fold_size
            test_start = train_end
            test_end = min((fold + 1) * fold_size, len(X))

            if test_end <= test_start:
                continue

            # Get indices in time order
            fold_train_idx = sorted_indices[:train_end]
            fold_test_idx = sorted_indices[test_start:test_end]

            X_fold_train = X.iloc[fold_train_idx]
            y_fold_train = y[fold_train_idx]
            X_fold_test = X.iloc[fold_test_idx]
            y_fold_test = y[fold_test_idx]
            odds_fold_test = odds[fold_test_idx]

            print(f"\n  Fold {fold}: Train={len(X_fold_train)}, Test={len(X_fold_test)}")

            # Use pre-tuned parameters from main training
            fold_preds = {}

            for name in ['XGBoost', 'LightGBM', 'CatBoost']:
                features = features_per_model[name]
                params = best_params[name]

                if is_regression:
                    if name == 'XGBoost':
                        model = XGBRegressor(**params, random_state=42, verbosity=0)
                    elif name == 'LightGBM':
                        model = LGBMRegressor(**params, random_state=42, verbose=-1)
                    else:
                        model = CatBoostRegressor(**params, random_state=42, verbose=0)
                    model.fit(X_fold_train[features], y_fold_train)
                    fold_preds[name] = model.predict(X_fold_test[features])
                else:
                    if name == 'XGBoost':
                        model = XGBClassifier(**params, random_state=42, verbosity=0)
                    elif name == 'LightGBM':
                        model = LGBMClassifier(**params, random_state=42, verbose=-1)
                    else:
                        model = CatBoostClassifier(**params, random_state=42, verbose=0)
                    model.fit(X_fold_train[features], y_fold_train)
                    proba = model.predict_proba(X_fold_test[features])[:, 1]

                    # Apply beta calibration if used
                    if calibration == 'beta':
                        # Quick calibration on fold
                        X_cal = X_fold_train[features].iloc[-fold_size//2:]
                        y_cal = y_fold_train[-fold_size//2:]
                        cal_proba = model.predict_proba(X_cal)[:, 1]
                        beta_cal = BetaCalibrator(method='abm')
                        beta_cal.fit(cal_proba, y_cal)
                        proba = beta_cal.transform(proba)

                    fold_preds[name] = proba

            # Stacking ensemble for this fold
            if is_regression:
                stack_input = np.column_stack([fold_preds[n] for n in ['XGBoost', 'LightGBM', 'CatBoost']])
                stack_pred = np.mean(stack_input, axis=1)
                fold_preds['Stacking'] = stack_pred
            else:
                stack_input = np.column_stack([fold_preds[n] for n in ['XGBoost', 'LightGBM', 'CatBoost']])
                stack_pred = np.mean(stack_input, axis=1)
                fold_preds['Stacking'] = stack_pred

            # Evaluate best strategy on this fold
            if not is_regression:
                best_thresh = best['threshold'] if best is not None else 0.5
                for model_name in ['XGBoost', 'LightGBM', 'CatBoost', 'Stacking']:
                    proba = fold_preds[model_name]
                    bet_mask = proba >= best_thresh
                    n_bets = bet_mask.sum()

                    if n_bets > 5:
                        wins = y_fold_test[bet_mask] == 1
                        profit = (wins * (odds_fold_test[bet_mask] - 1) - (~wins) * 1).sum()
                        roi = profit / n_bets * 100

                        wf_results.append({
                            'fold': fold,
                            'model': model_name,
                            'threshold': best_thresh,
                            'n_bets': n_bets,
                            'roi': roi,
                            'accuracy': (proba >= 0.5).astype(int)[bet_mask].mean() if n_bets > 0 else 0
                        })

        # Summarize walk-forward results
        if wf_results:
            wf_df = pd.DataFrame(wf_results)
            print("\n" + "-" * 50)
            print("Walk-Forward Summary by Model:")
            for model_name in ['XGBoost', 'LightGBM', 'CatBoost', 'Stacking']:
                model_wf = wf_df[wf_df['model'] == model_name]
                if len(model_wf) > 0:
                    avg_roi = model_wf['roi'].mean()
                    std_roi = model_wf['roi'].std()
                    total_bets = model_wf['n_bets'].sum()
                    print(f"  {model_name}: ROI={avg_roi:+.1f}% (+/-{std_roi:.1f}%), Bets={total_bets}")

            # Overall walk-forward performance
            stacking_wf = wf_df[wf_df['model'] == 'Stacking']
            if len(stacking_wf) > 0:
                walkforward_results = {
                    'avg_roi': float(stacking_wf['roi'].mean()),
                    'std_roi': float(stacking_wf['roi'].std()),
                    'total_bets': int(stacking_wf['n_bets'].sum()),
                    'n_folds': len(stacking_wf),
                    'all_folds': wf_df.to_dict('records')
                }
                print(f"\n  STACKING Walk-Forward: {walkforward_results['avg_roi']:+.1f}% avg ROI across {walkforward_results['n_folds']} folds")

    # Save results
    output = {
        'bet_type': bet_type,
        'target_name': target_name,
        'is_regression': is_regression,
        'base_rate': float(base_rate),
        'matches': len(df),
        'test_matches': len(y_test),
        'boruta_features': selected_features,  # Boruta-selected features
        'features_per_model': {k: v for k, v in features_per_model.items()},  # Save actual features
        'best_params': best_params,
        'calibration_method': calibration,
        # Best by ROI
        'best_strategy': best['strategy'] if best is not None else None,
        'best_roi': float(best['roi']) if best is not None else None,
        'best_p_profit': float(best['p_profit']) if best is not None else None,
        'best_bets': int(best['bets']) if best is not None else None,
        'best_sharpe': float(best['sharpe']) if best is not None else None,
        'best_sortino': float(best['sortino']) if best is not None else None,
        # Best by Sharpe (risk-adjusted)
        'best_sharpe_strategy': best_sharpe_row['strategy'] if best_sharpe_row is not None else None,
        'best_sharpe_value': float(best_sharpe_row['sharpe']) if best_sharpe_row is not None else None,
        'best_sharpe_roi': float(best_sharpe_row['roi']) if best_sharpe_row is not None else None,
        'best_sharpe_sortino': float(best_sharpe_row['sortino']) if best_sharpe_row is not None else None,
        # Walk-forward validation
        'walkforward': walkforward_results,
        'all_results': results_df.to_dict('records') if len(results_df) > 0 else []
    }

    output_path = f'experiments/outputs/{bet_type}_full_optimization.json'
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nSaved to {output_path}")

    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bet_type', type=str, required=True,
                       choices=['asian_handicap', 'home_win', 'over25', 'under25', 'away_win', 'btts',
                                'corners', 'shots', 'fouls', 'cards'])
    parser.add_argument('--n_trials', type=int, default=150,
                       help='Number of Optuna trials per model (default: 150)')
    parser.add_argument('--revalidate-features', action='store_true',
                       help='Two-pass feature selection: re-select features with tuned params')
    parser.add_argument('--walkforward', action='store_true',
                       help='Run walk-forward validation after training')
    parser.add_argument('--stepwise', action='store_true',
                       help='Use step-wise tuning from "Effective XGBoost" book (faster, 4x speedup)')
    parser.add_argument('--optimize-sharpe', action='store_true',
                       help='Optimize for Sharpe ratio instead of ROI (risk-adjusted returns)')
    parser.add_argument('--calibration', type=str, default='platt',
                       choices=['platt', 'isotonic', 'beta'],
                       help='Calibration method: platt (default), isotonic, or beta')
    args = parser.parse_args()

    run_pipeline(args.bet_type, args.n_trials, args.revalidate_features, args.walkforward, args.stepwise, args.optimize_sharpe, args.calibration)
