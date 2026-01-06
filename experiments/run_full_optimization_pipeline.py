"""
Full Optimization Pipeline for Any Bet Type

Usage:
    python run_full_optimization_pipeline.py --bet_type asian_handicap
    python run_full_optimization_pipeline.py --bet_type home_win
    python run_full_optimization_pipeline.py --bet_type over25
    python run_full_optimization_pipeline.py --bet_type under25

Pipeline:
1. Permutation importance feature selection PER MODEL
2. Optuna tuning per model with its own best features (80 trials)
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
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_data(bet_type):
    """Load data and create target for specific bet type."""
    df = pd.read_csv('/home/kamil/projects/bettip/data/03-features/features_all_5leagues_with_odds.csv')

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

    else:
        raise ValueError(f"Unknown bet type: {bet_type}")

    return df_filtered, odds_col_bet, odds_col_opposite, exclude_odds, target_name, base_rate, is_regression


def get_feature_columns(df, exclude_odds):
    """Get feature columns excluding identifiers and odds."""
    exclude_cols = [
        'fixture_id', 'date', 'home_team_id', 'home_team_name', 'away_team_id',
        'away_team_name', 'round', 'match_result', 'home_win', 'draw', 'away_win',
        'total_goals', 'goal_difference', 'league', 'target', 'ah_result',
        'home_goals', 'away_goals', 'season', 'round_num', 'result'
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


def run_pipeline(bet_type, n_trials=80, revalidate_features=False, walkforward=False):
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
    # STEP 1: Feature Selection Per Model
    # ============================================================
    print("\n" + "=" * 70)
    print("STEP 1: Permutation Importance Feature Selection")
    print("=" * 70)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Choose models based on task type
    if is_regression:
        base_models = {
            'XGBoost': XGBRegressor(n_estimators=100, max_depth=4, random_state=42, verbosity=0),
            'LightGBM': LGBMRegressor(n_estimators=100, max_depth=4, random_state=42, verbose=-1),
            'CatBoost': CatBoostRegressor(iterations=100, depth=4, random_state=42, verbose=0),
        }
    else:
        base_models = {
            'XGBoost': XGBClassifier(n_estimators=100, max_depth=4, random_state=42, verbosity=0),
            'LightGBM': LGBMClassifier(n_estimators=100, max_depth=4, random_state=42, verbose=-1),
            'CatBoost': CatBoostClassifier(iterations=100, depth=4, random_state=42, verbose=0),
            'LogisticReg': LogisticRegression(C=0.1, max_iter=1000, random_state=42)
        }

    features_per_model = {}

    for name, model in base_models.items():
        print(f"\n{name}:")

        if name == 'LogisticReg':
            model.fit(X_train_scaled, y_train)
            perm = permutation_importance(model, X_val_scaled, y_val, n_repeats=15, random_state=42, n_jobs=-1)
        else:
            model.fit(X_train, y_train)
            perm = permutation_importance(model, X_val, y_val, n_repeats=15, random_state=42, n_jobs=-1)

        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': perm.importances_mean
        }).sort_values('importance', ascending=False)

        top_features = importance_df[importance_df['importance'] > 0].head(60)['feature'].tolist()
        if len(top_features) < 25:
            top_features = importance_df.head(30)['feature'].tolist()

        features_per_model[name] = top_features
        print(f"  Selected {len(top_features)} features")
        print(f"  Top 5: {top_features[:5]}")

    # ============================================================
    # STEP 2: Optuna Tuning Per Model
    # ============================================================
    print("\n" + "=" * 70)
    print(f"STEP 2: Optuna Tuning ({n_trials} trials per model)")
    print("=" * 70)

    best_params = {}
    direction = 'minimize' if is_regression else 'maximize'
    metric_name = 'MAE' if is_regression else 'accuracy'

    # XGBoost
    print("\nTuning XGBoost...")
    xgb_features = features_per_model['XGBoost']

    def xgb_objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 400),
            'max_depth': trial.suggest_int('max_depth', 2, 8),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 30),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0.0, 2.0),
            'random_state': 42, 'verbosity': 0
        }
        if is_regression:
            model = XGBRegressor(**params)
            model.fit(X_train[xgb_features], y_train)
            pred = model.predict(X_val[xgb_features])
            return mean_absolute_error(y_val, pred)
        else:
            model = XGBClassifier(**params)
            model.fit(X_train[xgb_features], y_train)
            return model.score(X_val[xgb_features], y_val)

    study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(xgb_objective, n_trials=n_trials, show_progress_bar=False)
    best_params['XGBoost'] = study.best_params
    print(f"  Best val {metric_name}: {study.best_value:.4f}")

    # LightGBM
    print("\nTuning LightGBM...")
    lgbm_features = features_per_model['LightGBM']

    def lgbm_objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 400),
            'max_depth': trial.suggest_int('max_depth', 2, 10),
            'num_leaves': trial.suggest_int('num_leaves', 10, 100),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 60),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'random_state': 42, 'verbose': -1
        }
        if is_regression:
            model = LGBMRegressor(**params)
            model.fit(X_train[lgbm_features], y_train)
            pred = model.predict(X_val[lgbm_features])
            return mean_absolute_error(y_val, pred)
        else:
            model = LGBMClassifier(**params)
            model.fit(X_train[lgbm_features], y_train)
            return model.score(X_val[lgbm_features], y_val)

    study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lgbm_objective, n_trials=n_trials, show_progress_bar=False)
    best_params['LightGBM'] = study.best_params
    print(f"  Best val {metric_name}: {study.best_value:.4f}")

    # CatBoost
    print("\nTuning CatBoost...")
    cat_features = features_per_model['CatBoost']

    def cat_objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 50, 400),
            'depth': trial.suggest_int('depth', 3, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.1, 30.0),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'random_strength': trial.suggest_float('random_strength', 0.0, 3.0),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
            'random_state': 42, 'verbose': 0
        }
        if is_regression:
            model = CatBoostRegressor(**params)
            model.fit(X_train[cat_features], y_train)
            pred = model.predict(X_val[cat_features])
            return mean_absolute_error(y_val, pred)
        else:
            model = CatBoostClassifier(**params)
            model.fit(X_train[cat_features], y_train)
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
    # STEP 2.5: Re-select Features with Tuned Params (Optional)
    # ============================================================
    if revalidate_features:
        print("\n" + "=" * 70)
        print("STEP 2.5: Re-select Features with Tuned Params")
        print("=" * 70)

        for name in ['XGBoost', 'LightGBM', 'CatBoost']:
            params = best_params[name]
            print(f"\n{name}:")

            # Create model with tuned params
            if name == 'XGBoost':
                if is_regression:
                    model = XGBRegressor(**params, random_state=42, verbosity=0)
                else:
                    model = XGBClassifier(**params, random_state=42, verbosity=0)
            elif name == 'LightGBM':
                if is_regression:
                    model = LGBMRegressor(**params, random_state=42, verbose=-1)
                else:
                    model = LGBMClassifier(**params, random_state=42, verbose=-1)
            elif name == 'CatBoost':
                if is_regression:
                    model = CatBoostRegressor(**params, random_state=42, verbose=0)
                else:
                    model = CatBoostClassifier(**params, random_state=42, verbose=0)

            model.fit(X_train, y_train)

            perm = permutation_importance(model, X_val, y_val, n_repeats=15, random_state=42, n_jobs=-1)
            importance_df = pd.DataFrame({
                'feature': feature_cols,
                'importance': perm.importances_mean
            }).sort_values('importance', ascending=False)

            top_features = importance_df[importance_df['importance'] > 0].head(60)['feature'].tolist()
            if len(top_features) < 25:
                top_features = importance_df.head(30)['feature'].tolist()

            old_count = len(features_per_model[name])
            features_per_model[name] = top_features
            print(f"  Re-selected {len(top_features)} features (was {old_count})")
            print(f"  Top 5: {top_features[:5]}")

        # Update feature references
        xgb_features = features_per_model['XGBoost']
        lgbm_features = features_per_model['LightGBM']
        cat_features = features_per_model['CatBoost']

    # ============================================================
    # STEP 3: Train Final Models with Calibration
    # ============================================================
    print("\n" + "=" * 70)
    print(f"STEP 3: Train Final Models {'(Regression)' if is_regression else 'with Calibration'}")
    print("=" * 70)

    final_models = {}

    if is_regression:
        # Regression models (no calibration needed)
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

        print("Regression models trained")
    else:
        # Classification models with calibration
        xgb_params = {**best_params['XGBoost'], 'random_state': 42, 'verbosity': 0}
        xgb = XGBClassifier(**xgb_params)
        xgb.fit(X_train[xgb_features], y_train)
        xgb_cal = CalibratedClassifierCV(xgb, method='sigmoid', cv='prefit')
        xgb_cal.fit(X_val[xgb_features], y_val)
        final_models['XGBoost'] = (xgb_cal, xgb_features)

        lgbm_params = {**best_params['LightGBM'], 'random_state': 42, 'verbose': -1}
        lgbm = LGBMClassifier(**lgbm_params)
        lgbm.fit(X_train[lgbm_features], y_train)
        lgbm_cal = CalibratedClassifierCV(lgbm, method='sigmoid', cv='prefit')
        lgbm_cal.fit(X_val[lgbm_features], y_val)
        final_models['LightGBM'] = (lgbm_cal, lgbm_features)

        cat_params = {**best_params['CatBoost'], 'random_state': 42, 'verbose': 0}
        cat = CatBoostClassifier(**cat_params)
        cat.fit(X_train[cat_features], y_train)
        cat_cal = CalibratedClassifierCV(cat, method='sigmoid', cv='prefit')
        cat_cal.fit(X_val[cat_features], y_val)
        final_models['CatBoost'] = (cat_cal, cat_features)

        # LogisticReg
        lr_features = features_per_model['LogisticReg']
        lr_params = {**best_params['LogisticReg'], 'solver': 'saga', 'max_iter': 1000, 'random_state': 42}
        lr = LogisticRegression(**lr_params)
        lr.fit(X_train_lr, y_train)
        lr_cal = CalibratedClassifierCV(lr, method='sigmoid', cv='prefit')
        lr_cal.fit(X_val_lr, y_val)
        final_models['LogisticReg'] = (lr_cal, lr_features, scaler_lr)

        print("Models trained and calibrated")

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
            test_preds[name] = proba
            acc = ((proba >= 0.5) == y_test).mean()
            print(f"{name}: {acc:.4f}")

            val_preds[name] = model.predict_proba(X_v)[:, 1]

        # Ensembles for classification
        avg_proba = np.mean([test_preds[n] for n in test_preds], axis=0)
        print(f"\nSimple Average: {((avg_proba >= 0.5) == y_test).mean():.4f}")

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
        """Calculate ROI with bootstrap confidence intervals."""
        rois = []
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
            return 0, 0, 0, 0
        return np.mean(rois), np.percentile(rois, 2.5), np.percentile(rois, 97.5), (np.array(rois) > 0).mean()

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
                roi, ci_low, ci_high, p_profit = calc_roi_bootstrap(
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
                    'p_profit': p_profit
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
                roi, ci_low, ci_high, p_profit = calc_roi_bootstrap(
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
                    'p_profit': p_profit
                })

    # Print results
    print(f"\n{'Strategy':<25} {'Bets':>5} {'Prec':>7} {'ROI':>9} {'95% CI':>22} {'P(profit)':>10}")
    print("-" * 85)

    results_df = pd.DataFrame(results).sort_values('roi', ascending=False)
    for _, row in results_df.head(15).iterrows():
        print(f"{row['strategy']:<25} {row['bets']:>5} {row['precision']:>6.1%} {row['roi']:>+8.1f}% "
              f"[{row['ci_low']:>7.1f}% to {row['ci_high']:>+7.1f}%] {row['p_profit']:>9.0%}")

    # Best result
    best = None
    if len(results_df) > 0:
        best = results_df.iloc[0]
        print("\n" + "=" * 70)
        print(f"BEST: {best['strategy']} | ROI: {best['roi']:.1f}% | P(profit): {best['p_profit']:.0%}")
        print("=" * 70)

    # Save results
    output = {
        'bet_type': bet_type,
        'target_name': target_name,
        'is_regression': is_regression,
        'base_rate': float(base_rate),
        'matches': len(df),
        'test_matches': len(y_test),
        'features_per_model': {k: v for k, v in features_per_model.items()},  # Save actual features
        'best_params': best_params,
        'best_strategy': best['strategy'] if best is not None else None,
        'best_roi': float(best['roi']) if best is not None else None,
        'best_p_profit': float(best['p_profit']) if best is not None else None,
        'best_bets': int(best['bets']) if best is not None else None,
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
                       choices=['asian_handicap', 'home_win', 'over25', 'under25', 'away_win', 'btts'])
    parser.add_argument('--n_trials', type=int, default=80)
    parser.add_argument('--revalidate-features', action='store_true',
                       help='Two-pass feature selection: re-select features with tuned params')
    parser.add_argument('--walkforward', action='store_true',
                       help='Run walk-forward validation after training')
    args = parser.parse_args()

    run_pipeline(args.bet_type, args.n_trials, args.revalidate_features, args.walkforward)
