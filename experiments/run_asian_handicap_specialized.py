"""
Specialized Asian Handicap Optimization

Key insights from research:
1. AH market is highly efficient - need to find value vs the LINE, not just predict winner
2. Goal MARGIN prediction is key - regression approach
3. Line-specific opportunities (some lines are mispriced more often)
4. Expected margin vs actual line = value signal

Approach:
1. Create margin-focused features (xG differential, scoring patterns, defensive strength)
2. Train regression model to predict goal margin
3. Compare predicted margin to bookmaker's line
4. Bet when our predicted margin differs significantly from line
5. Analyze which line types offer more value
"""

import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_ah_data():
    """Load data with AH odds."""
    df = pd.read_csv('/home/kamil/projects/bettip/data/03-features/features_all_5leagues_with_odds.csv')

    # Filter to matches with AH data
    ah_df = df[df['ah_line'].notna() & df['avg_ah_home'].notna()].copy()

    # Target: actual goal margin (home - away)
    ah_df['goal_margin'] = ah_df['goal_difference']  # This is home_score - away_score

    # AH result
    ah_df['ah_result'] = ah_df['goal_margin'] + ah_df['ah_line']
    ah_df['home_covers'] = (ah_df['ah_result'] > 0).astype(int)
    ah_df['away_covers'] = (ah_df['ah_result'] < 0).astype(int)

    return ah_df


def create_margin_features(df):
    """Create features specifically for goal margin prediction."""

    # The AH line itself is the bookmaker's expected margin (inverted)
    # If line is -0.5, bookmaker expects home to win by ~0.5 goals
    df['bookie_expected_margin'] = -df['ah_line']

    # Margin-focused features (if they exist)
    margin_features = []

    # Core margin predictors
    core_features = [
        # Attack vs Defense
        'home_attack_strength', 'away_attack_strength',
        'home_defense_strength', 'away_defense_strength',

        # Goal scoring patterns
        'home_goals_scored_ema', 'away_goals_scored_ema',
        'home_goals_conceded_ema', 'away_goals_conceded_ema',
        'home_goals_scored_last_n', 'away_goals_scored_last_n',
        'home_goals_conceded_last_n', 'away_goals_conceded_last_n',

        # Expected goals
        'home_xg_poisson', 'away_xg_poisson',

        # Form and momentum
        'home_avg_goal_diff', 'away_avg_goal_diff',
        'home_total_goal_diff', 'away_total_goal_diff',
        'home_season_gd', 'away_season_gd',
        'season_gd_diff',

        # Strength indicators
        'position_diff', 'ppg_diff',
        'home_elo', 'away_elo', 'elo_diff',
        'home_win_prob_elo', 'away_win_prob_elo',

        # Poisson probabilities
        'poisson_home_win_prob', 'poisson_away_win_prob', 'poisson_draw_prob',

        # Rest and schedule
        'home_rest_days', 'away_rest_days', 'rest_days_diff',

        # Season context
        'home_pts_to_leader', 'away_pts_to_leader',
        'home_pts_to_cl', 'away_pts_to_cl',
        'home_pts_to_safety', 'away_pts_to_safety',
        'match_importance', 'home_importance', 'away_importance',

        # Home/away specific
        'home_home_wins', 'home_home_draws', 'home_home_losses',
        'away_away_wins', 'away_away_draws', 'away_away_losses',
        'home_home_goals_scored', 'home_home_goals_conceded',
        'away_away_goals_scored', 'away_away_goals_conceded',

        # Streaks
        'home_unbeaten_streak', 'away_unbeaten_streak',
        'home_scoring_streak', 'away_scoring_streak',
        'home_clean_sheet_streak', 'away_clean_sheet_streak',

        # Referee
        'ref_avg_goals', 'ref_home_win_pct', 'ref_away_win_pct',
    ]

    for f in core_features:
        if f in df.columns:
            margin_features.append(f)

    # Create composite margin features
    if 'home_attack_strength' in df.columns and 'away_defense_strength' in df.columns:
        df['home_attack_vs_away_def'] = df['home_attack_strength'] - df['away_defense_strength']
        margin_features.append('home_attack_vs_away_def')

    if 'away_attack_strength' in df.columns and 'home_defense_strength' in df.columns:
        df['away_attack_vs_home_def'] = df['away_attack_strength'] - df['home_defense_strength']
        margin_features.append('away_attack_vs_home_def')

    if 'home_goals_scored_ema' in df.columns and 'home_goals_conceded_ema' in df.columns:
        df['home_net_goals_ema'] = df['home_goals_scored_ema'] - df['home_goals_conceded_ema']
        margin_features.append('home_net_goals_ema')

    if 'away_goals_scored_ema' in df.columns and 'away_goals_conceded_ema' in df.columns:
        df['away_net_goals_ema'] = df['away_goals_scored_ema'] - df['away_goals_conceded_ema']
        margin_features.append('away_net_goals_ema')

    if 'home_net_goals_ema' in df.columns and 'away_net_goals_ema' in df.columns:
        df['net_goals_diff'] = df['home_net_goals_ema'] - df['away_net_goals_ema']
        margin_features.append('net_goals_diff')

    # Expected margin from Poisson
    if 'home_xg_poisson' in df.columns and 'away_xg_poisson' in df.columns:
        df['xg_margin'] = df['home_xg_poisson'] - df['away_xg_poisson']
        margin_features.append('xg_margin')

    # League average adjustments
    if 'home_avg_goal_diff' in df.columns and 'away_avg_goal_diff' in df.columns:
        df['form_margin'] = df['home_avg_goal_diff'] - df['away_avg_goal_diff']
        margin_features.append('form_margin')

    return df, margin_features


def train_margin_model(X_train, y_train, X_val, y_val, n_trials=50):
    """Train regression models to predict goal margin."""

    print("\n--- Training Margin Prediction Models ---")

    models = {}
    predictions = {}

    # XGBoost Regression
    print("\nTuning XGBoost Regressor...")
    def xgb_objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 2, 8),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 30),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'random_state': 42, 'verbosity': 0
        }
        model = XGBRegressor(**params)
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        # Minimize MAE
        return np.mean(np.abs(pred - y_val))

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(xgb_objective, n_trials=n_trials, show_progress_bar=False)

    xgb = XGBRegressor(**study.best_params, random_state=42, verbosity=0)
    xgb.fit(X_train, y_train)
    models['XGBoost'] = xgb
    predictions['XGBoost'] = xgb.predict(X_val)
    print(f"  XGBoost MAE: {np.mean(np.abs(predictions['XGBoost'] - y_val)):.3f}")

    # LightGBM Regression
    print("\nTuning LightGBM Regressor...")
    def lgbm_objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 2, 10),
            'num_leaves': trial.suggest_int('num_leaves', 10, 100),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 60),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'random_state': 42, 'verbose': -1
        }
        model = LGBMRegressor(**params)
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        return np.mean(np.abs(pred - y_val))

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lgbm_objective, n_trials=n_trials, show_progress_bar=False)

    lgbm = LGBMRegressor(**study.best_params, random_state=42, verbose=-1)
    lgbm.fit(X_train, y_train)
    models['LightGBM'] = lgbm
    predictions['LightGBM'] = lgbm.predict(X_val)
    print(f"  LightGBM MAE: {np.mean(np.abs(predictions['LightGBM'] - y_val)):.3f}")

    # CatBoost Regression
    print("\nTuning CatBoost Regressor...")
    def cat_objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 50, 300),
            'depth': trial.suggest_int('depth', 3, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.1, 30.0),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'random_state': 42, 'verbose': 0
        }
        model = CatBoostRegressor(**params)
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        return np.mean(np.abs(pred - y_val))

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(cat_objective, n_trials=n_trials, show_progress_bar=False)

    cat = CatBoostRegressor(**study.best_params, random_state=42, verbose=0)
    cat.fit(X_train, y_train)
    models['CatBoost'] = cat
    predictions['CatBoost'] = cat.predict(X_val)
    print(f"  CatBoost MAE: {np.mean(np.abs(predictions['CatBoost'] - y_val)):.3f}")

    # Ridge Regression (baseline)
    print("\nTraining Ridge Regression...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_scaled, y_train)
    models['Ridge'] = (ridge, scaler)
    predictions['Ridge'] = ridge.predict(X_val_scaled)
    print(f"  Ridge MAE: {np.mean(np.abs(predictions['Ridge'] - y_val)):.3f}")

    # Ensemble
    predictions['Ensemble'] = np.mean([predictions['XGBoost'], predictions['LightGBM'], predictions['CatBoost']], axis=0)
    print(f"\n  Ensemble MAE: {np.mean(np.abs(predictions['Ensemble'] - y_val)):.3f}")

    return models, predictions


def value_betting_strategy(predicted_margin, ah_line, actual_margin, odds_home, odds_away,
                          value_threshold=0.25, min_edge=0.0):
    """
    Value-based AH betting strategy.

    Bet home covers when: predicted_margin > -ah_line + value_threshold
    Bet away covers when: predicted_margin < -ah_line - value_threshold

    The value_threshold represents how much our prediction must differ from the line.
    """
    results = []

    n = len(predicted_margin)

    # Calculate edge: difference between our predicted margin and bookmaker's implied margin
    bookie_margin = -ah_line  # If line is -0.5, bookie expects home to win by 0.5
    edge = predicted_margin - bookie_margin

    for i in range(n):
        bet = None
        odds = None

        # Bet home covers if we predict margin higher than line suggests
        if edge[i] > value_threshold and edge[i] > min_edge:
            bet = 'home'
            odds = odds_home[i]
            win = actual_margin[i] + ah_line[i] > 0
        # Bet away covers if we predict margin lower than line suggests
        elif edge[i] < -value_threshold and edge[i] < -min_edge:
            bet = 'away'
            odds = odds_away[i]
            win = actual_margin[i] + ah_line[i] < 0

        if bet:
            results.append({
                'bet': bet,
                'odds': odds,
                'win': win,
                'edge': abs(edge[i]),
                'predicted_margin': predicted_margin[i],
                'ah_line': ah_line[i],
                'actual_margin': actual_margin[i]
            })

    return pd.DataFrame(results)


def calc_roi_bootstrap(wins, odds, n_boot=1000):
    """Calculate ROI with bootstrap CI."""
    if len(wins) == 0:
        return 0, 0, 0, 0

    wins = np.array(wins)
    odds = np.array(odds)

    rois = []
    for _ in range(n_boot):
        idx = np.random.choice(len(wins), len(wins), replace=True)
        w, o = wins[idx], odds[idx]
        profit = (w * (o - 1) - (~w) * 1).sum()
        rois.append(profit / len(w) * 100)

    return np.mean(rois), np.percentile(rois, 2.5), np.percentile(rois, 97.5), (np.array(rois) > 0).mean()


def run_ah_optimization():
    """Run the full AH optimization pipeline."""

    print("=" * 80)
    print("SPECIALIZED ASIAN HANDICAP OPTIMIZATION")
    print("=" * 80)

    # Load data
    df = load_ah_data()
    print(f"\nTotal matches with AH data: {len(df)}")
    print(f"AH line range: {df['ah_line'].min():.2f} to {df['ah_line'].max():.2f}")
    print(f"Home covers rate: {df['home_covers'].mean():.1%}")
    print(f"Away covers rate: {df['away_covers'].mean():.1%}")

    # Create margin features
    df, margin_features = create_margin_features(df)
    print(f"\nMargin features created: {len(margin_features)}")

    # Prepare data
    X = df[margin_features].copy()
    y_margin = df['goal_margin'].values
    ah_line = df['ah_line'].values
    odds_home = df['avg_ah_home'].values
    odds_away = df['avg_ah_away'].values
    dates = pd.to_datetime(df['date'])

    # Fill missing values
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())

    # Time-based split
    sorted_idx = dates.argsort()
    n = len(X)
    train_idx = sorted_idx[:int(0.6*n)]
    val_idx = sorted_idx[int(0.6*n):int(0.8*n)]
    test_idx = sorted_idx[int(0.8*n):]

    X_train, y_train = X.iloc[train_idx].values, y_margin[train_idx]
    X_val, y_val = X.iloc[val_idx].values, y_margin[val_idx]
    X_test, y_test = X.iloc[test_idx].values, y_margin[test_idx]

    ah_line_test = ah_line[test_idx]
    odds_home_test = odds_home[test_idx]
    odds_away_test = odds_away[test_idx]

    print(f"\nTrain: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Train margin models
    models, val_predictions = train_margin_model(X_train, y_train, X_val, y_val, n_trials=50)

    # Get test predictions
    print("\n--- Test Set Predictions ---")
    test_preds = {}
    for name, model in models.items():
        if name == 'Ridge':
            model, scaler = model
            pred = model.predict(scaler.transform(X_test))
        else:
            pred = model.predict(X_test)
        test_preds[name] = pred
        mae = np.mean(np.abs(pred - y_test))
        print(f"{name}: MAE = {mae:.3f}")

    # Ensemble prediction
    test_preds['Ensemble'] = np.mean([test_preds['XGBoost'], test_preds['LightGBM'], test_preds['CatBoost']], axis=0)
    print(f"Ensemble: MAE = {np.mean(np.abs(test_preds['Ensemble'] - y_test)):.3f}")

    # Evaluate value betting strategies
    print("\n" + "=" * 80)
    print("VALUE BETTING STRATEGIES")
    print("=" * 80)

    results = []

    for model_name, pred in test_preds.items():
        for value_thresh in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]:
            bets_df = value_betting_strategy(
                pred, ah_line_test, y_test,
                odds_home_test, odds_away_test,
                value_threshold=value_thresh
            )

            if len(bets_df) < 30:
                continue

            wins = bets_df['win'].values
            odds = bets_df['odds'].values

            roi, ci_low, ci_high, p_profit = calc_roi_bootstrap(wins, odds)

            results.append({
                'model': model_name,
                'threshold': value_thresh,
                'bets': len(bets_df),
                'win_rate': wins.mean(),
                'avg_odds': odds.mean(),
                'roi': roi,
                'ci_low': ci_low,
                'ci_high': ci_high,
                'p_profit': p_profit,
                'home_bets': (bets_df['bet'] == 'home').sum(),
                'away_bets': (bets_df['bet'] == 'away').sum()
            })

    results_df = pd.DataFrame(results).sort_values('roi', ascending=False)

    print(f"\n{'Model':<12} {'Thresh':>6} {'Bets':>6} {'Win%':>7} {'ROI':>9} {'95% CI':>24} {'P(profit)':>10}")
    print("-" * 90)

    for _, row in results_df.head(20).iterrows():
        ci_str = f"[{row['ci_low']:+.1f}% to {row['ci_high']:+.1f}%]"
        print(f"{row['model']:<12} {row['threshold']:>6.2f} {row['bets']:>6} {row['win_rate']:>6.1%} "
              f"{row['roi']:>+8.1f}% {ci_str:>24} {row['p_profit']:>9.0%}")

    # Analyze by line type
    print("\n" + "=" * 80)
    print("ANALYSIS BY AH LINE TYPE")
    print("=" * 80)

    best_model = results_df.iloc[0]['model'] if len(results_df) > 0 else 'Ensemble'
    best_pred = test_preds[best_model]

    line_results = []
    for line_type in [(-3, -1.5), (-1.5, -0.5), (-0.5, 0), (0, 0.5), (0.5, 1.5), (1.5, 3)]:
        mask = (ah_line_test >= line_type[0]) & (ah_line_test < line_type[1])
        if mask.sum() < 30:
            continue

        bets_df = value_betting_strategy(
            best_pred[mask], ah_line_test[mask], y_test[mask],
            odds_home_test[mask], odds_away_test[mask],
            value_threshold=0.25
        )

        if len(bets_df) < 10:
            continue

        wins = bets_df['win'].values
        odds = bets_df['odds'].values
        roi = (wins * (odds - 1) - (~wins) * 1).sum() / len(wins) * 100

        line_results.append({
            'line_range': f"{line_type[0]:+.1f} to {line_type[1]:+.1f}",
            'matches': mask.sum(),
            'bets': len(bets_df),
            'win_rate': wins.mean(),
            'roi': roi
        })

    print(f"\n{'Line Range':<18} {'Matches':>8} {'Bets':>6} {'Win%':>7} {'ROI':>9}")
    print("-" * 55)
    for r in line_results:
        print(f"{r['line_range']:<18} {r['matches']:>8} {r['bets']:>6} {r['win_rate']:>6.1%} {r['roi']:>+8.1f}%")

    # Compare to classification approach
    print("\n" + "=" * 80)
    print("COMPARISON: REGRESSION VS CLASSIFICATION")
    print("=" * 80)

    # Classification baseline (from original approach)
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier

    y_class = (y_margin + ah_line > 0).astype(int)  # Home covers
    y_class_train = y_class[train_idx]
    y_class_val = y_class[val_idx]
    y_class_test = y_class[test_idx]

    clf = XGBClassifier(n_estimators=100, max_depth=4, random_state=42, verbosity=0)
    clf.fit(X_train, y_class_train)
    clf_proba = clf.predict_proba(X_test)[:, 1]
    clf_pred = (clf_proba >= 0.5).astype(int)

    # Classification ROI
    clf_wins = y_class_test[clf_pred == 1] == 1
    clf_odds = odds_home_test[clf_pred == 1]
    clf_profit = (clf_wins * (clf_odds - 1) - (~clf_wins) * 1).sum()
    clf_roi = clf_profit / clf_pred.sum() * 100 if clf_pred.sum() > 0 else 0

    print(f"\nClassification (predict home covers):")
    print(f"  Bets: {clf_pred.sum()}, Win rate: {clf_wins.mean():.1%}, ROI: {clf_roi:+.1f}%")

    # Best regression-based
    if len(results_df) > 0:
        best = results_df.iloc[0]
        print(f"\nRegression (value betting):")
        print(f"  Best: {best['model']} with threshold {best['threshold']}")
        print(f"  Bets: {best['bets']}, Win rate: {best['win_rate']:.1%}, ROI: {best['roi']:+.1f}%")
        print(f"  P(profit): {best['p_profit']:.0%}")

    # Save results
    output = {
        'approach': 'regression_value_betting',
        'total_matches': len(df),
        'test_matches': len(X_test),
        'model_maes': {name: float(np.mean(np.abs(test_preds[name] - y_test))) for name in test_preds},
        'best_strategy': f"{results_df.iloc[0]['model']} thresh={results_df.iloc[0]['threshold']}" if len(results_df) > 0 else None,
        'best_roi': float(results_df.iloc[0]['roi']) if len(results_df) > 0 else None,
        'best_p_profit': float(results_df.iloc[0]['p_profit']) if len(results_df) > 0 else None,
        'best_bets': int(results_df.iloc[0]['bets']) if len(results_df) > 0 else None,
        'all_results': results_df.to_dict('records') if len(results_df) > 0 else [],
        'line_analysis': line_results,
        'classification_baseline': {
            'bets': int(clf_pred.sum()),
            'win_rate': float(clf_wins.mean()),
            'roi': float(clf_roi)
        }
    }

    output_path = 'experiments/outputs/asian_handicap_specialized.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to {output_path}")

    return output


if __name__ == '__main__':
    run_ah_optimization()
