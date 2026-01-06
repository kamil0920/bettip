"""
Improved Asian Handicap Optimization

Improvements:
1. New AH-specific features (margin volatility, consistency, form margin)
2. Feature selection using permutation importance
3. Line-specific models
4. Higher confidence thresholds
5. Ensemble of approaches
"""

import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


def create_ah_features(df):
    """Create advanced features for AH margin prediction."""

    df = df.copy()

    # ============================================================
    # 1. MARGIN-BASED FEATURES
    # ============================================================

    # Expected margin from different sources
    if 'home_xg_poisson' in df.columns and 'away_xg_poisson' in df.columns:
        df['xg_margin'] = df['home_xg_poisson'] - df['away_xg_poisson']

    if 'elo_diff' in df.columns:
        # Elo-based expected margin (elo diff correlates with margin)
        df['elo_expected_margin'] = df['elo_diff'] / 400  # Normalize

    if 'season_gd_diff' in df.columns:
        # Season goal difference normalized by games
        df['season_margin_per_game'] = df['season_gd_diff'] / (df['round_number'] + 1)

    # ============================================================
    # 2. FORM MARGIN FEATURES
    # ============================================================

    if 'home_avg_goal_diff' in df.columns and 'away_avg_goal_diff' in df.columns:
        df['form_margin'] = df['home_avg_goal_diff'] - df['away_avg_goal_diff']

    if 'home_goals_scored_ema' in df.columns and 'away_goals_conceded_ema' in df.columns:
        df['home_attack_vs_away_def'] = df['home_goals_scored_ema'] - df['away_goals_conceded_ema']

    if 'away_goals_scored_ema' in df.columns and 'home_goals_conceded_ema' in df.columns:
        df['away_attack_vs_home_def'] = df['away_goals_scored_ema'] - df['home_goals_conceded_ema']

    if 'home_attack_vs_away_def' in df.columns and 'away_attack_vs_home_def' in df.columns:
        df['attack_def_margin'] = df['home_attack_vs_away_def'] - df['away_attack_vs_home_def']

    # ============================================================
    # 3. CONSISTENCY/VOLATILITY FEATURES
    # ============================================================

    # Use goals scored/conceded variance as proxy for margin volatility
    if 'home_goals_scored_last_n' in df.columns and 'home_goals_scored_ema' in df.columns:
        df['home_scoring_consistency'] = 1 - abs(df['home_goals_scored_last_n'] - df['home_goals_scored_ema']) / (df['home_goals_scored_ema'] + 0.1)

    if 'away_goals_scored_last_n' in df.columns and 'away_goals_scored_ema' in df.columns:
        df['away_scoring_consistency'] = 1 - abs(df['away_goals_scored_last_n'] - df['away_goals_scored_ema']) / (df['away_goals_scored_ema'] + 0.1)

    # ============================================================
    # 4. LINE MOVEMENT FEATURES (if available)
    # ============================================================

    if 'ah_line' in df.columns and 'ah_line_close' in df.columns:
        df['line_movement'] = df['ah_line_close'] - df['ah_line']
        df['line_moved_towards_home'] = (df['line_movement'] < 0).astype(int)
        df['line_moved_towards_away'] = (df['line_movement'] > 0).astype(int)
        df['line_movement_abs'] = abs(df['line_movement'])

    # ============================================================
    # 5. LEAGUE-SPECIFIC FEATURES
    # ============================================================

    # League encoding (if league column exists)
    if 'league' in df.columns:
        league_stats = df.groupby('league')['goal_difference'].agg(['mean', 'std']).reset_index()
        league_stats.columns = ['league', 'league_avg_margin', 'league_margin_std']
        df = df.merge(league_stats, on='league', how='left')

        # League home advantage
        league_home_adv = df.groupby('league')['home_win'].mean().reset_index()
        league_home_adv.columns = ['league', 'league_home_win_rate']
        df = df.merge(league_home_adv, on='league', how='left')

    # ============================================================
    # 6. COMPOSITE MARGIN PREDICTION
    # ============================================================

    # Weighted combination of margin signals
    margin_components = []
    weights = []

    if 'elo_expected_margin' in df.columns:
        margin_components.append('elo_expected_margin')
        weights.append(0.3)
    if 'xg_margin' in df.columns:
        margin_components.append('xg_margin')
        weights.append(0.3)
    if 'form_margin' in df.columns:
        margin_components.append('form_margin')
        weights.append(0.2)
    if 'season_margin_per_game' in df.columns:
        margin_components.append('season_margin_per_game')
        weights.append(0.2)

    if margin_components:
        # Normalize weights
        weights = np.array(weights) / sum(weights)
        df['composite_expected_margin'] = sum(
            df[col].fillna(0) * w for col, w in zip(margin_components, weights)
        )

    # ============================================================
    # 7. BOOKMAKER EFFICIENCY FEATURES
    # ============================================================

    if 'ah_line' in df.columns:
        # The line is bookmaker's prediction (inverted)
        df['bookie_expected_margin'] = -df['ah_line']

        # How close is our composite to bookmaker's estimate?
        if 'composite_expected_margin' in df.columns:
            df['margin_edge'] = df['composite_expected_margin'] - df['bookie_expected_margin']

    # ============================================================
    # 8. CATEGORICAL LINE FEATURES
    # ============================================================

    if 'ah_line' in df.columns:
        # Line categories
        df['is_heavy_favorite'] = (df['ah_line'] <= -1.5).astype(int)
        df['is_moderate_favorite'] = ((df['ah_line'] > -1.5) & (df['ah_line'] <= -0.5)).astype(int)
        df['is_slight_favorite'] = ((df['ah_line'] > -0.5) & (df['ah_line'] < 0)).astype(int)
        df['is_even_match'] = (abs(df['ah_line']) <= 0.25).astype(int)
        df['is_underdog'] = (df['ah_line'] > 0.5).astype(int)

        # Half-goal vs integer line
        df['is_half_line'] = (df['ah_line'] % 1 != 0).astype(int)
        df['is_quarter_line'] = ((df['ah_line'] * 4) % 1 == 0).astype(int) & (df['ah_line'] % 0.5 != 0)

    return df


def get_feature_columns(df):
    """Get feature columns for modeling."""

    exclude_cols = [
        'fixture_id', 'date', 'home_team_id', 'home_team_name', 'away_team_id',
        'away_team_name', 'round', 'match_result', 'home_win', 'draw', 'away_win',
        'total_goals', 'goal_difference', 'league', 'goal_margin', 'ah_result',
        'home_covers', 'away_covers',
        # Exclude betting odds to prevent leakage
        'ah_line', 'avg_ah_home', 'avg_ah_away', 'b365_ah_home', 'b365_ah_away',
        'pinnacle_ah_home', 'pinnacle_ah_away', 'max_ah_home', 'max_ah_away',
        'ah_line_close', 'avg_ah_home_close', 'avg_ah_away_close',
        'b365_home_open', 'b365_draw_open', 'b365_away_open',
        'avg_home_open', 'avg_draw_open', 'avg_away_open',
        'avg_over25', 'avg_under25', 'b365_over25', 'b365_under25',
        'bookie_expected_margin',  # This uses ah_line
    ]

    feature_cols = [c for c in df.columns if c not in exclude_cols
                    and df[c].dtype in ['float64', 'int64', 'int32', 'float32']
                    and 'b365' not in c.lower() and 'pinnacle' not in c.lower()
                    and not c.startswith('max_') and not c.startswith('avg_')]

    # Remove features with too many missing values
    feature_cols = [c for c in feature_cols if df[c].notna().sum() > len(df) * 0.5]

    return feature_cols


def select_features(X_train, y_train, X_val, y_val, feature_cols, n_features=50):
    """Select best features using permutation importance."""

    print(f"\nSelecting top {n_features} features from {len(feature_cols)}...")

    model = XGBRegressor(n_estimators=100, max_depth=5, random_state=42, verbosity=0)
    model.fit(X_train, y_train)

    perm = permutation_importance(model, X_val, y_val, n_repeats=15, random_state=42, n_jobs=-1)

    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': perm.importances_mean
    }).sort_values('importance', ascending=False)

    selected = importance_df.head(n_features)['feature'].tolist()

    print(f"Top 10 features: {selected[:10]}")

    return selected


def train_margin_models(X_train, y_train, X_val, y_val, n_trials=40):
    """Train regression models with Optuna tuning."""

    models = {}

    # XGBoost
    print("\nTuning XGBoost...")
    def xgb_obj(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 400),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
            'random_state': 42, 'verbosity': 0
        }
        model = XGBRegressor(**params)
        model.fit(X_train, y_train)
        return np.mean(np.abs(model.predict(X_val) - y_val))

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(xgb_obj, n_trials=n_trials, show_progress_bar=False)
    models['XGBoost'] = XGBRegressor(**study.best_params, random_state=42, verbosity=0)
    models['XGBoost'].fit(X_train, y_train)
    print(f"  XGBoost MAE: {study.best_value:.4f}")

    # LightGBM
    print("\nTuning LightGBM...")
    def lgbm_obj(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 400),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'num_leaves': trial.suggest_int('num_leaves', 15, 80),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15),
            'random_state': 42, 'verbose': -1
        }
        model = LGBMRegressor(**params)
        model.fit(X_train, y_train)
        return np.mean(np.abs(model.predict(X_val) - y_val))

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lgbm_obj, n_trials=n_trials, show_progress_bar=False)
    models['LightGBM'] = LGBMRegressor(**study.best_params, random_state=42, verbose=-1)
    models['LightGBM'].fit(X_train, y_train)
    print(f"  LightGBM MAE: {study.best_value:.4f}")

    # CatBoost
    print("\nTuning CatBoost...")
    def cat_obj(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 100, 400),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.5, 20.0),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15),
            'random_state': 42, 'verbose': 0
        }
        model = CatBoostRegressor(**params)
        model.fit(X_train, y_train)
        return np.mean(np.abs(model.predict(X_val) - y_val))

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(cat_obj, n_trials=n_trials, show_progress_bar=False)
    models['CatBoost'] = CatBoostRegressor(**study.best_params, random_state=42, verbose=0)
    models['CatBoost'].fit(X_train, y_train)
    print(f"  CatBoost MAE: {study.best_value:.4f}")

    return models


def calc_roi_bootstrap(wins, odds, n_boot=1000):
    """Calculate ROI with bootstrap CI."""
    if len(wins) == 0:
        return 0, 0, 0, 0

    wins = np.array(wins).astype(bool)
    odds = np.array(odds)

    profit = (wins * (odds - 1) - (~wins) * 1).sum()
    base_roi = profit / len(wins) * 100

    rois = []
    for _ in range(n_boot):
        idx = np.random.choice(len(wins), len(wins), replace=True)
        p = (wins[idx] * (odds[idx] - 1) - (~wins[idx]) * 1).sum()
        rois.append(p / len(wins) * 100)

    return base_roi, np.percentile(rois, 2.5), np.percentile(rois, 97.5), (np.array(rois) > 0).mean()


def run_improved_ah_optimization():
    """Run the improved AH optimization pipeline."""

    print("=" * 90)
    print("IMPROVED ASIAN HANDICAP OPTIMIZATION")
    print("=" * 90)

    # Load data
    df = pd.read_csv('/home/kamil/projects/bettip/data/03-features/features_all_5leagues_with_odds.csv')
    df = df[df['ah_line'].notna() & df['avg_ah_home'].notna()].copy()
    df['goal_margin'] = df['goal_difference']

    print(f"\nTotal matches: {len(df)}")

    # Create advanced features
    print("\n--- Creating AH-specific features ---")
    df = create_ah_features(df)

    # Get feature columns
    feature_cols = get_feature_columns(df)
    print(f"Total features available: {len(feature_cols)}")

    # Prepare data
    X = df[feature_cols].copy()
    y = df['goal_margin'].values
    ah_line = df['ah_line'].values
    odds_home = df['avg_ah_home'].values
    odds_away = df['avg_ah_away'].values
    dates = pd.to_datetime(df['date'])

    # Fill missing values
    for col in X.columns:
        X[col] = X[col].fillna(X[col].median())

    # Time split
    sorted_idx = dates.argsort()
    n = len(X)
    train_idx = sorted_idx[:int(0.6*n)]
    val_idx = sorted_idx[int(0.6*n):int(0.8*n)]
    test_idx = sorted_idx[int(0.8*n):]

    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_val, y_val = X.iloc[val_idx], y[val_idx]
    X_test, y_test = X.iloc[test_idx], y[test_idx]

    ah_line_test = ah_line[test_idx]
    odds_home_test = odds_home[test_idx]
    odds_away_test = odds_away[test_idx]

    print(f"\nTrain: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Feature selection
    print("\n" + "=" * 90)
    print("STEP 1: FEATURE SELECTION")
    print("=" * 90)

    selected_features = select_features(
        X_train.values, y_train, X_val.values, y_val,
        feature_cols, n_features=50
    )

    X_train_sel = X_train[selected_features].values
    X_val_sel = X_val[selected_features].values
    X_test_sel = X_test[selected_features].values

    # Train models
    print("\n" + "=" * 90)
    print("STEP 2: MODEL TRAINING (40 Optuna trials each)")
    print("=" * 90)

    models = train_margin_models(X_train_sel, y_train, X_val_sel, y_val, n_trials=40)

    # Get predictions
    print("\n" + "=" * 90)
    print("STEP 3: TEST SET EVALUATION")
    print("=" * 90)

    predictions = {}
    for name, model in models.items():
        pred = model.predict(X_test_sel)
        predictions[name] = pred
        mae = np.mean(np.abs(pred - y_test))
        print(f"{name}: MAE = {mae:.4f}")

    # Ensemble
    predictions['Ensemble'] = np.mean([predictions['XGBoost'], predictions['LightGBM'], predictions['CatBoost']], axis=0)
    print(f"Ensemble: MAE = {np.mean(np.abs(predictions['Ensemble'] - y_test)):.4f}")

    # Calculate edge
    bookie_margin = -ah_line_test

    # Betting optimization
    print("\n" + "=" * 90)
    print("STEP 4: BETTING OPTIMIZATION")
    print("=" * 90)

    results = []

    # Strategy 1: Global value betting
    print("\n--- Strategy 1: Global Value Betting ---")

    for model_name, pred in predictions.items():
        edge = pred - bookie_margin

        for thresh in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
            # Home bets
            home_mask = edge > thresh
            if home_mask.sum() >= 20:
                home_wins = (y_test[home_mask] + ah_line_test[home_mask] > 0)
                home_odds = odds_home_test[home_mask]
                roi, ci_low, ci_high, p_profit = calc_roi_bootstrap(home_wins, home_odds)
                results.append({
                    'strategy': f'{model_name} Home edge>{thresh}',
                    'model': model_name,
                    'type': 'home',
                    'threshold': thresh,
                    'line_filter': 'all',
                    'bets': int(home_mask.sum()),
                    'win_rate': float(home_wins.mean()),
                    'roi': float(roi),
                    'ci_low': float(ci_low),
                    'ci_high': float(ci_high),
                    'p_profit': float(p_profit)
                })

            # Away bets
            away_mask = edge < -thresh
            if away_mask.sum() >= 20:
                away_wins = (y_test[away_mask] + ah_line_test[away_mask] < 0)
                away_odds = odds_away_test[away_mask]
                roi, ci_low, ci_high, p_profit = calc_roi_bootstrap(away_wins, away_odds)
                results.append({
                    'strategy': f'{model_name} Away edge>{thresh}',
                    'model': model_name,
                    'type': 'away',
                    'threshold': thresh,
                    'line_filter': 'all',
                    'bets': int(away_mask.sum()),
                    'win_rate': float(away_wins.mean()),
                    'roi': float(roi),
                    'ci_low': float(ci_low),
                    'ci_high': float(ci_high),
                    'p_profit': float(p_profit)
                })

    # Strategy 2: Line-specific betting
    print("\n--- Strategy 2: Line-Specific Betting ---")

    line_filters = {
        'heavy_fav': (ah_line_test <= -1.5),
        'mod_fav': (ah_line_test > -1.5) & (ah_line_test <= -0.5),
        'slight_fav': (ah_line_test > -0.5) & (ah_line_test < 0),
        'even': (ah_line_test >= -0.25) & (ah_line_test <= 0.25),
        'underdog': (ah_line_test > 0.5),
    }

    for line_name, line_mask in line_filters.items():
        if line_mask.sum() < 50:
            continue

        for model_name in ['Ensemble']:
            pred = predictions[model_name]
            edge = pred - bookie_margin

            for thresh in [0.2, 0.3, 0.4, 0.5]:
                # Home bets within line filter
                home_mask = line_mask & (edge > thresh)
                if home_mask.sum() >= 15:
                    home_wins = (y_test[home_mask] + ah_line_test[home_mask] > 0)
                    home_odds = odds_home_test[home_mask]
                    roi, ci_low, ci_high, p_profit = calc_roi_bootstrap(home_wins, home_odds)
                    results.append({
                        'strategy': f'{line_name} Home edge>{thresh}',
                        'model': model_name,
                        'type': 'home',
                        'threshold': thresh,
                        'line_filter': line_name,
                        'bets': int(home_mask.sum()),
                        'win_rate': float(home_wins.mean()),
                        'roi': float(roi),
                        'ci_low': float(ci_low),
                        'ci_high': float(ci_high),
                        'p_profit': float(p_profit)
                    })

                # Away bets within line filter
                away_mask = line_mask & (edge < -thresh)
                if away_mask.sum() >= 15:
                    away_wins = (y_test[away_mask] + ah_line_test[away_mask] < 0)
                    away_odds = odds_away_test[away_mask]
                    roi, ci_low, ci_high, p_profit = calc_roi_bootstrap(away_wins, away_odds)
                    results.append({
                        'strategy': f'{line_name} Away edge>{thresh}',
                        'model': model_name,
                        'type': 'away',
                        'threshold': thresh,
                        'line_filter': line_name,
                        'bets': int(away_mask.sum()),
                        'win_rate': float(away_wins.mean()),
                        'roi': float(roi),
                        'ci_low': float(ci_low),
                        'ci_high': float(ci_high),
                        'p_profit': float(p_profit)
                    })

    # Print results
    results_df = pd.DataFrame(results).sort_values('roi', ascending=False)

    print(f"\n{'Strategy':<35} {'Bets':>5} {'Win%':>7} {'ROI':>9} {'95% CI':>22} {'P(profit)':>10}")
    print("-" * 95)

    for _, row in results_df.head(25).iterrows():
        ci_str = f"[{row['ci_low']:+.1f}% to {row['ci_high']:+.1f}%]"
        print(f"{row['strategy']:<35} {row['bets']:>5} {row['win_rate']:>6.1%} "
              f"{row['roi']:>+8.1f}% {ci_str:>22} {row['p_profit']:>9.0%}")

    # Show profitable strategies
    print("\n" + "=" * 90)
    print("PROFITABLE STRATEGIES (P(profit) >= 70%)")
    print("=" * 90)

    profitable = results_df[results_df['p_profit'] >= 0.7].sort_values('roi', ascending=False)

    if len(profitable) > 0:
        for _, row in profitable.iterrows():
            print(f"\n{row['strategy']}")
            print(f"  Bets: {row['bets']}, Win rate: {row['win_rate']:.1%}")
            print(f"  ROI: {row['roi']:+.1f}% (95% CI: {row['ci_low']:+.1f}% to {row['ci_high']:+.1f}%)")
            print(f"  P(profit): {row['p_profit']:.0%}")
    else:
        print("\nNo strategies with P(profit) >= 70%")

        # Show best strategies anyway
        print("\nBest strategies found:")
        for _, row in results_df.head(5).iterrows():
            print(f"\n{row['strategy']}")
            print(f"  ROI: {row['roi']:+.1f}%, P(profit): {row['p_profit']:.0%}")

    # Save results
    output = {
        'total_matches': len(df),
        'test_matches': len(y_test),
        'selected_features': selected_features,
        'model_maes': {name: float(np.mean(np.abs(predictions[name] - y_test))) for name in predictions},
        'best_strategy': results_df.iloc[0]['strategy'] if len(results_df) > 0 else None,
        'best_roi': float(results_df.iloc[0]['roi']) if len(results_df) > 0 else None,
        'best_p_profit': float(results_df.iloc[0]['p_profit']) if len(results_df) > 0 else None,
        'all_results': results_df.to_dict('records')
    }

    output_path = 'experiments/outputs/ah_improved_optimization.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to {output_path}")

    return output


if __name__ == '__main__':
    run_improved_ah_optimization()
