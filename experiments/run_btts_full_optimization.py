"""
BTTS Full Optimization Pipeline

Same flow as Away Win optimization:
1. Permutation importance feature selection PER MODEL
2. Optuna tuning per model with its own best features
3. Probability calibration
4. Stacking ensemble
5. Betting optimization
"""
import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

print("=" * 70)
print("BTTS FULL OPTIMIZATION PIPELINE")
print("=" * 70)

# ============================================================
# STEP 0: Load and prepare data
# ============================================================
print("\n" + "=" * 70)
print("STEP 0: Data Preparation")
print("=" * 70)

df = pd.read_csv('/home/kamil/projects/bettip/data/03-features/features_all_5leagues_with_odds.csv')

# Load BTTS target
import pyarrow.parquet as pq

leagues = ['premier_league', 'la_liga', 'serie_a', 'bundesliga', 'ligue_1']
all_matches = []

for league in leagues:
    league_dir = Path(f'data/01-raw/{league}')
    for season_dir in league_dir.iterdir():
        if season_dir.is_dir():
            matches_file = season_dir / 'matches.parquet'
            if matches_file.exists():
                matches_df = pd.read_parquet(matches_file)
                if 'fixture.id' in matches_df.columns:
                    matches_df['fixture_id'] = matches_df['fixture.id']
                if 'goals.home' in matches_df.columns and 'goals.away' in matches_df.columns:
                    sub = matches_df[['fixture_id', 'goals.home', 'goals.away']].copy()
                    all_matches.append(sub)

goals_df = pd.concat(all_matches, ignore_index=True).dropna()
goals_df['fixture_id'] = goals_df['fixture_id'].astype(int)
goals_df['btts'] = ((goals_df['goals.home'] > 0) & (goals_df['goals.away'] > 0)).astype(int)

df['fixture_id'] = df['fixture_id'].astype(int)
df = df.merge(goals_df[['fixture_id', 'btts']], on='fixture_id', how='left')
df_btts = df[df['btts'].notna()].copy()

print(f"Matches: {len(df_btts)}")
btts_rate = df_btts['btts'].mean()
print(f"BTTS rate: {btts_rate:.1%}")

# Create additional BTTS-specific features
df_btts = df_btts.sort_values('date').reset_index(drop=True)

# Scoring probability proxies
if 'home_goals_scored_ema' in df_btts.columns:
    df_btts['home_scoring_rate'] = df_btts['home_goals_scored_ema'] / 1.5
if 'away_goals_scored_ema' in df_btts.columns:
    df_btts['away_scoring_rate'] = df_btts['away_goals_scored_ema'] / 1.5
if 'home_goals_conceded_ema' in df_btts.columns:
    df_btts['home_concede_rate'] = df_btts['home_goals_conceded_ema'] / 1.5
if 'away_goals_conceded_ema' in df_btts.columns:
    df_btts['away_concede_rate'] = df_btts['away_goals_conceded_ema'] / 1.5

# Composite features
if all(c in df_btts.columns for c in ['home_scoring_rate', 'away_concede_rate', 'away_scoring_rate', 'home_concede_rate']):
    df_btts['home_scores_prob'] = (df_btts['home_scoring_rate'] * df_btts['away_concede_rate']).clip(0, 1)
    df_btts['away_scores_prob'] = (df_btts['away_scoring_rate'] * df_btts['home_concede_rate']).clip(0, 1)
    df_btts['btts_composite'] = df_btts['home_scores_prob'] * df_btts['away_scores_prob']

if 'home_goals_scored_ema' in df_btts.columns and 'away_goals_scored_ema' in df_btts.columns:
    df_btts['total_attack'] = df_btts['home_goals_scored_ema'] + df_btts['away_goals_scored_ema']
    df_btts['min_attack'] = df_btts[['home_goals_scored_ema', 'away_goals_scored_ema']].min(axis=1)

if 'home_goals_conceded_ema' in df_btts.columns and 'away_goals_conceded_ema' in df_btts.columns:
    df_btts['total_concede'] = df_btts['home_goals_conceded_ema'] + df_btts['away_goals_conceded_ema']

if 'home_xg_poisson' in df_btts.columns and 'away_xg_poisson' in df_btts.columns:
    df_btts['xg_product'] = df_btts['home_xg_poisson'] * df_btts['away_xg_poisson']
    df_btts['xg_sum'] = df_btts['home_xg_poisson'] + df_btts['away_xg_poisson']
    df_btts['xg_min'] = df_btts[['home_xg_poisson', 'away_xg_poisson']].min(axis=1)

if 'home_scoring_streak' in df_btts.columns and 'away_scoring_streak' in df_btts.columns:
    df_btts['combined_scoring_streak'] = df_btts['home_scoring_streak'] + df_btts['away_scoring_streak']

if 'home_clean_sheet_streak' in df_btts.columns and 'away_clean_sheet_streak' in df_btts.columns:
    df_btts['max_cs_streak'] = df_btts[['home_clean_sheet_streak', 'away_clean_sheet_streak']].max(axis=1)

# League BTTS rate
league_btts = df_btts.groupby('league')['btts'].mean()
df_btts['league_btts_rate'] = df_btts['league'].map(league_btts)

# All features
exclude_cols = [
    'fixture_id', 'date', 'home_team_id', 'home_team_name', 'away_team_id',
    'away_team_name', 'round', 'match_result', 'home_win', 'draw', 'away_win',
    'total_goals', 'goal_difference', 'league', 'btts',
]

feature_cols = [c for c in df_btts.columns if c not in exclude_cols]
feature_cols = [c for c in feature_cols if 'b365' not in c.lower()
                and 'avg_' not in c[:4]
                and 'max_' not in c[:4]
                and 'pinnacle' not in c.lower()
                and '_ah_' not in c]
feature_cols = [c for c in feature_cols if df_btts[c].notna().sum() > len(df_btts) * 0.5]

print(f"Total features available: {len(feature_cols)}")

# Prepare data
X = df_btts[feature_cols].copy()
y = df_btts['btts'].values.astype(int)
dates = pd.to_datetime(df_btts['date'])

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

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# ============================================================
# STEP 1: Permutation Importance Feature Selection PER MODEL
# ============================================================
print("\n" + "=" * 70)
print("STEP 1: Permutation Importance Feature Selection (Per Model)")
print("=" * 70)

# Scale for LogisticRegression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

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
        'importance': perm.importances_mean,
        'std': perm.importances_std
    }).sort_values('importance', ascending=False)

    # Select features with positive importance
    top_features = importance_df[importance_df['importance'] > 0].head(60)['feature'].tolist()
    if len(top_features) < 25:
        top_features = importance_df.head(30)['feature'].tolist()

    features_per_model[name] = top_features
    print(f"  Selected {len(top_features)} features")
    print(f"  Top 5: {top_features[:5]}")

# ============================================================
# STEP 2: Optuna Hyperparameter Tuning (Per Model)
# ============================================================
print("\n" + "=" * 70)
print("STEP 2: Optuna Hyperparameter Tuning (Per Model)")
print("=" * 70)

best_params = {}
N_TRIALS = 80

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
    model = XGBClassifier(**params)
    model.fit(X_train[xgb_features], y_train)
    return model.score(X_val[xgb_features], y_val)

study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(xgb_objective, n_trials=N_TRIALS, show_progress_bar=False)
best_params['XGBoost'] = study.best_params
print(f"  Best val accuracy: {study.best_value:.4f}")

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
    model = LGBMClassifier(**params)
    model.fit(X_train[lgbm_features], y_train)
    return model.score(X_val[lgbm_features], y_val)

study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(lgbm_objective, n_trials=N_TRIALS, show_progress_bar=False)
best_params['LightGBM'] = study.best_params
print(f"  Best val accuracy: {study.best_value:.4f}")

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
        'border_count': trial.suggest_int('border_count', 32, 255),
        'random_state': 42, 'verbose': 0
    }
    model = CatBoostClassifier(**params)
    model.fit(X_train[cat_features], y_train)
    return model.score(X_val[cat_features], y_val)

study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(cat_objective, n_trials=N_TRIALS, show_progress_bar=False)
best_params['CatBoost'] = study.best_params
print(f"  Best val accuracy: {study.best_value:.4f}")

# Logistic Regression
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
study.optimize(lr_objective, n_trials=N_TRIALS, show_progress_bar=False)
best_params['LogisticReg'] = study.best_params
print(f"  Best val accuracy: {study.best_value:.4f}")

# ============================================================
# STEP 3: Train Final Models with Calibration
# ============================================================
print("\n" + "=" * 70)
print("STEP 3: Train Final Models with Calibration")
print("=" * 70)

final_models = {}

# XGBoost
xgb_params = {**best_params['XGBoost'], 'random_state': 42, 'verbosity': 0}
xgb = XGBClassifier(**xgb_params)
xgb.fit(X_train[xgb_features], y_train)
xgb_cal = CalibratedClassifierCV(xgb, method='sigmoid', cv='prefit')
xgb_cal.fit(X_val[xgb_features], y_val)
final_models['XGBoost'] = (xgb_cal, xgb_features)
print(f"XGBoost trained with {len(xgb_features)} features")

# LightGBM
lgbm_params = {**best_params['LightGBM'], 'random_state': 42, 'verbose': -1}
lgbm = LGBMClassifier(**lgbm_params)
lgbm.fit(X_train[lgbm_features], y_train)
lgbm_cal = CalibratedClassifierCV(lgbm, method='sigmoid', cv='prefit')
lgbm_cal.fit(X_val[lgbm_features], y_val)
final_models['LightGBM'] = (lgbm_cal, lgbm_features)
print(f"LightGBM trained with {len(lgbm_features)} features")

# CatBoost
cat_params = {**best_params['CatBoost'], 'random_state': 42, 'verbose': 0}
cat = CatBoostClassifier(**cat_params)
cat.fit(X_train[cat_features], y_train)
cat_cal = CalibratedClassifierCV(cat, method='sigmoid', cv='prefit')
cat_cal.fit(X_val[cat_features], y_val)
final_models['CatBoost'] = (cat_cal, cat_features)
print(f"CatBoost trained with {len(cat_features)} features")

# Logistic Regression
lr_params = {**best_params['LogisticReg'], 'solver': 'saga', 'max_iter': 1000, 'random_state': 42}
lr = LogisticRegression(**lr_params)
lr.fit(X_train_lr, y_train)
lr_cal = CalibratedClassifierCV(lr, method='sigmoid', cv='prefit')
lr_cal.fit(X_val_lr, y_val)
final_models['LogisticReg'] = (lr_cal, lr_features, scaler_lr)
print(f"LogisticReg trained with {len(lr_features)} features")

# ============================================================
# STEP 4: Get Predictions and Create Ensemble
# ============================================================
print("\n" + "=" * 70)
print("STEP 4: Evaluation")
print("=" * 70)

# Test predictions
test_preds = {}
for name, model_data in final_models.items():
    if name == 'LogisticReg':
        model, features, sc = model_data
        X_t = sc.transform(X_test[features])
    else:
        model, features = model_data
        X_t = X_test[features]

    proba = model.predict_proba(X_t)[:, 1]
    test_preds[name] = proba
    acc = ((proba >= 0.5) == y_test).mean()
    print(f"{name}: accuracy = {acc:.4f}")

# Validation predictions for stacking
val_preds = {}
for name, model_data in final_models.items():
    if name == 'LogisticReg':
        model, features, sc = model_data
        X_v = sc.transform(X_val[features])
    else:
        model, features = model_data
        X_v = X_val[features]
    val_preds[name] = model.predict_proba(X_v)[:, 1]

# Simple average
avg_proba = np.mean([test_preds[n] for n in test_preds], axis=0)
print(f"\nSimple Average: {((avg_proba >= 0.5) == y_test).mean():.4f}")

# Stacking
X_stack_val = np.column_stack([val_preds[n] for n in ['XGBoost', 'LightGBM', 'CatBoost', 'LogisticReg']])
X_stack_test = np.column_stack([test_preds[n] for n in ['XGBoost', 'LightGBM', 'CatBoost', 'LogisticReg']])

meta = RidgeClassifierCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0])
meta.fit(X_stack_val, y_val)
stack_proba = 1 / (1 + np.exp(-meta.decision_function(X_stack_test)))
print(f"Stacking: {((stack_proba >= 0.5) == y_test).mean():.4f}")

# ============================================================
# STEP 5: Betting Optimization
# ============================================================
print("\n" + "=" * 70)
print("STEP 5: Betting Optimization")
print("=" * 70)

# Calibrated odds
MARGIN = 0.05
BTTS_YES_ODDS = 1 / (btts_rate + MARGIN * btts_rate)
BTTS_NO_ODDS = 1 / ((1-btts_rate) + MARGIN * (1-btts_rate))
print(f"\nOdds: BTTS Yes = {BTTS_YES_ODDS:.2f}, BTTS No = {BTTS_NO_ODDS:.2f}")
print(f"Break-even: Yes = {1/BTTS_YES_ODDS:.1%}, No = {1/BTTS_NO_ODDS:.1%}")

def calc_roi_bootstrap(pred, actual, odds, n_boot=1000):
    rois = []
    for _ in range(n_boot):
        idx = np.random.choice(len(pred), len(pred), replace=True)
        p, a, o = pred[idx], actual[idx], odds[idx]
        mask = p == 1
        if mask.sum() == 0:
            continue
        wins = a[mask] == 1
        profit = (wins * (o[mask] - 1) - (~wins) * 1).sum()
        rois.append(profit / mask.sum() * 100)
    if not rois:
        return 0, 0, 0, 0
    return np.mean(rois), np.percentile(rois, 2.5), np.percentile(rois, 97.5), (np.array(rois) > 0).mean()

results = []

# Test all callibration and thresholds for BTTS Yes
for model_name in ['XGBoost', 'LightGBM', 'CatBoost', 'LogisticReg', 'Average', 'Stacking']:
    if model_name == 'Average':
        proba = avg_proba
    elif model_name == 'Stacking':
        proba = stack_proba
    else:
        proba = test_preds[model_name]

    for thresh in [0.5, 0.55, 0.6, 0.65, 0.7]:
        pred = (proba >= thresh).astype(int)
        n_bets = pred.sum()
        if n_bets < 20:
            continue

        odds = np.full(len(y_test), BTTS_YES_ODDS)
        prec = y_test[pred == 1].mean()
        roi, ci_low, ci_high, p_profit = calc_roi_bootstrap(pred, y_test, odds)

        results.append({
            'strategy': f'{model_name} Yes >= {thresh}',
            'model': model_name,
            'threshold': thresh,
            'bets': n_bets,
            'precision': prec,
            'roi': roi,
            'ci_low': ci_low,
            'ci_high': ci_high,
            'p_profit': p_profit
        })

# BTTS No strategies
for model_name in ['Stacking', 'Average']:
    if model_name == 'Average':
        proba = avg_proba
    else:
        proba = stack_proba

    no_proba = 1 - proba
    for thresh in [0.55, 0.6, 0.65, 0.7]:
        pred = (no_proba >= thresh).astype(int)
        n_bets = pred.sum()
        if n_bets < 20:
            continue

        btts_no = 1 - y_test
        odds = np.full(len(y_test), BTTS_NO_ODDS)
        prec = btts_no[pred == 1].mean()
        roi, ci_low, ci_high, p_profit = calc_roi_bootstrap(pred, btts_no, odds)

        results.append({
            'strategy': f'{model_name} No >= {thresh}',
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
for _, row in results_df.head(20).iterrows():
    print(f"{row['strategy']:<25} {row['bets']:>5} {row['precision']:>6.1%} {row['roi']:>+8.1f}% "
          f"[{row['ci_low']:>7.1f}% to {row['ci_high']:>+7.1f}%] {row['p_profit']:>9.0%}")

# Best result
if len(results_df) > 0:
    best = results_df.iloc[0]
    print("\n" + "=" * 70)
    print(f"BEST STRATEGY: {best['strategy']}")
    print(f"ROI: {best['roi']:.1f}% | P(profit): {best['p_profit']:.0%} | Bets: {best['bets']}")
    print(f"Precision: {best['precision']:.1%} (break-even: {1/BTTS_YES_ODDS:.1%})")
    print("=" * 70)

# Save results
output = {
    'btts_rate': float(btts_rate),
    'odds': {'btts_yes': float(BTTS_YES_ODDS), 'btts_no': float(BTTS_NO_ODDS)},
    'features_per_model': {k: v if isinstance(v, list) else v[1] for k, v in features_per_model.items()},
    'best_params': {k: {kk: float(vv) if isinstance(vv, (int, float)) else vv for kk, vv in v.items()} for k, v in best_params.items()},
    'model_accuracies': {n: float(((test_preds[n] >= 0.5) == y_test).mean()) for n in test_preds},
    'ensemble_accuracy': float(((stack_proba >= 0.5) == y_test).mean()),
    'best_strategy': best['strategy'],
    'best_roi': float(best['roi']),
    'best_p_profit': float(best['p_profit']),
    'best_bets': int(best['bets']),
    'all_results': results_df.to_dict('records')
}

with open('experiments/outputs/btts_full_optimization.json', 'w') as f:
    json.dump(output, f, indent=2)

print("\nResults saved to experiments/outputs/btts_full_optimization.json")
