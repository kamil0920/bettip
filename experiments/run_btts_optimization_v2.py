"""
BTTS Optimization v2 - With calibrated odds based on actual BTTS rates.

Uses market-calibrated odds estimation since historical BTTS odds
aren't available from football-data.co.uk or api-football.
"""
import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import RidgeClassifierCV
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

print("=" * 70)
print("BTTS OPTIMIZATION v2 - Calibrated Odds")
print("=" * 70)

# Load data
df = pd.read_csv('/home/kamil/projects/bettip/data/03-features/features_all_5leagues_with_odds.csv')
print(f"Loaded {len(df)} matches")

# Load raw match data for BTTS target
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
                    all_matches.append(matches_df[['fixture_id', 'goals.home', 'goals.away']].copy())

goals_df = pd.concat(all_matches, ignore_index=True).dropna()
goals_df['fixture_id'] = goals_df['fixture_id'].astype(int)
goals_df['btts'] = ((goals_df['goals.home'] > 0) & (goals_df['goals.away'] > 0)).astype(int)

# Merge
df['fixture_id'] = df['fixture_id'].astype(int)
df = df.merge(goals_df[['fixture_id', 'btts']], on='fixture_id', how='left')
df_btts = df[df['btts'].notna()].copy()

print(f"Matches with BTTS data: {len(df_btts)}")
btts_rate = df_btts['btts'].mean()
print(f"BTTS Yes rate: {btts_rate:.1%}")

# Calculate calibrated odds based on actual rates
# Assuming 5% bookmaker margin
MARGIN = 0.05
true_yes_prob = btts_rate
true_no_prob = 1 - btts_rate

# Apply margin proportionally
margin_yes = MARGIN * true_yes_prob / (true_yes_prob + true_no_prob)
margin_no = MARGIN * true_no_prob / (true_yes_prob + true_no_prob)

BTTS_YES_ODDS = 1 / (true_yes_prob + margin_yes)
BTTS_NO_ODDS = 1 / (true_no_prob + margin_no)

print(f"\nCalibrated odds:")
print(f"  BTTS Yes: {BTTS_YES_ODDS:.2f} (break-even: {1/BTTS_YES_ODDS:.1%})")
print(f"  BTTS No: {BTTS_NO_ODDS:.2f} (break-even: {1/BTTS_NO_ODDS:.1%})")

# Features
exclude_cols = [
    'fixture_id', 'date', 'home_team_id', 'home_team_name', 'away_team_id',
    'away_team_name', 'round', 'match_result', 'home_win', 'draw', 'away_win',
    'total_goals', 'goal_difference', 'league', 'btts',
]

# Remove all odds columns
feature_cols = [c for c in df_btts.columns if c not in exclude_cols]
feature_cols = [c for c in feature_cols if 'b365' not in c.lower()
                and 'avg_' not in c[:4]
                and 'max_' not in c[:4]
                and 'pinnacle' not in c.lower()
                and '_ah_' not in c]
feature_cols = [c for c in feature_cols if df_btts[c].notna().sum() > len(df_btts) * 0.5]

print(f"\nFeatures: {len(feature_cols)}")

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

print(f"\nTrain: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
print(f"Train BTTS rate: {y_train.mean():.1%}, Test BTTS rate: {y_test.mean():.1%}")

# Feature Selection
print("\n" + "=" * 70)
print("STEP 1: Feature Selection (Permutation Importance)")
print("=" * 70)

xgb_init = XGBClassifier(n_estimators=100, max_depth=4, random_state=42, verbosity=0)
xgb_init.fit(X_train, y_train)
perm = permutation_importance(xgb_init, X_val, y_val, n_repeats=10, random_state=42, n_jobs=-1)

importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': perm.importances_mean
}).sort_values('importance', ascending=False)

selected_features = importance_df[importance_df['importance'] > 0].head(50)['feature'].tolist()
if len(selected_features) < 20:
    selected_features = importance_df.head(30)['feature'].tolist()

print(f"Selected {len(selected_features)} features")
print("\nTop 10:")
for _, row in importance_df.head(10).iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

# Hyperparameter tuning with Optuna
print("\n" + "=" * 70)
print("STEP 2: Optuna Hyperparameter Tuning")
print("=" * 70)

def xgb_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 2, 6),
        'min_child_weight': trial.suggest_int('min_child_weight', 5, 50),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'random_state': 42,
        'verbosity': 0
    }
    model = XGBClassifier(**params)
    model.fit(X_train[selected_features], y_train)
    return model.score(X_val[selected_features], y_val)

study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(xgb_objective, n_trials=50, show_progress_bar=False)
best_xgb_params = study.best_params
print(f"XGBoost best val accuracy: {study.best_value:.4f}")

def lgbm_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 2, 6),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'random_state': 42,
        'verbose': -1
    }
    model = LGBMClassifier(**params)
    model.fit(X_train[selected_features], y_train)
    return model.score(X_val[selected_features], y_val)

study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(lgbm_objective, n_trials=50, show_progress_bar=False)
best_lgbm_params = study.best_params
print(f"LightGBM best val accuracy: {study.best_value:.4f}")

# Train final callibration
print("\n" + "=" * 70)
print("STEP 3: Train Calibrated Models")
print("=" * 70)

xgb = XGBClassifier(**best_xgb_params, random_state=42, verbosity=0)
xgb.fit(X_train[selected_features], y_train)
xgb_cal = CalibratedClassifierCV(xgb, method='sigmoid', cv='prefit')
xgb_cal.fit(X_val[selected_features], y_val)

lgbm = LGBMClassifier(**best_lgbm_params, random_state=42, verbose=-1)
lgbm.fit(X_train[selected_features], y_train)
lgbm_cal = CalibratedClassifierCV(lgbm, method='sigmoid', cv='prefit')
lgbm_cal.fit(X_val[selected_features], y_val)

cat = CatBoostClassifier(iterations=200, depth=4, l2_leaf_reg=10, learning_rate=0.05, random_state=42, verbose=0)
cat.fit(X_train[selected_features], y_train)
cat_cal = CalibratedClassifierCV(cat, method='sigmoid', cv='prefit')
cat_cal.fit(X_val[selected_features], y_val)

# Stacking ensemble
xgb_val = xgb_cal.predict_proba(X_val[selected_features])[:, 1]
lgbm_val = lgbm_cal.predict_proba(X_val[selected_features])[:, 1]
cat_val = cat_cal.predict_proba(X_val[selected_features])[:, 1]

X_stack_val = np.column_stack([xgb_val, lgbm_val, cat_val])
meta = RidgeClassifierCV(alphas=[0.01, 0.1, 1.0, 10.0])
meta.fit(X_stack_val, y_val)

# Test predictions
xgb_test = xgb_cal.predict_proba(X_test[selected_features])[:, 1]
lgbm_test = lgbm_cal.predict_proba(X_test[selected_features])[:, 1]
cat_test = cat_cal.predict_proba(X_test[selected_features])[:, 1]

X_stack_test = np.column_stack([xgb_test, lgbm_test, cat_test])
stack_proba = 1 / (1 + np.exp(-meta.decision_function(X_stack_test)))
avg_proba = (xgb_test + lgbm_test + cat_test) / 3

print(f"\nTest Accuracy:")
print(f"  XGBoost: {((xgb_test >= 0.5) == y_test).mean():.4f}")
print(f"  LightGBM: {((lgbm_test >= 0.5) == y_test).mean():.4f}")
print(f"  CatBoost: {((cat_test >= 0.5) == y_test).mean():.4f}")
print(f"  Stacking: {((stack_proba >= 0.5) == y_test).mean():.4f}")
print(f"  Average: {((avg_proba >= 0.5) == y_test).mean():.4f}")

# Betting simulation
print("\n" + "=" * 70)
print("STEP 4: Betting Simulation")
print("=" * 70)
print(f"\nUsing calibrated odds: BTTS Yes = {BTTS_YES_ODDS:.2f}, BTTS No = {BTTS_NO_ODDS:.2f}")

def calculate_roi_bootstrap(predictions, actual, odds, n_bootstrap=1000):
    n = len(predictions)
    rois = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        p, a, o = predictions[idx], actual[idx], odds[idx]
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

# BTTS Yes strategies (using stacking)
print("\nBTTS YES strategies:")
for thresh in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75]:
    pred = (stack_proba >= thresh).astype(int)
    n_bets = pred.sum()
    if n_bets < 20:
        continue

    odds = np.full(len(y_test), BTTS_YES_ODDS)
    precision = y_test[pred == 1].mean()
    roi, ci_low, ci_high, p_profit = calculate_roi_bootstrap(pred, y_test, odds)

    results.append({
        'strategy': f'BTTS Yes >= {thresh}',
        'bets': n_bets,
        'precision': precision,
        'roi': roi,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'p_profit': p_profit
    })

# BTTS No strategies
print("\nBTTS NO strategies:")
for thresh in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75]:
    no_proba = 1 - stack_proba
    pred = (no_proba >= thresh).astype(int)
    n_bets = pred.sum()
    if n_bets < 20:
        continue

    btts_no = 1 - y_test
    odds = np.full(len(y_test), BTTS_NO_ODDS)
    precision = btts_no[pred == 1].mean()
    roi, ci_low, ci_high, p_profit = calculate_roi_bootstrap(pred, btts_no, odds)

    results.append({
        'strategy': f'BTTS No >= {thresh}',
        'bets': n_bets,
        'precision': precision,
        'roi': roi,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'p_profit': p_profit
    })

# Print results
print(f"\n{'Strategy':<20} {'Bets':>6} {'Prec':>8} {'ROI':>10} {'95% CI':>22} {'P(profit)':>10}")
print("-" * 80)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('roi', ascending=False)

for _, row in results_df.iterrows():
    print(f"{row['strategy']:<20} {row['bets']:>6} {row['precision']:>7.1%} {row['roi']:>9.1f}% "
          f"[{row['ci_low']:>7.1f}% to {row['ci_high']:>+7.1f}%] {row['p_profit']:>9.0%}")

if len(results_df) > 0:
    best = results_df.iloc[0]
    print("\n" + "=" * 70)
    print(f"BEST: {best['strategy']} | ROI: {best['roi']:.1f}% | P(profit): {best['p_profit']:.0%}")
    print("=" * 70)

    # Check if profitable
    if best['roi'] > 0 and best['p_profit'] >= 0.9:
        print("\n✅ BTTS betting appears PROFITABLE with calibrated odds!")
    elif best['roi'] > 0:
        print("\n⚠️ BTTS shows positive ROI but needs more validation")
    else:
        print("\n❌ BTTS NOT profitable with calibrated odds")

# Save results
output = {
    'bet_type': 'btts',
    'btts_rate': float(btts_rate),
    'calibrated_odds': {
        'btts_yes': float(BTTS_YES_ODDS),
        'btts_no': float(BTTS_NO_ODDS)
    },
    'test_accuracy': {
        'xgboost': float(((xgb_test >= 0.5) == y_test).mean()),
        'lightgbm': float(((lgbm_test >= 0.5) == y_test).mean()),
        'stacking': float(((stack_proba >= 0.5) == y_test).mean())
    },
    'best_strategy': best['strategy'] if len(results_df) > 0 else None,
    'best_roi': float(best['roi']) if len(results_df) > 0 else None,
    'best_p_profit': float(best['p_profit']) if len(results_df) > 0 else None,
    'results': results_df.to_dict('records') if len(results_df) > 0 else []
}

with open('/home/kamil/projects/bettip/experiments/outputs/btts_optimization_v2.json', 'w') as f:
    json.dump(output, f, indent=2)

print("\nResults saved to experiments/outputs/btts_optimization_v2.json")
