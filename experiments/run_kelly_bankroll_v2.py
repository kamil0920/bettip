"""
Kelly Criterion Bankroll Management System v2

Key insight: Kelly requires well-calibrated probabilities.
If probabilities are overconfident, Kelly will bet too aggressively and lose.

This version:
- Uses higher probability thresholds
- Compares Kelly vs Flat betting properly
- Applies stricter bet limits
"""
import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

print("=" * 70)
print("KELLY CRITERION BANKROLL MANAGEMENT v2")
print("=" * 70)

# Load data
df = pd.read_csv('/home/kamil/projects/bettip/data/03-features/features_all_5leagues_with_odds.csv')

# Load best params
with open('/home/kamil/projects/bettip/experiments/outputs/optuna_best_params.json', 'r') as f:
    best_params = json.load(f)

with open('/home/kamil/projects/bettip/experiments/outputs/features_per_model.json', 'r') as f:
    features_per_model = json.load(f)

# Prepare data
df_with_odds = df[df['avg_away_open'].notna() & (df['avg_away_open'] > 1)].copy()

exclude_cols = [
    'fixture_id', 'date', 'home_team_id', 'home_team_name', 'away_team_id',
    'away_team_name', 'round', 'match_result', 'home_win', 'draw', 'away_win',
    'total_goals', 'goal_difference', 'league',
    'b365_home_open', 'b365_draw_open', 'b365_away_open',
    'avg_home_open', 'avg_draw_open', 'avg_away_open',
    'b365_home_close', 'b365_draw_close', 'b365_away_close',
    'avg_home_close', 'avg_draw_close', 'avg_away_close'
]

feature_cols = [c for c in df_with_odds.columns if c not in exclude_cols]
feature_cols = [c for c in feature_cols if df_with_odds[c].notna().sum() > len(df_with_odds) * 0.5]

X = df_with_odds[feature_cols].copy()
y = df_with_odds['away_win'].values
dates = pd.to_datetime(df_with_odds['date'])

for col in X.columns:
    if X[col].isna().any():
        X[col] = X[col].fillna(X[col].median())

# Time-based split
sorted_indices = dates.argsort()
n = len(X)
train_idx = sorted_indices[:int(0.6*n)]
val_idx = sorted_indices[int(0.6*n):int(0.8*n)]
test_idx = sorted_indices[int(0.8*n):]

X_train, y_train = X.iloc[train_idx], y[train_idx]
X_val, y_val = X.iloc[val_idx], y[val_idx]
X_test, y_test = X.iloc[test_idx], y[test_idx]

# Train models
print("\nTraining calibrated ensemble models...")

final_models = {}

# XGBoost
xgb_features = [f for f in features_per_model['XGBoost'] if f in X_train.columns]
xgb_params = {k: v for k, v in best_params['XGBoost'].items()}
xgb_params.update({'random_state': 42, 'verbosity': 0})
xgb = XGBClassifier(**xgb_params)
xgb.fit(X_train[xgb_features], y_train)
xgb_cal = CalibratedClassifierCV(xgb, method='sigmoid', cv='prefit')
xgb_cal.fit(X_val[xgb_features], y_val)
final_models['XGBoost'] = (xgb_cal, xgb_features)

# LightGBM
lgbm_features = [f for f in features_per_model['LightGBM'] if f in X_train.columns]
lgbm_params = {k: v for k, v in best_params['LightGBM'].items()}
lgbm_params.update({'random_state': 42, 'verbose': -1})
lgbm = LGBMClassifier(**lgbm_params)
lgbm.fit(X_train[lgbm_features], y_train)
lgbm_cal = CalibratedClassifierCV(lgbm, method='sigmoid', cv='prefit')
lgbm_cal.fit(X_val[lgbm_features], y_val)
final_models['LightGBM'] = (lgbm_cal, lgbm_features)

# CatBoost
cat_features = [f for f in features_per_model['CatBoost'] if f in X_train.columns]
cat_params = {k: v for k, v in best_params['CatBoost'].items()}
cat_params.update({'random_state': 42, 'verbose': 0})
cat = CatBoostClassifier(**cat_params)
cat.fit(X_train[cat_features], y_train)
cat_cal = CalibratedClassifierCV(cat, method='sigmoid', cv='prefit')
cat_cal.fit(X_val[cat_features], y_val)
final_models['CatBoost'] = (cat_cal, cat_features)

# Logistic Regression
lr_features = [f for f in features_per_model['LogisticReg'] if f in X_train.columns]
scaler_lr = StandardScaler()
X_train_lr = scaler_lr.fit_transform(X_train[lr_features])
X_val_lr = scaler_lr.transform(X_val[lr_features])
lr_params = {k: v for k, v in best_params['LogisticReg'].items()}
lr_params.update({'solver': 'saga', 'max_iter': 1000, 'random_state': 42})
lr = LogisticRegression(**lr_params)
lr.fit(X_train_lr, y_train)
lr_cal = CalibratedClassifierCV(lr, method='sigmoid', cv='prefit')
lr_cal.fit(X_val_lr, y_val)
final_models['LogisticReg'] = (lr_cal, lr_features, scaler_lr)

# Get predictions
test_preds = {}
for name, model_data in final_models.items():
    if name == 'LogisticReg':
        model, features, sc = model_data
        X_t = sc.transform(X_test[features])
    else:
        model, features = model_data
        X_t = X_test[features]
    test_preds[name] = model.predict_proba(X_t)[:, 1]

# Stacking
val_preds_stack = {}
for name, model_data in final_models.items():
    if name == 'LogisticReg':
        model, features, sc = model_data
        X_v = sc.transform(X_val[features])
    else:
        model, features = model_data
        X_v = X_val[features]
    val_preds_stack[name] = model.predict_proba(X_v)[:, 1]

X_stack_val = np.column_stack([val_preds_stack[n] for n in ['XGBoost', 'LightGBM', 'CatBoost', 'LogisticReg']])
X_stack_test = np.column_stack([test_preds[n] for n in ['XGBoost', 'LightGBM', 'CatBoost', 'LogisticReg']])

meta_ridge = RidgeClassifierCV(alphas=[0.01, 0.1, 1.0, 10.0])
meta_ridge.fit(X_stack_val, y_val)
stack_proba = 1 / (1 + np.exp(-meta_ridge.decision_function(X_stack_test)))

# Get test data
test_df = df_with_odds.iloc[test_idx].copy().reset_index(drop=True)
test_odds = test_df['avg_away_open'].values
test_actual = y_test
test_dates = pd.to_datetime(test_df['date']).values

print(f"Test matches: {len(test_df)}")

# Betting functions
def kelly_fraction(prob, odds):
    """Calculate Kelly fraction with edge adjustment."""
    b = odds - 1
    q = 1 - prob
    kelly = (prob * b - q) / b
    return max(0, kelly)

def simulate_betting(proba, actual, odds, dates, strategy='flat',
                     bet_size=0.02, kelly_frac=0.25, max_bet=0.05,
                     threshold=0.45, initial_bankroll=1000):
    """
    Simulate betting with different strategies.

    Strategies:
    - 'flat': Fixed percentage of initial bankroll
    - 'flat_current': Fixed percentage of current bankroll
    - 'kelly': Kelly criterion with fraction
    """
    bankroll = initial_bankroll
    peak = bankroll
    max_dd = 0
    history = []

    for i in range(len(proba)):
        if proba[i] < threshold:
            continue

        # Determine bet size
        if strategy == 'flat':
            bet = initial_bankroll * bet_size
        elif strategy == 'flat_current':
            bet = bankroll * bet_size
        elif strategy == 'kelly':
            k = kelly_fraction(proba[i], odds[i])
            bet = bankroll * min(k * kelly_frac, max_bet)
        else:
            bet = initial_bankroll * bet_size

        if bet <= 0 or bet > bankroll:
            continue

        # Resolve bet
        if actual[i] == 1:
            profit = bet * (odds[i] - 1)
        else:
            profit = -bet

        bankroll += profit

        # Track drawdown
        peak = max(peak, bankroll)
        dd = (peak - bankroll) / peak * 100
        max_dd = max(max_dd, dd)

        history.append({
            'bankroll': bankroll,
            'bet': bet,
            'profit': profit,
            'prob': proba[i],
            'odds': odds[i],
            'won': actual[i] == 1
        })

    if len(history) == 0:
        return None

    wins = sum(1 for h in history if h['won'])

    return {
        'final_bankroll': bankroll,
        'profit': bankroll - initial_bankroll,
        'roi': (bankroll - initial_bankroll) / initial_bankroll * 100,
        'bets': len(history),
        'wins': wins,
        'win_rate': wins / len(history) * 100,
        'max_drawdown': max_dd,
        'avg_bet': np.mean([h['bet'] for h in history]),
        'history': history
    }

# Test different strategies
print("\n" + "=" * 70)
print("BANKROLL SIMULATIONS")
print("=" * 70)

initial = 1000
all_results = []

# Test multiple thresholds and strategies
for threshold in [0.45, 0.50, 0.55]:
    print(f"\n{'='*70}")
    print(f"THRESHOLD: {threshold}")
    print(f"{'='*70}")

    strategies = [
        ('Flat 1% (of initial)', 'flat', 0.01, None, None),
        ('Flat 2% (of initial)', 'flat', 0.02, None, None),
        ('Flat 3% (of initial)', 'flat', 0.03, None, None),
        ('Flat 2% (of current)', 'flat_current', 0.02, None, None),
        ('Quarter Kelly (max 5%)', 'kelly', None, 0.25, 0.05),
        ('1/8 Kelly (max 3%)', 'kelly', None, 0.125, 0.03),
        ('1/16 Kelly (max 2%)', 'kelly', None, 0.0625, 0.02),
    ]

    print(f"\n{'Strategy':<25} {'Bets':>5} {'WinRate':>8} {'ROI':>10} {'MaxDD':>8} {'Final':>10}")
    print("-" * 70)

    for name, strat, bet_size, kelly_frac, max_bet in strategies:
        result = simulate_betting(
            stack_proba, test_actual, test_odds, test_dates,
            strategy=strat, bet_size=bet_size, kelly_frac=kelly_frac, max_bet=max_bet,
            threshold=threshold, initial_bankroll=initial
        )

        if result is None:
            continue

        print(f"{name:<25} {result['bets']:>5} {result['win_rate']:>7.1f}% "
              f"{result['roi']:>+9.1f}% {result['max_drawdown']:>7.1f}% ${result['final_bankroll']:>9,.0f}")

        all_results.append({
            'threshold': threshold,
            'strategy': name,
            **{k: v for k, v in result.items() if k != 'history'}
        })

# Find best strategies
print("\n" + "=" * 70)
print("BEST STRATEGIES BY CATEGORY")
print("=" * 70)

results_df = pd.DataFrame(all_results)

# Best ROI
best_roi = results_df.sort_values('roi', ascending=False).iloc[0]
print(f"\n🏆 HIGHEST ROI: {best_roi['strategy']} @ threshold {best_roi['threshold']}")
print(f"   ROI: {best_roi['roi']:+.1f}% | {best_roi['bets']} bets | Max DD: {best_roi['max_drawdown']:.1f}%")

# Lowest drawdown with positive ROI
positive_roi = results_df[results_df['roi'] > 0]
if len(positive_roi) > 0:
    safest = positive_roi.sort_values('max_drawdown').iloc[0]
    print(f"\n🛡️ SAFEST (positive ROI): {safest['strategy']} @ threshold {safest['threshold']}")
    print(f"   ROI: {safest['roi']:+.1f}% | {safest['bets']} bets | Max DD: {safest['max_drawdown']:.1f}%")

# Best risk-adjusted (Sharpe-like)
results_df['risk_adj'] = results_df['roi'] / (results_df['max_drawdown'] + 1)
best_risk_adj = results_df.sort_values('risk_adj', ascending=False).iloc[0]
print(f"\n⚖️ BEST RISK-ADJUSTED: {best_risk_adj['strategy']} @ threshold {best_risk_adj['threshold']}")
print(f"   ROI: {best_risk_adj['roi']:+.1f}% | {best_risk_adj['bets']} bets | Max DD: {best_risk_adj['max_drawdown']:.1f}%")

# Save results
output = {
    'test_matches': len(test_df),
    'initial_bankroll': initial,
    'all_results': all_results,
    'recommendations': {
        'highest_roi': {
            'strategy': best_roi['strategy'],
            'threshold': float(best_roi['threshold']),
            'roi': float(best_roi['roi']),
            'max_drawdown': float(best_roi['max_drawdown'])
        },
        'best_risk_adjusted': {
            'strategy': best_risk_adj['strategy'],
            'threshold': float(best_risk_adj['threshold']),
            'roi': float(best_risk_adj['roi']),
            'max_drawdown': float(best_risk_adj['max_drawdown'])
        }
    }
}

with open('/home/kamil/projects/bettip/experiments/outputs/kelly_bankroll_v2.json', 'w') as f:
    json.dump(output, f, indent=2)

print("\n" + "=" * 70)
print("✅ Results saved to experiments/outputs/kelly_bankroll_v2.json")
print("=" * 70)
