# Niche Betting Model Pipeline

This document describes the standardized pipeline for training and optimizing niche betting models (corners, cards, etc.).

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│ STEP 1: DATA PREPARATION                                        │
│ - Load all available features (~253)                            │
│ - Merge with target data (corners, cards, etc.)                │
│ - Temporal train/val/test split (60/20/20)                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 2: BORUTA FEATURE SELECTION                                │
│ - Identify statistically relevant features                      │
│ - Reduce from ~253 to ~30-50 features                          │
│ - Saves computation in later steps                              │
│ - Uses Random Forest with shadow features                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 3: MODEL ARCHITECTURE COMPARISON                           │
│ Individual models:                                              │
│ - XGBoost                                                       │
│ - LightGBM                                                      │
│ - CatBoost                                                      │
│ - RandomForest                                                  │
│                                                                 │
│ Ensemble models:                                                │
│ - VotingClassifier (simple average)                            │
│ - StackingClassifier + LogisticRegression meta                 │
│ - StackingClassifier + XGBoost meta                            │
│                                                                 │
│ Metrics: Brier score, AUC, Accuracy, ROI                       │
│ Select best architecture for this bet type                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 4: HYPERPARAMETER TUNING                                   │
│ - Only tune the winning model architecture                      │
│ - Use Optuna or GridSearchCV                                    │
│ - Optimize for log_loss or brier_score                         │
│ - TimeSeriesSplit cross-validation                             │
│                                                                 │
│ Key hyperparameters per model:                                  │
│ - XGBoost: max_depth, learning_rate, n_estimators, reg_lambda  │
│ - LightGBM: num_leaves, learning_rate, min_child_samples       │
│ - CatBoost: depth, l2_leaf_reg, iterations                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 5: SHAP FEATURE VALIDATION                                 │
│ - Calculate SHAP values from tuned model                        │
│ - Validate feature importance matches Boruta selection          │
│ - Remove features with near-zero SHAP importance               │
│ - Document top predictive features                              │
│                                                                 │
│ Why SHAP instead of 2nd Boruta:                                │
│ - Uses actual tuned model weights                               │
│ - Model-specific (XGB SHAP ≠ RF SHAP)                          │
│ - More interpretable results                                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 6: PROBABILITY CALIBRATION                                 │
│ Methods:                                                        │
│ - Platt scaling (sigmoid) - good for most cases                │
│ - Isotonic regression - more flexible, needs more data         │
│ - Beta calibration - good for imbalanced classes               │
│                                                                 │
│ Validation:                                                     │
│ - Calibration curves                                            │
│ - Expected Calibration Error (ECE)                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 7: BUSINESS METRICS OPTIMIZATION                           │
│                                                                 │
│ Threshold optimization:                                         │
│ - Test thresholds: 0.50, 0.55, 0.60, 0.65, 0.70, 0.75         │
│ - For both OVER and UNDER directions                           │
│                                                                 │
│ Key metrics:                                                    │
│ - ROI (Return on Investment)                                   │
│ - Precision (win rate)                                         │
│ - Number of bets (volume)                                      │
│ - Bootstrap 95% CI for ROI                                     │
│ - P(profit > 0) from bootstrap                                 │
│                                                                 │
│ Risk management:                                                │
│ - Kelly criterion for bet sizing                               │
│ - Fractional Kelly (0.25-0.5x) for safety                     │
│ - Maximum drawdown analysis                                     │
│                                                                 │
│ CLV validation:                                                 │
│ - Track closing line value                                     │
│ - Positive CLV = sustainable edge                              │
└─────────────────────────────────────────────────────────────────┘
```

## Implementation Details

### Step 1: Data Preparation

```python
# Load main features
features_df = pd.read_csv('data/03-features/features_all_5leagues_with_odds.csv')

# Merge with target-specific data
# For corners: merge with match_stats.parquet
# For cards: merge with events.parquet

# Temporal split (NEVER random shuffle for time series)
train_df = df.iloc[:int(0.6*n)]  # 2019-2022
val_df = df.iloc[int(0.6*n):int(0.8*n)]  # 2023
test_df = df.iloc[int(0.8*n):]  # 2024-2025
```

### Step 2: Boruta Selection

```python
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, max_depth=5, n_jobs=-1)
boruta = BorutaPy(rf, n_estimators='auto', max_iter=100)
boruta.fit(X_train.values, y_train)

selected_features = [f for f, s in zip(feature_cols, boruta.support_) if s]
```

### Step 3: Model Comparison

```python
models = {
    'xgboost': XGBClassifier(...),
    'lightgbm': LGBMClassifier(...),
    'catboost': CatBoostClassifier(...),
    'random_forest': RandomForestClassifier(...),
    'stacking_lr': StackingClassifier(..., final_estimator=LogisticRegression()),
    'stacking_xgb': StackingClassifier(..., final_estimator=XGBClassifier()),
}

# Compare on validation set
for name, model in models.items():
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_val)[:, 1]
    brier = brier_score_loss(y_val, proba)
    roi = calculate_roi(proba, y_val, odds)
```

### Step 4: Hyperparameter Tuning

```python
import optuna

def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 10.0),
    }
    model = XGBClassifier(**params)
    # Use TimeSeriesSplit CV
    cv = TimeSeriesSplit(n_splits=5)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='neg_brier_score')
    return scores.mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
```

### Step 5: SHAP Analysis

```python
import shap

explainer = shap.TreeExplainer(tuned_model)
shap_values = explainer.shap_values(X_test)

# Get feature importance
importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': np.abs(shap_values).mean(axis=0)
}).sort_values('importance', ascending=False)

# Keep features with importance > threshold
final_features = importance[importance['importance'] > 0.001]['feature'].tolist()
```

### Step 6: Calibration

```python
from sklearn.calibration import CalibratedClassifierCV

# Platt scaling (sigmoid)
calibrated = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
calibrated.fit(X_val, y_val)

# Validate calibration
from sklearn.calibration import calibration_curve
prob_true, prob_pred = calibration_curve(y_test, proba, n_bins=10)
```

### Step 7: Business Optimization

```python
def optimize_thresholds(proba, y_test, odds_over, odds_under):
    results = []

    for direction in ['over', 'under']:
        for threshold in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
            if direction == 'over':
                mask = proba >= threshold
                wins = y_test[mask] == 1
                odds = odds_over
            else:
                mask = (1 - proba) >= threshold
                wins = y_test[mask] == 0
                odds = odds_under

            n_bets = mask.sum()
            if n_bets < 20:
                continue

            # Calculate ROI
            profit = (wins * (odds - 1) - (~wins)).sum()
            roi = profit / n_bets * 100

            # Bootstrap confidence interval
            rois = bootstrap_roi(proba, y_test, odds, threshold, direction)

            results.append({
                'direction': direction,
                'threshold': threshold,
                'n_bets': n_bets,
                'roi': roi,
                'ci_low': np.percentile(rois, 2.5),
                'ci_high': np.percentile(rois, 97.5),
                'p_profit': (np.array(rois) > 0).mean(),
            })

    return pd.DataFrame(results)
```

## Output Format

Each pipeline run produces:

1. **Model artifact**: Trained and calibrated model (pickle/joblib)
2. **Feature list**: Selected features with importance scores
3. **Results JSON**:
   ```json
   {
     "bet_type": "corners_over_10_5",
     "model_architecture": "stacking_xgb",
     "features_selected": 24,
     "best_strategy": {
       "direction": "under",
       "threshold": 0.70,
       "roi": 35.7,
       "ci_95": [28.2, 43.1],
       "p_profit": 1.0
     },
     "all_strategies": [...]
   }
   ```

## Best Practices

1. **Never look at test set** during feature selection or tuning
2. **Use temporal splits** - never random shuffle match data
3. **Calibrate separately** from training data
4. **Bootstrap everything** - single-point estimates are unreliable
5. **Track CLV** - closing line value validates real edge
6. **Fractional Kelly** - never use full Kelly (too aggressive)

## Niche Bet Types

| Bet Type | Target | Lines | Key Features |
|----------|--------|-------|--------------|
| Corners | total_corners | 9.5, 10.5, 11.5 | referee patterns, shots, possession |
| Cards | total_yellows | 3.5, 4.5, 5.5 | referee cards avg, team discipline |
| BTTS | btts | yes/no | attack strength, defense weakness |
| Goals | total_goals | 2.5, 3.5 | xG, form, H2H |

## File Locations

- Pipeline script: `experiments/run_niche_optimization_pipeline.py`
- Model outputs: `experiments/outputs/{bet_type}_optimization.json`
- Trained models: `models/{bet_type}_model.joblib`
- Feature lists: `experiments/outputs/{bet_type}_features.json`
