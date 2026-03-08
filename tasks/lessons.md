# Lessons Learned — Anti-Overfit Retraining (S51, Mar 7-8, 2026)

## CI Log Analysis Findings

### 1. Temporal Leakage Is REDUCED But NOT Eliminated

The adversarial filter successfully removes the worst temporal features, but remaining features still carry temporal signal. Evidence:

| Market | Pass 0 AUC | Final Mean AUC | Features Removed | Features Remaining |
|--------|-----------|----------------|------------------|--------------------|
| btts | 0.979 | 0.913 | 13 | 17 |
| away_win | 0.904 | 0.778 | 15 | 15 |
| over25 | 0.886 | 0.770 | 18 | 12 |
| home_win | 0.848 | 0.755 | 18 | 12 |
| cornershc_over_15 | 0.829 | 0.776 | 7 | 23 |
| cornershc_over_05 | 0.813 | 0.753 | 8 | 21 |

**Root cause**: Many features (kurtosis, skewness, damped_trend, hurst, first_diff) are inherently non-stationary by construction. They encode "which time period" even after the most leaky ones are removed.

**Implication**: Expect some backtest-to-live degradation. Markets with fewer features (corners at 5) are safest. Markets with 20+ features and high adversarial AUC (btts at 0.913) carry the most risk.

### 2. Universal Distribution Shift (KS Test) — ALL Folds, ALL Markets

Every single walk-forward fold across all markets shows "Significant distribution shift" warnings. The worst cases:

| Market | Fold | Shifted Features | Total Features | Shift Ratio |
|--------|------|-----------------|----------------|-------------|
| away_win_h1 | 0 | 20 | 20 | 100% |
| away_win | 0 | 13 | 15 | 87% |
| home_win | 0 | 11 | 12 | 92% |
| away_win | 3 | 10 | 15 | 67% |
| over25 | 0 | 8 | 12 | 67% |

**Most frequently shifting features across markets:**
- `fouls_hurst_diff` — KS stats 39-75, appears in 4/5 folds for over25
- `home_corners_damped_trend` — top shifter in 5/5 cornershc markets
- `away_corners_kurtosis` — KS stats 39-52, appears in home_win + away_win
- `away_corners_damped_trend` — KS stat 97 in cornershc_over_25 Fold 1
- `ref_home_win_pct` — KS stat 87 in cornershc_over_15 Fold 4
- `away_cards_damped_trend` — KS stat 72 in away_win_h1 Fold 4
- `home_shots_skewness` — KS stats 32-52, appears in home_win + over25

**Implication**: Higher-order statistical features (kurtosis, skewness, hurst) are inherently non-stationary. They survive adversarial filtering because they're not the WORST leakers, but they still shift significantly between train and test periods.

### 3. Conformal Calibration Failed on ALL Two-Stage Models

All 3 H2H markets (home_win, away_win, over25) show:
```
WARNING: Conformal calibration failed: TwoStageLightGBM instance not fitted
WARNING: Conformal calibration failed: TwoStageCatBoost instance not fitted
```

**Impact**: Deployed two-stage models lack uncertainty bounds. The uncertainty_roi metric for these markets may be unreliable.

**Action needed**: Investigate why TwoStageModel instances are "not fitted" during conformal calibration step. Likely a timing/ordering bug in the pipeline.

### 4. Cornershc Over/Under Pairs Are Near-Clones (SUSPICIOUS)

The adversarial filter produces **byte-for-byte identical** results for paired markets:
- `cornershc_over_05` = `cornershc_under_05`: same 8 features removed, same AUCs (0.813 -> 0.693)
- `cornershc_over_15` = `cornershc_under_15`: same 7 features removed, same AUCs (0.829 -> 0.722)

Selected features are also nearly identical between over/under variants of the same line.

**Risk**: If P(over) + P(under) predictions use the same features with similar weights, they may not sum to a sensible value. The model is essentially making the same prediction for both sides of the line, which is logically inconsistent.

**Investigation needed**: Check if live predictions for over_X + under_X at the same line ever both exceed 0.5 probability simultaneously, which would indicate the model doesn't understand these are complementary events.

### 5. All Cornershc + away_win_h1 Use Fake Odds (Default 2.50)

Six markets report "odds column not found, using default 2.50":
- cornershc_over_05, cornershc_over_15, cornershc_over_25
- cornershc_under_05, cornershc_under_15
- away_win_h1

**Impact**: All ROI numbers for these markets are computed against fixed 2.50 odds. ROI is meaningless — only precision is trustworthy for deployment decisions. This is already noted in our evaluation criteria but worth repeating: never make deployment decisions for these markets based on ROI.

### 6. btts: 83% Data Filtered Out

24,550 rows loaded but only 4,126 remain after filtering missing/invalid odds (20,424 rows removed). This is unique to btts — no other market loses this much data.

**Impact**:
- Training on 17% of data increases temporal concentration
- Higher adversarial AUC (0.913) may be partly an artifact of smaller, more temporally concentrated dataset
- Holdout may not represent full distribution of btts outcomes

### 7. Stacking Weight Collapse Is Systemic

6 of 7 markets analyzed show complete stacking weight collapse to catboost (lgb=0.0, xgb=0.0). Only btts has healthy weights (lgb=0.34, cat=0.96, xgb=0.01).

**Implication**: For most markets, "stacking ensemble" is just rescaled catboost. The `temporal_blend` and `average` strategies compensate by equal-weighting base models, but the stacking meta-learner adds no diversity. The 3-model ensemble may not be worth the compute cost.

### 8. away_win CI Crosses Zero

away_win holdout ROI 95% CI: [-9.2%, +55.8%] — explicitly flagged by pipeline as "not significantly profitable". Only 22 holdout bets (barely above 20 minimum).

### 9. over25 ECE Near Limit

over25 calibration ECE=0.0926, close to 0.10 deployment limit. May drift above threshold in production. Monitor weekly.

## What IS Clean

- No explicit data leakage found in any log
- No NaN/inf/degenerate values in model outputs
- No zero-fill or feature mismatch warnings
- Fake zero cards detector found 0 matches (data quality OK)
- Feature counts reduced to 5-24 (anti-overfit effective)
- All deployed ECE values under 0.10
- No "suspicious", "unexpected", or "anomaly" warnings

## Action Items

### High Priority
- [ ] Investigate conformal calibration failure on two-stage models (code bug)
- [ ] Check cornershc over/under prediction consistency (do P(over) + P(under) make sense?)
- [ ] Monitor over25 ECE for drift (currently 0.0926, limit 0.10)
- [ ] Consider removing highest-shift features (fouls_hurst_diff, home_corners_damped_trend) from future runs

### Medium Priority
- [ ] Investigate btts 83% data loss — can we recover more rows with fallback odds?
- [ ] Evaluate if stacking is worth keeping vs single catboost (6/7 markets collapsed)
- [ ] away_win: marginal deployment — watch live performance closely, consider disabling if first 10 bets lose

### Low Priority
- [ ] Consider adding feature stationarity check (ADF test) to feature engineering pipeline
- [ ] Investigate if kurtosis/skewness features should be replaced with more stable alternatives
- [ ] Study if damped_trend features can be made stationary via differencing

## Per-Market Risk Assessment

| Market | Risk Level | Reasons |
|--------|-----------|---------|
| corners_over_85 | LOW | 5 features, lowest adversarial AUC, ECE=0.009 |
| corners_under_95 | LOW | 5 features, low ECE, strong precision |
| corners_under_105 | LOW | 5 features, low adversarial signal |
| cornershc_over_25 | LOW | 9 features, catboost won directly |
| home_win_h1 | LOW | 14 features, 93.2% precision, 118 HO bets |
| ht_under_05 | LOW-MED | 14 features, 96.4% precision, but only 56 HO bets |
| home_win | MEDIUM | 12 features, odds_threshold helped, but distribution shift in kurtosis/skewness features |
| over25 | MEDIUM | 12 features, ECE=0.0926 near limit |
| btts | MEDIUM-HIGH | 17 features, adversarial AUC=0.913 (highest), 83% data filtered, aggressive regularization triggered |
| away_win | HIGH | 15 features, CI crosses zero, only 22 HO bets, 3/5 folds HIGH SHIFT |
| cardshc/cards markets | MEDIUM | 16-24 features, fake odds, precision is primary signal |
| cornershc_over/under pairs | MEDIUM | Potential prediction inconsistency between over/under sides |
