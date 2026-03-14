# Lessons Learned — Anti-Overfit Retraining (S51, Mar 7-9, 2026)

## Calibration Overfit Fix (Mar 15, 2026)

### Root Cause
Sigmoid calibration (CalibratedClassifierCV with cv="prefit") was fitted on the SAME data the base model trained on. This produced degenerate steep sigmoids with |a| = 25-40 (normal is 1-5), mapping raw 0.57 → calibrated 0.80 and raw 0.60 → calibrated 0.91. Result: corners model 0/4 live, 8W-13L (38.1%) across Mar 14 picks.

### Fix Pattern
ALL calibration paths that use cv="prefit" now split training data 80/20: train base model on first 80%, calibrate on held-out last 20%. Constants: `_CAL_FRACTION = 0.2`, `_MIN_CAL_SAMPLES = 50`.

### Rule
**NEVER calibrate on the same data the model trained on** when using cv="prefit". The only correct pattern for prefit is: model.fit(X_train), then CalibratedClassifierCV(model, cv="prefit").fit(X_cal) where X_cal was NOT seen during training. Non-prefit cv (e.g., cv=5) already handles this via internal cross-validation.

### Affected Locations (10 in run_sniper_optimization.py)
1. WF default path — Optuna trial fitting
2. WF default post-hoc (beta/temperature/VA)
3. WF temporal_blend base models
4. WF temporal_blend post-hoc
5. WF MAPIE conformal
6. WF tb2 full-history
7. WF tb2 recent-only
8. HO evaluation base models + post-hoc
9. HO tb2 full-history
10. HO tb2 recent-only
Plus model saving (already had 80/20 split but was using WRONG portion)

### Production Safety Nets Added
- `model_loader.py`: |a| > 10 steep sigmoid detection → falls back to uncalibrated
- `generate_daily_recommendations.py`: niche market probs capped at 0.95

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

### 3. Conformal Calibration — PARTIALLY FIXED (Mar 9)

**Bug 1 (FIXED)**: `TwoStageModel` used `_is_fitted` (no trailing underscore) instead of sklearn convention `is_fitted_`. `check_is_fitted()` always failed → "not fitted" error. Fixed by adding `self.is_fitted_ = True` in `fit()` (commit c63aa45).

**Bug 2 (OPEN)**: Two-stage models lack `classes_` attribute required by MAPIE's `MapieClassifier`. Error: `"does not contain 'classes_' attribute"`. CatBoost/XGB/LGB base models work fine — only two-stage wrappers affected.

**Current status**: Conformal calibration works for catboost/lightgbm/xgboost models. Two-stage models still skip conformal. Non-blocking since deployed models use temporal_blend/catboost strategies (not two-stage).

**Remaining fix**: Add `self.classes_ = np.array([0, 1])` after `fit()` in `TwoStageModel`.

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

### 9. over25 ECE Near Limit — FIXED (Mar 9)

~~over25 calibration ECE=0.0926, close to 0.10 deployment limit.~~ Wave 1 retry with `max_rfe_features=30,mrmr=20` reduced to **ECE=0.035**. No longer a concern.

### 10. Feature Bloat Without max_rfe_features+mrmr (Mar 9)

Wave 1 initial run forgot `max_rfe_features=30,mrmr=20` flags → RFECV kept 37/80/87 features for home_win/over25/away_win. This is **confirmed overfitting** — adversarial AUC was high.

Wave 1 retry with correct flags → 5/5/6 features. Results dramatically improved:
- **over25**: 87→5 features, ECE 0.0926→0.035, precision 93.8%, HO ROI +84.4%
- **away_win**: 80→6 features, precision 92.6%, HO ROI +64.8%
- **home_win**: 37→5 features, but only 2 HO bets — insufficient for evaluation

**Lesson**: `max_rfe_features=30,mrmr=20` is MANDATORY for H2H markets. Without it, RFECV keeps 80+ features = guaranteed overfit.

### 11. Deployment Config Format Bug — 20/29 Markets Silently Blocked (Mar 9)

`generate_daily_recommendations.py`'s `_check_deployment_gates()` expects `holdout_metrics` as a nested dict:
```python
holdout = market_cfg.get("holdout_metrics", {})
n_bets = holdout.get("n_bets", 0)
```

But 20/29 markets stored holdout data as flat fields (`holdout_roi`, `holdout_n_bets`) → `n_bets` always 0 → market silently skipped.

**Impact**: Most deployed markets were producing ZERO predictions despite being configured. Fixed by converting all to nested `holdout_metrics` dict format.

### 12. Data Freshness Gap (Mar 9)

HF Hub raw data was 9-16 days behind (last data: Feb 21-28 depending on league). Features up to Feb 28.
- `collect-match-data.yaml` workflow only runs Mon-Thu → weekend matches not collected until Monday
- All Wave 1-3 training runs used data missing March 1-8 matches
- Stale fixtures appeared in recommendations because parquet data had NS-status matches that were already played/rescheduled
- **No live API cross-check** in `--local` mode to filter out already-played matches

Triggered data collection pipeline (run 22855844814) with `days_back=21` to catch up.

## What IS Clean

- No explicit data leakage found in any log
- No NaN/inf/degenerate values in model outputs
- No zero-fill or feature mismatch warnings
- Fake zero cards detector found 0 matches (data quality OK)
- Feature counts reduced to 5-24 (anti-overfit effective)
- All deployed ECE values under 0.10
- No "suspicious", "unexpected", or "anomaly" warnings

### 13. Workflow Dispatch Stagger — 5 MINUTES MINIMUM (Mar 10)

**CRITICAL**: When dispatching multiple `gh workflow run` commands, MUST use `sleep 300` (5 minutes) between each dispatch. NOT 2 minutes, NOT 120 seconds.

**Reason**: HF Hub rate-limits parallel downloads. All matrix jobs in a workflow dispatch hit HF Hub simultaneously to download features parquet + model files. Multiple concurrent workflows amplify this to 10-25 simultaneous downloads, causing HTTP 429 errors that fail jobs silently.

**Pattern**:
```bash
gh workflow run sniper-optimization.yaml ... && sleep 300 && gh workflow run sniper-optimization.yaml ...
```

**Never**: `sleep 120`, `sleep 60`, or dispatching without delay between runs.

### 14. ALWAYS Validate Model Picks with Web Research Before Betting (Mar 14, 2026)

**Evidence from Mar 14 live session (9 bets, 4W 5L, -1,194 PLN):**

The 3 bets selected and validated via web research went **3/3 (+170 PLN)**:
- Oviedo U2.5 @ 1.65 → WIN (9/10 home U2.5 confirmed by web stats)
- RM Cards HC Elche (-0.5) @ 1.83 → WIN (Elche highest foul rate in La Liga confirmed)
- RM Corners U10.5 @ 1.60 → WIN (9/10 H2H under 10.5 confirmed)

The 6 bets selected by model/conformal alone (no web validation) went **1/5 (-1,363 PLN)**:
- WHU Cards U2.5 (best conformal, 500 PLN) → LOSS
- RM Fouls U23.5 (2nd best conformal, 350 PLN) → LOSS
- Lorient Cards U3.5 → LOSS
- Celtic Corners O9.5 (98.5% model prob) → LOSS
- Kilmarnock BTTS (research said SKIP, placed anyway) → LOSS

Skip decisions validated by web research also went **3/4 correct**:
- Monaco Corners U8.5 (SKIP: Brest concedes 7.58 corners away) → would have LOST (12 corners)
- WHU Cornershc U15 (SKIP: thin market) → would have LOST (16 corners)
- WHU Away Win (SKIP: City fatigue) → would have LOST (1-1 draw)
- Arouca Away Win (SKIP: odds too short) → would have WON but only +14.5% edge

**Full 64-pick analysis (all recs, not just placed bets):**
- 31W 27L (53.4% win rate) — barely above breakeven
- Corners model: 8W 13L (38.1%) — BROKEN, needs investigation
- Cards model: 14W 9L (60.9%) — decent
- Under 2.5: 3W 1L (75%) — strong
- BTTS: 0W 2L — avoid
- Top 10 model misses ALL had >93% probability — extreme overconfidence

**RULE: Before placing ANY bet, validate with web research:**
1. Search for team form (last 5 matches), H2H stats, injury news
2. Check if real bookmaker odds match model assumptions
3. If web research contradicts model → SKIP the bet
4. If web research confirms model → increase confidence, size appropriately
5. Conformal lower bound alone is NOT sufficient for bet selection

### 15. Corners Model is Broken Live — 0/4 Since Deployment (Mar 14, 2026)

Live results for corners_over_95 model: **0W 4L** including Celtic O9.5 at 98.5% model probability (actual: 5 corners).
Full corners market across all 64 recs: **8W 13L (38.1%)** — worse than random.
Inter O8.5/O9.5/O10.5/O11.5 ALL lost (actual: 5 corners, model said 92-99%).

**Do NOT bet corners markets until model is retrained and validated with live data.**

### 16. Stake Sizing Must Follow Convergence, Not Conformal Alone (Mar 14, 2026)

Mar 14 showed inverted sizing: biggest stakes (500, 350 PLN) on pure-conformal picks that lost,
smallest stakes (100-150 PLN) on web-research-validated picks that won.

**New sizing rule**: Highest stakes only when BOTH model conformal AND web research agree.
Model-only or conformal-only picks get minimum stakes until live track record is established.

## Action Items

### High Priority
- [x] ~~Investigate conformal calibration failure on two-stage models~~ — PARTIAL FIX: `is_fitted_` attribute added (commit c63aa45). CatBoost/LGB/XGB work. Two-stage still needs `classes_` attribute.
- [ ] Fix two-stage model `classes_` attribute for MAPIE conformal calibration
- [ ] Check cornershc over/under prediction consistency (do P(over) + P(under) make sense?)
- [x] ~~Monitor over25 ECE for drift~~ — FIXED: Wave 1 retry reduced ECE from 0.0926 to 0.035
- [x] ~~Remove highest-shift features~~ — DONE: 6 non-stationary features added to EXCLUDE_COLUMNS in run_sniper_optimization.py

### Medium Priority
- [ ] Investigate btts 83% data loss — can we recover more rows with fallback odds?
- [ ] Evaluate if stacking is worth keeping vs single catboost (6/7 markets collapsed)
- [ ] away_win: marginal deployment — watch live performance closely, consider disabling if first 10 bets lose
- [ ] Retrain home_win — Wave 1 retry only got 2 HO bets, needs different approach (more folds? longer history?)
- [ ] Add date guard to recommendations pipeline — filter out matches that have already been played
- [ ] Automate data collection on weekends (currently Mon-Thu only)

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
| home_win | HIGH | 5 features after retry, but only 2 HO bets — insufficient for evaluation. Needs different approach. |
| over25 | LOW-MED | **Improved**: 5 features (was 12), ECE=0.035 (was 0.0926), 93.8% precision, +84.4% HO ROI |
| btts | MEDIUM-HIGH | 17 features, adversarial AUC=0.913 (highest), 83% data filtered, aggressive regularization triggered |
| away_win | MEDIUM | **Improved**: 6 features (was 15), 92.6% precision, +64.8% HO ROI. Still watch live perf. |
| cardshc/cards markets | MEDIUM | 16-24 features, fake odds, precision is primary signal |
| cornershc_over/under pairs | MEDIUM | Potential prediction inconsistency between over/under sides |
