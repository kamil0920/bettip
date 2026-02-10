# Session Summary — Feb 10, 2026 (Session 7: Job Results + Deployment Rebuild + Pipeline Fixes)

## What Was Done This Session

### 1. Analyzed All 7 Optimization Jobs (A–G)

Initial 7 jobs: 4 failed (HF Hub rate limiting from simultaneous launches), retried as C2/D2/E2/F2. Jobs A & B timed out on feature_optimize. All others completed.

| Job | Run ID | Status | Result Summary |
|-----|--------|--------|----------------|
| **A** | 21856436230 | **TIMED OUT** | feature_optimize hit 5h55m at 33/75 trials (~11 min/trial). No results. |
| **B** | 21856438698 | **TIMED OUT** | feature_optimize hit 5h55m at 32/75 trials. No results. |
| **C** | 21856440867→retried | **Partial** | cards_over_35 ✅, shots_over_285 ✅, cards_over_55 ✘ (no models saved) |
| **D** | 21856442651→retried (D2) | **Success** | All 5 niche markets completed. decay=0.005 worked. |
| **E** | 21856468685→retried (E2) | **Success** | home_win +119.9%, over25 +93.2%, corners_o85 +59.6%, shots_o225 +119.1% |
| **F** | 21856523950→retried (F2) | **Success** | home_win auto-RFE completed |
| **G** | 21856567292 | **Success** | fouls_over_225 +135.0% WF. corners_o85/shots_o225 also produced results. |

### 2. Deployment Config Wiped — Rebuilt From Scratch

Multiple aggregate steps from retry runs uploaded empty `sniper_deployment.json` to HF Hub, then orphan cleanup deleted all previously deployed models. **Total loss of deployment.**

**Rebuilt:** Wrote `/tmp/build_deployment.py` to parse all 17 sniper result JSONs across 8 artifact dirs, select best per market, build deployment config. Uploaded 41 model files + 12 model_params + deployment config to HF Hub.

### 3. 12 Markets Deployed (Best Per Market)

| Market | Source Job | Strategy | WF ROI | WF Bets | HO ROI | HO Bets | Confidence |
|--------|-----------|----------|--------|---------|--------|---------|------------|
| **home_win** | E (odds-thresh) | stacking | +119.9% | 135 | **+113.5%** | 48 | ⭐⭐⭐ HIGH |
| **over25** | E (odds-thresh) | lightgbm | +93.2% | 119 | **+100.0%** | 40 | ⭐⭐⭐ HIGH |
| **corners_o85** | E2/G | agreement | +59.6% | 6371 | **+57.0%** | 1207 | ⭐⭐⭐ HIGH |
| **shots_o225** | E2/G | catboost | +119.1% | 307 | **+102.8%** | 159 | ⭐⭐⭐ HIGH |
| **cards_o35** | C | agreement | +81.9% | 2569 | **+66.1%** | 462 | ⭐⭐⭐ HIGH |
| **shots_o285** | C | disagree_bal | +87.5% | 15 | **+87.5%** | 8 | ⭐⭐ MED |
| **corners** | D | lightgbm | +65.4% | 31 | **+55.4%** | 37 | ⭐⭐ MED |
| **fouls** | D | lightgbm | +110.9% | 104 | 150% (1 bet) | 1 | ⭐ LOW |
| **shots** | D2 | stacking | +122.2% | 27 | — | 0 | ⭐ LOW |
| **cards** | D | agreement | +81.8% | 22 | — | 0 | ⭐ LOW |
| **fouls_o225** | G | agreement | +135.0% | 50 | — | 0 | ⭐ LOW |
| **btts** | D2 | stacking | +65.8% | 200 | **-100%** | 3 | ⚠️ BROKEN |

### 4. Pipeline Bug Fixes (2 commits)

**Fix 1 — RiskConfig kwarg** (commit `f7ef3ab`):
- `generate_daily_recommendations.py` line 167: `max_stake_fraction=` → `max_stake_per_bet=`
- Pipeline crashed on first run with `TypeError`

**Fix 2 — Model discovery for line variants** (commit `6d64fcb`):
- `model_loader.py`: Added `MODEL_VARIANTS` (7 entries including two_stage), added dynamic glob scan for `*_over_*_*.joblib`
- Models discovered: 22 → 41 (all line variant models now found)
- 608 tests pass

**Fix 3 — Stacking sigmoid bug (ROOT CAUSE of 95% BTTS)**:
- `generate_daily_recommendations.py`: Ridge meta-learner is trained in probability space (0-1) and outputs are clipped. But deployment applied sigmoid `1/(1+exp(-raw))` as if weights were log-odds. With large Ridge coefficients (catboost=5.79), sigmoid inflates any reasonable probability to 93%+.
- **Fix**: Replaced sigmoid with normalized weighted average (`raw / sum(weights)`). Also added fastai + two_stage models to model_map (were excluded before).
- Before: BTTS prob=0.47 → sigmoid(5.79×0.47)=**0.938** (broken)
- After: BTTS prob=0.47 → weighted_avg=**0.47** (correct)
- Affects ALL stacking markets (btts, home_win, shots), not just BTTS
- 610 tests pass (2 new regression tests added)

### 5. Prediction Run Results (Feb 10 — 5 matches)

7 recommendations generated: 4 BTTS + 3 CORNERS_O8.5

**Issues identified:**
- **BTTS probabilities unrealistic (95%+)** — holdout was 0/3, stacking collapsed to catboost only (weight 5.79, rest ~0). Likely miscalibrated.
- **CORNERS_O8.5 no real odds** — pipeline couldn't find odds from The Odds API. Strong model (HO +57%) but unbettable without manual odds check.
- **Hearts vs Hibernian DEGRADED** — Scottish Premiership data gaps (56-66% zero features).
- **Two-stage models fail** — `predict_proba()` needs `odds` argument. Pre-existing interface mismatch, non-blocking (4/6 base models still work for stacking).
- **Most markets below threshold** — thin Monday slate, models correctly conservative.

---

## Key Answers From Jobs A–G

| Question | Answer |
|----------|--------|
| Does feature optimize beat R144 params? | **UNKNOWN** — Jobs A/B timed out at 33/75 trials (75 trials × 11min = 12.5h >> 5h55m limit) |
| Can cards/btts be rescued via feature tuning? | **UNKNOWN** — Same timeout issue |
| Do failed variants work? (Job C) | **YES** for cards_o35 (+66.1% HO) and shots_o285 (+87.5% HO). cards_o55 still fails. |
| Does decay=0.005 help niche? (Job D) | **YES** — all 5 niche markets improved vs previous. corners +65.4%, cards +81.8% |
| Odds-dependent thresholds? (Job E) | **YES** — best H2H results: home_win +113.5% HO, over25 +100% HO |
| RFECV optimal features? (Job F) | **Inconclusive** — Job E already won for home_win/over25 |
| Feature opt multi-line? (Job G) | **YES for fouls_o225** (+135% WF). corners_o85 and shots_o225 may come from E2 instead. |

### Feature Optimize Timeout Problem — ROOT CAUSE

```
Workflow timeout: 355 min (5h55m)
Time per trial:   ~10-11 min (feature generation + CV evaluation)
75 trials:        75 × 11 = 825 min = 13.75h → IMPOSSIBLE
50 trials:        50 × 11 = 550 min = 9.2h   → IMPOSSIBLE
30 trials:        30 × 11 = 330 min = 5.5h    → RISKY (10 min margin)
25 trials:        25 × 11 = 275 min = 4.6h    → SAFE
25 trials (2 CV): 25 × 7  = 175 min = 2.9h    → COMFORTABLE
```

**Solution:** Use `n_feature_trials=25` with `n_feature_folds=2` for all future feature optimize jobs. Gets ~70% of search coverage at ~3h runtime, leaving ~3h margin for sniper phase.

### Stagger Lesson Reinforced
Original 7 simultaneous `gh workflow run` calls caused 4/7 to fail at "Download data from HF" (rate limiting). Must use `sleep 60` between launches.

---

## What To Do Next — Planned New Sniper Jobs

### Priority 1: Fix BTTS (Broken Holdout + Miscalibrated)

**Problem:** BTTS predicts 95% for all matches. Holdout 0/3 (-100% ROI). Stacking collapsed to catboost-only (weight 5.79, rest ~0). Sigmoid calibration likely over-shooting.

**Job 1a — BTTS with isotonic calibration:**
```
Markets: btts
feature_params_mode: best
calibration_method: isotonic
adversarial_filter: 2,10,0.75
sample_weight_decay: 0.005
n_trials: 150
```

**Job 1b — BTTS without decay (control):**
```
Markets: btts
feature_params_mode: best
calibration_method: isotonic
adversarial_filter: 2,10,0.75
sample_weight_decay: (default/empty = 0.002)
n_trials: 150
```

Compare: which calibration + decay combo produces realistic BTTS probabilities and positive holdout?

### Priority 2: Feature Optimize (Retry with Reduced Trials)

**Job 2a — Feature Opt H2H (was Job A, timed out):**
```
Markets: home_win, over25
feature_params_mode: optimize
n_feature_trials: 25
n_feature_folds: 2
adversarial_filter: 5,15,0.65
sample_weight_decay: 0.005
n_trials: 75
```
Compare vs Job E deployed (home_win +119.9%, over25 +93.2%)

**Job 2b — Feature Opt Weak Markets (was Job B, timed out):**
```
Markets: cards, btts
feature_params_mode: optimize
n_feature_trials: 25
n_feature_folds: 2
adversarial_filter: 2,10,0.75
calibration_method: isotonic  (especially for btts)
n_trials: 150
```

### Priority 3: Expand Profitable Line Variants

cards_over_35 is the standout discovery: HO +66.1% on 462 bets. Test adjacent lines.

**Job 3 — New Line Variants:**
```
Markets: cards_over_55, shots_over_265, fouls_over_265, corners_over_105
feature_params_mode: best
adversarial_filter: 2,10,0.75
n_trials: 150
```
cards_over_55 failed in Job C (no models saved) — retry.

### Priority 4: Strengthen Weak Deployed Markets

fouls (HO 1 bet), shots (no HO), cards (no HO), fouls_o225 (no HO) need more data/better optimization.

**Job 4 — Niche with isotonic + odds-threshold:**
```
Markets: fouls, shots, cards
feature_params_mode: best
calibration_method: isotonic
odds_threshold: true
threshold_alpha: 0.2
adversarial_filter: 2,10,0.75
sample_weight_decay: 0.005
n_trials: 150
```

### Code Fix Needed: Two-Stage Model Interface

`TwoStageModel.predict_proba()` requires `odds` argument. `ModelLoader.predict()` calls `model.predict_proba(X)` without it. Fix: detect TwoStageModel and pass odds from features_df.

This would enable all 6 home_win base models in stacking (currently 4/6). Low priority since home_win already has +113.5% HO with 4 models.

### Code Fix Desired: Corners O8.5 Odds

The Odds API doesn't return corners O8.5 odds. Need to check if API-Football odds endpoint covers this, or add a different odds source. Without odds, the model's best signal (HO +57%, 1207 bets) is unbettable via automation.

---

## Execution Plan

Launch jobs with 60s stagger:
```bash
# Job 1a: BTTS isotonic+decay
gh workflow run ... -f bet_types=btts -f calibration_method=isotonic -f adversarial_filter=2,10,0.75 -f sample_weight_decay=0.005 -f n_trials=150

sleep 60

# Job 1b: BTTS isotonic (default decay)
gh workflow run ... -f bet_types=btts -f calibration_method=isotonic -f adversarial_filter=2,10,0.75 -f n_trials=150

sleep 60

# Job 2a: Feature opt H2H
gh workflow run ... -f bet_types=home_win,over25 -f feature_params_mode=optimize -f n_feature_trials=25 -f n_feature_folds=2 -f adversarial_filter=5,15,0.65 -f sample_weight_decay=0.005 -f n_trials=75

sleep 60

# Job 2b: Feature opt weak
gh workflow run ... -f bet_types=cards,btts -f feature_params_mode=optimize -f n_feature_trials=25 -f n_feature_folds=2 -f adversarial_filter=2,10,0.75 -f calibration_method=isotonic -f n_trials=150

sleep 60

# Job 3: New line variants
gh workflow run ... -f bet_types=cards_over_55,shots_over_265,fouls_over_265,corners_over_105 -f adversarial_filter=2,10,0.75 -f n_trials=150

sleep 60

# Job 4: Niche isotonic + odds-threshold
gh workflow run ... -f bet_types=fouls,shots,cards -f calibration_method=isotonic -f odds_threshold=true -f threshold_alpha=0.2 -f adversarial_filter=2,10,0.75 -f sample_weight_decay=0.005 -f n_trials=150
```

**Estimated times:**
| Job | Feature Opt | Sniper | Total Est |
|-----|------------|--------|-----------|
| 1a (btts isotonic) | — | ~2h | ~2h |
| 1b (btts control) | — | ~2h | ~2h |
| 2a (feat H2H) | ~3h | ~2h | ~5h |
| 2b (feat weak) | ~3h | ~3h | ~6h (tight!) |
| 3 (new lines) | — | ~3h | ~3h |
| 4 (niche isotonic) | — | ~3h | ~3h |

Job 2b is the tightest — btts might push total past timeout. Consider using `feature_params_lock=btts` if cards is the priority, or split into separate runs.

---

## Files Changed This Session

| # | File | Action |
|---|------|--------|
| 1 | `experiments/generate_daily_recommendations.py` | FIX `max_stake_fraction` → `max_stake_per_bet` (commit f7ef3ab) |
| 2 | `src/ml/model_loader.py` | FIX model discovery: added `MODEL_VARIANTS`, dynamic glob scan (commit 6d64fcb) |
| 3 | `config/sniper_deployment.json` (HF Hub) | REBUILT from scratch — 12 markets, 41 models |
| 4 | `/tmp/build_deployment.py` | CREATE — deployment rebuild script |
| 5 | `experiments/generate_daily_recommendations.py` | FIX stacking sigmoid→weighted avg, add fastai/two_stage to model_map |
| 6 | `tests/unit/test_daily_recommendations_strategies.py` | ADD 2 regression tests for sigmoid fix + fastai inclusion |

---

# Session Summary — Feb 9, 2026 (Session 5: Temporal Leakage Cleanup + Clean Deployment)

## What Was Done This Session

### Analyzed Post-Fix Optimization Runs (R137, R138, R139)
Downloaded and analyzed artifacts from all 3 post-fix optimization jobs:

| Run | Job | Markets | Key Result |
|-----|-----|---------|------------|
| **R137** (21827497725) | F: Niche + all fixes | corners,btts,cards,fouls,shots | All 5 completed, adversarial filter active |
| **R138** (21827499191) | G: H2H + all fixes | home_win,over25,under25 | All 3 completed |
| **R139** (21827500715) | H: Aggressive decay | home_win,over25,fouls | 2/3 completed (fouls timed out) |

**Key findings:**
- Adversarial feature filtering working — removed 5-15 leaky features per market
- Stacking weights all non-negative (fix confirmed)
- H2H markets (home_win, over25) ROI collapsed vs R90 leaked models, confirming temporal leakage
- R139 decay=0.005 improved over25 vs R138 (+45.5% vs +17.2%)
- under25 has no real edge post-fix (WF +0.7%)

### Temporal Leakage Cleanup — Replaced ALL Deployed Models
**Decision:** Remove all R90-era models that benefited from temporal leakage. Accept lower but honest ROI numbers.

Created `scripts/update_deployment_config.py` to programmatically update deployment config from sniper result JSONs (includes JSON repair for truncated artifact files).

### New Deployed Models (Feb 9 — Clean Post-Fix)

| Market | Source | Model | WF ROI | Models | Status |
|--------|--------|-------|--------|--------|--------|
| home_win | R139 | catboost | +13.7% | 1 | ENABLED |
| over25 | R139 | lightgbm | +45.5% | 1 | ENABLED |
| corners | R137 | stacking | +48.7% | 4 | ENABLED |
| btts | R137 | lightgbm | +63.8% | 1 | ENABLED |
| cards | R137 | disagree_balanced_filtered | +62.9% | 4 | ENABLED |
| fouls | R137 | catboost | +150.0% | 1 | ENABLED |
| shots | R137 | disagree_aggressive_filtered | +130.8% | 4 | ENABLED |
| under25 | — | — | +0.7% | — | DISABLED |
| away_win | — | — | — | — | DISABLED |

### Uploaded to HF Hub
- 16 model `.joblib` files (R137 niche + R139 H2H)
- Updated `config/sniper_deployment.json`

### Implemented Pipeline Enhancements (5-item plan)
1. **Kelly criterion** wired into `generate_daily_recommendations.py` (was existing code, now integrated)
2. **Vig removal** extended to all markets (over25/under25/btts 2-way, niche markets)
3. **Monte Carlo stress test** script (`scripts/monte_carlo_stress_test.py`)
4. **Poisson GLM feature engineer** (`src/features/engineers/ratings.py`)
5. **Bayesian shrinkage form engineer** (`src/features/engineers/form.py`)

---

## What To Do Next

### Immediate
1. **Monitor live predictions** — clean models should produce honest (lower) probabilities
2. **Re-run multi-line variants** (cards_over_35, corners_over_85) with post-fix code
3. **Fix SHAP string bug** for niche markets (partially addressed in Session 4)

### Future Experiments
- **Feature optimize** for under25 — see if dedicated tuning can recover edge
- **Dedicated home_win/away_win investigation** — why do H2H models collapse post-fix?
- **MC stress test analysis** — run on historical settled bets to validate Kelly sizing
- **Bayesian shrinkage + Poisson GLM** — require sniper re-run to measure impact

---

## Files Created/Modified This Session

| # | File | Action |
|---|------|--------|
| 1 | `scripts/update_deployment_config.py` | CREATE — JSON repair + config update from sniper results |
| 2 | `config/sniper_deployment.json` | UPDATE — all markets replaced with clean post-fix models |
| 3 | `experiments/generate_daily_recommendations.py` | MODIFY — Kelly sizing + extended vig removal |
| 4 | `src/odds/odds_features.py` | ADD `remove_vig_2way()` utility |
| 5 | `src/features/engineers/ratings.py` | ADD `PoissonGLMFeatureEngineer` class |
| 6 | `src/features/engineers/form.py` | ADD `BayesianFormFeatureEngineer` class |
| 7 | `src/features/registry.py` | REGISTER new engineers |
| 8 | `scripts/monte_carlo_stress_test.py` | CREATE — MC simulation script |
| 9 | `tests/unit/test_kelly_vig.py` | CREATE — tests for Kelly + vig removal |
| 10 | `tests/unit/test_bayesian_form.py` | CREATE — tests for Bayesian shrinkage |

---

# Session Summary — Feb 9, 2026 (Session 4: Model Quality Fixes + New Jobs)

## What Was Done This Session

### Analyzed Pre-Fix Runs (21817145922 + 21823653021)
- **Run 21817145922** (niche: btts, corners, fouls, shots, cards): All sniper jobs succeeded, aggregate step failed with `KeyError: 'model'` in `generate_deployment_config.py:340`
- **Run 21823653021** (home_win): Same aggregate failure
- Confirmed adversarial AUC ~1.0 and extreme/negative stacking weights in both runs (pre-fix behavior)
- Results: fouls +125.8% ROI, shots +109.2%, btts +100%, cards +75.5%, home_win +13.4%

### Implemented Model Quality Improvement Plan (commit 5db780a)

**P0 Bug Fixes:**
1. **Referee feature bug** (`src/features/engineers/external.py`): Added `pd.notna()` check before `isinstance(str)` — fixes silent Series ambiguity crash
2. **SHAP string parsing** (`src/utils/data_io.py`): Centralized bracketed scientific notation cleaning (e.g. `'[5.07E-1]'`) in `load_features()` — fixes SHAP for shots/corners
3. **Deployment config KeyError** (`scripts/generate_deployment_config.py`): Changed direct `cfg['model']` to `.get()` with defaults — fixes aggregate step failure
4. **Min odds raised to 1.5** across all 9 markets in `config/sniper_deployment.json` — eliminates toxic 1.3-1.5 odds bracket (2.3% hit rate on 43 historical bets)

**P1 Improvements:**
5. **Adversarial feature filtering** (`experiments/run_sniper_optimization.py`): New `_adversarial_filter()` function — pre-screens and removes temporally leaky features before training (iterative LGB-based, max 2 passes, AUC>0.75 trigger, 5-feature safety floor)
6. **Non-negative stacking weights** (`experiments/run_sniper_optimization.py`): Replaced `RidgeClassifierCV` with `Ridge(positive=True)` — prevents extreme/inverted meta-learner coefficients (e.g. LGB=-6.16, CB=+10.03)
7. **Calibration validation** (new `src/ml/calibration_validator.py`): ECE check after calibration with 0.10 threshold — logs warning and tries alternatives if calibration is poor
8. **Workflow inputs** (`.github/workflows/sniper-optimization.yaml`): Added `adversarial_filter` (bool, default true) and `sample_weight_decay` (string) inputs

**Tests:**
9. Created `tests/unit/test_adversarial_filter.py` with 11 tests covering all changes — all 546 tests pass

### Uploaded Updated Config to HF Hub
- `config/sniper_deployment.json` with min_odds=1.5 uploaded — live in production immediately

### Launched 3 New Optimization Jobs (with all fixes)

| Run ID | Job | Markets | Trials | Key Feature |
|--------|-----|---------|--------|-------------|
| **21827497725** | F: Niche + all fixes | fouls,shots,corners,cards,btts | 150 | Adversarial filter + non-neg stacking |
| **21827499191** | G: H2H + all fixes | home_win,over25,under25 | 75 | Adversarial filter + non-neg stacking |
| **21827500715** | H: Aggressive decay | home_win,over25,fouls | 75 | decay=0.005 (4.5-month half-life) |

---

## What To Watch In New Results
- **Adversarial AUC** should drop below 1.0 (leaky features removed)
- **Stacking weights** should all be non-negative
- **`calibration_validation`** field should show ECE < 0.10 for well-calibrated markets
- **`adversarial_validation.filter`** field shows which features were removed and why
- Compare WF ROI / holdout ROI against pre-fix runs to measure improvement

## What To Do Next

### Immediate: Analyze Jobs F/G/H When Complete (~2-4h)
1. Download artifacts, compare vs pre-fix runs and currently deployed models
2. Check adversarial filter diagnostics — which features were removed?
3. Check stacking weights — are they all non-negative?
4. Compare Job H (decay=0.005) vs Job G (default decay) for home_win/over25
5. Deploy any markets that improved

### Still Pending From Session 3
- **Job 1** (21799666956) — niche with fixed data, check results
- **Job 2** (21799667493) — multi-line variants test
- **Job 3** (21799667864) — shots feature optimize
- **R115** (21797059071) — under25 feature optimize

### Future Experiments
- **Auto-RFE on shots** — RFECV to prune features down to optimal count
- **Calibration comparison** — isotonic vs beta vs sigmoid systematically
- **CatBoost merge** — two-phase approach for shots/corners
- **Seed diversity** — measure result variance across seeds

---

## Files Changed This Session

| # | File | Action |
|---|------|--------|
| 1 | `src/features/engineers/external.py` | FIX referee pd.notna() check |
| 2 | `src/utils/data_io.py` | FIX centralized bracketed string cleaning |
| 3 | `scripts/generate_deployment_config.py` | FIX KeyError on missing 'model' key |
| 4 | `experiments/run_sniper_optimization.py` | ADD adversarial filter, non-neg stacking, calibration validation |
| 5 | `.github/workflows/sniper-optimization.yaml` | ADD adversarial_filter + sample_weight_decay inputs |
| 6 | `src/ml/calibration_validator.py` | CREATE — ECE validation module |
| 7 | `tests/unit/test_adversarial_filter.py` | CREATE — 11 tests for all changes |
| 8 | `config/sniper_deployment.json` | UPDATE min_odds to 1.5 (uploaded to HF Hub) |

---

# Session Summary — Feb 8, 2026 (Session 3: R112 Analysis + Deploy + New Jobs)

## What Was Done This Session

### R112/R113/R114 Analysis
Analyzed all completed optimization runs vs deployed models:
- **shots (R112)**: BETTER — HO +114.3% vs deployed R102 +91.7%, Sharpe 1.275. **DEPLOYED.**
- **btts (R112)**: BETTER — HO +43.9% on 139 bets vs deployed R93 +26.4%. **DEPLOYED.**
- corners (R112): SIMILAR to R104
- cards (R112): SIMILAR to R104
- fouls (R112): WORSE (only 3 HO bets)
- home_win (R113): WORSE than R90
- over25 (R114): WORSE than R90
- away_win (R113): STAY DISABLED (empty holdout)

### Deployed R112 Models to HF Hub
- **shots**: temporal_blend (3 models: lightgbm, xgboost, fastai), threshold 0.65, 98 features, sigmoid calibration
- **btts**: xgboost (single model), threshold 0.60, 98 features, sigmoid calibration
- Updated `config/sniper_deployment.json` on HF Hub

### Triggered 3 New Optimization Jobs (parallel)

| Run ID | Job | Markets | Trials | Purpose |
|--------|-----|---------|--------|---------|
| **21799666956** | Job 1: Niche fixed data | corners,shots,fouls,cards,btts | 150 | Phase B — measure data quality fix impact |
| **21799667493** | Job 2: Multi-line test | cards_over_35,cards_over_55,corners_over_85,corners_over_105 | 50 | First test of line variants |
| **21799667864** | Job 3: Shots feature optimize | shots | 75 feat + 150 sniper | Feature tuning on best niche market |

- R115 (under25 feature optimize, run 21797059071) still in progress

---

## Current Deployed Models (Updated Feb 8) — SUPERSEDED by Session 5 (Feb 9)
*See Session 5 table above for latest clean post-fix deployment.*

---

## What To Do Next

### Immediate: Analyze New Jobs When Complete (~2-4h)
1. **Job 1 results** — compare niche markets vs R112 to measure data quality fix impact
2. **Job 2 results** — check if multi-line variants produce viable models (new profitable markets?)
3. **Job 3 results** — compare shots with optimized features vs R112 shots
4. **R115 results** — under25 feature optimize, check when complete
5. Deploy any markets that improved

### Phase D: Expansion League Data — Upload Wednesday
- Raw data available on second computer → upload via `upload_data.py`
- After upload, full +129% data boost realized for niche markets

### Future Experiments
- **Auto-RFE on shots** — RFECV to prune 98 features down to optimal count
- **Calibration comparison** — isotonic vs beta vs sigmoid on shots/btts
- **CatBoost merge** — two-phase approach for shots/corners
- **Seed diversity** — measure result variance across seeds
- **Higher trials for fouls** — 300 trials to overcome sampling noise

---

# Session Summary — Feb 8, 2026 (Session 2: Data Quality Fix)

## What Was Done This Session

### Niche Market Data Quality Fix (commit c70a1cc)
Feb 7 predictions went 3/10 (30%) — fouls 2/5, corners 0/4, btts 1/1. Investigation revealed two data quality issues causing niche models to train on far less data than available:

**Issue 1 — Column name mismatch:** Bundesliga/Ligue 1 `match_stats.parquet` files use old API-Football names (`home_corner_kicks`, `home_total_shots`), but niche feature engineers read parquet directly and expect `home_corners`, `home_shots`. The rename logic existed in the pipeline but the engineers bypassed it via their own `_load_match_stats()`. This silently dropped ~2,500 rows.

**Issue 2 — Missing expansion leagues:** Niche engineers hardcoded 5 leagues. Expansion leagues (Eredivisie, Portuguese Liga, Turkish Super Lig, Belgian Pro League, Scottish Premiership) have match data but their match_stats were never collected. ~6,000 potential rows missing.

**Impact:** Corners/shots trained on ~6,500 rows instead of potential ~15,000 (+129%).

**Fix (9 files, 5 tests):**
1. `src/data_collection/match_stats_utils.py` (**NEW**) — shared `normalize_match_stats_columns()` utility
2. `src/features/engineers/corners.py` — add normalize after `pd.read_parquet()`, expand to `EUROPEAN_LEAGUES`
3. `src/features/engineers/niche_markets.py` — same fix in `FoulsFeatureEngineer`, `ShotsFeatureEngineer`, `CardsFeatureEngineer`
4. `src/pipelines/feature_eng_pipeline.py` — replaced 12-line inline rename with shared utility
5. `src/features/regeneration.py` — same deduplication
6. `src/data_collection/match_stats_collector.py` — same deduplication
7. `scripts/collect_all_stats.py` — expanded to `EUROPEAN_LEAGUES`, added `--leagues` CLI arg
8. `.github/workflows/collect-match-data.yaml` — added "Collect match statistics" step after match collection
9. `tests/unit/test_bugfixes.py` — 5 new tests for column normalization

**Data pipeline triggered** (run 21798539185) with `league=all` to backfill expansion league match_stats. Immediate benefit: Bundesliga/Ligue 1 data now included (+2,500 rows). Expansion leagues will accumulate match_stats over daily workflow runs.

### Data Pipeline Results (run 21798539185)
- 5,511 new match_stats collected (ekstraklasa 600, liga_mx 1,455, mls 2,225, ligue_1 351, la_liga 82)
- **API limit (500/day) hit during MLS** — expansion leagues (eredivisie, portuguese_liga, etc.) got 0 match_stats because match collector ate most budget first
- Features generated for all 13 leagues — 18,154 rows merged and uploaded to HF Hub
- **Expansion league raw data**: available on second computer, will be uploaded Wednesday

### Referee Cache Bug Fix
Bundesliga/Ligue 1 feature generation "failed" due to crash in `_build_referee_stats_cache` (features themselves saved correctly before crash). Root cause: after `pd.concat` of old + new season match_stats, old seasons have `home_corner_kicks` and new seasons have `home_corners`. After normalization, both become `home_corners` — duplicate column. Then `_safe_sum` calls `pd.to_numeric(df['home_corners'])` which returns a DataFrame instead of Series.

**Fix:**
- `normalize_match_stats_columns()` — detect when rename target already exists, coalesce old column into new and drop old
- `_safe_sum()` — defensive guard: if `df[col]` returns DataFrame (duplicate cols), take `.iloc[:, 0]`
- Added test: `test_coalesces_duplicate_columns_after_concat`

---

# Session Summary — Feb 8, 2026 (Session 1: Multi-Line + Runs)

## What Was Done This Session

### Multi-Line Support for Niche Markets (commit 7d1b076)
Each niche market (cards, corners, shots, fouls) previously supported only one hardcoded over/under line. Bookmakers offer multiple lines per market — training separate models per line captures different risk/reward profiles (e.g., Cards Over 3.5 = higher hit rate/shorter odds vs Over 5.5 = lower hit rate/better odds).

**Design:** Flat bet_type keys with backward compatibility. Existing base markets unchanged.

**12 new line variants added:**

| Market | Base (unchanged) | New variants |
|--------|-----------------|--------------|
| Cards | `cards` (over 4.5) | `cards_over_35`, `cards_over_55`, `cards_over_65` |
| Corners | `corners` (over 9.5) | `corners_over_85`, `corners_over_105`, `corners_over_115` |
| Shots | `shots` (over 24.5) | `shots_over_225`, `shots_over_265`, `shots_over_285` |
| Fouls | `fouls` (over 24.5) | `fouls_over_225`, `fouls_over_265`, `fouls_over_285` |

**Files changed (6):**
1. `src/ml/betting_strategies.py` — Parameterized `NicheMarketStrategy` with `line` arg; added `default_line`, dynamic `name`/`odds_column`/`bet_side` to all 4 subclasses; fixed inconsistencies (corners 10.5→9.5, fouls 26.5→24.5); added registry entries + `NICHE_LINE_LOOKUP` + `BASE_MARKET_MAP`; updated `get_strategy()` factory
2. `experiments/run_sniper_optimization.py` — Added 12 `BET_TYPES` entries + `BASE_MARKET_MAP` for feature params sharing
3. `experiments/generate_daily_recommendations.py` — Added odds column mappings, baselines, and labels for all variants
4. `.github/workflows/sniper-optimization.yaml` — Updated `--no-two-stage` regex, niche detection regex, feature params path mapping (`sed` strips `_over_[0-9]+$`), known_markets lists (model upload + orphan cleanup)
5. `scripts/cleanup_orphan_models.py` — Added 12 variants to `KNOWN_MARKETS`
6. `tests/unit/test_niche_market_lines.py` (new) — 49 tests

**How line variants work:**
- **Opt-in only** — default workflow `bet_types` unchanged (9 original markets). Variants run only when explicitly passed
- **Share features** — `cards_over_35` loads `config/feature_params/cards.yaml` (same base features, different classification target)
- **Share odds column** — all variants of a market share the same odds column (e.g., `cards_over_avg`)
- **Independent models** — each variant trains its own set of models (e.g., `cards_over_35_lightgbm.joblib`)
- **No extra time** — each variant runs same models/trials as the base market

**Also fixed:** Corners default line was 10.5 in strategy class but 9.5 in sniper optimizer. Fouls was 26.5 in strategy but 24.5 in sniper. Both now aligned to sniper values.

### Overnight/Morning Optimization Runs (pre-commit, on old code)

**R109** (run 21789207368) — Feature optimize + sniper for over25, under25, btts:
- Feature optimize completed for all 3 (under25 3h54m, over25 4h2m, btts 4h)
- over25 sniper: completed (4h17m)
- under25 sniper: completed (4h31m)
- btts sniper: **TIMED OUT at 5h** — likely too many trials with optimized features
- Aggregate skipped (btts incomplete)

**R110** (run 21789209091) — fouls + cards with feature_params=optimize:
- Both **TIMED OUT at 5h** — feature optimize for niche markets + sniper in single 5h window is too tight

**R112** (run 21797059433) — Niche markets (corners, btts, shots, cards, fouls):
- All 5 completed successfully: corners 1h13m, shots 1h6m, fouls 1h7m, cards 1h29m, btts 1h47m
- Artifacts available: `sniper-all-results-112`

**R113** (run 21797059881) — home_win + away_win:
- home_win: completed (1h48m)
- away_win: **still running** (~2h in)

**R114** (run 21797060329) — over25 standalone:
- Completed (1h51m)

**R115** (run 21797059071) — under25 with feature_params=optimize:
- Feature optimize **still running** (~2h in)

---

## Previous Runs Summary (Completed)

| Run | Markets | Result |
|-----|---------|--------|
| R112 | corners,shots,fouls,cards,btts | shots+btts deployed, rest similar/worse |
| R113 | home_win,away_win | Both worse, not deployed |
| R114 | over25 | Worse, not deployed |
| R115 | under25 (feature optimize) | Still running |

---

# Session Summary — Feb 7, 2026 (Morning)

## What Was Done This Session

### Phase 1: Enable Referee/Player/Roster Caches for Predictions (commit ee1c631)
Daily predictions were running without referee, player stats, and team roster caches — all ref-dependent features (cards, fouls, corners) used league-average defaults instead of actual referee tendencies.

- **1A**: Added `data/cache/**` to `download_data.py` allow_patterns — caches now downloaded from HF Hub before predictions
- **1B**: Added `cache/**` to `upload_data.py` allow_patterns — caches built during weekly feature engineering now uploaded
- **1C**: Added "Fetch referee assignments from API-Football" step to `prematch-intelligence.yaml` — runs after schedule fetch, before odds fetch. Groups matches by league, 1 API call per league (~5-10 calls total), updates referee field in `today_schedule.json`

**Verification (run 21777445227):**
- Referee step: 15/18 matches updated, 3 API calls (premier_league, bundesliga, serie_a)
- Cache loaded: 69 referees, 11,430 players, 4,546 roster entries (previously all showed "Using defaults" warnings)

### Phase 2: Add Expansion Leagues to Data Collection (commit e4e259b)
5 expansion leagues (belgian_pro_league, eredivisie, portuguese_liga, scottish_premiership, turkish_super_lig) were added to `src/leagues.py` and prematch workflow on Feb 4 but never added to the data collection workflow.

Updated `.github/workflows/collect-match-data.yaml`:
- **CONFIGURED_LEAGUES**: added all 5 expansion leagues
- **Schedule**: spread 13 leagues across Mon-Thu (3-4/day), niche odds moved to Fri
- **Auto-select case**: Mon=premier_league+la_liga+eredivisie, Tue=serie_a+bundesliga+portuguese_liga, Wed=ligue_1+ekstraklasa+belgian_pro_league+scottish_premiership, Thu=turkish_super_lig+liga_mx+mls, Fri=niche_odds
- **Manual dispatch dropdown**: added all 7 missing leagues (5 expansion + liga_mx + mls)
- **Feature merge + niche odds rotation**: updated to include all 13 leagues

### Phase 3: Expansion Leagues — Blocked on Raw Data
- Expansion leagues have feature data on HF Hub (`features_all_5leagues_with_odds.parquet`, 19K rows) but **raw parquet files** (`data/01-raw/{league}/2025/matches.parquet`) are missing from HF Hub
- Raw data was collected on the second computer but never uploaded
- Without raw data, match scheduler can't find today's matches for these leagues
- **Action needed**: push raw data from second computer via `upload_data.py`

---

## What To Do Next

### 1. Upload Expansion League Raw Data — PRIORITY
- From second computer, run `uv run python entrypoints/upload_data.py`
- This will push `data/01-raw/{league}/2025/matches.parquet` for all expansion leagues
- After upload, prematch workflow will automatically include these leagues in daily schedule + referee fetch

### 2. Verify Expansion Leagues Working
- After raw data upload, trigger `prematch-intelligence.yaml` manually
- Check logs for expansion leagues appearing in schedule (no more "No parquet file" warnings)
- Verify referee fetch covers expansion leagues too

---

# Session Summary — Feb 6, 2026 (Evening)

## What Was Done This Session

### Phase 1: Code Fixes (commit 35d2fc0)
- **1A**: Fixed odds-adjusted threshold bug — was only in Optuna, now in grid search + holdout + walk-forward
- **1B**: Added beta calibration to Optuna search space (sigmoid+BetaCalibrator post-hoc)
- **1C**: Added sample weights to walk-forward training (was missing, mismatch with Optuna)
- **1D**: Widened over25 threshold search to [0.65..0.85]

### Phase 2: Optimization Runs
- **R90**: home_win, away_win, over25, under25 with odds-threshold (alpha=0.2)
- **R93**: fouls, shots, btts, cards, away_win with R89 best feature params + odds-threshold

### Phase 3: Deployment (Late Afternoon)
- **8 markets enabled** (was 3 from R86):
  - home_win (R90): temporal_blend, HO +128.3%, Sharpe 1.78
  - over25 (R90): average, HO +104.5%, Sharpe 1.03
  - under25 (R90): disagree_balanced_filtered, HO +64.5%
  - fouls (R93): temporal_blend, WF +128.5% (181 bets)
  - shots (R93): catboost + **beta calibration**, HO +60.7%
  - cards (R93): lightgbm + isotonic, WF +68.5% (214 bets)
  - btts (R93): average, HO +26.4% (275 bets, high volume)
  - corners (R86): LightGBM, HO +18.75% (awaiting upgrade)
- away_win: DISABLED (empty/tiny holdout in both R90 and R93)
- All models + deployment config uploaded to HF Hub

### Phase 4: Workflow Fixes
- Extended under25 threshold floor to 0.60 (commit 615b416)
- Bumped feature_optimize timeout to 300min, default trials to 50 (commit 615b416)
- Fixed YAML syntax error in HF fallback for locked markets (commit cebf47e)

### Phase 5: R94/R95 Analysis & Corners Deployment (Evening)
- **R94** (run 21751283638) = Corners 50-trial feature optimize — **COMPLETED**
  - corners: disagree_conservative_filtered, CV +56.4%, WF +46.3%, HO +11.9%
  - Massive WF improvement over R86 (+46.3% vs +18.75%)
  - **Deployed to HF Hub**: 4 models (lightgbm, catboost, xgboost, fastai) + feature params + updated deployment config
- **R95** (run 21757118076) = home_win/over25/under25 with R89 best feature params — **COMPLETED**
  - home_win: temporal_blend, CV +114.3%, HO +89.4% — **worse than R90 (+128.3%)**
  - over25: xgboost, CV +94.4%, HO +65.5% — **worse than R90 (+104.5%)**
  - under25: lightgbm, CV +68.0%, HO +23.1% — **similar to R90, weak HO**
  - Conclusion: R89 "best" feature params did NOT help these markets vs R90 defaults
  - under25 correctly used new 0.60 threshold floor
- **Seed validation run launched** (run 21765572745): seed=123, all 8 markets, feature_params=best, odds_threshold=true

### Phase 6: Prematch Intelligence Bug Fixes (Evening)
Analyzed today's prematch workflow runs — only 2 Telegram messages were sent (expected more). Found and fixed 3 issues:

- **6A**: Fixed lineup fetch crash (commit 0730a8e) — `pre_kickoff_repredict.py` called `client.get()` which doesn't exist on `FootballAPIClient`. Changed to `client._make_request('/fixtures/lineups', ...)`. All 3 lineup fetches today failed with `AttributeError`.
- **6B**: Fixed misleading log message (commit bde1ffc) — `match_scheduler.py` logged "Enabled markets from strategies.yaml" even when markets came from `sniper_deployment.json`. Changed to generic "Enabled markets".
- **6C**: Set `THE_ODDS_API_KEY` as GitHub Actions secret — was missing, so pre-match odds from The Odds API were never fetched. Now set in repo secrets.

**Why only 2 Telegram messages today:**
1. Morning run (7 AM) sent 1 Telegram — but only had 3 markets (shots, fouls, corners) because Phase 3 deployment happened later that afternoon
2. Lineup collection ran 6 times (hourly at :30) — only 1 run found matches in window, but lineup fetch crashed due to bug 6A → no re-prediction → no second Telegram
3. `generate_daily_recommendations.py` also sent 1 Telegram with sniper recommendations

---

## Deployed Models (see Session 3 table above for latest)

---

## What Was Done Next (Feb 8 follow-up)
- Seed validation (R96), prematch fixes, R89 analysis — handled in Feb 8 Session 1
- Under25 feature optimize run (R115) launched
- Multi-line niche market support implemented (see Feb 8 Session 1 above)
- Niche market data quality fix: column mismatch + expansion league collection (see Feb 8 Session 2 above)

---

## Key Files
- Deployment config: `config/sniper_deployment.json` (also on HF Hub)
- Feature params: `config/feature_params/*.yaml` (R89 optimized, also on HF)
- Sniper script: `experiments/run_sniper_optimization.py`
- CI workflow: `.github/workflows/sniper-optimization.yaml`
- Prematch workflow: `.github/workflows/prematch-intelligence.yaml`
- Pre-kickoff script: `scripts/pre_kickoff_repredict.py` (lineup fetch fixed)
- R94 artifacts: `data/artifacts/sniper-all-results-94/`
- R95 artifacts: `data/artifacts/sniper-all-results-95/`

## Environment Notes
- `gh` CLI installed at `~/.local/bin/gh` (no sudo needed)
- Auth via `GH_TOKEN` env var from `.env`
- GitHub repo: `kamil0920/bettip`
- HF repo: `czlowiekZplanety/bettip-data`
- GitHub secrets set: `THE_ODDS_API_KEY`, `HF_TOKEN`, `API_FOOTBALL_KEY`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`
