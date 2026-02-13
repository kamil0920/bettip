# Bettip — Current State & Next Steps (Feb 13, 2026)

## Session 15 — Boosting-Only Baseline + Feature Refresh + Two-Stage Niche (PLANNED)

### Code Changes (pre-launch)

1. **`--no-fastai` CLI flag** — Exclude FastAI from model list for boosting-only experiments
2. **`no_fastai` CI input** — Workflow-level toggle for FastAI exclusion
3. **`force_two_stage_niche` CI input** — Override auto-disable of two-stage models for niche markets
4. **Feature trial defaults bumped** — 30→50 trials, 2→3 folds for feature param optimization

### Rationale

**Key insight from S14 analysis:** FastAI dominates stacking weights (65-95%) in 12/14 deployed markets but contributes to instability (cards_o35 collapse was fastai-only model). Two-stage models have NEVER been tested on niche markets (CI auto-disables them). Feature params were last optimized with only 25 trials/2 folds — now we have 50/3.

### Jobs (5 runs)

| Job | Run | Markets | Key Lever | Purpose |
|-----|-----|---------|-----------|---------|
| 1 | — | fouls, shots, corners, cards, btts | Feature refresh (50t, 3f) | **PRODUCTION**: Better params with 2x budget |
| 2 | — | fouls_o225, shots_o225, corners_o85, cards_o35, shots_o265 | Feature refresh (50t, 3f) | **PRODUCTION**: Line variant params refresh |
| 3 | — | away_win | Deep optimize (200 trials, feat opt, odds-threshold) | **EXPERIMENTAL**: Push paper→production |
| 4 | — | cards, corners, btts, fouls | `--no-fastai` boosting-only | **KNOWLEDGE**: Is FastAI helping or hurting? |
| 5 | — | home_win, shots, fouls_o225 | seed=123 | **VALIDATION**: Seed robustness check |

### Job Details

**Job 1 — Niche base feature refresh**
```
markets: fouls,shots,corners,cards,btts
feature_params_mode: optimize
n_feature_trials: 50, n_feature_folds: 3
n_trials: 150, walkforward: true, shap: true
sample_weights: true, adversarial_filter: 2,10,0.75
```
Hypothesis: 2x trial budget + 3 folds (vs 25 trials/2 folds from S7) finds better feature configs.

**Job 2 — Niche lines feature refresh**
```
markets: fouls_over_225,shots_over_225,corners_over_85,cards_over_35,shots_over_265
feature_params_mode: optimize
n_feature_trials: 50, n_feature_folds: 3
n_trials: 150, walkforward: true, shap: true
sample_weights: true, adversarial_filter: 2,10,0.75
```
Same as Job 1 but for line variant markets. Cards_o35 is the key market to watch (current deployed from S8).

**Job 3 — away_win deep optimization**
```
markets: away_win
feature_params_mode: optimize
n_feature_trials: 75
n_trials: 200, walkforward: true, shap: true
odds_threshold: true, threshold_alpha: 0.2
sample_weights: true, adversarial_filter: 2,10,0.75
```
Hypothesis: away_win showed +120.6% HO on 17 bets in R220. Extra budget + odds-threshold could push to production.

**Job 4 — Boosting-only baseline (NO FASTAI)**
```
markets: cards,corners,btts,fouls
feature_params_mode: best
no_fastai: true
n_trials: 150, walkforward: true, shap: true
sample_weights: true, adversarial_filter: 2,10,0.75
```
**CRITICAL EXPERIMENT**: FastAI gets 65-95% stacking weight in most markets. If boosting-only ensembles match or beat deployed performance, FastAI is a liability, not an asset. This answers the single most important open question about our ensemble architecture.

**Job 5 — Seed robustness validation**
```
markets: home_win,shots,fouls_over_225
feature_params_mode: best
seed: 123
n_trials: 150, walkforward: true, shap: true
sample_weights: true
```
Hypothesis: Top 3 performers should produce <15pp ROI divergence across seeds. S14 showed fouls had 26.9pp divergence — overfitting flag. This validates whether home_win (R220), shots (S13), fouls_o225 (S13) are seed-robust.

### Expected Outcomes

| Job | If better → | If worse → |
|-----|-------------|------------|
| 1 | Deploy updated feature params for improved niche markets | Confirm current params are near-optimal |
| 2 | Deploy line variant improvements, especially cards_o35 | Confirm S8 configs still best |
| 3 | Graduate away_win from paper to production | Keep paper trading, investigate why |
| 4 | **Paradigm shift**: drop FastAI from ensemble, simplify pipeline | FastAI confirmed essential despite weight collapse |
| 5 | Deployed configs are robust (seed-independent) | Flag overfitting, investigate which markets are seed-sensitive |

### What We Learn (regardless of outcome)

- **Job 1+2**: Whether 50 trials + 3 folds is a meaningful improvement over 25 trials + 2 folds for feature params
- **Job 3**: Whether away_win has genuine signal or R220 was lucky
- **Job 4**: Whether FastAI's high stacking weight reflects true alpha or meta-learner exploitation
- **Job 5**: Which markets are seed-robust vs seed-fragile (informs future optimization strategy)

---

## Session 14 — Feature Param Expansion + Adversarial-Cleaned catboost_merge (COMPLETE)

### Code Changes (committed 9a57e9d, pushed to main)

1. **BET_TYPE_PARAM_PRIORITIES expanded** — Added `half_life_days`, `h2h_matches`, `goal_diff_lookback`, `home_away_form_window` to all markets (4 params never tuned in 13 sessions).
2. **catboost_merge Phase 2 adversarial filter FIXED** — Was checking `== "true"` instead of parsing `"passes,features,auc"` like Phase 1. All S13 catboost_merge Phase 2 results were confounded.
3. **Temperature calibration added to Optuna** — Search space now `["sigmoid", "beta", "temperature"]`.
4. **Search space boundaries expanded** — `ema_span` max→35, `home_away_form_window`→[1,15], `goal_diff_lookback`→[1,15], `elo_home_advantage` max→350.
5. **Temperature calibration fix** — Map temperature to sigmoid for CalibratedClassifierCV, apply post-hoc (94f3f9c).

### Jobs (3 batches, 8 runs total)

| Job | Run ID | Markets | Key Lever | Status |
|-----|--------|---------|-----------|--------|
| 3 | R208 | corners,corners_o85,cards,cards_o35 | Aggressive adversarial (5,15,0.65) | **DONE — ALL REGRESSED** |
| 5 | R210 | fouls,fouls_o225,shots,shots_o225 | decay=0.008 + seed=123 | **DONE — 0.005 WINS** |
| 1 | CANCELLED | fouls,shots,corners,cards,btts | 4 new feature params + catboost_merge | Superseded by R218 |
| 2a | CANCELLED | shots_o225,shots_o265 | 4 new feature params + catboost_merge | Superseded by R219 |
| 2b | CANCELLED | fouls_o225,corners_o85,cards_o35 | 4 new feature params + catboost_merge | Superseded by R219 |
| 4 | CANCELLED | home_win,away_win,over25,under25 | 200 trials + feature params + odds-threshold | Superseded by R220 |
| temp | R212-R216 | 7 markets (3 used temperature) | Temperature calibration fix test | **DONE — NON-EVENT** |
| R218 | R218 | fouls,shots,corners,cards,btts | Feature params + workflow fix | **DONE** |
| R219 | R219 | corners_o85,shots_o225,shots_o265,fouls_o225,cards_o35 | Feature params + workflow fix | **DONE** |
| R220 | R220 | home_win,away_win,over25,under25 | Feature params + workflow fix | **DONE** |

### Batch 1 Results (R208+R210, original S14 jobs)

**Job 3 — Aggressive adversarial (5,15,0.65): ALL REGRESSED**
- corners: +47.1% WF (vs deployed +65.4%) — **regression**
- corners_o85: +51.7% WF (vs deployed +59.6%) — **regression**
- cards: +60.2% WF (vs deployed +81.8%) — **regression**
- cards_o35: +71.3% WF (vs deployed +81.9%) — **regression**
- Aggressive adversarial removes 37-48% features — too destructive for weak-signal markets
- **DEFINITIVELY CLOSED** (2x confirmed: S9 + S14)

**Job 5 — decay=0.008 + seed=123: 0.005 WINS**
- fouls: +96.1% WF (seed=123) vs deployed +139.6% (seed=42) — **26.9pp seed divergence (overfitting flag)**
- fouls_o225: +115.3% WF vs deployed +127.6% — regression
- shots: +108.2% WF vs deployed +113.1% — flat
- shots_o225: +106.5% WF vs deployed +119.1% — regression
- decay=0.008 does NOT beat 0.005 (faster decay loses too much historical signal)
- Fouls >20pp seed divergence = **confirmed overfitting concern**

### Batch 2 Results (R212-R216, temperature calibration fix)

Temperature calibration appeared in only **3/21 market-run combos**. All 3 WF ROI worse than deployed. Temperature calibration is a **non-event** — it never wins Optuna selection and the 3 times it did, it underperformed. No deployments.

### Batch 3 Results (R218-R220, feature params + workflow fix) — THE MAIN RESULTS

**R218 — Niche base markets (fouls, shots, corners, cards, btts):**
- **fouls**: average +136.5% WF, 0 HO bets → **KEEP deployed** (+139.6%)
- **shots**: LGB +95.5% WF, +80.0% HO (50b) → **KEEP deployed** (+113.1%/+118.8%)
- **corners**: LGB +55.3% WF, +39.2% HO (97b) → **KEEP deployed** (+65.4%)
- **cards**: stacking +81.8% WF, +87.5% HO (4b) → **KEEP** (tiny HO)
- **btts**: LGB +76.8% WF, +100% HO (5b) → **KEEP deployed** (+108.3%)

**R219 — Niche line variants (corners_o85, shots_o225, shots_o265, fouls_o225, cards_o35):**
- **corners_o85**: agreement +60.4% WF, +55.1% HO (1193b) → **KEEP** (flat)
- **shots_o225**: disagree_cons +124.8% WF, +101.8% HO (114b) → **KEEP** (marginal)
- **shots_o265**: catboost +98.1% WF, +71.2% HO (73b) → **KEEP** (WF-HO gap = overfit)
- **fouls_o225**: average +124.3% WF, +123.7% HO (19b) → **KEEP** (flat)
- **cards_o35**: fastai +59.0% WF, +22.9% HO (669b) → **DEGRADED** (from +81.9%/+66.1%)

**R220 — H2H markets (home_win, away_win, over25, under25):**
- **home_win**: temporal_blend +111.5% WF, +114.6% HO (99b) → **UPGRADED** (2x HO volume, same ROI)
- **away_win**: catboost +113.6% WF, +120.6% HO (17b) → **PAPER TRADE** (first viable HO ever)
- **over25**: disagree_cons +82.1% WF, +52.2% HO (23b) → **KEEP deployed** (feature params hurt, 4x confirmed)
- **under25**: disagree_cons +64.0% WF, +150% HO (2b) → **DISABLE** (5x confirmed intractable)

### Deployment Actions Taken

1. **home_win UPGRADED** to R220 temporal_blend (models copied, deployment config updated, feature params updated)
2. **away_win** set up for paper trading with R220 catboost model
3. Search space boundaries expanded (`ema_span`→35, `home_away_form_window`→[1,15], `goal_diff_lookback`→[1,15], `elo_home_advantage`→350)

### Bug Fixes (during S14)

| Bug | Fix | Commit |
|-----|-----|--------|
| Prematch odds lookup pandas Series truth value | `is None` check instead of `or` | f3e2785 |
| 21 model files missing from HF Hub | Uploaded 8 models, removed 3 phantom config refs | a8a3f02 |
| Temperature calibration in Optuna (sigmoid map + post-hoc) | Map to sigmoid for CalibratedClassifierCV | 94f3f9c |

---

## Session 13 — catboost_merge + Niche Levers (COMPLETE)

### Code Changes (committed a73fe2c, pushed to main)

1. **Normalization made market-specific** — z-score normalization ON for 13 niche markets, OFF for H2H.
2. **generate_deployment_config.py fixed** — handles array-format JSON, correct field mappings.
3. **feature_eng_pipeline.py** — removed blanket normalization call.

### Results (5 jobs: R196-R200)

| Job | Markets | Lever | Key Results |
|-----|---------|-------|-------------|
| 1 (R196) | fouls, cards, shots, corners, btts | catboost_merge | **shots UPGRADED** catboost HO +118.8% (40 bets, Sharpe 1.42) |
| 2 (R197) | fouls_o225, cards_o35, corners_o85, shots_o225 | catboost_merge | **fouls_o225 UPGRADED** average WF +127.6%, HO +129.2% |
| 3 (R198) | home_win, over25, away_win, under25 | catboost_merge | H2H REGRESSED: home_win +22% vs deployed +120%, over25 +13% vs +93% |
| 4 (R199) | corners, fouls, shots_o265, cards | beta calibration | fouls +131.6% catboost (no upgrade — HO only 1 bet) |
| 5 (R200) | corners, btts, cards_o35 | odds-threshold α=0.2 | No improvements over deployed |

### Deployment Decisions

| Market | Decision | Rationale |
|--------|----------|-----------|
| **shots** | **UPGRADED** → catboost, HO +118.8% | Deployed had 0 holdout, 1 WF fold. New has Sharpe 1.42, 2 WF folds |
| **fouls_o225** | **UPGRADED** → average, WF +127.6% | +8pp WF improvement. CatBoost got stacking weight via catboost_merge |
| home_win | KEEP | Massive regression (H2H + catboost_merge = bad) |
| over25 | KEEP | Massive regression |
| fouls | KEEP | R196 lightgbm +121.6% WF but current deployed +139.6% is better |
| cards | KEEP | R196 average +64.3% ≈ deployed +81.8% |
| btts | KEEP | R196 xgboost +64.2% vs deployed +60.4% — marginal, not enough to switch |
| corners | KEEP | R196 catboost +74.2% (33 bets) better WF but no HO improvement |
| shots_o265 | KEEP | R199 catboost HO +88.9% vs deployed +93.5% — actually worse |
| cards_o35 | KEEP | R197 agreement +74.7% vs deployed +81.9% — regression |
| corners_o85 | KEEP | R197 agreement +62.1% vs deployed +59.6% — marginal |
| shots_o225 | KEEP | R197 temporal_blend +108.3% vs deployed +119.1% — regression |

### S13 Key Findings

1. **catboost_merge worked** — Phase 2 ran successfully (first time, post-2842dd9 fix). CatBoost earned non-zero stacking weight in 8/13 markets tested.
2. **catboost_merge + H2H = disaster** — home_win, over25, away_win all massively regressed. Two-stage models still only viable path for H2H.
3. **catboost_merge + niche = mixed** — Helped shots (+118.8% HO) and fouls_o225 (+8pp WF), but didn't help other niche markets.
4. **Adversarial AUC extreme** — 0.89-0.99 across ALL markets. Train/test easily distinguished. Phase 2 skips adversarial filtering which confounds comparison.
5. **Beta calibration (R199)** — No clear wins. fouls +131.6% catboost but HO had only 1 bet.
6. **Odds-threshold (R200)** — No improvements on corners/btts/cards_o35.

---

## Session 12 — Normalization A/B Test (COMPLETE)

### Results

| Run | Markets | Key Finding |
|-----|---------|-------------|
| 1 (21925789425) | fouls, cards, shots, corners, btts | Niche: fouls +29pp, btts +11pp, corners +10pp |
| 2 (21925817964) | home_win, over25 | H2H REGRESSED: home_win -92pp, over25 -84pp |
| 3 (21925856923) | fouls_o225, fouls_o265, fouls_o285, cards_o35 | Mixed line variant results |
| 4 (21925884302) | corners_o85, shots_o225, shots_o265, shots_o285 | Some lines improved |
| 5 (21926002210) | away_win, under25 | away_win deployed (new), under25 still dead |

**Key finding:** Adversarial AUC NOT reduced by normalization (still 0.97-1.0). Normalization changes feature distributions but structural temporal patterns remain. Decision: make it market-specific (niche ON, H2H OFF).

---

## Session 11 — Three Untested Levers (COMPLETE)

### Results

| Market | S11 Best | S11 WF ROI | S11 HO ROI | S11 HO Bets | Deployed WF | Verdict |
|--------|----------|-----------|-----------|-------------|-------------|---------|
| **fouls** | stacking | **+139.6%** | Empty | 0 | +110.9% | **DEPLOYED (+29pp)** |
| **shots** | xgboost | +103.1% | **+104.5%** | 66 | +122.2% | **DEPLOYED (HO excellent)** |
| **btts** | agreement | **+108.3%** | Empty | 0 | +61.1% | **DEPLOYED (+47pp)** |
| corners | catboost | +84.2% | +20.0% | 25 | +65.4% | SKIP (HO weak) |
| cards | agreement | +64.3% | +150% | 1 | +81.8% | SKIP (regressed -17pp) |

### Key Findings
- **Odds-threshold + decay = best combo for fouls** (+139.6% WF stacking)
- **Beta calibration fixed btts stacking weights** (balanced ensemble for first time)
- **Shots HO +104.5% on 66 bets** (Sharpe=1.076) — strongest holdout in portfolio

---

## Session 10 — catboost_merge Experiment (FAILED — bug prevented Phase 2)

S10 launched 5 catboost_merge runs but ALL had `catboost_merge` step skipped due to transitive skip cascade bug. Fix committed in 2842dd9 but AFTER S10 runs. S13 is the first real test.

---

## Current Deployment (14 markets on HF Hub)

| Market | Strategy | WF ROI | HO ROI | HO Bets | Source | Fastai % |
|--------|----------|--------|--------|---------|--------|----------|
| **home_win** | temporal_blend | +111.5% | +114.6% | 99 | **S14 R220** | — |
| **over25** | lightgbm | +93.2% | +100.0% | 40 | S7 | 71% |
| **fouls** | temporal_blend→stacking | +139.6% | — | 0 | S11 | ~100% |
| **shots** | catboost | +113.1% | +118.8% | 40 | **S13** | 55% |
| **btts** | stacking→agreement | +108.3% | — | 0 | S11 | balanced |
| **cards** | agreement | +81.8% | — | 0 | S8 | 83% |
| **corners** | lightgbm | +65.4% | +55.4% | 37 | S7 | 80% |
| **fouls_o225** | average | +127.6% | +129.2% | 12 | **S13** | 97% |
| **fouls_o265** | lightgbm | +120.4% | +87.5% | 4 | S9 | 99% |
| **shots_o225** | catboost | +119.1% | +102.8% | 159 | S8 | 52% |
| **shots_o265** | disagree_agg | +83.8% | +93.5% | 31 | S9b | 55% |
| **shots_o285** | disagree_bal | +87.5% | +87.5% | 8 | S7 | 83% |
| **corners_o85** | agreement | +59.6% | +57.0% | 1207 | S8 | 85% |
| **cards_o35** | agreement | +81.9% | +66.1% | 462 | S8 | 81% |

**Niche market thresholds:** fouls 0.80, btts 0.60, corners 0.60, shots 0.75, cards 0.55, fouls_o225 0.80, fouls_o265 0.80, corners_o85 0.60, shots_o225 0.75, shots_o265 0.55, shots_o285 0.55, cards_o35 0.60

---

## Confirmed Levers (what works)

| Lever | Evidence | Use For |
|-------|----------|---------|
| **Odds-threshold α=0.2** | home_win +113.5% HO, over25 +100% HO, fouls +139.6% WF | H2H + fouls |
| **decay=0.005** | All niche improved (S7, S9b, S11) | Niche markets |
| **Beta calibration** | btts weights balanced (S11) | Markets with fastai collapse |
| **Agreement ensembles** | cards_o35 +66.1% HO (462), corners_o85 +57% HO (1207) | High-volume lines |
| **Z-score normalization (niche only)** | fouls +29pp, btts +11pp, corners +10pp (S12) | Niche markets |
| **catboost_merge (niche)** | S13: shots +118.8% HO, fouls_o225 +8pp WF | Niche only (hurts H2H) |

## Confirmed Failures (do NOT re-test)

| Lever | Evidence | Notes |
|-------|----------|-------|
| Normalization for H2H | S12 — home_win -92pp, over25 -84pp | Only helps niche |
| Aggressive adversarial for niche | S9 J3 + **S14 J3** — all 4 regressed both times | 2x confirmed, even WITH normalization |
| Decay 0.008 (vs 0.005) | S14 J5 — all 4 markets worse | 0.005 is optimal |
| Auto-RFE for weight collapse | S9 J5 — 12/17 still fastai-dominated | RFECV didn't help |
| Feature optimize for H2H (except home_win) | S8, R89, R95, **R220 over25** all worse | 4x confirmed (over25); home_win benefited in R220 |
| Fouls seed robustness | S14 J5 — 26.9pp divergence (seed=42 vs 123) | >20pp threshold = overfitting flag |
| under25 | S7, S9b, S10, S11, **S14 R220** — near-zero WF ROI | 5x confirmed intractable |
| cards_o65, corners_o115 | S9b — dead markets | Don't re-test |
| catboost_merge for H2H | S13 — home_win -98pp, over25 -80pp | Only helps niche |
| Temperature calibration | S14 R212-R216 — 0/3 markets improved | Never wins Optuna selection |
| Feature params for over25 | S8, R89, R95, **S14 R220** all worse | 4x confirmed harmful |
| cards_o35 degrading | S14 R219 — +22.9% HO vs deployed +66.1% | Monitor; fastai-dominated run |

---

## Bugs Fixed (Recent)

| Bug | Commit | Session |
|-----|--------|---------|
| Normalization market-specific + config parser fix | a73fe2c | S13 |
| Rolling z-score normalization feature | 82abaf5 | S12 |
| catboost_merge skip cascade (always() fix) | 2842dd9 | S10 |
| TwoStageModel interface + sigmoid calibration | 2264d4d | S10 |
| Isotonic calibration collapse detection | 04594c0 | S10 |
| Boolean input comparison (CLI string vs UI bool) | 440ed8b | S10 |

---

## Code Fixes Still Needed

### Two-Stage Model Interface (low priority)
`TwoStageModel.predict_proba()` requires `odds` argument but `ModelLoader.predict()` doesn't pass it. Fixed in commit 2264d4d.

### Corners O8.5 Odds (medium priority)
The Odds API doesn't return corners O8.5 odds. Need alternative odds source.

### Fastai Weight Collapse (partially addressed by S13 catboost_merge)
catboost_merge gave CatBoost non-zero stacking weight in 8/13 niche markets. But fastai still dominates most (>90% in many). If needed:
- Constrained stacking weights (max weight per model)
- Different meta-learner (Lasso, ElasticNet)
- Simple average fallback when fastai > 80% weight

---

## Key Files

| Purpose | Path |
|---------|------|
| Deployment config | `config/sniper_deployment.json` (+ HF Hub) |
| Feature params | `config/feature_params/*.yaml` (+ HF Hub) |
| Normalization module | `src/features/normalization.py` |
| Config manager (NICHE_NORMALIZE_MARKETS) | `src/features/config_manager.py` |
| Sniper optimizer | `experiments/run_sniper_optimization.py` |
| Daily recommendations | `experiments/generate_daily_recommendations.py` |
| CI: Sniper workflow | `.github/workflows/sniper-optimization.yaml` |
| Deployment config generator (FIXED) | `scripts/generate_deployment_config.py` |
| Model loader | `src/ml/model_loader.py` |
| Upload/download HF | `entrypoints/upload_data.py` / `download_data.py` |
