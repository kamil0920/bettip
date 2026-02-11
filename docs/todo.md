# Bettip — Current State & Next Steps (Feb 11, 2026)

## Session 10 — catboost_merge Experiment (5 jobs running)

### Motivation: Fastai Weight Collapse

Deep analysis of all 15 deployed markets revealed **fastai dominates stacking weights in 14/15 markets** (71-99% of total weight). xgboost has ZERO weight in every single market. This means stacking ensembles are essentially fastai-only predictions, which limits diversity and reliability.

**catboost_merge** gives CatBoost dedicated hyperparameter tuning (50 trials) in a separate Phase 2, instead of competing for trials with other models. This is the first time catboost_merge has actually run (S9 boolean bug prevented it).

### Key Findings That Shaped Job Design

1. **S9b Phase 1 natural experiment** (no-catboost runs):
   - cards_o35 DROPPED 8% without catboost (+73.8% vs deployed +81.9%) → catboost was contributing
   - corners_o85 unchanged (+59.8% vs +59.6%) → catboost wasn't needed
   - shots_o225 unchanged (+117.9% vs +119.1%) → stacking works without catboost
   - home_win/over25: only two-stage models produced bets → catboost won't help H2H

2. **Stacking weight analysis** (fastai % of total weight):
   - 90-99%: fouls_o265, shots
   - 80-89%: home_win, corners_o85, cards, fouls, shots_o285, btts, cards_o35, corners
   - 70-79%: over25, fouls_o285
   - 50-59%: shots_o265, shots_o225

3. **Stacking underperforms agreement** in most niche markets — only shots_o225 benefits from stacking

### Running Jobs

| Job | Run ID | Markets | Hypothesis | Config |
|-----|--------|---------|-----------|--------|
| A | 21905899097 | cards_o35, corners_o85, fouls_o225 | 4th catboost vote improves agreement precision | catboost_merge + decay=0.005 |
| B | 21905933212 | fouls, cards, shots | Break fastai monopoly (83-100% dominance) | catboost_merge + decay=0.005 |
| C | 21905966780 | btts, under25 | Dedicated CB tuning enables ensemble for 2-way | catboost_merge + odds-threshold α=0.2 |
| D | 21905996541 | shots_o225, shots_o265, shots_o285 | Improve shot line variant ensembles | catboost_merge + decay=0.005 |
| E | 21906030520 | fouls_o265, fouls_o285 | Improve fouls line variant ensembles | catboost_merge + decay=0.005 |

All: adversarial=2,10,0.75, n_trials=150, save_models=true, upload_results=true, only_if_better=false.

**Timing:** Phase 1 ~2-4h + Phase 2 ~1-1.5h per job. All have separate 5h55m timeouts per phase.

### What catboost_merge CANNOT help (excluded)

- **home_win, over25**: Only two-stage models produce WF bets. Base models (lgb/xgb/fastai/catboost) all get zero bets after threshold filtering. Adding catboost changes nothing.
- **away_win**: Disabled, tiny holdout.

### Expected Outcomes

| Scenario | What it means | Next step |
|----------|---------------|-----------|
| Job A improves agreement markets | Dedicated CB tuning adds diversity to consensus | Always use catboost_merge for agreement markets |
| Job B breaks fastai monopoly | Stacking problem is solvable via better CB params | catboost_merge becomes standard for all |
| Nothing improves | Fastai dominance is a Ridge meta-learner problem | Investigate alternative stacking (constrained weights, different meta-learner) |
| Job C helps btts | odds-threshold + catboost_merge = winning 2-way combo | Test on other 2-way markets |

---

## Session 9b — COMPLETE (Feb 11)

### Results Summary

All 5 runs completed. Upload to HF Hub skipped (YAML boolean bug, fixed commit 558be59). Deployed manually.

| Job | Market | Best Model | WF ROI | vs Deployed | Verdict |
|-----|--------|-----------|--------|-------------|---------|
| 1v2 | home_win | two_stage_xgb | +16.7% | -103% | SKIP |
| 1v2 | over25 | two_stage_xgb | +15.5% | -78% | SKIP |
| 2v2 | corners_o85 | stacking | +59.8% | ~SAME | SKIP |
| 2v2 | cards_o35 | agreement | +73.8% | -8% | SKIP |
| 2v2 | shots_o225 | average | +117.9% | ~SAME | SKIP |
| **A** | **btts** | **xgboost** | **+61.1%** | +0.7% | **DEPLOYED** |
| A | under25 | two_stage_lgb | +20.2% | NEW | SKIP (thin) |
| **B** | **fouls_o225** | **agreement** | **+127.7%** | +8.2% | **DEPLOYED** |
| **B** | **shots_o265** | **xgboost** | **+84.3%** | +0.5% | **DEPLOYED** |
| B | shots_o285 | — | — | — | DEAD |
| **C** | **fouls_o285** | **average** | **+49.1%** | NEW | **DEPLOYED** |
| C | cards_o65 | xgboost | -18.3% | — | DEAD |
| C | corners_o115 | xgboost | +10.2% | — | DEAD |

---

## Current Deployment (15 markets on HF Hub)

| Market | Strategy | WF ROI | WF Bets | HO ROI | HO Bets | Source | Fastai % |
|--------|----------|--------|---------|--------|---------|--------|----------|
| **home_win** | stacking | +119.9% | 135 | +113.5% | 48 | S7 | 88% |
| **over25** | lightgbm | +93.2% | 119 | +100.0% | 40 | S7 | 71% |
| **fouls** | temporal_blend | +110.9% | 104 | +150% | 1 | S7 | 100% |
| **shots** | stacking | +122.2% | 27 | — | 0 | S7 | 90% |
| **fouls_o225** | agreement | +127.7% | 42 | — | — | S9b | 49% |
| **fouls_o265** | lightgbm | +120.4% | 31 | +87.5% | 4 | S9 | 99% |
| **fouls_o285** | average | +49.1% | 72 | +25.0% | 6 | S9b | 71% |
| **shots_o225** | catboost | +119.1% | 307 | +102.8% | 159 | S8 | 52% |
| **shots_o265** | xgboost | +84.3% | 88 | +78.6% | 21 | S9b | 55% |
| **shots_o285** | disagree_bal | +87.5% | 15 | +87.5% | 8 | S7 | 83% |
| **corners_o85** | agreement | +59.6% | 6371 | +57.0% | 1207 | S8 | 85% |
| **cards_o35** | agreement | +81.9% | 2569 | +66.1% | 462 | S8 | 81% |
| **cards** | agreement | +81.8% | 22 | — | 0 | S8 | 83% |
| **corners** | lightgbm | +65.4% | 31 | +55.4% | 37 | S7 | 80% |
| **btts** | xgboost | +61.1% | 45 | — | — | S9b | 82% |
| under25 | — | +20.2% | 5 | — | — | DISABLED | — |
| away_win | — | — | — | — | — | DISABLED | — |

---

## Bugs Fixed (Sessions 9-9b)

| Bug | Commit | Impact |
|-----|--------|--------|
| Orphan cleanup deleting models for unrelated markets | b95296e | 17 models lost → recovered |
| Numpy int32 JSON serialization crash | 82703af | Truncated result files |
| YAML boolean comparison (`== 'true'` vs `== true`) | 558be59 | catboost_merge + Upload skipped on all runs |

---

## Confirmed Levers (what works)

| Lever | Evidence | Use For |
|-------|----------|---------|
| **Odds-threshold α=0.2** | home_win +113.5% HO, over25 +100% HO | 2-way markets |
| **Aggressive adversarial (5,15,0.65)** | Best H2H improvement | H2H only |
| **decay=0.005** | All base niche improved (S7), line variants improved (S9b) | Niche markets |
| **Agreement ensembles** | cards_o35 +66.1% HO (462), corners_o85 +57% HO (1207), fouls_o225 +127.7% WF | High-volume lines |
| **Standard adversarial (2,10,0.75)** | Baseline that works everywhere | All markets |
| **catboost_merge** | UNTESTED (boolean bug prevented it). S10 is the first real test. | TBD |

## Confirmed Failures (do NOT re-test)

| Lever | Evidence | Notes |
|-------|----------|-------|
| Aggressive adversarial for niche | S9 J3 — all 4 regressed | Only helps H2H |
| Auto-RFE for weight collapse | S9 J5 — 12/17 still fastai-dominated | RFECV didn't help |
| Feature optimize for H2H | S8 J2a, R89, R95 all worse | 3x confirmed |
| Isotonic for BTTS | S8 — no improvement | Stacking itself was broken |
| cards_o65 | S9b — WF -18.3% | Dead market |
| corners_o115 | S9b — WF +10.2% | Dead market |
| shots_o285 with decay | S9b — 0 bets optimized | Dead line |
| catboost_merge for H2H | S9b Phase 1 showed base models produce 0 WF bets | Only two-stage works for H2H |

---

## Code Fixes Still Needed

### Two-Stage Model Interface (low priority)
`TwoStageModel.predict_proba()` requires `odds` argument but `ModelLoader.predict()` doesn't pass it. Would enable 6/6 home_win models in stacking (currently 4/6). Low impact since home_win already +113.5% HO.

### Corners O8.5 Odds (medium priority)
The Odds API doesn't return corners O8.5 odds. Need alternative odds source. Without odds, corners_o85 model (+57% HO, 1207 bets) is unbettable via automation.

### Fastai Weight Collapse (research needed)
If S10 catboost_merge doesn't fix the problem, investigate:
- Constrained stacking weights (max weight per model)
- Different meta-learner (Lasso, ElasticNet, simple average fallback)
- Fastai ensemble penalty (reduce weight if fastai WF bets < threshold)

---

## Session History (Condensed)

### Session 10 (Feb 11) — catboost_merge Experiment
- Deep analysis: fastai dominates 14/15 markets (71-99% stacking weight), xgboost=0 everywhere
- 5 jobs launched testing catboost_merge across ALL 13 niche markets + btts/under25
- First time catboost_merge has actually executed (S9 boolean bug prevented it)

### Session 9b (Feb 11) — S9 Re-Runs + New Jobs
- Fixed 3 bugs: orphan cleanup scope, numpy int32 serialization, YAML boolean comparisons
- 5 jobs completed. Deployed 4 markets: fouls_o225, shots_o265, fouls_o285 (new), btts
- Dead markets: cards_o65, corners_o115, shots_o285 (with decay)

### Session 9 (Feb 11) — catboost_merge + Systematic Tests
- 5 jobs testing catboost_merge, aggressive adversarial for niche, odds-threshold for lines, auto-RFE
- catboost_merge Phase 2 skipped (YAML bug). Auto-RFE didn't fix fastai weight collapse
- Deployed 2 markets: fouls_o265 (+120.4%), fouls_o225 (first HO +118.8%)

### Session 8 (Feb 10) — Mass Optimization
- Deployed corners_o85 (+57% HO, 1207 bets), cards_o35 (+66.1% HO, 462 bets), shots_o225 (+102.8% HO)
- Agreement strategy proven for high-volume line variants

### Session 7 (Feb 10) — Deployment Rebuild + Pipeline Fixes
- Rebuilt deployment from scratch after retry wipe. 12 markets deployed
- decay=0.005 helped all niche. Odds-threshold best for H2H

### Session 5 (Feb 9) — Temporal Leakage Cleanup
- Replaced all R90-era models with clean versions. Added Kelly, Poisson GLM, Bayesian features

### Session 4 (Feb 9) — Model Quality Fixes
- Adversarial filtering, non-negative stacking, calibration validation

### Session 2-3 (Feb 8) — Data Quality + Multi-Line
- Fixed niche data quality (+129% data). 12 new line variants

### Session 1 (Feb 8) — Multi-Line Support
- Added multi-line niche market support

---

## Key Files

| Purpose | Path |
|---------|------|
| Deployment config | `config/sniper_deployment.json` (+ HF Hub) |
| Feature params | `config/feature_params/*.yaml` (+ HF Hub) |
| Sniper optimizer | `experiments/run_sniper_optimization.py` |
| Daily recommendations | `experiments/generate_daily_recommendations.py` |
| CI: Sniper workflow | `.github/workflows/sniper-optimization.yaml` |
| CI: Prematch workflow | `.github/workflows/prematch-intelligence.yaml` |
| CI: Data collection | `.github/workflows/collect-match-data.yaml` |
| Deployment config updater | `scripts/update_deployment_config.py` |
| Deployment config generator | `scripts/generate_deployment_config.py` |
| Model loader | `src/ml/model_loader.py` |
| Upload/download HF | `entrypoints/upload_data.py` / `download_data.py` |
