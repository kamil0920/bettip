# Bettip — Current State & Next Steps (Feb 12, 2026)

## Session 13 — catboost_merge + Niche Levers (5 jobs RUNNING)

### Code Changes (committed a73fe2c, pushed to main)

1. **Normalization made market-specific** — z-score normalization ON for 13 niche markets (fouls/cards/shots/corners/btts + all line variants), OFF for H2H (home_win/away_win/over25/under25). S12 showed normalization hurts H2H (-92pp home_win, -84pp over25) but helps niche (+29pp fouls, +11pp btts, +10pp corners).
2. **generate_deployment_config.py fixed** — was silently skipping array-format JSON (`sniper_all_*.json`), wrong field names (`best_roi` → `roi`, `best_bets` → `n_bets`, `selected_features` → `optimal_features`). Now handles both dict and array formats, maps SniperResult fields correctly, extracts holdout metrics from sub-dict.
3. **feature_eng_pipeline.py** — removed blanket normalization call. Raw features stored in parquet; normalization happens per-market during sniper optimization via `regeneration.py`.

### Running Jobs

| Job | Run ID | Markets | Lever | Est. time |
|-----|--------|---------|-------|-----------|
| 1 | 21938483698 | fouls, cards, shots, corners, btts | catboost_merge + decay=0.005 | ~3h (Phase 1+2) |
| 2 | 21938512005 | fouls_o225, cards_o35, corners_o85, shots_o225 | catboost_merge + decay=0.005 | ~3h (Phase 1+2) |
| 3 | 21938541486 | home_win, over25, away_win, under25 | catboost_merge (n_trials=75) | ~4h (Phase 1+2) |
| 4 | 21938570348 | corners, cards, fouls, shots_o265 | beta calibration + decay=0.005 | ~2.5h |
| 5 | 21938599508 | corners, btts, cards_o35 | odds-threshold α=0.2 + decay=0.005 | ~2.5h |

All: save_models=true, upload_results=true, only_if_better=true, walkforward=true, shap=true.

**NOTE:** All niche markets will automatically have z-score normalization enabled (code change in config_manager.py). This is the first time catboost_merge actually runs with the bug fix (commit 2842dd9 added `always()` to break skip cascade).

### What to Check When Jobs Finish

1. **catboost_merge actually ran?** — Check jobs 1-3: `catboost_merge` step should show `completed/success` not `skipped`
2. **Stacking weight rebalancing** — Did CatBoost get non-zero weight? Compare vs deployed weights
3. **Normalization + catboost_merge combo** — First time these two levers are combined for niche
4. **Beta calibration on new markets** — Did it fix fastai dominance in corners/cards/fouls like it did for btts?
5. **Odds-threshold on corners/btts/cards_o35** — Did it help like it did for fouls/shots?
6. **H2H catboost_merge** — Previous S9b analysis said base models produce 0 WF bets for H2H (only two-stage works). Verify this is still true.

### Analysis Commands

```bash
# Check job status
GH_TOKEN=... gh run list --repo kamil0920/bettip --limit 7

# Check specific job's catboost_merge step
GH_TOKEN=... gh run view <RUN_ID> --repo kamil0920/bettip --json jobs --jq '.jobs[] | "\(.name)\t\(.conclusion)"'

# Download logs (large runs need curl+zip method)
GH_TOKEN=... curl -L -H "Authorization: Bearer $GH_TOKEN" \
  "https://api.github.com/repos/kamil0920/bettip/actions/runs/<RUN_ID>/logs" -o logs.zip && unzip logs.zip
```

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

## Current Deployment (16 markets on HF Hub)

| Market | Strategy | WF ROI | HO ROI | HO Bets | Source | Fastai % |
|--------|----------|--------|--------|---------|--------|----------|
| **home_win** | stacking | +119.9% | +113.5% | 48 | S7 | 88% |
| **over25** | lightgbm | +93.2% | +100.0% | 40 | S7 | 71% |
| **fouls** | temporal_blend→stacking | +139.6% | — | 0 | S11 | ~100% |
| **shots** | stacking→xgboost | +103.1% | +104.5% | 66 | S11 | 90% |
| **btts** | stacking→agreement | +108.3% | — | 0 | S11 | balanced |
| **cards** | agreement | +81.8% | — | 0 | S8 | 83% |
| **corners** | lightgbm | +65.4% | +55.4% | 37 | S7 | 80% |
| **fouls_o225** | agreement | +127.7% | +118.8% | 56 | S9b | 49% |
| **fouls_o265** | lightgbm | +120.4% | +87.5% | 4 | S9 | 99% |
| **fouls_o285** | average | +49.1% | +25.0% | 6 | S9b | 71% |
| **shots_o225** | catboost | +119.1% | +102.8% | 159 | S8 | 52% |
| **shots_o265** | xgboost | +84.3% | +93.5% | 31 | S9b | 55% |
| **shots_o285** | disagree_bal | +87.5% | +87.5% | 8 | S7 | 83% |
| **corners_o85** | agreement | +59.6% | +57.0% | 1207 | S8 | 85% |
| **cards_o35** | agreement | +81.9% | +66.1% | 462 | S8 | 81% |
| under25 | DISABLED | — | — | — | — | — |
| away_win | DISABLED | — | — | — | — | — |

**Niche market thresholds:** fouls 0.80, btts 0.60, corners 0.60, shots 0.75, cards 0.55, fouls_o225 0.80, fouls_o265 0.80, fouls_o285 0.55, corners_o85 0.60, shots_o225 0.75, shots_o265 0.55, shots_o285 0.55, cards_o35 0.60

---

## Confirmed Levers (what works)

| Lever | Evidence | Use For |
|-------|----------|---------|
| **Odds-threshold α=0.2** | home_win +113.5% HO, over25 +100% HO, fouls +139.6% WF | H2H + fouls |
| **decay=0.005** | All niche improved (S7, S9b, S11) | Niche markets |
| **Beta calibration** | btts weights balanced (S11) | Markets with fastai collapse |
| **Agreement ensembles** | cards_o35 +66.1% HO (462), corners_o85 +57% HO (1207) | High-volume lines |
| **Z-score normalization (niche only)** | fouls +29pp, btts +11pp, corners +10pp (S12) | Niche markets |
| **catboost_merge** | FIRST REAL TEST IN S13 | TBD |

## Confirmed Failures (do NOT re-test)

| Lever | Evidence | Notes |
|-------|----------|-------|
| Normalization for H2H | S12 — home_win -92pp, over25 -84pp | Only helps niche |
| Aggressive adversarial for niche | S9 J3 — all 4 regressed | Only helps H2H |
| Auto-RFE for weight collapse | S9 J5 — 12/17 still fastai-dominated | RFECV didn't help |
| Feature optimize for H2H | S8, R89, R95 all worse | 3x confirmed |
| under25 | S7, S9b, S10, S11 — near-zero WF ROI | 4x confirmed intractable |
| cards_o65, corners_o115 | S9b — dead markets | Don't re-test |

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

### Fastai Weight Collapse (S13 will test catboost_merge fix)
If S13 catboost_merge doesn't help, investigate:
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
