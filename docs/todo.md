# Bettip — Current State & Next Steps (Feb 11, 2026)

## Currently Running Jobs (5 active)

### Session 9b — New Jobs (launched ~07:03 UTC)

| Job | Run ID | Markets | Key Lever | Status |
|-----|--------|---------|-----------|--------|
| A | 21895799398 | btts, under25 | odds-threshold α=0.2 | in_progress |
| B | 21895824491 | fouls_o225, shots_o265, shots_o285 | decay=0.005 | in_progress |
| C | 21895847039 | fouls_o285, cards_o65, corners_o115 | exploration | in_progress |

**Parameters:**

| Job | odds_threshold | decay | adversarial | n_trials | Hypothesis |
|-----|---------------|-------|-------------|----------|------------|
| A | YES (α=0.2) | default | 2,10,0.75 | 150 | #1 lever for 2-way markets, NEVER tested on btts/under25 |
| B | no | 0.005 | 2,10,0.75 | 150 | decay helped all base niche, never tested on these line variants |
| C | no | default | 2,10,0.75 | 150 | fouls family 2/2 lines work, explore 3 new lines |

All: only_if_better=false, save_models=true, upload_results=true.

### Session 9 Re-Runs — catboost_merge (launched ~06:55 UTC)

| Job | Run ID | Markets | Config | Status |
|-----|--------|---------|--------|--------|
| 1v2 | 21895626347 | home_win, over25 | catboost_merge + aggressive adversarial (5,15,0.65) + decay=0.005, 75 trials | in_progress |
| 2v2 | 21895649762 | corners_o85, cards_o35, shots_o225 | catboost_merge + standard adversarial (2,10,0.75), 150 trials | in_progress |

---

## Baselines to Beat (Session 9b)

| Market | Current WF ROI | Current HO ROI | Source |
|--------|---------------|----------------|--------|
| btts | +60.4% | +25% (4 bets) | S7 Job D |
| under25 | +0.7% | — | DISABLED |
| fouls_o225 | +116.6% | +118.8% (56 bets) | S9 Job 5 |
| shots_o265 | +83.8% | +93.5% (26 bets) | S8 Job 3 |
| shots_o285 | +87.5% | +87.5% (8 bets) | S7 Job C |
| fouls_o285 | — | — | NEW (first test) |
| cards_o65 | — | — | NEW (first test) |
| corners_o115 | — | — | NEW (first test) |

---

## Current Deployment (14 markets on HF Hub)

| Market | Strategy | WF ROI | WF Bets | HO ROI | HO Bets | Confidence |
|--------|----------|--------|---------|--------|---------|------------|
| **home_win** | stacking | +119.9% | 135 | +113.5% | 48 | HIGH |
| **over25** | lightgbm | +93.2% | 119 | +100.0% | 40 | HIGH |
| **corners_o85** | agreement | +59.6% | 6371 | +57.0% | 1207 | HIGH |
| **shots_o225** | catboost | +119.1% | 307 | +102.8% | 159 | HIGH |
| **cards_o35** | agreement | +81.9% | 2569 | +66.1% | 462 | HIGH |
| **fouls_o265** | lightgbm | +120.4% | — | +87.5% | 4 | MED (S9 deployed) |
| **fouls_o225** | temporal_blend | +116.6% | — | +118.8% | 56 | MED (S9 deployed) |
| **shots_o285** | disagree_bal | +87.5% | 15 | +87.5% | 8 | MED |
| **shots_o265** | — | +83.8% | — | +93.5% | 26 | MED |
| **corners** | lightgbm | +65.4% | 31 | +55.4% | 37 | MED |
| **fouls** | lightgbm | +110.9% | 104 | 150% | 1 | LOW |
| **shots** | stacking | +122.2% | 27 | — | 0 | LOW |
| **cards** | agreement | +81.8% | 22 | — | 0 | LOW |
| **btts** | stacking | +60.4% | 200 | +25% | 4 | LOW (needs improvement) |
| under25 | — | +0.7% | — | — | — | DISABLED |
| away_win | — | — | — | — | — | DISABLED |

---

## When Jobs Complete — Analysis Steps

1. Download artifacts: `gh run download <run_id> --repo kamil0920/bettip -D /tmp/jobs_s9b/<job>`
2. Parse sniper result JSONs, compare vs baselines above
3. Deploy winners: WF ROI > deployed AND WF Bets >= 30 AND HO ROI > 0%
4. Sync models + deployment config to HF Hub together

### Conditional Job D

IF catboost_merge (Jobs 1v2/2v2) improves results → launch:
```
Markets: btts, fouls, shots, cards
catboost_merge=true, odds_threshold=true, threshold_alpha=0.2
adversarial_filter=2,10,0.75, n_trials=150
```

---

## Confirmed Levers (what works)

| Lever | Evidence | Use For |
|-------|----------|---------|
| **Odds-threshold α=0.2** | home_win +113.5% HO, over25 +100% HO | 2-way markets |
| **Aggressive adversarial (5,15,0.65)** | Best H2H improvement | H2H only |
| **decay=0.005** | All 5 base niche improved in S7 Job D | Niche markets |
| **Agreement ensembles** | cards_o35 HO +66.1% (462 bets), corners_o85 HO +57% (1207 bets) | High-volume lines |
| **Standard adversarial (2,10,0.75)** | Baseline that works everywhere | All markets |

## Confirmed Failures (do NOT re-test)

| Lever | Evidence | Notes |
|-------|----------|-------|
| Aggressive adversarial for niche | S9 J3 — all 4 regressed | Only helps H2H |
| Auto-RFE for weight collapse | S9 J5 — 12/17 still fastai-dominated | RFECV didn't help |
| Feature optimize for H2H | S8 J2a, R89, R95 all worse | 3x confirmed |
| Isotonic for BTTS | S8 — no improvement | Stacking itself was broken |

---

## Code Fixes Still Needed

### Two-Stage Model Interface (low priority)
`TwoStageModel.predict_proba()` requires `odds` argument but `ModelLoader.predict()` doesn't pass it. Would enable 6/6 home_win models in stacking (currently 4/6). Low impact since home_win already +113.5% HO.

### Corners O8.5 Odds (medium priority)
The Odds API doesn't return corners O8.5 odds. Need alternative odds source. Without odds, corners_o85 model (+57% HO, 1207 bets) is unbettable via automation.

---

## Session History (Condensed)

### Session 9 (Feb 11) — catboost_merge + Systematic Tests

5 jobs launched testing catboost_merge, aggressive adversarial for niche, odds-threshold for lines, auto-RFE.

**Bug found:** catboost_merge Phase 2 SKIPPED due to YAML boolean bug (`== true` vs `== 'true'`). Fixed commit 2849715.

**Results:**

| Market | Job | WF ROI | HO ROI | vs Deployed | Verdict |
|--------|-----|--------|--------|-------------|---------|
| home_win | J1 | -7.6% | -100% | -127.5 WF | CATASTROPHIC |
| over25 | J1 | +11.3% | — | -81.9 WF | REGRESSION |
| btts | J3 | +51.8% | +25% | -8.6 WF | WORSE |
| cards | J3 | +19.9% | — | -61.9 WF | NO MODELS |
| corners | J3 | +53.8% | +41.4% | -11.6 WF | WORSE |
| fouls | J3 | +84.5% | — | -26.4 WF | WORSE |
| cards_o35 | J4 | +80.5% | +68.4% | -1.3 WF | ~SAME |
| corners_o85 | J4 | +56.7% | +59.1% | -2.9 WF | ~SAME |
| shots_o225 | J2 | +113.8% | +109.9% | -5.3 WF | MARGINAL |
| **fouls_o265** | **J4** | **+120.4%** | **+87.5%** | **+29.1 WF** | **DEPLOYED** |
| shots_o265 | J4 | +70.8% | +92.3% | -13.0 WF | WORSE |
| **fouls_o225** | **J5** | **+116.6%** | **+118.8%** | **NEW HO** | **DEPLOYED** |
| shots_o285 | J5 | +76.5% | +150% | -11.0 WF | UNRELIABLE |

### Session 7 (Feb 10) — Deployment Rebuild + Pipeline Fixes

- Analyzed 7 optimization jobs (A-G). 4 failed from HF Hub rate limiting, retried.
- Deployment config wiped by retry aggregate steps → rebuilt from scratch with 41 models across 12 markets.
- Fixed: RiskConfig kwarg (f7ef3ab), model discovery for line variants (6d64fcb), stacking sigmoid bug (cd2d072/8d4e0a5).
- Key finding: decay=0.005 helped all niche. Odds-threshold α=0.2 best for H2H. Feature optimize timed out (75 trials impossible in 5h55m).

### Session 5 (Feb 9) — Temporal Leakage Cleanup

- Replaced ALL R90-era deployed models with clean post-fix versions.
- Implemented Kelly criterion, vig removal, Monte Carlo stress test, Poisson GLM features, Bayesian shrinkage.

### Session 4 (Feb 9) — Model Quality Fixes

- Fixed referee feature bug, SHAP string parsing, deployment config KeyError, min odds→1.5.
- Added adversarial feature filtering, non-negative stacking weights, calibration validation.
- Launched 3 post-fix optimization jobs.

### Session 2-3 (Feb 8) — Data Quality + Multi-Line

- Fixed niche market data quality (column mismatch + expansion leagues → +129% data).
- Implemented 12 new line variants (cards/corners/shots/fouls over different lines).
- Deployed shots (R112) and btts (R112) models.

### Session 1 (Feb 8) — Multi-Line Support

- Added multi-line niche market support (12 new bet_types).
- Analyzed R109-R115 optimization runs.

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
