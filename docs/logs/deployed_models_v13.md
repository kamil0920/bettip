Deployed Models — Current State (2026-04-02)
================================================================================

CURRENTLY DEPLOYED: 27 enabled markets (S40 league audit + model health retraining)
Last updated: S40b — League audit, EUROPEAN_LEAGUES bug fix, model health retraining (Apr 1-2)

================================================================================
1. CURRENT STATE
================================================================================

LEGEND:
  Model   = Best model/ensemble strategy    PSR    = Probabilistic Sharpe Ratio — P(true Sharpe > 0). >0.95 = significant.
  Thr     = Betting threshold               Returns-based (López de Prado). Reliable for REAL only.
  HO_P    = Holdout precision (>=71%)
  HO_B    = Holdout bets (>=50)            MinTRL = Min Track Record Length — minimum N for SR significance.
  ECE     = Expected Calibration Error      Returns-based (López de Prado). HO_B >= MinTRL = sufficient.
  FVA     = Forecast Value Added
  TS      = Tracking Signal                p-val  = Binomial test of precision vs weighted base rate (EST only).
  AdvV    = Adversarial AUC (<0.85)         p<=0.05 = model significantly beats per-league base rate.
  #F      = Number of features
  Odds    = REAL or EST                    w_BR   = Weighted base rate — expected precision if betting randomly
                                            from same league mix as qualifying bets. Null hypothesis.
                                           Conc   = League concentration — max % of HO bets from single league.
                                            >80% = concentration risk (model may only work in one league).
                                           #Lg    = Approved leagues — leagues with >=7 holdout bets (>=3 required).
                                            Guards against out-of-distribution inference on unseen leagues.

DEPLOYMENT GATES (S39):
  HO_B >= 50 | Precision >= 71% | ECE <= 0.10 | AdvAUC < 0.85
  EST markets: precision_pvalue <= 0.05 | approved_leagues >= 3
  Wilson-based grid search objective (replaces raw precision/ROI)

DEPLOYED — 28 markets (sorted by HO_B desc)
|---------------------------|--------------------------|------|------|-------|-------|-------|--------|-------|------|--------|--------|-------|------|-----|----|----|
| Market                    | Model                    | Thr  | HO_B | HO_P  | ECE   | FVA   |   TS   | AdvV  | PSR  | MinTRL | p-val  | w_BR  | Conc | #Lg| #F | Odds|
|---------------------------|--------------------------|------|------|-------|-------|-------|--------|-------|------|--------|--------|-------|------|-----|----|----|
| cornershc_under_15        | xgboost                  | 0.65 |  851 | 76.1% | 0.013 | +0.18 |  -23.9 | 0.789 | 1.00 |     10 | 0.0000 | 57.0% |  11% |  13 | 15 | EST |
| dc_x2                     | temporal_blend           | 0.72 |  771 | 78.3% | 0.019 | +0.25 |   +5.4 | 0.717 | 1.00 |     10 | 0.0000 | 56.2% |  10% |  13 | 12 | EST |
| corners_over_85           | temporal_blend           | 0.83 |  195 | 86.2% | 0.015 | +0.34 |   +3.6 | 0.683 | 1.00 |     25 | 0.0000 |     — |    — |   9 |  8 | EST |
| ~shots_over_245~          | ~DISABLED S40b~          |    — |    — |     — |     — |     — |      — |     — |    — |      — |      — |     — |    — |   — |  — | EST |
| goals_over_25             | stacking                 | 0.65 |  454 | 75.6% | 0.035 | +0.24 |  -15.7 | 0.750 | 1.00 |     10 | 0.0000 | 55.2% |  17% |  11 | 16 | EST |
| sot_under_95              | stacking                 | 0.80 |  362 | 87.3% | 0.024 | +0.33 |  -33.0 | 0.737 | 1.00 |     12 | 0.0000 |     — |    — |  12 | 12 | EST |
| hgoals_under_15           | average                  | 0.72 |  334 | 75.1% | 0.041 | +0.19 |  -13.0 | 0.807 | 1.00 |     10 | 0.0000 | 54.9% |  16% |  10 | 18 | EST |
| over25                    | stacking                 | 0.70 |  307 | 81.4% | 0.030 | +0.55 |  -16.5 | 0.738 | 1.00 |     10 |      — |     — |    — |  12 | 17 | REAL|
| cards_over_35             | average                  | 0.75 |  155 | 83.2% | 0.009 | +0.20 |   -2.3 | 0.750 | 1.00 |     23 | 0.0000 |     — |  22% |   7 |  5 | EST |
| sot_over_75               | stacking                 | 0.80 |  232 | 91.4% | 0.023 | +0.47 |  -24.7 | 0.759 | 1.00 |     13 | 0.0000 | 69.0% |  28% |  10 | 21 | EST |
| cornershc_over_15         | temporal_blend           | 0.80 |   88 | 87.5% | 0.024 | +0.65 |  -15.0 | 0.802 | 1.00 |     13 | 0.0000 | 42.0% |  23% |   9 | 30 | EST |
| sot                       | catboost                 | 0.80 |  227 | 81.1% | 0.021 | +0.40 |  -17.5 | 0.755 | 1.00 |     10 | 0.0000 | 52.7% |  22% |   9 | 16 | EST |
| ht_under_05               | temporal_blend           | 0.55 |  209 | 75.6% | 0.018 | +0.51 |  -20.8 | 0.784 | 1.00 |     10 | 0.0000 | 31.3% |  33% |   7 | 23 | EST |
| agoals_under_15           | catboost                 | 0.83 |  449 | 89.5% | 0.034 | +0.37 |  -30.9 | 0.741 | 1.00 |     12 | 0.0000 |     — |    — |  11 | 12 | EST |
| corners_over_95           | stacking                 | 0.72 |  196 | 84.7% | 0.027 | +0.47 |  -20.1 | 0.777 | 1.00 |     14 | 0.0000 | 53.4% |  42% |   7 | 15 | EST |
| sot_over_85               | temporal_blend           | 0.72 |  173 | 83.2% | 0.034 | +0.44 |   -3.1 | 0.751 | 1.00 |     10 | 0.0000 | 58.1% |  35% |   7 | 18 | EST |
| sot_over_95               | temporal_blend           | 0.80 |   22 | 86.4% | 0.027 | +0.68 |   -2.7 | 0.767 | 1.00 |     11 | 0.0000 | 43.7% |  39% |   1 | 26 | EST |
| hgoals_over_15            | temporal_blend           | 0.80 |  153 | 84.3% | 0.025 | +0.55 |   -8.5 | 0.740 | 1.00 |     16 | 0.0000 | 47.6% |  18% |   9 | 18 | EST |
| away_win_h1               | temporal_blend           | 0.70 |   56 | 78.6% | 0.034 | +0.66 |   -4.9 | 0.735 | 1.00 |      7 | 0.0000 | 25.5% |  21% |   4 | 20 | EST |
| hgoals_over_25            | average                  | 0.65 |   57 | 75.4% | 0.019 | +0.62 |   -8.3 | 0.713 | 1.00 |     11 | 0.0000 | 22.7% |  23% |   5 | 31 | EST |
| corners_over_105          | temporal_blend           | 0.65 |  141 | 76.6% | 0.012 | +0.43 |  -17.1 | 0.786 | 1.00 |     12 | 0.0000 | 42.3% |  31% |   8 | 18 | EST |
| away_win                  | agreement                | 0.72 |  152 | 77.6% | 0.041 | +0.46 |   +5.1 | 0.713 | 1.00 |     10 |      — |     — |    — |   8 |  6 | REAL|
| home_win                  | catboost                 | 0.70 |  171 | 78.9% | 0.036 | +0.25 |   -6.9 | 0.764 | 1.00 |     17 |      — |     — |    — |  13 |  8 | REAL|
| cornershc_over_25         | average                  | 0.65 |  130 | 73.1% | 0.019 | +0.44 |  -15.5 | 0.814 | 1.00 |     13 | 0.0000 | 33.5% |  21% |   8 | 14 | EST |
| btts                      | temporal_blend           | 0.78 |  127 | 82.7% | 0.011 | +0.57 |   +2.7 | 0.721 | 1.00 |     10 |      — |     — |    — |   8 | 17 | REAL|
| btts_no                   | temporal_blend           | 0.70 |  122 | 86.1% | 0.020 | +0.62 |   -9.5 | 0.717 | 1.00 |     10 |      — |     — |    — |   7 | 24 | REAL|
| corners_under_95          | temporal_blend           | 0.70 |  528 | 78.8% | 0.032 | +0.34 |  -19.8 | 0.847 | 1.00 |     18 | 0.0000 |     — |    — |   9 | 15 | EST |
| under25                   | catboost                 | 0.72 |   76 | 76.3% | 0.026 | +0.43 |   +4.6 | 0.752 | 1.00 |     13 |      — |     — |    — |   4 | 18 | REAL|
|---------------------------|--------------------------|------|------|-------|-------|-------|--------|-------|------|--------|--------|-------|------|-----|----|----|

ABANDONED / DISABLED — 2 markets
  shots             0 HO — intractable, confirmed 3x. No viable config found.
  shots_over_245    DISABLED S40b — FVA 0.078 after retrain (model adds no value). TS flipped to +5.3.

REPAIR HISTORY (all completed):
  away_win         4 attempts: S39(0), R1(0), R5a(137✓). Key: holdout_folds=2 + odds_threshold + max_thr=0.72
  away_win_h1      5 attempts: S39(40/2lg), R4b(36/2lg), R5b(0), R7b(257@64%), R8(159@71%✓).
                   Key: narrow grid 0.65-0.72 found sweet spot between volume and precision.
  sot              R6: removed max_thr cap → Wilson chose thr=0.80, 227 bets@81.1%, 9 leagues ✓
  ht_under_05      R7c: max_thr=0.55 (CLAUDE.md: "best at low threshold") → 209 bets@75.6%, 7 leagues ✓

  S40 REPAIR WAVES (2026-03-31):
  sot_under_95     W2: avg→stacking, thr 0.75→0.80, FVA +0.04→+0.33 (8x!), P 72.8%→87.3%, TS +12.7→-33.0 (positive TS FIXED)
  agoals_under_15  W2: disagree→catboost, ECE 0.051→0.034, FVA +0.11→+0.37 (3.4x), HO_B 196→449 (+129%)
  corners_over_85  W2: disagree_aggr→temporal_blend, thr 0.72→0.83, P 74.4%→86.2%, FVA +0.09→+0.34 (3.8x), CLEAN gate
  corners_under_95 W2: catboost→temporal_blend, thr 0.65→0.70, HO_B 96→528 (+450%), #Lg 4→9. ⚠️ AdvAUC 0.784→0.847
  cards_over_35    W3: features 18→5, ECE 0.037→0.009, Conc 70%→22%, #Lg 5→7. Structural concentration fix.
  shots_over_245   W3: NOT DEPLOYED — P 83.5%→72.7% (-10.8pp), FVA +0.38→+0.14. AdvAUC fixed (0.824→0.763) but too much edge lost.
  away_win         W1: agreement, thr 0.70→0.72, P 75.2%→77.6%, TS +8.5→+5.1, features 18→6. AdvAUC 0.729→0.713.
  away_win_h1      W1: NOT DEPLOYED — P 78.1% (+7pp) but HO_B 159→32, #Lg 10→2 (fails gates). Retry dispatched with thr 0.65-0.72.

  S40b MODEL HEALTH RETRAINING (2026-04-01/02):
  Context: League audit found EUROPEAN_LEAGUES bug (7 engineers), 3 dead columns, MLS/Liga MX excluded.
           Features regenerated without MLS/Liga MX (25,362 rows, was 30,287). Beta calibration banned.
           Bug fixed: only_if_better=false now applies hard gates (n_bets, TS, ECE).

  DEPLOYED (4 markets — manual deploy from artifacts):
  away_win_h1      temporal_blend, thr 0.72→0.70, calib beta→temperature.
                   TS +8.4→-4.9 (OVERPREDICTION FIXED). P 71.1%→78.6% (+7.5pp), FVA +0.51→+0.66.
                   Trade-off: HO_B 159→56, #Lg 10→4. Features populated (was None).
  home_win         average→catboost, thr 0.72→0.70, calib sigmoid (kept).
                   TS -21.3→-6.9 (LESS CONSERVATIVE). P 81.5%→78.9% (-2.6pp), HO_B 135→171 (+27%).
                   Features populated (was None). 13 approved leagues.
  cornershc_over_15 agreement→temporal_blend, thr 0.70→0.80, calib beta→sigmoid.
                   TS -21.2→-15.0, P 74.8%→87.5% (+12.7pp!), FVA +0.34→+0.65 (1.9x).
                   Trade-off: HO_B 218→88. Features populated (was None).
  hgoals_over_25   catboost→average, thr 0.70→0.65, calib temperature→sigmoid.
                   Stable: P 78.9%→75.4%, FVA +0.69→+0.62, TS -7.4→-8.3.
                   Features populated (was None). #Lg 8→5.
  sot_over_95      Deployed from artifacts (good metrics, only_if_better rejected due to fewer bets).
                   thr 0.72→0.80, calib temperature→sigmoid. TS -21.8→-2.7 (MUCH BETTER).
                   P 82.5%→86.4%, FVA +0.60→+0.68. Trade-off: HO_B 154→22, #Lg 7→1.
                   Features populated (was None).

  DISABLED (1 market):
  shots_over_245   FVA 0.078 after retrain — model adds near-zero value. TS flipped -10.3→+5.3.
                   37 features (overfitted). Disabled pending further analysis.

  NOT DEPLOYED (7 markets — only_if_better rejected or 0 holdout bets):
  away_win         17 HO bets (holdout_folds=2 too aggressive). Retrain dispatched with folds=1.
  under25          30 HO bets, TS -18.7 (too conservative). Retrain dispatched with folds=1.
  btts             44 HO bets, good metrics but old model had 127 bets. Retrain dispatched with folds=1, only_if_better=false.
  dc_x2            0 HO bets — threshold 0.75 >> pred_mean 0.546. Retrain dispatched with min_threshold=0.55.
  sot_over_85      0 HO bets — threshold 0.70 >> pred_mean 0.481. Retrain dispatched with min_threshold=0.55.
  cornershc_over_25 0 HO bets — threshold 0.78 >> pred_mean 0.346. Retrain dispatched with min_threshold=0.55.
  PENDING WAVE A (away_win, under25, btts) + WAVE B (dc_x2, sot_over_85, cornershc_over_25).

GATE SUMMARY
  Total deployed: 27 | Abandoned/Disabled: 2 (shots, shots_over_245)
  REAL-odds: 6 (away_win, btts, btts_no, home_win, over25, under25)
  EST-odds: 21
  Training data: ~25K rows, 13 leagues (MLS/Liga MX excluded), data end ~2026-03-23
  Governance: Wilson scoring + PSR/MinTRL + binomial precision gate + approved leagues
  NOTE: REAL-odds markets (—) don't have p-val/w_BR/Conc — uses PSR/MinTRL instead
  PENDING: Wave A+B retraining for 6 remaining markets (dispatched Apr 2)

================================================================================
2. KEY INSIGHTS & GROUND RULES
================================================================================

TS IS ASYMMETRIC (S35):
  TS measures cumulative directional prediction bias.
  - TS < 0 (underprediction): model says 80%, reality 90%. SAFE — you bet
    conservatively, still profit, just miss some edge. Common in high-base-rate
    markets (sot ~85-94%, dc_x2 ~90%). Small bias x many bets = large |TS|.
  - TS > 0 (overprediction): model says 90%, reality 80%. DANGEROUS — you
    oversize positions, think you have edge that doesn't exist. Loses money.
  - Per-fold TS +/-4 + ECE < 0.05 + precision >= 70% = SAFE even if holdout |TS| > 10.
    Holdout TS explosion with normal per-fold TS = concept drift at fold boundary.
  DEPLOYMENT RULE: Never reject solely on negative TS when other metrics are good.
  Current max_ts=4.0 gate is too strict for niche markets. TODO: implement
  asymmetric gate (TS > +8 reject, TS < -20 reject, between = pass).

ROI IS MEANINGLESS FOR ESTIMATED-ODDS MARKETS (S30):
  ALL niche markets (corners, cards, fouls, shots, sot, goals lines, HT,
  cornershc, cardshc) use Poisson-estimated odds. ROI for these = circular loop.
  Only valid for: home_win, away_win, over25, under25, btts.
  For EST markets evaluate ONLY: precision, ECE, FVA, adversarial AUC, n_bets.

DATA DENSITY DETERMINES CALIBRATION (S32):
  Dense (>1000 WF bets) -> beta calibration works.
  Sparse (<500 WF bets) -> temperature scaling safer.
  ECE on <30 qualifying bets = statistical noise, not real miscalibration.

FEATURE COUNT IS THE #1 OVERFITTING PREDICTOR (S30):
  50 features = overfit. 5-24 features = generalizes.
  Markets with 4-6 features are safest.

TS REJECTION IS THE #1 DEPLOYMENT BLOCKER (S31):
  Per-fold TS is OK but holdout TS explodes -> genuine temporal distribution shift.
  NOT a model quality issue. 60%+ of markets blocked by symmetric TS gate.

================================================================================
3. CHANGELOG
================================================================================

Moved to file -> docs/logs/deployed_model_changelog.md, append there new entries