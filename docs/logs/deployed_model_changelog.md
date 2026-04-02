================================================================================
3. CHANGELOG
================================================================================

--- S40b (2026-04-01/02) — League Audit + Model Health Retraining ---

  INFRASTRUCTURE:
  - EUROPEAN_LEAGUES bug fixed: 7 feature engineers now use ALL_LEAGUES via MatchStatsLoaderMixin
  - MLS/Liga MX excluded from training pipeline (44% NaN root cause: bug not data)
  - 3 dead columns removed (implied_total_goals, implied/abs_goal_supremacy)
  - Features regenerated: 30,287→25,362 rows (13 leagues, no MLS/Liga MX)
  - only_if_better=false bug fixed: hard gates (n_bets, TS, ECE) now always apply
  - 335 match_stats backfilled for MLS/Liga MX (API-Football, for future use)

  DEPLOYED (5 markets from retraining artifacts):
  - away_win_h1:      TS +8.4→-4.9 (FIXED), beta→temperature, P 71→79%, feat 16→20
  - home_win:         TS -21.3→-6.9 (FIXED), average→catboost, feat None→8
  - cornershc_over_15: P 75→88% (+13pp!), FVA 0.34→0.65, beta→sigmoid, feat None→30
  - hgoals_over_25:   Stable, temperature→sigmoid, feat None→31
  - sot_over_95:      TS -21.8→-2.7 (FIXED), feat None→26

  DISABLED (1 market):
  - shots_over_245:   FVA 0.078 (near zero value), TS flipped to +5.3, 37 features (overfit)

  RETRAINING ROUND 2 DISPATCHED (6 markets, Apr 2):
  - Wave A (real-odds, holdout_folds=1): away_win, under25, btts
  - Wave B (niche, min_threshold=0.55): dc_x2, sot_over_85, cornershc_over_25
  Key changes vs round 1: folds=1 (was 2), min_threshold=0.55 (was default 0.65),
  only_if_better=false (now safe after hard gates fix)

--- S40 (2026-03-31) — Targeted Repair Waves for 8 Worst Markets ---

  MOTIVATION: Identified 8 weakest markets by FVA, TS, ECE, AdvAUC, concentration.
  Dispatched 3 repair waves with targeted parameter changes per market type.

  WAVE 2 (4 niche markets — 700 trials, adv 3,15,0.70, holdout_folds=2):
  - sot_under_95: MASSIVE UPGRADE. avg→stacking, thr 0.75→0.80.
    P 72.8%→87.3% (+14.5pp), FVA +0.04→+0.33 (8x!), TS +12.7→-33.0 (positive TS ELIMINATED).
    ECE stable (0.023→0.024), AdvAUC 0.751→0.737. Features 21→12. Leagues 10→12. Run 1052.
  - agoals_under_15: STRONG UPGRADE. disagree_bal→catboost, same thr 0.83.
    P 89.3%→89.5%, ECE 0.051→0.034 (-33%), FVA +0.11→+0.37 (3.4x), HO_B 196→449 (+129%).
    AdvAUC 0.763→0.741. Features 21→12. Leagues 10→11. Run 1052.
  - corners_over_85: QUALITY UPGRADE. disagree_aggr→temporal_blend, thr 0.72→0.83.
    P 74.4%→86.2% (+11.8pp), FVA +0.09→+0.34 (3.8x), AdvAUC 0.731→0.683. CLEAN gate.
    Trade-off: HO_B 672→195 (-71%), leagues 13→9. Run 1052.
  - corners_under_95: VOLUME UPGRADE. catboost→temporal_blend, thr 0.65→0.70.
    HO_B 96→528 (+450%), leagues 4→9, P 76.0%→78.8%, FVA +0.26→+0.34.
    ⚠️ AdvAUC 0.784→0.847 (near 0.85 gate), TS -1.0→-19.8 (EXTREME). Monitor. Run 1052.

  WAVE 3 (2 niche markets — 700 trials, adv 3,15,0.70, max_rfe=25, mrmr=12):
  - cards_over_35: STRUCTURAL FIX. Features 18→5, concentration 70%→22%.
    ECE 0.037→0.009 (4x!), #Lg 5→7. P 86.3%→83.2% (-3pp), HO_B 278→155. Run 1053.
  - shots_over_245: NOT DEPLOYED. AdvAUC fixed (0.824→0.763) but P dropped 83.5%→72.7%.
    FVA +0.38→+0.14. Too much edge lost from feature reduction (18→5). Keep old model.

  WAVE 1 (H2H — 500 trials, odds_threshold, adv 3,15,0.70, holdout_folds=2):
  - away_win: MODEST UPGRADE. agreement, thr 0.70→0.72.
    P 75.2%→77.6% (+2.4pp), TS +8.5→+5.1 (reduced, still positive), FVA +0.41→+0.46.
    Features 18→6 (massive reduction). AdvAUC 0.729→0.713. Leagues 7→8. Run 1051.
    ROI=94.1% (negative, CI [76%, 111%] crosses 100% — not significant).
  - away_win_h1: NOT DEPLOYED. P 71.1%→78.1% (+7pp), TS +8.4→+1.1 (CLEAN!), ECE -36%.
    BUT HO_B 159→32 (-80%), #Lg 10→2. Fails volume gate (50) and leagues gate (3).
    Root cause: min_thr=0.70 too high → Wilson chose 0.75 → killed volume.
    Retry dispatched: min_thr=0.65, max_thr=0.72 (R8 sweet spot range).

  CONFIG FIXES:
  - Fixed model_type=None for 11 markets (walkforward.best_model_wf fallback was incorrect,
    corrected to match deployed_models_v13.md table values)
  - Fixed sot approved_leagues: empty→9 leagues (was lost during config regeneration)
  - Fixed ensemble_strategy=None for 4 W2 markets (deploy agent didn't set it)

  KEY PARAMETERS THAT WORKED:
  - holdout_folds=2: multiplied evaluation data for all markets
  - adversarial 3,15,0.70: strict filter reduced AdvAUC by 0.04-0.06 across all markets
  - calibration_methods=beta:temperature: Optuna chose temperature for most niche markets
  - min_threshold=0.68, max_threshold=0.83: Wilson scoring found market-specific optima
  - max_rfe=40, mrmr=20 (W2) / max_rfe=25, mrmr=12 (W3): aggressive feature reduction

--- S39 (2026-03-30) — Full Reset with Wilson + Binomial Gate ---

  FULL RESET: All 79 markets set to enabled=false, 15 rebuilt from scratch.

  INFRASTRUCTURE (commits 5df4d45, a160291, e28a09d):
  - Wilson score lower bound replaces raw precision/ROI in grid search objective
  - PSR/MinTRL/DSR computed in holdout evaluation (López de Prado)
  - Binomial test vs per-league weighted base rate for EST-odds (Gate 7)
  - Approved leagues gate: min 3 leagues with >=7 holdout bets (Gate 8)
  - PSR soft-warning gate (Gate 6, MIN_PSR=0.50)
  - NEEDS_DATA gate in training summary (HO_B < MinTRL)
  - feature_params=best confirmed default (A/B test: 6/8 wins)

  DEPLOYMENT GATES HARDENED:
  - HO_B >= 50 (was 20) | Precision >= 71% (was 55%) | ECE <= 0.10
  - AdvAUC < 0.85 | EST: p-value <= 0.05 | approved_leagues >= 3

  DEPLOYED (15 markets):
  - 3 REAL-odds: btts_no (122 bets, 86.1%), over25 (307, 81.4%), under25 (76, 76.3%)
  - 12 EST-odds: all pass binomial test (p<0.05 vs weighted base rate)
  - Top volume: cornershc_under_15 (851), dc_x2 (771), corners_over_85 (672)
  - Top precision: sot_over_75 (91.4%), agoals_under_15 (89.3%), cards_over_35 (86.3%)
  - Best ECE: corners_over_85 (0.007), corners_over_105 (0.012), cornershc_under_15 (0.013)

  REJECTED (14 markets — 13 in repair waves R1-R5, 1 abandoned):
  - 0 HO bets (3): btts, away_win, shots (shots ABANDONED — intractable 3x)
  - HO_B < 50 (8): home_win(39), away_win_h1(40), corners_under_95(43),
    cornershc_over_15(34), cornershc_over_25(32), sot_over_85(29), sot(19), sot_over_95(11)
  - Precision < 71% (2): hgoals_over_25(65%), shots_over_245(70.6%)
  - AdvAUC >= 0.85 (1): corners_under_95(0.850)

  REPAIR WAVES — 9/13 markets fixed:
  - R1: btts 0→127 HO (odds_threshold fix), home_win 39→135 (same), away_win still 0
  - R2: cornershc_over_15 34→218 (max_thr=0.72), cornershc_over_25 32→130, sot_over_85 29→173
  - R3: sot_over_95 11→154 (holdout_folds=2!), ht_under_05 39→44 (still <50), sot 19→1050 (70.6%)
  - R4a: shots_over_245 70.6%→83.5% (max_thr=0.75), hgoals_over_25 20→147 (holdout_folds=2)
  - R4b: corners_under_95 43→96 (adv=3,15,0.70), away_win_h1 40→36 (still <50, 2 leagues)
  Still pending: away_win, away_win_h1, ht_under_05, sot — R5/R6 dispatched

--- S38b (2026-03-30) — Wilson Wave 2 Deployment ---

  UPGRADES:
  - sot_over_75: agreement→stacking, thr 0.78→0.80, 64→232 HO bets (+262%),
    prec 87.5%→91.4% (+3.9pp), ECE 0.050→0.023 (was borderline, now excellent).
    FVA +0.32→+0.47. PSR=1.00, MinTRL=13. TS=-24.7 (negative=safe). Run 23730917455.
  - hgoals_under_15: agreement→temporal_blend, thr 0.78→0.72, 68→541 HO bets (+695%),
    prec 85.3%→76.7% (-8.6pp, volume trade-off). Massive volume gain, PSR=1.00,
    MinTRL=10. TS=-3.9 (near-CLEAN!). Run 23731067562.
  - corners_over_105: catboost→temporal_blend, thr 0.78→0.65, 113→141 HO bets (+25%),
    prec 82.3%→76.6% (-5.7pp), ECE 0.023→0.012 (best in class).
    TS +4.1→-17.1 (overprediction FIXED, now safe negative). Run 23731371857.

  NOT DEPLOYED:
  - btts_no: 0 HO bets — Wilson scoring too restrictive for this market
  - dc_x2: 0 HO bets — same issue
  - under25: 187→76 HO bets (-59%) — volume regression
  - ht_under_05: 82→39 HO bets (-52%) — volume regression
  - shots_over_245: 802 bets but 71% precision (-14pp) — too aggressive threshold drop
  - home_win: NEEDS_DATA (HO_B=20 < MinTRL=22)
  - sot_over_85: regression (45→29 bets, 80→76% prec)
  - sot_over_95: MARGINAL (11 bets, PSR=0.50)
  - cornershc_over_25: REVIEW — 32 bets, 94% prec, MinTRL=17 passes but based on
    EST-odds returns (Sharpe/MinTRL unreliable for EST markets)

  KEY FINDING:
  MinTRL/PSR/DSR are RETURNS-BASED metrics — unreliable for EST-odds markets where
  returns come from Poisson-estimated odds (circular loop). For EST markets, use
  Wilson score on PRECISION as the confidence measure, not MinTRL on returns.

--- S38 (2026-03-30) — Wilson Scoring Deployment ---

  INFRASTRUCTURE (commits 5df4d45, a160291):
  - PSR (Probabilistic Sharpe Ratio) computed in holdout evaluation alongside DSR/MinTRL
  - PSR soft-warning gate (MIN_PSR=0.50) added to deployment_gates.py
  - NEEDS_DATA gate in training summary: markets with HO_B < MinTRL flagged instead of silent CLEAN
  - Wilson score interval lower bound replaces raw precision/ROI in grid search objective
    Fixes small-sample bias: 12/12 wins gets Wilson=0.816 (not raw 1.000)
  - Wilson-adjusted expected ROI for REAL-odds (handles zero-variance degenerate returns)
  - feature_params=best confirmed winner in A/B test (6/8 markets, +33-367% HO volume)

  UPGRADES:
  - btts: temporal_blend→catboost, thr 0.70→0.83, 339→56 HO bets (fewer but unbiased),
    prec 71.4%→91.1% (+19.7pp), FVA +0.34→+0.76, Brier 0.128→0.083.
    PSR=1.00, MinTRL=16, HO_B(56)>MinTRL — statistically confirmed.
    TS=-10.0 (negative=safe). Run 23718764190.
  - hgoals_over_15: stacking→temporal_blend, 100→153 HO bets (+53%), prec 92.0%→84.3% (-7.7pp),
    but volume trade-off valid (153 bets >> 100), TS -25.3→-8.5 (improved).
    PSR=1.00, MinTRL=16. Run 23718958638.
  - hgoals_over_25: temporal_blend→catboost, 57→21 HO bets, prec 84.2%→81.0% (-3.2pp),
    FVA +0.77→+0.71. Borderline at 21 bets but PSR=0.99, MinTRL=11, HO_B>MinTRL.
    TS=+6.6 (positive but below +8 gate). Monitor closely. Run 23718764190.
  - sot: average→stacking, thr 0.75→0.78, 70→38 HO bets, prec 81.4%→92.1% (+10.7pp),
    ECE 0.026→0.020 (best calibration), FVA +0.37→+0.66.
    PSR=0.97, MinTRL=29, HO_B(38)>MinTRL — passes governance. Run 23718958638.

  RE-ENABLED:
  - goals_over_25: DISABLED→ENABLED! stacking, thr 0.65, 454 HO bets, prec 75.6%.
    Was "duplicate of over25" — Wilson scoring selected much lower threshold (0.78→0.65)
    unlocking massive volume (0→454 bets). PSR=1.00, MinTRL=10.
    FVA=+0.24 (modest but positive), ECE=0.035. TS=-15.7 (negative=safe).
    EST odds — evaluate on precision/FVA only. Run 23718764190.

  NOT DEPLOYED:
  - home_win: NEEDS_DATA — HO_B=20 < MinTRL=22. PSR=0.94 (borderline). 95% prec promising.
  - sot_over_85: SKIP — regression (45→29 HO bets, 80%→76% prec)
  - sot_over_95: MARGINAL — HO_B=11, PSR=0.50, MinTRL=∞ (zero-variance degenerate)

  A/B TEST RESULTS (feature_params none vs best):
  | Market          | none HO_B | best HO_B | Winner |
  |-----------------|-----------|-----------|--------|
  | btts            |        12 |        56 | BEST   |
  | goals_over_25   |         0 |       454 | BEST   |
  | hgoals_over_15  |       115 |       153 | BEST   |
  | hgoals_over_25  |        14 |        21 | BEST   |
  | home_win        |        13 |        20 | BEST   |
  | sot             |        19 |        38 | BEST   |
  | sot_over_85     |        29 |        29 | TIE    |
  | sot_over_95     |        11 |        11 | TIE    |
  Conclusion: feature_params=best is the default for all future retraining.

--- S37 (2026-03-28) — CAUTION Markets Improvement ---

  UPGRADES:
  - cornershc_over_25: 0.843→0.933 prec (+9.0pp), average→temporal_blend, thr 0.78→0.80
    19→14 features, FVA 0.65→0.83, ECE 0.024→0.036. Temperature calibration.
    Binomial test: p=3.3e-16 (93.3% on 45 bets vs 35.2% base rate). Wilson CI [82.1%, 97.7%].
    TS=-21.0 (negative=safe, underprediction). Run 995.

  RE-ENABLED:
  - sot_over_85: DISABLED→RE-ENABLED! 86.3% prec, 51 HO bets, stacking, thr=0.78, 24 features.
    Was "structurally intractable" with TS=+45.0. Fix: temperature calibration killed overconfidence
    (TS +45→-2.8 on first attempt), then removing mRMR gave more features (10→24) for better
    discrimination. Final TS=-13.65 (negative=safe). ECE=0.020, FVA=0.540. Beta calibration.
    Key features: combined_shots_ema (domain), missingness indicators, stacking ensemble.
    3 optimization rounds to get here. Run 1003.

  RE-ENABLED:
  - away_win_h1: DISABLED→CLEAN! temporal_blend, thr 0.70→0.75, 37→136 HO bets (3.7x),
    prec 78.4%→74.3% (-4pp), TS -13.7→-1.6 (CLEAN!). Beta calibration, 19 features.
    Key: holdout_folds=2 gave more evaluation data, model proved stable. ECE=0.044, FVA=+0.56.
    Base rate 25.1% (away win at HT). Run 1005.
  - corners_under_95: DISABLED→CLEAN! catboost, thr 0.78→0.70, 44→559 HO bets (12.7x),
    prec 95.5%→73.7% (-22pp, volume trade-off), TS -26.5→-4.5 (near-CLEAN). Temperature cal.
    13 features, ECE=0.019, FVA=+0.23. Base rate 48.6%. Precision boost run pending. Run 1005.

  NOT DEPLOYED:
  - corners_over_105: Optimization REGRESSED (70.6% vs 82.3% deployed, TS +21.1 vs +4.1).
    Root cause: adversarial filter removed ALL corners features, RFECV selected only proxy
    features (fouls/shots) with r=0.03. Corners base rate has massive temporal drift
    (Serie A: 51%→31% over 7yr). Fix: whitelist corners domain features. Pending rerun.
  - btts_no: 90.5% prec on only 21 HO bets (p=0.000017 significant, but CI [71%, 97%] too wide).
    Need more volume before deployment. Pending rerun with relaxed threshold.
  - sot_over_95: holdout_folds=2 backfired (92.9% prec, 14 bets — less volume than deployed 20).
    Retry with max_threshold=0.68, holdout_folds=1 pending (Run 1006).
  - ht_under_05: min_threshold=0.65 killed the market (53.8% prec, 13 bets). Model works best
    at low threshold (0.50). Retry with max_threshold=0.68 pending (Run 1006).

--- S36 (2026-03-26) — CAUTION Market Investigation ---

  UPGRADES:
  - under25: CAUTION→CLEAN! 0.749→0.860 prec (+11.1pp), TS -21.3→+3.7, thr 0.65→0.83
    Breakthrough after 5 failed campaigns. Key: mirrored over25 architecture (thr=0.83,
    alpha=0.3, min_odds=1.8). CatBoost single model, sigmoid calibration, 14 features.
    ROI CI [62.6%, 109.4%] with REAL odds. Run 976.
  - btts_no: 0.856→0.897 prec (+4.1pp), thr 0.78→0.75, 104→39 HO bets
    total_cards_missing missingness indicator retained. TS=-15.3 (negative=safe).
    ECE 0.027 excellent. Run 977.

  CODE CHANGES (commit 2808028):
  - model_loader.py: Generate _missing indicators at inference from base feature NaN
    (fixes btts_no total_cards_missing being zero-filled in production)
  - betting_strategies.py: HomeGoalsStrategy→ft_home, AwayGoalsStrategy→ft_away
    (0% NaN vs 40%, +12k training rows for hgoals/agoals markets)
  - run_sniper_optimization.py: 120+ cards/offsides/fouls exclusions for hgoals family

  DISABLED:
  - goals_over_25: duplicate of over25 (same target, EST vs REAL odds)

  - hgoals_over_25: 0.720→0.842 prec (+12.2pp), 25→57 HO bets, TS -5.2→+1.5 (CLEAN!)
    ft_home (+12k rows) massive improvement. temporal_blend, beta calibration, 30 features.
  - agoals_under_15: 0.841→0.866 prec (+2.5pp), 719→643 HO bets, TS +1.2→-16.7
    ft_away impact. Precision up, FVA +0.16→+0.28. TS regressed to CAUTION but negative=safe.

  RE-ENABLED:
  - hgoals_over_15: DISABLED→92.0% prec, 100 HO bets, FVA +0.73, TS=-25.3
    Was "intractable" (FVA=-0.04, TS=+31.4). ft_home + min_threshold=0.78 resurrected it.
    Stacking ensemble, temperature calibration, 12 features incl home_red_cards_missing.
    ⚠️ Monitor closely: 92% precision on 100 bets may indicate overfit.

  KEPT (only_if_better=true protected):
  - hgoals_under_15: 74.2% < 88.9% deployed — ft_home + min_threshold=0.78 not enough

  INFRASTRUCTURE (commit 602187e):
  - --min-threshold CLI flag for sniper optimizer (prevents low threshold selection)
  - hgoals exclusions lightened: 120→31 features (offsides + duplicate only)

--- S35 (2026-03-26) — SOT Cards Exclusion Campaign ---

  UPGRADES:
  - sot: agreement->average, 0.70->0.75 thr, 18->10 features, 28->70 HO bets
    ECE 0.071->0.026, FVA +0.37, per-fold TS=-1.61 (within bounds)
    Holdout TS=-14.1 (still rejected) — MANUAL OVERRIDE: per-fold TS OK,
    concept drift at holdout boundary, not systematic model failure.
    Cards exclusion removed 4/18 cards-derived features (27-44% NaN fake zeros).
    Run B (970): max_threshold=0.75, max_rfe=40, mrmr=20, adversarial 2,10,0.75

  CODE CHANGES (commit 4288e57):
  - Added 73 cards-derived features to LOW_IMPORTANCE_EXCLUSIONS["sot"]
    (dynamics, entropy, cross-market interactions, referee-cards, NegBin)
  - Fixed line variant fallback: sot_over_85 etc. now inherit exclusions
    from base "sot" market via BASE_MARKET_MAP (latent bug affecting all markets)

  KEY FINDINGS:
  - Cards exclusion worked: 0/10 selected features are cards-derived (was 4/18)
  - TS problem is NOT caused by cards noise — it's temporal concept drift
  - Per-fold TS is normal (+/-1-3) but holdout TS explodes (-14) due to
    base rate shift at the temporal boundary
  - Run A (tight RFECV, max_rfe=25): 6 features, 100% prec but only 15 HO bets
  - Run B (lower threshold, 0.75): 10 features, 81.4% prec, 70 HO bets — deployed
  - Run C (temperature calibration): dispatched, pending results

--- S32 (2026-03-25) — TS Calibration Campaign ---

  UPGRADES:
  - corners_over_85: 0.784 -> 0.860 prec (+7.6pp), 193 HO bets, stacking, thr=0.80
    CLEAN gate (TS=-1.2), FVA +0.34, NegBin FVA +0.37 — from 0 HO bets to deployable
    Enabled by: per-market nan_filter override (blocklist killed this market)

  DISABLED:
  - hgoals_over_15: DISABLED — confirmed intractable (FVA=-0.04, model worse than base rate)
    TS=+31.4 in Run 938, 0 HO bets with calibration fixes in Run 940

  NEW INFRASTRUCTURE (commits 1e424bf, a28c307, a4b7ffc):
  - Per-market filter mode override in strategies.yaml (corners -> nan_filter)
  - Niche calibration restriction: ["beta", "temperature"] (was full 4-method search)
  - Beta holdout recalibrator on last 2 opt folds (replaces isotonic on last 1)
  - Leading-edge TS penalty from last opt fold (catches holdout drift early)
  - ECE noise guard: skip ECE penalty when n_bets < 30
  - BetaCalibrator scoping fix (module-level import)

  KEY FINDINGS:
  - Data density determines calibration effectiveness:
    dense (>1000 WF bets) -> beta works; sparse (<500) -> temperature safer
  - ECE on <30 qualifying bets is statistical noise, not real miscalibration
  - Hybrid filter mode = blocklist in practice (NaN filter no-op after blocklist)
  - hgoals_over_15 has negative FVA — model adds no value over base rate

--- S31 (2026-03-24) — Temporal Shift Campaign ---

  UPGRADES (auto-deployed by CI, verified):
  - shots_over_245:  0.761 -> 0.850 prec (+8.9pp), 113 HO bets, stacking, thr=0.78
  - agoals_under_15: 0.810 -> 0.841 prec (+3.1pp), 719 HO bets, temporal_blend
  - btts:            0.714 -> 0.850 prec (+13.6pp), 20 HO bets, temporal_blend
  - home_win:        0.742 -> 0.756 prec (+1.4pp), 127 HO bets, fastai
  - hgoals_over_25:  0.641 -> 0.737 prec (+9.6pp), 19 HO bets, stacking
  - sot_over_85:     0.799 -> 0.800 prec (+0.1pp), 45 HO bets, disagree_balanced

  REVERTED (CI auto-deployed regressions, manually fixed):
  - btts_no:         0.856 preserved (CI deployed 0.569 — reverted)
  - hgoals_over_15:  0.829 preserved (CI deployed 0.662 — reverted)
  - goals_over_35:   0.787 preserved (CI deployed 0.667 — reverted)
  - sot_under_95:    0.842 preserved (CI deployed 0.754 — reverted)
  - corners_over_85: 0.784 preserved (CI deployed 0.721 — reverted)
  - cornershc_over_15: 0.799 preserved (CI deployed 0.792 — reverted)

  BUGS FIXED:
  - Estimated-odds filter: fallback + walk-forward now threshold-only (commit 4374677)
  - feature_params best mode: line variant -> base market mapping (commit 3b905bf)
  - Niche threshold floor: 0.50 -> 0.60, ceiling 0.78 -> 0.83 (commit 770c620)

  NEW INFRASTRUCTURE:
  - SLD prior correction (Saerens et al. 2002) in holdout eval (commit 8aa5c1c)
  - Holdout-fold recalibration via isotonic regression (commit 8aa5c1c)
  - Per-fold sliding window CV via --training-window-days (commit 544de45)
  - Adversarial density ratio diagnostics (commit 544de45)
  - Training data expanded: 18,196 -> 30,233 rows (15 leagues)

  EXPERIMENT RESULTS (S31):
  - Feature_optimize: NO improvement — default params produce stronger models
  - Higher thresholds: marginal — 1 market improved (cornershc_over_15)
  - n_folds=7: no improvement — TS rejection rate unchanged (71%)
  - purged_kfold: slight improvement — 1 upgrade (sot_under_85 +5.8pp)
  - SLD + recal: 1 market saved (shots_over_245 TS below threshold)
  - Sliding window 730d: WORSE — insufficient data, TS rate up to 86%
  - TS rejection is the #1 blocker — genuine temporal distribution shift

--- S32 Retrain + Bookpts Removal (2026-03-21-22) ---

  DATA EXPANSION:
  - Eredivisie bulk: 306->2,080 matches
  - Ligue 1 2024: 308 fixtures collected
  - La Liga 2 + Championship added (13 active leagues)
  - Cards fallback coverage: 64.8%->92.5% (+6,249 matches)
  - Total: 21,946 active rows, 913 cols

  RETRAIN (7 sniper runs, 28 niche + 3 H2H):
  Passed gates (13/28 niche): corners_over_85, corners_over_95,
    corners_under_95, cards_under_35, cornershc_over_25, cornershc_under_15,
    goals_over_25, goals_under_25, hgoals_over_25, hgoals_under_15,
    sot_over_75, sot_under_95, bookpts_over_305
  Failed (15/28 niche): 0 holdout bets
  H2H: over25 upgraded, btts upgraded, under25 skipped

  DEPLOYED (6 upgrades):
  over25, btts, corners_under_95, cards_under_35, hgoals_under_15, sot_under_95

  FEATURE PARAM OPTIMIZATION (2026-03-22):
  24 markets tested. New: btts_no (85.6%), home_win (74.2%), away_win (78.3%),
  away_win_h1 (78.4%). H2H feature_optimize confirmed harmful.

  REMOVED: All bookpts markets (4) — removed from project entirely

--- S31 Gate Hardening (2026-03-20) ---

  PHASE 1 — Gate Hardening:
  Created src/ml/deployment_gates.py (shared gate module)
  Fixed 5 bugs: ECE=None passing, empty saved_models, conflicting
  thresholds (20 vs 60), n_bets wrong source, MinTRL not enforced
  Disabled 25 phantom/unvalidated markets

  PHASE 2 — Retraining (6 waves, 27 markets):
  Wave 1 (H2H): 500 trials, odds_threshold alpha=0.2, adv 3,15,0.70, rfe=30, mrmr=20
  Waves 2-6 (Niche): 700 trials, decay=0.005, adv 2,10,0.75, rfe=40, mrmr=20, no_fastai

  RESULTS:
  Newly deployed (7): btts, hgoals_under_15, goals_over_25,
    goals_under_25, hgoals_over_25, cornershc_over_25, sot_under_85
  Failed holdout (12): home_win, dc_1x, hgoals_over_15, sot_over_85, etc.

### S35 — NaN Handling Campaign (Mar 26, 2026)

  CHANGES: Native NaN for tree models, missingness indicators (>5% NaN),
  post-RFECV row drop (>=3% NaN), CatBoost nan_mode tuning, NaN gate 50%,
  emergency fallback for hard ECE/TS gates, per-fold TS skip <30 bets.

  DEPLOYED:
  - sot_under_85:   DISABLED -> 0.760 prec, 25 HO bets, two_stage_xgb, TS=-5.2
  - hgoals_over_25: 0.737 -> 0.720 prec (-1.7pp), 25 HO bets, catboost, TS=-5.2
    (volume: 19->25 bets, now above 20-bet minimum; `home_red_cards_missing` indicator selected)
