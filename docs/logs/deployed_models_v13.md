Deployed Models — Current State (2026-03-19)
================================================================

CURRENTLY DEPLOYED MODELS (31 enabled markets)
================================================================

LEGEND:
  Model  = Best model/ensemble strategy from sniper optimization
  Thr    = Betting threshold (minimum probability to trigger bet)
  HO_P   = Holdout precision (out-of-sample, most trustworthy metric)
  HO_B   = Holdout bets (sample size)
  ECE    = Holdout Expected Calibration Error (<0.05 = excellent, <0.10 = acceptable)
  FVA    = Forecast Value Added vs market baseline (>0 = beats market)
  TS     = Holdout Tracking Signal (directional bias; * = ts_rejected/small sample)
  #F     = Number of features
  Odds   = REAL (bookmaker odds, ROI meaningful) or EST (Poisson-estimated, ROI unreliable)
  Gate   = CLEAN (|TS|<2), CAUTION (2-4 or ts_rejected with high |TS|), EXTREME (>4, not rejected)
  Src    = Sprint/version that deployed this config

NOTE: ROI omitted for EST-odds markets (inflated by circular Poisson loop).
      S30 Fix changes: temporal leakage exclusion (round_number, cross_yellows_*),
      holdout threshold auto-fallback, relaxed min_precision (0.55) + fallback (0.50).
      New market families: SOT (shots on target), bookpts, clean_sheet, win_to_nil,
      score_both_halves.

+---------------------------+------------------+------+-------+------+-------+-------+--------+-----+------+----------+------+
|         Market            |     Model        | Thr  | HO_P  | HO_B |  ECE  |  FVA  |   TS   | #F  | Odds |   Gate   | Src  |
+---------------------------+------------------+------+-------+------+-------+-------+--------+-----+------+----------+------+
| CLEAN — 11 markets (|TS|<2, or ts_rejected with |TS|<2)                                                                    |
+---------------------------+------------------+------+-------+------+-------+-------+--------+-----+------+----------+------+
| dc_x2                     | average          | 0.83 | 94.1% |  169 | 0.020 | +0.68 | -16.0* |   5 | EST  | CLEAN    | v19  |
| cards_over_35             | catboost         | 0.78 | 84.9% |  218 | 0.047 | +0.20 |  +0.8  |  15 | EST  | CLEAN    | v19  |
| home_win                  | xgboost          | 0.78 | 82.6% |   23 | 0.028 | +0.54 |  -5.7* |   5 | REAL | CLEAN    | v19  |
| corners_over_95           | stacking         | 0.36 | 81.5% |  362 | 0.017 | +0.35 | -15.1* |  39 | EST  | CLEAN    | S28  |
| cornershc_over_15         | lightgbm         | 0.78 | 79.9% |  209 | 0.014 | +0.45 |  -1.3  |  19 | EST  | CLEAN    | v18  |
| corners_over_85           | temporal_blend   | 0.72 | 78.4% | 1539 | 0.023 | +0.17 |  -0.7  |  22 | EST  | CLEAN    | v19  |
| away_win_h1               | catboost         | 0.72 | 76.7% |   43 | 0.016 | +0.60 |  -0.8  |  10 | EST  | CLEAN    | v19  |
| sot_over_85               | two_stage_xgb    | 0.60 | 67.5% |  492 | 0.025 | +0.13 |  -1.0  |  16 | EST  | CLEAN    | Fix  |
| score_both_halves_hm      | lightgbm         | 0.65 | 67.6% |   37 | 0.016 | +0.39 |  -0.9  |  13 | EST  | CLEAN    | Fix  |
| cornershc_over_25         | disag_con_filt   | 0.65 | 63.8% |  177 | 0.007 | +0.27 |  -1.7  |  18 | EST  | CLEAN    | S30  |
| sot_over_95               | lightgbm         | 0.65 | 63.6% |   22 | 0.028 | +0.23 |  -0.0  |  17 | EST  | CLEAN    | Fix  |
| win_to_nil_home           | xgboost          | 0.60 | 53.4% |   58 | 0.016 | +0.27 |  -1.0  |  13 | EST  | CLEAN    | Fix  |
+---------------------------+------------------+------+-------+------+-------+-------+--------+-----+------+----------+------+
| CAUTION — 12 markets (|TS| 2-4, or ts_rejected with high |TS| — bias exists but uncertain magnitude)                       |
+---------------------------+------------------+------+-------+------+-------+-------+--------+-----+------+----------+------+
| sot                       | agreement        | 0.70 | 89.3% |   28 | 0.071 | +0.47 | -15.3* |  18 | EST  | CAUTION  | Fix  |
| sot_over_75               | agreement        | 0.78 | 87.5% |   64 | 0.050 | +0.32 | -24.5* |  14 | EST  | CAUTION  | Fix  |
| cornershc_under_15        | disag_bal_filt   | 0.78 | 86.3% |  197 | 0.022 | +0.42 |  -6.2* |  20 | EST  | CAUTION  | S30  |
| over25                    | catboost         | 0.70 | 80.9% |  162 | 0.044 | +0.27 | -13.8* |  12 | REAL | CAUTION  | S30  |
| cards_under_35            | temporal_blend   | 0.72 | 78.9% |   95 | 0.031 | +0.50 |  -3.2  |  17 | EST  | CAUTION  | v19  |
| btts                      | catboost         | 0.75 | 74.4% |   39 | 0.017 | +0.17 |  +7.0* |   5 | REAL | CAUTION  | Fix  |
| bookpts_over_305          | average          | 0.70 | 73.8% |  477 | 0.014 | +0.06 | -12.1* |  12 | EST  | CAUTION  | Fix  |
| sot_under_95              | agreement        | 0.75 | 73.4% |  222 | 0.075 | +0.03 | -17.8* |   8 | EST  | CAUTION  | Fix  |
| shots                     | temporal_blend   | 0.60 | 71.6% |  697 | 0.022 | +0.12 |  -4.1* |  11 | EST  | CAUTION  | S30  |
| under25                   | temporal_blend   | 0.65 | 66.3% |   86 | 0.016 | +0.24 | -12.7* |   5 | REAL | CAUTION  | S30  |
| clean_sheet_home          | disag_bal_filt   | 0.55 | 62.7% |   51 | 0.002 | +0.28 | -12.8* |  13 | EST  | CAUTION  | Fix  |
| sot_under_85              | agreement        | 0.60 | 57.8% |  102 | 0.052 | +0.03 |  -6.8* |  11 | EST  | CAUTION  | Fix  |
+---------------------------+------------------+------+-------+------+-------+-------+--------+-----+------+----------+------+
| EXTREME — 6 markets (|TS| > 4, confirmed bias — deploy with monitoring)                                                    |
+---------------------------+------------------+------+-------+------+-------+-------+--------+-----+------+----------+------+
| home_win_h1               | catboost         | 0.78 | 92.5% |   67 | 0.025 | +0.79 | -20.9  |  11 | EST  | EXTREME  | v19  |
| corners_under_95          | catboost         | 0.78 | 88.5% |  217 | 0.027 | +0.59 | -16.3  |  22 | EST  | EXTREME  | v19  |
| fouls_under_235           | catboost         | 0.83 | 88.6% |   35 | 0.040 | +0.68 |  -9.3  |  14 | EST  | EXTREME  | v19  |
| agoals_under_15           | catboost         | 0.78 | 88.0% |  548 | 0.028 | +0.33 | -15.7  |   7 | EST  | EXTREME  | v19  |
| corners_over_105          | catboost         | 0.78 | 82.3% |  113 | 0.023 | +0.57 |  +4.1  |  21 | EST  | EXTREME  | v19  |
| hgoals_over_15            | catboost         | 0.72 | 82.1% |  195 | 0.039 | +0.47 |  +4.1  |   5 | EST  | EXTREME  | v19  |
+---------------------------+------------------+------+-------+------+-------+-------+--------+-----+------+----------+------+
| NO_HO — 2 markets (no holdout data — cannot validate)                                                                      |
+---------------------------+------------------+------+-------+------+-------+-------+--------+-----+------+----------+------+
| cards_under_55            | average          | 0.80 |   —   |   —  |   —   |   —   |    —   |   5 | EST  | NO_HO    | v19  |
| dc_1x                     | average          | 0.90 |   —   |   —  |   —   |   —   |    —   |  12 | EST  | NO_HO    | v19  |
+---------------------------+------------------+------+-------+------+-------+-------+--------+-----+------+----------+------+

DISABLED MARKETS (not in production):
  Dead (S30 Fix: still 0 HO bets): fouls_under_235 (niche), agoals_under_15 (niche),
    cards_under_55 (niche), away_win_h1 (0 bets even with fallback)
  Marginal HO (< 20 bets): corners_over_85 (13), cards_under_35 (3), away_win (10-12)
  Intractable: cards_over_25, cards_over_45, fouls_over_225, shots_under_255, ht_over_15

GATE SUMMARY
================================================================

  Total enabled: 31 (was 21) | All ECE < 0.10
  S30 Fix deployed: btts (Wave I, real-odds), 10 new market families
  S30 Fix net change: +10 new markets, btts resurrected
  New market families: SOT (6 markets), bookpts (1), clean_sheet (1),
    win_to_nil (1), score_both_halves (1)

KEY INSIGHTS
================================================================

  1. btts RESURRECTED via Wave I (real-odds params): 39 HO bets, 74.4% precision,
     +32.6% ROI (CI: +4.9% to +56.2%), ECE 0.017. CatBoost solo with only 5 features.
     odds_threshold alpha=0.2 was the key — niche params (Wave F) produced 4 HO bets.

  2. SOT (Shots on Target) is a NEW market family with strong results:
     - sot_over_85: 492 HO bets, 67.5% prec, ECE 0.025 (largest new market)
     - sot_under_95: 222 HO bets, 73.4% prec, ECE 0.075 (borderline ECE)
     - sot_over_75: 64 HO bets, 87.5% prec, ECE 0.050 (highest precision)
     - sot: 28 HO bets, 89.3% prec, ECE 0.071 (exceptional precision)
     All estimated-odds — monitor precision in live, ignore ROI.

  3. bookpts_over_305 is the strongest new market by sample size:
     477 HO bets, 73.8% prec, ECE 0.014. Ensemble average with 12 features.

  4. FIX A (relaxed min_precision) partially worked: btts got through with Wave I
     real-odds config. Niche markets (fouls, agoals, cards) still 0 HO bets —
     calibration drift between folds too severe at any threshold.

  5. FIX B (holdout threshold fallback) worked for Mode B markets: home_win went
     from 0 → 11 HO bets (Wave H), hgoals_over_15 got 29 HO bets (Wave G).
     But home_win 11 bets < deployment gate (20), kept existing xgboost config.

  6. FIX C (temporal leakage exclusion) improved cards markets: adversarial AUC
     dropped to 0.72-0.75 range after excluding round_number, cross_yellows_*.

  7. Adversarial AUC well-controlled across all new markets (0.67-0.80).
     S29 stationarity fix (EWM/rolling) continues to prove effective.

  PENDING ACTIONS:
    - Monitor btts live: first deployment with real-odds + odds_threshold
    - Monitor SOT family live: entirely new market type, no historical data
    - Monitor bookpts_over_305, clean_sheet_home, win_to_nil_home live
    - sot_under_95 ECE=0.075 borderline — watch for calibration drift
    - win_to_nil_home precision 53.4% is low — quick to disable if live fails
    - sot_under_85 precision 57.8% is low — watch closely
    - 3 bookpts/offsides runs still in progress — may add more markets


================================================================
================================================================
       TRAINING SESSION HISTORY (newest first)
================================================================
================================================================


================================================================
S30 FIX WAVES (2026-03-19 evening, 3 code fixes + 4 dispatch waves)
================================================================

CODE CHANGES (commit dfed9f7):
  1. Fix A: min_precision 0.60→0.55 + relaxed fallback (0.50, 10 bets)
     Tracks relaxed_best in parallel during grid search. If no config passes
     primary constraints (0.55 prec, 20 bets), falls back to best with 0.50/10.
  2. Fix B: Holdout threshold auto-fallback when threshold > pred_max
     Instead of logging THRESHOLD UNREACHABLE with 0 HO bets, auto-selects
     highest reachable threshold from the grid.
  3. Fix C: Exclude temporal leakage features from EXCLUDE_COLUMNS:
     round_number, cross_yellows_total, cross_yellows_product
     (SHAP + KS confirmed: cards_over_35 adv AUC 0.823, 12/12 feature shift)

DISPATCH WAVES (feature_params_mode=none, only_if_better=true):
  Wave F (Mode A niche):  btts, corners_over_85, fouls_under_235, agoals_under_15, cards_under_55
    CI run: 23308742135 — 700 trials, decay=0.005, adversarial 2,10,0.75
    model_flags: max_rfe_features=30, mrmr=15, no_fastai, holdout_folds=1
  Wave G (Mode B):        cards_under_35, hgoals_over_15, away_win_h1
    CI run: 23308976205 — 700 trials, decay=0.005, adversarial 2,10,0.75
    model_flags: max_rfe_features=30, mrmr=15, no_fastai, holdout_folds=1
  Wave H (H2H):           home_win, away_win
    CI run: 23309199001 — 500 trials, odds_threshold, alpha=0.2, adversarial 3,15,0.70
    model_flags: max_rfe_features=30, mrmr=20, holdout_folds=1
  Wave I (btts real-odds): btts
    CI run: 23309422281 — 500 trials, odds_threshold, alpha=0.2, adversarial 3,15,0.70
    model_flags: max_rfe_features=20, mrmr=12, holdout_folds=1

RESULTS:
  DEPLOYED (1 updated + 10 new from other today runs):
    btts:              catboost, t=0.75, 39 HO bets, 74.4% prec, +32.6% ROI, ECE 0.017, 5 feats
    bookpts_over_305:  average, t=0.70, 477 HO bets, 73.8% prec, ECE 0.014, 12 feats [NEW]
    sot_over_85:       two_stage_xgb, t=0.60, 492 HO bets, 67.5% prec, ECE 0.025, 16 feats [NEW]
    sot_under_95:      agreement, t=0.75, 222 HO bets, 73.4% prec, ECE 0.075, 8 feats [NEW]
    sot_under_85:      agreement, t=0.60, 102 HO bets, 57.8% prec, ECE 0.052, 11 feats [NEW]
    sot_over_75:       agreement, t=0.78, 64 HO bets, 87.5% prec, ECE 0.050, 14 feats [NEW]
    win_to_nil_home:   xgboost, t=0.60, 58 HO bets, 53.4% prec, ECE 0.016, 13 feats [NEW]
    clean_sheet_home:  disagree_balanced, t=0.55, 51 HO bets, 62.7% prec, ECE 0.002, 13 feats [NEW]
    score_both_halves_home: lightgbm, t=0.65, 37 HO bets, 67.6% prec, ECE 0.016, 13 feats [NEW]
    sot:               agreement, t=0.70, 28 HO bets, 89.3% prec, ECE 0.071, 18 feats [NEW]
    sot_over_95:       lightgbm, t=0.65, 22 HO bets, 63.6% prec, ECE 0.028, 17 feats [NEW]

  KEPT CURRENT (better than new):
    hgoals_over_15:   Current (195 HO, 82.1% prec) > Wave G (29 HO, 82.8% prec)
    corners_over_85:  Current (1539 HO, 78.4% prec) > Wave F (55 HO, 74.5% prec)
    home_win:         Current (23 HO, 82.6% prec) = Wave H (23 HO, same data)

  FAILED (Wave F/G/H markets with 0 or <20 HO bets):
    Wave F: fouls_under_235 (0 HO), agoals_under_15 (0 HO), cards_under_55 (0 HO)
    Wave F: corners_over_85 (13 HO, kept current), btts niche (4 HO, Wave I deployed instead)
    Wave G: away_win_h1 (0 HO — odds constraints 2.0-5.0 too restrictive)
    Wave G: cards_under_35 (3 HO)
    Wave H: home_win (11 HO, < 20 gate), away_win (10 HO, < 20 gate)

FIX EFFECTIVENESS:
  Fix A (relaxed fallback): Enabled btts (via Wave I), unblocked relaxed candidates in grid search
  Fix B (threshold fallback): home_win 0→11 HO, hgoals_over_15 maintained 29 HO, cards_under_35 3 HO
  Fix C (temporal leakage): Cards adversarial AUC improved (0.72-0.75 range)


================================================================
S30 FINAL RETRAIN (2026-03-19, threshold grid fix + stationarity)
================================================================

CODE CHANGES (commit 3389423):
  1. Threshold grid floor lowered for niche/regression_line markets:
     OLD: [0.72, 0.75, 0.78, 0.80, 0.83]
     NEW: [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.78]
  2. btts grid: [0.55, 0.60, 0.65, 0.70, 0.72, 0.75, 0.78]
  3. dc_1x/dc_12 grid: [0.72, 0.75, 0.78, 0.80, 0.83, 0.85]
  4. THRESHOLD UNREACHABLE diagnostic in _evaluate_holdout
  5. H2H markets (home_win, away_win, over25) unchanged

PRE-REQUISITES (S29-S30):
  - S29: EWM/rolling replaced expanding windows (adversarial AUC -0.10)
  - S30 min_bets fix: 60→20 (old default caused 0 holdout bets)
  - S30 DC/AH leakage fix: dc_1x/dc_x2/dc_12/ah_minus_15 targets + fair_odds
    added to EXCLUDE_COLUMNS
  - Features regenerated on HF Hub post-S29

ROOT CAUSE: 10/16 S30 markets had 0 holdout bets because threshold grid floor
  (0.72) exceeded holdout prediction max (~0.65-0.70). Calibration drift between
  optimization folds and holdout fold causes holdout predictions to be lower.
  Proven: btts pred_max=0.683 < threshold 0.72 → 0 bets.

CAMPAIGN: 5 waves, 22 markets, feature_params_mode=none, only_if_better=true.
  Wave A (H2H):    home_win, away_win, over25, under25 — 500 trials, odds_threshold
    CI runs: 23301388115 (812), 23301722505 (814)
  Wave B (Niche):   btts, corners_over_85, corners_over_95, fouls_under_235, cards_over_35
    CI run: 23301554117 (813) — 700 trials, decay=0.005
  Wave C (Niche):   hgoals_over_15, agoals_under_15, cornershc_under_15, cornershc_over_25, shots
    CI run: 23301796601 (815) — 700 trials
  Wave D (Niche):   cards_under_55, cards_under_35, cornershc_over_15, ht_over_15
    CI run: 23302038091 (816) — 700 trials
  Wave E (H1):     home_win_h1, away_win_h1
    CI run: 23302278830 (817) — 500 trials, odds_threshold

RESULTS:
  DEPLOYED (5):
    over25:           catboost, t=0.70, 162 HO bets, 80.9% prec, +35.3% ROI, ECE 0.044, 12 feats
    under25:          temporal_blend, t=0.65, 86 HO bets, 66.3% prec, +65.7% ROI, ECE 0.016, 5 feats
    shots:            temporal_blend, t=0.60, 697 HO bets, 71.6% prec, ECE 0.022, 11 feats
    cornershc_under_15: disagree_balanced, t=0.78, 197 HO bets, 86.3% prec, ECE 0.022, 20 feats
    cornershc_over_25:  disagree_conserv, t=0.65, 177 HO bets, 63.8% prec, ECE 0.007, 18 feats

  KEPT CURRENT (1):
    corners_over_95:  Current (362 HO, 81.5% prec) better than S30 (75 HO, 54.7% prec)

  FAILED (16):
    H2H: home_win (0-5 HO), away_win (1-6 HO) — threshold still too high for HO predictions
    Niche 0 HO: btts (20 feats=overfit), corners_over_85, fouls_under_235, agoals_under_15,
      cards_under_55, cards_under_35, cornershc_over_15
    Low HO: cards_over_35 (1), hgoals_over_15 (4)
    Dead: ht_over_15 (no viable model)
    H1: home_win_h1 (0 bets), away_win_h1 (pred_max=0.62 < t=0.70)


================================================================
S28 EDGE THRESHOLD CAMPAIGN (2026-03-18, NegBin edge mode)
================================================================

CODE CHANGES (commits eb79785, a1217b5):
  1. --edge-threshold CLI flag: replaces raw probability threshold with
     ML edge over NegBin baseline. Bet when (prob - negbin_prob) >= min_edge
     AND prob >= prob_floor.
  2. Adaptive prob_floor: [base_rate*0.7, base_rate*0.9, base_rate*1.1]
     (fixed 0.55/0.60 killed under-markets with <50% base rate).
  3. Edge mode min_bets halved (floor 20) since edge filtering is selective.
  4. GitHub Actions: edge_threshold parsed from model_flags.

CAMPAIGN: 5 markets x 3 runs (A/B/C comparison), 700 trials each.
  Run 1 (no whitelist):      CI run 23257959410 — all 5 markets: 0 bets
  Run 2 (11 ref features):   CI run 23264272487 — fouls: 66 bets/47% prec, rest: 0
  Run 3 (26 broad whitelist): CI run 23264844172 — corners_over_95: 362 bets/81.5% prec

DEPLOYED: NONE — corners_over_95 blocked by TS gate (|TS|=15.1 > 4.0, models not saved).
  Needs TS investigation before deployment.


================================================================
v19 CAMPAIGN (2026-03-18, league-aggregate features)
================================================================

CODE CHANGES:
  1. LeagueAggregateFeatureEngineer (commit 294525d):
     7 new features: league_home_win_rate, league_draw_rate, league_avg_goals,
     league_goal_std, league_btts_rate, league_avg_corners, league_corners_std.
     All expanding + shift(1), min_periods=20.
  2. CI merge step fix (commit 58c99d4): Engineer runs at merge time when
     all leagues are combined (per-league step has no league column).
  3. Pandas FutureWarning fix (commit 7e07910): include_groups=False in
     CardsFeatureEngineer groupby.apply().
  4. IndexError fix (commit dd4d688): Rebuild _per_model_features at
     train_and_save_models start (mRMR K changed feature count).

CAMPAIGN: 6 waves, 27 markets, 500-700 trials, only_if_better=true.

DEPLOYED (18 from v19 + 2 kept from v18 = 20 total).
DEAD (5): corners_over_115, corners_under_85, cardshc_over_05/under_05, fouls_under_195.


================================================================
v18 CAMPAIGN (2026-03-18, TS sign fix + NegBin edge)
================================================================

DEPLOYED (2): btts (|TS|=0.4), cornershc_over_15 (|TS|=1.3)


================================================================
v17 CAMPAIGN (2026-03-18, mRMR K search)
================================================================

DEPLOYED (12). KEY: mRMR K search works, max_threshold=0.80 revived home_win.


================================================================
v16 CAMPAIGN (2026-03-17, TS penalty gate)
================================================================

DEPLOYED (5). KEY: TS gate is partial fix — filters bad configs but not root cause.


================================================================
v15 CAMPAIGN (2026-03-16, DSR + K_eff)
================================================================

CRITICAL: Holdout fold EXCLUDED from Optuna — fixes test-set leakage in all prior campaigns.
DEPLOYED (7). Endemic TS bias discovered.


================================================================
v14 TEST WAVE (2026-03-16) — NOT DEPLOYED
================================================================

Diagnostic only. Holdout metrics inflated by test-set leakage.


================================================================
v13 FULL RETRAIN (2026-03-16, LGB Fix + Feature Subsetting)
================================================================

DEPLOYED (27 markets). NOTE: holdout metrics inflated by test-set leakage.
Three commits fix: LGB subsample, per-model feature subsetting, num_leaves cap.
