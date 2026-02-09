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

## Current Deployed Models (Updated Feb 8)

| Market | Source | Model | Key Metric | Threshold |
|--------|--------|-------|------------|-----------|
| home_win | R90 | temporal_blend | HO +128.3%, Sharpe 1.78 | 0.80 |
| over25 | R90 | average | HO +104.5%, Sharpe 1.03 | 0.75 |
| under25 | R90 | disagree_balanced_filtered | HO +64.5% | 0.65 |
| **shots** | **R112** | **temporal_blend** | **HO +114.3%, Sharpe 1.28** | **0.65** |
| **btts** | **R112** | **xgboost** | **HO +43.9% (139 bets)** | **0.60** |
| fouls | R104 | catboost | WF +137% | 0.80 |
| cards | R104 | disagree_aggressive | WF +76% | 0.60 |
| corners | R104 | disagree_aggressive | WF +55% | 0.60 |
| away_win | — | DISABLED | — | — |

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
