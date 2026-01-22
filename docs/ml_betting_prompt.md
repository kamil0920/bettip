# ML Betting Analysis Prompt

## Role & Context

You are a senior ML researcher + sports-betting quant. Be ruthless in detecting methodological issues (data leakage, wrong validation, target mistakes, lookahead bias) and focus on maximizing out-of-sample betting profit while keeping validation scientifically sound (proper backtesting).

**Goal:** Extract the maximum value from my data, compare several models (each predicts a different event/market), and at the end present a ranking of approaches by both ML quality and profitability metrics (EV/ROI) from betting simulation.

---

## Dataset Description

| Attribute | Value |
|-----------|-------|
| **Sport** | Football (Soccer) |
| **Leagues** | Premier League, La Liga, Serie A, Bundesliga, Ligue 1 |
| **Time Range** | 2019-08-09 to 2026-01-18 (6.5 seasons) |
| **Granularity** | Match-level (one row per fixture) |
| **Total Matches** | 10,399 |
| **Features** | 394 columns |
| **Timezone** | UTC |

---

## Target Definitions

| Model | Target | Type | Description | Horizon |
|-------|--------|------|-------------|---------|
| **BTTS** | `btts` | Binary classification | Both teams score (home_goals > 0 AND away_goals > 0) | Pre-match |
| **Away Win** | `away_win` | Binary classification | Away team wins (away_goals > home_goals) | Pre-match |
| **Corners** | `total_corners > line` | Binary classification | Lines: 9.5, 10.5, 11.5 | Pre-match |
| **Cards** | `total_cards > line` | Binary classification | Lines: 3.5, 4.5 | Pre-match |
| **Shots** | `total_shots > line` | Binary classification | Lines: 22.5, 24.5, 26.5 | Pre-match |
| **Fouls** | `total_fouls > line` | Binary classification | Lines: 22.5, 24.5, 26.5 | Pre-match |
| **Asian Handicap** | `goal_margin` | Regression | home_goals - away_goals | Pre-match |

---

## Bookmaker Odds Columns

| Source | Columns | Type |
|--------|---------|------|
| **API-Football** | `odds_home_prob`, `odds_draw_prob`, `odds_away_prob` | Implied probabilities |
| **API-Football** | `ah_line`, `ah_line_close`, `avg_ah_away` | Asian Handicap |
| **SportMonks** | `sm_corners_over_odds`, `sm_corners_under_odds`, `sm_corners_line` | Corners market |
| **SportMonks** | `sm_cards_over_odds`, `sm_cards_under_odds`, `sm_cards_line` | Cards market |
| **SportMonks** | `sm_shots_over_odds`, `sm_shots_under_odds`, `sm_shots_line` | Shots market |
| **SportMonks** | `sm_btts_yes_odds`, `sm_btts_no_odds` | BTTS market |

**Odds Timestamp:** Pre-match collection time (not closing line).

---

## Staking Rules

| Parameter | Value |
|-----------|-------|
| **Method** | Fractional Kelly |
| **Bankroll Fraction** | 2% |
| **Min Odds** | 1.30 |
| **Max Odds** | 10.0 |
| **Max Stake/Bet** | 5% of bankroll |
| **Daily Stop-Loss** | 10% |
| **Daily Take-Profit** | 20% |

---

## Hard Rules (Must Not Break)

1. **Validation must be time-based** (walk-forward / expanding window). No random split for time series data.
2. **Every proposed improvement must be validated immediately:** "change → pipeline → OOS results → compare to baseline".
3. **No feature may use information not available at bet time** (strict anti-leakage rule).
4. **Always report both:**
   - Predictive metrics (AUC, LogLoss, Brier, ECE calibration)
   - Betting metrics (EV, ROI, yield, max drawdown, Sharpe/Sortino)
   - Computed on the same OOS splits.

---

## Feature Engineering (394 Features)

### Feature Categories

| Category | Count | Examples |
|----------|-------|----------|
| **Form/Momentum** | 67 | `*_wins_last_n`, `*_goals_scored_ema`, `*_streak` |
| **Composite/Derived** | 58 | `elo_diff`, `xg_diff`, `goal_diff_advantage` |
| **Goals** | 39 | `home_goals_scored_last_n`, `away_goals_conceded_ema` |
| **Corners** | 37 | `home_corners_won_ema`, `ref_corners_avg` |
| **Cards** | 37 | `home_avg_yellows`, `away_avg_reds` |
| **Shots** | 36 | `home_shots_ema`, `combined_shots_ema` |
| **Odds/Market** | 33 | `ah_line`, `odds_overround`, `line_movement_magnitude` |
| **Fouls** | 30 | `home_fouls_committed_ema`, `ref_fouls_avg` |
| **Referee** | 15 | `ref_cards_avg`, `ref_fouls_bias`, `ref_corners_avg` |
| **Position/Table** | 14 | `home_league_position`, `pts_to_leader`, `position_diff` |
| **Attack/Defense** | 14 | `home_attack_strength`, `away_defense_strength` |
| **Home/Away Venue** | 14 | `home_home_wins`, `away_away_goals_scored` |
| **Season** | 8 | `home_season_ppg`, `season_gd_diff`, `season_phase` |
| **Rest/Schedule** | 7 | `home_rest_days`, `rest_days_diff`, `away_short_rest` |
| **BTTS-specific** | 7 | `home_clean_sheet_streak`, `btts_int_sot_product` |
| **ELO** | 6 | `home_elo`, `away_elo`, `elo_diff`, `*_win_prob_elo` |
| **Poisson** | 5 | `home_xg_poisson`, `poisson_home_win_prob` |
| **H2H** | 4 | `h2h_home_wins`, `h2h_avg_goals` |

### Feature Engineering Methods (18 Engineers)

| Engineer | Features | Method |
|----------|----------|--------|
| **ELORatingFeatureEngineer** | `home_elo`, `away_elo`, `elo_diff` | Dynamic ELO with K-factor decay |
| **PoissonFeatureEngineer** | `*_xg_poisson`, `poisson_*_prob` | Poisson regression on goals |
| **TeamFormFeatureEngineer** | `*_wins_last_n`, `*_goals_*_last_n` | Rolling window (n=5) |
| **ExponentialMovingAverageFeatureEngineer** | `*_ema` | EMA with α=0.3 |
| **HomeAwayFormFeatureEngineer** | `home_home_*`, `away_away_*` | Venue-specific splits |
| **HeadToHeadFeatureEngineer** | `h2h_*` | Last 10 H2H meetings |
| **RefereeFeatureEngineer** | `ref_*` | Per-referee historical stats |
| **LeaguePositionFeatureEngineer** | `*_league_position`, `pts_to_*` | Current standings |
| **RestDaysFeatureEngineer** | `*_rest_days` | Days since last match |
| **CornerFeatureEngineer** | `*_corners_*_ema` | Corner-specific rolling stats |
| **FoulsFeatureEngineer** | `*_fouls_committed_ema` | Foul-specific rolling stats |
| **CardsFeatureEngineer** | `*_avg_yellows`, `*_avg_reds` | Card-specific rolling stats |
| **ShotsFeatureEngineer** | `*_shots_ema` | Shots rolling stats |
| **CrossMarketFeatureEngineer** | `btts_int_*`, `shots_int_*` | Cross-market interactions |
| **TeamRatingFeatureEngineer** | `*_attack_strength`, `*_defense_strength` | Attack/defense indices |
| **GoalDifferenceFeatureEngineer** | `*_avg_goal_diff` | GD-based features |
| **DisciplineFeatureEngineer** | `discipline_diff` | Aggregated card/foul tendency |

### Target Columns (Actual Match Outcomes)

| Column | Type | Description |
|--------|------|-------------|
| `home_goals` | float | Goals scored by home team |
| `away_goals` | float | Goals scored by away team |
| `result` | float | 1=home win, 0=draw, -1=away win |
| `total_corners` | float | Total match corners |
| `total_cards` | float | Total yellow + red cards |
| `total_shots` | float | Total shots by both teams |
| `total_fouls` | float | Total fouls committed |

### Anti-Leakage Safeguards

- All features use **only pre-match information**
- Form stats computed from matches **before** current date
- ELO/ratings updated **after** each match (not including current)
- Referee stats exclude current match
- Test: `pytest tests/test_data_leakage.py`

---

## Validation Strategy

### Primary Split (Holdout)

```
┌─────────────────────────────────────────────────────────────────────┐
│                        10,399 matches                               │
├────────────────────┬───────────────────┬────────────────────────────┤
│   TRAIN (60%)      │    VAL (20%)      │      TEST (20%)            │
│   ~6,239 matches   │   ~2,080 matches  │     ~2,080 matches         │
│   2019-08 → 2023   │   2023 → 2024     │     2024 → 2026-01         │
└────────────────────┴───────────────────┴────────────────────────────┘
        ↓                    ↓                      ↓
  Feature selection    Hyperparameter         Final evaluation
  + Model training     tuning (Optuna)        + Betting backtest
```

**Split Method:** Time-based (chronological sort, no shuffle)

### Walk-Forward Validation (Optional)

```
Fold 1: Train [0-20%]  → Test [20-40%]
Fold 2: Train [0-40%]  → Test [40-60%]   (expanding window)
Fold 3: Train [0-60%]  → Test [60-80%]
Fold 4: Train [0-80%]  → Test [80-100%]
```

- 5 folds with expanding training window
- Minimum 30 test samples per fold
- Reports: avg ROI, std ROI, total bets across folds

---

## Model Pipeline Architecture

### Pipeline Steps

| Step | Name | Description |
|------|------|-------------|
| **1** | Feature Selection | Leshy (LightGBM-native Boruta) or BoostAGroota, selects ~40-80 features |
| **2** | Optuna Tuning | 150 trials/model for XGBoost, LightGBM, CatBoost, LogisticRegression |
| **2.5** | Feature Revalidation | Optional: BoostAGroota with tuned XGBoost params |
| **3** | Train + Calibration | Platt (sigmoid), Isotonic, or Beta calibration |
| **4** | SHAP Analysis | TreeExplainer for feature importance validation |
| **5** | Evaluation | Per-model accuracy, ECE, Brier; Simple Average & Stacking ensembles |
| **5.5** | Precision Metrics | Yield curve at different confidence thresholds |
| **6** | Betting Optimization | Threshold sweep with bootstrap ROI, Sharpe, Sortino |
| **7** | Walk-Forward | Optional: 5-fold expanding window OOS validation |

### Model Ensemble

```
Features (40-80) → [XGBoost, LightGBM, CatBoost, LogisticReg] (all calibrated)
                           ↓
                   [Simple Average, Stacking (RidgeClassifierCV)]
                           ↓
                   Betting Filter (proba >= threshold)
```

### Calibration Methods

| Method | Description | Best For |
|--------|-------------|----------|
| **Platt** | Sigmoid fit on probabilities | Well-calibrated base models |
| **Isotonic** | Non-parametric, monotonic | Miscalibrated models |
| **Beta** | 3-param beta distribution | Overconfident models |

---

## Current Baseline Results

| Market | Model | Expected ROI | P(Profit) | Status |
|--------|-------|--------------|-----------|--------|
| **Fouls** | CatBoost | +52.9% | 0.82 | Enabled (paper trading validated) |
| **Shots** | RandomForest | +46.2% | 0.75 | Enabled |
| **Corners** | Stacking XGB | +34.5% | 0.70 | Enabled |
| **Cards** | XGBoost | +25.4% | 0.70 | Enabled |
| **BTTS** | CatBoost | +16.0% | 0.98 | Enabled (needs recalibration) |
| **Away Win** | CatBoost | +14.5% | 1.0 | Enabled |
| **Asian Handicap** | Ensemble | +20.8% | 0.97 | Disabled (negative live ROI) |
| **Home Win** | XGBoost | -2.6% | 0.20 | Disabled |
| **Over 2.5** | LightGBM | Negative | - | Disabled (model broken) |

---

## Known Drift/Changes

| Period | Issue |
|--------|-------|
| 2020-21 | COVID empty stadiums (home advantage reduced) |
| 2021+ | VAR expansion affecting cards/fouls |
| Ongoing | Tactical evolution (pressing intensity) |

---

## Step-by-Step Instructions

### Step 1 — Data Quality Audit

Run a checklist and report concrete issues and fixes:
- Missing data, duplicates, outliers, inconsistent dtypes, constant columns, rare categories
- Timestamp consistency (event_time < feature_time < bet_time)
- Distribution stability over time (drift) and season/rule changes
- Identify suspicious "too good to be true" features (leakage signals)

### Step 2 — Baselines + Correct Validation

Build baselines per target:
- Simple model (logistic regression / LightGBM) + minimal features
- Proper time-based validation + metric report
- Probability calibration (reliability, Brier) — betting needs calibrated probabilities

### Step 3 — Betting Layer (EV + Backtest)

Define and apply:
- Implied probability from odds: `p_imp = 1 / odds`
- EV per bet: compute using `p_model`, stake, and decimal-odds payout
- Bet filtering: place bets only when edge > threshold; test multiple thresholds
- OOS backtest on every fold; aggregate results

### Step 4 — Iterative Improvement (with Immediate Validation)

For each model, run iterations (max 5 rounds):
1. Feature engineering (lags, rolling stats, Elo/Poisson, form, opponent strength) — without leakage
2. Feature selection / regularization
3. Hyperparameter tuning (avoiding overfitting)
4. Calibration (isotonic/Platt/Beta) and measure impact on EV/ROI

After each round: show "baseline vs new pipeline" table on same splits.

### Step 5 — Model Comparison + Recommendation

Produce:
1. Comparison table for all models: predictive metrics + EV/ROI + max drawdown + number of bets
2. Ranking (separately: "best prediction" and "best profit")
3. Recommendation: what to deploy, thresholds to use, key risks (overfitting, drift, liquidity)

---

## Output Format

1. **First:** Ask clarification questions (max 10)
2. **Then:** Execute steps 1–5
3. **If something cannot be computed without data:** Provide exact Python code/pseudocode specifying required files/columns

---

## Quick Reference Answers

| # | Question | Answer |
|---|----------|--------|
| 1 | Sports/leagues + timestamps? | Football: EPL, La Liga, Serie A, Bundesliga, Ligue 1. Match timestamps reliable (UTC). Odds are pre-match. |
| 2 | Odds type? | Pre-match odds from API-Football and SportMonks. Not closing line. |
| 3 | Profit computation? | Fractional Kelly (2%), min odds 1.30, max odds 10.0, max 5%/bet |
| 4 | Observation unit? | Match (one row per fixture, 10,399 total) |
| 5 | Target types? | Binary classification (BTTS, away_win, over/under lines) + Regression (goal_margin) |
| 6 | Markets + multiple lines? | 1X2, Asian Handicap, BTTS, O/U goals, Corners/Cards/Shots/Fouls with multiple lines |
| 7 | Current models + baseline? | XGB/LGB/CatBoost ensemble. Best: Fouls +52.9% ROI, Corners +34.5%, Shots +46.2% |
| 8 | History + drift? | 6.5 seasons (2019-2026). Drift: COVID 2020-21, VAR expansion, tactical changes |
