# Bettip - Football Betting Prediction System

## Complete Project Documentation

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Directory Structure](#directory-structure)
4. [Core Modules](#core-modules)
5. [Pipelines](#pipelines)
6. [Configuration](#configuration)
7. [Betting Strategies](#betting-strategies)
8. [Data Flow](#data-flow)
9. [MLOps (MLflow + Metaflow)](#mlops)
10. [CI/CD](#cicd)
11. [Quick Start](#quick-start)
12. [File Reference](#file-reference)

---

## 1. Project Overview

**Bettip** is an end-to-end machine learning system for football match prediction and betting recommendations. It covers data collection, feature engineering, model training, and automated predictions for multiple betting markets.

### Supported Leagues
- Premier League (England)
- La Liga (Spain)
- Serie A (Italy)
- Bundesliga (Germany)
- Ligue 1 (France)

### Supported Bet Types
| Bet Type | Status | ROI | P(profit) |
|----------|--------|-----|-----------|
| Asian Handicap | ✅ Enabled | +20.8% | 97% |
| BTTS (Both Teams To Score) | ✅ Enabled | +16.0% | 98% |
| Away Win | ✅ Enabled | +14.5% | 100% |
| Over 2.5 Goals | ❌ Disabled | +4.6% | 75% |
| Under 2.5 Goals | ❌ Disabled | +5.5% | 64% |
| Home Win | ❌ Disabled | -2.6% | 20% |

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              BETTIP ARCHITECTURE                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐                 │
│  │ Data Sources │     │   Storage    │     │   Outputs    │                 │
│  ├──────────────┤     ├──────────────┤     ├──────────────┤                 │
│  │ API-Football │     │ Hugging Face │     │ MLflow UI    │                 │
│  │ Football-Data│     │ Local Parquet│     │ Predictions  │                 │
│  │ The Odds API │     │ MLflow DB    │     │ JSON Recs    │                 │
│  └──────┬───────┘     └──────┬───────┘     └──────────────┘                 │
│         │                    │                     ▲                        │
│         ▼                    ▼                     │                        │
│  ┌─────────────────────────────────────────────────┴──────────────────────┐ │
│  │                          PIPELINE STAGES                                │ │
│  │                                                                         │ │
│  │  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   │ │
│  │  │ Collect │ → │Preproc. │ → │Features │ → │ Train   │ → │Inference│   │ │
│  │  │  Data   │   │         │   │ Eng.    │   │ Models  │   │         │   │ │
│  │  └─────────┘   └─────────┘   └─────────┘   └─────────┘   └─────────┘   │ │
│  │       │             │             │             │             │        │ │
│  │       ▼             ▼             ▼             ▼             ▼        │ │
│  │   01-raw/      02-preproc/   03-features/   mlruns/     04-predict/    │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                         ORCHESTRATION                                   │ │
│  │                                                                         │ │
│  │   Metaflow (DAG Pipelines)  +  MLflow (Experiment Tracking)            │ │
│  │   GitHub Actions (CI/CD)    +  Hugging Face Hub (Data Storage)         │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Directory Structure

```
bettip/
├── config/                    # Configuration files
│   ├── strategies.yaml        # Betting strategies & thresholds
│   ├── prod.yaml              # Production config
│   ├── premier_league.yaml    # League-specific configs
│   ├── la_liga.yaml
│   ├── serie_a.yaml
│   ├── bundesliga.yaml
│   └── ligue_1.yaml
│
├── src/                       # Core source code
│   ├── data_collection/       # API clients & collectors
│   ├── preprocessing/         # Data validation & cleaning
│   ├── features/              # Feature engineering
│   ├── ml/                    # ML models & training
│   ├── odds/                  # Betting odds handling
│   ├── pipelines/             # Pipeline orchestration
│   └── config_loader.py       # Configuration utilities
│
├── entrypoints/               # CLI entry points
│   ├── run_pipeline.py        # Main orchestration
│   ├── collect.py             # Data collection
│   ├── preprocess.py          # Preprocessing
│   ├── features.py            # Feature generation
│   ├── train.py               # Model training
│   ├── inference.py           # Predictions
│   ├── fetch_odds.py          # Odds fetching
│   ├── download_data.py       # HF Hub download
│   └── upload_data.py         # HF Hub upload
│
├── flows/                     # Metaflow pipelines
│   ├── betting_flow.py        # Training pipeline
│   └── daily_inference_flow.py# Daily predictions
│
├── experiments/               # ML experimentation
│   ├── run_full_optimization_pipeline.py
│   ├── run_ah_improved.py
│   ├── run_btts_full_optimization.py
│   └── outputs/               # Experiment results
│
├── data/                      # Data directories
│   ├── 01-raw/                # Raw API data
│   ├── 02-preprocessed/       # Cleaned data
│   ├── 03-features/           # ML features
│   ├── 04-predictions/        # Predictions
│   └── odds-cache/            # Cached odds
│
├── tests/                     # Test suite
│   ├── unit/
│   └── integration/
│
├── .github/workflows/         # CI/CD
│   ├── collect-match-data.yaml
│   └── train-and-predict.yaml
│
├── docs/                      # Documentation
├── outputs/                   # Pipeline outputs
├── mlruns/                    # MLflow artifacts
└── cloudformation/            # AWS infrastructure
```

---

## 4. Core Modules

### 4.1 Data Collection (`src/data_collection/`)

| File | Purpose |
|------|---------|
| `api_client.py` | API-Football client with rate limiting |
| `match_collector.py` | Match data collection orchestration |
| `scheduler.py` | Scheduled update management |

### 4.2 Preprocessing (`src/preprocessing/`)

| File | Purpose |
|------|---------|
| `loaders.py` | Data loading utilities |
| `parsers.py` | API response parsing |
| `processors.py` | Data transformation |
| `validators.py` | Data validation rules |
| `extractors.py` | Feature extraction |
| `writers.py` | Output writers |
| `factory.py` | Processor factory pattern |

### 4.3 Features (`src/features/`)

| File | Purpose |
|------|---------|
| `engineers.py` | **18+ feature engineers** (main file, ~2000 lines) |
| `cleaners.py` | Data cleaning |
| `loaders.py` | Multi-file loading |
| `merger.py` | Feature merging |
| `interfaces.py` | Base interfaces |

**Feature Engineers in `engineers.py`:**
- `TeamFormEngineer` - Recent form (wins/draws/losses)
- `TeamStatsEngineer` - Season statistics
- `EMAFormEngineer` - Exponential moving averages
- `ELORatingEngineer` - ELO ratings
- `PoissonEngineer` - Expected goals (xG)
- `HeadToHeadEngineer` - H2H history
- `HomeAwayFormEngineer` - Home/away specific
- `RestDaysEngineer` - Rest between matches
- `LeaguePositionEngineer` - Table position
- `StreakEngineer` - Win/loss streaks
- `GoalDifferenceEngineer` - Goal metrics
- `MatchImportanceEngineer` - Match importance
- `RefereeEngineer` - Referee statistics
- `FormationEngineer` - Team formations
- `CoachEngineer` - Manager data
- And more...

### 4.4 ML (`src/ml/`)

| File | Purpose |
|------|---------|
| `models.py` | Model factory (RF, XGB, LGB, CatBoost, LR) |
| `mlflow_config.py` | MLflow configuration & registry |
| `experiment.py` | Experiment tracking |
| `metrics.py` | Betting-specific metrics |
| `tuning.py` | Hyperparameter optimization |
| `ensemble.py` | Ensemble methods |
| `business_targets.py` | Betting target variables |

### 4.5 Odds (`src/odds/`)

| File | Purpose |
|------|---------|
| `football_data_loader.py` | Football-data.co.uk odds |
| `btts_odds_loader.py` | BTTS odds handling |
| `odds_features.py` | Odds-derived features |
| `odds_merger.py` | Merge odds with matches |

### 4.6 Pipelines (`src/pipelines/`)

| File | Purpose |
|------|---------|
| `preprocessing_pipeline.py` | Raw → Preprocessed |
| `feature_eng_pipeline.py` | Preprocessed → Features |
| `betting_training_pipeline.py` | Full training pipeline |
| `betting_inference_pipeline.py` | Predictions & recommendations |
| `training_pipeline.py` | Legacy training |
| `inference_pipeline.py` | Legacy inference |

---

## 5. Pipelines

### 5.1 Training Pipeline

```python
from src.pipelines.betting_training_pipeline import BettingTrainingPipeline, TrainingConfig

config = TrainingConfig(
    data_path='data/03-features/features_all_5leagues_with_odds.csv',
    strategies_path='config/strategies.yaml',
    output_dir='outputs/training',
    n_optuna_trials=80
)

pipeline = BettingTrainingPipeline(config)
results = pipeline.run(['asian_handicap', 'btts', 'away_win'])
```

**Pipeline Steps:**
1. Load data and create targets
2. Add bet-type-specific features
3. Feature selection (permutation importance)
4. Model tuning (Optuna)
5. Train ensemble (XGBoost, LightGBM, CatBoost)
6. Calibrate probabilities
7. Evaluate with bootstrap CI
8. Register models to MLflow

### 5.2 Inference Pipeline

```python
from src.pipelines.betting_inference_pipeline import BettingInferencePipeline, InferenceConfig

config = InferenceConfig(
    strategies_path='config/strategies.yaml',
    models_dir='outputs/callibration',
    bankroll=1000.0
)

pipeline = BettingInferencePipeline(config)
recommendations = pipeline.run(fixtures_df)
```

### 5.3 Metaflow Pipelines

```bash
# Training flow (parallel by bet type)
python flows/betting_flow.py run

# Daily inference flow
python flows/daily_inference_flow.py run

# Show DAG
python flows/betting_flow.py show
```

---

## 6. Configuration

### 6.1 Strategies Configuration (`config/strategies.yaml`)

```yaml
strategies:
  asian_handicap:
    enabled: true
    approach: "regression"
    model_type: "ensemble"
    target: "goal_margin"
    line_filter:
      min: -4.0
      max: -1.5
    bet_side: "away"
    edge_threshold: 0.3
    expected_roi: 20.8
    p_profit: 0.97

  btts:
    enabled: true
    approach: "classification"
    model_type: "catboost"
    target: "btts"
    probability_threshold: 0.6
    expected_roi: 16.0
    p_profit: 0.98

  away_win:
    enabled: true
    approach: "classification"
    model_type: "catboost"
    target: "away_win"
    probability_threshold: 0.45
    expected_roi: 14.5
    p_profit: 1.0

risk_management:
  max_daily_bets: 10
  max_stake_per_bet: 0.05
  stop_loss_daily: 0.10
  take_profit_daily: 0.20

pilot:
  paper_trading: true
  min_bets_before_live: 100
  performance_threshold: 0.05
```

### 6.2 League Configuration (`config/premier_league.yaml`)

```yaml
league: "premier_league"
seasons: [2020, 2021, 2022, 2023, 2024, 2025]

data:
  raw_dir: "data/01-raw"
  preprocessed_dir: "data/02-preprocessed"
  features_dir: "data/03-features"

preprocessing:
  batch_size: 500
  include_player_features: true

features:
  form_window: 5
  ema_span: 10
  include_h2h: true
```

---

## 7. Betting Strategies

### 7.1 Asian Handicap (ROI: +20.8%)

**Approach:** Regression-based value betting
- Predict goal margin (not binary outcome)
- Compare to bookmaker's line
- Bet when edge > 0.3 goals

**Target:** Heavy favorites (line ≤ -1.5), bet Away

**Key Features:**
- `margin_edge` - Our prediction vs bookmaker
- `composite_expected_margin` - Weighted combo
- `season_margin_per_game`
- `away_win_prob_elo`

### 7.2 BTTS (ROI: +16.0%)

**Approach:** Classification
- Predict probability both teams score
- Bet Yes when probability ≥ 0.6

**Key Features:**
- `btts_composite` - Combined scoring probability
- `home_scores_prob`, `away_scores_prob`
- `total_attack`, `min_defense`
- `league_btts_rate`

### 7.3 Away Win (ROI: +14.5%)

**Approach:** Classification
- Predict away win probability
- Bet when probability ≥ 0.45

**Key Features:**
- `ah_line_close` - Asian handicap line
- `position_diff` - League position difference
- `away_season_ppg` - Away points per game
- `elo_diff` - ELO rating difference

---

## 8. Data Flow

```
API-Football                  Football-Data.co.uk
     │                              │
     ▼                              ▼
┌─────────────┐              ┌─────────────┐
│  Matches    │              │   Odds      │
│  Lineups    │              │  (1X2, OU)  │
│  Events     │              │  (AH, BTTS) │
│  Stats      │              └──────┬──────┘
└──────┬──────┘                     │
       │                            │
       ▼                            │
data/01-raw/                        │
  └── {league}/{season}/            │
       ├── matches.parquet          │
       ├── lineups.parquet          │
       ├── events.parquet           │
       └── player_stats.parquet     │
       │                            │
       ▼                            │
data/02-preprocessed/               │
  └── {league}/{season}/            │
       └── [cleaned parquet]        │
       │                            │
       ▼                            ▼
data/03-features/  ◄────────  Merge Odds
  ├── features_{league}.csv
  ├── features_{league}_with_odds.csv
  └── features_all_5leagues_with_odds.csv
       │
       ▼
┌──────────────────┐
│  Model Training  │
│  (XGB, LGB, Cat) │
└────────┬─────────┘
         │
         ▼
mlruns/  &  MLflow Registry
  ├── Experiments
  ├── Models
  └── Metrics
         │
         ▼
data/04-predictions/
  └── recommendations.json
```

---

## 9. MLOps

### 9.1 MLflow

**Tracking:** `sqlite:///mlflow.db`

```bash
# View experiments
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Access at http://localhost:5000
```

**Model Registry:**
- `asian_handicap_xgboost`
- `asian_handicap_lightgbm`
- `asian_handicap_catboost`
- `btts_catboost`
- `away_win_catboost`
- etc.

### 9.2 Metaflow

**Training Flow:**
```bash
python flows/betting_flow.py run --n_trials 80
python flows/betting_flow.py show
python flows/betting_flow.py resume
```

**Daily Inference:**
```bash
python flows/daily_inference_flow.py run --bankroll 1000
```

---

## 10. CI/CD

### 10.1 Data Pipeline (`.github/workflows/collect-match-data.yaml`)

**Schedule:** Weekly (Monday 3 AM UTC)

**Steps:**
1. Verify secrets
2. Download data from HF Hub
3. Collect match data (API-Football)
4. Preprocess data
5. Generate features
6. Fetch odds
7. Merge all leagues
8. Upload to HF Hub

### 10.2 ML Pipeline (`.github/workflows/train-and-predict.yaml`)

**Trigger:** After data pipeline or manual

**Steps:**
1. Download data from HF Hub
2. Train models (Optuna tuning)
3. Register to MLflow
4. Generate predictions
5. Save artifacts

---

## 11. Quick Start

### Installation

```bash
# Clone repository
git clone <repo-url>
cd bettip

# Install dependencies
uv sync
```

### Run Training

```bash
# Full training (all enabled strategies)
python entrypoints/run_pipeline.py --mode train \
    --data data/03-features/features_all_5leagues_with_odds.csv

# Specific bet types
python entrypoints/run_pipeline.py --mode train \
    --data data/03-features/features_all_5leagues_with_odds.csv \
    --bet-types asian_handicap away_win
```

### Run Inference

```bash
python entrypoints/run_pipeline.py --mode inference \
    --fixtures data/04-predictions/upcoming_fixtures.csv
```

### Using Metaflow

```bash
# Show pipeline DAG
python flows/betting_flow.py show

# Run training
python flows/betting_flow.py run

# Daily predictions
python flows/daily_inference_flow.py run
```

---

## 12. File Reference

### Core Source Files

| File | Lines | Description |
|------|-------|-------------|
| `src/features/engineers.py` | ~2000 | Feature engineering (18+ engineers) |
| `src/data_collection/match_collector.py` | ~590 | Match data collection |
| `src/pipelines/feature_eng_pipeline.py` | ~510 | Feature pipeline |
| `src/pipelines/betting_training_pipeline.py` | ~600 | Training pipeline |
| `src/pipelines/betting_inference_pipeline.py` | ~450 | Inference pipeline |
| `src/ml/mlflow_config.py` | ~200 | MLflow configuration |
| `src/data_collection/api_client.py` | ~300 | API client with rate limiting |
| `src/config_loader.py` | ~220 | Configuration loader |
| `src/odds/football_data_loader.py` | ~250 | Odds data loader |

### Entrypoints

| File | Description |
|------|-------------|
| `entrypoints/run_pipeline.py` | Main orchestration script |
| `entrypoints/collect.py` | Data collection CLI |
| `entrypoints/preprocess.py` | Preprocessing CLI |
| `entrypoints/features.py` | Feature generation CLI |
| `entrypoints/train.py` | Training CLI |
| `entrypoints/inference.py` | Inference CLI |
| `entrypoints/fetch_odds.py` | Odds fetching CLI |
| `entrypoints/download_data.py` | HF Hub download |
| `entrypoints/upload_data.py` | HF Hub upload |

### Metaflow Flows

| File | Description |
|------|-------------|
| `flows/betting_flow.py` | Training pipeline (parallel by bet type) |
| `flows/daily_inference_flow.py` | Daily predictions |

### Configuration Files

| File | Description |
|------|-------------|
| `config/strategies.yaml` | Betting strategies & thresholds |
| `config/prod.yaml` | Production configuration |
| `config/premier_league.yaml` | Premier League config |
| `config/la_liga.yaml` | La Liga config |
| `config/serie_a.yaml` | Serie A config |
| `config/bundesliga.yaml` | Bundesliga config |
| `config/ligue_1.yaml` | Ligue 1 config |

### Experiment Scripts

| File | Description |
|------|-------------|
| `experiments/run_full_optimization_pipeline.py` | Generic optimization for all bet types |
| `experiments/run_ah_improved.py` | Asian Handicap with new features |
| `experiments/run_btts_full_optimization.py` | BTTS optimization |
| `experiments/run_asian_handicap_specialized.py` | AH regression approach |

### Test Files

| File | Description |
|------|-------------|
| `tests/unit/test_features.py` | Feature engineering tests |
| `tests/unit/test_preprocessing.py` | Preprocessing tests |
| `tests/integration/test_pipelines.py` | Pipeline integration tests |
| `tests/test_data_leakage.py` | Data leakage prevention tests |

### CI/CD Workflows

| File | Description |
|------|-------------|
| `.github/workflows/collect-match-data.yaml` | Data collection pipeline |
| `.github/workflows/train-and-predict.yaml` | ML training pipeline |

---

## Version History

- **v1.0** - Initial data collection and preprocessing
- **v2.0** - Feature engineering (18+ engineers)
- **v3.0** - ML training with multiple models
- **v4.0** - Betting optimization (Away Win +12.5%)
- **v5.0** - Full optimization pipeline (Asian Handicap +20.8%)
- **v6.0** - MLflow + Metaflow integration

---

## Contact & Support

- Issues: GitHub Issues
- Data: Hugging Face Hub

---

*Last Updated: January 2026*
