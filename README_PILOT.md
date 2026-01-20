# Bettip Pilot Guide

## Overview

This document describes how to run the betting prediction pipeline for the pilot deployment.

## Quick Start

### 1. Install Dependencies

```bash
uv sync
```

### 2. Run Full Training Pipeline

```bash
# Train all enabled strategies (Asian Handicap, BTTS, Away Win)
python entrypoints/run_pipeline.py --mode train \
    --data data/03-features/features_all_5leagues_with_odds.csv
```

### 3. Generate Predictions

```bash
# Run inference on upcoming fixtures
python entrypoints/run_pipeline.py --mode inference \
    --fixtures data/04-predictions/upcoming_fixtures.csv
```

## Pipeline Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `full` | Preprocess + Features + Train | Initial setup or full rebuild |
| `train` | Train models only | Retraining with new data |
| `inference` | Predictions only | Daily predictions |
| `daily` | Preprocess + Features + Inference | Daily automated updates |

## Enabled Betting Strategies

Based on backtesting results:

| Strategy | ROI | P(profit) | Description |
|----------|-----|-----------|-------------|
| **Asian Handicap** | +20.8% | 97% | Heavy favorites, bet away when edge > 0.3 |
| **BTTS** | +16.0% | 98% | Both teams to score, CatBoost >= 0.6 |
| **Away Win** | +14.5% | 100% | CatBoost >= 0.45 |

## MLflow Tracking

Models and experiments are tracked with MLflow:

```bash
# View MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Access at: http://localhost:5000

## Configuration Files

- `config/strategies.yaml` - Betting strategies and thresholds
- `config/prod.yaml` - Production configuration
- `config/{league}.yaml` - League-specific settings

## GitHub Actions Automation

Two workflows are configured:

1. **Data Pipeline** (Weekly + Manual)
   - Collects match data from API
   - Preprocesses and generates features
   - Fetches betting odds
   - Uploads to Hugging Face

2. **ML Pipeline** (After data pipeline + Manual)
   - Trains models with Optuna optimization
   - Registers models to MLflow
   - Generates predictions

## Pilot Mode Settings

From `config/strategies.yaml`:

```yaml
pilot:
  paper_trading: true       # No real money
  min_bets_before_live: 100 # Validate first
  performance_threshold: 0.05 # Need 5% ROI
```

## Example Commands

```bash
# Train specific bet types
python entrypoints/run_pipeline.py --mode train \
    --data data/03-features/features_all_5leagues_with_odds.csv \
    --bet-types asian_handicap away_win

# Train with fewer Optuna trials (faster)
python -c "
from src.pipelines.betting_training_pipeline import *
config = TrainingConfig(
    data_path='data/03-features/features_all_5leagues_with_odds.csv',
    strategies_path='config/strategies.yaml',
    output_dir='outputs/training',
    n_optuna_trials=20  # Quick test
)
pipeline = BettingTrainingPipeline(config)
pipeline.run(['away_win'])
"

# Check MLflow for best callibration
python -c "
from src.ml.mlflow_config import get_mlflow_manager
mgr = get_mlflow_manager()
for bt in ['asian_handicap', 'away_win', 'btts']:
    run = mgr.get_best_run(metric='best_roi')
    if run:
        print(f'{bt}: ROI={run.data.metrics.get(\"best_roi\", 0):.1f}%')
"
```

## Output Files

- `outputs/training/training_summary.json` - Training results
- `outputs/recommendations/latest.json` - Betting recommendations
- `mlflow.db` - MLflow tracking database
- `mlruns/` - MLflow artifacts

## Monitoring

Key metrics to monitor during pilot:

1. **ROI** - Should be positive for enabled strategies
2. **P(profit)** - Confidence level (>70% recommended)
3. **Bet count** - Sufficient sample size
4. **Edge** - Predicted probability vs implied odds

## Risk Management

From `config/strategies.yaml`:

```yaml
risk_management:
  max_daily_bets: 10
  max_stake_per_bet: 0.05  # 5% of bankroll
  stop_loss_daily: 0.10    # Stop if 10% down
  take_profit_daily: 0.20  # Lock in if 20% up
```

## Troubleshooting

### No features file found
```bash
# Regenerate features
python entrypoints/run_pipeline.py --mode features
```

### MLflow errors
```bash
# Reset MLflow
rm mlflow.db
rm -rf mlruns/
python entrypoints/run_pipeline.py --mode train ...
```

### Model loading fails
```bash
# Check registered callibration
python -c "
from mlflow.tracking import MlflowClient
client = MlflowClient()
for rm in client.search_registered_models():
    print(rm.name)
"
```
