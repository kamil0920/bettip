#!/usr/bin/env python3
"""
Full Betting Analysis Pipeline
==============================
Comprehensive ML analysis for sports betting with proper validation,
data quality checks, and betting simulation.

Handles:
- 7 betting markets (BTTS, Away Win, Corners, Cards, Shots, Fouls, Asian Handicap)
- Walk-forward validation with 5 expanding folds
- Fractional Kelly staking (2%)
- Bootstrap confidence intervals
- Calibration analysis
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import json
from collections import defaultdict

# ML
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    roc_auc_score, log_loss, brier_score_loss,
    mean_squared_error, mean_absolute_error
)
import lightgbm as lgb
import xgboost as xgb

np.random.seed(42)

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class BettingConfig:
    """Global betting configuration."""
    kelly_fraction: float = 0.02
    min_odds: float = 1.30
    max_odds: float = 10.0
    max_stake_pct: float = 0.05
    daily_stop_loss: float = 0.10
    edge_thresholds: List[float] = field(default_factory=lambda: [0.02, 0.03, 0.05, 0.07, 0.10])
    min_bets_for_valid: int = 30

CONFIG = BettingConfig()

# Post-match columns that MUST be excluded from features
POST_MATCH_COLUMNS = [
    'match_result', 'home_win', 'draw', 'away_win', 'result',
    'total_goals', 'goal_difference', 'home_goals', 'away_goals',
    'btts', 'goal_margin', 'gd_form_diff',
    'total_corners', 'home_corners', 'away_corners',
    'total_cards', 'home_cards', 'away_cards',
    'total_shots', 'home_shots', 'away_shots',
    'home_shots_on_target', 'away_shots_on_target',
    'total_fouls', 'home_fouls', 'away_fouls',
    'corners_o10', 'corners_o9', 'cards_o4', 'shots_o24', 'fouls_o26',
]

ID_COLUMNS = [
    'fixture_id', 'date', 'home_team_id', 'home_team_name', 'away_team_id',
    'away_team_name', 'round', 'league', 'sm_fixture_id', 'round_number'
]

# =============================================================================
# MODEL DEFINITIONS
# =============================================================================

MODELS = {
    'away_win': {
        'name': 'Away Win',
        'target': 'away_win',
        'type': 'binary',
        'odds_col': 'b365_away_open',
        'min_samples': 500,
        'description': 'Predict away team victories'
    },
    'corners_o10': {
        'name': 'Corners O10.5',
        'target_col': 'corners_o10',
        'target_source': 'total_corners',
        'line': 10.5,
        'type': 'binary',
        'odds_col': 'sm_corners_over_odds',
        'min_samples': 500,
        'description': 'Predict total corners over 10.5'
    },
    'corners_o9': {
        'name': 'Corners O9.5',
        'target_col': 'corners_o9',
        'target_source': 'total_corners',
        'line': 9.5,
        'type': 'binary',
        'odds_col': 'sm_corners_over_odds',
        'min_samples': 500,
        'description': 'Predict total corners over 9.5'
    },
    'cards_o4': {
        'name': 'Cards O4.5',
        'target_col': 'cards_o4',
        'target_source': 'total_cards',
        'line': 4.5,
        'type': 'binary',
        'odds_col': 'sm_cards_over_odds',
        'min_samples': 500,
        'description': 'Predict total cards over 4.5'
    },
    'fouls_o26': {
        'name': 'Fouls O26.5',
        'target_col': 'fouls_o26',
        'target_source': 'total_fouls',
        'line': 26.5,
        'type': 'binary',
        'odds_col': None,  # No odds available
        'min_samples': 500,
        'description': 'Predict total fouls over 26.5'
    },
    'shots_o24': {
        'name': 'Shots O24.5',
        'target_col': 'shots_o24',
        'target_source': 'total_shots',
        'line': 24.5,
        'type': 'binary',
        'odds_col': 'sm_shots_over_odds',
        'min_samples': 100,  # Lower due to limited data
        'description': 'Predict total shots over 24.5'
    },
}

# Note: BTTS excluded due to corrupted odds data (overround=0.44)

# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(filepath: str) -> pd.DataFrame:
    """Load and prepare dataset."""
    df = pd.read_csv(filepath, low_memory=False)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # Create binary targets
    if 'total_corners' in df.columns:
        df['corners_o10'] = (df['total_corners'] > 10.5).astype(float)
        df['corners_o9'] = (df['total_corners'] > 9.5).astype(float)
        df.loc[df['total_corners'].isna(), ['corners_o10', 'corners_o9']] = np.nan

    if 'total_cards' in df.columns:
        df['cards_o4'] = (df['total_cards'] > 4.5).astype(float)
        df.loc[df['total_cards'].isna(), 'cards_o4'] = np.nan

    if 'total_shots' in df.columns:
        df['shots_o24'] = (df['total_shots'] > 24.5).astype(float)
        df.loc[df['total_shots'].isna(), 'shots_o24'] = np.nan

    if 'total_fouls' in df.columns:
        df['fouls_o26'] = (df['total_fouls'] > 26.5).astype(float)
        df.loc[df['total_fouls'].isna(), 'fouls_o26'] = np.nan

    return df


def get_features(df: pd.DataFrame) -> List[str]:
    """Get safe feature columns."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    exclude = set(POST_MATCH_COLUMNS + ID_COLUMNS)

    # Remove constant columns
    const_cols = [c for c in numeric_cols if df[c].nunique() <= 1]
    exclude.update(const_cols)

    features = [c for c in numeric_cols if c not in exclude]

    # Remove high-missing columns
    features = [c for c in features if df[c].notna().mean() > 0.5]

    return features


# =============================================================================
# VALIDATION
# =============================================================================

def create_walk_forward_splits(n: int, n_folds: int = 5, train_pct: float = 0.60,
                                min_test: int = 30) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Create walk-forward expanding window splits."""
    initial_train = int(n * train_pct)
    remaining = n - initial_train
    fold_size = remaining // n_folds

    splits = []
    for i in range(n_folds):
        train_end = initial_train + i * fold_size
        test_start = train_end
        test_end = min(test_start + fold_size, n)

        if test_end - test_start < min_test:
            continue

        splits.append((np.arange(0, train_end), np.arange(test_start, test_end)))

    return splits


# =============================================================================
# METRICS
# =============================================================================

def calc_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Calculate Expected Calibration Error."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i+1])
        if mask.sum() > 0:
            avg_pred = y_prob[mask].mean()
            avg_true = y_true[mask].mean()
            ece += mask.sum() * np.abs(avg_pred - avg_true)
    return ece / len(y_true) if len(y_true) > 0 else 0


def calc_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict:
    """Calculate ML metrics."""
    mask = ~(np.isnan(y_true) | np.isnan(y_prob))
    y_true, y_prob = y_true[mask], y_prob[mask]

    if len(y_true) < 10:
        return {'error': 'Insufficient samples'}

    return {
        'auc': roc_auc_score(y_true, y_prob),
        'logloss': log_loss(y_true, np.clip(y_prob, 1e-7, 1-1e-7)),
        'brier': brier_score_loss(y_true, y_prob),
        'ece': calc_ece(y_true, y_prob),
        'n': len(y_true),
        'base_rate': y_true.mean()
    }


# =============================================================================
# BETTING SIMULATION
# =============================================================================

def kelly_stake(p_model: float, odds: float) -> float:
    """Calculate fractional Kelly stake."""
    if odds <= 1 or pd.isna(odds) or p_model <= 0 or p_model >= 1:
        return 0.0

    q = 1 - p_model
    b = odds - 1
    kelly = (b * p_model - q) / b

    if kelly <= 0:
        return 0.0

    return min(kelly * CONFIG.kelly_fraction, CONFIG.max_stake_pct)


def simulate_betting(y_true: np.ndarray, y_prob: np.ndarray, odds: np.ndarray,
                     edge_threshold: float) -> Dict:
    """Simulate betting with Kelly staking."""
    results = {
        'n_bets': 0, 'n_wins': 0, 'total_staked': 0.0, 'total_return': 0.0,
        'returns': [], 'max_dd': 0.0
    }

    bankroll = 1.0
    peak = 1.0

    for i in range(len(y_true)):
        if pd.isna(odds[i]) or odds[i] < CONFIG.min_odds or odds[i] > CONFIG.max_odds:
            continue
        if pd.isna(y_prob[i]) or pd.isna(y_true[i]):
            continue

        p_implied = 1 / odds[i]
        edge = y_prob[i] - p_implied

        if edge < edge_threshold:
            continue

        stake = kelly_stake(y_prob[i], odds[i])
        if stake <= 0:
            continue

        stake_amount = bankroll * stake
        results['n_bets'] += 1
        results['total_staked'] += stake_amount

        if y_true[i] == 1:
            profit = stake_amount * (odds[i] - 1)
            results['n_wins'] += 1
        else:
            profit = -stake_amount

        results['total_return'] += profit
        bankroll += profit
        results['returns'].append(profit)

        peak = max(peak, bankroll)
        results['max_dd'] = max(results['max_dd'], (peak - bankroll) / peak)

    # Summary stats
    if results['n_bets'] > 0:
        results['roi'] = results['total_return'] / results['total_staked'] * 100
        results['win_rate'] = results['n_wins'] / results['n_bets'] * 100

        returns = np.array(results['returns'])
        if len(returns) > 1 and returns.std() > 0:
            results['sharpe'] = (returns.mean() / returns.std()) * np.sqrt(252)
            neg = returns[returns < 0]
            results['sortino'] = (returns.mean() / neg.std()) * np.sqrt(252) if len(neg) > 0 else 0
        else:
            results['sharpe'] = results['sortino'] = 0
    else:
        results['roi'] = results['win_rate'] = results['sharpe'] = results['sortino'] = 0

    return results


def bootstrap_ci(y_true, y_prob, odds, threshold, n_boot=500, ci=0.95):
    """Bootstrap confidence interval for ROI."""
    rois = []
    n = len(y_true)

    for _ in range(n_boot):
        idx = np.random.choice(n, size=n, replace=True)
        res = simulate_betting(y_true[idx], y_prob[idx], odds[idx], threshold)
        if res['n_bets'] >= 10:
            rois.append(res['roi'])

    if len(rois) < 10:
        return (np.nan, np.nan)

    alpha = (1 - ci) / 2
    return (np.percentile(rois, alpha * 100), np.percentile(rois, (1 - alpha) * 100))


# =============================================================================
# MODEL TRAINING
# =============================================================================

def train_lgb(X_train, y_train, X_val, y_val) -> Tuple[Any, Dict]:
    """Train LightGBM classifier."""
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42
    }

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        params, train_data,
        num_boost_round=500,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )

    importance = dict(zip(X_train.columns, model.feature_importance(importance_type='gain')))
    return model, importance


def calibrate(y_cal, probs_cal, probs_test):
    """Isotonic calibration."""
    iso = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds='clip')
    iso.fit(probs_cal, y_cal)
    return iso.predict(probs_test)


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def analyze_model(df: pd.DataFrame, model_key: str, features: List[str]) -> Dict:
    """Run full analysis for a single model."""

    config = MODELS[model_key]
    print(f"\n{'='*60}")
    print(f"MODEL: {config['name']}")
    print(f"{'='*60}")

    # Get target column
    target_col = config.get('target_col', config.get('target'))
    if target_col not in df.columns:
        return {'error': f'Target column {target_col} not found'}

    # Filter to valid rows
    df_model = df[df[target_col].notna()].copy()
    print(f"  Samples with target: {len(df_model):,}")

    if len(df_model) < config['min_samples']:
        return {'error': f'Insufficient samples: {len(df_model)} < {config["min_samples"]}'}

    # Prepare features
    available_features = [f for f in features if f in df_model.columns]
    df_model = df_model.dropna(subset=available_features, thresh=int(len(available_features) * 0.5))

    for f in available_features:
        df_model[f] = df_model[f].fillna(df_model[f].median())

    print(f"  Features: {len(available_features)}")
    print(f"  Final samples: {len(df_model):,}")

    # Walk-forward CV
    splits = create_walk_forward_splits(len(df_model), n_folds=5)
    print(f"  Walk-forward folds: {len(splits)}")

    all_preds, all_true, all_odds = [], [], []
    fold_metrics = []
    feature_importance = defaultdict(float)

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        X_train = df_model.iloc[train_idx][available_features]
        y_train = df_model.iloc[train_idx][target_col].values
        X_test = df_model.iloc[test_idx][available_features]
        y_test = df_model.iloc[test_idx][target_col].values

        # Split train into train/val
        val_size = int(len(train_idx) * 0.15)
        X_tr, y_tr = X_train.iloc[:-val_size], y_train[:-val_size]
        X_val, y_val = X_train.iloc[-val_size:], y_train[-val_size:]

        # Train
        model, importance = train_lgb(X_tr, y_tr, X_val, y_val)

        for k, v in importance.items():
            feature_importance[k] += v

        # Predict and calibrate
        y_prob_val = model.predict(X_val)
        y_prob_test = model.predict(X_test)
        y_prob_cal = calibrate(y_val, y_prob_val, y_prob_test)

        # Metrics
        metrics = calc_metrics(y_test, y_prob_cal)
        metrics['fold'] = fold_idx + 1
        fold_metrics.append(metrics)

        # Store for betting
        all_preds.extend(y_prob_cal)
        all_true.extend(y_test)

        if config['odds_col'] and config['odds_col'] in df_model.columns:
            all_odds.extend(df_model.iloc[test_idx][config['odds_col']].values)
        else:
            all_odds.extend([np.nan] * len(test_idx))

        print(f"  Fold {fold_idx+1}: AUC={metrics.get('auc', 0):.3f}, ECE={metrics.get('ece', 0):.3f}")

    # Convert to arrays
    all_preds = np.array(all_preds)
    all_true = np.array(all_true)
    all_odds = np.array(all_odds)

    # Overall metrics
    overall = calc_metrics(all_true, all_preds)

    # Betting simulation
    betting = {}
    best_roi = -np.inf
    best_threshold = None

    if config['odds_col']:
        for threshold in CONFIG.edge_thresholds:
            result = simulate_betting(all_true, all_preds, all_odds, threshold)
            betting[f'edge_{threshold}'] = result

            if result['n_bets'] >= CONFIG.min_bets_for_valid and result['roi'] > best_roi:
                best_roi = result['roi']
                best_threshold = threshold

        if best_threshold:
            ci = bootstrap_ci(all_true, all_preds, all_odds, best_threshold)
            betting['best'] = {
                'threshold': best_threshold,
                'roi': best_roi,
                'ci_95': ci
            }

    # Top features
    top_features = dict(sorted(feature_importance.items(), key=lambda x: -x[1])[:20])

    return {
        'model': config['name'],
        'description': config['description'],
        'n_samples': len(df_model),
        'target': target_col,
        'fold_metrics': fold_metrics,
        'overall': overall,
        'betting': betting,
        'top_features': top_features
    }


def run_full_analysis(data_path: str, output_dir: str = 'experiments/outputs/full_analysis'):
    """Run complete analysis pipeline."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("FULL BETTING ANALYSIS PIPELINE")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    df = load_data(data_path)
    print(f"Loaded {len(df):,} matches ({df['date'].min().date()} to {df['date'].max().date()})")

    # Get features
    features = get_features(df)
    print(f"Safe features: {len(features)}")

    # Run analysis for each model
    results = {}
    for model_key in MODELS:
        try:
            results[model_key] = analyze_model(df, model_key, features)
        except Exception as e:
            print(f"  ERROR: {e}")
            results[model_key] = {'error': str(e)}

    # Save results
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(convert(v) for v in obj)
        return obj

    with open(output_path / 'results.json', 'w') as f:
        json.dump(convert(results), f, indent=2)

    # Print summary
    print("\n" + "=" * 110)
    print("RESULTS SUMMARY")
    print("=" * 110)
    print(f"{'Model':<18} {'AUC':<7} {'Brier':<7} {'ECE':<7} {'Threshold':<10} {'Bets':<7} {'Win%':<7} {'ROI%':<10} {'Sharpe':<8} {'95% CI':<20}")
    print("-" * 110)

    summary_rows = []
    for model_key, result in results.items():
        if 'error' in result:
            print(f"{model_key:<18} ERROR: {result['error']}")
            continue

        m = result['overall']
        b = result.get('betting', {})
        best = b.get('best', {})

        if best:
            th = best['threshold']
            stats = b.get(f'edge_{th}', {})
            ci = best.get('ci_95', (np.nan, np.nan))
            ci_str = f"[{ci[0]:.1f}, {ci[1]:.1f}]" if not np.isnan(ci[0]) else "N/A"
        else:
            th = 'N/A'
            stats = {}
            ci_str = 'N/A'

        row = {
            'model': result['model'],
            'auc': m.get('auc', 0),
            'brier': m.get('brier', 0),
            'ece': m.get('ece', 0),
            'threshold': th,
            'n_bets': stats.get('n_bets', 0),
            'win_rate': stats.get('win_rate', 0),
            'roi': stats.get('roi', 0),
            'sharpe': stats.get('sharpe', 0),
            'ci': ci_str
        }
        summary_rows.append(row)

        print(f"{row['model']:<18} {row['auc']:<7.3f} {row['brier']:<7.3f} {row['ece']:<7.3f} "
              f"{str(row['threshold']):<10} {row['n_bets']:<7} {row['win_rate']:<7.1f} "
              f"{row['roi']:<10.1f} {row['sharpe']:<8.2f} {row['ci']:<20}")

    # Rankings
    print("\n" + "=" * 60)
    print("RANKINGS")
    print("=" * 60)

    # By ROI
    profitable = [r for r in summary_rows if r['roi'] > 0 and r['n_bets'] >= 30]
    if profitable:
        by_roi = sorted(profitable, key=lambda x: -x['roi'])
        print("\nTop by ROI (min 30 bets):")
        for i, r in enumerate(by_roi[:3], 1):
            print(f"  {i}. {r['model']}: {r['roi']:.1f}% ROI ({r['n_bets']} bets)")

    # By Sharpe
    with_sharpe = [r for r in summary_rows if r['sharpe'] > 0 and r['n_bets'] >= 30]
    if with_sharpe:
        by_sharpe = sorted(with_sharpe, key=lambda x: -x['sharpe'])
        print("\nTop by Sharpe (min 30 bets):")
        for i, r in enumerate(by_sharpe[:3], 1):
            print(f"  {i}. {r['model']}: {r['sharpe']:.2f} Sharpe ({r['roi']:.1f}% ROI)")

    # By AUC
    by_auc = sorted(summary_rows, key=lambda x: -x['auc'])
    print("\nTop by AUC:")
    for i, r in enumerate(by_auc[:3], 1):
        print(f"  {i}. {r['model']}: {r['auc']:.3f} AUC")

    print(f"\nResults saved to {output_path / 'results.json'}")

    return results


if __name__ == '__main__':
    import sys
    data_path = sys.argv[1] if len(sys.argv) > 1 else 'data/03-features/features_with_sportmonks_odds.csv'
    run_full_analysis(data_path)
