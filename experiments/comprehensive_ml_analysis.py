#!/usr/bin/env python3
"""
Comprehensive ML Analysis Pipeline for Sports Betting
=====================================================
Implements 5-step analysis: Data Audit → Baselines → Betting Layer →
Iterative Improvements → Comparison & Recommendations

Walk-forward validation with 5 expanding folds, fractional Kelly staking.
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

# ML
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    roc_auc_score, log_loss, brier_score_loss,
    mean_squared_error, mean_absolute_error
)
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import lightgbm as lgb
import xgboost as xgb

# Feature selection
try:
    from arfs.feature_selection import BorutaShap
    HAS_ARFS = True
except ImportError:
    HAS_ARFS = False

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ModelConfig:
    """Configuration for a betting model."""
    name: str
    target: str
    target_type: str  # 'binary' or 'regression'
    line: Optional[float] = None  # For over/under markets
    odds_col: Optional[str] = None  # For implied probability
    min_samples: int = 500  # Minimum samples required

# Define all 7 models
# Note: Use actual decimal odds columns, not probability columns
MODEL_CONFIGS = {
    'btts': ModelConfig('BTTS', 'btts', 'binary', odds_col='sm_btts_yes_odds'),
    'away_win': ModelConfig('Away Win', 'away_win', 'binary', odds_col='b365_away_open'),  # B365 decimal odds
    'corners_o10': ModelConfig('Corners O10.5', 'total_corners', 'binary', line=10.5, odds_col='sm_corners_over_odds'),
    'corners_o9': ModelConfig('Corners O9.5', 'total_corners', 'binary', line=9.5, odds_col='sm_corners_over_odds'),
    'cards_o4': ModelConfig('Cards O4.5', 'total_cards', 'binary', line=4.5, odds_col='sm_cards_over_odds'),
    'shots_o24': ModelConfig('Shots O24.5', 'total_shots', 'binary', line=24.5, odds_col='sm_shots_over_odds'),
    'fouls_o26': ModelConfig('Fouls O26.5', 'total_fouls', 'binary', line=26.5),  # No odds available
    'asian_handicap': ModelConfig('Asian Handicap', 'goal_margin', 'regression', odds_col='ah_line'),
}

# Leaky columns to ALWAYS exclude
LEAKY_COLUMNS = [
    'match_result', 'home_win', 'draw', 'away_win', 'total_goals', 'goal_difference',
    'home_goals', 'away_goals', 'result', 'btts', 'total_corners', 'total_cards',
    'total_shots', 'total_fouls', 'home_shots', 'away_shots', 'home_shots_on_target',
    'away_shots_on_target', 'home_cards', 'away_cards', 'gd_form_diff'
]

ID_COLUMNS = [
    'fixture_id', 'date', 'home_team_id', 'home_team_name', 'away_team_id',
    'away_team_name', 'round', 'league', 'sm_fixture_id', 'round_number'
]

# Betting parameters
KELLY_FRACTION = 0.02  # 2% fractional Kelly
MIN_ODDS = 1.30
MAX_ODDS = 10.0
MAX_STAKE_PCT = 0.05  # 5% max per bet
EDGE_THRESHOLDS = [0.02, 0.03, 0.05, 0.07, 0.10]  # Sweep these

# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================

def load_and_prepare_data(filepath: str) -> pd.DataFrame:
    """Load dataset and prepare for analysis."""
    df = pd.read_csv(filepath, low_memory=False)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # Create goal_margin for Asian Handicap
    if 'home_goals' in df.columns and 'away_goals' in df.columns:
        df['goal_margin'] = df['home_goals'] - df['away_goals']

    # Create binary targets for over/under markets
    if 'total_corners' in df.columns:
        df['corners_o10'] = (df['total_corners'] > 10.5).astype(float)
        df['corners_o9'] = (df['total_corners'] > 9.5).astype(float)
    if 'total_cards' in df.columns:
        df['cards_o4'] = (df['total_cards'] > 4.5).astype(float)
    if 'total_shots' in df.columns:
        df['shots_o24'] = (df['total_shots'] > 24.5).astype(float)
    if 'total_fouls' in df.columns:
        df['fouls_o26'] = (df['total_fouls'] > 26.5).astype(float)

    return df


def get_safe_features(df: pd.DataFrame, model_key: str) -> List[str]:
    """Get safe features for a model, excluding leaky and target-related columns."""
    config = MODEL_CONFIGS[model_key]

    # Get numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # CRITICAL: All post-match outcomes must be excluded
    # These are known at match end, not pre-match
    post_match_outcomes = [
        # Direct match results
        'match_result', 'home_win', 'draw', 'away_win', 'result',
        'total_goals', 'goal_difference', 'home_goals', 'away_goals',
        'btts', 'goal_margin', 'gd_form_diff',
        # Match statistics (only known after match)
        'total_corners', 'home_corners', 'away_corners',
        'total_cards', 'home_cards', 'away_cards',
        'total_shots', 'home_shots', 'away_shots',
        'home_shots_on_target', 'away_shots_on_target',
        'total_fouls', 'home_fouls', 'away_fouls',
        # Derived binary targets
        'corners_o10', 'corners_o9', 'cards_o4', 'shots_o24', 'fouls_o26',
    ]

    # Columns to exclude
    exclude = set(post_match_outcomes + ID_COLUMNS + LEAKY_COLUMNS)

    # Remove constant columns
    const_cols = [c for c in numeric_cols if df[c].nunique() <= 1]
    exclude.update(const_cols)

    safe = [c for c in numeric_cols if c not in exclude]

    # Remove high-missing columns (>50% missing)
    safe = [c for c in safe if df[c].notna().mean() > 0.5]

    return safe


def create_walk_forward_splits(
    df: pd.DataFrame,
    n_folds: int = 5,
    train_pct: float = 0.60,
    min_test_samples: int = 30
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create walk-forward expanding window splits.

    Initial training: first 60% of data
    Then expand training window while sliding test window forward.
    """
    n = len(df)
    initial_train_size = int(n * train_pct)
    remaining = n - initial_train_size
    test_fold_size = remaining // n_folds

    splits = []
    for i in range(n_folds):
        train_end = initial_train_size + i * test_fold_size
        test_start = train_end
        test_end = min(test_start + test_fold_size, n)

        if test_end - test_start < min_test_samples:
            continue

        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)
        splits.append((train_idx, test_idx))

    return splits


# =============================================================================
# METRICS
# =============================================================================

def calculate_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Calculate Expected Calibration Error."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i+1])
        if mask.sum() > 0:
            avg_pred = y_prob[mask].mean()
            avg_true = y_true[mask].mean()
            ece += mask.sum() * np.abs(avg_pred - avg_true)
    return ece / len(y_true)


def calculate_metrics(y_true: np.ndarray, y_prob: np.ndarray, is_binary: bool = True) -> Dict:
    """Calculate comprehensive metrics."""
    metrics = {}

    if is_binary:
        # Handle NaN values
        mask = ~(np.isnan(y_true) | np.isnan(y_prob))
        y_true = y_true[mask]
        y_prob = y_prob[mask]

        if len(y_true) < 10:
            return {'error': 'Insufficient samples'}

        metrics['auc'] = roc_auc_score(y_true, y_prob)
        metrics['logloss'] = log_loss(y_true, np.clip(y_prob, 1e-7, 1-1e-7))
        metrics['brier'] = brier_score_loss(y_true, y_prob)
        metrics['ece'] = calculate_ece(y_true, y_prob)
        metrics['n_samples'] = len(y_true)
        metrics['base_rate'] = y_true.mean()
    else:
        # Regression metrics
        mask = ~(np.isnan(y_true) | np.isnan(y_prob))
        y_true = y_true[mask]
        y_prob = y_prob[mask]

        metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_prob))
        metrics['mae'] = mean_absolute_error(y_true, y_prob)
        metrics['n_samples'] = len(y_true)

    return metrics


# =============================================================================
# BETTING SIMULATION
# =============================================================================

def implied_probability(odds: float) -> float:
    """Convert decimal odds to implied probability."""
    if odds <= 1 or pd.isna(odds):
        return np.nan
    return 1 / odds


def kelly_stake(p_model: float, odds: float, fraction: float = KELLY_FRACTION) -> float:
    """Calculate fractional Kelly stake."""
    if odds <= 1 or pd.isna(odds) or p_model <= 0 or p_model >= 1:
        return 0.0

    q = 1 - p_model
    b = odds - 1  # Net odds

    # Kelly formula: (bp - q) / b
    kelly = (b * p_model - q) / b

    if kelly <= 0:
        return 0.0

    # Apply fraction and cap
    stake = min(kelly * fraction, MAX_STAKE_PCT)
    return stake


def simulate_betting(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    odds: np.ndarray,
    edge_threshold: float = 0.03
) -> Dict:
    """
    Simulate betting with Kelly staking.

    Returns detailed backtest metrics.
    """
    results = {
        'n_bets': 0,
        'n_wins': 0,
        'total_staked': 0.0,
        'total_return': 0.0,
        'max_drawdown': 0.0,
        'returns': []
    }

    bankroll = 1.0
    peak_bankroll = 1.0
    max_dd = 0.0

    for i in range(len(y_true)):
        if pd.isna(odds[i]) or odds[i] < MIN_ODDS or odds[i] > MAX_ODDS:
            continue
        if pd.isna(y_prob[i]) or pd.isna(y_true[i]):
            continue

        p_implied = implied_probability(odds[i])
        edge = y_prob[i] - p_implied

        if edge < edge_threshold:
            continue

        stake = kelly_stake(y_prob[i], odds[i])
        if stake <= 0:
            continue

        stake_amount = bankroll * stake
        results['n_bets'] += 1
        results['total_staked'] += stake_amount

        # Determine outcome
        won = y_true[i] == 1
        if won:
            profit = stake_amount * (odds[i] - 1)
            results['n_wins'] += 1
        else:
            profit = -stake_amount

        results['total_return'] += profit
        bankroll += profit
        results['returns'].append(profit)

        # Track drawdown
        peak_bankroll = max(peak_bankroll, bankroll)
        dd = (peak_bankroll - bankroll) / peak_bankroll
        max_dd = max(max_dd, dd)

    results['max_drawdown'] = max_dd

    # Calculate summary stats
    if results['n_bets'] > 0:
        results['roi'] = results['total_return'] / results['total_staked'] * 100
        results['win_rate'] = results['n_wins'] / results['n_bets'] * 100
        results['avg_odds'] = np.mean([o for o in odds if not pd.isna(o) and MIN_ODDS <= o <= MAX_ODDS])

        # Sharpe ratio (annualized assuming daily bets)
        returns = np.array(results['returns'])
        if len(returns) > 1:
            results['sharpe'] = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
            results['sortino'] = (returns.mean() / returns[returns < 0].std()) * np.sqrt(252) if (returns < 0).sum() > 0 else 0
        else:
            results['sharpe'] = 0
            results['sortino'] = 0
    else:
        results['roi'] = 0
        results['win_rate'] = 0
        results['sharpe'] = 0
        results['sortino'] = 0

    return results


def bootstrap_roi_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    odds: np.ndarray,
    edge_threshold: float,
    n_bootstrap: int = 1000,
    ci: float = 0.95
) -> Tuple[float, float]:
    """Calculate bootstrap confidence interval for ROI."""
    rois = []
    n = len(y_true)

    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        result = simulate_betting(y_true[idx], y_prob[idx], odds[idx], edge_threshold)
        if result['n_bets'] > 0:
            rois.append(result['roi'])

    if len(rois) < 10:
        return (np.nan, np.nan)

    alpha = (1 - ci) / 2
    lower = np.percentile(rois, alpha * 100)
    upper = np.percentile(rois, (1 - alpha) * 100)
    return (lower, upper)


# =============================================================================
# MODEL TRAINING
# =============================================================================

def train_lgb_baseline(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    is_binary: bool = True
) -> Tuple[Any, Dict]:
    """Train LightGBM baseline with early stopping."""

    params = {
        'objective': 'binary' if is_binary else 'regression',
        'metric': 'auc' if is_binary else 'rmse',
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
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )

    # Get feature importance
    importance = dict(zip(X_train.columns, model.feature_importance(importance_type='gain')))

    return model, importance


def calibrate_probabilities(
    y_cal: np.ndarray,
    probs_cal: np.ndarray,
    probs_test: np.ndarray,
    method: str = 'isotonic'
) -> np.ndarray:
    """Calibrate probabilities using isotonic regression or Platt scaling."""
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression as LR

    if method == 'isotonic':
        # Isotonic regression
        iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
        iso.fit(probs_cal, y_cal)
        return iso.predict(probs_test)
    else:
        # Platt scaling
        lr = LR(solver='lbfgs')
        lr.fit(probs_cal.reshape(-1, 1), y_cal)
        return lr.predict_proba(probs_test.reshape(-1, 1))[:, 1]


# =============================================================================
# MAIN ANALYSIS FUNCTIONS
# =============================================================================

def run_baseline_analysis(
    df: pd.DataFrame,
    model_key: str,
    output_dir: Path
) -> Dict:
    """Run baseline analysis with walk-forward CV for a single model."""

    config = MODEL_CONFIGS[model_key]
    print(f"\n{'='*60}")
    print(f"BASELINE: {config.name}")
    print(f"{'='*60}")

    # Determine target column
    if config.line is not None:
        target_col = f"{config.target.replace('total_', '')}_o{int(config.line) if config.line == int(config.line) else config.line}".replace('.', '')
        if target_col not in df.columns:
            # Create it
            target_col = model_key
    else:
        target_col = config.target

    # Filter to rows with valid target
    if target_col not in df.columns:
        print(f"  ERROR: Target column '{target_col}' not found")
        return {'error': f'Target column not found: {target_col}'}

    df_model = df[df[target_col].notna()].copy()
    print(f"  Samples with target: {len(df_model):,}")

    if len(df_model) < config.min_samples:
        print(f"  ERROR: Insufficient samples ({len(df_model)} < {config.min_samples})")
        return {'error': 'Insufficient samples'}

    # Get safe features
    features = get_safe_features(df_model, model_key)
    print(f"  Features: {len(features)}")

    # Handle missing features
    df_model = df_model.dropna(subset=features, thresh=int(len(features) * 0.5))
    for f in features:
        df_model[f] = df_model[f].fillna(df_model[f].median())

    print(f"  Final samples: {len(df_model):,}")

    # Create walk-forward splits
    splits = create_walk_forward_splits(df_model, n_folds=5)
    print(f"  Walk-forward folds: {len(splits)}")

    # Run walk-forward validation
    fold_results = []
    all_preds = []
    all_true = []
    all_odds = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        X_train = df_model.iloc[train_idx][features]
        y_train = df_model.iloc[train_idx][target_col].values
        X_test = df_model.iloc[test_idx][features]
        y_test = df_model.iloc[test_idx][target_col].values

        # Split training into train/val for early stopping
        val_size = int(len(train_idx) * 0.15)
        X_tr = X_train.iloc[:-val_size]
        y_tr = y_train[:-val_size]
        X_val = X_train.iloc[-val_size:]
        y_val = y_train[-val_size:]

        # Train model
        is_binary = config.target_type == 'binary'
        model, importance = train_lgb_baseline(X_tr, y_tr, X_val, y_val, is_binary)

        # Get predictions
        if is_binary:
            y_prob = model.predict(X_test)
            y_prob_val = model.predict(X_val)

            # Calibrate using validation set
            y_prob_cal = calibrate_probabilities(y_val, y_prob_val, y_prob, method='isotonic')
        else:
            y_prob = model.predict(X_test)
            y_prob_cal = y_prob

        # Calculate metrics
        metrics = calculate_metrics(y_test, y_prob_cal, is_binary)
        metrics['fold'] = fold_idx + 1
        metrics['train_size'] = len(train_idx)
        metrics['test_size'] = len(test_idx)

        # Get odds for betting simulation
        if config.odds_col and config.odds_col in df_model.columns:
            test_odds = df_model.iloc[test_idx][config.odds_col].values
        else:
            test_odds = np.full(len(test_idx), np.nan)

        fold_results.append(metrics)
        all_preds.extend(y_prob_cal)
        all_true.extend(y_test)
        all_odds.extend(test_odds)

        print(f"  Fold {fold_idx + 1}: AUC={metrics.get('auc', 'N/A'):.3f}, "
              f"ECE={metrics.get('ece', 'N/A'):.3f}, n={len(test_idx)}")

    # Aggregate results
    all_preds = np.array(all_preds)
    all_true = np.array(all_true)
    all_odds = np.array(all_odds)

    # Overall metrics
    overall_metrics = calculate_metrics(all_true, all_preds, config.target_type == 'binary')

    # Betting simulation across different thresholds
    betting_results = {}
    best_roi = -np.inf
    best_threshold = None

    if config.odds_col:
        for threshold in EDGE_THRESHOLDS:
            bet_result = simulate_betting(all_true, all_preds, all_odds, threshold)
            betting_results[f'edge_{threshold}'] = bet_result

            if bet_result['n_bets'] >= 30 and bet_result['roi'] > best_roi:
                best_roi = bet_result['roi']
                best_threshold = threshold

        # Bootstrap CI for best threshold
        if best_threshold:
            ci_lower, ci_upper = bootstrap_roi_ci(
                all_true, all_preds, all_odds, best_threshold, n_bootstrap=500
            )
            betting_results['best'] = {
                'threshold': best_threshold,
                'roi': best_roi,
                'roi_ci_95': (ci_lower, ci_upper)
            }

    result = {
        'model': config.name,
        'target': target_col,
        'n_samples': len(df_model),
        'n_folds': len(splits),
        'fold_results': fold_results,
        'overall_metrics': overall_metrics,
        'betting_results': betting_results,
        'feature_importance': dict(sorted(importance.items(), key=lambda x: -x[1])[:30])
    }

    return result


def run_full_analysis(data_path: str, output_dir: str = 'experiments/outputs/comprehensive'):
    """Run full analysis pipeline for all models."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("COMPREHENSIVE ML ANALYSIS PIPELINE")
    print("=" * 80)
    print(f"Output directory: {output_path}")

    # Load data
    print("\nLoading data...")
    df = load_and_prepare_data(data_path)
    print(f"Loaded {len(df):,} matches from {df['date'].min().date()} to {df['date'].max().date()}")

    # Run analysis for each model
    results = {}

    # Models to run (skip shots due to insufficient data)
    models_to_run = ['btts', 'away_win', 'corners_o10', 'corners_o9',
                     'cards_o4', 'fouls_o26']  # Skip shots_o24, asian_handicap for now

    for model_key in models_to_run:
        try:
            result = run_baseline_analysis(df, model_key, output_path)
            results[model_key] = result
        except Exception as e:
            print(f"  ERROR: {e}")
            results[model_key] = {'error': str(e)}

    # Save results
    results_file = output_path / 'baseline_results.json'

    # Convert numpy types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(convert_types(v) for v in obj)
        return obj

    with open(results_file, 'w') as f:
        json.dump(convert_types(results), f, indent=2)

    print(f"\nResults saved to {results_file}")

    # Print summary table
    print("\n" + "=" * 100)
    print("BASELINE RESULTS SUMMARY")
    print("=" * 100)
    print(f"{'Model':<20} {'AUC':<8} {'Brier':<8} {'ECE':<8} {'Best Th':<10} {'Bets':<8} {'ROI%':<10} {'Win%':<8}")
    print("-" * 100)

    for model_key, result in results.items():
        if 'error' in result:
            print(f"{model_key:<20} ERROR: {result['error']}")
            continue

        metrics = result['overall_metrics']
        betting = result.get('betting_results', {})
        best = betting.get('best', {})
        best_th = best.get('threshold', 'N/A')

        # Get betting stats for best threshold
        if best_th != 'N/A':
            bet_stats = betting.get(f'edge_{best_th}', {})
            roi = bet_stats.get('roi', 0)
            n_bets = bet_stats.get('n_bets', 0)
            win_rate = bet_stats.get('win_rate', 0)
        else:
            roi, n_bets, win_rate = 0, 0, 0

        print(f"{result['model']:<20} "
              f"{metrics.get('auc', 0):<8.3f} "
              f"{metrics.get('brier', 0):<8.3f} "
              f"{metrics.get('ece', 0):<8.3f} "
              f"{best_th if best_th != 'N/A' else 'N/A':<10} "
              f"{n_bets:<8} "
              f"{roi:<10.1f} "
              f"{win_rate:<8.1f}")

    return results


if __name__ == '__main__':
    import sys

    data_path = sys.argv[1] if len(sys.argv) > 1 else 'data/03-features/features_with_sportmonks_odds.csv'
    results = run_full_analysis(data_path)
