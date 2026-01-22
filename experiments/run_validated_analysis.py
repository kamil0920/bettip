#!/usr/bin/env python3
"""
Validated ML Analysis Pipeline for Sports Betting
==================================================
This script implements the complete analysis with all fixes applied:
- Excludes corrupted BTTS odds (overround=0.44)
- Uses correct lines from sm_*_line columns
- Proper walk-forward validation with 5 expanding folds
- Kelly staking with fractional sizing (2%)

Run: python experiments/run_validated_analysis.py

Author: Validated analysis based on comprehensive audit
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
from collections import defaultdict

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

np.random.seed(42)

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class Config:
    """Analysis configuration."""
    kelly_fraction: float = 0.02
    min_odds: float = 1.30
    max_odds: float = 10.0
    max_stake_pct: float = 0.05
    edge_thresholds: List[float] = None
    min_bets: int = 30
    train_pct: float = 0.60
    n_folds: int = 5

    def __post_init__(self):
        if self.edge_thresholds is None:
            self.edge_thresholds = [0.02, 0.03, 0.05, 0.07, 0.10]

CONFIG = Config()

# Columns that leak post-match information
LEAKY_COLUMNS = [
    'match_result', 'home_win', 'draw', 'away_win', 'result',
    'total_goals', 'goal_difference', 'home_goals', 'away_goals',
    'btts', 'goal_margin', 'gd_form_diff',
    'total_corners', 'home_corners', 'away_corners',
    'total_cards', 'home_cards', 'away_cards',
    'total_shots', 'home_shots', 'away_shots',
    'home_shots_on_target', 'away_shots_on_target',
    'total_fouls', 'home_fouls', 'away_fouls',
]

ID_COLUMNS = [
    'fixture_id', 'date', 'home_team_id', 'home_team_name', 'away_team_id',
    'away_team_name', 'round', 'league', 'sm_fixture_id', 'round_number'
]

# Model definitions - ONLY models with valid odds
MODELS = {
    'away_win': {
        'name': 'Away Win',
        'target': 'away_win',
        'odds_col': 'b365_away_open',
        'line': None,
        'min_samples': 500,
    },
    'corners_o9.5': {
        'name': 'Corners O9.5',  # Matches sm_corners_line=9.5
        'target_source': 'total_corners',
        'line': 9.5,
        'odds_col': 'sm_corners_over_odds',
        'min_samples': 500,
    },
    'cards_o4.5': {
        'name': 'Cards O4.5',  # Matches sm_cards_line=4.5
        'target_source': 'total_cards',
        'line': 4.5,
        'odds_col': 'sm_cards_over_odds',
        'min_samples': 500,
    },
    'fouls_o26.5': {
        'name': 'Fouls O26.5',
        'target_source': 'total_fouls',
        'line': 26.5,
        'odds_col': None,  # No betting odds available
        'min_samples': 500,
    },
}

# NOTE: BTTS excluded (corrupted odds), Shots excluded (line mismatch)

# =============================================================================
# DATA LOADING & FEATURE ENGINEERING
# =============================================================================

def load_data(filepath: str) -> pd.DataFrame:
    """Load and prepare dataset with proper target creation."""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath, low_memory=False)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # Create binary targets for over/under markets
    for model_key, cfg in MODELS.items():
        if cfg.get('line') is not None:
            source = cfg['target_source']
            if source in df.columns:
                target_col = f"target_{model_key.replace('.', '_')}"
                df[target_col] = (df[source] > cfg['line']).astype(float)
                df.loc[df[source].isna(), target_col] = np.nan

    print(f"Loaded {len(df):,} matches from {df['date'].min().date()} to {df['date'].max().date()}")
    return df


def get_features(df: pd.DataFrame) -> List[str]:
    """Get safe features excluding leaky and ID columns."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Exclude leaky columns
    exclude = set(LEAKY_COLUMNS + ID_COLUMNS)

    # Also exclude derived target columns
    exclude.update([c for c in df.columns if c.startswith('target_')])

    # Remove constant columns
    const_cols = [c for c in numeric_cols if df[c].nunique() <= 1]
    exclude.update(const_cols)

    features = [c for c in numeric_cols if c not in exclude]

    # Remove high-missing columns (>50%)
    features = [c for c in features if df[c].notna().mean() > 0.5]

    return features


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cross-market interaction features."""
    df = df.copy()

    # ELO x Form interaction
    if 'elo_diff' in df.columns and 'home_wins_last_n' in df.columns:
        df['elo_x_form'] = df['elo_diff'] * (df['home_wins_last_n'] - df['away_wins_last_n'])

    # Rest x Position interaction
    if 'rest_days_diff' in df.columns and 'position_diff' in df.columns:
        df['rest_x_position'] = df['rest_days_diff'] * df['position_diff']

    # Odds confidence (how lopsided the market is)
    if 'odds_home_prob' in df.columns and 'odds_away_prob' in df.columns:
        df['odds_confidence'] = abs(df['odds_home_prob'] - df['odds_away_prob'])

    # Attack intensity (expected corners x shots)
    if 'expected_total_corners' in df.columns and 'expected_total_shots' in df.columns:
        df['attack_intensity'] = df['expected_total_corners'] * df['expected_total_shots']

    return df


# =============================================================================
# BETTING SIMULATION
# =============================================================================

def kelly_stake(p_model: float, odds: float, fraction: float = CONFIG.kelly_fraction,
                max_stake: float = CONFIG.max_stake_pct) -> float:
    """Calculate fractional Kelly stake."""
    if odds <= 1 or p_model <= 0 or p_model >= 1:
        return 0.0

    b = odds - 1
    kelly = (b * p_model - (1 - p_model)) / b

    if kelly <= 0:
        return 0.0

    return min(kelly * fraction, max_stake)


def backtest(y_true: np.ndarray, y_prob: np.ndarray, odds: np.ndarray,
             threshold: float) -> Dict:
    """Simulate betting with Kelly staking."""
    results = {
        'n_bets': 0, 'n_wins': 0,
        'total_staked': 0.0, 'total_return': 0.0,
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

        if edge < threshold:
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

    # Calculate summary stats
    if results['n_bets'] > 0:
        results['roi'] = results['total_return'] / results['total_staked'] * 100
        results['win_rate'] = results['n_wins'] / results['n_bets'] * 100

        returns = np.array(results['returns'])
        if len(returns) > 1 and returns.std() > 0:
            results['sharpe'] = (returns.mean() / returns.std()) * np.sqrt(252)
        else:
            results['sharpe'] = 0
    else:
        results['roi'] = results['win_rate'] = results['sharpe'] = 0

    return results


def bootstrap_ci(y_true: np.ndarray, y_prob: np.ndarray, odds: np.ndarray,
                 threshold: float, n_boot: int = 500, ci: float = 0.95) -> Tuple[float, float]:
    """Calculate bootstrap confidence interval for ROI."""
    rois = []
    n = len(y_true)

    for _ in range(n_boot):
        idx = np.random.choice(n, size=n, replace=True)
        res = backtest(y_true[idx], y_prob[idx], odds[idx], threshold)
        if res['n_bets'] >= 10:
            rois.append(res['roi'])

    if len(rois) < 10:
        return (np.nan, np.nan)

    alpha = (1 - ci) / 2
    return (np.percentile(rois, alpha * 100), np.percentile(rois, (1 - alpha) * 100))


# =============================================================================
# MODEL TRAINING
# =============================================================================

def train_stacking_ensemble(X_train: pd.DataFrame, y_train: np.ndarray,
                            X_val: pd.DataFrame, y_val: np.ndarray,
                            X_test: pd.DataFrame) -> np.ndarray:
    """Train LGB + XGB + CatBoost stacking ensemble."""

    # Convert to numpy
    X_tr = X_train.values
    X_v = X_val.values
    X_te = X_test.values

    # Train base models
    lgb_model = lgb.LGBMClassifier(
        n_estimators=200, num_leaves=31, learning_rate=0.05,
        colsample_bytree=0.8, verbose=-1, random_state=42
    )
    lgb_model.fit(X_tr, y_train, eval_set=[(X_v, y_val)],
                  callbacks=[lgb.early_stopping(30, verbose=False)])

    xgb_model = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        colsample_bytree=0.8, verbosity=0, random_state=42
    )
    xgb_model.fit(X_tr, y_train, eval_set=[(X_v, y_val)], verbose=False)

    cat_model = CatBoostClassifier(
        iterations=200, depth=6, learning_rate=0.05,
        verbose=False, random_state=42
    )
    cat_model.fit(X_tr, y_train, eval_set=[(X_v, y_val)],
                  early_stopping_rounds=30, verbose=False)

    # Get base predictions
    preds_lgb_val = lgb_model.predict_proba(X_v)[:, 1]
    preds_xgb_val = xgb_model.predict_proba(X_v)[:, 1]
    preds_cat_val = cat_model.predict_proba(X_v)[:, 1]

    preds_lgb_test = lgb_model.predict_proba(X_te)[:, 1]
    preds_xgb_test = xgb_model.predict_proba(X_te)[:, 1]
    preds_cat_test = cat_model.predict_proba(X_te)[:, 1]

    # Stack with logistic regression
    stack_val = np.column_stack([preds_lgb_val, preds_xgb_val, preds_cat_val])
    stack_test = np.column_stack([preds_lgb_test, preds_xgb_test, preds_cat_test])

    meta = LogisticRegression(C=1.0, solver='lbfgs')
    meta.fit(stack_val, y_val)

    return meta.predict_proba(stack_test)[:, 1]


def train_lgb_simple(X_train: pd.DataFrame, y_train: np.ndarray,
                     X_val: pd.DataFrame, y_val: np.ndarray,
                     X_test: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
    """Train simple LightGBM model."""

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
        'seed': 42
    }

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        params, train_data,
        num_boost_round=300,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )

    importance = dict(zip(X_train.columns, model.feature_importance(importance_type='gain')))

    return model.predict(X_test), importance


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def calc_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Calculate Expected Calibration Error."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i+1])
        if mask.sum() > 0:
            ece += mask.sum() * np.abs(y_prob[mask].mean() - y_true[mask].mean())
    return ece / len(y_true) if len(y_true) > 0 else 0


def analyze_model(df: pd.DataFrame, model_key: str, features: List[str],
                  use_stacking: bool = False) -> Dict:
    """Run complete analysis for a single model."""

    cfg = MODELS[model_key]
    print(f"\n{'='*60}")
    print(f"MODEL: {cfg['name']} {'(Stacking)' if use_stacking else '(LightGBM)'}")
    print(f"{'='*60}")

    # Get target column
    if cfg.get('line') is not None:
        target_col = f"target_{model_key.replace('.', '_')}"
    else:
        target_col = cfg['target']

    if target_col not in df.columns:
        return {'error': f'Target column {target_col} not found'}

    # Filter to valid rows
    df_model = df[df[target_col].notna()].copy()
    print(f"  Samples with target: {len(df_model):,}")

    if len(df_model) < cfg['min_samples']:
        return {'error': f'Insufficient samples: {len(df_model)}'}

    # Prepare features
    avail_features = [f for f in features if f in df_model.columns]
    df_model = df_model.dropna(subset=avail_features, thresh=int(len(avail_features) * 0.5))

    for f in avail_features:
        df_model[f] = df_model[f].fillna(df_model[f].median())

    print(f"  Features: {len(avail_features)}")
    print(f"  Final samples: {len(df_model):,}")

    # Walk-forward CV
    n = len(df_model)
    train_size = int(n * CONFIG.train_pct)
    fold_size = (n - train_size) // CONFIG.n_folds

    all_preds, all_true, all_odds = [], [], []
    fold_metrics = []
    feature_importance = defaultdict(float)

    for fold in range(CONFIG.n_folds):
        train_end = train_size + fold * fold_size
        test_start = train_end
        test_end = min(test_start + fold_size, n)

        if test_end - test_start < 30:
            continue

        X_train = df_model.iloc[:train_end][avail_features]
        y_train = df_model.iloc[:train_end][target_col].values
        X_test = df_model.iloc[test_start:test_end][avail_features]
        y_test = df_model.iloc[test_start:test_end][target_col].values

        # Split for validation
        val_size = int(train_end * 0.15)
        X_tr, y_tr = X_train.iloc[:-val_size], y_train[:-val_size]
        X_val, y_val = X_train.iloc[-val_size:], y_train[-val_size:]

        # Train model
        if use_stacking:
            y_prob_test = train_stacking_ensemble(X_tr, y_tr, X_val, y_val, X_test)
            y_prob_val = train_stacking_ensemble(X_tr, y_tr, X_val, y_val, X_val)
        else:
            y_prob_test, importance = train_lgb_simple(X_tr, y_tr, X_val, y_val, X_test)
            y_prob_val = train_lgb_simple(X_tr, y_tr, X_val, y_val, X_val)[0]
            for k, v in importance.items():
                feature_importance[k] += v

        # Calibrate
        iso = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds='clip')
        iso.fit(y_prob_val, y_val)
        y_prob_cal = iso.predict(y_prob_test)

        # Calculate fold metrics
        mask = ~(np.isnan(y_test) | np.isnan(y_prob_cal))
        if mask.sum() >= 10:
            auc = roc_auc_score(y_test[mask], y_prob_cal[mask])
            ece = calc_ece(y_test[mask], y_prob_cal[mask])
            fold_metrics.append({'fold': fold + 1, 'auc': auc, 'ece': ece, 'n': mask.sum()})
            print(f"  Fold {fold + 1}: AUC={auc:.3f}, ECE={ece:.3f}, n={mask.sum()}")

        all_preds.extend(y_prob_cal)
        all_true.extend(y_test)

        if cfg['odds_col'] and cfg['odds_col'] in df_model.columns:
            all_odds.extend(df_model.iloc[test_start:test_end][cfg['odds_col']].values)
        else:
            all_odds.extend([np.nan] * (test_end - test_start))

    # Convert to arrays
    all_preds = np.array(all_preds)
    all_true = np.array(all_true)
    all_odds = np.array(all_odds)

    # Overall metrics
    mask = ~(np.isnan(all_true) | np.isnan(all_preds))
    overall = {
        'auc': roc_auc_score(all_true[mask], all_preds[mask]),
        'brier': brier_score_loss(all_true[mask], all_preds[mask]),
        'logloss': log_loss(all_true[mask], np.clip(all_preds[mask], 1e-7, 1-1e-7)),
        'ece': calc_ece(all_true[mask], all_preds[mask]),
        'n': mask.sum(),
        'base_rate': all_true[mask].mean()
    }

    # Betting simulation
    betting = {}
    best_roi = -np.inf
    best_threshold = None

    if cfg['odds_col']:
        for threshold in CONFIG.edge_thresholds:
            result = backtest(all_true, all_preds, all_odds, threshold)
            betting[f'edge_{threshold}'] = result

            if result['n_bets'] >= CONFIG.min_bets and result['roi'] > best_roi:
                best_roi = result['roi']
                best_threshold = threshold

        if best_threshold:
            ci = bootstrap_ci(all_true, all_preds, all_odds, best_threshold)
            betting['best'] = {
                'threshold': best_threshold,
                'roi': best_roi,
                'ci_95': ci
            }

    return {
        'model': cfg['name'],
        'n_samples': len(df_model),
        'fold_metrics': fold_metrics,
        'overall': overall,
        'betting': betting,
        'top_features': dict(sorted(feature_importance.items(), key=lambda x: -x[1])[:20])
    }


def run_full_pipeline(data_path: str, output_dir: str = 'experiments/outputs/validated_analysis'):
    """Run the complete validated analysis pipeline."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("VALIDATED ML ANALYSIS PIPELINE")
    print("=" * 80)
    print(f"Output directory: {output_path}")

    # Load and prepare data
    df = load_data(data_path)
    df = add_interaction_features(df)

    # Get features
    features = get_features(df)
    print(f"Safe features: {len(features)}")

    # Add interaction features to feature list
    interaction_features = ['elo_x_form', 'rest_x_position', 'odds_confidence', 'attack_intensity']
    features.extend([f for f in interaction_features if f in df.columns])

    # Run analysis for each model (both simple and stacking)
    results = {}

    for model_key in MODELS:
        # Simple LightGBM
        results[f'{model_key}_lgb'] = analyze_model(df, model_key, features, use_stacking=False)

        # Stacking ensemble (only for models with odds)
        if MODELS[model_key]['odds_col']:
            results[f'{model_key}_stack'] = analyze_model(df, model_key, features, use_stacking=True)

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

    # Print summary table
    print("\n" + "=" * 110)
    print("FINAL RESULTS SUMMARY")
    print("=" * 110)
    print(f"{'Model':<25} {'AUC':<7} {'Brier':<7} {'ECE':<7} {'Threshold':<10} {'Bets':<7} "
          f"{'Win%':<7} {'ROI%':<10} {'95% CI':<20}")
    print("-" * 110)

    for key, result in results.items():
        if 'error' in result:
            print(f"{key:<25} ERROR: {result['error']}")
            continue

        m = result['overall']
        b = result.get('betting', {})
        best = b.get('best', {})

        if best:
            th = best['threshold']
            stats = b.get(f'edge_{th}', {})
            ci = best.get('ci_95', (np.nan, np.nan))
            ci_str = f"[{ci[0]:.1f}, {ci[1]:.1f}]" if ci and not np.isnan(ci[0]) else "N/A"
        else:
            th = 'N/A'
            stats = {}
            ci_str = 'N/A'

        print(f"{result['model']:<25} {m.get('auc', 0):<7.3f} {m.get('brier', 0):<7.3f} "
              f"{m.get('ece', 0):<7.3f} {str(th):<10} {stats.get('n_bets', 0):<7} "
              f"{stats.get('win_rate', 0):<7.1f} {stats.get('roi', 0):<10.1f} {ci_str:<20}")

    print("\n" + "=" * 80)
    print("DEPLOYMENT RECOMMENDATION")
    print("=" * 80)

    # Check if any model is profitable
    profitable = []
    for key, result in results.items():
        if 'error' in result:
            continue
        best = result.get('betting', {}).get('best', {})
        if best and best.get('ci_95'):
            ci = best['ci_95']
            if ci[0] > 0:  # Lower bound > 0
                profitable.append((key, best['roi'], ci))

    if profitable:
        print("PROFITABLE MODELS FOUND:")
        for name, roi, ci in profitable:
            print(f"  - {name}: {roi:.1f}% ROI, 95% CI [{ci[0]:.1f}%, {ci[1]:.1f}%]")
    else:
        print("⛔ NO PROFITABLE MODELS FOUND")
        print("   All models show negative or statistically insignificant ROI")
        print("   DO NOT DEPLOY for live betting")

    print(f"\nResults saved to {output_path / 'results.json'}")

    return results


if __name__ == '__main__':
    import sys

    data_path = sys.argv[1] if len(sys.argv) > 1 else 'data/03-features/features_with_sportmonks_odds.csv'

    results = run_full_pipeline(data_path)
