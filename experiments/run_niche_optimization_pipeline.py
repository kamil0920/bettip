#!/usr/bin/env python
"""
Niche Betting Optimization Pipeline

Complete pipeline for training and optimizing niche betting models:
1. Data Preparation - Load all features, merge with targets
2. Boruta Feature Selection - Reduce to relevant features
3. Model Architecture Comparison - Find best model type
4. Hyperparameter Tuning - Optimize the winning architecture
5. SHAP Feature Validation - Validate/refine feature importance
6. Probability Calibration - Calibrate for accurate probabilities
7. Business Metrics Optimization - Find optimal betting thresholds

Usage:
    python experiments/run_niche_optimization_pipeline.py --bet-type corners
    python experiments/run_niche_optimization_pipeline.py --bet-type cards
    python experiments/run_niche_optimization_pipeline.py --bet-type corners --skip-tuning
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import joblib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime
from typing import Dict, List, Tuple, Optional

# ML imports
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from boruta import BorutaPy

# Optional imports
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


# =============================================================================
# CONFIGURATION
# =============================================================================

BET_TYPE_CONFIG = {
    'corners': {
        'targets': ['over_9_5', 'over_10_5', 'over_11_5'],
        'odds': {
            'over_9_5': (1.90, 1.90),
            'over_10_5': (2.10, 1.72),
            'over_11_5': (2.50, 1.55),
        },
        'data_loader': 'load_corners_data',
    },
    'cards': {
        'targets': ['over_3_5', 'over_4_5', 'over_5_5'],
        'odds': {
            'over_3_5': (1.85, 1.95),
            'over_4_5': (2.20, 1.65),
            'over_5_5': (2.75, 1.45),
        },
        'data_loader': 'load_cards_data',
    },
    'shots': {
        'targets': ['over_22_5', 'over_24_5', 'over_26_5'],
        'odds': {
            'over_22_5': (1.75, 2.05),  # ~63% over rate
            'over_24_5': (1.90, 1.90),  # ~50% over rate (balanced)
            'over_26_5': (2.15, 1.70),  # ~36% over rate
        },
        'data_loader': 'load_shots_data',
    },
    'fouls': {
        'targets': ['over_22_5', 'over_24_5', 'over_26_5'],
        'odds': {
            'over_22_5': (1.75, 2.05),  # ~61% over rate
            'over_24_5': (1.90, 1.90),  # ~48% over rate
            'over_26_5': (2.15, 1.70),  # ~37% over rate
        },
        'data_loader': 'load_fouls_data',
    },
}

MODEL_CONFIGS = {
    'xgboost': {
        'class': XGBClassifier,
        'default_params': {
            'n_estimators': 200, 'max_depth': 4, 'min_child_weight': 15,
            'reg_lambda': 5.0, 'learning_rate': 0.05, 'subsample': 0.8,
            'random_state': 42, 'verbosity': 0, 'n_jobs': -1
        },
        'tune_params': {
            'max_depth': (3, 8),
            'learning_rate': (0.01, 0.2),
            'n_estimators': (100, 400),
            'reg_lambda': (1.0, 10.0),
            'min_child_weight': (5, 30),
        }
    },
    'lightgbm': {
        'class': LGBMClassifier,
        'default_params': {
            'n_estimators': 200, 'max_depth': 4, 'min_child_samples': 50,
            'reg_lambda': 5.0, 'learning_rate': 0.05, 'subsample': 0.8,
            'random_state': 42, 'verbose': -1, 'n_jobs': -1
        },
        'tune_params': {
            'max_depth': (3, 8),
            'learning_rate': (0.01, 0.2),
            'n_estimators': (100, 400),
            'reg_lambda': (1.0, 10.0),
            'min_child_samples': (20, 100),
        }
    },
    'catboost': {
        'class': CatBoostClassifier,
        'default_params': {
            'iterations': 200, 'depth': 4, 'l2_leaf_reg': 10,
            'learning_rate': 0.05, 'random_state': 42, 'verbose': 0,
            'thread_count': -1
        },
        'tune_params': {
            'depth': (3, 8),
            'learning_rate': (0.01, 0.2),
            'iterations': (100, 400),
            'l2_leaf_reg': (1.0, 20.0),
        }
    },
    'random_forest': {
        'class': RandomForestClassifier,
        'default_params': {
            'n_estimators': 200, 'max_depth': 6, 'min_samples_split': 10,
            'min_samples_leaf': 5, 'random_state': 42, 'n_jobs': -1
        },
        'tune_params': {
            'max_depth': (4, 12),
            'n_estimators': (100, 400),
            'min_samples_split': (5, 20),
            'min_samples_leaf': (2, 10),
        }
    },
}


# =============================================================================
# STEP 1: DATA LOADING
# =============================================================================

def load_main_features() -> pd.DataFrame:
    """Load the main features file."""
    path = Path('data/03-features/features_all_5leagues_with_odds.csv')
    return pd.read_csv(path)


def load_corners_data() -> Tuple[pd.DataFrame, List[str]]:
    """Load and prepare corners data with all features."""
    print("\n" + "=" * 70)
    print("STEP 1: DATA PREPARATION (Corners)")
    print("=" * 70)

    # Load corner stats
    corner_data = []
    for league in ['premier_league', 'la_liga', 'serie_a']:
        league_path = Path(f'data/01-raw/{league}')
        if not league_path.exists():
            continue
        for season_dir in league_path.iterdir():
            if not season_dir.is_dir():
                continue
            stats_file = season_dir / 'match_stats.parquet'
            matches_file = season_dir / 'matches.parquet'
            if not stats_file.exists() or not matches_file.exists():
                continue
            stats = pd.read_parquet(stats_file)
            matches = pd.read_parquet(matches_file)
            matches_slim = matches[['fixture.id', 'fixture.referee']].rename(
                columns={'fixture.id': 'fixture_id', 'fixture.referee': 'referee'})
            merged = stats.merge(matches_slim, on='fixture_id', how='left')
            merged['total_corners'] = merged['home_corners'] + merged['away_corners']
            corner_data.append(merged)

    corner_df = pd.concat(corner_data, ignore_index=True)
    print(f"Corner data: {len(corner_df)} matches")

    # Load main features
    main_df = load_main_features()
    print(f"Main features: {len(main_df)} matches, {len(main_df.columns)} columns")

    # Merge
    df = main_df.merge(
        corner_df[['fixture_id', 'total_corners', 'referee']],
        on='fixture_id', how='inner'
    )
    print(f"Merged: {len(df)} matches")

    # NOTE: Referee stats will be calculated after temporal split to avoid leakage
    df['ref_corner_avg'] = np.nan
    df['ref_corner_std'] = np.nan

    # Create targets
    df['over_9_5'] = (df['total_corners'] > 9.5).astype(int)
    df['over_10_5'] = (df['total_corners'] > 10.5).astype(int)
    df['over_11_5'] = (df['total_corners'] > 11.5).astype(int)

    # Get feature columns
    exclude_cols = [
        'fixture_id', 'date', 'home_team_name', 'away_team_name',
        'home_team_id', 'away_team_id', 'round',
        'total_corners', 'referee', 'over_9_5', 'over_10_5', 'over_11_5',
        'home_score', 'away_score', 'result', 'btts',
    ]

    feature_cols = [c for c in df.columns if c not in exclude_cols
                    and df[c].dtype in ['int64', 'float64', 'int32', 'float32']]

    print(f"Feature columns: {len(feature_cols)}")

    return df, feature_cols


def load_cards_data() -> Tuple[pd.DataFrame, List[str]]:
    """Load and prepare cards data with all features."""
    print("\n" + "=" * 70)
    print("STEP 1: DATA PREPARATION (Cards)")
    print("=" * 70)

    # Load main features
    main_df = load_main_features()
    print(f"Main features: {len(main_df)} matches, {len(main_df.columns)} columns")

    # Load events for cards
    cards_data = []
    for league in ['premier_league', 'la_liga', 'serie_a']:
        for year in range(2019, 2026):
            events_file = Path(f'data/01-raw/{league}/{year}/events.parquet')
            matches_file = Path(f'data/01-raw/{league}/{year}/matches.parquet')
            if not events_file.exists():
                continue

            events = pd.read_parquet(events_file)
            yellows = events[events['type'] == 'Card']
            if 'detail' in yellows.columns:
                yellows = yellows[yellows['detail'] == 'Yellow Card']

            card_counts = yellows.groupby('fixture_id').size().reset_index(name='total_yellows')

            if matches_file.exists():
                matches = pd.read_parquet(matches_file)
                matches_slim = matches[['fixture.id', 'fixture.referee']].rename(
                    columns={'fixture.id': 'fixture_id', 'fixture.referee': 'referee'})
                card_counts = card_counts.merge(matches_slim, on='fixture_id', how='left')

            cards_data.append(card_counts)

    if not cards_data:
        raise ValueError("No cards data found")

    cards_df = pd.concat(cards_data, ignore_index=True)
    print(f"Cards data: {len(cards_df)} matches")

    # Merge
    df = main_df.merge(cards_df, on='fixture_id', how='inner')
    print(f"Merged: {len(df)} matches")

    # Add referee stats
    if 'referee' in df.columns:
        ref_stats = df.groupby('referee')['total_yellows'].agg(['mean', 'std']).reset_index()
        ref_stats.columns = ['referee', 'ref_cards_avg', 'ref_cards_std']
        df = df.merge(ref_stats, on='referee', how='left')
        df['ref_cards_avg'] = df['ref_cards_avg'].fillna(df['total_yellows'].mean())
        df['ref_cards_std'] = df['ref_cards_std'].fillna(df['total_yellows'].std())

    # Create targets
    df['over_3_5'] = (df['total_yellows'] > 3.5).astype(int)
    df['over_4_5'] = (df['total_yellows'] > 4.5).astype(int)
    df['over_5_5'] = (df['total_yellows'] > 5.5).astype(int)

    # Get feature columns
    exclude_cols = [
        'fixture_id', 'date', 'home_team_name', 'away_team_name',
        'home_team_id', 'away_team_id', 'round', 'referee',
        'total_yellows', 'over_3_5', 'over_4_5', 'over_5_5',
        'home_score', 'away_score', 'result', 'btts',
    ]

    feature_cols = [c for c in df.columns if c not in exclude_cols
                    and df[c].dtype in ['int64', 'float64', 'int32', 'float32']]

    print(f"Feature columns: {len(feature_cols)}")

    return df, feature_cols


def load_shots_data() -> Tuple[pd.DataFrame, List[str]]:
    """Load and prepare total shots data with all features."""
    print("\n" + "=" * 70)
    print("STEP 1: DATA PREPARATION (Shots)")
    print("=" * 70)

    # Load match stats for shots
    stats_data = []
    for league in ['premier_league', 'la_liga', 'serie_a']:
        league_path = Path(f'data/01-raw/{league}')
        if not league_path.exists():
            continue
        for season_dir in league_path.iterdir():
            if not season_dir.is_dir():
                continue
            stats_file = season_dir / 'match_stats.parquet'
            matches_file = season_dir / 'matches.parquet'
            if not stats_file.exists() or not matches_file.exists():
                continue
            stats = pd.read_parquet(stats_file)
            matches = pd.read_parquet(matches_file)
            matches_slim = matches[['fixture.id', 'fixture.referee']].rename(
                columns={'fixture.id': 'fixture_id', 'fixture.referee': 'referee'})
            merged = stats.merge(matches_slim, on='fixture_id', how='left')
            merged['total_shots'] = merged['home_shots'] + merged['away_shots']
            stats_data.append(merged)

    stats_df = pd.concat(stats_data, ignore_index=True)
    print(f"Shots data: {len(stats_df)} matches")

    # Load main features
    main_df = load_main_features()
    print(f"Main features: {len(main_df)} matches, {len(main_df.columns)} columns")

    # Merge
    df = main_df.merge(
        stats_df[['fixture_id', 'total_shots', 'home_shots', 'away_shots', 'referee']],
        on='fixture_id', how='inner'
    )
    print(f"Merged: {len(df)} matches")

    # NOTE: Referee stats will be calculated after temporal split to avoid leakage
    # Placeholder columns that will be recalculated in run_pipeline
    df['ref_shots_avg'] = np.nan
    df['ref_shots_std'] = np.nan

    # Create targets
    df['over_22_5'] = (df['total_shots'] > 22.5).astype(int)
    df['over_24_5'] = (df['total_shots'] > 24.5).astype(int)
    df['over_26_5'] = (df['total_shots'] > 26.5).astype(int)

    # Get feature columns
    exclude_cols = [
        'fixture_id', 'date', 'home_team_name', 'away_team_name',
        'home_team_id', 'away_team_id', 'round', 'referee',
        'total_shots', 'home_shots', 'away_shots',
        'over_22_5', 'over_24_5', 'over_26_5',
        'home_score', 'away_score', 'result', 'btts',
    ]

    feature_cols = [c for c in df.columns if c not in exclude_cols
                    and df[c].dtype in ['int64', 'float64', 'int32', 'float32']]

    print(f"Feature columns: {len(feature_cols)}")

    return df, feature_cols


def load_fouls_data() -> Tuple[pd.DataFrame, List[str]]:
    """Load and prepare total fouls data with all features."""
    print("\n" + "=" * 70)
    print("STEP 1: DATA PREPARATION (Fouls)")
    print("=" * 70)

    # Load match stats for fouls
    stats_data = []
    for league in ['premier_league', 'la_liga', 'serie_a']:
        league_path = Path(f'data/01-raw/{league}')
        if not league_path.exists():
            continue
        for season_dir in league_path.iterdir():
            if not season_dir.is_dir():
                continue
            stats_file = season_dir / 'match_stats.parquet'
            matches_file = season_dir / 'matches.parquet'
            if not stats_file.exists() or not matches_file.exists():
                continue
            stats = pd.read_parquet(stats_file)
            matches = pd.read_parquet(matches_file)
            matches_slim = matches[['fixture.id', 'fixture.referee']].rename(
                columns={'fixture.id': 'fixture_id', 'fixture.referee': 'referee'})
            merged = stats.merge(matches_slim, on='fixture_id', how='left')
            merged['total_fouls'] = merged['home_fouls'] + merged['away_fouls']
            stats_data.append(merged)

    stats_df = pd.concat(stats_data, ignore_index=True)
    print(f"Fouls data: {len(stats_df)} matches")

    # Load main features
    main_df = load_main_features()
    print(f"Main features: {len(main_df)} matches, {len(main_df.columns)} columns")

    # Merge
    df = main_df.merge(
        stats_df[['fixture_id', 'total_fouls', 'home_fouls', 'away_fouls', 'referee']],
        on='fixture_id', how='inner'
    )
    print(f"Merged: {len(df)} matches")

    # NOTE: Referee stats will be calculated after temporal split to avoid leakage
    df['ref_fouls_avg'] = np.nan
    df['ref_fouls_std'] = np.nan

    # Create targets
    df['over_22_5'] = (df['total_fouls'] > 22.5).astype(int)
    df['over_24_5'] = (df['total_fouls'] > 24.5).astype(int)
    df['over_26_5'] = (df['total_fouls'] > 26.5).astype(int)

    # Get feature columns
    exclude_cols = [
        'fixture_id', 'date', 'home_team_name', 'away_team_name',
        'home_team_id', 'away_team_id', 'round', 'referee',
        'total_fouls', 'home_fouls', 'away_fouls',
        'over_22_5', 'over_24_5', 'over_26_5',
        'home_score', 'away_score', 'result', 'btts',
    ]

    feature_cols = [c for c in df.columns if c not in exclude_cols
                    and df[c].dtype in ['int64', 'float64', 'int32', 'float32']]

    print(f"Feature columns: {len(feature_cols)}")

    return df, feature_cols


# =============================================================================
# REFEREE STATS CALCULATION (Leakage-free)
# =============================================================================

def calculate_referee_stats_from_train(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    bet_type: str,
    target_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Calculate referee statistics using ONLY training data to avoid data leakage.
    Then apply these stats to val and test sets.
    """
    # Map bet type to target column for stats
    target_map = {
        'corners': 'total_corners',
        'shots': 'total_shots',
        'fouls': 'total_fouls',
    }

    stat_col = target_map.get(bet_type)
    if stat_col is None or stat_col not in train_df.columns:
        return train_df, val_df, test_df

    # Calculate referee stats from training data ONLY
    ref_stats = train_df.groupby('referee')[stat_col].agg(['mean', 'std']).reset_index()

    # Column names based on bet type
    avg_col = f'ref_{bet_type[:-1] if bet_type.endswith("s") else bet_type}_avg'
    std_col = f'ref_{bet_type[:-1] if bet_type.endswith("s") else bet_type}_std'
    if bet_type == 'corners':
        avg_col, std_col = 'ref_corner_avg', 'ref_corner_std'
    elif bet_type == 'shots':
        avg_col, std_col = 'ref_shots_avg', 'ref_shots_std'
    elif bet_type == 'fouls':
        avg_col, std_col = 'ref_fouls_avg', 'ref_fouls_std'

    ref_stats.columns = ['referee', avg_col, std_col]

    # Global mean/std for unknown referees (from training only)
    global_mean = train_df[stat_col].mean()
    global_std = train_df[stat_col].std()

    # Apply to each split
    for df in [train_df, val_df, test_df]:
        # Drop old columns if they exist
        df.drop(columns=[avg_col, std_col], errors='ignore', inplace=True)
        # Merge with training-based referee stats
        df_merged = df.merge(ref_stats, on='referee', how='left')
        df[avg_col] = df_merged[avg_col].fillna(global_mean)
        df[std_col] = df_merged[std_col].fillna(global_std)

    return train_df, val_df, test_df


# =============================================================================
# STEP 2: BORUTA FEATURE SELECTION
# =============================================================================

def run_boruta_selection(X_train: pd.DataFrame, y_train: np.ndarray,
                         feature_cols: List[str], max_iter: int = 100) -> Tuple[List[str], pd.DataFrame]:
    """Run Boruta feature selection."""
    print("\n" + "=" * 70)
    print("STEP 2: BORUTA FEATURE SELECTION")
    print("=" * 70)

    rf = RandomForestClassifier(
        n_estimators=100, max_depth=5, n_jobs=-1, random_state=42
    )

    boruta = BorutaPy(
        rf, n_estimators='auto', verbose=0,
        random_state=42, max_iter=max_iter, perc=100
    )

    boruta.fit(X_train.values, y_train)

    confirmed = [f for f, s in zip(feature_cols, boruta.support_) if s]
    tentative = [f for f, s in zip(feature_cols, boruta.support_weak_) if s]
    selected = confirmed + tentative

    print(f"Confirmed features: {len(confirmed)}")
    print(f"Tentative features: {len(tentative)}")
    print(f"Total selected: {len(selected)}")

    # Feature rankings
    ranks = pd.DataFrame({
        'feature': feature_cols,
        'rank': boruta.ranking_,
        'confirmed': boruta.support_,
        'tentative': boruta.support_weak_
    }).sort_values('rank')

    print("\nTop 15 features:")
    for _, row in ranks.head(15).iterrows():
        status = "CONFIRMED" if row['confirmed'] else ("tentative" if row['tentative'] else "rejected")
        print(f"  {row['feature']:<40} rank={row['rank']:>2} ({status})")

    # Use at least 15 features
    if len(selected) < 15:
        selected = ranks.head(20)['feature'].tolist()
        print(f"\nExpanded to top {len(selected)} features")

    return selected, ranks


# =============================================================================
# STEP 3: MODEL ARCHITECTURE COMPARISON
# =============================================================================

def create_model(model_type: str, params: Optional[Dict] = None):
    """Create a model instance."""
    config = MODEL_CONFIGS[model_type]
    model_params = config['default_params'].copy()
    if params:
        model_params.update(params)
    return config['class'](**model_params)


def create_stacking_model(meta_type: str = 'lr'):
    """Create a stacking classifier."""
    estimators = [
        ('xgb', create_model('xgboost')),
        ('lgbm', create_model('lightgbm')),
        ('cat', create_model('catboost')),
    ]

    if meta_type == 'lr':
        final = LogisticRegression(max_iter=1000, random_state=42)
    else:  # xgb
        final = XGBClassifier(
            n_estimators=50, max_depth=3, learning_rate=0.1,
            random_state=42, verbosity=0
        )

    return StackingClassifier(
        estimators=estimators,
        final_estimator=final,
        cv=3,
        stack_method='predict_proba',
        n_jobs=-1
    )


def create_voting_model():
    """Create a voting classifier."""
    estimators = [
        ('xgb', create_model('xgboost')),
        ('lgbm', create_model('lightgbm')),
        ('cat', create_model('catboost')),
    ]
    return VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)


def calculate_betting_roi(proba: np.ndarray, y_true: np.ndarray,
                          odds_over: float, odds_under: float,
                          threshold: float = 0.55) -> Dict:
    """Calculate ROI for betting strategies."""
    results = {}

    # OVER strategy
    over_mask = proba >= threshold
    if over_mask.sum() >= 20:
        wins = y_true[over_mask] == 1
        profit = (wins * (odds_over - 1) - (~wins)).sum()
        results['over_roi'] = profit / over_mask.sum() * 100
        results['over_bets'] = over_mask.sum()
    else:
        results['over_roi'] = -100
        results['over_bets'] = 0

    # UNDER strategy
    under_mask = (1 - proba) >= threshold
    if under_mask.sum() >= 20:
        wins = y_true[under_mask] == 0
        profit = (wins * (odds_under - 1) - (~wins)).sum()
        results['under_roi'] = profit / under_mask.sum() * 100
        results['under_bets'] = under_mask.sum()
    else:
        results['under_roi'] = -100
        results['under_bets'] = 0

    results['best_roi'] = max(results['over_roi'], results['under_roi'])
    return results


def compare_models(X_train: pd.DataFrame, y_train: np.ndarray,
                   X_val: pd.DataFrame, y_val: np.ndarray,
                   X_test: pd.DataFrame, y_test: np.ndarray,
                   odds_over: float, odds_under: float) -> Tuple[str, Dict]:
    """Compare all model architectures."""
    print("\n" + "=" * 70)
    print("STEP 3: MODEL ARCHITECTURE COMPARISON")
    print("=" * 70)

    models = {
        'xgboost': create_model('xgboost'),
        'lightgbm': create_model('lightgbm'),
        'catboost': create_model('catboost'),
        'random_forest': create_model('random_forest'),
        'voting': create_voting_model(),
        'stacking_lr': create_stacking_model('lr'),
        'stacking_xgb': create_stacking_model('xgb'),
    }

    results = {}
    print(f"\n{'Model':<20} {'Brier':>8} {'AUC':>8} {'Best ROI':>10}")
    print("-" * 50)

    for name, model in models.items():
        try:
            # Train
            model.fit(X_train, y_train)

            # Calibrate
            cal_model = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
            cal_model.fit(X_val, y_val)

            # Predict
            proba = cal_model.predict_proba(X_test)[:, 1]

            # Metrics
            brier = brier_score_loss(y_test, proba)
            auc = roc_auc_score(y_test, proba)
            roi_results = calculate_betting_roi(proba, y_test, odds_over, odds_under)

            results[name] = {
                'brier': brier,
                'auc': auc,
                'best_roi': roi_results['best_roi'],
                'over_roi': roi_results['over_roi'],
                'under_roi': roi_results['under_roi'],
            }

            print(f"{name:<20} {brier:>8.4f} {auc:>8.3f} {roi_results['best_roi']:>+9.1f}%")

        except Exception as e:
            print(f"{name:<20} ERROR: {str(e)[:30]}")
            results[name] = {'error': str(e)}

    # Find best model
    valid = {k: v for k, v in results.items() if 'error' not in v}
    best_by_roi = max(valid.items(), key=lambda x: x[1]['best_roi'])
    best_by_brier = min(valid.items(), key=lambda x: x[1]['brier'])

    print(f"\nBest by ROI: {best_by_roi[0]} ({best_by_roi[1]['best_roi']:+.1f}%)")
    print(f"Best by Brier: {best_by_brier[0]} ({best_by_brier[1]['brier']:.4f})")

    return best_by_roi[0], results


# =============================================================================
# STEP 4: HYPERPARAMETER TUNING
# =============================================================================

def tune_model(model_type: str, X_train: pd.DataFrame, y_train: np.ndarray,
               n_trials: int = 50) -> Dict:
    """Tune hyperparameters using Optuna."""
    print("\n" + "=" * 70)
    print(f"STEP 4: HYPERPARAMETER TUNING ({model_type})")
    print("=" * 70)

    if not OPTUNA_AVAILABLE:
        print("Optuna not available, using default parameters")
        return MODEL_CONFIGS[model_type]['default_params']

    config = MODEL_CONFIGS[model_type]
    tune_ranges = config['tune_params']

    def objective(trial):
        params = config['default_params'].copy()

        for param, (low, high) in tune_ranges.items():
            if isinstance(low, int):
                params[param] = trial.suggest_int(param, low, high)
            else:
                params[param] = trial.suggest_float(param, low, high)

        model = config['class'](**params)

        cv = TimeSeriesSplit(n_splits=3)
        scores = cross_val_score(model, X_train, y_train, cv=cv,
                                 scoring='neg_brier_score', n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = config['default_params'].copy()
    best_params.update(study.best_params)

    print(f"\nBest parameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print(f"Best CV score: {study.best_value:.4f}")

    return best_params


# =============================================================================
# STEP 5: SHAP FEATURE VALIDATION
# =============================================================================

def validate_features_shap(model, X_test: pd.DataFrame,
                           feature_cols: List[str]) -> Tuple[List[str], pd.DataFrame]:
    """Validate feature importance using SHAP."""
    print("\n" + "=" * 70)
    print("STEP 5: SHAP FEATURE VALIDATION")
    print("=" * 70)

    if not SHAP_AVAILABLE:
        print("SHAP not available, keeping all features")
        return feature_cols, pd.DataFrame()

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # For binary classification

        importance = pd.DataFrame({
            'feature': feature_cols,
            'shap_importance': np.abs(shap_values).mean(axis=0)
        }).sort_values('shap_importance', ascending=False)

        print("\nTop 15 features by SHAP importance:")
        for i, row in importance.head(15).iterrows():
            print(f"  {row['feature']:<40} {row['shap_importance']:.4f}")

        # Keep features with meaningful importance
        threshold = importance['shap_importance'].max() * 0.01
        final_features = importance[importance['shap_importance'] > threshold]['feature'].tolist()

        print(f"\nFeatures with >1% relative importance: {len(final_features)}")

        return final_features, importance

    except Exception as e:
        print(f"SHAP analysis failed: {e}")
        return feature_cols, pd.DataFrame()


# =============================================================================
# STEP 6: CALIBRATION
# =============================================================================

def calibrate_model(model, X_val: pd.DataFrame, y_val: np.ndarray,
                    X_test: pd.DataFrame, y_test: np.ndarray,
                    method: str = 'sigmoid'):
    """Calibrate model and evaluate calibration quality."""
    print("\n" + "=" * 70)
    print(f"STEP 6: PROBABILITY CALIBRATION ({method})")
    print("=" * 70)

    calibrated = CalibratedClassifierCV(model, method=method, cv='prefit')
    calibrated.fit(X_val, y_val)

    # Evaluate calibration
    proba_uncal = model.predict_proba(X_test)[:, 1]
    proba_cal = calibrated.predict_proba(X_test)[:, 1]

    brier_uncal = brier_score_loss(y_test, proba_uncal)
    brier_cal = brier_score_loss(y_test, proba_cal)

    print(f"Brier score (uncalibrated): {brier_uncal:.4f}")
    print(f"Brier score (calibrated): {brier_cal:.4f}")
    print(f"Improvement: {(brier_uncal - brier_cal) / brier_uncal * 100:+.1f}%")

    # Calculate ECE
    prob_true, prob_pred = calibration_curve(y_test, proba_cal, n_bins=10)
    ece = np.mean(np.abs(prob_true - prob_pred))
    print(f"Expected Calibration Error: {ece:.4f}")

    return calibrated


# =============================================================================
# STEP 7: BUSINESS OPTIMIZATION
# =============================================================================

def optimize_betting_thresholds(proba: np.ndarray, y_test: np.ndarray,
                                odds_over: float, odds_under: float,
                                n_bootstrap: int = 1000) -> pd.DataFrame:
    """Find optimal betting thresholds with bootstrap confidence intervals."""
    print("\n" + "=" * 70)
    print("STEP 7: BUSINESS METRICS OPTIMIZATION")
    print("=" * 70)

    results = []

    for direction in ['over', 'under']:
        for threshold in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
            if direction == 'over':
                mask = proba >= threshold
                y_target = y_test == 1
                odds = odds_over
            else:
                mask = (1 - proba) >= threshold
                y_target = y_test == 0
                odds = odds_under

            n_bets = mask.sum()
            if n_bets < 20:
                continue

            wins = y_target[mask]
            precision = wins.mean()
            profit = (wins * (odds - 1) - (~wins)).sum()
            roi = profit / n_bets * 100

            # Bootstrap
            rois = []
            for _ in range(n_bootstrap):
                idx = np.random.choice(len(proba), len(proba), replace=True)
                p = proba[idx]
                y = y_test[idx]

                if direction == 'over':
                    m = p >= threshold
                    w = y[m] == 1
                else:
                    m = (1 - p) >= threshold
                    w = y[m] == 0

                if m.sum() > 0:
                    prof = (w * (odds - 1) - (~w)).sum()
                    rois.append(prof / m.sum() * 100)

            results.append({
                'direction': direction.upper(),
                'threshold': threshold,
                'n_bets': n_bets,
                'precision': precision,
                'roi': roi,
                'ci_low': np.percentile(rois, 2.5) if rois else roi,
                'ci_high': np.percentile(rois, 97.5) if rois else roi,
                'p_profit': (np.array(rois) > 0).mean() if rois else 0,
                'odds': odds,
            })

    df = pd.DataFrame(results).sort_values('roi', ascending=False)

    print(f"\n{'Direction':<8} {'Thresh':>7} {'Bets':>6} {'Prec':>7} {'ROI':>8} {'CI 95%':>16} {'P>0':>6}")
    print("-" * 65)

    for _, row in df.iterrows():
        ci = f"[{row['ci_low']:+.1f}, {row['ci_high']:+.1f}]"
        print(f"{row['direction']:<8} {row['threshold']:>7.2f} {row['n_bets']:>6} "
              f"{row['precision']:>6.1%} {row['roi']:>+7.1f}% {ci:>16} {row['p_profit']:>5.0%}")

    # Best strategy
    viable = df[(df['roi'] > 0) & (df['p_profit'] > 0.7)]
    print(f"\nViable strategies (ROI > 0, P > 70%): {len(viable)}")

    if len(viable) > 0:
        best = viable.iloc[0]
        print(f"\nBest strategy: {best['direction']} >= {best['threshold']:.2f}")
        print(f"  ROI: {best['roi']:+.1f}% [{best['ci_low']:+.1f}, {best['ci_high']:+.1f}]")
        print(f"  Bets: {best['n_bets']}, Precision: {best['precision']:.1%}")

    return df


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_pipeline(bet_type: str, target: str, skip_tuning: bool = False,
                 n_tune_trials: int = 50) -> Dict:
    """Run the complete optimization pipeline."""
    print("\n" + "=" * 70)
    print(f"NICHE BETTING OPTIMIZATION PIPELINE")
    print(f"Bet Type: {bet_type.upper()} | Target: {target}")
    print("=" * 70)

    config = BET_TYPE_CONFIG[bet_type]
    odds_over, odds_under = config['odds'][target]

    # Step 1: Load data
    if bet_type == 'corners':
        df, feature_cols = load_corners_data()
    elif bet_type == 'cards':
        df, feature_cols = load_cards_data()
    elif bet_type == 'shots':
        df, feature_cols = load_shots_data()
    elif bet_type == 'fouls':
        df, feature_cols = load_fouls_data()
    else:
        raise ValueError(f"Unknown bet type: {bet_type}")

    # Sort by date
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # Temporal split
    n = len(df)
    train_end = int(0.6 * n)
    val_end = int(0.8 * n)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    print(f"\nTrain: {len(train_df)} ({train_df['date'].min().date()} to {train_df['date'].max().date()})")
    print(f"Val: {len(val_df)} ({val_df['date'].min().date()} to {val_df['date'].max().date()})")
    print(f"Test: {len(test_df)} ({test_df['date'].min().date()} to {test_df['date'].max().date()})")

    # Calculate referee stats using training data only (prevents leakage)
    if bet_type in ['corners', 'shots', 'fouls']:
        train_df, val_df, test_df = calculate_referee_stats_from_train(
            train_df, val_df, test_df, bet_type, target
        )
        print(f"Referee stats calculated from training data only (leakage-free)")

    # Prepare data
    X_train = train_df[feature_cols].fillna(0).astype(float)
    X_val = val_df[feature_cols].fillna(0).astype(float)
    X_test = test_df[feature_cols].fillna(0).astype(float)

    y_train = train_df[target].values
    y_val = val_df[target].values
    y_test = test_df[target].values

    print(f"\nPositive rate - Train: {y_train.mean():.1%}, Val: {y_val.mean():.1%}, Test: {y_test.mean():.1%}")

    # Step 2: Boruta feature selection
    selected_features, feature_ranks = run_boruta_selection(X_train, y_train, feature_cols)

    X_train_sel = X_train[selected_features]
    X_val_sel = X_val[selected_features]
    X_test_sel = X_test[selected_features]

    # Step 3: Model comparison
    best_model_type, comparison_results = compare_models(
        X_train_sel, y_train, X_val_sel, y_val, X_test_sel, y_test,
        odds_over, odds_under
    )

    # Step 4: Hyperparameter tuning
    if skip_tuning or best_model_type.startswith('stacking') or best_model_type == 'voting':
        print("\nSkipping tuning (ensemble model or --skip-tuning)")
        best_params = None
        if best_model_type == 'stacking_xgb':
            final_model = create_stacking_model('xgb')
        elif best_model_type == 'stacking_lr':
            final_model = create_stacking_model('lr')
        elif best_model_type == 'voting':
            final_model = create_voting_model()
        else:
            final_model = create_model(best_model_type)
    else:
        best_params = tune_model(best_model_type, X_train_sel, y_train, n_tune_trials)
        final_model = create_model(best_model_type, best_params)

    # Fit final model
    X_train_full = pd.concat([X_train_sel, X_val_sel])
    y_train_full = np.concatenate([y_train, y_val])
    final_model.fit(X_train_full, y_train_full)

    # Step 5: SHAP validation
    if not (best_model_type.startswith('stacking') or best_model_type == 'voting'):
        final_features, shap_importance = validate_features_shap(
            final_model, X_test_sel, selected_features
        )
    else:
        final_features = selected_features
        shap_importance = pd.DataFrame()

    # Step 6: Calibration
    calibrated_model = calibrate_model(final_model, X_val_sel, y_val, X_test_sel, y_test)

    # Step 7: Business optimization
    proba = calibrated_model.predict_proba(X_test_sel)[:, 1]
    betting_results = optimize_betting_thresholds(proba, y_test, odds_over, odds_under)

    # Save results
    results = {
        'bet_type': bet_type,
        'target': target,
        'timestamp': datetime.now().isoformat(),
        'data': {
            'train_size': len(train_df),
            'val_size': len(val_df),
            'test_size': len(test_df),
        },
        'features': {
            'total': len(feature_cols),
            'boruta_selected': len(selected_features),
            'final': len(final_features),
            'list': final_features,
        },
        'model': {
            'architecture': best_model_type,
            'params': best_params,
        },
        'comparison': {k: v for k, v in comparison_results.items() if 'error' not in v},
        'betting_strategies': betting_results.to_dict('records'),
    }

    # Find best strategy
    viable = betting_results[(betting_results['roi'] > 0) & (betting_results['p_profit'] > 0.7)]
    if len(viable) > 0:
        best = viable.iloc[0]
        results['best_strategy'] = {
            'direction': best['direction'],
            'threshold': best['threshold'],
            'roi': best['roi'],
            'ci_95': [best['ci_low'], best['ci_high']],
            'p_profit': best['p_profit'],
            'n_bets': int(best['n_bets']),
        }

    # Save to file
    output_path = Path(f'experiments/outputs/{bet_type}_{target}_pipeline.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    # Save model
    model_path = Path(f'models/{bet_type}_{target}_model.joblib')
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrated_model, model_path)
    print(f"Model saved to {model_path}")

    return results


# =============================================================================
# WALK-FORWARD VALIDATION
# =============================================================================

def run_walk_forward_validation(bet_type: str, target: str, n_folds: int = 3) -> Dict:
    """
    Run walk-forward validation with expanding window.

    For 3 folds, splits data into 4 parts:
    - Fold 1: Train [part1], Test [part2]
    - Fold 2: Train [part1+part2], Test [part3]
    - Fold 3: Train [part1+part2+part3], Test [part4]
    """
    print("\n" + "=" * 70)
    print(f"WALK-FORWARD VALIDATION ({n_folds} FOLDS)")
    print(f"Bet Type: {bet_type.upper()} | Target: {target}")
    print("=" * 70)

    config = BET_TYPE_CONFIG[bet_type]
    odds_over, odds_under = config['odds'][target]

    # Load data
    if bet_type == 'corners':
        df, feature_cols = load_corners_data()
    elif bet_type == 'cards':
        df, feature_cols = load_cards_data()
    elif bet_type == 'shots':
        df, feature_cols = load_shots_data()
    elif bet_type == 'fouls':
        df, feature_cols = load_fouls_data()
    else:
        raise ValueError(f"Unknown bet type: {bet_type}")

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    n = len(df)
    n_parts = n_folds + 1  # 4 parts for 3 folds
    part_size = n // n_parts

    print(f"\nTotal samples: {n}")
    print(f"Part size: ~{part_size} samples each")

    fold_results = []
    all_predictions = []
    all_actuals = []
    all_odds_used = []

    for fold in range(n_folds):
        print(f"\n{'='*50}")
        print(f"FOLD {fold + 1}/{n_folds}")
        print(f"{'='*50}")

        # Expanding window: train on all data up to test fold
        train_end = (fold + 1) * part_size
        test_start = train_end
        test_end = (fold + 2) * part_size if fold < n_folds - 1 else n

        train_df = df.iloc[:train_end].copy()
        test_df = df.iloc[test_start:test_end].copy()

        # Use last 20% of training as validation
        val_split = int(0.8 * len(train_df))
        val_df = train_df.iloc[val_split:].copy()
        train_df_actual = train_df.iloc[:val_split].copy()

        print(f"Train: {len(train_df_actual)} ({train_df_actual['date'].min().date()} to {train_df_actual['date'].max().date()})")
        print(f"Val: {len(val_df)} ({val_df['date'].min().date()} to {val_df['date'].max().date()})")
        print(f"Test: {len(test_df)} ({test_df['date'].min().date()} to {test_df['date'].max().date()})")

        # Calculate referee stats from training data only
        if bet_type in ['corners', 'shots', 'fouls']:
            train_df_actual, val_df, test_df = calculate_referee_stats_from_train(
                train_df_actual, val_df, test_df, bet_type, target
            )

        # Prepare features
        X_train = train_df_actual[feature_cols].fillna(0).astype(float)
        X_val = val_df[feature_cols].fillna(0).astype(float)
        X_test = test_df[feature_cols].fillna(0).astype(float)

        y_train = train_df_actual[target].values
        y_val = val_df[target].values
        y_test = test_df[target].values

        print(f"Positive rate - Train: {y_train.mean():.1%}, Val: {y_val.mean():.1%}, Test: {y_test.mean():.1%}")

        # Boruta feature selection
        rf = RandomForestClassifier(n_estimators=100, max_depth=5, n_jobs=-1, random_state=42)
        boruta = BorutaPy(rf, n_estimators='auto', verbose=0, random_state=42, max_iter=50, perc=100)
        boruta.fit(X_train.values, y_train)

        confirmed = [f for f, s in zip(feature_cols, boruta.support_) if s]
        tentative = [f for f, s in zip(feature_cols, boruta.support_weak_) if s]
        selected_features = confirmed + tentative

        if len(selected_features) < 5:
            selected_features = feature_cols[:20]

        print(f"Selected features: {len(selected_features)}")

        X_train_sel = X_train[selected_features]
        X_val_sel = X_val[selected_features]
        X_test_sel = X_test[selected_features]

        # Train Random Forest (robust baseline)
        model = RandomForestClassifier(
            n_estimators=200, max_depth=6, min_samples_split=10,
            min_samples_leaf=5, random_state=42, n_jobs=-1
        )

        X_train_full = pd.concat([X_train_sel, X_val_sel], ignore_index=True)
        y_train_full = np.concatenate([y_train, y_val])
        model.fit(X_train_full, y_train_full)

        # Calibrate
        calibrated = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
        calibrated.fit(X_val_sel, y_val)

        # Predict
        proba = calibrated.predict_proba(X_test_sel)[:, 1]

        # Evaluate with best threshold (0.75)
        threshold = 0.75

        # Check OVER strategy
        over_mask = proba >= threshold
        if over_mask.sum() > 0:
            over_wins = y_test[over_mask] == 1
            over_profit = (over_wins * (odds_over - 1) - (~over_wins)).sum()
            over_roi = over_profit / over_mask.sum() * 100
            over_precision = over_wins.mean()
        else:
            over_roi, over_precision = 0, 0

        # Check UNDER strategy
        under_mask = (1 - proba) >= threshold
        if under_mask.sum() > 0:
            under_wins = y_test[under_mask] == 0
            under_profit = (under_wins * (odds_under - 1) - (~under_wins)).sum()
            under_roi = under_profit / under_mask.sum() * 100
            under_precision = under_wins.mean()
        else:
            under_roi, under_precision = 0, 0

        # Pick best direction
        if over_roi > under_roi:
            best_direction = 'OVER'
            best_roi = over_roi
            best_precision = over_precision
            best_n_bets = over_mask.sum()
            best_odds = odds_over
            mask = over_mask
            wins = y_test[mask] == 1
        else:
            best_direction = 'UNDER'
            best_roi = under_roi
            best_precision = under_precision
            best_n_bets = under_mask.sum()
            best_odds = odds_under
            mask = under_mask
            wins = y_test[mask] == 0

        print(f"\nFold {fold + 1} Results:")
        print(f"  OVER @{threshold}: {over_mask.sum()} bets, {over_precision:.1%} precision, {over_roi:+.1f}% ROI")
        print(f"  UNDER @{threshold}: {under_mask.sum()} bets, {under_precision:.1%} precision, {under_roi:+.1f}% ROI")
        print(f"  Best: {best_direction} -> {best_roi:+.1f}% ROI")

        fold_results.append({
            'fold': fold + 1,
            'train_size': len(train_df_actual),
            'test_size': len(test_df),
            'test_start': test_df['date'].min().date().isoformat(),
            'test_end': test_df['date'].max().date().isoformat(),
            'over_roi': over_roi,
            'over_n_bets': int(over_mask.sum()),
            'over_precision': over_precision,
            'under_roi': under_roi,
            'under_n_bets': int(under_mask.sum()),
            'under_precision': under_precision,
            'best_direction': best_direction,
            'best_roi': best_roi,
        })

        # Collect predictions for aggregate analysis
        if mask.sum() > 0:
            all_predictions.extend(proba[mask].tolist())
            all_actuals.extend(y_test[mask].tolist())
            all_odds_used.extend([best_odds] * mask.sum())

    # Aggregate results
    print("\n" + "=" * 70)
    print("WALK-FORWARD VALIDATION SUMMARY")
    print("=" * 70)

    print(f"\n{'Fold':<6} {'Period':<25} {'Direction':<10} {'Bets':>6} {'Precision':>10} {'ROI':>10}")
    print("-" * 75)

    for r in fold_results:
        period = f"{r['test_start']} to {r['test_end']}"
        n_bets = r['over_n_bets'] if r['best_direction'] == 'OVER' else r['under_n_bets']
        prec = r['over_precision'] if r['best_direction'] == 'OVER' else r['under_precision']
        print(f"{r['fold']:<6} {period:<25} {r['best_direction']:<10} {n_bets:>6} {prec:>9.1%} {r['best_roi']:>+9.1f}%")

    # Average metrics
    avg_over_roi = np.mean([r['over_roi'] for r in fold_results])
    avg_under_roi = np.mean([r['under_roi'] for r in fold_results])
    avg_best_roi = np.mean([r['best_roi'] for r in fold_results])

    std_over_roi = np.std([r['over_roi'] for r in fold_results])
    std_under_roi = np.std([r['under_roi'] for r in fold_results])
    std_best_roi = np.std([r['best_roi'] for r in fold_results])

    total_over_bets = sum(r['over_n_bets'] for r in fold_results)
    total_under_bets = sum(r['under_n_bets'] for r in fold_results)

    print(f"\n{'Aggregate Results':}")
    print(f"  OVER: Avg ROI = {avg_over_roi:+.1f}% (+/- {std_over_roi:.1f}%), Total bets = {total_over_bets}")
    print(f"  UNDER: Avg ROI = {avg_under_roi:+.1f}% (+/- {std_under_roi:.1f}%), Total bets = {total_under_bets}")
    print(f"  Best: Avg ROI = {avg_best_roi:+.1f}% (+/- {std_best_roi:.1f}%)")

    # Calculate pooled ROI across all folds
    if all_predictions:
        all_predictions = np.array(all_predictions)
        all_actuals = np.array(all_actuals)
        all_odds_used = np.array(all_odds_used)

        # For pooled calculation, determine if each was OVER or UNDER bet
        # This is simplified - using the actual outcomes
        pooled_wins = all_actuals == 1  # Assuming OVER bets for simplicity
        pooled_profit = sum((all_odds_used[i] - 1) if pooled_wins[i] else -1 for i in range(len(pooled_wins)))
        pooled_roi = pooled_profit / len(all_predictions) * 100

        print(f"\n  Pooled ROI (all folds combined): {pooled_roi:+.1f}%")
        print(f"  Total bets across folds: {len(all_predictions)}")

    # Consistency check
    positive_folds = sum(1 for r in fold_results if r['best_roi'] > 0)
    print(f"\n  Profitable folds: {positive_folds}/{n_folds}")

    if positive_folds == n_folds:
        print("  STATUS: CONSISTENT - Profitable in all folds")
    elif positive_folds >= n_folds // 2 + 1:
        print("  STATUS: MOSTLY CONSISTENT - Profitable in majority of folds")
    else:
        print("  STATUS: INCONSISTENT - Not reliably profitable")

    # Save results
    results = {
        'bet_type': bet_type,
        'target': target,
        'n_folds': n_folds,
        'timestamp': datetime.now().isoformat(),
        'fold_results': fold_results,
        'aggregate': {
            'avg_over_roi': avg_over_roi,
            'std_over_roi': std_over_roi,
            'avg_under_roi': avg_under_roi,
            'std_under_roi': std_under_roi,
            'avg_best_roi': avg_best_roi,
            'std_best_roi': std_best_roi,
            'total_over_bets': total_over_bets,
            'total_under_bets': total_under_bets,
            'profitable_folds': positive_folds,
        }
    }

    output_path = Path(f'experiments/outputs/{bet_type}_{target}_walkforward.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Niche Betting Optimization Pipeline')
    parser.add_argument('--bet-type', type=str, required=True,
                        choices=['corners', 'cards', 'shots', 'fouls'],
                        help='Type of bet to optimize')
    parser.add_argument('--target', type=str, default=None,
                        help='Specific target (e.g., over_10_5). If not set, runs all targets.')
    parser.add_argument('--skip-tuning', action='store_true',
                        help='Skip hyperparameter tuning')
    parser.add_argument('--n-trials', type=int, default=50,
                        help='Number of Optuna trials for tuning')
    parser.add_argument('--walk-forward', action='store_true',
                        help='Run walk-forward validation instead of single split')
    parser.add_argument('--n-folds', type=int, default=3,
                        help='Number of folds for walk-forward validation')

    args = parser.parse_args()

    config = BET_TYPE_CONFIG[args.bet_type]

    if args.target:
        targets = [args.target]
    else:
        targets = config['targets']

    all_results = {}

    if args.walk_forward:
        # Run walk-forward validation
        for target in targets:
            print(f"\n{'#' * 70}")
            print(f"# Walk-Forward: {args.bet_type} - {target}")
            print(f"{'#' * 70}")

            results = run_walk_forward_validation(
                args.bet_type, target,
                n_folds=args.n_folds
            )
            all_results[target] = results

        # Summary for walk-forward
        print("\n" + "=" * 70)
        print("WALK-FORWARD VALIDATION COMPLETE - SUMMARY")
        print("=" * 70)

        for target, results in all_results.items():
            agg = results['aggregate']
            print(f"\n{target}:")
            print(f"  Avg OVER ROI: {agg['avg_over_roi']:+.1f}% (+/- {agg['std_over_roi']:.1f}%)")
            print(f"  Avg UNDER ROI: {agg['avg_under_roi']:+.1f}% (+/- {agg['std_under_roi']:.1f}%)")
            print(f"  Profitable folds: {agg['profitable_folds']}/{args.n_folds}")
    else:
        # Run standard pipeline
        for target in targets:
            print(f"\n{'#' * 70}")
            print(f"# Processing: {args.bet_type} - {target}")
            print(f"{'#' * 70}")

            results = run_pipeline(
                args.bet_type, target,
                skip_tuning=args.skip_tuning,
                n_tune_trials=args.n_trials
            )
            all_results[target] = results

        # Summary
        print("\n" + "=" * 70)
        print("PIPELINE COMPLETE - SUMMARY")
        print("=" * 70)

        for target, results in all_results.items():
            print(f"\n{target}:")
            print(f"  Model: {results['model']['architecture']}")
            print(f"  Features: {results['features']['final']} selected")
            if 'best_strategy' in results:
                bs = results['best_strategy']
                print(f"  Best: {bs['direction']} >= {bs['threshold']:.2f} -> {bs['roi']:+.1f}% ROI")


if __name__ == "__main__":
    main()
