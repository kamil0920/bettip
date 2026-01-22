#!/usr/bin/env python3
"""
Full ML Betting Analysis Pipeline

Implements the 5-step framework:
1. Data Quality Audit
2. Baselines + Correct Validation
3. Betting Layer (EV + Backtest)
4. Iterative Improvement
5. Model Comparison + Recommendations

Usage:
    python experiments/full_ml_analysis.py --step 1    # Run specific step
    python experiments/full_ml_analysis.py --all       # Run all steps
"""
import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Output directory
OUTPUT_DIR = project_root / 'experiments/outputs/full_analysis'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_features() -> pd.DataFrame:
    """Load the main features dataset."""
    paths = [
        project_root / 'data/03-features/features_with_sportmonks_odds.csv',
        project_root / 'data/03-features/features_all_5leagues_with_odds.csv',
    ]
    for path in paths:
        if path.exists():
            print(f"Loading: {path.name}")
            df = pd.read_csv(path)
            df['date'] = pd.to_datetime(df['date'])
            return df
    raise FileNotFoundError("No features file found")


# =============================================================================
# STEP 1: DATA QUALITY AUDIT
# =============================================================================

def step1_data_quality_audit(df: pd.DataFrame) -> Dict:
    """
    Comprehensive data quality audit.

    Checks:
    - Missing data patterns
    - Duplicates
    - Outliers
    - Constant columns
    - Data types
    - Timestamp consistency
    - Distribution drift
    - Potential leakage signals
    """
    print("\n" + "=" * 70)
    print("STEP 1: DATA QUALITY AUDIT")
    print("=" * 70)

    results = {
        'timestamp': datetime.now().isoformat(),
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'issues': [],
        'warnings': [],
    }

    # 1.1 Basic Statistics
    print("\n1.1 Basic Statistics")
    print("-" * 40)
    print(f"Total matches: {len(df):,}")
    print(f"Total columns: {len(df.columns)}")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")

    if 'league' in df.columns:
        print(f"Leagues: {df['league'].nunique()}")
        print(df['league'].value_counts().to_string())

    results['date_range'] = {
        'start': str(df['date'].min().date()),
        'end': str(df['date'].max().date()),
    }

    # 1.2 Missing Data Analysis
    print("\n1.2 Missing Data Analysis")
    print("-" * 40)

    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        'missing_count': missing,
        'missing_pct': missing_pct
    }).sort_values('missing_pct', ascending=False)

    high_missing = missing_df[missing_df['missing_pct'] > 30]
    if len(high_missing) > 0:
        print(f"Columns with >30% missing: {len(high_missing)}")
        print(high_missing.head(10).to_string())
        results['warnings'].append(f"{len(high_missing)} columns have >30% missing data")
    else:
        print("No columns with >30% missing data")

    results['missing_data'] = {
        'high_missing_cols': len(high_missing),
        'total_missing_values': int(missing.sum()),
    }

    # 1.3 Duplicates Check
    print("\n1.3 Duplicates Check")
    print("-" * 40)

    if 'fixture_id' in df.columns:
        dup_fixtures = df['fixture_id'].duplicated().sum()
        print(f"Duplicate fixture_ids: {dup_fixtures}")
        if dup_fixtures > 0:
            results['issues'].append(f"{dup_fixtures} duplicate fixture_ids found")

    # Check for duplicate rows (excluding date columns)
    key_cols = ['home_team_name', 'away_team_name', 'date'] if 'home_team_name' in df.columns else []
    if key_cols:
        dup_rows = df.duplicated(subset=key_cols).sum()
        print(f"Duplicate matches (same teams, same date): {dup_rows}")
        if dup_rows > 0:
            results['issues'].append(f"{dup_rows} duplicate matches found")

    # 1.4 Constant Columns
    print("\n1.4 Constant/Near-Constant Columns")
    print("-" * 40)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    constant_cols = []
    near_constant_cols = []

    for col in numeric_cols:
        nunique = df[col].nunique()
        if nunique <= 1:
            constant_cols.append(col)
        elif nunique <= 3 and len(df) > 1000:
            near_constant_cols.append(col)

    print(f"Constant columns (1 unique value): {len(constant_cols)}")
    if constant_cols:
        print(f"  {constant_cols[:10]}")
    print(f"Near-constant columns (<=3 unique values): {len(near_constant_cols)}")

    results['constant_columns'] = constant_cols
    results['near_constant_columns'] = near_constant_cols

    # 1.5 Outlier Detection
    print("\n1.5 Outlier Detection (IQR method)")
    print("-" * 40)

    outlier_cols = []
    for col in numeric_cols[:50]:  # Check first 50 numeric columns
        if df[col].notna().sum() < 100:
            continue
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df[col] < Q1 - 3*IQR) | (df[col] > Q3 + 3*IQR)).sum()
        if outliers > len(df) * 0.05:  # More than 5% outliers
            outlier_cols.append((col, outliers, outliers/len(df)*100))

    if outlier_cols:
        print(f"Columns with >5% extreme outliers: {len(outlier_cols)}")
        for col, count, pct in sorted(outlier_cols, key=lambda x: -x[2])[:5]:
            print(f"  {col}: {count} ({pct:.1f}%)")
        results['warnings'].append(f"{len(outlier_cols)} columns have >5% extreme outliers")
    else:
        print("No columns with excessive outliers")

    # 1.6 Target Variable Analysis
    print("\n1.6 Target Variable Analysis")
    print("-" * 40)

    targets = {
        'home_goals': 'Goals (home)',
        'away_goals': 'Goals (away)',
        'total_corners': 'Corners',
        'total_fouls': 'Fouls',
        'home_shots': 'Shots (home)',
        'away_shots': 'Shots (away)',
    }

    target_stats = {}
    for col, name in targets.items():
        if col in df.columns:
            valid = df[col].notna().sum()
            mean = df[col].mean()
            std = df[col].std()
            print(f"{name}: n={valid}, mean={mean:.2f}, std={std:.2f}")
            target_stats[col] = {'count': int(valid), 'mean': float(mean), 'std': float(std)}

    results['target_stats'] = target_stats

    # 1.7 Timestamp Consistency
    print("\n1.7 Timestamp/Date Consistency")
    print("-" * 40)

    df_sorted = df.sort_values('date')
    date_gaps = df_sorted['date'].diff().dt.days
    max_gap = date_gaps.max()
    print(f"Max gap between matches: {max_gap} days")

    # Check for matches per season
    df['season'] = df['date'].dt.year
    matches_per_season = df.groupby('season').size()
    print(f"Matches per year:\n{matches_per_season.to_string()}")

    # 1.8 Distribution Drift Analysis
    print("\n1.8 Distribution Drift (Early vs Recent)")
    print("-" * 40)

    mid_date = df['date'].median()
    early = df[df['date'] < mid_date]
    recent = df[df['date'] >= mid_date]

    drift_cols = []
    for col in ['home_goals', 'away_goals', 'total_corners', 'total_fouls']:
        if col in df.columns and df[col].notna().sum() > 100:
            early_mean = early[col].mean()
            recent_mean = recent[col].mean()
            pct_change = (recent_mean - early_mean) / early_mean * 100 if early_mean != 0 else 0

            # KS test for distribution difference
            try:
                ks_stat, ks_pval = stats.ks_2samp(early[col].dropna(), recent[col].dropna())
                if ks_pval < 0.01:
                    drift_cols.append((col, pct_change, ks_pval))
                    print(f"{col}: early={early_mean:.2f}, recent={recent_mean:.2f} ({pct_change:+.1f}%), KS p={ks_pval:.4f} *DRIFT*")
                else:
                    print(f"{col}: early={early_mean:.2f}, recent={recent_mean:.2f} ({pct_change:+.1f}%), KS p={ks_pval:.4f}")
            except:
                pass

    if drift_cols:
        results['warnings'].append(f"Significant drift detected in {len(drift_cols)} target variables")

    # 1.9 Potential Leakage Detection
    print("\n1.9 Potential Leakage Detection")
    print("-" * 40)

    # Check for suspiciously high correlations with targets
    leakage_suspects = []

    target_cols = ['home_goals', 'away_goals', 'result', 'home_win', 'away_win']
    feature_cols = [c for c in numeric_cols if c not in target_cols and 'goals' not in c.lower()]

    for target in target_cols:
        if target not in df.columns:
            continue
        for feat in feature_cols[:100]:  # Check first 100 features
            try:
                corr = df[[feat, target]].dropna().corr().iloc[0, 1]
                if abs(corr) > 0.8:
                    leakage_suspects.append((feat, target, corr))
            except:
                pass

    if leakage_suspects:
        print(f"Suspicious high correlations (|r| > 0.8): {len(leakage_suspects)}")
        for feat, target, corr in sorted(leakage_suspects, key=lambda x: -abs(x[2]))[:10]:
            print(f"  {feat} <-> {target}: r={corr:.3f}")
        results['warnings'].append(f"{len(leakage_suspects)} potential leakage signals detected")
    else:
        print("No obvious leakage signals detected")

    # Check for odds-based features that might leak
    odds_features = [c for c in df.columns if 'odds' in c.lower() or 'line' in c.lower()]
    print(f"\nOdds-related columns: {len(odds_features)}")
    if odds_features:
        print(f"  Examples: {odds_features[:5]}")

    # 1.10 Summary
    print("\n" + "=" * 70)
    print("STEP 1 SUMMARY")
    print("=" * 70)

    print(f"\nTotal Issues: {len(results['issues'])}")
    for issue in results['issues']:
        print(f"  [ISSUE] {issue}")

    print(f"\nTotal Warnings: {len(results['warnings'])}")
    for warning in results['warnings']:
        print(f"  [WARNING] {warning}")

    # Feature count by category
    print("\nFeature Categories:")
    categories = {
        'ELO': [c for c in df.columns if 'elo' in c.lower()],
        'Form': [c for c in df.columns if 'last_n' in c.lower() or 'streak' in c.lower()],
        'EMA': [c for c in df.columns if '_ema' in c.lower()],
        'Corners': [c for c in df.columns if 'corner' in c.lower()],
        'Cards': [c for c in df.columns if 'card' in c.lower() or 'yellow' in c.lower()],
        'Shots': [c for c in df.columns if 'shot' in c.lower()],
        'Fouls': [c for c in df.columns if 'foul' in c.lower()],
        'Referee': [c for c in df.columns if 'ref_' in c.lower()],
        'Odds': [c for c in df.columns if 'odds' in c.lower() or 'prob' in c.lower()],
        'Position': [c for c in df.columns if 'position' in c.lower() or 'pts' in c.lower()],
    }

    for cat, cols in categories.items():
        print(f"  {cat}: {len(cols)} features")

    results['feature_categories'] = {k: len(v) for k, v in categories.items()}

    # Save results
    output_file = OUTPUT_DIR / 'step1_data_quality_audit.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_file}")

    return results


# =============================================================================
# STEP 2: BASELINES + CORRECT VALIDATION
# =============================================================================

def step2_baselines(df: pd.DataFrame) -> Dict:
    """
    Build baselines per target with proper time-based validation.

    For each market:
    - Train simple model (LightGBM)
    - Use time-based split (60/20/20)
    - Report: AUC, LogLoss, Brier, ECE
    """
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
    from sklearn.calibration import calibration_curve
    from lightgbm import LGBMClassifier, LGBMRegressor

    print("\n" + "=" * 70)
    print("STEP 2: BASELINES + CORRECT VALIDATION")
    print("=" * 70)

    results = {
        'timestamp': datetime.now().isoformat(),
        'markets': {},
    }

    # Define markets and targets
    markets = {
        'BTTS': {'target': 'btts', 'type': 'classification'},
        'Away Win': {'target': 'away_win', 'type': 'classification'},
        'Home Win': {'target': 'home_win', 'type': 'classification'},
        'Corners O9.5': {'target': 'corners_over_9_5', 'type': 'classification'},
        'Corners O10.5': {'target': 'corners_over_10_5', 'type': 'classification'},
        'Shots O22.5': {'target': 'shots_over_22_5', 'type': 'classification'},
        'Shots O24.5': {'target': 'shots_over_24_5', 'type': 'classification'},
        'Fouls O22.5': {'target': 'fouls_over_22_5', 'type': 'classification'},
        'Fouls O24.5': {'target': 'fouls_over_24_5', 'type': 'classification'},
        'Asian Handicap': {'target': 'goal_difference', 'type': 'regression'},
    }

    # Create target columns if they don't exist
    if 'total_corners' in df.columns:
        df['corners_over_9_5'] = (df['total_corners'] > 9.5).astype(int)
        df['corners_over_10_5'] = (df['total_corners'] > 10.5).astype(int)

    if 'home_shots' in df.columns and 'away_shots' in df.columns:
        df['total_shots'] = df['home_shots'] + df['away_shots']
        df['shots_over_22_5'] = (df['total_shots'] > 22.5).astype(int)
        df['shots_over_24_5'] = (df['total_shots'] > 24.5).astype(int)

    if 'home_fouls' in df.columns and 'away_fouls' in df.columns:
        df['total_fouls'] = df['home_fouls'] + df['away_fouls']
        df['fouls_over_22_5'] = (df['total_fouls'] > 22.5).astype(int)
        df['fouls_over_24_5'] = (df['total_fouls'] > 24.5).astype(int)

    if 'btts' not in df.columns and 'home_goals' in df.columns:
        df['btts'] = ((df['home_goals'] > 0) & (df['away_goals'] > 0)).astype(int)

    if 'goal_difference' not in df.columns and 'home_goals' in df.columns:
        df['goal_difference'] = df['home_goals'] - df['away_goals']

    # Get feature columns (exclude targets, identifiers, and LEAKY columns)
    # LEAKY = actual match outcomes (current match stats, not historical)
    exclude_cols = [
        # Identifiers
        'fixture_id', 'date', 'home_team_id', 'away_team_id', 'home_team_name', 'away_team_name',
        'league', 'season', 'round', 'time',
        # LEAKY: Match result columns
        'home_goals', 'away_goals', 'result', 'match_result', 'goal_difference', 'total_goals',
        'home_win', 'away_win', 'draw', 'btts',
        # LEAKY: Current match stats (not historical)
        'home_shots', 'away_shots', 'total_shots',
        'home_corners', 'away_corners', 'total_corners',
        'home_fouls', 'away_fouls', 'total_fouls',
        'home_cards', 'away_cards', 'total_cards',
        'home_shots_on_target', 'away_shots_on_target',
        # Created targets
        'corners_over_9_5', 'corners_over_10_5', 'corners_over_11_5',
        'shots_over_22_5', 'shots_over_24_5', 'shots_over_26_5',
        'fouls_over_22_5', 'fouls_over_24_5', 'fouls_over_26_5',
        'cards_over_3_5', 'cards_over_4_5',
    ]

    feature_cols = [c for c in df.columns if c not in exclude_cols]
    feature_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

    print(f"\nUsing {len(feature_cols)} features")

    # Sort by date for time-based split
    df = df.sort_values('date').reset_index(drop=True)
    n = len(df)
    train_end = int(0.6 * n)
    val_end = int(0.8 * n)

    print(f"\nTime-based split:")
    print(f"  Train: {train_end} matches ({df.iloc[0]['date'].date()} to {df.iloc[train_end-1]['date'].date()})")
    print(f"  Val: {val_end - train_end} matches ({df.iloc[train_end]['date'].date()} to {df.iloc[val_end-1]['date'].date()})")
    print(f"  Test: {n - val_end} matches ({df.iloc[val_end]['date'].date()} to {df.iloc[-1]['date'].date()})")

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    print("\n" + "-" * 70)
    print(f"{'Market':<20} {'Base Rate':>10} {'AUC':>8} {'LogLoss':>10} {'Brier':>8} {'ECE':>8}")
    print("-" * 70)

    for market_name, config in markets.items():
        target_col = config['target']

        if target_col not in df.columns:
            continue

        # Filter to valid data
        valid_mask = df[target_col].notna()
        if valid_mask.sum() < 1000:
            print(f"{market_name:<20} Insufficient data ({valid_mask.sum()} rows)")
            continue

        # Prepare data
        X_train = train_df[feature_cols].fillna(0)
        X_val = val_df[feature_cols].fillna(0)
        X_test = test_df[feature_cols].fillna(0)

        y_train = train_df[target_col].fillna(0)
        y_val = val_df[target_col].fillna(0)
        y_test = test_df[target_col].fillna(0)

        # Skip if target is all same value
        if y_train.nunique() <= 1 or y_test.nunique() <= 1:
            continue

        try:
            if config['type'] == 'classification':
                base_rate = y_test.mean()

                model = LGBMClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42,
                    verbose=-1
                )
                model.fit(X_train, y_train)

                y_pred_proba = model.predict_proba(X_test)[:, 1]

                auc = roc_auc_score(y_test, y_pred_proba)
                logloss = log_loss(y_test, y_pred_proba)
                brier = brier_score_loss(y_test, y_pred_proba)

                # Calculate ECE (Expected Calibration Error)
                prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=10, strategy='uniform')
                ece = np.mean(np.abs(prob_true - prob_pred))

                print(f"{market_name:<20} {base_rate:>10.1%} {auc:>8.3f} {logloss:>10.4f} {brier:>8.4f} {ece:>8.3f}")

                results['markets'][market_name] = {
                    'type': 'classification',
                    'base_rate': float(base_rate),
                    'auc': float(auc),
                    'log_loss': float(logloss),
                    'brier': float(brier),
                    'ece': float(ece),
                    'test_samples': int(len(y_test)),
                }

            else:  # regression
                model = LGBMRegressor(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42,
                    verbose=-1
                )
                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)

                mse = np.mean((y_test - y_pred) ** 2)
                mae = np.mean(np.abs(y_test - y_pred))

                print(f"{market_name:<20} {'N/A':>10} {'N/A':>8} MSE={mse:.3f} MAE={mae:.3f}")

                results['markets'][market_name] = {
                    'type': 'regression',
                    'mse': float(mse),
                    'mae': float(mae),
                    'test_samples': int(len(y_test)),
                }

        except Exception as e:
            print(f"{market_name:<20} Error: {str(e)[:40]}")

    print("-" * 70)

    # Save results
    output_file = OUTPUT_DIR / 'step2_baselines.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_file}")

    return results


# =============================================================================
# STEP 3: BETTING LAYER (EV + BACKTEST)
# =============================================================================

def step3_betting_layer(df: pd.DataFrame) -> Dict:
    """
    Betting simulation with EV calculation and threshold optimization.

    - Implied probability from odds: p_imp = 1 / odds
    - EV = (prob_model * odds) - 1
    - Bet when edge > threshold
    - Backtest on OOS data
    """
    from lightgbm import LGBMClassifier
    from sklearn.calibration import CalibratedClassifierCV

    print("\n" + "=" * 70)
    print("STEP 3: BETTING LAYER (EV + BACKTEST)")
    print("=" * 70)

    results = {
        'timestamp': datetime.now().isoformat(),
        'staking': {
            'method': 'Fractional Kelly',
            'fraction': 0.02,
            'min_odds': 1.30,
            'max_odds': 10.0,
            'max_stake': 0.05,
        },
        'markets': {},
    }

    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)
    n = len(df)
    train_end = int(0.6 * n)
    val_end = int(0.8 * n)

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    # Get feature columns (exclude LEAKY columns - current match outcomes)
    exclude_cols = [
        # Identifiers
        'fixture_id', 'date', 'home_team_id', 'away_team_id', 'home_team_name', 'away_team_name',
        'league', 'season', 'round', 'time',
        # LEAKY: Match result columns
        'home_goals', 'away_goals', 'result', 'match_result', 'goal_difference', 'total_goals',
        'home_win', 'away_win', 'draw', 'btts',
        # LEAKY: Current match stats
        'home_shots', 'away_shots', 'total_shots',
        'home_corners', 'away_corners', 'total_corners',
        'home_fouls', 'away_fouls', 'total_fouls',
        'home_cards', 'away_cards', 'total_cards',
        'home_shots_on_target', 'away_shots_on_target',
        # Created targets
        'corners_over_9_5', 'corners_over_10_5', 'corners_over_11_5',
        'shots_over_22_5', 'shots_over_24_5', 'shots_over_26_5',
        'fouls_over_22_5', 'fouls_over_24_5', 'fouls_over_26_5',
    ]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    feature_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

    # Markets with odds columns
    markets = [
        {
            'name': 'BTTS',
            'target': 'btts',
            'odds_col': 'sm_btts_yes_odds',
            'default_odds': 1.85,
        },
        {
            'name': 'Corners O9.5',
            'target': 'corners_over_9_5',
            'odds_col': None,
            'default_odds': 1.90,
        },
        {
            'name': 'Corners O10.5',
            'target': 'corners_over_10_5',
            'odds_col': None,
            'default_odds': 2.10,
        },
        {
            'name': 'Shots O22.5',
            'target': 'shots_over_22_5',
            'odds_col': None,
            'default_odds': 1.75,
        },
        {
            'name': 'Shots O24.5',
            'target': 'shots_over_24_5',
            'odds_col': None,
            'default_odds': 1.90,
        },
        {
            'name': 'Fouls O22.5',
            'target': 'fouls_over_22_5',
            'odds_col': None,
            'default_odds': 1.75,
        },
        {
            'name': 'Away Win',
            'target': 'away_win',
            'odds_col': None,
            'default_odds': 3.50,
        },
    ]

    # Create target columns
    if 'total_corners' in df.columns:
        df['corners_over_9_5'] = (df['total_corners'] > 9.5).astype(int)
        df['corners_over_10_5'] = (df['total_corners'] > 10.5).astype(int)
    if 'home_shots' in df.columns and 'away_shots' in df.columns:
        df['total_shots'] = df['home_shots'] + df['away_shots']
        df['shots_over_22_5'] = (df['total_shots'] > 22.5).astype(int)
        df['shots_over_24_5'] = (df['total_shots'] > 24.5).astype(int)
    if 'home_fouls' in df.columns and 'away_fouls' in df.columns:
        df['total_fouls'] = df['home_fouls'] + df['away_fouls']
        df['fouls_over_22_5'] = (df['total_fouls'] > 22.5).astype(int)
        df['fouls_over_24_5'] = (df['total_fouls'] > 24.5).astype(int)
    if 'btts' not in df.columns and 'home_goals' in df.columns:
        df['btts'] = ((df['home_goals'] > 0) & (df['away_goals'] > 0)).astype(int)

    # Update splits with new columns
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    print(f"\nBacktest period: {test_df['date'].min().date()} to {test_df['date'].max().date()}")
    print(f"Test matches: {len(test_df)}")

    print("\n" + "-" * 90)
    print(f"{'Market':<20} {'Threshold':>10} {'Bets':>8} {'Win%':>8} {'ROI':>10} {'Profit':>10} {'Sharpe':>8}")
    print("-" * 90)

    for market in markets:
        target_col = market['target']

        if target_col not in df.columns or df[target_col].notna().sum() < 1000:
            continue

        try:
            # Train model
            X_train = train_df[feature_cols].fillna(0)
            y_train = train_df[target_col].fillna(0)
            X_val = val_df[feature_cols].fillna(0)
            y_val = val_df[target_col].fillna(0)
            X_test = test_df[feature_cols].fillna(0)
            y_test = test_df[target_col].fillna(0)

            if y_train.nunique() <= 1:
                continue

            # Train and calibrate
            model = LGBMClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, verbose=-1)
            model.fit(X_train, y_train)

            calibrated = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
            calibrated.fit(X_val, y_val)

            # Get predictions
            y_pred_proba = calibrated.predict_proba(X_test)[:, 1]

            # Get odds
            if market['odds_col'] and market['odds_col'] in test_df.columns:
                odds = test_df[market['odds_col']].fillna(market['default_odds'])
            else:
                odds = pd.Series([market['default_odds']] * len(test_df))

            # Test multiple thresholds
            best_result = None
            best_roi = -999

            for threshold in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
                # Select bets
                mask = (y_pred_proba >= threshold) & (odds >= 1.30) & (odds <= 10.0)

                if mask.sum() < 10:
                    continue

                bet_probs = y_pred_proba[mask]
                bet_outcomes = y_test.values[mask]
                bet_odds = odds.values[mask]

                # Calculate results
                n_bets = len(bet_outcomes)
                n_wins = bet_outcomes.sum()
                win_rate = n_wins / n_bets

                # Profit calculation (flat stake)
                profits = np.where(bet_outcomes == 1, bet_odds - 1, -1)
                total_profit = profits.sum()
                roi = (total_profit / n_bets) * 100

                # Sharpe ratio (daily returns approximation)
                if len(profits) > 10:
                    sharpe = profits.mean() / profits.std() if profits.std() > 0 else 0
                else:
                    sharpe = 0

                if roi > best_roi:
                    best_roi = roi
                    best_result = {
                        'threshold': threshold,
                        'n_bets': n_bets,
                        'win_rate': win_rate,
                        'roi': roi,
                        'profit': total_profit,
                        'sharpe': sharpe,
                        'avg_odds': bet_odds.mean(),
                        'avg_prob': bet_probs.mean(),
                    }

            if best_result:
                print(f"{market['name']:<20} {best_result['threshold']:>10.2f} {best_result['n_bets']:>8d} "
                      f"{best_result['win_rate']:>7.1%} {best_result['roi']:>9.1f}% "
                      f"{best_result['profit']:>9.1f}u {best_result['sharpe']:>8.2f}")

                results['markets'][market['name']] = best_result

        except Exception as e:
            print(f"{market['name']:<20} Error: {str(e)[:50]}")

    print("-" * 90)

    # Summary
    profitable = [m for m, r in results['markets'].items() if r['roi'] > 0]
    print(f"\nProfitable markets: {len(profitable)}/{len(results['markets'])}")
    print(f"Markets: {', '.join(profitable)}")

    total_bets = sum(r['n_bets'] for r in results['markets'].values())
    total_profit = sum(r['profit'] for r in results['markets'].values())
    print(f"Total bets: {total_bets}, Total profit: {total_profit:.1f} units")

    # Save results
    output_file = OUTPUT_DIR / 'step3_betting_layer.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_file}")

    return results


# =============================================================================
# STEP 4: ITERATIVE IMPROVEMENT
# =============================================================================

def step4_iterative_improvement(df: pd.DataFrame) -> Dict:
    """
    Iterative model improvement with feature selection, tuning, and calibration.

    For each market:
    1. Feature selection (importance-based)
    2. Hyperparameter tuning
    3. Calibration comparison
    4. Report improvement vs baseline
    """
    from lightgbm import LGBMClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import roc_auc_score, brier_score_loss
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print("\n" + "=" * 70)
    print("STEP 4: ITERATIVE IMPROVEMENT")
    print("=" * 70)

    results = {
        'timestamp': datetime.now().isoformat(),
        'markets': {},
    }

    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)
    n = len(df)
    train_end = int(0.6 * n)
    val_end = int(0.8 * n)

    # Create target columns
    if 'total_corners' in df.columns:
        df['corners_over_9_5'] = (df['total_corners'] > 9.5).astype(int)
        df['corners_over_10_5'] = (df['total_corners'] > 10.5).astype(int)
    if 'home_shots' in df.columns and 'away_shots' in df.columns:
        df['total_shots'] = df['home_shots'] + df['away_shots']
        df['shots_over_22_5'] = (df['total_shots'] > 22.5).astype(int)
        df['shots_over_24_5'] = (df['total_shots'] > 24.5).astype(int)
    if 'home_fouls' in df.columns and 'away_fouls' in df.columns:
        df['total_fouls'] = df['home_fouls'] + df['away_fouls']
        df['fouls_over_22_5'] = (df['total_fouls'] > 22.5).astype(int)
        df['fouls_over_24_5'] = (df['total_fouls'] > 24.5).astype(int)
    if 'btts' not in df.columns and 'home_goals' in df.columns:
        df['btts'] = ((df['home_goals'] > 0) & (df['away_goals'] > 0)).astype(int)

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    # Get feature columns (exclude LEAKY columns - current match outcomes)
    exclude_cols = [
        # Identifiers
        'fixture_id', 'date', 'home_team_id', 'away_team_id', 'home_team_name', 'away_team_name',
        'league', 'season', 'round', 'time',
        # LEAKY: Match result columns
        'home_goals', 'away_goals', 'result', 'match_result', 'goal_difference', 'total_goals',
        'home_win', 'away_win', 'draw', 'btts',
        # LEAKY: Current match stats
        'home_shots', 'away_shots', 'total_shots',
        'home_corners', 'away_corners', 'total_corners',
        'home_fouls', 'away_fouls', 'total_fouls',
        'home_cards', 'away_cards', 'total_cards',
        'home_shots_on_target', 'away_shots_on_target',
        # Created targets
        'corners_over_9_5', 'corners_over_10_5', 'corners_over_11_5',
        'shots_over_22_5', 'shots_over_24_5', 'shots_over_26_5',
        'fouls_over_22_5', 'fouls_over_24_5', 'fouls_over_26_5',
    ]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    feature_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

    markets = [
        ('Fouls O22.5', 'fouls_over_22_5', 1.75),
        ('Shots O24.5', 'shots_over_24_5', 1.90),
        ('Corners O10.5', 'corners_over_10_5', 2.10),
        ('BTTS', 'btts', 1.85),
    ]

    for market_name, target_col, default_odds in markets:
        print(f"\n{'='*60}")
        print(f"Optimizing: {market_name}")
        print(f"{'='*60}")

        if target_col not in df.columns:
            print(f"Target column {target_col} not found")
            continue

        X_train = train_df[feature_cols].fillna(0)
        y_train = train_df[target_col].fillna(0)
        X_val = val_df[feature_cols].fillna(0)
        y_val = val_df[target_col].fillna(0)
        X_test = test_df[feature_cols].fillna(0)
        y_test = test_df[target_col].fillna(0)

        if y_train.nunique() <= 1:
            continue

        # 4.1 Baseline
        print("\n4.1 Baseline model...")
        baseline_model = LGBMClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, verbose=-1)
        baseline_model.fit(X_train, y_train)
        baseline_auc = roc_auc_score(y_test, baseline_model.predict_proba(X_test)[:, 1])
        print(f"  Baseline AUC: {baseline_auc:.4f}")

        # 4.2 Feature Selection
        print("\n4.2 Feature selection (importance-based)...")
        importances = pd.Series(baseline_model.feature_importances_, index=feature_cols)
        top_features = importances.nlargest(50).index.tolist()
        print(f"  Selected {len(top_features)} features")

        X_train_sel = X_train[top_features]
        X_val_sel = X_val[top_features]
        X_test_sel = X_test[top_features]

        # 4.3 Hyperparameter Tuning
        print("\n4.3 Hyperparameter tuning (30 trials)...")

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'num_leaves': trial.suggest_int('num_leaves', 10, 100),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0),
                'verbose': -1,
                'random_state': 42,
            }
            model = LGBMClassifier(**params)
            model.fit(X_train_sel, y_train)
            return roc_auc_score(y_val, model.predict_proba(X_val_sel)[:, 1])

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=30, show_progress_bar=False)

        best_params = study.best_params
        print(f"  Best params: {best_params}")

        # 4.4 Train tuned model
        print("\n4.4 Training tuned model...")
        tuned_model = LGBMClassifier(**best_params, verbose=-1, random_state=42)
        tuned_model.fit(X_train_sel, y_train)
        tuned_auc = roc_auc_score(y_test, tuned_model.predict_proba(X_test_sel)[:, 1])
        print(f"  Tuned AUC: {tuned_auc:.4f}")

        # 4.5 Calibration comparison
        print("\n4.5 Calibration comparison...")

        calibration_results = {}
        for method in ['sigmoid', 'isotonic']:
            calibrated = CalibratedClassifierCV(tuned_model, method=method, cv='prefit')
            calibrated.fit(X_val_sel, y_val)
            proba = calibrated.predict_proba(X_test_sel)[:, 1]
            brier = brier_score_loss(y_test, proba)
            calibration_results[method] = {'brier': brier, 'model': calibrated}
            print(f"  {method}: Brier={brier:.4f}")

        best_calibration = min(calibration_results.items(), key=lambda x: x[1]['brier'])
        best_method = best_calibration[0]
        best_model = best_calibration[1]['model']

        # 4.6 Betting backtest with tuned model
        print("\n4.6 Betting backtest...")
        proba = best_model.predict_proba(X_test_sel)[:, 1]

        best_roi = -999
        best_config = None

        for threshold in [0.55, 0.60, 0.65, 0.70, 0.75]:
            mask = proba >= threshold
            if mask.sum() < 10:
                continue

            bet_outcomes = y_test.values[mask]
            n_bets = len(bet_outcomes)
            profits = np.where(bet_outcomes == 1, default_odds - 1, -1)
            roi = (profits.sum() / n_bets) * 100

            if roi > best_roi:
                best_roi = roi
                best_config = {
                    'threshold': threshold,
                    'n_bets': n_bets,
                    'win_rate': bet_outcomes.mean(),
                    'roi': roi,
                    'profit': profits.sum(),
                }

        if best_config:
            print(f"  Best: threshold={best_config['threshold']}, bets={best_config['n_bets']}, "
                  f"win_rate={best_config['win_rate']:.1%}, ROI={best_config['roi']:+.1f}%")

        # 4.7 Summary
        print(f"\nImprovement Summary for {market_name}:")
        print(f"  AUC: {baseline_auc:.4f} -> {tuned_auc:.4f} ({(tuned_auc-baseline_auc)*100:+.2f}pp)")
        print(f"  Features: {len(feature_cols)} -> {len(top_features)}")
        print(f"  Calibration: {best_method}")
        if best_config:
            print(f"  Best ROI: {best_config['roi']:+.1f}%")

        results['markets'][market_name] = {
            'baseline_auc': float(baseline_auc),
            'tuned_auc': float(tuned_auc),
            'improvement': float(tuned_auc - baseline_auc),
            'best_params': best_params,
            'calibration_method': best_method,
            'n_features': len(top_features),
            'betting': best_config,
        }

    # Save results
    output_file = OUTPUT_DIR / 'step4_iterative_improvement.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n\nResults saved to: {output_file}")

    return results


# =============================================================================
# STEP 5: MODEL COMPARISON + RECOMMENDATIONS
# =============================================================================

def step5_recommendations(df: pd.DataFrame) -> Dict:
    """
    Final model comparison and deployment recommendations.

    1. Load all previous step results
    2. Create comparison table
    3. Rank by prediction quality and profitability
    4. Generate deployment recommendations
    """
    print("\n" + "=" * 70)
    print("STEP 5: MODEL COMPARISON + RECOMMENDATIONS")
    print("=" * 70)

    results = {
        'timestamp': datetime.now().isoformat(),
        'recommendations': [],
    }

    # Load previous results
    step2_file = OUTPUT_DIR / 'step2_baselines.json'
    step3_file = OUTPUT_DIR / 'step3_betting_layer.json'
    step4_file = OUTPUT_DIR / 'step4_iterative_improvement.json'

    step2 = json.load(open(step2_file)) if step2_file.exists() else {}
    step3 = json.load(open(step3_file)) if step3_file.exists() else {}
    step4 = json.load(open(step4_file)) if step4_file.exists() else {}

    # Combine results
    markets = set()
    markets.update(step2.get('markets', {}).keys())
    markets.update(step3.get('markets', {}).keys())
    markets.update(step4.get('markets', {}).keys())

    comparison = []
    for market in sorted(markets):
        row = {'market': market}

        # Step 2 (baseline)
        if market in step2.get('markets', {}):
            s2 = step2['markets'][market]
            row['baseline_auc'] = s2.get('auc', 0)
            row['baseline_brier'] = s2.get('brier', 0)
            row['base_rate'] = s2.get('base_rate', 0)

        # Step 3 (betting)
        if market in step3.get('markets', {}):
            s3 = step3['markets'][market]
            row['betting_roi'] = s3.get('roi', 0)
            row['betting_bets'] = s3.get('n_bets', 0)
            row['betting_winrate'] = s3.get('win_rate', 0)
            row['betting_threshold'] = s3.get('threshold', 0)

        # Step 4 (improved)
        if market in step4.get('markets', {}):
            s4 = step4['markets'][market]
            row['tuned_auc'] = s4.get('tuned_auc', 0)
            row['improvement'] = s4.get('improvement', 0)
            if s4.get('betting'):
                row['tuned_roi'] = s4['betting'].get('roi', 0)
                row['tuned_bets'] = s4['betting'].get('n_bets', 0)

        comparison.append(row)

    comparison_df = pd.DataFrame(comparison)

    print("\n" + "=" * 100)
    print("COMPARISON TABLE")
    print("=" * 100)

    # Display comparison
    print(f"\n{'Market':<20} {'Base Rate':>10} {'AUC':>8} {'Tuned AUC':>10} {'ROI':>10} {'Tuned ROI':>10} {'Bets':>8}")
    print("-" * 100)

    for _, row in comparison_df.iterrows():
        print(f"{row['market']:<20} "
              f"{row.get('base_rate', 0):>9.1%} "
              f"{row.get('baseline_auc', 0):>8.3f} "
              f"{row.get('tuned_auc', 0):>10.3f} "
              f"{row.get('betting_roi', 0):>9.1f}% "
              f"{row.get('tuned_roi', 0):>9.1f}% "
              f"{row.get('tuned_bets', row.get('betting_bets', 0)):>8.0f}")

    print("-" * 100)

    # Ranking by profitability
    print("\n" + "=" * 70)
    print("RANKING BY PROFITABILITY (ROI)")
    print("=" * 70)

    roi_col = 'tuned_roi' if 'tuned_roi' in comparison_df.columns else 'betting_roi'
    if roi_col in comparison_df.columns:
        ranked = comparison_df.dropna(subset=[roi_col]).sort_values(roi_col, ascending=False)

        print(f"\n{'Rank':<6} {'Market':<20} {'ROI':>10} {'Bets':>8} {'Status':>15}")
        print("-" * 60)

        for i, (_, row) in enumerate(ranked.iterrows(), 1):
            roi = row.get(roi_col, 0)
            bets = row.get('tuned_bets', row.get('betting_bets', 0))
            status = "DEPLOY" if roi > 10 and bets >= 20 else ("MONITOR" if roi > 0 else "DISABLED")

            print(f"{i:<6} {row['market']:<20} {roi:>9.1f}% {bets:>8.0f} {status:>15}")

            results['recommendations'].append({
                'rank': i,
                'market': row['market'],
                'roi': roi,
                'bets': int(bets),
                'status': status,
            })

    # Final recommendations
    print("\n" + "=" * 70)
    print("DEPLOYMENT RECOMMENDATIONS")
    print("=" * 70)

    deploy = [r for r in results['recommendations'] if r['status'] == 'DEPLOY']
    monitor = [r for r in results['recommendations'] if r['status'] == 'MONITOR']
    disabled = [r for r in results['recommendations'] if r['status'] == 'DISABLED']

    print(f"\nDEPLOY ({len(deploy)} markets) - Production ready:")
    for r in deploy:
        print(f"  - {r['market']}: ROI={r['roi']:+.1f}%, Bets={r['bets']}")

    print(f"\nMONITOR ({len(monitor)} markets) - Paper trade first:")
    for r in monitor:
        print(f"  - {r['market']}: ROI={r['roi']:+.1f}%, Bets={r['bets']}")

    print(f"\nDISABLED ({len(disabled)} markets) - Negative or insufficient edge:")
    for r in disabled:
        print(f"  - {r['market']}: ROI={r['roi']:+.1f}%, Bets={r['bets']}")

    # Risk warnings
    print("\n" + "-" * 70)
    print("RISK WARNINGS")
    print("-" * 70)
    print("1. Backtest ROI != Live ROI (execution slippage, odds movement)")
    print("2. Markets with <50 bets have high variance")
    print("3. Monitor for drift (COVID era, rule changes, tactical evolution)")
    print("4. Use fractional Kelly (2-5%) to manage bankroll risk")
    print("5. Set daily stop-loss (10%) and take-profit (20%)")

    # Save results
    output_file = OUTPUT_DIR / 'step5_recommendations.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_file}")

    # Save comparison table
    comparison_df.to_csv(OUTPUT_DIR / 'model_comparison.csv', index=False)
    print(f"Comparison table saved to: {OUTPUT_DIR / 'model_comparison.csv'}")

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Full ML Betting Analysis Pipeline')
    parser.add_argument('--step', type=int, choices=[1, 2, 3, 4, 5], help='Run specific step')
    parser.add_argument('--all', action='store_true', help='Run all steps')
    args = parser.parse_args()

    print("=" * 70)
    print("FULL ML BETTING ANALYSIS PIPELINE")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Load data
    df = load_features()
    print(f"Loaded {len(df)} matches with {len(df.columns)} columns")

    if args.all or args.step == 1:
        step1_data_quality_audit(df)

    if args.all or args.step == 2:
        step2_baselines(df)

    if args.all or args.step == 3:
        step3_betting_layer(df)

    if args.all or args.step == 4:
        step4_iterative_improvement(df)

    if args.all or args.step == 5:
        step5_recommendations(df)

    print("\n" + "=" * 70)
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == '__main__':
    main()
