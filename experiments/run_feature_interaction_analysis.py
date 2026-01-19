#!/usr/bin/env python
"""
Feature Interaction Analysis

Comprehensive analysis of feature interactions using:
1. xgbfir - XGBoost Feature Interaction Ranking
2. SHAP interaction values
3. Partial Dependence Plots (PDP)

This helps identify which feature pairs interact strongly,
enabling creation of explicit interaction features.

Usage:
    python experiments/run_feature_interaction_analysis.py --market fouls
    python experiments/run_feature_interaction_analysis.py --market corners
    python experiments/run_feature_interaction_analysis.py --market all
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from xgboost import XGBClassifier
import xgbfir
import shap
from sklearn.model_selection import TimeSeriesSplit
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt

# Output directory
OUTPUT_DIR = Path("experiments/outputs/feature_interactions")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_features_data():
    """Load the main features file and merge with match stats for targets."""
    features_path = Path("data/03-features/features_all_5leagues_with_odds.csv")
    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")

    df = pd.read_csv(features_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    print(f"Loaded {len(df)} matches from {df['date'].min()} to {df['date'].max()}")

    # Load match stats from raw data to get targets
    leagues = ['premier_league', 'la_liga', 'serie_a', 'bundesliga', 'ligue_1']
    seasons = ['2024', '2025', '2023', '2022', '2021', '2020', '2019']

    all_stats = []
    for league in leagues:
        for season in seasons:
            stats_path = Path(f"data/01-raw/{league}/{season}/match_stats.parquet")
            if stats_path.exists():
                try:
                    stats = pd.read_parquet(stats_path)
                    if 'fixture_id' in stats.columns:
                        all_stats.append(stats)
                except Exception as e:
                    pass

    if all_stats:
        stats_df = pd.concat(all_stats, ignore_index=True)
        print(f"Loaded {len(stats_df)} match stats records")

        # Compute totals
        if 'home_fouls' in stats_df.columns:
            stats_df['total_fouls'] = stats_df['home_fouls'] + stats_df['away_fouls']
        if 'home_corners' in stats_df.columns:
            stats_df['total_corners'] = stats_df['home_corners'] + stats_df['away_corners']
        if 'home_shots' in stats_df.columns:
            stats_df['total_shots'] = stats_df['home_shots'] + stats_df['away_shots']

        # Merge with features
        merge_cols = ['fixture_id']
        target_cols = ['total_fouls', 'total_corners', 'total_shots',
                       'home_fouls', 'away_fouls', 'home_corners', 'away_corners',
                       'home_shots', 'away_shots']
        available_cols = [c for c in target_cols if c in stats_df.columns]

        if 'fixture_id' in df.columns and available_cols:
            df = df.merge(stats_df[merge_cols + available_cols].drop_duplicates(),
                         on='fixture_id', how='left', suffixes=('', '_stats'))
            print(f"Merged match stats. Fouls available: {df['total_fouls'].notna().sum()}")

    return df


def get_market_config(market: str) -> dict:
    """Get target and feature config for each market."""
    configs = {
        'fouls': {
            'target_col': 'total_fouls',
            'target_line': 26.5,
            'key_features': [
                'ref_fouls_avg', 'ref_fouls_bias', 'ref_cards_avg',
                'home_fouls_ema', 'away_fouls_ema',
                'home_avg_yellows', 'away_avg_yellows',
                'home_elo', 'away_elo', 'elo_diff',
                'is_derby', 'match_importance',
                'home_league_position', 'away_league_position',
            ],
            'exclude_patterns': [
                'fixture_id', 'date', 'season', 'home_team', 'away_team',
                'home_goals', 'away_goals', 'total_goals', 'result',
                'home_win', 'draw', 'away_win', 'btts',
                'total_fouls', 'home_fouls', 'away_fouls',  # Target leakage
                'total_corners', 'home_corners', 'away_corners',
                'total_cards', 'total_shots',
            ]
        },
        'corners': {
            'target_col': 'total_corners',
            'target_line': 9.5,
            'key_features': [
                'ref_corners_avg', 'ref_corners_bias',
                'home_corners_won_ema', 'away_corners_won_ema',
                'home_shots_ema', 'away_shots_ema',
                'home_elo', 'away_elo', 'elo_diff',
                'home_attack_strength', 'away_attack_strength',
            ],
            'exclude_patterns': [
                'fixture_id', 'date', 'season', 'home_team', 'away_team',
                'home_goals', 'away_goals', 'total_goals', 'result',
                'home_win', 'draw', 'away_win', 'btts',
                'total_corners', 'home_corners', 'away_corners',  # Target leakage
                'total_fouls', 'total_cards', 'total_shots',
            ]
        },
        'shots': {
            'target_col': 'total_shots',
            'target_line': 24.5,
            'key_features': [
                'home_shots_ema', 'away_shots_ema',
                'home_shots_on_target_ema', 'away_shots_on_target_ema',
                'home_attack_strength', 'away_attack_strength',
                'home_elo', 'away_elo', 'elo_diff',
            ],
            'exclude_patterns': [
                'fixture_id', 'date', 'season', 'home_team', 'away_team',
                'home_goals', 'away_goals', 'total_goals', 'result',
                'home_win', 'draw', 'away_win', 'btts',
                'total_shots', 'home_shots', 'away_shots',  # Target leakage
                'total_corners', 'total_fouls', 'total_cards',
            ]
        },
        'btts': {
            'target_col': 'btts',
            'target_line': None,  # Binary target
            'key_features': [
                'home_attack_strength', 'away_attack_strength',
                'home_defense_strength', 'away_defense_strength',
                'home_xg_poisson', 'away_xg_poisson',
                'home_clean_sheet_rate', 'away_clean_sheet_rate',
                'home_elo', 'away_elo', 'elo_diff',
            ],
            'exclude_patterns': [
                'fixture_id', 'date', 'season', 'home_team', 'away_team',
                'home_goals', 'away_goals', 'total_goals', 'result',
                'home_win', 'draw', 'away_win',
                'btts',  # Target
            ]
        },
    }
    return configs.get(market, configs['fouls'])


def prepare_data(df: pd.DataFrame, config: dict) -> tuple:
    """Prepare features and target for analysis."""
    target_col = config['target_col']
    target_line = config['target_line']

    # Check if target exists
    if target_col not in df.columns:
        print(f"Target column '{target_col}' not found, computing from components...")
        if target_col == 'total_fouls' and 'home_fouls' in df.columns:
            df['total_fouls'] = df['home_fouls'] + df['away_fouls']
        elif target_col == 'total_corners' and 'home_corners' in df.columns:
            df['total_corners'] = df['home_corners'] + df['away_corners']
        elif target_col == 'total_shots' and 'home_shots' in df.columns:
            df['total_shots'] = df['home_shots'] + df['away_shots']

    # Filter rows with target
    df_valid = df[df[target_col].notna()].copy()
    print(f"Valid rows with target: {len(df_valid)}")

    # Create binary target
    if target_line is not None:
        y = (df_valid[target_col] > target_line).astype(int)
    else:
        y = df_valid[target_col].astype(int)

    # Select features
    exclude_cols = []
    for pattern in config['exclude_patterns']:
        exclude_cols.extend([c for c in df_valid.columns if pattern in c.lower()])

    feature_cols = [c for c in df_valid.columns
                    if c not in exclude_cols
                    and df_valid[c].dtype in ['int64', 'float64']
                    and df_valid[c].notna().mean() > 0.5]  # At least 50% non-null

    X = df_valid[feature_cols].copy()

    # Fill NaN with median
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())

    print(f"Features: {len(feature_cols)}, Target balance: {y.mean():.2%}")
    return X, y, feature_cols


def run_xgbfir_analysis(X: pd.DataFrame, y: pd.Series, market: str):
    """Run xgbfir feature interaction analysis."""
    print(f"\n{'='*60}")
    print(f"Running xgbfir Analysis for {market.upper()}")
    print(f"{'='*60}")

    # Train XGBoost model
    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X, y)

    # Run xgbfir
    output_file = OUTPUT_DIR / f"xgbfir_{market}.xlsx"
    xgbfir.saveXgbFI(
        model,
        feature_names=X.columns.tolist(),
        OutputXlsxFile=str(output_file)
    )
    print(f"Saved xgbfir results to: {output_file}")

    # Parse and summarize top interactions
    try:
        interactions_df = pd.read_excel(output_file, sheet_name='Interaction Depth 1')
        print(f"\nTop 15 Feature Interactions (Depth 1):")
        print("-" * 50)
        for i, row in interactions_df.head(15).iterrows():
            print(f"{i+1}. {row['Interaction']} (Gain: {row['Gain']:.4f})")

        # Save as JSON for easier parsing
        top_interactions = interactions_df.head(30).to_dict('records')
        with open(OUTPUT_DIR / f"xgbfir_{market}_top30.json", 'w') as f:
            json.dump(top_interactions, f, indent=2)

    except Exception as e:
        print(f"Could not parse interactions: {e}")

    return model


def run_shap_interaction_analysis(model, X: pd.DataFrame, market: str, n_samples: int = 500):
    """Run SHAP interaction values analysis."""
    print(f"\n{'='*60}")
    print(f"Running SHAP Interaction Analysis for {market.upper()}")
    print(f"{'='*60}")

    # Sample for speed
    if len(X) > n_samples:
        X_sample = X.sample(n=n_samples, random_state=42)
    else:
        X_sample = X

    # Create explainer - use predict_proba for compatibility
    try:
        # Use the model's predict function wrapped for SHAP
        explainer = shap.Explainer(model.predict_proba, X_sample, algorithm='permutation')
        shap_values = explainer(X_sample).values[:, :, 1]  # Get class 1 values
    except Exception as e1:
        print(f"Permutation explainer failed: {e1}")
        try:
            # Try KernelExplainer as last resort
            explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X_sample, 100))
            shap_values = explainer.shap_values(X_sample)[1]  # Class 1
        except Exception as e2:
            print(f"KernelExplainer also failed: {e2}")
            print("Skipping SHAP analysis due to compatibility issues")
            return

    # Summary plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"shap_summary_{market}.png", dpi=150)
    plt.close()
    print(f"Saved SHAP summary plot")

    # Get interaction values (slower)
    print("Computing SHAP interaction values (this may take a while)...")
    try:
        shap_interaction_values = explainer.shap_interaction_values(X_sample)

        # Compute mean absolute interaction for each pair
        n_features = X_sample.shape[1]
        interaction_matrix = np.zeros((n_features, n_features))

        for i in range(n_features):
            for j in range(n_features):
                if i != j:
                    interaction_matrix[i, j] = np.abs(shap_interaction_values[:, i, j]).mean()

        # Get top interactions
        interactions = []
        for i in range(n_features):
            for j in range(i+1, n_features):
                interactions.append({
                    'feature1': X_sample.columns[i],
                    'feature2': X_sample.columns[j],
                    'interaction_strength': interaction_matrix[i, j] + interaction_matrix[j, i]
                })

        interactions_df = pd.DataFrame(interactions)
        interactions_df = interactions_df.sort_values('interaction_strength', ascending=False)

        print(f"\nTop 15 SHAP Feature Interactions:")
        print("-" * 60)
        for i, row in interactions_df.head(15).iterrows():
            print(f"{row['feature1']} x {row['feature2']}: {row['interaction_strength']:.4f}")

        # Save
        interactions_df.head(50).to_csv(OUTPUT_DIR / f"shap_interactions_{market}.csv", index=False)
        print(f"Saved SHAP interactions to: shap_interactions_{market}.csv")

        # Interaction heatmap for top features
        top_features = interactions_df.head(20)['feature1'].unique()[:10]
        top_idx = [list(X_sample.columns).index(f) for f in top_features if f in X_sample.columns]

        if len(top_idx) >= 5:
            plt.figure(figsize=(10, 8))
            sub_matrix = interaction_matrix[np.ix_(top_idx, top_idx)]
            plt.imshow(sub_matrix, cmap='YlOrRd')
            plt.xticks(range(len(top_idx)), [X_sample.columns[i] for i in top_idx], rotation=45, ha='right')
            plt.yticks(range(len(top_idx)), [X_sample.columns[i] for i in top_idx])
            plt.colorbar(label='Interaction Strength')
            plt.title(f'SHAP Interaction Heatmap - {market.upper()}')
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / f"shap_interaction_heatmap_{market}.png", dpi=150)
            plt.close()
            print(f"Saved SHAP interaction heatmap")

    except Exception as e:
        print(f"SHAP interaction analysis failed: {e}")
        print("Falling back to regular SHAP values only")


def run_pdp_analysis(model, X: pd.DataFrame, config: dict, market: str):
    """Run Partial Dependence Plot analysis."""
    print(f"\n{'='*60}")
    print(f"Running PDP Analysis for {market.upper()}")
    print(f"{'='*60}")

    # Get key features that exist in X
    key_features = [f for f in config['key_features'] if f in X.columns]

    if len(key_features) < 3:
        print(f"Not enough key features found. Available: {key_features}")
        # Fall back to top importance features
        importances = pd.Series(model.feature_importances_, index=X.columns)
        key_features = importances.nlargest(6).index.tolist()

    print(f"Analyzing features: {key_features[:6]}")

    # Sample for speed
    X_sample = X.sample(n=min(1000, len(X)), random_state=42)

    # Individual PDPs
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, feature in enumerate(key_features[:6]):
        try:
            PartialDependenceDisplay.from_estimator(
                model, X_sample, features=[feature],
                ax=axes[i], kind='both'
            )
            axes[i].set_title(f'{feature}')
        except Exception as e:
            axes[i].text(0.5, 0.5, f"Error: {str(e)[:30]}", ha='center', va='center')

    plt.suptitle(f'Partial Dependence Plots - {market.upper()}', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"pdp_{market}.png", dpi=150)
    plt.close()
    print(f"Saved PDP plots")

    # 2D PDP for top interaction (if we have enough features)
    if len(key_features) >= 2:
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            PartialDependenceDisplay.from_estimator(
                model, X_sample,
                features=[(key_features[0], key_features[1])],
                ax=ax, kind='average'
            )
            plt.title(f'2D PDP: {key_features[0]} x {key_features[1]} - {market.upper()}')
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / f"pdp_2d_{market}.png", dpi=150)
            plt.close()
            print(f"Saved 2D PDP plot")
        except Exception as e:
            print(f"2D PDP failed: {e}")


def analyze_market(market: str, df: pd.DataFrame):
    """Run full analysis for a single market."""
    print(f"\n{'#'*70}")
    print(f"# ANALYZING MARKET: {market.upper()}")
    print(f"{'#'*70}")

    config = get_market_config(market)

    try:
        X, y, feature_cols = prepare_data(df, config)

        if len(X) < 100:
            print(f"Not enough data for {market}: {len(X)} samples")
            return

        # Run all analyses
        model = run_xgbfir_analysis(X, y, market)
        run_shap_interaction_analysis(model, X, market)
        run_pdp_analysis(model, X, config, market)

        # Save feature importance
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        importance_df.to_csv(OUTPUT_DIR / f"feature_importance_{market}.csv", index=False)

        print(f"\nTop 20 Features by Importance:")
        print("-" * 40)
        for i, row in importance_df.head(20).iterrows():
            print(f"{row['feature']}: {row['importance']:.4f}")

    except Exception as e:
        print(f"Error analyzing {market}: {e}")
        import traceback
        traceback.print_exc()


def create_summary_report():
    """Create a summary report of all analyses."""
    print(f"\n{'='*70}")
    print("Creating Summary Report")
    print(f"{'='*70}")

    report = {
        'generated_at': datetime.now().isoformat(),
        'markets_analyzed': [],
        'top_interactions': {},
        'recommendations': []
    }

    # Load xgbfir results
    for market in ['fouls', 'corners', 'shots', 'btts']:
        json_file = OUTPUT_DIR / f"xgbfir_{market}_top30.json"
        if json_file.exists():
            with open(json_file) as f:
                interactions = json.load(f)
            report['markets_analyzed'].append(market)
            report['top_interactions'][market] = interactions[:10]

    # Generate recommendations
    report['recommendations'] = [
        "Consider creating explicit interaction features for top pairs",
        "Use PDP plots to identify optimal feature thresholds",
        "Features with high SHAP interaction might benefit from binning",
        "Low-importance features can be candidates for removal"
    ]

    # Save report
    with open(OUTPUT_DIR / "interaction_analysis_report.json", 'w') as f:
        json.dump(report, f, indent=2)

    print(f"Summary report saved to: {OUTPUT_DIR / 'interaction_analysis_report.json'}")

    # Print summary
    print(f"\nAnalysis Complete!")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Files generated:")
    for f in sorted(OUTPUT_DIR.glob("*")):
        print(f"  - {f.name}")


def main():
    parser = argparse.ArgumentParser(description='Feature Interaction Analysis')
    parser.add_argument('--market', type=str, default='fouls',
                       choices=['fouls', 'corners', 'shots', 'btts', 'all'],
                       help='Market to analyze')
    args = parser.parse_args()

    # Load data
    df = load_features_data()

    # Run analysis
    if args.market == 'all':
        for market in ['fouls', 'corners', 'shots', 'btts']:
            analyze_market(market, df)
    else:
        analyze_market(args.market, df)

    # Create summary
    create_summary_report()


if __name__ == "__main__":
    main()
