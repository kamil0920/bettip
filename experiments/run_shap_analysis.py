#!/usr/bin/env python3
"""
SHAP (SHapley Additive exPlanations) feature importance analysis.

This script provides detailed feature importance analysis using SHAP values,
which explain how each feature contributes to individual predictions.

Usage:
    uv run python experiments/run_shap_analysis.py
    uv run python experiments/run_shap_analysis.py --model xgboost --features-file features_v2.csv
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.config_loader import load_config
from src.ml.models import ModelFactory


EXCLUDE_COLUMNS = [
    "fixture_id", "date", "home_team_id", "home_team_name",
    "away_team_id", "away_team_name", "round",
    "home_win", "draw", "away_win", "match_result",
    "total_goals", "goal_difference"
]


def setup_logging() -> None:
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run SHAP feature importance analysis"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/local.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="xgboost",
        choices=["xgboost", "lightgbm", "catboost", "random_forest"],
        help="Model type to analyze",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="home_win",
        help="Target variable",
    )
    parser.add_argument(
        "--features-file",
        type=str,
        default="features_v2.csv",
        help="Features file name",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/shap",
        help="Directory for output plots",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of top features to display",
    )
    return parser.parse_args()


def load_data(config, features_file: str, target: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
    """Load and prepare data."""
    features_path = config.get_features_dir() / features_file

    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")

    df = pd.read_csv(features_path)
    df = df.sort_values("date")

    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLUMNS]
    X = df[feature_cols].fillna(0)
    y = df[target]

    # Time-based split
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    return X_train, X_test, y_train, y_test, feature_cols


def run_shap_analysis(
    config_path: str,
    model_type: str,
    target: str,
    features_file: str,
    output_dir: str,
    top_n: int,
) -> Dict:
    """Run SHAP analysis and generate visualizations."""
    logger = logging.getLogger(__name__)

    logger.info("=" * 70)
    logger.info("SHAP FEATURE IMPORTANCE ANALYSIS")
    logger.info("=" * 70)
    logger.info(f"Model: {model_type}")
    logger.info(f"Target: {target}")
    logger.info(f"Features file: {features_file}")
    logger.info("=" * 70)

    # Load data
    config = load_config(config_path)
    X_train, X_test, y_train, y_test, feature_cols = load_data(
        config, features_file, target
    )

    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    logger.info(f"Features: {len(feature_cols)}")

    # Train model
    logger.info(f"\nTraining {model_type}...")
    model = ModelFactory.create(model_type)
    model.fit(X_train, y_train)

    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    logger.info(f"Train accuracy: {train_acc:.4f}")
    logger.info(f"Test accuracy: {test_acc:.4f}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Calculate SHAP values
    logger.info("\nCalculating SHAP values...")

    if model_type == "xgboost":
        # Fix for XGBoost 3.x + SHAP compatibility issue
        # Use predict_proba function with KernelExplainer or sample-based Explainer
        background = shap.sample(X_train, 100)  # Sample background data
        explainer = shap.Explainer(model.predict_proba, background)
        shap_values = explainer(X_test).values
        # Use positive class (index 1) for binary classification
        if len(shap_values.shape) == 3:
            shap_values = shap_values[:, :, 1]
    elif model_type in ["lightgbm", "catboost"]:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
    else:
        # For other models, use TreeExplainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

    # Handle binary classification (use positive class)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # 1. Summary plot (bar)
    logger.info("\nGenerating SHAP summary bar plot...")
    plt.figure(figsize=(12, 10))
    shap.summary_plot(
        shap_values,
        X_test,
        feature_names=feature_cols,
        plot_type="bar",
        max_display=top_n,
        show=False,
    )
    plt.title(f"SHAP Feature Importance - {model_type.upper()}")
    plt.tight_layout()
    plt.savefig(output_path / f"shap_importance_bar_{model_type}.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 2. Summary plot (beeswarm)
    logger.info("Generating SHAP beeswarm plot...")
    plt.figure(figsize=(12, 10))
    shap.summary_plot(
        shap_values,
        X_test,
        feature_names=feature_cols,
        max_display=top_n,
        show=False,
    )
    plt.title(f"SHAP Feature Impact - {model_type.upper()}")
    plt.tight_layout()
    plt.savefig(output_path / f"shap_beeswarm_{model_type}.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 3. Calculate mean absolute SHAP values
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_importance = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False)

    # Save feature importance to CSV
    feature_importance.to_csv(output_path / f"shap_importance_{model_type}.csv", index=False)

    # 4. Print top features
    logger.info("\n" + "=" * 70)
    logger.info(f"TOP {top_n} FEATURES BY SHAP IMPORTANCE")
    logger.info("=" * 70)
    logger.info(f"{'Rank':<6} {'Feature':<40} {'Mean |SHAP|':<15}")
    logger.info("-" * 65)

    for i, row in feature_importance.head(top_n).iterrows():
        rank = feature_importance.index.get_loc(i) + 1
        logger.info(f"{rank:<6} {row['feature']:<40} {row['mean_abs_shap']:.6f}")

    # 5. Group features by category and analyze
    logger.info("\n" + "=" * 70)
    logger.info("FEATURE GROUP IMPORTANCE")
    logger.info("=" * 70)

    feature_groups = {
        "ELO": ["home_elo", "away_elo", "elo_diff", "home_win_prob_elo", "away_win_prob_elo"],
        "Poisson": ["home_xg_poisson", "away_xg_poisson", "xg_diff", "home_attack_strength",
                    "home_defense_strength", "away_attack_strength", "away_defense_strength",
                    "poisson_home_win_prob", "poisson_draw_prob", "poisson_away_win_prob"],
        "Form": [c for c in feature_cols if "last_n" in c],
        "H2H": [c for c in feature_cols if "h2h" in c],
        "EMA": [c for c in feature_cols if "_ema" in c and "poisson" not in c.lower()],
        "Goal Diff": [c for c in feature_cols if "goal_diff" in c],
    }

    group_importance = {}
    for group_name, group_features in feature_groups.items():
        existing = [f for f in group_features if f in feature_cols]
        if existing:
            group_shap = feature_importance[feature_importance["feature"].isin(existing)]["mean_abs_shap"].sum()
            group_importance[group_name] = {
                "total_shap": group_shap,
                "n_features": len(existing),
                "avg_shap": group_shap / len(existing),
            }

    # Sort by total importance
    sorted_groups = sorted(group_importance.items(), key=lambda x: x[1]["total_shap"], reverse=True)

    logger.info(f"{'Group':<15} {'Total SHAP':<15} {'Avg SHAP':<15} {'#Features':<10}")
    logger.info("-" * 55)
    for group_name, stats in sorted_groups:
        logger.info(
            f"{group_name:<15} {stats['total_shap']:.6f}       "
            f"{stats['avg_shap']:.6f}       {stats['n_features']:<10}"
        )

    # 6. Dependence plots for top features
    logger.info("\nGenerating dependence plots for top 3 features...")
    top_features = feature_importance.head(3)["feature"].tolist()

    for feat in top_features:
        feat_idx = feature_cols.index(feat)
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            feat_idx,
            shap_values,
            X_test,
            feature_names=feature_cols,
            show=False,
        )
        plt.title(f"SHAP Dependence: {feat}")
        plt.tight_layout()
        safe_name = feat.replace("/", "_")
        plt.savefig(output_path / f"shap_dependence_{safe_name}_{model_type}.png", dpi=150, bbox_inches="tight")
        plt.close()

    logger.info("\n" + "=" * 70)
    logger.info("SHAP ANALYSIS COMPLETED")
    logger.info("=" * 70)
    logger.info(f"Output saved to: {output_path}")
    logger.info("Generated files:")
    logger.info(f"  - shap_importance_bar_{model_type}.png")
    logger.info(f"  - shap_beeswarm_{model_type}.png")
    logger.info(f"  - shap_importance_{model_type}.csv")
    logger.info(f"  - shap_dependence_*_{model_type}.png")
    logger.info("=" * 70)

    return {
        "feature_importance": feature_importance,
        "group_importance": group_importance,
        "test_accuracy": test_acc,
    }


def main() -> int:
    """Main entry point."""
    setup_logging()
    args = parse_args()

    try:
        run_shap_analysis(
            config_path=args.config,
            model_type=args.model,
            target=args.target,
            features_file=args.features_file,
            output_dir=args.output_dir,
            top_n=args.top_n,
        )
        return 0
    except Exception as e:
        logging.exception(f"SHAP analysis failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
