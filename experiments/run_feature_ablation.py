#!/usr/bin/env python3
"""
Run feature ablation experiments.

This script tests different feature combinations to understand
which features contribute most to prediction accuracy.

Ablation study approach:
1. Run with ALL features (baseline)
2. Run WITHOUT each feature group (measure drop)
3. Run with ONLY each feature group (measure standalone value)

Usage:
    uv run python experiments/run_feature_ablation.py
    uv run python experiments/run_feature_ablation.py --model xgboost
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.config_loader import load_config
from src.ml.experiment import Experiment, ExperimentConfig
from sklearn.model_selection import train_test_split

FEATURE_GROUPS = {
    "form": [
        "home_wins_last_n", "home_draws_last_n", "home_losses_last_n",
        "home_goals_scored_last_n", "home_goals_conceded_last_n", "home_points_last_n",
        "away_wins_last_n", "away_draws_last_n", "away_losses_last_n",
        "away_goals_scored_last_n", "away_goals_conceded_last_n", "away_points_last_n",
    ],
    "h2h": [
        "h2h_home_wins", "h2h_draws", "h2h_away_wins", "h2h_avg_goals",
    ],
    "ema": [
        "home_goals_scored_ema", "home_goals_conceded_ema", "home_points_ema",
        "away_goals_scored_ema", "away_goals_conceded_ema", "away_points_ema",
    ],
    "team_stats": [
        "home_avg_rating", "home_team_shots", "home_team_shots_on",
        "home_team_passes", "home_team_key_passes", "home_avg_pass_accuracy",
        "home_team_tackles", "home_team_fouls", "home_team_yellows", "home_team_reds",
        "away_avg_rating", "away_team_shots", "away_team_shots_on",
        "away_team_passes", "away_team_key_passes", "away_avg_pass_accuracy",
        "away_team_tackles", "away_team_fouls", "away_team_yellows", "away_team_reds",
    ],
}

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
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run feature ablation experiments"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/local.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="xgboost",
        help="Model type to use (default: xgboost)"
    )
    parser.add_argument(
        "--target",
        type=str,
        default="home_win",
        help="Target variable"
    )
    parser.add_argument(
        "--features-file",
        type=str,
        default="features.csv",
        help="Features file name"
    )
    return parser.parse_args()


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Get all feature columns from dataframe."""
    return [col for col in df.columns if col not in EXCLUDE_COLUMNS]


def get_columns_for_group(all_columns: List[str], group_name: str) -> List[str]:
    """Get columns that belong to a feature group."""
    group_patterns = FEATURE_GROUPS.get(group_name, [])
    return [col for col in all_columns if col in group_patterns]


def run_single_experiment(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    model_type: str,
    experiment_name: str,
    run_name: str,
    tags: Dict[str, str],
) -> Dict:
    """Run a single experiment and return results."""
    config = ExperimentConfig(
        experiment_name=experiment_name,
        run_name=run_name,
        model_type=model_type,
        tags=tags,
    )

    experiment = Experiment(config)
    result = experiment.run(X_train, X_test, y_train, y_test)

    return {
        "run_name": run_name,
        "n_features": X_train.shape[1],
        "accuracy": result["metrics"].accuracy,
        "f1": result["metrics"].f1,
        "run_id": result["run_id"],
    }


def run_ablation_study(
    config_path: str,
    model_type: str,
    target: str,
    features_file: str,
) -> None:
    """
    Run feature ablation study.

    Tests:
    1. All features (baseline)
    2. Without each feature group
    3. Only each feature group
    """
    logger = logging.getLogger(__name__)

    logger.info("=" * 70)
    logger.info("FEATURE ABLATION STUDY")
    logger.info("=" * 70)
    logger.info(f"Model: {model_type}")
    logger.info(f"Target: {target}")
    logger.info("=" * 70)

    config = load_config(config_path)
    features_path = config.get_features_dir() / features_file
    df = pd.read_csv(features_path)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    all_features = get_feature_columns(df)
    logger.info(f"Total features: {len(all_features)}")

    present_groups = {}
    for group_name, group_cols in FEATURE_GROUPS.items():
        cols_present = [c for c in group_cols if c in all_features]
        if cols_present:
            present_groups[group_name] = cols_present
            logger.info(f"  {group_name}: {len(cols_present)} features")

    # Prepare base data
    X = df[all_features].fillna(0)
    y = df[target]

    # Time-based split
    if "date" in df.columns:
        df_sorted = df.sort_values("date")
        split_idx = int(len(df_sorted) * 0.8)
        train_idx = df_sorted.index[:split_idx]
        test_idx = df_sorted.index[split_idx:]
    else:
        train_idx, test_idx = train_test_split(
            df.index, test_size=0.2, random_state=42
        )

    y_train = y.loc[train_idx]
    y_test = y.loc[test_idx]

    experiment_name = f"feature-ablation-{target}"
    results = []

    # 1. Baseline: ALL features
    logger.info("\n[1] Running BASELINE (all features)...")
    X_train_all = X.loc[train_idx]
    X_test_all = X.loc[test_idx]

    baseline_result = run_single_experiment(
        X_train_all, X_test_all, y_train, y_test,
        model_type, experiment_name,
        run_name="baseline-all-features",
        tags={"ablation_type": "baseline", "features": "all"}
    )
    results.append({"experiment": "ALL FEATURES", **baseline_result})
    baseline_accuracy = baseline_result["accuracy"]

    # 2. Without each feature group
    logger.info("\n[2] Running WITHOUT each feature group...")
    for group_name, group_cols in present_groups.items():
        remaining_features = [f for f in all_features if f not in group_cols]

        if len(remaining_features) == 0:
            logger.warning(f"  Skipping 'without {group_name}' - no features left")
            continue

        logger.info(f"  Without {group_name} ({len(group_cols)} features removed)...")

        X_train_subset = X.loc[train_idx, remaining_features]
        X_test_subset = X.loc[test_idx, remaining_features]

        result = run_single_experiment(
            X_train_subset, X_test_subset, y_train, y_test,
            model_type, experiment_name,
            run_name=f"without-{group_name}",
            tags={"ablation_type": "without", "removed_group": group_name}
        )

        drop = baseline_accuracy - result["accuracy"]
        results.append({
            "experiment": f"WITHOUT {group_name.upper()}",
            "drop_from_baseline": drop,
            **result
        })

    # 3. Only each feature group
    logger.info("\n[3] Running with ONLY each feature group...")
    for group_name, group_cols in present_groups.items():
        logger.info(f"  Only {group_name} ({len(group_cols)} features)...")

        X_train_subset = X.loc[train_idx, group_cols]
        X_test_subset = X.loc[test_idx, group_cols]

        result = run_single_experiment(
            X_train_subset, X_test_subset, y_train, y_test,
            model_type, experiment_name,
            run_name=f"only-{group_name}",
            tags={"ablation_type": "only", "feature_group": group_name}
        )

        results.append({
            "experiment": f"ONLY {group_name.upper()}",
            **result
        })

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("ABLATION STUDY RESULTS")
    logger.info("=" * 70)
    logger.info(f"{'Experiment':<30} {'Features':<10} {'Accuracy':<12} {'Drop':<10}")
    logger.info("-" * 65)

    for r in results:
        exp = r["experiment"]
        n_feat = r["n_features"]
        acc = f"{r['accuracy']:.4f}"
        drop = r.get("drop_from_baseline")
        drop_str = f"{drop:+.4f}" if drop is not None else "-"
        logger.info(f"{exp:<30} {n_feat:<10} {acc:<12} {drop_str:<10}")

    logger.info("=" * 70)

    # Feature group importance ranking
    logger.info("\nFEATURE GROUP IMPORTANCE (by accuracy drop when removed):")
    logger.info("-" * 50)

    drops = [(r["experiment"].replace("WITHOUT ", ""), r.get("drop_from_baseline", 0))
             for r in results if "WITHOUT" in r["experiment"]]
    drops_sorted = sorted(drops, key=lambda x: x[1], reverse=True)

    for i, (group, drop) in enumerate(drops_sorted, 1):
        importance = "HIGH" if drop > 0.02 else "MEDIUM" if drop > 0.01 else "LOW"
        logger.info(f"  {i}. {group}: {drop:+.4f} ({importance})")

    logger.info("\n" + "=" * 70)
    logger.info("View detailed results: mlflow ui")


def main() -> int:
    """Main entry point."""
    setup_logging()
    args = parse_args()

    try:
        run_ablation_study(
            config_path=args.config,
            model_type=args.model,
            target=args.target,
            features_file=args.features_file,
        )
        return 0
    except Exception as e:
        logging.exception(f"Ablation study failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
