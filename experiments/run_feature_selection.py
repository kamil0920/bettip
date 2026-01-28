#!/usr/bin/env python3
"""
Experiment with feature selection - train on top N features.

Usage:
    # Train on top 25 features (by importance)
    uv run python experiments/run_feature_selection.py --top-n 25

    # Train on specific feature groups only
    uv run python experiments/run_feature_selection.py --groups elo form

    # Exclude certain feature groups
    uv run python experiments/run_feature_selection.py --exclude ema team_stats

    # Compare different numbers of features
    uv run python experiments/run_feature_selection.py --compare-top 10 15 20 25 30
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

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

FEATURE_GROUPS = {
    "elo": ["home_elo", "away_elo", "elo_diff", "home_win_prob_elo", "away_win_prob_elo"],
    "poisson": ["home_xg_poisson", "away_xg_poisson", "xg_diff", "home_attack_strength",
                "home_defense_strength", "away_attack_strength", "away_defense_strength",
                "poisson_home_win_prob", "poisson_draw_prob", "poisson_away_win_prob"],
    "form": ["home_wins_last_n", "home_draws_last_n", "home_losses_last_n",
             "home_goals_scored_last_n", "home_goals_conceded_last_n", "home_points_last_n",
             "away_wins_last_n", "away_draws_last_n", "away_losses_last_n",
             "away_goals_scored_last_n", "away_goals_conceded_last_n", "away_points_last_n"],
    "h2h": ["h2h_home_wins", "h2h_draws", "h2h_away_wins", "h2h_avg_goals"],
    "ema": ["home_goals_scored_ema", "home_goals_conceded_ema", "home_points_ema",
            "away_goals_scored_ema", "away_goals_conceded_ema", "away_points_ema"],
    "team_stats": ["home_rating_ema", "away_rating_ema", "home_shots_total_ema",
                   "away_shots_total_ema", "home_shots_on_ema", "away_shots_on_ema",
                   "home_passes_total_ema", "away_passes_total_ema", "home_passes_key_ema",
                   "away_passes_key_ema", "home_passes_accuracy_ema", "away_passes_accuracy_ema",
                   "home_tackles_total_ema", "away_tackles_total_ema", "home_fouls_committed_ema",
                   "away_fouls_committed_ema"],
    "goal_diff": ["home_avg_goal_diff", "away_avg_goal_diff", "home_total_goal_diff",
                  "away_total_goal_diff", "goal_diff_advantage"],
}


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Feature selection experiments")
    parser.add_argument("--config", default="config/local.yaml")
    parser.add_argument("--features-file", default="features_v2.csv")
    parser.add_argument("--model", default="lightgbm", choices=["lightgbm", "xgboost", "catboost", "random_forest", "logistic_regression"])
    parser.add_argument("--target", default="home_win")

    parser.add_argument("--top-n", type=int, help="Use top N features by importance")
    parser.add_argument("--groups", nargs="+", help="Use only these feature groups (elo, poisson, form, h2h, ema, team_stats, goal_diff)")
    parser.add_argument("--exclude", nargs="+", help="Exclude these feature groups")
    parser.add_argument("--compare-top", type=int, nargs="+", help="Compare different top-N values")
    parser.add_argument("--features", nargs="+", help="Use exactly these features")

    return parser.parse_args()


def load_data(config, features_file: str, target: str):
    """Load data and return X, y with time-based split."""
    from src.utils.data_io import load_features
    features_path = config.get_features_dir() / features_file
    df = load_features(features_path).sort_values("date")

    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLUMNS]
    X = df[feature_cols].fillna(0)
    y = df[target]

    split_idx = int(len(df) * 0.8)
    return X.iloc[:split_idx], X.iloc[split_idx:], y.iloc[:split_idx], y.iloc[split_idx:], feature_cols


def get_feature_importance(model, feature_cols: List[str]) -> pd.DataFrame:
    """Get feature importance from trained model."""
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance = abs(model.coef_).mean(axis=0) if len(model.coef_.shape) > 1 else abs(model.coef_)
    else:
        return pd.DataFrame({"feature": feature_cols, "importance": [1] * len(feature_cols)})

    return pd.DataFrame({
        "feature": feature_cols,
        "importance": importance
    }).sort_values("importance", ascending=False)


def train_and_evaluate(X_train, X_test, y_train, y_test, model_type: str):
    """Train model and return metrics."""
    model = ModelFactory.create(model_type)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    return model, acc, f1


def select_features_by_groups(all_features: List[str], groups: List[str]) -> List[str]:
    """Select features belonging to specified groups."""
    selected = []
    for group in groups:
        if group in FEATURE_GROUPS:
            selected.extend([f for f in FEATURE_GROUPS[group] if f in all_features])
    return selected


def exclude_features_by_groups(all_features: List[str], exclude: List[str]) -> List[str]:
    """Exclude features from specified groups."""
    excluded = []
    for group in exclude:
        if group in FEATURE_GROUPS:
            excluded.extend(FEATURE_GROUPS[group])
    return [f for f in all_features if f not in excluded]


def main():
    setup_logging()
    args = parse_args()
    logger = logging.getLogger(__name__)

    config = load_config(args.config)
    X_train, X_test, y_train, y_test, all_features = load_data(
        config, args.features_file, args.target
    )

    logger.info("=" * 70)
    logger.info("FEATURE SELECTION EXPERIMENTS")
    logger.info("=" * 70)
    logger.info(f"Model: {args.model}")
    logger.info(f"Total features available: {len(all_features)}")
    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")

    logger.info("\nTraining on ALL features to get importance ranking...")
    model_full, acc_full, f1_full = train_and_evaluate(
        X_train, X_test, y_train, y_test, args.model
    )
    importance_df = get_feature_importance(model_full, all_features)
    logger.info(f"All features - Accuracy: {acc_full:.4f}, F1: {f1_full:.4f}")

    if args.compare_top:
        logger.info("\n" + "=" * 70)
        logger.info("COMPARING TOP-N FEATURES")
        logger.info("=" * 70)

        results = [{"n_features": len(all_features), "accuracy": acc_full, "f1": f1_full, "label": "ALL"}]

        for n in sorted(args.compare_top):
            top_features = importance_df.head(n)["feature"].tolist()

            model, acc, f1 = train_and_evaluate(
                X_train[top_features], X_test[top_features], y_train, y_test, args.model
            )
            results.append({"n_features": n, "accuracy": acc, "f1": f1, "label": f"TOP-{n}"})
            logger.info(f"Top {n:3d} features - Accuracy: {acc:.4f}, F1: {f1:.4f}")

        logger.info("\n" + "-" * 50)
        logger.info("SUMMARY:")
        results_df = pd.DataFrame(results).sort_values("f1", ascending=False)
        for _, row in results_df.iterrows():
            logger.info(f"  {row['label']:<10} ({row['n_features']:2d} features): F1={row['f1']:.4f}, Acc={row['accuracy']:.4f}")

        return

    if args.top_n:
        logger.info(f"\n" + "=" * 70)
        logger.info(f"TRAINING ON TOP {args.top_n} FEATURES")
        logger.info("=" * 70)

        top_features = importance_df.head(args.top_n)["feature"].tolist()

        logger.info("Selected features:")
        for i, f in enumerate(top_features, 1):
            imp = importance_df[importance_df["feature"] == f]["importance"].values[0]
            logger.info(f"  {i:2d}. {f:<40} (importance: {imp:.4f})")

        model, acc, f1 = train_and_evaluate(
            X_train[top_features], X_test[top_features], y_train, y_test, args.model
        )

        logger.info(f"\nResults: Accuracy: {acc:.4f}, F1: {f1:.4f}")
        logger.info(f"vs ALL features: Accuracy: {acc_full:.4f}, F1: {f1_full:.4f}")
        logger.info(f"Difference: Accuracy: {acc - acc_full:+.4f}, F1: {f1 - f1_full:+.4f}")
        return

    if args.groups:
        logger.info(f"\n" + "=" * 70)
        logger.info(f"TRAINING ON GROUPS: {', '.join(args.groups)}")
        logger.info("=" * 70)

        selected = select_features_by_groups(all_features, args.groups)
        logger.info(f"Selected {len(selected)} features from groups: {args.groups}")

        model, acc, f1 = train_and_evaluate(
            X_train[selected], X_test[selected], y_train, y_test, args.model
        )

        logger.info(f"\nResults: Accuracy: {acc:.4f}, F1: {f1:.4f}")
        logger.info(f"vs ALL features: Accuracy: {acc_full:.4f}, F1: {f1_full:.4f}")
        return

    # Option 4: Exclude feature groups
    if args.exclude:
        logger.info(f"\n" + "=" * 70)
        logger.info(f"TRAINING WITHOUT GROUPS: {', '.join(args.exclude)}")
        logger.info("=" * 70)

        remaining = exclude_features_by_groups(all_features, args.exclude)
        logger.info(f"Using {len(remaining)} features (excluded {len(all_features) - len(remaining)})")

        model, acc, f1 = train_and_evaluate(
            X_train[remaining], X_test[remaining], y_train, y_test, args.model
        )

        logger.info(f"\nResults: Accuracy: {acc:.4f}, F1: {f1:.4f}")
        logger.info(f"vs ALL features: Accuracy: {acc_full:.4f}, F1: {f1_full:.4f}")
        return

    # Option 5: Use specific features
    if args.features:
        logger.info(f"\n" + "=" * 70)
        logger.info(f"TRAINING ON SPECIFIC FEATURES")
        logger.info("=" * 70)

        valid_features = [f for f in args.features if f in all_features]
        invalid = [f for f in args.features if f not in all_features]

        if invalid:
            logger.warning(f"Invalid features (not found): {invalid}")

        logger.info(f"Using {len(valid_features)} features: {valid_features}")

        model, acc, f1 = train_and_evaluate(
            X_train[valid_features], X_test[valid_features], y_train, y_test, args.model
        )

        logger.info(f"\nResults: Accuracy: {acc:.4f}, F1: {f1:.4f}")
        return

    # Default: show feature importance and available groups
    logger.info("\n" + "=" * 70)
    logger.info("FEATURE IMPORTANCE RANKING (Top 20)")
    logger.info("=" * 70)
    for i, row in importance_df.head(20).iterrows():
        rank = importance_df.index.get_loc(i) + 1
        logger.info(f"  {rank:2d}. {row['feature']:<40} {row['importance']:.4f}")

    logger.info("\n" + "=" * 70)
    logger.info("AVAILABLE FEATURE GROUPS")
    logger.info("=" * 70)
    for group, features in FEATURE_GROUPS.items():
        present = [f for f in features if f in all_features]
        logger.info(f"  {group:<12}: {len(present)} features")

    logger.info("\n" + "=" * 70)
    logger.info("USAGE EXAMPLES")
    logger.info("=" * 70)
    logger.info("  # Train on top 25 features:")
    logger.info("  uv run python experiments/run_feature_selection.py --top-n 25")
    logger.info("")
    logger.info("  # Compare top 10, 20, 30 features:")
    logger.info("  uv run python experiments/run_feature_selection.py --compare-top 10 20 30")
    logger.info("")
    logger.info("  # Use only ELO and form features:")
    logger.info("  uv run python experiments/run_feature_selection.py --groups elo form")
    logger.info("")
    logger.info("  # Exclude team_stats (might be overfitting):")
    logger.info("  uv run python experiments/run_feature_selection.py --exclude team_stats")


if __name__ == "__main__":
    main()
