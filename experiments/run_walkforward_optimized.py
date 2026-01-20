#!/usr/bin/env python3
"""
Walk-Forward Validation using optimized model configs.

Loads optimal features and parameters from iterative optimization
and runs walk-forward validation for all callibration.

Usage:
    uv run python experiments/run_walkforward_optimized.py
    uv run python experiments/run_walkforward_optimized.py --config-dir experiments/outputs/iterative_optimization_100trials
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.config_loader import load_config
from src.ml.models import ModelFactory


EXCLUDE_COLUMNS = [
    "fixture_id", "date", "home_team_id", "home_team_name",
    "away_team_id", "away_team_name", "round", "round_num", "season",
    "home_win", "draw", "away_win", "match_result",
    "total_goals", "goal_difference", "league"
]

MODELS = ["lightgbm", "xgboost", "catboost", "random_forest", "logistic_regression"]


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Walk-forward with optimized configs")
    parser.add_argument("--config", default="config/local.yaml")
    parser.add_argument("--features-file", default="features_v2.csv")
    parser.add_argument("--target", default="home_win")
    parser.add_argument("--config-dir", default="experiments/outputs/iterative_optimization_100trials",
                       help="Directory with optimized model configs")
    parser.add_argument("--start-season", type=int, default=2023)
    parser.add_argument("--start-round", type=int, default=19)
    parser.add_argument("--output-dir", default="experiments/outputs/walkforward_optimized")
    parser.add_argument("--retrain-every", type=int, default=1)
    parser.add_argument("--callibration", nargs="+", default=MODELS,
                       help="Models to evaluate")
    return parser.parse_args()


def load_optimized_config(config_dir: Path, model_type: str) -> Optional[Dict]:
    """Load optimized config for a model."""
    config_file = config_dir / f"{model_type}_config.json"
    if not config_file.exists():
        return None
    with open(config_file) as f:
        return json.load(f)


def load_and_prepare_data(config, features_file: str) -> pd.DataFrame:
    """Load data and add season/round columns."""
    features_path = config.get_features_dir() / features_file
    df = pd.read_csv(features_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df['round_num'] = df['round'].str.extract(r'(\d+)').astype(int)
    df['season'] = df['date'].apply(lambda x: x.year if x.month >= 8 else x.year - 1)
    return df


def get_matchdays(df: pd.DataFrame, start_season: int, start_round: int) -> List[Tuple[int, int]]:
    """Get list of (season, round) tuples for walk-forward validation."""
    matchdays = []
    for season in sorted(df['season'].unique()):
        if season < start_season:
            continue
        season_df = df[df['season'] == season]
        rounds = sorted(season_df['round_num'].unique())
        for round_num in rounds:
            if season == start_season and round_num < start_round:
                continue
            matchdays.append((season, round_num))
    return matchdays


def run_walkforward_for_model(
    df: pd.DataFrame,
    model_type: str,
    features: List[str],
    params: Dict[str, Any],
    target: str,
    matchdays: List[Tuple[int, int]],
    retrain_every: int,
    logger: logging.Logger
) -> Dict:
    """Run walk-forward validation for a single model."""

    results = []
    all_y_true = []
    all_y_pred = []
    all_y_proba = []

    model = None

    for i, (season, round_num) in enumerate(matchdays):
        # Split data
        train_mask = (
            (df['season'] < season) |
            ((df['season'] == season) & (df['round_num'] < round_num))
        )
        test_mask = (df['season'] == season) & (df['round_num'] == round_num)

        train_df = df[train_mask]
        test_df = df[test_mask]

        if len(test_df) == 0:
            continue

        # Filter to available features
        available_features = [f for f in features if f in df.columns]

        X_train = train_df[available_features].fillna(0)
        y_train = train_df[target]
        X_test = test_df[available_features].fillna(0)
        y_test = test_df[target]

        # Train or reuse model
        should_retrain = (model is None or i % retrain_every == 0)

        if should_retrain:
            model = ModelFactory.create(model_type, params=params)
            model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)
        y_proba = None
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)

        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)

        results.append({
            'season': season,
            'round': round_num,
            'n_matches': len(test_df),
            'accuracy': acc,
            'correct': int(acc * len(test_df)),
        })

        all_y_true.extend(y_test.values)
        all_y_pred.extend(y_pred)
        if y_proba is not None:
            all_y_proba.extend(y_proba)

    # Calculate overall metrics
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_y_proba = np.array(all_y_proba) if all_y_proba else None

    overall_acc = accuracy_score(all_y_true, all_y_pred)
    overall_f1 = f1_score(all_y_true, all_y_pred, average='weighted')

    overall_logloss = None
    if all_y_proba is not None and len(all_y_proba) > 0:
        try:
            overall_logloss = log_loss(all_y_true, all_y_proba)
        except:
            pass

    return {
        'model': model_type,
        'n_features': len(available_features),
        'n_matchdays': len(results),
        'n_matches': len(all_y_true),
        'accuracy': overall_acc,
        'f1': overall_f1,
        'log_loss': overall_logloss,
        'results_per_matchday': results,
    }


def main():
    args = parse_args()
    setup_logging()
    logger = logging.getLogger(__name__)

    config_dir = Path(args.config_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("WALK-FORWARD VALIDATION (OPTIMIZED MODELS)")
    logger.info("=" * 70)
    logger.info(f"Config dir: {config_dir}")
    logger.info(f"Start: Season {args.start_season}, Round {args.start_round}")

    # Load data
    config = load_config(args.config)
    df = load_and_prepare_data(config, args.features_file)

    # Get matchdays
    matchdays = get_matchdays(df, args.start_season, args.start_round)
    logger.info(f"Matchdays to evaluate: {len(matchdays)}")

    # Run walk-forward for each model
    all_results = []

    for model_type in args.models:
        logger.info(f"\n{'='*50}")
        logger.info(f"MODEL: {model_type.upper()}")
        logger.info(f"{'='*50}")

        # Load optimized config
        opt_config = load_optimized_config(config_dir, model_type)

        if opt_config is None:
            logger.warning(f"No optimized config found for {model_type}, using defaults")
            feature_cols = [c for c in df.columns if c not in EXCLUDE_COLUMNS]
            params = None
        else:
            feature_cols = opt_config['features']
            params = opt_config.get('params', {})
            logger.info(f"  Features: {len(feature_cols)}")
            logger.info(f"  Params: {params}")

        # Run walk-forward
        result = run_walkforward_for_model(
            df=df,
            model_type=model_type,
            features=feature_cols,
            params=params,
            target=args.target,
            matchdays=matchdays,
            retrain_every=args.retrain_every,
            logger=logger
        )

        all_results.append(result)

        logger.info(f"\n  Results:")
        logger.info(f"    Accuracy: {result['accuracy']:.4f} ({result['accuracy']:.2%})")
        logger.info(f"    F1: {result['f1']:.4f}")
        if result['log_loss']:
            logger.info(f"    Log Loss: {result['log_loss']:.4f}")
        logger.info(f"    Matches: {result['n_matches']}")

        # Save per-model results
        results_df = pd.DataFrame(result['results_per_matchday'])
        results_df.to_csv(output_dir / f"walkforward_{model_type}.csv", index=False)

    # Create summary
    summary_data = []
    for r in all_results:
        summary_data.append({
            'model': r['model'],
            'n_features': r['n_features'],
            'n_matches': r['n_matches'],
            'accuracy': r['accuracy'],
            'f1': r['f1'],
            'log_loss': r['log_loss'],
        })

    summary_df = pd.DataFrame(summary_data).sort_values('accuracy', ascending=False)
    summary_df.to_csv(output_dir / "walkforward_summary.csv", index=False)

    # Print final summary
    logger.info("\n" + "=" * 70)
    logger.info("WALK-FORWARD VALIDATION SUMMARY")
    logger.info("=" * 70)

    for _, row in summary_df.iterrows():
        logger.info(f"\n{row['model'].upper()}")
        logger.info(f"  Accuracy: {row['accuracy']:.4f} ({row['accuracy']:.2%})")
        logger.info(f"  F1: {row['f1']:.4f}")
        logger.info(f"  Features: {row['n_features']}")

    best = summary_df.iloc[0]
    logger.info("\n" + "-" * 50)
    logger.info(f"BEST MODEL: {best['model'].upper()}")
    logger.info(f"  Walk-Forward Accuracy: {best['accuracy']:.2%}")
    logger.info(f"  Walk-Forward F1: {best['f1']:.4f}")

    logger.info(f"\nResults saved to: {output_dir}")

    return summary_df


if __name__ == "__main__":
    main()
