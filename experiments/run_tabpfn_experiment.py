#!/usr/bin/env python3
"""
TabPFN Foundation Model Experiment

Quick benchmark of TabPFN (Nature 2025) against current GBDT models.
TabPFN requires zero hyperparameter tuning and excels on datasets ≤10K samples.

Usage:
    uv run python experiments/run_tabpfn_experiment.py
    uv run python experiments/run_tabpfn_experiment.py --bet-types away_win btts
    uv run python experiments/run_tabpfn_experiment.py --per-league
"""

import argparse
import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Reuse sniper paths and configs
_SPORTMONKS_BACKUP = Path("data/sportmonks_backup/features_with_sportmonks_odds_FULL.parquet")
_SPORTMONKS_STANDARD = Path("data/03-features/features_with_sportmonks_odds.parquet")
FEATURES_FILE = _SPORTMONKS_BACKUP if _SPORTMONKS_BACKUP.exists() else _SPORTMONKS_STANDARD
OUTPUT_DIR = Path("experiments/outputs/tabpfn_experiment")

# Import bet type configs from sniper
from experiments.run_sniper_optimization import BET_TYPES


def load_data(bet_type: str, league: Optional[str] = None) -> tuple:
    """Load features and prepare X, y, odds arrays for a bet type."""
    from src.utils.data_io import load_features

    df = load_features(FEATURES_FILE)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    config = BET_TYPES[bet_type]
    target = config["target"]

    # Derive target if needed
    if target not in df.columns:
        if target == "under25":
            df["under25"] = (df.get("total_goals", df.get("home_goals", 0).fillna(0) + df.get("away_goals", 0).fillna(0)) < 2.5).astype(int)
        elif target == "over25":
            df["over25"] = (df.get("total_goals", df.get("home_goals", 0).fillna(0) + df.get("away_goals", 0).fillna(0)) > 2.5).astype(int)

    if target not in df.columns:
        raise ValueError(f"Target {target} not found in data")

    # Filter by league if requested
    if league and "league" in df.columns:
        df = df[df["league"] == league].reset_index(drop=True)

    # Get odds column
    odds_col = config.get("odds_column", "best_odds")
    if odds_col not in df.columns:
        for fallback in ["best_odds", "avg_odds", "odds"]:
            if fallback in df.columns:
                odds_col = fallback
                break

    # Select numeric features (exclude target, odds, metadata)
    exclude_patterns = config.get("exclude_patterns", [])
    exclude_cols = {"date", "league", "season", "home_team", "away_team", "fixture_id",
                    target, odds_col, "result", "home_goals", "away_goals", "total_goals"}

    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c not in exclude_cols
                    and not any(p in c for p in exclude_patterns)]

    df_clean = df.dropna(subset=[target]).reset_index(drop=True)

    X = df_clean[feature_cols].fillna(0).values
    y = df_clean[target].values.astype(int)
    odds = df_clean[odds_col].fillna(2.0).values if odds_col in df_clean.columns else np.full(len(y), 2.0)

    return X, y, odds, feature_cols, df_clean


def run_walkforward_comparison(
    X: np.ndarray,
    y: np.ndarray,
    odds: np.ndarray,
    n_folds: int = 5,
    feature_cols: Optional[List[str]] = None,
) -> Dict:
    """Run walk-forward validation comparing TabPFN vs XGBoost baseline."""
    try:
        from tabpfn import TabPFNClassifier
    except ImportError:
        raise ImportError("tabpfn is required. Install with: uv pip install 'bettip[dl]'")

    from xgboost import XGBClassifier

    n_samples = len(y)
    fold_size = n_samples // (n_folds + 1)

    results = {"tabpfn": [], "xgboost_default": []}

    for fold in range(n_folds):
        train_end = (fold + 1) * fold_size
        test_start = train_end
        test_end = min(test_start + fold_size, n_samples)

        X_train, y_train = X[:train_end], y[:train_end]
        X_test, y_test = X[test_start:test_end], y[test_start:test_end]
        odds_test = odds[test_start:test_end]

        if len(X_train) < 50 or len(X_test) < 20:
            continue

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # TabPFN has a 10K sample limit and 500 feature limit
        max_train = min(len(X_train_s), 10000)
        max_features = min(X_train_s.shape[1], 500)

        X_train_tabpfn = X_train_s[-max_train:, :max_features]
        y_train_tabpfn = y_train[-max_train:]
        X_test_tabpfn = X_test_s[:, :max_features]

        fold_result = {"fold": fold, "n_train": len(X_train), "n_test": len(X_test)}

        # --- TabPFN ---
        try:
            tabpfn = TabPFNClassifier(device="cpu", N_ensemble_configurations=32)
            tabpfn.fit(X_train_tabpfn, y_train_tabpfn)
            tabpfn_probs = tabpfn.predict_proba(X_test_tabpfn)[:, 1]

            results["tabpfn"].append({
                **fold_result,
                "log_loss": float(log_loss(y_test, tabpfn_probs)),
                "brier": float(brier_score_loss(y_test, tabpfn_probs)),
                "auc": float(roc_auc_score(y_test, tabpfn_probs)),
                **_betting_metrics(y_test, tabpfn_probs, odds_test),
            })
        except Exception as e:
            logger.warning(f"  TabPFN fold {fold} failed: {e}")

        # --- XGBoost default (no tuning, for fair comparison) ---
        try:
            xgb_model = XGBClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                random_state=42, verbosity=0
            )
            calibrated = CalibratedClassifierCV(xgb_model, method="sigmoid", cv=3)
            calibrated.fit(X_train_s, y_train)
            xgb_probs = calibrated.predict_proba(X_test_s)[:, 1]

            results["xgboost_default"].append({
                **fold_result,
                "log_loss": float(log_loss(y_test, xgb_probs)),
                "brier": float(brier_score_loss(y_test, xgb_probs)),
                "auc": float(roc_auc_score(y_test, xgb_probs)),
                **_betting_metrics(y_test, xgb_probs, odds_test),
            })
        except Exception as e:
            logger.warning(f"  XGBoost fold {fold} failed: {e}")

    return results


def _betting_metrics(
    y_true: np.ndarray, probs: np.ndarray, odds: np.ndarray, threshold: float = 0.55
) -> Dict:
    """Calculate betting ROI metrics at a given probability threshold."""
    mask = (probs >= threshold) & (odds >= 1.5) & (odds <= 6.0)
    n_bets = mask.sum()
    if n_bets < 3:
        return {"n_bets": int(n_bets), "precision": 0.0, "roi": 0.0}

    wins = y_true[mask] == 1
    profit = (wins * (odds[mask] - 1) - (~wins) * 1).sum()
    return {
        "n_bets": int(n_bets),
        "precision": float(wins.mean()),
        "roi": float(profit / n_bets * 100),
    }


def summarize_results(results: Dict, bet_type: str) -> Dict:
    """Summarize walk-forward results per model."""
    summary = {}
    for model_name, folds in results.items():
        if not folds:
            continue
        df = pd.DataFrame(folds)
        summary[model_name] = {
            "avg_log_loss": float(df["log_loss"].mean()),
            "avg_brier": float(df["brier"].mean()),
            "avg_auc": float(df["auc"].mean()),
            "avg_roi": float(df["roi"].mean()),
            "total_bets": int(df["n_bets"].sum()),
            "avg_precision": float(df["precision"].mean()),
            "n_folds": len(df),
        }

    logger.info(f"\n{'='*60}")
    logger.info(f"TabPFN Experiment Results: {bet_type}")
    logger.info(f"{'='*60}")
    for model_name, stats in summary.items():
        logger.info(f"\n{model_name}:")
        logger.info(f"  Log Loss:  {stats['avg_log_loss']:.4f}")
        logger.info(f"  Brier:     {stats['avg_brier']:.4f}")
        logger.info(f"  AUC:       {stats['avg_auc']:.4f}")
        logger.info(f"  ROI:       {stats['avg_roi']:.1f}%")
        logger.info(f"  Precision: {stats['avg_precision']:.3f}")
        logger.info(f"  Bets:      {stats['total_bets']}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="TabPFN Experiment")
    parser.add_argument("--bet-types", nargs="+",
                        default=["away_win", "btts", "over25"],
                        help="Bet types to evaluate")
    parser.add_argument("--per-league", action="store_true",
                        help="Also run per-league subsets")
    parser.add_argument("--n-folds", type=int, default=5)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for bet_type in args.bet_types:
        if bet_type not in BET_TYPES:
            logger.warning(f"Unknown bet type: {bet_type}, skipping")
            continue

        logger.info(f"\n{'#'*60}")
        logger.info(f"Running TabPFN experiment for: {bet_type}")
        logger.info(f"{'#'*60}")

        try:
            X, y, odds, feature_cols, df = load_data(bet_type)
            logger.info(f"  Data: {X.shape[0]} samples, {X.shape[1]} features")
            logger.info(f"  Class balance: {y.mean():.3f}")

            results = run_walkforward_comparison(X, y, odds, n_folds=args.n_folds, feature_cols=feature_cols)
            summary = summarize_results(results, bet_type)
            all_results[bet_type] = {"summary": summary, "folds": results}

            # Per-league analysis
            if args.per_league and "league" in df.columns:
                for league in df["league"].unique():
                    try:
                        X_l, y_l, odds_l, _, _ = load_data(bet_type, league=league)
                        if len(y_l) < 200:
                            continue
                        logger.info(f"\n  League: {league} ({len(y_l)} samples)")
                        r = run_walkforward_comparison(X_l, y_l, odds_l, n_folds=min(3, args.n_folds))
                        s = summarize_results(r, f"{bet_type}/{league}")
                        all_results[f"{bet_type}/{league}"] = {"summary": s, "folds": r}
                    except Exception as e:
                        logger.warning(f"  League {league} failed: {e}")

        except Exception as e:
            logger.error(f"Failed for {bet_type}: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"tabpfn_results_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults saved to: {output_path}")

    # Print winner summary
    logger.info(f"\n{'='*60}")
    logger.info("WINNER SUMMARY")
    logger.info(f"{'='*60}")
    for bet_type, data in all_results.items():
        if "/" in bet_type:
            continue  # Skip per-league in summary
        summary = data.get("summary", {})
        if "tabpfn" in summary and "xgboost_default" in summary:
            tabpfn_ll = summary["tabpfn"]["avg_log_loss"]
            xgb_ll = summary["xgboost_default"]["avg_log_loss"]
            winner = "TabPFN" if tabpfn_ll < xgb_ll else "XGBoost"
            delta = abs(tabpfn_ll - xgb_ll)
            logger.info(f"  {bet_type}: {winner} wins (log_loss delta: {delta:.4f})")


if __name__ == "__main__":
    main()
