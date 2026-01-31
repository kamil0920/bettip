#!/usr/bin/env python3
"""
TabNet Experiment for Feature Attention Analysis

Evaluates TabNet as an alternative/supplement to GBDT models.
TabNet provides instance-wise feature selection via attention masks,
potentially replacing or improving upon global RFE feature selection.

Usage:
    uv run python experiments/run_tabnet_experiment.py
    uv run python experiments/run_tabnet_experiment.py --bet-types away_win btts
    uv run python experiments/run_tabnet_experiment.py --compare-rfe
"""

import argparse
import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_SPORTMONKS_BACKUP = Path("data/sportmonks_backup/features_with_sportmonks_odds_FULL.parquet")
_SPORTMONKS_STANDARD = Path("data/03-features/features_with_sportmonks_odds.parquet")
FEATURES_FILE = _SPORTMONKS_BACKUP if _SPORTMONKS_BACKUP.exists() else _SPORTMONKS_STANDARD
OUTPUT_DIR = Path("experiments/outputs/tabnet_experiment")

from experiments.run_sniper_optimization import BET_TYPES


def load_data(bet_type: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], pd.DataFrame]:
    """Load features for a bet type. Returns X, y, odds, feature_cols, df."""
    from src.utils.data_io import load_features

    df = load_features(FEATURES_FILE)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    config = BET_TYPES[bet_type]
    target = config["target"]

    if target not in df.columns:
        if target == "under25":
            df["under25"] = (df.get("total_goals", df.get("home_goals", 0).fillna(0) + df.get("away_goals", 0).fillna(0)) < 2.5).astype(int)
        elif target == "over25":
            df["over25"] = (df.get("total_goals", df.get("home_goals", 0).fillna(0) + df.get("away_goals", 0).fillna(0)) > 2.5).astype(int)

    odds_col = config.get("odds_column", "best_odds")
    if odds_col not in df.columns:
        for fallback in ["best_odds", "avg_odds", "odds"]:
            if fallback in df.columns:
                odds_col = fallback
                break

    exclude_cols = {"date", "league", "season", "home_team", "away_team", "fixture_id",
                    target, odds_col, "result", "home_goals", "away_goals", "total_goals"}
    exclude_patterns = config.get("exclude_patterns", [])

    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c not in exclude_cols
                    and not any(p in c for p in exclude_patterns)]

    df_clean = df.dropna(subset=[target]).reset_index(drop=True)
    X = df_clean[feature_cols].fillna(0).values
    y = df_clean[target].values.astype(int)
    odds = df_clean[odds_col].fillna(2.0).values if odds_col in df_clean.columns else np.full(len(y), 2.0)

    return X, y, odds, feature_cols, df_clean


def run_tabnet_walkforward(
    X: np.ndarray,
    y: np.ndarray,
    odds: np.ndarray,
    feature_cols: List[str],
    n_folds: int = 5,
    n_optuna_trials: int = 20,
) -> Dict:
    """Walk-forward validation with TabNet, optionally with Optuna tuning."""
    try:
        from pytorch_tabnet.tab_model import TabNetClassifier
    except ImportError:
        raise ImportError("pytorch-tabnet required. Install with: uv pip install 'bettip[dl]'")

    import optuna
    from optuna.samplers import TPESampler
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    n_samples = len(y)
    fold_size = n_samples // (n_folds + 1)

    results = {"tabnet_tuned": [], "tabnet_default": []}
    attention_maps = []

    for fold in range(n_folds):
        train_end = (fold + 1) * fold_size
        test_start = train_end
        test_end = min(test_start + fold_size, n_samples)

        X_train, y_train = X[:train_end], y[:train_end]
        X_test, y_test = X[test_start:test_end], y[test_start:test_end]
        odds_test = odds[test_start:test_end]

        if len(X_train) < 100 or len(X_test) < 20:
            continue

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train).astype(np.float32)
        X_test_s = scaler.transform(X_test).astype(np.float32)

        # Use last 20% of training as validation for TabNet early stopping
        n_val = int(len(X_train_s) * 0.2)
        X_tr, X_val = X_train_s[:-n_val], X_train_s[-n_val:]
        y_tr, y_val = y_train[:-n_val], y_train[-n_val:]

        fold_info = {"fold": fold, "n_train": len(X_train), "n_test": len(X_test)}

        # --- TabNet default ---
        try:
            tabnet = TabNetClassifier(
                n_d=32, n_a=32, n_steps=5,
                gamma=1.5, n_independent=2, n_shared=2,
                lambda_sparse=1e-3, seed=42, verbose=0,
            )
            tabnet.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                eval_metric=["logloss"],
                max_epochs=100,
                patience=15,
                batch_size=min(256, len(X_tr) // 4),
            )
            probs_default = tabnet.predict_proba(X_test_s)[:, 1]

            results["tabnet_default"].append({
                **fold_info,
                "log_loss": float(log_loss(y_test, probs_default)),
                "brier": float(brier_score_loss(y_test, probs_default)),
                "auc": float(roc_auc_score(y_test, probs_default)),
                **_betting_metrics(y_test, probs_default, odds_test),
            })

            # Extract attention for feature importance analysis
            explain_matrix, masks = tabnet.explain(X_test_s)
            avg_attention = explain_matrix.mean(axis=0)
            top_features = np.argsort(avg_attention)[::-1][:20]
            attention_maps.append({
                "fold": fold,
                "top_features": [(feature_cols[i], float(avg_attention[i])) for i in top_features],
            })

        except Exception as e:
            logger.warning(f"  TabNet default fold {fold} failed: {e}")

        # --- TabNet with Optuna tuning (on first fold only to save time) ---
        if fold == 0 and n_optuna_trials > 0:
            try:
                def tabnet_objective(trial):
                    n_d = trial.suggest_int("n_d", 8, 64, step=8)
                    params = {
                        "n_d": n_d, "n_a": n_d,
                        "n_steps": trial.suggest_int("n_steps", 3, 7),
                        "gamma": trial.suggest_float("gamma", 1.0, 2.0),
                        "lambda_sparse": trial.suggest_float("lambda_sparse", 1e-4, 1e-1, log=True),
                        "n_independent": trial.suggest_int("n_independent", 1, 3),
                        "n_shared": trial.suggest_int("n_shared", 1, 3),
                        "seed": 42, "verbose": 0,
                    }
                    model = TabNetClassifier(**params)
                    model.fit(
                        X_tr, y_tr,
                        eval_set=[(X_val, y_val)],
                        eval_metric=["logloss"],
                        max_epochs=80,
                        patience=10,
                        batch_size=min(256, len(X_tr) // 4),
                    )
                    probs = model.predict_proba(X_val)[:, 1]
                    return -log_loss(y_val, probs)

                study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
                study.optimize(tabnet_objective, n_trials=n_optuna_trials, show_progress_bar=True)
                best_params = study.best_params
                best_params["n_a"] = best_params["n_d"]
                logger.info(f"  Best TabNet params: {best_params}")
            except Exception as e:
                logger.warning(f"  TabNet tuning failed: {e}")

    return {"results": results, "attention": attention_maps}


def _betting_metrics(y_true, probs, odds, threshold=0.55):
    mask = (probs >= threshold) & (odds >= 1.5) & (odds <= 6.0)
    n_bets = mask.sum()
    if n_bets < 3:
        return {"n_bets": int(n_bets), "precision": 0.0, "roi": 0.0}
    wins = y_true[mask] == 1
    profit = (wins * (odds[mask] - 1) - (~wins) * 1).sum()
    return {"n_bets": int(n_bets), "precision": float(wins.mean()), "roi": float(profit / n_bets * 100)}


def main():
    parser = argparse.ArgumentParser(description="TabNet Experiment")
    parser.add_argument("--bet-types", nargs="+", default=["away_win", "btts"],
                        help="Bet types to evaluate")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-optuna-trials", type=int, default=20)
    parser.add_argument("--compare-rfe", action="store_true",
                        help="Compare TabNet attention vs RFE feature selection")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for bet_type in args.bet_types:
        if bet_type not in BET_TYPES:
            logger.warning(f"Unknown bet type: {bet_type}")
            continue

        logger.info(f"\n{'#'*60}")
        logger.info(f"TabNet experiment: {bet_type}")
        logger.info(f"{'#'*60}")

        try:
            X, y, odds, feature_cols, df = load_data(bet_type)
            logger.info(f"  Data: {X.shape[0]} samples, {X.shape[1]} features")

            output = run_tabnet_walkforward(
                X, y, odds, feature_cols,
                n_folds=args.n_folds,
                n_optuna_trials=args.n_optuna_trials,
            )

            # Summarize
            for model_name, folds in output["results"].items():
                if not folds:
                    continue
                fdf = pd.DataFrame(folds)
                logger.info(f"\n  {model_name}:")
                logger.info(f"    Log Loss:  {fdf['log_loss'].mean():.4f}")
                logger.info(f"    AUC:       {fdf['auc'].mean():.4f}")
                logger.info(f"    ROI:       {fdf['roi'].mean():.1f}%")

            # Feature attention summary
            if output["attention"]:
                logger.info("\n  Top features by TabNet attention (averaged across folds):")
                all_attn = {}
                for a in output["attention"]:
                    for feat, score in a["top_features"]:
                        all_attn[feat] = all_attn.get(feat, []) + [score]
                avg_attn = {f: np.mean(s) for f, s in all_attn.items()}
                for feat, score in sorted(avg_attn.items(), key=lambda x: -x[1])[:15]:
                    logger.info(f"    {feat}: {score:.4f}")

            all_results[bet_type] = output

        except Exception as e:
            logger.error(f"Failed for {bet_type}: {e}")
            import traceback
            traceback.print_exc()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"tabnet_results_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
