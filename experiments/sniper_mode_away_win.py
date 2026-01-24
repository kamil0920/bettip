#!/usr/bin/env python3
"""
Sniper Mode: Away Win Precision Optimizer

Goal: Achieve 90% precision on away_win predictions with minimal bet volume.
Strategy: Quality over quantity - find the conditions where we almost never lose.

Usage:
    python experiments/sniper_mode_away_win.py
    python experiments/sniper_mode_away_win.py --target-precision 0.85
    python experiments/sniper_mode_away_win.py --min-bets 10
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from itertools import product

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import precision_score, accuracy_score
from sklearn.linear_model import RidgeClassifierCV

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Paths
MODELS_DIR = Path("models")
FEATURES_FILE = Path("data/03-features/features_all_5leagues_with_odds.csv")
OUTPUT_DIR = Path("experiments/outputs/sniper_mode")


@dataclass
class SniperConfig:
    """Configuration for a sniper mode strategy."""
    name: str
    primary_model: str
    primary_threshold: float
    require_consensus: bool = False
    consensus_models: List[str] = None
    consensus_threshold: float = 0.5
    min_odds: float = 1.5
    max_odds: float = 10.0
    leagues: List[str] = None  # None = all leagues
    min_position_diff: int = None  # Away team must be X positions higher

    def __post_init__(self):
        if self.consensus_models is None:
            self.consensus_models = []
        if self.leagues is None:
            self.leagues = []


@dataclass
class SniperResult:
    """Result of a sniper mode backtest."""
    config: Dict
    total_bets: int
    wins: int
    losses: int
    precision: float
    roi: float
    avg_odds: float
    min_odds_seen: float
    max_odds_seen: float
    by_league: Dict[str, Dict]
    sample_bets: List[Dict]
    meets_target: bool
    # Per-bet details for bootstrap calculation
    bet_outcomes: List[int] = None  # 1=win, 0=loss for each bet
    bet_odds: List[float] = None    # Odds for each bet


class SniperModeOptimizer:
    """
    Optimizer for finding high-precision away_win betting configurations.

    Iterates through combinations of:
    - Probability thresholds
    - Multi-model consensus requirements
    - Odds filters
    - League filters
    - Position difference filters
    """

    def __init__(
        self,
        target_precision: float = 0.90,
        min_bets: int = 10,
    ):
        self.target_precision = target_precision
        self.min_bets = min_bets
        self.models = {}
        self.features_df = None
        self.results = []
        # Ensemble components
        self.stacking_meta = None
        self.ensemble_predictions = {}  # Cache for stacking/average predictions

    def load_models(self) -> Dict[str, Any]:
        """Load all available away_win models."""
        model_files = {
            "catboost": MODELS_DIR / "away_win_catboost.joblib",
            "lightgbm": MODELS_DIR / "away_win_lightgbm.joblib",
            "xgboost": MODELS_DIR / "away_win_xgboost.joblib",
            "logreg": MODELS_DIR / "away_win_logisticreg.joblib",
        }

        for name, path in model_files.items():
            if path.exists():
                try:
                    model_data = joblib.load(path)
                    self.models[name] = model_data
                    logger.info(f"Loaded {name} model from {path}")
                except Exception as e:
                    logger.warning(f"Failed to load {name}: {e}")

        if not self.models:
            raise RuntimeError("No models found! Run optimization pipeline first.")

        return self.models

    def load_features(self) -> pd.DataFrame:
        """Load feature data for backtesting."""
        if not FEATURES_FILE.exists():
            raise FileNotFoundError(f"Features file not found: {FEATURES_FILE}")

        df = pd.read_csv(FEATURES_FILE)
        logger.info(f"Loaded {len(df)} matches from {FEATURES_FILE}")

        # Ensure we have required columns
        required = ["away_win", "avg_away_open", "league"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Create target
        if "away_win" not in df.columns and "result" in df.columns:
            df["away_win"] = (df["result"] == "A").astype(int)

        self.features_df = df
        return df

    def train_stacking_ensemble(self, df: pd.DataFrame = None) -> None:
        """
        Train stacking meta-learner and compute ensemble predictions.

        Uses first 70% of data to train the meta-learner, then predicts on all data.
        This avoids look-ahead bias while still using all data for backtesting.
        """
        if df is None:
            df = self.features_df

        if df is None:
            raise ValueError("No data available. Load features first.")

        base_models = ["catboost", "lightgbm", "xgboost", "logreg"]
        available_models = [m for m in base_models if m in self.models]

        if len(available_models) < 2:
            logger.warning("Need at least 2 base models for stacking ensemble")
            return

        logger.info(f"Training stacking ensemble with {len(available_models)} base models...")

        # Get predictions from all base models
        all_preds = {}
        for model_name in available_models:
            preds = self.get_model_predictions(model_name, df)
            if preds is not None:
                all_preds[model_name] = preds

        if len(all_preds) < 2:
            logger.warning("Could not get predictions from enough models")
            return

        # Create stacked features matrix
        model_names = list(all_preds.keys())
        X_stack = np.column_stack([all_preds[m] for m in model_names])
        y = df["away_win"].values if "away_win" in df.columns else None

        if y is None:
            logger.warning("No target column found for training stacking")
            return

        # Use first 70% for training meta-learner (temporal split)
        n_train = int(len(df) * 0.7)
        X_train, y_train = X_stack[:n_train], y[:n_train]

        # Train meta-learner
        self.stacking_meta = RidgeClassifierCV(
            alphas=[0.001, 0.01, 0.1, 1.0, 10.0],
            cv=5
        )
        self.stacking_meta.fit(X_train, y_train)

        # Get stacking predictions for ALL data
        decision = self.stacking_meta.decision_function(X_stack)
        stacking_proba = 1 / (1 + np.exp(-decision))  # Sigmoid

        # Simple average ensemble
        average_proba = np.mean(X_stack, axis=1)

        # Cache predictions
        self.ensemble_predictions = {
            "stacking": stacking_proba,
            "average": average_proba,
            "model_names": model_names,
        }

        # Log training accuracy
        train_acc = ((stacking_proba[:n_train] >= 0.5) == y_train).mean()
        test_acc = ((stacking_proba[n_train:] >= 0.5) == y[n_train:]).mean()
        logger.info(f"Stacking meta-learner: train_acc={train_acc:.1%}, test_acc={test_acc:.1%}")
        logger.info(f"Meta-learner weights: {dict(zip(model_names, self.stacking_meta.coef_[0]))}")

    def get_model_predictions(
        self,
        model_name: str,
        X: pd.DataFrame
    ) -> np.ndarray:
        """Get probability predictions from a model (including stacking/average ensembles)."""
        # Handle ensemble predictions (stacking, average)
        if model_name in ["stacking", "average"]:
            if model_name not in self.ensemble_predictions:
                # Need to train stacking first
                if self.stacking_meta is None:
                    self.train_stacking_ensemble(X)
                if model_name not in self.ensemble_predictions:
                    return None
            return self.ensemble_predictions[model_name]

        model_data = self.models.get(model_name)
        if model_data is None:
            return None

        model = model_data.get("model")
        features = model_data.get("features", [])
        scaler = model_data.get("scaler")
        calibrator = model_data.get("calibrator")

        # Select features - must have ALL features for the model
        available_features = [f for f in features if f in X.columns]
        missing_features = [f for f in features if f not in X.columns]

        if missing_features:
            logger.debug(f"{model_name}: Missing {len(missing_features)} features, creating with zeros")

        # Create dataframe with all required features
        X_model = pd.DataFrame(index=X.index)
        for f in features:
            if f in X.columns:
                X_model[f] = X[f].values
            else:
                # Fill missing features with 0 (neutral value)
                X_model[f] = 0

        # Handle missing values
        X_model = X_model.fillna(0)

        # Scale if scaler exists
        if scaler is not None:
            try:
                X_model = pd.DataFrame(scaler.transform(X_model), columns=features, index=X_model.index)
            except Exception as e:
                logger.debug(f"Scaler failed: {e}")

        # Predict probabilities
        try:
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_model)[:, 1]
            else:
                probs = model.predict(X_model)
        except Exception as e:
            logger.warning(f"Prediction failed for {model_name}: {e}")
            return None

        # Calibrate if calibrator exists
        if calibrator is not None:
            try:
                probs = calibrator.predict_proba(probs.reshape(-1, 1))[:, 1]
            except Exception:
                pass  # Use uncalibrated if calibrator fails

        return probs

    def evaluate_config(
        self,
        config: SniperConfig,
        df: pd.DataFrame = None
    ) -> SniperResult:
        """Evaluate a sniper configuration on historical data."""
        if df is None:
            df = self.features_df.copy()

        # Get primary model predictions
        primary_probs = self.get_model_predictions(config.primary_model, df)
        if primary_probs is None:
            return None

        # Start with primary threshold filter
        mask = primary_probs >= config.primary_threshold

        # Add consensus filter if required
        if config.require_consensus and config.consensus_models:
            for consensus_model in config.consensus_models:
                consensus_probs = self.get_model_predictions(consensus_model, df)
                if consensus_probs is not None:
                    mask &= consensus_probs >= config.consensus_threshold

        # Add odds filter
        if "avg_away_open" in df.columns:
            odds = df["avg_away_open"].values
            mask &= (odds >= config.min_odds) & (odds <= config.max_odds)
        elif "odds_away_prob" in df.columns:
            # Convert implied prob to odds
            odds = 1 / df["odds_away_prob"].values
            mask &= (odds >= config.min_odds) & (odds <= config.max_odds)
        else:
            odds = np.ones(len(df)) * 3.0  # Default odds estimate

        # Add league filter
        if config.leagues and "league" in df.columns:
            league_mask = df["league"].isin(config.leagues)
            mask &= league_mask

        # Add position diff filter
        if config.min_position_diff is not None and "position_diff" in df.columns:
            # Positive position_diff means away team is higher in table
            mask &= df["position_diff"].values >= config.min_position_diff

        # Get filtered data
        filtered_idx = np.where(mask)[0]
        n_bets = len(filtered_idx)

        if n_bets == 0:
            return SniperResult(
                config=asdict(config),
                total_bets=0,
                wins=0,
                losses=0,
                precision=0.0,
                roi=0.0,
                avg_odds=0.0,
                min_odds_seen=0.0,
                max_odds_seen=0.0,
                by_league={},
                sample_bets=[],
                meets_target=False,
            )

        # Calculate results
        y_true = df["away_win"].values[filtered_idx]
        bet_odds = odds[filtered_idx] if len(odds) == len(df) else np.ones(n_bets) * 3.0

        wins = int(y_true.sum())
        losses = n_bets - wins
        precision = wins / n_bets if n_bets > 0 else 0

        # Calculate ROI
        # Win: odds - 1, Loss: -1
        returns = np.where(y_true == 1, bet_odds - 1, -1)
        roi = returns.mean() * 100 if len(returns) > 0 else 0

        # By league breakdown
        by_league = {}
        if "league" in df.columns:
            for league in df["league"].unique():
                league_mask_full = (df["league"] == league).values
                league_idx = np.where(mask & league_mask_full)[0]
                if len(league_idx) > 0:
                    league_wins = df["away_win"].values[league_idx].sum()
                    by_league[league] = {
                        "bets": len(league_idx),
                        "wins": int(league_wins),
                        "precision": league_wins / len(league_idx),
                    }

        # Sample bets for inspection
        sample_bets = []
        sample_idx = filtered_idx[:10]  # First 10 bets
        for idx in sample_idx:
            row = df.iloc[idx]
            sample_bets.append({
                "home_team": row.get("home_team", "Unknown"),
                "away_team": row.get("away_team", "Unknown"),
                "league": row.get("league", "Unknown"),
                "away_win": int(row["away_win"]),
                "prob": float(primary_probs[idx]),
                "odds": float(bet_odds[list(filtered_idx).index(idx)]) if idx in filtered_idx else 0,
            })

        return SniperResult(
            config=asdict(config),
            total_bets=n_bets,
            wins=wins,
            losses=losses,
            precision=precision,
            roi=roi,
            avg_odds=float(bet_odds.mean()) if len(bet_odds) > 0 else 0,
            min_odds_seen=float(bet_odds.min()) if len(bet_odds) > 0 else 0,
            max_odds_seen=float(bet_odds.max()) if len(bet_odds) > 0 else 0,
            by_league=by_league,
            sample_bets=sample_bets,
            meets_target=precision >= self.target_precision and n_bets >= self.min_bets,
            bet_outcomes=y_true.astype(int).tolist(),
            bet_odds=bet_odds.tolist(),
        )

    def grid_search(self) -> List[SniperResult]:
        """
        Search through configurations to find ones meeting target precision.
        """
        logger.info(f"Starting grid search for {self.target_precision:.0%} precision...")

        # Train stacking ensemble first (if we have enough models)
        if len(self.models) >= 2:
            self.train_stacking_ensemble()

        # Define search space - include stacking and average ensembles
        base_models = ["catboost", "lightgbm", "xgboost", "logreg"]
        primary_models = [m for m in base_models if m in self.models]

        # Add ensemble methods if stacking was trained
        if self.stacking_meta is not None:
            primary_models.extend(["stacking", "average"])
            logger.info(f"Added stacking and average ensembles to search space")

        thresholds = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
        consensus_options = [False, True]
        consensus_thresholds = [0.50, 0.55, 0.60]
        odds_ranges = [
            (1.5, 10.0),   # All odds
            (2.0, 5.0),    # Medium odds
            (2.5, 4.5),    # Tighter medium
            (3.0, 6.0),    # Higher odds only
        ]

        results = []
        configs_tested = 0

        # Ensemble models (already combining multiple models, no consensus needed)
        ensemble_models = {"stacking", "average"}

        for primary_model in primary_models:
            for threshold in thresholds:
                for require_consensus in consensus_options:
                    for min_odds, max_odds in odds_ranges:
                        # Ensemble models skip consensus (they're already ensembles)
                        if primary_model in ensemble_models and require_consensus:
                            continue

                        # Without consensus
                        if not require_consensus:
                            config = SniperConfig(
                                name=f"{primary_model}_{threshold:.2f}_odds_{min_odds}-{max_odds}",
                                primary_model=primary_model,
                                primary_threshold=threshold,
                                require_consensus=False,
                                min_odds=min_odds,
                                max_odds=max_odds,
                            )
                            result = self.evaluate_config(config)
                            if result:
                                results.append(result)
                            configs_tested += 1
                        else:
                            # With consensus - only use base models as consensus
                            # Don't use ensembles as consensus (redundant)
                            other_models = [m for m in base_models if m != primary_model and m in self.models]
                            for consensus_model in other_models:
                                for consensus_thresh in consensus_thresholds:
                                    config = SniperConfig(
                                        name=f"{primary_model}_{threshold:.2f}+{consensus_model}_{consensus_thresh:.2f}",
                                        primary_model=primary_model,
                                        primary_threshold=threshold,
                                        require_consensus=True,
                                        consensus_models=[consensus_model],
                                        consensus_threshold=consensus_thresh,
                                        min_odds=min_odds,
                                        max_odds=max_odds,
                                    )
                                    result = self.evaluate_config(config)
                                    if result:
                                        results.append(result)
                                    configs_tested += 1

        logger.info(f"Tested {configs_tested} configurations")

        # Sort by precision (descending), then by bets (descending)
        results.sort(key=lambda x: (x.precision, x.total_bets), reverse=True)

        self.results = results
        return results

    def find_best_configs(
        self,
        top_n: int = 20
    ) -> List[SniperResult]:
        """Find configurations that meet target precision."""
        if not self.results:
            self.grid_search()

        # Filter to those meeting criteria
        meeting_target = [
            r for r in self.results
            if r.precision >= self.target_precision and r.total_bets >= self.min_bets
        ]

        if meeting_target:
            logger.info(f"Found {len(meeting_target)} configs meeting {self.target_precision:.0%} precision!")
            return meeting_target[:top_n]
        else:
            # Return best we found
            logger.warning(f"No configs met {self.target_precision:.0%} target. Showing best found:")
            best = [r for r in self.results if r.total_bets >= self.min_bets]
            return best[:top_n]

    def print_results(self, results: List[SniperResult], top_n: int = 20):
        """Print results in a nice format."""
        print("\n" + "="*80)
        print(f"SNIPER MODE RESULTS - Target: {self.target_precision:.0%} precision, min {self.min_bets} bets")
        print("="*80)

        # Header
        print(f"\n{'Config':<50} {'Bets':>6} {'Wins':>6} {'Prec':>8} {'ROI':>8} {'Target':>8}")
        print("-"*80)

        for i, result in enumerate(results[:top_n], 1):
            target_met = "✓" if result.meets_target else "✗"
            print(
                f"{result.config['name']:<50} "
                f"{result.total_bets:>6} "
                f"{result.wins:>6} "
                f"{result.precision:>7.1%} "
                f"{result.roi:>7.1f}% "
                f"{target_met:>8}"
            )

        # Best result details
        if results:
            best = results[0]
            print("\n" + "="*80)
            print("BEST CONFIGURATION DETAILS")
            print("="*80)
            print(f"Config: {best.config['name']}")
            print(f"Primary model: {best.config['primary_model']} >= {best.config['primary_threshold']:.0%}")
            if best.config.get('require_consensus'):
                print(f"Consensus: {best.config['consensus_models']} >= {best.config['consensus_threshold']:.0%}")
            print(f"Odds range: {best.config['min_odds']} - {best.config['max_odds']}")
            print(f"\nResults:")
            print(f"  Total bets: {best.total_bets}")
            print(f"  Wins: {best.wins}, Losses: {best.losses}")
            print(f"  Precision: {best.precision:.1%}")
            print(f"  ROI: {best.roi:.1f}%")
            print(f"  Avg odds: {best.avg_odds:.2f}")

            if best.by_league:
                print(f"\nBy League:")
                for league, stats in sorted(best.by_league.items(), key=lambda x: x[1]['bets'], reverse=True):
                    print(f"  {league}: {stats['bets']} bets, {stats['wins']} wins ({stats['precision']:.0%})")

            if best.sample_bets:
                print(f"\nSample Bets:")
                for bet in best.sample_bets[:5]:
                    result_str = "✓ WIN" if bet['away_win'] else "✗ LOSS"
                    print(f"  {bet['away_team']} @ {bet['home_team']}: prob={bet['prob']:.0%}, {result_str}")

    def save_results(self, results: List[SniperResult], filename: str = None):
        """Save results to JSON."""
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sniper_away_win_{timestamp}.json"

        output_path = OUTPUT_DIR / filename

        output = {
            "generated_at": datetime.now().isoformat(),
            "target_precision": self.target_precision,
            "min_bets": self.min_bets,
            "total_configs_tested": len(self.results),
            "configs_meeting_target": len([r for r in results if r.meets_target]),
            "results": [asdict(r) for r in results],
        }

        with open(output_path, "w") as f:
            json.dump(output, f, indent=2, default=str)

        logger.info(f"Saved results to {output_path}")
        return output_path


def calc_bootstrap_metrics(
    outcomes: np.ndarray,
    odds: np.ndarray,
    n_boot: int = 1000,
) -> Dict[str, float]:
    """
    Calculate ROI with bootstrap confidence intervals, Sharpe and Sortino ratios.

    Args:
        outcomes: Array of bet outcomes (1=win, 0=loss)
        odds: Array of odds for each bet
        n_boot: Number of bootstrap iterations

    Returns:
        Dict with roi, ci_low, ci_high, p_profit, sharpe, sortino
    """
    if len(outcomes) == 0 or len(odds) == 0:
        return {
            'roi': 0.0, 'ci_low': 0.0, 'ci_high': 0.0,
            'p_profit': 0.0, 'sharpe': 0.0, 'sortino': 0.0
        }

    outcomes = np.array(outcomes)
    odds = np.array(odds)

    # Per-bet returns (win: odds-1, loss: -1)
    per_bet_returns = np.where(outcomes == 1, odds - 1, -1)

    # Bootstrap for ROI confidence intervals
    rois = []
    for _ in range(n_boot):
        idx = np.random.choice(len(per_bet_returns), len(per_bet_returns), replace=True)
        boot_returns = per_bet_returns[idx]
        rois.append(boot_returns.mean() * 100)

    # Sharpe ratio (mean / std)
    mean_return = np.mean(per_bet_returns)
    std_return = np.std(per_bet_returns)
    sharpe = mean_return / std_return if std_return > 0 else 0

    # Sortino ratio (mean / downside std)
    downside_returns = per_bet_returns[per_bet_returns < 0]
    downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0
    sortino = mean_return / downside_std if downside_std > 0 else (mean_return if mean_return > 0 else 0)

    return {
        'roi': float(np.mean(rois)),
        'ci_low': float(np.percentile(rois, 2.5)),
        'ci_high': float(np.percentile(rois, 97.5)),
        'p_profit': float((np.array(rois) > 0).mean()),
        'sharpe': float(sharpe),
        'sortino': float(sortino),
    }


def walk_forward_validate(
    optimizer: 'SniperModeOptimizer',
    config: SniperConfig,
    n_folds: int = 5,
) -> Dict[str, Any]:
    """
    Perform walk-forward validation on a sniper configuration.

    This is the TRUE test - train on past data, test on future data.

    Args:
        optimizer: The optimizer instance with loaded data
        config: Configuration to validate
        n_folds: Number of time-based folds

    Returns:
        Dict with fold-by-fold results and aggregate statistics
    """
    df = optimizer.features_df.copy()

    # Sort by date if available
    date_col = None
    for col in ["date", "match_date", "kickoff", "fixture_date"]:
        if col in df.columns:
            date_col = col
            break

    if date_col:
        df = df.sort_values(date_col).reset_index(drop=True)
    else:
        logger.warning("No date column found, using index order (may not be chronological)")

    n_samples = len(df)
    fold_size = n_samples // (n_folds + 1)  # +1 because first fold is training only

    fold_results = []
    all_outcomes = []  # Track all bet outcomes (1=win, 0=loss)
    all_odds = []      # Track all bet odds

    logger.info(f"Walk-forward validation with {n_folds} folds...")
    logger.info(f"Total samples: {n_samples}, fold size: ~{fold_size}")

    for fold in range(n_folds):
        # Training data: all data up to this fold
        train_end = (fold + 1) * fold_size
        # Test data: next fold
        test_start = train_end
        test_end = min(test_start + fold_size, n_samples)

        if test_end <= test_start:
            continue

        test_df = df.iloc[test_start:test_end].copy()

        # Evaluate on test fold (models already trained, just filtering)
        result = optimizer.evaluate_config(config, test_df)

        if result and result.total_bets > 0:
            fold_results.append({
                "fold": fold + 1,
                "test_start_idx": test_start,
                "test_end_idx": test_end,
                "n_bets": result.total_bets,
                "wins": result.wins,
                "losses": result.losses,
                "precision": result.precision,
                "roi": result.roi,
                "avg_odds": result.avg_odds,
            })

            # Track per-bet outcomes and odds for bootstrap calculation
            if result.bet_outcomes and result.bet_odds:
                all_outcomes.extend(result.bet_outcomes)
                all_odds.extend(result.bet_odds)

            logger.info(
                f"  Fold {fold + 1}: {result.wins}/{result.total_bets} = "
                f"{result.precision:.1%} precision, ROI: {result.roi:.1f}%"
            )

    # Aggregate statistics
    total_bets = sum(f["n_bets"] for f in fold_results)
    total_wins = sum(f["wins"] for f in fold_results)

    if total_bets > 0:
        overall_precision = total_wins / total_bets
        precisions = [f["precision"] for f in fold_results if f["n_bets"] > 0]
        rois = [f["roi"] for f in fold_results if f["n_bets"] > 0]

        # Calculate bootstrap metrics (P(profit), Sharpe, Sortino)
        bootstrap_metrics = calc_bootstrap_metrics(
            np.array(all_outcomes),
            np.array(all_odds),
            n_boot=1000
        )

        return {
            "config": config.name,
            "n_folds": len(fold_results),
            "fold_results": fold_results,
            "total_bets": total_bets,
            "total_wins": total_wins,
            "total_losses": total_bets - total_wins,
            "overall_precision": overall_precision,
            "avg_precision": np.mean(precisions) if precisions else 0,
            "std_precision": np.std(precisions) if len(precisions) > 1 else 0,
            "min_precision": min(precisions) if precisions else 0,
            "max_precision": max(precisions) if precisions else 0,
            "avg_roi": np.mean(rois) if rois else 0,
            "std_roi": np.std(rois) if len(rois) > 1 else 0,
            "all_folds_profitable": all(f["precision"] > 0.5 for f in fold_results),
            "meets_target": overall_precision >= optimizer.target_precision,
            # Bootstrap metrics (like full optimization)
            "p_profit": bootstrap_metrics["p_profit"],
            "sharpe": bootstrap_metrics["sharpe"],
            "sortino": bootstrap_metrics["sortino"],
            "roi_ci_low": bootstrap_metrics["ci_low"],
            "roi_ci_high": bootstrap_metrics["ci_high"],
        }
    else:
        return {
            "config": config.name,
            "n_folds": 0,
            "fold_results": [],
            "total_bets": 0,
            "overall_precision": 0,
            "meets_target": False,
            "p_profit": 0,
            "sharpe": 0,
            "sortino": 0,
            "roi_ci_low": 0,
            "roi_ci_high": 0,
        }


def main():
    parser = argparse.ArgumentParser(description="Sniper Mode: Away Win Precision Optimizer")
    parser.add_argument(
        "--target-precision",
        type=float,
        default=0.90,
        help="Target precision (default: 0.90 = 90%%)"
    )
    parser.add_argument(
        "--min-bets",
        type=int,
        default=10,
        help="Minimum bets required (default: 10)"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Show top N results (default: 20)"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run walk-forward validation on top configs"
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of folds for walk-forward validation (default: 5)"
    )
    args = parser.parse_args()

    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║                    SNIPER MODE: AWAY WIN                        ║
║                                                                  ║
║  Target: {args.target_precision:.0%} precision                                        ║
║  Philosophy: Quality over quantity - only bet when we're sure   ║
╚══════════════════════════════════════════════════════════════════╝
    """)

    optimizer = SniperModeOptimizer(
        target_precision=args.target_precision,
        min_bets=args.min_bets,
    )

    # Load data
    logger.info("Loading models...")
    optimizer.load_models()

    logger.info("Loading features...")
    optimizer.load_features()

    # Run optimization
    logger.info("Running grid search...")
    optimizer.grid_search()

    # Find best configs
    best_configs = optimizer.find_best_configs(top_n=args.top_n)

    # Print and save results
    optimizer.print_results(best_configs, top_n=args.top_n)
    optimizer.save_results(best_configs)

    # Summary
    meeting_target = [r for r in best_configs if r.meets_target]
    if meeting_target:
        print(f"\n✓ SUCCESS: Found {len(meeting_target)} configurations meeting {args.target_precision:.0%} precision!")
        print(f"  Best: {meeting_target[0].precision:.1%} precision with {meeting_target[0].total_bets} bets")
    else:
        best = best_configs[0] if best_configs else None
        if best:
            print(f"\n✗ Target not met. Best found: {best.precision:.1%} precision with {best.total_bets} bets")
            gap = args.target_precision - best.precision
            print(f"  Gap to target: {gap:.1%}")
            print(f"\n  Suggestions to reach {args.target_precision:.0%}:")
            print(f"    1. Raise threshold further (may reduce volume)")
            print(f"    2. Add more consensus models")
            print(f"    3. Filter by specific leagues or conditions")
            print(f"    4. Add new predictive features")

    # Walk-forward validation
    if args.validate and best_configs:
        print("\n" + "="*80)
        print("WALK-FORWARD VALIDATION (True Out-of-Sample Test)")
        print("="*80)
        print(f"Testing top {min(10, len(best_configs))} configurations with {args.n_folds} folds...\n")

        validation_results = []

        for i, result in enumerate(best_configs[:10], 1):
            # Recreate config from result
            cfg = result.config
            config = SniperConfig(
                name=cfg["name"],
                primary_model=cfg["primary_model"],
                primary_threshold=cfg["primary_threshold"],
                require_consensus=cfg.get("require_consensus", False),
                consensus_models=cfg.get("consensus_models", []),
                consensus_threshold=cfg.get("consensus_threshold", 0.5),
                min_odds=cfg.get("min_odds", 1.5),
                max_odds=cfg.get("max_odds", 10.0),
            )

            wf_result = walk_forward_validate(optimizer, config, n_folds=args.n_folds)
            validation_results.append(wf_result)

        # Print validation results
        print("\n" + "-"*100)
        print(f"{'Config':<40} {'Bets':>5} {'Prec':>7} {'ROI':>8} {'P(profit)':>10} {'Sharpe':>8} {'Sortino':>8}")
        print("-"*100)

        for vr in validation_results:
            if vr["total_bets"] > 0:
                print(
                    f"{vr['config']:<40} "
                    f"{vr['total_bets']:>5} "
                    f"{vr['overall_precision']:>6.1%} "
                    f"{vr['avg_roi']:>7.1f}% "
                    f"{vr.get('p_profit', 0):>9.1%} "
                    f"{vr.get('sharpe', 0):>+7.3f} "
                    f"{vr.get('sortino', 0):>+7.3f}"
                )

        # Find best validated config
        valid_results = [v for v in validation_results if v["total_bets"] >= args.min_bets]
        if valid_results:
            best_validated = max(valid_results, key=lambda x: x["overall_precision"])

            print("\n" + "="*80)
            print("BEST VALIDATED CONFIGURATION")
            print("="*80)
            print(f"Config: {best_validated['config']}")
            print(f"\nWalk-Forward Results ({best_validated['n_folds']} folds):")
            print(f"  Total bets: {best_validated['total_bets']}")
            print(f"  Total wins: {best_validated['total_wins']}")
            print(f"  Overall precision: {best_validated['overall_precision']:.1%}")
            print(f"  Precision range: {best_validated['min_precision']:.1%} - {best_validated['max_precision']:.1%} (±{best_validated['std_precision']:.1%})")
            print(f"  Avg ROI: {best_validated['avg_roi']:.1f}% (±{best_validated['std_roi']:.1f}%)")
            print(f"  ROI 95% CI: [{best_validated.get('roi_ci_low', 0):.1f}%, {best_validated.get('roi_ci_high', 0):.1f}%]")
            print(f"\nRisk-Adjusted Metrics:")
            print(f"  P(profit): {best_validated.get('p_profit', 0):.1%}")
            print(f"  Sharpe Ratio: {best_validated.get('sharpe', 0):+.3f}")
            print(f"  Sortino Ratio: {best_validated.get('sortino', 0):+.3f}")
            print(f"  All folds profitable: {'Yes' if best_validated['all_folds_profitable'] else 'No'}")

            print(f"\nFold-by-Fold Breakdown:")
            for fold in best_validated["fold_results"]:
                status = "✓" if fold["precision"] >= args.target_precision else "✗"
                print(
                    f"  Fold {fold['fold']}: {fold['wins']}/{fold['n_bets']} = "
                    f"{fold['precision']:.0%} precision, ROI: {fold['roi']:.1f}% {status}"
                )

            if best_validated["overall_precision"] >= args.target_precision:
                print(f"\n✓ VALIDATED: {best_validated['overall_precision']:.1%} precision holds in walk-forward!")
            else:
                print(f"\n✗ WARNING: Precision dropped to {best_validated['overall_precision']:.1%} in walk-forward")
                print(f"  This suggests overfitting. Consider:")
                print(f"    - Raising thresholds further")
                print(f"    - Adding more consensus requirements")
                print(f"    - Using more conservative odds ranges")

        # Save validation results
        validation_output = {
            "generated_at": datetime.now().isoformat(),
            "target_precision": args.target_precision,
            "n_folds": args.n_folds,
            "validation_results": validation_results,
            # Best config summary (like full optimization output)
            "best_config": best_validated["config"] if valid_results else None,
            "best_precision": best_validated["overall_precision"] if valid_results else None,
            "best_roi": best_validated["avg_roi"] if valid_results else None,
            "best_p_profit": best_validated.get("p_profit", 0) if valid_results else None,
            "best_sharpe": best_validated.get("sharpe", 0) if valid_results else None,
            "best_sortino": best_validated.get("sortino", 0) if valid_results else None,
            "best_bets": best_validated["total_bets"] if valid_results else None,
            "meets_target": best_validated.get("meets_target", False) if valid_results else False,
        }

        val_path = OUTPUT_DIR / f"sniper_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(val_path, "w") as f:
            json.dump(validation_output, f, indent=2, default=str)
        logger.info(f"Saved validation results to {val_path}")

        # Print summary similar to full optimization
        print("\n" + "="*80)
        print("SNIPER OPTIMIZATION SUMMARY")
        print("="*80)
        if valid_results:
            print(f"  Best Strategy: {best_validated['config']}")
            print(f"  Precision: {best_validated['overall_precision']:.1%}")
            print(f"  ROI: {best_validated['avg_roi']:.1f}%")
            print(f"  P(profit): {best_validated.get('p_profit', 0):.1%}")
            print(f"  Sharpe: {best_validated.get('sharpe', 0):+.3f}")
            print(f"  Sortino: {best_validated.get('sortino', 0):+.3f}")
            print(f"  Bets: {best_validated['total_bets']}")
        else:
            print("  No valid configurations found")


if __name__ == "__main__":
    main()
