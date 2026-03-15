#!/usr/bin/env python3
"""Pool conformal taus across markets for data-poor niche markets.

Reads per-market conformal_tau and conformal_residuals from model artifacts
(joblib files), applies CrossMarketConformalPooler, and updates
sniper_deployment.json with conformal_tau_pooled for data-poor markets
(< POOL_CANDIDATES_THRESHOLD holdout bets).

Falls back to synthetic residuals when raw residuals are not available
(backward compatibility with models saved before residual storage).

Usage:
    python scripts/pool_conformal_taus.py
    python scripts/pool_conformal_taus.py --models-dir models/
    python scripts/pool_conformal_taus.py --alpha 0.05
    python scripts/pool_conformal_taus.py --deployment-config config/sniper_deployment.json
    python scripts/pool_conformal_taus.py --pool-threshold 30
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration.conformal_pooling import (
    MIN_SAMPLES_PER_MARKET,
    CrossMarketConformalPooler,
)

logger = logging.getLogger(__name__)

# Markets with fewer holdout bets than this get pooled tau
POOL_CANDIDATES_THRESHOLD: int = 50


def _load_residuals_from_artifact(
    model_path: Path,
) -> Optional[List[float]]:
    """Load conformal_residuals from a joblib model artifact.

    Args:
        model_path: Path to a .joblib model artifact file.

    Returns:
        List of residual values, or None if not found or load fails.
    """
    try:
        import joblib

        model_data = joblib.load(model_path)
        if isinstance(model_data, dict):
            residuals = model_data.get("conformal_residuals")
            if residuals is not None:
                return list(residuals)
    except Exception as e:
        logger.debug(f"Could not load residuals from {model_path}: {e}")
    return None


def _synthesize_residuals(
    tau: float,
    alpha: float = 0.10,
    n_synthetic: int = 100,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Synthesize approximate (preds, actuals) from a scalar tau.

    When raw residuals are unavailable (models saved before residual storage),
    approximate the residual distribution as uniform [0, scale * 2] where
    scale = tau / quantile_level. This is a rough approximation that allows
    pooling to function with legacy models.

    Args:
        tau: Per-market conformal tau value.
        alpha: Significance level used when tau was computed.
        n_synthetic: Number of synthetic samples to generate.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (synthetic_preds, synthetic_actuals) arrays.
    """
    rng = np.random.RandomState(seed)
    quantile_level = 1.0 - alpha
    # tau = (1-alpha) quantile of (preds - actuals)
    # Approximate scale as tau / quantile_level
    scale = tau / quantile_level if quantile_level > 0 else tau
    # Synthesize residuals as uniform [0, scale * 2] to approximate distribution
    residuals = rng.uniform(0, scale * 2, n_synthetic)
    # Split into preds/actuals: preds = residuals, actuals = 0
    return residuals, np.zeros(n_synthetic)


def collect_market_data(
    config: dict,
    models_dir: Path,
    alpha: float,
) -> Tuple[
    Dict[str, Tuple[np.ndarray, np.ndarray]],
    List[str],
    Dict[str, str],
]:
    """Collect per-market predictions/actuals for pooling.

    Tries to load raw residuals from model artifacts first. Falls back to
    synthetic residuals derived from the scalar conformal_tau.

    Args:
        config: Parsed deployment config dict.
        models_dir: Directory containing .joblib model files.
        alpha: Significance level for synthetic residual synthesis.

    Returns:
        Tuple of:
        - market_arrays: Dict mapping market name to (preds, actuals) arrays.
        - data_poor_markets: List of market names with < POOL_CANDIDATES_THRESHOLD holdout bets.
        - sources: Dict mapping market name to data source ("artifact" or "synthetic").
    """
    market_arrays: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    data_poor_markets: List[str] = []
    sources: Dict[str, str] = {}

    for market, mconfig in config.get("markets", {}).items():
        if not mconfig.get("enabled", False):
            continue

        tau = mconfig.get("conformal_tau", 0)
        n_bets = mconfig.get("holdout_metrics", {}).get("n_bets", 0)

        if n_bets < POOL_CANDIDATES_THRESHOLD:
            data_poor_markets.append(market)

        if tau <= 0:
            logger.debug(f"Skipping {market}: no conformal_tau")
            continue

        # Try loading raw residuals from model artifacts
        residuals_loaded = False
        saved_models = mconfig.get("saved_models", [])
        for model_file in saved_models:
            model_path = models_dir / model_file
            if model_path.exists():
                raw_residuals = _load_residuals_from_artifact(model_path)
                if raw_residuals is not None and len(raw_residuals) >= MIN_SAMPLES_PER_MARKET:
                    residuals_arr = np.array(raw_residuals, dtype=np.float64)
                    # Residuals are (preds - actuals). Reconstruct as preds=residuals, actuals=0.
                    market_arrays[market] = (residuals_arr, np.zeros(len(residuals_arr)))
                    sources[market] = "artifact"
                    residuals_loaded = True
                    logger.info(
                        f"  {market}: loaded {len(raw_residuals)} residuals from {model_file}"
                    )
                    break

        # Fall back to synthetic residuals from scalar tau
        if not residuals_loaded:
            preds, actuals = _synthesize_residuals(tau, alpha=alpha)
            if len(preds) >= MIN_SAMPLES_PER_MARKET:
                market_arrays[market] = (preds, actuals)
                sources[market] = "synthetic"
                logger.info(
                    f"  {market}: using synthetic residuals (tau={tau:.4f}, "
                    f"no raw residuals in artifacts)"
                )

    return market_arrays, data_poor_markets, sources


def pool_conformal_taus(
    config_path: Path,
    models_dir: Path,
    alpha: float = 0.10,
    pool_threshold: int = POOL_CANDIDATES_THRESHOLD,
) -> Dict[str, float]:
    """Run cross-market conformal pooling and update deployment config.

    Args:
        config_path: Path to sniper_deployment.json.
        models_dir: Directory containing .joblib model files.
        alpha: Significance level for conformal prediction.
        pool_threshold: Markets with fewer holdout bets get pooled tau.

    Returns:
        Dict mapping market name to its pooled conformal tau (only data-poor markets).
    """
    global POOL_CANDIDATES_THRESHOLD
    POOL_CANDIDATES_THRESHOLD = pool_threshold

    with open(config_path) as f:
        config = json.load(f)

    market_arrays, data_poor_markets, sources = collect_market_data(
        config, models_dir, alpha
    )

    if len(market_arrays) < 2:
        logger.warning(
            f"Only {len(market_arrays)} market(s) with tau available "
            f"-- pooling requires at least 2. Skipping."
        )
        return {}

    # Build pooler from collected data
    pooler = CrossMarketConformalPooler()
    for market, (preds, actuals) in market_arrays.items():
        pooler.add_market(market, preds, actuals)

    pooler.fit(alpha=alpha)
    all_pooled_taus = pooler.get_tau_per_market()

    # Write pooled taus only for data-poor markets
    updated: Dict[str, float] = {}
    markets_cfg = config.get("markets", {})

    for market in data_poor_markets:
        if market in all_pooled_taus and market in markets_cfg:
            pooled_tau = all_pooled_taus[market]
            markets_cfg[market]["conformal_tau_pooled"] = round(pooled_tau, 6)
            updated[market] = pooled_tau
            per_market_tau = markets_cfg[market].get("conformal_tau", 0)
            source = sources.get(market, "unknown")
            logger.info(
                f"  {market}: per-market tau={per_market_tau:.4f}, "
                f"pooled tau={pooled_tau:.4f} (source: {source})"
            )

    # Save updated config
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")

    logger.info(
        f"Updated {len(updated)} data-poor market(s) with pooled conformal tau "
        f"(threshold: <{pool_threshold} holdout bets)"
    )

    return updated


def print_summary(
    updated: Dict[str, float],
    config_path: Path,
    alpha: float,
    pool_threshold: int,
) -> None:
    """Print a formatted summary of pooling results.

    Args:
        updated: Dict of market name to pooled tau for data-poor markets.
        config_path: Path to deployment config that was updated.
        alpha: Significance level used.
        pool_threshold: Holdout bet threshold for data-poor classification.
    """
    with open(config_path) as f:
        config = json.load(f)

    markets_cfg = config.get("markets", {})

    print()
    print("=" * 70)
    print(f"CONFORMAL POOLING SUMMARY  (alpha={alpha}, pool_threshold={pool_threshold})")
    print("=" * 70)
    print(
        f"  {'Market':<30s} {'HO bets':>8s}  {'tau':>10s}  {'pooled':>10s}  {'Status':<10s}"
    )
    print("  " + "-" * 68)

    for market in sorted(markets_cfg.keys()):
        mconfig = markets_cfg[market]
        if not mconfig.get("enabled", False):
            continue
        n_bets = mconfig.get("holdout_metrics", {}).get("n_bets", 0)
        tau = mconfig.get("conformal_tau", 0)
        pooled = mconfig.get("conformal_tau_pooled")
        status = "POOLED" if market in updated else ("--" if tau <= 0 else "per-mkt")
        pooled_str = f"{pooled:.6f}" if pooled is not None else "--"
        tau_str = f"{tau:.6f}" if tau > 0 else "--"
        print(f"  {market:<30s} {n_bets:>8d}  {tau_str:>10s}  {pooled_str:>10s}  {status:<10s}")

    print("  " + "-" * 68)
    print(f"  Data-poor markets updated: {len(updated)}")
    print("=" * 70)
    print()


def main() -> int:
    """Entry point for cross-market conformal pooling."""
    parser = argparse.ArgumentParser(
        description=(
            "Pool conformal taus across markets for data-poor niche markets. "
            "Reads per-market conformal_tau from deployment config and raw "
            "residuals from model artifacts, applies CrossMarketConformalPooler, "
            "and writes conformal_tau_pooled for data-poor markets."
        ),
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Directory containing .joblib model files (default: models/)",
    )
    parser.add_argument(
        "--deployment-config",
        type=str,
        default="config/sniper_deployment.json",
        help="Path to deployment config (default: config/sniper_deployment.json)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.10,
        help="Significance level for conformal prediction (default: 0.10)",
    )
    parser.add_argument(
        "--pool-threshold",
        type=int,
        default=POOL_CANDIDATES_THRESHOLD,
        help=(
            f"Markets with fewer holdout bets than this get pooled tau "
            f"(default: {POOL_CANDIDATES_THRESHOLD})"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be updated without writing to deployment config",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    config_path = PROJECT_ROOT / args.deployment_config
    models_dir = PROJECT_ROOT / args.models_dir

    if not config_path.exists():
        logger.error(f"Deployment config not found: {config_path}")
        return 1

    if args.dry_run:
        logger.info("DRY RUN — will not write to deployment config")
        # Load config, collect data, fit pooler, print summary without saving
        with open(config_path) as f:
            config = json.load(f)
        market_arrays, data_poor_markets, sources = collect_market_data(
            config, models_dir, args.alpha
        )
        if len(market_arrays) < 2:
            logger.warning("Not enough markets for pooling")
            return 0
        pooler = CrossMarketConformalPooler()
        for market, (preds, actuals) in market_arrays.items():
            pooler.add_market(market, preds, actuals)
        pooler.fit(alpha=args.alpha)
        all_taus = pooler.get_tau_per_market()
        updated = {m: all_taus[m] for m in data_poor_markets if m in all_taus}
        # Temporarily inject pooled taus for display
        for market, tau in updated.items():
            config["markets"][market]["conformal_tau_pooled"] = round(tau, 6)
        # Write temp for summary display, then restore
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            json.dump(config, tmp, indent=2)
            tmp_path = Path(tmp.name)
        print_summary(updated, tmp_path, args.alpha, args.pool_threshold)
        tmp_path.unlink()
        return 0

    updated = pool_conformal_taus(
        config_path=config_path,
        models_dir=models_dir,
        alpha=args.alpha,
        pool_threshold=args.pool_threshold,
    )

    print_summary(updated, config_path, args.alpha, args.pool_threshold)

    return 0


if __name__ == "__main__":
    sys.exit(main())
