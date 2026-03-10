#!/usr/bin/env python3
"""
Compute cross-market conformal pooling thresholds.

Reads holdout prediction CSVs from sniper optimization, fits a
CrossMarketConformalPooler across all markets, and writes the
per-market conformal_tau_pooled into the deployment config.

Usage:
    python scripts/compute_conformal_pooling.py
    python scripts/compute_conformal_pooling.py --source experiments/outputs/sniper_results/
    python scripts/compute_conformal_pooling.py --alpha 0.05
    python scripts/compute_conformal_pooling.py --deployment-config config/sniper_deployment.json
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration.conformal_pooling import CrossMarketConformalPooler

logger = logging.getLogger(__name__)


def find_holdout_csvs(source_dir: Path) -> dict[str, Path]:
    """Glob for holdout_preds_*.csv and extract market names.

    Args:
        source_dir: Directory to search for holdout prediction CSVs.

    Returns:
        Mapping of market name to CSV path, sorted alphabetically.
    """
    csvs: dict[str, Path] = {}
    for path in sorted(source_dir.glob("holdout_preds_*.csv")):
        # Filename format: holdout_preds_{market}.csv
        market = path.stem.replace("holdout_preds_", "")
        if market:
            csvs[market] = path
    return csvs


def load_holdout_arrays(
    csv_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Load prob and actual arrays from a holdout predictions CSV.

    Args:
        csv_path: Path to the holdout CSV (columns: prob, actual, ...).

    Returns:
        Tuple of (probabilities, actuals) as float64 arrays.

    Raises:
        ValueError: If required columns are missing or arrays are empty.
    """
    df = pd.read_csv(csv_path)

    for col in ("prob", "actual"):
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {csv_path}")

    probs = df["prob"].to_numpy(dtype=np.float64)
    actuals = df["actual"].to_numpy(dtype=np.float64)

    if len(probs) == 0:
        raise ValueError(f"Empty holdout CSV: {csv_path}")

    return probs, actuals


def update_deployment_config(
    config_path: Path,
    tau_per_market: dict[str, float],
) -> dict:
    """Load deployment config, inject conformal_tau_pooled per market, and save.

    Args:
        config_path: Path to sniper_deployment.json.
        tau_per_market: Mapping of market name to pooled conformal tau.

    Returns:
        The updated config dict.
    """
    with open(config_path) as f:
        config = json.load(f)

    markets = config.get("markets", {})
    updated: list[str] = []

    for market, tau in tau_per_market.items():
        if market in markets:
            markets[market]["conformal_tau_pooled"] = round(tau, 6)
            updated.append(market)
        else:
            logger.warning(
                f"Market '{market}' has pooled tau but is not in deployment config — skipping"
            )

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"Updated {len(updated)} market(s) in {config_path}")
    return config


def print_summary(
    tau_per_market: dict[str, float],
    n_samples: dict[str, int],
    alpha: float,
) -> None:
    """Print a formatted summary table of per-market conformal tau values.

    Args:
        tau_per_market: Mapping of market name to pooled tau.
        n_samples: Mapping of market name to holdout sample count.
        alpha: Significance level used for fitting.
    """
    print()
    print("=" * 60)
    print(f"CONFORMAL POOLING SUMMARY  (alpha={alpha})")
    print("=" * 60)
    print(f"  {'Market':<30s} {'N':>6s}  {'tau_pooled':>12s}")
    print("  " + "-" * 50)

    for market in sorted(tau_per_market):
        tau = tau_per_market[market]
        n = n_samples.get(market, 0)
        print(f"  {market:<30s} {n:>6d}  {tau:>12.6f}")

    print("  " + "-" * 50)
    total_n = sum(n_samples.values())
    mean_tau = np.mean(list(tau_per_market.values()))
    print(f"  {'TOTAL / MEAN':<30s} {total_n:>6d}  {mean_tau:>12.6f}")
    print("=" * 60)
    print()


def main() -> int:
    """Run cross-market conformal pooling and update deployment config."""
    parser = argparse.ArgumentParser(
        description="Compute cross-market conformal pooling thresholds from holdout predictions."
    )
    parser.add_argument(
        "--source",
        type=str,
        default="experiments/outputs/sniper_results/",
        help="Directory containing holdout_preds_*.csv files (default: experiments/outputs/sniper_results/)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.10,
        help="Significance level for conformal prediction (default: 0.10)",
    )
    parser.add_argument(
        "--deployment-config",
        type=str,
        default="config/sniper_deployment.json",
        help="Path to deployment config to update (default: config/sniper_deployment.json)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    source_dir = PROJECT_ROOT / args.source
    config_path = PROJECT_ROOT / args.deployment_config

    # 1. Find holdout CSVs
    csvs = find_holdout_csvs(source_dir)
    if not csvs:
        logger.error(f"No holdout_preds_*.csv files found in {source_dir}")
        return 1

    logger.info(f"Found {len(csvs)} holdout prediction file(s) in {source_dir}")

    # 2. Build pooler and add each market
    pooler = CrossMarketConformalPooler()
    n_samples: dict[str, int] = {}

    for market, csv_path in sorted(csvs.items()):
        try:
            probs, actuals = load_holdout_arrays(csv_path)
            pooler.add_market(market, probs, actuals)
            n_samples[market] = len(probs)
            logger.info(f"  Added {market}: {len(probs)} samples")
        except (ValueError, KeyError) as e:
            logger.warning(f"  Skipping {market}: {e}")

    if not n_samples:
        logger.error("No valid holdout data loaded — nothing to fit")
        return 1

    # 3. Fit and get per-market tau
    pooler.fit(alpha=args.alpha)
    tau_per_market = pooler.get_tau_per_market()

    logger.info(
        f"Fitted conformal pooler across {len(tau_per_market)} market(s) at alpha={args.alpha}"
    )

    # 4. Update deployment config
    if not config_path.exists():
        logger.error(f"Deployment config not found: {config_path}")
        return 1

    update_deployment_config(config_path, tau_per_market)

    # 5. Print summary
    print_summary(tau_per_market, n_samples, args.alpha)

    return 0


if __name__ == "__main__":
    sys.exit(main())
