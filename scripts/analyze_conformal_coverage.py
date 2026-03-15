#!/usr/bin/env python3
"""Analyze conformal coverage on all historical settled bets.

Reads recommendation CSVs from ``data/05-recommendations/`` and the
prediction ledger at ``data/preds/predictions.parquet`` to find
settled bets that carry conformal data (``conformal_tau``,
``conformal_lower``).

DATA AVAILABILITY NOTES
-----------------------
- ``conformal_tau`` and ``conformal_lower`` are present in rec CSVs
  generated since ~Mar 2026 (8 files at the time of writing).
- **VA interval data (va_lower, va_upper) is NOT persisted** in
  recommendation CSVs or the prediction ledger.  It is computed at
  prediction time but only used inside ``MarketHealthReport``.
  Until VA columns are added to the CSV output in
  ``generate_daily_recommendations.py``, the ``va_consistency`` and
  ``width_accuracy_correlation`` checks cannot run on production data.
- As of Mar 15, 2026, zero settled bets have conformal data — all
  rec CSVs with ``conformal_tau`` are still pending settlement.
  This script will produce useful output once those bets settle.

Usage::

    python scripts/analyze_conformal_coverage.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.monitoring.conformal_coverage import (
    one_sided_coverage,
    rolling_coverage,
    va_consistency,
    width_accuracy_correlation,
)

REC_DIR = PROJECT_ROOT / "data" / "05-recommendations"
LEDGER_PATH = PROJECT_ROOT / "data" / "preds" / "predictions.parquet"


def _load_settled_with_conformal() -> pd.DataFrame:
    """Collect settled bets that carry conformal_tau from rec CSVs.

    The prediction ledger (parquet) does not store conformal columns,
    so we scan all ``rec_*.csv`` files and keep rows that are both
    settled (result W or L) and have a non-null ``conformal_tau > 0``.

    Returns
    -------
    pd.DataFrame
        Combined dataframe of settled bets with conformal data.
        Empty if no matching rows exist.
    """
    frames: List[pd.DataFrame] = []
    for csv_path in sorted(REC_DIR.rglob("rec_*.csv")):
        if "_week" in csv_path.name:
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue

        if "conformal_tau" not in df.columns:
            continue
        if "result" not in df.columns:
            continue

        settled = df[df["result"].isin(["W", "L"])].copy()
        if settled.empty:
            continue

        has_tau = settled["conformal_tau"].notna() & (settled["conformal_tau"] > 0)
        settled = settled[has_tau]
        if settled.empty:
            continue

        settled["source_file"] = csv_path.name
        frames.append(settled)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _print_header(title: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def analyze_conformal_tau_coverage(df: pd.DataFrame) -> None:
    """Analyze one-sided conformal coverage overall and per market."""
    _print_header("CONFORMAL TAU COVERAGE")

    probs = df["probability"].astype(float).values
    actuals = (df["result"] == "W").astype(float).values
    taus = df["conformal_tau"].astype(float).values

    # Overall
    overall = one_sided_coverage(probs, taus, actuals)
    alert_str = " << ALERT" if overall["alert"] else " (OK)"
    print(
        f"\n  Overall: coverage={overall['empirical_coverage']:.1%} "
        f"(nominal={overall['nominal_coverage']:.0%}, "
        f"gap={overall['coverage_gap']:+.1%}, "
        f"n={overall['n_bets']}){alert_str}"
    )

    # Per market
    print("\n  Per-market breakdown:")
    by_market = defaultdict(list)
    for _, row in df.iterrows():
        by_market[row.get("market", "unknown")].append(row)

    for market in sorted(by_market.keys()):
        rows = by_market[market]
        m_probs = np.array([float(r["probability"]) for r in rows])
        m_taus = np.array([float(r["conformal_tau"]) for r in rows])
        m_actuals = np.array([1.0 if r["result"] == "W" else 0.0 for r in rows])
        if len(m_actuals) < 3:
            continue
        result = one_sided_coverage(m_probs, m_taus, m_actuals)
        flag = " << ALERT" if result["alert"] else ""
        print(
            f"    {market:30s}: coverage={result['empirical_coverage']:.1%} "
            f"(n={result['n_bets']}){flag}"
        )


def analyze_rolling_coverage(df: pd.DataFrame) -> None:
    """Show rolling coverage summary for the full bet sequence."""
    _print_header("ROLLING COVERAGE (window=50)")

    probs = df["probability"].astype(float).values
    actuals = (df["result"] == "W").astype(float).values
    taus = df["conformal_tau"].astype(float).values
    scores = probs - actuals

    if len(scores) < 50:
        print(f"\n  Insufficient bets for rolling analysis ({len(scores)} < 50)")
        return

    rolling = rolling_coverage(scores, taus, window=50)
    valid = rolling[~np.isnan(rolling)]
    print(f"\n  Rolling coverage (last 50 bets at each point):")
    print(f"    Min:  {np.min(valid):.1%}")
    print(f"    Max:  {np.max(valid):.1%}")
    print(f"    Mean: {np.mean(valid):.1%}")
    print(f"    Std:  {np.std(valid):.1%}")
    print(f"    Current (latest window): {valid[-1]:.1%}")


def analyze_va_data(df: pd.DataFrame) -> None:
    """Check for VA interval data and run consistency + correlation."""
    _print_header("VENN-ABERS INTERVAL ANALYSIS")

    has_va_lower = "va_lower" in df.columns
    has_va_upper = "va_upper" in df.columns

    if not (has_va_lower and has_va_upper):
        print(
            "\n  ** VA data (va_lower, va_upper) is NOT persisted in "
            "recommendation CSVs. **"
        )
        print(
            "  To enable this analysis, add va_lower, va_upper, and "
            "va_width columns"
        )
        print(
            "  to the recommendation CSV output in "
            "experiments/generate_daily_recommendations.py."
        )
        print("  Until then, va_consistency and width_accuracy_correlation")
        print("  cannot be computed on production data.")
        return

    va_lo = df["va_lower"].astype(float).values
    va_hi = df["va_upper"].astype(float).values
    va_mask = ~(np.isnan(va_lo) | np.isnan(va_hi))

    if va_mask.sum() < 10:
        print(f"\n  Only {va_mask.sum()} bets with VA data — insufficient")
        return

    actuals = (df["result"] == "W").astype(float).values

    # VA consistency
    consist = va_consistency(va_lo[va_mask], va_hi[va_mask], actuals[va_mask])
    print(f"\n  VA consistency rate: {consist['consistency_rate']:.1%}")
    print(f"  Mean interval width: {consist['mean_width']:.4f}")
    print(f"  n_bets: {consist['n_bets']}")

    # Width-accuracy correlation (the MOST IMPORTANT check)
    probs = df["probability"].astype(float).values
    widths = va_hi[va_mask] - va_lo[va_mask]
    corr = width_accuracy_correlation(widths, actuals[va_mask], probs[va_mask])
    print(f"\n  Width-accuracy correlation:")
    print(f"    Spearman rho: {corr['spearman_rho']:.4f}")
    print(f"    p-value:      {corr['p_value']:.4f}")
    if corr["informative"]:
        print("    --> VA width IS informative — proceed with Phases 2-5")
    else:
        print("    --> VA width is NOT informative — Phases 2-5 may be wasted effort")


def analyze_conformal_lower(df: pd.DataFrame) -> None:
    """Analyze conformal_lower calibration if column is present."""
    _print_header("CONFORMAL LOWER BOUND ANALYSIS")

    if "conformal_lower" not in df.columns:
        print("\n  conformal_lower column not found.")
        return

    cl = df["conformal_lower"].astype(float)
    actuals = (df["result"] == "W").astype(float)
    valid_mask = cl.notna() & (cl > 0)

    if valid_mask.sum() < 10:
        print(f"\n  Only {valid_mask.sum()} bets with conformal_lower — insufficient")
        return

    cl_vals = cl[valid_mask].values
    act_vals = actuals[valid_mask].values

    print(f"\n  Bets with conformal_lower: {valid_mask.sum()}")
    print(f"  Mean conformal_lower:      {np.mean(cl_vals):.4f}")
    print(f"  Actual win rate:           {np.mean(act_vals):.4f}")

    # Check: is actual win rate >= mean conformal_lower?
    # (conformal_lower is the guaranteed lower probability bound)
    if np.mean(act_vals) >= np.mean(cl_vals):
        print("  --> Actual win rate >= mean conformal_lower (consistent)")
    else:
        print("  --> Actual win rate < mean conformal_lower (potential issue)")


def main() -> None:
    """Run retroactive conformal coverage analysis."""
    print("Conformal Coverage Analysis")
    print("=" * 60)

    # Load settled bets with conformal data
    df = _load_settled_with_conformal()

    if df.empty:
        print("\nNo settled bets with conformal_tau found.")
        print()
        print("Possible reasons:")
        print("  1. Conformal predictions were added recently (~Mar 2026)")
        print("     and none of those bets have settled yet.")
        print("  2. Rec CSVs with conformal_tau exist but results are")
        print("     still pending (result column is NaN/empty).")
        print()

        # Report what conformal data exists even if unsettled
        unsettled_count = 0
        for csv_path in sorted(REC_DIR.rglob("rec_*.csv")):
            if "_week" in csv_path.name:
                continue
            try:
                csv_df = pd.read_csv(csv_path)
            except Exception:
                continue
            if "conformal_tau" in csv_df.columns:
                has_tau = csv_df["conformal_tau"].notna() & (
                    csv_df["conformal_tau"] > 0
                )
                unsettled_count += has_tau.sum()

        if unsettled_count > 0:
            print(
                f"Found {unsettled_count} unsettled predictions with "
                f"conformal_tau — re-run after settlement."
            )
        else:
            print("No predictions with conformal_tau found at all.")
        return

    print(f"\nSettled bets with conformal data: {len(df)}")
    print(f"Win rate: {(df['result'] == 'W').mean():.1%}")
    print(f"Markets: {sorted(df['market'].unique())}")

    # Run all analyses
    analyze_conformal_tau_coverage(df)
    analyze_rolling_coverage(df)
    analyze_va_data(df)
    analyze_conformal_lower(df)

    # Summary
    _print_header("DECISION GATE")
    print()
    print("  Check the results above against these criteria:")
    print()
    print("  | Result                                  | Next step              |")
    print("  |-----------------------------------------|------------------------|")
    print("  | VA width correlates (rho>0, p<0.05)     | Proceed to Phases 2+4  |")
    print("  | VA width doesn't correlate              | Skip Phases 2-5        |")
    print("  | Coverage severely under-nominal (<80%)  | Phases 2+3 urgent      |")
    print("  | Coverage near-nominal (88-92%)          | Current approach works |")
    print()


if __name__ == "__main__":
    main()
