#!/usr/bin/env python3
"""
Live performance report: backtest vs live metrics comparison.

Scans settled recommendation CSVs, calculates per-market live metrics
(win rate, ROI, ECE, Sharpe), and compares against backtest metrics
from the deployment config.

Usage:
    python scripts/live_performance_report.py
    python scripts/live_performance_report.py --since 2026-01-15
    python scripts/live_performance_report.py --json-only
    python scripts/live_performance_report.py --rec-dir /tmp/pmi/data/05-recommendations
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Project root (two levels up from scripts/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REC_DIR = PROJECT_ROOT / "data" / "05-recommendations"
DEFAULT_DEPLOYMENT_CONFIG = PROJECT_ROOT / "config" / "sniper_deployment.json"
DEFAULT_OUTPUT_JSON = DEFAULT_REC_DIR / "live_report.json"


# ---------------------------------------------------------------------------
# Market normalization (mirrors analyze_prediction_errors.py logic)
# ---------------------------------------------------------------------------

def normalize_market_label(row: pd.Series) -> str:
    """Map raw CSV market/bet_type/line to deployment config market key."""
    m = str(row.get("market", "")).upper()
    bt = str(row.get("bet_type", "")).upper()
    line = row.get("line")

    if m == "CARDS":
        if bt == "OVER" and pd.notna(line):
            return f"cards_over_{str(line).replace('.', '')}"
        if bt == "UNDER" and pd.notna(line):
            return f"cards_under_{str(line).replace('.', '')}"
        return "cards"
    elif m == "CORNERS":
        if bt == "OVER" and pd.notna(line):
            return f"corners_over_{str(line).replace('.', '')}"
        if bt == "UNDER" and pd.notna(line):
            return f"corners_under_{str(line).replace('.', '')}"
        return "corners"
    elif m == "SHOTS":
        if bt == "OVER" and pd.notna(line):
            return f"shots_over_{str(line).replace('.', '')}"
        if bt == "UNDER" and pd.notna(line):
            return f"shots_under_{str(line).replace('.', '')}"
        return "shots"
    elif m == "FOULS":
        if bt == "OVER" and pd.notna(line):
            return f"fouls_over_{str(line).replace('.', '')}"
        if bt == "UNDER" and pd.notna(line):
            return f"fouls_under_{str(line).replace('.', '')}"
        return "fouls"
    elif m in ("HOME_WIN", "MATCH_RESULT"):
        return "home_win"
    elif m == "AWAY_WIN" or bt == "AWAY_WIN":
        return "away_win"
    elif m == "OVER_2.5" or bt == "OVER_2.5":
        return "over25"
    elif m == "UNDER_2.5" or bt == "UNDER_2.5":
        return "under25"
    elif m == "BTTS":
        return "btts"
    else:
        return f"{m}_{bt}".lower()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_recommendations(rec_dir: Path, since: str | None = None) -> pd.DataFrame:
    """Load and deduplicate all rec_*.csv files under rec_dir.

    Deduplication is by (date, fixture_id, market, bet_type, line) to handle
    the same CSVs appearing under multiple daily-predictions-* artifact dirs.
    """
    csv_files = sorted(rec_dir.rglob("rec_*.csv"))
    if not csv_files:
        logger.error("No rec_*.csv files found under %s", rec_dir)
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            frames.append(df)
        except Exception as exc:
            logger.warning("Skipping %s: %s", f, exc)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    logger.info(
        "Loaded %d rows from %d CSV files (before dedup)", len(combined), len(csv_files)
    )

    # Deduplicate: same fixture + same bet should appear only once
    dedup_cols = ["date", "fixture_id", "market", "bet_type", "line"]
    available_dedup = [c for c in dedup_cols if c in combined.columns]
    if available_dedup:
        combined = combined.drop_duplicates(subset=available_dedup, keep="first")
        logger.info("After dedup: %d rows", len(combined))

    # Date filter
    if since:
        combined["_date_parsed"] = pd.to_datetime(combined["date"], errors="coerce")
        since_dt = pd.to_datetime(since)
        combined = combined[combined["_date_parsed"] >= since_dt].copy()
        combined.drop(columns=["_date_parsed"], inplace=True)
        logger.info("After --since %s filter: %d rows", since, len(combined))

    return combined


def load_deployment_config(path: Path) -> dict[str, Any]:
    """Load sniper_deployment.json and return the markets dict."""
    if not path.exists():
        logger.error("Deployment config not found: %s", path)
        return {}
    with open(path) as f:
        data = json.load(f)
    return data.get("markets", data)


# ---------------------------------------------------------------------------
# Metric calculations
# ---------------------------------------------------------------------------

def calculate_ece(probabilities: np.ndarray, outcomes: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error.

    Bins predictions into n_bins uniform buckets and computes weighted average
    absolute difference between predicted probability and actual outcome rate.
    """
    if len(probabilities) == 0:
        return float("nan")

    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probabilities >= bins[i]) & (probabilities < bins[i + 1])
        if mask.sum() > 0:
            avg_prob = probabilities[mask].mean()
            avg_outcome = outcomes[mask].mean()
            ece += mask.sum() * abs(avg_prob - avg_outcome)
    return ece / len(probabilities)


def calculate_sharpe(returns: np.ndarray) -> float:
    """Sharpe ratio of per-bet returns (mean / std).

    Returns 0.0 when standard deviation is near-zero or there are no returns.
    """
    if len(returns) < 2:
        return 0.0
    std = float(np.std(returns))
    if std < 1e-9:
        return 0.0
    return float(np.mean(returns) / std)


def compute_per_bet_returns(odds: np.ndarray, outcomes: np.ndarray) -> np.ndarray:
    """Compute per-bet flat-stake returns.

    Win: (odds - 1), i.e. profit per unit staked.
    Loss: -1, i.e. lose the stake.
    """
    returns = np.where(outcomes == 1, odds - 1.0, -1.0)
    return returns


def compute_market_metrics(df: pd.DataFrame) -> dict[str, Any]:
    """Compute live metrics for a single market's settled bets."""
    n_bets = len(df)
    if n_bets == 0:
        return {}

    wins = int(df["won"].sum())
    losses = n_bets - wins
    win_rate = wins / n_bets

    # ROI: total PnL / total staked (flat 1-unit stakes)
    odds = df["odds"].values.astype(float)
    outcomes = df["won"].values.astype(float)
    returns = compute_per_bet_returns(odds, outcomes)
    total_pnl = float(returns.sum())
    roi = (total_pnl / n_bets) * 100  # percentage

    # Sharpe ratio of individual bet returns
    sharpe = calculate_sharpe(returns)

    # ECE (only for bets with valid probability)
    prob_mask = df["probability"].notna() & (df["probability"] > 0)
    if prob_mask.sum() > 0:
        probs = df.loc[prob_mask, "probability"].values.astype(float)
        outs = df.loc[prob_mask, "won"].values.astype(float)
        ece = calculate_ece(probs, outs)
    else:
        ece = float("nan")

    return {
        "n_bets": n_bets,
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate * 100, 2),
        "roi": round(roi, 2),
        "total_pnl": round(total_pnl, 2),
        "sharpe": round(sharpe, 4),
        "ece": round(ece, 6) if not np.isnan(ece) else None,
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def build_report(
    rec_dir: Path,
    deployment_path: Path,
    since: str | None = None,
) -> dict[str, Any]:
    """Build the full live performance report."""
    # Load data
    all_recs = load_recommendations(rec_dir, since=since)
    if all_recs.empty:
        logger.error("No recommendation data found.")
        return {"error": "No recommendation data found", "markets": {}}

    deployment = load_deployment_config(deployment_path)

    # Normalize market labels
    all_recs["market_label"] = all_recs.apply(normalize_market_label, axis=1)

    # Filter to settled bets only (result == WON or LOST)
    # Handle both 'result' and 'status' column naming
    result_col = "result" if "result" in all_recs.columns else "status"
    settled = all_recs[all_recs[result_col].isin(["WON", "LOST", "W", "L"])].copy()
    settled["won"] = settled[result_col].isin(["WON", "W"]).astype(int)

    logger.info(
        "Total rows: %d, Settled: %d, Pending: %d",
        len(all_recs),
        len(settled),
        len(all_recs) - len(settled),
    )

    if settled.empty:
        logger.warning("No settled bets found.")
        return {
            "generated_at": datetime.now().isoformat(),
            "total_recs": len(all_recs),
            "total_settled": 0,
            "markets": {},
        }

    # Ensure odds are numeric, filter out invalid odds
    settled["odds"] = pd.to_numeric(settled["odds"], errors="coerce")
    settled["probability"] = pd.to_numeric(settled["probability"], errors="coerce")

    # Remove bets with missing or zero odds (can't compute valid ROI)
    invalid_odds = settled["odds"].isna() | (settled["odds"] <= 0)
    if invalid_odds.sum() > 0:
        logger.warning(
            "Excluding %d bets with missing/zero odds (invalid for ROI calculation)",
            invalid_odds.sum(),
        )
    settled = settled[~invalid_odds].copy()

    # Per-market analysis
    markets_report: dict[str, Any] = {}
    alerts: list[str] = []

    for market_label in sorted(settled["market_label"].unique()):
        mdf = settled[settled["market_label"] == market_label]
        live = compute_market_metrics(mdf)
        if not live:
            continue

        # Pull backtest metrics from deployment config
        bt_config = deployment.get(market_label, {})
        bt_roi = bt_config.get("roi")
        bt_holdout = bt_config.get("holdout_metrics", {})
        bt_precision = bt_holdout.get("precision")
        bt_ece = bt_holdout.get("ece")
        bt_sharpe = bt_holdout.get("sharpe")
        bt_enabled = bt_config.get("enabled", False)

        backtest: dict[str, Any] = {}
        if bt_roi is not None:
            backtest["wf_roi"] = bt_roi
        if bt_holdout:
            backtest["ho_roi"] = bt_holdout.get("roi")
            backtest["ho_precision"] = (
                round(bt_precision * 100, 2) if bt_precision is not None else None
            )
            backtest["ho_n_bets"] = bt_holdout.get("n_bets")
            backtest["ho_sharpe"] = bt_sharpe
            backtest["ho_ece"] = bt_ece
        backtest["enabled"] = bt_enabled

        # Calibration drift flag: live ECE > 2x backtest ECE
        calibration_drift = False
        if (
            live.get("ece") is not None
            and bt_ece is not None
            and bt_ece > 0
            and live["ece"] > 2 * bt_ece
        ):
            calibration_drift = True
            alerts.append(
                f"CALIBRATION DRIFT: {market_label} "
                f"(live ECE={live['ece']:.4f} > 2x backtest ECE={bt_ece:.4f})"
            )

        markets_report[market_label] = {
            "live": live,
            "backtest": backtest,
            "calibration_drift": calibration_drift,
        }

    # Aggregate totals
    total_bets = len(settled)
    total_wins = int(settled["won"].sum())
    total_odds = settled["odds"].values.astype(float)
    total_outcomes = settled["won"].values.astype(float)
    total_returns = compute_per_bet_returns(total_odds, total_outcomes)
    total_pnl = float(total_returns.sum())
    overall_roi = (total_pnl / total_bets) * 100 if total_bets > 0 else 0

    report = {
        "generated_at": datetime.now().isoformat(),
        "since": since,
        "total_recs": len(all_recs),
        "total_settled": total_bets,
        "total_pending": len(all_recs) - total_bets,
        "overall": {
            "win_rate": round(total_wins / total_bets * 100, 2) if total_bets else 0,
            "roi": round(overall_roi, 2),
            "total_pnl": round(total_pnl, 2),
            "sharpe": round(calculate_sharpe(total_returns), 4),
        },
        "markets": markets_report,
        "alerts": alerts,
    }

    return report


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def format_value(val: Any, fmt: str = ".2f") -> str:
    """Format a numeric value, returning 'N/A' for None/NaN."""
    if val is None:
        return "N/A"
    if isinstance(val, float) and np.isnan(val):
        return "N/A"
    return f"{val:{fmt}}"


def print_markdown_report(report: dict[str, Any]) -> None:
    """Print the report as markdown tables to stdout."""
    print("=" * 100)
    print("LIVE PERFORMANCE REPORT: BACKTEST vs LIVE")
    print("=" * 100)
    print(f"Generated: {report['generated_at']}")
    if report.get("since"):
        print(f"Since: {report['since']}")
    print(
        f"Total recommendations: {report['total_recs']} | "
        f"Settled: {report['total_settled']} | "
        f"Pending: {report['total_pending']}"
    )

    overall = report.get("overall", {})
    print(
        f"\nOverall: Win Rate={overall.get('win_rate', 0):.1f}%, "
        f"ROI={overall.get('roi', 0):.1f}%, "
        f"PnL={overall.get('total_pnl', 0):.1f}u, "
        f"Sharpe={overall.get('sharpe', 0):.3f}"
    )

    # Alerts
    alerts = report.get("alerts", [])
    if alerts:
        print(f"\n{'!' * 80}")
        print("ALERTS:")
        for a in alerts:
            print(f"  >> {a}")
        print(f"{'!' * 80}")

    markets = report.get("markets", {})
    if not markets:
        print("\nNo settled market data available.")
        return

    # --- Table 1: Volume & Returns ---
    print("\n" + "-" * 100)
    print("1. PER-MARKET VOLUME & RETURNS")
    print("-" * 100)
    header = (
        f"{'Market':<22} {'Enabled':>7} {'Bets':>5} {'W':>4} {'L':>4} "
        f"{'Win%':>7} {'Live ROI':>9} {'BT WF ROI':>10} {'BT HO ROI':>10} "
        f"{'PnL':>8}"
    )
    print(header)
    print("-" * len(header))

    for mkt in sorted(markets.keys()):
        data = markets[mkt]
        live = data["live"]
        bt = data.get("backtest", {})
        enabled = "YES" if bt.get("enabled") else "NO"
        bt_wf_roi = format_value(bt.get("wf_roi"), ".1f")
        bt_ho_roi = format_value(bt.get("ho_roi"), ".1f")
        print(
            f"{mkt:<22} {enabled:>7} {live['n_bets']:>5} {live['wins']:>4} "
            f"{live['losses']:>4} {live['win_rate']:>6.1f}% "
            f"{live['roi']:>8.1f}% {bt_wf_roi:>10} {bt_ho_roi:>10} "
            f"{live['total_pnl']:>7.1f}u"
        )

    # --- Table 2: Precision / Win Rate Comparison ---
    print("\n" + "-" * 100)
    print("2. PRECISION: BACKTEST vs LIVE")
    print("-" * 100)
    header2 = (
        f"{'Market':<22} {'Live Win%':>10} {'BT HO Prec%':>12} "
        f"{'Delta':>8} {'Live Sharpe':>12} {'BT HO Sharpe':>13}"
    )
    print(header2)
    print("-" * len(header2))

    for mkt in sorted(markets.keys()):
        data = markets[mkt]
        live = data["live"]
        bt = data.get("backtest", {})
        bt_prec = bt.get("ho_precision")
        delta = ""
        if bt_prec is not None:
            d = live["win_rate"] - bt_prec
            delta = f"{d:+.1f}pp"
        bt_sharpe_str = format_value(bt.get("ho_sharpe"), ".3f")
        bt_prec_str = format_value(bt_prec, ".1f")
        print(
            f"{mkt:<22} {live['win_rate']:>9.1f}% {bt_prec_str:>12} "
            f"{delta:>8} {live['sharpe']:>12.3f} {bt_sharpe_str:>13}"
        )

    # --- Table 3: Calibration (ECE) ---
    print("\n" + "-" * 100)
    print("3. CALIBRATION: LIVE ECE vs BACKTEST ECE")
    print("-" * 100)
    header3 = (
        f"{'Market':<22} {'Live ECE':>10} {'BT HO ECE':>10} "
        f"{'Ratio':>8} {'Drift?':>8}"
    )
    print(header3)
    print("-" * len(header3))

    for mkt in sorted(markets.keys()):
        data = markets[mkt]
        live = data["live"]
        bt = data.get("backtest", {})
        live_ece = live.get("ece")
        bt_ece = bt.get("ho_ece")
        drift = "YES" if data.get("calibration_drift") else ""

        if live_ece is not None and bt_ece is not None and bt_ece > 0:
            ratio = f"{live_ece / bt_ece:.2f}x"
        else:
            ratio = "N/A"

        live_ece_str = format_value(live_ece, ".4f")
        bt_ece_str = format_value(bt_ece, ".4f")
        print(
            f"{mkt:<22} {live_ece_str:>10} {bt_ece_str:>10} "
            f"{ratio:>8} {drift:>8}"
        )

    print("\n" + "=" * 100)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate live performance report: backtest vs live metrics."
    )
    parser.add_argument(
        "--rec-dir",
        type=Path,
        default=DEFAULT_REC_DIR,
        help="Root directory to scan for rec_*.csv files (default: data/05-recommendations/)",
    )
    parser.add_argument(
        "--deployment-config",
        type=Path,
        default=DEFAULT_DEPLOYMENT_CONFIG,
        help="Path to sniper_deployment.json (default: config/sniper_deployment.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help="Path for JSON output (default: data/05-recommendations/live_report.json)",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Suppress markdown table output, only write JSON.",
    )
    parser.add_argument(
        "--since",
        type=str,
        default=None,
        help="Filter bets to this date or later (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    report = build_report(
        rec_dir=args.rec_dir,
        deployment_path=args.deployment_config,
        since=args.since,
    )

    # Write JSON
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("JSON report written to %s", args.output)

    # Print markdown table
    if not args.json_only:
        print_markdown_report(report)


if __name__ == "__main__":
    main()
