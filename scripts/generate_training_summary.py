#!/usr/bin/env python3
"""Generate a rich training summary table from sniper optimization JSON results.

Produces a markdown table matching the deployed_models format with:
Market | Model | Thr | WF_P | WF_B | HO_P | HO_B | ECE | FVA | TS | #F | Odds | Gate

Usage:
    python scripts/generate_training_summary.py <results_dir> [--output SUMMARY.md]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


MIN_HO_BETS_CLEAN: int = 100  # Hard gate: CLEAN requires >= 100 HO bets


def classify_gate(
    ho_n_bets: int,
    ts: Optional[float],
    ts_rejected: Optional[bool],
    ece: Optional[float],
) -> str:
    """Classify market into deployment gate category.

    Asymmetric TS: positive TS (overprediction) is dangerous,
    negative TS (underprediction) is safe/conservative.

    CLEAN:      TS < +4 AND HO_B >= 100 (includes negative TS — safe)
    INCUBATION: TS < +4 but HO_B < 100 (statistically insufficient)
    CAUTION:    TS >= +4 (overpredicting, dangerous)
    WARNING:    TS <= -15 (deeply underpredicting, safe but too conservative)
    NO_HO:      no holdout data
    """
    if ho_n_bets == 0:
        return "NO_HO"
    if ece is not None and ece > 0.10:
        return "FAIL_ECE"
    if ts is None:
        return "NO_TS"
    # Overprediction (TS > 0) is dangerous — hard gate
    if ts >= 4.0:
        return "CAUTION"
    # Deep underprediction — safe but needs recalibration
    if ts <= -15.0:
        return "WARNING"
    # TS in safe range but insufficient sample size
    if ho_n_bets < MIN_HO_BETS_CLEAN:
        return "INCUBATION"
    return "CLEAN"


def parse_result(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract summary fields from a sniper result JSON."""
    bet_type = data.get("bet_type")
    if not bet_type:
        return None

    ho = data.get("holdout_metrics") or {}
    estimated = data.get("estimated_odds", False)

    # Walk-forward: top-level precision/roi/n_bets are WF best
    wf_precision = data.get("precision") or 0
    wf_roi = data.get("roi") or 0
    wf_bets = data.get("n_bets") or 0

    # Holdout
    ho_precision = ho.get("precision") or 0
    ho_bets = ho.get("n_bets") or 0
    ho_ece = ho.get("ece")
    ho_brier = ho.get("brier")
    ho_log_loss = ho.get("log_loss")
    ho_fva = ho.get("fva")
    ho_ts = ho.get("tracking_signal")
    ts_rejected = ho.get("ts_rejected", False)
    ho_roi = ho.get("roi") or 0
    ho_roi_ci_lower = ho.get("roi_ci_lower")
    ho_sharpe = ho.get("sharpe")

    gate = classify_gate(ho_bets, ho_ts, ts_rejected, ho_ece)

    return {
        "bet_type": bet_type,
        "model": data.get("best_model") or "N/A",
        "threshold": data.get("best_threshold") or 0,
        "wf_precision": wf_precision,
        "wf_bets": wf_bets,
        "wf_roi": wf_roi,
        "ho_precision": ho_precision,
        "ho_bets": ho_bets,
        "ho_roi": ho_roi,
        "ho_roi_ci_lower": ho_roi_ci_lower,
        "ho_ece": ho_ece,
        "ho_brier": ho_brier,
        "ho_log_loss": ho_log_loss,
        "ho_fva": ho_fva,
        "ho_ts": ho_ts,
        "ts_rejected": ts_rejected,
        "ho_sharpe": ho_sharpe,
        "n_features": data.get("n_features") or 0,
        "adv_auc": data.get("adversarial_auc_mean") or 0,
        "estimated_odds": estimated,
        "calibration": data.get("calibration_method") or "",
        "gate": gate,
        "saved_models": data.get("saved_models"),
    }


def fmt(val: Optional[float], fmt_str: str, fallback: str = "—") -> str:
    """Format a value with fallback for None."""
    if val is None:
        return fallback
    return fmt_str.format(val)


def generate_summary(results: List[Dict[str, Any]]) -> str:
    """Generate markdown summary from parsed results."""
    lines: List[str] = []

    # Sort: deployable first (by ho_bets desc), then failed (by bet_type)
    deployable = [r for r in results if r["ho_bets"] >= 20 and r["gate"] != "FAIL_ECE"]
    marginal = [r for r in results if 0 < r["ho_bets"] < 20]
    failed = [r for r in results if r["ho_bets"] == 0]

    deployable.sort(key=lambda x: -x["ho_bets"])
    marginal.sort(key=lambda x: -x["ho_bets"])
    failed.sort(key=lambda x: x["bet_type"])

    # Header
    header = (
        "| Market | Model | Thr | WF_P | WF_B | HO_P | HO_B | ECE | Brier | LL | "
        "FVA | TS | #F | Odds | Gate |"
    )
    sep = (
        "|--------|-------|-----|------|------|------|------|-----|-------|----|----|"
        "----|------|------|------|"
    )

    lines.append("")
    lines.append("### Training Results")
    lines.append("")
    lines.append(
        "Legend: WF=Walk-Forward (optimization folds), HO=Holdout (out-of-sample), "
        "FVA=Forecast Value Added, TS=Tracking Signal (* = ts_rejected)"
    )
    lines.append(
        f"Gate: CLEAN (|TS|<4 & HO>={MIN_HO_BETS_CLEAN}), "
        f"INCUBATION (|TS|<4 but HO<{MIN_HO_BETS_CLEAN}), "
        "CAUTION (rejected+|TS|>=4), EXTREME (|TS|>=4, confirmed)"
    )
    lines.append(
        "**ROI is MEANINGLESS for EST-odds markets.** Evaluate on precision, ECE, FVA."
    )
    lines.append("")
    lines.append(header)
    lines.append(sep)

    def row(r: Dict[str, Any]) -> str:
        odds = "EST" if r["estimated_odds"] else "REAL"
        ts_star = "*" if r["ts_rejected"] else ""
        ts_str = fmt(r["ho_ts"], "{:+.1f}", "—") + ts_star
        ece_str = fmt(r["ho_ece"], "{:.3f}", "—")
        brier_str = fmt(r["ho_brier"], "{:.4f}", "—")
        ll_str = fmt(r["ho_log_loss"], "{:.4f}", "—")
        fva_str = fmt(r["ho_fva"], "{:+.2f}", "—")
        ho_p_str = f"{r['ho_precision']:.1%}" if r["ho_bets"] > 0 else "—"
        wf_p_str = f"{r['wf_precision']:.1%}" if r["wf_bets"] > 0 else "—"

        return (
            f"| {r['bet_type']} | {r['model']} | {r['threshold']:.2f} | "
            f"{wf_p_str} | {r['wf_bets']} | {ho_p_str} | {r['ho_bets']} | "
            f"{ece_str} | {brier_str} | {ll_str} | "
            f"{fva_str} | {ts_str} | {r['n_features']} | {odds} | {r['gate']} |"
        )

    if deployable:
        lines.append(f"| **DEPLOYABLE ({len(deployable)})** | | | | | | | | | | | | | | |")
        for r in deployable:
            lines.append(row(r))

    if marginal:
        lines.append(f"| **MARGINAL HO <20 ({len(marginal)})** | | | | | | | | | | | | | | |")
        for r in marginal:
            lines.append(row(r))

    if failed:
        lines.append(f"| **FAILED 0 HO ({len(failed)})** | | | | | | | | | | | | | | |")
        for r in failed:
            lines.append(row(r))

    # Summary stats
    lines.append("")
    lines.append("### Summary")
    lines.append("")
    lines.append(f"- **Total markets:** {len(results)}")
    lines.append(f"- **Deployable (HO >= 20, ECE < 0.10):** {len(deployable)}")
    lines.append(f"- **Marginal (0 < HO < 20):** {len(marginal)}")
    lines.append(f"- **Failed (0 HO bets):** {len(failed)}")

    clean = sum(1 for r in results if r["gate"] == "CLEAN")
    incubation = sum(1 for r in results if r["gate"] == "INCUBATION")
    caution = sum(1 for r in results if r["gate"] == "CAUTION")
    extreme = sum(1 for r in results if r["gate"] == "EXTREME")
    lines.append(
        f"- **Gate breakdown:** {clean} CLEAN, {incubation} INCUBATION, "
        f"{caution} CAUTION, {extreme} EXTREME"
    )

    # Separate summary by odds source (REAL vs EST)
    real_markets = [r for r in deployable if not r["estimated_odds"]]
    est_markets = [r for r in deployable if r["estimated_odds"]]

    lines.append("")
    lines.append("### Odds Source Breakdown (Deployable Only)")
    lines.append("")
    lines.append(
        "**ROI is only meaningful for REAL-odds markets.** "
        "EST-odds markets must be evaluated on Precision, Brier, LogLoss, ECE."
    )
    lines.append("")

    for label, group in [("REAL", real_markets), ("EST", est_markets)]:
        if not group:
            lines.append(f"- **{label} odds:** 0 markets")
            continue
        avg_prec = sum(r["ho_precision"] for r in group) / len(group)
        avg_bets = sum(r["ho_bets"] for r in group) / len(group)
        brier_vals = [r["ho_brier"] for r in group if r.get("ho_brier") is not None]
        avg_brier = sum(brier_vals) / len(brier_vals) if brier_vals else None
        ll_vals = [r["ho_log_loss"] for r in group if r.get("ho_log_loss") is not None]
        avg_ll = sum(ll_vals) / len(ll_vals) if ll_vals else None
        brier_str = f", avg Brier={avg_brier:.4f}" if avg_brier is not None else ""
        ll_str = f", avg LL={avg_ll:.4f}" if avg_ll is not None else ""

        lines.append(
            f"- **{label} odds:** {len(group)} markets, "
            f"avg HO_P={avg_prec:.1%}, avg HO_B={avg_bets:.0f}"
            f"{brier_str}{ll_str}"
        )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate training summary table")
    parser.add_argument("results_dir", help="Directory with sniper JSON results")
    parser.add_argument("--output", "-o", default=None, help="Output file (default: stdout)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        print(f"Error: {results_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    # Find all per-market JSON files (not sniper_all_*)
    json_files = sorted(
        f
        for f in results_dir.glob("sniper_*_2026*.json")
        if not f.name.startswith("sniper_all_")
    )

    if not json_files:
        print("No sniper result JSON files found", file=sys.stderr)
        sys.exit(1)

    results = []
    for jf in json_files:
        try:
            with open(jf) as f:
                data = json.load(f)
            if isinstance(data, list):
                for entry in data:
                    parsed = parse_result(entry)
                    if parsed:
                        results.append(parsed)
            elif isinstance(data, dict):
                parsed = parse_result(data)
                if parsed:
                    results.append(parsed)
        except Exception as e:
            print(f"Error parsing {jf}: {e}", file=sys.stderr)

    summary = generate_summary(results)

    if args.output:
        with open(args.output, "a") as f:
            f.write(summary)
            f.write("\n")
        print(f"Summary appended to {args.output}")
    else:
        print(summary)


if __name__ == "__main__":
    main()
