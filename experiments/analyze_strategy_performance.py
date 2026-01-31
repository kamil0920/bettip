#!/usr/bin/env python3
"""
Analyze strategy performance from strategy_results.jsonl.

Prints comparison table per market showing bets, wins, precision, and ROI
for each strategy variant.

Usage:
    python experiments/analyze_strategy_performance.py
    python experiments/analyze_strategy_performance.py --market over25
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
RESULTS_FILE = project_root / "data" / "06-prematch" / "strategy_results.jsonl"


def load_results() -> list:
    """Load all result entries from JSONL."""
    if not RESULTS_FILE.exists():
        print(f"No results file found: {RESULTS_FILE}")
        return []

    entries = []
    with open(RESULTS_FILE) as f:
        for line in f:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def analyze(market_filter: str = None):
    """Print strategy comparison tables."""
    entries = load_results()
    if not entries:
        return

    # Group by market
    by_market = defaultdict(list)
    for e in entries:
        by_market[e["market"]].append(e)

    markets = [market_filter] if market_filter else sorted(by_market.keys())

    for market in markets:
        market_entries = by_market.get(market, [])
        if not market_entries:
            print(f"\nNo results for market: {market}")
            continue

        deployed = market_entries[0].get("deployed", "unknown")

        # Collect per-strategy stats
        stats = defaultdict(lambda: {"bets": 0, "wins": 0, "pnl": 0.0, "no_bet": 0})

        for entry in market_entries:
            for strat_name, strat_data in entry.get("strategies", {}).items():
                outcome = strat_data.get("outcome")
                if outcome is None:
                    continue
                s = stats[strat_name]
                if outcome == "WON":
                    s["bets"] += 1
                    s["wins"] += 1
                    s["pnl"] += strat_data.get("pnl", 0)
                elif outcome == "LOST":
                    s["bets"] += 1
                    s["pnl"] += strat_data.get("pnl", 0)
                elif outcome == "NO_BET":
                    s["no_bet"] += 1

        if not stats:
            continue

        print(f"\nMARKET: {market} (deployed={deployed}, {len(market_entries)} matches)")
        print(f"{'Strategy':<20} | {'Bets':>5} | {'Wins':>5} | {'Precision':>10} | {'ROI':>8}")
        print("-" * 62)

        sorted_strats = sorted(
            stats.items(),
            key=lambda x: (x[1]["pnl"] / x[1]["bets"] * 100) if x[1]["bets"] > 0 else -999,
            reverse=True,
        )

        for strat_name, s in sorted_strats:
            bets = s["bets"]
            wins = s["wins"]
            precision = f"{wins/bets*100:.1f}%" if bets > 0 else "N/A"
            roi = f"{s['pnl']/bets*100:+.1f}%" if bets > 0 else "N/A"
            marker = " *" if strat_name == deployed else ""
            print(f"{strat_name:<20} | {bets:>5} | {wins:>5} | {precision:>10} | {roi:>8}{marker}")

        print("* = currently deployed")


def main():
    parser = argparse.ArgumentParser(description="Analyze strategy performance")
    parser.add_argument("--market", type=str, help="Filter to specific market")
    args = parser.parse_args()

    analyze(market_filter=args.market)


if __name__ == "__main__":
    main()
