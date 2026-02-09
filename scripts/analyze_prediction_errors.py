#!/usr/bin/env python3
"""
Error analysis on settled recommendations.

Analyzes ~800 settled recommendations from rec_*.csv files to find
systematic failure patterns across markets, leagues, calibration buckets,
and confidence levels.

Usage:
    python scripts/analyze_prediction_errors.py \
        --rec-dir /tmp/pmi_feb9/pre-kickoff-175/data/05-recommendations \
        [--strategy-scores /tmp/pmi_feb9/pre-kickoff-175/data/06-prematch/strategy_scores.jsonl]
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load_recommendations(rec_dir: str) -> pd.DataFrame:
    """Load and merge all rec_*.csv files."""
    files = sorted(glob.glob(f"{rec_dir}/rec_*.csv"))
    if not files:
        print(f"ERROR: No rec_*.csv files found in {rec_dir}")
        sys.exit(1)

    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)

    # Unify probability column: some files use 'probability', others 'our_prob'
    df["prob"] = df["probability"].combine_first(df.get("our_prob"))

    # Keep only settled bets
    settled = df[df["status"].isin(["WON", "LOST"])].copy()
    settled["won"] = (settled["status"] == "WON").astype(int)

    print(f"Loaded {len(df)} total rows from {len(files)} files")
    print(f"Settled: {len(settled)} (WON={settled['won'].sum()}, LOST={(~settled['won'].astype(bool)).sum()})")
    print(f"Pending/Unknown: {len(df) - len(settled)}")
    return settled


def load_strategy_scores(path: str) -> pd.DataFrame:
    """Load strategy_scores.jsonl."""
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return pd.DataFrame(records)


def normalize_market(df: pd.DataFrame) -> pd.DataFrame:
    """Create a unified market label combining market + bet_type + line."""
    def _label(row):
        m = str(row.get("market", "")).upper()
        bt = str(row.get("bet_type", "")).upper()
        line = row.get("line")

        if m == "CARDS":
            if bt == "OVER" and pd.notna(line):
                return f"cards_over_{str(line).replace('.', '')}"
            elif bt == "UNDER" and pd.notna(line):
                return f"cards_under_{str(line).replace('.', '')}"
            return "cards"
        elif m == "CORNERS":
            if bt == "OVER" and pd.notna(line):
                return f"corners_over_{str(line).replace('.', '')}"
            return "corners"
        elif m == "SHOTS":
            if bt == "OVER" and pd.notna(line):
                return f"shots_over_{str(line).replace('.', '')}"
            return "shots"
        elif m == "FOULS":
            if bt == "OVER" and pd.notna(line):
                return f"fouls_over_{str(line).replace('.', '')}"
            elif bt == "UNDER" and pd.notna(line):
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

    df["market_label"] = df.apply(_label, axis=1)
    return df


# ─── Analysis functions ────────────────────────────────────────────────

def analysis_per_market(df: pd.DataFrame) -> pd.DataFrame:
    """Per-market hit rate, ROI, volume."""
    grouped = df.groupby("market_label").agg(
        n_bets=("won", "count"),
        wins=("won", "sum"),
        total_pnl=("pnl", "sum"),
    )
    grouped["hit_rate"] = (grouped["wins"] / grouped["n_bets"] * 100).round(1)
    grouped["roi_pct"] = (grouped["total_pnl"] / grouped["n_bets"] * 100).round(1)
    grouped = grouped.sort_values("n_bets", ascending=False)
    return grouped


def analysis_calibration(df: pd.DataFrame) -> pd.DataFrame:
    """Calibration: predicted probability buckets vs actual hit rate."""
    has_prob = df.dropna(subset=["prob"])
    if has_prob.empty:
        return pd.DataFrame()

    bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    labels = ["0.50-0.60", "0.60-0.70", "0.70-0.80", "0.80-0.90", "0.90-1.00"]
    has_prob = has_prob.copy()
    has_prob["prob_bucket"] = pd.cut(has_prob["prob"], bins=bins, labels=labels, include_lowest=True)

    cal = has_prob.groupby("prob_bucket", observed=False).agg(
        n_bets=("won", "count"),
        wins=("won", "sum"),
        avg_predicted=("prob", "mean"),
    )
    cal["actual_hit_rate"] = (cal["wins"] / cal["n_bets"]).round(3)
    cal["avg_predicted"] = cal["avg_predicted"].round(3)
    cal["calibration_gap"] = (cal["avg_predicted"] - cal["actual_hit_rate"]).round(3)
    return cal


def analysis_per_league(df: pd.DataFrame) -> pd.DataFrame:
    """Per-league hit rate."""
    grouped = df.groupby("league").agg(
        n_bets=("won", "count"),
        wins=("won", "sum"),
        total_pnl=("pnl", "sum"),
    )
    grouped["hit_rate"] = (grouped["wins"] / grouped["n_bets"] * 100).round(1)
    grouped["roi_pct"] = (grouped["total_pnl"] / grouped["n_bets"] * 100).round(1)
    grouped = grouped.sort_values("n_bets", ascending=False)
    return grouped


def analysis_market_league(df: pd.DataFrame, min_bets: int = 3) -> pd.DataFrame:
    """Per market+league breakdown — find worst combinations."""
    grouped = df.groupby(["market_label", "league"]).agg(
        n_bets=("won", "count"),
        wins=("won", "sum"),
        total_pnl=("pnl", "sum"),
    )
    grouped["hit_rate"] = (grouped["wins"] / grouped["n_bets"] * 100).round(1)
    grouped["roi_pct"] = (grouped["total_pnl"] / grouped["n_bets"] * 100).round(1)
    grouped = grouped[grouped["n_bets"] >= min_bets].sort_values("roi_pct")
    return grouped


def analysis_edge_vs_outcome(df: pd.DataFrame) -> pd.DataFrame:
    """Does higher predicted edge correlate with higher win rate?"""
    has_edge = df.dropna(subset=["edge"])
    if has_edge.empty:
        return pd.DataFrame()

    has_edge = has_edge.copy()
    bins = [0, 10, 20, 30, 50, 100]
    labels = ["0-10%", "10-20%", "20-30%", "30-50%", "50%+"]
    has_edge["edge_bucket"] = pd.cut(has_edge["edge"], bins=bins, labels=labels, include_lowest=True)

    grouped = has_edge.groupby("edge_bucket", observed=False).agg(
        n_bets=("won", "count"),
        wins=("won", "sum"),
        avg_edge=("edge", "mean"),
        total_pnl=("pnl", "sum"),
    )
    grouped["hit_rate"] = (grouped["wins"] / grouped["n_bets"] * 100).round(1)
    grouped["roi_pct"] = (grouped["total_pnl"] / grouped["n_bets"] * 100).round(1)
    return grouped


def analysis_confidence(df: pd.DataFrame) -> pd.DataFrame:
    """HIGH vs MEDIUM vs LOW confidence hit rates."""
    has_conf = df.dropna(subset=["confidence"])
    if has_conf.empty:
        return pd.DataFrame()

    grouped = has_conf.groupby("confidence").agg(
        n_bets=("won", "count"),
        wins=("won", "sum"),
        total_pnl=("pnl", "sum"),
        avg_prob=("prob", "mean"),
    )
    grouped["hit_rate"] = (grouped["wins"] / grouped["n_bets"] * 100).round(1)
    grouped["roi_pct"] = (grouped["total_pnl"] / grouped["n_bets"] * 100).round(1)
    grouped["avg_prob"] = grouped["avg_prob"].round(3)
    return grouped


def analysis_cards_deepdive(df: pd.DataFrame) -> None:
    """Deep dive into cards market failures."""
    cards = df[df["market_label"].str.startswith("cards")]
    if cards.empty:
        print("  No settled cards bets found.")
        return

    # Check for settlement data issue: all actual=0.0 means settlement is broken
    if "actual" in cards.columns:
        zero_actual = (cards["actual"] == 0.0).sum()
        if zero_actual == len(cards):
            print("\n  *** WARNING: ALL cards bets have actual=0.0 ***")
            print("  This indicates a SETTLEMENT DATA BUG, not model failure.")
            print("  The settlement logic likely can't find cards stats and defaults to 0.")
            print("  Cards hit rates below are UNRELIABLE — ignore until settlement is fixed.\n")

    print(f"\n  Cards bets total: {len(cards)}")
    print(f"  Won: {cards['won'].sum()}, Lost: {(~cards['won'].astype(bool)).sum()}")
    print(f"  Hit rate: {cards['won'].mean() * 100:.1f}%")
    print(f"  Total PnL: {cards['pnl'].sum():.1f}")

    # By sub-market
    sub = cards.groupby("market_label").agg(
        n=("won", "count"), wins=("won", "sum"), pnl=("pnl", "sum"),
    )
    sub["hit_rate"] = (sub["wins"] / sub["n"] * 100).round(1)
    print(f"\n  Per sub-market:")
    print(sub.to_string(index=True))

    # By league
    league = cards.groupby("league").agg(
        n=("won", "count"), wins=("won", "sum"), pnl=("pnl", "sum"),
    )
    league["hit_rate"] = (league["wins"] / league["n"] * 100).round(1)
    league = league.sort_values("n", ascending=False)
    print(f"\n  Per league (cards only):")
    print(league.to_string(index=True))

    # By bet_type (OVER vs UNDER)
    bt = cards.groupby("bet_type").agg(
        n=("won", "count"), wins=("won", "sum"), pnl=("pnl", "sum"),
    )
    bt["hit_rate"] = (bt["wins"] / bt["n"] * 100).round(1)
    print(f"\n  OVER vs UNDER:")
    print(bt.to_string(index=True))

    # Calibration for cards specifically
    has_prob = cards.dropna(subset=["prob"])
    if not has_prob.empty:
        bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        labels = ["0.50-0.60", "0.60-0.70", "0.70-0.80", "0.80-0.90", "0.90-1.00"]
        has_prob = has_prob.copy()
        has_prob["prob_bucket"] = pd.cut(has_prob["prob"], bins=bins, labels=labels, include_lowest=True)
        cal = has_prob.groupby("prob_bucket", observed=False).agg(
            n=("won", "count"), wins=("won", "sum"), avg_pred=("prob", "mean"),
        )
        cal["actual_hit"] = (cal["wins"] / cal["n"]).round(3)
        cal["avg_pred"] = cal["avg_pred"].round(3)
        print(f"\n  Cards calibration:")
        print(cal.to_string(index=True))

    # Feb 8 specifically
    feb8 = cards[cards["date"] == "2026-02-08"]
    if not feb8.empty:
        print(f"\n  === Feb 8 Deep Dive ===")
        print(f"  Feb 8 cards: {len(feb8)} bets, Won={feb8['won'].sum()}, Lost={(~feb8['won'].astype(bool)).sum()}")
        print(f"  Hit rate: {feb8['won'].mean() * 100:.1f}%")
        if not feb8.dropna(subset=["prob"]).empty:
            print(f"  Avg predicted prob: {feb8['prob'].mean():.3f}")
        print(f"\n  Feb 8 individual bets:")
        cols = ["home_team", "away_team", "league", "bet_type", "line", "odds", "prob", "status"]
        available_cols = [c for c in cols if c in feb8.columns]
        print(feb8[available_cols].to_string(index=False))


def analysis_time_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Weekly trend of hit rate and PnL."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["week"] = df["date"].dt.isocalendar().week.astype(str).astype(int)
    grouped = df.groupby("week").agg(
        n_bets=("won", "count"),
        wins=("won", "sum"),
        total_pnl=("pnl", "sum"),
    )
    grouped["hit_rate"] = (grouped["wins"] / grouped["n_bets"] * 100).round(1)
    grouped["roi_pct"] = (grouped["total_pnl"] / grouped["n_bets"] * 100).round(1)
    return grouped


def analysis_odds_range(df: pd.DataFrame) -> pd.DataFrame:
    """Hit rate by odds bucket."""
    has_odds = df.dropna(subset=["odds"])
    has_odds = has_odds[has_odds["odds"] > 0].copy()
    if has_odds.empty:
        return pd.DataFrame()

    bins = [1.0, 1.3, 1.5, 1.7, 2.0, 3.0]
    labels = ["1.0-1.3", "1.3-1.5", "1.5-1.7", "1.7-2.0", "2.0+"]
    has_odds["odds_bucket"] = pd.cut(has_odds["odds"], bins=bins, labels=labels, include_lowest=True)

    grouped = has_odds.groupby("odds_bucket", observed=False).agg(
        n_bets=("won", "count"),
        wins=("won", "sum"),
        avg_odds=("odds", "mean"),
        total_pnl=("pnl", "sum"),
    )
    grouped["hit_rate"] = (grouped["wins"] / grouped["n_bets"] * 100).round(1)
    grouped["roi_pct"] = (grouped["total_pnl"] / grouped["n_bets"] * 100).round(1)
    grouped["implied_prob"] = (1 / grouped["avg_odds"] * 100).round(1)
    return grouped


# ─── Strategy scores analysis ──────────────────────────────────────────

def analysis_strategy_scores(path: str) -> None:
    """Analyze strategy score distribution from JSONL."""
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))

    df = pd.DataFrame(records)
    print(f"\n  Total strategy entries: {len(df)}")
    print(f"  Markets covered: {df['market'].nunique()}")
    print(f"  Date range: {df['date'].min()} → {df['date'].max()}")

    # How often does each strategy pass?
    pass_counts = {}
    for _, row in df.iterrows():
        strategies = row.get("strategies", {})
        for name, strat in strategies.items():
            if name not in pass_counts:
                pass_counts[name] = {"total": 0, "pass": 0}
            pass_counts[name]["total"] += 1
            if strat.get("pass"):
                pass_counts[name]["pass"] += 1

    print("\n  Strategy pass rates:")
    for name, counts in sorted(pass_counts.items(), key=lambda x: x[1]["total"], reverse=True):
        rate = counts["pass"] / counts["total"] * 100 if counts["total"] > 0 else 0
        print(f"    {name:30s}: {counts['pass']:4d}/{counts['total']:4d} ({rate:.1f}%)")

    # How many have real odds?
    real_odds = df["has_real_odds"].sum()
    print(f"\n  With real odds: {real_odds}/{len(df)} ({real_odds / len(df) * 100:.1f}%)")


# ─── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyze settled recommendation errors")
    parser.add_argument("--rec-dir", required=True, help="Directory containing rec_*.csv files")
    parser.add_argument("--strategy-scores", help="Path to strategy_scores.jsonl")
    args = parser.parse_args()

    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 160)

    print("=" * 80)
    print("BETTIP ERROR ANALYSIS REPORT")
    print("=" * 80)

    df = load_recommendations(args.rec_dir)
    df = normalize_market(df)

    # 1. Per-market hit rate
    print("\n" + "─" * 80)
    print("1. PER-MARKET HIT RATE & ROI")
    print("─" * 80)
    market_stats = analysis_per_market(df)
    print(market_stats.to_string())

    # 2. Calibration
    print("\n" + "─" * 80)
    print("2. CALIBRATION (predicted prob vs actual hit rate)")
    print("─" * 80)
    cal = analysis_calibration(df)
    if not cal.empty:
        print(cal.to_string())
        overconfident = cal[cal["calibration_gap"] > 0.1]
        if not overconfident.empty:
            print("\n  ⚠ OVERCONFIDENT buckets (gap > 10pp):")
            print(overconfident.to_string())
    else:
        print("  No probability data available.")

    # 3. Per-league
    print("\n" + "─" * 80)
    print("3. PER-LEAGUE BREAKDOWN")
    print("─" * 80)
    league_stats = analysis_per_league(df)
    print(league_stats.to_string())

    # 4. Market x League (worst combos)
    print("\n" + "─" * 80)
    print("4. WORST MARKET x LEAGUE COMBINATIONS (min 3 bets)")
    print("─" * 80)
    ml = analysis_market_league(df, min_bets=3)
    print(ml.head(20).to_string())

    # 5. Edge vs outcome
    print("\n" + "─" * 80)
    print("5. EDGE vs OUTCOME")
    print("─" * 80)
    edge = analysis_edge_vs_outcome(df)
    if not edge.empty:
        print(edge.to_string())
    else:
        print("  No edge data available.")

    # 6. Confidence levels
    print("\n" + "─" * 80)
    print("6. CONFIDENCE LEVEL ANALYSIS")
    print("─" * 80)
    conf = analysis_confidence(df)
    if not conf.empty:
        print(conf.to_string())
    else:
        print("  No confidence data available.")

    # 7. Cards deep-dive
    print("\n" + "─" * 80)
    print("7. CARDS DEEP-DIVE")
    print("─" * 80)
    analysis_cards_deepdive(df)

    # 8. Time trend
    print("\n" + "─" * 80)
    print("8. WEEKLY TREND")
    print("─" * 80)
    trend = analysis_time_trend(df)
    print(trend.to_string())

    # 9. Odds range analysis
    print("\n" + "─" * 80)
    print("9. ODDS RANGE ANALYSIS")
    print("─" * 80)
    odds = analysis_odds_range(df)
    if not odds.empty:
        print(odds.to_string())
    else:
        print("  No odds data available.")

    # 10. Strategy scores
    if args.strategy_scores and Path(args.strategy_scores).exists():
        print("\n" + "─" * 80)
        print("10. STRATEGY SCORES ANALYSIS")
        print("─" * 80)
        analysis_strategy_scores(args.strategy_scores)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    total_pnl = df["pnl"].sum()
    overall_hr = df["won"].mean() * 100
    print(f"  Overall: {len(df)} settled bets, {overall_hr:.1f}% hit rate, PnL={total_pnl:.1f}")
    print(f"  Best market:  {market_stats['roi_pct'].idxmax()} (ROI {market_stats['roi_pct'].max():.1f}%)")
    print(f"  Worst market: {market_stats['roi_pct'].idxmin()} (ROI {market_stats['roi_pct'].min():.1f}%)")
    if not league_stats.empty:
        worst_league = league_stats[league_stats["n_bets"] >= 5]["roi_pct"]
        if not worst_league.empty:
            print(f"  Worst league (5+ bets): {worst_league.idxmin()} (ROI {worst_league.min():.1f}%)")


if __name__ == "__main__":
    main()
