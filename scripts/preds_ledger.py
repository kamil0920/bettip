#!/usr/bin/env python3
"""Unified prediction ledger management.

Commands:
    import   - Import rec CSV(s) into the ledger
    backfill - Import all historical rec CSVs
    settle   - Update results from settled rec CSVs
    query    - Filter and display ledger entries
    summary  - Per-market breakdown
    add-bet  - Add a bet to my_bets.csv
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LEDGER_DIR = PROJECT_ROOT / "data" / "preds"
LEDGER_PATH = LEDGER_DIR / "predictions.parquet"
BETS_PATH = LEDGER_DIR / "my_bets.csv"
REC_DIR = PROJECT_ROOT / "data" / "05-recommendations"

# Canonical columns for the ledger
LEDGER_COLS = [
    "prediction_id", "run_id", "generated_at",
    "date", "fixture_id", "home_team", "away_team", "league",
    "market", "bet_type", "line", "probability", "edge", "edge_source",
    "odds", "kelly_stake", "model", "threshold", "referee",
    "result", "actual", "settled_at",
]

# Columns from rec CSVs that map directly
REC_DIRECT_COLS = [
    "date", "home_team", "away_team", "league", "market", "bet_type",
    "line", "odds", "probability", "edge", "edge_source", "kelly_stake",
    "referee", "fixture_id", "result", "actual",
]

# V2 schema mapping (older rec files with different column names)
V2_COL_MAP = {
    "our_prob": "probability",
    "edge_pct": "edge",
    "side": "bet_type",
    "status": "result",
    "actual_value": "actual",
    "start_time": "date",
}

V2_RESULT_MAP = {"WON": "W", "LOST": "L", "PENDING": "", "PUSH": "P"}


def _load_ledger() -> pd.DataFrame:
    """Load existing ledger or return empty DataFrame."""
    if LEDGER_PATH.exists():
        return pd.read_parquet(LEDGER_PATH)
    return pd.DataFrame(columns=LEDGER_COLS)


def _save_ledger(df: pd.DataFrame) -> None:
    """Save ledger to parquet."""
    LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(LEDGER_PATH, index=False)


def _make_prediction_id(row: pd.Series) -> str:
    """Create unique prediction ID from date + fixture_id + market."""
    date_str = str(row.get("date", ""))[:10].replace("-", "")
    fixture = str(row.get("fixture_id", "")).split(".")[0]
    market = str(row.get("market", "")).upper()
    return f"{date_str}_{fixture}_{market}"


def _extract_run_id(filepath: Path) -> int:
    """Extract run number from artifact directory name (daily-predictions-NNN)."""
    for part in filepath.parts:
        if part.startswith("daily-predictions-"):
            try:
                return int(part.split("-")[-1])
            except ValueError:
                pass
    return 0


def _normalize_rec(df: pd.DataFrame, filepath: Path) -> pd.DataFrame:
    """Normalize a rec CSV to ledger schema, handling v1 and v2 formats."""
    if df.empty:
        return pd.DataFrame(columns=LEDGER_COLS)

    # Detect v2 schema (has rec_id, our_prob columns)
    if "rec_id" in df.columns or "our_prob" in df.columns:
        df = df.rename(columns=V2_COL_MAP)
        if "result" in df.columns:
            df["result"] = df["result"].map(V2_RESULT_MAP).fillna(df["result"])
        # Extract date from start_time if it was mapped
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime(
                "%Y-%m-%d"
            )

    # Ensure all expected columns exist
    for col in LEDGER_COLS:
        if col not in df.columns:
            df[col] = ""

    # Fill computed fields
    run_id = _extract_run_id(filepath)
    df["run_id"] = run_id
    df["generated_at"] = (
        df.get("created_at", pd.Series(dtype=str))
        if "created_at" in df.columns
        else datetime.now().isoformat()
    )
    df["prediction_id"] = df.apply(_make_prediction_id, axis=1)

    return df[LEDGER_COLS]


def _find_all_rec_csvs() -> list[Path]:
    """Find all rec_*.csv files recursively under data/05-recommendations/."""
    files = sorted(REC_DIR.rglob("rec_*.csv"))
    # Skip week summary files
    return [f for f in files if "_week" not in f.name]


def import_to_ledger(filepath: str | Path, quiet: bool = False) -> int:
    """Import a single rec CSV into the ledger. Returns count of new rows added."""
    filepath = Path(filepath)
    if not filepath.exists():
        if not quiet:
            print(f"File not found: {filepath}")
        return 0

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        if not quiet:
            print(f"Error reading {filepath}: {e}")
        return 0

    if df.empty:
        return 0

    normalized = _normalize_rec(df, filepath)
    ledger = _load_ledger()

    # Deduplicate on prediction_id
    existing_ids = set(ledger["prediction_id"]) if not ledger.empty else set()
    new_rows = normalized[~normalized["prediction_id"].isin(existing_ids)]

    if new_rows.empty:
        if not quiet:
            print(f"  {filepath.name}: 0 new (all duplicates)")
        return 0

    if ledger.empty:
        ledger = new_rows.copy()
    else:
        ledger = pd.concat([ledger, new_rows], ignore_index=True)
    _save_ledger(ledger)

    if not quiet:
        print(f"  {filepath.name}: +{len(new_rows)} new predictions")
    return len(new_rows)


def cmd_import(args: argparse.Namespace) -> None:
    """Import one or more rec CSVs."""
    files = []
    for pattern in args.file:
        p = Path(pattern)
        if p.exists():
            files.append(p)
        else:
            # Try glob
            matches = sorted(Path(".").glob(pattern))
            files.extend(matches)

    if not files:
        print("No files found to import.")
        return

    total_new = 0
    for f in files:
        total_new += import_to_ledger(f)

    ledger = _load_ledger()
    print(f"\nImported {total_new} new predictions. Ledger total: {len(ledger)}")


def cmd_backfill(args: argparse.Namespace) -> None:
    """Import all historical rec CSVs."""
    files = _find_all_rec_csvs()
    print(f"Found {len(files)} rec CSV files to backfill...")

    total_new = 0
    for f in files:
        total_new += import_to_ledger(f)

    ledger = _load_ledger()
    print(f"\nBackfill complete: {total_new} new predictions. Ledger total: {len(ledger)}")


def cmd_settle(args: argparse.Namespace) -> None:
    """Update results in ledger from settled rec CSVs."""
    ledger = _load_ledger()
    if ledger.empty:
        print("Ledger is empty. Run backfill first.")
        return

    # Find unsettled predictions
    unsettled = ledger[ledger["result"].isin(["", None]) | ledger["result"].isna()]
    if unsettled.empty:
        print("All predictions already settled.")
        return

    print(f"{len(unsettled)} unsettled predictions...")

    # Build lookup from all rec CSVs
    settled_count = 0
    for csv_path in _find_all_rec_csvs():
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue

        normalized = _normalize_rec(df, csv_path)
        settled_rows = normalized[
            normalized["result"].isin(["W", "L", "P"])
            & normalized["prediction_id"].isin(unsettled["prediction_id"])
        ]

        for _, row in settled_rows.iterrows():
            mask = ledger["prediction_id"] == row["prediction_id"]
            ledger.loc[mask, "result"] = row["result"]
            ledger.loc[mask, "actual"] = row["actual"]
            ledger.loc[mask, "settled_at"] = datetime.now().isoformat()
            settled_count += 1

    _save_ledger(ledger)
    remaining = len(ledger[ledger["result"].isin(["", None]) | ledger["result"].isna()])
    print(f"Settled {settled_count} predictions. {remaining} still pending.")


def cmd_query(args: argparse.Namespace) -> None:
    """Query and filter ledger entries."""
    ledger = _load_ledger()
    if ledger.empty:
        print("Ledger is empty. Run backfill first.")
        return

    df = ledger.copy()

    if args.market:
        df = df[df["market"].str.upper().str.contains(args.market.upper())]
    if args.league:
        df = df[df["league"].str.contains(args.league, case=False)]
    if args.month:
        df = df[df["date"].str.startswith(args.month)]
    if args.date:
        df = df[df["date"] == args.date]

    if df.empty:
        print("No matching predictions.")
        return

    # Summary stats
    total = len(df)
    settled = df[df["result"].isin(["W", "L", "P"])]
    wins = len(settled[settled["result"] == "W"])
    losses = len(settled[settled["result"] == "L"])
    win_rate = (wins / len(settled) * 100) if len(settled) > 0 else 0
    avg_prob = df["probability"].astype(float, errors="ignore").mean()
    avg_edge = df["edge"].astype(float, errors="ignore").mean()

    print(f"\n{'='*60}")
    print(f"  Predictions: {total}  |  Settled: {len(settled)}  |  Pending: {total - len(settled)}")
    print(f"  Wins: {wins}  |  Losses: {losses}  |  Win rate: {win_rate:.1f}%")
    print(f"  Avg probability: {avg_prob:.3f}  |  Avg edge: {avg_edge:.1f}%")
    print(f"{'='*60}")

    # Show recent entries
    display_cols = ["date", "home_team", "away_team", "market", "probability", "edge", "odds", "result"]
    available = [c for c in display_cols if c in df.columns]
    print(df[available].tail(args.limit).to_string(index=False))


def cmd_summary(args: argparse.Namespace) -> None:
    """Per-market summary of ledger."""
    ledger = _load_ledger()
    if ledger.empty:
        print("Ledger is empty. Run backfill first.")
        return

    df = ledger.copy()
    if args.month:
        df = df[df["date"].str.startswith(args.month)]

    rows = []
    for market in sorted(df["market"].unique()):
        mdf = df[df["market"] == market]
        settled = mdf[mdf["result"].isin(["W", "L", "P"])]
        wins = len(settled[settled["result"] == "W"])
        win_rate = (wins / len(settled) * 100) if len(settled) > 0 else 0

        # ROI calculation (only for bets with real odds > 1)
        roi = 0.0
        if len(settled) > 0:
            try:
                odds_vals = settled["odds"].astype(float)
                with_odds = settled[odds_vals > 1.0]
                if len(with_odds) > 0:
                    o = with_odds["odds"].astype(float)
                    r = with_odds["result"]
                    pnl = sum(
                        (o.iloc[i] - 1) if r.iloc[i] == "W" else -1
                        for i in range(len(with_odds))
                    )
                    roi = (pnl / len(with_odds)) * 100
            except (ValueError, TypeError):
                pass

        rows.append({
            "market": market,
            "total": len(mdf),
            "settled": len(settled),
            "wins": wins,
            "win_rate": f"{win_rate:.0f}%",
            "avg_edge": f"{mdf['edge'].astype(float, errors='ignore').mean():.1f}%",
            "roi": f"{roi:+.1f}%",
        })

    summary = pd.DataFrame(rows)
    print(f"\n{'='*70}")
    if args.month:
        print(f"  Prediction Summary — {args.month}")
    else:
        print("  Prediction Summary — All Time")
    print(f"{'='*70}")
    print(summary.to_string(index=False))
    print(f"\n  Total: {len(df)} predictions across {len(df['market'].unique())} markets")


def cmd_add_bet(args: argparse.Namespace) -> None:
    """Add a bet to my_bets.csv."""
    LEDGER_DIR.mkdir(parents=True, exist_ok=True)

    implied_prob = 1.0 / args.odds if args.odds > 0 else 0
    ev_pct = (args.model_prob * (args.odds - 1) - (1 - args.model_prob)) * 100

    row = {
        "date": args.date or datetime.now().strftime("%Y-%m-%d"),
        "home_team": args.match.split(" vs ")[0] if " vs " in args.match else args.match,
        "away_team": args.match.split(" vs ")[1] if " vs " in args.match else "",
        "league": args.league or "",
        "market": args.market,
        "odds": args.odds,
        "stake": args.stake,
        "bet_type": args.bet_type or "single",
        "parlay_id": args.parlay_id or "",
        "model_prob": args.model_prob,
        "implied_prob": round(implied_prob, 4),
        "ev_pct": round(ev_pct, 1),
        "result": "",
        "actual": "",
        "pnl": "",
        "notes": args.notes or "",
    }

    if BETS_PATH.exists():
        bets = pd.read_csv(BETS_PATH)
        if bets.empty:
            bets = pd.DataFrame([row])
        else:
            bets = pd.concat([bets, pd.DataFrame([row])], ignore_index=True)
    else:
        bets = pd.DataFrame([row])
    bets.to_csv(BETS_PATH, index=False)

    print(f"Added bet: {args.match} — {args.market} @ {args.odds}")
    print(f"  Stake: {args.stake} PLN  |  Model: {args.model_prob:.1%}  |  EV: {ev_pct:+.1f}%")
    print(f"  Total bets: {len(bets)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prediction ledger management")
    sub = parser.add_subparsers(dest="command")

    # import
    p_import = sub.add_parser("import", help="Import rec CSV(s) into ledger")
    p_import.add_argument("--file", nargs="+", required=True, help="CSV file(s) or glob patterns")

    # backfill
    sub.add_parser("backfill", help="Import all historical rec CSVs")

    # settle
    sub.add_parser("settle", help="Update results from settled rec CSVs")

    # query
    p_query = sub.add_parser("query", help="Query ledger entries")
    p_query.add_argument("--market", help="Filter by market (e.g., CARDS_U3.5)")
    p_query.add_argument("--league", help="Filter by league")
    p_query.add_argument("--month", help="Filter by month (YYYY-MM)")
    p_query.add_argument("--date", help="Filter by exact date (YYYY-MM-DD)")
    p_query.add_argument("--limit", type=int, default=20, help="Max rows to display")

    # summary
    p_summary = sub.add_parser("summary", help="Per-market summary")
    p_summary.add_argument("--month", help="Filter by month (YYYY-MM)")

    # add-bet
    p_bet = sub.add_parser("add-bet", help="Add a bet to my_bets.csv")
    p_bet.add_argument("--match", required=True, help='Match (e.g., "Levante vs Valencia")')
    p_bet.add_argument("--market", required=True, help='Market (e.g., "Corners U10.5")')
    p_bet.add_argument("--odds", type=float, required=True, help="Bookmaker odds")
    p_bet.add_argument("--stake", type=float, required=True, help="Stake in PLN")
    p_bet.add_argument("--model-prob", type=float, default=0.0, help="Model probability")
    p_bet.add_argument("--date", help="Match date (YYYY-MM-DD, default: today)")
    p_bet.add_argument("--league", help="League name")
    p_bet.add_argument("--bet-type", default="single", help="single or parlay")
    p_bet.add_argument("--parlay-id", help="Parlay group ID (e.g., P1)")
    p_bet.add_argument("--notes", help="Optional notes")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    commands = {
        "import": cmd_import,
        "backfill": cmd_backfill,
        "settle": cmd_settle,
        "query": cmd_query,
        "summary": cmd_summary,
        "add-bet": cmd_add_bet,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
