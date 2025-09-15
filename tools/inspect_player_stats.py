#!/usr/bin/env python3
"""
inspect_player_stats.py

Quick inspector for player_stats.parquet produced by normalize_season.py.

Usage examples:
  python3 tools/inspect_player_stats.py --season 2025
  python3 tools/inspect_player_stats.py --file data.json/seasons/2025/player_stats.parquet
"""
from pathlib import Path
import argparse
import pandas as pd
import json
import textwrap

TARGET_COLS = [
    'player_id', 'player_name', 'player_photo', 'offsides', 'games_minutes',
    'games_number', 'games_position', 'games_rating', 'games_captain',
    'games_substitute', 'shots_total', 'shots_on', 'goals_total',
    'goals_conceded', 'goals_assists', 'goals_saves', 'passes_total',
    'passes_key', 'passes_accuracy', 'tackles_total', 'tackles_blocks',
    'tackles_interceptions', 'duels_total', 'duels_won',
    'dribbles_attempts', 'dribbles_success', 'dribbles_past', 'fouls_drawn',
    'fouls_committed', 'cards_yellow', 'cards_red', 'penalty_won',
    'penalty_commited', 'penalty_scored', 'penalty_missed', 'penalty_saved',
    'fixture_id'
]

def show_sample(series, max_items=5):
    nonnull = series.dropna()
    if nonnull.empty:
        return []
    # prefer small unique sample
    uniq = pd.Series(nonnull.unique())
    return uniq.head(max_items).tolist()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, help="Season year (used to build default path)")
    ap.add_argument("--file", type=str, help="Full path to player_stats.parquet (overrides --season)")
    args = ap.parse_args()

    if args.file:
        fp = Path(args.file)
    elif args.season:
        fp = Path("data.json") / "seasons" / str(args.season) / "player_stats.parquet"
    else:
        ap.error("Either --file or --season must be provided")

    if not fp.exists():
        raise SystemExit(f"ERROR: file not found: {fp}")

    print(f"Loading: {fp} ...")
    df = pd.read_parquet(fp)
    print("Shape:", df.shape)
    print()

    cols = list(df.columns)
    print("Total columns:", len(cols))
    print("First 40 columns:", cols[:40])
    print()

    # Target columns summary
    print("=== Target columns summary ===")
    missing = []
    present = []
    for c in TARGET_COLS:
        if c in df.columns:
            present.append(c)
            s = df[c]
            dtype = s.dtype
            nonnull = s.dropna().shape[0]
            uniq_count = s.nunique(dropna=True)
            samples = show_sample(s, max_items=6)
            print(f"- {c:25} dtype={dtype:10} non-null={nonnull:5} uniques={uniq_count:4}  samples={samples}")
        else:
            missing.append(c)
            print(f"- {c:25} MISSING")

    print()
    print(f"Present target cols: {len(present)}, Missing target cols: {len(missing)}")
    if missing:
        print("Missing (first 20):", missing[:20])
    print()

    # Show a quick cross-check: top 10 rows with canonical target subset (if present)
    print("=== Example rows (head) for available target columns ===")
    show_cols = [c for c in TARGET_COLS if c in df.columns][:20]
    if show_cols:
        with pd.option_context('display.max_columns', 40, 'display.width', 200):
            print(df[show_cols].head(10).to_string(index=False))
    else:
        print("No target columns present to preview.")

    print()

    # Extra flattened columns not in target (helpful to spot aliases/paths to add)
    extra_cols = [c for c in cols if c not in TARGET_COLS]
    print("=== Extra flattened columns (sample) ===")
    print(f"Total extra columns: {len(extra_cols)}")
    # print the first 60 extra names in a readable block
    for c in extra_cols[:60]:
        print(" -", c)
    if len(extra_cols) > 60:
        print(f" ... and {len(extra_cols)-60} more")

    # For debugging: show the 'raw' column sample if present
    if 'raw' in df.columns:
        print()
        print("=== Sample 'raw' JSON snippets (first 3 non-null) ===")
        nonnull_raw = df['raw'].dropna().head(6)
        for i, val in enumerate(nonnull_raw):
            print(f"--- raw sample {i+1} ---")
            try:
                parsed = json.loads(val) if isinstance(val, str) else val
                pretty = json.dumps(parsed, indent=2) if isinstance(parsed, (dict, list)) else str(parsed)
                print(textwrap.shorten(pretty, width=1000, placeholder="..."))
            except Exception:
                print(str(val)[:1000])

if __name__ == "__main__":
    main()
