#!/usr/bin/env python3
"""
Build team rosters cache from raw lineups.parquet files.

Identifies expected starters per team by analyzing recent lineup history.
A player is an "expected starter" if they started >= 3 of the team's last 10 matches.

Zero API calls — purely local data aggregation.

Usage:
    python scripts/build_team_rosters_cache.py
    python scripts/build_team_rosters_cache.py --lookback 10 --min-starts 3
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def build_team_rosters_cache(
    raw_data_dir: Path = Path("data/01-raw"),
    player_stats_cache_path: Path = Path("data/cache/player_stats.parquet"),
    output_path: Path = Path("data/cache/team_rosters.parquet"),
    lookback_matches: int = 10,
    min_starts: int = 3,
) -> pd.DataFrame:
    """
    Build team rosters cache identifying expected starters.

    Args:
        raw_data_dir: Root directory containing raw league data.
        player_stats_cache_path: Path to player stats cache (for avg_rating lookup).
        output_path: Where to write the output parquet.
        lookback_matches: Number of recent matches to consider per team.
        min_starts: Minimum starts in lookback window to be considered expected starter.

    Returns:
        DataFrame with expected starters per team.
    """
    # Load player stats cache for rating lookup
    player_ratings = {}
    if player_stats_cache_path.exists():
        ps_df = pd.read_parquet(player_stats_cache_path)
        for _, row in ps_df.iterrows():
            pid = int(row["player_id"])
            player_ratings[pid] = {
                "avg_rating": row.get("avg_rating", 6.0),
                "position": row.get("position", ""),
            }
        logger.info(f"Loaded {len(player_ratings)} player ratings from cache")
    else:
        logger.warning(f"Player stats cache not found: {player_stats_cache_path}")

    all_records = []

    # Discover all lineups.parquet files
    parquet_files = sorted(raw_data_dir.glob("*/*/lineups.parquet"))
    if not parquet_files:
        logger.warning(f"No lineups.parquet files found in {raw_data_dir}")
        return pd.DataFrame()

    logger.info(f"Found {len(parquet_files)} lineups.parquet files")

    # Process each league — collect all lineups then take most recent per team
    league_data = {}  # league -> list of (fixture_id, date, team_name, player_id, player_name, pos, type)

    for pf in parquet_files:
        season = pf.parent.name
        league = pf.parent.parent.name

        try:
            df = pd.read_parquet(pf)
        except Exception as e:
            logger.warning(f"Failed to read {pf}: {e}")
            continue

        # Filter to actual player rows
        df = df.dropna(subset=["id", "fixture_id"])
        if df.empty:
            continue

        # Only starters
        starters = df[df["type"] == "StartXI"].copy()
        if starters.empty:
            continue

        if league not in league_data:
            league_data[league] = []

        for _, row in starters.iterrows():
            league_data[league].append({
                "fixture_id": int(row["fixture_id"]),
                "date": row.get("date", ""),
                "team_name": row.get("team_name", ""),
                "player_id": int(row["id"]),
                "player_name": row.get("name", ""),
                "position": row.get("pos", ""),
                "season": season,
            })

        logger.info(f"  {league}/{season}: {len(starters)} starter entries")

    # For each league, find expected starters per team from most recent matches
    for league, records in league_data.items():
        ldf = pd.DataFrame(records)

        # Sort by date descending to get most recent matches first
        ldf = ldf.sort_values("date", ascending=False)

        for team_name, team_df in ldf.groupby("team_name"):
            # Get unique fixture IDs (most recent first)
            fixtures = team_df["fixture_id"].unique()[:lookback_matches]
            recent = team_df[team_df["fixture_id"].isin(fixtures)]

            # Count starts per player
            starts_count = recent.groupby("player_id").agg(
                starts=("fixture_id", "nunique"),
                player_name=("player_name", "first"),
                position=("position", "first"),
            ).reset_index()

            # Filter to expected starters
            expected = starts_count[starts_count["starts"] >= min_starts]

            for _, row in expected.iterrows():
                pid = int(row["player_id"])
                rating_info = player_ratings.get(pid, {})

                all_records.append({
                    "team_name": team_name,
                    "player_id": pid,
                    "player_name": row["player_name"],
                    "starts_in_last_n": int(row["starts"]),
                    "avg_rating": rating_info.get("avg_rating", 6.0),
                    "position": rating_info.get("position", row["position"]),
                })

        logger.info(
            f"  {league}: {ldf['team_name'].nunique()} teams processed"
        )

    if not all_records:
        logger.warning("No roster records collected")
        return pd.DataFrame()

    result = pd.DataFrame(all_records)

    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)
    logger.info(
        f"Saved {len(result)} expected starters "
        f"across {result['team_name'].nunique()} teams to {output_path}"
    )

    return result


def main():
    parser = argparse.ArgumentParser(description="Build team rosters cache")
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/01-raw",
        help="Raw data directory",
    )
    parser.add_argument(
        "--player-stats-cache",
        type=str,
        default="data/cache/player_stats.parquet",
        help="Path to player stats cache for rating lookup",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/cache/team_rosters.parquet",
        help="Output parquet path",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=10,
        help="Number of recent matches to consider",
    )
    parser.add_argument(
        "--min-starts",
        type=int,
        default=3,
        help="Minimum starts to be considered expected starter",
    )
    args = parser.parse_args()

    result = build_team_rosters_cache(
        raw_data_dir=Path(args.raw_dir),
        player_stats_cache_path=Path(args.player_stats_cache),
        output_path=Path(args.output),
        lookback_matches=args.lookback,
        min_starts=args.min_starts,
    )

    if result.empty:
        print("No roster data found")
        return 1

    print(f"\nTeam rosters cache built: {len(result)} expected starters")
    print(f"Teams: {result['team_name'].nunique()}")
    print(f"\nStarters per team:")
    per_team = result.groupby("team_name").size()
    print(f"  Mean: {per_team.mean():.1f}, Min: {per_team.min()}, Max: {per_team.max()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
