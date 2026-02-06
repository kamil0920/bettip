#!/usr/bin/env python3
"""
Build player stats cache from raw player_stats.parquet files.

Reads existing data/01-raw/{league}/{season}/player_stats.parquet files,
aggregates per-player statistics, and writes data/cache/player_stats.parquet.

Zero API calls — purely local data aggregation.

Usage:
    python scripts/build_player_stats_cache.py
    python scripts/build_player_stats_cache.py --min-matches 5
    python scripts/build_player_stats_cache.py --output data/cache/player_stats.parquet
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

# Season recency weights: more recent seasons count more
SEASON_WEIGHTS = {
    "2024": 2.0,
    "2023": 1.5,
    "2022": 1.0,
    "2021": 1.0,
    "2020": 1.0,
    "2019": 0.8,
}
DEFAULT_WEIGHT = 0.8


def build_player_stats_cache(
    raw_data_dir: Path = Path("data/01-raw"),
    output_path: Path = Path("data/cache/player_stats.parquet"),
    min_matches: int = 3,
) -> pd.DataFrame:
    """
    Build player stats cache from raw data.

    Args:
        raw_data_dir: Root directory containing raw league data.
        output_path: Where to write the output parquet.
        min_matches: Minimum matches for a player to be included.

    Returns:
        DataFrame with aggregated player statistics.
    """
    all_records = []

    # Discover all player_stats.parquet files
    parquet_files = sorted(raw_data_dir.glob("*/*/player_stats.parquet"))
    if not parquet_files:
        logger.warning(f"No player_stats.parquet files found in {raw_data_dir}")
        return pd.DataFrame()

    logger.info(f"Found {len(parquet_files)} player_stats.parquet files")

    for pf in parquet_files:
        # Extract league and season from path: data/01-raw/{league}/{season}/player_stats.parquet
        season = pf.parent.name
        league = pf.parent.parent.name

        try:
            df = pd.read_parquet(pf)
        except Exception as e:
            logger.warning(f"Failed to read {pf}: {e}")
            continue

        # Filter to actual player rows (non-null player id and fixture_id)
        df = df.dropna(subset=["id", "fixture_id"])
        if df.empty:
            continue

        # Parse rating from string to float
        if "games.rating" in df.columns:
            df["rating"] = pd.to_numeric(df["games.rating"], errors="coerce")
        else:
            df["rating"] = np.nan

        # Parse minutes
        if "games.minutes" in df.columns:
            df["minutes"] = pd.to_numeric(df["games.minutes"], errors="coerce").fillna(0)
        else:
            df["minutes"] = 0

        # Get season weight
        weight = SEASON_WEIGHTS.get(season, DEFAULT_WEIGHT)

        # Aggregate per player within this season/league file
        for player_id, grp in df.groupby("id"):
            player_id = int(player_id)
            name = grp["name"].dropna().iloc[0] if not grp["name"].dropna().empty else ""
            position = ""
            if "games.position" in grp.columns:
                pos_vals = grp["games.position"].dropna()
                if not pos_vals.empty:
                    position = pos_vals.mode().iloc[0] if len(pos_vals.mode()) > 0 else pos_vals.iloc[0]

            # Only count matches where player actually played (minutes > 0)
            played = grp[grp["minutes"] > 0]
            matches = len(played)
            if matches == 0:
                continue

            total_minutes = float(played["minutes"].sum())
            ratings = played["rating"].dropna()
            avg_rating = float(ratings.mean()) if not ratings.empty else 6.0

            # Goals and assists (handle string "None" and NaN values)
            if "goals.total" in played.columns:
                goals = float(pd.to_numeric(played["goals.total"], errors="coerce").fillna(0).sum())
            else:
                goals = 0.0
            if "goals.assists" in played.columns:
                assists = float(pd.to_numeric(played["goals.assists"], errors="coerce").fillna(0).sum())
            else:
                assists = 0.0

            # Per-90 stats (avoid division by zero)
            goals_per_90 = (goals / total_minutes) * 90 if total_minutes > 0 else 0.0
            assists_per_90 = (assists / total_minutes) * 90 if total_minutes > 0 else 0.0

            all_records.append({
                "player_id": player_id,
                "player_name": name,
                "position": position,
                "league": league,
                "season": season,
                "matches_played": matches,
                "total_minutes": int(total_minutes),
                "avg_rating": avg_rating,
                "goals_per_90": goals_per_90,
                "assists_per_90": assists_per_90,
                "weight": weight,
            })

        logger.info(
            f"  {league}/{season}: {len(df)} rows, "
            f"{df['id'].nunique()} unique players"
        )

    if not all_records:
        logger.warning("No player records collected")
        return pd.DataFrame()

    records_df = pd.DataFrame(all_records)
    logger.info(
        f"Collected {len(records_df)} player-season records "
        f"for {records_df['player_id'].nunique()} unique players"
    )

    # Aggregate across seasons with weighting
    aggregated = _aggregate_across_seasons(records_df, min_matches)

    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    aggregated.to_parquet(output_path, index=False)
    logger.info(f"Saved {len(aggregated)} players to {output_path}")

    return aggregated


def _aggregate_across_seasons(df: pd.DataFrame, min_matches: int) -> pd.DataFrame:
    """Aggregate player stats across seasons with recency weighting."""
    results = []

    for player_id, grp in df.groupby("player_id"):
        total_matches = int(grp["matches_played"].sum())
        if total_matches < min_matches:
            continue

        total_minutes = int(grp["total_minutes"].sum())

        # Weighted average rating (by season weight × matches)
        weights = grp["weight"] * grp["matches_played"]
        avg_rating = float(np.average(grp["avg_rating"], weights=weights))

        # Weighted per-90 stats
        goals_per_90 = float(np.average(grp["goals_per_90"], weights=weights))
        assists_per_90 = float(np.average(grp["assists_per_90"], weights=weights))

        # Most recent name and position
        latest = grp.sort_values("season", ascending=False).iloc[0]

        results.append({
            "player_id": int(player_id),
            "player_name": latest["player_name"],
            "avg_rating": round(avg_rating, 2),
            "total_minutes": total_minutes,
            "matches_played": total_matches,
            "goals_per_90": round(goals_per_90, 4),
            "assists_per_90": round(assists_per_90, 4),
            "position": latest["position"],
        })

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Build player stats cache")
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/01-raw",
        help="Raw data directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/cache/player_stats.parquet",
        help="Output parquet path",
    )
    parser.add_argument(
        "--min-matches",
        type=int,
        default=3,
        help="Minimum matches to include a player",
    )
    args = parser.parse_args()

    result = build_player_stats_cache(
        raw_data_dir=Path(args.raw_dir),
        output_path=Path(args.output),
        min_matches=args.min_matches,
    )

    if result.empty:
        print("No players found")
        return 1

    print(f"\nPlayer stats cache built: {len(result)} players")
    print(f"Columns: {result.columns.tolist()}")
    print(f"\nPosition distribution:")
    print(result["position"].value_counts().to_string())
    print(f"\nRating distribution:")
    print(result["avg_rating"].describe().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
